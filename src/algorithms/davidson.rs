/*!

# Davidson Diagonalization

The Davidson method is suitable for diagonal-dominant symmetric matrices,
that are quite common in certain scientific problems like [electronic
structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson
method could be not practical for other kind of symmetric matrices.

The current implementation uses a general davidson algorithm, meaning
that it compute all the requested eigenvalues simultaneusly using a variable
size block approach. The family of Davidson algorithm only differ in the way
that the correction vector is computed.

Available correction methods are:
 * **DPR**: Diagonal-Preconditioned-Residue
 * **GJD**: Generalized Jacobi Davidson

*/

use super::{DavidsonCorrection, SpectrumTarget};
use crate::utils;
use crate::MGS;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray::stack;
use ndarray_stats::QuantileExt;
use std::iter::FromIterator;
use std::error;
use std::fmt;
use std::time::Instant;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::distributions::Uniform;

/// Structure containing the initial configuration data
struct Config {
    method: DavidsonCorrection,
    spectrum_target: SpectrumTarget,
    tolerance: f64,
    max_iters: usize,
    max_search_space: usize,
    init_dim: usize,   // Initial dimension of the subpace
    update_dim: usize, // number of vector to add to the search space
}
impl Config {
    /// Choose sensible default values for the davidson algorithm, where:
    /// * `nvalues` - Number of eigenvalue/eigenvector pairs to compute
    /// * `dim` - dimension of the matrix to diagonalize
    /// * `method` - Either DPR or GJD
    /// * `target` Lowest, highest or somewhere in the middle portion of the spectrum
    /// * `tolerance` Numerical tolerance to reach convergence
    fn new(
        nvalues: usize,
        dim: usize,
        method: DavidsonCorrection,
        target: SpectrumTarget,
        tolerance: f64,
    ) -> Self {
        let max_search_space = if nvalues * 10 < dim {
            nvalues * 10
        } else {
            dim
        };
        Config {
            method,
            spectrum_target: target,
            tolerance,
            max_iters: 100,
            max_search_space,
            init_dim: nvalues * 2,
            update_dim: nvalues * 2,
        }
    }
}
#[derive(Debug, PartialEq)]
pub struct DavidsonError;

impl fmt::Display for DavidsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Davidson Algorithm did not converge!")
    }
}

impl error::Error for DavidsonError {}

/// Structure with the configuration data
pub struct Davidson {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

impl Davidson {
    /// The new static method takes the following arguments:
    /// * `h` - A highly diagonal symmetric matrix
    /// * `nvalues` - the number of eigenvalues/eigenvectors pair to compute
    /// * `method` Either DPR or GJD
    /// * `spectrum_target` Lowest or Highest part of the spectrum
    /// * `tolerance` numerical tolerance.
    pub fn new(
        h: ArrayView2<f64>,
        nvalues: usize,
        method: DavidsonCorrection,
        spectrum_target: SpectrumTarget,
        tolerance: f64,
    ) -> Result<Self, DavidsonError> {
        let time: Instant = Instant::now();
        // Initial configuration
        let conf = Config::new(nvalues, h.nrows(), method, spectrum_target, tolerance);

        // Initial subpace
        let mut dim_sub = conf.init_dim;
        // 1.1 Select the initial orthogonal subspace
        let mut basis = Self::generate_subspace(h.diag(), &conf);
        //let mut basis: Array2<f64> = Array::random((h.nrows(), dim_sub), Uniform::new(0.0, 1.0));
        // 1.2 Select the correction to use
        let corrector = CorrectionMethod::new(h.view(), conf.method);
        println!("basis {:2.1}", basis);
        // 2. Generate subspace matrix problem by projecting into the basis
        let first_subspace: ArrayView2<f64> = basis.slice(s![.., 0..dim_sub]);
        let mut matrix_subspace: Array2<f64> = h.dot(&first_subspace);
        let mut matrix_proj: Array2<f64> = first_subspace.t().dot(&matrix_subspace);

        // Outer loop block Davidson schema
        let mut result = Err(DavidsonError);
        utils::print_davidson_init(conf.max_iters, nvalues, tolerance);
        for i in 0..conf.max_iters {
            let ord_sort = !matches!(conf.spectrum_target, SpectrumTarget::Highest);
            println!("matrix proj {}", matrix_proj);
            let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix_proj.eigh(UPLO::Upper).unwrap();
            let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = utils::sort_eigenpairs(eigenvalues, eigenvectors, ord_sort);
            println!("EIGENVALUES {} {}", i, eigenvalues);
            // 4. Check for convergence
            // 4.1 Compute the residues
            let ritz_vectors = basis.slice(s![.., 0..dim_sub]).dot(&eigenvectors.slice(s![.., 0..dim_sub]));
            let residues = Self::compute_residues(ritz_vectors.view(), matrix_subspace.view(), eigenvalues.view(), eigenvectors.view());

            // 4.2 Check Converge for each pair eigenvalue/eigenvector
            let errors: Array1<f64> = Array::from_iter(
                residues
                    .slice(s![.., 0..nvalues])
                    .axis_iter(Axis(1))
                    .map(|col| col.norm()),
            );

            let total_error: f64 = errors.norm();
            let max_error: f64 = *errors.max().unwrap();
            utils::print_davidson_iteration(i, 0, 0, total_error, max_error);

            // 4.3 Check if all eigenvalues/eigenvectors have converged
            if errors.iter().all(|&x| x < conf.tolerance) && i > 1{
                result = Ok(Self::create_results(
                    eigenvalues.view(),
                    ritz_vectors.view(),
                    nvalues,
                ));
                break;
            }

            // 5. Update subspace basis set
            // 5.1 Add the correction vectors to the current basis
            if dim_sub + conf.update_dim <= conf.max_search_space {
                let correction =
                    corrector.compute_correction(residues.view(), eigenvalues.view(), ritz_vectors.view());
                update_subspace(basis.view_mut(), correction.view(), (dim_sub, dim_sub + conf.update_dim));

                // 6. Orthogonalize the subspace
                MGS::orthonormalize(basis.view_mut(), dim_sub, dim_sub + conf.update_dim);

                // Update projected matrix
                matrix_subspace = {
                    let additional_subspace: Array2<f64> = Array::zeros((matrix_subspace.nrows(), conf.update_dim));
                    let mut tmp = stack![Axis(1), matrix_subspace, additional_subspace];
                    let new_block = h.dot(&basis.slice(s![.., dim_sub..(dim_sub+conf.update_dim)]));
                    tmp.slice_mut(s![.., dim_sub..(dim_sub+conf.update_dim)]).assign(&new_block);
                    tmp
                };

                matrix_proj = {
                    let new_dim: usize = dim_sub + conf.update_dim;
                    let new_subspace: ArrayView2<f64> = basis.slice(s![.., 0..new_dim]);
                    let new_block = new_subspace.t().dot(&matrix_subspace.slice(s![.., dim_sub..(dim_sub+conf.update_dim)]));
                    let mut tmp: Array2<f64> = Array::zeros((new_dim, new_dim));
                    tmp.slice_mut(s![0..dim_sub, 0..dim_sub]).assign(&matrix_proj);
                    tmp.slice_mut(s![.., dim_sub..]).assign(&new_block);
                    tmp.slice_mut(s![dim_sub.., ..]).assign(&new_block.t());
                    tmp
                };
                // update counter
                dim_sub += conf.update_dim;

            // 5.2 Otherwise reduce the basis of the subspace to the current
            // correction
            } else {
                dim_sub = conf.init_dim;
                basis.fill(0.0);
                update_subspace(basis.view_mut(), ritz_vectors.view(), (0, dim_sub));
                // Update projected matrix
                matrix_subspace = h.dot(&basis.slice(s![.., 0..dim_sub]));
                matrix_proj = basis.slice(s![.., 0..dim_sub]).t().dot(&matrix_subspace);
            }
            // Check number of iterations
            if i > conf.max_iters {
                break;
            }
        }
        utils::print_davidson_end(result.is_ok(), time);
        result
    }

    /// Extract the requested eigenvalues/eigenvectors pairs
    fn create_results(
        subspace_eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
        nvalues: usize,
    ) -> Davidson {
        Davidson {
            eigenvalues: subspace_eigenvalues.slice(s![0..nvalues]).to_owned(),
            eigenvectors: ritz_vectors.slice(s![.., 0..nvalues]).to_owned(),
        }
    }

    /// Residue vectors
    fn compute_residues(
        ritz_vectors: ArrayView2<f64>,
        matrix_subspace: ArrayView2<f64>,
        eigenvalues: ArrayView1<f64>,
        eigenvectors: ArrayView2<f64>,
    ) -> Array2<f64> {
        let vs: Array2<f64> = matrix_subspace.dot(&eigenvectors);
        let guess: Array2<f64> = ritz_vectors.dot(&Array::from_diag(&eigenvalues));
        vs - guess
    }

    /// Generate initial orthonormal subspace
    fn generate_subspace(diag: ArrayView1<f64>, conf: &Config) -> Array2<f64> {
        if is_sorted(diag) && conf.spectrum_target == SpectrumTarget::Lowest {
            let mut mtx: Array2<f64> = Array::eye(conf.max_search_space.clone());
            if &conf.max_search_space < &diag.len() {
                let zero_block: Array2<f64> = Array2::zeros((diag.len() - conf.max_search_space.clone()
                                                             , conf.max_search_space.clone()));
                mtx = stack![Axis(0), mtx, zero_block];
            }
            mtx
        } else {
            let order: Vec<usize> = utils::argsort(diag.view());
            println!("ORDER {:?}", order);
            let mut mtx: Array2<f64> = Array2::zeros([diag.len(), conf.max_search_space]);
            for (idx, i) in order.into_iter().enumerate() {
                if idx < conf.max_search_space {
                    mtx[[i, idx]] = 1.0;
                }
            }
            mtx
        }
    }
}

/// Structure containing the correction methods
struct CorrectionMethod<'a>
{
    /// The initial target matrix
    target: ArrayView2<'a, f64>,
    /// Method used to compute the correction
    method: DavidsonCorrection,
}

impl<'a> CorrectionMethod<'a>
{
    fn new(target: ArrayView2<'a, f64>, method: DavidsonCorrection) -> Self {
        Self { target, method }
    }

    /// compute the correction vectors using either DPR or GJD
    fn compute_correction(
        &self,
        residues: ArrayView2<f64>,
        eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
    ) -> Array2<f64> {
        match self.method {
            DavidsonCorrection::DPR => self.compute_dpr_correction(residues, eigenvalues),
            DavidsonCorrection::GJD => {
                self.compute_gjd_correction(residues, eigenvalues, ritz_vectors)
            }
        }
    }

    /// Use the Diagonal-Preconditioned-Residue (DPR) method to compute the correction
    fn compute_dpr_correction(
        &self,
        residues: ArrayView2<f64>,
        eigenvalues: ArrayView1<f64>,
    ) -> Array2<f64> {
        let d = self.target.diag();
        let mut correction: Array2<f64> = Array::zeros((self.target.nrows(), residues.ncols()));
        for (k, lambda) in eigenvalues.iter().enumerate() {
            let tmp: Array1<f64> = Array::from_elem(self.target.nrows(), *lambda) - &d;
            let rs = &residues.column(k) / &tmp;
            correction.slice_mut(s![.., k]).assign(&rs);
        }
        correction
    }

    /// Use the Generalized Jacobi Davidson (GJD) to compute the correction
    fn compute_gjd_correction(
        &self,
        residues: ArrayView2<f64>,
        eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
    ) -> Array2<f64> {
        let dimx: usize = self.target.nrows();
        let dimy: usize = residues.ncols();
        let id: Array2<f64> = Array2::eye(dimx);
        let ones: Array1<f64> = Array1::ones(dimx);
        let mut correction: Array2<f64> = Array2::zeros((dimx, dimy));
        let diag = self.target.diag();
        for (k, r) in ritz_vectors.axis_iter(Axis(1)).enumerate() {
            // Create the components of the linear system
            let t1 = &id - r.dot(&r.t());
            let mut t2 = self.target.clone().to_owned();
            let val = &diag - &(eigenvalues[k] * &ones);
            t2.diag_mut().assign(&val);
            let arr = t1.dot(&t2.dot(&t1.slice(s![0..dimx, ..])));
            // Solve the linear system
            let decomp = arr.factorize_into().unwrap();
            let b = -1.0 * &residues.column(k);
            let x = decomp.solve_into(b).unwrap();
            correction.slice_mut(s![.., k]).assign(&x);
        }
        correction
    }
}

/// Update the subpace with new vectors
fn update_subspace(mut basis: ArrayViewMut2<f64>, vectors: ArrayView2<f64>, range: (usize, usize)) {
    let (start, end): (usize, usize) = range;
    basis.slice_mut(s![.., start..end]).assign(&vectors.slice(s![.., 0..end - start]));
}

fn sort_diagonal(rs: &mut Vec<f64>, conf: &Config) {
    match conf.spectrum_target {
        SpectrumTarget::Lowest => utils::sort_vector(rs, true),
        SpectrumTarget::Highest => utils::sort_vector(rs, false),
        _ => panic!("Not implemented error!"),
    }
}

/// Check if a vector is sorted in ascending order
fn is_sorted(xs: ArrayView1<f64>) -> bool {
    for k in 1..xs.len() {
        if xs[k] < xs[k - 1] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;

    #[test]
    fn test_update_subspace() {
        let mut arr = Array2::from_elem((3, 3), 1.);
        let brr = Array::zeros([3, 2]);
        super::update_subspace( arr.view_mut(), brr.view(), (0, 2));
        assert_eq!(arr.column(1).sum(), 0.);
        assert_eq!(arr.column(2).sum(), 3.);
    }
}
