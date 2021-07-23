use eigenvalues::{Davidson, utils};
use eigenvalues::utils::{generate_random_sparse_symmetric};
use ndarray_linalg::*;
use ndarray::prelude::*;
use std::time::Instant;
use log::LevelFilter;
use std::io::Write;
use env_logger::Builder;

fn main() {
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, LevelFilter::Info)
        .init();

    let matrix = generate_random_sparse_symmetric(2000, 10, 0.005);

    let tolerance = 1e-6;
    let eigh_start = Instant::now();

    let (u, v): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    println!("Elapsed time for eigh routine {}", eigh_start.elapsed().as_secs_f32());

    // Compute the first 5 lowest eigenvalues/eigenvectors using the DPR method
    let dav_start = Instant::now();

    let guess: Array2<f64> = make_guess(matrix.diag(), 6);
    let n_roots: usize = 4;
    let dav = Davidson::new(matrix, guess, n_roots, tolerance, 60).unwrap();

    println!("Elapsed time for Davidson routine {}", dav_start.elapsed().as_secs_f32());

    println!("Computed eigenvalues:\n{}", dav.eigenvalues);
    println!("Expected eigenvalues:\n{}", u.slice(s![0..n_roots]));

    println!("The scalar product of the first eigenvector :{:8.4e}", v.column(0).dot(&dav.eigenvectors.column(0)));


}

pub fn make_guess(diag: ArrayView1<f64>, dim: usize) -> Array2<f64> {
    let order: Vec<usize> = utils::argsort(diag.view());
    let mut mtx: Array2<f64> = Array2::zeros([diag.len(), dim]);
    for (idx, i) in order.into_iter().enumerate() {
        if idx < dim {
            mtx[[i, idx]] = 1.0;
        }
    }
    mtx
}

