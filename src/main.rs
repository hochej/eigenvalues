
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs};
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

    let matrix = generate_random_sparse_symmetric(1000, 5, 0.05);
    println!("{:5.3}", matrix.slice(s![..10, ..10]));
    let tolerance = 1e-6;
    let eigh_start = Instant::now();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, true);
    println!("EIGH {}", eigh_start.elapsed().as_secs_f32());
    let spectrum_target = SpectrumTarget::Lowest;
    // Compute the first 5 lowest eigenvalues/eigenvectors using the DPR method
    let dav_start = Instant::now();
    let dav = Davidson::new(
        matrix.view(), 8, DavidsonCorrection::DPR, spectrum_target, tolerance).unwrap();
    println!("DAVIDSON {}", dav_start.elapsed().as_secs_f32());
    println!("Computed eigenvalues:\n{}", dav.eigenvalues);
    println!("Expected eigenvalues:\n{}", eigenvalues.slice(s![0..5]));
    let x = eigenvectors.column(0);
    let y = dav.eigenvectors.column(0);
    println!("parallel:{}", x.dot(&y));
}
