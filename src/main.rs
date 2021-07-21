
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};
use eigenvalues::utils::{generate_random_sparse_symmetric, sort_eigenpairs};
use ndarray_linalg::*;
use ndarray::prelude::*;
use std::time::Instant;
use log::LevelFilter;
use std::io::Write;
use env_logger::Builder;
use ndarray_linalg::lobpcg::LobpcgResult;

fn main() {
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, LevelFilter::Info)
        .init();

    let matrix = generate_random_sparse_symmetric(2000, 10, 0.0005);
    println!("{:5.3}", matrix.slice(s![.., ..]));
    let tolerance = 1e-6;
    let eigh_start = Instant::now();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = matrix.eigh(UPLO::Upper).unwrap();
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, true);
    println!("EIGH {}", eigh_start.elapsed().as_secs_f32());
    let spectrum_target = SpectrumTarget::Lowest;
    // Compute the first 5 lowest eigenvalues/eigenvectors using the DPR method
    let dav_start = Instant::now();
    let dav = Davidson::new(
        matrix.view(), 1, DavidsonCorrection::DPR, spectrum_target, tolerance).unwrap();
    println!("DAVIDSON {}", dav_start.elapsed().as_secs_f32());
    println!("Computed eigenvalues:\n{}", dav.eigenvalues);
    println!("Expected eigenvalues:\n{}", eigenvalues.slice(s![0..5]));
    let x = eigenvectors.column(0);
    let y = dav.eigenvectors.column(0);
    println!("parallel:{}", x.dot(&y));
    let lobpcg_start = Instant::now();
    do_lobpcg(matrix.view());
    println!("LOBPCG {}", lobpcg_start.elapsed().as_secs_f32());

}


fn do_lobpcg(matrix: ArrayView2<f64>) -> () {
    let x:Array2<f64> = ndarray_linalg::generate::random((matrix.dim().0,8));
    let result = lobpcg::lobpcg(|y| matrix.dot(&y),x,|_| {},None,1e-9,600,lobpcg::TruncatedOrder::Smallest);
    match result {
        LobpcgResult::Ok(vals, _, r_norms) | LobpcgResult::Err(vals, _, r_norms, _) => {
            // check convergence
            for (i, norm) in r_norms.into_iter().enumerate() {
                if norm > 1e-5 {
                    println!("==== Assertion Failed ====");
                    println!("The {}th eigenvalue estimation did not converge!", i);
                    panic!("Too large deviation of residual norm: {} > 0.01", norm);
                }
            }
            println!("eigenvalues {}",vals.mapv(f64::sqrt));
        }
        LobpcgResult::NoResult(err) => panic!("Did not converge: {:?}", err),
    }
}