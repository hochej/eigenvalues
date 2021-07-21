use eigenvalues::algorithms::davidson::Davidson;
use eigenvalues::utils::generate_diagonal_dominant;
use eigenvalues::utils::{sort_eigenpairs, test_eigenpairs};
use eigenvalues::{DavidsonCorrection, SpectrumTarget};
use ndarray_linalg::*;
use ndarray::prelude::*;

#[test]
fn test_davidson_lowest() {
    let arr = generate_diagonal_dominant(200, 0.005);
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = arr.eigh(UPLO::Upper).unwrap();
    let eig: (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, true);
    let spectrum_target = SpectrumTarget::Lowest;
    let tolerance = 1.0e-4;

    println!("running DPR");
    let dav = Davidson::new(
        arr.view(),
        2,
        DavidsonCorrection::DPR,
        spectrum_target.clone(),
        tolerance,
    )
    .unwrap();
    println!("Davidson eigenvalues {}", dav.eigenvalues);
    println!("Eigh eigenvalues {}", eig.0);
    test_eigenpairs(eig.clone(), (dav.eigenvalues, dav.eigenvectors), 2);
    println!("running GJD");
    let dav = Davidson::new(
        arr.view(),
        10,
        DavidsonCorrection::GJD,
        spectrum_target,
        tolerance,
    )
    .unwrap();
    println!("Davidson eigenvalues {}", dav.eigenvalues);
    println!("Eigh eigenvalues {}", eig.0);
    test_eigenpairs(eig, (dav.eigenvalues, dav.eigenvectors), 2);
}

#[test]
fn test_davidson_unsorted() {
    // Test the algorithm when the diagonal is unsorted
    let mut arr = generate_diagonal_dominant(100, 0.005);
    let tolerance = 1.0e-6;
    let vs: Array1<f64> = array![3.0, 2.0, 4.0, 1.0, 5.0, 6.0, 7.0, 8.0];
    arr.diag_mut().assign(&vs);
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = arr.eigh(UPLO::Upper).unwrap();
    let eig: (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, true);
    let dav = Davidson::new(
        arr.view(),
        1,
        DavidsonCorrection::DPR,
        SpectrumTarget::Lowest,
        tolerance,
    )
    .unwrap();
    test_eigenpairs(eig, (dav.eigenvalues, dav.eigenvectors), 1);
}

#[test]
fn test_davidson_highest() {
    // Test the compution of the highest eigenvalues
    let dim = 200;
    let nvalues = 2;
    let tolerance = 1.0e-4;
    let arr = generate_diagonal_dominant(dim, 0.005);
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = arr.eigh(UPLO::Upper).unwrap();
    let eig: (Array1<f64>, Array2<f64>) = sort_eigenpairs(eigenvalues, eigenvectors, false);

    let target = SpectrumTarget::Highest;
    println!("running DPR");
    let dav = Davidson::new(
        arr.view(),
        nvalues,
        DavidsonCorrection::DPR,
        target.clone(),
        tolerance,
    )
    .unwrap();
    //test_eigenpairs(eig.clone(), (dav.eigenvalues, dav.eigenvectors), nvalues);
    println!("running GJD");
    let dav = Davidson::new(
        arr.view(),
        nvalues,
        DavidsonCorrection::GJD,
        target,
        tolerance,
    )
    .unwrap();
    test_eigenpairs(eig, (dav.eigenvalues, dav.eigenvectors), nvalues);
}
