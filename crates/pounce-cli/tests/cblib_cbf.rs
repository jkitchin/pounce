//! CBLIB exponential-cone benchmark tier: parse real `.cbf` instances from
//! the Conic Benchmark Library, map them to a pounce conic program, and solve
//! them through the non-symmetric (exp-cone) HSDE driver.
//!
//! These are the literal geometric-program instances from the source papers
//! (Demberg `demb761`, Beck `beck751`, Fang `fang88`), the gold-standard
//! broad validation called for in `dev-notes/hsde.md`. Published reference
//! objectives are unavailable (the CBLIB solution files 404), so correctness
//! is cross-checked against an independent smooth NLP in `cblib_vs_nlp.rs`;
//! this file checks that the parse → map → solve pipeline reaches a verified
//! optimum on each instance.

use pounce_cli::cbf;
use pounce_convex::{solve_socp_ipm, QpOptions, QpStatus};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parse, map, and solve a CBLIB instance; return `(status, cbf_objective)`.
fn solve_instance(text: &str) -> (QpStatus, f64) {
    let model = cbf::parse(text).expect("parse CBF");
    let cp = model.to_conic().expect("map to conic");
    let opts = QpOptions {
        max_iter: 500,
        ..QpOptions::default()
    };
    let sol = solve_socp_ipm(&cp.prob, &cp.cones, &opts, backend);
    let obj = cp.cbf_objective(sol.obj, model.minimize);
    (sol.status, obj)
}

const DEMB761: &str = include_str!("data/cblib/demb761.cbf");
const BECK751: &str = include_str!("data/cblib/beck751.cbf");
const FANG88: &str = include_str!("data/cblib/fang88.cbf");

#[test]
fn demb761_solves_to_optimum() {
    let (status, obj) = solve_instance(DEMB761);
    assert_eq!(status, QpStatus::Optimal, "demb761 status");
    assert!(obj.is_finite(), "demb761 objective finite: {obj}");
}

#[test]
fn beck751_solves_to_optimum() {
    let (status, obj) = solve_instance(BECK751);
    assert_eq!(status, QpStatus::Optimal, "beck751 status");
    assert!(obj.is_finite(), "beck751 objective finite: {obj}");
}

#[test]
fn fang88_solves_to_optimum() {
    let (status, obj) = solve_instance(FANG88);
    assert_eq!(status, QpStatus::Optimal, "fang88 status");
    assert!(obj.is_finite(), "fang88 objective finite: {obj}");
}
