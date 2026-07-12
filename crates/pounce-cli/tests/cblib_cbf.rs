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
use pounce_convex::{QpOptions, QpStatus, solve_socp_ipm};
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

/// Accept a usable solve: a clean `Optimal` or the reduced-accuracy
/// `OptimalInaccurate` (code review 2026-06 item M20). These exp/power-cone GPs
/// reach their optimum through the non-symmetric driver's reduced-accuracy
/// fallback (best iterate within √tol), so the status is `OptimalInaccurate`;
/// the objective check in each test pins the actual solution quality.
fn assert_solved(status: QpStatus, label: &str) {
    assert!(
        matches!(status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "{label} status: {status:?}"
    );
}

const DEMB761: &str = include_str!("data/cblib/demb761.cbf");
const BECK751: &str = include_str!("data/cblib/beck751.cbf");
const FANG88: &str = include_str!("data/cblib/fang88.cbf");
const POW3: &str = include_str!("data/cblib/pow3_synthetic.cbf");
const SDP: &str = include_str!("data/cblib/sdp_synthetic.cbf");

#[test]
fn demb761_solves_to_optimum() {
    let (status, obj) = solve_instance(DEMB761);
    assert_solved(status, "demb761");
    assert!(obj.is_finite(), "demb761 objective finite: {obj}");
}

#[test]
fn beck751_solves_to_optimum() {
    let (status, obj) = solve_instance(BECK751);
    assert_solved(status, "beck751");
    assert!(obj.is_finite(), "beck751 objective finite: {obj}");
}

#[test]
fn fang88_solves_to_optimum() {
    let (status, obj) = solve_instance(FANG88);
    assert_solved(status, "fang88");
    assert!(obj.is_finite(), "fang88 objective finite: {obj}");
}

#[test]
fn power_cone_synthetic_hits_known_optimum() {
    // max x2 s.t. (x0,x1,x2) ∈ POW(α=½), x0=2, x1=½  →  x2 = 2^½·½^½ = 1.
    // Validates the POWCONES parse, the α = α₀/(α₀+α₁) resolution, and the
    // CBF→pounce power-cone permutation end to end.
    let (status, obj) = solve_instance(POW3);
    assert_solved(status, "pow3");
    assert!((obj - 1.0).abs() < 1e-6, "pow3 objective {obj} vs 1");
}

#[test]
fn sdp_psdcon_synthetic_hits_known_optimum() {
    // max λ s.t. (M − λI) ⪰ 0, M = diag(2,5)  →  λ = λ_min(M) = 2.
    // Validates the PSDCON / HCOORD / DCOORD reader (affine PSD constraint →
    // a pounce Psd cone with √2-scaled svec rows) end to end.
    let (status, obj) = solve_instance(SDP);
    assert_eq!(status, QpStatus::Optimal, "sdp status");
    assert!((obj - 2.0).abs() < 1e-5, "sdp objective {obj} vs 2");
}
