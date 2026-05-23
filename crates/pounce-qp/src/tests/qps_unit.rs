//! QPS reader tests — parse → build QpProblem → solve →
//! verify the recovered solution matches the closed-form answer.
//! The on-ramp for §8.1 Maros-Mészáros benchmarking.

use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::qps::parse_qps;
use crate::solver::{ParametricActiveSetSolver, QpSolver};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()))
}

#[test]
fn parse_qps_recovers_basic_metadata_for_tiny_qp() {
    // A 2-variable, 1-constraint convex QP in QPS form:
    //   min ½(x₁² + x₂²) − x₁ − 2x₂   s.t.  x₁ + x₂ ≤ 1,
    //                                       0 ≤ x_i ≤ 1
    let text = "\
NAME          TINY
ROWS
 N  COST
 L  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
RHS
    RHS       C1         1.0
BOUNDS
 LO BND       X1         0.0
 UP BND       X1         1.0
 LO BND       X2         0.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";

    let model = parse_qps(text).unwrap();
    assert_eq!(model.name, "TINY");
    assert_eq!(model.n, 2);
    assert_eq!(model.m, 1);
    assert_eq!(model.var_names, vec!["X1", "X2"]);
    assert_eq!(model.row_names, vec!["C1"]);
    assert_eq!(model.g, vec![-1.0, -2.0]);
    assert_eq!(model.bl, vec![pounce_common::types::NLP_LOWER_BOUND_INF]);
    assert_eq!(model.bu, vec![1.0]);
    assert_eq!(model.xl, vec![0.0, 0.0]);
    assert_eq!(model.xu, vec![1.0, 1.0]);
    assert_eq!(model.h_irow.len(), 2);
    assert_eq!(model.a_irow.len(), 2);
}

#[test]
fn parse_qps_round_trip_solves_to_expected_optimum() {
    // Same TINY problem as above.
    //
    // Closed form:
    //   Unconstrained min: x = (1, 2). Violates x ≤ 1 and the
    //   sum constraint. Bounds clamp x₂ = 1; x₁ = 1 clamps to xu.
    //   Sum: x₁ + x₂ = 2 > 1. So the sum constraint binds.
    //   Reduce to x₁ + x₂ = 1 with 0 ≤ x_i ≤ 1. Substitute
    //   x₂ = 1 − x₁, minimize over x₁ ∈ [0, 1]:
    //     f(x₁) = ½(x₁² + (1 − x₁)²) − x₁ − 2(1 − x₁)
    //           = x₁² − x₁ + ½ + x₁ − 2 = x₁² − 1.5
    //   Minimum at x₁ = 0, x₂ = 1. f = -1.5.
    let text = "\
NAME          TINY
ROWS
 N  COST
 L  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
RHS
    RHS       C1         1.0
BOUNDS
 LO BND       X1         0.0
 UP BND       X1         1.0
 LO BND       X2         0.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";

    let model = parse_qps(text).unwrap();

    // Wrap the parsed model in pounce-linalg sparse types so the
    // existing solver can consume it. SymTMatrixSpace and
    // GenTMatrixSpace own their indices via `Rc`; the values
    // arrays are set in place after construction.
    let h_space = SymTMatrixSpace::new(model.n as i32, model.h_irow.clone(), model.h_jcol.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&model.h_val);

    let a_space = GenTMatrixSpace::new(
        model.m as i32,
        model.n as i32,
        model.a_irow.clone(),
        model.a_jcol.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&model.a_val);

    let qp = QpProblem {
        n: model.n,
        m: model.m,
        h: &h,
        g: &model.g,
        a: &a,
        bl: &model.bl,
        bu: &model.bu,
        xl: &model.xl,
        xu: &model.xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    // The optimum (0, 1) lies at a corner: x₁ = xl_1 = 0,
    // x₂ = xu_2 = 1, and the general constraint binds.
    assert!((sol.x[0] - 0.0).abs() < 1e-6, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", sol.x[1]);
    assert!(
        (sol.obj - (-1.5)).abs() < 1e-6,
        "obj = {} but expected -1.5",
        sol.obj
    );
}

#[test]
fn parse_qps_rejects_ranges_section() {
    let text = "\
NAME          T
ROWS
 N  C
 L  R1
COLUMNS
    X1        C          1.0   R1         1.0
RHS
    RHS       R1         5.0
RANGES
    RNG       R1         3.0
ENDATA
";
    let err = parse_qps(text).unwrap_err();
    assert!(
        err.contains("RANGES"),
        "expected error about RANGES but got: {err}"
    );
}

#[test]
fn parse_qps_rejects_mip_markers() {
    let text = "\
NAME          T
ROWS
 N  C
COLUMNS
    MARKER1   'MARKER'                   'INTORG'
    X1        C          1.0
    MARKER2   'MARKER'                   'INTEND'
ENDATA
";
    let err = parse_qps(text).unwrap_err();
    assert!(
        err.contains("MIP") || err.contains("MARKER"),
        "expected MIP/MARKER rejection but got: {err}"
    );
}

#[test]
fn parse_qps_rejects_binary_variable_bound() {
    let text = "\
NAME          T
ROWS
 N  C
COLUMNS
    X1        C          1.0
BOUNDS
 BV BND       X1
ENDATA
";
    let err = parse_qps(text).unwrap_err();
    assert!(err.contains("BV"), "expected BV rejection but got: {err}");
}
