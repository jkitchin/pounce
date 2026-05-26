//! §8.1 published-optimum comparison framework — Maros-Mészáros-
//! flavoured QPS fixtures with hand-derived closed-form optima.
//!
//! The actual Maros-Mészáros .qps set (138 problems, sizes
//! `n ∈ [2, 12955]`) is gated on external oracle distribution
//! and pounce-cutest-style FFI to qpOASES / OSQP for tolerance
//! cross-checking. This file ships the **framework** — a small
//! `compare_qps_to_published(text, x_star, f_star, …)` helper
//! plus five hand-crafted .qps fixtures covering the problem
//! shapes that appear across the MM set:
//!
//!   1. Pure-equality QP (LP-like; classic MM `QPCBOEI1` shape).
//!   2. Box-constrained quadratic with no general constraints.
//!   3. Inequality + equality mix (canonical KKT exercise).
//!   4. Two-sided ineq via RANGES (MM `QPCBOEI2` shape).
//!   5. Indefinite Hessian with PD reduced (exercises §4.5
//!      inertia control through the .qps pipeline).
//!
//! Each fixture's optimum was hand-derived from the KKT system;
//! see the per-test docstring for the calculation. When the real
//! MM set is wired in (Phase 5a follow-up requiring the .qps
//! distribution), the helper here is the call-site every entry
//! drops into.

use pounce_common::types::Number;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::qps::parse_qps;
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

/// Parse a .qps string, solve, and assert the result matches the
/// supplied closed-form optimum. The §8.1 framework entry point.
///
/// - `text`: a complete MM-style .qps record (`NAME` line through
///   `ENDATA`).
/// - `x_star`, `f_star`: published or hand-derived optima.
/// - `x_tol`, `f_tol`: absolute tolerances. Pick scale-appropriate.
/// - `inertia`: `Psd` for convex problems, `Indefinite` to exercise
///   §4.5 inertia control.
fn compare_qps_to_published(
    text: &str,
    x_star: &[Number],
    f_star: Number,
    x_tol: Number,
    f_tol: Number,
    inertia: HessianInertia,
) {
    let model = parse_qps(text).expect("qps parse");
    let n = model.n;
    let m = model.m;
    let h_space = SymTMatrixSpace::new(n as i32, model.h_irow.clone(), model.h_jcol.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&model.h_val);
    let a_space = GenTMatrixSpace::new(
        m as i32,
        n as i32,
        model.a_irow.clone(),
        model.a_jcol.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&model.a_val);
    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &model.g,
        a: &a,
        bl: &model.bl,
        bu: &model.bu,
        xl: &model.xl,
        xu: &model.xu,
        hessian_inertia: inertia,
    };
    let mut solver = new_solver();
    let sol = solver
        .solve(&qp, None, &QpOptions::default())
        .expect("qp solve");
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.x.len(), x_star.len());
    for (i, (&got, &want)) in sol.x.iter().zip(x_star.iter()).enumerate() {
        assert!(
            (got - want).abs() < x_tol,
            "x[{i}] = {got}, want {want} (|err| = {})",
            (got - want).abs(),
        );
    }
    assert!(
        (sol.obj - f_star).abs() < f_tol,
        "f* = {}, want {f_star} (|err| = {})",
        sol.obj,
        (sol.obj - f_star).abs(),
    );
}

// ─────────────────────────────────────────────────────────────
// #1. Pure-equality QP — n=3, m=1.
//
//   min ½(x₁² + x₂² + x₃²) − x₁ − 2x₂ + 3x₃
//   s.t. x₁ + x₂ + x₃ = 1
//
// KKT: x_i − g_i − λ = 0 for each i.
//   x₁ = 1 + λ,  x₂ = 2 + λ,  x₃ = −3 + λ.
//   sum = 0 + 3λ = 1 ⇒ λ = 1/3.
//   x* = (4/3, 7/3, −8/3), f* = computed numerically below.
// ─────────────────────────────────────────────────────────────

#[test]
fn mm_pure_equality_n3_m1() {
    let text = "\
NAME          PURE_EQ
ROWS
 N  COST
 E  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
    X3        COST        3.0   C1         1.0
RHS
    RHS       C1         1.0
BOUNDS
 FR BND       X1
 FR BND       X2
 FR BND       X3
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
    X3        X3         1.0
ENDATA
";
    // x* = (4/3, 7/3, -8/3).
    let x_star = [4.0 / 3.0, 7.0 / 3.0, -8.0 / 3.0];
    // f* = ½(x₁² + x₂² + x₃²) − x₁ − 2x₂ + 3x₃
    //    = ½(16/9 + 49/9 + 64/9) − 4/3 − 14/3 − 24/3
    //    = ½ · 129/9 − 42/3
    //    = 129/18 − 252/18
    //    = -123/18 = -41/6.
    let f_star = -41.0 / 6.0;
    compare_qps_to_published(text, &x_star, f_star, 1e-7, 1e-7, HessianInertia::Psd);
}

// ─────────────────────────────────────────────────────────────
// #2. Box-constrained QP — n=2, m=0.
//
//   min ½((x₁-2)² + (x₂+1)²)
//   s.t. -1 ≤ x_i ≤ 1
//
// Unconstrained min: x = (2, -1). Project onto the box:
//   x₁ clamps to upper bound 1; x₂ already at lower bound -1.
//   x* = (1, -1), f* = ½(1 + 0) = 0.5.
// Encoding: g = (-2, +1), QUADOBJ = diag(1, 1), constant = ½(4+1) = 2.5.
// The QPS QUADOBJ doesn't carry a constant, so we encode g and
// the Hessian; f_qps = ½xᵀHx + gᵀx + 2.5 (the constant from
// expanding the squares). pounce-qp's obj only includes
// `½xᵀHx + gᵀx` so f*_pqp = 0.5 − 2.5 = -2.
// ─────────────────────────────────────────────────────────────

#[test]
fn mm_box_constrained_n2_m0() {
    let text = "\
NAME          BOX
ROWS
 N  COST
COLUMNS
    X1        COST       -2.0
    X2        COST        1.0
BOUNDS
 LO BND       X1        -1.0
 UP BND       X1         1.0
 LO BND       X2        -1.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";
    let x_star = [1.0, -1.0];
    let f_star = -2.0;
    compare_qps_to_published(text, &x_star, f_star, 1e-7, 1e-7, HessianInertia::Psd);
}

// ─────────────────────────────────────────────────────────────
// #3. Mixed inequality + box, with a binding inequality.
// n=2, m=1.
//
//   min ½(x₁² + x₂²) − x₁ − 2x₂
//   s.t. x₁ + x₂ ≤ 1,  x ≥ 0
//
// Unconstrained min: x = (1, 2), sum = 3 > 1 ⇒ ineq binds.
// At sum = 1: KKT x₁ - 1 + λ = 0, x₂ - 2 + λ = 0, x₁+x₂=1.
// (1-λ) + (2-λ) = 1 ⇒ λ = 1. x* = (0, 1). λ ≥ 0 ✓; x ≥ 0 ✓.
// f* = ½(0 + 1) − 0 − 2 = -1.5.
// ─────────────────────────────────────────────────────────────

#[test]
fn mm_mixed_inequality_n2_m1() {
    let text = "\
NAME          MIXED
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
 LO BND       X2         0.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";
    let x_star = [0.0, 1.0];
    let f_star = -1.5;
    compare_qps_to_published(text, &x_star, f_star, 1e-7, 1e-7, HessianInertia::Psd);
}

// ─────────────────────────────────────────────────────────────
// #4. Two-sided inequality via RANGES — n=2, m=1.
// Same problem as #3 but expressed as -∞ ≤ x₁+x₂ ≤ 1, encoded
// via RANGES on an E row to give bl = -∞, bu = 1. (We can't use
// RANGES on E to encode pure ≤; instead use L with a range to
// get bl_constrained, bu = rhs.)
//
// Actually RANGES on L row `aᵀx ≤ b` adds a lower bound:
//    bl = b - |range|, bu = b.
// With b = 1, range = 100, we get -99 ≤ x₁+x₂ ≤ 1.
//
// Same KKT as #3 (only the upper side binds since x ≥ 0 keeps
// x₁+x₂ ≥ 0).
// ─────────────────────────────────────────────────────────────

#[test]
fn mm_ranges_l_row_two_sided() {
    let text = "\
NAME          RANGES
ROWS
 N  COST
 L  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
RHS
    RHS       C1         1.0
RANGES
    RNG       C1         100.0
BOUNDS
 LO BND       X1         0.0
 LO BND       X2         0.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";
    let x_star = [0.0, 1.0];
    let f_star = -1.5;
    compare_qps_to_published(text, &x_star, f_star, 1e-7, 1e-7, HessianInertia::Psd);
}

// ─────────────────────────────────────────────────────────────
// #5. Indefinite Hessian, PD reduced — n=2, m=1.
//
//   min ½ x_1² − ½ x_2²  + 0·x_2     (i.e. H = diag(1, -1))
//   s.t. x_1 + x_2 = 1
//        free variables
//
// The unconstrained Lagrangian L = ½ x_1² − ½ x_2² + λ(x_1+x_2 - 1).
// KKT: x_1 + λ = 0; -x_2 + λ = 0; x_1+x_2 = 1.
// ⇒ λ = -x_1 = x_2, so x_2 = -x_1, x_1+x_2 = 0. Contradiction.
// → Problem is unbounded below along x_2 → -∞ once x_1+x_2 = 1
//    is satisfied: x_1 = 1 - x_2, f = ½(1-x_2)² - ½ x_2² = ½ - x_2.
//
// To make it well-posed, add x_2 ≤ 2 ⇒ upper-bound active.
// Then x_2 = 2, x_1 = -1, f = ½ - 2 = -1.5.
// Reduced Hessian on the null space of (1,1) (direction (1,-1)):
//    z·H·zᵀ = 1·1 - (-1)·(-1) = 0. Marginal — let's instead use
//    H = diag(1, -2) (positive reduced):
//    z·H·zᵀ on (1, -1): 1·1 + 1·(-2) = -1. Still indefinite reduced.
//
// Let's use H = diag(2, -1). Reduced on (1,-1):
//    2 + (-1)·1 = 1 > 0. Good.
//
//   min x_1² − ½ x_2² + 0·x_2
//   s.t. x_1 + x_2 = 1,  x_2 ≤ 2,  x_1 free
//
// Lagrangian: 2 x_1 + λ = 0;  -x_2 + λ - μ = 0 (with μ ≥ 0); x_1+x_2=1; x_2 ≤ 2.
// Case μ = 0: λ = x_2 = -2 x_1, so x_1 = -x_2/2. Then x_1+x_2 = x_2/2 = 1
//   ⇒ x_2 = 2, x_1 = -1. Check x_2 ≤ 2 with equality — degenerate
//   between cases. Pick the bound to be loose (x_u = 3):
//
//   x_u = 3: μ = 0 case yields x_2 = 2, x_1 = -1. x_2 < 3 ✓.
//   f = 1 + 0·(-1) - ½·4 + 0 = 1 - 2 = -1.
// ─────────────────────────────────────────────────────────────

#[test]
fn mm_indefinite_hessian_pd_reduced() {
    let text = "\
NAME          INDEFH
ROWS
 N  COST
 E  C1
COLUMNS
    X1        C1         1.0
    X2        C1         1.0
RHS
    RHS       C1         1.0
BOUNDS
 FR BND       X1
 MI BND       X2
 UP BND       X2         3.0
QUADOBJ
    X1        X1         2.0
    X2        X2        -1.0
ENDATA
";
    let x_star = [-1.0, 2.0];
    let f_star = -1.0;
    // Indefinite Hessian — exercises §4.5 inertia control.
    compare_qps_to_published(
        text,
        &x_star,
        f_star,
        1e-7,
        1e-7,
        HessianInertia::Indefinite,
    );
}
