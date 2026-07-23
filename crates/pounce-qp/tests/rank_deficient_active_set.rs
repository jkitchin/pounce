//! Linear-independence guard: the active-set engine must pin and solve an
//! active set whose constraint normals are **rank-deficient** — redundant
//! equality rows, the degenerate shape a pure interior-point method hands the
//! LP-crossover bridge on the NETLIB GEN family.
//!
//! Before the guard landed, both the cold-start equality factor
//! (`cold_general_initial`) and the warm-start hint factor
//! (`solve_with_working_set`) assembled the full pinned KKT in one shot; a
//! rank-deficient constraint block makes that saddle matrix singular, and the
//! §4.5 inertia control only shifts the H block, so it could never rescue a
//! rank-deficient *constraint* block — the solve returned `Err`. The guard
//! prunes the active set to a maximal linearly-independent subset (the dropped
//! rows are linear combinations of the kept ones, hence satisfied at any
//! consistent point) and retries, so these solves now reach the exact vertex.
//!
//! Fixture (shared): minimize  −x₁ − 2x₂ − 3x₃  on the plane
//!   x₁ + x₂ + x₃ = 3            (row 1)
//!   2x₁ + 2x₂ + 2x₃ = 6         (row 2 — exactly 2× row 1, redundant: rank 1)
//! with x ≥ 0. Pushing mass onto the richest coefficient gives the unique
//! optimal vertex x* = (0, 0, 3), f* = −9.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_qp::{
    AntiCyclingChoice, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver,
    QpStatus,
};
use std::rc::Rc;

const NEG_INF: f64 = -1e20;
const POS_INF: f64 = 1e20;

fn opts() -> QpOptions {
    QpOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: 1000,
        ..QpOptions::default()
    }
}

/// Warm path: `solve_with_working_set` is handed a hint that pins BOTH
/// (redundant) equality rows plus the two binding lower bounds — a
/// rank-deficient pinned set. The guard prunes the redundant equality and
/// recovers the exact vertex.
#[test]
fn warm_start_prunes_redundant_equality_rows() {
    let h_space = SymTMatrixSpace::new(3, Vec::new(), Vec::new());
    let h = SymTMatrix::new(Rc::clone(&h_space));

    // Two equality rows over 3 variables; row 2 = 2 × row 1 (rank 1).
    let a_space = GenTMatrixSpace::new(2, 3, vec![1, 1, 1, 2, 2, 2], vec![1, 2, 3, 1, 2, 3]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

    let g = [-1.0, -2.0, -3.0];
    let bl = [3.0, 6.0];
    let bu = [3.0, 6.0];
    let xl = [0.0, 0.0, 0.0];
    let xu = [POS_INF; 3];

    let qp = QpProblem {
        n: 3,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    // Hint: both equalities active (redundant!) + x₁, x₂ at their lower bound.
    let working = WorkingSet {
        constraints: vec![ConsStatus::Equality, ConsStatus::Equality],
        bounds: vec![
            BoundStatus::AtLower,
            BoundStatus::AtLower,
            BoundStatus::Inactive,
        ],
    };

    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver
        .solve_with_working_set(&qp, &working, &opts())
        .expect("rank-deficient warm start must be repaired, not error");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0]).abs() < 1e-8, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1]).abs() < 1e-8, "x[1] = {}", sol.x[1]);
    assert!((sol.x[2] - 3.0).abs() < 1e-8, "x[2] = {}", sol.x[2]);
    assert!((sol.obj - (-9.0)).abs() < 1e-8, "obj = {}", sol.obj);
    // The dropped redundant equality is still satisfied: 2·Σx = 6.
    let sum = sol.x.iter().sum::<f64>();
    assert!(
        (2.0 * sum - 6.0).abs() < 1e-7,
        "redundant row violated: {sum}"
    );
}

/// Equality+bounds path (#313): a strictly convex QP whose ONLY general
/// constraints are equalities, one row an exact scalar multiple of another,
/// with finite variable bounds and no inequality row. This routes through
/// `solve_equality_plus_bounds`, which pins every equality row in each KKT and
/// (before the fix) propagated the rank-deficient factor's failure straight to
/// the user as `InternalError` / exit 1. The path now delegates to the
/// rank-deficiency-aware `solve_general` and reaches the exact optimum.
///
/// Fixture (the issue's reproduction, n = 3):
///   min  ½(x₀² + x₁² + x₂²) − x₀ − 2x₁ − 3x₂
///   s.t.   x₀ +  x₁       = 2          (row 1)
///         2x₀ + 2x₁       = 4          (row 2 — exactly 2× row 1, redundant)
///         −10 ≤ x ≤ 10
/// Closed form: x* = (0.5, 1.5, 3.0), f* = −6.75 (all bounds inactive).
#[test]
fn equality_plus_bounds_prunes_exact_rank_deficient_rows() {
    // H = I₃ (strictly convex): three unit diagonal entries.
    let h_space = SymTMatrixSpace::new(3, vec![1, 2, 3], vec![1, 2, 3]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[1.0, 1.0, 1.0]);

    // Rows 1,2: x₀+x₁ (=2) and 2x₀+2x₁ (=4) — exactly 2× row 1 (rank 1).
    let a_space = GenTMatrixSpace::new(2, 3, vec![1, 1, 2, 2], vec![1, 2, 1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 2.0, 2.0]);

    let g = [-1.0, -2.0, -3.0];
    let bl = [2.0, 4.0];
    let bu = [2.0, 4.0];
    let xl = [-10.0, -10.0, -10.0];
    let xu = [10.0, 10.0, 10.0];

    let qp = QpProblem {
        n: 3,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver
        .solve(&qp, None, &opts())
        .expect("exact rank-deficient equality+bounds QP must solve, not error (#313)");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0] - 0.5).abs() < 1e-8, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.5).abs() < 1e-8, "x[1] = {}", sol.x[1]);
    assert!((sol.x[2] - 3.0).abs() < 1e-8, "x[2] = {}", sol.x[2]);
    assert!((sol.obj - (-6.75)).abs() < 1e-8, "obj = {}", sol.obj);
    // The dropped redundant equality is still satisfied: 2·(x₀+x₁) = 4.
    assert!(
        (2.0 * (sol.x[0] + sol.x[1]) - 4.0).abs() < 1e-7,
        "redundant row violated"
    );
}

/// Cold path: `solve(qp, None, ..)` routes through `cold_general_initial`,
/// which pins ALL equality rows up front. With redundant equalities that
/// factor is singular; the guard prunes the redundant row so the cold start
/// succeeds, and the add/drop loop reaches the vertex. A general inequality
/// row (`x₃ ≤ 5`, slack at the optimum) forces the general-constraint path.
#[test]
fn cold_start_prunes_redundant_equality_rows() {
    let h_space = SymTMatrixSpace::new(3, Vec::new(), Vec::new());
    let h = SymTMatrix::new(Rc::clone(&h_space));

    // Rows 1,2: redundant equalities (rank 1). Row 3: x₃ ≤ 5 (inequality).
    let a_space = GenTMatrixSpace::new(3, 3, vec![1, 1, 1, 2, 2, 2, 3], vec![1, 2, 3, 1, 2, 3, 3]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0]);

    let g = [-1.0, -2.0, -3.0];
    let bl = [3.0, 6.0, NEG_INF];
    let bu = [3.0, 6.0, 5.0];
    let xl = [0.0, 0.0, 0.0];
    let xu = [POS_INF; 3];

    let qp = QpProblem {
        n: 3,
        m: 3,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver
        .solve(&qp, None, &opts())
        .expect("rank-deficient cold start must be repaired, not error");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0]).abs() < 1e-8, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1]).abs() < 1e-8, "x[1] = {}", sol.x[1]);
    assert!((sol.x[2] - 3.0).abs() < 1e-8, "x[2] = {}", sol.x[2]);
    assert!((sol.obj - (-9.0)).abs() < 1e-8, "obj = {}", sol.obj);
}
