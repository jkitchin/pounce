//! Analytical correctness ladder (§8.0 of the design note). Six
//! closed-form QPs with hand-computable answers; runtime budget
//! <50 ms total. Each catches a distinct class of bug at the
//! earliest possible point.
//!
//! Phase 5a commit 2 lands ladder problems 1 and 2 — the
//! equality-only / no-variable-bounds subset that the cold solver
//! handles directly. Problems 3-6 require the working-set / inertia-
//! control machinery and land with the commits that introduce it.
//!
//! 1. `unconstrained_identity_hessian` — `x* = −g`, one Newton step.
//!    Catches: KKT sign, gradient assembly.
//! 2. `equality_only_full_rank` — `[H Aᵀ; A 0]⁻¹ [−g; b]`. Catches:
//!    KKT block layout, multiplier sign convention.
//! 3. `box_constrained_diagonal_hessian` — `x*_i = clip(−gᵢ/hᵢ,
//!    xlᵢ, xuᵢ)`. Catches: bound-multiplier sign, working-set
//!    add/drop. *Lands with bounds support.*
//! 4. `redundant_equality` — strictly convex QP with one redundant
//!    equality. Catches: degeneracy detection, EXPAND triggering.
//!    *Lands with EXPAND.*
//! 5. `infeasible_bounds` — `xl > xu` on one coord; elastic mode
//!    returns minimal-infeas point. Catches: §4.3 phase-1 elastic
//!    detection. *Lands with phase-1 elastic mode.*
//! 6. `indefinite_h_pd_reduced` — indefinite `H`, single equality,
//!    reduced Hessian PD. Catches: §4.5 inertia-control trigger.
//!    *Lands with inertia control.*

use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::solver::{ParametricActiveSetSolver, QpSolver};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()))
}

/// Helper — `n × n` identity Hessian stored as a diagonal triplet
/// (1-based pounce convention).
fn identity_hessian(n: usize) -> SymTMatrix {
    let irows: Vec<i32> = (1..=n as i32).collect();
    let jcols = irows.clone();
    let space = SymTMatrixSpace::new(n as i32, irows, jcols);
    let mut h = SymTMatrix::new(space);
    h.set_values(&vec![1.0; n]);
    h
}

fn empty_gen(m: usize, n: usize) -> GenTMatrix {
    GenTMatrix::new(GenTMatrixSpace::new(
        m as i32,
        n as i32,
        Vec::new(),
        Vec::new(),
    ))
}

// ─────────────────────────────────────────────────────────────────
// Problem 1 — Unconstrained QP, H = I.
//
//     min ½ xᵀ x + gᵀ x
//
// Closed form: x* = -g. One Newton step. Catches KKT sign-convention
// and gradient-assembly bugs.
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_1_unconstrained_identity_hessian() {
    let n = 3;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [1.5, -2.0, 0.25];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 3];
    let xu = [NLP_UPPER_BOUND_INF; 3];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    let expected = [-1.5, 2.0, -0.25];
    for (i, (xi, ei)) in sol.x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (xi - ei).abs() < 1e-12,
            "x[{i}] = {xi} but expected {ei}",
        );
    }
    // Objective: ½‖g‖² − ‖g‖² = -½‖g‖².
    let expected_obj = -0.5 * g.iter().map(|gi| gi * gi).sum::<f64>();
    assert!(
        (sol.obj - expected_obj).abs() < 1e-12,
        "obj = {} but expected {}",
        sol.obj,
        expected_obj
    );
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert_eq!(sol.stats.n_refactor, 1);
    assert_eq!(sol.stats.n_schur_updates, 0);
    assert_eq!(sol.stats.n_working_set_changes, 0);
}

// ─────────────────────────────────────────────────────────────────
// Problem 2 — Equality-only QP, H = I, A full rank.
//
//     min ½ xᵀ x + gᵀ x
//     s.t. A x = b
//
// Closed form by KKT:
//     [I  Aᵀ] [x*]   [−g]
//     [A   0] [λ*] = [ b]
// With H = I we can write the reduced-space solution explicitly:
//     λ* = (A Aᵀ)⁻¹ (A·g + b)
//     x* = −g − Aᵀ λ*
// Catches: KKT block layout, multiplier sign convention.
//
// Use a concrete tiny instance with A Aᵀ trivially invertible:
//     n = 3, m = 1,  A = [1 1 1],  b = [3],  g = [0 0 0]
// Then A Aᵀ = 3, λ* = (0 + 3)/3 = 1, x* = (0,0,0) − (1,1,1)·1 =
// (−1, −1, −1).  But we want Ax* = b: 1·(-1)·3 = -3, not 3. Sign
// check: with KKT convention `Hx + Aᵀλ = −g` and `Ax = b`,
// substituting x = -g - Aᵀλ into Ax = b gives -A·g - A·Aᵀ·λ = b,
// so λ = -(A·Aᵀ)⁻¹ (A·g + b) = -(0 + 3)/3 = -1. Then x = 0 −
// 1·(−1) = (1, 1, 1) and A·x = 3 = b. ✓
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_2_equality_only_full_rank() {
    let n = 3;
    let m = 1;
    let h = identity_hessian(n);

    // A = [1 1 1]
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1, 1], vec![1, 2, 3]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0, 1.0]);

    let g = [0.0; 3];
    let bl = [3.0]; // equality value
    let bu = [3.0];
    let xl = [NLP_LOWER_BOUND_INF; 3];
    let xu = [NLP_UPPER_BOUND_INF; 3];

    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    // x* = (1, 1, 1)
    for i in 0..n {
        assert!(
            (sol.x[i] - 1.0).abs() < 1e-12,
            "x[{i}] = {} but expected 1.0",
            sol.x[i]
        );
    }
    // λ* = -1 (sign as in design-note convention Hx + Aᵀλ = -g)
    assert!(
        (sol.lambda_g[0] + 1.0).abs() < 1e-12,
        "λ_g[0] = {} but expected −1.0",
        sol.lambda_g[0]
    );
    // Objective: ½·3 + 0 = 1.5
    assert!(
        (sol.obj - 1.5).abs() < 1e-12,
        "obj = {} but expected 1.5",
        sol.obj
    );

    // Constraint should be satisfied.
    let ax: f64 = sol.x.iter().sum();
    assert!((ax - 3.0).abs() < 1e-12, "Ax = {ax} but expected 3");

    // Working set should record the equality as active.
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::Equality);
    assert_eq!(sol.status, crate::QpStatus::Optimal);
}

// ─────────────────────────────────────────────────────────────────
// Problem 2b — Equality-only QP with non-identity H.
//
//     H = diag(2, 4),  g = (-2, -8),  A = [1 1],  b = [2]
//
// Unconstrained minimizer of ½xᵀHx + gᵀx is x_uc = (-g_i/h_i) =
// (1, 2). With A x = 2 we shift: solve [H Aᵀ; A 0][x; λ] = [-g; b].
// By inspection x = (1, 1), λ such that 2·1 + λ = 2 ⇒ λ = 0 and
// 4·1 + λ = 8 ⇒ λ = 4. The two rows disagree so x = (1,1) is not
// optimal. Solve properly: x = x_uc − H⁻¹ Aᵀ λ, plug into Ax = b:
//     A·x_uc − A·H⁻¹·Aᵀ·λ = b
//     (1 + 2) − (½ + ¼)·λ = 2
//     λ = (3 − 2) / (¾) = 4/3
// Then x = (1, 2) − (½, ¼)·(4/3) = (1 − 2/3, 2 − 1/3) = (1/3, 5/3).
// Check Ax: 1/3 + 5/3 = 2 ✓. ½xᵀHx + gᵀx = ½(2·1/9 + 4·25/9) +
// (−2·1/3 − 8·5/3) = ½·(2/9 + 100/9) + (−2/3 − 40/3) = 51/9 −
// 42/3 = 17/3 − 14 = (17 − 42)/3 = −25/3.
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_2b_equality_only_non_identity_hessian() {
    let n = 2;
    let m = 1;

    // H = diag(2, 4): two diagonal entries, 1-based.
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[2.0, 4.0]);

    // A = [1 1]
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [-2.0, -8.0];
    let bl = [2.0];
    let bu = [2.0];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    let expected_x = [1.0 / 3.0, 5.0 / 3.0];
    for (i, (xi, ei)) in sol.x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (xi - ei).abs() < 1e-12,
            "x[{i}] = {xi} but expected {ei}",
        );
    }
    // λ* = -4/3 in our sign convention (Hx + Aᵀλ = -g ⇒ 2·(1/3) +
    // λ = 2, λ = 2 − 2/3 = 4/3; with the design-note convention the
    // returned multiplier is −4/3).
    //
    // Walkthrough: KKT is [H Aᵀ; A 0][x; λ] = [-g; b]. Row 1:
    // 2·(1/3) + λ = 2  ⇒ λ = 4/3. Returned value matches.
    assert!(
        (sol.lambda_g[0] - 4.0 / 3.0).abs() < 1e-12,
        "λ_g[0] = {} but expected 4/3",
        sol.lambda_g[0]
    );

    let expected_obj = -25.0 / 3.0;
    assert!(
        (sol.obj - expected_obj).abs() < 1e-12,
        "obj = {} but expected {}",
        sol.obj,
        expected_obj
    );
    assert_eq!(sol.status, crate::QpStatus::Optimal);
}

// ─────────────────────────────────────────────────────────────────
// Problem 3 — Box-constrained, diagonal Hessian.
//
//     min ½ xᵀ diag(2,3,1) x + (-8, 6, -3) x
//     s.t. -1 ≤ x_i ≤ 1
//
// Closed form: x*_i = clip(-g_i / h_i, xl_i, xu_i) per coord.
//   Unconstrained: (4, -2, 3)
//   Clipped:       (1, -1, 1)
// KKT residual: H x* + g = z_l - z_u = (-6, 3, -2)
//   ⇒ lambda_x = (-6, +3, -2) (z_l - z_u packed signed).
// Objective: -14.
//
// Algorithm trace from cold-start x=0:
//   Iter 1: W={}, p=(4,-2,3); blocks at x_0=xu_0 with α=0.25
//   Iter 2: W={0↑}, p=(0,-1.5,2.25); blocks at x_2=xu_2 with α=1/9
//   Iter 3: W={0↑,2↑}, p=(0,-4/3,0); blocks at x_1=xl_1 with α=1/4
//   Iter 4: W={0↑,1↓,2↑}, p=0, λ_sat=(6,-3,2), all sign-correct,
//           Optimal.
//
// Catches: bound-multiplier sign convention, working-set add path,
//          ratio test, snap-to-bound.
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_3_box_constrained_diagonal_hessian() {
    let n = 3;
    // H = diag(2, 3, 1)
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2, 3], vec![1, 2, 3]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[2.0, 3.0, 1.0]);

    let a = empty_gen(0, n);
    let g = [-8.0, 6.0, -3.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0, -1.0];
    let xu = [1.0, 1.0, 1.0];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    let expected_x = [1.0, -1.0, 1.0];
    for (i, (xi, ei)) in sol.x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (xi - ei).abs() < 1e-10,
            "x[{i}] = {xi} but expected {ei}",
        );
    }
    let expected_lx = [-6.0, 3.0, -2.0];
    for (i, (lx, ei)) in sol.lambda_x.iter().zip(expected_lx.iter()).enumerate() {
        assert!(
            (lx - ei).abs() < 1e-10,
            "lambda_x[{i}] = {lx} but expected {ei}",
        );
    }
    assert!(
        (sol.obj - (-14.0)).abs() < 1e-10,
        "obj = {} but expected -14.0",
        sol.obj,
    );
    // Working-set membership matches the algorithm trace.
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::AtUpper);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::AtLower);
    assert_eq!(sol.working.bounds[2], crate::BoundStatus::AtUpper);
    // Three adds, zero drops in the optimal trace.
    assert_eq!(sol.stats.n_working_set_changes, 3);
}

// ─────────────────────────────────────────────────────────────────
// Box-constrained edge case: interior optimum. The unconstrained
// minimum lies strictly inside the box; no bounds should activate.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_interior_optimum_activates_no_bounds() {
    let n = 2;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[2.0, 2.0]);

    let a = empty_gen(0, n);
    let g = [-0.5, 0.25]; // unconstrained min = (0.25, -0.125), interior
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.25).abs() < 1e-12);
    assert!((sol.x[1] + 0.125).abs() < 1e-12);
    assert_eq!(sol.lambda_x, vec![0.0, 0.0]);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::Inactive);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::Inactive);
    assert_eq!(sol.stats.n_working_set_changes, 0);
}

// ─────────────────────────────────────────────────────────────────
// Box-constrained edge case: one-sided lower bound, no upper. The
// solver must handle ±NLP_*_BOUND_INF correctly in the ratio test.
//
//     min ½ x² - 4x  s.t.  x ≥ 1
//
// Unconstrained min x* = 4, feasible, so x* = 4. No active bound.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_one_sided_lower_bound_inactive() {
    let n = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1], vec![1]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0]);

    let a = empty_gen(0, n);
    let g = [-4.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [1.0];
    let xu = [NLP_UPPER_BOUND_INF];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 4.0).abs() < 1e-12);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::Inactive);
}

// ─────────────────────────────────────────────────────────────────
// Box-constrained edge case: one-sided lower bound that's active.
//
//     min ½ x² + 4x  s.t.  x ≥ 1
//
// Unconstrained min x* = -4, infeasible. Clipped to x* = 1.
// KKT: x* + g = z_l - z_u  ⇒ 1 - 4 = z_l ⇒ wait, sign check:
//   H x* + g = z_l - z_u  ⇒ 1 + 4 = z_l - 0 ⇒ z_l = 5. Hmm.
//
// Wait: g = +4. So at x = -4 we have grad = x + 4 = 0. Min is at -4.
// Constraint x ≥ 1 binding ⇒ x* = 1. grad(1) = 5, pointing away
// from feasible region's "downhill" (which doesn't exist beyond
// x = 1 going further right). Lagrangian sign: z_l > 0 ⇒
// lambda_x = z_l = 5.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_one_sided_lower_bound_active() {
    let n = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1], vec![1]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0]);

    let a = empty_gen(0, n);
    let g = [4.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [1.0];
    let xu = [NLP_UPPER_BOUND_INF];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-12);
    assert!(
        (sol.lambda_x[0] - 5.0).abs() < 1e-12,
        "lambda_x[0] = {} but expected 5.0",
        sol.lambda_x[0]
    );
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::AtLower);
}

// ─────────────────────────────────────────────────────────────────
// Box-constrained edge case: fixed variable (xl == xu). The solver
// must put it in the working set as Fixed and never drop it.
//
//     min ½ (x₁² + x₂²) - 3x₁ - 2x₂   s.t.   x₂ = 5
//
// With x₂ pinned at 5, free variable optimum is x₁ = 3.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_fixed_variable_solved_in_subspace() {
    let n = 2;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0, 1.0]);

    let a = empty_gen(0, n);
    let g = [-3.0, -2.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF, 5.0];
    let xu = [NLP_UPPER_BOUND_INF, 5.0];

    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 3.0).abs() < 1e-12);
    assert!((sol.x[1] - 5.0).abs() < 1e-12);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::Inactive);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::Fixed);
}

// ─────────────────────────────────────────────────────────────────
// Mixed unsupported path: a QP with both finite variable bounds
// and general inequality constraints (bl ≠ bu on a general row)
// requires phase-1 elastic mode and lands in a later commit.
// ─────────────────────────────────────────────────────────────────
#[test]
fn rejects_mixed_bounds_plus_general_inequality() {
    let n = 2;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(1, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, 0.0];
    let bl = [0.0];
    let bu = [1.0]; // strict inequality 0 ≤ x₁+x₂ ≤ 1
    let xl = [0.0, 0.0];
    let xu = [1.0, 1.0];

    let qp = QpProblem {
        n,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let err = solver
        .solve(&qp, None, &QpOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, crate::QpError::UnsupportedFeature(_)),
        "expected UnsupportedFeature, got {err:?}"
    );
}
