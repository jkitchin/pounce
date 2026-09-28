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

use crate::error::QpStatus;
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

/// Helper — `n × n` all-zero Hessian (no stored entries).
fn zero_hessian(n: usize) -> SymTMatrix {
    SymTMatrix::new(SymTMatrixSpace::new(n as i32, Vec::new(), Vec::new()))
}

/// Helper — diagonal Hessian from `diag`, storing only the nonzero
/// entries (1-based pounce convention). A zero entry is omitted, so
/// the resulting `H` is genuinely rank-deficient on that coordinate.
fn diag_hessian(diag: &[f64]) -> SymTMatrix {
    let mut irows = Vec::new();
    let mut jcols = Vec::new();
    let mut vals = Vec::new();
    for (k, &d) in diag.iter().enumerate() {
        if d != 0.0 {
            irows.push(k as i32 + 1);
            jcols.push(k as i32 + 1);
            vals.push(d);
        }
    }
    let space = SymTMatrixSpace::new(diag.len() as i32, irows, jcols);
    let mut h = SymTMatrix::new(space);
    h.set_values(&vals);
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
        assert!((xi - ei).abs() < 1e-12, "x[{i}] = {xi} but expected {ei}",);
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
// H1 regression — zero Hessian + linear objective is unbounded.
//
//     min gᵀx,  H = 0,  no constraints, no bounds.
//
// The equality-only KKT is the singular 0-matrix; inertia control
// shifts the H-diagonal by δ and solves `(δI) x = -g`, i.e.
// x = -g/δ. Pre-fix every caller dropped δ and declared this point
// `Optimal` — a δ-dependent garbage answer for a problem that is in
// fact unbounded below. The true (unshifted) stationarity residual is
// `δ·x = -g ≠ 0`, so the fix reports `QpStatus::Unbounded`.
// ─────────────────────────────────────────────────────────────────
#[test]
fn h1_zero_hessian_linear_objective_is_unbounded() {
    let n = 2;
    let h = zero_hessian(n);
    let a = empty_gen(0, n);
    let g = [1.0, -2.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

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
    assert_eq!(
        sol.status,
        crate::QpStatus::Unbounded,
        "min gᵀx with H=0 must be Unbounded, got status {:?} with x = {:?}",
        sol.status,
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────
// N1 regression — bounded singular QP must NOT be falsely Unbounded.
//
//     min ½·1e-6·x₁² − x₁,   x₂ free with zero curvature & zero grad
//     H = diag(1e-6, 0),  g = (−1, 0),  no constraints, no bounds.
//
// The x₁ direction is curved (h₁₁ = 1e-6 > 0) so the problem has a
// finite minimizer x₁* = −g₁/h₁₁ = 1e6, obj* = −5e5. The x₂ direction
// is flat but g₂ = 0, so there is NO descent ray — the QP is bounded.
//
// H is rank-deficient (the x₂ diagonal is structurally absent), so
// inertia control shifts by δ and solves the regularized system,
// giving a large-but-finite x₁ ≈ 1/(1e-6+δ). The OLD magnitude
// heuristic (`δ·‖x‖∞ > 1e-3·‖g‖∞`) fired on this `‖x‖∞ ≈ 1e6` and
// wrongly returned `Unbounded`: it cannot tell a large finite
// minimizer in a *curved* direction from a blow-up along a *flat*
// descent ray. The certified recession test rejects it because the
// dominant ray d = x/‖x‖ ≈ (1,0) has curvature dᵀHd ≈ 1e-6 ≈ ‖H‖
// (NOT a zero-curvature direction).
// ─────────────────────────────────────────────────────────────────
#[test]
fn n1_bounded_singular_qp_is_not_falsely_unbounded() {
    let n = 2;
    let h = diag_hessian(&[1e-6, 0.0]);
    let a = empty_gen(0, n);
    let g = [-1.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

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
    assert_eq!(
        sol.status,
        crate::QpStatus::Optimal,
        "bounded singular QP (curved x₁, flat-but-gradient-free x₂) must be \
         Optimal, got {:?} with x = {:?}",
        sol.status,
        sol.x
    );
    // Finite, negative objective (≈ −5e5 at the regularized minimizer).
    assert!(
        sol.obj.is_finite() && sol.obj < 0.0,
        "expected a finite negative objective, got {}",
        sol.obj
    );
    // Curved coordinate drives toward the large finite minimizer; flat
    // coordinate has no gradient so stays put.
    assert!(
        sol.x[0] > 1e5,
        "x₁ should approach the ≈1e6 minimizer, got {}",
        sol.x[0]
    );
    assert!(
        sol.x[1].abs() < 1e-3,
        "x₂ (flat, zero-gradient) should stay ≈0, got {}",
        sol.x[1]
    );
}

// ─────────────────────────────────────────────────────────────────
// Genuine unbounded WITH curvature in one coordinate (guard that the
// recession test still fires when only PART of the space is flat).
//
//     min ½ x₁² − x₂,   H = diag(1, 0),  g = (0, −1).
//
// The x₂ direction is flat (h₂₂ = 0) and g₂ = −1 drives descent along
// it without bound → Unbounded. The recession ray d = (0,1) has
// dᵀHd = 0 (zero curvature), is feasible, and g·d = −1 < 0.
// ─────────────────────────────────────────────────────────────────
#[test]
fn n1_partial_curvature_descent_ray_is_unbounded() {
    let n = 2;
    let h = diag_hessian(&[1.0, 0.0]);
    let a = empty_gen(0, n);
    let g = [0.0, -1.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

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
    assert_eq!(
        sol.status,
        crate::QpStatus::Unbounded,
        "min ½x₁²−x₂ has a flat descent ray along x₂ and must be Unbounded, \
         got {:?} with x = {:?}",
        sol.status,
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────
// F2(a) regression — δ discarded on the general active-set path.
//
//     min ½ x₁² − x₂,   s.t.  x₁ ≤ 5   (general inequality row)
//     H = diag(1, 0),  g = (0, −1),  A = [1 0],  bl = −∞, bu = 5.
//
// The inequality row routes this to `solve_general` (not the
// equality-only fast path). The QP is unbounded: x₂ runs to +∞ along
// the flat, gradient-driven direction, and the lone inequality only
// caps x₁ (a·p = 0 along the recession ray, so it never blocks). The
// inertia-control shift inside the active-set loop made every step
// finite, and with no blocking constraint the loop simply stepped
// forever and returned `MaxIter` — δ was discarded, so the recession
// ray was never certified. The fix runs the same certified recession
// test (zero curvature + feasible ray + descent) on the unblocked
// Newton step and reports `Unbounded`.
// ─────────────────────────────────────────────────────────────────
#[test]
fn f2_general_active_set_detects_unbounded_ray() {
    let n = 2;
    let m = 1;
    let h = diag_hessian(&[1.0, 0.0]);

    // A = [1 0] — the single row reads x₁.
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1], vec![1]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0]);

    let g = [0.0, -1.0];
    let bl = [NLP_LOWER_BOUND_INF]; // one-sided: x₁ ≤ 5
    let bu = [5.0];
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
    assert_eq!(
        sol.status,
        crate::QpStatus::Unbounded,
        "min ½x₁²−x₂ s.t. x₁≤5 is unbounded along x₂ and must be Unbounded, \
         got {:?} with x = {:?}",
        sol.status,
        sol.x
    );

    // Same problem on the opt-in Schur-update path.
    let schur_opts = QpOptions {
        use_schur_updates: true,
        ..QpOptions::default()
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &schur_opts).unwrap();
    assert_eq!(
        sol.status,
        crate::QpStatus::Unbounded,
        "Schur path: min ½x₁²−x₂ s.t. x₁≤5 must be Unbounded, got {:?} with x = {:?}",
        sol.status,
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────
// Bounded ill-conditioned QP: the certificate must NOT fire on a soft
// (small-but-real curvature) mode.
//
//     min ½ x₁² + ½·10⁻⁴ x₂² − x₂,   H = diag(1, 1e-4, 0),
//     g = (0, −1, 0) — true minimum −5000 at x₂ = 10⁴.
//
// x₃ is structurally flat (h₃₃ = 0), so the inertia shift fires
// (δ > 0) and the certificate runs; the candidate ray is dominated by
// the *soft* x₂ mode, whose curvature (1e-4) is 4 orders below the
// stiffest entry but real — the minimizer is finite. An earlier
// `dᵀHd ≤ 1e-3·‖H‖` curvature clause certified this `Unbounded` with
// obj = −∞ on all three paths (the pre-certificate code answered
// Optimal). The structural-zero floor (`‖Hd‖∞ ≤ 1e-10·‖H‖`) must
// reject the ray and report the finite optimum.
// ─────────────────────────────────────────────────────────────────
#[test]
fn soft_mode_bounded_qp_is_not_falsely_unbounded() {
    let n = 3;
    let h = diag_hessian(&[1.0, 1e-4, 0.0]);
    let g = [0.0, -1.0, 0.0];
    let xl = [NLP_LOWER_BOUND_INF; 3];
    let xu = [NLP_UPPER_BOUND_INF; 3];

    let check = |sol: &crate::QpSolution, path: &str| {
        assert_eq!(
            sol.status,
            crate::QpStatus::Optimal,
            "{path}: bounded soft-mode QP must be Optimal, got {:?} with x = {:?}",
            sol.status,
            sol.x
        );
        // True minimum −5000 at x₂ = 1e4; the δ-regularized solve sits
        // within O(δ/λ) of it.
        assert!(
            (sol.obj + 5000.0).abs() < 5.0,
            "{path}: expected obj ≈ −5000, got {}",
            sol.obj
        );
        assert!(
            (sol.x[1] - 1e4).abs() < 10.0,
            "{path}: expected x₂ ≈ 1e4, got {}",
            sol.x[1]
        );
    };

    // Unconstrained (one-shot equality path, m = 0).
    let a = empty_gen(0, n);
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
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
    check(&sol, "one-shot");

    // With a non-binding inequality row (x₁ ≤ 5) → general active-set
    // path, and the same on the opt-in Schur-update path.
    let a_space = GenTMatrixSpace::new(1, n as i32, vec![1], vec![1]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0]);
    let bl = [NLP_LOWER_BOUND_INF];
    let bu = [5.0];
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
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    check(&sol, "general");

    let schur_opts = QpOptions {
        use_schur_updates: true,
        ..QpOptions::default()
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &schur_opts).unwrap();
    check(&sol, "schur");
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
        assert!((xi - ei).abs() < 1e-12, "x[{i}] = {xi} but expected {ei}",);
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
        assert!((xi - ei).abs() < 1e-10, "x[{i}] = {xi} but expected {ei}",);
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
// Equality + bounds, bound-feasible equality solution.
//
//     min ½‖x‖²
//     s.t. x₁ + x₂ + x₃ = 0.6,   −1 ≤ x_i ≤ 1
//
// Equality-relaxed KKT: x_i = −λ, Σx_i = 0.6 ⇒ λ = −0.2,
// x* = (0.2, 0.2, 0.2). All interior; no bounds activate.
// lambda_g = -0.2 (our convention); lambda_x = 0.
// ─────────────────────────────────────────────────────────────────
#[test]
fn eq_plus_bounds_interior_optimum() {
    let n = 3;
    let m = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2, 3], vec![1, 2, 3]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0; 3]);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1, 1], vec![1, 2, 3]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0, 1.0]);

    let g = [0.0; 3];
    let bl = [0.6];
    let bu = [0.6];
    let xl = [-1.0; 3];
    let xu = [1.0; 3];

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
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    for (i, &xi) in sol.x.iter().enumerate() {
        assert!((xi - 0.2).abs() < 1e-10, "x[{i}] = {xi} but expected 0.2",);
    }
    assert!(
        (sol.lambda_g[0] - (-0.2)).abs() < 1e-10,
        "lambda_g[0] = {} but expected -0.2",
        sol.lambda_g[0]
    );
    for (i, &lx) in sol.lambda_x.iter().enumerate() {
        assert!(lx.abs() < 1e-10, "lambda_x[{i}] = {lx} but expected 0");
    }
    for (i, &b) in sol.working.bounds.iter().enumerate() {
        assert_eq!(
            b,
            crate::BoundStatus::Inactive,
            "bound {i} should be inactive"
        );
    }
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::Equality);
}

// ─────────────────────────────────────────────────────────────────
// Equality + bounds, equality solution lies exactly on a bound but
// LICQ holds (eq row and bound row are independent). The bound is
// initialized as active; the inner loop produces a marginal
// multiplier (≈ 0) and declares optimal.
//
//     min ½(x₁² + x₂²) - x₂
//     s.t. x₁ + x₂ = 1,   0 ≤ x₁ ≤ 1   (x₂ free)
//
// Equality-relaxed:
//   row 1: x₁ + λ = 0  → x₁ = −λ
//   row 2: x₂ − 1 + λ = 0 → x₂ = 1 − λ
//   eq:    x₁ + x₂ = 1  → −λ + 1 − λ = 1 ⇒ λ = 0
//   ⇒ x* = (0, 1).  x₁ = xl_1 exactly (binds), x₂ free.
// LICQ holds: A = [1 1], E = [1 0] are independent.
// ─────────────────────────────────────────────────────────────────
#[test]
fn eq_plus_bounds_bound_active_at_init_marginal_multiplier() {
    let n = 2;
    let m = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0, 1.0]);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, -1.0];
    let bl = [1.0];
    let bu = [1.0];
    let xl = [0.0, NLP_LOWER_BOUND_INF];
    let xu = [1.0, NLP_UPPER_BOUND_INF];

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
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    assert!((sol.x[0] - 0.0).abs() < 1e-10, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-10, "x[1] = {}", sol.x[1]);
    // Multiplier on the equality should match the relaxed solve
    // (λ = 0 in our convention).
    assert!(sol.lambda_g[0].abs() < 1e-10);
    // x_1 starts AtLower (snapped at init). Marginal — multiplier
    // can be either zero or close to it.
    assert!(sol.lambda_x[0].abs() < 1e-10);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::AtLower);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::Inactive);
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::Equality);
}

// ─────────────────────────────────────────────────────────────────
// Equality + bounds, equality solution is bound-INfeasible. Commit
// 4 originally returned `UnsupportedFeature`; once §4.3 elastic
// landed, the cold path falls through to elastic and the solve
// completes. This is the Phase 5c-c24 fix: the eq+bounds branch
// now recovers via elastic instead of erroring.
// ─────────────────────────────────────────────────────────────────
#[test]
fn eq_plus_bounds_with_infeasible_relaxed_init_recovers_via_elastic() {
    let n = 2;
    let m = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0, 1.0]);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[2.0, 1.0]);

    //   min ½(x₁² + x₂²)   s.t.   2x₁ + x₂ = 1,
    //                              −1 ≤ x₁ ≤ 1,
    //                              0.5 ≤ x₂ ≤ 1.
    //
    // Equality-relaxed unbounded min: x = (0.4, 0.2) (violates
    // x₂ ≥ 0.5). With the bound enforced: x₂ = 0.5, x₁ = 0.25;
    // λ_eq = 0.125, μ_xl[1] = 0.375. Closed form verified by hand.
    let g = [0.0, 0.0];
    let bl = [1.0];
    let bu = [1.0];
    let xl = [-1.0, 0.5];
    let xu = [1.0, 1.0];

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
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.25).abs() < 1e-7, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.5).abs() < 1e-7, "x[1] = {}", sol.x[1]);
}

// ─────────────────────────────────────────────────────────────────
// General inequality, equality-relaxed solution is feasible at
// the lower bound: commit 5's cold path solves it directly.
//
//     min ½‖x‖² + x₁ + x₂   s.t.   −2 ≤ x₁ + x₂ ≤ 5,  no bounds
//
// Eq-relaxed (no equality rows): H x = −g ⇒ x = (−1, −1).
// Constraint: a·x = −2 = bl. Activated AtLower at init. Inner
// loop confirms p = 0; multiplier check: x₁ + g₁ + λ = 0 ⇒
// −1 + 1 + λ = 0 ⇒ λ = 0. Marginal, no drop. Optimal.
// ─────────────────────────────────────────────────────────────────
#[test]
fn general_ineq_cold_eq_relaxed_at_lower_bound() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [1.0, 1.0];
    let bl = [-2.0];
    let bu = [5.0];
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
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] + 1.0).abs() < 1e-10);
    assert!((sol.x[1] + 1.0).abs() < 1e-10);
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::AtLower);
}

// ─────────────────────────────────────────────────────────────────
// Warm-start at the optimum of a binding-inequality QP — the
// canonical case for commit 5's machinery.
//
//     min ½‖x‖² + x₁ + x₂   s.t.   x₁ + x₂ ≥ −1   (no bounds)
//
// Unconstrained min: (−1, −1). Inequality violated (−2 < −1) ⇒
// constraint binds. True optimum on x₁+x₂ = −1:
//   ∂L/∂xᵢ = xᵢ + 1 + λ = 0 ⇒ xᵢ = −1 − λ
//   Eq:  2(−1 − λ) = −1 ⇒ λ = −0.5
//   x* = (−0.5, −0.5),  λ_g = −0.5.
//
// Warm-starting at (x*, W = {cons AtLower}) should converge in
// one inner-loop iteration with zero working-set changes.
// ─────────────────────────────────────────────────────────────────
#[test]
fn warm_start_general_ineq_at_optimum_returns_in_one_iter() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [1.0, 1.0];
    let bl = [-1.0];
    let bu = [NLP_UPPER_BOUND_INF];
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

    let ws = crate::QpWarmStart {
        x: vec![-0.5, -0.5],
        lambda_g: vec![-0.5],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::Inactive; 2],
            constraints: vec![crate::ConsStatus::AtLower],
        },
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    assert!((sol.x[0] + 0.5).abs() < 1e-10, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] + 0.5).abs() < 1e-10, "x[1] = {}", sol.x[1]);
    assert!(
        (sol.lambda_g[0] + 0.5).abs() < 1e-10,
        "lambda_g[0] = {}",
        sol.lambda_g[0]
    );
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::AtLower);
    assert_eq!(sol.stats.n_working_set_changes, 0);
}

// ─────────────────────────────────────────────────────────────────
// Warm-start with an extra bound wrongly in the working set —
// algorithm must drop it, then re-solve to the true optimum.
// This is the "drop" path of the active-set inner loop, end-to-end.
//
//     min ½(x₁² + x₂²) − ½ x₁   s.t.   0 ≤ x_i ≤ 1
//
// Unconstrained min: (0.5, 0). Box-feasible (x₁ interior, x₂ at
// xl_2). True optimum has W = {x₂ AtLower}.
//
// Warm-start at x = (0, 0) with W = {x₁ AtLower, x₂ AtLower}:
//   Iter 1: p = 0, multipliers (0.5, 0). Drop x₁ (λ > 0 violates
//           "≤ 0 at lower").
//   Iter 2: W = {x₂}. Step p = (0.5, 0), full step, x → (0.5, 0).
//   Iter 3: p = 0, multipliers OK. Optimal.
// ─────────────────────────────────────────────────────────────────
#[test]
fn warm_start_with_wrong_bound_in_working_set_drops_it() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);

    let g = [-0.5, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [0.0, 0.0];
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

    let ws = crate::QpWarmStart {
        x: vec![0.0, 0.0],
        lambda_g: vec![],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::AtLower, crate::BoundStatus::AtLower],
            constraints: vec![],
        },
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-10, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.0).abs() < 1e-10, "x[1] = {}", sol.x[1]);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::Inactive);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::AtLower);
    assert_eq!(sol.stats.n_working_set_changes, 1);
}

// ─────────────────────────────────────────────────────────────────
// §4.2 Schur-complement path produces the same answer as the
// refactor-per-iteration path. Opts.use_schur_updates flips the
// dispatch; the solver's correctness must be invariant under the
// switch.
//
//     min ½‖x‖² + x₁ + x₂   s.t.   x₁ + x₂ ≥ −1
//
// True optimum: x* = (−0.5, −0.5), λ_g = −0.5, obj = 0.25.
// Warm-start at the optimum to keep the iteration count tiny and
// to exercise both apply_change (for the initial slot activation
// from warm-start consistency) and solve.
// ─────────────────────────────────────────────────────────────────
#[test]
fn schur_path_matches_refactor_path_on_binding_ineq() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [1.0, 1.0];
    let bl = [-1.0];
    let bu = [NLP_UPPER_BOUND_INF];
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

    let ws = crate::QpWarmStart {
        x: vec![-0.5, -0.5],
        lambda_g: vec![-0.5],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::Inactive; 2],
            constraints: vec![crate::ConsStatus::AtLower],
        },
    };

    // Default (refactor-per-iter):
    let mut solver = new_solver();
    let sol_default = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();
    assert_eq!(sol_default.status, crate::QpStatus::Optimal);

    // Schur:
    let mut opts_schur = QpOptions::default();
    opts_schur.use_schur_updates = true;
    let sol_schur = solver.solve(&qp, Some(&ws), &opts_schur).unwrap();
    assert_eq!(sol_schur.status, crate::QpStatus::Optimal);

    // Both must agree on x, λ_g, working set, obj to 1e-9.
    for (i, (&a, &b)) in sol_default.x.iter().zip(sol_schur.x.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "x[{i}] default={a} schur={b}",);
    }
    assert!((sol_default.lambda_g[0] - sol_schur.lambda_g[0]).abs() < 1e-9);
    assert!((sol_default.obj - sol_schur.obj).abs() < 1e-9);
    assert_eq!(sol_default.working, sol_schur.working);
}

// ─────────────────────────────────────────────────────────────────
// Schur path agrees with refactor path on the drop-then-restep
// case (warm_start_with_wrong_bound_in_working_set_drops_it
// translated to use_schur_updates=true).
// ─────────────────────────────────────────────────────────────────
#[test]
fn schur_path_matches_refactor_path_on_drop_test() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-0.5, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [0.0, 0.0];
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
    let ws = crate::QpWarmStart {
        x: vec![0.0, 0.0],
        lambda_g: vec![],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::AtLower, crate::BoundStatus::AtLower],
            constraints: vec![],
        },
    };

    let mut solver = new_solver();
    let mut opts_schur = QpOptions::default();
    opts_schur.use_schur_updates = true;
    let sol = solver.solve(&qp, Some(&ws), &opts_schur).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-9, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.0).abs() < 1e-9, "x[1] = {}", sol.x[1]);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::Inactive);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::AtLower);
    // Schur stats: at least one rank-2 update was applied.
    assert!(
        sol.stats.n_schur_updates > 0,
        "expected ≥1 Schur update, got {}",
        sol.stats.n_schur_updates
    );
}

// PR #50 review C2 — multi-step Schur cross-check. Bumps the
// existing 2-step coverage to a sequence with both adds and a
// drop in between, validating that the running `K_W⁻¹ b == b`
// round-trip survives interleaved sign updates. Compares the
// final primal against the refactor-per-iteration path (the
// strongest possible cross-check — agreement at the optimum
// implies the cumulative rank-2 updates produced a numerically
// correct backsolve at every intermediate step).
#[test]
fn schur_multi_step_add_drop_add_matches_fresh_factor() {
    let n = 3;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-0.25, -0.6, -0.9];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [0.0, 0.0, 0.0];
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
    // Warm start that forces several add/drop cycles: pin all
    // three at AtLower to start, then the solver must drop the
    // ones whose gradient is negative.
    let ws = crate::QpWarmStart {
        x: vec![0.0, 0.0, 0.0],
        lambda_g: vec![],
        lambda_x: vec![0.0, 0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![
                crate::BoundStatus::AtLower,
                crate::BoundStatus::AtLower,
                crate::BoundStatus::AtLower,
            ],
            constraints: vec![],
        },
    };

    let mut solver = new_solver();

    let mut opts_default = QpOptions::default();
    opts_default.use_schur_updates = false;
    let sol_default = solver.solve(&qp, Some(&ws), &opts_default).unwrap();
    assert_eq!(sol_default.status, crate::QpStatus::Optimal);

    let mut solver_b = new_solver();
    let mut opts_schur = QpOptions::default();
    opts_schur.use_schur_updates = true;
    let sol_schur = solver_b.solve(&qp, Some(&ws), &opts_schur).unwrap();
    assert_eq!(sol_schur.status, crate::QpStatus::Optimal);

    // Cross-check: same primal, same objective.
    for i in 0..n {
        assert!(
            (sol_default.x[i] - sol_schur.x[i]).abs() < 1e-9,
            "x[{i}]: refactor = {}, schur = {}",
            sol_default.x[i],
            sol_schur.x[i],
        );
    }
    assert!((sol_default.obj - sol_schur.obj).abs() < 1e-9);
    // Multiple Schur updates happened (multi-step coverage).
    assert!(
        sol_schur.stats.n_schur_updates >= 2,
        "expected ≥2 Schur updates, got {}",
        sol_schur.stats.n_schur_updates
    );
}

// ─────────────────────────────────────────────────────────────────
// `solve_with_working_set` API: caller supplies just a working
// set (not a primal `x`), pounce-qp computes a feasible primal
// compatible with that set internally, then runs the standard
// active-set loop. The §6 SQP integration uses this when each
// outer iteration's QP has a fresh constraint RHS.
//
//     min ½(x² + y²) − x − 2y  s.t.  x + y = 1
//
// Closed form: x* = (0, 1), λ_g = 1. Working set: cons[0]
// Equality.
// ─────────────────────────────────────────────────────────────────
#[test]
fn solve_with_working_set_recovers_optimum_from_active_set_seed() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);
    let g = [-1.0, -2.0];
    let bl = [1.0];
    let bu = [1.0];
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

    let working = crate::WorkingSet {
        bounds: vec![crate::BoundStatus::Inactive; 2],
        constraints: vec![crate::ConsStatus::Equality],
    };

    let mut solver = new_solver();
    let sol = solver
        .solve_with_working_set(&qp, &working, &QpOptions::default())
        .unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.0).abs() < 1e-10);
    assert!((sol.x[1] - 1.0).abs() < 1e-10);
    // KKT at x* = (0, 1): Hx + Aᵀλ = -g ⇒ (0, 1) + (λ, λ) = (1, 2)
    // ⇒ λ = 1. The pounce-qp convention returns lambda_g with this
    // sign (positive when the equality "pulls" upward).
    assert!(
        (sol.lambda_g[0] - 1.0).abs() < 1e-10,
        "lambda_g[0] = {} (expected 1.0)",
        sol.lambda_g[0]
    );
}

// ─────────────────────────────────────────────────────────────────
// EXPAND (Harris-style two-pass) ratio-test selection.
//
// At a degenerate intersection where multiple constraints would
// activate at the same α, the strict-min ratio test picks the
// first-encountered constraint (lowest index). Harris picks the
// one with the largest |a·p|, which avoids cycling at degenerate
// vertices because the chosen direction "actually moves" with
// the step.
//
//     min ½‖x‖² − x₁ − x₂   s.t.   x₁ ≤ 0.5, x₂ ≤ 0.5
//
// Unconstrained min (1, 1). Both bounds active at optimum. From
// x = (0, 0), p = (1, 1). Both bounds hit at α = 0.5 — a true tie
// in ratio. With `Expand` we pick the larger |p_i| = 1, which is
// either one (tie). With `None` / `Bland` we pick the lower
// index, i.e. x₁'s bound.
// Both strategies converge to the same optimum (0.5, 0.5); the
// test verifies that, and that the two strategies pick valid
// blockers.
// ─────────────────────────────────────────────────────────────────
#[test]
fn anti_cycling_expand_two_pass_converges_at_degenerate_vertex() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-1.0, -1.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF];
    let xu = [0.5, 0.5];

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

    // Default (steepest-violation drop + first-min ratio test):
    let mut solver = new_solver();
    let sol_default = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol_default.status, crate::QpStatus::Optimal);
    assert!((sol_default.x[0] - 0.5).abs() < 1e-10);
    assert!((sol_default.x[1] - 0.5).abs() < 1e-10);

    // EXPAND (Harris-style):
    let opts_expand = crate::QpOptions {
        anti_cycling: crate::AntiCyclingChoice::Expand,
        ..QpOptions::default()
    };
    let sol_expand = solver.solve(&qp, None, &opts_expand).unwrap();
    assert_eq!(sol_expand.status, crate::QpStatus::Optimal);
    assert!((sol_expand.x[0] - 0.5).abs() < 1e-10);
    assert!((sol_expand.x[1] - 0.5).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────
// Full GMSW EXPAND with τ-growth on the same degenerate-vertex
// problem from anti_cycling_expand_two_pass... — verifies that
// the τ-relaxation + snap-reset machinery is wired and doesn't
// break correctness on standard problems. (Cycling-pathology
// stress-tests need very large iteration counts to actually
// trigger τ_max overflow; this test just exercises the
// τ-growth code path.)
// ─────────────────────────────────────────────────────────────────
#[test]
fn expand_tau_growth_does_not_break_correctness() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-1.0, -1.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF];
    let xu = [0.5, 0.5];
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

    // Use EXPAND with a deliberately tight τ_max so the snap
    // reset triggers within a few iterations.
    let opts = crate::QpOptions {
        anti_cycling: crate::AntiCyclingChoice::Expand,
        expand_tol_initial: 1e-12,
        expand_tol_growth: 1e-8, // grow fast
        expand_tol_max: 1e-7,    // hit ceiling within ~10 iters
        ..crate::QpOptions::default()
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &opts).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-9);
    assert!((sol.x[1] - 0.5).abs() < 1e-9);
}

// ─────────────────────────────────────────────────────────────────
// EXPAND must NOT route a problem with non-degenerate single
// blocker differently from the default — the Harris test
// degenerates to single-blocker selection.
// ─────────────────────────────────────────────────────────────────
#[test]
fn anti_cycling_expand_single_blocker_matches_default() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-2.0, -1.0]; // unconstrained min (2, 1); only x₁ blocks
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [1.0, NLP_UPPER_BOUND_INF];

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
    let opts_expand = crate::QpOptions {
        anti_cycling: crate::AntiCyclingChoice::Expand,
        ..QpOptions::default()
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &opts_expand).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-10);
    assert!((sol.x[1] - 1.0).abs() < 1e-10);
    assert_eq!(sol.working.bounds[0], crate::BoundStatus::AtUpper);
    assert_eq!(sol.working.bounds[1], crate::BoundStatus::Inactive);
}

// ─────────────────────────────────────────────────────────────────
// Bland's-rule drop selection (§4.4 anti-cycling fallback).
//
// Box QP with two wrong-sign bounds in the warm-start working
// set. The steepest-violation rule picks the larger-magnitude
// violation first; Bland's picks the lower-indexed one. Both
// converge to the same optimum but record different first-drop
// behavior. The test pins which constraint the algorithm drops
// first.
//
//     min ½(x₁² + x₂²) − 0.25 x₁ − 0.5 x₂   s.t.   0 ≤ x_i ≤ 1
//
// Unconstrained min: (0.25, 0.5). Box-feasible, no bound active.
// Warm-start at (0, 0) with W = {x₁ AtLower, x₂ AtLower}: both
// multipliers wrong-sign (λ_sat[x₁] = 0.25, λ_sat[x₂] = 0.5).
// Steepest violation picks x₂ (larger λ); Bland picks x₁
// (smaller index).
// ─────────────────────────────────────────────────────────────────
#[test]
fn anti_cycling_bland_picks_lowest_indexed_violation() {
    let n = 2;
    let h = identity_hessian(n);
    let a = empty_gen(0, n);
    let g = [-0.25, -0.5];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [0.0, 0.0];
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
    let ws = crate::QpWarmStart {
        x: vec![0.0, 0.0],
        lambda_g: vec![],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::AtLower, crate::BoundStatus::AtLower],
            constraints: vec![],
        },
    };

    // Steepest violation (default): drops x₂ first, then x₁.
    let mut solver = new_solver();
    let sol_default = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();
    assert_eq!(sol_default.status, crate::QpStatus::Optimal);
    assert!((sol_default.x[0] - 0.25).abs() < 1e-10);
    assert!((sol_default.x[1] - 0.5).abs() < 1e-10);

    // Bland's: drops x₁ first, then x₂. Same optimum, possibly
    // different iteration count (could be more or fewer; we just
    // pin the OPTIMUM is reached).
    let opts_bland = crate::QpOptions {
        anti_cycling: crate::AntiCyclingChoice::Bland,
        ..QpOptions::default()
    };
    let sol_bland = solver.solve(&qp, Some(&ws), &opts_bland).unwrap();
    assert_eq!(sol_bland.status, crate::QpStatus::Optimal);
    assert!((sol_bland.x[0] - 0.25).abs() < 1e-10);
    assert!((sol_bland.x[1] - 0.5).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────
// Cold start through l1-elastic (§4.3): an inequality QP whose
// eq-relaxed solution is infeasible. The cold-init in
// solve_general now returns Ok(None), triggering solve_elastic,
// which augments with two slacks per row and re-solves from a
// slack-feasible warm start.
//
//     min ½‖x‖²   s.t.   x₁ + x₂ ≥ 1,  no bounds
//
// Eq-relaxed (0, 0) violates the constraint. Elastic mode finds
// the true optimum on x₁+x₂=1: by symmetry x = (0.5, 0.5),
// λ_g = -1 (∂L/∂x_i = x_i + λ = 0 ⇒ λ = -x_i = -0.5… wait, let
// me redo: at optimum x = (0.5, 0.5), Hx + Aᵀλ = 0
// ⇒ 0.5 + λ = 0 ⇒ λ = -0.5).
// Slacks zero ⇒ status = Optimal (not Infeasible).
// ─────────────────────────────────────────────────────────────────
#[test]
fn general_ineq_solved_via_l1_elastic_when_cold_infeasible() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, 0.0];
    let bl = [1.0];
    let bu = [NLP_UPPER_BOUND_INF];
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
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-6, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.5).abs() < 1e-6, "x[1] = {}", sol.x[1]);
    assert!(sol.stats.used_phase1, "elastic mode should have been used");
    assert_eq!(sol.working.constraints[0], crate::ConsStatus::AtLower);
}

// ─────────────────────────────────────────────────────────────────
// Ladder #6 — indefinite H with PD reduced Hessian.
//
//     H = diag(-1, 2),  g = (0, 0),  A = [1 1],  b = 1   (equality)
//
// H is indefinite (eigenvalues -1, 2) but the reduced Hessian on
// null(A) = span{(1, -1)} is
//     dᵀ H d  =  (1)·(-1)·(1) + (-1)·(2)·(-1)  =  -1 + 2  =  1  > 0
// so the saddle-point system [H Aᵀ; A 0] has the canonical
// (n, m, 0) = (2, 1, 0) inertia by Wright's theorem and FERAL
// reports `number_of_neg_evals = 1` — no shift needed.
//
// Closed form: ∇_x L = Hx + Aᵀλ = 0 ⇒ (-x₁ + λ, 2x₂ + λ) = 0
// ⇒ x₁ = λ, x₂ = -λ/2. Eq: x₁ + x₂ = 1 ⇒ λ - λ/2 = λ/2 = 1
// ⇒ λ = 2. So x = (2, -1), λ_g = 2.
// Objective: ½·(-1·4 + 2·1) + 0 = ½·(-2) = -1.
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_6_indefinite_h_with_pd_reduced_hessian() {
    let n = 2;
    let m = 1;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[-1.0, 2.0]);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, 0.0];
    let bl = [1.0];
    let bu = [1.0];
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
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    assert!((sol.x[0] - 2.0).abs() < 1e-10, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] + 1.0).abs() < 1e-10, "x[1] = {}", sol.x[1]);
    assert!(
        (sol.lambda_g[0] - 2.0).abs() < 1e-10,
        "lambda_g[0] = {}",
        sol.lambda_g[0]
    );
    assert!(
        (sol.obj + 1.0).abs() < 1e-10,
        "obj = {} but expected -1.0",
        sol.obj
    );
}

// ─────────────────────────────────────────────────────────────────
// Inertia-control shift path — H with a hard zero direction that
// the saddle theorem doesn't cover. Without shift the KKT factor
// reports `WrongInertia`; with shift `H ← H + δI` the reduced
// Hessian becomes PD and the factor succeeds.
//
//     H = diag(0, 1),  g = (0, -2),  no constraints
//
// H is PSD but not PD — the (1, 0) direction has zero curvature.
// Crucially `g` has **no component along that null direction**
// (g₁ = 0), so the objective is *bounded below* despite the
// singular Hessian: any x₁ leaves the objective unchanged, and the
// minimum `-2` is attained at x₂ = 2 for every x₁. The shift is
// needed only to make the singular KKT factorable; the regularized
// solution is x₁ = -g₁/δ = 0, x₂ = 2/(1 + δ) ≈ 2, which stays
// finite as δ → 0. The H1 re-verification (`δ·‖x‖∞ ≤ 1e-3·‖g‖∞`)
// therefore keeps `Optimal` here — distinguishing this bounded
// singular problem from the genuinely-unbounded one in
// `h1_zero_hessian_linear_objective_is_unbounded`, where g *does*
// drive the null direction and x blows up to ≈ 1/δ.
// ─────────────────────────────────────────────────────────────────
#[test]
fn inertia_control_shift_succeeds_on_psd_singular_hessian() {
    let n = 2;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[0.0, 1.0]); // singular: zero in (1,1)

    let a = empty_gen(0, n);
    let g = [0.0, -2.0]; // no descent along the null direction ⇒ bounded
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);
    // The shift adds δI to the *whole* H-block, so x₂ becomes
    // `2/(1 + δ) ≈ 2 - 2δ`. δ_initial = 1e-8 ⇒ error ≈ 2e-8.
    // The 1e-6 tolerance is loose enough to accept this minor
    // PD-direction perturbation (the standard cost of Tikhonov-
    // style regularization).
    assert!((sol.x[1] - 2.0).abs() < 1e-6, "x[1] = {}", sol.x[1]);
    // x₁ stays ≈ 0: g has no component along the null direction, so
    // the regularizer does not blow it up (this is what keeps the
    // problem bounded and the status `Optimal`).
    assert!(
        sol.x[0].abs() < 1e-6,
        "x[0] = {} should stay ≈ 0 (g has no null-direction component)",
        sol.x[0]
    );
}

// ─────────────────────────────────────────────────────────────────
// Ladder #5 — infeasibility certification via l1-elastic.
//
//     min ½ x²   s.t.   x ≥ 5,  x ≤ 3,  x free
//
// No x satisfies both constraints. Elastic mode minimizes
//     ½x² + γ·(v_l + v_u)
// s.t. x + v_l ≥ 5, x − v_u ≤ 3, v_l, v_u ≥ 0.
// Closed form (γ large enough): x = 3, v_l = 2, v_u = 0; the
// penalty term equals γ·2. Status reported: Infeasible.
// ─────────────────────────────────────────────────────────────────
#[test]
fn problem_5_infeasibility_certified_by_elastic_mode() {
    let n = 1;
    let m = 2;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1], vec![1]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0]);

    // Two rows in A, both with one nonzero at column 1.
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 2], vec![1, 1]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0];
    let bl = [5.0, NLP_LOWER_BOUND_INF];
    let bu = [NLP_UPPER_BOUND_INF, 3.0];
    let xl = [NLP_LOWER_BOUND_INF];
    let xu = [NLP_UPPER_BOUND_INF];

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

    assert_eq!(
        sol.status,
        crate::QpStatus::Infeasible,
        "expected Infeasible; got {:?}",
        sol.status
    );
    assert!(
        sol.stats.used_phase1,
        "elastic mode should have run for an infeasible problem"
    );
    // Minimal-l1 elastic minimum sits at x = 3 (the upper-side
    // constraint binds with zero slack; the lower-side absorbs
    // a violation of 2).
    assert!(
        (sol.x[0] - 3.0).abs() < 1e-6,
        "x = {} but expected 3.0 (the minimum-violation point)",
        sol.x[0]
    );
}

// ─────────────────────────────────────────────────────────────────
// L15 regression (dev-notes/code-review-2026-06.md): `solve_elastic`
// hard-called `solve_general`, ignoring `opts.use_schur_updates` — so
// an infeasible problem solved with the Schur path silently fell back
// to the refactor path. Same infeasible problem as
// `problem_5_infeasibility_certified_by_elastic_mode`, but with
// `use_schur_updates = true`: the elastic recovery must now route the
// augmented solve through `solve_general_schur`, which records ≥1
// rank-2 Schur update in the stats (the refactor path reports 0).
// ─────────────────────────────────────────────────────────────────
#[test]
fn l15_elastic_honors_use_schur_updates() {
    let n = 1;
    let m = 2;
    let h_space = SymTMatrixSpace::new(n as i32, vec![1], vec![1]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0]);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 2], vec![1, 1]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0];
    let bl = [5.0, NLP_LOWER_BOUND_INF];
    let bu = [NLP_UPPER_BOUND_INF, 3.0];
    let xl = [NLP_LOWER_BOUND_INF];
    let xu = [NLP_UPPER_BOUND_INF];

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

    let mut opts = QpOptions::default();
    opts.use_schur_updates = true;

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &opts).unwrap();

    // Same minimal-l1 infeasibility certificate as the refactor path.
    assert_eq!(sol.status, crate::QpStatus::Infeasible);
    assert!(sol.stats.used_phase1, "elastic mode should have run");
    assert!((sol.x[0] - 3.0).abs() < 1e-6, "x = {}", sol.x[0]);
    // The Schur path was actually taken inside the elastic recovery:
    // the refactor path leaves n_schur_updates == 0.
    assert!(
        sol.stats.n_schur_updates > 0,
        "elastic solve should have used the Schur path (≥1 update), got {}",
        sol.stats.n_schur_updates
    );
}

// ─────────────────────────────────────────────────────────────────
// M5 regression (dev-notes/code-review-2026-06.md): a warm start can
// drive `solve` to report `Optimal` at a point that violates an
// equality row the caller left `Inactive`.
//
//     min ½‖x‖²   s.t.   x₁ + x₂ = 2   (no bounds)
//
// True optimum: x* = (1, 1), obj = 1. We warm-start at x = (0, 0)
// with the single equality row marked `Inactive` (not `Equality`).
//
// Pre-fix: the inner loop sees no active rows, computes p = −Hx − g
// = 0, declares KKT-stationarity, finds no active row to drop, and
// returns `Optimal` at (0, 0) — which violates x₁ + x₂ = 2 by 2.0
// (the ratio test would have `continue`d past the equality row even
// if it had been reached). Post-fix: the feasibility audit catches
// the violation and recovers through elastic mode, returning the
// true feasible optimum (1, 1).
// ─────────────────────────────────────────────────────────────────
#[test]
fn m5_warm_start_inactive_equality_is_not_a_false_optimal() {
    let n = 2;
    let m = 1;
    let h = identity_hessian(n);

    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, 0.0];
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

    // Infeasible warm start with the equality row left Inactive.
    let ws = crate::QpWarmStart {
        x: vec![0.0, 0.0],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0, 0.0],
        working: crate::WorkingSet {
            bounds: vec![crate::BoundStatus::Inactive; 2],
            constraints: vec![crate::ConsStatus::Inactive],
        },
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();

    // The returned point MUST satisfy the equality (this is the
    // assertion that fails pre-fix: x = (0, 0) ⇒ residual 2.0).
    let residual = (sol.x[0] + sol.x[1] - 2.0).abs();
    assert!(
        residual < 1e-6,
        "returned x = ({}, {}) violates x₁+x₂=2 by {residual}; status = {:?}",
        sol.x[0],
        sol.x[1],
        sol.status
    );

    // And it must be the true optimum, reported feasible.
    assert_eq!(
        sol.status,
        crate::QpStatus::Optimal,
        "status = {:?}",
        sol.status
    );
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", sol.x[1]);
}

// ─────────────────────────────────────────────────────────────────
// Regression — #282: collapsed-cone QP (m/n ≫ 1, no interior).
//
//     min ½‖x‖² − eᵀx   s.t.  a_iᵀ x ≤ 0,  i = 1..40,   x ∈ R⁵
//
// The `a_i` are the 40 random unit vectors from the issue's seed=7
// draw. Their positive hull spans R⁵, so the feasible set collapses
// to exactly {0}: every one of the 40 rows is active at the unique
// feasible (hence optimal) point x* = 0, there is no interior (Slater
// fails), and the multipliers are wildly non-unique. This is the
// m/n ≥ 5 geometry on which the active-set phase-1 previously stalled
// and `solve_elastic` FALSELY returned `QpStatus::Infeasible` — a
// confident infeasibility certificate on a trivially feasible problem
// (0 satisfies G·0 = 0 ≤ 0 exactly). A feasible problem has no
// Farkas certificate, so this must NEVER report Infeasible.
//
// Post-fix contract: the phase-2 recovery re-solve inside
// `solve_elastic` finds the feasible optimum, so this now solves to
// x* = 0. At minimum it must not be `Infeasible`.
#[rustfmt::skip]
const CONE_G_40X5: [[f64; 5]; 40] = [
        [1.14006678131625184e-03, 2.76867807829154766e-01, -2.54062195171605498e-01, -8.25372027725340573e-01, -4.21374339594566771e-01],
        [-5.36974168392638451e-01, 3.25676127509080815e-02, 7.25723256257718385e-01, -2.66528618129953432e-01, -3.35985630828422255e-01],
        [4.38997742127722024e-01, 3.19843081494841475e-01, 9.44725290135275397e-02, -8.33887924882038223e-01, -2.62155607295105966e-02],
        [2.49321106852515434e-01, -4.82007074679814573e-01, -1.64091390583346458e-01, -6.81738501474345227e-01, -4.62401121084694000e-01],
        [-8.11421143044696369e-01, -1.03575112805048219e-01, -5.58404358725853700e-01, 1.19512107651655708e-01, 6.90605017943466265e-02],
        [-7.23560679054214839e-02, -9.74171702273510354e-01, -2.08513897092891864e-01, -1.87734444174779899e-02, 4.38589378648868503e-02],
        [-6.64223361298640680e-01, -2.07390019913612778e-01, -4.24769648402377020e-01, -3.51111713108238688e-01, 4.60530147395027289e-01],
        [-6.03852707244763143e-01, -2.43188561128928268e-02, 6.61322952428854971e-01, -4.36400704547353746e-01, -8.35277130770765708e-02],
        [6.01805229660406851e-02, 3.47481129845238490e-02, -6.67406618362186177e-01, 4.14809615867338041e-02, 7.40282789811933850e-01],
        [-5.62706290823368427e-01, 3.12562911284687661e-01, 4.34098129567316582e-02, -2.33306833707185496e-01, 7.27564132047228251e-01],
        [4.92745774123164670e-01, -7.75253537733193832e-01, 4.81693525823559052e-02, 3.72788107293740179e-01, -1.22033990543938875e-01],
        [3.68100127518360298e-01, -3.58539550587552713e-02, 3.59657665183738762e-01, 7.75387887491390315e-01, -3.64193324806801388e-01],
        [1.43029200307004833e-01, -3.26213278910249094e-01, 8.96092527514146392e-02, -8.35899603731578456e-01, -4.07884271362461026e-01],
        [-9.20693441849489841e-02, 4.21765029621002263e-01, 5.37421016676730035e-01, -6.21094990497102262e-01, -3.72903686310155635e-01],
        [2.59964180454159921e-01, -8.00672493357831638e-01, -1.86129134816472641e-01, -3.90956594418991402e-02, 5.05143205285614405e-01],
        [3.91433013352545434e-01, -1.85786786208968019e-01, -2.09271767620131860e-01, -1.42057130037411244e-01, 8.65036742115945501e-01],
        [-6.35768278605074677e-01, -4.51072679541090371e-01, 5.23719349193674688e-01, -1.79386784137966104e-01, -2.93036787192245907e-01],
        [-6.20419558137837224e-01, -6.41625969630275298e-03, -2.47028617662692684e-01, 6.49411917480900769e-01, 3.63702387923147130e-01],
        [-1.86838595351474547e-02, 5.17235641142353875e-01, -2.63012622090559312e-01, 8.14202128156561900e-01, -4.17852262205643723e-03],
        [1.93182237796024975e-01, -4.27468613166594069e-01, 1.14800228635058829e-01, -5.59034819809572259e-01, -6.73982333227931107e-01],
        [-1.17980755449147934e-01, -3.48710022882760096e-01, 6.35682844528384638e-02, 8.69813447191548961e-01, -3.22281711521505476e-01],
        [-7.21389612485113241e-01, 2.37483446058063874e-01, 5.70010935233716376e-01, -2.03956745912026505e-01, -2.38092039677780698e-01],
        [5.17896517823822178e-01, 3.83306127173157929e-01, -7.62085900882474743e-01, -5.83770701200977335e-02, 2.60155157130209612e-02],
        [-6.19470204409825920e-01, 1.52645735597002091e-01, -5.04017283114570946e-01, 5.71052768017808465e-01, 1.13231001556570662e-01],
        [3.75967106335250364e-02, -2.48814201647233474e-01, -4.99329828984094101e-02, -8.41021664292754823e-01, -4.76305823833071185e-01],
        [1.20941880247688829e-01, -7.09494660858058013e-01, 2.82191828599218375e-01, -5.82009446862597479e-01, 2.52236324605687789e-01],
        [-3.68616566806838908e-01, 3.39621557399881369e-01, 5.70916072818533071e-02, -6.70023426941934064e-01, 5.44599100395676272e-01],
        [8.14334163869365413e-01, -3.71692565309879111e-02, -1.54718923059518371e-01, -9.02992896756755065e-02, -5.50805236953577193e-01],
        [6.91286299467434806e-01, -3.41615035101577813e-01, -3.22115942537574140e-02, -4.99182181180119533e-01, -3.93956829042291901e-01],
        [-6.25724962287801612e-01, 6.15609427348765004e-01, -7.54594529259716429e-02, 4.73029170374398955e-01, 6.52529449723616029e-03],
        [-6.79721432790381996e-01, -3.19778002474718492e-01, -5.48385826323768866e-01, 7.79081627016873449e-03, -3.67332399343851701e-01],
        [-1.24237452922767083e-01, -5.71051038712015790e-01, -3.34222157385254459e-01, 6.85165114521146257e-01, -2.78046905814759548e-01],
        [-4.55172408708694409e-01, 1.45662218221790513e-01, 6.07679776033102859e-01, -6.27867986560743829e-01, -9.00426442611968192e-02],
        [-3.14207144671359184e-01, -8.75441286162504162e-01, 3.65348136671841850e-01, -1.16544845253235881e-02, 3.55153029055668285e-02],
        [-4.94737715802489930e-01, 2.99076760391308261e-01, -3.54654619258104375e-01, -9.39765104507929716e-02, -7.28818359043428510e-01],
        [-6.40415580750529290e-01, 7.03308500489194133e-01, -2.67047953910354630e-01, 1.53602681386187184e-01, -1.77944838326449424e-02],
        [-4.49377543891295694e-01, -5.17440180270609229e-01, 6.41840749076716399e-01, -3.07500845201134188e-01, -1.54269782890148588e-01],
        [1.46151556953761329e-02, 7.73791509159409974e-01, 4.47573217287683900e-01, 2.51636842233827152e-01, -3.70661857188114952e-01],
        [-6.85952351330849863e-01, 4.71307548357392969e-01, 4.79704550876281721e-01, -6.98418537506430709e-02, 2.68968819565169270e-01],
        [3.62260953182630574e-01, 3.85320044238508408e-01, 4.27134457981805493e-01, -2.11214986933055432e-01, 7.02310365441593309e-01],
];

#[test]
fn collapsed_cone_no_interior_not_false_infeasible() {
    let n = 5usize;
    let m = 40usize;

    let h = identity_hessian(n);
    let g = vec![-1.0_f64; n]; // ½‖x‖² − eᵀx

    // A = G (m×n), rows a_iᵀ x ≤ 0.
    let mut irows = Vec::with_capacity(m * n);
    let mut jcols = Vec::with_capacity(m * n);
    let mut vals = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            irows.push(i as i32 + 1);
            jcols.push(j as i32 + 1);
            vals.push(CONE_G_40X5[i][j]);
        }
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, irows, jcols);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&vals);

    let bl = vec![NLP_LOWER_BOUND_INF; m];
    let bu = vec![0.0_f64; m];
    let xl = vec![NLP_LOWER_BOUND_INF; n];
    let xu = vec![NLP_UPPER_BOUND_INF; n];

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

    // The core #282 guarantee: NEVER a confident infeasibility
    // certificate on this feasible problem.
    assert_ne!(
        sol.status,
        QpStatus::Infeasible,
        "collapsed-cone QP is feasible (x = 0 satisfies G·0 = 0 ≤ 0) —          must not certify Infeasible"
    );

    // Stronger post-fix guarantee: the recovery path solves it to the
    // true optimum x* = 0.
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal, got {}",
        sol.status
    );
    let x_inf = sol.x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(x_inf < 1e-6, "x* should be 0, got ‖x‖∞ = {x_inf:.3e}");
    // Feasible to tolerance: G·x ≤ 0.
    let ax = crate::kkt::a_times_x(qp.a, &sol.x, m);
    let max_viol = ax.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_viol < 1e-6,
        "G·x* must be ≤ 0, max row = {max_viol:.3e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Nonconvex step QP falsely certified infeasible (gh#484 follow-up).
//
// This is HS071's SQP step QP, linearized at `x* + 1e-3·e₁` — one
// millimetre from the NLP's own solution — with the exact ∇²L as the
// Hessian, which is what `SqpHessianSource::Exact` (the default) hands
// the QP solver. `H` is indefinite: three zeros on its diagonal.
//
// The QP is emphatically feasible. `p = (0, -0.0300768, 0, 0.1)` sits
// well inside the box, satisfies the equality row exactly, and clears
// the inequality row with a slack of 1.66. Yet `solve_elastic` returned
// `QpStatus::Infeasible`, and the SQP driver turned that into
// `Infeasible_Problem_Detected` at iteration 0.
//
// Two premises of the residual-slack certificate failed at once:
//
//  * It is a *global* claim — "the minimal l1 infeasibility is
//    positive" — but an active-set solve of a NONCONVEX elastic problem
//    stops at a local KKT point. With γ = 1e6 turning a ~1e-7 slack into
//    ~0.1 of apparent objective, phase-1 settled at a far box vertex
//    carrying a cancelling `(v_l, v_u)` pair, missing `feas_tol` by a
//    factor of two: 1.95e-9 against 1e-9.
//  * The phase-2 recovery that exists to catch exactly this seeded a
//    COLD working set, marking the equality row `Inactive`. The warm
//    inner loop cannot pull an Inactive equality in, so the row went
//    unenforced and recovery "converged" to `Optimal` while violating
//    it by 7.8 — never rescuing anything on any QP with an equality.
#[rustfmt::skip]
const HS071_STEP_QP_G: [f64; 4] = [
    14.573655005576034, 1.3794082930783524, 2.3794082930783524, 9.565149621547352,
];
#[rustfmt::skip]
const HS071_STEP_QP_A: [f64; 8] = [
    25.005270924065304, 5.270925976501263, 6.543912442918067, 18.127534138759138,
    2.0,                9.48799927585525,  7.642299967239455,  2.758816586156705,
];
/// Lower triangle in `(0,0), (1,0), (1,1), (2,0), …` order.
#[rustfmt::skip]
const HS071_STEP_QP_H: [f64; 10] = [
    2.758816586156705,
    1.3794082930783524, 0.0,
    1.3794082930783524, 0.0, 0.0,
    10.565149621547352, 1.0, 1.0, 0.0,
];

#[test]
fn nonconvex_step_qp_near_nlp_solution_not_false_infeasible() {
    let n = 4usize;
    let m = 2usize;

    let mut h_irows = Vec::new();
    let mut h_jcols = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            h_irows.push(i as i32 + 1);
            h_jcols.push(j as i32 + 1);
        }
    }
    let h_space = SymTMatrixSpace::new(n as i32, h_irows, h_jcols);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&HS071_STEP_QP_H);

    let mut a_irows = Vec::new();
    let mut a_jcols = Vec::new();
    for i in 0..m {
        for j in 0..n {
            a_irows.push(i as i32 + 1);
            a_jcols.push(j as i32 + 1);
        }
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, a_irows, a_jcols);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&HS071_STEP_QP_A);

    let g = HS071_STEP_QP_G.to_vec();
    // Row 0: one-sided inequality. Row 1: equality (bl == bu).
    let bl = vec![-0.005270924065303717, -0.00948700098781785];
    let bu = vec![NLP_UPPER_BOUND_INF, -0.00948700098781785];
    let xl = vec![
        0.0,
        -3.743999637927625,
        -2.8211499836197276,
        -0.37940829307835244,
    ];
    let xu = vec![
        4.0,
        0.256000362072375,
        1.1788500163802724,
        3.6205917069216476,
    ];

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    // Sanity: the witness really is feasible, so the assertions below
    // rest on arithmetic rather than on trusting the solver.
    let witness = [0.0, -0.030_076_800_314_444_076, 0.0, 0.1];
    let row = |i: usize| -> f64 {
        (0..n)
            .map(|j| HS071_STEP_QP_A[i * n + j] * witness[j])
            .sum()
    };
    assert!(
        row(0) >= bl[0],
        "witness must satisfy row 0: {} < {}",
        row(0),
        bl[0]
    );
    assert!(
        (row(1) - bl[1]).abs() < 1e-12,
        "witness must satisfy the equality row: {} vs {}",
        row(1),
        bl[1]
    );
    for j in 0..n {
        assert!(witness[j] >= xl[j] && witness[j] <= xu[j], "witness in box");
    }

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    assert_ne!(
        sol.status,
        QpStatus::Infeasible,
        "feasible QP (witness clears row 0 by {:.3} and satisfies the \
         equality exactly) must never be certified Infeasible",
        row(0) - bl[0]
    );

    // Stronger post-fix contract: the convex feasibility phase-1 hands
    // the recovery a usable seed and phase-2 converges from it.
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal, got {}",
        sol.status
    );

    let ax: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|j| HS071_STEP_QP_A[i * n + j] * sol.x[j]).sum())
        .collect();
    let tol = QpOptions::default().feas_tol;
    assert!(
        ax[0] >= bl[0] - tol,
        "row 0 violated: {} < {}",
        ax[0],
        bl[0]
    );
    assert!(
        (ax[1] - bl[1]).abs() <= tol,
        "equality row violated by {:.3e}",
        (ax[1] - bl[1]).abs()
    );
    for j in 0..n {
        assert!(
            sol.x[j] >= xl[j] - tol && sol.x[j] <= xu[j] + tol,
            "x[{j}] = {} outside [{}, {}]",
            sol.x[j],
            xl[j],
            xu[j]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Penalty bias read as an infeasibility certificate (gh#484 follow-up,
// round 2). Found by the property-based probe in `adversary/fuzz/`.
//
// A feasible QP — the witness below satisfies every row and bound in
// exact float arithmetic — with an indefinite `H`, two free variables,
// and a near-parallel row pair. The first fix for the false-certificate
// defect refuted a certificate only when its convex feasibility phase-1
// landed within `feas_tol` of the rows, and here it cannot: the phase-1
// minimizes `½‖x − r‖² + γ‖v‖₁`, whose proximal term competes with the
// penalty, so its optimum carries a residual up to `‖x̂ − r‖²/(2γ)`.
// With `γ = 1e6` and a box of this size that ceiling is ~1e-5 — four
// orders above `feas_tol`. The phase-1 stopped a few 1e-6 short, was
// judged "not feasible", and the certificate went out.
//
// The bias is now computed rather than tripped over: a residual is only
// allowed to certify once it exceeds `D²/(2γ)`, and γ is escalated until
// either the residual clears that bar or the point clears `feas_tol`.
#[rustfmt::skip]
const BIAS_QP_G: [f64; 3] = [
    -0.7213796209979861, -7.231392669305645, 4.234973671720205,
];
#[rustfmt::skip]
const BIAS_QP_A: [f64; 9] = [
    -0.7911384941828317, 0.03702927627662023, 2.095070462604718,
     1.934754937738557,  2.8782196768168857,  0.7027825799349712,
     1.9935114267903111, 2.9656260009633773,  0.7241229757025899,
];
/// Feasibility witness. Satisfies every row and bound exactly.
#[rustfmt::skip]
const BIAS_QP_WITNESS: [f64; 3] = [
    1.1096278255655565, 0.09990985992069845, -2.530165115509102,
];

#[test]
fn penalty_bias_is_not_an_infeasibility_certificate() {
    let n = 3usize;
    let m = 3usize;

    // Indefinite H with a zero diagonal — the exact-∇²L shape.
    let mut h_irows = Vec::new();
    let mut h_jcols = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            h_irows.push(i as i32 + 1);
            h_jcols.push(j as i32 + 1);
        }
    }
    let h_space = SymTMatrixSpace::new(n as i32, h_irows, h_jcols);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[-1.7, 2.4, 0.0, -0.9, 1.3, 3.1]);

    let mut a_irows = Vec::new();
    let mut a_jcols = Vec::new();
    for i in 0..m {
        for j in 0..n {
            a_irows.push(i as i32 + 1);
            a_jcols.push(j as i32 + 1);
        }
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, a_irows, a_jcols);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&BIAS_QP_A);

    let g = BIAS_QP_G.to_vec();
    // Row 1 is an equality; rows 0 and 2 are one- and two-sided. Two of
    // the three variables are free — the case where the box gives no
    // bound on how far a feasible point can sit.
    let bl = vec![-8.239786780104733, 0.6562644717578805, 0.6762003356215169];
    let bu = vec![-4.110301012358298, 0.6562644717578805, NLP_UPPER_BOUND_INF];
    let xl = vec![-5.64448797643853, -1.7243973211195898, NLP_LOWER_BOUND_INF];
    let xu = vec![1.1096278255655565, NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF];

    // The instance's feasibility is arithmetic, not an assumption.
    for i in 0..m {
        let ax: f64 = (0..n)
            .map(|j| BIAS_QP_A[i * n + j] * BIAS_QP_WITNESS[j])
            .sum();
        assert!(
            ax >= bl[i] - 1e-12 && (bu[i] >= NLP_UPPER_BOUND_INF || ax <= bu[i] + 1e-12),
            "witness must satisfy row {i}: {ax} not in [{}, {}]",
            bl[i],
            bu[i]
        );
    }
    for j in 0..n {
        assert!(
            BIAS_QP_WITNESS[j] >= xl[j] && BIAS_QP_WITNESS[j] <= xu[j],
            "witness must be in the box at {j}"
        );
    }

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    assert_ne!(
        sol.status,
        QpStatus::Infeasible,
        "feasible QP certified Infeasible — a phase-1 residual bounded by \
         the penalty bias is not a Farkas certificate"
    );
}

// ─────────────────────────────────────────────────────────────────────
// The cold fast paths returned `Optimal` at infeasible points.
//
// `solve` has a feasibility audit (M5) for solves that converge to a
// constraint-violating point and label it `Optimal` — but it guarded only
// the `solve_general` branch. The three cold fast paths at the bottom
// (`solve_equality_only`, `solve_box_constrained`,
// `solve_equality_plus_bounds`) returned straight to the caller, and
// nothing else checked them.
//
// The smallest possible infeasible QP falls into exactly that gap:
// `aᵀx = c₁` and `aᵀx = c₂` with `c₁ ≠ c₂` is all-equality, so it has no
// general inequality and takes no warm start, and `solve` routes it to
// `solve_equality_plus_bounds` — which came back `Optimal` at a point
// violating both rows by 2.9, at every tolerance from 1e-9 to 1e-2.
//
// Data is the exact instance the property-based probe in `adversary/fuzz/`
// found. Hand-built equivalents do *not* reproduce it: the free variable
// (`x₀` unbounded below) and the indefinite `H` are both load-bearing, and
// a tidy bounded version with a PSD Hessian passes on the unfixed solver.
#[test]
fn inconsistent_equalities_are_not_reported_optimal() {
    let n = 2usize;
    let m = 2usize;

    let h_space = SymTMatrixSpace::new(n as i32, vec![1, 2, 2], vec![1, 1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[-2.7913153552489676, -0.9863948932143245, 0.0]);
    let g = vec![-5.466878209095613, 8.48954475911776];

    // Both rows are the same functional. It cannot take two values.
    let row = [-2.5934781082241516_f64, 1.957129692504897];
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, vec![1, 1, 2, 2], vec![1, 2, 1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[row[0], row[1], row[0], row[1]]);

    let bl = vec![-5.590699548529652_f64, 0.20152723379558513];
    let bu = bl.clone();
    let xl = vec![NLP_LOWER_BOUND_INF, -2.3141811629330657];
    let xu = vec![3.5443427682446997_f64, 3.3442587552272114];

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();

    // The point is that `Optimal` must not be claimed at a point that
    // violates the problem. A non-committal status is honest here;
    // certifying `Infeasible` would be better still. Claiming a solution
    // is the one thing that is wrong.
    if sol.status == QpStatus::Optimal {
        let res: Vec<f64> = (0..m)
            .map(|i| row[0] * sol.x[0] + row[1] * sol.x[1] - bl[i])
            .collect();
        panic!(
            "returned Optimal on an inconsistent equality system; \
             row residuals {res:?} at x = {:?}",
            sol.x
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// A rank-deficient equality block escaped as a hard error.
//
// `cold_general_initial` prunes a rank-deficient equality set to an
// independent subset and retries. The prune was single-shot, and
// `independent_active_subset` is a *numerical* rank test whose answer
// depends on the shift the factorization settled at — so the pruned subset
// could itself be rejected at the next δ. Here it was: four equality rows
// pruned to two, and the retry's own masked-deficiency guard then found
// only one of those two independent. The retry's `?` propagated
// `LinearSolverFailure("pinned KKT constraint block is rank-deficient …")`
// to the caller — a hard error where the elastic path had a perfectly good
// answer to give.
//
// Again the probe's exact instance. The 1e6 row scaling and the free `x₀`
// are what push the rank test into disagreeing with itself; a uniformly
// scaled, fully bounded version does not reproduce it.
#[test]
fn repeatedly_rank_deficient_equalities_do_not_error() {
    let n = 5usize;
    let m = 4usize;

    let mut h_irows = Vec::new();
    let mut h_jcols = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            h_irows.push(i as i32 + 1);
            h_jcols.push(j as i32 + 1);
        }
    }
    let h_space = SymTMatrixSpace::new(n as i32, h_irows, h_jcols);
    let mut h = SymTMatrix::new(h_space);
    #[rustfmt::skip]
    h.set_values(&[
        -1.6548767961092978,
         2.064557047637633,  -3.650251250460926,
         0.0,                -2.55214992299658,   0.0,
         0.0,                -2.6125066492612103, 0.0, 2.686020283211974,
        -2.702216601033217,  -0.3677125985327869, 2.1594310219402475,
         1.1562910960930806,  0.0,
    ]);

    // Rows 0 and 2 are identical; row 1 is scaled ~1e6 against the rest.
    #[rustfmt::skip]
    let vals: [f64; 20] = [
         2.4098251714297536, 1.4030741454122708, 1.2932054542069062,
        -2.6693323747422992, 0.19214071677022115,
        -1772507.6707250883, 952702.1423816633,  2701610.653564656,
         2181005.4733584146, 957385.1779632831,
         2.4098251714297536, 1.4030741454122708, 1.2932054542069062,
        -2.6693323747422992, 0.19214071677022115,
        -2.5443770494332574, -1.8411314201953106, -1.4152153780449328,
         2.3312007522090887, -2.6027526355747406,
    ];
    let mut a_irows = Vec::new();
    let mut a_jcols = Vec::new();
    for i in 0..m {
        for j in 0..n {
            a_irows.push(i as i32 + 1);
            a_jcols.push(j as i32 + 1);
        }
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, a_irows, a_jcols);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&vals);

    let bl = vec![
        84.05103893366257_f64,
        2663155354.6247716,
        433.9344680755099,
        -1383.6721463359931,
    ];
    let bu = bl.clone();
    let xl = vec![
        NLP_LOWER_BOUND_INF,
        388.1155777186471,
        389.1225518670195,
        389.0009787098146,
        392.18204924864364,
    ];
    let xu = vec![
        NLP_UPPER_BOUND_INF,
        395.0630589179779,
        394.1037660233893,
        395.7242293003824,
        394.0346410306801,
    ];
    let g = vec![
        -5.805815991484248_f64,
        -3.5780398581700146,
        2.9453592415953995,
        5.212960644579471,
        9.266168148934472,
    ];

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut solver = new_solver();
    // The contract is that it returns *an answer*. A rank-deficient active
    // set is the solver's to prune, not the caller's to receive as a
    // linear-algebra failure.
    let sol = solver
        .solve(&qp, None, &QpOptions::default())
        .expect("rank-deficient equalities must not surface as a hard error");

    // And whatever it says, `Optimal` has to mean feasible. (Rows 0 and 2
    // are identical with different right-hand sides, so nothing can be.)
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "returned Optimal on an inconsistent equality system at x = {:?}",
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────────
// gh#971 — a consistent redundant equality row drove the elastic
// multipliers to the `±γ` penalty cap.
//
// Row 3 = row 1 + row 2, `b` consistent by construction, box `[-2, 2]`,
// rank-1 `H`. The cold homotopy's corrector falls through to l1-elastic,
// where every row has its own slack pair: the dependent row no longer makes
// anything singular, so no rank guard fires, and `λ` slides along
// `null(Aᵀ) = span(1, 1, −1)` until a row reaches `γ = 1e6`. The engine
// returned `MaxIter` with `λ = (1e6, 1e6, −1e6)` and `|Ax − b| ≈ 3e-6`.
//
// `solve_elastic` now rank-reveals the equality rows and runs on an
// independent subset — but only keeps that answer when the dropped rows
// hold at it. The second test is the reason for that condition: move row 3's
// right-hand side and the block is contradictory while its rank is the same,
// so the prune by itself would return the two-row optimum as `Optimal` at a
// point violating row 3 by 0.5.
// Data: seed 155 of the issue's `numpy.random.default_rng` generator.
fn issue_971_qp_parts(shift: f64) -> (SymTMatrix, GenTMatrix, Vec<f64>, Vec<f64>) {
    let v = [
        -1.8273390143890733_f64,
        -0.24535960128069498,
        0.18582556752464716,
        0.4180888279857321,
    ];
    let (mut hi, mut hj, mut hv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..4 {
        for j in 0..=i {
            hi.push(i as i32 + 1);
            hj.push(j as i32 + 1);
            hv.push(v[i] * v[j]);
        }
    }
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(4, hi, hj));
    h.set_values(&hv);
    #[rustfmt::skip]
    let rows: [[f64; 4]; 3] = [
        [1.8207619861986173, -2.7456402747587245, 0.7995850935329225, 0.8712283527052398],
        [0.9770683785838647, -0.9210003765043308, -0.40674360602557913, -0.10211300182846439],
        [2.797830364782482, -3.666640651263055, 0.3928414875073434, 0.7691153508767754],
    ];
    let (mut ai, mut aj, mut av) = (Vec::new(), Vec::new(), Vec::new());
    for (i, r) in rows.iter().enumerate() {
        for (j, &x) in r.iter().enumerate() {
            ai.push(i as i32 + 1);
            aj.push(j as i32 + 1);
            av.push(x);
        }
    }
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(3, 4, ai, aj));
    a.set_values(&av);
    let b = vec![
        -0.37147458000233335,
        -0.08041693825355356,
        -0.45189151825588647 + shift,
    ];
    let g = vec![
        0.41533064386674734,
        -1.389847552821441,
        -0.41502780761130875,
        0.1039154844102442,
    ];
    (h, a, b, g)
}

#[test]
fn issue_971_redundant_equality_keeps_multipliers_bounded() {
    let (h, a, b, g) = issue_971_qp_parts(0.0);
    let (xl, xu) = (vec![-2.0; 4], vec![2.0; 4]);
    let qp = QpProblem {
        n: 4,
        m: 3,
        h: &h,
        g: &g,
        a: &a,
        bl: &b,
        bu: &b,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("solve");
    assert_eq!(sol.status, QpStatus::Optimal, "λ = {:?}", sol.lambda_g);
    let lmax = sol.lambda_g.iter().fold(0.0_f64, |w, l| w.max(l.abs()));
    assert!(
        lmax < 10.0,
        "λ drifted to the elastic cap: {:?}",
        sol.lambda_g
    );
    let ax = crate::kkt::a_times_x(&a, &sol.x, 3);
    for i in 0..3 {
        assert!(
            (ax[i] - b[i]).abs() < 1e-9,
            "row {i}: |Ax − b| = {:e}",
            (ax[i] - b[i]).abs()
        );
    }
    // Clarabel's optimum for this QP.
    assert!(
        (sol.obj - -0.2499623364970595).abs() < 1e-8,
        "obj {}",
        sol.obj
    );
}

#[test]
fn issue_971_inconsistent_dependent_equality_is_not_pruned_to_optimal() {
    let (h, a, b, g) = issue_971_qp_parts(0.5);
    let (xl, xu) = (vec![-2.0; 4], vec![2.0; 4]);
    let qp = QpProblem {
        n: 4,
        m: 3,
        h: &h,
        g: &g,
        a: &a,
        bl: &b,
        bu: &b,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("solve");
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "contradictory equalities reported optimal at x = {:?}",
        sol.x
    );
}
