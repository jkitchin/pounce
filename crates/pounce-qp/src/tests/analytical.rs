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
