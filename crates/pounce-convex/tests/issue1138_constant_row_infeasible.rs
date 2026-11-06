//! gh#1138: a constraint row with no `x` in it whose constant its cone
//! forbids is primal infeasible, with an exact certificate, through every
//! convex IPM entry point.
//!
//! The adversary's case is `s = h − Gx = (−1, x₁, x₂) ∈ SOC`, i.e. `‖x‖ ≤ −1`.
//! With `c = 0` it was certified; with any nonzero `c` it ended
//! `NumericalFailure` after ~155 iterations. The certificate's multiplier
//! sits on the head row, which has no `G` entry, so it never enters the
//! column magnitudes the Farkas margin measures the residual against, and
//! the margin read about 1 at every iterate. An orthant row `0 ≤ −1` failed
//! the same way, through `solve_qp_ipm` too.
//!
//! The screen proves these before the iteration. Every certificate here is
//! checked exactly: `Aᵀy + Gᵀz = 0` and `bᵀy + hᵀz < 0`. The cases that must
//! *not* fire are pinned beside them: a feasible constant head, a feasible
//! SOC whose constant tail row carries a negative `h`, a violation inside the
//! tolerance, and a non-finite constant.

use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpSolution, QpStatus, QpWarmStart, Triplet, solve_qp_ipm,
    solve_qp_ipm_warm, solve_socp_ipm, solve_socp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `s = h − Gx` with `G = [0 0; −I]`: row 0 has no `x`, rows 1–2 are `x`.
fn constant_head(c: [f64; 2], h: [f64; 3]) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![],
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(1, 0, -1.0), Triplet::new(2, 1, -1.0)],
        h: h.to_vec(),
        lb: vec![],
        ub: vec![],
    }
}

/// `Aᵀy + Gᵀz`, `bᵀy + hᵀz`.
fn farkas(prob: &QpProblem, y: &[f64], z: &[f64]) -> (Vec<f64>, f64) {
    let mut q = vec![0.0; prob.n];
    for t in &prob.a {
        q[t.col] += t.val * y[t.row];
    }
    for t in &prob.g {
        q[t.col] += t.val * z[t.row];
    }
    let v = prob.b.iter().zip(y).map(|(b, y)| b * y).sum::<f64>()
        + prob.h.iter().zip(z).map(|(h, z)| h * z).sum::<f64>();
    (q, v)
}

fn assert_certified(prob: &QpProblem, sol: &QpSolution, what: &str) {
    assert_eq!(sol.status, QpStatus::PrimalInfeasible, "{what}: {sol:?}");
    assert_eq!(sol.iters, 0, "{what}: proved before the iteration");
    let (q, v) = farkas(prob, &sol.y, &sol.z);
    assert!(q.iter().all(|&qi| qi == 0.0), "{what}: Aᵀy + Gᵀz = {q:?}");
    assert!(v < 0.0, "{what}: bᵀy + hᵀz = {v}");
}

#[test]
fn a_negative_constant_soc_head_is_infeasible_for_every_objective() {
    let cones = [ConeSpec::SecondOrder(3)];
    for c in [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [-3.0, 2.0]] {
        for h0 in [-1.0, -1e-3, -1e3] {
            let prob = constant_head(c, [h0, 0.0, 0.0]);
            let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
            assert_certified(&prob, &sol, &format!("c={c:?} h0={h0}"));
            assert_eq!(sol.z, vec![1.0, 0.0, 0.0], "z = e₀ ∈ SOC*");
        }
    }
    // The tail's own offsets do not matter: `s₀ = h₀ < 0` alone is the proof.
    let prob = constant_head([1.0, 1.0], [-1.0, 2.0, 3.0]);
    let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
    assert_certified(&prob, &sol, "tail offsets");
}

#[test]
fn a_violated_constant_orthant_row_is_infeasible_through_both_entry_points() {
    let prob = constant_head([1.0, 1.0], [-1.0, 0.0, 0.0]);
    let opts = QpOptions::default();
    let qp = solve_qp_ipm(&prob, &opts, backend);
    assert_certified(&prob, &qp, "solve_qp_ipm");
    let socp = solve_socp_ipm(&prob, &[ConeSpec::Nonneg(3)], &opts, backend);
    assert_certified(&prob, &socp, "solve_socp_ipm, orthant");
    assert_eq!(qp.z, vec![1.0, 0.0, 0.0]);
}

#[test]
fn a_violated_constant_equality_row_is_infeasible() {
    // Row 0 of `A` is empty with `b₀ = 2`: `0 = 2`.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![Triplet::new(1, 0, 1.0)],
        b: vec![2.0, 0.5],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_certified(&prob, &sol, "0 = 2");
    assert_eq!(sol.y, vec![-1.0, 0.0]);
}

#[test]
fn a_constant_soc_block_outside_the_cone_is_infeasible() {
    // `(1, 2) ∈ SOC` is false, and neither row has `x` in it. Row 2 makes
    // the problem otherwise ordinary.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(2, 0, -1.0)],
        h: vec![1.0, 2.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(2), ConeSpec::Nonneg(1)];
    let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
    assert_certified(&prob, &sol, "constant SOC (1, 2)");
    assert_eq!(sol.z, vec![1.0, -1.0, 0.0], "z on the boundary of SOC*");
}

/// Every certificate type through every entry point it can reach (#1142
/// review): the orthant row and the empty equality row through both QP and
/// SOCP entries, cold and warm; the SOC head and the all-constant SOC
/// through both SOCP entries.
#[test]
fn every_certificate_type_through_every_entry_point() {
    let opts = QpOptions::default();
    let warm_for = |p: &QpProblem| QpWarmStart {
        x: vec![0.1; p.n],
        y: vec![0.0; p.m_eq()],
        z: vec![0.5; p.m_ineq()],
        z_lb: vec![0.0; p.n],
        z_ub: vec![0.0; p.n],
    };
    let all = |prob: &QpProblem, cones: &[ConeSpec], qp_too: bool, what: &str| {
        let w = warm_for(prob);
        assert_certified(prob, &solve_socp_ipm(prob, cones, &opts, backend), what);
        assert_certified(
            prob,
            &solve_socp_ipm_warm(prob, cones, &w, &opts, backend),
            what,
        );
        if qp_too {
            assert_certified(prob, &solve_qp_ipm(prob, &opts, backend), what);
            assert_certified(prob, &solve_qp_ipm_warm(prob, &opts, &w, backend), what);
        }
    };
    // Orthant row `0 ≤ −1`.
    let orth = constant_head([1.0, 1.0], [-1.0, 0.0, 0.0]);
    all(&orth, &[ConeSpec::Nonneg(3)], true, "orthant row");
    // Empty equality row `0 = 2`, beside an ordinary orthant block.
    let eq = QpProblem {
        a: vec![Triplet::new(1, 0, 1.0)],
        b: vec![2.0, 0.5],
        ..constant_head([1.0, 1.0], [1.0, 0.0, 0.0])
    };
    all(&eq, &[ConeSpec::Nonneg(3)], true, "equality row");
    all(
        &eq,
        &[ConeSpec::SecondOrder(3)],
        false,
        "equality row, SOC present",
    );
    // SOC head `‖x‖ ≤ −1`.
    all(&orth, &[ConeSpec::SecondOrder(3)], false, "SOC head");
    // All-constant SOC `(1, 2) ∉ K`.
    let cst = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(2, 0, -1.0)],
        h: vec![1.0, 2.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    all(
        &cst,
        &[ConeSpec::SecondOrder(2), ConeSpec::Nonneg(1)],
        false,
        "all-constant SOC",
    );
}

/// The other branch: feasible models with constant rows are solved, not
/// screened.
#[test]
fn feasible_constant_rows_are_left_to_the_solve() {
    let opts = QpOptions::default();
    let soc = [ConeSpec::SecondOrder(3)];
    // `‖x‖ ≤ 1`, min x₁ + x₂ = −√2.
    let prob = constant_head([1.0, 1.0], [1.0, 0.0, 0.0]);
    let sol = solve_socp_ipm(&prob, &soc, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "{sol:?}");
    assert!((sol.obj + 2f64.sqrt()).abs() < 1e-6, "obj {}", sol.obj);
    // `‖x‖ ≤ 0`: feasible at x = 0.
    let prob = constant_head([1.0, 1.0], [0.0, 0.0, 0.0]);
    let sol = solve_socp_ipm(&prob, &soc, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "{sol:?}");

    // A constant *tail* row with a negative constant, in a feasible SOC:
    // `s = (x, −1) ∈ SOC` is `x ≥ 1`, min x = 1. Its `hᵢzᵢ` can be negative
    // at a dual point, which is why only heads and all-constant blocks are
    // read.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0)],
        h: vec![0.0, -1.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_socp_ipm(&prob, &[ConeSpec::SecondOrder(2)], &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "{sol:?}");
    assert!((sol.obj - 1.0).abs() < 1e-6, "obj {}", sol.obj);
}

/// The tolerance is presolve's empty-row one, `EMPTY_ROW_TOL (1e-9) × (1 +
/// scale)` (#1142 review): a constant at rounding level is not read as a
/// conflict, and the iteration then finds the point feasible to its own
/// tolerance. Anything clearly above it is certified at any magnitude, so
/// scaling the row does not flip the verdict (`infeas_tol`, an absolute
/// `1e-7`, let `0 ≤ −5e-8` stand down into the gh#1138 failure).
#[test]
fn the_tolerance_is_presolves_and_the_verdict_survives_scaling() {
    let opts = QpOptions::default();
    let soc = [ConeSpec::SecondOrder(3)];
    for h0 in [-5e-8, -5e-6, -2e-9] {
        let prob = constant_head([1.0, 1.0], [h0, 0.0, 0.0]);
        assert_certified(&prob, &solve_socp_ipm(&prob, &soc, &opts, backend), "socp");
        assert_certified(&prob, &solve_qp_ipm(&prob, &opts, backend), "qp");
    }
    let prob = constant_head([1.0, 1.0], [-5e-10, 0.0, 0.0]);
    let sol = solve_socp_ipm(&prob, &soc, &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "inside the tolerance: {sol:?}"
    );
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "inside the tolerance: {sol:?}"
    );
}

/// A constant head with *some* constant tail rows beside rows that depend
/// on `x` (#1142 review): `s = (1, x, 2) ∈ SOC`, i.e. `‖(x, 2)‖ ≤ 1`. The
/// certificate is `z = (1, 0, −1)`, all on rows with no `G` entry.
#[test]
fn a_constant_head_with_constant_tail_rows_is_infeasible() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(1, 0, -1.0)],
        h: vec![1.0, 0.0, 2.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_socp_ipm(
        &prob,
        &[ConeSpec::SecondOrder(3)],
        &QpOptions::default(),
        backend,
    );
    assert_certified(&prob, &sol, "‖(x, 2)‖ ≤ 1");
    assert_eq!(sol.z, vec![1.0, 0.0, -1.0]);
}

/// Extreme constants (#1142 review): the tail norm is computed scaled, so
/// `(1e10, 1e160)` gets a valid certificate rather than `z = (1, −0)` with
/// `hᵀz > 0`; and a non-finite constant is left to the iteration and its
/// finiteness guard, never certified with a NaN multiplier.
#[test]
fn extreme_constants_give_a_valid_certificate_or_none() {
    let opts = QpOptions::default();
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(2, 0, -1.0)],
        h: vec![1e10, 1e160, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(2), ConeSpec::Nonneg(1)];
    let sol = solve_socp_ipm(&prob, &cones, &opts, backend);
    assert_certified(&prob, &sol, "(1e10, 1e160)");

    let prob = constant_head([1.0, 1.0], [1.0, f64::INFINITY, 0.0]);
    let sol = solve_socp_ipm(&prob, &[ConeSpec::SecondOrder(3)], &opts, backend);
    assert_ne!(sol.status, QpStatus::PrimalInfeasible, "{sol:?}");
    assert!(sol.z.iter().all(|v| v.is_finite()), "{sol:?}");
}
