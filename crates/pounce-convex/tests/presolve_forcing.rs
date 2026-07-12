//! Forcing-constraint presolve: a row whose activity range touches its
//! RHS pins every involved variable to a bound. Correctness is checked by
//! verifying the postsolved `(x, y, z, z_lb, z_ub)` is a valid KKT point
//! of the *original* problem — not by comparing duals to a direct solve,
//! because a forcing constraint's multiplier is generally **not unique**
//! (it ranges over an interval), so two valid solves can report different
//! — both correct — duals. The primal of a strictly convex QP is unique,
//! so that we do compare.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const TOL: f64 = 1e-5;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> QpSolution {
    solve_with_presolve(prob, |reduced| {
        solve_qp_ipm(reduced, &QpOptions::default(), backend)
    })
}

/// Assert `sol` satisfies the KKT system of `prob` to `TOL`:
/// primal feasibility, dual feasibility (z, z_lb, z_ub ≥ 0),
/// stationarity `Px + c + Aᵀy + Gᵀz + z_ub − z_lb = 0`, and
/// complementarity on every inequality and bound.
fn assert_kkt(prob: &QpProblem, sol: &QpSolution) {
    let n = prob.n;
    let me = prob.m_eq();
    let mi = prob.m_ineq();

    // Primal feasibility.
    let mut ax = vec![0.0; me];
    prob.a_mul(&sol.x, &mut ax);
    for i in 0..me {
        assert!(
            (ax[i] - prob.b[i]).abs() < TOL,
            "Ax=b row {i}: {} vs {}",
            ax[i],
            prob.b[i]
        );
    }
    let mut gx = vec![0.0; mi];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..mi {
        assert!(
            gx[i] <= prob.h[i] + TOL,
            "Gx≤h row {i}: {} vs {}",
            gx[i],
            prob.h[i]
        );
    }
    for i in 0..n {
        assert!(
            sol.x[i] >= prob.lb_of(i) - TOL && sol.x[i] <= prob.ub_of(i) + TOL,
            "box {i}: {} in [{}, {}]",
            sol.x[i],
            prob.lb_of(i),
            prob.ub_of(i)
        );
    }

    // Dual feasibility.
    for (i, &zi) in sol.z.iter().enumerate() {
        assert!(zi >= -TOL, "z[{i}] = {zi} < 0");
    }
    for i in 0..n {
        assert!(sol.z_lb[i] >= -TOL, "z_lb[{i}] = {} < 0", sol.z_lb[i]);
        assert!(sol.z_ub[i] >= -TOL, "z_ub[{i}] = {} < 0", sol.z_ub[i]);
    }

    // Stationarity: Px + c + Aᵀy + Gᵀz + z_ub − z_lb = 0.
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        let stat = g[i] + sol.z_ub[i] - sol.z_lb[i];
        assert!(stat.abs() < TOL, "stationarity[{i}] = {stat}");
    }

    // Complementarity.
    for i in 0..mi {
        assert!(
            (sol.z[i] * (prob.h[i] - gx[i])).abs() < TOL,
            "ineq comp {i}: z={} slack={}",
            sol.z[i],
            prob.h[i] - gx[i]
        );
    }
    for i in 0..n {
        assert!(
            (sol.z_lb[i] * (sol.x[i] - prob.lb_of(i))).abs() < TOL,
            "lb comp {i}"
        );
        assert!(
            (sol.z_ub[i] * (prob.ub_of(i) - sol.x[i])).abs() < TOL,
            "ub comp {i}"
        );
    }
}

fn forcing_rows(prob: &QpProblem) -> usize {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => ps.stats().forcing_rows,
        _ => 0,
    }
}

#[test]
fn inequality_forcing_to_lower_bounds() {
    // min ½‖x‖² − 2x0 − 3x1  s.t.  x0 + x1 ≤ 0,  0 ≤ x ≤ 5.
    // min-activity of x0+x1 over the box is 0 = h ⇒ forces x0 = x1 = 0.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-2.0, -3.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![0.0],
        lb: vec![0.0, 0.0],
        ub: vec![5.0, 5.0],
    };
    assert_eq!(
        forcing_rows(&prob),
        1,
        "the row should be detected as forcing"
    );

    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.x[0].abs() < TOL && sol.x[1].abs() < TOL,
        "x pinned to 0: {:?}",
        sol.x
    );
    assert_kkt(&prob, &sol);
    // Primal matches the direct solve (unique for strictly convex P).
    let d = direct(&prob);
    assert!((sol.x[0] - d.x[0]).abs() < TOL && (sol.x[1] - d.x[1]).abs() < TOL);
    assert!(
        (sol.obj - d.obj).abs() < TOL,
        "obj {} vs {}",
        sol.obj,
        d.obj
    );
}

#[test]
fn inequality_forcing_with_mixed_signs() {
    // x0 − x1 ≤ −5 with 0 ≤ x0 ≤ 5, 0 ≤ x1 ≤ 5: min activity of x0 − x1 is
    // 0 − 5 = −5 = h ⇒ forces x0 = 0 (lower), x1 = 5 (upper).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-5.0],
        lb: vec![0.0, 0.0],
        ub: vec![5.0, 5.0],
    };
    assert_eq!(forcing_rows(&prob), 1);
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.x[0]).abs() < TOL && (sol.x[1] - 5.0).abs() < TOL,
        "x={:?}",
        sol.x
    );
    assert_kkt(&prob, &sol);
}

#[test]
fn equality_forcing_min_vertex() {
    // x0 + 2x1 = 0 with 0 ≤ x ≤ 4: min activity 0 = b ⇒ x0 = x1 = 0.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-1.0, -1.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 2.0)],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![4.0, 4.0],
    };
    assert_eq!(forcing_rows(&prob), 1);
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.x[0].abs() < TOL && sol.x[1].abs() < TOL,
        "x={:?}",
        sol.x
    );
    assert_kkt(&prob, &sol);
}

#[test]
fn equality_forcing_max_vertex() {
    // x0 + x1 = 8 with 0 ≤ x ≤ 4: max activity 4+4 = 8 = b ⇒ x0 = x1 = 4.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![1.0, 5.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![8.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![4.0, 4.0],
    };
    assert_eq!(forcing_rows(&prob), 1);
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.x[0] - 4.0).abs() < TOL && (sol.x[1] - 4.0).abs() < TOL,
        "x={:?}",
        sol.x
    );
    assert_kkt(&prob, &sol);
}

#[test]
fn overlapping_forcing_rows_resolved_by_fixpoint() {
    // Two forcing rows sharing x1: x0+x1 ≤ 0 and x1+x2 ≤ 0 (box [0,5]).
    // A single round can only fire one (disjoint-column rule); the fixpoint
    // fires the second next round once x1 is fixed — and the composed
    // postsolve recovers a valid KKT point with both rows' multipliers.
    let prob = QpProblem {
        n: 3,
        p_lower: (0..3).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: vec![-2.0, -3.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0), // x0 + x1 ≤ 0
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0), // x1 + x2 ≤ 0  (shares x1)
        ],
        h: vec![0.0, 0.0],
        lb: vec![0.0; 3],
        ub: vec![5.0; 3],
    };
    // Both rows forcing ⇒ all three variables pinned to 0.
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    for i in 0..3 {
        assert!(
            sol.x[i].abs() < 1e-6,
            "x[{i}]={} (all pinned to 0)",
            sol.x[i]
        );
    }
    assert_kkt(&prob, &sol);
}

#[test]
fn forcing_combined_with_other_rows() {
    // A forcing inequality x0 + x1 ≤ 0 (pins x0=x1=0) alongside a live
    // inequality x2 + x3 ≤ 3, on a strictly convex objective. Checks that
    // forcing coexists with kept rows and the recovered KKT is valid.
    let prob = QpProblem {
        n: 4,
        p_lower: (0..4).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: vec![-2.0, -3.0, -1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0), // forcing: x0+x1 ≤ 0
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0), // live: x2+x3 ≤ 3
        ],
        h: vec![0.0, 3.0],
        lb: vec![0.0; 4],
        ub: vec![5.0; 4],
    };
    assert_eq!(forcing_rows(&prob), 1);
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.x[0].abs() < TOL && sol.x[1].abs() < TOL,
        "forced x={:?}",
        &sol.x[..2]
    );
    assert_kkt(&prob, &sol);
    let d = direct(&prob);
    for i in 0..4 {
        assert!(
            (sol.x[i] - d.x[i]).abs() < TOL,
            "x[{i}]: {} vs {}",
            sol.x[i],
            d.x[i]
        );
    }
}
