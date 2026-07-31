//! gh #413 — the parametric homotopy must not hand back a prediction built on a
//! point that has fallen off the feasible set, and the driver must still have a
//! working cold start when it doesn't.
//!
//! #412 turned the §4.2 homotopy on by default and, in the same change, turned
//! `pounce-convex`'s simplex phase-1 vertex seed *off* whenever it is on. The
//! justification was the homotopy module's own claim that `x(t)` is feasible for
//! the `t`-problem "at every point on the path by construction".
//!
//! It is not. The path's primal ratio test can only *prevent* a violation — a
//! row whose gap has already gone negative yields a negative `dt`, which the
//! test discards — so a row it fails to cap stays violated for the remainder of
//! the path and drifts further out. Measured on Maros-Mészáros `QSHARE2B`, 14
//! rows were crossed uncapped on a single path (10 of them because the
//! rank-repair tabu hides a row from the ratio test, 4 because two events
//! coincide in `t`), and the worst grew `8e-2 -> 0.4 -> 7.5 -> 11 -> 22` on the
//! way to `t = 1`.
//!
//! The damage was not a slow solve, it was a *seedless* one: the working set at
//! `t = 1` pinned an infeasible vertex, the corrector's warm-start pre-check
//! rejected it, and with no simplex seed left to fall back on the engine
//! cold-started the l1-elastic phase-1 that `pounce-qp` documents as not
//! terminating on this problem family. `QSHARE2B` burned its whole iteration
//! budget to return `4854` (published optimum `11703.7`) still carrying a
//! constraint violation of `20`; with the seed it solves in 52 iterations.
//!
//! These tests cover the three pieces of the repair that are reachable without
//! the benchmark set:
//!
//! * an infeasible QP is still certified `PrimalInfeasible` on the *seeded*
//!   route — which it was not, because `cold_general_initial` never re-checked
//!   an equality row its own rank guard had pruned;
//! * a degenerate QP with a rank-deficient active set — the geometry that makes
//!   the path abandon — is still solved, and solved correctly;
//! * a solve that reports `Optimal` reports a genuinely feasible point.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_active_set,
    solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve_with(prob: &QpProblem, engine: &ActiveSetOverrides) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set(prob, &QpOptions::default(), engine, &mut mk)
}

fn solve(prob: &QpProblem) -> QpSolution {
    solve_with(prob, &ActiveSetOverrides::default())
}

/// Worst violation of `a·x` against `[b, b]` rows, `G x <= h` rows, and the box.
fn max_violation(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut worst: f64 = 0.0;
    let mut ax = vec![0.0; prob.b.len()];
    for t in &prob.a {
        ax[t.row] += t.val * x[t.col];
    }
    for (i, &axi) in ax.iter().enumerate() {
        worst = worst.max((axi - prob.b[i]).abs());
    }
    let mut gx = vec![0.0; prob.h.len()];
    for t in &prob.g {
        gx[t.row] += t.val * x[t.col];
    }
    for (i, &gxi) in gx.iter().enumerate() {
        worst = worst.max(gxi - prob.h[i]);
    }
    for (j, &xj) in x.iter().enumerate() {
        worst = worst.max(prob.lb[j] - xj).max(xj - prob.ub[j]);
    }
    worst
}

/// `x₀ + x₁ = 1` together with `x₀ + x₁ = 3`, solved on the route the #413
/// seeded retry uses — the homotopy explicitly off.
///
/// The two rows are linearly *dependent* and mutually *inconsistent*. Pinning
/// both makes the KKT singular, so `cold_general_initial`'s rank guard prunes
/// one and recovers a point satisfying the survivor. Its feasibility sweep then
/// skipped every `bl == bu` row outright — on the reasoning that a pinned
/// equality is satisfied by construction, which is true only for the rows that
/// were actually *kept*. The pruned row, violated by 2, was never looked at, so
/// the caller ran phase-2 on an infeasible iterate and reported the model as a
/// solver failure instead of routing to the elastic phase-1 that certifies it
/// infeasible.
///
/// This was masked for as long as the homotopy always reached `t = 1`: the
/// corrector's own pre-check caught the bad point and sent it to elastic. It
/// surfaced the moment the path started abandoning early.
#[test]
fn contradictory_equalities_are_infeasible_on_the_seeded_route() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![1.0, 3.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };

    let seeded = ActiveSetOverrides {
        use_homotopy: Some(false),
        ..Default::default()
    };
    assert_eq!(
        solve_with(&prob, &seeded).status,
        QpStatus::PrimalInfeasible,
        "homotopy off: a pruned, inconsistent equality must route to elastic"
    );
    // And on the default route, which is what #412 shipped.
    assert_eq!(solve(&prob).status, QpStatus::PrimalInfeasible);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

/// A degenerate QP whose optimal active set is rank-deficient: four constraints
/// meet at the single feasible point `(1, 1)`, two of them duplicates.
///
/// This is the geometry that drives the path into its rank-repair branch — the
/// branch whose tabu is what hides rows from the primal ratio test. Whatever the
/// path decides to do here, the driver must still return the right answer.
#[test]
fn degenerate_rank_deficient_vertex_still_solves() {
    // min ½‖x − (3,3)‖²  s.t.  x₀ + x₁ ≤ 2 (×2 duplicated), x₀ ≤ 1, x₁ ≤ 1.
    // The three distinct rows all bind at (1,1), which is the unique optimum.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-3.0, -3.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, 1.0),
            Triplet::new(3, 1, 1.0),
        ],
        h: vec![2.0, 2.0, 1.0, 1.0],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };

    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "obj={}", sol.obj);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-6 && (sol.x[1] - 1.0).abs() < 1e-6,
        "expected the degenerate vertex (1,1), got {:?}",
        sol.x
    );
    assert!(
        max_violation(&prob, &sol.x) < 1e-6,
        "reported Optimal at an infeasible point (violation {:.3e})",
        max_violation(&prob, &sol.x)
    );
}

/// The contract the path guard exists to protect, stated end-to-end: an
/// `Optimal` verdict must come with a feasible point and the IPM's objective.
///
/// Run over a small ladder of degenerate / duplicated-row instances, because the
/// failure this guards against was silent — #413's engine returned `Optimal`-
/// shaped answers that were `4854` against a true `11703.7`, and it was only
/// visible by checking the point rather than the status.
#[test]
fn optimal_is_never_reported_at_an_infeasible_point() {
    // Each entry duplicates its rows a different number of times, which is what
    // pushes the active set at the optimum past full rank.
    for dup in 1..=4usize {
        let mut g = Vec::new();
        for k in 0..dup {
            g.push(Triplet::new(k, 0, 1.0));
            g.push(Triplet::new(k, 1, 2.0));
        }
        g.push(Triplet::new(dup, 0, -1.0));
        let mut h = vec![4.0; dup];
        h.push(0.0);

        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 1.0)],
            c: vec![-8.0, -3.0],
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![0.0, 0.0],
            ub: vec![5.0, 5.0],
        };

        let sol = solve(&prob);
        let ipm = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(ipm.status, QpStatus::Optimal, "IPM oracle, dup={dup}");

        if sol.status == QpStatus::Optimal {
            assert!(
                max_violation(&prob, &sol.x) < 1e-6,
                "dup={dup}: Optimal at an infeasible point (violation {:.3e})",
                max_violation(&prob, &sol.x)
            );
            assert!(
                (sol.obj - ipm.obj).abs() <= 1e-6 * (1.0 + ipm.obj.abs()),
                "dup={dup}: Optimal with obj={} against IPM {}",
                sol.obj,
                ipm.obj
            );
        }
    }
}
