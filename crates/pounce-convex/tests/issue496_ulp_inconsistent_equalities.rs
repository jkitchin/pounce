//! gh#496 — a false primal-infeasible certificate on rank-deficient
//! equality rows that are consistent to within floating-point rounding.
//!
//! Two equality rows on one column, each a singleton, are redundant: both
//! pin `x0`. Presolve fixes `x0` from the first and substitutes it into the
//! second, which empties to `0 = residual`. When the two right-hand sides
//! were *computed* rather than typed — a duplicated balance, an alias plus
//! its defining equation, a unit conversion stated twice — that residual is
//! a rounding artifact of size ~1e-17, not a contradiction. Testing it
//! against exact zero rejected the model at one ULP, at iteration 0, with a
//! confident infeasibility certificate.
//!
//! The check now scales with the magnitude of the terms that cancelled, so
//! rounding-level residuals pass and real conflicts still fail.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn with_presolve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_with_presolve(prob, |reduced| {
        solve_qp_ipm(reduced, &QpOptions::default(), backend)
    })
}

/// `min −x0 s.t. a0·x0 = b0, a1·x0 = b1, −10 ≤ x0 ≤ 10`.
fn two_row_lp(a0: f64, b0: f64, a1: f64, b1: f64) -> QpProblem {
    QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![-1.0],
        a: vec![Triplet::new(0, 0, a0), Triplet::new(1, 0, a1)],
        b: vec![b0, b1],
        g: vec![],
        h: vec![],
        lb: vec![-10.0],
        ub: vec![10.0],
    }
}

/// The exact instance from the issue: the two rows imply values of `x0`
/// that differ by 6.94e-18 — one ULP at that magnitude.
const A0: f64 = -2.830268;
const B0: f64 = 0.13596324445199998;
const A1: f64 = 2.470924;
const B1: f64 = -0.11870071803600002;

/// The reduction must not fire. `x0 = b0/a0` satisfies both rows to ~1e-17,
/// so the model is feasible and the LP has a finite optimum.
#[test]
fn ulp_inconsistent_redundant_equalities_are_feasible() {
    let prob = two_row_lp(A0, B0, A1, B1);

    // The two rows really are inconsistent in exact arithmetic — this is
    // the rounding-level disagreement the old check rejected, not a
    // problem that happens to be exactly consistent.
    let x_from_row0 = B0 / A0;
    let x_from_row1 = B1 / A1;
    assert_ne!(x_from_row0, x_from_row1, "instance lost its 1-ULP gap");
    assert!((x_from_row0 - x_from_row1).abs() < 1e-16);

    assert!(
        !matches!(presolve(&prob), PresolveOutcome::Infeasible),
        "presolve wrongly certified infeasibility at 1 ULP (gh#496)"
    );

    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "obj={}", sol.obj);
    assert!(
        (sol.x[0] - x_from_row0).abs() < 1e-9,
        "x0={} expected ≈{x_from_row0}",
        sol.x[0]
    );
    assert!((sol.obj - (-x_from_row0)).abs() < 1e-9, "obj={}", sol.obj);
}

/// The certificate must not be recoverable by scaling either: the residual
/// tolerance is relative to the size of the terms that cancelled, so
/// multiplying both rows by 1000 (which multiplies the residual by 1000)
/// changes nothing.
#[test]
fn ulp_inconsistency_stays_feasible_under_scaling() {
    for s in [1e-3, 1.0, 1e3, 1e6] {
        let prob = two_row_lp(A0 * s, B0 * s, A1 * s, B1 * s);
        assert!(
            !matches!(presolve(&prob), PresolveOutcome::Infeasible),
            "scale {s}: presolve wrongly certified infeasibility"
        );
        assert_eq!(with_presolve(&prob).status, QpStatus::Optimal, "scale {s}");
    }
}

/// Exactly consistent variants were never affected; they must stay optimal.
#[test]
fn exactly_consistent_redundant_equalities_still_optimal() {
    for (name, prob) in [
        ("exact rhs", two_row_lp(A0, B0, A1, A1 * (B0 / A0))),
        ("duplicate row", two_row_lp(A0, B0, A0, B0)),
        ("clean integers", two_row_lp(2.0, 2.0, 3.0, 3.0)),
        ("both rhs zero", two_row_lp(A0, 0.0, A1, 0.0)),
    ] {
        assert_eq!(
            with_presolve(&prob).status,
            QpStatus::Optimal,
            "{name} regressed"
        );
    }
}

/// The widened tolerance must not blunt real detection: rows that disagree
/// by anything a solve could act on are still certified infeasible at
/// presolve time.
#[test]
fn genuinely_inconsistent_equalities_still_infeasible() {
    for gap in [1e-6, 1e-3, 1.0, 100.0] {
        let prob = two_row_lp(A0, B0, A1, B1 - gap);
        assert!(
            matches!(presolve(&prob), PresolveOutcome::Infeasible),
            "gap {gap}: real conflict went undetected"
        );
        assert_eq!(
            with_presolve(&prob).status,
            QpStatus::PrimalInfeasible,
            "gap {gap}"
        );
    }
}

/// Deterministic splitmix64, so the sweep below reproduces bit for bit.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The generator that surfaced this: build the instance *around* a known
/// feasible point, so both right-hand sides are computed (`bᵢ = aᵢ·x*`)
/// rather than typed, and their implied values of `x0` agree only to the
/// last bit or two. Before the fix this rejected roughly 1 instance in 250;
/// every one of them is feasible by construction.
#[test]
fn random_redundant_equalities_around_a_feasible_point() {
    let mut state: u64 = 987_654_321;
    let draw = |lo: f64, hi: f64, state: &mut u64| {
        // 53-bit uniform in [0,1), mapped to [lo, hi).
        let u = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    };

    for trial in 0..2000 {
        let x_star = draw(-1.0, 1.0, &mut state);
        let a0 = draw(-3.0, 3.0, &mut state);
        let a1 = draw(-3.0, 3.0, &mut state);
        if a0.abs() < 1e-3 || a1.abs() < 1e-3 {
            continue; // a near-zero pivot is a conditioning question, not this one
        }
        // Right-hand sides computed from the feasible point — the rounding
        // in each product is what makes the pair inexactly redundant.
        let prob = two_row_lp(a0, a0 * x_star, a1, a1 * x_star);

        assert!(
            !matches!(presolve(&prob), PresolveOutcome::Infeasible),
            "trial {trial}: a0={a0} a1={a1} x*={x_star} wrongly certified infeasible"
        );
        let sol = with_presolve(&prob);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "trial {trial}: a0={a0} a1={a1} x*={x_star}"
        );
        assert!(
            (sol.x[0] - x_star).abs() < 1e-9,
            "trial {trial}: x0={} expected ≈{x_star}",
            sol.x[0]
        );
    }
}

/// The inequality analogue, end to end: `x0 = b0/a0` together with an
/// inequality the same point meets only to the last bit. `0 ≤ −1.4e-17` is
/// a rounding artifact, not a violation, and the model must solve.
///
/// This shape passed before the fix too — the activity-bound pass sees the
/// row while `x0` is already fixed, and *it* has always used a tolerance
/// (`ACTIVITY_TOL`), so it drops the row as redundant before `build_rows`
/// ever looks. The `0 ≤ rhs` check in `build_rows` is the backstop for the
/// rows that reach it instead (fixings made *after* the activity pass —
/// forcing rows, dominated columns, bound tightening); it used to disagree
/// with the activity pass by rejecting at exact zero, and now agrees. So
/// this is a guard against that divergence reopening, not a reproducer.
///
/// The orientation is chosen from the residual's actual sign rather than
/// assumed — with the other one the residual lands on the nonnegative side
/// and there is nothing to reject.
#[test]
fn ulp_violated_emptied_inequality_is_feasible() {
    let x0 = B0 / A0; // the value presolve fixes from the equality row
    let residual = B1 - A1 * x0; // `a1·x0 ≤ b1` leaves `0 ≤ residual`
    assert_ne!(residual, 0.0, "instance lost its rounding residual");
    let sign = if residual < 0.0 { 1.0 } else { -1.0 };
    assert!(
        (sign * residual).abs() < 1e-16 && sign * residual < 0.0,
        "expected a rounding-level negative residual, got {}",
        sign * residual
    );

    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![-1.0],
        a: vec![Triplet::new(0, 0, A0)],
        b: vec![B0],
        g: vec![Triplet::new(0, 0, sign * A1)],
        h: vec![sign * B1],
        lb: vec![-10.0],
        ub: vec![10.0],
    };
    assert!(
        !matches!(presolve(&prob), PresolveOutcome::Infeasible),
        "emptied inequality wrongly certified infeasible at 1 ULP"
    );
    assert_eq!(with_presolve(&prob).status, QpStatus::Optimal);
}
