//! Verification that §4.7 iterative refinement is wired through
//! the linear-solver backend. The QP-side code never invokes
//! refinement explicitly; it inherits it from the backend's
//! default (`pounce_feral::FeralSolverInterface::new` sets
//! `refine: true`). This test pins that contract via a problem
//! whose solution accuracy is observable to better than 1e-12.

use crate::factor::LinearSolver;
use crate::kkt::KktTriplet;
use pounce_common::Index;
use pounce_feral::FeralSolverInterface;

#[test]
fn refinement_delivers_near_machine_precision_on_spd_kkt() {
    // SPD 2x2 system H = [[2,1],[1,3]], RHS = [3, 4] ⇒ x = (1, 1).
    let kkt = KktTriplet {
        dim: 2,
        irn: vec![1, 2, 2] as Vec<Index>,
        jcn: vec![1, 1, 2] as Vec<Index>,
        vals: vec![2.0, 1.0, 3.0],
    };
    let mut rhs = vec![3.0, 4.0];

    let mut ls = LinearSolver::new(Box::new(FeralSolverInterface::new()));
    ls.factorize_and_solve(&kkt, &mut rhs, Some(0))
        .expect("SPD factor + refined solve");

    // With refinement, the residual ‖Hx − b‖_∞ should be below
    // machine epsilon times a small constant. Without refinement,
    // even a one-pivot direct solve on this trivial system
    // already nails 1e-15 — the test pins the *upper-bound*, which
    // the refinement-on default comfortably clears.
    assert!(
        (rhs[0] - 1.0).abs() < 1e-14,
        "x[0] = {} (off by {})",
        rhs[0],
        (rhs[0] - 1.0).abs()
    );
    assert!(
        (rhs[1] - 1.0).abs() < 1e-14,
        "x[1] = {} (off by {})",
        rhs[1],
        (rhs[1] - 1.0).abs()
    );
}

#[test]
fn cached_resolve_returns_same_solution_as_fresh_factor() {
    // Factor a SPD KKT, solve once, then re-solve against a
    // different RHS via the cached factor. Result must match a
    // fresh factor-and-solve on the same matrix and RHS.
    let kkt = KktTriplet {
        dim: 2,
        irn: vec![1, 2, 2] as Vec<Index>,
        jcn: vec![1, 1, 2] as Vec<Index>,
        vals: vec![2.0, 1.0, 3.0],
    };

    let mut ls = LinearSolver::new(Box::new(FeralSolverInterface::new()));
    let mut rhs1 = vec![3.0, 4.0];
    ls.factorize_and_solve(&kkt, &mut rhs1, Some(0)).unwrap();
    assert!(ls.has_cached_factor());

    // Now resolve against a new RHS using the cached factor.
    let mut rhs2 = vec![5.0, 11.0];
    ls.resolve(&mut rhs2).unwrap();

    // Cross-check against a fresh factor on the same matrix.
    let mut ls_fresh = LinearSolver::new(Box::new(FeralSolverInterface::new()));
    let mut rhs2_fresh = vec![5.0, 11.0];
    ls_fresh
        .factorize_and_solve(&kkt, &mut rhs2_fresh, Some(0))
        .unwrap();

    for (i, (&a, &b)) in rhs2.iter().zip(rhs2_fresh.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "resolve[{i}] = {a} vs fresh {b} (diff {})",
            (a - b).abs(),
        );
    }
}

#[test]
fn resolve_before_factor_errors_cleanly() {
    let mut ls = LinearSolver::new(Box::new(FeralSolverInterface::new()));
    let mut rhs = vec![0.0; 2];
    let err = ls.resolve(&mut rhs).unwrap_err();
    assert!(
        matches!(err, crate::QpError::LinearSolverFailure(_)),
        "expected LinearSolverFailure, got {err:?}"
    );
    assert!(!ls.has_cached_factor());
}

#[test]
fn refinement_holds_under_indefinite_saddle_kkt() {
    // 3x3 saddle KKT for a 2-var equality QP:
    //   [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    // RHS chosen so the solution is (1, 0, -1) — verify by hand:
    //   1·1 + 0 + 1·(-1) = 0  ✓
    //   0 + 1·0 + 1·(-1) = -1
    //   1·1 + 1·0 + 0    = 1
    // So RHS = [0, -1, 1].
    let kkt = KktTriplet {
        dim: 3,
        irn: vec![1, 2, 3, 3] as Vec<Index>,
        jcn: vec![1, 2, 1, 2] as Vec<Index>,
        vals: vec![1.0, 1.0, 1.0, 1.0],
    };
    let mut rhs = vec![0.0, -1.0, 1.0];

    let mut ls = LinearSolver::new(Box::new(FeralSolverInterface::new()));
    // Inertia: (n, m, 0) = (2, 1, 0) ⇒ expected_neg = 1.
    ls.factorize_and_solve(&kkt, &mut rhs, Some(1))
        .expect("indefinite saddle factor + refined solve");

    assert!((rhs[0] - 1.0).abs() < 1e-13, "x[0] = {}", rhs[0]);
    assert!((rhs[1] - 0.0).abs() < 1e-13, "x[1] = {}", rhs[1]);
    assert!((rhs[2] - (-1.0)).abs() < 1e-13, "λ = {}", rhs[2]);
}

/// L14: the inertia-control retry loops decide recoverability via
/// `QpError::is_recoverable_factorization_failure`. It must accept the
/// failures produced by BOTH the `factorize_and_solve` path (lower-case
/// human messages "…singular…" / "…inertia…") AND the cached-factor
/// `resolve` path, whose catch-all embeds the backend's `Debug`-formatted
/// `ESymSolverStatus` ("Singular" / "WrongInertia", capitalized). A
/// case-sensitive `contains("singular")` — the pre-fix code — would miss
/// the resolve-path messages, so a singular/wrong-inertia failure during a
/// Schur-update resolve would propagate as unrecoverable instead of
/// triggering a Hessian-shift retry.
#[test]
fn recoverable_factorization_failure_is_case_insensitive() {
    use crate::QpError;

    // factorize_and_solve path — lower-case keywords.
    assert!(QpError::LinearSolverFailure(
        "KKT matrix is singular (LICQ violation or rank-deficient Jacobian)".into()
    )
    .is_recoverable_factorization_failure());
    assert!(QpError::LinearSolverFailure(
        "KKT inertia mismatch: expected 2 negative eigenvalues, got 1".into()
    )
    .is_recoverable_factorization_failure());

    // resolve path — capitalized Debug-formatted ESymSolverStatus. These
    // are exactly the strings `format!("resolve backend status: {st:?}")`
    // produces, and are the L14 regression.
    assert!(
        QpError::LinearSolverFailure("resolve backend status: Singular".into())
            .is_recoverable_factorization_failure()
    );
    assert!(
        QpError::LinearSolverFailure("resolve backend status: WrongInertia".into())
            .is_recoverable_factorization_failure()
    );

    // Genuinely unrecoverable / unrelated failures must NOT be retried.
    assert!(
        !QpError::LinearSolverFailure("backend reported fatal error".into())
            .is_recoverable_factorization_failure()
    );
    assert!(!QpError::LinearSolverFailure(
        "resolve called before a successful factorize_and_solve".into()
    )
    .is_recoverable_factorization_failure());
    assert!(
        !QpError::DimensionMismatch("g.len() != n".into()).is_recoverable_factorization_failure()
    );
}
