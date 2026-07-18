//! Soundness under a **misidentified active set**.
//!
//! The emitter picks the active set from a *float* slack (`QpInput::active_tol`),
//! which is a guess. The question this file settles: can a wrong guess ever
//! yield a **passing certificate for a non-optimal point**, or only a refusal?
//!
//! If the former, it is a soundness bug that outranks everything else in the
//! certificate pipeline — a green `lake build` would then attest a false claim.
//!
//! The argument that it cannot: the guess only *proposes* an active set. What
//! is then verified, exactly over ℚ, is the full KKT system —
//!
//!   * `λ ≥ 0`                       (dual sign)
//!   * `A x ≥ b`                     (primal feasibility)
//!   * `λᵢ · slackᵢ = 0`             (complementarity)
//!   * `E x = d`                     (equality residual)
//!   * `Q x + c = Aᵀλ + Eᵀμ`         (stationarity)
//!
//! — plus `Q ⪰ 0` via the `LDLᵀ` witness. Those are precisely the hypotheses of
//! `global_min_of_kkt` on the Lean side. For a convex QP they are *sufficient*
//! for global optimality, so any certificate that clears all of them is sound
//! no matter how its active set was chosen. A bad guess can only produce a KKT
//! system whose exact solution fails one of the gates.
//!
//! These tests check that empirically rather than trusting the argument.
//!
//! # Verdict
//!
//! **No.** Across a sweep of `active_tol` from `0` to `f64::MAX` and of
//! perturbed float solutions, every outcome is either a refusal or a
//! certificate for the exact true optimum. A misidentified active set costs
//! availability, never soundness.
//!
//! # These tests were mutation-tested
//!
//! Passing assertions prove nothing on their own — this pipeline has already
//! produced two tests that could not fail (`certify_refuses_off_slice`, and a
//! drift check whose `sed` mutation silently never applied). So the guards were
//! deliberately broken to confirm these tests notice:
//!
//! | Mutation | Result |
//! |---|---|
//! | disable `RefineError::InactiveViolated` | tests still pass |
//! | disable `RefineError::NegativeDual` | tests still pass |
//! | disable **both** that and `SelfCheck("primal feasibility")` | **2 tests fail**, reporting `x₀ = 0` (the infeasible unconstrained minimum) |
//!
//! The first two surviving is not vacuity — it is **defense in depth**. The
//! emitter checks KKT twice, independently: once inside `refine.rs` as the
//! exact solve proceeds, and again in `emit.rs` as a final self-check gate over
//! the assembled certificate. Removing either leaves the other standing.
//! Removing both lets an infeasible point through, and these tests catch it.

#![allow(clippy::unwrap_used)]

use num_rational::BigRational;
use num_traits::One;
use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_certificate};

const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: ZERO_HASH.to_string(),
        sol_sha256: ZERO_HASH.to_string(),
        solver: "pounce 0.9.0".to_string(),
    }
}

fn half(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

/// `min x₀² + x₁²  s.t.  x₀ + x₁ ≥ 1,  x₀ ≥ -5`.
///
/// True optimum `x* = (1/2, 1/2)`, `f = 1/2`, with **only the first constraint
/// active**. The second is slack by 5.5 at the optimum, which makes it exactly
/// the row a too-generous `active_tol` will wrongly pull into the active set.
fn two_constraint_qp(x_float: Vec<f64>, active_tol: f64) -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![0.0, 0.0],
        constant: 0.0,
        constraints: vec![
            LinearConstraint {
                name: "c0".to_string(),
                coeffs: vec![1.0, 1.0],
                lower: 1.0,
                upper: f64::INFINITY,
            },
            LinearConstraint {
                name: "c1".to_string(),
                coeffs: vec![1.0, 0.0],
                lower: -5.0,
                upper: f64::INFINITY,
            },
        ],
        var_lower: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float,
        active_tol,
    }
}

/// The one invariant that matters: **if a certificate is emitted at all, its
/// candidate is the true optimum.** Anything else is a soundness failure.
fn assert_sound(input: &QpInput, label: &str) {
    match emit_certificate(input, &meta()) {
        // Refusal is always acceptable — the emitter is allowed to give up.
        Err(_) => {}
        Ok(cert) => {
            let x0 = cert.candidate.x[0].inner().clone();
            let x1 = cert.candidate.x[1].inner().clone();
            let obj = cert.candidate.objective.inner().clone();
            assert_eq!(
                x0,
                half(1, 2),
                "{label}: emitted a certificate with x₀ = {x0}, expected 1/2 — \
                 a passing certificate for a non-optimal point is a SOUNDNESS BUG"
            );
            assert_eq!(x1, half(1, 2), "{label}: emitted x₁ = {x1}, expected 1/2");
            assert_eq!(obj, half(1, 2), "{label}: emitted objective = {obj}");
            assert_eq!(cert.verdict, "global-min", "{label}: unexpected verdict");
        }
    }
}

#[test]
fn misidentified_active_set_never_certifies_a_non_optimal_point() {
    // Sweep active_tol over many orders of magnitude, from "nothing is active"
    // to "everything is active", including the values that wrongly capture c1.
    let tols = [
        0.0,
        1e-15,
        1e-12,
        1e-9,
        1e-7,
        1e-4,
        1e-2,
        0.5,
        1.0,
        5.4,
        5.5,
        5.6,
        1e3,
        1e6,
        f64::MAX,
    ];
    for tol in tols {
        let input = two_constraint_qp(vec![0.5, 0.5], tol);
        assert_sound(&input, &format!("active_tol = {tol:e}"));
    }
}

#[test]
fn perturbed_float_solution_never_certifies_a_non_optimal_point() {
    // A bad x_float shifts which rows *look* active. Perturbations range from
    // solver-grade noise to grossly wrong points.
    let deltas = [
        0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.4, 1.0, 5.0, -0.3, -1.0, -6.0,
    ];
    for d in deltas {
        for tol in [0.0, 1e-9, 1e-7, 1e-3, 1.0, 1e6] {
            let input = two_constraint_qp(vec![0.5 + d, 0.5 - d], tol);
            assert_sound(&input, &format!("delta = {d:e}, active_tol = {tol:e}"));
        }
        // Asymmetric perturbation: moves off the constraint surface entirely.
        for tol in [0.0, 1e-7, 1.0, 1e6] {
            let input = two_constraint_qp(vec![0.5 + d, 0.5 + d], tol);
            assert_sound(&input, &format!("both +{d:e}, active_tol = {tol:e}"));
        }
    }
}

/// The specific misidentification the problem was built to invite: a tolerance
/// large enough to call the slack row `c1` active. The exact KKT solve with
/// both rows active forces `x₀ = -5`, hence `x₁ = 6`, whose stationarity
/// demands a negative multiplier — so the dual-sign gate must reject it.
#[test]
fn over_generous_tolerance_is_refused_not_silently_wrong() {
    let input = two_constraint_qp(vec![0.5, 0.5], 1e6);
    let got = emit_certificate(&input, &meta());
    assert!(
        got.is_err(),
        "active_tol = 1e6 marks the slack row active; the resulting KKT point \
         is (-5, 6), not the optimum — this must be refused, got {got:?}"
    );
}

/// `active_tol = 0` does **not** mean "no row is active".
///
/// This surprised the first version of this test. At `x_float = (0.5, 0.5)` the
/// row `x₀ + x₁ ≥ 1` has float slack exactly `0.0`, and `|0.0| ≤ 0.0` holds, so
/// the row is still correctly identified and the emitter produces the right
/// certificate. Zero tolerance is fine whenever the float point lands exactly
/// on the constraint surface — which, for an active row, is the common case.
#[test]
fn zero_tolerance_still_finds_a_row_the_float_point_sits_exactly_on() {
    let input = two_constraint_qp(vec![0.5, 0.5], 0.0);
    let cert = emit_certificate(&input, &meta())
        .expect("slack is exactly 0.0 here, so c0 is still active at tol = 0");
    assert_eq!(cert.candidate.x[0].inner().clone(), half(1, 2));
    assert_eq!(cert.witnesses.active_set, vec![0]);
}

/// The genuine too-tight case: a float point strictly *off* the constraint, so
/// no row is within tolerance. The exact solve then returns the unconstrained
/// minimum `(0, 0)`, which violates `x₀ + x₁ ≥ 1` — the feasibility gate must
/// catch it rather than emit a certificate for an infeasible point.
#[test]
fn empty_active_set_is_refused_not_silently_wrong() {
    let input = two_constraint_qp(vec![0.6, 0.6], 0.0);
    let got = emit_certificate(&input, &meta());
    assert!(
        got.is_err(),
        "no row is within tolerance, so the unconstrained minimum (0,0) results; \
         it is infeasible and must be refused, got {got:?}"
    );
}

/// Degenerate case: a redundant duplicate of the active row makes the KKT
/// matrix singular. That must surface as a refusal, not a panic and not a
/// certificate — an exact linear solve returning `None` is a normal outcome.
#[test]
fn redundant_active_rows_are_refused_not_panicked() {
    let mut input = two_constraint_qp(vec![0.5, 0.5], 1e-7);
    input.constraints.push(LinearConstraint {
        name: "c0_dup".to_string(),
        coeffs: vec![1.0, 1.0],
        lower: 1.0,
        upper: f64::INFINITY,
    });
    // Whatever happens, it must not be an unsound certificate.
    assert_sound(&input, "duplicated active row");
}

/// Sanity anchor: with a sane tolerance the emitter *does* succeed, so the
/// tests above are not vacuously passing on universal refusal. Without this,
/// an emitter that rejected everything would satisfy every assertion here —
/// the same "test that cannot fail" trap found elsewhere in this pipeline.
#[test]
fn the_well_posed_case_still_succeeds() {
    let input = two_constraint_qp(vec![0.5, 0.5], 1e-7);
    let cert = emit_certificate(&input, &meta())
        .expect("the correctly-identified active set must still produce a certificate");
    assert_eq!(cert.candidate.x[0].inner().clone(), half(1, 2));
    assert_eq!(cert.candidate.x[1].inner().clone(), half(1, 2));
    // And the dual on the active row is exactly 1.
    assert_eq!(cert.witnesses.duals[0].inner().clone(), BigRational::one());
}
