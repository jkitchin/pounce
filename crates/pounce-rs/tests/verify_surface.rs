//! Independent solution verification is reachable from the library facade.
//!
//! `pounce verify` argues in its own header that a checker matters because
//! the solver is "a tool an agent calls" and trust should not rest on the
//! solver's exit string. But the check lived in `crates/pounce-cli/src/`, so
//! an agent driving `pounce-rs` — exactly the case the argument describes —
//! could not reach it. These tests are that reachability, and they cover the
//! direction that matters: a wrong claim must be *rejected*.

use pounce_rs::diagnostics::{SolutionClaim, VerifyOptions, VerifyProvenance, verify_tnlp};
use pounce_rs::prelude::*;

/// min (x₀ − 1)² + (x₁ − 2)²  s.t.  x₀ + x₁ == 3,  0 ≤ x ≤ 5.
/// Optimum (1, 2), where the equality is already satisfied.
struct Quad;

impl TNLP for Quad {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        b.g_l[0] = 3.0;
        b.g_u[0] = 3.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn check(claim: SolutionClaim, opts: VerifyOptions) -> pounce_rs::diagnostics::VerifyOutcome {
    let mut t = Quad;
    verify_tnlp(
        &mut t,
        &claim,
        &[],
        &[],
        &VerifyProvenance::default(),
        &opts,
    )
    .expect("the model and the claim must both be readable")
}

/// The optimum verifies: feasible, and first-order optimal once the
/// equality's dual is supplied.
#[test]
fn the_true_optimum_verifies() {
    let o = check(
        SolutionClaim {
            x: vec![1.0, 2.0],
            lambda: vec![0.0],
            ..SolutionClaim::default()
        },
        VerifyOptions::default(),
    );
    assert!(o.feasible, "x = (1, 2) satisfies x₀ + x₁ = 3 and 0 ≤ x ≤ 5");
    assert!(o.verified);
    assert_eq!(o.optimal, Some(true), "∇f vanishes at the optimum");
}

/// A claim that violates the equality is rejected. This is the whole point
/// of the checker: it re-derives the answer instead of believing it.
#[test]
fn an_infeasible_claim_is_rejected() {
    let o = check(
        SolutionClaim {
            x: vec![0.0, 0.0],
            ..SolutionClaim::default()
        },
        VerifyOptions::default(),
    );
    assert!(!o.feasible, "x₀ + x₁ = 0, not 3");
    assert!(!o.verified);
    assert!(o.max_con_violation > 2.9);
    let worst = o.worst_con.expect("the offending row must be named");
    assert_eq!(worst.index, 0);
}

/// A claim outside its declared variable bounds is rejected even though it
/// satisfies the equality — bounds are checked, not assumed.
#[test]
fn a_claim_outside_its_bounds_is_rejected() {
    let o = check(
        SolutionClaim {
            x: vec![-1.0, 4.0],
            ..SolutionClaim::default()
        },
        VerifyOptions::default(),
    );
    assert!(!o.verified, "x₀ = −1 is below its lower bound of 0");
    assert!((o.max_bound_violation - 1.0).abs() < 1e-12);
}

/// A fabricated claim carrying NaN must register an infinite violation
/// rather than collapsing to "feasible" through `f64::max`. This is the
/// adversarial case the checker exists for.
#[test]
fn a_nan_claim_cannot_pass_the_feasibility_gate() {
    let o = check(
        SolutionClaim {
            x: vec![Number::NAN, 2.0],
            ..SolutionClaim::default()
        },
        VerifyOptions::default(),
    );
    assert!(!o.feasible, "a NaN primal is not a point at all");
    assert!(!o.verified);
    assert!(o.max_bound_violation.is_infinite());
}

/// `require_optimal` is the stricter gate: a feasible but non-stationary
/// point passes feasibility and fails verification.
#[test]
fn require_optimal_rejects_a_feasible_but_suboptimal_point() {
    let claim = SolutionClaim {
        x: vec![0.0, 3.0],
        lambda: vec![0.0],
        ..SolutionClaim::default()
    };
    let lenient = check(claim.clone(), VerifyOptions::default());
    assert!(lenient.feasible, "0 + 3 = 3 and both are within bounds");
    assert!(lenient.verified, "feasibility alone is the default gate");

    let strict = check(
        claim,
        VerifyOptions {
            require_optimal: true,
            ..VerifyOptions::default()
        },
    );
    assert!(strict.feasible);
    assert_eq!(strict.optimal, Some(false), "∇f does not vanish at (0, 3)");
    assert!(!strict.verified, "--require-optimal must reject it");
}

/// A claim of the wrong length is an error, not a silent verdict.
#[test]
fn a_wrong_length_claim_is_an_error() {
    let mut t = Quad;
    let err = verify_tnlp(
        &mut t,
        &SolutionClaim {
            x: vec![1.0],
            ..SolutionClaim::default()
        },
        &[],
        &[],
        &VerifyProvenance::default(),
        &VerifyOptions::default(),
    )
    .expect_err("a 1-value claim for a 2-variable problem must be refused");
    assert!(
        err.contains("1 primal values") && err.contains("2 variables"),
        "{err}"
    );
}
