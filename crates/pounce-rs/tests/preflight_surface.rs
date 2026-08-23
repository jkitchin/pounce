//! The starting-point preflight is reachable from the library facade.
//!
//! `check_tnlp` was always generic over `&mut dyn TNLP` — no `.nl`, no CLI —
//! but it lived in `crates/pounce-cli/src/check_x0.rs`, so the only way to
//! ask "will this starting point survive iteration 0" was to be the `pounce`
//! binary. These are the three answers an embedder actually needs: the fatal
//! case, the clean case, and the silent one (the solver moved my point).

use pounce_rs::diagnostics::{PreflightOptions, X0Override, check_tnlp};
use pounce_rs::prelude::*;

/// min 1/x₀ + x₁  s.t.  x₀ + x₁ = 1, x ≥ 0.
/// With x₀ starting at 0 this is the canonical `Invalid_Number_Detected`
/// trap: the objective is not finite at the starting point.
struct DomainTrap {
    x0: Vec<Number>,
}

impl TNLP for DomainTrap {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(1.0 / x[0] + x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = -1.0 / (x[0] * x[0]);
        g[1] = 1.0;
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
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn preflight(x0: Vec<Number>) -> pounce_rs::diagnostics::PreflightOutcome {
    let mut t = DomainTrap { x0 };
    check_tnlp(
        &mut t,
        &[],
        &[],
        None,
        "library".to_string(),
        &PreflightOptions::default(),
    )
    .expect("preflight must evaluate")
}

/// A non-finite gradient at x₀ is fatal — the solve would abort with
/// `Invalid_Number_Detected`, and an embedder can now see that first.
#[test]
fn non_finite_derivative_at_x0_is_reported_fatal() {
    let o = preflight(vec![0.0, 1.0]);
    assert!(o.fatal, "1/x₀ at x₀ = 0 must be reported fatal");
    assert!(
        o.grad_nonfinite_count > 0,
        "the offending gradient entry must be named, got {:?}",
        o.grad_nonfinite
    );
}

/// A clean interior point is not fatal and reports no non-finite entries.
#[test]
fn clean_interior_point_is_not_fatal() {
    let o = preflight(vec![0.5, 0.5]);
    assert!(!o.fatal);
    assert_eq!(o.grad_nonfinite_count, 0);
    assert_eq!(o.g_nonfinite_count, 0);
    assert_eq!(o.n_bound_violations, 0);
}

/// The interior clamp moves a point sitting exactly on a bound. This is the
/// "the solver silently moved my starting point" case, and it is the one an
/// embedder has no other way to observe.
#[test]
fn a_point_on_its_bound_is_reported_as_clamped() {
    let o = preflight(vec![0.0, 1.0]);
    assert!(o.n_on_bounds > 0, "x₀ = 0 sits on its lower bound");
    assert!(
        o.n_clamp_moved > 0 && o.max_clamp_move > 0.0,
        "the bound_push clamp must be reported as moving the point"
    );
}

/// A caller-supplied starting point overrides the model's own, and the
/// override is what gets checked — the library equivalent of `--x0-file`.
#[test]
fn a_caller_supplied_x0_overrides_the_models_own() {
    let mut t = DomainTrap { x0: vec![0.5, 0.5] };
    let opts = PreflightOptions {
        x0: Some(X0Override {
            x: vec![0.0, 1.0],
            source: "caller".to_string(),
        }),
        ..PreflightOptions::default()
    };
    let o = check_tnlp(&mut t, &[], &[], None, "library".to_string(), &opts)
        .expect("preflight must evaluate");
    assert!(
        o.fatal,
        "the override, not the model's clean x0, is checked"
    );
    assert_eq!(o.x0_source, "caller");
}

/// A wrong-length override is refused rather than silently truncated.
#[test]
fn a_wrong_length_x0_override_is_refused() {
    let mut t = DomainTrap { x0: vec![0.5, 0.5] };
    let opts = PreflightOptions {
        x0: Some(X0Override {
            x: vec![0.5],
            source: "caller".to_string(),
        }),
        ..PreflightOptions::default()
    };
    let err = check_tnlp(&mut t, &[], &[], None, "library".to_string(), &opts)
        .expect_err("a 1-value override for a 2-variable problem must be refused");
    assert!(
        err.contains("caller") && err.contains("1 values") && err.contains("2 variables"),
        "the override's own label must name which starting point is wrong: {err}"
    );
}
