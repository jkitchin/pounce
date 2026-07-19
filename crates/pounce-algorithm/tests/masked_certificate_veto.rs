//! gh #200 — a termination certificate masked by an extreme objective scale.
//!
//! Gradient-based objective scaling picks `df = nlp_scaling_max_gradient /
//! max‖∇f‖`, floored at `nlp_scaling_min_value = 1e-8`. A separable quartic has
//! an enormous initial gradient and a gradient that vanishes *cubically* toward
//! its minimum, so `df` pins at the floor and stays there while the true
//! gradient collapses. The strict test runs on the scaled aggregate, and
//! crosses `tol` while the iterate is still far from the minimum: the solver
//! certifies optimality at a clearly non-optimal point.
//!
//! This is the benchmark failure (`quartc`, `dqrtic`) reduced to a
//! self-contained problem, so the regression is covered with no benchmark data.
//! Upstream Ipopt has the same behaviour, which is why the fix is opt-out-able
//! via `obj_scale_certificate_threshold = 0`.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const N: usize = 1000;

/// `f(x) = Σᵢ (xᵢ − i)⁴`, unconstrained, started at `xᵢ = 2`.
///
/// Minimum `0` at `xᵢ = i`. Separable and convex, so nothing about the geometry
/// is hard — the only difficulty is the scaling pathology. At the start the
/// largest component gradient is `4·(2 − 999)³ ≈ 4e9`, which drives `df` to the
/// 1e-8 floor.
#[derive(Default)]
struct SeparableQuartic;

impl TNLP for SeparableQuartic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as i32,
            m: 0,
            nnz_jac_g: 0,
            // Diagonal Hessian: the problem is separable.
            nnz_h_lag: N as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; N]);
        b.x_u.copy_from_slice(&[2.0e19; N]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[2.0; N]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            x.iter()
                .enumerate()
                .map(|(i, xi)| (xi - i as Number).powi(4))
                .sum(),
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (i, (gi, xi)) in g.iter_mut().zip(x).enumerate() {
            *gi = 4.0 * (xi - i as Number).powi(3);
        }
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..N {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                for (i, v) in values.iter_mut().enumerate() {
                    *v = obj_factor * 12.0 * (x[i] - i as Number).powi(2);
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve(opts: &[(&str, Number)]) -> (ApplicationReturnStatus, Number, Number, usize) {
    solve_capped(opts, None)
}

fn solve_capped(
    opts: &[(&str, Number)],
    max_iter: Option<i32>,
) -> (ApplicationReturnStatus, Number, Number, usize) {
    let mut app = IpoptApplication::new();
    for (k, v) in opts {
        app.options_mut()
            .set_numeric_value(k, *v, true, false)
            .unwrap();
    }
    if let Some(m) = max_iter {
        app.options_mut()
            .set_integer_value("max_iter", m, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeparableQuartic));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    (
        status,
        s.final_objective,
        s.final_unscaled_kkt_error,
        s.iteration_count as usize,
    )
}

/// The bug, and the fix. With the veto disabled the solver stops early and
/// calls it optimal; with it enabled the run continues to the true minimum and
/// the certificate it finally issues is honest.
#[test]
fn masked_certificate_is_refused_and_the_run_reaches_the_true_minimum() {
    let (off_status, off_obj, off_err, off_iter) =
        solve(&[("obj_scale_certificate_threshold", 0.0)]);
    let (on_status, on_obj, on_err, on_iter) = solve(&[]);

    eprintln!(
        "veto off: {off_status:?} obj={off_obj:.6e} unscaled_err={off_err:.3e} iters={off_iter}\n\
         veto on : {on_status:?} obj={on_obj:.6e} unscaled_err={on_err:.3e} iters={on_iter}"
    );

    // Guard the premise: without the veto this problem must actually exhibit
    // the bug, otherwise the test below proves nothing. It reports success at a
    // point whose objective is far from zero and whose UNSCALED KKT error is
    // enormous — the scaled one is what passed.
    assert!(
        matches!(off_status, ApplicationReturnStatus::SolveSucceeded),
        "premise: opt-out should reproduce the false certificate, got {off_status:?}"
    );
    assert!(
        off_obj > 1.0,
        "premise: opt-out should stop far from the minimum, got obj {off_obj:.6e}"
    );
    assert!(
        off_err > 1e-3,
        "premise: the refused point should be grossly non-stationary unscaled, got {off_err:.3e}"
    );

    // The fix: same problem, default options, now genuinely solved.
    assert!(
        matches!(on_status, ApplicationReturnStatus::SolveSucceeded),
        "veto run should still end in a strict certificate, got {on_status:?}"
    );
    assert!(
        on_obj < 1e-6,
        "veto run should reach the true minimum, got obj {on_obj:.6e}"
    );
    assert!(
        on_obj < off_obj,
        "veto run must not be worse than the opt-out run"
    );
    // Continuing costs iterations — that is the whole trade.
    assert!(on_iter > off_iter, "expected extra iterations to be spent");
}

/// The opt-out is bit-for-bit upstream behaviour, which is the escape hatch for
/// anyone who needs Ipopt parity (Ipopt reports this class of problem as
/// optimal at the early point too).
#[test]
fn threshold_zero_restores_the_upstream_stop() {
    let (status, obj, _, _) = solve(&[("obj_scale_certificate_threshold", 0.0)]);
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(
        obj > 1.0,
        "opt-out should keep the early stop, got {obj:.6e}"
    );
}

/// The safety property that lets the veto run without predicting whether a stop
/// is really false: if continuing does not pan out, the refused point comes
/// back with the status it would originally have had.
///
/// Capping `max_iter` below the true convergence iteration forces exactly that
/// path. The result must be the refused certificate — never a bare
/// `Maximum_Iterations_Exceeded`, and never a point worse than the one the
/// solver was about to return.
#[test]
fn a_veto_that_does_not_pan_out_restores_the_refused_certificate() {
    let (off_status, off_obj, _, off_iter) = solve(&[("obj_scale_certificate_threshold", 0.0)]);
    assert!(matches!(
        off_status,
        ApplicationReturnStatus::SolveSucceeded
    ));

    // Stop a couple of iterations after the veto would fire, well short of the
    // true minimum.
    let cap = (off_iter + 2) as i32;
    let (status, obj, _, _) = solve_capped(&[], Some(cap));
    eprintln!("capped at {cap} iters: {status:?} obj={obj:.6e} (refused point was {off_obj:.6e})");

    assert!(
        !matches!(status, ApplicationReturnStatus::MaximumIterationsExceeded),
        "a stalled veto must fall back to the refused point, not surface a bare failure"
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status after a stalled veto: {status:?}"
    );
    assert!(
        obj <= off_obj * (1.0 + 1e-9),
        "fallback point {obj:.6e} is worse than the refused point {off_obj:.6e}"
    );
}
