//! gh #200: post-optimal sensitivity must survive a masked-certificate fallback.
//!
//! `PdSensBacksolver` reuses the solver's **held factor**, so a mechanism that
//! restores an earlier iterate after the run has travelled on raises a fair
//! question: does the factor still belong to the point being returned? If not,
//! a consumer would receive a `dx/dp` that does not correspond to the solution
//! it was handed — a silently wrong answer rather than a wrong status.
//!
//! Measured: it does. On a well-posed masked problem where the veto is cut off
//! and must fall back, the sensitivity is bit-identical to the run that never
//! vetoed at all.
//!
//! One caveat worth recording, because it cost real time to chase. An earlier
//! version of this probe used an objective with a `−k·√(1+y²)` term, which is
//! **unbounded below**; the sensitivity there differed by nine orders of
//! magnitude between the two arms. That is not a defect. With no minimum, the
//! KKT system is singular along the unbounded direction and the "sensitivity"
//! is an artifact of whichever inertia-correction perturbation happened to be
//! applied — the quantity is not defined, so the two arms are not obliged to
//! agree. The problem below has a genuine minimum, which is what makes the
//! comparison meaningful.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::PdSensBacksolver;
use std::cell::RefCell;
use std::rc::Rc;

/// `f(x) = Σ (xᵢ − a)⁴`, minimum 0 at `xᵢ = a`. A large `a` makes the initial
/// gradient enormous, pinning the objective scale at its floor, which is what
/// arms the veto.
struct MaskedQuartic {
    a: Number,
}

const N: usize = 2;

impl TNLP for MaskedQuartic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as i32,
            m: 0,
            nnz_jac_g: 0,
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
        sp.x.copy_from_slice(&[0.0; N]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(x.iter().map(|v| (v - self.a).powi(4)).sum())
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for (gi, xi) in g.iter_mut().zip(x) {
            *gi = 4.0 * (xi - self.a).powi(3);
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        o: Number,
        _l: Option<&[Number]>,
        _nl: bool,
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
                    *v = o * 12.0 * (x[i] - self.a).powi(2);
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve(
    threshold: Number,
    max_iter: i32,
) -> (ApplicationReturnStatus, Number, i32, Option<Vec<Number>>) {
    let out: Rc<RefCell<Option<Vec<Number>>>> = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&out);
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", max_iter, true, false)
        .unwrap();
    app.initialize().unwrap();
    app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
        if let Ok(bs) = PdSensBacksolver::new(data, cq, nlp, pd) {
            let mut rhs = vec![0.0; N];
            rhs[0] = 1.0;
            let mut lhs = vec![0.0; N];
            if bs.solve_scaled_space(&rhs, &mut lhs) {
                *sink.borrow_mut() = Some(lhs);
            }
        }
    }));
    let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(MaskedQuartic { a: 1e5 }));
    let status = app.optimize_tnlp(t);
    let s = app.statistics();
    let (obj, iters) = (s.final_objective, s.iteration_count);
    drop(app);
    (
        status,
        obj,
        iters,
        Rc::try_unwrap(out).ok().and_then(|c| c.into_inner()),
    )
}

#[test]
fn sensitivity_after_a_masked_certificate_fallback_matches_the_unvetoed_run() {
    // The run that never vetoes: stops at the refused point, and its held
    // factor belongs to that point by construction.
    let (base_status, base_obj, base_iters, base_sens) = solve(0.0, 300);
    assert!(
        matches!(base_status, ApplicationReturnStatus::SolveSucceeded),
        "premise: baseline should stop with a certificate, got {base_status:?}"
    );
    let base_sens = base_sens.expect("premise: sensitivity ran on the baseline");

    // The vetoed run, cut off at the baseline's own iteration count so it
    // cannot converge on its own and must fall back to that same point.
    let (veto_status, veto_obj, _, veto_sens) = solve(1e-4, base_iters);
    let veto_sens = veto_sens.expect("sensitivity did not run after the fallback");

    eprintln!(
        "baseline {base_status:?} f={base_obj:.6e} it={base_iters} sens={base_sens:?}\n\
         fallback {veto_status:?} f={veto_obj:.6e} sens={veto_sens:?}"
    );

    assert!(
        (veto_obj - base_obj).abs() <= 1e-9 * base_obj.abs().max(1.0),
        "premise: the fallback should return the baseline point"
    );
    for (i, (b, v)) in base_sens.iter().zip(&veto_sens).enumerate() {
        let scale = b.abs().max(v.abs()).max(1.0);
        assert!(
            (b - v).abs() <= 1e-9 * scale,
            "component {i}: sensitivity after the fallback is {v:.12e} but the unvetoed run at \
             the same point gives {b:.12e} — the held factor no longer describes the returned \
             solution"
        );
    }
}
