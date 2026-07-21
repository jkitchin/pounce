//! Regression tests for issue #248: spurious UNBOUNDED (`DivergingIterates`)
//! on a bounded NLP.
//!
//! `DivergingIterates` is Ipopt's *unboundedness* verdict (it maps to the
//! AMPL 300 "unbounded" range). The divergence guard fires when
//! `max_i |x_i| > diverging_iterates_tol`, but a large `|x|` alone does not
//! prove unboundedness: under severe objective ill-scaling the normal-mode
//! IPM can take a large excursion on a problem that is bounded below with a
//! finite optimum (MINLPLib `jit1`). Worse, if every variable is boxed the
//! feasible region is bounded and unboundedness is structurally impossible.
//!
//! The fix gates the guard on a structural check: a diverging iterate is only
//! reported as unbounded when some over-threshold component is heading toward
//! a side with no finite bound. These tests force the guard to fire (by
//! lowering `diverging_iterates_tol` well below the iterates) and assert:
//!   * a fully-boxed problem is NOT reported unbounded (issue #248), and
//!   * a problem with a genuinely free, diverging variable still is (so the
//!     `diverging_iterates_tol` option remains wired — the #191 guarantee).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min Σ cᵢ/xᵢ + Σ dᵢ·xᵢ` — badly-scaled convex objective (the shape of
/// MINLPLib `jit1`'s continuous relaxation). `ub` controls whether the
/// variables carry a finite upper bound (a bounded box) or run to `+∞` (an
/// upper bound at or beyond the `1e19` infinity sentinel).
struct IllScaled {
    n: usize,
    c: Vec<Number>,
    d: Vec<Number>,
    lo: Number,
    ub: Number,
    x0: Number,
    final_obj: Option<Number>,
}

impl IllScaled {
    fn new(n: usize, lo: Number, ub: Number, x0: Number) -> Self {
        let c = (0..n).map(|i| 10f64.powi((i % 7) as i32)).collect(); // 1 .. 1e6
        let d = (0..n).map(|i| 10f64.powi((i % 4) as i32)).collect(); // 1 .. 1e3
        IllScaled {
            n,
            c,
            d,
            lo,
            ub,
            x0,
            final_obj: None,
        }
    }
}

impl TNLP for IllScaled {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: self.n as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..self.n {
            b.x_l[i] = self.lo;
            b.x_u[i] = self.ub;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for i in 0..self.n {
            sp.x[i] = self.x0;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..self.n {
            f += self.c[i] / x[i] + self.d[i] * x[i];
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.n {
            g[i] = -self.c[i] / (x[i] * x[i]) + self.d[i];
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
                for i in 0..self.n {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h needs x");
                for i in 0..self.n {
                    values[i] = obj_factor * (2.0 * self.c[i] / (x[i] * x[i] * x[i]));
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
    }
}

fn solve(inst: IllScaled, diverging_tol: Number) -> (ApplicationReturnStatus, usize, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("diverging_iterates_tol", diverging_tol, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(inst));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    (status, s.iteration_count as usize, s.final_objective)
}

/// A fully-boxed, badly-scaled problem must never be reported UNBOUNDED,
/// even when `diverging_iterates_tol` is set below the iterates: a bounded
/// box cannot be unbounded. Before the fix the running divergence guard
/// aborted with `DivergingIterates` on magnitude alone.
#[test]
fn boxed_illscaled_is_not_reported_unbounded() {
    // Box [1e-3, 100]; drop the divergence threshold to 10, well inside the
    // box, so the magnitude-only guard would have tripped.
    let inst = IllScaled::new(25, 1e-3, 100.0, 50.0);
    let (status, _iters, obj) = solve(inst, 10.0);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "boxed problem must not report DivergingIterates (issue #248); got {status:?} obj={obj}",
    );
}

/// A free variable whose objective has a *finite* optimum above the
/// threshold must not be reported unbounded either — this is the raw `jit1`
/// case (free variables, low `diverging_iterates_tol`). The magnitude-only
/// guard, and even the structural guard alone, tag it unbounded; the growth-
/// persistence check recognises the iterate settles (like `jit1`, whose
/// `|x|` peaks then recedes) and lets it converge.
#[test]
fn free_variable_with_finite_optimum_is_not_reported_unbounded() {
    // `min Σ cᵢ/xᵢ + dᵢ·xᵢ` with dᵢ > 0 has the finite minimiser
    // xᵢ* = √(cᵢ/dᵢ); variables are unbounded above (free upper side). Drop
    // the threshold well below x* so the guard would trip on magnitude.
    let inst = IllScaled::new(6, 1e-3, 2e19, 1.0);
    let (status, _iters, obj) = solve(inst, 1e-2);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a free variable with a finite optimum must not report DivergingIterates \
         (issue #248); got {status:?} obj={obj}",
    );
}

/// The companion guarantee (#191): the option is still wired and a genuinely
/// unbounded problem is still reported `DivergingIterates`. Here the linear
/// term is negative and the variables are unbounded above, so the objective
/// decreases without bound (`Σ cᵢ/xᵢ − |dᵢ|·xᵢ → −∞`) and the iterate rides a
/// real recession ray — the persistence streak accumulates and fires.
#[test]
fn genuinely_unbounded_still_triggers_diverging() {
    let mut inst = IllScaled::new(4, 1e-3, 2e19, 1.0);
    // Negative linear coefficients ⇒ objective unbounded below as x → +∞.
    inst.d = vec![-1e3; inst.n];
    let (status, _iters, _obj) = solve(inst, 1e-2);
    assert!(
        matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a genuinely unbounded problem past diverging_iterates_tol must report \
         DivergingIterates; got {status:?}",
    );
}
