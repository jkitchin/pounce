//! Regression tests for issue #252: spurious UNBOUNDED (`DivergingIterates`)
//! on an **unbounded-box** subproblem that still has a finite optimum — the
//! follow-up to #248.
//!
//! #248 stopped POUNCE mislabelling jit1's fully-boxed / free-variable *root*
//! relaxation as unbounded. But its guard only asked whether the diverging
//! iterate (a) heads toward a side with no finite bound and (b) keeps growing
//! for a few iterations. jit1's spatial-B&B *node* subproblems carry variables
//! with `ub = +∞` (integer-tightened boxes), so (a) holds, and under the linear
//! tail's 1e7-scale ill-scaling the transient excursion climbs past enough
//! doublings to satisfy (b) as well — so every node was reported UNBOUNDED even
//! though cyipopt/Ipopt find each node's finite optimum.
//!
//! The distinguishing fact a *local* solver can check is the objective: a
//! genuine recession ray drives `f → −∞` with a per-step drop that keeps up as
//! `|x|` grows geometrically, whereas an excursion converging to a finite
//! optimum has a per-step drop that decelerates toward zero. These tests force
//! the guard to fire (low `diverging_iterates_tol`, the kind a B&B driver sets
//! to abort runaway nodes) on `ub = +∞` problems with finite optima and assert
//! POUNCE returns the optimum, while the companion genuinely-unbounded ray is
//! still reported `DivergingIterates`.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min Σ cᵢ/xᵢ + Σ dᵢ·xᵢ` — the shape of jit1's continuous relaxation, with
/// jit1-scale ill-scaling: tiny `1/x` coefficients and a linear tail up to
/// `1e7`. Bounds are per-variable so a B&B *node* can be modelled: some lower
/// bounds raised by integer branching, upper bounds left at `+∞`.
struct NodeIllScaled {
    n: usize,
    c: Vec<Number>,
    d: Vec<Number>,
    lo: Vec<Number>,
    ub: Vec<Number>,
    x0: Number,
    final_obj: Option<Number>,
}

impl NodeIllScaled {
    /// `all_free_above` leaves every `ub = +∞`; the lower bounds default to a
    /// tiny positive floor with `raise_every_third` lifting a third of them
    /// (the integer-tightened, `ub = +∞` variables the issue calls out).
    fn new(n: usize, raise_lb_to: Number) -> Self {
        // c: 1e-3 .. 1e3 ; d (linear tail): 1 .. 1e7 — jit1-scale ill-scaling.
        let c = (0..n).map(|i| 10f64.powi(((i % 7) as i32) - 3)).collect();
        let d = (0..n).map(|i| 10f64.powi((i % 8) as i32)).collect();
        let lo = (0..n)
            .map(|i| if i % 3 == 0 { raise_lb_to } else { 1e-6 })
            .collect();
        NodeIllScaled {
            n,
            c,
            d,
            lo,
            ub: vec![2e19; n], // > 1e19 infinity sentinel ⇒ ub = +∞
            x0: 1.0,
            final_obj: None,
        }
    }

    /// The closed-form optimum of `Σ cᵢ/xᵢ + dᵢ·xᵢ` on the box: the
    /// unconstrained minimiser is `xᵢ* = √(cᵢ/dᵢ)`, clamped up to any raised
    /// lower bound (all `dᵢ > 0`, so the objective is convex in each `xᵢ`).
    fn optimum(&self) -> Number {
        (0..self.n)
            .map(|i| {
                let xstar = (self.c[i] / self.d[i]).sqrt().max(self.lo[i]);
                self.c[i] / xstar + self.d[i] * xstar
            })
            .sum()
    }
}

impl TNLP for NodeIllScaled {
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
            b.x_l[i] = self.lo[i];
            b.x_u[i] = self.ub[i];
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for i in 0..self.n {
            sp.x[i] = self.x0.max(self.lo[i]);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            (0..self.n)
                .map(|i| self.c[i] / x[i] + self.d[i] * x[i])
                .sum(),
        )
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

fn solve(
    inst: Rc<RefCell<NodeIllScaled>>,
    diverging_tol: Number,
) -> (ApplicationReturnStatus, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("diverging_iterates_tol", diverging_tol, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = inst;
    let status = app.optimize_tnlp(tnlp);
    let obj = app.statistics().final_objective;
    (status, obj)
}

/// An `ub = +∞` node subproblem with a finite optimum must not be reported
/// UNBOUNDED even when `diverging_iterates_tol` is low enough to trip on the
/// (finite) excursion. Before #252 the growth-only streak fired here — the
/// linear tail's ill-scaling makes the transient excursion climb past four
/// doublings — and every jit1 B&B node came back UNBOUNDED. The objective
/// descent-deceleration gate recognises the excursion is settling onto a
/// finite floor and lets the solve converge.
#[test]
fn unbounded_box_node_with_finite_optimum_is_not_unbounded() {
    for &n in &[6usize, 12, 20] {
        for &raise in &[0.0f64, 1.0, 5.0] {
            let inst = Rc::new(RefCell::new(NodeIllScaled::new(n, raise)));
            let expected = inst.borrow().optimum();
            let (status, obj) = solve(Rc::clone(&inst), 1.0);
            assert!(
                !matches!(status, ApplicationReturnStatus::DivergingIterates),
                "n={n} raise_lb={raise}: reported DivergingIterates on an ub=+inf \
                 subproblem with a finite optimum (issue #252); got obj={obj}",
            );
            assert!(
                (obj - expected).abs() / expected < 1e-4,
                "n={n} raise_lb={raise}: objective {obj} is not the finite optimum {expected}",
            );
        }
    }
}

/// The #252 gate must not swallow genuine unboundedness: with a negative
/// linear tail the objective `Σ cᵢ/xᵢ − |dᵢ|·xᵢ → −∞` as `x → +∞` on the
/// `ub = +∞` box, so the iterate rides a real recession ray whose per-step
/// objective drop keeps accelerating. That still trips `DivergingIterates`.
#[test]
fn genuinely_unbounded_box_still_reports_diverging() {
    let inst = Rc::new(RefCell::new(NodeIllScaled::new(6, 0.0)));
    inst.borrow_mut().d = vec![-1e3; 6];
    inst.borrow_mut().lo = vec![1e-6; 6];
    let (status, _obj) = solve(Rc::clone(&inst), 1e-2);
    assert!(
        matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a genuinely unbounded-below objective on an ub=+inf box must still \
         report DivergingIterates; got {status:?}",
    );
}
