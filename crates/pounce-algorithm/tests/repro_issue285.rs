//! Regression tests for issue #285: the NLP-path divergence detector missed
//! an unbounded LP whose recession ray lies in `null(A_eq)` over **free**
//! variables.
//!
//! The `1e20` `diverging_iterates_tol` magnitude guard (issues #248 / #252)
//! only fires once `|x|_∞` crosses `1e20`, and only accumulates its streak on
//! *geometric* growth. A recession ray in an equality null space over free
//! variables (`min −x0  s.t.  x0 − x1 = 0`, ray `d = (1, 1)`) is walked out by
//! regularized zero-Hessian Newton steps at a bounded, roughly *linear* rate,
//! so `|x|` never reaches `1e20` within `max_iter` and the geometric streak
//! never accumulates — POUNCE reported `Maximum_Iterations_Exceeded` while
//! Ipopt reported `Diverging_Iterates` on the identical model.
//!
//! The fix adds a second, independent unboundedness path: a *checked
//! recession-ray proof* active from a far lower magnitude floor. A genuinely
//! feasible iterate of large norm already witnesses that the feasible region
//! is unbounded; the proof additionally certifies the escape direction lies in
//! `null(A_eq)`, is not blocked by any finitely-bounded inequality, heads
//! toward an unbounded variable side, and strictly lowers the objective. That
//! makes it a proof, not a magnitude heuristic — so it must fire on the
//! unbounded shape yet stay silent on every bounded control below.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// A dense linear program `min cᵀx  s.t.  gₗ ≤ A x ≤ gᵤ,  xₗ ≤ x ≤ xᵤ`.
/// General enough to express both the `null(A_eq)` unbounded ray and its
/// bounded controls (add a variable bound, or a second pinning equality).
struct DenseLp {
    n: usize,
    m: usize,
    c: Vec<Number>,
    a: Vec<Number>, // row-major m×n
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    x0: Vec<Number>,
    final_obj: Option<Number>,
}

impl DenseLp {
    fn a(&self, i: usize, j: usize) -> Number {
        self.a[i * self.n + j]
    }
}

impl TNLP for DenseLp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32,
            nnz_h_lag: 0, // linear objective + constraints ⇒ zero Hessian
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((0..self.n).map(|j| self.c[j] * x[j]).sum())
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&self.c);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.m {
            g[i] = (0..self.n).map(|j| self.a(i, j) * x[j]).sum();
        }
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
                let mut k = 0;
                for i in 0..self.m {
                    for j in 0..self.n {
                        irow[k] = i as i32;
                        jcol[k] = j as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&self.a);
            }
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
        // Zero Hessian (LP): no structure entries, nothing to fill.
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
    }
}

fn solve(inst: Rc<RefCell<DenseLp>>, max_iter: i32) -> (ApplicationReturnStatus, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("max_iter", max_iter, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = inst;
    let status = app.optimize_tnlp(tnlp);
    let obj = app.statistics().final_objective;
    (status, obj)
}

const INF: Number = 2e19; // > 1e19 infinity sentinel

/// The headline case: `min −x0  s.t.  x0 − x1 = 0`, both variables free. The
/// recession ray `d = (1, 1)` lies in `null(A_eq)`, `cᵀd = −1 < 0`. Before the
/// fix this ran to `Maximum_Iterations_Exceeded`; it must now report
/// `DivergingIterates` — POUNCE's (Ipopt `ApplicationReturnStatus`) unbounded
/// verdict — matching Ipopt, scipy/HiGHS and Clarabel on the same model.
#[test]
fn null_aeq_free_var_unbounded_reports_diverging() {
    // Reproducible across the iteration budgets the adversary exercised.
    for &max_iter in &[300, 3000] {
        let inst = Rc::new(RefCell::new(DenseLp {
            n: 2,
            m: 1,
            c: vec![-1.0, 0.0],
            a: vec![1.0, -1.0],
            g_l: vec![0.0],
            g_u: vec![0.0],
            x_l: vec![-INF, -INF],
            x_u: vec![INF, INF],
            x0: vec![0.0, 0.0],
            final_obj: None,
        }));
        let (status, obj) = solve(Rc::clone(&inst), max_iter);
        assert!(
            matches!(status, ApplicationReturnStatus::DivergingIterates),
            "max_iter={max_iter}: a recession ray in null(A_eq) over free \
             variables must report DivergingIterates (issue #285); got \
             {status:?} obj={obj}",
        );
    }
}

/// Bounded control A — identical structure, but `x0` carries a finite upper
/// bound. The ray is now capped; the optimum is `x0 = x1 = 10`, `obj = −10`.
/// Must solve, never falsely report unbounded. Pins that the new recession
/// path's free-to-escape gate keys on the *variable bound*.
#[test]
fn null_aeq_bounded_by_var_upper_is_optimal() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 2,
        m: 1,
        c: vec![-1.0, 0.0],
        a: vec![1.0, -1.0],
        g_l: vec![0.0],
        g_u: vec![0.0],
        x_l: vec![-INF, -INF],
        x_u: vec![10.0, INF],
        x0: vec![0.0, 0.0],
        final_obj: None,
    }));
    let (status, obj) = solve(Rc::clone(&inst), 3000);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a bounded variant (x0 ≤ 10) must not report DivergingIterates \
         (issue #285); got {status:?} obj={obj}",
    );
    assert!(
        (obj - (-10.0)).abs() < 1e-5,
        "bounded variant objective {obj} is not the finite optimum -10",
    );
}

/// Bounded control B — the same free variables and one equality, plus a second
/// equality `x0 + x1 = 4` that fully determines `x0 = x1 = 2` (`obj = −2`). The
/// feasible set is a single point: bounded despite free variable bounds. Must
/// solve to the optimum, never falsely report unbounded. Pins that the
/// recession path's `null(A_eq)` direction / feasibility gate is not fooled by
/// free variables when the equalities themselves pin the region.
#[test]
fn free_vars_with_pinning_equalities_is_optimal() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 2,
        m: 2,
        c: vec![-1.0, 0.0],
        a: vec![1.0, -1.0, 1.0, 1.0],
        g_l: vec![0.0, 4.0],
        g_u: vec![0.0, 4.0],
        x_l: vec![-INF, -INF],
        x_u: vec![INF, INF],
        x0: vec![0.0, 0.0],
        final_obj: None,
    }));
    let (status, obj) = solve(Rc::clone(&inst), 3000);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a bounded, fully-pinned variant must not report DivergingIterates \
         (issue #285); got {status:?} obj={obj}",
    );
    assert!(
        (obj - (-2.0)).abs() < 1e-5,
        "pinned variant objective {obj} is not the finite optimum -2",
    );
}

/// Bounded control C — the escape direction is blocked by a *finitely-bounded
/// inequality*, not a variable bound: `min −x0  s.t.  x0 ≤ 1e8`, `x` free. The
/// variable is free (so the free-to-escape gate alone would admit it), but the
/// inequality caps the ray, so the problem is bounded with optimum `−1e8`. Pins
/// that [`recession_blocked_by_inequality`] prevents a false positive here.
#[test]
fn free_var_capped_by_inequality_is_optimal() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 1,
        m: 1,
        c: vec![-1.0],
        a: vec![1.0],
        g_l: vec![-INF],
        g_u: vec![1e8],
        x_l: vec![-INF],
        x_u: vec![INF],
        x0: vec![0.0],
        final_obj: None,
    }));
    let (status, obj) = solve(Rc::clone(&inst), 3000);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a free variable capped by an inequality must not report \
         DivergingIterates (issue #285); got {status:?} obj={obj}",
    );
    assert!(
        (obj - (-1e8)).abs() / 1e8 < 1e-4,
        "inequality-capped variant objective {obj} is not the finite optimum -1e8",
    );
}
