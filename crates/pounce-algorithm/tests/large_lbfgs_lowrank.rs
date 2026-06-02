//! Large-`n` limited-memory L-BFGS through the low-rank
//! Sherman-Morrison-Woodbury path.
//!
//! `LimMemQuasiNewtonUpdater` publishes a `LowRankUpdateSymMatrix` and the
//! solve goes through `LowRankAugSystemSolver` — no dense `n×n` is ever
//! formed. At `n = 2500` this guards that the limited-memory path scales:
//! a dense Hessian rebuild here would be `O(n²)`, and the point of the
//! low-rank assembler is `O(n·m)` storage. The problem is a
//! well-conditioned separable convex quadratic with a single equality
//! constraint and an interior optimum, so it converges in a handful of
//! iterations and the test stays fast.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const N: usize = 2500;

fn target(i: usize) -> Number {
    0.5 * (0.1 * i as Number).sin()
}

#[derive(Default)]
struct SeparableQp {
    final_obj: Option<Number>,
    final_x0: Option<Number>,
}

impl TNLP for SeparableQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as i32,
            m: 1,
            nnz_jac_g: N as i32,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -10.0);
        b.x_u.iter_mut().for_each(|v| *v = 10.0);
        // Equality: Σ x_i = Σ a_i (so x* = a is feasible and optimal).
        let rhs: Number = (0..N).map(target).sum();
        b.g_l[0] = rhs;
        b.g_u[0] = rhs;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(0.5 * (0..N).map(|i| (x[i] - target(i)).powi(2)).sum::<Number>())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..N {
            g[i] = x[i] - target(i);
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x.iter().sum();
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
                for j in 0..N {
                    irow[j] = 0;
                    jcol[j] = j as i32;
                }
            }
            SparsityRequest::Values { values } => {
                values.iter_mut().for_each(|v| *v = 1.0);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
        if !sol.x.is_empty() {
            self.final_x0 = Some(sol.x[0]);
        }
    }
}

#[test]
fn large_separable_qp_solves_with_lbfgs_lowrank() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(SeparableQp::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);

    let stats = app.statistics();
    eprintln!(
        "large L-BFGS (n={N}): status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );

    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    // Optimum is x* = a, objective 0.
    let obj = tnlp_concrete.borrow().final_obj.unwrap();
    assert!(obj < 1e-6, "final objective {obj} not near 0");
    let x0 = tnlp_concrete.borrow().final_x0.unwrap();
    assert!(
        (x0 - target(0)).abs() < 1e-3,
        "x[0]={x0} expected {}",
        target(0)
    );
}

// ---------------------------------------------------------------------------
// Large-n with *active bounds* at the solution — the ill-conditioned regime.
//
// minimize 0.5 Σ (xᵢ − aᵢ)²  s.t.  0 ≤ xᵢ ≤ 1,  with aᵢ ∈ [−2, 2].
//
// The unconstrained minimizer aᵢ lies outside [0,1] for most i, so the
// optimum is xᵢ* = clamp(aᵢ, 0, 1): a large fraction of the bounds are
// active, driving the corresponding slacks → 0 and the barrier terms
// (Σ_l, Σ_u) → ∞ as μ → 0. This is exactly the near-singular reduced
// Hessian I worried the indirect SMW step/inertia computation might not
// handle at scale. The known closed-form optimum lets us assert the
// solve lands on it, not merely that it converges.

const M: usize = 2500;

fn a_of(i: usize) -> Number {
    2.0 * (0.1 * (i as Number + 1.0)).sin()
}

fn clamp01(v: Number) -> Number {
    v.clamp(0.0, 1.0)
}

#[derive(Default)]
struct BoundActiveQp {
    final_obj: Option<Number>,
    final_x: Option<Vec<Number>>,
}

impl TNLP for BoundActiveQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: M as i32,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = 0.0);
        b.x_u.iter_mut().for_each(|v| *v = 1.0);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.5);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(0.5 * (0..M).map(|i| (x[i] - a_of(i)).powi(2)).sum::<Number>())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..M {
            g[i] = x[i] - a_of(i);
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

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
        self.final_x = Some(sol.x.to_vec());
    }
}

#[test]
fn large_bound_active_qp_solves_with_lbfgs_lowrank() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(BoundActiveQp::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);

    let stats = app.statistics();
    eprintln!(
        "large bound-active L-BFGS (n={M}): status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );

    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    // Closed-form optimum: xᵢ* = clamp(aᵢ, 0, 1).
    let obj_star: Number = 0.5
        * (0..M)
            .map(|i| (clamp01(a_of(i)) - a_of(i)).powi(2))
            .sum::<Number>();
    let obj = tnlp_concrete.borrow().final_obj.unwrap();
    assert!(
        (obj - obj_star).abs() < 1e-4,
        "final objective {obj} != closed-form {obj_star}",
    );

    let x = tnlp_concrete.borrow().final_x.clone().unwrap();
    // Spot-check one variable in each regime (bounds are interior-point,
    // so active vars approach the bound to within barrier tolerance).
    // i=0:  a≈0.20  → interior, x*≈0.20.
    // i=10: a≈1.78  → upper-active, x*≈1.
    // i=35: a≈-0.88 → lower-active, x*≈0.
    assert!((x[0] - clamp01(a_of(0))).abs() < 1e-3, "x[0]={}", x[0]);
    assert!(
        x[10] > 1.0 - 1e-4,
        "upper-active x[10]={} (a={})",
        x[10],
        a_of(10)
    );
    assert!(
        x[35] < 1e-4,
        "lower-active x[35]={} (a={})",
        x[35],
        a_of(35)
    );
}
