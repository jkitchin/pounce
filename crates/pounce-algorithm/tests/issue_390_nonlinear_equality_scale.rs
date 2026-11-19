//! gh #390 — the verdict on a *nonlinear* equality contradiction must not
//! depend on how large the rows are written.
//!
//! `x*y == 1` with `x + y == 0.5` asks for two reals whose sum is `0.5` and
//! whose product is `1` — the roots of `t² - 0.5t + 1`, discriminant
//! `0.25 - 4 < 0`. There are none. Refuting it needs the discriminant, not
//! intervals: each row is individually satisfiable over the box and interval
//! propagation narrows neither variable, so neither the DOF gate's linear
//! bound-propagation probe (#389) nor FBBT can prove it. The verdict has to
//! come from the runtime feasibility test.
//!
//! That test was absolute on equality rows. POUNCE folds the right-hand side
//! into `c(x) = 0`, so `|c_i|` *is* the violation and carries no independent
//! magnitude — the scale-relative measure added for inequality rows in #386
//! skipped the `c` block entirely. Scaling both rows by `1e-8` shrank the
//! residuals under `constr_viol_tol` and the solve stopped at
//! `x = y ≈ 2e-14` — a point that satisfies neither row — and called it
//! `Solve_Succeeded`.
//!
//! Both directions are covered on purpose. A relative measure is *stricter*
//! than an absolute one on small rows, so a fix aimed only at the missed
//! infeasibility could quietly start refusing genuine solutions;
//! [`feasible_twin_solves_at_every_scale`] is that guard.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Row scalings swept by every test here. Multiplying both rows by `s > 0`
/// leaves the solution set exactly unchanged, so the verdict must not move.
const SCALES: [i32; 10] = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6];

/// `min x² + y²  s.t.  s·x·y == s·1,  s·(x + y) == s·sum,  x, y in [-10, 10]`.
///
/// `sum = 0.5` is contradictory (discriminant `< 0`); `sum = 2.5` is solvable
/// (`{x, y} = {2, 0.5}`).
struct ProductAndSum {
    scale: Number,
    sum: Number,
    start: [Number; 2],
}

impl ProductAndSum {
    /// The contradictory model, started at `(1, 1)`. That start satisfies the
    /// product row exactly and misses the sum row, which is what drove the
    /// pre-fix run to stop at a sub-tolerance non-solution and report success
    /// at the small-scale end.
    fn contradictory(scale: Number) -> Self {
        Self {
            scale,
            sum: 0.5,
            start: [1.0, 1.0],
        }
    }

    /// The feasible twin, started at `(0.5, -0.5)` — a start from which this
    /// nonconvex model converges to its genuine solution at every scale in
    /// [`SCALES`]. (From `(1, 1)` the *feasible* twin lands in a
    /// locally-infeasible basin at several scales both before and after this
    /// change, so it says nothing about the fix and is not used here.)
    fn feasible(scale: Number) -> Self {
        Self {
            scale,
            sum: 2.5,
            start: [0.5, -0.5],
        }
    }
}

impl TNLP for ProductAndSum {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, -10.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0]);
        // Equalities: g_l == g_u. Both right-hand sides are non-zero, so both
        // rows have a declared magnitude the relative measure can divide by (a
        // homogeneous row has none and keeps the absolute test).
        b.g_l.copy_from_slice(&[self.scale, self.scale * self.sum]);
        b.g_u.copy_from_slice(&[self.scale, self.scale * self.sum]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.start);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        // Honest classification: the product row is nonlinear, the sum row is
        // linear. Bound propagation gets to see the linear row and still
        // cannot prove anything — the contradiction is in the coupling.
        types.copy_from_slice(&[Linearity::NonLinear, Linearity::Linear]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.scale * x[0] * x[1];
        g[1] = self.scale * (x[0] + x[1]);
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("jacobian values need x");
                values[0] = self.scale * x[1];
                values[1] = self.scale * x[0];
                values[2] = self.scale;
                values[3] = self.scale;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // Lower triangle: (0,0), (1,0), (1,1).
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let l0 = lambda.map(|l| l[0]).unwrap_or(0.0);
                values[0] = 2.0 * obj_factor;
                // ∂²(s·x·y)/∂x∂y = s; the sum row contributes nothing.
                values[1] = l0 * self.scale;
                values[2] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// All-default options — in particular `presolve` stays at its default "no",
/// the configuration the issue was filed against.
fn solve(problem: ProductAndSum) -> ApplicationReturnStatus {
    solve_with(problem, false)
}

/// `solve`, optionally with `presolve=yes`. Every sweep below runs both legs:
/// with presolve off nothing here reaches Phase 2's redundant-row removal,
/// which under an absolute tolerance dropped the down-scaled linear sum row
/// and reported this model solved (gh #1197).
fn solve_with(problem: ProductAndSum, presolve: bool) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    if presolve {
        app.options_mut()
            .set_string_value("presolve", "yes", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(problem));
    app.optimize_tnlp(tnlp)
}

/// `FeasiblePointFound` counts, in **both** directions. It is a success-band
/// answer — `Ipopt 3.14.19: Found feasible point for square problem`, AMPL
/// `objno` code 2, which every band table reads as SOLVED — and this model is
/// square (two variables, two equality rows), so it is a verdict the driver
/// can actually reach here. Leaving it out is what let the square-problem
/// path (gh#508) reintroduce this issue's exact defect at `1e-4` without any
/// test in this file noticing: the run reported a feasible point for a system
/// with no solution, and `is_success` said that was not a success.
///
/// Real Ipopt does return it on this model at that scale, having compared an
/// unscaled violation against an absolute `constr_viol_tol`. That is the
/// comparison this issue exists to refuse.
fn is_success(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded
            | ApplicationReturnStatus::SolvedToAcceptableLevel
            | ApplicationReturnStatus::FeasiblePointFound
    )
}

/// The defect, stated as the property it violated: a model with no solution
/// must never be reported as solved, at any row scale. Before the fix this
/// failed at `1e-12`, `1e-10` and `1e-8` (`Solve_Succeeded`) and at `1e-6`
/// (`Solved_To_Acceptable_Level`).
#[test]
fn contradictory_equalities_are_never_reported_solved() {
    for presolve in [false, true] {
        for k in SCALES {
            let status = solve_with(ProductAndSum::contradictory(10.0_f64.powi(k)), presolve);
            assert!(
                !is_success(status),
                "row scale 1e{k}, presolve={presolve}: `x*y == 1, x + y == 0.5` \
                 has no real solution, yet the solve reported {status:?}"
            );
        }
    }
}

/// And at the small-scale end it now reaches the *positive* verdict, not just
/// a non-answer: the relative measure keeps the violation visible, so rapid
/// infeasibility detection can do its job where the absolute tolerance had
/// erased the evidence. (At `1e-4` and above this driver path exits
/// `Restoration_Failed` — a pre-existing non-verdict, unchanged by this fix,
/// which is why the sweep above asserts the invariant rather than a status.)
#[test]
fn contradictory_equalities_are_diagnosed_at_small_row_scales() {
    for presolve in [false, true] {
        for k in [-12, -10, -8, -6] {
            assert_eq!(
                solve_with(ProductAndSum::contradictory(10.0_f64.powi(k)), presolve),
                ApplicationReturnStatus::InfeasibleProblemDetected,
                "row scale 1e{k}, presolve={presolve}"
            );
        }
    }
}

/// The accepting direction. A relative measure is *stricter* than an absolute
/// one on small rows, so it must not start refusing certificates on a model
/// that genuinely has a solution.
#[test]
fn feasible_twin_solves_at_every_scale() {
    for presolve in [false, true] {
        for k in SCALES {
            let status = solve_with(ProductAndSum::feasible(10.0_f64.powi(k)), presolve);
            assert!(
                is_success(status),
                "row scale 1e{k}, presolve={presolve}: `x*y == 1, x + y == 2.5` \
                 has the solution (2, 0.5); got {status:?}"
            );
        }
    }
}
