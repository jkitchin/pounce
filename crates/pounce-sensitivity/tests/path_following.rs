//! Walking the perturbation through more than one breakpoint.
//!
//! `parametric_cpp.rs` covers the walk on upstream's own example, where
//! exactly one variable crosses. This file is a QP whose crossings
//! interact: holding one variable at its bound turns the other's
//! direction.
//!
//! The reference is a full re-solve at the perturbed parameter, which
//! for a QP is the exact answer. What the file pins is a measured
//! identity, not a separation: both bound-aware modes reproduce the
//! re-solve here, because once they settle on the same final active
//! set the QP solution under that set is affine in the parameter, and
//! the endpoint of the walk's piecewise path coincides with the
//! base-point repair's single step. What the walk adds on this model
//! is the breakpoint record.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::Solver;

/// Off-diagonal weight in the objective. Nonzero is the whole point:
/// with it, holding one variable at a bound turns the other one's
/// direction, so the crossings interact.
const COUPLING: Number = 0.5;

/// A two-variable QP with both variables bounded above, and the
/// parameter carried as a third variable pinned by an equality so the
/// sensitivity machinery can perturb it the usual way.
///
/// ```text
/// min  0.5 x1^2 + 0.5 x2^2 + COUPLING x1 x2 - p x1 - 2 p x2
/// s.t. g0 = p               (the pin)
///      0 <= x1 <= 1,  0 <= x2 <= 1
/// ```
///
/// The unconstrained stationary point moves with `p`, and because `x2`
/// carries twice the linear pull it reaches its upper bound first.
/// Holding `x2` there changes what `x1` wants, which is what the
/// coupling is for.
struct CoupledQp {
    p_nominal: Number,
}

impl TNLP for CoupledQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 1.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = self.p_nominal;
        b.g_u[0] = self.p_nominal;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.1;
        sp.x[1] = 0.1;
        sp.x[2] = self.p_nominal;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        Some(0.5 * x1 * x1 + 0.5 * x2 * x2 + COUPLING * x1 * x2 - p * x1 - 1.5 * p * x2)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        g[0] = x1 + COUPLING * x2 - p;
        g[1] = x2 + COUPLING * x1 - 1.5 * p;
        g[2] = -x1 - 1.5 * x2;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
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
        // Lower triangle of the objective Hessian: (0,0) and (1,1),
        // (1,0) for the coupling, and (2,0) / (2,1) for the parameter's
        // bilinear terms.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 1, 0, 0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor * COUPLING;
                values[3] = -obj_factor;
                values[4] = -1.5 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn configured() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

/// Solve at `p` and return `(x1, x2)`.
fn solve_at(p: Number) -> [Number; 2] {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(CoupledQp { p_nominal: p })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at p={p} failed: {status:?}",
    );
    let x = solver.converged().expect("converged state").x.clone();
    [x[0], x[1]]
}

/// Solve at `p0`, then take both bound-aware steps to `p0 + dp`.
fn both_steps(p0: Number, dp: Number) -> ([Number; 2], [Number; 2], usize) {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(CoupledQp { p_nominal: p0 })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "base solve failed: {status:?}",
    );
    let base = solver.converged().expect("converged state").x.clone();
    let (fixed, _) = solver
        .parametric_step_bounded(&[0], &[dp], 8)
        .expect("parametric_step_bounded");
    let (walked, segs) = solver
        .parametric_step_path(&[0], &[dp], 8)
        .expect("parametric_step_path");
    (
        [base[0] + fixed[0], base[1] + fixed[1]],
        [base[0] + walked[0], base[1] + walked[1]],
        segs.len(),
    )
}

#[test]
fn both_modes_reproduce_the_resolve_across_two_interacting_crossings() {
    // p from 0.3 to 1.6. x2 reaches its upper bound first because it
    // carries the larger linear pull, and holding it there turns x1's
    // direction, which brings x1 to its own bound later in the path.
    let truth = solve_at(1.6);
    let (fixed, walked, segments) = both_steps(0.3, 1.3);

    assert!(
        segments >= 2,
        "the walk should cross both bounds, got {segments} segment(s)",
    );
    let err_walk = (walked[0] - truth[0])
        .abs()
        .max((walked[1] - truth[1]).abs());
    let err_fixed = (fixed[0] - truth[0]).abs().max((fixed[1] - truth[1]).abs());
    assert!(
        err_walk < 1e-6,
        "the walk should reproduce the re-solve on a QP, off by {err_walk} \
         (walk {walked:?} against {truth:?})",
    );
    assert!(
        err_fixed < 1e-6,
        "the base-point repair lands on the same final active set here, \
         so it should also reproduce the re-solve, off by {err_fixed} \
         (fixed {fixed:?} against {truth:?})",
    );
}

#[test]
fn the_walk_reproduces_the_resolve_where_one_bound_is_reached() {
    // A perturbation big enough for x2 to reach its bound and no more.
    let truth = solve_at(0.9);
    let (_, walked, segments) = both_steps(0.3, 0.6);
    assert_eq!(segments, 1, "one crossing expected");
    let err = (walked[0] - truth[0])
        .abs()
        .max((walked[1] - truth[1]).abs());
    assert!(
        err < 1e-6,
        "off by {err}: walk {walked:?} against {truth:?}"
    );
}
