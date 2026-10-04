//! gh#986 review item 10: an SQP working set is published in the caller's
//! full variable space and mapped through each solve's fixed-variable
//! elimination.
//!
//! The issue's TinyQP: `min (x1-0.6)^2 + (x2-0.3)^2  s.t.  x1 + x2 <= 0.8`,
//! box `[0, 1]^2`. A child fixes `x1 = 0` (`ub = 0`); the optimum is then
//! `(0, 0.3)`. The first pass made the dimension mismatch a cold solve: the
//! parent's working set (full `n = 2`) was dropped against the child's
//! reduced `n = 1`, and the child's own working set was published reduced,
//! so a sibling fixing `x2` instead (also `n = 1`) would have applied it to
//! the wrong column.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::sqp::SqpIterates;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_qp::{BoundStatus, ConsStatus, WorkingSet};
use std::cell::RefCell;
use std::rc::Rc;

struct TinyQp {
    lb: [Number; 2],
    ub: [Number; 2],
    x0: [Number; 2],
    x: Vec<Number>,
}

impl TNLP for TinyQp {
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
        b.x_l.copy_from_slice(&self.lb);
        b.x_u.copy_from_slice(&self.ub);
        b.g_l[0] = -1e20;
        b.g_u[0] = 0.8;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 0.6).powi(2) + (x[1] - 0.3).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 0.6);
        g[1] = 2.0 * (x[1] - 0.3);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..2].copy_from_slice(&[0, 0]);
                jcol[..2].copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values[..2].copy_from_slice(&[1.0, 1.0]),
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..2].copy_from_slice(&[0, 1]);
                jcol[..2].copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[..2].copy_from_slice(&[2.0 * of, 2.0 * of])
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.x = s.x.to_vec();
    }
}

fn app() -> IpoptApplication {
    let mut a = IpoptApplication::new();
    a.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    a.options_mut()
        .set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    a.initialize().unwrap();
    a
}

fn solve(
    a: &mut IpoptApplication,
    lb: [Number; 2],
    ub: [Number; 2],
    x0: [Number; 2],
) -> (ApplicationReturnStatus, Vec<Number>) {
    let p = Rc::new(RefCell::new(TinyQp {
        lb,
        ub,
        x0,
        x: Vec::new(),
    }));
    let s = a.optimize_tnlp(p.clone());
    let x = p.borrow().x.clone();
    (s, x)
}

fn warm(x0: [Number; 2], ws: WorkingSet) -> SqpIterates {
    SqpIterates {
        x: x0.to_vec(),
        lambda_g: vec![0.0],
        lambda_x: vec![0.0; 2],
        working: Some(ws),
    }
}

#[test]
fn a_child_that_fixes_a_variable_gets_the_parent_working_set_and_publishes_full_space() {
    let mut a = app();
    let (s, xp) = solve(&mut a, [0.0, 0.0], [1.0, 1.0], [0.5, 0.5]);
    assert_eq!(s, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        (xp[0] - 0.55).abs() < 1e-6 && (xp[1] - 0.25).abs() < 1e-6,
        "{xp:?}"
    );
    let parent_ws = a
        .last_sqp_working_set()
        .cloned()
        .expect("parent working set");
    assert_eq!(parent_ws.bounds.len(), 2);

    // Child: x1 fixed at 0.
    a.set_sqp_warm_start(warm([0.0, 0.3], parent_ws));
    let (s, xc) = solve(&mut a, [0.0, 0.0], [0.0, 1.0], [0.0, 0.3]);
    assert_eq!(s, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        a.statistics().sqp_warm_working_set_applied,
        "the parent's working set must reach the child (mapped), not be dropped"
    );
    assert!(xc[0].abs() < 1e-12 && (xc[1] - 0.3).abs() < 1e-6, "{xc:?}");
    // Published in the caller's space: two entries, the fixed one `Fixed`.
    let child_ws = a
        .last_sqp_working_set()
        .cloned()
        .expect("child working set");
    assert_eq!(child_ws.bounds.len(), 2, "{child_ws:?}");
    assert_eq!(child_ws.bounds[0], BoundStatus::Fixed);

    // Sibling: x2 fixed at 0.3 instead; the child's full-space set maps
    // onto the sibling's remaining column (x1), not onto "column 0 of n = 1".
    a.set_sqp_warm_start(warm([0.5, 0.3], child_ws));
    let (s, xs) = solve(&mut a, [0.0, 0.3], [1.0, 0.3], [0.5, 0.3]);
    assert_eq!(s, ApplicationReturnStatus::SolveSucceeded);
    assert!(a.statistics().sqp_warm_working_set_applied);
    assert!(
        (xs[0] - 0.5).abs() < 1e-6 && (xs[1] - 0.3).abs() < 1e-12,
        "{xs:?}"
    );
    let sib_ws = a.last_sqp_working_set().cloned().unwrap();
    assert_eq!(sib_ws.bounds.len(), 2);
    assert_eq!(sib_ws.bounds[1], BoundStatus::Fixed);
}

/// A *reduced-space* working set (a caller that kept one from before this
/// fix, or built one by hand) cannot be placed: when the solve that produced
/// it fixed different variables, it is dropped and the solve is cold and
/// correct, instead of applied to the wrong columns.
#[test]
fn a_reduced_space_working_set_from_a_different_elimination_is_dropped() {
    let mut a = app();
    // Child fixing x1 (the last elimination this application saw).
    let (s, _) = solve(&mut a, [0.0, 0.0], [0.0, 1.0], [0.0, 0.3]);
    assert_eq!(s, ApplicationReturnStatus::SolveSucceeded);
    // A reduced (n = 1) working set claiming the remaining bound is at its
    // upper bound, handed to a sibling that fixes x2 instead.
    let reduced = WorkingSet {
        bounds: vec![BoundStatus::AtUpper],
        constraints: vec![ConsStatus::Inactive],
    };
    a.set_sqp_warm_start(SqpIterates {
        x: vec![0.5],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0],
        working: Some(reduced),
    });
    let (s, xs) = solve(&mut a, [0.0, 0.3], [1.0, 0.3], [0.5, 0.3]);
    assert_eq!(s, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        !a.statistics().sqp_warm_working_set_applied,
        "a reduced working set from a different elimination must be dropped"
    );
    assert!(
        (xs[0] - 0.5).abs() < 1e-6 && (xs[1] - 0.3).abs() < 1e-12,
        "{xs:?}"
    );
}
