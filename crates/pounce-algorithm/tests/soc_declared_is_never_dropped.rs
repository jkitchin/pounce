//! Workstream C (`dev-notes/research/nlp-soc-design.md`): a declared
//! second-order cone is never dropped. Until the interior-point method
//! carries cones natively, a solve with cones declared is refused before the
//! model is evaluated — the alternative, solving with the cone rows read as
//! the free rows they are declared as, returns the unconstrained answer.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::soc::SocRows;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x1 + x2` s.t. `(1, x1, x2) ∈ K` written as free rows `g = (x0, x1,
/// x2)` with `x0 = 1` fixed by bounds. With the cone the answer is
/// `x1 = x2 = −1/√2`; without it the model is unbounded.
struct Disk;

impl TNLP for Disk {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 3,
            nnz_jac_g: 3,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[1.0, -1e19, -1e19]);
        b.x_u.copy_from_slice(&[1.0, 1e19, 1e19]);
        b.g_l.fill(-1e19);
        b.g_u.fill(1e19);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 0.0, 0.0]);
        true
    }
    fn eval_f(&mut self, _: &[Number], _: bool) -> Option<Number> {
        panic!("the model must not be evaluated")
    }
    fn eval_grad_f(&mut self, _: &[Number], _: bool, _: &mut [Number]) -> bool {
        panic!("the model must not be evaluated")
    }
    fn eval_g(&mut self, _: &[Number], _: bool, _: &mut [Number]) -> bool {
        panic!("the model must not be evaluated")
    }
    fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, _: SparsityRequest<'_>) -> bool {
        panic!("the model must not be evaluated")
    }
    fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
}

#[test]
fn a_solve_with_declared_cones_is_refused_until_the_ipm_carries_them() {
    let mut app = IpoptApplication::new();
    app.set_soc_constraints(vec![SocRows {
        rows: vec![0, 1, 2],
    }]);
    let status = app.optimize_tnlp(Rc::new(RefCell::new(Disk)));
    assert_eq!(status, ApplicationReturnStatus::InvalidProblemDefinition);
    // Clearing the declaration clears the refusal path.
    app.set_soc_constraints(Vec::new());
    assert!(app.soc_constraints().is_empty());
}
