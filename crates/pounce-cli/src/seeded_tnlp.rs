//! TNLP wrapper that overrides the starting point with a seed captured
//! by the interactive debugger, so `resolve` can re-run the solve from
//! the current iterate with new options (a primal warm start).
//!
//! Every method forwards to the inner TNLP exactly like
//! [`crate::counting_tnlp::CountingTnlp`]; only [`TNLP::get_starting_point`]
//! is overridden. The seed is applied **only when its length matches the
//! starting-point buffer** — if presolve or fixed-variable elimination
//! changed the coordinate count, we fall back to the problem's own start
//! rather than seed into the wrong space.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, MetaData, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct SeededTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    seed_x: Vec<Number>,
}

impl SeededTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, seed_x: Vec<Number>) -> Self {
        Self { inner, seed_x }
    }
}

impl TNLP for SeededTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if self.seed_x.len() == sp.x.len() {
            // `init_x` / `init_z` / `init_lambda` are inputs (which
            // buffers the solver wants); for the initial point the solver
            // always reads `x`, so writing the primal seed here suffices.
            // We don't have warm duals, so we leave z/lambda untouched.
            sp.x.copy_from_slice(&self.seed_x);
            true
        } else {
            // Coordinate count changed (presolve / fixed-variable
            // elimination); fall back to the problem's own start.
            self.inner.borrow_mut().get_starting_point(sp)
        }
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.borrow_mut().eval_jac_g(x, new_x, mode)
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.inner
            .borrow_mut()
            .finalize_solution(sol, ip_data, ip_cq)
    }
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.borrow_mut().get_scaling_parameters(req)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }
    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        self.inner.borrow_mut().get_list_of_nonlinear_variables(pos)
    }
    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con)
    }
}
