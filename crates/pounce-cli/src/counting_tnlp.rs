//! TNLP wrapper that counts evaluation calls so the CLI can mirror
//! Ipopt's end-of-run "Number of … evaluations = N" summary block.
//!
//! All eight required TNLP methods (and `intermediate_callback`) are
//! forwarded transparently to the inner TNLP. The counters live in
//! `Cell<i32>`s on the wrapper itself, so the CLI can read them via
//! `Rc<RefCell<CountingTnlp>>::borrow()` after the solve completes.
//!
//! The wrapper does not count *every* call — calls that pass an
//! `irow/jcol`-only `SparsityRequest::Structure` (the symbolic
//! sparsity-pattern call, not the values call) don't represent a real
//! Jacobian / Hessian evaluation, mirroring the way Ipopt reports
//! these numbers.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, MetaData, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub struct CountingTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    pub n_obj: Cell<i32>,
    pub n_grad_f: Cell<i32>,
    pub n_g: Cell<i32>,
    pub n_jac_g: Cell<i32>,
    pub n_h: Cell<i32>,
    /// Primal `x` and constraint duals `lambda` captured at
    /// `finalize_solution`, in the original-problem space the inner TNLP
    /// presents. The CLI uses this as a fallback solution source for the
    /// active-set SQP route, whose solve bypasses the IPM-only
    /// `on_converged` hook the `.sol` / JSON writers normally read.
    captured_solution: RefCell<Option<(Vec<Number>, Vec<Number>)>>,
}

impl CountingTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>) -> Self {
        Self {
            inner,
            n_obj: Cell::new(0),
            n_grad_f: Cell::new(0),
            n_g: Cell::new(0),
            n_jac_g: Cell::new(0),
            n_h: Cell::new(0),
            captured_solution: RefCell::new(None),
        }
    }

    /// The `(x, lambda)` captured at the last `finalize_solution`, if any.
    pub fn captured_solution(&self) -> Option<(Vec<Number>, Vec<Number>)> {
        self.captured_solution.borrow().clone()
    }
}

impl TNLP for CountingTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.borrow_mut().get_starting_point(sp)
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.n_obj.set(self.n_obj.get() + 1);
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.n_grad_f.set(self.n_grad_f.get() + 1);
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.n_g.set(self.n_g.get() + 1);
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        // Only the values call counts as a real Jacobian evaluation;
        // the symbolic Structure call is bookkeeping.
        if matches!(mode, SparsityRequest::Values { .. }) {
            self.n_jac_g.set(self.n_jac_g.get() + 1);
        }
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
        if matches!(mode, SparsityRequest::Values { .. }) {
            self.n_h.set(self.n_h.get() + 1);
        }
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        *self.captured_solution.borrow_mut() = Some((sol.x.to_vec(), sol.lambda.to_vec()));
        self.inner
            .borrow_mut()
            .finalize_solution(sol, ip_data, ip_cq);
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
