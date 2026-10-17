//! [`CaptureTnlp`]: the TNLP wrapper `IpoptLocalSolve` hands the
//! application, so it can read each solve's answer from `finalize_solution`.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, InfeasibilityProof, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// The `(x, lambda)` of the last `finalize_solution` a [`CaptureTnlp`] saw.
pub(crate) type FinalSlot = Rc<RefCell<Option<(Vec<Number>, Vec<Number>)>>>;

/// Forwards every method to `inner` unchanged and records the
/// `finalize_solution` payload in `slot`.
///
/// That payload is what the application reports as the answer: the full
/// `x` after `honor_original_bounds`, and `lambda` in the user's
/// `∇f + λ·∇g + z = 0` convention. When an attempt loses and an earlier
/// one's answer is restored from the floor, the restore re-delivers it
/// through `finalize_solution` too, so the slot always holds the answer
/// the status describes.
pub(crate) struct CaptureTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    slot: FinalSlot,
}

impl CaptureTnlp {
    pub(crate) fn new(inner: Rc<RefCell<dyn TNLP>>, slot: FinalSlot) -> Self {
        Self { inner, slot }
    }
}

impl TNLP for CaptureTnlp {
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
        // The last delivery wins: a floor replay re-delivers the winning
        // attempt's payload after the losing one's (gh#884).
        *self.slot.borrow_mut() = Some((sol.x.to_vec(), sol.lambda.to_vec()));
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
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }
    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    /// Transparent decorator: forward the constant-derivative proofs, or
    /// `OrigIpoptNlp` asks this wrapper — which knows no algebra — and
    /// gets the declining default, so gh #588 Q6's reuse never engages on
    /// any model the CLI solves. Neither wrapper changes a row, a
    /// variable or a derivative's value, so both directions of the proof
    /// carry through unchanged.
    fn derivative_proofs(&mut self) -> pounce_nlp::constant_derivatives::DerivativeProofs {
        self.inner.borrow_mut().derivative_proofs()
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

    /// Transparent decorator: forward the presolve infeasibility proof so a
    /// certified-empty feasible region is not swallowed on the way up.
    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }
    fn is_presolve_wrapper(&self) -> bool {
        self.inner.borrow().is_presolve_wrapper()
    }
    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.borrow().scaling_factors()
    }
}
