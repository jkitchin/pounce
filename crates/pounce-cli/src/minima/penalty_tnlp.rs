//! Objective-augmenting TNLP wrappers for the repulsion `--minima`
//! strategies (flooding / deflation / tunneling).
//!
//! Each wrapper forwards every TNLP method to an inner problem but adds an
//! analytic penalty term to `eval_f` / `eval_grad_f` (closed-form value and
//! gradient). The Hessian is **declined** (`eval_h` returns `false`, the
//! documented quasi-Newton signal): the driver runs these solves under
//! `hessian_approximation = limited-memory`, so the dense augmented Hessian
//! never has to be assembled. The clean-objective *polish* solve that
//! follows each accepted minimum keeps the exact Hessian.
//!
//! The kernels mirror `_gauss_terms` / `_pole_terms` in
//! `python/pounce/_minima.py` (value + gradient only — the Hessian terms
//! there are unused once the solve goes quasi-Newton).

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, MetaData, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// A repulsion kernel evaluated at the current iterate. Centers are the
/// found minima; the per-dimension scales make the bumps/poles anisotropic.
#[derive(Debug, Clone)]
pub enum Kernel {
    /// Σ Aₖ·exp(−½‖(x−cₖ)/σ‖²) — flooding's Gaussian bumps. `amps` is one
    /// height per center; `inv_sigma2` is the per-dimension 1/σ².
    Gauss {
        centers: Vec<Vec<Number>>,
        amps: Vec<Number>,
        inv_sigma2: Vec<Number>,
    },
    /// Σ η·(‖(x−cₖ)/ℓ‖²+soft)^(−q) — deflation/tunneling's softened poles.
    /// `q = power/2`; `inv_len2` is the per-dimension 1/ℓ².
    Pole {
        centers: Vec<Vec<Number>>,
        eta: Number,
        q: Number,
        soft: Number,
        inv_len2: Vec<Number>,
    },
}

impl Kernel {
    /// Penalty value at `x`.
    pub fn value(&self, x: &[Number]) -> Number {
        match self {
            Kernel::Gauss {
                centers,
                amps,
                inv_sigma2,
            } => {
                let mut val = 0.0;
                for (c, &a) in centers.iter().zip(amps) {
                    let mut q = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - c[i];
                        q += inv_sigma2[i] * d * d;
                    }
                    val += a * (-0.5 * q).exp();
                }
                val
            }
            Kernel::Pole {
                centers,
                eta,
                q,
                soft,
                inv_len2,
            } => {
                let mut val = 0.0;
                for c in centers {
                    let mut r2 = *soft;
                    for i in 0..x.len() {
                        let d = x[i] - c[i];
                        r2 += inv_len2[i] * d * d;
                    }
                    val += eta * r2.powf(-*q);
                }
                val
            }
        }
    }

    /// Add the penalty gradient at `x` into `grad` (in place).
    pub fn add_grad(&self, x: &[Number], grad: &mut [Number]) {
        match self {
            Kernel::Gauss {
                centers,
                amps,
                inv_sigma2,
            } => {
                for (c, &a) in centers.iter().zip(amps) {
                    let mut q = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - c[i];
                        q += inv_sigma2[i] * d * d;
                    }
                    let g = a * (-0.5 * q).exp();
                    // grad += -g · (m∘d)
                    for i in 0..x.len() {
                        let md = inv_sigma2[i] * (x[i] - c[i]);
                        grad[i] -= g * md;
                    }
                }
            }
            Kernel::Pole {
                centers,
                eta,
                q,
                soft,
                inv_len2,
            } => {
                for c in centers {
                    let mut r2 = *soft;
                    for i in 0..x.len() {
                        let d = x[i] - c[i];
                        r2 += inv_len2[i] * d * d;
                    }
                    // coef1 = -2q·η·r2^{-q-1}
                    let coef1 = -2.0 * q * eta * r2.powf(-*q - 1.0);
                    for i in 0..x.len() {
                        let md = inv_len2[i] * (x[i] - c[i]);
                        grad[i] += coef1 * md;
                    }
                }
            }
        }
    }
}

/// Forwarding TNLP that adds a [`Kernel`] to the objective (flooding,
/// deflation). The constraints, Jacobian, bounds, and starting point are
/// untouched — only the objective the solver sees changes.
pub struct PenaltyTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    kernel: Kernel,
}

impl PenaltyTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, kernel: Kernel) -> Self {
        Self { inner, kernel }
    }
}

impl TNLP for PenaltyTnlp {
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
        let base = self.inner.borrow_mut().eval_f(x, new_x)?;
        Some(base + self.kernel.value(x))
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        if !self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f) {
            return false;
        }
        self.kernel.add_grad(x, grad_f);
        true
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
        // Forward to the inner Hessian so the NLP gets a valid sparsity
        // *structure* (otherwise `OrigIpoptNlp` has no `h_space` and the
        // limited-memory path panics). Penalty solves always run under
        // `hessian_approximation = limited-memory`, so the *values* requested
        // here are never consumed by the quasi-Newton update — the analytic
        // penalty Hessian therefore never has to be assembled.
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

/// Forwarding TNLP for the tunneling objective `T(x) = (f(x) − f_ref)² +
/// pole(x)`: a constant-height tunnel away from known minima (Levy &
/// Montalvo 1985). Only the objective changes; the Hessian is declined.
pub struct TunnelTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    f_ref: Number,
    pole: Kernel,
}

impl TunnelTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, f_ref: Number, pole: Kernel) -> Self {
        Self { inner, f_ref, pole }
    }
}

impl TNLP for TunnelTnlp {
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
        let base = self.inner.borrow_mut().eval_f(x, new_x)?;
        let d = base - self.f_ref;
        Some(d * d + self.pole.value(x))
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        // ∇T = 2·(f − f_ref)·∇f + ∇pole
        let base = match self.inner.borrow_mut().eval_f(x, new_x) {
            Some(v) => v,
            None => return false,
        };
        if !self.inner.borrow_mut().eval_grad_f(x, false, grad_f) {
            return false;
        }
        let scale = 2.0 * (base - self.f_ref);
        for g in grad_f.iter_mut() {
            *g *= scale;
        }
        self.pole.add_grad(x, grad_f);
        true
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
        // See `PenaltyTnlp::eval_h`: forward the inner sparsity so the NLP
        // has a valid `h_space`; the limited-memory solve never reads values.
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
