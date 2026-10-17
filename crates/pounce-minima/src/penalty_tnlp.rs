//! Objective-augmenting TNLP wrappers for the repulsion `--minima`
//! strategies (flooding / deflation / tunneling).
//!
//! Each wrapper forwards every TNLP method to an inner problem but adds an
//! analytic penalty term to `eval_f` / `eval_grad_f` (closed-form value and
//! gradient).
//!
//! [`PenaltyTnlp`] can also carry the exact augmented Hessian — the base
//! problem's plus the kernel's, appended as a dense lower triangle (the
//! triplet conversion sums the duplicate entries). The search uses it when
//! the base problem has a Hessian and is small enough for a dense block,
//! which is what `pounce.find_minima(hess=...)` always did; otherwise the
//! solve runs under `hessian_approximation = limited-memory` and only the
//! base sparsity is forwarded. [`TunnelTnlp`] is always quasi-Newton. The
//! clean-objective *polish* solve that follows each accepted minimum keeps
//! the exact Hessian either way.
//!
//! The kernels are `_gauss_terms` / `_pole_terms`'s value, gradient and
//! Hessian.

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

    /// Add the penalty Hessian at `x`, times `factor`, into `lower`: the
    /// packed lower triangle, row by row (`(0,0), (1,0), (1,1), (2,0), …`),
    /// length `n(n+1)/2`.
    pub fn add_hess_lower(&self, x: &[Number], factor: Number, lower: &mut [Number]) {
        let n = x.len();
        match self {
            Kernel::Gauss {
                centers,
                amps,
                inv_sigma2,
            } => {
                for (c, &a) in centers.iter().zip(amps) {
                    let mut q = 0.0;
                    for i in 0..n {
                        let d = x[i] - c[i];
                        q += inv_sigma2[i] * d * d;
                    }
                    // g·(md mdᵀ − diag(m)), md = m∘d
                    let g = factor * a * (-0.5 * q).exp();
                    let mut k = 0;
                    for i in 0..n {
                        let mdi = inv_sigma2[i] * (x[i] - c[i]);
                        for j in 0..=i {
                            let mdj = inv_sigma2[j] * (x[j] - c[j]);
                            let diag = if i == j { inv_sigma2[i] } else { 0.0 };
                            lower[k] += g * (mdi * mdj - diag);
                            k += 1;
                        }
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
                    for i in 0..n {
                        let d = x[i] - c[i];
                        r2 += inv_len2[i] * d * d;
                    }
                    // 4q(q+1)η·r2^{-q-2}·p pᵀ − 2qη·r2^{-q-1}·diag(m), p = m∘d
                    let outer = factor * 4.0 * q * (q + 1.0) * eta * r2.powf(-*q - 2.0);
                    let diag_c = factor * 2.0 * q * eta * r2.powf(-*q - 1.0);
                    let mut k = 0;
                    for i in 0..n {
                        let pi = inv_len2[i] * (x[i] - c[i]);
                        for j in 0..=i {
                            let pj = inv_len2[j] * (x[j] - c[j]);
                            let diag = if i == j { diag_c * inv_len2[i] } else { 0.0 };
                            lower[k] += outer * pi * pj - diag;
                            k += 1;
                        }
                    }
                }
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
    /// `Some((n, base_nnz_h, index_offset))` when this wrapper reports the
    /// exact augmented Hessian; `None` forwards the base Hessian untouched
    /// for a quasi-Newton solve.
    exact: Option<(usize, usize, Index)>,
}

impl PenaltyTnlp {
    /// The quasi-Newton form: the base Hessian is forwarded only so the NLP
    /// has a sparsity structure, and its values are never read.
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, kernel: Kernel) -> Self {
        Self {
            inner,
            kernel,
            exact: None,
        }
    }

    /// The exact form: the Hessian is the base problem's plus the kernel's,
    /// the latter appended as a dense `n(n+1)/2` lower triangle. Returns
    /// `None` when the base problem reports no dimensions.
    pub fn with_exact_hessian(inner: Rc<RefCell<dyn TNLP>>, kernel: Kernel) -> Option<Self> {
        let info = inner.borrow_mut().get_nlp_info()?;
        let offset = match info.index_style {
            pounce_nlp::tnlp::IndexStyle::Fortran => 1,
            pounce_nlp::tnlp::IndexStyle::C => 0,
        };
        Some(Self {
            inner,
            kernel,
            exact: Some((info.n as usize, info.nnz_h_lag as usize, offset)),
        })
    }
}

impl TNLP for PenaltyTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let mut info = self.inner.borrow_mut().get_nlp_info()?;
        if let Some((n, _, _)) = self.exact {
            info.nnz_h_lag += (n * (n + 1) / 2) as Index;
        }
        Some(info)
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
        let Some((n, base_nnz, offset)) = self.exact else {
            // Quasi-Newton: forward to the inner Hessian so the NLP gets a
            // valid sparsity *structure* (otherwise `OrigIpoptNlp` has no
            // `h_space` and the limited-memory path panics). The values
            // requested here are never consumed by the quasi-Newton update.
            return self
                .inner
                .borrow_mut()
                .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode);
        };
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let (ib, it) = irow.split_at_mut(base_nnz);
                let (jb, jt) = jcol.split_at_mut(base_nnz);
                if !self.inner.borrow_mut().eval_h(
                    x,
                    new_x,
                    obj_factor,
                    lambda,
                    new_lambda,
                    SparsityRequest::Structure { irow: ib, jcol: jb },
                ) {
                    return false;
                }
                let mut k = 0;
                for i in 0..n {
                    for j in 0..=i {
                        it[k] = i as Index + offset;
                        jt[k] = j as Index + offset;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let (vb, vt) = values.split_at_mut(base_nnz);
                if !self.inner.borrow_mut().eval_h(
                    x,
                    new_x,
                    obj_factor,
                    lambda,
                    new_lambda,
                    SparsityRequest::Values { values: vb },
                ) {
                    return false;
                }
                vt.fill(0.0);
                let Some(x) = x else { return false };
                self.kernel.add_hess_lower(x, obj_factor, vt);
                true
            }
        }
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
    fn wants_intermediate_callback(&self) -> bool {
        self.inner.borrow().wants_intermediate_callback()
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
    fn wants_intermediate_callback(&self) -> bool {
        self.inner.borrow().wants_intermediate_callback()
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

#[cfg(test)]
mod tests {
    use super::Kernel;

    /// Central differences of `value` must match `add_grad`, for both
    /// kernels, with widths spanning 2.5 orders (ported from
    /// `test_minima_scaling.py::test_anisotropic_kernel_derivatives`).
    #[test]
    fn kernel_gradients_match_finite_differences_of_the_value() {
        let centers = vec![vec![0.1, 0.2, 0.3, 0.4], vec![-0.2, 0.1, 0.0, 0.5]];
        let widths: [f64; 4] = [0.3, 1.0, 5.0, 100.0];
        let inv2: Vec<f64> = widths.iter().map(|w| 1.0 / (w * w)).collect();
        let kernels = [
            Kernel::Gauss {
                centers: centers.clone(),
                amps: vec![2.0, 2.0],
                inv_sigma2: inv2.clone(),
            },
            Kernel::Pole {
                centers,
                eta: 1.5,
                q: 1.0,
                soft: 1e-3,
                inv_len2: inv2,
            },
        ];
        let x = [0.15, 0.25, 0.2, 0.35];
        let h = 1e-6;
        for kernel in &kernels {
            let mut g = vec![0.0; 4];
            kernel.add_grad(&x, &mut g);
            let scale = g.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
            for i in 0..4 {
                let (mut xp, mut xm) = (x, x);
                xp[i] += h;
                xm[i] -= h;
                let fd = (kernel.value(&xp) - kernel.value(&xm)) / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() < 1e-6 * scale,
                    "{kernel:?} [{i}]: {} vs {fd}",
                    g[i]
                );
            }
        }
    }

    /// Central differences of `add_grad` must match `add_hess_lower`, for
    /// both kernels, at an anisotropic point away from every center.
    #[test]
    fn kernel_hessians_match_finite_differences_of_the_gradient() {
        let centers = vec![vec![0.3, -0.2, 0.1], vec![-0.4, 0.5, 0.0]];
        let kernels = [
            Kernel::Gauss {
                centers: centers.clone(),
                amps: vec![1.5, 0.7],
                inv_sigma2: vec![4.0, 9.0, 1.0],
            },
            Kernel::Pole {
                centers,
                eta: 1.0,
                q: 1.0,
                soft: 1e-3,
                inv_len2: vec![25.0, 4.0, 16.0],
            },
        ];
        let x = [0.1, 0.05, -0.2];
        let n = x.len();
        let h = 1e-6;
        for kernel in &kernels {
            let mut lower = vec![0.0; n * (n + 1) / 2];
            kernel.add_hess_lower(&x, 2.0, &mut lower);
            let mut k = 0;
            for i in 0..n {
                let (mut xp, mut xm) = (x, x);
                xp[i] += h;
                xm[i] -= h;
                let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
                kernel.add_grad(&xp, &mut gp);
                kernel.add_grad(&xm, &mut gm);
                for j in 0..=i {
                    let fd = 2.0 * (gp[j] - gm[j]) / (2.0 * h);
                    let tol = 1e-5 * fd.abs().max(1.0);
                    assert!(
                        (lower[k] - fd).abs() < tol,
                        "{kernel:?} ({i},{j}): {} vs {fd}",
                        lower[k]
                    );
                    k += 1;
                }
            }
        }
    }
}
