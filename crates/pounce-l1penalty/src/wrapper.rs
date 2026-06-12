//! [`L1PenaltyBarrierTnlp`] — TNLP wrapper implementing the
//! Thierry-Biegler ℓ₁-exact penalty-barrier reformulation.
//!
//! Variable layout: `[x(n_orig), p(m_eq), n(m_eq)]`. Constraint layout
//! is unchanged from the inner TNLP (same `m` rows in the same order;
//! equality rows have `c_i(x) − p_k + n_k = g_i`, inequality rows pass
//! through unchanged).
//!
//! ## Initial slack seeding
//!
//! For each equality row `i` with target `g_i`, the violation at the
//! user's `x_0` is `r_i = c_i(x_0) − g_i`. We seed
//!
//! ```text
//! p_k = max(0, +r_i) + floor
//! n_k = max(0, −r_i) + floor
//! ```
//!
//! with `floor = 1e-4` so the IPM has a strictly positive interior on
//! the slack bounds (`z = μ/s` requires `s > 0`).
//!
//! ## Hessian
//!
//! The penalty term `ρ · 1ᵀ(p + n)` is **linear** in `(p, n)`, and the
//! constraint contribution `−p + n` is **linear** in `(p, n)`. So the
//! augmented Hessian is exactly the inner Hessian over the original
//! `n_orig` variables; the `(p, n)` block is zero (the IPM's barrier
//! contributes its own `μ/s²` curvature there automatically).
//!
//! ## Borrow discipline
//!
//! The inner TNLP is held as `Rc<RefCell<dyn TNLP>>`. Every method
//! `borrow_mut()`s for the duration of the call. The wrapper's own
//! `&mut self` methods never re-enter via `Rc::clone`, so the borrow
//! is uncontended.
//!
//! ## Reference
//!
//! ripopt 0.8.0 `src/l1_penalty_barrier_nlp.rs` (commit `7847bba9`).
//! Algorithmic equivalence to ripopt's `L1PenaltyBarrierNlp` is the
//! port goal; see [`tests`] for the structural-invariant tests.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, MetaData, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};

/// TNLP wrapper that turns `min f(x) s.t. c(x) = g_target, x ∈ [x_L, x_U]`
/// into the augmented ℓ₁-penalty-barrier formulation; see module docs.
///
/// Constructed by [`L1PenaltyBarrierTnlp::new`]; `Rc::new(RefCell::new(_))`
/// it before passing to `IpoptApplication::optimize_tnlp`.
pub struct L1PenaltyBarrierTnlp {
    inner: Rc<RefCell<dyn TNLP>>,

    n_orig: usize,
    m: usize,
    m_eq: usize,

    /// Indices into the inner constraint vector that are equality
    /// rows, in ascending order. `eq_rows.len() == m_eq`.
    eq_rows: Vec<usize>,

    /// Reverse map: `row_to_slot[i] == Some(k)` iff inner row `i` is
    /// equality row `eq_rows[k]`; `None` otherwise. Reserved for
    /// Phase 2's algorithm-side wiring (multiplier mapping).
    #[allow(dead_code)]
    row_to_slot: Vec<Option<usize>>,

    /// Equality-row targets, indexed by `k ∈ 0..m_eq`. `g_target[k] ==
    /// inner.g_l[eq_rows[k]] == inner.g_u[eq_rows[k]]`. Reserved for
    /// Phase 3's BNW outer loop, which reads the actual residual
    /// `c(x*) − g_target` to drive ρ.
    #[allow(dead_code)]
    g_target: Vec<Number>,

    /// Number of nonzeros in the inner constraint Jacobian.
    inner_jac_nnz: usize,

    /// Initial seeded values for the slack variables `p` and `n`,
    /// length `m_eq` each.
    p_init: Vec<Number>,
    n_init: Vec<Number>,

    /// Penalty weight `ρ`. Set per inner solve; the Phase-3 driver
    /// updates this via `set_rho` between outer iterations.
    rho: Number,

    /// Cached inner index style — needed to translate inner Jacobian
    /// row indices into the equality-row matching test (the inner
    /// reports indices in its own style; the wrapper reports the same
    /// `index_style` so the augmented (irow, jcol) buffer the IPM
    /// gives us is in the same convention).
    index_style: IndexStyle,

    // ---- Phase-3 state-capture fields (populated by finalize_solution) ----
    /// When `true`, [`Self::finalize_solution`] only captures the
    /// final augmented Solution into the fields below and does **not**
    /// forward to the inner TNLP. The Phase-3 outer driver sets this
    /// so it can run multiple inner solves at escalating ρ values and
    /// only call `inner.finalize_solution` once, after the loop. When
    /// `false` (the Phase-1/2 default for direct one-shot wrapper use),
    /// the wrapper does its own back-projection and forwards.
    defer_inner_finalize: bool,
    /// Captured: the truncated (original-space) `x*` from the most
    /// recent `finalize_solution` call. Empty if never called.
    last_x_trunc: Vec<Number>,
    /// Captured: `Σ p_k + n_k` of the augmented slacks. `NaN` if
    /// `finalize_solution` hasn't run yet.
    last_slack_sum: Number,
    /// Captured: `‖λ_eq‖∞` across the inner equality-row multipliers.
    /// Reads `sol.lambda[i]` for each `i` in `eq_rows`; the BNW driver
    /// uses this for the ρ-steering update.
    last_y_eq_inf_norm: Number,
    /// Captured: status reported by the inner IPM at the most recent
    /// inner-solve termination.
    last_status: Option<SolverReturn>,
    /// Captured: inner-space `λ` of length `m_inner` (just the input
    /// `sol.lambda` cloned — the wrapper does not modify it; the
    /// driver forwards it on the final inner finalize).
    last_lambda: Vec<Number>,
    /// Captured: inner-space `z_l` truncated to `n_orig`.
    last_z_l_trunc: Vec<Number>,
    /// Captured: inner-space `z_u` truncated to `n_orig`.
    last_z_u_trunc: Vec<Number>,
}

impl L1PenaltyBarrierTnlp {
    /// Wrap `inner` with fixed penalty weight `rho`.
    ///
    /// Calls `inner.get_nlp_info`, `get_bounds_info`,
    /// `get_starting_point`, and `eval_g` to detect equality rows and
    /// seed `(p, n)` from the constraint violation at `x_0`. If any
    /// of these fail, returns `None` — the caller should not enable
    /// the wrapper for that TNLP.
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, rho: Number) -> Option<Self> {
        let info = inner.borrow_mut().get_nlp_info()?;
        let n = info.n as usize;
        let m = info.m as usize;
        let inner_jac_nnz = info.nnz_jac_g as usize;
        let index_style = info.index_style;

        // Discover equality rows by reading inner bounds.
        let mut x_l = vec![Number::NEG_INFINITY; n];
        let mut x_u = vec![Number::INFINITY; n];
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        let ok = inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        });
        if !ok {
            return None;
        }
        let mut eq_rows = Vec::new();
        let mut g_target = Vec::new();
        let mut row_to_slot = vec![None; m];
        for i in 0..m {
            if g_l[i] == g_u[i] && g_l[i].is_finite() {
                row_to_slot[i] = Some(eq_rows.len());
                eq_rows.push(i);
                g_target.push(g_l[i]);
            }
        }
        let m_eq = eq_rows.len();

        // Seed (p, n) from the inner violation at the user's x_0.
        let mut x0 = vec![0.0; n];
        let mut z_l_ignore = vec![0.0; n];
        let mut z_u_ignore = vec![0.0; n];
        let mut lambda_ignore = vec![0.0; m];
        let sp_ok = inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x0,
            init_z: false,
            z_l: &mut z_l_ignore,
            z_u: &mut z_u_ignore,
            init_lambda: false,
            lambda: &mut lambda_ignore,
        });
        if !sp_ok {
            return None;
        }
        let mut g0 = vec![0.0; m];
        if m > 0 && !inner.borrow_mut().eval_g(&x0, true, &mut g0) {
            return None;
        }
        let floor: Number = 1e-4;
        let mut p_init = vec![floor; m_eq];
        let mut n_init = vec![floor; m_eq];
        for (k, &row) in eq_rows.iter().enumerate() {
            let viol = g0[row] - g_target[k];
            if viol > 0.0 {
                p_init[k] = viol + floor;
            } else if viol < 0.0 {
                n_init[k] = -viol + floor;
            }
        }

        Some(Self {
            inner,
            n_orig: n,
            m,
            m_eq,
            eq_rows,
            row_to_slot,
            g_target,
            inner_jac_nnz,
            p_init,
            n_init,
            rho,
            index_style,
            defer_inner_finalize: false,
            last_x_trunc: Vec::new(),
            last_slack_sum: Number::NAN,
            last_y_eq_inf_norm: 0.0,
            last_status: None,
            last_lambda: Vec::new(),
            last_z_l_trunc: Vec::new(),
            last_z_u_trunc: Vec::new(),
        })
    }

    /// Phase-3: update ρ between outer iterations.
    pub fn set_rho(&mut self, rho: Number) {
        self.rho = rho;
    }

    /// Phase-3 outer-loop accessor: control whether
    /// [`Self::finalize_solution`] forwards to the inner TNLP. When
    /// `true`, finalize captures state but doesn't forward — the
    /// driver does the single forward after the outer loop.
    pub fn set_defer_inner_finalize(&mut self, defer: bool) {
        self.defer_inner_finalize = defer;
    }

    /// Phase-3 accessors used by the BNW driver to read state captured
    /// by the most recent `finalize_solution` call.
    pub fn last_slack_sum(&self) -> Number {
        self.last_slack_sum
    }
    pub fn last_y_eq_inf_norm(&self) -> Number {
        self.last_y_eq_inf_norm
    }
    pub fn last_x_trunc(&self) -> &[Number] {
        &self.last_x_trunc
    }
    pub fn last_status(&self) -> Option<SolverReturn> {
        self.last_status
    }
    pub fn last_lambda(&self) -> &[Number] {
        &self.last_lambda
    }
    pub fn last_z_l_trunc(&self) -> &[Number] {
        &self.last_z_l_trunc
    }
    pub fn last_z_u_trunc(&self) -> &[Number] {
        &self.last_z_u_trunc
    }
    /// `true` iff [`Self::finalize_solution`] has run at least once.
    pub fn has_solution(&self) -> bool {
        self.last_status.is_some()
    }

    /// Original (un-augmented) variable count. Used by the algorithm-
    /// side driver to truncate the result back to user-visible space.
    pub fn n_orig(&self) -> usize {
        self.n_orig
    }

    /// Equality-row indices into the inner constraint vector, in
    /// ascending order. The Phase-3 BNW outer loop reads
    /// `lambda[eq_rows[k]]` to drive the ρ update.
    pub fn eq_rows(&self) -> &[usize] {
        &self.eq_rows
    }

    /// Slack count `m_eq`. Augmented variable count is
    /// `n_orig + 2 * m_eq`.
    pub fn m_eq(&self) -> usize {
        self.m_eq
    }

    /// Current penalty weight ρ.
    pub fn rho(&self) -> Number {
        self.rho
    }
}

impl TNLP for L1PenaltyBarrierTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        // Pass through inner Hessian nnz unchanged: the augmented
        // Hessian is exactly the inner Hessian over the original
        // variables (penalty term and constraint contribution are both
        // linear in (p, n)).
        let inner_info = self.inner.borrow_mut().get_nlp_info()?;
        let n_aug = self.n_orig + 2 * self.m_eq;
        let nnz_jac_aug = self.inner_jac_nnz + 2 * self.m_eq;
        Some(NlpInfo {
            n: n_aug as Index,
            m: self.m as Index,
            nnz_jac_g: nnz_jac_aug as Index,
            nnz_h_lag: inner_info.nnz_h_lag,
            index_style: self.index_style,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let n = self.n_orig;
        debug_assert_eq!(b.x_l.len(), n + 2 * self.m_eq);
        debug_assert_eq!(b.x_u.len(), n + 2 * self.m_eq);
        let inner_ok = self.inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut b.x_l[..n],
            x_u: &mut b.x_u[..n],
            g_l: b.g_l,
            g_u: b.g_u,
        });
        if !inner_ok {
            return false;
        }
        // Slack bounds: (0, +∞) for both p and n.
        for k in 0..2 * self.m_eq {
            b.x_l[n + k] = 0.0;
            b.x_u[n + k] = Number::INFINITY;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        debug_assert_eq!(sp.x.len(), n + 2 * m_eq);
        let inner_ok = self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: sp.init_x,
            x: &mut sp.x[..n],
            init_z: sp.init_z,
            // The inner cares about z for its original variables only;
            // the augmented z entries for slack bounds are seeded by
            // the IPM's default (μ at iter 0). We pass through the
            // first n entries of the wrapper's z buffers.
            z_l: &mut sp.z_l[..n],
            z_u: &mut sp.z_u[..n],
            init_lambda: sp.init_lambda,
            lambda: sp.lambda,
        });
        if !inner_ok {
            return false;
        }
        sp.x[n..n + m_eq].copy_from_slice(&self.p_init);
        sp.x[n + m_eq..n + 2 * m_eq].copy_from_slice(&self.n_init);
        // Slack-side z entries: leave at whatever the IPM defaults to
        // (Phase 2 wiring will revisit if needed).
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        let n = self.n_orig;
        let f_inner = self.inner.borrow_mut().eval_f(&x[..n], new_x)?;
        let mut acc: Number = 0.0;
        for k in 0..2 * self.m_eq {
            acc += x[n + k];
        }
        Some(f_inner + self.rho * acc)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        let n = self.n_orig;
        let inner_ok = self
            .inner
            .borrow_mut()
            .eval_grad_f(&x[..n], new_x, &mut grad_f[..n]);
        if !inner_ok {
            return false;
        }
        for k in 0..2 * self.m_eq {
            grad_f[n + k] = self.rho;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        let inner_ok = self.inner.borrow_mut().eval_g(&x[..n], new_x, g);
        if !inner_ok {
            return false;
        }
        // For each equality row i with slot k, the augmented constraint
        // is c_i(x) − p_k + n_k = g_target_k.  We report c_i(x) − p_k +
        // n_k here; the IPM compares against (g_l_i, g_u_i) which the
        // wrapper passes through unchanged from the inner (so the
        // equality target is g_target_k on both sides).
        for (k, &row) in self.eq_rows.iter().enumerate() {
            g[row] = g[row] - x[n + k] + x[n + m_eq + k];
        }
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        let inner_nnz = self.inner_jac_nnz;
        // The inner TNLP only knows about its original `n` variables, so
        // hand it the original-space slice of x — exactly as eval_f /
        // eval_grad_f / eval_g / eval_h do. Forwarding the full augmented
        // slice (length n + 2*m_eq) breaks any inner that checks x.len()
        // or iterates the slice.
        let inner_x = x.map(|xa| &xa[..n]);
        let n_idx_offset: Index = match self.index_style {
            IndexStyle::C => 0,
            IndexStyle::Fortran => 1,
        };

        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                debug_assert_eq!(irow.len(), inner_nnz + 2 * m_eq);
                debug_assert_eq!(jcol.len(), inner_nnz + 2 * m_eq);
                let inner_ok = self.inner.borrow_mut().eval_jac_g(
                    inner_x,
                    new_x,
                    SparsityRequest::Structure {
                        irow: &mut irow[..inner_nnz],
                        jcol: &mut jcol[..inner_nnz],
                    },
                );
                if !inner_ok {
                    return false;
                }
                // -1 entries for p_k at column n + k (in inner's index
                // style, so add n_idx_offset for Fortran-1-based).
                for (k, &row) in self.eq_rows.iter().enumerate() {
                    let row_idx = (row as Index) + n_idx_offset;
                    irow[inner_nnz + k] = row_idx;
                    jcol[inner_nnz + k] = (n as Index) + (k as Index) + n_idx_offset;
                }
                // +1 entries for n_k at column n + m_eq + k.
                for (k, &row) in self.eq_rows.iter().enumerate() {
                    let row_idx = (row as Index) + n_idx_offset;
                    irow[inner_nnz + m_eq + k] = row_idx;
                    jcol[inner_nnz + m_eq + k] =
                        (n as Index) + (m_eq as Index) + (k as Index) + n_idx_offset;
                }
                true
            }
            SparsityRequest::Values { values } => {
                debug_assert_eq!(values.len(), inner_nnz + 2 * m_eq);
                let inner_ok = self.inner.borrow_mut().eval_jac_g(
                    inner_x,
                    new_x,
                    SparsityRequest::Values {
                        values: &mut values[..inner_nnz],
                    },
                );
                if !inner_ok {
                    return false;
                }
                for k in 0..m_eq {
                    values[inner_nnz + k] = -1.0;
                    values[inner_nnz + m_eq + k] = 1.0;
                }
                true
            }
        }
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
        // Augmented Hessian == inner Hessian over original variables.
        // Pass the original-space slice of x through; lambda is the
        // same length (m unchanged) so it passes straight through —
        // the equality-row multiplier is the same scalar in both
        // augmented and inner formulations because the slack
        // contributions to c(x) are linear.
        let n = self.n_orig;
        let inner_x = x.map(|xa| &xa[..n]);
        self.inner
            .borrow_mut()
            .eval_h(inner_x, new_x, obj_factor, lambda, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        // Always capture: the Phase-3 driver reads these between
        // outer iterations. Forwarding to the inner is conditional on
        // `defer_inner_finalize` so the driver can run multiple inner
        // solves and only forward once at the end.
        let n = self.n_orig;
        let m_eq = self.m_eq;
        let aug_len = n + 2 * m_eq;
        debug_assert!(sol.x.len() >= aug_len, "augmented x too short");
        self.last_x_trunc.clear();
        self.last_x_trunc.extend_from_slice(&sol.x[..n]);
        self.last_slack_sum = if m_eq > 0 {
            sol.x[n..aug_len].iter().sum()
        } else {
            0.0
        };
        self.last_y_eq_inf_norm = self
            .eq_rows
            .iter()
            .map(|&i| sol.lambda.get(i).copied().unwrap_or(0.0).abs())
            .fold(0.0_f64, Number::max);
        self.last_status = Some(sol.status);
        self.last_lambda.clear();
        self.last_lambda.extend_from_slice(sol.lambda);
        self.last_z_l_trunc.clear();
        self.last_z_l_trunc
            .extend_from_slice(&sol.z_l[..n.min(sol.z_l.len())]);
        self.last_z_u_trunc.clear();
        self.last_z_u_trunc
            .extend_from_slice(&sol.z_u[..n.min(sol.z_u.len())]);

        if self.defer_inner_finalize {
            // Phase-3 outer-loop mode: driver handles the single inner
            // forward after the loop. Don't call inner.finalize_solution.
            return;
        }

        // Phase-1/2 one-shot mode: do the back-projection here.
        let x_trunc = &sol.x[..n];
        let f_inner = self
            .inner
            .borrow_mut()
            .eval_f(x_trunc, true)
            .unwrap_or(sol.obj_value);
        let mut g_inner = vec![0.0; self.m];
        if self.m > 0 {
            let _ = self.inner.borrow_mut().eval_g(x_trunc, false, &mut g_inner);
        }
        let inner_sol = Solution {
            status: sol.status,
            x: x_trunc,
            z_l: &sol.z_l[..n.min(sol.z_l.len())],
            z_u: &sol.z_u[..n.min(sol.z_u.len())],
            g: &g_inner,
            lambda: sol.lambda,
            obj_value: f_inner,
        };
        self.inner
            .borrow_mut()
            .finalize_solution(inner_sol, ip_data, ip_cq);
    }

    // --- Pass-through optional methods ---

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        // The wrapper introduces 2*m_eq new variables (p, n) but does
        // not name them — the inner's metadata describes only the
        // first n_orig entries. Phase 2 may extend var with default
        // names; for now pass through and let downstream code handle
        // unnamed entries.
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn get_scaling_parameters(&mut self, _req: ScalingRequest<'_>) -> bool {
        // Defer scaling support: the wrapper's slack variables and
        // augmented objective term should be scaled consistently with
        // the inner; ripopt's reference also passes through. Phase 2.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal equality-only test problem:
    /// `min x[0]^2 + x[1]^2  s.t. x[0] + x[1] = 1`. Optimum at
    /// `(0.5, 0.5)`, `f* = 0.5`.
    struct EqOnly;

    impl TNLP for EqOnly {
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
            for v in b.x_l.iter_mut() {
                *v = -1e19;
            }
            for v in b.x_u.iter_mut() {
                *v = 1e19;
            }
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0;
            sp.x[1] = 0.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[0] + x[1] * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
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
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 0;
                    jcol[1] = 1;
                    true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 1.0;
                    values[1] = 1.0;
                    true
                }
            }
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
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 1;
                    jcol[1] = 1;
                    true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 2.0 * obj_factor;
                    values[1] = 2.0 * obj_factor;
                    true
                }
            }
        }
        fn finalize_solution(
            &mut self,
            _sol: Solution<'_>,
            _ip_data: &IpoptData,
            _ip_cq: &IpoptCq,
        ) {
        }
    }

    fn wrap(rho: Number) -> L1PenaltyBarrierTnlp {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(EqOnly));
        L1PenaltyBarrierTnlp::new(inner, rho).expect("wrapper construction")
    }

    #[test]
    fn dimensions_lift() {
        let mut w = wrap(1.0);
        let info = w.get_nlp_info().expect("get_nlp_info");
        // n_aug = 2 + 2*1, m unchanged, jac_nnz = 2 + 2*1 = 4, hess unchanged.
        assert_eq!(info.n, 4);
        assert_eq!(info.m, 1);
        assert_eq!(info.nnz_jac_g, 4);
        assert_eq!(info.nnz_h_lag, 2);
        assert_eq!(w.n_orig(), 2);
        assert_eq!(w.m_eq(), 1);
        assert_eq!(w.eq_rows(), &[0]);
    }

    #[test]
    fn bounds_lift_with_slack_nonneg() {
        let mut w = wrap(1.0);
        let mut x_l = vec![0.0; 4];
        let mut x_u = vec![0.0; 4];
        let mut g_l = vec![0.0; 1];
        let mut g_u = vec![0.0; 1];
        assert!(w.get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        }));
        // Original vars unbounded.
        assert!(x_l[0] <= -1e18 && x_u[0] >= 1e18);
        assert!(x_l[1] <= -1e18 && x_u[1] >= 1e18);
        // Slacks: lower 0, upper +∞.
        assert_eq!(x_l[2], 0.0);
        assert_eq!(x_l[3], 0.0);
        assert!(x_u[2].is_infinite() && x_u[2] > 0.0);
        assert!(x_u[3].is_infinite() && x_u[3] > 0.0);
        // Constraint bounds passed through.
        assert_eq!(g_l[0], 1.0);
        assert_eq!(g_u[0], 1.0);
    }

    #[test]
    fn objective_includes_penalty_term() {
        let mut w = wrap(2.5);
        // x = (1, 1, 0.3, 0.7) — pure penalty contribution should be
        // 2.5 * (0.3 + 0.7) = 2.5; original f = 1 + 1 = 2.0.
        let x = [1.0, 1.0, 0.3, 0.7];
        let f = w.eval_f(&x, true).unwrap();
        assert!((f - (2.0 + 2.5 * 1.0)).abs() < 1e-12, "got {}", f);
    }

    #[test]
    fn gradient_slack_entries_are_rho() {
        let mut w = wrap(3.0);
        let x = [0.5, 0.5, 0.0, 0.0];
        let mut g = vec![0.0; 4];
        assert!(w.eval_grad_f(&x, true, &mut g));
        assert_eq!(g[0], 1.0); // 2*0.5
        assert_eq!(g[1], 1.0);
        assert_eq!(g[2], 3.0); // ρ
        assert_eq!(g[3], 3.0);
    }

    #[test]
    fn constraint_value_includes_minus_p_plus_n() {
        let mut w = wrap(1.0);
        // x = (0.4, 0.5, 0.2, 0.3): inner c = 0.9; aug c = 0.9 - 0.2 + 0.3 = 1.0.
        let x = [0.4, 0.5, 0.2, 0.3];
        let mut g = vec![0.0; 1];
        assert!(w.eval_g(&x, true, &mut g));
        assert!((g[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jacobian_structure_appends_slack_entries() {
        let mut w = wrap(1.0);
        let mut irow = vec![0i32; 4];
        let mut jcol = vec![0i32; 4];
        assert!(w.eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            },
        ));
        // First two entries from inner: (0, 0) and (0, 1).
        assert_eq!((irow[0], jcol[0]), (0, 0));
        assert_eq!((irow[1], jcol[1]), (0, 1));
        // Slack p_0 at column 2 (n_orig + 0).
        assert_eq!((irow[2], jcol[2]), (0, 2));
        // Slack n_0 at column 3 (n_orig + m_eq + 0).
        assert_eq!((irow[3], jcol[3]), (0, 3));
    }

    #[test]
    fn jacobian_values_slack_entries_are_minus1_plus1() {
        let mut w = wrap(1.0);
        let x = [0.4, 0.5, 0.2, 0.3];
        let mut vals = vec![0.0; 4];
        assert!(w.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Values { values: &mut vals },
        ));
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], -1.0);
        assert_eq!(vals[3], 1.0);
    }

    #[test]
    fn jacobian_passes_inner_only_original_x() {
        // Regression for M8: the wrapper must hand the inner TNLP only the
        // original-variable slice of x in eval_jac_g, exactly as eval_f /
        // eval_grad_f / eval_g / eval_h do. Previously the full augmented
        // slice (length n_orig + 2*m_eq) leaked through, so an inner that
        // checks x.len() or iterates the slice would misbehave.
        use std::cell::Cell;

        struct LenSpy {
            seen_len: Rc<Cell<usize>>,
        }
        impl TNLP for LenSpy {
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
                for v in b.x_l.iter_mut() {
                    *v = -1e19;
                }
                for v in b.x_u.iter_mut() {
                    *v = 1e19;
                }
                b.g_l[0] = 1.0;
                b.g_u[0] = 1.0;
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x[0] = 0.0;
                sp.x[1] = 0.0;
                true
            }
            fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
                Some(x[0] * x[0] + x[1] * x[1])
            }
            fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
                true
            }
            fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
                g[0] = x[0] + x[1];
                true
            }
            fn eval_jac_g(
                &mut self,
                x: Option<&[Number]>,
                _new_x: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                // Record the length of x the wrapper handed us.
                if let Some(xs) = x {
                    self.seen_len.set(xs.len());
                }
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow[0] = 0;
                        jcol[0] = 0;
                        irow[1] = 0;
                        jcol[1] = 1;
                        true
                    }
                    SparsityRequest::Values { values } => {
                        values[0] = 1.0;
                        values[1] = 1.0;
                        true
                    }
                }
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
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow[0] = 0;
                        jcol[0] = 0;
                        irow[1] = 1;
                        jcol[1] = 1;
                        true
                    }
                    SparsityRequest::Values { values } => {
                        values[0] = 2.0 * obj_factor;
                        values[1] = 2.0 * obj_factor;
                        true
                    }
                }
            }
            fn finalize_solution(
                &mut self,
                _sol: Solution<'_>,
                _ip_data: &IpoptData,
                _ip_cq: &IpoptCq,
            ) {
            }
        }

        let seen = Rc::new(Cell::new(0usize));
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(LenSpy {
            seen_len: Rc::clone(&seen),
        }));
        let mut w = L1PenaltyBarrierTnlp::new(inner, 1.0).expect("wrapper construction");

        // Augmented x has length n_orig + 2*m_eq = 2 + 2 = 4.
        let x = [0.4, 0.5, 0.2, 0.3];
        let mut vals = vec![0.0; 4];
        assert!(w.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Values { values: &mut vals },
        ));
        // The inner must have seen only its 2 original variables, not 4.
        assert_eq!(
            seen.get(),
            2,
            "inner eval_jac_g received x of length {} but expected the {} original vars only",
            seen.get(),
            2
        );
    }

    #[test]
    fn hessian_passes_through_unchanged() {
        let mut w = wrap(1.0);
        // Structure call.
        let mut irow = vec![0i32; 2];
        let mut jcol = vec![0i32; 2];
        assert!(w.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            },
        ));
        assert_eq!((irow[0], jcol[0]), (0, 0));
        assert_eq!((irow[1], jcol[1]), (1, 1));

        // Values call: with obj_factor=2, expect [4, 4].
        let x = [0.0, 0.0, 0.0, 0.0];
        let lambda = [0.0];
        let mut vals = vec![0.0; 2];
        assert!(w.eval_h(
            Some(&x),
            true,
            2.0,
            Some(&lambda),
            true,
            SparsityRequest::Values { values: &mut vals },
        ));
        assert_eq!(vals[0], 4.0);
        assert_eq!(vals[1], 4.0);
    }

    #[test]
    fn starting_point_seeds_slacks_from_violation() {
        // EqOnly's x_0 = (0, 0); inner c(x_0) = 0; target = 1; viol = -1.
        // Expect p ≈ floor (since viol < 0) and n ≈ |viol| + floor = 1 + 1e-4.
        let mut w = wrap(1.0);
        let mut x = vec![0.0; 4];
        let mut z_l = vec![0.0; 4];
        let mut z_u = vec![0.0; 4];
        let mut lam = vec![0.0; 1];
        assert!(w.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: false,
            lambda: &mut lam,
        }));
        assert_eq!(x[0], 0.0);
        assert_eq!(x[1], 0.0);
        assert!((x[2] - 1e-4).abs() < 1e-12, "p got {}", x[2]);
        assert!((x[3] - (1.0 + 1e-4)).abs() < 1e-12, "n got {}", x[3]);
    }

    #[test]
    fn equality_row_detection_skips_inequalities() {
        struct Mixed;
        impl TNLP for Mixed {
            fn get_nlp_info(&mut self) -> Option<NlpInfo> {
                Some(NlpInfo {
                    n: 1,
                    m: 3,
                    nnz_jac_g: 3,
                    nnz_h_lag: 1,
                    index_style: IndexStyle::C,
                })
            }
            fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
                b.x_l[0] = -1e19;
                b.x_u[0] = 1e19;
                // row 0: equality at 0
                b.g_l[0] = 0.0;
                b.g_u[0] = 0.0;
                // row 1: inequality 1 ≤ . ≤ 2
                b.g_l[1] = 1.0;
                b.g_u[1] = 2.0;
                // row 2: equality at 5
                b.g_l[2] = 5.0;
                b.g_u[2] = 5.0;
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x[0] = 0.0;
                true
            }
            fn eval_f(&mut self, _x: &[Number], _: bool) -> Option<Number> {
                Some(0.0)
            }
            fn eval_grad_f(&mut self, _x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g[0] = 0.0;
                true
            }
            fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g[0] = x[0];
                g[1] = x[0];
                g[2] = x[0];
                true
            }
            fn eval_jac_g(
                &mut self,
                _x: Option<&[Number]>,
                _: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow[0] = 0;
                        jcol[0] = 0;
                        irow[1] = 1;
                        jcol[1] = 0;
                        irow[2] = 2;
                        jcol[2] = 0;
                        true
                    }
                    SparsityRequest::Values { values } => {
                        values[0] = 1.0;
                        values[1] = 1.0;
                        values[2] = 1.0;
                        true
                    }
                }
            }
            fn finalize_solution(&mut self, _sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
        }
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mixed));
        let mut w = L1PenaltyBarrierTnlp::new(inner, 1.0).unwrap();
        // Two equality rows (0 and 2), so m_eq = 2.
        assert_eq!(w.m_eq(), 2);
        assert_eq!(w.eq_rows(), &[0, 2]);
        // Augmented n = 1 + 2*2 = 5.
        let info = w.get_nlp_info().unwrap();
        assert_eq!(info.n, 5);
    }

    /// Like [`EqOnly`] but lets a chosen seed-phase callback report
    /// failure, so we can check `new()` honors its documented "returns
    /// `None` if any of these fail" contract instead of silently
    /// seeding `(p, n)` from zeroed `x_0`/`g_0`.
    struct SeedFails {
        fail_starting_point: bool,
        fail_eval_g: bool,
    }

    impl TNLP for SeedFails {
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
            for v in b.x_l.iter_mut() {
                *v = -1e19;
            }
            for v in b.x_u.iter_mut() {
                *v = 1e19;
            }
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0;
            sp.x[1] = 0.0;
            !self.fail_starting_point
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[0] + x[1] * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            !self.fail_eval_g
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 0;
                    jcol[1] = 1;
                    true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 1.0;
                    values[1] = 1.0;
                    true
                }
            }
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
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 1;
                    jcol[1] = 1;
                    true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 2.0 * obj_factor;
                    values[1] = 2.0 * obj_factor;
                    true
                }
            }
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn new_returns_none_when_starting_point_fails() {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeedFails {
            fail_starting_point: true,
            fail_eval_g: false,
        }));
        assert!(
            L1PenaltyBarrierTnlp::new(inner, 1.0).is_none(),
            "new() must reject a TNLP whose get_starting_point fails \
             instead of seeding (p, n) from a zeroed x_0"
        );
    }

    #[test]
    fn new_returns_none_when_eval_g_fails() {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeedFails {
            fail_starting_point: false,
            fail_eval_g: true,
        }));
        assert!(
            L1PenaltyBarrierTnlp::new(inner, 1.0).is_none(),
            "new() must reject a TNLP whose eval_g fails instead of \
             seeding (p, n) from a zeroed violation"
        );
    }

    #[test]
    fn new_succeeds_when_seed_callbacks_succeed() {
        // Control: the same mock with both callbacks succeeding wraps fine,
        // so the None results above are attributable to the failures.
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeedFails {
            fail_starting_point: false,
            fail_eval_g: false,
        }));
        assert!(L1PenaltyBarrierTnlp::new(inner, 1.0).is_some());
    }
}
