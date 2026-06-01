//! Local NLP upper bounds: a tape→`TNLP` adapter feeding the filter
//! line-search interior-point solver in `pounce-algorithm`.
//!
//! At a node, polishing the relaxation point with a genuine local solve yields
//! a feasible point of the *original* problem far closer to a local minimum
//! than the relaxation solution alone — a sharper incumbent, hence more
//! pruning. First derivatives are the exact reverse-mode gradients from
//! [`crate::ad`]; the Lagrangian Hessian is finite-differenced from those
//! gradients (the local solve only needs a usable Newton direction, not an
//! exact Hessian). The Jacobian's structural sparsity is read off the tapes;
//! the Hessian is declared dense (problems here are low-dimensional).

use crate::ad::{accumulate_gradient, gradient, referenced_vars};
use crate::expr::eval;
use crate::problem::GlobalProblem;
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_nlp::FbbtTape;
use std::cell::RefCell;
use std::rc::Rc;

/// Owns clones of the problem tapes (so the adapter is `'static`, as
/// `optimize_tnlp` requires) plus the node box and starting point.
struct TapeTnlp {
    n: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    x0: Vec<f64>,
    objective: FbbtTape,
    con_tapes: Vec<FbbtTape>,
    con_lo: Vec<f64>,
    con_hi: Vec<f64>,
    /// `(constraint row, variable col)` Jacobian entries, grouped by row.
    jac: Vec<(usize, usize)>,
    final_x: Option<Vec<f64>>,
}

impl TapeTnlp {
    /// Lagrangian gradient `obj_factor·∇f + Σ λⱼ ∇gⱼ` at `x`.
    fn lagrangian_grad(&self, x: &[f64], obj_factor: f64, lambda: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.n];
        accumulate_gradient(&self.objective, x, obj_factor, &mut g);
        for (j, t) in self.con_tapes.iter().enumerate() {
            let lj = lambda.get(j).copied().unwrap_or(0.0);
            if lj != 0.0 {
                accumulate_gradient(t, x, lj, &mut g);
            }
        }
        g
    }
}

impl TNLP for TapeTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n;
        Some(NlpInfo {
            n: n as Index,
            m: self.con_tapes.len() as Index,
            nnz_jac_g: self.jac.len() as Index,
            nnz_h_lag: (n * (n + 1) / 2) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.lo);
        b.x_u.copy_from_slice(&self.hi);
        b.g_l.copy_from_slice(&self.con_lo);
        b.g_u.copy_from_slice(&self.con_hi);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for i in 0..self.n {
            sp.x[i] = self.x0[i].clamp(self.lo[i], self.hi[i]);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let f = eval(&self.objective, x);
        f.is_finite().then_some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        grad_f.iter_mut().for_each(|g| *g = 0.0);
        accumulate_gradient(&self.objective, x, 1.0, grad_f);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (j, t) in self.con_tapes.iter().enumerate() {
            g[j] = eval(t, x);
        }
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
                for (k, &(r, c)) in self.jac.iter().enumerate() {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                // Walk grouped-by-row entries, recomputing each row's gradient
                // once as the constraint index advances.
                let mut cur_row = usize::MAX;
                let mut grad = vec![0.0; self.n];
                for (k, &(r, c)) in self.jac.iter().enumerate() {
                    if r != cur_row {
                        grad = gradient(&self.con_tapes[r], x, self.n);
                        cur_row = r;
                    }
                    values[k] = grad[c];
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..n {
                    for j in 0..=i {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                let lambda = lambda.unwrap_or(&[]);
                // Dense Lagrangian Hessian by central differences of the
                // (exact) Lagrangian gradient.
                let mut hmat = vec![0.0; n * n];
                let mut xp = x.to_vec();
                for kcol in 0..n {
                    let hk = 1e-6 * (1.0 + x[kcol].abs());
                    xp[kcol] = x[kcol] + hk;
                    let gp = self.lagrangian_grad(&xp, obj_factor, lambda);
                    xp[kcol] = x[kcol] - hk;
                    let gm = self.lagrangian_grad(&xp, obj_factor, lambda);
                    xp[kcol] = x[kcol];
                    for i in 0..n {
                        hmat[i * n + kcol] = (gp[i] - gm[i]) / (2.0 * hk);
                    }
                }
                let mut k = 0;
                for i in 0..n {
                    for j in 0..=i {
                        // Symmetrize the finite-difference estimate.
                        values[k] = 0.5 * (hmat[i * n + j] + hmat[j * n + i]);
                        k += 1;
                    }
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        if sol.x.len() == self.n {
            self.final_x = Some(sol.x.to_vec());
        }
    }
}

/// Local-solve `prob` over the box `[lo, hi]` from start `x0`, capped at
/// `max_iter` interior-point iterations. Returns the solver's final point (its
/// feasibility/quality is judged by the caller). `None` if the solve produced
/// no usable point.
pub(crate) fn local_solve(
    prob: &GlobalProblem,
    lo: &[f64],
    hi: &[f64],
    x0: &[f64],
    max_iter: usize,
) -> Option<Vec<f64>> {
    let mut jac: Vec<(usize, usize)> = Vec::new();
    for (j, con) in prob.constraints.iter().enumerate() {
        for v in referenced_vars(&con.tape) {
            jac.push((j, v));
        }
    }
    let tnlp = Rc::new(RefCell::new(TapeTnlp {
        n: prob.n_vars,
        lo: lo.to_vec(),
        hi: hi.to_vec(),
        x0: x0.to_vec(),
        objective: prob.objective.clone(),
        con_tapes: prob.constraints.iter().map(|c| c.tape.clone()).collect(),
        con_lo: prob.constraints.iter().map(|c| c.lo).collect(),
        con_hi: prob.constraints.iter().map(|c| c.hi).collect(),
        jac,
        final_x: None,
    }));

    let mut app = IpoptApplication::new();
    let _ = app
        .options_mut()
        .set_integer_value("max_iter", max_iter as Index, true, true);
    let _ = app
        .options_mut()
        .set_integer_value("print_level", 0, true, true);
    if app.initialize().is_err() {
        return None;
    }
    let dynt: Rc<RefCell<dyn TNLP>> = tnlp.clone();
    let _ = app.optimize_tnlp(dynt);
    let result = tnlp.borrow().final_x.clone();
    result
}
