//! `SqpAlgorithm` — active-set SQP outer loop. Consumes an
//! `SqpProblemSpec` for evaluation; delegates the QP subproblem
//! solve to `pounce_qp::ParametricActiveSetSolver`.
//!
//! Outer loop (Nocedal-Wright §18 standard SQP):
//! 1. Evaluate `f, ∇f, c, ∇c, ∇²L` at `x_k`.
//! 2. Build the QP via `SqpQpData::build`.
//! 3. Solve the QP via `pounce-qp` (warm-started by the previous
//!    `WorkingSet` when available).
//! 4. KKT-error check on `x_k` (before stepping) — if all
//!    component tolerances are met, declare optimal.
//! 5. Globalization step acceptance via either the Fletcher-
//!    Leyffer 2002 filter (`SqpGlobalization::Filter`, default)
//!    or the Han-Powell l1-merit (`SqpGlobalization::L1Elastic`),
//!    both backtracking on α.
//! 6. Take `α·p`; promote `(x_k + α p, λ_g, λ_x)` to the next
//!    iterate and carry the QP's `WorkingSet` for the next solve.

use crate::sqp::bfgs::DampedBfgs;
use crate::sqp::filter::{SqpFilter, filter_line_search};
use crate::sqp::iterates::SqpIterates;
use crate::sqp::lbfgs::LBfgs;
use crate::sqp::line_search::l1_merit_line_search;
use crate::sqp::options::{SqpGlobalization, SqpHessianSource, SqpOptions};
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::{SqpQpData, Triplet};
use crate::sqp::result::{SqpError, SqpResult, SqpStatus};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_qp::{HessianInertia, ParametricActiveSetSolver, QpOptions, QpSolver, QpStatus};

/// SQP-side algorithm driver.
pub struct SqpAlgorithm {
    qp_solver: ParametricActiveSetSolver,
    qp_opts: QpOptions,
    opts: SqpOptions,
    iterates: Option<SqpIterates>,
    /// Filter for Fletcher-Leyffer globalization; reset at the
    /// top of each `optimize` call. Unused when
    /// `opts.globalization = L1Elastic`.
    filter: SqpFilter,
}

impl SqpAlgorithm {
    pub fn new(qp_solver: ParametricActiveSetSolver, opts: SqpOptions) -> Self {
        Self {
            qp_solver,
            qp_opts: QpOptions::default(),
            opts,
            iterates: None,
            filter: SqpFilter::new(),
        }
    }

    /// Override the per-call QP-solver options. Defaults are the
    /// `pounce_qp::QpOptions::default()` (which include the
    /// `use_schur_updates = false` and `anti_cycling = Expand`
    /// from Phase 5a.2). Callers can pin tighter tolerances or
    /// flip `use_schur_updates = true` for warm-started workloads.
    pub fn with_qp_options(mut self, qp_opts: QpOptions) -> Self {
        self.qp_opts = qp_opts;
        self
    }

    pub fn options(&self) -> &SqpOptions {
        &self.opts
    }

    pub fn iterates(&self) -> Option<&SqpIterates> {
        self.iterates.as_ref()
    }

    /// Run the SQP loop to convergence (or `max_iter`). Cold-starts
    /// the iterate from `nlp.x_init()` and an empty working set.
    pub fn optimize<N: SqpProblemSpec>(&mut self, nlp: &mut N) -> Result<SqpResult, SqpError> {
        self.optimize_with_warm_start(nlp, None)
    }

    /// Warm-start variant. `warm = Some(prev)` seeds the iterate
    /// from `prev.{x, lambda_g, lambda_x, working}` instead of the
    /// NLP's cold defaults. Dimensions are validated against the
    /// problem; any mismatch is fatal. The QP solver consumes
    /// `warm.working` (when present) via `solve_with_working_set`.
    ///
    /// `warm = None` is equivalent to [`Self::optimize`].
    ///
    /// Implements the §6 design-note warm-start contract: the
    /// tuple `(x, λ_g, λ_x, 𝒲)`. The Hessian carry-forward
    /// (damped-BFGS / L-BFGS state) is *not* part of the warm-start
    /// payload — each `optimize` call rebuilds its own Hessian
    /// approximation from scratch.
    pub fn optimize_with_warm_start<N: SqpProblemSpec>(
        &mut self,
        nlp: &mut N,
        warm: Option<SqpIterates>,
    ) -> Result<SqpResult, SqpError> {
        let n = nlp.n();
        let m = nlp.m();
        let (xl, xu) = nlp.variable_bounds();
        let (bl_c, bu_c) = nlp.constraint_bounds();
        if xl.len() != n || xu.len() != n {
            return Err(SqpError::DimensionMismatch(format!(
                "variable_bounds length must be n = {n}"
            )));
        }
        if bl_c.len() != m || bu_c.len() != m {
            return Err(SqpError::DimensionMismatch(format!(
                "constraint_bounds length must be m = {m}"
            )));
        }

        let mut iter = match warm {
            Some(w) => {
                if w.x.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.x length {} must equal n = {n}",
                        w.x.len()
                    )));
                }
                if w.lambda_g.len() != m {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.lambda_g length {} must equal m = {m}",
                        w.lambda_g.len()
                    )));
                }
                if w.lambda_x.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.lambda_x length {} must equal n = {n}",
                        w.lambda_x.len()
                    )));
                }
                if let Some(ws) = w.working.as_ref() {
                    ws.validate_dims(n, m).map_err(SqpError::QpFailure)?;
                }
                w
            }
            None => {
                let mut cold = SqpIterates::cold(n, m);
                let x_init = nlp.x_init();
                if x_init.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "x_init length must be n = {n}"
                    )));
                }
                cold.x = x_init;
                cold
            }
        };

        let mut n_qp_solves: u32 = 0;
        let mut final_stationarity = 0.0;
        let mut final_constr_viol = 0.0;
        // l1-merit penalty parameter ν, adapted across iterations
        // by `l1_merit_line_search`. Initialized from
        // `SqpOptions::l1_penalty`.
        let mut nu = self.opts.l1_penalty;
        // Reset filter state at the top of each optimize call.
        self.filter = SqpFilter::new();
        // Cache the most recent f(x) and c(x) so we don't
        // re-evaluate them after a successful line search (the
        // LS already computed them at the new iterate).
        let mut f_cached: Option<Number> = None;
        let mut c_cached: Option<Vec<Number>> = None;

        // Damped-BFGS state, allocated only if needed. The
        // matrix is updated at the END of each iteration (after
        // we have x_new and the next ∇L), then queried at the
        // TOP of the next iteration to populate the QP Hessian.
        let mut bfgs: Option<DampedBfgs> =
            if matches!(self.opts.hessian, SqpHessianSource::DampedBfgs) {
                Some(DampedBfgs::new(n))
            } else {
                None
            };
        let mut lbfgs: Option<LBfgs> = if matches!(self.opts.hessian, SqpHessianSource::Lbfgs) {
            Some(LBfgs::new(n, self.opts.lbfgs_max_history.max(1) as usize))
        } else {
            None
        };

        for outer in 0..self.opts.max_iter {
            let grad_f = nlp.eval_grad_f(&iter.x);
            let c_vals = c_cached.take().unwrap_or_else(|| nlp.eval_c(&iter.x));
            let f_curr = f_cached.take().unwrap_or_else(|| nlp.eval_f(&iter.x));
            let jac_c = nlp.eval_jac_c(&iter.x);
            let hess_lag = match self.opts.hessian {
                SqpHessianSource::Exact => nlp.eval_hess_lag(&iter.x, &iter.lambda_g),
                SqpHessianSource::DampedBfgs => {
                    let bfgs = bfgs.as_mut().expect("DampedBfgs state initialized above");
                    // Update on the *current* (x, ∇L). The
                    // very first iteration's update is a no-op
                    // (no previous pair); the matrix stays I.
                    let grad_lag = compute_grad_lag(&grad_f, &jac_c, &iter.lambda_g, n);
                    bfgs.update(&iter.x, &grad_lag);
                    bfgs.as_triplet()
                }
                SqpHessianSource::Lbfgs => {
                    let lb = lbfgs.as_mut().expect("LBfgs state initialized above");
                    let grad_lag = compute_grad_lag(&grad_f, &jac_c, &iter.lambda_g, n);
                    lb.update(&iter.x, &grad_lag);
                    lb.as_triplet()
                }
            };

            // KKT check uses the current iterate's evaluations.
            let kkt = check_kkt(
                n, m, &iter, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, &jac_c,
            );
            final_stationarity = kkt.stationarity;
            final_constr_viol = kkt.constr_viol;

            #[cfg(test)]
            if self.opts.print_level >= 1 {
                tracing::debug!(target: "pounce::sqp",
                    "[sqp k={outer:3}] x={:?} f={:.4e} ‖c‖={:.2e} stat={:.2e} ν={:.2e}",
                    iter.x.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
                    f_curr,
                    kkt.constr_viol,
                    kkt.stationarity,
                    nu,
                );
            }

            if kkt.stationarity <= self.opts.dual_inf_tol
                && kkt.constr_viol <= self.opts.constr_viol_tol
            {
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: SqpStatus::Optimal,
                    n_iter: outer,
                    n_qp_solves,
                    final_stationarity,
                    final_constr_viol,
                    working_set: iter.working,
                });
            }

            let qp_data = SqpQpData::build(
                &iter.x,
                &grad_f,
                &c_vals,
                &bl_c,
                &bu_c,
                &xl,
                &xu,
                jac_c,
                hess_lag,
                self.hessian_inertia(),
            );
            let qp = qp_data.as_qp();

            // Warm-start from the previous QP's working set when
            // available. Pounce-qp's `solve_with_working_set`
            // internally computes a feasible primal compatible
            // with the supplied set (it satisfies every active
            // row exactly) — necessary because each SQP
            // linearization shifts the QP's constraint RHS by
            // `-c(x_k)`, so the previous QP's *primal* doesn't
            // carry over even when the active set does.
            let sol = if let Some(prev_w) = iter.working.as_ref() {
                self.qp_solver
                    .solve_with_working_set(&qp, prev_w, &self.qp_opts)?
            } else {
                self.qp_solver.solve(&qp, None, &self.qp_opts)?
            };
            n_qp_solves += 1;

            match sol.status {
                QpStatus::Optimal => {}
                QpStatus::Infeasible => {
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status: SqpStatus::InfeasibleSubproblem,
                        n_iter: outer,
                        n_qp_solves,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
                // The QP subproblem neither solved nor certified
                // infeasibility. `MaxIter` / `NumericalError` mean the
                // active-set QP could not resolve the (typically extremely
                // degenerate) step subproblem — the m/n ≫ 1 collapsed-cone
                // geometry of #282. Terminate the SQP with an HONEST
                // non-committal status rather than a hard error, and — the
                // point of #282 — WITHOUT ever asserting infeasibility on a
                // problem we have not certified infeasible.
                QpStatus::MaxIter | QpStatus::NumericalError => {
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status: SqpStatus::QpStepFailed,
                        n_iter: outer,
                        n_qp_solves,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
                // `Unbounded` on a step QP is a genuine pathology (an
                // indefinite/negative-curvature ray); keep the historical
                // hard-error behavior.
                other => {
                    return Err(SqpError::QpFailure(
                        pounce_qp::QpError::LinearSolverFailure(format!(
                            "QP subproblem returned status {other}"
                        )),
                    ));
                }
            }

            #[cfg(test)]
            if self.opts.print_level >= 1 {
                let p_inf = sol.x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                tracing::debug!(target: "pounce::sqp",
                    "         qp: ‖p‖_inf={:.3e} ‖λ_g_qp‖_inf={:.3e}",
                    p_inf,
                    sol.lambda_g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                );
            }
            // Globalization: l1-merit backtracking (Han-Powell)
            // or filter (Fletcher-Leyffer 2002). The two share
            // the same backtracking shell + acceptance API; the
            // filter keeps state across iterations on
            // `self.filter`.
            let ls = match self.opts.globalization {
                SqpGlobalization::L1Elastic => l1_merit_line_search(
                    nlp,
                    &iter.x,
                    &sol.x,
                    &sol.lambda_g,
                    &grad_f,
                    f_curr,
                    &c_vals,
                    &bl_c,
                    &bu_c,
                    &xl,
                    &xu,
                    nu,
                    &self.opts,
                ),
                SqpGlobalization::Filter => filter_line_search(
                    nlp,
                    &mut self.filter,
                    &iter.x,
                    &sol.x,
                    f_curr,
                    &c_vals,
                    &bl_c,
                    &bu_c,
                    &xl,
                    &xu,
                    nu,
                    &self.opts,
                ),
            };
            #[cfg(test)]
            if self.opts.print_level >= 1 {
                tracing::debug!(target: "pounce::sqp",
                    "         ls: α={:.3e} ν={:.3e} ok={} f_new={:.3e}",
                    ls.alpha, ls.nu, ls.success, ls.f_new
                );
            }
            if !ls.success {
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: SqpStatus::LineSearchFailed,
                    n_iter: outer,
                    n_qp_solves,
                    final_stationarity,
                    final_constr_viol,
                    working_set: Some(sol.working),
                });
            }
            iter.x = ls.x_new;
            for (l, &lq) in iter.lambda_g.iter_mut().zip(sol.lambda_g.iter()) {
                *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
            }
            for (l, &lq) in iter.lambda_x.iter_mut().zip(sol.lambda_x.iter()) {
                *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
            }
            iter.working = Some(sol.working);
            nu = ls.nu;
            f_cached = Some(ls.f_new);
            c_cached = Some(ls.c_new);
        }

        let obj = nlp.eval_f(&iter.x);
        self.iterates = Some(iter.clone());
        Ok(SqpResult {
            x: iter.x,
            lambda_g: iter.lambda_g,
            lambda_x: iter.lambda_x,
            obj,
            status: SqpStatus::MaxIter,
            n_iter: self.opts.max_iter,
            n_qp_solves,
            final_stationarity,
            final_constr_viol,
            working_set: iter.working,
        })
    }

    fn hessian_inertia(&self) -> HessianInertia {
        match self.opts.hessian {
            // Exact ∇²L is indefinite on nonconvex NLPs; let the
            // QP solver's §4.5 inertia control handle it.
            crate::sqp::SqpHessianSource::Exact => HessianInertia::Indefinite,
            // Damped BFGS and L-BFGS are PSD by construction.
            crate::sqp::SqpHessianSource::DampedBfgs => HessianInertia::Psd,
            crate::sqp::SqpHessianSource::Lbfgs => HessianInertia::Psd,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct KktError {
    pub stationarity: Number,
    pub constr_viol: Number,
}

/// Lagrangian gradient `∇L(x, λ_g) = ∇f(x) + J_c(x)ᵀ λ_g` at the
/// current iterate. Used by the damped-BFGS update.
fn compute_grad_lag(
    grad_f: &[Number],
    jac_c: &Triplet,
    lambda_g: &[Number],
    n: usize,
) -> Vec<Number> {
    let mut out = grad_f.to_vec();
    debug_assert_eq!(out.len(), n);
    for k in 0..jac_c.irow.len() {
        let row_i = (jac_c.irow[k] - 1) as usize;
        let col_j = (jac_c.jcol[k] - 1) as usize;
        out[col_j] += jac_c.vals[k] * lambda_g[row_i];
    }
    out
}

fn check_kkt(
    n: usize,
    m: usize,
    iter: &SqpIterates,
    grad_f: &[Number],
    c_vals: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    jac_c: &crate::sqp::qp_assembly::Triplet,
) -> KktError {
    // Constraint violation: max(0, bl - c, c - bu) on every row,
    // plus bound violation on every variable.
    let mut viol = 0.0_f64;
    for i in 0..m {
        let lo = if bl_c[i] > NLP_LOWER_BOUND_INF {
            (bl_c[i] - c_vals[i]).max(0.0)
        } else {
            0.0
        };
        let hi = if bu_c[i] < NLP_UPPER_BOUND_INF {
            (c_vals[i] - bu_c[i]).max(0.0)
        } else {
            0.0
        };
        viol = viol.max(lo).max(hi);
    }
    for i in 0..n {
        let lo = if xl[i] > NLP_LOWER_BOUND_INF {
            (xl[i] - iter.x[i]).max(0.0)
        } else {
            0.0
        };
        let hi = if xu[i] < NLP_UPPER_BOUND_INF {
            (iter.x[i] - xu[i]).max(0.0)
        } else {
            0.0
        };
        viol = viol.max(lo).max(hi);
    }

    // Stationarity: ∇f + Jᵀ λ_g − λ_x. pounce-qp's KKT is
    // `Hx + Aᵀλ_qp + (lower-bound multiplier) e_i − (upper-bound
    // multiplier) e_i = -g`. Since `λ_x = z_l − z_u` packs the
    // bound-multiplier sign, the variable-bound term enters the
    // stationarity check with a negative sign — i.e. at the
    // optimum `∇f + Jᵀ λ_g = λ_x`.
    let mut stat = vec![0.0; n];
    for (s, &g) in stat.iter_mut().zip(grad_f.iter()) {
        *s = g;
    }
    // Add Jᵀ λ_g
    for k in 0..jac_c.irow.len() {
        let i = (jac_c.irow[k] - 1) as usize; // 0-based row in c
        let j = (jac_c.jcol[k] - 1) as usize; // 0-based col in x
        stat[j] += jac_c.vals[k] * iter.lambda_g[i];
    }
    // Subtract λ_x
    for (s, &lx) in stat.iter_mut().zip(iter.lambda_x.iter()) {
        *s -= lx;
    }
    let stat_max = stat.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

    KktError {
        stationarity: stat_max,
        constr_viol: viol,
    }
}
