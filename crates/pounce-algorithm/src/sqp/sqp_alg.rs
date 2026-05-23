//! `SqpAlgorithm` — active-set SQP outer loop. Consumes an
//! `SqpProblemSpec` for evaluation; delegates the QP subproblem
//! solve to `pounce_qp::ParametricActiveSetSolver`. Globalization
//! is full-step in commit 3; the filter line search lands in
//! commit 5.
//!
//! Outer loop (Nocedal-Wright §18 standard SQP):
//! 1. Evaluate `f, ∇f, c, ∇c, ∇²L` at `x_k`.
//! 2. Build the QP via `SqpQpData::build`.
//! 3. Solve the QP via `pounce-qp` (warm-started by the previous
//!    `WorkingSet` when available).
//! 4. KKT-error check on `x_k` (before stepping) — if all
//!    component tolerances are met, declare optimal.
//! 5. Take the QP step `p`; promote `(x_k + p, λ_g_qp, λ_x_qp)`
//!    to the next iterate.
//! 6. Carry the QP's `WorkingSet` forward for warm-start.

use crate::sqp::iterates::SqpIterates;
use crate::sqp::options::SqpOptions;
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::SqpQpData;
use crate::sqp::result::{SqpError, SqpResult, SqpStatus};
use pounce_common::types::{Number, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpSolver, QpStatus, QpWarmStart,
};

/// SQP-side algorithm driver.
pub struct SqpAlgorithm {
    qp_solver: ParametricActiveSetSolver,
    qp_opts: QpOptions,
    opts: SqpOptions,
    iterates: Option<SqpIterates>,
}

impl SqpAlgorithm {
    pub fn new(qp_solver: ParametricActiveSetSolver, opts: SqpOptions) -> Self {
        Self {
            qp_solver,
            qp_opts: QpOptions::default(),
            opts,
            iterates: None,
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

    /// Run the SQP loop to convergence (or `max_iter`).
    pub fn optimize<N: SqpProblemSpec>(&mut self, nlp: &mut N) -> Result<SqpResult, SqpError> {
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

        let mut iter = SqpIterates::cold(n, m);
        let x_init = nlp.x_init();
        if x_init.len() != n {
            return Err(SqpError::DimensionMismatch(format!(
                "x_init length must be n = {n}"
            )));
        }
        iter.x = x_init;

        let mut n_qp_solves: u32 = 0;
        let mut final_stationarity = 0.0;
        let mut final_constr_viol = 0.0;

        for outer in 0..self.opts.max_iter {
            let grad_f = nlp.eval_grad_f(&iter.x);
            let c_vals = nlp.eval_c(&iter.x);
            let jac_c = nlp.eval_jac_c(&iter.x);
            let hess_lag = nlp.eval_hess_lag(&iter.x, &iter.lambda_g);

            // KKT check uses the current iterate's evaluations.
            let kkt = check_kkt(
                n, m, &iter, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, &jac_c,
            );
            final_stationarity = kkt.stationarity;
            final_constr_viol = kkt.constr_viol;

            if kkt.stationarity <= self.opts.dual_inf_tol
                && kkt.constr_viol <= self.opts.constr_viol_tol
            {
                let obj = nlp.eval_f(&iter.x);
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj,
                    status: SqpStatus::Optimal,
                    n_iter: outer,
                    n_qp_solves,
                    final_stationarity,
                    final_constr_viol,
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

            // Warm-start from the previous working set when the
            // dimensions match — they do unless the user changed
            // the NLP between solves (which the c3 loop doesn't
            // support).
            let ws_opt: Option<QpWarmStart> = iter.working.as_ref().map(|w| QpWarmStart {
                x: vec![0.0; qp.n],
                lambda_g: iter.lambda_g.clone(),
                lambda_x: iter.lambda_x.clone(),
                working: w.clone(),
            });
            let sol = self.qp_solver.solve(&qp, ws_opt.as_ref(), &self.qp_opts)?;
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
                    });
                }
                other => {
                    return Err(SqpError::QpFailure(
                        pounce_qp::QpError::LinearSolverFailure(format!(
                            "QP subproblem returned status {other}"
                        )),
                    ));
                }
            }

            // Take the full QP step (Phase 5b commit 3 has no
            // line search). The QP primal `sol.x` IS the step
            // direction `p`; the QP multipliers become the new
            // SQP multipliers (per Nocedal-Wright §18.4 — the
            // QP duals are the linearized NLP duals).
            for (xi, &pi) in iter.x.iter_mut().zip(sol.x.iter()) {
                *xi += pi;
            }
            iter.lambda_g = sol.lambda_g;
            iter.lambda_x = sol.lambda_x;
            iter.working = Some(sol.working);
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

    // Stationarity: ∇f + Jᵀ λ_g + λ_x. (The pounce-qp convention
    // is Hx + Aᵀλ = −g, so the sign convention here matches.)
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
    // Add λ_x
    for (s, &lx) in stat.iter_mut().zip(iter.lambda_x.iter()) {
        *s += lx;
    }
    let stat_max = stat.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

    KktError {
        stationarity: stat_max,
        constr_viol: viol,
    }
}
