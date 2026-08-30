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
use pounce_linalg::triplet::GenTMatrix;
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus, WorkingSet,
};

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
            qp_opts: QpOptions::sqp_subproblem(),
            opts,
            iterates: None,
            filter: SqpFilter::new(),
        }
    }

    /// Override the per-call QP-solver options. Defaults are
    /// `pounce_qp::QpOptions::sqp_subproblem()` — `QpOptions::default()`
    /// with second-order certification off, for the reason given there
    /// (which include the
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
        // Inner active-set work: adds + drops summed over every step QP
        // solved in this call. This — not the outer iteration count — is
        // what a warm start is trying to reduce, and on a QP-shaped NLP
        // (one outer iteration by construction) it is the *only* thing
        // that moves. Second-order-correction QPs are not counted: they
        // are solved inside the line search, which does not surface its
        // subproblem stats.
        let mut n_qp_working_set_changes: u32 = 0;
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
        // Previous iterate's `(x, ∇f, ∇c)`, kept so the quasi-Newton
        // curvature pair can difference `∇L` at a single fixed multiplier
        // (see [`curvature_pair`]). Storing `∇L` directly — as the older
        // `DampedBfgs::update(x, ∇L)` form did — bakes in the multiplier
        // that was current at the time, which is precisely the bug.
        let mut prev_point: Option<(Vec<Number>, Vec<Number>, Triplet)> = None;

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

        // Iteration-0 curvature probe (issue #358 tail).
        //
        // `DampedBfgs::update` sizes the identity seed from the first
        // `(s, y)` pair — but that pair only exists at iteration 1, and
        // iteration **0** already solves a QP against `B`. With `B = I`
        // on a problem where `‖∇²L‖ ≫ 1`, that first step overshoots the
        // Newton step by `~cond(∇²L)`; the filter (empty, and `θ` tiny at
        // a near-feasible start) accepts the objective-blowing step, the
        // iterate is flung to `‖x‖ ~ 1e3`, and the huge `(s, y)` pairs
        // that follow drive `B` so ill-conditioned that the QP subproblem
        // itself fails a few iterations later (`QpStepFailed`, surfacing
        // as `Search_Direction_Becomes_Too_Small`).
        //
        // Fix the scale *before* that first QP with one extra gradient
        // evaluation: step a short distance along the steepest-descent
        // direction, difference the gradients, and seed `B = γI` with the
        // resulting Rayleigh quotient `γ = sᵀy / sᵀs`. For a quadratic
        // this is exactly the curvature along the probe direction, and it
        // lies in `[λ_min(∇²L), λ_max(∇²L)]`.
        //
        // The probe differences the *objective* gradient, so it estimates
        // `∇²f` — which equals the Lagrangian Hessian `∇²L = ∇²f + Σλᵢ∇²cᵢ`
        // only when the constraint-curvature term vanishes, i.e. when every
        // constraint is linear (or there are none). That condition is
        // exactly the #358 family, and it is *not* cosmetic: on the Maratos
        // problem (`min 2(x₁²+x₂²−1) − x₁ s.t. x₁²+x₂²=1`) `∇²f = 4I` while
        // `∇²L ≈ I` at the solution, so seeding the objective curvature
        // would over-scale `B` fourfold and cost that solve its convergence.
        //
        // Detect linearity directly rather than trusting a declaration:
        // compare the constraint Jacobian at the probe point with the one
        // at `x`. Identical ⇒ `∇c` is constant ⇒ constraints are linear ⇒
        // the objective Hessian *is* the Lagrangian Hessian and the probe
        // is exact. Otherwise leave the identity seed alone and let the
        // rank-2 updates (which see the true `∇L`) do the work.
        if let Some(b) = bfgs.as_mut() {
            let g0 = nlp.eval_grad_f(&iter.x);
            let g_norm = g0.iter().map(|v| v * v).sum::<Number>().sqrt();
            if g_norm.is_finite() && g_norm > 0.0 {
                // Absolute probe length, scaled by the iterate so the step
                // is meaningful in the problem's own units but always tiny
                // relative to it. `1e-7` keeps the gradient difference well
                // above f64 roundoff without leaving the local model.
                let x_scale = iter.x.iter().map(|v| v.abs()).fold(1.0, f64::max);
                let eps = 1e-7 * x_scale;
                let step: Vec<Number> = g0.iter().map(|gi| -eps * gi / g_norm).collect();
                let x_probe: Vec<Number> =
                    iter.x.iter().zip(step.iter()).map(|(a, d)| a + d).collect();
                // Constant-Jacobian (linear-constraint) check, per above.
                let linear_constraints = m == 0 || {
                    let j0 = nlp.eval_jac_c(&iter.x);
                    let j1 = nlp.eval_jac_c(&x_probe);
                    j0.vals.len() == j1.vals.len()
                        && j0.vals.iter().zip(j1.vals.iter()).all(|(a, c)| {
                            // Relative comparison: a linear constraint
                            // reproduces its Jacobian bit-for-bit, so this
                            // only tolerates evaluation noise.
                            let scale = a.abs().max(c.abs()).max(1.0);
                            (a - c).abs() <= 1e-12 * scale
                        })
                };
                if linear_constraints {
                    let g1 = nlp.eval_grad_f(&x_probe);
                    let s_y: Number = step
                        .iter()
                        .zip(g1.iter().zip(g0.iter()))
                        .map(|(si, (a, bg))| si * (a - bg))
                        .sum();
                    let s_s: Number = step.iter().map(|v| v * v).sum();
                    if s_s > 0.0 && s_y.is_finite() {
                        // A non-positive quotient means the probe direction
                        // has non-positive curvature (nonconvex or
                        // numerically flat); `seed_scale` ignores it, leaving
                        // the identity seed rather than a meaningless or
                        // negative scale.
                        b.seed_scale(s_y / s_s);
                    }
                }
            }
        }

        // Bounded so a curvature escape that keeps returning to the same
        // neighbourhood cannot spin: each one costs a fresh descent, and a
        // handful is far more than any of the measured models needs (one).
        const MAX_SECOND_ORDER_ESCAPES: u32 = 8;
        let mut escapes: u32 = 0;

        for outer in 0..self.opts.max_iter {
            let grad_f = nlp.eval_grad_f(&iter.x);
            let c_vals = c_cached.take().unwrap_or_else(|| nlp.eval_c(&iter.x));
            let f_curr = f_cached.take().unwrap_or_else(|| nlp.eval_f(&iter.x));
            let jac_c = nlp.eval_jac_c(&iter.x);
            let hess_lag = match self.opts.hessian {
                SqpHessianSource::Exact => nlp.eval_hess_lag(&iter.x, &iter.lambda_g),
                SqpHessianSource::DampedBfgs => {
                    let bfgs = bfgs.as_mut().expect("DampedBfgs state initialized above");
                    if let Some((s, y)) =
                        curvature_pair(prev_point.as_ref(), &iter, &grad_f, &jac_c, n)
                    {
                        bfgs.update_sy(&s, &y);
                    }
                    bfgs.as_triplet()
                }
                SqpHessianSource::Lbfgs => {
                    let lb = lbfgs.as_mut().expect("LBfgs state initialized above");
                    if let Some((s, y)) =
                        curvature_pair(prev_point.as_ref(), &iter, &grad_f, &jac_c, n)
                    {
                        lb.update_sy(&s, &y);
                    }
                    lb.as_triplet()
                }
            };

            // Remember this iterate's `(x, ∇f, ∇c)` so the next
            // iteration can build its curvature pair at a fixed
            // multiplier. See `curvature_pair`.
            prev_point = Some((iter.x.clone(), grad_f.clone(), jac_c.clone()));

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

            // `sqp_tol` and `sqp_dual_inf_tol` are both registered and both
            // documented as a max-norm tolerance on the stationarity
            // residual, but only `dual_inf_tol` was ever read — `opts.tol`
            // (default 1e-8) was dead, so the loose 1e-4 governed alone and
            // silently capped attainable accuracy (max x-error `7e-5` on the
            // #358 sweep). Honor both by requiring the tighter, which is the
            // only reading under which neither option is a no-op. Restores
            // `~5e-9` worst-case accuracy for ~10% more iterations. Same
            // registered-but-inert defect family as gh #360.
            let stationarity_tol = self.opts.tol.min(self.opts.dual_inf_tol);
            if kkt.stationarity <= stationarity_tol && kkt.constr_viol <= self.opts.constr_viol_tol
            {
                // First-order KKT is necessary and not sufficient. Before
                // reporting success on an indefinite Lagrangian, look for a
                // feasible direction of negative curvature *at the converged
                // multipliers* and, where one is found, step along it and keep
                // going (gh #856).
                //
                // `nonconvex_qp.nl` under `algorithm=active-set-sqp` is the
                // case: `min x₀x₁ s.t. x₀+x₁ = 2, 0 ≤ x ≤ 4`, on whose
                // feasible segment `f(x₀) = x₀(2−x₀)` is concave, so the
                // `(1, 1)` this converges to at `f = 1` is the constrained
                // **maximum** and the minimum is `0` at either endpoint. It was
                // reported `Solve_Succeeded`. The escape below moves to
                // `(2, 0)`, where a bound joins the active set, the null space
                // closes and `f = 0` certifies.
                //
                // Refuted **by exhibition**, exactly as gh #848 does one layer
                // down: the direction is only acted on after stepping along it
                // and finding the true objective lower, so the curvature search
                // is free to be approximate and a direction it gets wrong
                // costs an evaluation rather than a wrong answer.
                //
                // The Hessian is the **exact** `∇²L`, taken here even when the
                // steps were driven by a quasi-Newton one. That is not an
                // optimization detail, it is what makes the check exist at all
                // under `limited-memory`: a damped-BFGS or L-BFGS matrix is
                // positive definite by construction, so searching it for
                // negative curvature can only ever find none, and gating the
                // check on `SqpHessianSource::Exact` left the L-BFGS leg
                // certifying the same constrained maximum. `eval_hess_lag` is
                // a required method of `SqpProblemSpec`, so it is always
                // callable; this costs one Hessian evaluation per converged
                // solve, and an implementation that has none to give returns
                // an empty triplet, which finds no curvature and changes
                // nothing.
                //
                // The L-BFGS leg is not exotic coverage: the Python frontend
                // and the CasADi plugin both select `limited-memory` on their
                // own whenever no exact Lagrangian Hessian is available.
                let exact_hess = if matches!(self.opts.hessian, SqpHessianSource::Exact) {
                    None
                } else {
                    Some(nlp.eval_hess_lag(&iter.x, &iter.lambda_g))
                };
                if escapes < MAX_SECOND_ORDER_ESCAPES
                    && let Some(d) = negative_curvature_at_kkt_point(
                        n,
                        m,
                        &iter.x,
                        exact_hess.as_ref().unwrap_or(&hess_lag),
                        &jac_c,
                        &c_vals,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        self.opts.constr_viol_tol,
                    )
                    && let Some(next) = exhibit_better_point(
                        nlp,
                        &iter.x,
                        &d,
                        f_curr,
                        &xl,
                        &xu,
                        &bl_c,
                        &bu_c,
                        self.opts.constr_viol_tol,
                    )
                {
                    tracing::debug!(target: "pounce::sqp",
                        "first-order KKT point refuted at second order; stepping \
                         along negative curvature to a strictly better feasible \
                         point (gh #856)");
                    escapes += 1;
                    iter.x = next;
                    iter.working = None;
                    f_cached = None;
                    c_cached = None;
                    prev_point = None;
                    continue;
                }
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: SqpStatus::Optimal,
                    n_iter: outer,
                    n_qp_solves,
                    n_qp_working_set_changes,
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

            // Scale-relative inner-QP tolerances (issue #358 tail).
            //
            // `QpOptions::{feas_tol, opt_tol}` are **absolute** (1e-9 each).
            // That is a sane default for a standalone `solve_qp` on
            // well-scaled data, but this QP is an *inner* subproblem whose
            // data inherits the NLP's scale: with `‖∇f‖ ~ 1e3` and
            // `‖B‖ ~ 1e3`, an absolute 1e-9 is ~1e-12 *relative* — at the
            // f64 noise floor. The active-set solver then cannot certify
            // its own optimality, burns its whole iteration budget, and
            // returns `MaxIter`; the driver reports `QpStepFailed`, which
            // surfaces to the user as `Search_Direction_Becomes_Too_Small`
            // on a QP that is trivially solvable. This is what stalled the
            // ill-conditioned tail of #358 even once the Hessian scale was
            // fixed by the probe above.
            //
            // Scale both tolerances by the QP data magnitude, so the inner
            // solve is asked for a *relative* accuracy it can actually
            // reach. Nothing is lost in the answer: the SQP outer loop
            // still gates optimality on the true, unscaled NLP KKT
            // residuals (`dual_inf_tol` / `constr_viol_tol`) at the top of
            // each iteration, so a sloppier inner step can only cost an
            // extra outer iteration — never a false `Optimal`. Measured on
            // a 500-instance convex-QP sweep this converts 34 failures into
            // successes with a *bit-for-bit identical* error distribution
            // (median 6e-11, max true constraint violation 4e-11).
            //
            // The `SCALE_MAX` clamp bounds the relaxation on pathological
            // data (a quasi-Newton `B` that has blown up); it does not bind
            // on any problem in the sweep.
            const SCALE_MAX: Number = 1e6;
            let g_inf = grad_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let b_inf = qp_data
                .h
                .values()
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            let base = self.qp_opts.clone();
            let scale = g_inf.max(b_inf).clamp(1.0, SCALE_MAX);
            if scale > 1.0 {
                self.qp_opts.opt_tol = base.opt_tol * scale;
                self.qp_opts.feas_tol = base.feas_tol * scale;
            }

            // Warm-start from the previous QP's working set when
            // available. Pounce-qp's `solve_with_working_set`
            // internally computes a feasible primal compatible
            // with the supplied set (it satisfies every active
            // row exactly) — necessary because each SQP
            // linearization shifts the QP's constraint RHS by
            // `-c(x_k)`, so the previous QP's *primal* doesn't
            // carry over even when the active set does.
            let warm_started = iter.working.is_some();
            let mut sol = if let Some(prev_w) = iter.working.as_ref() {
                // A warm solve that *errors* falls back to cold rather than
                // aborting the SQP (gh #855). The cold-start fallback below
                // already exists for a warm solve that comes back `MaxIter` /
                // `NumericalError`, on the reasoning that the carried-over
                // working set can be a poor guess; a hard `Err` from the same
                // call is the **stronger** form of that signal and was the one
                // case the fallback could not see, because `?` propagated out
                // of the whole algorithm first.
                //
                // The error that motivates this says so itself: `eigena2`
                // under `algorithm=active-set-sqp` reaches outer iteration 17
                // and the warm solve fails with "pinned KKT constraint block
                // is rank-deficient (inertia shift masked a singular
                // constraint block); prune to a linearly-independent subset".
                // That is a statement about the *pinned set*, which is exactly
                // what a warm start supplies and a cold start rebuilds. The
                // SQP exited `Internal_Error` / `solve_result_num=500` -- "the
                // solver broke, retry" -- on a model whose objective it can
                // report; with the cold re-solve it ends
                // `Maximum_Iterations_Exceeded` at `obj = 82.5177`, against
                // the NLP arm's 82.5 on the same file.
                //
                // The cold error is still propagated: if a clean start fails
                // too, the failure is not about the working set and there is
                // nothing further to try here.
                match self
                    .qp_solver
                    .solve_with_working_set(&qp, prev_w, &self.qp_opts)
                {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::debug!(target: "pounce::sqp",
                            "warm-started step QP failed hard ({e:?}); re-solving \
                             from cold, since the carried working set is what a \
                             cold start rebuilds (gh #855)");
                        n_qp_solves += 1;
                        self.qp_solver.solve(&qp, None, &self.qp_opts)?
                    }
                }
            } else {
                self.qp_solver.solve(&qp, None, &self.qp_opts)?
            };
            n_qp_solves += 1;
            n_qp_working_set_changes += sol.stats.n_working_set_changes;

            // Cold-start fallback: a warm start seeds the QP with the
            // previous iterate's working set, which is usually a big
            // win but can occasionally strand the active-set solver at
            // its iteration limit (or a numerical breakdown) on a QP
            // that is perfectly solvable from a clean start — e.g.
            // when a quasi-Newton Hessian has drifted enough that the
            // carried-over active set is a poor guess. Rather than
            // give up with `QpStepFailed`, re-solve once from cold;
            // this is what rescues the curved-constraint SQP runs of
            // issue #349 that previously reported
            // `Search_Direction_Becomes_Too_Small`.
            if warm_started && matches!(sol.status, QpStatus::MaxIter | QpStatus::NumericalError) {
                let cold = self.qp_solver.solve(&qp, None, &self.qp_opts)?;
                n_qp_solves += 1;
                n_qp_working_set_changes += cold.stats.n_working_set_changes;
                // `Unbounded` is accepted as well as `Optimal`, and it is
                // the point of gh #855. A cold re-solve that comes back
                // `Unbounded` has *found something* — an unblocked direction
                // of negative curvature in the null space of its working set
                // — and taking only `Optimal` threw that away, leaving `sol`
                // on the original `MaxIter`. The unbounded-model fallback
                // below is gated on `sol.status == Unbounded`, so the
                // δ-shifted proximal step written for exactly this situation
                // was unreachable from a retry.
                //
                // Accepting it is safe because the fallback does not trust it
                // either: it re-tests the ray against the true NLP (gh #388)
                // and only takes the proximal branch when the certificate
                // does *not* survive. A spurious `Unbounded` therefore costs
                // one re-test, not a wrong verdict.
                //
                // Nothing else can be accepted here: `Infeasible` from a cold
                // solve would contradict the warm one on the same subproblem
                // without a tie-breaker, and `MaxIter` / `NumericalError` are
                // what we already have.
                //
                // COVERAGE, stated because it matters: no fixture in the CLI
                // corpus reaches this branch. Swept under
                // `algorithm=active-set-sqp`, the retries fire on three
                // fixtures (`cresc4`, `eigena2`, `jit1_boxed`) and return only
                // `MaxIter` or `Optimal`, including under `sqp_qp_max_iter`
                // forced down to 2. gh #855 observed the `Unbounded` return on
                // `eigena2` in a build carrying second-order certification for
                // the step subproblem, which is gh #856's subject and does not
                // exist here yet. It is kept rather than dropped because it is
                // not redundant -- nothing else makes the unbounded-model
                // fallback reachable from a retry -- which is the opposite of
                // the gh #846 case, where a second arm already rejected
                // everything the removed one would have.
                if matches!(cold.status, QpStatus::Optimal | QpStatus::Unbounded) {
                    sol = cold;
                }
            }

            // Quasi-Newton reset fallback (issue #358 tail). If the QP
            // still cannot be solved, the usual culprit is not the
            // linearization but the *approximated* Hessian: a damped-BFGS
            // matrix that has accumulated enough drift (typically after a
            // large early step on an ill-conditioned problem) to make the
            // step subproblem numerically unsolvable. That is recoverable
            // — throwing away the accumulated curvature and retrying from
            // a scaled identity almost always yields a usable step —
            // whereas the alternative is aborting an otherwise healthy
            // solve with `QpStepFailed`, which the user sees as
            // `Search_Direction_Becomes_Too_Small` on a trivially solvable
            // problem. Rebuild the subproblem around the reset Hessian and
            // re-solve once, from cold (the carried working set belongs to
            // the discarded model).
            let mut qp_data = qp_data;
            if matches!(sol.status, QpStatus::MaxIter | QpStatus::NumericalError)
                && let Some(b) = bfgs.as_mut()
            {
                b.reset_to_scale();
                qp_data = SqpQpData::build(
                    &iter.x,
                    &grad_f,
                    &c_vals,
                    &bl_c,
                    &bu_c,
                    &xl,
                    &xu,
                    nlp.eval_jac_c(&iter.x),
                    b.as_triplet(),
                    self.hessian_inertia(),
                );
                let retry = self
                    .qp_solver
                    .solve(&qp_data.as_qp(), None, &self.qp_opts)?;
                n_qp_solves += 1;
                n_qp_working_set_changes += retry.stats.n_working_set_changes;
                // Same as the cold retry above (gh #855): an `Unbounded`
                // verdict is a finding, not a failure, and discarding it hid
                // the proximal fallback from this branch too.
                //
                // The subproblem stays consistent: this branch runs only
                // while `sol.status` is `MaxIter`/`NumericalError`, so it
                // cannot fire after the cold retry has been accepted, and
                // `qp_data` is rebound above — so the fallback below re-solves
                // the *reset-Hessian* subproblem that produced this verdict,
                // not the discarded one.
                if matches!(retry.status, QpStatus::Optimal | QpStatus::Unbounded) {
                    sol = retry;
                }
            }

            // Unbounded-model fallback (gh #423). The step QP being
            // unbounded below is a statement about the *linearization*, so
            // re-test the ray against the true NLP (gh #388) — and when it
            // does not survive, do not stop there. An unbounded model on a
            // bounded NLP is not a dead end; it is the textbook signal that
            // the model needs regularizing (Nocedal-Wright §18.4), and δ
            // from §4.5 inertia control already *is* that regularization.
            // So re-solve declining the certificate: the same subproblem,
            // the same shift, but the unblocked direction takes the
            // δ-shifted proximal step instead of certifying recession.
            //
            // This is not a corner case. A nonconvex NLP with `m = 0` and
            // no finite bounds has *nothing that can ever block* a
            // negative-curvature direction, so every indefinite iterate
            // produces this certificate. gh #419 gave those iterates a real
            // step where a bound exists and left them with none where one
            // does not: a chain of coupled double wells (`n = 12`, `m = 0`)
            // that converged to f = 0.027424 in 24 iterations died at
            // iteration 1 with `QpStepFailed` at f = 26.03. The proximal
            // step is slow — that slowness is what #416 was about — but it
            // is a step, and it is only reached here where the alternative
            // is no step at all.
            let mut ray_certified = false;
            if sol.status == QpStatus::Unbounded {
                ray_certified = sol.unbounded_ray.as_ref().is_some_and(|d| {
                    ray_certifies_unbounded(
                        nlp,
                        &iter.x,
                        d,
                        f_curr,
                        &grad_f,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        self.opts.constr_viol_tol,
                    )
                });
                if !ray_certified {
                    tracing::debug!(target: "pounce::sqp",
                        "unbounded step QP whose recession ray does not survive \
                         re-testing against the NLP — re-solving for the δ-shifted \
                         proximal step (gh #423)");
                    let prox_opts = QpOptions {
                        certify_recession_ray: false,
                        ..self.qp_opts.clone()
                    };
                    let prox = self.qp_solver.solve(&qp_data.as_qp(), None, &prox_opts)?;
                    n_qp_solves += 1;
                    if prox.status == QpStatus::Optimal {
                        sol = prox;
                    }
                }
            }
            self.qp_opts = base;

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
                        n_qp_working_set_changes,
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
                QpStatus::MaxIter | QpStatus::TimeLimit | QpStatus::NumericalError => {
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    // Report *which* of the two it was. Both are honest
                    // non-committal failures making no infeasibility claim
                    // (#282), but only the budget one is actionable by the
                    // user, and merging them hid the dominant Maros-Mészáros
                    // failure mode behind a step-size verdict. See
                    // `SqpStatus::QpIterationLimit`.
                    // `TimeLimit` rides with `MaxIter`: it is the same kind of
                    // outcome (a budget ran out, no claim about the problem),
                    // and it is the closest honest report `SqpStatus` can make
                    // today. Nothing here reaches it yet — the SQP never sets
                    // `QpOptions::time_limit` on a step subproblem, so the
                    // active-set solver has no deadline to cross — but leaving
                    // the arm unhandled would make the first caller that does
                    // set one land in `QpStepFailed`, i.e. "the step broke
                    // down", which would be false.
                    let status = if matches!(sol.status, QpStatus::MaxIter | QpStatus::TimeLimit) {
                        SqpStatus::QpIterationLimit
                    } else {
                        SqpStatus::QpStepFailed
                    };
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status,
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
                // The step QP is unbounded below along a certified
                // recession ray of the LOCAL model (zero curvature,
                // feasible for every step length, strict descent). That
                // is a statement about the linearization, not yet about
                // the NLP — so it was re-tested against the true
                // objective and constraints above. If it survived, the
                // NLP itself is unbounded and we say so with the same
                // `Diverging_Iterates` verdict every other POUNCE path
                // returns. If it did not, the proximal re-solve above
                // already had its chance to produce a step, and reaching
                // here means even that failed: the QP simply could not
                // produce a usable step, which is `QpStepFailed`.
                //
                // Neither outcome is a hard error. This used to return
                // `QpFailure(LinearSolverFailure("QP subproblem returned
                // status unbounded"))`, which surfaced to AMPL / Pyomo /
                // GAMS consumers as `Internal_Error` /
                // `solve_result_num=500` — "the solver broke" — on a
                // model that is merely unbounded (`300`), and named a
                // linear-solver failure when no linear solver had failed
                // (gh #388).
                QpStatus::Unbounded => {
                    let certified = ray_certified;
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status: if certified {
                            SqpStatus::Unbounded
                        } else {
                            SqpStatus::QpStepFailed
                        },
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
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
            //
            // Both are handed a second-order-correction (SOC)
            // provider (the Maratos remedy). When the full step
            // (α = 1) is rejected because it increased the
            // constraint violation, the line search calls this
            // closure with `c(x_k + p)` to obtain a corrected full
            // step. We build that step by re-solving the SAME QP
            // with the general-constraint RHS re-centered on the
            // trial-point constraint values: the original QP models
            // `c(x_k) + A p`, and the SOC replaces `c(x_k)` by
            // `c(x_k + p) − A p`, so the correction subproblem
            // targets the true (curved) violation at the trial
            // point (Nocedal-Wright §18.11). The just-solved working
            // set warm-starts the correction. Only meaningful with
            // general constraints (`m > 0`).
            //
            // Pre-computed `A p` for the RHS re-centering:
            let a_p = if m > 0 {
                mat_vec_gen(&qp_data.a, &sol.x, m)
            } else {
                Vec::new()
            };
            let mut n_soc_solves: u32 = 0;
            // Working set from the SOC subproblem, kept so that a
            // taken SOC step warm-starts the next iteration from the
            // active set that actually describes `x + p_soc` (not the
            // original QP's set, which belongs to the rejected step).
            let mut soc_working: Option<WorkingSet> = None;
            let ls = {
                let qp_solver = &mut self.qp_solver;
                let qp_opts = &self.qp_opts;
                let qp_data_ref = &qp_data;
                let c_curr_ref = &c_vals;
                let a_p_ref = &a_p;
                let sol_working = &sol.working;
                let n_soc = &mut n_soc_solves;
                let soc_working_slot = &mut soc_working;
                let mut soc = |c_trial: &[Number]| -> Option<crate::sqp::line_search::SocStep> {
                    let mm = qp_data_ref.m;
                    // Re-center the general-constraint RHS on the
                    // trial-point violation, preserving ±∞ sentinels.
                    let mut bl_soc = qp_data_ref.bl.clone();
                    let mut bu_soc = qp_data_ref.bu.clone();
                    for i in 0..mm {
                        let delta = c_curr_ref[i] - c_trial[i] + a_p_ref[i];
                        if qp_data_ref.bl[i] > NLP_LOWER_BOUND_INF {
                            bl_soc[i] = qp_data_ref.bl[i] + delta;
                        }
                        if qp_data_ref.bu[i] < NLP_UPPER_BOUND_INF {
                            bu_soc[i] = qp_data_ref.bu[i] + delta;
                        }
                    }
                    let qp_soc = QpProblem {
                        n: qp_data_ref.n,
                        m: qp_data_ref.m,
                        h: &qp_data_ref.h,
                        g: &qp_data_ref.g,
                        a: &qp_data_ref.a,
                        bl: &bl_soc,
                        bu: &bu_soc,
                        xl: &qp_data_ref.xl,
                        xu: &qp_data_ref.xu,
                        hessian_inertia: qp_data_ref.hessian_inertia,
                    };
                    let sol_soc = qp_solver
                        .solve_with_working_set(&qp_soc, sol_working, qp_opts)
                        .ok()?;
                    *n_soc += 1;
                    if sol_soc.status == QpStatus::Optimal {
                        *soc_working_slot = Some(sol_soc.working);
                        Some(crate::sqp::line_search::SocStep {
                            p: sol_soc.x,
                            lambda_g: sol_soc.lambda_g,
                            lambda_x: sol_soc.lambda_x,
                        })
                    } else {
                        None
                    }
                };
                let soc_ref: Option<crate::sqp::line_search::SocProvider<'_>> =
                    if m > 0 { Some(&mut soc) } else { None };
                match self.opts.globalization {
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
                        soc_ref,
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
                        soc_ref,
                    ),
                }
            };
            n_qp_solves += n_soc_solves;
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
                    n_qp_working_set_changes,
                    final_stationarity,
                    final_constr_viol,
                    working_set: Some(sol.working),
                });
            }
            iter.x = ls.x_new;
            match ls.soc_duals {
                Some((soc_lg, soc_lx)) => {
                    // A second-order-correction step was taken (α = 1
                    // on the SOC subproblem). Adopt the SOC
                    // subproblem's own multipliers and working set so
                    // `(step, multipliers, active set)` stay a
                    // consistent triple — required for the quasi-
                    // Newton Hessian update to stay well-conditioned
                    // and for the next QP to warm-start correctly.
                    iter.lambda_g = soc_lg;
                    iter.lambda_x = soc_lx;
                    iter.working = soc_working.take().or(Some(sol.working));
                }
                None => {
                    for (l, &lq) in iter.lambda_g.iter_mut().zip(sol.lambda_g.iter()) {
                        *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
                    }
                    for (l, &lq) in iter.lambda_x.iter_mut().zip(sol.lambda_x.iter()) {
                        *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
                    }
                    iter.working = Some(sol.working);
                }
            }
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
            n_qp_working_set_changes,
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

/// gh #388: does the step QP's certified recession ray certify the **NLP**
/// unbounded below?
///
/// The inner QP hands back a direction `d` that is a recession ray *of the
/// linearization at `x`*: `∇²L d ≈ 0`, `d` feasible for the linearized
/// constraints at every step length, `∇q(x)ᵀd < 0`. On an LP or a QP that
/// linearization is exact and `d` is a recession ray of the original
/// problem; on a general NLP it need not be — the constraints curve back
/// and the objective can turn around. The two cases must not share a
/// status, so we settle it by evaluation rather than by faith: walk the
/// ray and check, at the **true** `f` and `c`, that
///
///  1. every probe point is *feasible* (variable bounds and constraint
///     bounds, the latter with a roundoff allowance that grows with the
///     row scale so a linear row evaluated at `‖x‖ ~ 1e12` is not failed
///     on cancellation noise), and
///  2. the objective keeps falling at **at least half** the initial linear
///     rate `∇f(x)ᵀd` — not merely falling. A ray that decelerates is
///     settling onto a finite optimum, the same distinction the IPM's
///     divergence guard draws (#248/#252/#285).
///
/// Probes span twelve decades of step length, so a "pass" is a family of
/// genuinely feasible points whose objective marches to `−∞` at a linear
/// rate over `1e12`. Anything short of that — one infeasible probe, one
/// decelerating decade, a NaN — returns `false` and the caller reports the
/// non-committal `QpStepFailed` instead. False negatives cost an honest
/// "no step" status; a false positive would tell a modeler their bounded
/// model is unbounded, so the asymmetry is deliberate.
///
/// `dir` need not be normalized (it is rescaled to unit max-norm here, so
/// the probe lengths are in the iterate's own units).
#[allow(clippy::too_many_arguments)]
fn ray_certifies_unbounded<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    dir: &[Number],
    f_x: Number,
    grad_f: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    constr_viol_tol: Number,
) -> bool {
    /// Step lengths along the unit-max-norm ray, spanning twelve decades.
    const PROBES: [Number; 7] = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12];
    /// Roundoff allowance per unit of `row_scale · ‖x‖∞` when checking a
    /// constraint at a far-out probe: comfortably above f64 epsilon
    /// (`2.2e-16`) to absorb accumulation over a row, far below anything
    /// a real violation would produce.
    const ROUNDOFF_REL: Number = 1e-12;

    let n = x.len();
    if dir.len() != n || grad_f.len() != n || !f_x.is_finite() {
        return false;
    }
    let scale = dir.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    let d: Vec<Number> = dir.iter().map(|v| v / scale).collect();

    // Descent of the TRUE objective along the ray. The QP certified this
    // for its own (possibly quasi-Newton) model gradient; re-derive it
    // from `∇f(x)` so the rate we hold the probes to is the real one.
    let slope: Number = grad_f.iter().zip(d.iter()).map(|(g, di)| g * di).sum();
    let g_norm = grad_f.iter().map(|v| v * v).sum::<Number>().sqrt();
    // Numerically meaningful (not roundoff-scale) descent; a NaN slope
    // fails the `is_finite` guard rather than sneaking past the comparison.
    let descent_bar = -1e-9 * g_norm.max(1.0);
    if !slope.is_finite() || slope >= descent_bar {
        return false;
    }

    // Per-row `max_j |∂c_i/∂x_j|`, the scale a linear row's value grows
    // with along the ray — the basis for the roundoff allowance in (1).
    let m = bl_c.len();
    let mut row_scale: Vec<Number> = vec![0.0; m];
    {
        let jac = nlp.eval_jac_c(x);
        for k in 0..jac.vals.len() {
            let i = (jac.irow[k] - 1) as usize;
            row_scale[i] = row_scale[i].max(jac.vals[k].abs());
        }
    }

    for &t in PROBES.iter() {
        let xt: Vec<Number> = x.iter().zip(d.iter()).map(|(xi, di)| xi + t * di).collect();
        if xt.iter().any(|v| !v.is_finite()) {
            return false;
        }

        // (1a) Variable bounds. These are exact linear rows in the probe's
        // own arithmetic, so the tolerance stays tight.
        for i in 0..n {
            let tol = 1e-9 * (1.0 + xt[i].abs());
            if xl[i] > NLP_LOWER_BOUND_INF && xt[i] < xl[i] - tol {
                return false;
            }
            if xu[i] < NLP_UPPER_BOUND_INF && xt[i] > xu[i] + tol {
                return false;
            }
        }

        // (1b) Constraint bounds, at the true (possibly nonlinear) `c`.
        let x_inf = xt.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let c = nlp.eval_c(&xt);
        if c.len() != m {
            return false;
        }
        for i in 0..m {
            if c[i].is_nan() {
                return false;
            }
            let tol =
                constr_viol_tol.max(0.0) * (1.0 + c[i].abs()) + ROUNDOFF_REL * row_scale[i] * x_inf;
            if bl_c[i] > NLP_LOWER_BOUND_INF && c[i] < bl_c[i] - tol {
                return false;
            }
            if bu_c[i] < NLP_UPPER_BOUND_INF && c[i] > bu_c[i] + tol {
                return false;
            }
        }

        // (2) Sustained (non-decelerating) descent. `-inf` passes: an
        // objective that has already overflowed downward is not evidence
        // against unboundedness.
        let f_t = nlp.eval_f(&xt);
        if f_t.is_nan() || f_t > f_x + 0.5 * slope * t {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct KktError {
    pub stationarity: Number,
    pub constr_viol: Number,
}

/// Sparse `A · p` for an `m × n` general-constraint Jacobian stored
/// as a `GenTMatrix` (1-based triplet indices). Used to re-center
/// the second-order-correction QP's RHS on the trial point.
fn mat_vec_gen(a: &GenTMatrix, p: &[Number], m: usize) -> Vec<Number> {
    let mut out = vec![0.0; m];
    let irows = a.irows();
    let jcols = a.jcols();
    let vals = a.values();
    for k in 0..vals.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        out[i] += vals[k] * p[j];
    }
    out
}

/// Build the quasi-Newton curvature pair `(s, y)` for the step from the
/// previous iterate to the current one, differencing `∇L` at a **single,
/// fixed multiplier** (Nocedal-Wright §18.3):
///
/// ```text
///     s = x_k − x_{k−1}
///     y = ∇L(x_k, λ_k) − ∇L(x_{k−1}, λ_k)        ← the SAME λ_k twice
/// ```
///
/// Returns `None` on the first iteration (no previous point yet).
///
/// **Why the fixed multiplier matters (gh #361).** The previous code held
/// `∇L(x_{k−1}, λ_{k−1})` inside the Hessian object and differenced against
/// `∇L(x_k, λ_k)`, giving
///
/// ```text
///     y = (∇f_k − ∇f_{k−1}) + (J_kᵀλ_k − J_{k−1}ᵀλ_{k−1})
/// ```
///
/// For **linear** constraints `J` is constant, so that second group collapses
/// to `Aᵀ(λ_k − λ_{k−1})` — pure *multiplier* difference, carrying no
/// curvature information at all. Since the true `∇²L` equals `∇²f` there, the
/// whole term is spurious, and it feeds a divergent loop: a perturbed `B`
/// yields a worse QP multiplier, which injects a larger error into the next
/// `y`, which corrupts `B` further. On equality-constrained QPs (where `λ` is
/// sign-free and can swing hard) the multiplier was observed oscillating and
/// growing exponentially — `−13, 19, −69, 104, −145, 581, −1320, 3176, …` —
/// while `x` itself sat on the exact optimum. The solve then burned its whole
/// iteration budget and exited `Maximum_Iterations_Exceeded` *at the right
/// answer*, because the stationarity residual is computed from that garbage
/// multiplier.
///
/// Using one multiplier at both points makes the term telescope to
/// `Σλᵏᵢ(∇cᵢ(x_k) − ∇cᵢ(x_{k−1}))`, which is the genuine constraint-curvature
/// contribution: it vanishes identically for linear constraints (as it must)
/// and is retained for nonlinear ones.
fn curvature_pair(
    prev: Option<&(Vec<Number>, Vec<Number>, Triplet)>,
    iter: &SqpIterates,
    grad_f: &[Number],
    jac_c: &Triplet,
    n: usize,
) -> Option<(Vec<Number>, Vec<Number>)> {
    let (prev_x, prev_grad_f, prev_jac) = prev?;
    let s: Vec<Number> = iter
        .x
        .iter()
        .zip(prev_x.iter())
        .map(|(a, b)| a - b)
        .collect();
    // Both evaluated at the *current* multiplier `iter.lambda_g`.
    let lag_curr = compute_grad_lag(grad_f, jac_c, &iter.lambda_g, n);
    let lag_prev = compute_grad_lag(prev_grad_f, prev_jac, &iter.lambda_g, n);
    let y: Vec<Number> = lag_curr
        .iter()
        .zip(lag_prev.iter())
        .map(|(a, b)| a - b)
        .collect();
    Some((s, y))
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
/// Walk `d` from `x` and return a strictly better feasible point, or `None`.
///
/// This is what turns a curvature *direction* into a refutation. gh #848
/// established the shape one layer down: a feasible point with a strictly
/// lower objective is proof needing no theory, so the search that produced
/// `d` may be as approximate as it likes — a direction it gets wrong costs
/// the evaluations below and nothing else.
///
/// The step length is the distance to the first blocking variable bound,
/// halved back until the *nonlinear* constraints are satisfied to
/// `constr_viol_tol`. That backtrack is the difference between this and the
/// QP-level version: `d` lies in the null space of the *linearized* active
/// constraints, which holds them exactly only where they are linear.
/// `nonconvex_qp`'s equality is linear, so the first trial is accepted there;
/// a curved constraint gives up its step rather than trading feasibility for
/// objective, which the SQP's own merit function would then have to undo.
#[allow(clippy::too_many_arguments)]
fn exhibit_better_point<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    d: &[Number],
    f_curr: Number,
    xl: &[Number],
    xu: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    constr_viol_tol: Number,
) -> Option<Vec<Number>> {
    let n = x.len();
    let dn = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !(dn > 0.0) || !dn.is_finite() {
        return None;
    }
    // Both signs: curvature is even, so `-d` descends wherever `d` does, and
    // only one of them may have room before a bound.
    for sign in [1.0_f64, -1.0] {
        let mut alpha = f64::INFINITY;
        for i in 0..n {
            let di = sign * d[i];
            if di > 1e-12 * dn && xu[i] < f64::INFINITY {
                alpha = alpha.min((xu[i] - x[i]) / di);
            }
            if di < -1e-12 * dn && xl[i] > f64::NEG_INFINITY {
                alpha = alpha.min((x[i] - xl[i]) / -di);
            }
        }
        // Unbounded in this direction is not this function's business -- the
        // unbounded-model fallback owns that -- so cap and keep going.
        if !alpha.is_finite() {
            alpha = 1.0 / dn;
        }
        for _ in 0..24 {
            if !(alpha > 0.0) {
                break;
            }
            let trial: Vec<Number> = (0..n)
                .map(|i| (x[i] + sign * alpha * d[i]).clamp(xl[i], xu[i]))
                .collect();
            let viol = nlp
                .eval_c(&trial)
                .iter()
                .enumerate()
                .fold(0.0_f64, |a, (j, &cj)| {
                    a.max((bl_c[j] - cj).max(0.0)).max((cj - bu_c[j]).max(0.0))
                });
            if viol <= constr_viol_tol {
                let f_trial = nlp.eval_f(&trial);
                // Strictly better by more than the objective's own scale can
                // round, so this cannot fire on noise at a genuine optimum.
                if f_trial.is_finite() && f_trial < f_curr - 1e-10 * (1.0 + f_curr.abs()) {
                    return Some(trial);
                }
                break;
            }
            alpha *= 0.5;
        }
    }
    None
}

/// A feasible direction of negative curvature at a *converged* first-order
/// point, or `None` when the point survives the second-order test (gh #856).
///
/// # Why this runs at convergence and not at each step
///
/// gh #848 gave standalone QP solves a second-order screen. Applying the same
/// screen to the SQP's **step** subproblem is wrong, and gh #856 has the
/// counterexample: that QP is a local model built from the *current*
/// multiplier estimates, and its second-order verdict is not the NLP's. At
/// SQP iteration 0 the multipliers are still zero, so the exact Lagrangian
/// Hessian is `∇²f`; started at HS071's own `x*` the step QP's working set
/// leaves a one-dimensional null space on which `dᵀHd = -4.05e-2`, and the
/// point is refuted — correctly for that model, wrongly for the NLP, whose
/// reduced Hessian at `x*` is positive once the multipliers have converged.
///
/// At *convergence* that objection disappears: the multipliers are the
/// converged ones, so `∇²L` is the Hessian the second-order condition is
/// actually about. This is the same distinction gh #856 draws when it says
/// "with the converged multipliers the reduced Hessian is positive" — the
/// check is meaningful exactly where it is run.
///
/// # What it computes
///
/// `Z` is an orthonormal basis for the null space of the active constraint
/// normals — equality rows, active inequality rows and active bounds — taken
/// from the eigenvectors of `BᵀB` whose eigenvalue is negligible against the
/// largest. The reduced Hessian `ZᵀHZ` is then eigen-decomposed, and a
/// sufficiently negative eigenvalue yields `d = Z v`: a direction that holds
/// every active constraint to first order and along which the objective
/// curves down.
///
/// Returns `None` when there is no negative curvature, when the active set
/// leaves no degrees of freedom, or when either eigensolve fails to converge
/// — never a direction it is unsure of, since the caller acts on it.
#[allow(clippy::too_many_arguments)]
fn negative_curvature_at_kkt_point(
    n: usize,
    m: usize,
    x: &[Number],
    hess_lag: &Triplet,
    jac_c: &Triplet,
    c_vals: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    tol: Number,
) -> Option<Vec<Number>> {
    // Dense and `O(n³)`, so it is bounded rather than let loose on a large
    // model. It runs once, at convergence, on the way to reporting success —
    // and skipping it returns exactly today's answer, so the ceiling costs
    // coverage and never correctness. (Contrast gh #849, where the analogous
    // ceiling silently withdrew a *guarantee*.)
    const MAX_N: usize = 512;
    if n == 0 || n > MAX_N {
        return None;
    }

    // The active set. A bound or row counts as active when the iterate sits
    // on it to within the same tolerance the convergence test just used, so
    // this is the set the verdict was issued about.
    let mut rows: Vec<Vec<Number>> = Vec::new();
    for j in 0..m {
        let lo_active = bl_c[j] > f64::NEG_INFINITY && (c_vals[j] - bl_c[j]).abs() <= tol;
        let hi_active = bu_c[j] < f64::INFINITY && (bu_c[j] - c_vals[j]).abs() <= tol;
        if lo_active || hi_active {
            let mut r = vec![0.0; n];
            // `pounce_linalg` triplets are **1-based** (see `triplet.rs`).
            for k in 0..jac_c.vals.len() {
                if jac_c.irow[k] as usize == j + 1 {
                    r[jac_c.jcol[k] as usize - 1] += jac_c.vals[k];
                }
            }
            rows.push(r);
        }
    }
    for i in 0..n {
        let on_lo = xl[i] > f64::NEG_INFINITY && (x[i] - xl[i]).abs() <= tol;
        let on_hi = xu[i] < f64::INFINITY && (xu[i] - x[i]).abs() <= tol;
        if on_lo || on_hi {
            let mut r = vec![0.0; n];
            r[i] = 1.0;
            rows.push(r);
        }
    }

    // Null space of the active normals, from the eigenvectors of `BᵀB`.
    let mut z: Vec<Number> = Vec::new();
    let n_dof;
    if rows.is_empty() {
        n_dof = n;
        z = vec![0.0; n * n];
        for i in 0..n {
            z[i * n + i] = 1.0;
        }
    } else {
        let mut btb = vec![0.0; n * n];
        for r in &rows {
            for a in 0..n {
                if r[a] == 0.0 {
                    continue;
                }
                for c in 0..n {
                    btb[c * n + a] += r[a] * r[c];
                }
            }
        }
        let (mut ev, mut evec) = (vec![0.0; n], vec![0.0; n * n]);
        if !pounce_linalg::symmetric_eigen(&btb, n, &mut ev, &mut evec) {
            return None;
        }
        let lam_max = ev.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let cut = 1e-9 * lam_max.max(1.0);
        for (j, &lam) in ev.iter().enumerate() {
            if lam.abs() <= cut {
                z.extend_from_slice(&evec[j * n..(j + 1) * n]);
            }
        }
        n_dof = z.len() / n;
    }
    if n_dof == 0 {
        return None;
    }

    // `H Z`, then `Zᵀ (H Z)`.
    let mut hz = vec![0.0; n * n_dof];
    for k in 0..n_dof {
        let (zc, out) = (&z[k * n..(k + 1) * n], &mut hz[k * n..(k + 1) * n]);
        // Stored as one triangle, so an off-diagonal entry contributes to
        // both of its rows.
        for e in 0..hess_lag.vals.len() {
            let (r, c, v) = (
                hess_lag.irow[e] as usize - 1,
                hess_lag.jcol[e] as usize - 1,
                hess_lag.vals[e],
            );
            out[r] += v * zc[c];
            if r != c {
                out[c] += v * zc[r];
            }
        }
    }
    let mut rh = vec![0.0; n_dof * n_dof];
    for a in 0..n_dof {
        for b in 0..n_dof {
            rh[b * n_dof + a] = (0..n).map(|i| z[a * n + i] * hz[b * n + i]).sum();
        }
    }

    let (mut ev, mut evec) = (vec![0.0; n_dof], vec![0.0; n_dof * n_dof]);
    if !pounce_linalg::symmetric_eigen(&rh, n_dof, &mut ev, &mut evec) {
        return None;
    }
    // Relative to the Hessian's own scale, so this cannot fire on the
    // rounding noise of a genuinely positive-semidefinite reduced Hessian.
    let h_scale = hess_lag
        .vals
        .iter()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        .max(1.0);
    if ev[0] >= -1e-8 * h_scale {
        return None;
    }
    // `d = Z v` with `v` the eigenvector of the most negative eigenvalue --
    // column 0 of the column-major `evec`, since the eigensolver returns them
    // in ascending order.
    let mut d = vec![0.0; n];
    for (i, di) in d.iter_mut().enumerate() {
        *di = (0..n_dof).map(|k| z[k * n + i] * evec[k]).sum();
    }
    Some(d)
}

pub(crate) fn check_kkt(
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
