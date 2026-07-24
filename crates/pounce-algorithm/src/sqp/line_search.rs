//! l1-merit backtracking line search for SQP. The classic
//! Han-Powell scheme:
//!
//! ```text
//!     φ(x; ν) = f(x) + ν · violation(x)
//!     violation(x) = Σ_i max(bl_i − c_i, 0) + max(c_i − bu_i, 0)
//!                  + Σ_j max(xl_j − x_j, 0) + max(x_j − xu_j, 0)
//! ```
//!
//! Step `p` is a descent direction of `φ(·; ν)` whenever
//! `ν ≥ ‖λ_g‖_∞` (Nocedal-Wright §18.4). We adapt `ν` at every
//! iteration as `ν ← max(ν, ‖λ_g_qp‖_∞ + buffer)` so the QP-
//! derived multipliers are always dominated.
//!
//! Backtracking is plain Armijo:
//!
//! ```text
//!     φ(x + αp) ≤ φ(x) + η · α · D_p φ(x; ν)
//! ```
//!
//! with predicted derivative
//!
//! ```text
//!     D_p φ ≈ ∇f(x)ᵀ p − ν · violation(x)
//! ```
//!
//! (correct under the standard assumption that the QP step
//! reduces the linearized constraint violation to zero).
//!
//! Phase 5b commit 5 deliverable. The filter alternative
//! (Fletcher-Leyffer 2002) is opt-in via
//! `SqpGlobalization::Filter` and lands as a follow-up.

use crate::sqp::options::SqpOptions;
use crate::sqp::problem::SqpProblemSpec;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};

/// l1 constraint + bound violation `‖max(bl − c, 0) + max(c − bu, 0)‖_1`
/// (plus the same for variable bounds). Infinite bounds are treated
/// as never violated.
pub fn l1_violation(
    x: &[Number],
    c_vals: &[Number],
    bl: &[Number],
    bu: &[Number],
    xl: &[Number],
    xu: &[Number],
) -> Number {
    let mut v = 0.0_f64;
    for (i, &ci) in c_vals.iter().enumerate() {
        if bl[i] > NLP_LOWER_BOUND_INF {
            v += (bl[i] - ci).max(0.0);
        }
        if bu[i] < NLP_UPPER_BOUND_INF {
            v += (ci - bu[i]).max(0.0);
        }
    }
    for (j, &xj) in x.iter().enumerate() {
        if xl[j] > NLP_LOWER_BOUND_INF {
            v += (xl[j] - xj).max(0.0);
        }
        if xu[j] < NLP_UPPER_BOUND_INF {
            v += (xj - xu[j]).max(0.0);
        }
    }
    v
}

pub struct LineSearchResult {
    pub alpha: Number,
    pub nu: Number,
    pub x_new: Vec<Number>,
    pub f_new: Number,
    pub c_new: Vec<Number>,
    pub success: bool,
    /// Present only when the accepted step was a second-order
    /// correction. Carries the SOC subproblem's own multipliers
    /// `(λ_g, λ_x)`, which the driver must adopt **verbatim**
    /// instead of interpolating the original QP's multipliers: the
    /// step actually taken is the SOC step, so a consistent
    /// `(step, multipliers)` pair is required for the quasi-Newton
    /// Hessian update to stay well-conditioned.
    pub soc_duals: Option<(Vec<Number>, Vec<Number>)>,
}

/// A second-order correction returned by a [`SocProvider`]: the
/// corrected full step together with the correction subproblem's
/// own multipliers, so the driver can keep step and duals
/// consistent.
pub struct SocStep {
    pub p: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
}

/// Second-order-correction (SOC) provider — the Maratos remedy
/// (Nocedal-Wright §18.11; Fletcher-Leyffer 2002). Given the
/// constraint values `c(x + p)` at the rejected **full** step, it
/// returns a corrected full step `p_soc` (recomputed against the
/// constraint curvature at the trial point), or `None` if the
/// correction subproblem could not be solved.
///
/// The line searches call it at most once, on the first (α = 1)
/// trial, and only when that step *increases* the constraint
/// violation — the signature of the Maratos effect, where a good
/// Newton step is rejected because the linearized constraints
/// under-predict the true (curved) violation. The provider itself
/// is built by the SQP driver ([`crate::sqp::sqp_alg`]), which owns
/// the QP solver and the linearization data.
pub type SocProvider<'a> = &'a mut (dyn FnMut(&[Number]) -> Option<SocStep> + 'a);

/// A second-order-correction step is accepted only if it reduces the
/// constraint violation to at most this fraction of the uncorrected
/// full-step violation (Wächter-Biegler 2006 §3.3, `κ_soc`). Keeps
/// the SOC strictly a feasibility-improving correction, so a step
/// that merely lowers the merit/objective without fixing the
/// linearization error is rejected in favor of ordinary
/// backtracking.
pub(crate) const KAPPA_SOC: Number = 0.99;

/// A second-order correction is a *small* perturbation of the QP
/// step — the min-norm correction has `‖p̂‖ = O(‖p‖²)`, so near the
/// solution `‖p_soc‖ ≈ ‖p‖`. We therefore reject any "correction"
/// that grows the step beyond this multiple of `‖p‖_∞`: far from the
/// solution, a quasi-Newton Hessian can make the re-solved SOC
/// subproblem return a much longer step that overshoots and
/// destabilizes the iteration, which is not what a correction should
/// do. Bounding the growth keeps the Maratos remedy local without a
/// separate trust region.
pub(crate) const SOC_MAX_STEP_GROWTH: Number = 2.0;

/// `‖v‖_∞`.
pub(crate) fn inf_norm(v: &[Number]) -> Number {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Adapt `ν` against the QP multiplier magnitude and run Armijo
/// backtracking on the l1 merit function. Returns the accepted
/// step length, the updated ν, and the resulting trial state
/// (`x_new`, `f_new`, `c_new`) so the caller doesn't have to
/// re-evaluate.
#[allow(clippy::too_many_arguments)]
pub fn l1_merit_line_search<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    p: &[Number],
    qp_lambda_g: &[Number],
    grad_f: &[Number],
    f_curr: Number,
    c_curr: &[Number],
    bl: &[Number],
    bu: &[Number],
    xl: &[Number],
    xu: &[Number],
    current_nu: Number,
    opts: &SqpOptions,
    mut soc: Option<SocProvider<'_>>,
) -> LineSearchResult {
    // ν adaptation (Han-Powell): dominate the QP multipliers by
    // an additive safety margin, then clamp at l1_penalty_max so
    // a pathological |λ_qp| spike doesn't blow the merit into a
    // regime where Armijo always fails. Nocedal-Wright §18.4
    // recommends `ν ≥ ‖λ‖_∞`; we use `+ l1_penalty_safety` to
    // give the test a comfortable inequality.
    let lambda_inf = qp_lambda_g.iter().map(|l| l.abs()).fold(0.0_f64, f64::max);
    let nu = current_nu
        .max(lambda_inf + opts.l1_penalty_safety)
        .min(opts.l1_penalty_max);

    let viol_curr = l1_violation(x, c_curr, bl, bu, xl, xu);
    let phi_curr = f_curr + nu * viol_curr;

    let grad_p: Number = grad_f.iter().zip(p.iter()).map(|(g, pi)| g * pi).sum();
    // Predicted decrease: linear-objective contribution minus
    // the violation we expect the QP to eliminate.
    let predicted = grad_p - nu * viol_curr;
    let eta = 1e-4_f64;

    let mut alpha = 1.0_f64;
    let mut x_trial = vec![0.0; x.len()];
    let mut last_f = f_curr;
    let mut last_c = c_curr.to_vec();
    let mut first_trial = true;
    while alpha > opts.bt_min_alpha {
        for (xt, (&xi, &pi)) in x_trial.iter_mut().zip(x.iter().zip(p.iter())) {
            *xt = xi + alpha * pi;
        }
        let f_trial = nlp.eval_f(&x_trial);
        let c_trial = nlp.eval_c(&x_trial);
        let viol_trial = l1_violation(&x_trial, &c_trial, bl, bu, xl, xu);
        let phi_trial = f_trial + nu * viol_trial;
        last_f = f_trial;
        last_c.clone_from(&c_trial);

        let target = phi_curr + eta * alpha * predicted;
        // Standard Armijo sufficient-decrease (Nocedal-Wright
        // §3.1). The earlier `|| phi_trial < phi_curr` fallback
        // (PR #50 review C3) accepted *any* descent and
        // effectively bypassed the inequality on nonconvex
        // problems where `predicted ≥ 0` makes the Armijo target
        // monotone-non-decreasing. We now gate the fallback on
        // `predicted >= 0` only — i.e. fall back to "any merit
        // decrease wins" only when the predicted derivative is
        // not a descent direction, which is the case the original
        // fallback was intended to cover (cf. Wächter-Biegler 2006
        // §3.3 backtracking rule).
        let armijo_ok = if predicted < 0.0 {
            phi_trial <= target
        } else {
            phi_trial < phi_curr
        };
        #[cfg(test)]
        if opts.print_level >= 2 {
            tracing::debug!(target: "pounce::sqp",
                "         ls trial α={alpha:.3e} phi_t={phi_trial:.4e} \
                 phi_c={phi_curr:.4e} target={target:.4e} pred={predicted:.3e} \
                 grad_p={grad_p:.3e} viol_c={viol_curr:.3e} viol_t={viol_trial:.3e} \
                 ok={armijo_ok}"
            );
        }
        if armijo_ok {
            return LineSearchResult {
                alpha,
                nu,
                x_new: x_trial,
                f_new: f_trial,
                c_new: c_trial,
                success: true,
                soc_duals: None,
            };
        }

        // Second-order correction (Maratos remedy). Attempted once,
        // on the full step (α = 1), and only when that step made the
        // constraint violation worse — otherwise a smaller α already
        // makes ordinary progress and no correction is needed. The
        // corrected full step `x + p_soc` is tested against the same
        // Armijo condition; on rejection we discard it and fall back
        // to plain backtracking on the original direction `p`.
        if first_trial {
            first_trial = false;
            if viol_trial > viol_curr {
                if let Some(soc_fn) = soc.as_deref_mut() {
                    if let Some(step) = soc_fn(&c_trial) {
                        // Reject an overshooting "correction" (see
                        // [`SOC_MAX_STEP_GROWTH`]) before spending an
                        // evaluation on it.
                        if inf_norm(&step.p) <= SOC_MAX_STEP_GROWTH * inf_norm(p) {
                            let mut x_soc = vec![0.0; x.len()];
                            for (xt, (&xi, &pi)) in
                                x_soc.iter_mut().zip(x.iter().zip(step.p.iter()))
                            {
                                *xt = xi + pi;
                            }
                            let f_soc = nlp.eval_f(&x_soc);
                            let c_soc = nlp.eval_c(&x_soc);
                            let viol_soc = l1_violation(&x_soc, &c_soc, bl, bu, xl, xu);
                            let phi_soc = f_soc + nu * viol_soc;
                            // Same Armijo gate as above, at α = 1.
                            let armijo_soc = if predicted < 0.0 {
                                phi_soc <= phi_curr + eta * predicted
                            } else {
                                phi_soc < phi_curr
                            };
                            // Feasibility safeguard (Wächter-Biegler
                            // 2006 §3.3): only take the correction if
                            // it genuinely reduces the constraint
                            // violation relative to the uncorrected
                            // full step — otherwise it is not a
                            // second-order *correction* and we fall
                            // back to ordinary backtracking on `p`.
                            let soc_ok = armijo_soc && viol_soc <= KAPPA_SOC * viol_trial;
                            if soc_ok {
                                return LineSearchResult {
                                    alpha: 1.0,
                                    nu,
                                    x_new: x_soc,
                                    f_new: f_soc,
                                    c_new: c_soc,
                                    success: true,
                                    soc_duals: Some((step.lambda_g, step.lambda_x)),
                                };
                            }
                        }
                    }
                }
            }
        }

        alpha *= opts.bt_reduction;
    }

    LineSearchResult {
        alpha,
        nu,
        x_new: x_trial,
        f_new: last_f,
        c_new: last_c,
        success: false,
        soc_duals: None,
    }
}
