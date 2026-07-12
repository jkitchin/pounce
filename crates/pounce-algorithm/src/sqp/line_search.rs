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
            };
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
    }
}
