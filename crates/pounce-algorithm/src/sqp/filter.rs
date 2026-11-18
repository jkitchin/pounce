//! Filter-based globalization for SQP (Fletcher-Leyffer 2002,
//! *Math. Prog.* **91**: "Nonlinear programming without a penalty
//! function"). Sibling to [`crate::sqp::line_search`]; same
//! backtracking shell, different acceptance criterion.
//!
//! The filter is a set of `(θ, φ)` pairs from past iterates,
//! where
//!
//! ```text
//!     θ = l1 constraint + bound violation
//!     φ = f(x)                       (the original objective)
//! ```
//!
//! Trial `(θ_t, φ_t)` is *acceptable to the filter* iff it is
//! not dominated by any filter entry — for every entry
//! `(θ_e, φ_e)`:
//!
//! ```text
//!     θ_t ≤ (1 − γ_θ) θ_e   OR   φ_t ≤ φ_e − γ_φ · θ_e
//! ```
//!
//! Additionally a trial must demonstrate **sufficient progress**
//! against the current iterate (one of θ or φ decreases by a
//! margin), so the filter doesn't collect arbitrarily-close
//! points.
//!
//! On acceptance we add `(θ_curr · (1 − γ_θ), φ_curr − γ_φ · θ_curr)`
//! to the filter (the standard margin convention), removing any
//! filter entries that the new pair dominates.
//!
//! Near feasibility (`θ ≤ θ_min`) a descent step that satisfies the
//! Wächter-Biegler switching condition is an **f-type** step instead:
//! it must reduce `φ` by an Armijo fraction of `∇fᵀp`, and it does not
//! augment the filter. That switching rule was missing until gh#1173,
//! and without it a feasible iterate (`θ = 0`) accepted any feasible
//! trial however far uphill it went. See [`filter_line_search`].
//!
//! ## First-step caveat (PR #50 review C4)
//!
//! At iteration 0 the filter is empty: no filter entry can
//! dominate a trial point, so the dominance test passes
//! unconditionally. The full step is therefore accepted on the
//! first iteration regardless of how badly the trial point
//! increases the constraint violation, provided the
//! sufficient-progress test against the *current* iterate also
//! passes. In practice this is harmless because (a) `θ_curr`
//! starts large at a typical cold-start point so the sufficient-
//! progress test is meaningful from the first step, and (b) the
//! QP step is locally feasible by construction so the first
//! trial is usually a good step. Callers that demand a hard
//! first-step constraint-violation safeguard should use
//! `SqpGlobalization::L1Elastic` instead (Han-Powell ν dominates
//! `|λ_qp|_∞` from iter 0 onward).

use crate::sqp::line_search::{
    KAPPA_SOC, LineSearchResult, SOC_MAX_STEP_GROWTH, SocProvider, inf_norm, l1_violation,
};
use crate::sqp::options::SqpOptions;
use crate::sqp::problem::SqpProblemSpec;
use pounce_common::Number;

/// Filter `F` — a set of `(θ, φ)` pairs.
#[derive(Debug, Clone, Default)]
pub struct SqpFilter {
    entries: Vec<(Number, Number)>,
    pub gamma_theta: Number,
    pub gamma_phi: Number,
    /// `θ_min` of the switching condition (see [`filter_line_search`]):
    /// `THETA_MIN_FACT · max(1, θ_0)`, fixed from the violation of the first
    /// iterate this filter sees. `None` until then.
    theta_min: Option<Number>,
}

impl SqpFilter {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            gamma_theta: 1e-5,
            gamma_phi: 1e-5,
            theta_min: None,
        }
    }

    /// Is `(θ, φ)` not dominated by any filter entry?
    pub fn accepts(&self, theta: Number, phi: Number) -> bool {
        for &(t_e, p_e) in &self.entries {
            // Acceptance: at least one of the two conditions
            // must hold for EACH entry.
            let ok = theta <= (1.0 - self.gamma_theta) * t_e || phi <= p_e - self.gamma_phi * t_e;
            if !ok {
                return false;
            }
        }
        true
    }

    /// Add the current iterate to the filter (with margin), and
    /// purge any entries dominated by the new pair.
    pub fn add(&mut self, theta_curr: Number, phi_curr: Number) {
        let new_theta = theta_curr * (1.0 - self.gamma_theta);
        let new_phi = phi_curr - self.gamma_phi * theta_curr;
        self.entries
            .retain(|&(t_e, p_e)| !(new_theta <= t_e && new_phi <= p_e));
        self.entries.push((new_theta, new_phi));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Switching-condition constants (Wächter-Biegler 2006 §2.3, Ipopt's
/// defaults): an iteration is an *f-type* step — judged by Armijo decrease
/// of the objective rather than by the filter — when its search direction
/// descends and `α·(−∇fᵀp)^S_PHI > DELTA·θ^S_THETA` at a violation `θ`
/// no larger than `θ_min`.
const SWITCH_S_PHI: Number = 2.3;
const SWITCH_S_THETA: Number = 1.1;
const SWITCH_DELTA: Number = 1.0;
/// `θ_min = THETA_MIN_FACT · max(1, θ_0)` (Ipopt's `theta_min_fact`).
const THETA_MIN_FACT: Number = 1e-4;
/// Armijo constant of an f-type step. The ℓ1-merit arm uses the same value.
const ETA_PHI: Number = 1e-4;

/// Filter-line-search backtracking. Same return shape as
/// [`crate::sqp::line_search::l1_merit_line_search`] so the
/// caller dispatches by `SqpGlobalization` choice. The `nu`
/// field of the returned `LineSearchResult` carries no
/// meaning here (the filter has no penalty parameter); we
/// echo back `current_nu` unchanged so the caller's state
/// machine doesn't have to special-case it.
///
/// `grad_p` is `∇f(x)ᵀp`, the predicted first-order change of the objective
/// along `p`. It selects between the two acceptance rules (gh#1173):
///
/// * **f-type** — `grad_p < 0`, `θ_curr ≤ θ_min` and the switching condition
///   holds: the trial must satisfy Armijo,
///   `φ_t ≤ φ_curr + η·α·∇fᵀp`, and the filter is *not* augmented.
/// * **h-type** — otherwise: sufficient progress in `θ` or `φ` and filter
///   acceptance, and the current iterate joins the filter.
///
/// Without the f-type branch a feasible iterate (`θ_curr = 0`) had nothing
/// guarding the objective: `θ_t ≤ (1 − γ_θ)·0` holds for every feasible
/// trial, so "sufficient progress" passed on `θ` alone and any step was
/// taken. On `min sin(3x) + ½x²` from `x = 0.05` the first step went from
/// `f = 0.15` to `f = 9.48`, and the run wandered to `f = 1403`. An uphill
/// step on a model with `m = 0` and no bounds is now rejected, and a
/// θ-progress claim needs `θ_curr > 0` to mean anything.
#[allow(clippy::too_many_arguments)]
pub fn filter_line_search<N: SqpProblemSpec>(
    nlp: &mut N,
    filter: &mut SqpFilter,
    x: &[Number],
    p: &[Number],
    grad_p: Number,
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
    let theta_curr = l1_violation(x, c_curr, bl, bu, xl, xu);
    let phi_curr = f_curr;
    let theta_min = *filter
        .theta_min
        .get_or_insert(THETA_MIN_FACT * theta_curr.max(1.0));
    // Switching condition, less its `α` factor (applied per trial below).
    let descent = grad_p.is_finite() && grad_p < 0.0;
    let switch_lhs = if descent { (-grad_p).powf(SWITCH_S_PHI) } else { 0.0 };
    let switch_rhs = SWITCH_DELTA * theta_curr.powf(SWITCH_S_THETA);
    // Acceptance of a trial `(θ_t, φ_t)` reached at step length `α` (an SOC
    // trial is judged at `α = 1`, as in Wächter-Biegler §3.3). Returns
    // `Some(f_type)` on acceptance, `None` on rejection.
    let judge = |filter: &SqpFilter, theta_t: Number, phi_t: Number, alpha: Number| {
        if descent && theta_curr <= theta_min && alpha * switch_lhs > switch_rhs {
            (phi_t <= phi_curr + ETA_PHI * alpha * grad_p).then_some(true)
        } else {
            // `θ_t ≤ (1 − γ_θ)·θ_curr` is no progress at all when
            // `θ_curr = 0`: every feasible trial satisfies it.
            let theta_progress =
                theta_curr > 0.0 && theta_t <= (1.0 - filter.gamma_theta) * theta_curr;
            let phi_progress = phi_t <= phi_curr - filter.gamma_phi * theta_curr;
            ((theta_progress || phi_progress) && filter.accepts(theta_t, phi_t)).then_some(false)
        }
    };

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
        let theta_trial = l1_violation(&x_trial, &c_trial, bl, bu, xl, xu);
        let phi_trial = f_trial;
        last_f = f_trial;
        last_c.clone_from(&c_trial);

        // A trial point the model cannot be EVALUATED at is REJECTED, before
        // any acceptance test reads it. Inevaluable means `NaN`; see below for
        // why an infinity is deliberately not included.
        //
        // Every test below is a comparison, and every comparison against a
        // `NaN` is false — which rejects it in `phi_progress` but ACCEPTS it
        // in `SqpFilter::accepts`, whose dominance test asks whether some
        // filter entry dominates the trial. No entry can dominate a `NaN`,
        // because `phi >= entry.phi` is false for all of them, so the filter
        // reports "not dominated" and a step to a point where the objective
        // does not exist is taken as progress. It only needs `theta_trial` to
        // stay finite to get there, which it does whenever the constraints
        // evaluate and the objective does not.
        //
        // Observed on `cresc4` with a named `bound_relax_factor`: the very
        // first step, `‖p‖_inf = 2.0e1`, landed outside the model's domain and
        // was accepted at `α = 1` with `f_new = NaN`. gh#876's screen then
        // caught it at the top of the next iteration and reported
        // `Invalid_Number_Detected` — an honest verdict on a solve that had
        // already thrown the answer away, where backtracking would have kept
        // it (that leg solves the model to `0.8718975487` without the
        // widening).
        //
        // Rejecting rather than aborting is what makes this a repair: `alpha`
        // halves and the next trial is closer to an iterate that evaluated
        // fine, which is the ordinary remedy for a step that overshot. Only a
        // model that is non-finite at EVERY trial length exhausts the loop,
        // and that one genuinely has nowhere to go.
        // NaN only, NOT `!is_finite()`: `±inf` is a different condition and
        // must keep its existing path. On a minimization `f_trial = -inf` is
        // the objective DIVERGING, which is a real answer about the model --
        // `unbounded_cubic.nl` is that case, and gh#876's screen at the top of
        // the outer loop is what turns it into a verdict. Refusing those steps
        // would replace a diverging trajectory with a backtrack to
        // `Search_Direction_Becomes_Too_Small` at `f = -3.6e+184`, which
        // reports a stall where the truth is divergence. A `NaN` carries no
        // such information: it says the model could not be evaluated there.
        let trial_is_evaluable =
            !f_trial.is_nan() && !theta_trial.is_nan() && !c_trial.iter().any(|v| v.is_nan());
        // f-type: Armijo on φ. h-type: sufficient progress in θ or φ
        // against the *current* iterate, and not dominated by the filter.
        let verdict = if trial_is_evaluable {
            judge(filter, theta_trial, phi_trial, alpha)
        } else {
            None
        };

        if let Some(f_type) = verdict {
            // Accept. An h-type step adds the current iterate to the
            // filter; an f-type step does not (Wächter-Biegler 2006 §2.3).
            if !f_type {
                filter.add(theta_curr, phi_curr);
            }
            return LineSearchResult {
                alpha,
                nu: current_nu,
                x_new: x_trial,
                f_new: f_trial,
                c_new: c_trial,
                success: true,
                soc_duals: None,
            };
        }

        // Second-order correction (Maratos remedy — see
        // [`SocProvider`]). Attempted once, on the full step
        // (α = 1), and only when that step *increased* the
        // constraint violation: the full Newton step is a good step
        // that the filter rejects because the linearized constraints
        // under-predict the curved violation. The corrected full
        // step `x + p_soc` is subjected to the same progress +
        // filter-acceptance test; on rejection we discard it and
        // resume backtracking on the original direction `p`.
        if first_trial {
            first_trial = false;
            if theta_trial > theta_curr {
                if let Some(soc_fn) = soc.as_deref_mut() {
                    if let Some(step) = soc_fn(&c_trial) {
                        // Reject an overshooting "correction" (see
                        // [`SOC_MAX_STEP_GROWTH`]) before evaluating it.
                        if inf_norm(&step.p) <= SOC_MAX_STEP_GROWTH * inf_norm(p) {
                            let mut x_soc = vec![0.0; x.len()];
                            for (xt, (&xi, &pi)) in
                                x_soc.iter_mut().zip(x.iter().zip(step.p.iter()))
                            {
                                *xt = xi + pi;
                            }
                            let f_soc = nlp.eval_f(&x_soc);
                            let c_soc = nlp.eval_c(&x_soc);
                            let theta_soc = l1_violation(&x_soc, &c_soc, bl, bu, xl, xu);
                            let phi_soc = f_soc;
                            // Feasibility safeguard (Wächter-Biegler
                            // 2006 §3.3): the correction must actually
                            // reduce the violation below the
                            // uncorrected full step's, else fall back
                            // to backtracking.
                            let soc_feasible = theta_soc <= KAPPA_SOC * theta_trial;
                            // Same guard as the main trial above: the
                            // correction is a step too, and `filter.accepts`
                            // cannot reject a `NaN` on its own.
                            let soc_evaluable = !f_soc.is_nan()
                                && !theta_soc.is_nan()
                                && !c_soc.iter().any(|v| v.is_nan());
                            let soc_verdict = if soc_evaluable && soc_feasible {
                                judge(filter, theta_soc, phi_soc, 1.0)
                            } else {
                                None
                            };
                            if let Some(f_type) = soc_verdict {
                                if !f_type {
                                    filter.add(theta_curr, phi_curr);
                                }
                                return LineSearchResult {
                                    alpha: 1.0,
                                    nu: current_nu,
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
        nu: current_nu,
        x_new: x_trial,
        f_new: last_f,
        c_new: last_c,
        success: false,
        soc_duals: None,
    }
}
