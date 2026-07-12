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
//! This implementation skips Fletcher-Leyffer's full f-mode /
//! h-mode switching (which requires a `d_φ` predicted-derivative
//! and trigger constants) — it ships the dominance test +
//! sufficient-progress check, which is the SOTA filter's core
//! and matches every commonly-cited reduced filter SQP variant.
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

use crate::sqp::line_search::{LineSearchResult, l1_violation};
use crate::sqp::options::SqpOptions;
use crate::sqp::problem::SqpProblemSpec;
use pounce_common::Number;

/// Filter `F` — a set of `(θ, φ)` pairs.
#[derive(Debug, Clone, Default)]
pub struct SqpFilter {
    entries: Vec<(Number, Number)>,
    pub gamma_theta: Number,
    pub gamma_phi: Number,
}

impl SqpFilter {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            gamma_theta: 1e-5,
            gamma_phi: 1e-5,
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

/// Filter-line-search backtracking. Same return shape as
/// [`crate::sqp::line_search::l1_merit_line_search`] so the
/// caller dispatches by `SqpGlobalization` choice. The `nu`
/// field of the returned `LineSearchResult` carries no
/// meaning here (the filter has no penalty parameter); we
/// echo back `current_nu` unchanged so the caller's state
/// machine doesn't have to special-case it.
#[allow(clippy::too_many_arguments)]
pub fn filter_line_search<N: SqpProblemSpec>(
    nlp: &mut N,
    filter: &mut SqpFilter,
    x: &[Number],
    p: &[Number],
    f_curr: Number,
    c_curr: &[Number],
    bl: &[Number],
    bu: &[Number],
    xl: &[Number],
    xu: &[Number],
    current_nu: Number,
    opts: &SqpOptions,
) -> LineSearchResult {
    let theta_curr = l1_violation(x, c_curr, bl, bu, xl, xu);
    let phi_curr = f_curr;

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
        let theta_trial = l1_violation(&x_trial, &c_trial, bl, bu, xl, xu);
        let phi_trial = f_trial;
        last_f = f_trial;
        last_c.clone_from(&c_trial);

        // Sufficient progress: at least one of θ or φ strictly
        // decreased against the *current* iterate by the
        // configured margin.
        let theta_progress = theta_trial <= (1.0 - filter.gamma_theta) * theta_curr;
        let phi_progress = phi_trial <= phi_curr - filter.gamma_phi * theta_curr;
        let progress = theta_progress || phi_progress;

        if progress && filter.accepts(theta_trial, phi_trial) {
            // Accept. Add current to filter (Fletcher-Leyffer's
            // "h-mode" augmentation; we don't distinguish f-mode
            // here per the module-doc rationale).
            filter.add(theta_curr, phi_curr);
            return LineSearchResult {
                alpha,
                nu: current_nu,
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
        nu: current_nu,
        x_new: x_trial,
        f_new: last_f,
        c_new: last_c,
        success: false,
    }
}
