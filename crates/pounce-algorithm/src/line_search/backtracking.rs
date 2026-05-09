//! Backtracking line-search driver — port of
//! `Algorithm/IpBacktrackingLineSearch.{hpp,cpp}`.
//!
//! Owns the alpha-reduction loop, max-soc / second-order-correction
//! slot, watchdog mechanism, and the fallback to restoration. Phase 7
//! ships the alpha-loop for the filter line search; SOC and watchdog
//! land alongside the restoration phase (Phase 9).
//!
//! The contract with the acceptor is the trio
//! `(theta, phi, d_phi)` at the current iterate plus the trial
//! `(theta_trial, phi_trial)` per backtracking step. Trial-point
//! construction is `x_trial = x + α·dx`, `s_trial = s + α·ds`; the dual
//! step uses the same α for the filter acceptor (upstream
//! `IpBacktrackingLineSearch.cpp:702-728` — primal-dual share α
//! when no fraction-to-the-boundary truncation differs).
//!
//! `find_acceptable_trial_point` returns `Outcome::Accepted` on a
//! successful trial, `Outcome::TinyStep` when α drops below
//! `alpha_min`, and `Outcome::Failed` when the alpha loop exhausts
//! without acceptance (which the main loop maps to a restoration
//! attempt).

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::filter_acceptor::AcceptDecision;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::rc::Rc;

/// Outcome of the backtracking line search. Mirrors the booleans
/// upstream returns through `accept_` plus the `tiny_step_flag` on
/// `IpoptData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Trial point accepted at the recorded `alpha`.
    Accepted,
    /// `alpha` fell below `alpha_min_frac` × current α₀ ⇒ tiny step.
    /// Caller maps to `STEP_BECOMES_TINY` in upstream's exception flow.
    TinyStep,
    /// All α reductions rejected; the caller hands off to restoration.
    Failed,
}

pub struct BacktrackingLineSearch {
    pub acceptor: Box<dyn BacktrackingLsAcceptor>,
    pub alpha_red_factor: Number,
    pub max_soc: i32,
    /// Threshold for the SOC outer-loop convergence test
    /// `theta_trial <= kappa_soc * theta_soc_old`. Mirrors upstream's
    /// `kappa_soc` (default 0.99).
    pub kappa_soc: Number,
    /// SOC RHS variant. `0` = upstream default ("old"), `1` = scaled
    /// gradient-block variant. Both correspond to upstream's
    /// `soc_method` option.
    pub soc_method: i32,
    pub watchdog_shortened_iter_trigger: i32,
    pub watchdog_trial_iter_max: i32,
    /// Lower bound on α; below this we declare a tiny step (mirrors
    /// `alpha_min_frac` flow, `IpBacktrackingLineSearch.cpp:CalculateAlphaMin`).
    pub alpha_min: Number,
    /// Maximum trial-iteration cap before declaring failure.
    pub max_trials: i32,
}

impl BacktrackingLineSearch {
    pub fn new(acceptor: Box<dyn BacktrackingLsAcceptor>) -> Self {
        Self {
            acceptor,
            alpha_red_factor: 0.5,
            max_soc: 4,
            kappa_soc: 0.99,
            soc_method: 0,
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
            alpha_min: 1e-12,
            max_trials: 50,
        }
    }

    pub fn acceptor(&self) -> &dyn BacktrackingLsAcceptor {
        &*self.acceptor
    }

    pub fn acceptor_mut(&mut self) -> &mut dyn BacktrackingLsAcceptor {
        &mut *self.acceptor
    }

    /// Reset the acceptor state at the start of a new outer iteration.
    pub fn reset(&mut self) {
        self.acceptor.reset();
    }

    /// Try `alpha`, then `alpha * alpha_red_factor`, etc. until the
    /// acceptor returns `Accept` or α drops below `alpha_min`.
    ///
    /// On accept: writes the trial iterate to `data.trial`, records
    /// the final α on `data.info_alpha_primal`, and returns
    /// `Outcome::Accepted`. On tiny step / failure, leaves
    /// `data.trial` cleared.
    ///
    /// The acceptor is consulted via [`FilterLsAcceptor::check_acceptability`]
    /// when concrete; for any other `BacktrackingLsAcceptor` the loop
    /// degenerates to "accept first α that produces a finite phi/theta"
    /// (which lets non-filter acceptors land later without changing the
    /// driver's signature).
    pub fn find_acceptable_trial_point(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
        alpha_init: Number,
        alpha_dual: Number,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        search_dir: Option<&mut PdSearchDirCalc>,
    ) -> Outcome {
        // Snapshot phi and theta at the current iterate.
        let theta = cq.borrow().curr_constraint_violation();
        let phi = cq.borrow().curr_barrier_obj();
        let d_phi = self.compute_d_phi(cq, delta);

        // Per-outer-iteration acceptor hook (penalty acceptor uses it to
        // bump ν and cache linearization vectors; filter acceptor: no-op).
        self.acceptor.init_this_line_search(data, cq, delta);

        let curr = match data.borrow().curr.clone() {
            Some(c) => c,
            None => return Outcome::Failed,
        };

        // SOC plumbing — only when both the search-dir calc and the NLP
        // are wired through (so we can re-solve the corrector system).
        // SOC is attempted exactly once, after the *first* trial point
        // (alpha == alpha_init) is rejected and `theta_trial >= theta`
        // (constraint violation grew along the Newton step), mirroring
        // upstream `IpBacktrackingLineSearch.cpp:813`.
        let mut soc_search_dir = search_dir;
        // Allocate the persistent SOC accumulators only if SOC is wired.
        let (mut c_soc_buf, mut dms_soc_buf) = if soc_search_dir.is_some() && nlp.is_some() && self.max_soc > 0
        {
            let cq_ref = cq.borrow();
            let curr_c = cq_ref.curr_c();
            let curr_dms = cq_ref.curr_d_minus_s();
            let mut c_soc = curr_c.make_new();
            c_soc.copy(&*curr_c);
            let mut dms_soc = curr_dms.make_new();
            dms_soc.copy(&*curr_dms);
            (Some(c_soc), Some(dms_soc))
        } else {
            (None, None)
        };

        // Try alpha = alpha_init * alpha_red_factor^k.
        let mut alpha = alpha_init;
        // Track the last α actually tested and how many steps were
        // taken — used by the Failed branch to stamp `info_alpha_primal`
        // / `info_ls_count` before the restoration phase reads them
        // (mirrors `IpBacktrackingLineSearch.cpp:605-608`, which sets
        // these on `IpData()` immediately before
        // `resto_phase_->PerformRestoration()`).
        let mut last_alpha = alpha_init;
        let mut n_steps: i32 = 0;
        // Acceptor-driven dynamic alpha-min (mirrors
        // IpBacktrackingLineSearch.cpp:703 `CalculateAlphaMin`).
        // Floor at the driver's absolute `alpha_min` so we never let
        // the reduction loop run for ever on a problem where the
        // acceptor's formula bottoms out near zero.
        let acceptor_alpha_min = self.acceptor.calc_alpha_min(d_phi, theta);
        let alpha_min_eff = self.alpha_min.max(acceptor_alpha_min);
        for trial in 0..self.max_trials {
            if alpha < alpha_min_eff {
                let mut d = data.borrow_mut();
                d.trial = None;
                // NB: do NOT set `tiny_step_flag` here — that flag means
                // "step direction is too small to make progress" and is
                // checked by the main loop / mu update for clean
                // termination at convergence. An LS that exhausted its
                // alpha budget is a different condition and is handled
                // by the caller via restoration. Conflating the two
                // turns a recoverable LS failure into a premature
                // STOP_AT_TINY_STEP at the next iteration.
                d.info_alpha_primal = last_alpha;
                d.info_alpha_dual = 0.0;
                d.info_alpha_primal_char = 'R';
                d.info_ls_count = n_steps + 1;
                return Outcome::TinyStep;
            }
            last_alpha = alpha;
            n_steps = trial;

            // Build trial = curr + alpha * delta and stash on data.
            // Primal blocks (x, s, y_c, y_d) advance by `alpha`; dual
            // bound multipliers (z, v) advance by `alpha_dual` per the
            // upstream `PerformDualStep` split.
            let trial_iv = scaled_step(&curr, delta, alpha, alpha_dual);
            data.borrow_mut().set_trial(trial_iv);

            // Evaluate trial phi/theta. If non-finite, treat as reject.
            let theta_trial = cq.borrow().trial_constraint_violation();
            let phi_trial = cq.borrow().trial_barrier_obj();
            if !theta_trial.is_finite() || !phi_trial.is_finite() {
                alpha *= self.alpha_red_factor;
                continue;
            }

            let decision = self
                .acceptor
                .check_trial_point(alpha, theta, phi, d_phi, theta_trial, phi_trial);
            if decision == AcceptDecision::Accept {
                // Mirrors upstream
                // `IpBacktrackingLineSearch.cpp:839` — call
                // `UpdateForNextIteration` once on the accepted alpha
                // to (a) decide the info char and (b) conditionally
                // augment the filter. Pounce's filter is *only*
                // augmented here, never separately by the main loop.
                let mode = self
                    .acceptor
                    .update_for_next_iteration(alpha, theta, phi, d_phi, phi_trial);
                let mut d = data.borrow_mut();
                d.info_alpha_primal = alpha;
                d.info_alpha_dual = alpha_dual;
                d.info_ls_count = trial + 1;
                d.info_alpha_primal_char = mode;
                return Outcome::Accepted;
            }

            // Second-order correction: only after the *first* trial
            // point is rejected and only when the constraint violation
            // grew along the Newton step (theta_trial >= theta). Mirrors
            // `IpBacktrackingLineSearch.cpp:809-824`.
            if trial == 0
                && self.max_soc > 0
                && theta <= theta_trial
                && c_soc_buf.is_some()
                && dms_soc_buf.is_some()
            {
                let alpha_test = alpha;
                let mut count_soc: i32 = 0;
                let mut theta_soc_old: Number = 0.0;
                let mut theta_trial_local = theta_trial;
                let mut alpha_primal_soc = alpha;
                let mut soc_accepted = false;
                while count_soc < self.max_soc
                    && !soc_accepted
                    && (count_soc == 0 || theta_trial_local <= self.kappa_soc * theta_soc_old)
                {
                    theta_soc_old = theta_trial_local;
                    // Upstream `IpFilterLSAcceptor.cpp:569`:
                    // `c_soc->AddOneVector(1.0, *trial_c, alpha_primal_soc)`
                    // i.e. `c_soc := trial_c + alpha_primal_soc · c_soc`.
                    {
                        let cq_ref = cq.borrow();
                        let trial_c = cq_ref.trial_c();
                        let trial_dms = cq_ref.trial_d_minus_s();
                        if let Some(c_soc) = c_soc_buf.as_mut() {
                            c_soc.scal(alpha_primal_soc);
                            c_soc.axpy(1.0, &*trial_c);
                        }
                        if let Some(dms_soc) = dms_soc_buf.as_mut() {
                            dms_soc.scal(alpha_primal_soc);
                            dms_soc.axpy(1.0, &*trial_dms);
                        }
                    }
                    // Solve the SOC system.
                    let delta_soc_opt = {
                        let sd = soc_search_dir
                            .as_deref_mut()
                            .expect("SOC: search_dir is gated above");
                        let nlp_ref = nlp.expect("SOC: nlp is gated above");
                        let c_soc = c_soc_buf
                            .as_deref()
                            .expect("SOC: c_soc_buf is gated above");
                        let dms_soc = dms_soc_buf
                            .as_deref()
                            .expect("SOC: dms_soc_buf is gated above");
                        sd.compute_soc_step(
                            data,
                            cq,
                            nlp_ref,
                            c_soc,
                            dms_soc,
                            alpha_primal_soc,
                            self.soc_method,
                        )
                    };
                    let Some(delta_soc) = delta_soc_opt else {
                        break;
                    };
                    // alpha_primal_soc = primal_frac_to_the_bound(tau, delta_soc)
                    let tau = data.borrow().curr_tau;
                    alpha_primal_soc = cq.borrow().aff_step_alpha_primal_max(&delta_soc, tau);
                    // Build the SOC trial iterate. Per upstream
                    // `IpFilterLSAcceptor.cpp:626`, only the primal
                    // (x, s) blocks advance for the SOC retry; y and
                    // bound multipliers stay at their first-trial
                    // values from earlier in this same loop.
                    let mut trial_iv = curr.deep_copy();
                    trial_iv.x.axpy(alpha_primal_soc, &*delta_soc.x);
                    trial_iv.s.axpy(alpha_primal_soc, &*delta_soc.s);
                    // Dual blocks track the original step (not the
                    // SOC delta) so the multiplier values match what
                    // upstream's `SetTrialPrimalVariablesFromStep` +
                    // existing dual update produce.
                    trial_iv.y_c.axpy(alpha, &*delta.y_c);
                    trial_iv.y_d.axpy(alpha, &*delta.y_d);
                    trial_iv.z_l.axpy(alpha_dual, &*delta.z_l);
                    trial_iv.z_u.axpy(alpha_dual, &*delta.z_u);
                    trial_iv.v_l.axpy(alpha_dual, &*delta.v_l);
                    trial_iv.v_u.axpy(alpha_dual, &*delta.v_u);
                    let trial_iv = trial_iv.freeze();
                    data.borrow_mut().set_trial(trial_iv);
                    let theta_soc = cq.borrow().trial_constraint_violation();
                    let phi_soc = cq.borrow().trial_barrier_obj();
                    if !theta_soc.is_finite() || !phi_soc.is_finite() {
                        break;
                    }
                    let dec = self.acceptor.check_trial_point(
                        alpha_test, theta, phi, d_phi, theta_soc, phi_soc,
                    );
                    if dec == AcceptDecision::Accept {
                        let mode = self.acceptor.update_for_next_iteration(
                            alpha_test, theta, phi, d_phi, phi_soc,
                        );
                        let mut d = data.borrow_mut();
                        d.info_alpha_primal = alpha_primal_soc;
                        d.info_alpha_dual = alpha_dual;
                        d.info_ls_count = trial + 1;
                        d.info_alpha_primal_char = mode.to_ascii_uppercase();
                        return Outcome::Accepted;
                    }
                    count_soc += 1;
                    theta_trial_local = theta_soc;
                    soc_accepted = false;
                }
            }

            alpha *= self.alpha_red_factor;
        }

        // Exhausted retries without acceptance. Mirror upstream
        // `IpBacktrackingLineSearch.cpp:605-608`: stamp the failure α,
        // dual=0, char='R', ls=n_steps+1 onto IpData so
        // `RestoMinC_1Nrm::Set_info_*` (and pounce's equivalent in
        // `resto_inner_solver.rs`) can copy them onto the nested
        // IPM's data before the inner's first OutputIteration row
        // prints.
        let mut d = data.borrow_mut();
        d.trial = None;
        d.info_alpha_primal = last_alpha;
        d.info_alpha_dual = 0.0;
        d.info_alpha_primal_char = 'R';
        d.info_ls_count = n_steps + 1;
        Outcome::Failed
    }

    /// Directional derivative of the barrier objective along the step
    /// `delta`: `d_phi = ∇_x φ · dx + ∇_s φ · ds`.
    fn compute_d_phi(&self, cq: &IpoptCqHandle, delta: &IteratesVector) -> Number {
        let cq_ref = cq.borrow();
        let g_x = cq_ref.curr_grad_barrier_obj_x();
        let g_s = cq_ref.curr_grad_barrier_obj_s();
        g_x.dot(&*delta.x) + g_s.dot(&*delta.s)
    }

}

/// `out = curr + alpha * delta` for all eight components, returned as a
/// fresh `IteratesVector` with `Rc<dyn Vector>` slots. Mirrors
/// `IpoptData::SetTrialBoundMultipliersFromStep` + the primal step
/// path in upstream — both share the same scalar α here because
/// fraction-to-the-boundary truncation has already been folded into
/// `alpha_init` upstream.
fn scaled_step(
    curr: &IteratesVector,
    delta: &IteratesVector,
    alpha_primal: Number,
    alpha_dual: Number,
) -> IteratesVector {
    let mut out = curr.make_new_zeroed();
    out.add_one_vector(1.0, curr, 0.0); // out = curr
    out.x.axpy(alpha_primal, &*delta.x);
    out.s.axpy(alpha_primal, &*delta.s);
    out.y_c.axpy(alpha_primal, &*delta.y_c);
    out.y_d.axpy(alpha_primal, &*delta.y_d);
    out.z_l.axpy(alpha_dual, &*delta.z_l);
    out.z_u.axpy(alpha_dual, &*delta.z_u);
    out.v_l.axpy(alpha_dual, &*delta.v_l);
    out.v_u.axpy(alpha_dual, &*delta.v_u);
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterates_vector::IteratesVector;
    use crate::line_search::filter_acceptor::FilterLsAcceptor;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::Vector;
    use std::rc::Rc;

    fn dense(n: i32, vals: &[Number]) -> Rc<dyn Vector> {
        let mut v = DenseVectorSpace::new(n).make_new_dense();
        v.set(0.0);
        if !vals.is_empty() {
            v.values_mut().copy_from_slice(vals);
        }
        Rc::new(v)
    }

    fn iv_from(x: &[Number], s: &[Number]) -> IteratesVector {
        IteratesVector::new(
            dense(x.len() as i32, x),
            dense(s.len() as i32, s),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
        )
    }

    #[test]
    fn driver_constructs_with_defaults() {
        let bls = BacktrackingLineSearch::new(Box::new(FilterLsAcceptor::new()));
        assert_eq!(bls.alpha_red_factor, 0.5);
        assert_eq!(bls.max_soc, 4);
    }

    #[test]
    fn scaled_step_writes_curr_plus_alpha_delta() {
        // curr.x = (0,0), delta.x = (1,1) → at alpha=0.5, trial.x = (0.5, 0.5).
        let curr = iv_from(&[0.0, 0.0], &[0.0]);
        let delta = iv_from(&[1.0, 1.0], &[2.0]);
        let trial = scaled_step(&curr, &delta, 0.5, 0.5);
        let xv = trial
            .x
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(xv, vec![0.5, 0.5]);
        let sv = trial
            .s
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(sv, vec![1.0]); // 0.0 + 0.5 * 2.0
    }

    #[test]
    fn outcome_variants_are_distinct() {
        assert_ne!(Outcome::Accepted, Outcome::Failed);
        assert_ne!(Outcome::Accepted, Outcome::TinyStep);
        assert_ne!(Outcome::Failed, Outcome::TinyStep);
    }
}
