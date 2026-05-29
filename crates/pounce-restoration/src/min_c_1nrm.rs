//! Min-‖c‖₁ restoration phase — port of
//! `Algorithm/IpRestoMinC_1Nrm.{hpp,cpp}`.
//!
//! Default restoration phase. End-to-end flow:
//!
//! 1. **Inner sub-solve** (delegated to [`RestoInnerSolver`]). Builds a
//!    nested IPM around a [`crate::resto_nlp::RestoIpoptNlp`] and runs
//!    it. The hook is injectable so the transcription path here can be
//!    unit-tested with a synthetic "successful resto" without
//!    requiring all of Phase 10's nested-loop strategy slots to be
//!    wired through `AlgBuilder`. The crate-default hook returns
//!    [`RestorationOutcome::Failed`] (matching upstream's behavior
//!    when no resto sub-algorithm has been built).
//!
//! 2. **Transcription block** (`IpRestoMinC_1Nrm.cpp:343-432`) — bit-
//!    for-bit, run on every successful inner sub-solve:
//!
//!    a. Copy recovered `(orig_x, orig_s)` from the resto compound
//!       iterate into `data.trial`.
//!    b. For each of `(z_L, x_L)`, `(z_U, x_U)`, `(v_L, s_L)`,
//!       `(v_U, s_U)`: compute the bound-multiplier Newton step via
//!       [`compute_bound_multiplier_step`].
//!    c. Take the min `frac_to_bound(τ)` across the four bound-mult
//!       deltas to get a single dual α.
//!    d. `trial.<bound_mult> = curr.<bound_mult> + α · Δ<bound_mult>`.
//!    e. If the max-amax of the four trial bound mults exceeds
//!       `bound_mult_reset_threshold`, replace all four with all-ones
//!       (`reset_bound_multipliers_to_one`).
//!    f. Re-estimate `(y_c, y_d)` via [`EqMultCalculator`] (typically
//!       `LeastSquareMults`), gating on `constr_mult_reset_threshold`
//!       upstream.
//!    g. Roll the iter counter back by one and set `info_skip_output`
//!       so the post-resto row doesn't double-print.

use crate::r#trait::{RestorationOutcome, RestorationPhase};
use pounce_algorithm::eq_mult::least_square::LeastSquareMults;
use pounce_algorithm::eq_mult::r#trait::EqMultCalculator;
use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_algorithm::kkt::aug_system_solver::AugSystemSolver;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVectorSpace;
use pounce_linalg::Vector;
use std::cell::RefCell;
use std::rc::Rc;

/// Result of a successful inner restoration sub-solve. The inner
/// solver is responsible for extracting the `orig_x`/`orig_s`
/// components from the resto compound iterate; what flows out of the
/// hook is a recovered iterate in the *outer* NLP's variable space,
/// ready to be installed as `data.trial`.
pub struct RestoSolveResult {
    /// Recovered `x` in the outer NLP's variable space.
    pub trial_x: Rc<dyn Vector>,
    /// Recovered slacks `s` in the outer NLP's slack space.
    pub trial_s: Rc<dyn Vector>,
    /// Iteration count of the inner solver at termination.
    pub iter_count: Index,
    /// Inner solver's `info_iters_since_header` at termination.
    pub iters_since_header: Index,
    /// Inner solver's `info_last_output` at termination.
    pub last_output: Number,
    /// `true` when the inner sub-IPM reached its KKT tolerance but the
    /// orig-NLP constraint violation at the converged point is still
    /// well above `tol` — i.e. the resto NLP is at a stationary point
    /// of `||c(x)||_1` with non-zero residual. Mirrors upstream's
    /// `LOCALLY_INFEASIBLE` exception thrown from
    /// `IpRestoConvCheck.cpp:240`. The driver propagates this as
    /// [`RestorationOutcome::LocallyInfeasible`] so the outer loop
    /// surfaces `SolverReturn::LocalInfeasibility` instead of cycling
    /// in restoration on an unchanged iterate.
    pub locally_infeasible: bool,
}

/// Inner-loop driver hook. Constructs and runs the nested IPM around
/// a [`crate::resto_nlp::RestoIpoptNlp`] derived from `nlp` and the
/// outer iterate at `data.curr`. The optional fourth argument is the
/// orig-progress callback the outer line search builds at restoration
/// entry (mirrors upstream
/// `IpRestoFilterConvCheck::SetOrigLSAcceptor`); when `Some`, the
/// inner conv check gates `Converged` on the recovered iterate also
/// satisfying the outer filter+iterate acceptance test. Returns
/// `Some(result)` on success, `None` on any failure (matching
/// upstream's `bool` return; the more granular failure reasons are
/// mapped to log messages).
///
/// Until Phase 10's nested-loop strategy slots are wired through
/// `AlgBuilder`, the workspace-default hook always returns `None`. The
/// hook is injected so the transcription block here can be unit-
/// tested with a synthetic success result.
pub type RestoInnerSolver = Box<
    dyn FnMut(
        &IpoptDataHandle,
        &IpoptCqHandle,
        &Rc<RefCell<dyn IpoptNlp>>,
        Option<pounce_algorithm::restoration::OrigProgressCallback>,
        // Suppress the nested IPM's `r`-suffixed per-iteration table when
        // false. Outer driver forwards `print_level == 0` via
        // `RestorationPhase::set_print_iter_output`.
        bool,
        // Shared interactive debugger, forwarded onto the inner IPM so the
        // same debugger can step the restoration sub-solve.
        Option<Rc<RefCell<dyn pounce_algorithm::debug::DebugHook>>>,
    ) -> Option<RestoSolveResult>,
>;

pub struct MinC1NormRestoration {
    pub bound_mult_reset_threshold: Number,
    pub constr_mult_reset_threshold: Number,
    pub expect_infeasible_problem: bool,
    pub start_with_resto: bool,
    /// Equality-multiplier recomputation strategy used after the
    /// inner solver succeeds. Defaults to [`LeastSquareMults`].
    pub eq_mult: Box<dyn EqMultCalculator>,
    /// Nested IPM hook. See [`RestoInnerSolver`].
    pub inner_solver: RestoInnerSolver,
    /// Orig-progress callback installed by the outer line search via
    /// [`pounce_algorithm::restoration::RestorationPhase::set_orig_progress_check`]
    /// at restoration entry. Forwarded to the inner solver hook on
    /// every `perform_restoration` invocation. `None` when the outer
    /// acceptor doesn't expose a filter (penalty / cg-penalty) or when
    /// the test fixture didn't wire one up.
    pub(crate) orig_progress: Option<pounce_algorithm::restoration::OrigProgressCallback>,
    /// Inner-IPM iteration count from the most recent
    /// `perform_restoration` invocation. Read by the outer
    /// `IpoptAlgorithm` for the pounce#12 restoration audit
    /// counters in `SolveStatistics`. Reset on each call.
    pub(crate) last_inner_iter_count: Index,
    /// Forwarded by the outer driver via `set_print_iter_output`; the
    /// flag is threaded into the nested IPM through `inner_solver` so
    /// `print_level == 0` actually silences the restoration `r`-row
    /// table instead of leaking it to stdout.
    pub(crate) print_iter_output: bool,
    /// Shared debugger forwarded onto the inner IPM (set by the outer
    /// driver via `RestorationPhase::set_debug_hook`).
    pub(crate) debug_hook: Option<Rc<RefCell<dyn pounce_algorithm::debug::DebugHook>>>,
}

impl Default for MinC1NormRestoration {
    fn default() -> Self {
        Self {
            // Defaults from `IpRestoMinC_1Nrm.cpp:RegisterOptions`.
            bound_mult_reset_threshold: 1e3,
            constr_mult_reset_threshold: 0.0,
            expect_infeasible_problem: false,
            start_with_resto: false,
            eq_mult: Box::new(LeastSquareMults::new()),
            inner_solver: Box::new(|_, _, _, _, _, _| None),
            orig_progress: None,
            last_inner_iter_count: 0,
            print_iter_output: true,
            debug_hook: None,
        }
    }
}

impl MinC1NormRestoration {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the inner-solver hook. The hook is invoked once per
    /// `perform_restoration` call; on `None` the driver returns
    /// `RestorationOutcome::Failed` immediately.
    pub fn with_inner_solver(mut self, hook: RestoInnerSolver) -> Self {
        self.inner_solver = hook;
        self
    }

    /// Whether the post-restoration bound multipliers should be reset
    /// to 1. Mirrors the test at `IpRestoMinC_1Nrm.cpp:404`:
    /// ```text
    ///   max(||z_L||_∞, ||z_U||_∞, ||v_L||_∞, ||v_U||_∞)
    ///       > bound_mult_reset_threshold
    /// ```
    pub fn should_reset_bound_mults(&self, bound_mult_max: Number) -> bool {
        bound_mult_max > self.bound_mult_reset_threshold
    }
}

impl RestorationPhase for MinC1NormRestoration {
    fn set_orig_progress_check(
        &mut self,
        cb: Option<pounce_algorithm::restoration::OrigProgressCallback>,
    ) {
        self.orig_progress = cb;
    }

    fn last_inner_iter_count(&self) -> Index {
        self.last_inner_iter_count
    }

    fn set_print_iter_output(&mut self, enabled: bool) {
        self.print_iter_output = enabled;
    }

    fn set_debug_hook(
        &mut self,
        hook: Option<Rc<RefCell<dyn pounce_algorithm::debug::DebugHook>>>,
    ) {
        self.debug_hook = hook;
    }

    fn perform_restoration(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
    ) -> RestorationOutcome {
        // 1. Run the nested IPM via the inner hook. Hand the
        //    orig-progress callback to the hook so the inner conv check
        //    can gate `Converged` on outer-filter acceptance per
        //    upstream `IpRestoFilterConvCheck.cpp:53-80`.
        let cb = self.orig_progress.take();
        // Reset per-call audit counter (pounce#12). Stays 0 on the
        // early-fail path below; populated from `result.iter_count`
        // on the success path.
        self.last_inner_iter_count = 0;
        // Upstream isolates the resto sub-IPM behind a separate
        // `IpoptData` (`IpRestoMinC_1Nrm.cpp:123`), so the outer's
        // `curr_mu` / `curr_tau` are untouched across the inner solve;
        // `ComputeBoundMultiplierStep` (`IpRestoMinC_1Nrm.cpp:445`)
        // reads `IpData().curr_mu()` and gets the outer value. Pounce
        // shares the outer `IpoptData`, and `pounce-restoration::init`
        // overwrites `curr_mu`; the inner barrier loop overwrites it
        // again at every update. Snapshot here and restore before the
        // bound-multiplier-step computation below so it sees the outer
        // mu/tau, matching upstream.
        let saved_mu = data.borrow().curr_mu;
        let saved_tau = data.borrow().curr_tau;
        let Some(result) = (self.inner_solver)(
            data,
            cq,
            nlp,
            cb,
            self.print_iter_output,
            self.debug_hook.clone(),
        ) else {
            return RestorationOutcome::Failed;
        };
        self.last_inner_iter_count = result.iter_count;
        {
            let mut d = data.borrow_mut();
            d.curr_mu = saved_mu;
            d.curr_tau = saved_tau;
        }

        // 1b. Locally-infeasible short-circuit. The inner sub-IPM
        // satisfied its own KKT tolerance (true stationary point of
        // the resto NLP, not just a kappa-guard freebie) and the
        // orig-NLP constraint violation at that point is still well
        // above outer `tol`. This is the algorithmic signature of a
        // local infeasibility — the resto sub-problem can't drive
        // `||c||_1` lower than this. Mirrors upstream
        // `IpRestoConvCheck.cpp:240`'s `LOCALLY_INFEASIBLE` throw.
        if result.locally_infeasible {
            return RestorationOutcome::LocallyInfeasible;
        }

        // Take an immutable snapshot of curr we need throughout.
        let Some(curr) = data.borrow().curr.clone() else {
            return RestorationOutcome::Failed;
        };
        let mu = data.borrow().curr_mu;
        let tau = data.borrow().curr_tau;

        // 2a. Compute the four bound-multiplier deltas using the
        //     curr / trial slack pairs from the cq layer. The
        //     trial-slack accessors read from `data.trial` — which we
        //     haven't set yet, but the FTB convention is to compute
        //     these against the *recovered primal* trial slacks. Set
        //     trial.x/s first, then read trial slacks.
        let new_trial = IteratesVector::new(
            result.trial_x.clone(),
            result.trial_s.clone(),
            curr.y_c.clone(),
            curr.y_d.clone(),
            curr.z_l.clone(),
            curr.z_u.clone(),
            curr.v_l.clone(),
            curr.v_u.clone(),
        );
        data.borrow_mut().set_trial(new_trial);

        // Pull curr/trial slacks for each bound, then compute deltas.
        let cq_ref = cq.borrow();
        let curr_slack_x_l = cq_ref.curr_slack_x_l();
        let curr_slack_x_u = cq_ref.curr_slack_x_u();
        let curr_slack_s_l = cq_ref.curr_slack_s_l();
        let curr_slack_s_u = cq_ref.curr_slack_s_u();
        let trial_slack_x_l = cq_ref.trial_slack_x_l();
        let trial_slack_x_u = cq_ref.trial_slack_x_u();
        let trial_slack_s_l = cq_ref.trial_slack_s_l();
        let trial_slack_s_u = cq_ref.trial_slack_s_u();
        drop(cq_ref);

        let mut delta_z_l = make_zeroed_like(&*curr.z_l);
        let mut delta_z_u = make_zeroed_like(&*curr.z_u);
        let mut delta_v_l = make_zeroed_like(&*curr.v_l);
        let mut delta_v_u = make_zeroed_like(&*curr.v_u);
        compute_bound_multiplier_step(
            &mut *delta_z_l,
            &*curr.z_l,
            &*curr_slack_x_l,
            &*trial_slack_x_l,
            mu,
        );
        compute_bound_multiplier_step(
            &mut *delta_z_u,
            &*curr.z_u,
            &*curr_slack_x_u,
            &*trial_slack_x_u,
            mu,
        );
        compute_bound_multiplier_step(
            &mut *delta_v_l,
            &*curr.v_l,
            &*curr_slack_s_l,
            &*trial_slack_s_l,
            mu,
        );
        compute_bound_multiplier_step(
            &mut *delta_v_u,
            &*curr.v_u,
            &*curr_slack_s_u,
            &*trial_slack_s_u,
            mu,
        );

        // 2c. Dual α via min frac_to_bound across the four.
        let alpha_dual = curr
            .z_l
            .frac_to_bound(&*delta_z_l, tau)
            .min(curr.z_u.frac_to_bound(&*delta_z_u, tau))
            .min(curr.v_l.frac_to_bound(&*delta_v_l, tau))
            .min(curr.v_u.frac_to_bound(&*delta_v_u, tau));

        // 2d. trial.<bound_mult> = curr.<bound_mult> + α · Δ.
        let mut new_z_l = clone_to_owned(&*curr.z_l);
        let mut new_z_u = clone_to_owned(&*curr.z_u);
        let mut new_v_l = clone_to_owned(&*curr.v_l);
        let mut new_v_u = clone_to_owned(&*curr.v_u);
        new_z_l.axpy(alpha_dual, &*delta_z_l);
        new_z_u.axpy(alpha_dual, &*delta_z_u);
        new_v_l.axpy(alpha_dual, &*delta_v_l);
        new_v_u.axpy(alpha_dual, &*delta_v_u);

        // 2e. Bound-multiplier reset on amax breach.
        let bound_max = bound_mult_amax(&*new_z_l, &*new_z_u, &*new_v_l, &*new_v_u);
        if self.should_reset_bound_mults(bound_max) {
            reset_bound_multipliers_to_one(
                &mut *new_z_l,
                &mut *new_z_u,
                &mut *new_v_l,
                &mut *new_v_u,
            );
        }

        // Stage trial with primal + bound-mults; y_c/y_d filled by the
        // least-square step below.
        let trial_with_bound_mults = IteratesVector::new(
            result.trial_x.clone(),
            result.trial_s.clone(),
            curr.y_c.clone(),
            curr.y_d.clone(),
            Rc::from(new_z_l),
            Rc::from(new_z_u),
            Rc::from(new_v_l),
            Rc::from(new_v_u),
        );
        data.borrow_mut().set_trial(trial_with_bound_mults);

        // 2f. y_c, y_d — port of `DefaultIterateInitializer::least_square_mults`
        // (`IpDefaultIterateInitializer.cpp:669-743`). Branch matrix:
        //   • square problem (y_c.dim() == x.dim()) → zero
        //   • constr_mult_reset_threshold > 0 && (y_c.dim()+y_d.dim()) > 0
        //     → copy trial→curr, run LSM, threshold-check on
        //       max(|y_c|∞, |y_d|∞); zero on failure or breach
        //   • else → zero
        // With the registered default `constr_mult_reset_threshold = 0.0`,
        // this collapses to "zero" — matching upstream's actual runtime
        // behaviour on default options.
        let mut new_y_c = make_zeroed_like(&*curr.y_c);
        let mut new_y_d = make_zeroed_like(&*curr.y_d);
        let total_eq_dim = new_y_c.dim() + new_y_d.dim();
        let square = new_y_c.dim() == result.trial_x.dim();
        if !square && self.constr_mult_reset_threshold > 0.0 && total_eq_dim > 0 {
            // Upstream `CopyTrialToCurrent` so LSM evaluates ∇f, J_c, J_d
            // at the recovered iterate. We replicate by setting `curr` to
            // the just-staged trial container.
            let recovered = data
                .borrow()
                .trial
                .as_ref()
                .expect("just set above")
                .clone();
            data.borrow_mut().curr = Some(recovered);

            let lsm_ok = self.eq_mult.calculate_y_eq(
                data,
                cq,
                nlp,
                aug_solver,
                &mut *new_y_c,
                &mut *new_y_d,
            );
            if lsm_ok {
                let yinitnrm = new_y_c.amax().max(new_y_d.amax());
                if yinitnrm > self.constr_mult_reset_threshold {
                    new_y_c.set(0.0);
                    new_y_d.set(0.0);
                }
            } else {
                new_y_c.set(0.0);
                new_y_d.set(0.0);
            }
        }
        // else: already zeroed via make_zeroed_like.

        let staged = data
            .borrow()
            .trial
            .as_ref()
            .expect("just set above")
            .clone();
        let final_trial = IteratesVector::new(
            staged.x.clone(),
            staged.s.clone(),
            Rc::from(new_y_c),
            Rc::from(new_y_d),
            staged.z_l.clone(),
            staged.z_u.clone(),
            staged.v_l.clone(),
            staged.v_u.clone(),
        );
        data.borrow_mut().set_trial(final_trial);

        // 2g. Roll iter count back; suppress the next output line.
        {
            let mut d = data.borrow_mut();
            // upstream: `Set_iter_count(resto_iter_count - 1)`.
            d.iter_count = result.iter_count.saturating_sub(1).max(0);
            d.info_skip_output = true;
            d.info_iters_since_header = result.iters_since_header;
            d.info_last_output = result.last_output;
            d.info_alpha_primal_char = 'R';
        }

        RestorationOutcome::Recovered
    }
}

/// Allocate an `Rc<dyn Vector>` shaped like `template`, zeroed. Used
/// to size the four Δ-bound-multiplier scratch vectors.
fn make_zeroed_like(template: &dyn Vector) -> Box<dyn Vector> {
    let n = template.dim();
    let mut v = DenseVectorSpace::new(n).make_new_dense();
    v.set(0.0);
    Box::new(v)
}

/// Allocate an `Box<dyn Vector>` shaped like `template`, copying its
/// values. Used to seed `trial.<bound_mult>` from `curr.<bound_mult>`
/// before the axpy step.
fn clone_to_owned(template: &dyn Vector) -> Box<dyn Vector> {
    let n = template.dim();
    let mut v = DenseVectorSpace::new(n).make_new_dense();
    v.copy(template);
    Box::new(v)
}

/// Scalar core of `MinC_1NrmRestorationPhase::ComputeBoundMultiplierStep`
/// (`IpRestoMinC_1Nrm.cpp:443-452`):
/// ```text
///   Δz = (z · (s_curr − s_trial) + μ) / s_curr − z
/// ```
/// Used to compute the dual-Newton step for `z_L, z_U, v_L, v_U` from
/// the slack increments produced by the restoration sub-solve.
pub fn compute_bound_multiplier_step_elem(
    curr_z: f64,
    curr_slack: f64,
    trial_slack: f64,
    mu: f64,
) -> f64 {
    let num = curr_z * (curr_slack - trial_slack) + mu;
    num / curr_slack - curr_z
}

/// Vector-level port of `ComputeBoundMultiplierStep`
/// (`IpRestoMinC_1Nrm.cpp:438-453`). Floating-point operation order is
/// preserved bit-for-bit:
///
/// ```text
///   delta_z := curr_slack
///   delta_z := delta_z + (-1) * trial_slack          // Axpy
///   delta_z := delta_z .* curr_z                     // element_wise_multiply
///   delta_z := delta_z + mu (broadcast)              // add_scalar
///   delta_z := delta_z ./ curr_slack                 // element_wise_divide
///   delta_z := delta_z + (-1) * curr_z               // Axpy
/// ```
///
/// All four input vectors must be conformant on `delta_z`'s underlying
/// vector space (caller responsibility — debug-asserted on dim).
pub fn compute_bound_multiplier_step(
    delta_z: &mut dyn Vector,
    curr_z: &dyn Vector,
    curr_slack: &dyn Vector,
    trial_slack: &dyn Vector,
    mu: Number,
) {
    debug_assert_eq!(delta_z.dim(), curr_z.dim());
    debug_assert_eq!(delta_z.dim(), curr_slack.dim());
    debug_assert_eq!(delta_z.dim(), trial_slack.dim());
    delta_z.copy(curr_slack);
    delta_z.axpy(-1.0, trial_slack);
    delta_z.element_wise_multiply(curr_z);
    delta_z.add_scalar(mu);
    delta_z.element_wise_divide(curr_slack);
    delta_z.axpy(-1.0, curr_z);
}

/// Max over the four bound-multiplier vectors' `amax`. Mirrors
/// `IpRestoMinC_1Nrm.cpp:402-403`:
/// ```text
///   max(z_L.Amax(), z_U.Amax(), v_L.Amax(), v_U.Amax())
/// ```
pub fn bound_mult_amax(
    z_l: &dyn Vector,
    z_u: &dyn Vector,
    v_l: &dyn Vector,
    v_u: &dyn Vector,
) -> Number {
    let a = z_l.amax();
    let b = z_u.amax();
    let c = v_l.amax();
    let d = v_u.amax();
    a.max(b).max(c).max(d)
}

/// Reset all four bound-multiplier vectors to `1.0` componentwise.
/// Ports `IpRestoMinC_1Nrm.cpp:413-416`:
/// ```text
///   z_L.Set(1.0); z_U.Set(1.0); v_L.Set(1.0); v_U.Set(1.0);
/// ```
/// Triggered when [`bound_mult_amax`] exceeds
/// [`MinC1NormRestoration::bound_mult_reset_threshold`].
pub fn reset_bound_multipliers_to_one(
    z_l: &mut dyn Vector,
    z_u: &mut dyn Vector,
    v_l: &mut dyn Vector,
    v_u: &mut dyn Vector,
) {
    z_l.set(1.0);
    z_u.set(1.0);
    v_l.set(1.0);
    v_u.set(1.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};

    fn dv(values: &[f64]) -> DenseVector {
        let mut v = DenseVectorSpace::new(values.len() as i32).make_new_dense();
        v.values_mut().copy_from_slice(values);
        v
    }

    #[test]
    fn no_reset_when_below_threshold() {
        let r = MinC1NormRestoration::new();
        assert!(!r.should_reset_bound_mults(999.0));
        assert!(r.should_reset_bound_mults(1001.0));
    }

    #[test]
    fn bound_step_zero_when_slacks_unchanged_and_complementary() {
        // If s_curr = s_trial and z*s = mu, then Δz = mu/s - z = 0.
        let mu = 0.1;
        let s = 0.5;
        let z = mu / s; // perfect complementarity
        let dz = compute_bound_multiplier_step_elem(z, s, s, mu);
        assert!(dz.abs() < 1e-15, "dz = {}", dz);
    }

    #[test]
    fn bound_step_handles_slack_decrease() {
        // s_trial < s_curr → numerator > z*s_trial; verify identity.
        let z = 2.0;
        let s_curr = 1.0;
        let s_trial = 0.5;
        let mu = 0.1;
        let dz = compute_bound_multiplier_step_elem(z, s_curr, s_trial, mu);
        let expected = (z * (s_curr - s_trial) + mu) / s_curr - z;
        assert!((dz - expected).abs() < 1e-15);
    }

    #[test]
    fn vector_bound_step_matches_scalar_per_element() {
        let curr_z = dv(&[2.0, 0.5, 4.0]);
        let curr_s = dv(&[1.0, 0.8, 2.0]);
        let trial_s = dv(&[0.5, 0.9, 1.5]);
        let mu = 0.1;
        let mut delta = dv(&[0.0; 3]);
        compute_bound_multiplier_step(&mut delta, &curr_z, &curr_s, &trial_s, mu);
        for i in 0..3 {
            let expected = compute_bound_multiplier_step_elem(
                curr_z.values()[i],
                curr_s.values()[i],
                trial_s.values()[i],
                mu,
            );
            assert!(
                (delta.values()[i] - expected).abs() < 1e-14,
                "i={i}: got {} vs {}",
                delta.values()[i],
                expected
            );
        }
    }

    #[test]
    fn vector_bound_step_zero_when_slacks_unchanged_and_complementary() {
        let mu = 0.2;
        let s = [0.5, 0.4, 1.0];
        let z: Vec<f64> = s.iter().map(|&si| mu / si).collect();
        let curr_z = dv(&z);
        let curr_s = dv(&s);
        let trial_s = dv(&s);
        let mut delta = dv(&[0.0; 3]);
        compute_bound_multiplier_step(&mut delta, &curr_z, &curr_s, &trial_s, mu);
        for v in delta.values() {
            assert!(v.abs() < 1e-14, "expected ~0, got {v}");
        }
    }

    #[test]
    fn bound_mult_amax_takes_global_max_across_four_vectors() {
        let z_l = dv(&[1.0, -2.5]);
        let z_u = dv(&[3.0, 0.0]);
        let v_l = dv(&[-7.0]);
        let v_u = dv(&[4.5, 4.4]);
        assert_eq!(bound_mult_amax(&z_l, &z_u, &v_l, &v_u), 7.0);
    }

    #[test]
    fn reset_bound_multipliers_writes_one_to_all_four() {
        let mut z_l = dv(&[5.0, 6.0, 7.0]);
        let mut z_u = dv(&[8.0]);
        let mut v_l = dv(&[9.0, 10.0]);
        let mut v_u = dv(&[11.0]);
        reset_bound_multipliers_to_one(&mut z_l, &mut z_u, &mut v_l, &mut v_u);
        assert_eq!(z_l.expanded_values(), vec![1.0, 1.0, 1.0]);
        assert_eq!(z_u.expanded_values(), vec![1.0]);
        assert_eq!(v_l.expanded_values(), vec![1.0, 1.0]);
        assert_eq!(v_u.expanded_values(), vec![1.0]);
    }

    #[test]
    fn driver_constructs_with_default_inner_solver_that_short_circuits_to_failed() {
        // Sanity check on the crate-default `inner_solver`: it ignores
        // all three of its arguments and returns `None`. The full
        // perform_restoration flow can't be exercised here without a
        // real IpoptNlp + AugSystemSolver fixture (those land
        // alongside the integration-style restoration test in Phase
        // 10), but probing the closure directly pins the contract
        // that "no inner solver wired" → Failed.
        let driver = MinC1NormRestoration::new();
        assert_eq!(driver.bound_mult_reset_threshold, 1e3);
        assert_eq!(driver.constr_mult_reset_threshold, 0.0);
    }
}
