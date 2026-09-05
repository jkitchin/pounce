//! Lazy-cache layer — port of
//! `Algorithm/IpIpoptCalculatedQuantities.{hpp,cpp}`.
//!
//! Upstream's CQ object exposes ~80 cached quantities (`curr_f`,
//! `curr_grad_f`, `curr_jac_c`, `curr_grad_lag_x`, `curr_compl_*`,
//! `curr_nlp_error`, etc.). All of them are pure derivations from
//! `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` and the NLP function
//! evaluations.
//!
//! Phase 5 ships the priority subset needed by the KKT layer
//! (Phase 6) and the convergence check / line search (Phase 7).
//! Caching is intentionally deferred — every accessor recomputes its
//! value on each call. Tag-based invalidation lands once the inner
//! loop benchmarks justify the bookkeeping; correctness does not
//! depend on it.
//!
//! All accessors take `&self` and return `Rc<dyn Vector>`. NLP
//! evaluations require a brief `borrow_mut()` on the Nlp handle;
//! callers must not hold an outstanding `borrow()` across an
//! accessor call.

use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use pounce_common::cached::Cache;
use pounce_common::tagged::TaggedObject;
use pounce_common::types::Number;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::{Matrix, SymMatrix, Vector};
use std::cell::RefCell;
use std::rc::Rc;

/// Safety factor on the per-row noise floor of
/// [`IpoptCalculatedQuantities::row_noise_floor`]. The floor prices one
/// component of `x` at `eps · ‖x‖_∞` and passes it through the row at
/// `max_j |a_ij|`; the row's residual accumulates that over all of its
/// nonzeros, and the linear solve's conditioning widens it further, so the
/// bare product is short by a problem-dependent factor. `64` covers a typical
/// sparse row without reaching far enough to swallow a declared magnitude a
/// model could have meant: at the `‖x‖_∞ ~ 1` of a well-posed problem the
/// floor sits near
/// `1.4e-14`, still nine orders under `constr_viol_tol`'s default. Rows it
/// silences fall back on the absolute feasibility test, which is already
/// scale-invariant on a row whose declared magnitude is numerically zero.
///
/// Measured, and not a knife edge: over gh #446's 15 problems plus the
/// infeasibility-detection suites (`false_local_infeasibility`,
/// `infeasible_status_tol_invariance`, `issue_390_nonlinear_equality_scale`),
/// every value from `8` to `1024` gives the same verdicts. `1` is too small —
/// QSCSD1's rows are wide enough that the missing nonzero-count factor still
/// leaves its `2^-53` RHS above the bound — so `64` sits an order of magnitude
/// inside the band from either edge.
pub const ROW_NOISE_KAPPA: Number = 64.0;

/// Headroom on the representability floor `calculate_safe_slack` puts under a
/// slack so that `Σ = z/s` stays inside the double range (gh#655).
///
/// The bare requirement is `s ≥ z/f64::MAX`, which bounds `z/s` by `f64::MAX`
/// exactly — no room for the rounding in the divide itself, and none for the
/// fact that `Σ_x` *sums* a lower and an upper ratio into the same diagonal
/// entry. Dividing the floor's budget by `4` bounds each ratio by `MAX/4` and
/// so their sum by `MAX/2`, which leaves the KKT diagonal finite with a bit
/// to spare. The factor costs nothing anywhere it is not needed: the floor is
/// `z_max/4.5e307`, below every slack any non-pathological iterate carries.
const SIGMA_OVERFLOW_HEADROOM: Number = 4.0;

/// Calculated-quantities object. Holds shared handles on data and the
/// NLP; per-quantity caches live in `RefCell`s here.
pub struct IpoptCalculatedQuantities {
    data: IpoptDataHandle,
    nlp: Rc<RefCell<dyn IpoptNlp>>,

    /// Optimality scaling cap from `IpOptErrorConvCheck` defaults.
    pub s_max: Number,
    /// Damping coefficient for the bound-multiplier complementarity
    /// term (`kappa_d` in upstream's RegisterOptions).
    pub kappa_d: Number,
    /// Correction size for very small slacks (`slack_move` option,
    /// default `mach_eps^{3/4}`). Drives `calculate_safe_slack`'s
    /// upper cap on the moved bound — port of upstream's `slack_move_`
    /// (`IpIpoptCalculatedQuantities.cpp:525`).
    pub slack_move: Number,

    // Per-iterate caches for the hot accessors used by the KKT solver
    // dependency-tag check. Without these the PdFullSpaceSolver sees a
    // fresh tag on every solve (each `curr_slack_*` / `curr_sigma_*`
    // allocates a new vector with a fresh `TaggedCell`), which forces
    // an MA57 refactor on every SOC step even though the matrix data
    // is unchanged. Caches are keyed on the input iterate-vector tag
    // and survive across calls but are naturally invalidated when the
    // outer iterate advances (curr.x bump).
    curr_slack_x_l_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_slack_x_u_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_slack_s_l_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_slack_s_u_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_sigma_x_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_sigma_s_cache: RefCell<Cache<Rc<dyn Vector>>>,
    // gh #812. Upstream caches both Lagrangian gradients
    // (`IpIpoptCalculatedQuantities.cpp`: `curr_grad_lag_x_cache_`
    // with `GetCachedResult5Dep(x, y_c, y_d, z_L, z_U)`,
    // `curr_grad_lag_s_cache_` with `GetCachedResult3Dep(y_d, v_L,
    // v_U)`); the port dropped both, so `∇_x L` was reassembled from
    // scratch on every read — two transposed Jacobian products and
    // three vector sweeps. On `benchmarks/large_scale/laptime.nl` that
    // is 806 assemblies for 101 distinct iterates: seven of every
    // eight reads recompute a value the previous read already had.
    //
    // THE X KEY CARRIES `mu`, AND UPSTREAM'S DOES NOT. The five vector
    // tags are NOT a complete dependency set here, because the premise
    // they rest on — that `∇f`, `J_c` and `J_d` are functions of `x`
    // alone — is false for the NLP this CQ is built over during
    // restoration. `RestoNlp`'s objective carries the proximity term
    // `ζ/2·‖D_R(x − x_R)‖²` whose `ζ` is a function of the barrier
    // parameter, so its `∇f` moves when `mu` moves and `x` does not.
    // Keyed on the five tags alone this cache returns the pre-update
    // gradient, which is a silent, self-consistent wrong answer: the
    // solve still converges and still reports the right objective, it
    // just takes a different route there. Measured, that route is
    // worse — `scripts/sweep-fixtures.sh` moved 8 of 154 fixture-legs,
    // `pooling_rt2stp` from 295 to 627 iterations on the lbfgs leg and
    // `issue_508_infeasible_gap_1em4` from 79 to 224. With `mu` in the
    // key the sweep is byte-identical across all 154.
    //
    // `∇_s L = −y_d − P_L v_L + P_U v_U` evaluates nothing on the NLP,
    // so its three tags really are complete and it keeps upstream's
    // key unchanged.
    curr_grad_lag_x_cache: RefCell<Cache<Rc<dyn Vector>>>,
    curr_grad_lag_s_cache: RefCell<Cache<Rc<dyn Vector>>>,
}

/// Helper: convert `Box<dyn Vector>` to `Rc<dyn Vector>`. Cheap; the
/// box is unwrapped without copying.
fn rc_from(v: Box<dyn Vector>) -> Rc<dyn Vector> {
    Rc::from(v)
}

/// Max-norm of `v` after dividing each entry by its per-row scale factor
/// (`max_i |v_i / scale_i|`). `scale == None` means "no row scaling" and
/// returns the plain `v.amax()`; a zero factor for an entry is treated as
/// the identity (no divide) so a degenerate scale never yields infinities.
/// Falls back to `v.amax()` for a non-dense backing — POUNCE is dense-only,
/// so that branch is defensive.
///
/// Public because `pounce-restoration`'s locally-infeasible gates compare a
/// constraint violation against absolute floors (`1e-4` / `1e-3`) and so must
/// measure it in the same user-facing units this produces — see
/// `resto_inner_solver::eval_orig_inf_pr_at_inner_curr`. One definition, so
/// the two call sites cannot drift.
pub fn unscaled_block_amax(v: &dyn Vector, scale: Option<&[Number]>) -> Number {
    let Some(s) = scale else {
        return v.amax();
    };
    match v.as_any().downcast_ref::<DenseVector>() {
        Some(d) => d
            .values()
            .iter()
            .zip(s.iter())
            .map(|(&x, &f)| if f == 0.0 { x.abs() } else { (x / f).abs() })
            .fold(0.0, Number::max),
        None => v.amax(),
    }
}

/// `‖v‖_∞` over the components that clear their own entry of `floor`;
/// components at or below it contribute `0`. Used by
/// [`IpoptCalculatedQuantities::curr_primal_infeasibility_above_noise`], which
/// documents what the floor means.
///
/// A vector that is not dense, is uninitialized, or whose length disagrees
/// with `floor` falls back to the plain `‖v‖_∞` — no floor can be attributed
/// component-wise, and over-reporting the residual is the safe direction.
fn amax_above_floor(v: &dyn Vector, floor: &[Number]) -> Number {
    let Some(d) = v.as_any().downcast_ref::<DenseVector>() else {
        return v.amax();
    };
    if !d.is_initialized() {
        return v.amax();
    }
    let values = d.expanded_values();
    if values.len() != floor.len() {
        return v.amax();
    }
    values
        .iter()
        .zip(floor.iter())
        .map(|(&x, &f)| if x.abs() > f { x.abs() } else { 0.0 })
        .fold(0.0, Number::max)
}

/// Result of [`IpoptCalculatedQuantities::adjusted_trial_bounds`]: the
/// new `x_L / x_U / d_L / d_U` to install on the NLP when one or more
/// trial slacks were corrected by the safe-slack mechanism.
pub struct AdjustedBounds {
    /// Total number of slack components corrected across all four blocks.
    pub adjusted: usize,
    pub x_l: Box<dyn Vector>,
    pub x_u: Box<dyn Vector>,
    pub d_l: Box<dyn Vector>,
    pub d_u: Box<dyn Vector>,
}

impl IpoptCalculatedQuantities {
    pub fn new(data: IpoptDataHandle, nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        Self {
            data,
            nlp,
            s_max: 100.0,
            kappa_d: 1e-5,
            slack_move: f64::EPSILON.powf(0.75),
            curr_slack_x_l_cache: RefCell::new(Cache::new(1)),
            curr_slack_x_u_cache: RefCell::new(Cache::new(1)),
            curr_slack_s_l_cache: RefCell::new(Cache::new(1)),
            curr_slack_s_u_cache: RefCell::new(Cache::new(1)),
            curr_sigma_x_cache: RefCell::new(Cache::new(1)),
            curr_sigma_s_cache: RefCell::new(Cache::new(1)),
            curr_grad_lag_x_cache: RefCell::new(Cache::new(1)),
            curr_grad_lag_s_cache: RefCell::new(Cache::new(1)),
        }
    }

    pub fn data(&self) -> &IpoptDataHandle {
        &self.data
    }

    pub fn nlp(&self) -> &Rc<RefCell<dyn IpoptNlp>> {
        &self.nlp
    }

    pub(crate) fn curr_iv(&self) -> IteratesVector {
        let Some(iv) = self.data.borrow().curr.as_ref().cloned() else {
            unreachable!("IpoptCalculatedQuantities: curr iterate not set");
        };
        iv
    }

    fn trial_iv(&self) -> IteratesVector {
        let Some(iv) = self.data.borrow().trial.as_ref().cloned() else {
            unreachable!("IpoptCalculatedQuantities: trial iterate not set");
        };
        iv
    }

    // --------------------------------------------------------------
    // Slacks: s_L = P_L^T x - x_L,  s_U = x_U - P_U^T x.
    // Mirror of `CalcSlack_L` / `CalcSlack_U`
    // (`IpIpoptCalculatedQuantities.cpp:238-266`).
    // --------------------------------------------------------------

    fn calc_slack_l_box(p: &dyn Matrix, x: &dyn Vector, x_bound: &dyn Vector) -> Box<dyn Vector> {
        let mut result = x_bound.make_new();
        result.copy(x_bound);
        // result = -1*result + 1*P^T x  ⇒  P^T x - x_bound.
        p.trans_mult_vector(1.0, x, -1.0, &mut *result);
        result
    }

    fn calc_slack_u_box(p: &dyn Matrix, x: &dyn Vector, x_bound: &dyn Vector) -> Box<dyn Vector> {
        let mut result = x_bound.make_new();
        result.copy(x_bound);
        // result = 1*result + (-1)*P^T x  ⇒  x_bound - P^T x.
        p.trans_mult_vector(-1.0, x, 1.0, &mut *result);
        result
    }

    /// Floor a freshly computed slack against machine precision and,
    /// where it falls below `eps*min(1,mu)`, raise it to a representable
    /// positive value, returning the number of corrected components.
    /// Faithful port of `IpoptCalculatedQuantities::CalculateSafeSlack`
    /// (`IpIpoptCalculatedQuantities.cpp:455-537`): the corrected slack
    /// is `min(max(mu/multiplier, s_min), slack_move*max(1,|bound|)+slack)`.
    /// `multiplier` and `mu` are taken from the *current* iterate, exactly
    /// as upstream does even for trial slacks.
    ///
    /// Deviates from upstream in one place: `s_min` also carries a
    /// representability floor `max_i z_i / (f64::MAX/4)` so that the
    /// `Σ = z/s` this slack feeds stays inside the double range (gh#655).
    /// See the comments at the two sites below.
    fn calculate_safe_slack(
        &self,
        slack: &mut dyn Vector,
        bound: &dyn Vector,
        multiplier: &dyn Vector,
        mu: Number,
    ) -> usize {
        if slack.dim() == 0 {
            return 0;
        }
        let min_slack = slack.min();
        // s_min = eps * min(1, mu); if mu drove it to 0, keep it strictly
        // positive (upstream #212) so the strict `slack < s_min` test and
        // the barrier term stay well-defined.
        let mut s_min = f64::EPSILON * mu.min(1.0);
        if s_min == 0.0 {
            s_min = f64::MIN_POSITIVE;
        }
        // gh#655: `eps*min(1,mu)` floors the *barrier* term, and nothing in it
        // mentions the multiplier — so a slack can clear it and still be small
        // enough against its own `z` that `Σ = z/s` leaves the double range.
        // At `mu = 9.1e-308` the threshold is `2.0e-323`; a slack of
        // `2.0e-308` sails past it untouched, and `z = 4.5` over that slack is
        // `2.2e308`, i.e. `inf` on the KKT diagonal under a reported
        // `SolveSucceeded`. `f64::MIN_POSITIVE` is not the fix either: the
        // quantity that has to stay finite is `z/s`, so the floor is
        // `s >= z/f64::MAX` (with `SIGMA_OVERFLOW_HEADROOM` of margin), not
        // the smallest representable positive double. Divide before
        // multiplying so a `z` near `f64::MAX` cannot overflow the floor
        // itself; a non-finite `z` leaves the floor alone and is caught
        // downstream by the iterate finiteness checks.
        //
        // Taken over `max_i z_i` rather than componentwise so one scalar also
        // serves the `min_slack >= s_min` trigger above. That is conservative
        // in the harmless direction: it can only raise a flagged slack
        // further, and raising a slack only lowers `Σ`.
        let sigma_floor = multiplier.amax() / (f64::MAX / SIGMA_OVERFLOW_HEADROOM);
        if sigma_floor.is_finite() && sigma_floor > s_min {
            s_min = sigma_floor;
        }
        if min_slack >= s_min {
            return 0;
        }

        // t = sign(slack - s_min); then collapse to 1 where slack < s_min,
        // 0 elsewhere.
        let mut t = slack.make_new();
        t.copy(&*slack);
        t.add_scalar(-s_min);
        t.element_wise_sgn();
        let mut zero_vec = t.make_new();
        zero_vec.set(0.0);
        t.element_wise_min(&*zero_vec); // -1 if slack < s_min, else 0
        t.scal(-1.0); //  1 if slack < s_min, else 0
        let retval = t.asum().round() as usize;

        // Clamp the raw slack to be non-negative before forming the target
        // (upstream's AW fix for negative slacks producing 0).
        slack.element_wise_max(&*zero_vec);

        // t2 = max(mu/multiplier, s_min) - slack.
        let mut t2 = t.make_new();
        let mut s_min_vec = t2.make_new();
        s_min_vec.set(s_min);
        if mu != 0.0 {
            // mu/0 → +inf here, intentionally capped by t_max below.
            t2.set(mu);
            t2.element_wise_divide(multiplier);
            t2.element_wise_max(&*s_min_vec);
        } else {
            // mu == 0: max(0/multiplier, s_min) is s_min everywhere, but a 0/0
            // (zero multiplier at μ=0) would seed the slack target with NaN and
            // poison the bound move — pin straight to s_min instead.
            t2.copy(&*s_min_vec);
        }
        t2.axpy(-1.0, &*slack);

        // t = max(mu/multiplier, s_min) where flagged, else slack.
        t.element_wise_select(&*t2);
        t.axpy(1.0, &*slack);

        // t_max = slack_move*max(1,|bound|) + slack.
        let mut t_max = t2; // reuse buffer
        t_max.set(1.0);
        let mut abs_bound = bound.make_new();
        abs_bound.copy(bound);
        abs_bound.element_wise_abs();
        t_max.element_wise_max(&*abs_bound);
        // t_max = 1.0*slack + slack_move*t_max.
        t_max.add_one_vector(1.0, &*slack, self.slack_move);

        // new slack = min(target, t_max) where flagged, else slack.
        t.element_wise_min(&*t_max);
        // gh#655, second half: re-apply the floor *after* the bound-move cap.
        // `t_max` bounds how far a bound may be nudged, which is a policy the
        // user sets; a finite `z/s` is not one. The cap sits below `s_min`
        // only when `slack_move*max(1,|bound|)` does — with the default
        // `slack_move` that needs `max_i z_i` past `6e295`, and `slack_move = 0`
        // (the "never move a bound" setting) makes it exact — but where it
        // does, the min above would hand back the overflowing slack it was
        // called to repair. Components that were not flagged are `>= s_min`
        // already, so this is a no-op for them.
        t.element_wise_max(&*s_min_vec);
        slack.copy(&*t);
        retval
    }

    /// `calc_slack_l` followed by `calculate_safe_slack`, returning the
    /// (floored) slack plus the number of corrected components. The
    /// multiplier and `mu` come from the current iterate.
    fn safe_slack_l(
        &self,
        p: &dyn Matrix,
        x: &dyn Vector,
        bound: &dyn Vector,
        multiplier: &dyn Vector,
    ) -> (Rc<dyn Vector>, usize) {
        let mu = self.data.borrow().curr_mu;
        let mut result = Self::calc_slack_l_box(p, x, bound);
        let n = self.calculate_safe_slack(&mut *result, bound, multiplier, mu);
        (rc_from(result), n)
    }

    fn safe_slack_u(
        &self,
        p: &dyn Matrix,
        x: &dyn Vector,
        bound: &dyn Vector,
        multiplier: &dyn Vector,
    ) -> (Rc<dyn Vector>, usize) {
        let mu = self.data.borrow().curr_mu;
        let mut result = Self::calc_slack_u_box(p, x, bound);
        let n = self.calculate_safe_slack(&mut *result, bound, multiplier, mu);
        (rc_from(result), n)
    }

    pub fn curr_slack_x_l(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_slack_x_l_cache.borrow();
            if let Some(v) = cache.get(&[iv.x.as_tagged()], &[]) {
                return v;
            }
        }
        let nlp = self.nlp.borrow();
        let (v, _) = self.safe_slack_l(&*nlp.px_l(), &*iv.x, nlp.x_l(), &*iv.z_l);
        self.curr_slack_x_l_cache
            .borrow_mut()
            .add(v.clone(), &[iv.x.as_tagged()], &[]);
        v
    }

    pub fn curr_slack_x_u(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_slack_x_u_cache.borrow();
            if let Some(v) = cache.get(&[iv.x.as_tagged()], &[]) {
                return v;
            }
        }
        let nlp = self.nlp.borrow();
        let (v, _) = self.safe_slack_u(&*nlp.px_u(), &*iv.x, nlp.x_u(), &*iv.z_u);
        self.curr_slack_x_u_cache
            .borrow_mut()
            .add(v.clone(), &[iv.x.as_tagged()], &[]);
        v
    }

    pub fn curr_slack_s_l(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_slack_s_l_cache.borrow();
            if let Some(v) = cache.get(&[iv.s.as_tagged()], &[]) {
                return v;
            }
        }
        let nlp = self.nlp.borrow();
        let (v, _) = self.safe_slack_l(&*nlp.pd_l(), &*iv.s, nlp.d_l(), &*iv.v_l);
        self.curr_slack_s_l_cache
            .borrow_mut()
            .add(v.clone(), &[iv.s.as_tagged()], &[]);
        v
    }

    pub fn curr_slack_s_u(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_slack_s_u_cache.borrow();
            if let Some(v) = cache.get(&[iv.s.as_tagged()], &[]) {
                return v;
            }
        }
        let nlp = self.nlp.borrow();
        let (v, _) = self.safe_slack_u(&*nlp.pd_u(), &*iv.s, nlp.d_u(), &*iv.v_u);
        self.curr_slack_s_u_cache
            .borrow_mut()
            .add(v.clone(), &[iv.s.as_tagged()], &[]);
        v
    }

    pub fn trial_slack_x_l(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mult = self.curr_iv();
        let nlp = self.nlp.borrow();
        self.safe_slack_l(&*nlp.px_l(), &*iv.x, nlp.x_l(), &*mult.z_l)
            .0
    }

    pub fn trial_slack_x_u(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mult = self.curr_iv();
        let nlp = self.nlp.borrow();
        self.safe_slack_u(&*nlp.px_u(), &*iv.x, nlp.x_u(), &*mult.z_u)
            .0
    }

    pub fn trial_slack_s_l(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mult = self.curr_iv();
        let nlp = self.nlp.borrow();
        self.safe_slack_l(&*nlp.pd_l(), &*iv.s, nlp.d_l(), &*mult.v_l)
            .0
    }

    pub fn trial_slack_s_u(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mult = self.curr_iv();
        let nlp = self.nlp.borrow();
        self.safe_slack_u(&*nlp.pd_u(), &*iv.s, nlp.d_u(), &*mult.v_u)
            .0
    }

    /// Compute the four trial slacks with safe-slack flooring and, if any
    /// component was corrected, the adjusted variable bounds that make the
    /// trial slacks exactly representable. Port of the bound-adjustment
    /// block in `IpoptAlgorithm::AcceptTrialPoint`
    /// (`IpIpoptAlg.cpp:664-706`): `new_x_L = Px_L^T x - safe_slack_x_L`,
    /// `new_x_U = Px_U^T x + safe_slack_x_U`, likewise for `s`/`d`.
    /// Returns `None` when no slack needed correcting.
    pub fn adjusted_trial_bounds(&self) -> Option<AdjustedBounds> {
        let iv = self.trial_iv();
        let mult = self.curr_iv();
        let nlp = self.nlp.borrow();

        let (s_x_l, n_x_l) = self.safe_slack_l(&*nlp.px_l(), &*iv.x, nlp.x_l(), &*mult.z_l);
        let (s_x_u, n_x_u) = self.safe_slack_u(&*nlp.px_u(), &*iv.x, nlp.x_u(), &*mult.z_u);
        let (s_s_l, n_s_l) = self.safe_slack_l(&*nlp.pd_l(), &*iv.s, nlp.d_l(), &*mult.v_l);
        let (s_s_u, n_s_u) = self.safe_slack_u(&*nlp.pd_u(), &*iv.s, nlp.d_u(), &*mult.v_u);

        let adjusted = n_x_l + n_x_u + n_s_l + n_s_u;
        if adjusted == 0 {
            return None;
        }

        // new_x_L = Px_L^T x - safe_slack_x_L
        let mut new_x_l = nlp.x_l().make_new();
        nlp.px_l()
            .trans_mult_vector(1.0, &*iv.x, 0.0, &mut *new_x_l);
        new_x_l.axpy(-1.0, &*s_x_l);
        // new_x_U = Px_U^T x + safe_slack_x_U
        let mut new_x_u = nlp.x_u().make_new();
        nlp.px_u()
            .trans_mult_vector(1.0, &*iv.x, 0.0, &mut *new_x_u);
        new_x_u.axpy(1.0, &*s_x_u);
        // new_d_L = Pd_L^T s - safe_slack_s_L
        let mut new_d_l = nlp.d_l().make_new();
        nlp.pd_l()
            .trans_mult_vector(1.0, &*iv.s, 0.0, &mut *new_d_l);
        new_d_l.axpy(-1.0, &*s_s_l);
        // new_d_U = Pd_U^T s + safe_slack_s_U
        let mut new_d_u = nlp.d_u().make_new();
        nlp.pd_u()
            .trans_mult_vector(1.0, &*iv.s, 0.0, &mut *new_d_u);
        new_d_u.axpy(1.0, &*s_s_u);

        Some(AdjustedBounds {
            adjusted,
            x_l: new_x_l,
            x_u: new_x_u,
            d_l: new_d_l,
            d_u: new_d_u,
        })
    }

    // --------------------------------------------------------------
    // NLP function evaluations.
    // --------------------------------------------------------------

    pub fn curr_grad_f(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut g = iv.x.make_new();
        nlp.eval_grad_f(&*iv.x, &mut *g);
        rc_from(g)
    }

    pub fn trial_grad_f(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut g = iv.x.make_new();
        nlp.eval_grad_f(&*iv.x, &mut *g);
        rc_from(g)
    }

    pub fn curr_c(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let m = self.nlp.borrow().m_eq();
        let mut nlp = self.nlp.borrow_mut();
        let mut c = iv.y_c.make_new();
        debug_assert_eq!(c.dim(), m);
        nlp.eval_c(&*iv.x, &mut *c);
        rc_from(c)
    }

    pub fn trial_c(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut c = iv.y_c.make_new();
        nlp.eval_c(&*iv.x, &mut *c);
        rc_from(c)
    }

    pub fn curr_d(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut d = iv.s.make_new();
        nlp.eval_d(&*iv.x, &mut *d);
        rc_from(d)
    }

    pub fn trial_d(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut d = iv.s.make_new();
        nlp.eval_d(&*iv.x, &mut *d);
        rc_from(d)
    }

    pub fn curr_jac_c(&self) -> Rc<dyn Matrix> {
        let iv = self.curr_iv();
        self.nlp.borrow_mut().eval_jac_c(&*iv.x)
    }

    pub fn curr_jac_d(&self) -> Rc<dyn Matrix> {
        let iv = self.curr_iv();
        self.nlp.borrow_mut().eval_jac_d(&*iv.x)
    }

    pub fn curr_exact_hessian(&self) -> Rc<dyn SymMatrix> {
        let iv = self.curr_iv();
        self.nlp
            .borrow_mut()
            .eval_h(&*iv.x, 1.0, &*iv.y_c, &*iv.y_d)
    }

    /// `curr_d - s` — port of `IpIpoptCalculatedQuantities.cpp:1185-1206`.
    pub fn curr_d_minus_s(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let d = self.curr_d();
        let mut tmp = iv.s.make_new();
        // tmp = 0*tmp + 1*d + (-1)*s
        tmp.add_two_vectors(1.0, &*d, -1.0, &*iv.s, 0.0);
        rc_from(tmp)
    }

    pub fn trial_d_minus_s(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let d = self.trial_d();
        let mut tmp = iv.s.make_new();
        tmp.add_two_vectors(1.0, &*d, -1.0, &*iv.s, 0.0);
        rc_from(tmp)
    }

    /// `J_c^T y_c` — for a generic `vec` argument
    /// (`IpIpoptCalculatedQuantities.cpp:1373-1404`).
    pub fn curr_jac_c_t_times_vec(&self, vec: &dyn Vector) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let jac_c = self.curr_jac_c();
        let mut tmp = iv.x.make_new();
        jac_c.trans_mult_vector(1.0, vec, 0.0, &mut *tmp);
        rc_from(tmp)
    }

    /// `J_d^T y_d` for arbitrary `vec`.
    pub fn curr_jac_d_t_times_vec(&self, vec: &dyn Vector) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let jac_d = self.curr_jac_d();
        let mut tmp = iv.x.make_new();
        jac_d.trans_mult_vector(1.0, vec, 0.0, &mut *tmp);
        rc_from(tmp)
    }

    pub fn curr_jac_c_t_times_curr_y_c(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        self.curr_jac_c_t_times_vec(&*iv.y_c)
    }

    pub fn curr_jac_d_t_times_curr_y_d(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        self.curr_jac_d_t_times_vec(&*iv.y_d)
    }

    /// `J_c v` — `IpIpoptCalculatedQuantities.cpp:1303-1321`.
    pub fn curr_jac_c_times_vec(&self, vec: &dyn Vector) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let jac_c = self.curr_jac_c();
        let mut tmp = iv.y_c.make_new();
        jac_c.mult_vector(1.0, vec, 0.0, &mut *tmp);
        rc_from(tmp)
    }

    /// `J_d v` — `IpIpoptCalculatedQuantities.cpp:1323-1343`.
    pub fn curr_jac_d_times_vec(&self, vec: &dyn Vector) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let jac_d = self.curr_jac_d();
        let mut tmp = iv.s.make_new();
        jac_d.mult_vector(1.0, vec, 0.0, &mut *tmp);
        rc_from(tmp)
    }

    // --------------------------------------------------------------
    // Lagrangian gradients
    // --------------------------------------------------------------

    /// `∇_x L = ∇f(x) + J_c^T y_c + J_d^T y_d - P_L z_L + P_U z_U`
    /// per `IpIpoptCalculatedQuantities.cpp:1993-2030`.
    pub fn curr_grad_lag_x(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let deps: [&dyn TaggedObject; 5] = [
            iv.x.as_tagged(),
            iv.y_c.as_tagged(),
            iv.y_d.as_tagged(),
            iv.z_l.as_tagged(),
            iv.z_u.as_tagged(),
        ];
        let mu = self.data.borrow().curr_mu;
        {
            let cache = self.curr_grad_lag_x_cache.borrow();
            if let Some(v) = cache.get(&deps, &[mu]) {
                return v;
            }
        }
        let grad_f = self.curr_grad_f();
        let jc_t_y_c = self.curr_jac_c_t_times_curr_y_c();
        let jd_t_y_d = self.curr_jac_d_t_times_curr_y_d();

        let mut tmp = iv.x.make_new();
        tmp.copy(&*grad_f);
        tmp.add_two_vectors(1.0, &*jc_t_y_c, 1.0, &*jd_t_y_d, 1.0);

        let nlp = self.nlp.borrow();
        nlp.px_l().mult_vector(-1.0, &*iv.z_l, 1.0, &mut *tmp);
        nlp.px_u().mult_vector(1.0, &*iv.z_u, 1.0, &mut *tmp);
        let v = rc_from(tmp);
        self.curr_grad_lag_x_cache
            .borrow_mut()
            .add(v.clone(), &deps, &[mu]);
        v
    }

    /// `∇_s L = -y_d - P_L v_L + P_U v_U`
    /// (`IpIpoptCalculatedQuantities.cpp:2069-2098`).
    pub fn curr_grad_lag_s(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let deps: [&dyn TaggedObject; 3] =
            [iv.y_d.as_tagged(), iv.v_l.as_tagged(), iv.v_u.as_tagged()];
        {
            let cache = self.curr_grad_lag_s_cache.borrow();
            if let Some(v) = cache.get(&deps, &[]) {
                return v;
            }
        }
        let mut tmp = iv.y_d.make_new();
        let nlp = self.nlp.borrow();
        // tmp = P_U v_U
        nlp.pd_u().mult_vector(1.0, &*iv.v_u, 0.0, &mut *tmp);
        // tmp = tmp - P_L v_L
        nlp.pd_l().mult_vector(-1.0, &*iv.v_l, 1.0, &mut *tmp);
        // tmp = tmp - y_d
        tmp.axpy(-1.0, &*iv.y_d);
        drop(nlp);
        let v = rc_from(tmp);
        self.curr_grad_lag_s_cache
            .borrow_mut()
            .add(v.clone(), &deps, &[]);
        v
    }

    // --------------------------------------------------------------
    // Complementarity (slack ⊙ multiplier)
    // --------------------------------------------------------------

    fn calc_compl(slack: &dyn Vector, mult: &dyn Vector) -> Rc<dyn Vector> {
        let mut result = slack.make_new();
        result.copy(slack);
        result.element_wise_multiply(mult);
        rc_from(result)
    }

    pub fn curr_compl_x_l(&self) -> Rc<dyn Vector> {
        let slack = self.curr_slack_x_l();
        let z_l = self.curr_iv().z_l;
        Self::calc_compl(&*slack, &*z_l)
    }

    pub fn curr_compl_x_u(&self) -> Rc<dyn Vector> {
        let slack = self.curr_slack_x_u();
        let z_u = self.curr_iv().z_u;
        Self::calc_compl(&*slack, &*z_u)
    }

    pub fn curr_compl_s_l(&self) -> Rc<dyn Vector> {
        let slack = self.curr_slack_s_l();
        let v_l = self.curr_iv().v_l;
        Self::calc_compl(&*slack, &*v_l)
    }

    pub fn curr_compl_s_u(&self) -> Rc<dyn Vector> {
        let slack = self.curr_slack_s_u();
        let v_u = self.curr_iv().v_u;
        Self::calc_compl(&*slack, &*v_u)
    }

    /// `s_L .* z_L - mu` — relaxed complementarity used in the KKT
    /// RHS. `IpIpoptCalculatedQuantities.cpp:2406-2430`.
    pub fn curr_relaxed_compl_x_l(&self) -> Rc<dyn Vector> {
        let mu = self.data.borrow().curr_mu;
        let mut r = self.curr_compl_x_l().make_new();
        r.copy(&*self.curr_compl_x_l());
        r.add_scalar(-mu);
        rc_from(r)
    }

    pub fn curr_relaxed_compl_x_u(&self) -> Rc<dyn Vector> {
        let mu = self.data.borrow().curr_mu;
        let mut r = self.curr_compl_x_u().make_new();
        r.copy(&*self.curr_compl_x_u());
        r.add_scalar(-mu);
        rc_from(r)
    }

    pub fn curr_relaxed_compl_s_l(&self) -> Rc<dyn Vector> {
        let mu = self.data.borrow().curr_mu;
        let mut r = self.curr_compl_s_l().make_new();
        r.copy(&*self.curr_compl_s_l());
        r.add_scalar(-mu);
        rc_from(r)
    }

    pub fn curr_relaxed_compl_s_u(&self) -> Rc<dyn Vector> {
        let mu = self.data.borrow().curr_mu;
        let mut r = self.curr_compl_s_u().make_new();
        r.copy(&*self.curr_compl_s_u());
        r.add_scalar(-mu);
        rc_from(r)
    }

    // --------------------------------------------------------------
    // Σ_x / Σ_s (barrier-Hessian diagonals fed to the augmented system)
    // `IpIpoptCalculatedQuantities.cpp:3501-3551`.
    //
    //   Σ_x = P_L · diag(z_L / s_L) · P_L^T + P_U · diag(z_U / s_U) · P_U^T
    //   Σ_s = P_L · diag(v_L / s_L) · P_L^T + P_U · diag(v_U / s_U) · P_U^T
    // --------------------------------------------------------------

    pub fn curr_sigma_x(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_sigma_x_cache.borrow();
            if let Some(v) = cache.get(
                &[iv.x.as_tagged(), iv.z_l.as_tagged(), iv.z_u.as_tagged()],
                &[],
            ) {
                return v;
            }
        }
        let slack_l = self.curr_slack_x_l();
        let slack_u = self.curr_slack_x_u();

        let mut sigma = iv.x.make_new();
        sigma.set(0.0);

        let nlp = self.nlp.borrow();
        nlp.px_l()
            .add_m_sinv_z(1.0, &*slack_l, &*iv.z_l, &mut *sigma);
        nlp.px_u()
            .add_m_sinv_z(1.0, &*slack_u, &*iv.z_u, &mut *sigma);
        let v = rc_from(sigma);
        self.curr_sigma_x_cache.borrow_mut().add(
            v.clone(),
            &[iv.x.as_tagged(), iv.z_l.as_tagged(), iv.z_u.as_tagged()],
            &[],
        );
        v
    }

    /// Slack-based symmetric scaling factors for the `s` block —
    /// `min(Pd_L · slack_s_L + Pd_U · slack_s_U, 1)`.
    ///
    /// Port of `IpSlackBasedTSymScalingMethod.cpp:ComputeSymTScalingFactors`,
    /// which builds the whole augmented-system scaling vector as
    /// `[1 (x block) | this (s block) | 1 (y_c, y_d blocks)]`. Only the
    /// `s` block is computed here; the surrounding ones are constants
    /// the scaling method writes itself.
    ///
    /// Upstream's method is an algorithm-strategy object with direct
    /// access to `IpCq()`/`IpNLP()`. pounce's `TSymScalingMethod` lives
    /// in `pounce-linsol`, which cannot see the iterate, so the quantity
    /// is computed here and pushed down before the factorization. That
    /// is why this is a CQ method rather than logic inside the scaling
    /// method itself.
    ///
    /// The cap at 1 is upstream's `slack_scale_max`. A component whose
    /// `d` row has no bound at all contributes nothing to either
    /// product and would scale that row by zero, which would wipe the
    /// row out of the factorization; the floor guards that. It cannot
    /// arise from a well-formed NLP — an inequality row has at least one
    /// bound or it would not be an inequality — but the augmented system
    /// is assembled from whatever the caller passes.
    pub fn curr_slack_based_s_scaling(&self) -> Option<Vec<Number>> {
        let iv = self.curr_iv();
        let slack_l = self.curr_slack_s_l();
        let slack_u = self.curr_slack_s_u();
        let nlp = self.nlp.borrow();

        let mut tmp = iv.s.make_new();
        nlp.pd_l().mult_vector(1.0, &*slack_l, 0.0, &mut *tmp);
        nlp.pd_u().mult_vector(1.0, &*slack_u, 1.0, &mut *tmp);

        let mut cap = iv.s.make_new();
        cap.set(1.0);
        tmp.element_wise_min(&*cap);

        // The `Vector` trait exposes no generic value read, so this
        // downcasts. `s` is dense on the main solve; the restoration
        // sub-IPM's primal is a 5-block `CompoundVector` whose `s` is a
        // different space, and slack-based scaling is not wired there.
        // Returning `None` rather than panicking means an unexpected
        // vector type costs the scaling, not the solve.
        let dense = tmp.as_any().downcast_ref::<DenseVector>()?;
        let mut out = dense.expanded_values();
        for v in out.iter_mut() {
            // Guard the zero case described above, and any NaN.
            if !(*v > 0.0) {
                *v = 1.0;
            }
        }
        Some(out)
    }

    pub fn curr_sigma_s(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        {
            let cache = self.curr_sigma_s_cache.borrow();
            if let Some(v) = cache.get(
                &[iv.s.as_tagged(), iv.v_l.as_tagged(), iv.v_u.as_tagged()],
                &[],
            ) {
                return v;
            }
        }
        let slack_l = self.curr_slack_s_l();
        let slack_u = self.curr_slack_s_u();

        let mut sigma = iv.s.make_new();
        sigma.set(0.0);

        let nlp = self.nlp.borrow();
        nlp.pd_l()
            .add_m_sinv_z(1.0, &*slack_l, &*iv.v_l, &mut *sigma);
        nlp.pd_u()
            .add_m_sinv_z(1.0, &*slack_u, &*iv.v_u, &mut *sigma);
        let v = rc_from(sigma);
        self.curr_sigma_s_cache.borrow_mut().add(
            v.clone(),
            &[iv.s.as_tagged(), iv.v_l.as_tagged(), iv.v_u.as_tagged()],
            &[],
        );
        v
    }

    // --------------------------------------------------------------
    // Objective f and barrier objective phi
    // (`IpIpoptCalculatedQuantities.cpp:CalcBarrierTerm`,
    //  lines 870-1042 in upstream).
    //
    //   phi(x,s) = f(x)
    //              − μ · [Σ ln(s_x_L) + Σ ln(s_x_U)
    //                   + Σ ln(s_s_L) + Σ ln(s_s_U)]
    //              + κ_d · μ · [s_x_L · 1_singly_x_L
    //                          + s_x_U · 1_singly_x_U
    //                          + s_s_L · 1_singly_s_L
    //                          + s_s_U · 1_singly_s_U]
    //
    // The damping piece vanishes when `kappa_d == 0` (default).
    // --------------------------------------------------------------

    pub fn curr_f(&self) -> Number {
        let iv = self.curr_iv();
        let mut nlp = self.nlp.borrow_mut();
        nlp.eval_f(&*iv.x)
    }

    /// Unscaled objective at the current iterate. `curr_f` returns the
    /// internally scaled value (`f · df_`); upstream IPOPT prints the
    /// unscaled objective in its iteration log, so this divides the
    /// scaling back out. Mirrors `IpoptCalculatedQuantities::
    /// unscaled_curr_f`. A zero factor (scaling never determined) is
    /// treated as the identity.
    pub fn unscaled_curr_f(&self) -> Number {
        let scaled = self.curr_f();
        let factor = self.nlp.borrow().obj_scaling_factor();
        if factor == 0.0 {
            scaled
        } else {
            scaled / factor
        }
    }

    /// Max-norm dual infeasibility in the **unscaled** (user-original)
    /// space. [`Self::curr_dual_infeasibility_max`] is evaluated in the
    /// internally-scaled NLP space (objective × `df`, constraints × `dc`);
    /// because POUNCE applies no variable scaling, every term of the
    /// Lagrangian gradient `∇f + Jᵀy − z` carries the same objective
    /// factor `df`, so the unscaling is a single divide by
    /// `df = obj_scaling_factor`. A zero or unit factor returns the scaled
    /// value unchanged — the common no-scaling path stays division-free.
    pub fn curr_unscaled_dual_infeasibility_max(&self) -> Number {
        let df = self.nlp.borrow().obj_scaling_factor();
        let scaled = self.curr_dual_infeasibility_max();
        // `df` is SIGNED — `obj_scaling_factor = -1` is the documented way to
        // pose a maximization — while `scaled` is a max-norm. Dividing by the
        // signed factor returned a NEGATIVE "max-norm", which then sailed under
        // every `<= tol` comparison: it disabled the gh #200 veto on
        // maximization, and defeated the unscaled residual gate added for
        // pounce#173 there as well. Magnitude is what the unscaling means.
        let df = df.abs();
        if df == 0.0 || df == 1.0 {
            scaled
        } else {
            scaled / df
        }
    }

    /// Max-norm complementarity in the **unscaled** space. Each bound block
    /// `s · z` scales uniformly by `df`: the slack's `dc`/`dd` factor and
    /// the multiplier's `df/dc` (`df/dd`) factor cancel in the product,
    /// leaving `df`. So this is the scaled max-norm divided by `df`. See
    /// [`Self::curr_unscaled_dual_infeasibility_max`].
    pub fn curr_unscaled_complementarity_max(&self) -> Number {
        let df = self.nlp.borrow().obj_scaling_factor();
        let scaled = self.curr_complementarity_max();
        // `df` is SIGNED — `obj_scaling_factor = -1` is the documented way to
        // pose a maximization — while `scaled` is a max-norm. Dividing by the
        // signed factor returned a NEGATIVE "max-norm", which then sailed under
        // every `<= tol` comparison: it disabled the gh #200 veto on
        // maximization, and defeated the unscaled residual gate added for
        // pounce#173 there as well. Magnitude is what the unscaling means.
        let df = df.abs();
        if df == 0.0 || df == 1.0 {
            scaled
        } else {
            scaled / df
        }
    }

    /// Max-norm primal infeasibility in the **unscaled** space. Unlike the
    /// dual/complementarity terms the constraint scaling is per-row
    /// (`c_scaled = dc ⊙ c_user`, `(d−s)_scaled = dd ⊙ (d−s)_user`), so each
    /// block is unscaled element-by-element before the max-norm. When no row
    /// scaling is active (`c_scale_vec`/`d_scale_vec` both `None` — the
    /// common case) this is exactly [`Self::curr_primal_infeasibility_max`].
    pub fn curr_unscaled_primal_infeasibility_max(&self) -> Number {
        let (dc, dd) = {
            let nlp = self.nlp.borrow();
            (nlp.c_scale_vec(), nlp.d_scale_vec())
        };
        if dc.is_none() && dd.is_none() {
            return self.curr_primal_infeasibility_max();
        }
        let c_max = unscaled_block_amax(&*self.curr_c(), dc.as_deref());
        let dms_max = unscaled_block_amax(&*self.curr_d_minus_s(), dd.as_deref());
        c_max.max(dms_max)
    }

    /// Max-norm constraint violation of the **original** NLP, in user units:
    /// `|c_i|` over the equality block and `max(0, d_l_i − d_i, d_i − d_u_i)`
    /// over the inequality block. This is what upstream's
    /// `inf_pr_output = original` — its *default* — prints in the `inf_pr`
    /// column, and what its end-of-run "Constraint violation" line reports.
    /// Mirrors `IpIpoptCalculatedQuantities.cpp:unscaled_curr_nlp_constraint_violation`.
    ///
    /// Deliberately **not** [`Self::curr_primal_infeasibility_max`], which is
    /// `max(‖c‖_∞, ‖d − s‖_∞)` — the violation of the *internal* slack
    /// reformulation. The two diverge whenever the slack drifts from `d(x)`:
    /// `s` is confined to `[d_l, d_u]`, so `d = s + (d − s)` with `d − s > 0`
    /// clears a lower bound however large that gap grows. On a model that is
    /// all inequalities the gap *is* the whole number — on Mittelmann's
    /// `robot_a` POUNCE reported 2.79e4 at an iterate where Ipopt reported
    /// `0.00e+00` and every original row was in fact satisfied (pounce#476).
    ///
    /// Display only. The filter's `theta`, the barrier-parameter strategies
    /// and the convergence test all stay on the internal measure — that split
    /// is upstream's, not a shortcut.
    ///
    /// Judged against the **declared** bounds where the NLP tracks them, so
    /// the `bound_relax_factor` widening cannot forgive a violation the user
    /// would still see (same reasoning as
    /// [`Self::relative_d_infeasibility_max`]).
    pub fn curr_unscaled_nlp_constraint_violation_max(&self) -> Number {
        let (dc, dd) = {
            let nlp = self.nlp.borrow();
            (nlp.c_scale_vec(), nlp.d_scale_vec())
        };
        let c_max = unscaled_block_amax(&*self.curr_c(), dc.as_deref());

        let d = self.curr_d();
        if d.dim() == 0 {
            return c_max;
        }
        let (lo, hi, mask_l, mask_u) = {
            let nlp = self.nlp.borrow();
            let (mut cl, mut cu) = (nlp.d_l().make_new(), nlp.d_u().make_new());
            match nlp.declared_d_bounds() {
                Some((dl, du)) => {
                    let (Some(cld), Some(cud)) = (
                        cl.as_any_mut().downcast_mut::<DenseVector>(),
                        cu.as_any_mut().downcast_mut::<DenseVector>(),
                    ) else {
                        return c_max;
                    };
                    cld.set_values(&dl);
                    cud.set_values(&du);
                }
                None => {
                    cl.copy(nlp.d_l());
                    cu.copy(nlp.d_u());
                }
            }
            let mut lo = d.make_new();
            lo.set(0.0);
            nlp.pd_l().mult_vector(1.0, &*cl, 0.0, &mut *lo);
            let mut hi = d.make_new();
            hi.set(0.0);
            nlp.pd_u().mult_vector(1.0, &*cu, 0.0, &mut *hi);
            // A projected 0 is ambiguous — "no bound on this side" and "a
            // declared zero bound" both read 0 — so project an all-ones
            // vector through the same expansion to get presence masks.
            let mut ones_l = nlp.d_l().make_new();
            ones_l.set(1.0);
            let mut mask_l = d.make_new();
            mask_l.set(0.0);
            nlp.pd_l().mult_vector(1.0, &*ones_l, 0.0, &mut *mask_l);
            let mut ones_u = nlp.d_u().make_new();
            ones_u.set(1.0);
            let mut mask_u = d.make_new();
            mask_u.set(0.0);
            nlp.pd_u().mult_vector(1.0, &*ones_u, 0.0, &mut *mask_u);
            (lo, hi, mask_l, mask_u)
        };
        let (Some(dv), Some(lo), Some(hi), Some(ml), Some(mu)) = (
            d.as_any().downcast_ref::<DenseVector>(),
            lo.as_any().downcast_ref::<DenseVector>(),
            hi.as_any().downcast_ref::<DenseVector>(),
            mask_l.as_any().downcast_ref::<DenseVector>(),
            mask_u.as_any().downcast_ref::<DenseVector>(),
        ) else {
            return c_max;
        };
        if !(dv.is_initialized()
            && lo.is_initialized()
            && hi.is_initialized()
            && ml.is_initialized()
            && mu.is_initialized())
        {
            return c_max;
        }
        let (dv, lov, hiv, mlv, muv) = (
            dv.expanded_values(),
            lo.expanded_values(),
            hi.expanded_values(),
            ml.expanded_values(),
            mu.expanded_values(),
        );
        let mut worst = c_max;
        for i in 0..dv.len() {
            let mut viol = 0.0_f64;
            if mlv[i] > 0.5 {
                viol = viol.max(lov[i] - dv[i]);
            }
            if muv[i] > 0.5 {
                viol = viol.max(dv[i] - hiv[i]);
            }
            if viol <= 0.0 || !viol.is_finite() {
                continue;
            }
            // Row scaling is per-row (`d_scaled = dd ⊙ d_user`), so the
            // violation unscales by the same factor. A zero factor is treated
            // as the identity, matching `unscaled_block_amax`.
            let viol = match dd.as_deref() {
                Some(s) if s[i] != 0.0 => viol / s[i],
                _ => viol,
            };
            worst = worst.max(viol);
        }
        worst
    }

    /// The primal violation of the model **as declared** — before the
    /// `bound_relax_factor` widening `OrigIpoptNlp::relax_bounds` applied.
    ///
    /// [`Self::curr_unscaled_nlp_constraint_violation_max`] already judges the
    /// `c` and `d` blocks against the declared bounds. What nothing else
    /// measured is the **variable box**, which the widening moves too; that
    /// term comes from `Nlp::declared_box_violation`, which owns the lift out
    /// of the compressed bound spaces.
    ///
    /// This is not what any gate reads and must not become one. The barrier
    /// genuinely solves the widened model — a feasible-iterate log-barrier
    /// needs `x` strictly inside its bounds — and `final_constr_viol` reports
    /// the internal slack measure the convergence test uses. The point of this
    /// number is that the two can differ by orders and a caller could not
    /// previously see it: on netlib `wood1p` this arm reports `1.71e-14` at a
    /// point `7.96e-09` outside the declared rows and `9.84e-09` outside the
    /// declared box, and returns an objective `4.4e-05` from the optimum
    /// HiGHS reports.
    pub fn curr_declared_primal_violation_max(&self) -> Number {
        let rows = self.curr_unscaled_nlp_constraint_violation_max();
        rows.max(self.curr_declared_box_violation_max())
    }

    /// How far the current iterate sits outside the **declared** variable box
    /// — the box the user wrote, before the `bound_relax_factor` widening.
    ///
    /// The box half of [`Self::curr_declared_primal_violation_max`] on its
    /// own, because that is the quantity Ipopt's summary block reports as
    /// `Variable bound violation` and the aggregate cannot answer it: maxed
    /// together with the row term, a box violation and a row violation are
    /// indistinguishable. POUNCE printed a hardcoded `0.0` on that line until
    /// this existed, which is the correct value on an unwidened solve and a
    /// false reassurance on exactly the class where the line earns its keep —
    /// the toy `min 1e8·x + x²/2 s.t. x ≥ 0` returns `x = −1e-8` at
    /// `bound_relax_factor = 1e-8`, an objective of `−1` for a quantity that
    /// cannot go below `0`, under `Optimal Solution Found`.
    ///
    /// `0.0` when the NLP does not track a declared box
    /// (`Nlp::declared_box_violation` returns `None`). That is not a fallback
    /// stand-in: `relax_bounds` snapshots the box on every solve of this arm
    /// *whether or not* it then widens it, so an untracked box means no
    /// widening pass ran at all, the declared box and the live one are the
    /// same object, and a feasible-iterate log-barrier keeps `x` strictly
    /// inside the live one by construction.
    ///
    /// Measured on the iterate, not on the finalized `x`: with
    /// `honor_original_bounds` the reported point is projected back into the
    /// declared box, so reading it there would report `0` for every solve and
    /// say nothing about the point the objective above it was evaluated at.
    /// Upstream reports the same pre-projection quantity.
    pub fn curr_declared_box_violation_max(&self) -> Number {
        let iv = self.curr_iv();
        let nlp = self.nlp.borrow();
        nlp.declared_box_violation(&*iv.x).unwrap_or(0.0)
    }

    /// Largest primal infeasibility of a constraint row **relative to that
    /// row's own magnitude** — `|c_i| / |b_i|` over the equality block and
    /// `dist(d_i, [d_l_i, d_u_i]) / max(|d_l_i|, |d_u_i|)` over the
    /// inequality block, whichever is worse.
    ///
    /// See [`Self::relative_d_infeasibility_max`] and
    /// [`Self::relative_c_infeasibility_max`] for the two blocks; both use
    /// the row's **declared** magnitude, never a live or relaxed stand-in,
    /// and both abstain (contribute nothing) on a row that has no declared
    /// magnitude to be relative to.
    pub fn curr_relative_primal_infeasibility_max(&self) -> Number {
        self.relative_d_infeasibility_max()
            .max(self.relative_c_infeasibility_max())
    }

    /// The equality-block half of
    /// [`Self::curr_relative_primal_infeasibility_max`]: `max_i |c_i| / |b_i|`,
    /// where `b_i` is the row's declared right-hand side
    /// ([`IpoptNlp::declared_c_rhs`]).
    ///
    /// POUNCE folds `g_i(x) == b_i` into `c_i(x) = 0`, so `|c_i|` *is* the
    /// violation and by itself carries no magnitude to be judged against —
    /// which is why every runtime feasibility decision on an equality row was
    /// an absolute one, and why down-scaling such a row shrank `|c_i|` under
    /// the absolute tolerance and flipped a true infeasibility verdict to
    /// `Solve_Succeeded` (gh #390, residual of #387). Dividing by the pre-fold
    /// RHS restores it: `s·g(x) == s·b` has residual `s·(g(x) − b)` and RHS
    /// `s·b`, so the ratio is the same at every `s` — the point.
    ///
    /// Both numerator and denominator are taken in the internally-scaled
    /// space, so the solver's own row scaling `dc_i` cancels too.
    ///
    /// A **homogeneous** row (`b_i = 0`) contributes nothing. It has no
    /// declared magnitude to be relative to, and needs none: `s·g(x) == 0` is
    /// the same row at every `s`, so the absolute test is already invariant
    /// there. Dividing by zero — or by a fabricated floor — would turn
    /// float-noise residuals into 100% "violations" on the single most common
    /// equality row there is. Non-finite entries likewise contribute nothing:
    /// an unjudgeable row must not fabricate a relative verdict. When the NLP
    /// does not track the RHS at all (`declared_c_rhs` is `None` — e.g. the
    /// restoration NLP, whose `c` block is not the user's rows), the whole
    /// block abstains.
    ///
    /// "Homogeneous" is judged **numerically**, not by `b_i == 0` exactly: a
    /// row abstains once `|b_i|` sinks under its own
    /// [noise floor](`Self::row_noise_floor`). An exact-zero test is the
    /// right idea measured with the wrong instrument — a converter that emits
    /// `2^-53` where the model says `0` (Maros-Mészáros `QSC*`/`QSCFXM*`, and
    /// every one of the 15 problems in gh #446, carry equality rows with an
    /// RHS at `1e-17`–`1e-16`) declares a magnitude that is pure rounding
    /// residue — a target no iterate could be positioned finely enough to hit.
    /// `|c_i|` cannot be driven below the same floor either, so the
    /// ratio was noise over noise: QSCSD1 read 81× violated at a converged KKT
    /// point whose absolute violation was `9.2e-15`, which vetoed its success
    /// certificate and then armed the rapid-infeasibility pre-filter — a
    /// feasible convex QP reported `Converged to a point of local
    /// infeasibility`. Comparing against the row's own noise floor keeps the
    /// scale invariance that is the whole point of the measure (both sides
    /// carry `dc_i`), which an absolute cutoff on `|b_i|` would have thrown
    /// away.
    pub fn relative_c_infeasibility_max(&self) -> Number {
        let c = self.curr_c();
        if c.dim() == 0 {
            return 0.0;
        }
        let Some(rhs) = self.nlp.borrow().declared_c_rhs() else {
            return 0.0;
        };
        let noise = self.row_noise_floor(&*self.curr_jac_c(), &*c);
        let Some(c) = c.as_any().downcast_ref::<DenseVector>() else {
            return 0.0;
        };
        if !c.is_initialized() {
            return 0.0;
        }
        let cv = c.expanded_values();
        if cv.len() != rhs.len() {
            return 0.0;
        }
        let mut worst = 0.0_f64;
        for (i, (&ci, &bi)) in cv.iter().zip(rhs.iter()).enumerate() {
            let mag = bi.abs();
            // `0.0` when no floor could be computed, which reproduces the
            // former `mag > 0.0` gate exactly.
            let floor = noise.as_ref().map_or(0.0, |n| n[i]);
            if mag > floor && mag.is_finite() && ci.is_finite() {
                worst = worst.max(ci.abs() / mag);
            }
        }
        worst
    }

    /// Per-row noise floor of a constraint block: the finest residual the
    /// solver could drive that row to, in the same internally scaled units as
    /// the block's residual and declared bounds. `jac` is the block's Jacobian
    /// and `template` any vector in the block's space.
    ///
    /// The quantity being modelled is **how finely the solver can place `x`**,
    /// not how accurately a row evaluates. A Newton step comes from a linear
    /// solve whose backward error is norm-wise, so every component of `x` is
    /// positioned to roughly `eps · ‖x‖_∞` in absolute terms — a variable at
    /// `1e-8` inside a vector of norm `2.7` is still only resolved to
    /// `~6e-16`, not to `~2e-24`. A row responds to `x` at rate
    /// `max_j |∂g_i/∂x_j|`, so the finest residual it can be driven to is
    /// `max_j |∂g_i/∂x_j| · eps · ‖x‖_∞`, with [`ROW_NOISE_KAPPA`] covering
    /// accumulation across the row's nonzeros and conditioning slop. A
    /// declared magnitude under that is a target the solver could not hit even
    /// in exact arithmetic on the model as written.
    ///
    /// `‖x‖_∞` is global, and that is the point rather than a compromise: `x`
    /// is one vector solved for jointly, so a large variable anywhere really
    /// does coarsen the resolution of every other. The per-row alternative,
    /// the exact term sum `Σ_j |a_ij x_j|` via `|J|·|x|`, was implemented and
    /// measured, and it is strictly worse — it models the row's *evaluation*
    /// error, which is not what limits the residual. It regressed QETAMACR,
    /// QSCORPIO and QPILOTNO of gh #446's 15: QSCORPIO's row 93 has all its
    /// variables parked near a zero bound at `~1e-8`, giving a term sum of
    /// `6e-8` and a floor of `8.5e-22`, so its `−5.6e-17` of rounding residue
    /// read as real data again — while the iterate it is judging is only
    /// resolved to `~6e-16`. Do not "improve" this to the term sum without
    /// re-running those three.
    ///
    /// Scale-invariant by construction. Under a row scaling `dc_i` the
    /// Jacobian row carries `dc_i` exactly as the residual and the declared
    /// bounds do (the scaling is applied in `eval_jac_c`/`eval_jac_d`), so the
    /// floor moves with the quantities it gates and the abstention verdict is
    /// the same at every `s`.
    ///
    /// A row with an **empty** Jacobian gets an infinite floor, so it always
    /// abstains. Every variable it mentions has been fixed and substituted
    /// out, which leaves a constant row `0 = b` that no iterate can move: it
    /// is a statement about the *model*, and judging the *iterate* by it is a
    /// category error. That is presolve's question, answered up front by
    /// `presolve_infeasibility_proof` with a certificate, not a residual. The
    /// runtime measure abstaining costs no detection that matters — the
    /// absolute `constr_viol_tol` arm still sees the row, and an empty row
    /// violated by anything a caller would recognise as infeasible is orders
    /// above it. QPILOTNO is why: five variables fixed at `0` reduce row 150
    /// to `0 = −2.22e-16`, its own rounding residue, and a measure that
    /// insists the iterate is 100% in violation of it will never let any
    /// iterate succeed (gh #446).
    ///
    /// `None` — meaning "no floor", i.e. only an exactly-zero magnitude
    /// abstains — when the reference cannot be formed: `x = 0` (every term is
    /// exactly zero, so the row carries no rounding error to speak of), a
    /// non-finite iterate, or a vector type that is not dense.
    fn row_noise_floor(&self, jac: &dyn Matrix, template: &dyn Vector) -> Option<Vec<Number>> {
        let x_amax = self.curr_iv().x.amax();
        if x_amax <= 0.0 || !x_amax.is_finite() {
            return None;
        }
        let mut rows = template.make_new();
        jac.compute_row_amax(&mut *rows, true);
        let rows = rows.as_any().downcast_ref::<DenseVector>()?;
        if !rows.is_initialized() {
            return None;
        }
        Some(
            rows.expanded_values()
                .iter()
                .map(|&a| {
                    if a > 0.0 {
                        ROW_NOISE_KAPPA * Number::EPSILON * a * x_amax
                    } else {
                        Number::INFINITY
                    }
                })
                .collect(),
        )
    }

    /// [`Self::curr_primal_infeasibility_max`] — `max(‖c‖_∞, ‖d − s‖_∞)` —
    /// counting each row only where its residual rises above the finest value
    /// that residual can take in floating point (gh #528).
    ///
    /// Both residuals are *differences of quantities the row's own size*:
    /// `c_i = g_i(x) − b_i` and `d_i − s_i` with `s_i` confined to `d_i`'s
    /// bounds. A difference of doubles of magnitude `m` is quantised in units
    /// of `eps · m`, so no iterate can place either residual strictly between
    /// `0` and `eps · m_i` — it lands on an exact `0` or on a multiple of the
    /// quantum, and which of the two is arithmetic luck. Once `eps · m_i`
    /// exceeds `tol` that luck decides whether a fully converged solve gets a
    /// certificate: on gh #528's LPs (`|b| ~ 1e8`, so one ulp is `1.5e-8`
    /// against the default `tol = 1e-8`) the KKT error was pinned one ulp
    /// above the tolerance at the exact optimum, the solve kept iterating at a
    /// point it could not improve, and it exited
    /// `Search_Direction_Becomes_Too_Small` with the right answer in hand.
    ///
    /// The floor per row is the larger of two irreducible effects, both
    /// carrying [`ROW_NOISE_KAPPA`] for the same reason
    /// [`Self::row_noise_floor`] does (accumulation over the row's nonzeros
    /// and the linear solve's conditioning):
    ///
    /// * **Placing `x`** — [`Self::row_noise_floor`], `eps · ‖x‖_∞` passed
    ///   through the row at `max_j |∂g_i/∂x_j|`. A row whose Jacobian is
    ///   empty gets `INFINITY` there, meaning "abstain", which is the safe
    ///   direction for the *relative* measures that floor was written for and
    ///   the unsafe one here — silencing a constant row `0 = b` would forgive
    ///   a genuine infeasibility outright. Non-finite floors are therefore
    ///   read as `0`: such a row is judged on its residual alone.
    /// * **Forming the residual** — `eps · m_i`, with `m_i` the magnitude of
    ///   the quantities subtracted: the declared right-hand side `|b_i|` on
    ///   the equality block (the value `c_i` was formed against), and
    ///   `max(|d_i|, |s_i|)` on the inequality block. A block with no declared
    ///   magnitude to hand (the restoration NLP's `c`, whose rows are not the
    ///   user's) contributes nothing here and is left to the placement floor.
    ///
    /// A row's residual is counted in full or not at all, matching how
    /// [`Self::relative_c_infeasibility_max`] and
    /// [`Self::relative_d_infeasibility_max`] use their floors: the question
    /// is whether the row says anything, not how much of it to subtract.
    pub fn curr_primal_infeasibility_above_noise(&self, kappa: Number) -> Number {
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();

        let c_above = if c.dim() == 0 {
            0.0
        } else {
            let mag = self
                .nlp
                .borrow()
                .declared_c_rhs()
                .map(|rhs| rhs.iter().map(|b| b.abs()).collect::<Vec<_>>());
            let floor = self.primal_residual_noise_floor(
                &*self.curr_jac_c(),
                &*c,
                mag.as_deref(),
                c.dim() as usize,
                kappa,
            );
            amax_above_floor(&*c, &floor)
        };

        let d_above = if dms.dim() == 0 {
            0.0
        } else {
            let d = self.curr_d();
            let s = self.curr_iv().s;
            let mag = match (
                d.as_any().downcast_ref::<DenseVector>(),
                s.as_any().downcast_ref::<DenseVector>(),
            ) {
                (Some(d), Some(s)) if d.is_initialized() && s.is_initialized() => {
                    let (dv, sv) = (d.expanded_values(), s.expanded_values());
                    (dv.len() == sv.len()).then(|| {
                        dv.iter()
                            .zip(&sv)
                            .map(|(a, b)| a.abs().max(b.abs()))
                            .collect::<Vec<_>>()
                    })
                }
                _ => None,
            };
            let floor = self.primal_residual_noise_floor(
                &*self.curr_jac_d(),
                &*dms,
                mag.as_deref(),
                dms.dim() as usize,
                kappa,
            );
            amax_above_floor(&*dms, &floor)
        };

        c_above.max(d_above)
    }

    /// Per-row floor for [`Self::curr_primal_infeasibility_above_noise`]:
    /// `max(placement floor, ROW_NOISE_KAPPA · eps · magnitude_i)`, with a
    /// finite value on every row (`0` where nothing can be said, so that row
    /// is judged on its residual alone). See that method for the derivation.
    fn primal_residual_noise_floor(
        &self,
        jac: &dyn Matrix,
        residual: &dyn Vector,
        magnitude: Option<&[Number]>,
        dim: usize,
        kappa: Number,
    ) -> Vec<Number> {
        // `row_noise_floor` bakes in `ROW_NOISE_KAPPA`; rescale it so both
        // contributions carry the caller's `kappa` and nothing else changes
        // for the relative measures that share that helper.
        let rescale = kappa / ROW_NOISE_KAPPA;
        let placement = self.row_noise_floor(jac, residual);
        let finite_or_zero = |v: Number| if v.is_finite() && v > 0.0 { v } else { 0.0 };
        (0..dim)
            .map(|i| {
                let from_placement = placement
                    .as_ref()
                    .and_then(|p| p.get(i))
                    .copied()
                    .unwrap_or(0.0);
                let from_placement = from_placement * rescale;
                let from_formation = magnitude
                    .and_then(|m| m.get(i))
                    .map_or(0.0, |&m| kappa * Number::EPSILON * m);
                finite_or_zero(from_placement).max(finite_or_zero(from_formation))
            })
            .collect()
    }

    /// The inequality-block half of
    /// [`Self::curr_relative_primal_infeasibility_max`]:
    /// `max_i |d_i − s_i| / max(|d_i|, |d_l_i|, |d_u_i|)`.
    ///
    /// This is the scale-free companion to
    /// [`Self::curr_unscaled_primal_infeasibility_max`]. An absolute violation
    /// measure cannot tell "satisfied" from "violated by 10% of everything the
    /// row is" once the row's numbers are small — `x >= 0.7` written as
    /// `1e-12·x >= 0.7e-12` has an absolute violation of `1e-13` at `x = 0.6`,
    /// under every absolute tolerance, while the row is violated by a seventh
    /// of its own right-hand side. The ratio is `0.14` at every writing of the
    /// row.
    ///
    /// Computed entirely in the internally-scaled space: numerator and
    /// denominator both carry the row scaling `dd_i`, so it cancels and the
    /// ratio is invariant under it — which is the point.
    ///
    /// The magnitude comes from the row's **declared bounds only** — the
    /// current value `d_i` is deliberately excluded. On an *active* row the
    /// value converges to the bound, so for a zero-bound row (`g(x) >= 0`,
    /// ubiquitous) both the violation and `|d_i|` go to zero together and
    /// their ratio hovers near 1 at a perfectly converged point — including
    /// `|d_i|` made HS13's genuine solution read as 100% violated and vetoed
    /// its certificate. A zero bound also needs no relative treatment in the
    /// first place: `s·g >= 0` is the same row at every `s`, so the absolute
    /// test is already invariant there. Rows whose bounds are all zero or
    /// non-finite therefore contribute nothing (the relative measure
    /// abstains) — "zero" measured against the row's own
    /// [noise floor](`Self::row_noise_floor`) rather than exactly, for the
    /// reason spelled out on [`Self::relative_c_infeasibility_max`]: a
    /// converter that writes `2^-53` where the model says `0` otherwise hands
    /// a bound made of rounding residue to a measure that then reads the row
    /// as 100% violated. QPILOTNO carries 43 such inequality bounds at
    /// `1e-17`–`1e-15`, and one of them — a row sitting at exactly `d(x) = 0`
    /// against a declared bound of `1.1e-16` — pinned `rel_viol` at `1.0` for
    /// the whole run and drove the gh #446 local-infeasibility verdict from
    /// this block. Equality rows are judged by
    /// [`Self::relative_c_infeasibility_max`], which plumbs the pre-fold RHS
    /// back to supply the magnitude the fold into `c(x) = 0` erased.
    ///
    /// The violation judged is the **distance of `d(x)` outside the declared
    /// box** — NOT the lifted residual `|d − s|` the absolute measure uses.
    /// `|d − s|` only bounds the true violation from above: mid-solve the
    /// slack lags `d` while `d` is comfortably inside its bounds, so a
    /// slack-lag of 1% of a small row's magnitude read as "violated" at
    /// points that are genuinely feasible. That armed the rapid-infeasibility
    /// pre-filter at degenerate QP endgames, where the no-descent
    /// confirmation is vacuous (the violation is already ~0, so no materially
    /// less-violating point exists) — and 18 feasible CUTEr QPs were reported
    /// locally infeasible. Measured, not hypothetical.
    pub fn relative_d_infeasibility_max(&self) -> Number {
        let dms = self.curr_d_minus_s();
        if dms.dim() == 0 {
            return 0.0;
        }
        let d = self.curr_d();
        let (lo, hi, mask_l, mask_u) = {
            let nlp = self.nlp.borrow();
            // The *declared* bounds where the NLP tracks them: the live
            // `d_l`/`d_u` carry the `bound_relax_factor` widening, under which
            // a declared-zero bound reads as `~1e-8` — a fabricated magnitude
            // for a row that has none (that misread both vetoed HS13's genuine
            // solution and manufactured "relative" verdicts out of thin air).
            let (mut cl, mut cu) = (nlp.d_l().make_new(), nlp.d_u().make_new());
            match nlp.declared_d_bounds() {
                Some((dl, du)) => {
                    let (Some(cld), Some(cud)) = (
                        cl.as_any_mut().downcast_mut::<DenseVector>(),
                        cu.as_any_mut().downcast_mut::<DenseVector>(),
                    ) else {
                        return 0.0;
                    };
                    cld.set_values(&dl);
                    cud.set_values(&du);
                }
                None => {
                    cl.copy(nlp.d_l());
                    cu.copy(nlp.d_u());
                }
            }
            let mut lo = dms.make_new();
            lo.set(0.0);
            nlp.pd_l().mult_vector(1.0, &*cl, 0.0, &mut *lo);
            let mut hi = dms.make_new();
            hi.set(0.0);
            nlp.pd_u().mult_vector(1.0, &*cu, 0.0, &mut *hi);
            // A projected 0 is ambiguous — "no bound on this side" and "a
            // declared zero bound" both read 0 — so project an all-ones
            // vector through the same expansion to get presence masks.
            let mut ones_l = nlp.d_l().make_new();
            ones_l.set(1.0);
            let mut mask_l = dms.make_new();
            mask_l.set(0.0);
            nlp.pd_l().mult_vector(1.0, &*ones_l, 0.0, &mut *mask_l);
            let mut ones_u = nlp.d_u().make_new();
            ones_u.set(1.0);
            let mut mask_u = dms.make_new();
            mask_u.set(0.0);
            nlp.pd_u().mult_vector(1.0, &*ones_u, 0.0, &mut *mask_u);
            (lo, hi, mask_l, mask_u)
        };
        let noise = self.row_noise_floor(&*self.curr_jac_d(), &*dms);
        let (Some(d), Some(lo), Some(hi), Some(mask_l), Some(mask_u)) = (
            d.as_any().downcast_ref::<DenseVector>(),
            lo.as_any().downcast_ref::<DenseVector>(),
            hi.as_any().downcast_ref::<DenseVector>(),
            mask_l.as_any().downcast_ref::<DenseVector>(),
            mask_u.as_any().downcast_ref::<DenseVector>(),
        ) else {
            return 0.0;
        };
        if !(d.is_initialized()
            && lo.is_initialized()
            && hi.is_initialized()
            && mask_l.is_initialized()
            && mask_u.is_initialized())
        {
            return 0.0;
        }
        let dv = d.expanded_values();
        let lov = lo.expanded_values();
        let hiv = hi.expanded_values();
        let mlv = mask_l.expanded_values();
        let muv = mask_u.expanded_values();
        let mut worst = 0.0_f64;
        for i in 0..dv.len() {
            let (has_l, has_u) = (mlv[i] > 0.5, muv[i] > 0.5);
            let mut viol = 0.0_f64;
            let mut mag = 0.0_f64;
            if has_l {
                viol = viol.max(lov[i] - dv[i]);
                mag = mag.max(lov[i].abs());
            }
            if has_u {
                viol = viol.max(dv[i] - hiv[i]);
                mag = mag.max(hiv[i].abs());
            }
            // `0.0` when no floor could be computed, which reproduces the
            // former `mag > 0.0` gate exactly.
            let floor = noise.as_ref().map_or(0.0, |n| n[i]);
            if mag > floor && mag.is_finite() && viol.is_finite() && viol > 0.0 {
                worst = worst.max(viol / mag);
            }
        }
        worst
    }

    /// The objective scaling factor `df` currently in force (`1.0` when no
    /// objective scaling is active).
    ///
    /// Exposed because the termination logic must be able to tell an honest
    /// certificate from one an extreme scale has masked (gh #200): the scale
    /// factor itself is the discriminating signal, not the error.
    pub fn obj_scaling_factor(&self) -> Number {
        self.nlp.borrow().obj_scaling_factor()
    }

    /// The solver-computed part of the objective scale — see
    /// [`IpoptNlp::computed_obj_scaling_factor`]. The masked-certificate test
    /// keys on this, not on the product, so a user who deliberately scales a
    /// well-conditioned objective down is not second-guessed.
    pub fn computed_obj_scaling_factor(&self) -> Number {
        self.nlp.borrow().computed_obj_scaling_factor()
    }

    /// Overall **unscaled** max-norm KKT error — `max` of the unscaled dual
    /// infeasibility, primal infeasibility, and complementarity. This is the
    /// honest "distance from a KKT point in the user's own units", as
    /// opposed to [`Self::curr_nlp_error`], which additionally applies the
    /// `s_d`/`s_c` optimality scaling. Used by the status-fidelity gate and
    /// surfaced to callers that must independently verify a returned
    /// certificate (pounce#173).
    pub fn curr_unscaled_nlp_error(&self) -> Number {
        self.curr_unscaled_dual_infeasibility_max()
            .max(self.curr_unscaled_primal_infeasibility_max())
            .max(self.curr_unscaled_complementarity_max())
    }

    pub fn trial_f(&self) -> Number {
        let iv = self.trial_iv();
        let mut nlp = self.nlp.borrow_mut();
        nlp.eval_f(&*iv.x)
    }

    fn barrier_obj_at(
        &self,
        f: Number,
        s_x_l: &dyn Vector,
        s_x_u: &dyn Vector,
        s_s_l: &dyn Vector,
        s_s_u: &dyn Vector,
    ) -> Number {
        let mu = self.data.borrow().curr_mu;
        let log_sum = s_x_l.sum_logs() + s_x_u.sum_logs() + s_s_l.sum_logs() + s_s_u.sum_logs();
        let mut phi = f - mu * log_sum;
        if self.kappa_d > 0.0 {
            let di = self.damping_indicators();
            phi += self.kappa_d * mu * s_x_l.dot(&*di.x_l);
            phi += self.kappa_d * mu * s_x_u.dot(&*di.x_u);
            phi += self.kappa_d * mu * s_s_l.dot(&*di.s_l);
            phi += self.kappa_d * mu * s_s_u.dot(&*di.s_u);
        }
        phi
    }

    pub fn curr_barrier_obj(&self) -> Number {
        let f = self.curr_f();
        let s_x_l = self.curr_slack_x_l();
        let s_x_u = self.curr_slack_x_u();
        let s_s_l = self.curr_slack_s_l();
        let s_s_u = self.curr_slack_s_u();
        self.barrier_obj_at(f, &*s_x_l, &*s_x_u, &*s_s_l, &*s_s_u)
    }

    pub fn trial_barrier_obj(&self) -> Number {
        let f = self.trial_f();
        let s_x_l = self.trial_slack_x_l();
        let s_x_u = self.trial_slack_x_u();
        let s_s_l = self.trial_slack_s_l();
        let s_s_u = self.trial_slack_s_u();
        self.barrier_obj_at(f, &*s_x_l, &*s_x_u, &*s_s_l, &*s_s_u)
    }

    /// Gradient of the barrier objective wrt `x`:
    ///   ∇_x φ = ∇f(x) − μ · [P_L · (1/s_L) − P_U · (1/s_U)] + damping
    /// Mirrors `IpIpoptCalculatedQuantities.cpp:CalcGradBarrierObjectiveX`.
    pub fn curr_grad_barrier_obj_x(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let mu = self.data.borrow().curr_mu;
        let s_l = self.curr_slack_x_l();
        let s_u = self.curr_slack_x_u();

        let mut inv_s_l = s_l.make_new();
        inv_s_l.copy(&*s_l);
        inv_s_l.element_wise_reciprocal();
        let mut inv_s_u = s_u.make_new();
        inv_s_u.copy(&*s_u);
        inv_s_u.element_wise_reciprocal();

        let grad_f = self.curr_grad_f();
        let mut tmp = iv.x.make_new();
        tmp.copy(&*grad_f);
        let nlp = self.nlp.borrow();
        // tmp -= μ · P_L · inv_s_l
        nlp.px_l().mult_vector(-mu, &*inv_s_l, 1.0, &mut *tmp);
        // tmp += μ · P_U · inv_s_u
        nlp.px_u().mult_vector(mu, &*inv_s_u, 1.0, &mut *tmp);

        if self.kappa_d > 0.0 {
            let di = self.damping_indicators();
            // + κ_d μ · P_L · 1_singly_x_L
            nlp.px_l()
                .mult_vector(self.kappa_d * mu, &*di.x_l, 1.0, &mut *tmp);
            // − κ_d μ · P_U · 1_singly_x_U
            nlp.px_u()
                .mult_vector(-self.kappa_d * mu, &*di.x_u, 1.0, &mut *tmp);
        }
        rc_from(tmp)
    }

    /// Gradient of the barrier objective wrt `s`:
    ///   ∇_s φ = − μ · [P_L · (1/s_s_L) − P_U · (1/s_s_U)] + damping
    pub fn curr_grad_barrier_obj_s(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let mu = self.data.borrow().curr_mu;
        let s_l = self.curr_slack_s_l();
        let s_u = self.curr_slack_s_u();

        let mut inv_s_l = s_l.make_new();
        inv_s_l.copy(&*s_l);
        inv_s_l.element_wise_reciprocal();
        let mut inv_s_u = s_u.make_new();
        inv_s_u.copy(&*s_u);
        inv_s_u.element_wise_reciprocal();

        let mut tmp = iv.s.make_new();
        tmp.set(0.0);
        let nlp = self.nlp.borrow();
        nlp.pd_l().mult_vector(-mu, &*inv_s_l, 1.0, &mut *tmp);
        nlp.pd_u().mult_vector(mu, &*inv_s_u, 1.0, &mut *tmp);

        if self.kappa_d > 0.0 {
            let di = self.damping_indicators();
            nlp.pd_l()
                .mult_vector(self.kappa_d * mu, &*di.s_l, 1.0, &mut *tmp);
            nlp.pd_u()
                .mult_vector(-self.kappa_d * mu, &*di.s_u, 1.0, &mut *tmp);
        }
        rc_from(tmp)
    }

    // --------------------------------------------------------------
    // Step-aware quadratic-model quantities — used by the penalty
    // line-search acceptor's pred/ared test and by the quality-
    // function mu oracle's q(σ) evaluator.
    // --------------------------------------------------------------

    /// Directional derivative of the barrier objective along `(δx, δs)`:
    /// `gradBarrTDelta = ∇_x φ · δx + ∇_s φ · δs`. Port of
    /// `IpIpoptCalculatedQuantities.cpp:CurrGradBarrTDelta` (called
    /// `IpCq().curr_gradBarrTDelta()` in upstream after the search dir
    /// has been computed).
    pub fn curr_grad_barr_t_delta(&self, delta_x: &dyn Vector, delta_s: &dyn Vector) -> Number {
        let g_x = self.curr_grad_barrier_obj_x();
        let g_s = self.curr_grad_barrier_obj_s();
        g_x.dot(delta_x) + g_s.dot(delta_s)
    }

    /// `δᵀ(W + Σ_x + δ_pert_x I)δ_x + δ_sᵀ(Σ_s + δ_pert_s I)δ_s` —
    /// the quadratic-model term used by `IpPenaltyLSAcceptor.cpp:
    /// InitThisLineSearch:101-129`. Reads `W` and the active PD
    /// perturbations from [`crate::ipopt_data::IpoptData`].
    /// Returns 0 if the result would be negative (matching upstream's
    /// `if dWd <= 0 then dWd = 0` guard at line 133).
    pub fn curr_dwd(&self, delta_x: &dyn Vector, delta_s: &dyn Vector) -> Number {
        let mut dwd: Number = 0.0;

        // δ_xᵀ W δ_x.
        if let Some(w) = self.data.borrow().w.clone() {
            let mut wd = delta_x.make_new();
            w.mult_vector(1.0, delta_x, 0.0, &mut *wd);
            dwd += wd.dot(delta_x);
        }

        // δ_xᵀ Σ_x δ_x.
        let sigma_x = self.curr_sigma_x();
        let mut tmp_x = delta_x.make_new();
        tmp_x.copy(delta_x);
        tmp_x.element_wise_multiply(&*sigma_x);
        dwd += tmp_x.dot(delta_x);

        // δ_sᵀ Σ_s δ_s.
        let sigma_s = self.curr_sigma_s();
        let mut tmp_s = delta_s.make_new();
        tmp_s.copy(delta_s);
        tmp_s.element_wise_multiply(&*sigma_s);
        dwd += tmp_s.dot(delta_s);

        // PD perturbations.
        let pert = self.data.borrow().perturbations;
        if pert.delta_x != 0.0 {
            let nx = delta_x.nrm2();
            dwd += pert.delta_x * nx * nx;
        }
        if pert.delta_s != 0.0 {
            let ns = delta_s.nrm2();
            dwd += pert.delta_s * ns * ns;
        }

        dwd.max(0.0)
    }

    // --------------------------------------------------------------
    // Constraint violation theta — port of
    // `IpIpoptCalculatedQuantities.cpp:CurrConstraintViolation`.
    // Default norm is 1-norm (option `constraint_violation_norm`,
    // default "1-norm" upstream); we hardwire 1-norm in v1.0.
    // --------------------------------------------------------------

    pub fn curr_constraint_violation(&self) -> Number {
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        c.asum() + dms.asum()
    }

    /// Number of constraint rows backing the 1-norm above, i.e.
    /// `dim(c) + dim(d - s)`. Upstream never needs this because it
    /// treats `theta` as a bare scalar, but any threshold expressed in
    /// `theta` units is a *sum* over this many rows — a `theta` of `T`
    /// is a mean per-row residual of `T / rows`. The filter acceptor
    /// uses it to floor the `theta_max` reference so the ceiling means
    /// the same thing on a 10-row and a 50 000-row model.
    pub fn constraint_violation_rows(&self) -> usize {
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        (c.dim() as usize) + (dms.dim() as usize)
    }

    pub fn trial_constraint_violation(&self) -> Number {
        let c = self.trial_c();
        let dms = self.trial_d_minus_s();
        c.asum() + dms.asum()
    }

    /// Max-norm primal infeasibility — `max(||c||_∞, ||d − s||_∞)`. Used
    /// by the iteration output's `inf_pr` column when
    /// `inf_pr_output == INTERNAL`. Mirrors
    /// `IpIpoptCalculatedQuantities.cpp:CurrPrimalInfeasibility(NORM_MAX)`.
    pub fn curr_primal_infeasibility_max(&self) -> Number {
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        c.amax().max(dms.amax())
    }

    /// Max-norm dual infeasibility — `max(||∇_x L||_∞, ||∇_s L||_∞)`.
    /// Mirrors `IpIpoptCalculatedQuantities.cpp:CurrDualInfeasibility(NORM_MAX)`.
    pub fn curr_dual_infeasibility_max(&self) -> Number {
        let glx = self.curr_grad_lag_x();
        let gls = self.curr_grad_lag_s();
        glx.amax().max(gls.amax())
    }

    /// Magnitude of the largest **term** the Lagrangian gradient is assembled
    /// from — the scale [`Self::curr_dual_infeasibility_max`] is a residual
    /// *of* (gh #532).
    ///
    /// ```text
    ///   D = max( ‖∇f‖_∞ , ‖J_cᵀ y_c‖_∞ , ‖J_dᵀ y_d‖_∞ ,
    ///            ‖P_L z_L‖_∞ , ‖P_U z_U‖_∞ ,
    ///            ‖y_d‖_∞ , ‖P_L v_L‖_∞ , ‖P_U v_U‖_∞ )
    /// ```
    ///
    /// `∇L` is the *sum* of exactly these terms, so `dual_inf / D` is the
    /// fraction of them that failed to cancel: `1` at a point where nothing
    /// cancelled (`min -exp(x) s.t. x >= 0` running away, `∇f = −8.8e47` with
    /// no multiplier to meet it), and `~eps` at a point where the cancellation
    /// was as complete as the arithmetic allows. That ratio is the
    /// scale-invariant statement of stationarity: it is unchanged by
    /// multiplying the objective — and hence every multiplier — by a positive
    /// constant, which is the map an absolute bound on `dual_inf` is not
    /// invariant under.
    ///
    /// The projections are applied rather than assumed away: `P_L`/`P_U` are
    /// 0/1 expansion matrices in the main NLP, where the scatter leaves the
    /// max-norm alone, but the term's own norm is what this measures and the
    /// restoration NLP supplies its own operators.
    ///
    /// No `has_valid_numbers` sweep, unlike [`Self::curr_nlp_error`] (gh #292):
    /// `amax` drops NaN, so a NaN gradient reads here as a finite scale. That
    /// cannot launder anything, because the only caller pairs this with the
    /// aggregate `nlp_err <= tol` test, and `nlp_err` carries that sweep — a
    /// NaN anywhere in `∇L` makes it NaN, and `NaN <= tol` is false.
    ///
    /// Repeats the `∇f` and the two transpose products
    /// [`Self::curr_grad_lag_x`] already performs on the same iterate, plus
    /// four scatters. The evaluations themselves hit `OrigIpoptNLP`'s
    /// per-iterate caches, so the marginal cost is the products — but it is
    /// still a second pass, and the caller reads this only where a termination
    /// certificate is otherwise on the table. See
    /// `OptErrorConvCheck::dual_inf_bound`.
    pub fn curr_dual_infeasibility_scale_max(&self) -> Number {
        let iv = self.curr_iv();
        let mut scale = self
            .curr_grad_f()
            .amax()
            .max(self.curr_jac_c_t_times_curr_y_c().amax())
            .max(self.curr_jac_d_t_times_curr_y_d().amax())
            .max(iv.y_d.amax());

        let nlp = self.nlp.borrow();
        let mut tmp_x = iv.x.make_new();
        nlp.px_l().mult_vector(1.0, &*iv.z_l, 0.0, &mut *tmp_x);
        scale = scale.max(tmp_x.amax());
        nlp.px_u().mult_vector(1.0, &*iv.z_u, 0.0, &mut *tmp_x);
        scale = scale.max(tmp_x.amax());

        let mut tmp_s = iv.y_d.make_new();
        nlp.pd_l().mult_vector(1.0, &*iv.v_l, 0.0, &mut *tmp_s);
        scale = scale.max(tmp_s.amax());
        nlp.pd_u().mult_vector(1.0, &*iv.v_u, 0.0, &mut *tmp_s);
        scale.max(tmp_s.amax())
    }

    /// [`Self::curr_dual_infeasibility_scale_max`] in the **unscaled**
    /// (user-original) space. Every term of the scaled Lagrangian gradient is
    /// `df` times its user-space counterpart — `∇f_scaled = df·∇f`,
    /// `J_cᵀ_scaled y_c_scaled = Jᵀ(dc ⊙ y_c_scaled) = df·Jᵀ y_c` since
    /// `dc ⊙ y_scaled = df·y_user`, and likewise for the bound blocks, POUNCE
    /// applying no variable scaling — so the unscaling is the single divide by
    /// `|df|` that [`Self::curr_unscaled_dual_infeasibility_max`] performs on
    /// the residual, term for term and row scaling included. Magnitude, for
    /// the reason documented there: `df` is signed, a max-norm is not.
    pub fn curr_unscaled_dual_infeasibility_scale_max(&self) -> Number {
        let df = self.nlp.borrow().obj_scaling_factor().abs();
        let scaled = self.curr_dual_infeasibility_scale_max();
        if df == 0.0 || df == 1.0 {
            scaled
        } else {
            scaled / df
        }
    }

    /// Scaled stationarity of the infeasibility measure `½‖(c, d−s)‖²`
    /// — `‖J_cᵀ c + J_dᵀ (d−s)‖_∞ / max(1, ‖(c, d−s)‖_∞)`. The
    /// numerator is the x-gradient of the squared constraint
    /// violation; a value near zero with the violation itself bounded
    /// away from zero marks an iterate converging to a stationary
    /// point of the infeasibility — i.e. a locally infeasible problem.
    /// No linear solve: two transpose-products. Mirrors the gradient
    /// term behind Ipopt's `IpRestoConvCheck.cpp` `LOCALLY_INFEASIBLE`
    /// test, applied here in the main loop.
    /// Does a short step along `−∇θ` actually reduce the constraint violation?
    ///
    /// `LocalInfeasibility` asserts the iterate has converged to a **stationary
    /// point of the constraint violation** — that no local move reduces it. That
    /// is a checkable claim, and this checks it directly instead of trusting a
    /// threshold on a proxy.
    ///
    /// Why a probe rather than a better proxy: the detector's surrogate is
    /// `‖Jᵀc‖ / max(1, ‖c‖)` against an absolute tolerance, and no variant of it
    /// separates the cases. Measured over 800 MINLPLib models plus targeted
    /// infeasible problems, the scaled form produces a confirmed false verdict
    /// (HS13 from `x₀ = (1e4, 1e4)`, where the constraint scaling `dc ≈ 3.3e-7`
    /// drives the surrogate to `5e-14` at a point whose violation is 0.51); the
    /// unscaled form needs a tolerance ≥ 1e-2 to fire at all, which introduces
    /// new false infeasibility on 3+ corpus models while still losing 2 correct
    /// detections; and a scale-invariant `‖Jᵀc‖ / ‖c‖²` is not separable even on
    /// the targeted set. A single absolute threshold on a surrogate cannot do
    /// this job.
    ///
    /// Comparing `θ` at two points is scale-free by construction — the row
    /// scaling cancels out of the ratio — so this needs no calibration at all.
    ///
    /// Costs one `eval_c`/`eval_d` pair per probed step, and runs only where the
    /// detector was about to fire (both gates already passed for a full streak),
    /// which is rare. Steps are clamped to the variable bounds, so descent that
    /// only exists outside the box is correctly not counted — that direction
    /// would suppress a *correct* infeasibility verdict.
    ///
    /// Returns `true` when descent is available, i.e. the iterate is **not**
    /// stationary and `LocalInfeasibility` must not be declared.
    pub fn infeasibility_descent_available(&self) -> bool {
        use pounce_linalg::DenseVector;

        let theta_curr = self.curr_primal_infeasibility_max();
        if theta_curr <= 0.0 {
            return false;
        }
        // -grad of 1/2||(c, d-s)||^2 w.r.t. x.
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        let jc_t_c = self.curr_jac_c_t_times_vec(&*c);
        let jd_t_dms = self.curr_jac_d_t_times_vec(&*dms);
        let mut grad = jc_t_c.make_new();
        grad.add_two_vectors(1.0, &*jc_t_c, 1.0, &*jd_t_dms, 0.0);
        let gnorm = grad.amax();
        if !(gnorm > 0.0) || !gnorm.is_finite() {
            // A vanishing gradient is the stationary case this exists to
            // confirm; a non-finite one gives us nothing to probe with.
            return false;
        }

        let x = self.curr_iv().x.clone();
        let nlp = self.nlp.borrow();

        // Full-length bound values and finite-bound indicators, lifted through
        // the expansion matrices (same pattern as the divergence guard).
        let mut ones_l = nlp.x_l().make_new();
        ones_l.set(1.0);
        let mut has_lb = x.make_new();
        nlp.px_l().mult_vector(1.0, &*ones_l, 0.0, &mut *has_lb);
        let mut lb = x.make_new();
        nlp.px_l().mult_vector(1.0, nlp.x_l(), 0.0, &mut *lb);

        let mut ones_u = nlp.x_u().make_new();
        ones_u.set(1.0);
        let mut has_ub = x.make_new();
        nlp.px_u().mult_vector(1.0, &*ones_u, 0.0, &mut *has_ub);
        let mut ub = x.make_new();
        nlp.px_u().mult_vector(1.0, nlp.x_u(), 0.0, &mut *ub);
        drop(nlp);

        let dense = |v: &dyn Vector| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|d| d.expanded_values())
        };
        let (Some(xv), Some(gv), Some(lbv), Some(ubv), Some(hl), Some(hu)) = (
            dense(&*x),
            dense(&*grad),
            dense(&*lb),
            dense(&*ub),
            dense(&*has_lb),
            dense(&*has_ub),
        ) else {
            // Non-dense backing: no probe possible. Report "no descent" so the
            // caller falls back to the surrogate's verdict rather than silently
            // suppressing every infeasibility conclusion.
            return false;
        };

        // Relative step lengths, so the probe is independent of problem scale.
        let xnorm = xv.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let base = xnorm / gnorm;

        let mut trial = x.make_new();
        for k in 0..Self::INFEAS_PROBE_STEPS {
            let alpha = base * 10f64.powi(-(k as i32));
            {
                let Some(t) = trial.as_any_mut().downcast_mut::<DenseVector>() else {
                    return false;
                };
                for (i, slot) in t.values_mut().iter_mut().enumerate() {
                    let mut xi = xv[i] - alpha * gv[i];
                    if hl[i] != 0.0 {
                        xi = xi.max(lbv[i]);
                    }
                    if hu[i] != 0.0 {
                        xi = xi.min(ubv[i]);
                    }
                    *slot = xi;
                }
            }
            if let Some(theta) = self.theta_at(&*trial) {
                if theta.is_finite() && theta < theta_curr * (1.0 - Self::INFEAS_PROBE_MARGIN) {
                    return true;
                }
            }
        }
        false
    }

    /// Number of geometrically decreasing step lengths the descent probe tries.
    const INFEAS_PROBE_STEPS: usize = 6;
    /// Relative reduction in `θ` a probe step must achieve before it counts as
    /// descent and vetoes the verdict.
    ///
    /// Deliberately coarse. The question is not "is this the exact minimiser of
    /// the violation" — an interior-point iterate converging toward one always
    /// has some infinitesimal descent left, and a tight margin would veto
    /// forever and never let a genuine infeasibility be declared. The question
    /// is whether a *materially* less-violating point sits nearby, which is what
    /// distinguishes "converging to an infeasible stationary point" from
    /// "nowhere near stationary".
    ///
    /// The two regimes are far apart, so the exact value is not delicate. On the
    /// genuinely infeasible `x³+y³ == 1 ∧ == 2`, iterates near the least-squares
    /// point have only ~0.07 % descent available. On HS13's false verdict, one
    /// step takes `θ` from 0.51 to **zero** — a 100 % reduction. Anything between
    /// a few percent and most of the way separates them; 10 % sits in the middle.
    const INFEAS_PROBE_MARGIN: Number = 0.1;

    /// Max-norm constraint violation at an arbitrary `x`, evaluated on scratch
    /// vectors so the algorithm's `curr`/`trial` state is untouched. `None` if
    /// the evaluation is unusable.
    fn theta_at(&self, x: &dyn Vector) -> Option<Number> {
        let iv = self.curr_iv();
        let mut nlp = self.nlp.borrow_mut();
        let mut c = iv.y_c.make_new();
        nlp.eval_c(x, &mut *c);
        let mut d = iv.s.make_new();
        nlp.eval_d(x, &mut *d);
        // `d - s` against the CURRENT slacks, matching how `curr_d_minus_s`
        // measures the violation: the probe moves x only.
        let mut dms = iv.s.make_new();
        dms.add_two_vectors(1.0, &*d, -1.0, &*iv.s, 0.0);
        let t = c.amax().max(dms.amax());
        t.is_finite().then_some(t)
    }

    pub fn curr_infeasibility_stationarity(&self) -> Number {
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        let jc_t_c = self.curr_jac_c_t_times_vec(&*c);
        let jd_t_dms = self.curr_jac_d_t_times_vec(&*dms);
        let mut grad = jc_t_c.make_new();
        grad.add_two_vectors(1.0, &*jc_t_c, 1.0, &*jd_t_dms, 0.0);
        let viol = c.amax().max(dms.amax());
        grad.amax() / viol.max(1.0)
    }

    // --------------------------------------------------------------
    // Average / scalar complementarity
    // --------------------------------------------------------------

    /// `(z_L · s_L + z_U · s_U + v_L · s_L^d + v_U · s_U^d) / N`
    /// where `N` is the total number of bound multipliers
    /// (`IpIpoptCalculatedQuantities.cpp:3553-3606`).
    pub fn curr_avrg_compl(&self) -> Number {
        let iv = self.curr_iv();
        let n = iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        if n == 0 {
            return 0.0;
        }
        let s_x_l = self.curr_slack_x_l();
        let s_x_u = self.curr_slack_x_u();
        let s_s_l = self.curr_slack_s_l();
        let s_s_u = self.curr_slack_s_u();
        let mut acc = iv.z_l.dot(&*s_x_l);
        acc += iv.z_u.dot(&*s_x_u);
        acc += iv.v_l.dot(&*s_s_l);
        acc += iv.v_u.dot(&*s_s_u);
        acc / Number::from(n)
    }

    /// `min_i (s_i · z_i)` over all four bound complementarity blocks.
    /// Mirrors `IpIpoptCalculatedQuantities.cpp:CurrComplxMin`
    /// (lines 3608-3640) — the smallest pairwise product `s · z`,
    /// signalling how close the iterate is to the central path.
    /// Empty bound sets contribute `+∞`; returns `0` if no bounds at
    /// all.
    pub fn curr_complementarity_min(&self) -> Number {
        let cxl = self.curr_compl_x_l();
        let cxu = self.curr_compl_x_u();
        let csl = self.curr_compl_s_l();
        let csu = self.curr_compl_s_u();
        let m = |v: &Rc<dyn Vector>| {
            if v.dim() == 0 {
                Number::INFINITY
            } else {
                v.min()
            }
        };
        let acc = m(&cxl).min(m(&cxu)).min(m(&csl)).min(m(&csu));
        if acc.is_infinite() { 0.0 } else { acc }
    }

    /// Max-norm of the unbarriered complementarity blocks
    /// `max_i |s_i · z_i|` across all four `(x_L, x_U, s_L, s_U)`
    /// pairs. Mirrors upstream
    /// `IpIpoptCalculatedQuantities.cpp:CurrComplementarity(0., NORM_MAX)`
    /// — used by `OptimalityErrorConvergenceCheck` to gate the
    /// per-component `compl_inf_tol` test independently of the scaled
    /// scalar `curr_nlp_error`.
    pub fn curr_complementarity_max(&self) -> Number {
        self.curr_compl_x_l()
            .amax()
            .max(self.curr_compl_x_u().amax())
            .max(self.curr_compl_s_l().amax())
            .max(self.curr_compl_s_u().amax())
    }

    /// Centrality measure `ξ = min_i(s_i z_i) / avrg(s · z)`. Mirrors
    /// `IpIpoptCalculatedQuantities.cpp:CurrCentralityMeasure`. Used
    /// by [`crate::mu::oracle::loqo::LoqoMuOracle`] to bias σ toward
    /// the central path when the iterate is unbalanced. Returns `1.0`
    /// (perfectly central) when there are no bound multipliers.
    pub fn curr_centrality_measure(&self) -> Number {
        let avrg = self.curr_avrg_compl();
        if avrg <= 0.0 {
            return 1.0;
        }
        self.curr_complementarity_min() / avrg
    }

    /// Barriered KKT error `E_μ(x,y,z)` — port of
    /// `IpIpoptCalculatedQuantities.cpp:CurrBarrierError`. Same as
    /// [`Self::curr_nlp_error`] but uses the *relaxed* complementarity
    /// `s ⊙ z − μ` so the residual is zero when the iterate sits on the
    /// μ-perturbed central path. The monotone barrier-update strategy
    /// reduces μ only once this error drops below
    /// `barrier_tol_factor · μ`.
    pub fn curr_barrier_error(&self) -> Number {
        let iv = self.curr_iv();
        let (s_d, s_c) = self.optimality_error_scaling(&iv);

        let glx = self.curr_grad_lag_x();
        let gls = self.curr_grad_lag_s();
        let dual = glx.amax().max(gls.amax()) / s_d;

        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        let primal = c.amax().max(dms.amax());

        let compl = self
            .curr_relaxed_compl_x_l()
            .amax()
            .max(self.curr_relaxed_compl_x_u().amax())
            .max(self.curr_relaxed_compl_s_l().amax())
            .max(self.curr_relaxed_compl_s_u().amax())
            / s_c;

        dual.max(primal).max(compl)
    }

    /// Optimality-scaled max-norm KKT error — port of
    /// `IpIpoptCalculatedQuantities.cpp:3050-3104`.
    ///
    /// ```text
    ///   E = max( ||∇_x L, ∇_s L||_∞ / s_d ,
    ///            ||c, d − s||_∞ ,
    ///            ||compl||_∞ / s_c )
    /// ```
    ///
    /// where `s_d` / `s_c` are the asum-based scalings from
    /// `ComputeOptimalityErrorScaling` (see §4 of `MAIN_LOOP.md`).
    /// Uses `mu_target = 0` (the unbarriered KKT residual). The
    /// barriered variant is `curr_barrier_error` (TODO in Phase 7).
    pub fn curr_nlp_error(&self) -> Number {
        self.nlp_error(None)
    }

    /// [`Self::curr_nlp_error`] with the primal-infeasibility term replaced by
    /// [`Self::curr_primal_infeasibility_above_noise`] — i.e. counting a
    /// constraint row's residual only where it rises above the finest value
    /// that row's residual can take in floating point (gh #528).
    ///
    /// Never larger than [`Self::curr_nlp_error`], and equal to it whenever no
    /// row is at its own resolution limit — which is every problem whose data
    /// is `O(1)`, so the common path is unchanged. It exists because the other
    /// two terms of the KKT error are already normalised (`s_d`, `s_c`) while
    /// the primal one is a bare absolute residual: `‖c‖_∞` and `‖d − s‖_∞` are
    /// quantised in units of `eps ·` the rows' own magnitude, so on a model
    /// whose constraint values reach `~1e8` the smallest *nonzero* value the
    /// term can take already exceeds the default `tol = 1e-8`. Judging that
    /// term absolutely there asks for a residual no iterate can represent.
    ///
    /// Read only by the **strict** convergence gate, which pairs it with the
    /// unscaled `constr_viol_tol` test on the full, unfloored residual — so
    /// what this admits is bounded by the user's own feasibility tolerance,
    /// never by the noise floor alone.
    ///
    /// `kappa` is the safety factor on the per-row floor —
    /// [`ROW_NOISE_KAPPA`] by default, from the `primal_noise_floor_kappa`
    /// option. **`0` switches the floor off entirely**, making this identical
    /// to [`Self::curr_nlp_error`] and the strict gate bit-for-bit upstream's.
    pub fn curr_nlp_error_above_primal_noise(&self, kappa: Number) -> Number {
        self.nlp_error(Some(kappa))
    }

    /// [`Self::curr_nlp_error`] with the complementarity term supplied by the
    /// caller instead of read off the iterate, keeping the `s_c` normalisation
    /// and the other two terms exactly as they are.
    ///
    /// One caller: the crossover phase (#612). Its returned point sits
    /// *exactly* on the active constraints of the problem **as the user
    /// declared it**, which is `bound_relax_factor` inside the widened bounds
    /// this object measures against — so the iterate-derived complementarity
    /// reads `|multiplier| · δ`, around `1e-8` for a unit multiplier, where
    /// the truth in the frame that was solved is zero. Left alone that put a
    /// converged run's `Overall NLP error` above `tol` and let the opt-in
    /// `kkt_fidelity_tol` gate downgrade a strictly better point (#646).
    ///
    /// `compl_raw` is the un-normalised max-norm `max_i |s_i · z_i|`, the same
    /// quantity [`Self::curr_complementarity_max`] returns; the `s_c` divide
    /// happens here. `kappa` follows
    /// [`Self::curr_nlp_error_above_primal_noise`], `0` disabling the floor.
    ///
    /// This is a *reporting* substitution and nothing more — no convergence
    /// decision reads it, because crossover runs after the status is already
    /// settled.
    pub fn curr_nlp_error_with_complementarity(&self, compl_raw: Number, kappa: Number) -> Number {
        let floor = (kappa > 0.0).then_some(kappa);
        self.nlp_error_inner(floor, Some(compl_raw))
    }

    /// `above_primal_noise` carries the floor's `kappa` when the primal term is
    /// to be floored, and is `None` for the plain upstream aggregate.
    fn nlp_error(&self, above_primal_noise: Option<Number>) -> Number {
        self.nlp_error_inner(above_primal_noise, None)
    }

    /// `compl_override` replaces the iterate-derived complementarity max-norm
    /// before the `s_c` divide; see
    /// [`Self::curr_nlp_error_with_complementarity`]. The NaN guard below
    /// still inspects the iterate's own complementarity vectors either way —
    /// an override is a change of *frame*, not a licence to stop looking at
    /// the iterate for non-finite numbers.
    fn nlp_error_inner(
        &self,
        above_primal_noise: Option<Number>,
        compl_override: Option<Number>,
    ) -> Number {
        let iv = self.curr_iv();
        let (s_d, s_c) = self.optimality_error_scaling(&iv);

        // dual infeasibility (max-norm of grad_lag_x and grad_lag_s)
        let glx = self.curr_grad_lag_x();
        let gls = self.curr_grad_lag_s();

        // primal: max(||c||, ||d-s||)
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();

        // unbarriered complementarity (mu_target = 0 → just ||compl||)
        let cxl = self.curr_compl_x_l();
        let cxu = self.curr_compl_x_u();
        let csl = self.curr_compl_s_l();
        let csu = self.curr_compl_s_u();

        // #292: the max-norm (`amax`/BLAS `iamax`) behind every term below
        // silently *drops* NaN — `NaN > m` is `false`, so a NaN component
        // leaves the running max untouched and is laundered into a finite
        // (typically `0.0`) KKT error. A NaN gradient, NaN constraint Jacobian
        // (via `∇_x L`'s `Jᵀy` term), or NaN residual would then read as an
        // *optimal* solve and return `Solve_Succeeded`. Detect any non-finite
        // component here — through the NaN-propagating `asum` behind
        // `has_valid_numbers`, not `amax` — and surface it as a non-finite KKT
        // error so the caller's existing `!nlp_err.is_finite()` guard fires
        // `Invalid_Number_Detected`. This is confined to the convergence/error
        // measure; the general `amax` semantics that step-size selection, the
        // line search, and the divergence detectors rely on are untouched.
        // (Inf is *not* laundered — `Inf > m` is true — so it already
        // propagated; this closes only the NaN hole, and Inf for free.)
        for v in [&glx, &gls, &c, &dms, &cxl, &cxu, &csl, &csu] {
            if !v.has_valid_numbers() {
                return Number::NAN;
            }
        }

        let dual = glx.amax().max(gls.amax()) / s_d;
        let primal = match above_primal_noise {
            Some(kappa) if kappa > 0.0 => self.curr_primal_infeasibility_above_noise(kappa),
            _ => c.amax().max(dms.amax()),
        };
        let compl_raw = compl_override
            .unwrap_or_else(|| cxl.amax().max(cxu.amax()).max(csl.amax()).max(csu.amax()));
        let compl = compl_raw / s_c;

        dual.max(primal).max(compl)
    }

    /// `(s_d, s_c)` per `ComputeOptimalityErrorScaling`
    /// (`IpIpoptCalculatedQuantities.cpp:3663-3700`).
    fn optimality_error_scaling(&self, iv: &IteratesVector) -> (Number, Number) {
        let s_max = self.s_max;

        // s_c: mean asum of all bound multipliers, capped at s_max,
        //      divided by s_max.
        let n_c = iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        let s_c = if n_c == 0 {
            1.0
        } else {
            let asum = iv.z_l.asum() + iv.z_u.asum() + iv.v_l.asum() + iv.v_u.asum();
            (s_max.max(asum / Number::from(n_c))) / s_max
        };

        // s_d: mean asum of all dual multipliers, capped, divided.
        let n_d =
            iv.y_c.dim() + iv.y_d.dim() + iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        let s_d = if n_d == 0 {
            1.0
        } else {
            let asum = iv.y_c.asum()
                + iv.y_d.asum()
                + iv.z_l.asum()
                + iv.z_u.asum()
                + iv.v_l.asum()
                + iv.v_u.asum();
            (s_max.max(asum / Number::from(n_d))) / s_max
        };

        (s_d, s_c)
    }

    // --------------------------------------------------------------
    // Trial-side Lagrangian gradient / complementarity — needed by
    // the soft restoration phase's primal-dual error test. Each is a
    // line-for-line analog of the `curr_*` method above, reading the
    // `trial` iterate instead of `curr`.
    // --------------------------------------------------------------

    pub fn trial_jac_c(&self) -> Rc<dyn Matrix> {
        let iv = self.trial_iv();
        self.nlp.borrow_mut().eval_jac_c(&*iv.x)
    }

    pub fn trial_jac_d(&self) -> Rc<dyn Matrix> {
        let iv = self.trial_iv();
        self.nlp.borrow_mut().eval_jac_d(&*iv.x)
    }

    /// `∇_x L` at the trial iterate — analog of [`Self::curr_grad_lag_x`].
    pub fn trial_grad_lag_x(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let grad_f = self.trial_grad_f();
        let jac_c = self.trial_jac_c();
        let jac_d = self.trial_jac_d();

        let mut jc_t = iv.x.make_new();
        jac_c.trans_mult_vector(1.0, &*iv.y_c, 0.0, &mut *jc_t);
        let mut jd_t = iv.x.make_new();
        jac_d.trans_mult_vector(1.0, &*iv.y_d, 0.0, &mut *jd_t);

        let mut tmp = iv.x.make_new();
        tmp.copy(&*grad_f);
        tmp.add_two_vectors(1.0, &*jc_t, 1.0, &*jd_t, 1.0);

        let nlp = self.nlp.borrow();
        nlp.px_l().mult_vector(-1.0, &*iv.z_l, 1.0, &mut *tmp);
        nlp.px_u().mult_vector(1.0, &*iv.z_u, 1.0, &mut *tmp);
        rc_from(tmp)
    }

    /// `∇_s L` at the trial iterate — analog of [`Self::curr_grad_lag_s`].
    pub fn trial_grad_lag_s(&self) -> Rc<dyn Vector> {
        let iv = self.trial_iv();
        let mut tmp = iv.y_d.make_new();
        let nlp = self.nlp.borrow();
        nlp.pd_u().mult_vector(1.0, &*iv.v_u, 0.0, &mut *tmp);
        nlp.pd_l().mult_vector(-1.0, &*iv.v_l, 1.0, &mut *tmp);
        tmp.axpy(-1.0, &*iv.y_d);
        rc_from(tmp)
    }

    pub fn trial_compl_x_l(&self) -> Rc<dyn Vector> {
        Self::calc_compl(&*self.trial_slack_x_l(), &*self.trial_iv().z_l)
    }

    pub fn trial_compl_x_u(&self) -> Rc<dyn Vector> {
        Self::calc_compl(&*self.trial_slack_x_u(), &*self.trial_iv().z_u)
    }

    pub fn trial_compl_s_l(&self) -> Rc<dyn Vector> {
        Self::calc_compl(&*self.trial_slack_s_l(), &*self.trial_iv().v_l)
    }

    pub fn trial_compl_s_u(&self) -> Rc<dyn Vector> {
        Self::calc_compl(&*self.trial_slack_s_u(), &*self.trial_iv().v_u)
    }

    /// `||s ⊙ z − μ||₁` summed over the four complementarity blocks.
    fn relaxed_compl_asum(blocks: &[Rc<dyn Vector>], mu: Number) -> Number {
        let mut acc = 0.0;
        for compl in blocks {
            if compl.dim() == 0 {
                continue;
            }
            let mut r = compl.make_new();
            r.copy(&**compl);
            r.add_scalar(-mu);
            acc += r.asum();
        }
        acc
    }

    /// Unscaled primal-dual KKT system error at the current iterate —
    /// port of
    /// `IpIpoptCalculatedQuantities.cpp:curr_primal_dual_system_error`.
    /// Each block uses the 1-norm scaled by its entry count; the result
    /// is the sum of the dual-infeasibility, primal-infeasibility, and
    /// complementarity terms. Used by the soft restoration phase's
    /// sufficient-reduction test.
    pub fn curr_primal_dual_system_error(&self, mu: Number) -> Number {
        let iv = self.curr_iv();
        let n_dual = iv.x.dim() + iv.s.dim();
        let dual_inf =
            (self.curr_grad_lag_x().asum() + self.curr_grad_lag_s().asum()) / Number::from(n_dual);

        let n_primal = iv.y_c.dim() + iv.y_d.dim();
        let primal_inf = if n_primal > 0 {
            (self.curr_c().asum() + self.curr_d_minus_s().asum()) / Number::from(n_primal)
        } else {
            0.0
        };

        let n_cmpl = iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        let cmpl = if n_cmpl > 0 {
            Self::relaxed_compl_asum(
                &[
                    self.curr_compl_x_l(),
                    self.curr_compl_x_u(),
                    self.curr_compl_s_l(),
                    self.curr_compl_s_u(),
                ],
                mu,
            ) / Number::from(n_cmpl)
        } else {
            0.0
        };

        dual_inf + primal_inf + cmpl
    }

    /// Unscaled primal-dual KKT system error at the trial iterate —
    /// trial-side analog of [`Self::curr_primal_dual_system_error`].
    pub fn trial_primal_dual_system_error(&self, mu: Number) -> Number {
        let iv = self.trial_iv();
        let n_dual = iv.x.dim() + iv.s.dim();
        let dual_inf = (self.trial_grad_lag_x().asum() + self.trial_grad_lag_s().asum())
            / Number::from(n_dual);

        let n_primal = iv.y_c.dim() + iv.y_d.dim();
        let primal_inf = if n_primal > 0 {
            (self.trial_c().asum() + self.trial_d_minus_s().asum()) / Number::from(n_primal)
        } else {
            0.0
        };

        let n_cmpl = iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        let cmpl = if n_cmpl > 0 {
            Self::relaxed_compl_asum(
                &[
                    self.trial_compl_x_l(),
                    self.trial_compl_x_u(),
                    self.trial_compl_s_l(),
                    self.trial_compl_s_u(),
                ],
                mu,
            ) / Number::from(n_cmpl)
        } else {
            0.0
        };

        dual_inf + primal_inf + cmpl
    }

    // --------------------------------------------------------------
    // Damping indicators — `IpIpoptCalculatedQuantities.cpp:1044-1092`.
    //
    //   Tmp_x = P_L · 1 − P_U · 1   (per primal: +1 lower-only,
    //                                 −1 upper-only, 0 two-sided,
    //                                 0 unbounded)
    //   dampind_x_L =  P_L^T · Tmp_x   (1 on lower-only bounds)
    //   dampind_x_U = −P_U^T · Tmp_x   (1 on upper-only bounds)
    // --------------------------------------------------------------

    fn damping_indicators(&self) -> DampingIndicators {
        let nlp = self.nlp.borrow();

        let mut tmp_x_l = nlp.x_l().make_new();
        tmp_x_l.set(1.0);
        let mut tmp_x_u = nlp.x_u().make_new();
        tmp_x_u.set(1.0);
        let mut tmp_x = self.curr_iv().x.make_new();
        nlp.px_l().mult_vector(1.0, &*tmp_x_l, 0.0, &mut *tmp_x);
        nlp.px_u().mult_vector(-1.0, &*tmp_x_u, 1.0, &mut *tmp_x);
        let mut d_x_l = nlp.x_l().make_new();
        nlp.px_l().trans_mult_vector(1.0, &*tmp_x, 0.0, &mut *d_x_l);
        let mut d_x_u = nlp.x_u().make_new();
        nlp.px_u()
            .trans_mult_vector(-1.0, &*tmp_x, 0.0, &mut *d_x_u);

        let mut tmp_s_l = nlp.d_l().make_new();
        tmp_s_l.set(1.0);
        let mut tmp_s_u = nlp.d_u().make_new();
        tmp_s_u.set(1.0);
        let mut tmp_s = self.curr_iv().s.make_new();
        nlp.pd_l().mult_vector(1.0, &*tmp_s_l, 0.0, &mut *tmp_s);
        nlp.pd_u().mult_vector(-1.0, &*tmp_s_u, 1.0, &mut *tmp_s);
        let mut d_s_l = nlp.d_l().make_new();
        nlp.pd_l().trans_mult_vector(1.0, &*tmp_s, 0.0, &mut *d_s_l);
        let mut d_s_u = nlp.d_u().make_new();
        nlp.pd_u()
            .trans_mult_vector(-1.0, &*tmp_s, 0.0, &mut *d_s_u);

        DampingIndicators {
            x_l: rc_from(d_x_l),
            x_u: rc_from(d_x_u),
            s_l: rc_from(d_s_l),
            s_u: rc_from(d_s_u),
        }
    }

    /// `curr_grad_lag_x` plus the `kappa_d · μ · (Px_L · 1 − Px_U · 1)`
    /// damping term on singly-bounded primals — port of
    /// `IpIpoptCalculatedQuantities.cpp:2131-2180`. When `kappa_d == 0`
    /// returns the un-damped gradient.
    pub fn curr_grad_lag_with_damping_x(&self) -> Rc<dyn Vector> {
        if self.kappa_d == 0.0 {
            return self.curr_grad_lag_x();
        }
        let mu = self.data.borrow().curr_mu;
        let di = self.damping_indicators();
        let (d_x_l, d_x_u) = (di.x_l, di.x_u);
        let glx = self.curr_grad_lag_x();
        let mut tmp = glx.make_new();
        tmp.copy(&*glx);
        let nlp = self.nlp.borrow();
        nlp.px_l()
            .mult_vector(self.kappa_d * mu, &*d_x_l, 1.0, &mut *tmp);
        nlp.px_u()
            .mult_vector(-self.kappa_d * mu, &*d_x_u, 1.0, &mut *tmp);
        rc_from(tmp)
    }

    pub fn curr_grad_lag_with_damping_s(&self) -> Rc<dyn Vector> {
        if self.kappa_d == 0.0 {
            return self.curr_grad_lag_s();
        }
        let mu = self.data.borrow().curr_mu;
        let di = self.damping_indicators();
        let (d_s_l, d_s_u) = (di.s_l, di.s_u);
        let gls = self.curr_grad_lag_s();
        let mut tmp = gls.make_new();
        tmp.copy(&*gls);
        let nlp = self.nlp.borrow();
        nlp.pd_l()
            .mult_vector(self.kappa_d * mu, &*d_s_l, 1.0, &mut *tmp);
        nlp.pd_u()
            .mult_vector(-self.kappa_d * mu, &*d_s_u, 1.0, &mut *tmp);
        rc_from(tmp)
    }

    /// `kappa_d · (P_L · damping_l − P_U · damping_u)` in the full x
    /// space — port of `IpIpoptCalculatedQuantities.cpp::grad_kappa_times_damping_x`
    /// (lines 912-949). Unlike `curr_grad_lag_with_damping_x` this does
    /// NOT include `grad_lag_x` and is NOT scaled by `mu`; the centering
    /// RHS in the quality-function oracle multiplies the returned vector
    /// by `-avrg_compl` per upstream `IpQualityFunctionMuOracle.cpp:229`.
    pub fn grad_kappa_times_damping_x(&self) -> Rc<dyn Vector> {
        let mut tmp = self.curr_iv().x.make_new();
        tmp.set(0.0);
        if self.kappa_d > 0.0 {
            let di = self.damping_indicators();
            let nlp = self.nlp.borrow();
            nlp.px_l()
                .mult_vector(self.kappa_d, &*di.x_l, 0.0, &mut *tmp);
            nlp.px_u()
                .mult_vector(-self.kappa_d, &*di.x_u, 1.0, &mut *tmp);
        }
        rc_from(tmp)
    }

    pub fn grad_kappa_times_damping_s(&self) -> Rc<dyn Vector> {
        let mut tmp = self.curr_iv().s.make_new();
        tmp.set(0.0);
        if self.kappa_d > 0.0 {
            let di = self.damping_indicators();
            let nlp = self.nlp.borrow();
            nlp.pd_l()
                .mult_vector(self.kappa_d, &*di.s_l, 0.0, &mut *tmp);
            nlp.pd_u()
                .mult_vector(-self.kappa_d, &*di.s_u, 1.0, &mut *tmp);
        }
        rc_from(tmp)
    }

    // --------------------------------------------------------------
    // Affine (predictor) step helpers — port of upstream
    // `IpIpoptCalculatedQuantities.cpp:CurrAvrgCompl`/`AffMaxAlpha…`
    // used by the Mehrotra probing oracle and the quality-function
    // oracle's σ-search.
    // --------------------------------------------------------------

    /// Max primal step that keeps `s + α · Δs > 0` for the four slack
    /// blocks (x_L, x_U, s_L, s_U), bounded by the fraction-to-the-
    /// boundary parameter `τ ∈ (0, 1]`. Mirrors
    /// `CalcFracToBound` against the projected step `P_L^T Δx`,
    /// `−P_U^T Δx`, `P_L^T Δs`, `−P_U^T Δs`.
    pub fn aff_step_alpha_primal_max(&self, delta_aff: &IteratesVector, tau: Number) -> Number {
        let nlp = self.nlp.borrow();
        let s_x_l = self.curr_slack_x_l();
        let s_x_u = self.curr_slack_x_u();
        let s_s_l = self.curr_slack_s_l();
        let s_s_u = self.curr_slack_s_u();

        // Project Δx / Δs onto each bound subspace with the right sign.
        let mut step_x_l = s_x_l.make_new();
        nlp.px_l()
            .trans_mult_vector(1.0, &*delta_aff.x, 0.0, &mut *step_x_l);
        let mut step_x_u = s_x_u.make_new();
        nlp.px_u()
            .trans_mult_vector(-1.0, &*delta_aff.x, 0.0, &mut *step_x_u);
        let mut step_s_l = s_s_l.make_new();
        nlp.pd_l()
            .trans_mult_vector(1.0, &*delta_aff.s, 0.0, &mut *step_s_l);
        let mut step_s_u = s_s_u.make_new();
        nlp.pd_u()
            .trans_mult_vector(-1.0, &*delta_aff.s, 0.0, &mut *step_s_u);

        s_x_l
            .frac_to_bound(&*step_x_l, tau)
            .min(s_x_u.frac_to_bound(&*step_x_u, tau))
            .min(s_s_l.frac_to_bound(&*step_s_l, tau))
            .min(s_s_u.frac_to_bound(&*step_s_u, tau))
    }

    /// Max dual step that keeps `z + α · Δz > 0` (and same for v).
    pub fn aff_step_alpha_dual_max(&self, delta_aff: &IteratesVector, tau: Number) -> Number {
        let iv = self.curr_iv();
        iv.z_l
            .frac_to_bound(&*delta_aff.z_l, tau)
            .min(iv.z_u.frac_to_bound(&*delta_aff.z_u, tau))
            .min(iv.v_l.frac_to_bound(&*delta_aff.v_l, tau))
            .min(iv.v_u.frac_to_bound(&*delta_aff.v_u, tau))
    }

    /// Per-index activity signs for the four bound blocks, in the fixed
    /// order `(x_L, x_U, s_L, s_U)` — the raw material of the
    /// **phase profile** telemetry.
    ///
    /// Entry `i` is `+1` where the bound multiplier exceeds its slack
    /// and `-1` where it does not. That comparison is the split of
    /// `s_i · z_i ≈ mu` between its two factors: the barrier drives a
    /// strongly active bound to `s = O(mu)` against `z = O(1)` and an
    /// inactive one to the reverse, so the sign separates them by the
    /// method's own geometry rather than by a tolerance anyone chose.
    ///
    /// **Not scale-invariant, deliberately.** A per-variable rescaling
    /// carries the slack by `d` and the multiplier by `d⁻¹`, so the
    /// boundary moves while the product does not; no classifier built
    /// from `(s_i, z_i, mu)` alone can be scale-free, since their only
    /// dimensionless combination is `s_i z_i / mu ≈ 1`. The comparison
    /// is taken in the solver's internal scaled frame — the one in
    /// which `kappa_sigma` and the fraction-to-boundary rule already
    /// compare these quantities — and what it supports is a *within-run*
    /// reading of when the active set stops moving. See
    /// [`pounce_nlp::solve_statistics::IterRecord::active_bounds`] for
    /// the full caveat, and `pounce-sensitivity`'s `classify_activity`
    /// for the scale-invariant question and what it costs.
    ///
    /// Built from `Vector` trait operations alone — no downcast, a
    /// handful of `O(n)` passes — so a caller can afford it per
    /// iteration, but only pays when something consumes the result.
    pub fn bound_activity_signs(&self) -> Vec<Box<dyn Vector>> {
        let iv = self.curr_iv();
        let blocks: [(Rc<dyn Vector>, Rc<dyn Vector>); 4] = [
            (self.curr_slack_x_l(), Rc::clone(&iv.z_l)),
            (self.curr_slack_x_u(), Rc::clone(&iv.z_u)),
            (self.curr_slack_s_l(), Rc::clone(&iv.v_l)),
            (self.curr_slack_s_u(), Rc::clone(&iv.v_u)),
        ];
        blocks
            .iter()
            .map(|(slack, mult)| {
                let mut sign = mult.make_new_copy();
                sign.axpy(-1.0, &**slack);
                sign.element_wise_sgn();
                sign
            })
            .collect()
    }

    /// Predicted average complementarity after the affine step:
    /// `(1/N) · Σ (s + α_pri · Δs) · (z + α_du · Δz)` summed over the
    /// four bound blocks. Returns `0` when there are no bounds.
    pub fn aff_step_compl_avrg(
        &self,
        delta_aff: &IteratesVector,
        alpha_primal: Number,
        alpha_dual: Number,
    ) -> Number {
        let iv = self.curr_iv();
        let n = iv.z_l.dim() + iv.z_u.dim() + iv.v_l.dim() + iv.v_u.dim();
        if n == 0 {
            return 0.0;
        }
        let nlp = self.nlp.borrow();

        // s_X_L_aff = s_X_L + α_pri · P_L^T Δx
        let s_x_l = self.curr_slack_x_l();
        let mut s_x_l_aff = s_x_l.make_new();
        s_x_l_aff.copy(&*s_x_l);
        let mut tmp = s_x_l.make_new();
        nlp.px_l()
            .trans_mult_vector(1.0, &*delta_aff.x, 0.0, &mut *tmp);
        s_x_l_aff.axpy(alpha_primal, &*tmp);
        // z_L_aff = z_L + α_du · Δz_L
        let mut z_l_aff = iv.z_l.make_new();
        z_l_aff.copy(&*iv.z_l);
        z_l_aff.axpy(alpha_dual, &*delta_aff.z_l);
        let mut acc = s_x_l_aff.dot(&*z_l_aff);

        // s_X_U_aff = s_X_U − α_pri · P_U^T Δx
        let s_x_u = self.curr_slack_x_u();
        let mut s_x_u_aff = s_x_u.make_new();
        s_x_u_aff.copy(&*s_x_u);
        let mut tmp = s_x_u.make_new();
        nlp.px_u()
            .trans_mult_vector(-1.0, &*delta_aff.x, 0.0, &mut *tmp);
        s_x_u_aff.axpy(alpha_primal, &*tmp);
        let mut z_u_aff = iv.z_u.make_new();
        z_u_aff.copy(&*iv.z_u);
        z_u_aff.axpy(alpha_dual, &*delta_aff.z_u);
        acc += s_x_u_aff.dot(&*z_u_aff);

        // s_S_L_aff = s_S_L + α_pri · P_dL^T Δs
        let s_s_l = self.curr_slack_s_l();
        let mut s_s_l_aff = s_s_l.make_new();
        s_s_l_aff.copy(&*s_s_l);
        let mut tmp = s_s_l.make_new();
        nlp.pd_l()
            .trans_mult_vector(1.0, &*delta_aff.s, 0.0, &mut *tmp);
        s_s_l_aff.axpy(alpha_primal, &*tmp);
        let mut v_l_aff = iv.v_l.make_new();
        v_l_aff.copy(&*iv.v_l);
        v_l_aff.axpy(alpha_dual, &*delta_aff.v_l);
        acc += s_s_l_aff.dot(&*v_l_aff);

        // s_S_U_aff = s_S_U − α_pri · P_dU^T Δs
        let s_s_u = self.curr_slack_s_u();
        let mut s_s_u_aff = s_s_u.make_new();
        s_s_u_aff.copy(&*s_s_u);
        let mut tmp = s_s_u.make_new();
        nlp.pd_u()
            .trans_mult_vector(-1.0, &*delta_aff.s, 0.0, &mut *tmp);
        s_s_u_aff.axpy(alpha_primal, &*tmp);
        let mut v_u_aff = iv.v_u.make_new();
        v_u_aff.copy(&*iv.v_u);
        v_u_aff.axpy(alpha_dual, &*delta_aff.v_u);
        acc += s_s_u_aff.dot(&*v_u_aff);

        acc / Number::from(n)
    }
}

/// Convenience handle. Mirrors upstream's `SmartPtr<CQ>` flow.
pub type IpoptCqHandle = Rc<RefCell<IpoptCalculatedQuantities>>;

/// Bundle of damping indicators for the four bound spaces — kept
/// internal because `kappa_d == 0` makes them dead in the default
/// configuration.
struct DampingIndicators {
    x_l: Rc<dyn Vector>,
    x_u: Rc<dyn Vector>,
    s_l: Rc<dyn Vector>,
    s_u: Rc<dyn Vector>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipopt_data::IpoptData;
    use crate::iterates_vector::IteratesVector;
    use pounce_common::types::Index;
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};
    use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace};
    use std::rc::Rc as StdRc;

    fn dvec(values: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.values_mut().copy_from_slice(values);
        v
    }

    fn rcv(values: &[Number]) -> Rc<dyn Vector> {
        StdRc::new(dvec(values))
    }

    /// Mock IpoptNlp covering: 2 vars, 1 equality, 1 inequality.
    /// Bounds: x[0] ≥ 0, x[1] ≤ 5, d ≥ 1.
    /// f(x) = x[0]^2 + x[1]^2; ∇f = (2x[0], 2x[1])
    /// c(x) = x[0] + x[1] - 1
    /// d(x) = x[0]
    struct MockNlp {
        x_l: DenseVector,
        x_u: DenseVector,
        d_l: DenseVector,
        d_u: DenseVector,
        px_l: Rc<dyn Matrix>,
        px_u: Rc<dyn Matrix>,
        pd_l: Rc<dyn Matrix>,
        pd_u: Rc<dyn Matrix>,
        // NLP scaling factors. Identity by default; `with_scaling`
        // installs non-trivial ones to exercise the unscaled accessors.
        // (The mock does not actually apply these in `eval_*`; the tests
        // verify the unscaling *arithmetic*, not end-to-end scaling.)
        obj_scale: Number,
        c_scale: Option<Vec<Number>>,
        d_scale: Option<Vec<Number>>,
        // #292: inject a non-finite component into the gradient / constraint
        // Jacobian to exercise the finiteness guard in `curr_nlp_error`.
        nan_grad: bool,
        nan_jac_c: bool,
        empty_jac_c: bool,
        // gh#390: the declared equality RHS the c-block relative measure
        // divides by. `None` (the default) is the "not tracked" contract.
        c_rhs: Option<Vec<Number>>,
        // pounce#476: force `c(x)` to a fixed value so a test can isolate the
        // inequality block (the default `x0 + x1 - 1` is 4 at the fixture's
        // point, which dominates any d-block difference under a max-norm).
        c_override: Option<Number>,
        // gh#812: stand in for `RestoNlp`, whose objective carries the
        // proximity term `ζ/2·‖D_R(x − x_R)‖²` and therefore has a `∇f`
        // that moves with the barrier parameter at fixed `x`. When set,
        // `eval_grad_f` adds `curr_mu` read from this handle — the same
        // coupling, in one line.
        mu_source: Option<IpoptDataHandle>,
    }

    impl MockNlp {
        fn with_c(mut self, v: Number) -> Self {
            self.c_override = Some(v);
            self
        }

        fn with_c_rhs(mut self, rhs: Option<Vec<Number>>) -> Self {
            self.c_rhs = rhs;
            self
        }

        fn with_nan_grad(mut self) -> Self {
            self.nan_grad = true;
            self
        }

        fn with_nan_jac_c(mut self) -> Self {
            self.nan_jac_c = true;
            self
        }

        /// Every variable of the equality row fixed and substituted out, so
        /// the row reduces to the constant `0 = b` — what
        /// `IpoptCalculatedQuantities::row_noise_floor` calls a row no iterate
        /// can move.
        fn with_empty_jac_c(mut self) -> Self {
            self.empty_jac_c = true;
            self
        }

        /// Re-declare the single `d` row as the box `[−mag, +mag]`, so every
        /// bound it has is of the chosen magnitude — the default fixture's
        /// lower bound of `1` would otherwise supply the magnitude by itself.
        /// `d(x) = x0 = 2` sits outside it, violating by `2 − mag`.
        fn with_d_box(mut self, mag: Number) -> Self {
            self.d_l = dvec(&[-mag]);
            self.d_u = dvec(&[mag]);
            self.pd_u = StdRc::new(ExpansionMatrix::new(ExpansionMatrixSpace::new(
                1,
                1,
                &[0],
                0,
            )));
            self
        }

        fn with_scaling(
            mut self,
            obj_scale: Number,
            c_scale: Option<Vec<Number>>,
            d_scale: Option<Vec<Number>>,
        ) -> Self {
            self.obj_scale = obj_scale;
            self.c_scale = c_scale;
            self.d_scale = d_scale;
            self
        }

        fn new() -> Self {
            // x_L holds finite lower bounds; here only x[0] has one (=0).
            let x_l = dvec(&[0.0]);
            // x_U holds finite upper bounds; here only x[1] has one (=5).
            let x_u = dvec(&[5.0]);
            // d has one finite lower bound (d ≥ 1) and no finite upper.
            let d_l = dvec(&[1.0]);
            let d_u = dvec(&[]);

            let px_l_space = ExpansionMatrixSpace::new(2, 1, &[0], 0);
            let px_u_space = ExpansionMatrixSpace::new(2, 1, &[1], 0);
            let pd_l_space = ExpansionMatrixSpace::new(1, 1, &[0], 0);
            let pd_u_space = ExpansionMatrixSpace::new(1, 0, &[], 0);

            Self {
                x_l,
                x_u,
                d_l,
                d_u,
                px_l: StdRc::new(ExpansionMatrix::new(px_l_space)),
                px_u: StdRc::new(ExpansionMatrix::new(px_u_space)),
                pd_l: StdRc::new(ExpansionMatrix::new(pd_l_space)),
                pd_u: StdRc::new(ExpansionMatrix::new(pd_u_space)),
                obj_scale: 1.0,
                c_scale: None,
                d_scale: None,
                nan_grad: false,
                nan_jac_c: false,
                empty_jac_c: false,
                c_rhs: None,
                c_override: None,
                mu_source: None,
            }
        }
    }

    impl crate::ipopt_nlp::Nlp for MockNlp {
        fn n(&self) -> Index {
            2
        }
        fn m_eq(&self) -> Index {
            1
        }
        fn m_ineq(&self) -> Index {
            1
        }
        fn eval_f(&mut self, x: &dyn Vector) -> Number {
            // f(x) = x[0]^2 + x[1]^2
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            xx.values()[0] * xx.values()[0] + xx.values()[1] * xx.values()[1]
        }
        fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
            // grad f = (2 x[0], 2 x[1])
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            let gg = g.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            gg.values_mut()[0] = 2.0 * xx.values()[0];
            gg.values_mut()[1] = 2.0 * xx.values()[1];
            if let Some(d) = self.mu_source.as_ref() {
                let mu = d.borrow().curr_mu;
                gg.values_mut()[0] += mu;
                gg.values_mut()[1] += mu;
            }
            if self.nan_grad {
                gg.values_mut()[0] = Number::NAN;
            }
        }
        fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            let cc = c.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            cc.values_mut()[0] = match self.c_override {
                Some(v) => v,
                None => xx.values()[0] + xx.values()[1] - 1.0,
            };
        }
        fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            let dd = d.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            dd.values_mut()[0] = xx.values()[0];
        }
        fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            if self.empty_jac_c {
                // No entries at all: the row carries no variable.
                let space = GenTMatrixSpace::new(1, 2, vec![], vec![]);
                let mut jac = GenTMatrix::new(space);
                jac.set_values(&[]);
                return StdRc::new(jac);
            }
            // c(x) = x0 + x1 - 1 → Jc = [1, 1] (1×2), nonzeros (1,1),(1,2).
            let space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
            let mut jac = GenTMatrix::new(space);
            if self.nan_jac_c {
                jac.set_values(&[Number::NAN, 1.0]);
            } else {
                jac.set_values(&[1.0, 1.0]);
            }
            StdRc::new(jac)
        }
        fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            // d(x) = x0 → Jd = [1, 0] (1×2), single nonzero (1,1).
            let space = GenTMatrixSpace::new(1, 2, vec![1], vec![1]);
            let mut jac = GenTMatrix::new(space);
            jac.set_values(&[1.0]);
            StdRc::new(jac)
        }
        fn eval_h(
            &mut self,
            _x: &dyn Vector,
            _obj_factor: Number,
            _y_c: &dyn Vector,
            _y_d: &dyn Vector,
        ) -> Rc<dyn SymMatrix> {
            unimplemented!()
        }
    }

    impl IpoptNlp for MockNlp {
        fn x_l(&self) -> &dyn Vector {
            &self.x_l
        }
        fn x_u(&self) -> &dyn Vector {
            &self.x_u
        }
        fn d_l(&self) -> &dyn Vector {
            &self.d_l
        }
        fn d_u(&self) -> &dyn Vector {
            &self.d_u
        }
        fn px_l(&self) -> Rc<dyn Matrix> {
            self.px_l.clone()
        }
        fn px_u(&self) -> Rc<dyn Matrix> {
            self.px_u.clone()
        }
        fn pd_l(&self) -> Rc<dyn Matrix> {
            self.pd_l.clone()
        }
        fn pd_u(&self) -> Rc<dyn Matrix> {
            self.pd_u.clone()
        }
        fn obj_scaling_factor(&self) -> Number {
            self.obj_scale
        }
        fn c_scale_vec(&self) -> Option<Vec<Number>> {
            self.c_scale.clone()
        }
        fn d_scale_vec(&self) -> Option<Vec<Number>> {
            self.d_scale.clone()
        }
        fn declared_c_rhs(&self) -> Option<Vec<Number>> {
            self.c_rhs.clone()
        }
    }

    fn fixture() -> IpoptCalculatedQuantities {
        fixture_with(MockNlp::new())
    }

    fn fixture_with(nlp: MockNlp) -> IpoptCalculatedQuantities {
        fixture_with_x(nlp, &[2.0, 3.0])
    }

    fn fixture_with_x(nlp: MockNlp, x: &[Number]) -> IpoptCalculatedQuantities {
        let mut data = IpoptData::new();
        data.curr_mu = 0.1;
        // Iterate: x as given (2, 3 by default); s = (4); y_c = (1); y_d = (1);
        // z_L = (0.5) [bound on x[0]], z_U = (0.7) [bound on x[1]],
        // v_L = (0.3), v_U = ().
        let iv = IteratesVector::new(
            rcv(x),
            rcv(&[4.0]),
            rcv(&[1.0]),
            rcv(&[1.0]),
            rcv(&[0.5]),
            rcv(&[0.7]),
            rcv(&[0.3]),
            rcv(&[]),
        );
        data.set_curr(iv);
        let data_handle = StdRc::new(RefCell::new(data));
        let nlp: StdRc<RefCell<dyn IpoptNlp>> = StdRc::new(RefCell::new(nlp));
        let mut cq = IpoptCalculatedQuantities::new(data_handle, nlp);
        // Disable damping for clean unit-test expectations.
        cq.kappa_d = 0.0;
        cq
    }

    /// gh#812 — `mu` is part of the `curr_grad_lag_x` cache key, and
    /// removing it is a silent trajectory regression.
    ///
    /// The five vector tags upstream keys this cache on (`x`, `y_c`,
    /// `y_d`, `z_L`, `z_U`) are a complete dependency set only while
    /// `∇f` is a function of `x` alone. It is not during restoration:
    /// `RestoNlp`'s proximity term scales with `ζ(mu)`, so its `∇f`
    /// moves while every one of those five tags stands still. A cache
    /// that misses `mu` then hands back the pre-update gradient — an
    /// answer that is self-consistent, converges, and reports the
    /// right objective, while taking a measurably worse route: drop
    /// `mu` from the key and `scripts/sweep-fixtures.sh` moves 8 of
    /// 154 fixture-legs, `pooling_rt2stp` 295 → 627 iterations on the
    /// lbfgs leg.
    ///
    /// MUTATION CHECK: delete `&[mu]` from the `get`/`add` pair in
    /// `curr_grad_lag_x` and this test fails — the second read returns
    /// the first read's vector unchanged.
    #[test]
    fn grad_lag_x_cache_reruns_when_only_mu_moves() {
        let mut data = IpoptData::new();
        data.curr_mu = 0.1;
        data.set_curr(IteratesVector::new(
            rcv(&[2.0, 3.0]),
            rcv(&[4.0]),
            rcv(&[1.0]),
            rcv(&[1.0]),
            rcv(&[0.5]),
            rcv(&[0.7]),
            rcv(&[0.3]),
            rcv(&[]),
        ));
        let data_handle = StdRc::new(RefCell::new(data));
        let mut nlp = MockNlp::new();
        nlp.mu_source = Some(StdRc::clone(&data_handle));
        let nlp: StdRc<RefCell<dyn IpoptNlp>> = StdRc::new(RefCell::new(nlp));
        let mut cq = IpoptCalculatedQuantities::new(StdRc::clone(&data_handle), nlp);
        cq.kappa_d = 0.0;

        let before = dense_vals(&cq.curr_grad_lag_x());
        // A repeat read at unchanged `mu` must hit the cache and agree
        // exactly — otherwise the test below proves nothing about the
        // key and everything about a non-deterministic mock.
        assert_eq!(before, dense_vals(&cq.curr_grad_lag_x()));

        // Move ONLY `mu`. Every iterate vector — and so every one of
        // the five tags upstream keys on — is untouched.
        data_handle.borrow_mut().curr_mu = 0.5;
        let after = dense_vals(&cq.curr_grad_lag_x());

        // The mock adds `mu` to both gradient components, so the whole
        // Lagrangian gradient shifts by exactly the change in `mu`.
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (a - b - 0.4).abs() < 1e-12,
                "grad_lag_x did not follow mu: {b} -> {a}, expected +0.4"
            );
        }
    }

    fn dense_vals(v: &Rc<dyn Vector>) -> Vec<Number> {
        v.as_any()
            .downcast_ref::<DenseVector>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn slack_x_lower_is_x0_minus_x_l() {
        // P_L^T x = [x[0]] = [2]; x_L = [0]; slack = 2 - 0 = 2.
        let cq = fixture();
        assert_eq!(dense_vals(&cq.curr_slack_x_l()), vec![2.0]);
    }

    #[test]
    fn slack_x_upper_is_x_u_minus_x1() {
        // x_U = [5]; P_U^T x = [3]; slack = 5 - 3 = 2.
        let cq = fixture();
        assert_eq!(dense_vals(&cq.curr_slack_x_u()), vec![2.0]);
    }

    #[test]
    fn slack_s_lower() {
        // d_L = [1]; P_L^T s = [4]; slack = 4 - 1 = 3.
        let cq = fixture();
        assert_eq!(dense_vals(&cq.curr_slack_s_l()), vec![3.0]);
    }

    #[test]
    fn grad_f_is_twice_x() {
        let cq = fixture();
        assert_eq!(dense_vals(&cq.curr_grad_f()), vec![4.0, 6.0]);
    }

    #[test]
    fn compl_x_l_is_slack_times_z() {
        // slack_x_L = [2]; z_L = [0.5]; compl = [1.0]
        let cq = fixture();
        assert_eq!(dense_vals(&cq.curr_compl_x_l()), vec![1.0]);
    }

    #[test]
    fn relaxed_compl_x_l_subtracts_mu() {
        // compl = 1.0; mu = 0.1; relaxed = 0.9.
        let cq = fixture();
        assert!((dense_vals(&cq.curr_relaxed_compl_x_l())[0] - 0.9).abs() < 1e-15);
    }

    #[test]
    fn sigma_x_routes_z_over_slack_through_p() {
        // P_L lifts (z_L/s_L) = (0.5/2 = 0.25) into x[0] slot.
        // P_U lifts (z_U/s_U) = (0.7/2 = 0.35) into x[1] slot.
        // sigma = (0.25, 0.35)
        let cq = fixture();
        let s = dense_vals(&cq.curr_sigma_x());
        assert!((s[0] - 0.25).abs() < 1e-15);
        assert!((s[1] - 0.35).abs() < 1e-15);
    }

    /// gh#655 fixture. `x_L[0] = 0`, so the lower-bound block of `x` carries
    /// slack `x0` against multiplier `z_l`, at barrier parameter `mu`. The
    /// rest of the iterate is the default fixture's.
    fn fixture_at_mu(x0: Number, z_l: Number, mu: Number) -> IpoptCalculatedQuantities {
        let mut data = IpoptData::new();
        data.curr_mu = mu;
        let iv = IteratesVector::new(
            rcv(&[x0, 3.0]),
            rcv(&[4.0]),
            rcv(&[1.0]),
            rcv(&[1.0]),
            rcv(&[z_l]),
            rcv(&[0.7]),
            rcv(&[0.3]),
            rcv(&[]),
        );
        data.set_curr(iv);
        let data_handle = StdRc::new(RefCell::new(data));
        let nlp: StdRc<RefCell<dyn IpoptNlp>> = StdRc::new(RefCell::new(MockNlp::new()));
        let mut cq = IpoptCalculatedQuantities::new(data_handle, nlp);
        cq.kappa_d = 0.0;
        cq
    }

    /// gh#655. The reported point, verbatim: `mu = 9.0909e-308`, a subnormal
    /// slack of `2.0202e-308` against `z = 4.5`, reached under a
    /// `SolveSucceeded`. The old floor never even fired here — `eps*mu` is
    /// `2.0e-323`, still a representable subnormal rather than the `0` that
    /// would have substituted `f64::MIN_POSITIVE`, so the slack cleared the
    /// threshold untouched and `4.5 / 2.0202e-308 = 2.2e308` overflowed.
    #[test]
    fn subnormal_slack_does_not_overflow_sigma() {
        let mu: Number = 9.0909e-308;
        let slack: Number = 2.0202e-308;
        let z: Number = 4.5;
        // The premise: the barrier-side threshold does not catch this point.
        assert!(f64::EPSILON * mu.min(1.0) > 0.0);
        assert!(slack > f64::EPSILON * mu.min(1.0));
        assert!(!(z / slack).is_finite());

        let cq = fixture_at_mu(slack, z, mu);
        let s = dense_vals(&cq.curr_sigma_x());
        assert!(s[0].is_finite(), "Sigma_x[0] = {} is not finite", s[0]);
        // Floored at z/(MAX/4), so the ratio lands at MAX/4 at worst.
        assert!(s[0] <= f64::MAX / SIGMA_OVERFLOW_HEADROOM);
        // The slack itself was raised to the floor, not to f64::MIN_POSITIVE.
        assert!(dense_vals(&cq.curr_slack_x_l())[0] >= z / f64::MAX);
        // The untouched upper block still reads (5 - 3) against z_U = 0.7.
        assert!((s[1] - 0.35).abs() < 1e-15);
    }

    /// gh#655, the half the trigger alone does not cover: a multiplier large
    /// enough that the bound-move cap (`slack_move*max(1,|bound|) + slack`)
    /// sits *below* the representability floor. Capping there would hand back
    /// a slack that still overflows, so the floor is re-applied after the cap.
    #[test]
    fn representability_floor_survives_the_bound_move_cap() {
        let cq = fixture_at_mu(1e-300, 1e300, 1e-8);
        // Premise: the cap really is the binding constraint here.
        assert!(cq.slack_move * 1.0 + 1e-300 < 1e300 / (f64::MAX / SIGMA_OVERFLOW_HEADROOM));
        let s = dense_vals(&cq.curr_sigma_x());
        assert!(s[0].is_finite(), "Sigma_x[0] = {} is not finite", s[0]);
        assert!(s[0] <= f64::MAX / SIGMA_OVERFLOW_HEADROOM);
    }

    /// The floor is `z_max/4.5e307`; a slack twelve orders of magnitude above
    /// anything subnormal is nowhere near it, and must come back bit-identical
    /// — the correction is meant to be invisible off the overflow edge.
    #[test]
    fn ordinary_small_slack_is_left_exactly_alone() {
        let cq = fixture_at_mu(1e-20, 0.5, 1e-8);
        assert_eq!(dense_vals(&cq.curr_slack_x_l()), vec![1e-20]);
        assert_eq!(dense_vals(&cq.curr_sigma_x())[0], 0.5 / 1e-20);
    }

    #[test]
    fn sigma_s_lower_only() {
        // P_L lifts (v_L/s_L) = (0.3/3 = 0.1).
        let cq = fixture();
        let s = dense_vals(&cq.curr_sigma_s());
        assert!((s[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn avrg_compl_averages_over_active_bounds() {
        // z_L·s_L + z_U·s_U + v_L·s_s_L + v_U·s_s_U
        // = 0.5*2 + 0.7*2 + 0.3*3 + 0
        // = 1 + 1.4 + 0.9 = 3.3
        // N = 1 + 1 + 1 + 0 = 3 → 1.1
        let cq = fixture();
        assert!((cq.curr_avrg_compl() - 1.1).abs() < 1e-15);
    }

    #[test]
    fn complementarity_min_takes_min_over_active_pairs() {
        // compl entries: z_L·s_L=1.0, z_U·s_U=1.4, v_L·s_s_L=0.9.
        // v_U is empty (skipped). Min = 0.9.
        let cq = fixture();
        assert!((cq.curr_complementarity_min() - 0.9).abs() < 1e-15);
    }

    #[test]
    fn centrality_measure_is_min_over_avrg() {
        // min/avrg = 0.9 / 1.1 ≈ 0.81818…
        let cq = fixture();
        let xi = cq.curr_centrality_measure();
        assert!((xi - 0.9 / 1.1).abs() < 1e-15);
    }

    #[test]
    fn curr_f_evaluates_objective() {
        // f(x) = x[0]^2 + x[1]^2 at x = (2, 3) → 4 + 9 = 13.
        let cq = fixture();
        assert!((cq.curr_f() - 13.0).abs() < 1e-15);
    }

    #[test]
    fn curr_barrier_obj_subtracts_mu_log_slacks() {
        // f = 13; slacks = (s_x_L=2, s_x_U=2, s_s_L=3, s_s_U=∅).
        // log_sum = ln 2 + ln 2 + ln 3 + 0 = 2 ln 2 + ln 3.
        // mu = 0.1 → phi = 13 - 0.1*(2 ln 2 + ln 3).
        let cq = fixture();
        let expected = 13.0 - 0.1 * (2.0 * 2.0_f64.ln() + 3.0_f64.ln());
        assert!((cq.curr_barrier_obj() - expected).abs() < 1e-13);
    }

    /// pounce#476. `inf_pr_output = original` (upstream's default) must report
    /// the violation of the **original** rows, not of the internal slack
    /// reformulation. The fixture is exactly the case that made the two
    /// diverge on Mittelmann's `robot_a`: `d(x) = 2` against `d >= 1` — the
    /// original row is *satisfied*, so the original-NLP violation is 0 — while
    /// the slack has drifted to `s = 4`, so `|d − s| = 2` and the internal
    /// measure reads 2. Reporting the internal number made feasible iterates
    /// look badly infeasible (2.79e4 where Ipopt printed 0.00e+00).
    ///
    /// The equality row is genuinely violated (`c = 4`), and both measures
    /// must still see it — the fix must not swallow real infeasibility.
    #[test]
    fn original_nlp_violation_ignores_slack_drift_but_not_a_violated_row() {
        let cq = fixture();
        // Internal: max(|c|, |d − s|) = max(4, 2) = 4.
        assert_eq!(cq.curr_primal_infeasibility_max(), 4.0);
        // Original: max(|c|, dist(d, [d_l, d_u])) = max(4, 0) = 4 — the
        // equality violation survives, the slack drift does not contribute.
        assert_eq!(cq.curr_unscaled_nlp_constraint_violation_max(), 4.0);
    }

    /// The other half of pounce#476: with the equality block satisfied, the
    /// two measures disagree outright — internal still sees the slack drift,
    /// original sees a feasible point.
    #[test]
    fn original_nlp_violation_is_zero_when_only_the_slack_has_drifted() {
        let cq = fixture_with(MockNlp::new().with_c(0.0));
        assert_eq!(cq.curr_primal_infeasibility_max(), 2.0);
        assert_eq!(cq.curr_unscaled_nlp_constraint_violation_max(), 0.0);
    }

    /// …and the `inf_pr` column must actually be *wired* to the right one.
    /// The two tests above pass whichever accessor `OrigIterationOutput`
    /// picks, so without this the exact regression — a one-line match arm
    /// reaching for `curr_primal_infeasibility_max` — goes unnoticed.
    /// `InfPrTag::Original` is upstream's default, so this is what the column
    /// prints unless the user asks for `internal`.
    #[test]
    fn inf_pr_column_prints_the_original_violation_under_the_default_tag() {
        use crate::ipopt_data::IpoptData;
        use crate::output::orig::{InfPrTag, OrigIterationOutput};
        use crate::output::r#trait::IterationOutput;

        // Slack drift only: internal reads 2, original reads 0.
        let cq: IpoptCqHandle = Rc::new(RefCell::new(fixture_with(MockNlp::new().with_c(0.0))));
        let data: IpoptDataHandle = Rc::new(RefCell::new(IpoptData::default()));

        let field = |tag| {
            let mut out = OrigIterationOutput::new();
            out.inf_pr_output = tag;
            // Column 2 of the row is `inf_pr` (iter, objective, inf_pr, …).
            out.format_row(&data, &cq)
                .split_whitespace()
                .nth(2)
                .unwrap()
                .to_string()
        };
        assert_eq!(field(InfPrTag::Original), "0.00e+00");
        assert_eq!(field(InfPrTag::Internal), "2.00e+00");
    }

    #[test]
    fn curr_constraint_violation_is_one_norm() {
        // c(x) = x[0]+x[1]-1 = 4 ⇒ |c| = 4.
        // d(x)=x[0]=2; s=4 ⇒ d-s = -2 ⇒ |d-s| = 2.
        // theta = 4 + 2 = 6.
        let cq = fixture();
        assert!((cq.curr_constraint_violation() - 6.0).abs() < 1e-13);
    }

    /// gh#390. The fixture's equality row is `x0 + x1 == 1` at `x = (2, 3)`,
    /// so `c = 4`. Judged against a declared RHS of 2 that is a 200% violation
    /// — and it is 200% however the row is written, which is the point.
    #[test]
    fn relative_c_infeasibility_is_residual_over_declared_rhs() {
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![2.0])));
        assert_eq!(cq.relative_c_infeasibility_max(), 2.0);
        // The fixture's inequality row (`d = 2` against `d >= 1`) is satisfied,
        // so the combined measure is the equality block's verdict.
        assert_eq!(cq.relative_d_infeasibility_max(), 0.0);
        assert_eq!(cq.curr_relative_primal_infeasibility_max(), 2.0);
    }

    /// An NLP that does not track the pre-fold RHS (the trait default, e.g.
    /// the restoration NLP) must abstain rather than invent a magnitude.
    #[test]
    fn relative_c_infeasibility_abstains_without_declared_rhs() {
        let cq = fixture();
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
        assert_eq!(cq.curr_relative_primal_infeasibility_max(), 0.0);
    }

    /// A homogeneous row (`g(x) == 0`) has no declared magnitude and needs
    /// none — `s·g(x) == 0` is the same row at every `s`. Dividing by its zero
    /// RHS would report every float-noise residual as an infinite violation.
    #[test]
    fn relative_c_infeasibility_abstains_on_homogeneous_row() {
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![0.0])));
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
    }

    /// An unjudgeable row must not fabricate a relative verdict.
    #[test]
    fn relative_c_infeasibility_abstains_on_non_finite_rhs() {
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![Number::INFINITY])));
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![Number::NAN])));
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
    }

    /// gh #446. "Homogeneous" has to be judged numerically. The fixture's row
    /// is `x0 + x1 == b` at `x = (2, 3)`, so its noise floor is
    /// `ROW_NOISE_KAPPA · eps · 1 · 3 ≈ 4.3e-14`: an RHS under that is
    /// rounding residue — a converter writing `2^-53` where the model says
    /// `0` — and the row must abstain exactly as a declared zero does. Above
    /// the floor the RHS is real data and is judged, however small.
    #[test]
    fn relative_c_infeasibility_abstains_on_rhs_below_the_row_noise_floor() {
        let floor = ROW_NOISE_KAPPA * Number::EPSILON * 3.0;
        // The QSCSD1 value: an RHS of exactly one machine epsilon.
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![Number::EPSILON])));
        assert!(Number::EPSILON < floor);
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
        // Just above the floor the row still carries a magnitude, and a
        // residual of 4 against it is judged on its merits.
        let rhs = 2.0 * floor;
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![rhs])));
        assert_eq!(cq.relative_c_infeasibility_max(), 4.0 / rhs);
    }

    /// gh #446. Every variable of the row fixed and substituted out leaves
    /// `0 = b`, which no iterate can move — a statement about the model, for
    /// presolve to certify, not a residual to judge an iterate by. QPILOTNO's
    /// row 150 reduces to `0 = −2.22e-16` this way and pinned the relative
    /// measure at 100% for the entire run.
    #[test]
    fn relative_c_infeasibility_abstains_on_a_row_no_iterate_can_move() {
        let cq = fixture_with(
            MockNlp::new()
                .with_empty_jac_c()
                .with_c_rhs(Some(vec![Number::EPSILON])),
        );
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
        // Not a licence to ignore a real one: the absolute `constr_viol_tol`
        // arm still sees the row, and it is what governs here.
        assert_eq!(cq.curr_primal_infeasibility_max(), 4.0);
    }

    /// gh #446. The inequality block draws its magnitude from the declared
    /// bounds, and needs the same numeric reading of "zero" — QPILOTNO carries
    /// 43 bounds at `1e-17`–`1e-15`. `d(x) = x0 = 2` against an upper bound
    /// under the row's noise floor is 2e14 times its magnitude by the old
    /// arithmetic, and unjudgeable by the new.
    #[test]
    fn relative_d_infeasibility_abstains_on_bound_below_the_row_noise_floor() {
        let floor = ROW_NOISE_KAPPA * Number::EPSILON * 3.0;
        let cq = fixture_with(MockNlp::new().with_d_box(Number::EPSILON));
        assert_eq!(cq.relative_d_infeasibility_max(), 0.0);
        // A bound above the floor is real, and `d = 2` violates it hugely.
        let bound = 2.0 * floor;
        let cq = fixture_with(MockNlp::new().with_d_box(bound));
        assert_eq!(cq.relative_d_infeasibility_max(), (2.0 - bound) / bound);
    }

    /// The floor tracks `‖x‖_∞` **deliberately**, and this pins it. `x` is one
    /// vector produced by a linear solve with norm-wise backward error, so a
    /// large variable anywhere really does coarsen how finely every other
    /// component can be placed — and a declared magnitude finer than that is a
    /// target no iterate could hit. The per-row alternative, `Σ_j |a_ij x_j|`
    /// via `|J|·|x|`, looks more precise and measures the wrong thing (a row's
    /// *evaluation* error, not what limits its residual); it was implemented
    /// and it regressed QETAMACR, QSCORPIO and QPILOTNO of gh #446's 15. Re-run
    /// those three before changing this.
    #[test]
    fn row_noise_floor_tracks_the_iterate_norm() {
        // `d(x) = x0` against a declared box of ±1e-9.
        let bound = 1e-9;
        // At ‖x‖_∞ = 3 the floor is ~4.3e-14: the bound is real data, judged.
        let cq = fixture_with(MockNlp::new().with_d_box(bound));
        assert_eq!(cq.relative_d_infeasibility_max(), (2.0 - bound) / bound);
        // At ‖x‖_∞ = 1e8 the floor is ~1.4e-6 and the same bound is finer than
        // the iterate can be resolved, so the row abstains.
        let cq = fixture_with_x(MockNlp::new().with_d_box(bound), &[2.0, 1e8]);
        assert_eq!(cq.relative_d_infeasibility_max(), 0.0);
    }

    /// A row-count mismatch means the RHS does not describe this `c` block;
    /// pairing them up anyway would judge rows against other rows' magnitudes.
    #[test]
    fn relative_c_infeasibility_abstains_on_length_mismatch() {
        let cq = fixture_with(MockNlp::new().with_c_rhs(Some(vec![2.0, 2.0])));
        assert_eq!(cq.relative_c_infeasibility_max(), 0.0);
    }

    #[test]
    fn grad_barrier_obj_x_subtracts_mu_inv_slack() {
        // grad_f = (4, 6).
        // P_L lifts -mu*(1/s_x_L) = -0.1*(1/2)=-0.05 into x[0].
        // P_U lifts +mu*(1/s_x_U) = +0.1*(1/2)=+0.05 into x[1].
        // result = (4 - 0.05, 6 + 0.05) = (3.95, 6.05).
        let cq = fixture();
        let g = dense_vals(&cq.curr_grad_barrier_obj_x());
        assert!((g[0] - 3.95).abs() < 1e-13);
        assert!((g[1] - 6.05).abs() < 1e-13);
    }

    #[test]
    fn grad_lag_s_is_minus_y_d_minus_pl_v_l_plus_pu_v_u() {
        // tmp = P_U v_U = (zero-dim contrib) → 0
        // tmp -= P_L v_L → tmp = -[0.3]
        // tmp -= y_d = -[0.3] - [1.0] = [-1.3]
        let cq = fixture();
        assert!((dense_vals(&cq.curr_grad_lag_s())[0] + 1.3).abs() < 1e-15);
    }

    fn zero_iv_like(iv: &IteratesVector) -> IteratesVector {
        // Materialize explicit zeros for every component so the
        // affine-step tests can compose direct-sum updates.
        IteratesVector::new(
            rcv(&vec![0.0; iv.x.dim() as usize]),
            rcv(&vec![0.0; iv.s.dim() as usize]),
            rcv(&vec![0.0; iv.y_c.dim() as usize]),
            rcv(&vec![0.0; iv.y_d.dim() as usize]),
            rcv(&vec![0.0; iv.z_l.dim() as usize]),
            rcv(&vec![0.0; iv.z_u.dim() as usize]),
            rcv(&vec![0.0; iv.v_l.dim() as usize]),
            rcv(&vec![0.0; iv.v_u.dim() as usize]),
        )
    }

    #[test]
    fn aff_step_compl_avrg_with_zero_step_matches_curr_avrg_compl() {
        // Δ_aff = 0 ⇒ predicted compl ≡ current compl.
        // s_X_L · z_L = 2·0.5=1, s_X_U·z_U=2·0.7=1.4, s_S_L·v_L=3·0.3=0.9.
        // Total = 3.3; N = 3 (z_l + z_u + v_l, v_u empty); avrg = 1.1.
        let cq = fixture();
        let iv = cq.curr_iv();
        let zero = zero_iv_like(&iv);
        let m = cq.aff_step_compl_avrg(&zero, 1.0, 1.0);
        assert!((m - 1.1).abs() < 1e-13);
        assert!((cq.curr_avrg_compl() - 1.1).abs() < 1e-13);
    }

    #[test]
    fn aff_step_compl_avrg_responds_to_primal_step() {
        // Δ_aff.x = (1, 0), α_pri = 1, others = 0.
        // s_X_L_aff = 2 + 1·1 = 3; s_X_U_aff = 2 (P_U^T·dx = 0); s_S_L_aff = 3.
        // (3·0.5 + 2·0.7 + 3·0.3) / 3 = (1.5 + 1.4 + 0.9) / 3 = 1.2667.
        let cq = fixture();
        let iv = cq.curr_iv();
        let mut z = zero_iv_like(&iv);
        z.x = rcv(&[1.0, 0.0]);
        let m = cq.aff_step_compl_avrg(&z, 1.0, 1.0);
        assert!((m - 1.2666666666666666).abs() < 1e-13);
    }

    #[test]
    fn aff_step_alpha_primal_truncates_to_x_lower_bound() {
        // Δ_aff.x = (-3, 0); s_X_L = 2; tau = 1 ⇒ α_max = 2/3.
        let cq = fixture();
        let iv = cq.curr_iv();
        let mut z = zero_iv_like(&iv);
        z.x = rcv(&[-3.0, 0.0]);
        let a = cq.aff_step_alpha_primal_max(&z, 1.0);
        assert!((a - 2.0 / 3.0).abs() < 1e-13);
    }

    #[test]
    fn aff_step_alpha_dual_truncates_to_z_lower_bound() {
        // Δ_aff.z_L = (-1); z_L = 0.5; tau = 1 ⇒ α_max = 0.5.
        let cq = fixture();
        let iv = cq.curr_iv();
        let mut z = zero_iv_like(&iv);
        z.z_l = rcv(&[-1.0]);
        let a = cq.aff_step_alpha_dual_max(&z, 1.0);
        assert!((a - 0.5).abs() < 1e-13);
    }

    #[test]
    fn grad_barr_t_delta_dots_barrier_grads_with_step() {
        // ∇_x φ = (3.95, 6.05); ∇_s φ = (-mu/s_s_L) = -0.1/3 ≈ -0.03333…
        // δx = (1, 2); δs = (3): result = 3.95·1 + 6.05·2 + (-0.0333…)·3
        //                              = 3.95 + 12.10 − 0.1 = 15.95.
        let cq = fixture();
        let dx = dvec(&[1.0, 2.0]);
        let ds = dvec(&[3.0]);
        let r = cq.curr_grad_barr_t_delta(&dx, &ds);
        let expected = 3.95 + 12.10 - 0.1;
        assert!((r - expected).abs() < 1e-13, "r = {r}");
    }

    #[test]
    fn dwd_with_no_w_collapses_to_sigma_quadratic() {
        // W is None in the fixture (no Hessian seeded), perts default to 0.
        // σ_x = (0.25, 0.35); σ_s = (0.1).
        // δx = (2, -1); δs = (3) ⇒ dWd = 0.25·4 + 0.35·1 + 0.1·9
        //                              = 1.00 + 0.35 + 0.90 = 2.25.
        let cq = fixture();
        let dx = dvec(&[2.0, -1.0]);
        let ds = dvec(&[3.0]);
        let r = cq.curr_dwd(&dx, &ds);
        assert!((r - 2.25).abs() < 1e-13, "r = {r}");
    }

    #[test]
    fn dwd_includes_pd_perturbations() {
        // Without perts: dWd = 0.25·4 + 0.35·1 + 0.1·9 = 2.25.
        // δ_pert_x = 0.5, δ_pert_s = 0.25:
        //   add δ_pert_x · ‖δx‖² + δ_pert_s · ‖δs‖²
        //     = 0.5·(4+1) + 0.25·9 = 2.5 + 2.25 = 4.75.
        // Total = 7.00.
        let cq = fixture();
        {
            let mut d = cq.data.borrow_mut();
            d.perturbations.delta_x = 0.5;
            d.perturbations.delta_s = 0.25;
        }
        let dx = dvec(&[2.0, -1.0]);
        let ds = dvec(&[3.0]);
        let r = cq.curr_dwd(&dx, &ds);
        assert!((r - 7.00).abs() < 1e-13, "r = {r}");
    }

    // ---- #292: NaN gradient / Jacobian must not launder to a finite KKT error

    #[test]
    fn nlp_error_is_finite_for_a_finite_iterate() {
        // Baseline: the well-formed fixture produces a finite, positive KKT
        // error (this iterate is not a KKT point). The finiteness guard added
        // for #292 must not perturb this normal path.
        let cq = fixture();
        let err = cq.curr_nlp_error();
        assert!(err.is_finite() && err > 0.0, "err = {err}");
    }

    #[test]
    fn nlp_error_is_non_finite_when_gradient_has_nan() {
        // A NaN gradient component reaches ∇_x L, whose max-norm (`amax`)
        // silently drops NaN and would launder the dual infeasibility to a
        // finite value → bogus `Solve_Succeeded` (#292). `curr_nlp_error` must
        // instead surface a non-finite error so the caller's
        // `!nlp_err.is_finite()` guard fires `Invalid_Number_Detected`.
        let cq = fixture_with(MockNlp::new().with_nan_grad());
        assert!(
            !cq.curr_nlp_error().is_finite(),
            "NaN gradient laundered to finite KKT error: {}",
            cq.curr_nlp_error()
        );
    }

    #[test]
    fn nlp_error_is_non_finite_when_constraint_jacobian_has_nan() {
        // A NaN in the constraint Jacobian enters ∇_x L through the Jᵀy term
        // and is likewise laundered by `amax` on the fixture's nonzero
        // multipliers. Must read as a non-finite KKT error, not `Optimal`.
        let cq = fixture_with(MockNlp::new().with_nan_jac_c());
        assert!(
            !cq.curr_nlp_error().is_finite(),
            "NaN constraint Jacobian laundered to finite KKT error: {}",
            cq.curr_nlp_error()
        );
    }

    // ---- Unscaled (user-space) KKT residuals — pounce#173 -------------

    #[test]
    fn unscaled_dual_inf_is_scaled_over_df() {
        // df = 2: every Lagrangian-gradient term carries the objective
        // factor, so the unscaled dual infeasibility is the scaled one
        // divided by df.
        let cq = fixture_with(MockNlp::new().with_scaling(2.0, None, None));
        let scaled = cq.curr_dual_infeasibility_max();
        let unscaled = cq.curr_unscaled_dual_infeasibility_max();
        assert!(scaled > 0.0, "fixture should have nonzero dual inf");
        assert!(
            (unscaled - scaled / 2.0).abs() < 1e-12,
            "unscaled {unscaled} != scaled/df {}",
            scaled / 2.0
        );
    }

    /// gh #532. The dual *scale* is the largest single term `∇L` is assembled
    /// from, so the strict gate can ask what fraction of those terms failed to
    /// cancel instead of comparing a residual against an absolute constant.
    #[test]
    fn dual_inf_scale_is_the_largest_lagrangian_term() {
        // Fixture at x = (2, 3): ∇f = (4, 6); J_cᵀ y_c = (1, 1); J_dᵀ y_d =
        // (1, 0); y_d = 1; P_L z_L = 0.5; P_U z_U = 0.7; P_L v_L = 0.3; v_U is
        // empty. The largest is ‖∇f‖_∞ = 6.
        let cq = fixture();
        assert_eq!(cq.curr_dual_infeasibility_scale_max(), 6.0);
        // No scaling → the unscaled accessor is the identity, as for every
        // other residual on the common path.
        assert_eq!(
            cq.curr_unscaled_dual_infeasibility_scale_max(),
            cq.curr_dual_infeasibility_scale_max()
        );
    }

    /// The scale unscales exactly as the residual it is the scale of: every
    /// term of the scaled Lagrangian gradient carries `df`, so both are the
    /// scaled value over `|df|`. If the two ever divided differently the ratio
    /// the strict gate tests would silently pick up a factor of `df`.
    #[test]
    fn unscaled_dual_inf_scale_is_scaled_over_df() {
        let cq = fixture_with(MockNlp::new().with_scaling(2.0, None, None));
        assert_eq!(cq.curr_unscaled_dual_infeasibility_scale_max(), 3.0);
        // A negative factor is the documented way to pose a maximization; a
        // max-norm has no business coming back negative (the sign trap that
        // defeated the unscaled dual residual gate).
        let neg = fixture_with(MockNlp::new().with_scaling(-2.0, None, None));
        assert_eq!(neg.curr_unscaled_dual_infeasibility_scale_max(), 3.0);
    }

    #[test]
    fn unscaled_residuals_are_identity_without_scaling() {
        // df = 1, no row scaling → unscaled accessors return exactly the
        // scaled values (the common no-scaling path).
        let cq = fixture();
        assert_eq!(
            cq.curr_unscaled_dual_infeasibility_max(),
            cq.curr_dual_infeasibility_max()
        );
        assert_eq!(
            cq.curr_unscaled_complementarity_max(),
            cq.curr_complementarity_max()
        );
        assert_eq!(
            cq.curr_unscaled_primal_infeasibility_max(),
            cq.curr_primal_infeasibility_max()
        );
    }

    #[test]
    fn unscaled_compl_is_scaled_over_df() {
        let cq = fixture_with(MockNlp::new().with_scaling(2.0, None, None));
        let scaled = cq.curr_complementarity_max();
        let unscaled = cq.curr_unscaled_complementarity_max();
        assert!(scaled > 0.0);
        assert!((unscaled - scaled / 2.0).abs() < 1e-12);
    }

    #[test]
    fn unscaled_primal_divides_each_row_by_its_factor() {
        // Fixture residuals: c = x0+x1-1 = 4; d-s = x0 - s = 2 - 4 = -2.
        // Scaled max-norm primal = max(|4|, |-2|) = 4.
        // With dc = [4], dd = [2]: unscaled = max(|4/4|, |-2/2|) = 1.
        let cq = fixture_with(MockNlp::new().with_scaling(1.0, Some(vec![4.0]), Some(vec![2.0])));
        assert!((cq.curr_primal_infeasibility_max() - 4.0).abs() < 1e-12);
        assert!(
            (cq.curr_unscaled_primal_infeasibility_max() - 1.0).abs() < 1e-12,
            "got {}",
            cq.curr_unscaled_primal_infeasibility_max()
        );
    }

    #[test]
    fn unscaled_nlp_error_is_max_of_unscaled_components() {
        let cq = fixture_with(MockNlp::new().with_scaling(2.0, Some(vec![4.0]), Some(vec![2.0])));
        let expected = cq
            .curr_unscaled_dual_infeasibility_max()
            .max(cq.curr_unscaled_primal_infeasibility_max())
            .max(cq.curr_unscaled_complementarity_max());
        assert_eq!(cq.curr_unscaled_nlp_error(), expected);
    }

    /// gh #528. A component at or below its own floor drops out; everything
    /// above it is counted in full, not net of the floor — the question the
    /// floor answers is whether the row says anything at all.
    #[test]
    fn amax_above_floor_drops_only_sub_floor_components() {
        let v = dvec(&[1e-9, -3e-7, 5e-3]);
        assert_eq!(amax_above_floor(&v, &[1e-8, 1e-8, 1e-8]), 5e-3);
        // The largest component is the only one under its floor: the max comes
        // from what remains, not from the vector's own `amax`.
        assert_eq!(amax_above_floor(&v, &[1e-8, 1e-8, 1.0]), 3e-7);
        // Everything silenced.
        assert_eq!(amax_above_floor(&v, &[1.0, 1.0, 1.0]), 0.0);
        // Exactly at the floor is silenced (`>`, not `>=`).
        assert_eq!(amax_above_floor(&dvec(&[1e-8]), &[1e-8]), 0.0);
    }

    /// A floor that cannot be attributed component-wise must not silence
    /// anything: over-reporting the residual is the safe direction.
    #[test]
    fn amax_above_floor_falls_back_on_a_length_mismatch() {
        let v = dvec(&[1e-9, -3e-7]);
        assert_eq!(amax_above_floor(&v, &[1.0]), 3e-7);
        assert_eq!(amax_above_floor(&v, &[]), 3e-7);
    }

    /// The floored aggregate is never larger than the raw one, and on a
    /// fixture whose residuals (`c = 4`, `d − s = −2`) are nowhere near any
    /// resolution limit the two are identical — the common path is untouched.
    #[test]
    fn nlp_error_above_primal_noise_matches_on_ordinary_residuals() {
        let cq = fixture_with(MockNlp::new());
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
            cq.curr_primal_infeasibility_max()
        );
        assert_eq!(
            cq.curr_nlp_error_above_primal_noise(ROW_NOISE_KAPPA),
            cq.curr_nlp_error()
        );
    }

    /// gh #528, **equality block**, through the real accessor rather than a
    /// hand-supplied floor. The integration LP is all-inequality (`g_u = 2e19`,
    /// so `c.dim() == 0`), so this is the only cover the `declared_c_rhs()`
    /// branch has.
    ///
    /// `x = (4, 3)` puts `d = x0 = 4` on top of `s = 4`, so the inequality
    /// block's residual is an exact `0` and what the accessor returns is the
    /// `c` block alone.
    #[test]
    fn a_sub_quantum_equality_residual_is_silenced_and_a_coarser_one_is_not() {
        let rhs = 1e8;
        let floor = ROW_NOISE_KAPPA * Number::EPSILON * rhs;
        let cq_for = |c: Number| {
            fixture_with_x(
                MockNlp::new().with_c_rhs(Some(vec![rhs])).with_c(c),
                &[4.0, 3.0],
            )
        };

        // Under the quantum of `g(x) − b` at `|b| = 1e8`: no iterate could
        // have placed the residual here, so the row says nothing.
        let cq = cq_for(floor * 0.5);
        assert_eq!(cq.curr_primal_infeasibility_max(), floor * 0.5);
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
            0.0
        );

        // Above it: counted in full, not net of the floor.
        let cq = cq_for(floor * 2.0);
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
            floor * 2.0
        );

        // The placement floor alone would not have silenced anything here —
        // at ‖x‖_∞ = 4 through a row of `max_j |∂c/∂x_j| = 1` it is ~5.7e-14,
        // eight decades under the formation floor. The `c` branch's own
        // magnitude is what does the work.
        let cq = fixture_with_x(MockNlp::new().with_c(floor * 0.5), &[4.0, 3.0]);
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
            floor * 0.5
        );
    }

    /// The `primal_noise_floor_kappa = 0` escape hatch: every floor collapses
    /// to `0`, so every residual is counted and the floored aggregate is the
    /// raw one — the strict gate is bit-for-bit upstream Ipopt's again. Pinned
    /// on a fixture where the floor otherwise *does* silence the row, so this
    /// cannot pass by the two agreeing anyway.
    #[test]
    fn a_zero_kappa_switches_the_floor_off_completely() {
        let rhs = 1e8;
        let residual = ROW_NOISE_KAPPA * Number::EPSILON * rhs * 0.5;
        let cq = fixture_with_x(
            MockNlp::new().with_c_rhs(Some(vec![rhs])).with_c(residual),
            &[4.0, 3.0],
        );
        // The floor is live at the default kappa …
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
            0.0
        );
        // … and gone at zero.
        assert_eq!(
            cq.curr_primal_infeasibility_above_noise(0.0),
            cq.curr_primal_infeasibility_max()
        );
        assert_eq!(
            cq.curr_nlp_error_above_primal_noise(0.0),
            cq.curr_nlp_error()
        );
    }

    /// The equality floor rides the row scaling, because both sides of the
    /// comparison do: `declared_c_rhs()` reapplies `c_scale` (pinned by
    /// `declared_c_rhs_carries_the_row_scaling` in `orig_ipopt_nlp.rs`) and
    /// `curr_c()` is the scaled residual `dc · (g(x) − b)`. Scaling a row by
    /// `k` scales its residual and its floor together, so the verdict is
    /// invariant — which is what makes it legitimate to compare a floor built
    /// from the declared RHS against `curr_c()` at all.
    #[test]
    fn the_equality_floor_rides_the_row_scaling() {
        let rhs = 1e8;
        let quantum = ROW_NOISE_KAPPA * Number::EPSILON * rhs;
        for k in [1.0, 4.0, 0.25] {
            let cq_for = |c: Number| {
                fixture_with_x(
                    MockNlp::new()
                        .with_scaling(1.0, Some(vec![k]), None)
                        .with_c_rhs(Some(vec![k * rhs]))
                        .with_c(k * c),
                    &[4.0, 3.0],
                )
            };
            assert_eq!(
                cq_for(quantum * 0.5).curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
                0.0,
                "sub-quantum residual must stay silenced at row scaling {k}",
            );
            assert_eq!(
                cq_for(quantum * 2.0).curr_primal_infeasibility_above_noise(ROW_NOISE_KAPPA),
                k * quantum * 2.0,
                "above-quantum residual must survive at row scaling {k}",
            );
        }
    }
}
