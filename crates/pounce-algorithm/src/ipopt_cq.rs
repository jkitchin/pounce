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
use pounce_common::types::Number;
use pounce_linalg::{Matrix, SymMatrix, Vector};
use std::cell::RefCell;
use std::rc::Rc;

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
}

/// Helper: convert `Box<dyn Vector>` to `Rc<dyn Vector>`. Cheap; the
/// box is unwrapped without copying.
fn rc_from(v: Box<dyn Vector>) -> Rc<dyn Vector> {
    Rc::from(v)
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

        // t2 = max(mu/multiplier, s_min) - slack  (mu/0 → +inf, capped below)
        let mut t2 = t.make_new();
        t2.set(mu);
        t2.element_wise_divide(multiplier);
        let mut s_min_vec = t2.make_new();
        s_min_vec.set(s_min);
        t2.element_wise_max(&*s_min_vec);
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
        let grad_f = self.curr_grad_f();
        let jc_t_y_c = self.curr_jac_c_t_times_curr_y_c();
        let jd_t_y_d = self.curr_jac_d_t_times_curr_y_d();

        let mut tmp = iv.x.make_new();
        tmp.copy(&*grad_f);
        tmp.add_two_vectors(1.0, &*jc_t_y_c, 1.0, &*jd_t_y_d, 1.0);

        let nlp = self.nlp.borrow();
        nlp.px_l().mult_vector(-1.0, &*iv.z_l, 1.0, &mut *tmp);
        nlp.px_u().mult_vector(1.0, &*iv.z_u, 1.0, &mut *tmp);
        rc_from(tmp)
    }

    /// `∇_s L = -y_d - P_L v_L + P_U v_U`
    /// (`IpIpoptCalculatedQuantities.cpp:2069-2098`).
    pub fn curr_grad_lag_s(&self) -> Rc<dyn Vector> {
        let iv = self.curr_iv();
        let mut tmp = iv.y_d.make_new();
        let nlp = self.nlp.borrow();
        // tmp = P_U v_U
        nlp.pd_u().mult_vector(1.0, &*iv.v_u, 0.0, &mut *tmp);
        // tmp = tmp - P_L v_L
        nlp.pd_l().mult_vector(-1.0, &*iv.v_l, 1.0, &mut *tmp);
        // tmp = tmp - y_d
        tmp.axpy(-1.0, &*iv.y_d);
        rc_from(tmp)
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

    /// Scaled stationarity of the infeasibility measure `½‖(c, d−s)‖²`
    /// — `‖J_cᵀ c + J_dᵀ (d−s)‖_∞ / max(1, ‖(c, d−s)‖_∞)`. The
    /// numerator is the x-gradient of the squared constraint
    /// violation; a value near zero with the violation itself bounded
    /// away from zero marks an iterate converging to a stationary
    /// point of the infeasibility — i.e. a locally infeasible problem.
    /// No linear solve: two transpose-products. Mirrors the gradient
    /// term behind Ipopt's `IpRestoConvCheck.cpp` `LOCALLY_INFEASIBLE`
    /// test, applied here in the main loop.
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
        if acc.is_infinite() {
            0.0
        } else {
            acc
        }
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
        let iv = self.curr_iv();
        let (s_d, s_c) = self.optimality_error_scaling(&iv);

        // dual infeasibility (max-norm of grad_lag_x and grad_lag_s)
        let glx = self.curr_grad_lag_x();
        let gls = self.curr_grad_lag_s();
        let dual = glx.amax().max(gls.amax()) / s_d;

        // primal: max(||c||, ||d-s||)
        let c = self.curr_c();
        let dms = self.curr_d_minus_s();
        let primal = c.amax().max(dms.amax());

        // unbarriered complementarity (mu_target = 0 → just ||compl||)
        let compl = self
            .curr_compl_x_l()
            .amax()
            .max(self.curr_compl_x_u().amax())
            .max(self.curr_compl_s_l().amax())
            .max(self.curr_compl_s_u().amax())
            / s_c;

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
    }

    impl MockNlp {
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
        }
        fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            let cc = c.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            cc.values_mut()[0] = xx.values()[0] + xx.values()[1] - 1.0;
        }
        fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
            let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
            let dd = d.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            dd.values_mut()[0] = xx.values()[0];
        }
        fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            unimplemented!("not exercised in Phase 5 unit tests")
        }
        fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            unimplemented!("not exercised in Phase 5 unit tests")
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
    }

    fn fixture() -> IpoptCalculatedQuantities {
        let mut data = IpoptData::new();
        data.curr_mu = 0.1;
        // Iterate: x = (2, 3); s = (4); y_c = (1); y_d = (1);
        // z_L = (0.5) [bound on x[0]], z_U = (0.7) [bound on x[1]],
        // v_L = (0.3), v_U = ().
        let iv = IteratesVector::new(
            rcv(&[2.0, 3.0]),
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
        let nlp: StdRc<RefCell<dyn IpoptNlp>> = StdRc::new(RefCell::new(MockNlp::new()));
        let mut cq = IpoptCalculatedQuantities::new(data_handle, nlp);
        // Disable damping for clean unit-test expectations.
        cq.kappa_d = 0.0;
        cq
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

    #[test]
    fn curr_constraint_violation_is_one_norm() {
        // c(x) = x[0]+x[1]-1 = 4 ⇒ |c| = 4.
        // d(x)=x[0]=2; s=4 ⇒ d-s = -2 ⇒ |d-s| = 2.
        // theta = 4 + 2 = 6.
        let cq = fixture();
        assert!((cq.curr_constraint_violation() - 6.0).abs() < 1e-13);
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
}
