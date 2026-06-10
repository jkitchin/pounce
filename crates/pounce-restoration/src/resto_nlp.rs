//! Restoration NLP wrapper — port of
//! `Algorithm/IpRestoIpoptNLP.{hpp,cpp}`.
//!
//! Wraps the regular-phase NLP into the augmented restoration NLP
//! whose variables are `(x, n_c, p_c, n_d, p_d)`, objective is
//! `ρ * sum(n + p) + 0.5 * η(μ) * ||D_R (x - x_R)||_2^2`, and constraints
//! are `c(x) + n_c - p_c = 0`, `d(x) + n_d - p_d - s = 0` (the slacks enter
//! as `+n - p`, matching upstream `IpRestoIpoptNLP` and the
//! `restoration_constraint_{c,d}` implementations below).
//!
//! [`RestoIpoptNlp`] owns the 5-block `CompoundVectorSpace` for the
//! resto `x`, the dense `x_ref / D_R / D_R²` weight vectors, and the
//! `(rho, eta_factor)` configuration. Inherent `*_with_mu` methods
//! take a `&CompoundVector` for the resto-`x` and pass `mu` explicitly
//! (mirroring upstream's two-argument `f(x, mu)` /
//! `grad_f(x, mu)` overloads, which our base [`pounce_algorithm::ipopt_nlp::Nlp`]
//! trait does not directly expose). The `IpoptNlp` trait impl, the
//! 5-block bound spaces, and the `Px_L / Px_U / Pd_L / Pd_U` expansion
//! matrices land alongside the resto sub-builder.

use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::ipopt_nlp::{IpoptNlp, Nlp};
use pounce_common::types::{Index, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linalg::{
    CompoundMatrix, CompoundMatrixSpace, CompoundVector, CompoundVectorSpace, DenseVector,
    DenseVectorSpace, IdentityMatrix, Matrix, SymMatrix, Vector,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Block index of the original `x` variables inside the resto x-vector.
pub const BLOCK_X: Index = 0;
/// Block index of the equality-constraint negative slacks `n_c`.
pub const BLOCK_N_C: Index = 1;
/// Block index of the equality-constraint positive slacks `p_c`.
pub const BLOCK_P_C: Index = 2;
/// Block index of the inequality-constraint negative slacks `n_d`.
pub const BLOCK_N_D: Index = 3;
/// Block index of the inequality-constraint positive slacks `p_d`.
pub const BLOCK_P_D: Index = 4;

pub struct RestoIpoptNlp {
    /// Penalty coefficient ρ. Default from
    /// `IpRestoIpoptNLP.cpp:RegisterOptions`.
    pub rho: f64,
    /// `eta_factor` — the proximity-weight option. The actual η(μ)
    /// used in the objective is `eta_factor * sqrt(mu)`. Default
    /// matches upstream's `resto_proximity_weight`.
    pub eta_factor: f64,
    /// Re-evaluate the original objective at the restoration trial
    /// point so failures surface early. Mirrors
    /// `evaluate_orig_obj_at_resto_trial` (default true upstream).
    pub evaluate_orig_obj_at_resto_trial: bool,
    /// Dim of the original problem's `x`.
    pub n_orig: Index,
    /// Dim of the equality-constraint vector `c`.
    pub m_eq: Index,
    /// Dim of the inequality-constraint vector `d`.
    pub m_ineq: Index,
    /// 5-block compound space for the resto `x = (orig_x, n_c, p_c, n_d, p_d)`.
    pub x_space: Rc<CompoundVectorSpace>,
    /// `x_ref` — copy of the orig iterate's `x` at restoration entry.
    pub x_ref: Rc<DenseVector>,
    /// `D_R = 1 / max(1, |x_ref|)`.
    pub dr_x: Rc<DenseVector>,
    /// `D_R²` — elementwise square of `dr_x`.
    pub dr2_x: Rc<DenseVector>,
    /// Fallback `μ` for trait-side evaluations when no inner-data
    /// handle has been wired (legacy path; tests). Preferred path is
    /// `inner_data` below, mirroring upstream's `ip_data_->curr_mu()`.
    pub curr_mu: f64,
    /// Handle to the inner IPM's `IpoptData`. When `Some`, trait
    /// evaluations read `μ` from `data.curr_mu` (port of
    /// `IpRestoIpoptNLP.cpp:485` `ip_data_->curr_mu()`); when `None`,
    /// they fall back to the cached `self.curr_mu` field above so
    /// existing unit tests continue to drive eta via `set_curr_mu`.
    pub inner_data: Option<IpoptDataHandle>,
    /// The wrapped original NLP. The resto sub-builder calls
    /// [`Self::set_orig_nlp`] before the nested `IpoptAlgorithm` runs;
    /// `eval_c` / `eval_d` / `eval_jac_c` / `eval_jac_d` / `eval_h`
    /// delegate into it. Mirrors `RestoIpoptNlp::orig_ip_nlp_`.
    pub orig_nlp: Option<Rc<RefCell<dyn IpoptNlp>>>,
    /// Resto-side compressed lower-bound vector. 5-block compound:
    /// `[orig.x_l (n_xL_orig), n_c (m_eq), p_c (m_eq), n_d (m_ineq), p_d (m_ineq)]`.
    /// Slack blocks contain zeros (slacks `≥ 0`).
    pub x_l_resto: Option<Rc<CompoundVector>>,
    /// Resto-side compressed upper-bound vector. The four slack blocks
    /// have no upper bound, so this is a single dense block of dim
    /// `n_xU_orig` cloned from the orig.
    pub x_u_resto: Option<Rc<DenseVector>>,
    /// Resto-side `d_l` / `d_u` — same as orig (the inequality
    /// constraints' bounds are unchanged).
    pub d_l_resto: Option<Rc<DenseVector>>,
    pub d_u_resto: Option<Rc<DenseVector>>,
    /// Resto-side `Px_L`. 5×5 [`CompoundMatrix`]: `(0,0) = orig.px_l`,
    /// `(k, k) = I_{slack_dim}` for `k ∈ {1,2,3,4}` (all slack blocks
    /// have lower bounds = 0).
    pub px_l_resto: Option<Rc<dyn Matrix>>,
    /// Resto-side `Px_U`. 5×1 [`CompoundMatrix`]: `(0,0) = orig.px_u`.
    /// Slack blocks contribute no upper-bounded rows.
    pub px_u_resto: Option<Rc<dyn Matrix>>,
    /// Resto-side `Pd_L` / `Pd_U` — same as orig.
    pub pd_l_resto: Option<Rc<dyn Matrix>>,
    pub pd_u_resto: Option<Rc<dyn Matrix>>,
}

impl RestoIpoptNlp {
    /// Build a new resto NLP wrapper. `x_ref_vals` is a copy of the
    /// outer iterate's `x` at restoration entry; `RestoProximityWeights`
    /// is computed from it (`IpRestoIpoptNLP.cpp:412-440`).
    ///
    /// The 5-block `x_space` is laid out as
    /// `[orig_x (n_orig), n_c (m_eq), p_c (m_eq), n_d (m_ineq), p_d (m_ineq)]`.
    pub fn new(
        n_orig: Index,
        m_eq: Index,
        m_ineq: Index,
        x_ref_vals: &[f64],
        rho: f64,
        eta_factor: f64,
    ) -> Self {
        assert_eq!(x_ref_vals.len(), n_orig as usize);

        // Sub-spaces for each block.
        let orig_x_space = DenseVectorSpace::new(n_orig);
        let n_c_space = DenseVectorSpace::new(m_eq);
        let p_c_space = DenseVectorSpace::new(m_eq);
        let n_d_space = DenseVectorSpace::new(m_ineq);
        let p_d_space = DenseVectorSpace::new(m_ineq);

        // Build the 5-block CompoundVectorSpace.
        let total_dim = n_orig + 2 * m_eq + 2 * m_ineq;
        let x_space = CompoundVectorSpace::new(5, total_dim);
        let s = Rc::clone(&orig_x_space);
        x_space.set_comp(BLOCK_X, n_orig, move || {
            Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let s = Rc::clone(&n_c_space);
        x_space.set_comp(BLOCK_N_C, m_eq, move || {
            Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let s = Rc::clone(&p_c_space);
        x_space.set_comp(BLOCK_P_C, m_eq, move || {
            Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let s = Rc::clone(&n_d_space);
        x_space.set_comp(BLOCK_N_D, m_ineq, move || {
            Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let s = Rc::clone(&p_d_space);
        x_space.set_comp(BLOCK_P_D, m_ineq, move || {
            Box::new(DenseVector::new(Rc::clone(&s)))
        });

        // Materialize x_ref / dr_x / dr2_x as DenseVectors.
        let mut x_ref = DenseVector::new(Rc::clone(&orig_x_space));
        x_ref.set_values(x_ref_vals);
        let mut dr_x_buf = vec![0.0; n_orig as usize];
        build_dr_x(x_ref_vals, &mut dr_x_buf);
        let mut dr_x = DenseVector::new(Rc::clone(&orig_x_space));
        dr_x.set_values(&dr_x_buf);
        let mut dr2_x_buf = vec![0.0; n_orig as usize];
        build_dr2_x(&dr_x_buf, &mut dr2_x_buf);
        let mut dr2_x = DenseVector::new(Rc::clone(&orig_x_space));
        dr2_x.set_values(&dr2_x_buf);

        Self {
            rho,
            eta_factor,
            evaluate_orig_obj_at_resto_trial: true,
            n_orig,
            m_eq,
            m_ineq,
            x_space,
            x_ref: Rc::new(x_ref),
            dr_x: Rc::new(dr_x),
            dr2_x: Rc::new(dr2_x),
            curr_mu: 0.0,
            inner_data: None,
            orig_nlp: None,
            x_l_resto: None,
            x_u_resto: None,
            d_l_resto: None,
            d_u_resto: None,
            px_l_resto: None,
            px_u_resto: None,
            pd_l_resto: None,
            pd_u_resto: None,
        }
    }

    /// Wire in the wrapped original NLP. Mirrors the
    /// `orig_ip_nlp_` constructor argument upstream
    /// (`IpRestoIpoptNLP.cpp:64`). Required before any
    /// constraint / Jacobian / Hessian evaluation through the `Nlp`
    /// trait surface. Snapshots the orig's bound vectors and bound
    /// expansion matrices into resto-side wrappers (`x_l_resto`,
    /// `px_l_resto`, etc.) so the [`IpoptNlp`] trait accessors can
    /// hand them out by reference.
    pub fn set_orig_nlp(&mut self, orig: Rc<RefCell<dyn IpoptNlp>>) {
        debug_assert_eq!(orig.borrow().n(), self.n_orig);
        debug_assert_eq!(orig.borrow().m_eq(), self.m_eq);
        debug_assert_eq!(orig.borrow().m_ineq(), self.m_ineq);

        // Snapshot bound vectors and expansion matrices.
        let orig_ref = orig.borrow();
        let n_xl_orig = orig_ref.x_l().dim();
        let n_xu_orig = orig_ref.x_u().dim();
        let x_l_vals = clone_dense_values(orig_ref.x_l());
        let x_u_vals = clone_dense_values(orig_ref.x_u());
        let d_l_vals = clone_dense_values(orig_ref.d_l());
        let d_u_vals = clone_dense_values(orig_ref.d_u());
        let orig_px_l = orig_ref.px_l();
        let orig_px_u = orig_ref.px_u();
        let pd_l = orig_ref.pd_l();
        let pd_u = orig_ref.pd_u();
        drop(orig_ref);

        // x_u_resto: dense block of dim n_xU_orig. Slacks have no upper
        // bound so they contribute no rows here.
        let x_u_space = DenseVectorSpace::new(n_xu_orig);
        let mut x_u_dv = DenseVector::new(x_u_space);
        x_u_dv.set_values(&x_u_vals);
        self.x_u_resto = Some(Rc::new(x_u_dv));

        // d_l / d_u: copies of orig (resto inequality bounds = orig).
        // Bound vectors are packed (dim = number of lower/upper-bounded
        // inequalities), so size the spaces from `len()` not from
        // `self.m_ineq` — those agree only when every inequality has
        // both bounds finite.
        let d_l_space = DenseVectorSpace::new(d_l_vals.len() as Index);
        let mut d_l_dv = DenseVector::new(d_l_space);
        d_l_dv.set_values(&d_l_vals);
        self.d_l_resto = Some(Rc::new(d_l_dv));

        let d_u_space = DenseVectorSpace::new(d_u_vals.len() as Index);
        let mut d_u_dv = DenseVector::new(d_u_space);
        d_u_dv.set_values(&d_u_vals);
        self.d_u_resto = Some(Rc::new(d_u_dv));

        // x_l_resto: 5-block compound `[orig.x_l, 0, 0, 0, 0]` with
        // slack dims `[m_eq, m_eq, m_ineq, m_ineq]`.
        let x_l_total_dim = n_xl_orig + 2 * self.m_eq + 2 * self.m_ineq;
        let x_l_resto_space = CompoundVectorSpace::new(5, x_l_total_dim);
        let xl0_space = DenseVectorSpace::new(n_xl_orig);
        x_l_resto_space.set_comp(0, n_xl_orig, {
            let s = Rc::clone(&xl0_space);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let n_c_l_space = DenseVectorSpace::new(self.m_eq);
        x_l_resto_space.set_comp(1, self.m_eq, {
            let s = Rc::clone(&n_c_l_space);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        x_l_resto_space.set_comp(2, self.m_eq, {
            let s = Rc::clone(&n_c_l_space);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let n_d_l_space = DenseVectorSpace::new(self.m_ineq);
        x_l_resto_space.set_comp(3, self.m_ineq, {
            let s = Rc::clone(&n_d_l_space);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        x_l_resto_space.set_comp(4, self.m_ineq, {
            let s = Rc::clone(&n_d_l_space);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        let mut x_l_cv = CompoundVector::new(x_l_resto_space);
        // Block 0 ← orig.x_l; blocks 1..4 ← explicit zeros (a freshly
        // constructed `DenseVector` with dim>0 starts uninitialized, so
        // we materialize zeros here for downstream `expanded_values`).
        downcast_dense_mut(x_l_cv.comp_mut(0)).set_values(&x_l_vals);
        let zero_eq = vec![0.0; self.m_eq as usize];
        let zero_ineq = vec![0.0; self.m_ineq as usize];
        downcast_dense_mut(x_l_cv.comp_mut(1)).set_values(&zero_eq);
        downcast_dense_mut(x_l_cv.comp_mut(2)).set_values(&zero_eq);
        downcast_dense_mut(x_l_cv.comp_mut(3)).set_values(&zero_ineq);
        downcast_dense_mut(x_l_cv.comp_mut(4)).set_values(&zero_ineq);
        self.x_l_resto = Some(Rc::new(x_l_cv));

        // px_l_resto: 5×5 CompoundMatrix.
        let px_l_space = CompoundMatrixSpace::new_with_dims(
            vec![self.n_orig, self.m_eq, self.m_eq, self.m_ineq, self.m_ineq],
            vec![n_xl_orig, self.m_eq, self.m_eq, self.m_ineq, self.m_ineq],
        );
        let mut px_l_cm = CompoundMatrix::new(px_l_space);
        px_l_cm.set_comp(0, 0, orig_px_l);
        px_l_cm.set_comp(1, 1, Rc::new(IdentityMatrix::new(self.m_eq)));
        px_l_cm.set_comp(2, 2, Rc::new(IdentityMatrix::new(self.m_eq)));
        px_l_cm.set_comp(3, 3, Rc::new(IdentityMatrix::new(self.m_ineq)));
        px_l_cm.set_comp(4, 4, Rc::new(IdentityMatrix::new(self.m_ineq)));
        self.px_l_resto = Some(Rc::new(px_l_cm));

        // px_u_resto: 5×1 CompoundMatrix; only block (0,0) populated.
        let px_u_space = CompoundMatrixSpace::new_with_dims(
            vec![self.n_orig, self.m_eq, self.m_eq, self.m_ineq, self.m_ineq],
            vec![n_xu_orig],
        );
        let mut px_u_cm = CompoundMatrix::new(px_u_space);
        px_u_cm.set_comp(0, 0, orig_px_u);
        self.px_u_resto = Some(Rc::new(px_u_cm));

        // pd_l, pd_u: same as orig.
        self.pd_l_resto = Some(pd_l);
        self.pd_u_resto = Some(pd_u);

        self.orig_nlp = Some(orig);
    }

    /// Build a fresh resto-`x` compound vector with all five blocks
    /// freshly allocated.
    pub fn make_new_x(&self) -> CompoundVector {
        CompoundVector::new(Rc::clone(&self.x_space))
    }

    /// `Eta(μ) = eta_factor * sqrt(μ)`. Matches upstream's `Eta(mu)`
    /// inline at `IpRestoIpoptNLP.cpp:632`.
    pub fn eta(&self, mu: f64) -> f64 {
        self.eta_factor * mu.sqrt()
    }

    /// Set the `μ` to be used by the `Nlp`-trait `eval_f` /
    /// `eval_grad_f` calls (which don't carry `μ` themselves). The
    /// algorithm calls this once per iteration, before any of the
    /// resto-NLP evaluations.
    ///
    /// Upstream's `IpoptAlgorithm` instead routes through the
    /// two-argument `IpoptNLP::f(x, mu)` / `grad_f(x, mu)` overloads
    /// (`IpRestoIpoptNLP.cpp:462,504`); our base `Nlp` trait does not
    /// expose those, so we cache `μ` on the wrapper instead.
    pub fn set_curr_mu(&mut self, mu: f64) {
        self.curr_mu = mu;
    }

    /// Wire the inner IPM's `IpoptData` so the trait-side `eval_*`
    /// methods can read `μ` from it. Mirrors upstream's
    /// `RestoIpoptNLP::ip_data_` slot. When set, this takes precedence
    /// over the cached [`Self::curr_mu`] on every evaluation.
    pub fn set_inner_data(&mut self, data: IpoptDataHandle) {
        self.inner_data = Some(data);
    }

    /// Read `μ` from the wired inner data when present, else fall back
    /// to the cached [`Self::curr_mu`]. Used by the `Nlp`-trait
    /// evaluations.
    fn live_mu(&self) -> f64 {
        match &self.inner_data {
            Some(d) => d.borrow().curr_mu,
            None => self.curr_mu,
        }
    }

    /// Two-argument objective `f(x, μ)` mirroring
    /// `IpRestoIpoptNLP.cpp:462-501`. Reads each block of `x` via
    /// `expanded_values` so homogeneous (post-`Set(1.0)`) slacks work
    /// without materialization.
    pub fn f_with_mu(&self, x: &CompoundVector, mu: f64) -> f64 {
        debug_assert_eq!(x.n_comps(), 5);
        let x_only = downcast_dense(x.comp(BLOCK_X)).expanded_values();
        let n_c = downcast_dense(x.comp(BLOCK_N_C)).expanded_values();
        let p_c = downcast_dense(x.comp(BLOCK_P_C)).expanded_values();
        let n_d = downcast_dense(x.comp(BLOCK_N_D)).expanded_values();
        let p_d = downcast_dense(x.comp(BLOCK_P_D)).expanded_values();
        let s_sum = slack_sum(&n_c, &p_c, &n_d, &p_d);
        restoration_objective(
            self.rho,
            self.eta(mu),
            s_sum,
            &x_only,
            self.x_ref.values(),
            self.dr_x.values(),
        )
    }

    /// Two-argument gradient `∇f(x, μ)` mirroring
    /// `IpRestoIpoptNLP.cpp:504-525`. Writes:
    /// * the slack blocks (`n_c, p_c, n_d, p_d`) to `rho` (constant);
    /// * the orig-`x` block to `eta(μ) · D_R² · (x − x_ref)`.
    pub fn grad_f_with_mu(&self, x: &CompoundVector, mu: f64, out: &mut CompoundVector) {
        debug_assert_eq!(x.n_comps(), 5);
        debug_assert_eq!(out.n_comps(), 5);
        let x_only = downcast_dense(x.comp(BLOCK_X)).expanded_values();
        let eta = self.eta(mu);
        // Orig-x block.
        {
            let dest = downcast_dense_mut(out.comp_mut(BLOCK_X));
            let n = self.n_orig as usize;
            let mut buf = vec![0.0; n];
            restoration_grad_x(
                eta,
                &x_only,
                self.x_ref.values(),
                self.dr_x.values(),
                &mut buf,
            );
            dest.set_values(&buf);
        }
        // Slack blocks: g = rho.
        for &k in &[BLOCK_N_C, BLOCK_P_C, BLOCK_N_D, BLOCK_P_D] {
            out.comp_mut(k).set(self.rho);
        }
    }

    /// Resto-NLP equality constraint: `c_resto = c_orig(x) + n_c − p_c`.
    /// `c_orig` must already have been evaluated at the orig-`x` slice
    /// of `x` and supplied as a dense (or homogeneous) buffer of dim
    /// `m_eq`. Mirrors `IpRestoIpoptNLP.cpp:528-543`.
    pub fn c_resto(&self, c_orig: &dyn Vector, x: &CompoundVector, out: &mut dyn Vector) {
        debug_assert_eq!(c_orig.dim(), self.m_eq);
        debug_assert_eq!(out.dim(), self.m_eq);
        let c_orig = downcast_dense(c_orig).expanded_values();
        let n_c = downcast_dense(x.comp(BLOCK_N_C)).expanded_values();
        let p_c = downcast_dense(x.comp(BLOCK_P_C)).expanded_values();
        let mut buf = vec![0.0; self.m_eq as usize];
        restoration_constraint_c(&c_orig, &n_c, &p_c, &mut buf);
        downcast_dense_mut(out).set_values(&buf);
    }

    /// Resto-NLP inequality constraint: `d_resto = d_orig(x) + n_d − p_d`.
    /// Mirrors `IpRestoIpoptNLP.cpp:553-572`. The slack `s` is folded
    /// in later as part of the `(d − s)` residual.
    pub fn d_resto(&self, d_orig: &dyn Vector, x: &CompoundVector, out: &mut dyn Vector) {
        debug_assert_eq!(d_orig.dim(), self.m_ineq);
        debug_assert_eq!(out.dim(), self.m_ineq);
        let d_orig = downcast_dense(d_orig).expanded_values();
        let n_d = downcast_dense(x.comp(BLOCK_N_D)).expanded_values();
        let p_d = downcast_dense(x.comp(BLOCK_P_D)).expanded_values();
        let mut buf = vec![0.0; self.m_ineq as usize];
        restoration_constraint_d(&d_orig, &n_d, &p_d, &mut buf);
        downcast_dense_mut(out).set_values(&buf);
    }

    /// Initialize a resto-`x`: copy `x_ref` into the orig-`x` block
    /// and set the four slack blocks to 1.0. Mirrors the cold-start
    /// portion of `IpRestoIpoptNLP.cpp:251-254`; the proper
    /// closed-form slack values are then refined by
    /// [`crate::init::init_slack_pair`] once `c(x)` and `d(x) − s`
    /// are known.
    pub fn init_starting_x(&self, out: &mut CompoundVector) {
        debug_assert_eq!(out.n_comps(), 5);
        // orig_x ← x_ref
        downcast_dense_mut(out.comp_mut(BLOCK_X)).set_values(self.x_ref.values());
        // slacks ← 1.0
        for &k in &[BLOCK_N_C, BLOCK_P_C, BLOCK_N_D, BLOCK_P_D] {
            out.comp_mut(k).set(1.0);
        }
    }
}

/// `Nlp`-trait surface for the resto wrapper.
///
/// `eval_f` and `eval_grad_f` route through `f_with_mu` / `grad_f_with_mu`
/// using the cached `curr_mu` (set by [`Self::set_curr_mu`] before the
/// algorithm calls these). The constraint, Jacobian, and Hessian
/// methods delegate to the wrapped orig NLP plumbed in via
/// [`Self::set_orig_nlp`] and assemble the resto-side overlay
/// (slack-difference adjustment for `c`/`d`; signed identities at the
/// `n_*`/`p_*` columns of the Jacobians; proximity-Hessian sum block
/// for `H`).
impl Nlp for RestoIpoptNlp {
    fn n(&self) -> Index {
        self.x_space.dim()
    }

    fn m_eq(&self) -> Index {
        self.m_eq
    }

    fn m_ineq(&self) -> Index {
        self.m_ineq
    }

    fn eval_f(&mut self, x: &dyn Vector) -> Number {
        let cv = downcast_compound(x);
        self.f_with_mu(cv, self.live_mu())
    }

    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
        let cv = downcast_compound(x);
        let mu = self.live_mu();
        let cg = downcast_compound_mut(g);
        self.grad_f_with_mu(cv, mu, cg);
    }

    /// `eval_c(x_R) = c(x_orig) + n_c − p_c`. Delegates to the wrapped
    /// orig NLP for `c(x_orig)`, then adds the slack difference in
    /// place. Mirrors `IpRestoIpoptNLP.cpp:528-543`.
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
        let cv = downcast_compound(x);
        let orig = self
            .orig_nlp
            .as_ref()
            .expect("RestoIpoptNlp::eval_c called before set_orig_nlp")
            .clone();
        // Orig writes c(x_orig) directly into the caller's buffer.
        orig.borrow_mut().eval_c(cv.comp(BLOCK_X), c);
        let n_c_vals = downcast_dense(cv.comp(BLOCK_N_C)).expanded_values();
        let p_c_vals = downcast_dense(cv.comp(BLOCK_P_C)).expanded_values();
        let dest = downcast_dense_mut(c);
        for (i, v) in dest.values_mut().iter_mut().enumerate() {
            *v += n_c_vals[i] - p_c_vals[i];
        }
    }

    /// `eval_d(x_R) = d(x_orig) + n_d − p_d`. Same pattern as `eval_c`.
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
        let cv = downcast_compound(x);
        let orig = self
            .orig_nlp
            .as_ref()
            .expect("RestoIpoptNlp::eval_d called before set_orig_nlp")
            .clone();
        orig.borrow_mut().eval_d(cv.comp(BLOCK_X), d);
        let n_d_vals = downcast_dense(cv.comp(BLOCK_N_D)).expanded_values();
        let p_d_vals = downcast_dense(cv.comp(BLOCK_P_D)).expanded_values();
        let dest = downcast_dense_mut(d);
        for (i, v) in dest.values_mut().iter_mut().enumerate() {
            *v += n_d_vals[i] - p_d_vals[i];
        }
    }

    /// `∂c_R/∂x_R = [J_c | +I_{m_eq} | −I_{m_eq} | 0 | 0]`. Mirrors
    /// `IpRestoIpoptNLP.cpp:593-608`: BLOCK_N_C uses the default
    /// IdentityMatrix factor (+1), BLOCK_P_C uses `SetFactor(-1.0)`.
    /// This matches `c_R(x_R) = c(x_orig) + n_c − p_c`, i.e.
    /// `∂c/∂n_c = +I` and `∂c/∂p_c = −I`.
    ///
    /// v0.1 simplification: emit as a flat [`GenTMatrix`] with
    /// `m_eq` rows and `n_orig + 2*m_eq + 2*m_ineq` cols. Triplet
    /// stream concatenates the orig Jacobian's triplets (column
    /// indices unshifted, since they map into BLOCK_X cols 1..=n_orig)
    /// with `m_eq` `(i, n_orig + i, +1.0)` entries (the +I at
    /// BLOCK_N_C) and `m_eq` `(i, n_orig + m_eq + i, -1.0)` entries
    /// (the −I at BLOCK_P_C). Bit-equivalence with upstream's
    /// `CompoundMatrix` is a Phase-10 concern.
    fn eval_jac_c(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        let cv = downcast_compound(x);
        let orig = self
            .orig_nlp
            .as_ref()
            .expect("RestoIpoptNlp::eval_jac_c called before set_orig_nlp")
            .clone();
        let orig_jac_dyn = orig.borrow_mut().eval_jac_c(cv.comp(BLOCK_X));
        let orig_jac = orig_jac_dyn
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("RestoIpoptNlp::eval_jac_c: orig Jacobian must be a GenTMatrix in v0.1");

        let n_total_cols = self.n_orig + 2 * self.m_eq + 2 * self.m_ineq;
        let n_orig_nz = orig_jac.nonzeros() as usize;
        let n_id = self.m_eq as usize;
        let total_nz = n_orig_nz + 2 * n_id;

        let mut irows = Vec::with_capacity(total_nz);
        let mut jcols = Vec::with_capacity(total_nz);
        irows.extend_from_slice(orig_jac.irows());
        jcols.extend_from_slice(orig_jac.jcols());
        // +I block at columns [n_orig+1 .. n_orig+m_eq] (BLOCK_N_C)
        for i in 1..=self.m_eq {
            irows.push(i);
            jcols.push(self.n_orig + i);
        }
        // -I block at columns [n_orig+m_eq+1 .. n_orig+2*m_eq] (BLOCK_P_C)
        for i in 1..=self.m_eq {
            irows.push(i);
            jcols.push(self.n_orig + self.m_eq + i);
        }

        let space = GenTMatrixSpace::new(self.m_eq, n_total_cols, irows, jcols);
        let mut gen_t = GenTMatrix::new(space);
        let vals = gen_t.values_mut();
        vals[..n_orig_nz].copy_from_slice(orig_jac.values());
        for i in 0..n_id {
            vals[n_orig_nz + i] = 1.0;
        }
        for i in 0..n_id {
            vals[n_orig_nz + n_id + i] = -1.0;
        }

        Rc::new(gen_t)
    }

    /// `∂d_R/∂x_R = [J_d | 0 | 0 | +I_{m_ineq} | −I_{m_ineq}]`. Mirrors
    /// `IpRestoIpoptNLP.cpp:631-644`: BLOCK_N_D uses the default
    /// IdentityMatrix factor (+1), BLOCK_P_D uses `SetFactor(-1.0)`.
    /// This matches `d_R(x_R) = d(x_orig) + n_d − p_d`.
    ///
    /// v0.1 simplification: emit as a flat [`GenTMatrix`] with
    /// `m_ineq` rows and `n_orig + 2*m_eq + 2*m_ineq` cols, analogous
    /// to [`Self::eval_jac_c`]. +I goes at columns
    /// `n_orig + 2*m_eq + i` (BLOCK_N_D), and −I at
    /// `n_orig + 2*m_eq + m_ineq + i` (BLOCK_P_D).
    fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        let cv = downcast_compound(x);
        let orig = self
            .orig_nlp
            .as_ref()
            .expect("RestoIpoptNlp::eval_jac_d called before set_orig_nlp")
            .clone();
        let orig_jac_dyn = orig.borrow_mut().eval_jac_d(cv.comp(BLOCK_X));
        let orig_jac = orig_jac_dyn
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("RestoIpoptNlp::eval_jac_d: orig Jacobian must be a GenTMatrix in v0.1");

        let n_total_cols = self.n_orig + 2 * self.m_eq + 2 * self.m_ineq;
        let n_orig_nz = orig_jac.nonzeros() as usize;
        let n_id = self.m_ineq as usize;
        let total_nz = n_orig_nz + 2 * n_id;

        let mut irows = Vec::with_capacity(total_nz);
        let mut jcols = Vec::with_capacity(total_nz);
        irows.extend_from_slice(orig_jac.irows());
        jcols.extend_from_slice(orig_jac.jcols());
        let nd_col_base = self.n_orig + 2 * self.m_eq;
        // +I block at BLOCK_N_D columns
        for i in 1..=self.m_ineq {
            irows.push(i);
            jcols.push(nd_col_base + i);
        }
        // -I block at BLOCK_P_D columns
        for i in 1..=self.m_ineq {
            irows.push(i);
            jcols.push(nd_col_base + self.m_ineq + i);
        }

        let space = GenTMatrixSpace::new(self.m_ineq, n_total_cols, irows, jcols);
        let mut gen_t = GenTMatrix::new(space);
        let vals = gen_t.values_mut();
        vals[..n_orig_nz].copy_from_slice(orig_jac.values());
        for i in 0..n_id {
            vals[n_orig_nz + i] = 1.0;
        }
        for i in 0..n_id {
            vals[n_orig_nz + n_id + i] = -1.0;
        }

        Rc::new(gen_t)
    }

    /// `∇²L_R = blockdiag(∇²L_orig + obj_factor·η(μ)·D_R², 0, 0, 0, 0)`.
    /// Mirrors `IpRestoIpoptNLP.cpp:701-740`. The orig Hessian is
    /// evaluated with `obj_factor=0` because the orig objective `f(x)`
    /// does not appear in the resto NLP's Lagrangian — only the
    /// proximity term `½ η(μ)·||D_R(x − x_ref)||²`, whose Hessian is
    /// `obj_factor·η(μ)·diag(D_R²)`. The slack blocks contribute
    /// nothing (linear in the constraints, scalar in the objective).
    fn eval_h(
        &mut self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        // v0.1 simplification: emit the resto Hessian as a single flat
        // [`SymTMatrix`] of dim `n_orig + 2*m_eq + 2*m_ineq` with
        //
        // * the orig-NLP Hessian's lower-triangular triplets in rows
        //   `1..=n_orig` (un-shifted — they already live in the
        //   top-left block), and
        // * `n_orig` extra diagonal entries `(i, i, obj_factor·η(μ)·DR²[i])`
        //   for the proximity term.
        //
        // The slack-block diagonals are implicit zero (no triplets
        // emitted). Duplicate (i,i) entries from `orig_h`'s diagonal
        // and the proximity diagonal are summed by the linear-solver
        // backend, matching upstream's `IpTripletToCSRConverter`
        // dedup-by-summing semantics.
        //
        // Bit-equivalence with upstream's resto-side
        // `CompoundSymMatrix(SumSymMatrix(orig_h, η·DR²))` is a Phase
        // 10 concern; for v0.1 the flat triplet form is what
        // `StdAugSystemSolver` consumes.
        let cv = downcast_compound(x);
        let orig = self
            .orig_nlp
            .as_ref()
            .expect("RestoIpoptNlp::eval_h called before set_orig_nlp")
            .clone();
        let orig_h_dyn = orig.borrow_mut().eval_h(cv.comp(BLOCK_X), 0.0, y_c, y_d);
        let orig_h = orig_h_dyn
            .as_any()
            .downcast_ref::<SymTMatrix>()
            .expect("RestoIpoptNlp::eval_h: orig Hessian must be a SymTMatrix in v0.1");

        let n_total = self.n_orig + 2 * self.m_eq + 2 * self.m_ineq;
        let n_orig_nz = orig_h.nonzeros() as usize;
        let n_diag = self.n_orig as usize;
        let total_nz = n_orig_nz + n_diag;

        let mut irows = Vec::with_capacity(total_nz);
        let mut jcols = Vec::with_capacity(total_nz);
        irows.extend_from_slice(orig_h.irows());
        jcols.extend_from_slice(orig_h.jcols());
        for i in 1..=self.n_orig {
            irows.push(i);
            jcols.push(i);
        }

        let space = SymTMatrixSpace::new(n_total, irows, jcols);
        let mut sym_t = SymTMatrix::new(space);
        let eta = self.eta(self.live_mu());
        let scale = obj_factor * eta;
        let dr2_vals = self.dr2_x.values();
        let vals = sym_t.values_mut();
        vals[..n_orig_nz].copy_from_slice(orig_h.values());
        for i in 0..n_diag {
            vals[n_orig_nz + i] = scale * dr2_vals[i];
        }

        Rc::new(sym_t)
    }
}

/// `IpoptNlp` impl — exposes the resto-side bound vectors and bound
/// expansion matrices snapshotted in [`RestoIpoptNlp::set_orig_nlp`].
///
/// Layout (per upstream `IpRestoIpoptNLP.cpp:147-251`):
/// * `x_l = [orig.x_l (n_xL_orig); 0 (m_eq); 0 (m_eq); 0 (m_ineq); 0 (m_ineq)]`
///   — slack lower bounds are all zero.
/// * `x_u = orig.x_u` (slacks have no upper bound).
/// * `Px_L = blockdiag(orig.Px_L, I_{m_eq}, I_{m_eq}, I_{m_ineq}, I_{m_ineq})`.
/// * `Px_U` extracts only the orig-`x` rows that have an upper bound.
/// * `d_l / d_u / Pd_L / Pd_U` are unchanged from the orig (the
///   inequality constraints `d` are reused).
impl IpoptNlp for RestoIpoptNlp {
    fn x_l(&self) -> &dyn Vector {
        &**self
            .x_l_resto
            .as_ref()
            .expect("RestoIpoptNlp::x_l called before set_orig_nlp")
    }
    fn x_u(&self) -> &dyn Vector {
        &**self
            .x_u_resto
            .as_ref()
            .expect("RestoIpoptNlp::x_u called before set_orig_nlp")
    }
    fn d_l(&self) -> &dyn Vector {
        &**self
            .d_l_resto
            .as_ref()
            .expect("RestoIpoptNlp::d_l called before set_orig_nlp")
    }
    fn d_u(&self) -> &dyn Vector {
        &**self
            .d_u_resto
            .as_ref()
            .expect("RestoIpoptNlp::d_u called before set_orig_nlp")
    }
    fn px_l(&self) -> Rc<dyn Matrix> {
        self.px_l_resto
            .as_ref()
            .expect("RestoIpoptNlp::px_l called before set_orig_nlp")
            .clone()
    }
    fn px_u(&self) -> Rc<dyn Matrix> {
        self.px_u_resto
            .as_ref()
            .expect("RestoIpoptNlp::px_u called before set_orig_nlp")
            .clone()
    }
    fn pd_l(&self) -> Rc<dyn Matrix> {
        self.pd_l_resto
            .as_ref()
            .expect("RestoIpoptNlp::pd_l called before set_orig_nlp")
            .clone()
    }
    fn pd_u(&self) -> Rc<dyn Matrix> {
        self.pd_u_resto
            .as_ref()
            .expect("RestoIpoptNlp::pd_u called before set_orig_nlp")
            .clone()
    }
}

fn downcast_compound(v: &dyn Vector) -> &CompoundVector {
    v.as_any()
        .downcast_ref::<CompoundVector>()
        .expect("RestoIpoptNlp::Nlp expected a CompoundVector argument")
}

fn downcast_compound_mut(v: &mut dyn Vector) -> &mut CompoundVector {
    v.as_any_mut()
        .downcast_mut::<CompoundVector>()
        .expect("RestoIpoptNlp::Nlp expected a CompoundVector argument")
}

fn downcast_dense(v: &dyn Vector) -> &DenseVector {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .expect("RestoIpoptNlp expected a DenseVector component")
}

fn downcast_dense_mut(v: &mut dyn Vector) -> &mut DenseVector {
    v.as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("RestoIpoptNlp expected a DenseVector component")
}

/// Snapshot a `&dyn Vector` (assumed to be a `DenseVector`) into a
/// fresh `Vec<f64>`. Used by `set_orig_nlp` to copy orig-side bound
/// vectors out of the orig NLP's borrow scope.
fn clone_dense_values(v: &dyn Vector) -> Vec<f64> {
    downcast_dense(v).expanded_values()
}

/// Build the diagonal scaling `D_R` from a reference point `x_ref`.
/// Mirrors `IpRestoIpoptNLP.cpp:431-440`:
/// ```text
///   D_R[i] = 1 / max(1, |x_ref[i]|)
/// ```
/// Implemented elementwise on `&[f64]`; the caller plumbs the result
/// into a `DenseVector`.
pub fn build_dr_x(x_ref: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x_ref.len(), out.len());
    for (o, &xr) in out.iter_mut().zip(x_ref.iter()) {
        let m = xr.abs().max(1.0);
        *o = 1.0 / m;
    }
}

/// Build the squared diagonal `D_R²` from `dr_x`. Used to populate the
/// `DR2_x_` `DiagMatrix` referenced by the Hessian (`SumSymMatrix(orig_h,
/// DR2_x)` at upstream `IpRestoIpoptNLP.cpp:701`).
pub fn build_dr2_x(dr_x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(dr_x.len(), out.len());
    for (o, &d) in out.iter_mut().zip(dr_x.iter()) {
        *o = d * d;
    }
}

/// Per-restoration-entry constants captured once when the restoration
/// phase activates (`IpRestoIpoptNLP.cpp:412-440`):
/// * `x_ref` is a copy of the outer iterate's `x` at the moment of
///   restoration entry,
/// * `dr_x` is the `1 / max(1, |x_ref|)` reciprocal,
/// * `dr2_x` is the elementwise square of `dr_x`.
///
/// Held by [`RestoIpoptNlp`] for the lifetime of the resto sub-solve;
/// `Eta(mu)` is multiplied in at f/grad/Hessian assembly time.
#[derive(Debug, Clone)]
pub struct RestoProximityWeights {
    pub x_ref: Vec<f64>,
    pub dr_x: Vec<f64>,
    pub dr2_x: Vec<f64>,
}

impl RestoProximityWeights {
    /// Allocate from a reference point. Mirrors the trio of
    /// `IpRestoIpoptNLP.cpp:412-440`.
    pub fn from_x_ref(x_ref: &[f64]) -> Self {
        let n = x_ref.len();
        let mut dr_x = vec![0.0; n];
        let mut dr2_x = vec![0.0; n];
        build_dr_x(x_ref, &mut dr_x);
        build_dr2_x(&dr_x, &mut dr2_x);
        Self {
            x_ref: x_ref.to_vec(),
            dr_x,
            dr2_x,
        }
    }
}

/// Scalar core of `RestoIpoptNLP::f` at `IpRestoIpoptNLP.cpp:462-501`:
/// ```text
///   f = rho * sum(n_c + p_c + n_d + p_d)
///       + 0.5 * eta(mu) * ||D_R (x - x_R)||_2^2
/// ```
/// The slack-block sum is supplied as `slack_sum` (the caller computes
/// `n_c.sum() + p_c.sum() + n_d.sum() + p_d.sum()` from the
/// CompoundVector blocks). `dr_x` and `x_ref` are the diagonal
/// scaling and reference point built once at restoration entry.
pub fn restoration_objective(
    rho: f64,
    eta: f64,
    slack_sum: f64,
    x: &[f64],
    x_ref: &[f64],
    dr_x: &[f64],
) -> f64 {
    debug_assert_eq!(x.len(), x_ref.len());
    debug_assert_eq!(x.len(), dr_x.len());
    let mut sq = 0.0;
    for i in 0..x.len() {
        let d = dr_x[i] * (x[i] - x_ref[i]);
        sq += d * d;
    }
    rho * slack_sum + 0.5 * eta * sq
}

/// Vector-level c-block constraint: `c_resto = c_orig(x) + n_c − p_c`.
/// Mirrors `IpRestoIpoptNLP.cpp:528-543`. The caller has already
/// computed `c_orig` at the resto-trial's `x_only`.
pub fn restoration_constraint_c(c_orig: &[f64], n_c: &[f64], p_c: &[f64], out: &mut [f64]) {
    debug_assert_eq!(c_orig.len(), n_c.len());
    debug_assert_eq!(c_orig.len(), p_c.len());
    debug_assert_eq!(c_orig.len(), out.len());
    for i in 0..c_orig.len() {
        out[i] = c_orig[i] + n_c[i] - p_c[i];
    }
}

/// Vector-level d-block constraint: `d_resto = d_orig(x) + n_d − p_d`.
/// Mirrors `IpRestoIpoptNLP.cpp:553-572`. (The slack `s` is subtracted
/// later in the (d − s) residual, not here.)
pub fn restoration_constraint_d(d_orig: &[f64], n_d: &[f64], p_d: &[f64], out: &mut [f64]) {
    debug_assert_eq!(d_orig.len(), n_d.len());
    debug_assert_eq!(d_orig.len(), p_d.len());
    debug_assert_eq!(d_orig.len(), out.len());
    for i in 0..d_orig.len() {
        out[i] = d_orig[i] + n_d[i] - p_d[i];
    }
}

/// Initial slack values for the four restoration slack blocks. Mirrors
/// `IpRestoIpoptNLP.cpp:251-254` (upstream: `comp_x[k]->Set(1.0)`). The
/// `init` module's [`crate::init::init_slack_pair`] computes the proper
/// slack values once `c(x)` and `d(x) − s` are known; this helper
/// covers the cold-start path used before that loop runs.
pub fn fill_initial_slacks(slacks: &mut [f64]) {
    for v in slacks.iter_mut() {
        *v = 1.0;
    }
}

/// Compute `slack_sum = Σ n_c + Σ p_c + Σ n_d + Σ p_d`. The objective
/// `f` at upstream `IpRestoIpoptNLP.cpp:480-482` uses this as the
/// penalty term `rho * slack_sum`.
pub fn slack_sum(n_c: &[f64], p_c: &[f64], n_d: &[f64], p_d: &[f64]) -> f64 {
    let s = |v: &[f64]| v.iter().sum::<f64>();
    s(n_c) + s(p_c) + s(n_d) + s(p_d)
}

/// Scalar core of `RestoIpoptNLP::grad_f` at
/// `IpRestoIpoptNLP.cpp:504-525`:
/// * On the `n_c, p_c, n_d, p_d` blocks the gradient is `rho`.
/// * On the `x` block the gradient is
///   `eta(mu) * D_R^2 * (x - x_R)`.
///
/// The caller writes `rho` into the slack-block slots itself; this
/// helper fills the `x`-block slot.
pub fn restoration_grad_x(eta: f64, x: &[f64], x_ref: &[f64], dr_x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), x_ref.len());
    debug_assert_eq!(x.len(), dr_x.len());
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        let d = dr_x[i];
        out[i] = eta * d * d * (x[i] - x_ref[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dr_x_clamps_to_one_when_x_ref_small() {
        let x_ref = [0.0, 0.5, -0.99];
        let mut dr = [0.0; 3];
        build_dr_x(&x_ref, &mut dr);
        // |x_ref| < 1 → max(1, |x_ref|) = 1 → dr = 1.
        assert_eq!(dr, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn dr_x_takes_reciprocal_when_x_ref_large() {
        let x_ref = [2.0, -10.0];
        let mut dr = [0.0; 2];
        build_dr_x(&x_ref, &mut dr);
        assert_eq!(dr, [0.5, 0.1]);
    }

    #[test]
    fn objective_matches_hand_computation() {
        let x = [1.5, 2.0];
        let x_ref = [1.0, 2.0];
        let dr_x = [1.0, 0.5];
        let slack_sum = 3.0;
        let rho = 1e3;
        let eta = 1e-2;
        // |D_R(x - x_R)|^2 = (1*(0.5))^2 + (0.5*0)^2 = 0.25
        // f = 1000*3 + 0.5*0.01*0.25 = 3000 + 0.00125
        let f = restoration_objective(rho, eta, slack_sum, &x, &x_ref, &dr_x);
        assert!((f - (3000.0 + 0.00125)).abs() < 1e-12);
    }

    #[test]
    fn grad_x_zero_when_at_reference() {
        let x = [1.0, 2.0, 3.0];
        let x_ref = [1.0, 2.0, 3.0];
        let dr_x = [1.0, 1.0, 1.0];
        let mut g = [42.0; 3];
        restoration_grad_x(1e-2, &x, &x_ref, &dr_x, &mut g);
        assert_eq!(g, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn grad_x_squared_dr_scaling() {
        let x = [3.0];
        let x_ref = [1.0];
        let dr_x = [0.5]; // dr^2 = 0.25
        let mut g = [0.0; 1];
        restoration_grad_x(2.0, &x, &x_ref, &dr_x, &mut g);
        // g = 2 * 0.25 * (3 - 1) = 1.0
        assert!((g[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn dr2_x_squares_dr_elementwise() {
        let dr = [0.5, 1.0, 0.25];
        let mut out = [0.0; 3];
        build_dr2_x(&dr, &mut out);
        assert_eq!(out, [0.25, 1.0, 0.0625]);
    }

    #[test]
    fn proximity_weights_packs_x_ref_dr_dr2() {
        let w = RestoProximityWeights::from_x_ref(&[0.0, -2.0, 4.0]);
        assert_eq!(w.x_ref, [0.0, -2.0, 4.0]);
        // dr = 1/max(1,|x_ref|) = [1, 1/2, 1/4]
        assert_eq!(w.dr_x, [1.0, 0.5, 0.25]);
        // dr2 = elementwise square
        assert_eq!(w.dr2_x, [1.0, 0.25, 0.0625]);
    }

    #[test]
    fn constraint_c_combines_orig_n_p_with_correct_signs() {
        let c_orig = [1.0, -2.0];
        let n_c = [0.5, 1.0];
        let p_c = [0.25, 0.5];
        let mut out = [0.0; 2];
        restoration_constraint_c(&c_orig, &n_c, &p_c, &mut out);
        // c + n - p
        assert_eq!(out, [1.0 + 0.5 - 0.25, -2.0 + 1.0 - 0.5]);
    }

    #[test]
    fn constraint_d_combines_orig_n_p_with_correct_signs() {
        let d_orig = [3.0];
        let n_d = [1.0];
        let p_d = [0.5];
        let mut out = [0.0; 1];
        restoration_constraint_d(&d_orig, &n_d, &p_d, &mut out);
        assert_eq!(out, [3.5]);
    }

    #[test]
    fn fill_initial_slacks_sets_all_to_one() {
        let mut s = [0.0, -1.0, 5.0];
        fill_initial_slacks(&mut s);
        assert_eq!(s, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn slack_sum_adds_all_four_blocks() {
        let s = slack_sum(&[1.0, 2.0], &[3.0], &[4.0, 5.0], &[6.0]);
        assert_eq!(s, 21.0);
    }

    #[test]
    fn objective_with_proximity_weights_struct_matches_scalar_form() {
        // Demonstrate the struct + scalar core compose to upstream's f.
        let x_ref = [1.0, 2.0];
        let weights = RestoProximityWeights::from_x_ref(&x_ref);
        let n_c = [0.5];
        let p_c = [0.25];
        let n_d = [0.1, 0.2];
        let p_d = [0.05, 0.15];
        let x = [1.5, 2.0];
        let mu = 0.04;

        let nlp = RestoIpoptNlp::new(2, 1, 2, &x_ref, 1e3, 1.0);
        let eta = nlp.eta(mu);
        let s_sum = slack_sum(&n_c, &p_c, &n_d, &p_d);
        let f = restoration_objective(nlp.rho, eta, s_sum, &x, &weights.x_ref, &weights.dr_x);

        // Hand-compute: rho=1e3, eta = 1*sqrt(0.04) = 0.2,
        // slack_sum = 0.5+0.25+0.1+0.2+0.05+0.15 = 1.25,
        // dr = [1, 0.5], (x - x_ref) = [0.5, 0],
        // ||D_R(x - x_ref)||^2 = (1*0.5)^2 + (0.5*0)^2 = 0.25,
        // f = 1e3*1.25 + 0.5*0.2*0.25 = 1250 + 0.025
        assert!((f - 1250.025).abs() < 1e-12, "f = {f}");
    }

    #[test]
    fn eta_scales_by_sqrt_mu() {
        let nlp = RestoIpoptNlp::new(1, 0, 0, &[0.0], 1e3, 0.5);
        let mu = 0.04;
        // 0.5 * sqrt(0.04) = 0.5 * 0.2 = 0.1
        assert!((nlp.eta(mu) - 0.1).abs() < 1e-15);
    }

    fn new_nlp_2_1_1() -> RestoIpoptNlp {
        // Orig dims: n=2, m_eq=1, m_ineq=1.
        RestoIpoptNlp::new(2, 1, 1, &[1.0, 2.0], 1e3, 1.0)
    }

    #[test]
    fn new_builds_x_ref_dr_x_dr2_x_consistent() {
        let nlp = new_nlp_2_1_1();
        assert_eq!(nlp.x_ref.values(), &[1.0, 2.0]);
        // dr = 1/max(1,|x_ref|) = [1, 0.5]
        assert_eq!(nlp.dr_x.values(), &[1.0, 0.5]);
        // dr2 = elementwise square
        assert_eq!(nlp.dr2_x.values(), &[1.0, 0.25]);
    }

    #[test]
    fn x_space_has_five_blocks_with_correct_dims() {
        let nlp = new_nlp_2_1_1();
        assert_eq!(nlp.x_space.n_comp_spaces(), 5);
        assert_eq!(nlp.x_space.comp_dim(BLOCK_X), 2);
        assert_eq!(nlp.x_space.comp_dim(BLOCK_N_C), 1);
        assert_eq!(nlp.x_space.comp_dim(BLOCK_P_C), 1);
        assert_eq!(nlp.x_space.comp_dim(BLOCK_N_D), 1);
        assert_eq!(nlp.x_space.comp_dim(BLOCK_P_D), 1);
        // total dim = 2 + 1 + 1 + 1 + 1 = 6
        assert_eq!(nlp.x_space.dim(), 6);
    }

    #[test]
    fn init_starting_x_seeds_x_ref_and_unit_slacks() {
        let nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        let x0 = x
            .comp(BLOCK_X)
            .as_any()
            .downcast_ref::<DenseVector>()
            .unwrap();
        assert_eq!(x0.values(), &[1.0, 2.0]);
        // Slack blocks should be 1.0 (homogeneous after `set(1.0)`).
        for &k in &[BLOCK_N_C, BLOCK_P_C, BLOCK_N_D, BLOCK_P_D] {
            let s = x.comp(k).as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(s.expanded_values(), vec![1.0]);
        }
    }

    #[test]
    fn f_with_mu_matches_scalar_kernel() {
        let nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        // orig_x = [1.5, 2.0], slacks = 1.0 each.
        nlp.init_starting_x(&mut x);
        x.comp_mut(BLOCK_X)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[1.5, 2.0]);
        let mu = 0.04;
        let f = nlp.f_with_mu(&x, mu);

        // Expected: rho * (1+1+1+1) + 0.5 * 0.2 * (||D_R(x-x_ref)||^2)
        //   D_R(x-x_ref) = (1*0.5, 0.5*0) = (0.5, 0); norm^2 = 0.25
        //   f = 1000*4 + 0.5*0.2*0.25 = 4000 + 0.025 = 4000.025
        assert!((f - 4000.025).abs() < 1e-9, "f = {f}");
    }

    #[test]
    fn grad_f_with_mu_writes_eta_dr2_x_into_orig_block_and_rho_into_slacks() {
        let nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        x.comp_mut(BLOCK_X)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[3.0, 4.0]);

        let mut g = nlp.make_new_x();
        // Pre-fill with a sentinel so we can verify the slack blocks
        // get overwritten.
        for k in 0..5 {
            g.comp_mut(k).set(-99.0);
        }

        let mu = 1.0;
        nlp.grad_f_with_mu(&x, mu, &mut g);

        // eta = 1*sqrt(1) = 1; dr2 = [1, 0.25]; (x - x_ref) = [2, 2]
        // → grad_x = [1*1*2, 1*0.25*2] = [2.0, 0.5]
        let g0 = g
            .comp(BLOCK_X)
            .as_any()
            .downcast_ref::<DenseVector>()
            .unwrap();
        assert_eq!(g0.values(), &[2.0, 0.5]);

        // Slack blocks: g = rho.
        for &k in &[BLOCK_N_C, BLOCK_P_C, BLOCK_N_D, BLOCK_P_D] {
            let s = g.comp(k).as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(s.expanded_values(), vec![1e3]);
        }
    }

    #[test]
    fn c_resto_adds_n_c_minus_p_c_to_orig_c() {
        let nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        // n_c = 0.5, p_c = 0.25
        x.comp_mut(BLOCK_N_C)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[0.5]);
        x.comp_mut(BLOCK_P_C)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[0.25]);

        let space = DenseVectorSpace::new(1);
        let mut c_orig = DenseVector::new(Rc::clone(&space));
        c_orig.set_values(&[1.0]);
        let mut out = DenseVector::new(Rc::clone(&space));
        out.set_values(&[0.0]);

        nlp.c_resto(&c_orig, &x, &mut out);
        // 1.0 + 0.5 - 0.25 = 1.25
        assert_eq!(out.values(), &[1.25]);
    }

    #[test]
    fn d_resto_adds_n_d_minus_p_d_to_orig_d() {
        let nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        x.comp_mut(BLOCK_N_D)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[2.0]);
        x.comp_mut(BLOCK_P_D)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[0.5]);

        let space = DenseVectorSpace::new(1);
        let mut d_orig = DenseVector::new(Rc::clone(&space));
        d_orig.set_values(&[3.0]);
        let mut out = DenseVector::new(Rc::clone(&space));
        out.set_values(&[0.0]);

        nlp.d_resto(&d_orig, &x, &mut out);
        // 3.0 + 2.0 - 0.5 = 4.5
        assert_eq!(out.values(), &[4.5]);
    }

    #[test]
    fn nlp_trait_n_returns_compound_total_dim() {
        let nlp = new_nlp_2_1_1();
        // 2 (orig) + 1 (n_c) + 1 (p_c) + 1 (n_d) + 1 (p_d) = 6
        assert_eq!(<RestoIpoptNlp as Nlp>::n(&nlp), 6);
    }

    #[test]
    fn nlp_trait_m_eq_m_ineq_match_constructor_args() {
        let nlp = new_nlp_2_1_1();
        assert_eq!(<RestoIpoptNlp as Nlp>::m_eq(&nlp), 1);
        assert_eq!(<RestoIpoptNlp as Nlp>::m_ineq(&nlp), 1);
    }

    #[test]
    fn nlp_trait_eval_f_routes_through_curr_mu() {
        let mut nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        x.comp_mut(BLOCK_X)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[1.5, 2.0]);
        let mu = 0.04;
        let direct = nlp.f_with_mu(&x, mu);
        nlp.set_curr_mu(mu);
        let via_trait = <RestoIpoptNlp as Nlp>::eval_f(&mut nlp, &x);
        assert_eq!(direct, via_trait);
    }

    #[test]
    fn nlp_trait_eval_grad_f_routes_through_curr_mu() {
        let mut nlp = new_nlp_2_1_1();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        x.comp_mut(BLOCK_X)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[3.0, 4.0]);

        let mu = 1.0;
        let mut g_direct = nlp.make_new_x();
        for k in 0..5 {
            g_direct.comp_mut(k).set(0.0);
        }
        nlp.grad_f_with_mu(&x, mu, &mut g_direct);

        nlp.set_curr_mu(mu);
        let mut g_trait = nlp.make_new_x();
        for k in 0..5 {
            g_trait.comp_mut(k).set(0.0);
        }
        <RestoIpoptNlp as Nlp>::eval_grad_f(&mut nlp, &x, &mut g_trait);

        // Compare each block's expanded values.
        for k in 0..5 {
            let a = g_direct
                .comp(k)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .expanded_values();
            let b = g_trait
                .comp(k)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .expanded_values();
            assert_eq!(a, b, "block {k}");
        }
    }

    /// Minimal `Nlp` + `IpoptNlp` mock used by the orig-delegation
    /// tests below. Returns deterministic, hand-computable values for
    /// the four constraint/Jacobian/Hessian methods. `eval_f` /
    /// `eval_grad_f` are unused by the resto wrapper's delegation
    /// paths and are stubbed to panic so any accidental call is
    /// caught loudly.
    ///
    /// All orig variables are taken to be both lower- and upper-
    /// bounded (`n_xL = n_xU = n`), and the lone inequality has both
    /// bounds, so `Px_L = Px_U = I_n` and `Pd_L = Pd_U = I_{m_ineq}`.
    struct MockOrigNlp {
        n: Index,
        m_eq: Index,
        m_ineq: Index,
        x_l: DenseVector,
        x_u: DenseVector,
        d_l: DenseVector,
        d_u: DenseVector,
        px_l: Rc<dyn Matrix>,
        px_u: Rc<dyn Matrix>,
        pd_l: Rc<dyn Matrix>,
        pd_u: Rc<dyn Matrix>,
    }

    impl Nlp for MockOrigNlp {
        fn n(&self) -> Index {
            self.n
        }
        fn m_eq(&self) -> Index {
            self.m_eq
        }
        fn m_ineq(&self) -> Index {
            self.m_ineq
        }
        fn eval_f(&mut self, _x: &dyn Vector) -> Number {
            unreachable!("MockOrigNlp::eval_f is not used by the resto delegation tests")
        }
        fn eval_grad_f(&mut self, _x: &dyn Vector, _g: &mut dyn Vector) {
            unreachable!("MockOrigNlp::eval_grad_f is not used by the resto delegation tests")
        }
        fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
            // c(x) = [x[0] + x[1] - 3]
            let xv = downcast_dense(x).expanded_values();
            downcast_dense_mut(c).set_values(&[xv[0] + xv[1] - 3.0]);
        }
        fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
            // d(x) = [x[0] * x[1] - 0.5]
            let xv = downcast_dense(x).expanded_values();
            downcast_dense_mut(d).set_values(&[xv[0] * xv[1] - 0.5]);
        }
        fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            // ∂c/∂x = [1, 1] (1 row, 2 cols, dense triplets)
            let space = pounce_linalg::triplet::GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
            let mut m = pounce_linalg::triplet::GenTMatrix::new(space);
            m.set_values(&[1.0, 1.0]);
            Rc::new(m)
        }
        fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
            // ∂d/∂x = [x[1], x[0]]
            let xv = downcast_dense(x).expanded_values();
            let space = pounce_linalg::triplet::GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
            let mut m = pounce_linalg::triplet::GenTMatrix::new(space);
            m.set_values(&[xv[1], xv[0]]);
            Rc::new(m)
        }
        fn eval_h(
            &mut self,
            _x: &dyn Vector,
            obj_factor: Number,
            _y_c: &dyn Vector,
            y_d: &dyn Vector,
        ) -> Rc<dyn SymMatrix> {
            // Triplet form (lower-tri, 1-based): H[1,1] = obj_factor,
            // H[2,1] = y_d[0], H[2,2] = obj_factor.
            //
            // Mirrors `OrigIpoptNlp::eval_h` which produces a
            // [`SymTMatrix`] from the user TNLP's triplet sparsity —
            // the resto-side flat-Hessian path requires that shape.
            let yd = downcast_dense(y_d).expanded_values();
            let space =
                pounce_linalg::triplet::SymTMatrixSpace::new(2, vec![1, 2, 2], vec![1, 1, 2]);
            let mut h = pounce_linalg::triplet::SymTMatrix::new(space);
            h.set_values(&[obj_factor, yd[0], obj_factor]);
            Rc::new(h)
        }
    }

    impl IpoptNlp for MockOrigNlp {
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

    fn build_mock_orig(n: Index, m_eq: Index, m_ineq: Index) -> MockOrigNlp {
        let xl_space = DenseVectorSpace::new(n);
        let mut x_l = DenseVector::new(Rc::clone(&xl_space));
        x_l.set(0.0);
        let mut x_u = DenseVector::new(Rc::clone(&xl_space));
        x_u.set(10.0);
        let dl_space = DenseVectorSpace::new(m_ineq);
        let mut d_l = DenseVector::new(Rc::clone(&dl_space));
        d_l.set(-5.0);
        let mut d_u = DenseVector::new(Rc::clone(&dl_space));
        d_u.set(5.0);
        let px_l: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(n));
        let px_u: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(n));
        let pd_l: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(m_ineq));
        let pd_u: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(m_ineq));
        MockOrigNlp {
            n,
            m_eq,
            m_ineq,
            x_l,
            x_u,
            d_l,
            d_u,
            px_l,
            px_u,
            pd_l,
            pd_u,
        }
    }

    fn nlp_with_orig() -> RestoIpoptNlp {
        let mut nlp = new_nlp_2_1_1();
        let orig: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(build_mock_orig(2, 1, 1)));
        nlp.set_orig_nlp(orig);
        nlp
    }

    fn set_block_x(x: &mut CompoundVector, vals: &[f64]) {
        x.comp_mut(BLOCK_X)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(vals);
    }
    fn set_slack(x: &mut CompoundVector, block: Index, val: f64) {
        x.comp_mut(block)
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .unwrap()
            .set_values(&[val]);
    }

    #[test]
    fn nlp_eval_c_delegates_to_orig_then_adds_n_c_minus_p_c() {
        let mut nlp = nlp_with_orig();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        // x_orig = [2, 2] → c_orig = 2+2-3 = 1; n_c=0.7, p_c=0.2 → c_R=1.5
        set_block_x(&mut x, &[2.0, 2.0]);
        set_slack(&mut x, BLOCK_N_C, 0.7);
        set_slack(&mut x, BLOCK_P_C, 0.2);

        let c_space = DenseVectorSpace::new(1);
        let mut c_out = DenseVector::new(c_space);
        c_out.set_values(&[0.0]);
        <RestoIpoptNlp as Nlp>::eval_c(&mut nlp, &x, &mut c_out);
        assert!((c_out.values()[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn nlp_eval_d_delegates_to_orig_then_adds_n_d_minus_p_d() {
        let mut nlp = nlp_with_orig();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        // x_orig = [3, 4] → d_orig = 12 - 0.5 = 11.5; n_d=0.25, p_d=0.5 → 11.25
        set_block_x(&mut x, &[3.0, 4.0]);
        set_slack(&mut x, BLOCK_N_D, 0.25);
        set_slack(&mut x, BLOCK_P_D, 0.5);

        let d_space = DenseVectorSpace::new(1);
        let mut d_out = DenseVector::new(d_space);
        d_out.set_values(&[0.0]);
        <RestoIpoptNlp as Nlp>::eval_d(&mut nlp, &x, &mut d_out);
        assert!((d_out.values()[0] - 11.25).abs() < 1e-12);
    }

    #[test]
    fn nlp_eval_jac_c_assembles_flat_gent_with_signed_identities_at_n_c_p_c() {
        let mut nlp = nlp_with_orig();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        set_block_x(&mut x, &[1.0, 1.0]);

        let jac = <RestoIpoptNlp as Nlp>::eval_jac_c(&mut nlp, &x);
        assert_eq!(jac.n_rows(), 1);
        // Total cols = n_orig + 2*m_eq + 2*m_ineq = 2 + 1 + 1 + 1 + 1 = 6
        assert_eq!(jac.n_cols(), 6);

        let gt = jac
            .as_any()
            .downcast_ref::<pounce_linalg::triplet::GenTMatrix>()
            .unwrap();
        // Triplets: orig (2 entries at cols 1,2) + +I at col 3 (n_c) + -I
        // at col 4 (p_c). c_R = c + n_c − p_c → ∂c_R/∂n_c = +1,
        // ∂c_R/∂p_c = −1.
        let irows = gt.irows();
        let jcols = gt.jcols();
        let vals = gt.values();
        assert_eq!(gt.nonzeros() as usize, 4);
        assert_eq!(irows, &[1, 1, 1, 1]);
        assert_eq!(jcols, &[1, 2, 3, 4]);
        assert_eq!(vals, &[1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn nlp_eval_jac_d_assembles_flat_gent_with_signed_identities_at_n_d_p_d() {
        let mut nlp = nlp_with_orig();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        set_block_x(&mut x, &[1.5, 2.5]);

        let jac = <RestoIpoptNlp as Nlp>::eval_jac_d(&mut nlp, &x);
        assert_eq!(jac.n_rows(), 1);
        assert_eq!(jac.n_cols(), 6);

        let gt = jac
            .as_any()
            .downcast_ref::<pounce_linalg::triplet::GenTMatrix>()
            .unwrap();
        // Triplets: orig (2 entries at cols 1,2) + +I at col 5 (n_d) + -I
        // at col 6 (p_d). d_R = d + n_d − p_d → ∂d_R/∂n_d = +1,
        // ∂d_R/∂p_d = −1.
        let irows = gt.irows();
        let jcols = gt.jcols();
        let vals = gt.values();
        assert_eq!(gt.nonzeros() as usize, 4);
        assert_eq!(irows, &[1, 1, 1, 1]);
        assert_eq!(jcols, &[1, 2, 5, 6]);
        // orig vals: [x[1], x[0]] = [2.5, 1.5]
        assert_eq!(vals, &[2.5, 1.5, 1.0, -1.0]);
    }

    #[test]
    fn ipopt_nlp_x_l_is_five_block_with_orig_at_block_zero() {
        let nlp = nlp_with_orig();
        let x_l = <RestoIpoptNlp as IpoptNlp>::x_l(&nlp);
        // Total compressed lower-bound dim = n_xL_orig + 2*m_eq + 2*m_ineq
        // = 2 + 1 + 1 + 1 + 1 = 6.
        assert_eq!(x_l.dim(), 6);
        let cv = x_l.as_any().downcast_ref::<CompoundVector>().unwrap();
        assert_eq!(cv.n_comps(), 5);
        // Block 0 ← orig.x_l = [0, 0]; slack blocks = 0.
        let b0 = cv.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(b0.values(), &[0.0, 0.0]);
        for k in 1..5 {
            let bk = cv.comp(k).as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(bk.expanded_values(), vec![0.0]);
        }
    }

    #[test]
    fn ipopt_nlp_x_u_is_dense_clone_of_orig() {
        let nlp = nlp_with_orig();
        let x_u = <RestoIpoptNlp as IpoptNlp>::x_u(&nlp);
        // Slacks have no upper bound, so resto x_u dim = n_xU_orig = 2.
        assert_eq!(x_u.dim(), 2);
        let dv = x_u.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.values(), &[10.0, 10.0]);
    }

    #[test]
    fn ipopt_nlp_d_l_d_u_match_orig() {
        let nlp = nlp_with_orig();
        let d_l = <RestoIpoptNlp as IpoptNlp>::d_l(&nlp);
        let d_u = <RestoIpoptNlp as IpoptNlp>::d_u(&nlp);
        assert_eq!(
            d_l.as_any().downcast_ref::<DenseVector>().unwrap().values(),
            &[-5.0]
        );
        assert_eq!(
            d_u.as_any().downcast_ref::<DenseVector>().unwrap().values(),
            &[5.0]
        );
    }

    #[test]
    fn ipopt_nlp_px_l_is_block_diagonal_with_orig_then_identities() {
        let nlp = nlp_with_orig();
        let px_l = <RestoIpoptNlp as IpoptNlp>::px_l(&nlp);
        // Rows = total resto x = 6; cols = total resto x_L = 6.
        assert_eq!(px_l.n_rows(), 6);
        assert_eq!(px_l.n_cols(), 6);
        let cm = px_l.as_any().downcast_ref::<CompoundMatrix>().unwrap();
        // Block (0,0): orig.px_l (IdentityMatrix dim 2 in mock).
        assert!(cm.get_comp(0, 0).is_some());
        // Slack-block diagonals: identity each.
        for k in 1..5 {
            let blk = cm.get_comp(k, k).expect("slack-diagonal block");
            let id = blk.as_any().downcast_ref::<IdentityMatrix>().unwrap();
            assert_eq!(id.factor(), 1.0);
        }
        // Off-diagonal slack-row/col blocks remain zero.
        assert!(cm.get_comp(1, 0).is_none());
        assert!(cm.get_comp(0, 1).is_none());
    }

    #[test]
    fn ipopt_nlp_px_u_only_populates_orig_block() {
        let nlp = nlp_with_orig();
        let px_u = <RestoIpoptNlp as IpoptNlp>::px_u(&nlp);
        assert_eq!(px_u.n_rows(), 6);
        // Cols = n_xU_orig = 2.
        assert_eq!(px_u.n_cols(), 2);
        let cm = px_u.as_any().downcast_ref::<CompoundMatrix>().unwrap();
        assert!(cm.get_comp(0, 0).is_some());
        for k in 1..5 {
            assert!(cm.get_comp(k, 0).is_none(), "row {k} should be empty");
        }
    }

    #[test]
    fn ipopt_nlp_pd_l_pd_u_pass_through_orig() {
        let nlp = nlp_with_orig();
        let pd_l = <RestoIpoptNlp as IpoptNlp>::pd_l(&nlp);
        let pd_u = <RestoIpoptNlp as IpoptNlp>::pd_u(&nlp);
        // Mock orig uses IdentityMatrix(m_ineq=1) for both; resto
        // re-exports the same Rc.
        assert_eq!(pd_l.n_rows(), 1);
        assert_eq!(pd_l.n_cols(), 1);
        assert!(pd_l.as_any().downcast_ref::<IdentityMatrix>().is_some());
        assert!(pd_u.as_any().downcast_ref::<IdentityMatrix>().is_some());
    }

    #[test]
    fn nlp_eval_h_returns_flat_symt_with_orig_block_plus_proximity_diagonal() {
        let mut nlp = nlp_with_orig();
        let mut x = nlp.make_new_x();
        nlp.init_starting_x(&mut x);
        set_block_x(&mut x, &[1.0, 1.0]);
        nlp.set_curr_mu(1.0); // η = eta_factor · 1.0 = 1.0

        let yc_space = DenseVectorSpace::new(1);
        let mut y_c = DenseVector::new(Rc::clone(&yc_space));
        y_c.set_values(&[0.0]);
        let yd_space = DenseVectorSpace::new(1);
        let mut y_d = DenseVector::new(Rc::clone(&yd_space));
        y_d.set_values(&[0.0]);

        let h = <RestoIpoptNlp as Nlp>::eval_h(&mut nlp, &x, 1.0, &y_c, &y_d);
        // Total dim: 2 + 1 + 1 + 1 + 1 = 6.
        assert_eq!(h.n_rows(), 6);
        assert_eq!(h.n_cols(), 6);

        let sym_t = h
            .as_any()
            .downcast_ref::<pounce_linalg::triplet::SymTMatrix>()
            .expect("eval_h must return a flat SymTMatrix in v0.1");
        // Triplet layout: orig_h's lower-triangular nonzeros (their
        // 1-based indices stay <= n_orig=2) followed by n_orig=2
        // diagonal entries from the proximity term.
        let n_diag = 2;
        let total_nz = sym_t.nonzeros() as usize;
        assert!(
            total_nz >= n_diag,
            "expected at least {n_diag} diagonal entries"
        );
        for k in 0..n_diag {
            let idx = total_nz - n_diag + k;
            assert_eq!(sym_t.irows()[idx], (k + 1) as i32);
            assert_eq!(sym_t.jcols()[idx], (k + 1) as i32);
        }
        // All triplet entries (orig + proximity) live in rows/cols
        // <= n_orig — slack blocks contribute no nonzeros.
        for (&i, &j) in sym_t.irows().iter().zip(sym_t.jcols().iter()) {
            assert!(i <= 2, "row index {i} outside orig block");
            assert!(j <= 2, "col index {j} outside orig block");
        }
    }
}
