//! Restoration-phase iterate initializer — port of
//! `Algorithm/IpRestoIterateInitializer.{hpp,cpp}`.
//!
//! Runs once at the start of every restoration sub-solve, in the inner
//! IPM's first iteration. Produces an [`IteratesVector`] whose layout
//! matches the resto NLP:
//!
//! * `x` — 5-block [`CompoundVector`] `(orig_x, n_c, p_c, n_d, p_d)`
//!   with `orig_x ← outer.x` and the four slack blocks set by the
//!   closed-form [`init_slack_pair`].
//! * `s` — clone of the outer iterate's `s`.
//! * `y_c, y_d` — zeros (upstream `constr_mult_init_max=0` for resto;
//!   the optional [`pounce_algorithm::eq_mult::least_square::LeastSquareMults`]
//!   path lands once the inner aug-system bridge is wired).
//! * `z_l` — 5-block compound matching `x_l_resto`, components:
//!   * block 0: `min(rho, outer.z_l)` (rho-cap on the orig bounds);
//!   * blocks 1..4: `resto_mu / slack` for `(n_c, p_c, n_d, p_d)`.
//! * `z_u` — orig-shape, `min(rho, outer.z_u)` (slacks have no upper
//!   bound so this is a single dense block).
//! * `v_l, v_u` — `min(rho, outer.v_l)`, `min(rho, outer.v_u)` (the
//!   inequality-slack `s` bounds are unchanged from the outer NLP).
//!
//! The outer-iterate snapshot needed for the formula is threaded in via
//! [`RestoIterateInitializer::with_outer_snapshot`]; the inner-solver
//! hook in [`crate::min_c_1nrm`] populates it from the outer
//! `(IpoptData, IpoptCq)` before invoking the inner
//! `IpoptAlgorithm::optimize` loop.

use pounce_algorithm::init::r#trait::IterateInitializer;
use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_algorithm::kkt::aug_system_solver::AugSystemSolver;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::{CompoundVector, CompoundVectorSpace, Vector};
use std::cell::RefCell;
use std::rc::Rc;

use crate::resto_nlp::{BLOCK_N_C, BLOCK_N_D, BLOCK_P_C, BLOCK_P_D, BLOCK_X};

/// Snapshot of the outer iterate captured at restoration entry. The
/// inner-solver hook builds this from the outer `(IpoptData, IpoptCq)`
/// and stashes it on [`RestoIterateInitializer`] before the nested IPM
/// runs. All vectors are cheap-to-clone `Rc<dyn Vector>` handles —
/// the underlying storage is shared with the outer-data snapshots.
pub struct OuterIterateSnapshot {
    /// Outer barrier parameter `μ_outer`. Used for both
    /// `restoration_mu` and the slack-block multiplier formula
    /// `resto_mu / n_*` upstream
    /// (`IpRestoIterateInitializer.cpp:178-188`).
    pub mu: Number,
    /// Outer iterate `s` — copied verbatim into the inner `s`.
    pub s: Rc<dyn Vector>,
    /// Outer `z_L` (matches outer x_l shape).
    pub z_l: Rc<dyn Vector>,
    /// Outer `z_U`.
    pub z_u: Rc<dyn Vector>,
    /// Outer `v_L`.
    pub v_l: Rc<dyn Vector>,
    /// Outer `v_U`.
    pub v_u: Rc<dyn Vector>,
    /// `c(x_outer)` — the equality-constraint residual at the outer
    /// iterate. Drives `init_slack_pair` for `(n_c, p_c)`.
    pub c_vec: Rc<dyn Vector>,
    /// `d(x_outer) − s_outer` — the inequality-residual.
    /// Drives `init_slack_pair` for `(n_d, p_d)`.
    pub d_minus_s_vec: Rc<dyn Vector>,
}

pub struct RestoIterateInitializer {
    /// Penalty coefficient ρ — also caps the orig-bound multipliers
    /// per `IpRestoIterateInitializer.cpp:160-174`.
    pub rho: Number,
    /// Dim of the original problem's `x`. Cloned from the resto NLP at
    /// bundle-build time so we don't need to downcast `&dyn IpoptNlp`.
    pub n_orig: Index,
    /// Dim of the equality-constraint vector `c`.
    pub m_eq: Index,
    /// Dim of the inequality-constraint vector `d`.
    pub m_ineq: Index,
    /// Reference primal `x` (outer iterate at restoration entry).
    pub x_ref_vals: Vec<Number>,
    /// Caches the outer-iterate snapshot. The inner-solver hook calls
    /// [`Self::set_outer_snapshot`] once per restoration entry.
    pub outer: Option<OuterIterateSnapshot>,
}

impl RestoIterateInitializer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dims(n_orig: Index, m_eq: Index, m_ineq: Index, x_ref_vals: Vec<Number>) -> Self {
        Self {
            n_orig,
            m_eq,
            m_ineq,
            x_ref_vals,
            ..Self::default()
        }
    }

    pub fn set_outer_snapshot(&mut self, snap: OuterIterateSnapshot) {
        self.outer = Some(snap);
    }

    pub fn with_outer_snapshot(mut self, snap: OuterIterateSnapshot) -> Self {
        self.outer = Some(snap);
        self
    }

    pub fn with_rho(mut self, rho: Number) -> Self {
        self.rho = rho;
        self
    }
}

impl Default for RestoIterateInitializer {
    fn default() -> Self {
        Self {
            rho: 1e3,
            n_orig: 0,
            m_eq: 0,
            m_ineq: 0,
            x_ref_vals: Vec::new(),
            outer: None,
        }
    }
}

impl IterateInitializer for RestoIterateInitializer {
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        let Some(snap) = self.outer.as_ref() else {
            return false;
        };

        // Bound-vector dims (orig-shape) are read from the resto NLP's
        // IpoptNlp accessors; the rest of the dim/x_ref data is held
        // directly on this initializer (set at bundle-build time).
        // `x_l()` returns a 5-block CompoundVector; block 0 holds the
        // orig-`x` lower-bound rows, so `n_xl_orig = comp(0).dim()`.
        // `x_u()` is dense (slacks have no upper bound), so its `.dim()`
        // already equals `n_xu_orig`.
        let nlp_ref = nlp.borrow();
        let xl_compound = nlp_ref
            .x_l()
            .as_any()
            .downcast_ref::<CompoundVector>()
            .expect("RestoIpoptNlp::x_l must be a 5-block CompoundVector");
        let n_xl_orig = xl_compound.comp(0).dim();
        let n_xu_orig = nlp_ref.x_u().dim();
        let n_dl = nlp_ref.d_l().dim();
        let n_du = nlp_ref.d_u().dim();
        drop(nlp_ref);

        let _n_orig = self.n_orig;
        let m_eq = self.m_eq;
        let m_ineq = self.m_ineq;
        let x_ref_vals = self.x_ref_vals.clone();
        let x_space = build_x_space(self.n_orig, m_eq, m_ineq);

        // ---- inner barrier parameter ----
        let c_amax = snap.c_vec.amax();
        let d_minus_s_amax = snap.d_minus_s_vec.amax();
        let resto_mu = restoration_mu(snap.mu, c_amax, d_minus_s_amax);
        data.borrow_mut().curr_mu = resto_mu;

        // ---- primal x: 5-block compound ----
        let mut x = CompoundVector::new(Rc::clone(&x_space));
        // Block 0: orig_x ← outer.x_ref.
        downcast_dense_mut(x.comp_mut(BLOCK_X)).set_values(&x_ref_vals);

        // Slack blocks via init_slack_pair on each entry of c / d-s.
        let c_vals = expanded_dense_or_panic(&*snap.c_vec, "c residual");
        let dms_vals = expanded_dense_or_panic(&*snap.d_minus_s_vec, "d − s residual");

        let mut nc_vals = vec![0.0; m_eq as usize];
        let mut pc_vals = vec![0.0; m_eq as usize];
        for i in 0..m_eq as usize {
            let (n, p) = init_slack_pair(c_vals[i], resto_mu, self.rho);
            nc_vals[i] = n;
            pc_vals[i] = p;
        }
        let mut nd_vals = vec![0.0; m_ineq as usize];
        let mut pd_vals = vec![0.0; m_ineq as usize];
        for i in 0..m_ineq as usize {
            let (n, p) = init_slack_pair(dms_vals[i], resto_mu, self.rho);
            nd_vals[i] = n;
            pd_vals[i] = p;
        }
        downcast_dense_mut(x.comp_mut(BLOCK_N_C)).set_values(&nc_vals);
        downcast_dense_mut(x.comp_mut(BLOCK_P_C)).set_values(&pc_vals);
        downcast_dense_mut(x.comp_mut(BLOCK_N_D)).set_values(&nd_vals);
        downcast_dense_mut(x.comp_mut(BLOCK_P_D)).set_values(&pd_vals);

        // ---- primal s: clone outer.s ----
        let mut s = DenseVectorSpace::new(m_ineq).make_new_dense();
        let s_outer = expanded_dense_or_panic(&*snap.s, "s (inequality slacks)");
        s.set_values(&s_outer);

        // ---- y_c, y_d: zero ----
        let mut y_c = DenseVectorSpace::new(m_eq).make_new_dense();
        let mut y_d = DenseVectorSpace::new(m_ineq).make_new_dense();
        y_c.set_values(&vec![0.0; m_eq as usize]);
        y_d.set_values(&vec![0.0; m_ineq as usize]);

        // ---- z_l: 5-block compound (n_xl_orig, m_eq, m_eq, m_ineq, m_ineq) ----
        let z_l_total = n_xl_orig + 2 * m_eq + 2 * m_ineq;
        let z_l_space = build_z_l_space(n_xl_orig, m_eq, m_ineq, z_l_total);
        let mut z_l = CompoundVector::new(z_l_space);
        // block 0: min(rho, outer.z_l)
        let outer_zl_vals = expanded_dense_or_panic(&*snap.z_l, "z_l (lower-bound multipliers)");
        let mut zl0 = vec![0.0; n_xl_orig as usize];
        for (i, &v) in outer_zl_vals.iter().enumerate() {
            zl0[i] = self.rho.min(v);
        }
        downcast_dense_mut(z_l.comp_mut(0)).set_values(&zl0);
        // block 1: resto_mu / n_c (componentwise)
        downcast_dense_mut(z_l.comp_mut(1)).set_values(&divide_safe(resto_mu, &nc_vals));
        downcast_dense_mut(z_l.comp_mut(2)).set_values(&divide_safe(resto_mu, &pc_vals));
        downcast_dense_mut(z_l.comp_mut(3)).set_values(&divide_safe(resto_mu, &nd_vals));
        downcast_dense_mut(z_l.comp_mut(4)).set_values(&divide_safe(resto_mu, &pd_vals));

        // ---- z_u: orig-shape (slacks have no upper bound) ----
        let mut z_u = DenseVectorSpace::new(n_xu_orig).make_new_dense();
        let outer_zu_vals = expanded_dense_or_panic(&*snap.z_u, "z_u (upper-bound multipliers)");
        let mut zu = vec![0.0; n_xu_orig as usize];
        for (i, &v) in outer_zu_vals.iter().enumerate() {
            zu[i] = self.rho.min(v);
        }
        z_u.set_values(&zu);

        // ---- v_l, v_u: orig-shape (d-bounds unchanged by resto) ----
        let mut v_l = DenseVectorSpace::new(n_dl).make_new_dense();
        let outer_vl_vals = expanded_dense_or_panic(&*snap.v_l, "v_l (d lower-bound multipliers)");
        let mut vl = vec![0.0; n_dl as usize];
        for (i, &v) in outer_vl_vals.iter().enumerate() {
            vl[i] = self.rho.min(v);
        }
        v_l.set_values(&vl);

        let mut v_u = DenseVectorSpace::new(n_du).make_new_dense();
        let outer_vu_vals = expanded_dense_or_panic(&*snap.v_u, "v_u (d upper-bound multipliers)");
        let mut vu = vec![0.0; n_du as usize];
        for (i, &v) in outer_vu_vals.iter().enumerate() {
            vu[i] = self.rho.min(v);
        }
        v_u.set_values(&vu);

        let iv = IteratesVector::new(
            Rc::new(x),
            Rc::new(s),
            Rc::new(y_c),
            Rc::new(y_d),
            Rc::new(z_l),
            Rc::new(z_u),
            Rc::new(v_l),
            Rc::new(v_u),
        );

        if std::env::var_os("POUNCE_DBG_RESTO_INIT").is_some() {
            use pounce_linalg::Vector;
            fn dump3(label: &str, v: &dyn Vector) {
                tracing::debug!(target: "pounce::restoration",
                    "[PN_RESTO_INIT] {} amax={:.17e} asum={:.17e} nrm2={:.17e}",
                    label,
                    v.amax(),
                    v.asum(),
                    v.nrm2()
                );
            }
            fn dump2(label: &str, v: &dyn Vector) {
                tracing::debug!(target: "pounce::restoration",
                    "[PN_RESTO_INIT] {} amax={:.17e} asum={:.17e}",
                    label,
                    v.amax(),
                    v.asum()
                );
            }
            let cx = iv.x.as_any().downcast_ref::<CompoundVector>().unwrap();
            let czl = iv.z_l.as_any().downcast_ref::<CompoundVector>().unwrap();
            tracing::debug!(target: "pounce::restoration",
                "[PN_RESTO_INIT] resto_mu={:.17e} rho={:.17e}",
                resto_mu, self.rho
            );
            dump3("x_orig  ", &*cx.comp(0));
            dump3("nc      ", &*cx.comp(1));
            dump3("pc      ", &*cx.comp(2));
            dump3("nd      ", &*cx.comp(3));
            dump3("pd      ", &*cx.comp(4));
            dump2("zL_orig ", &*czl.comp(0));
            dump2("zL_nc   ", &*czl.comp(1));
            dump2("zL_pc   ", &*czl.comp(2));
            dump2("zL_nd   ", &*czl.comp(3));
            dump2("zL_pd   ", &*czl.comp(4));
            tracing::debug!(target: "pounce::restoration",
                "[PN_RESTO_INIT] y_c amax={:.17e} y_d amax={:.17e}",
                iv.y_c.amax(),
                iv.y_d.amax()
            );
        }

        data.borrow_mut().set_curr(iv);
        true
    }
}

/// Initial restoration `μ_R` per `IpRestoIterateInitializer.cpp:58`:
/// ```text
///   μ_R = max(curr_mu, ||c||_∞, ||d - s||_∞)
/// ```
pub fn restoration_mu(curr_mu: f64, c_amax: f64, d_minus_s_amax: f64) -> f64 {
    curr_mu.max(c_amax).max(d_minus_s_amax)
}

/// Elementwise quadratic root used by the slack initializer.
///
/// Mirrors `RestoIterateInitializer::solve_quadratic` at
/// `IpRestoIterateInitializer.cpp:216-230`:
/// ```text
///   v = sqrt(a*a + b) + a
/// ```
/// which is the positive root of `n² - 2·a·n - b = 0`.
pub fn solve_quadratic_elem(a: f64, b: f64) -> f64 {
    (a * a + b).sqrt() + a
}

/// Closed-form slack initializer — scalar version of the per-component
/// formula at `IpRestoIterateInitializer.cpp:79-115`:
/// ```text
///   a = μ_R / (2·ρ) − 0.5·c
///   b = c · μ_R / (2·ρ)
///   n = sqrt(a² + b) + a
///   p = c + n
/// ```
/// Returns `(n, p)` for one component. Same formula applies to both
/// the `c` (equality) and `d−s` (inequality) blocks.
pub fn init_slack_pair(c: f64, mu_r: f64, rho: f64) -> (f64, f64) {
    let half = mu_r / (2.0 * rho);
    let a = half - 0.5 * c;
    let b = c * half;
    let n = solve_quadratic_elem(a, b);
    let p = c + n;
    (n, p)
}

/// Build the 5-block CompoundVectorSpace for the resto-`x` variable.
/// Layout `[n_orig, m_eq, m_eq, m_ineq, m_ineq]`. Mirrors
/// [`crate::resto_nlp::RestoIpoptNlp::x_space`] without requiring a
/// downcast to the concrete resto NLP.
fn build_x_space(n_orig: Index, m_eq: Index, m_ineq: Index) -> Rc<CompoundVectorSpace> {
    let total_dim = n_orig + 2 * m_eq + 2 * m_ineq;
    let space = CompoundVectorSpace::new(5, total_dim);
    let s0 = DenseVectorSpace::new(n_orig);
    space.set_comp(BLOCK_X, n_orig, {
        let s = Rc::clone(&s0);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_eq = DenseVectorSpace::new(m_eq);
    space.set_comp(BLOCK_N_C, m_eq, {
        let s = Rc::clone(&s_eq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space.set_comp(BLOCK_P_C, m_eq, {
        let s = Rc::clone(&s_eq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_ineq = DenseVectorSpace::new(m_ineq);
    space.set_comp(BLOCK_N_D, m_ineq, {
        let s = Rc::clone(&s_ineq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space.set_comp(BLOCK_P_D, m_ineq, {
        let s = Rc::clone(&s_ineq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space
}

/// Build the 5-block CompoundVectorSpace matching `x_l_resto`'s shape.
/// Layout `[n_xl_orig, m_eq, m_eq, m_ineq, m_ineq]`. Used for `z_L`,
/// whose dim must match the lower-bound expansion vector.
fn build_z_l_space(
    n_xl_orig: Index,
    m_eq: Index,
    m_ineq: Index,
    total_dim: Index,
) -> Rc<CompoundVectorSpace> {
    let space = CompoundVectorSpace::new(5, total_dim);
    let s0 = DenseVectorSpace::new(n_xl_orig);
    space.set_comp(0, n_xl_orig, {
        let s = Rc::clone(&s0);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_eq = DenseVectorSpace::new(m_eq);
    space.set_comp(1, m_eq, {
        let s = Rc::clone(&s_eq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space.set_comp(2, m_eq, {
        let s = Rc::clone(&s_eq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_ineq = DenseVectorSpace::new(m_ineq);
    space.set_comp(3, m_ineq, {
        let s = Rc::clone(&s_ineq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space.set_comp(4, m_ineq, {
        let s = Rc::clone(&s_ineq);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    space
}

/// Componentwise `mu / x[i]`, with a tiny floor on the divisor so a
/// pathological zero slack doesn't produce inf. The init formula
/// guarantees `n_*`, `p_*` are strictly positive when `c` is finite, so
/// the floor only fires on degenerate inputs.
fn divide_safe(mu: Number, x: &[Number]) -> Vec<Number> {
    x.iter()
        .map(|&v| {
            if v.abs() < 1e-300 {
                mu / 1e-300
            } else {
                mu / v
            }
        })
        .collect()
}

fn downcast_dense_mut(v: &mut dyn Vector) -> &mut DenseVector {
    v.as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("RestoIterateInitializer expected a DenseVector component")
}

/// Expand an outer-snapshot vector block to a dense value slice,
/// panicking with a clear diagnostic if it is not a `DenseVector`.
///
/// The restoration NLP is built entirely from `DenseVector` blocks — the
/// *write* side already asserts this via `downcast_dense_mut`'s `expect`.
/// The read side previously did `…downcast_ref::<DenseVector>().map(…)
/// .unwrap_or_else(|| vec![0.0; dim])`, which silently replaced a
/// non-dense (e.g. compound) block with **zeros**: the restoration start
/// point would be seeded from a zero residual / zero multiplier with no
/// signal, masking the invariant violation. Failing loudly here is
/// strictly better and symmetric with the write side.
fn expanded_dense_or_panic(v: &dyn Vector, what: &str) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_else(|| {
            panic!("RestoIterateInitializer: outer {what} must be a DenseVector (got a non-dense block)")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal non-`DenseVector` `dyn Vector`: a 1-block compound
    /// whose sole component is dense. The compound itself does not
    /// downcast to `DenseVector`, exercising the failed-downcast path.
    fn make_compound(dim: Index) -> CompoundVector {
        let dspace = DenseVectorSpace::new(dim);
        let cspace = CompoundVectorSpace::new(1, dim);
        cspace.set_comp(0, dim, move || Box::new(dspace.make_new_dense()));
        CompoundVector::new(cspace)
    }

    #[test]
    fn expanded_dense_or_panic_returns_values_for_dense() {
        // Happy path: a real DenseVector (incl. the homogeneous case)
        // round-trips its values — guards against the diagnostic fix
        // breaking the normal, all-DenseVector restoration data.
        let mut v = DenseVectorSpace::new(3).make_new_dense();
        v.set_values(&[1.0, -2.0, 3.5]);
        assert_eq!(expanded_dense_or_panic(&v, "test"), vec![1.0, -2.0, 3.5]);
    }

    #[test]
    #[should_panic(expected = "must be a DenseVector")]
    fn expanded_dense_or_panic_panics_on_non_dense() {
        // Regression for M9: a non-DenseVector block must surface a
        // diagnostic, NOT be silently replaced with zeros. Pre-fix this
        // returned `vec![0.0; dim]` and did not panic.
        let cv = make_compound(3);
        let _ = expanded_dense_or_panic(&cv, "z_l (lower-bound multipliers)");
    }

    #[test]
    fn restoration_mu_takes_max() {
        assert_eq!(restoration_mu(0.1, 0.5, 0.2), 0.5);
        assert_eq!(restoration_mu(1.0, 0.5, 0.2), 1.0);
        assert_eq!(restoration_mu(0.1, 0.2, 0.5), 0.5);
    }

    #[test]
    fn solve_quadratic_zero_b_gives_2a_when_a_positive() {
        // sqrt(a^2) + a = 2a for a >= 0.
        assert!((solve_quadratic_elem(3.0, 0.0) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn solve_quadratic_zero_b_gives_zero_when_a_negative() {
        // sqrt(a^2) + a = |a| + a = 0 for a < 0.
        assert!(solve_quadratic_elem(-3.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn init_slack_pair_satisfies_quadratic_root() {
        let c = 0.5;
        let mu_r = 0.1;
        let rho = 1e3;
        let (n, _p) = init_slack_pair(c, mu_r, rho);
        let half = mu_r / (2.0 * rho);
        let a = half - 0.5 * c;
        let b = c * half;
        let resid = n * n - 2.0 * a * n - b;
        assert!(resid.abs() < 1e-15, "residual = {}", resid);
    }

    #[test]
    fn init_slack_pair_p_minus_n_equals_c() {
        let c = -0.7;
        let (n, p) = init_slack_pair(c, 0.05, 1e2);
        assert!((p - n - c).abs() < 1e-15);
    }

    #[test]
    fn init_slack_pair_nonnegative() {
        for &c in &[-1.0, -0.1, 0.0, 0.1, 1.0, 10.0] {
            let (n, p) = init_slack_pair(c, 0.1, 1e3);
            assert!(n >= 0.0, "n = {} for c = {}", n, c);
            assert!(p >= 0.0, "p = {} for c = {}", p, c);
        }
    }

    #[test]
    fn build_z_l_space_has_five_blocks_and_total_dim() {
        let space = build_z_l_space(3, 2, 1, 3 + 2 + 2 + 1 + 1);
        assert_eq!(space.n_comp_spaces(), 5);
        assert_eq!(space.dim(), 9);
    }

    #[test]
    fn divide_safe_handles_zero() {
        let r = divide_safe(0.5, &[1.0, 0.0, 2.0]);
        assert!((r[0] - 0.5).abs() < 1e-15);
        assert!(r[1].is_finite());
        assert!((r[2] - 0.25).abs() < 1e-15);
    }

    /// End-to-end exercise of [`RestoIterateInitializer::set_initial_iterates`].
    /// Verifies the constructed inner iterate has the expected shapes
    /// and that the slack-init formula is applied per block.
    mod end_to_end {
        use super::*;
        use crate::resto_nlp::RestoIpoptNlp;
        use pounce_algorithm::ipopt_cq::IpoptCalculatedQuantities;
        use pounce_algorithm::ipopt_data::IpoptData;
        use pounce_algorithm::kkt::aug_system_solver::{
            AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver,
        };
        use pounce_common::types::Index;
        use pounce_linalg::{IdentityMatrix, Matrix, SymMatrix};
        use pounce_linsol::status::ESymSolverStatus;
        use std::cell::RefCell;
        use std::rc::Rc;

        struct PanickyAugSolver;
        impl AugSystemSolver for PanickyAugSolver {
            fn provides_inertia(&self) -> bool {
                true
            }
            fn number_of_neg_evals(&self) -> Index {
                0
            }
            fn increase_quality(&mut self) -> bool {
                false
            }
            fn last_solve_status(&self) -> ESymSolverStatus {
                ESymSolverStatus::Success
            }
            fn solve(
                &mut self,
                _coeffs: &AugSysCoeffs<'_>,
                _rhs: &AugSysRhs<'_>,
                _sol: &mut AugSysSol<'_>,
                _check_neg_evals: bool,
                _num_neg_evals: Index,
            ) -> ESymSolverStatus {
                unreachable!("RestoIterateInitializer should not call aug_solver in v0.1")
            }
        }

        struct StubOrigNlp {
            x_l: DenseVector,
            x_u: DenseVector,
            d_l: DenseVector,
            d_u: DenseVector,
            px_l: Rc<dyn Matrix>,
            px_u: Rc<dyn Matrix>,
            pd_l: Rc<dyn Matrix>,
            pd_u: Rc<dyn Matrix>,
        }
        impl pounce_algorithm::ipopt_nlp::Nlp for StubOrigNlp {
            fn n(&self) -> Index {
                2
            }
            fn m_eq(&self) -> Index {
                1
            }
            fn m_ineq(&self) -> Index {
                1
            }
            fn eval_f(&mut self, _x: &dyn Vector) -> Number {
                0.0
            }
            fn eval_grad_f(&mut self, _x: &dyn Vector, _g: &mut dyn Vector) {}
            fn eval_c(&mut self, _x: &dyn Vector, _c: &mut dyn Vector) {}
            fn eval_d(&mut self, _x: &dyn Vector, _d: &mut dyn Vector) {}
            fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
                Rc::new(IdentityMatrix::new(1))
            }
            fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
                Rc::new(IdentityMatrix::new(1))
            }
            fn eval_h(
                &mut self,
                _x: &dyn Vector,
                _o: Number,
                _y_c: &dyn Vector,
                _y_d: &dyn Vector,
            ) -> Rc<dyn SymMatrix> {
                let s = pounce_linalg::DenseSymMatrixSpace::new(2);
                Rc::new(pounce_linalg::DenseSymMatrix::new(s))
            }
        }
        impl pounce_algorithm::ipopt_nlp::IpoptNlp for StubOrigNlp {
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

        fn build_resto_nlp() -> Rc<RefCell<dyn pounce_algorithm::ipopt_nlp::IpoptNlp>> {
            let xl_space = DenseVectorSpace::new(2);
            let mut x_l = DenseVector::new(Rc::clone(&xl_space));
            x_l.set_values(&[0.0, 0.0]);
            let mut x_u = DenseVector::new(Rc::clone(&xl_space));
            x_u.set_values(&[10.0, 10.0]);
            let dl_space = DenseVectorSpace::new(1);
            let mut d_l = DenseVector::new(Rc::clone(&dl_space));
            d_l.set_values(&[-5.0]);
            let mut d_u = DenseVector::new(Rc::clone(&dl_space));
            d_u.set_values(&[5.0]);
            let stub = StubOrigNlp {
                x_l,
                x_u,
                d_l,
                d_u,
                px_l: Rc::new(IdentityMatrix::new(2)),
                px_u: Rc::new(IdentityMatrix::new(2)),
                pd_l: Rc::new(IdentityMatrix::new(1)),
                pd_u: Rc::new(IdentityMatrix::new(1)),
            };
            let orig: Rc<RefCell<dyn pounce_algorithm::ipopt_nlp::IpoptNlp>> =
                Rc::new(RefCell::new(stub));
            let mut resto = RestoIpoptNlp::new(2, 1, 1, &[1.5, 2.5], 1e3, 1e-4);
            resto.set_orig_nlp(orig);
            Rc::new(RefCell::new(resto))
        }

        fn build_outer_snapshot() -> OuterIterateSnapshot {
            // Outer mu is small; ||c||_∞ is large → restoration_mu picks ||c||.
            let c_space = DenseVectorSpace::new(1);
            let mut c_vec = DenseVector::new(Rc::clone(&c_space));
            c_vec.set_values(&[2.0]);
            let mut dms = DenseVector::new(Rc::clone(&c_space));
            dms.set_values(&[0.5]);

            let s_space = DenseVectorSpace::new(1);
            let mut s = DenseVector::new(Rc::clone(&s_space));
            s.set_values(&[0.0]);

            let xl_space = DenseVectorSpace::new(2);
            let mut z_l = DenseVector::new(Rc::clone(&xl_space));
            z_l.set_values(&[5.0, 1e6]); // 1e6 will be capped by rho=1e3.
            let mut z_u = DenseVector::new(Rc::clone(&xl_space));
            z_u.set_values(&[2.0, 3.0]);

            let dl_space = DenseVectorSpace::new(1);
            let mut v_l = DenseVector::new(Rc::clone(&dl_space));
            v_l.set_values(&[7.0]);
            let mut v_u = DenseVector::new(Rc::clone(&dl_space));
            v_u.set_values(&[2.5e3]); // capped by rho.

            OuterIterateSnapshot {
                mu: 0.01,
                s: Rc::new(s),
                z_l: Rc::new(z_l),
                z_u: Rc::new(z_u),
                v_l: Rc::new(v_l),
                v_u: Rc::new(v_u),
                c_vec: Rc::new(c_vec),
                d_minus_s_vec: Rc::new(dms),
            }
        }

        fn build_data_and_cq(
            nlp: &Rc<RefCell<dyn pounce_algorithm::ipopt_nlp::IpoptNlp>>,
        ) -> (
            pounce_algorithm::ipopt_data::IpoptDataHandle,
            pounce_algorithm::ipopt_cq::IpoptCqHandle,
        ) {
            let data = Rc::new(RefCell::new(IpoptData::new()));
            let cq = Rc::new(RefCell::new(IpoptCalculatedQuantities::new(
                Rc::clone(&data),
                Rc::clone(nlp),
            )));
            (data, cq)
        }

        #[test]
        fn set_initial_iterates_builds_well_shaped_iv() {
            let nlp = build_resto_nlp();
            let (data, cq) = build_data_and_cq(&nlp);
            let mut init = RestoIterateInitializer::with_dims(2, 1, 1, vec![1.5, 2.5])
                .with_outer_snapshot(build_outer_snapshot())
                .with_rho(1e3);
            let mut aug = PanickyAugSolver;

            assert!(init.set_initial_iterates(&data, &cq, &nlp, &mut aug));

            let d = data.borrow();
            let curr = d.curr.as_ref().expect("curr installed");

            // mu_R = max(0.01, 2.0, 0.5) = 2.0.
            assert!((d.curr_mu - 2.0).abs() < 1e-15, "mu_R = {}", d.curr_mu);

            // x is a 5-block CompoundVector with dims (2, 1, 1, 1, 1).
            let xc = curr
                .x
                .as_any()
                .downcast_ref::<CompoundVector>()
                .expect("inner x must be CompoundVector");
            assert_eq!(xc.n_comps(), 5);
            assert_eq!(xc.comp(BLOCK_X).dim(), 2);
            assert_eq!(xc.comp(BLOCK_N_C).dim(), 1);
            assert_eq!(xc.comp(BLOCK_P_D).dim(), 1);

            // BLOCK_X copies x_ref.
            let x0 = xc
                .comp(BLOCK_X)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap();
            assert_eq!(x0.values(), &[1.5, 2.5]);

            // n_c, p_c match init_slack_pair(c=2, mu_R=2, rho=1e3).
            let (n_exp, p_exp) = init_slack_pair(2.0, 2.0, 1e3);
            let nc = xc
                .comp(BLOCK_N_C)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .values()[0];
            let pc = xc
                .comp(BLOCK_P_C)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .values()[0];
            assert!((nc - n_exp).abs() < 1e-15);
            assert!((pc - p_exp).abs() < 1e-15);

            // s is dim m_ineq, copied from outer.s = [0.0].
            assert_eq!(curr.s.dim(), 1);

            // z_l block 0 is min(rho, outer.z_l) = [min(1e3, 5), min(1e3, 1e6)] = [5, 1e3].
            let zl = curr.z_l.as_any().downcast_ref::<CompoundVector>().unwrap();
            assert_eq!(zl.n_comps(), 5);
            let zl0 = zl.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(zl0.values(), &[5.0, 1e3]);

            // z_l block 1 = mu_R / n_c.
            let zl1 = zl.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
            assert!((zl1.values()[0] - 2.0 / nc).abs() < 1e-12);

            // z_u capped by rho: outer was [2, 3] both well below.
            let zu = curr.z_u.as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(zu.values(), &[2.0, 3.0]);

            // v_u = min(rho=1e3, outer 2.5e3) = 1e3.
            let vu = curr.v_u.as_any().downcast_ref::<DenseVector>().unwrap();
            assert_eq!(vu.values(), &[1e3]);
        }

        #[test]
        fn set_initial_iterates_returns_false_without_outer_snapshot() {
            let nlp = build_resto_nlp();
            let (data, cq) = build_data_and_cq(&nlp);
            let mut init = RestoIterateInitializer::with_dims(2, 1, 1, vec![1.5, 2.5]);
            let mut aug = PanickyAugSolver;
            assert!(!init.set_initial_iterates(&data, &cq, &nlp, &mut aug));
        }
    }
}
