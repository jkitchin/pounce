//! Warm-start iterate initializer — port of
//! `IpWarmStartIterateInitializer.{hpp,cpp}`. Used when a previous
//! solve has left a trial point that should be reused.
//!
//! There are two callers we serve:
//!
//! * **`IpoptApplication::ReOptimizeTNLP`** (the upstream re-solve
//!   path) populates `data.curr` with the previous solve's iterate.
//!   We clamp multipliers and optionally override `mu`.
//! * **First solves from `OptimizeTNLP`** that opt into
//!   `warm_start_init_point=yes` to forward user-supplied
//!   primal/dual seeds via `TNLP::get_starting_point`. Here
//!   `data.curr` carries only dim metadata (uninitialized vectors);
//!   we pull seeds from the NLP, push primals/slacks into the bound
//!   interior with warm-start `bound_push`/`bound_frac`, and then
//!   apply the same multiplier clamps.
//!
//! Wired options today: `bound_push`, `bound_frac`,
//! `slack_bound_push`, `slack_bound_frac`, `mult_init_max`,
//! `target_mu`. The remaining knobs (`mult_bound_push`,
//! `entire_iterate`, `same_structure`) are stored but not yet
//! consumed.

use crate::alg_builder::WarmStartOptions;
use crate::init::default::push_x_into_interior;
use crate::init::r#trait::IterateInitializer;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::aug_system_solver::AugSystemSolver;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::Vector;
use std::cell::RefCell;
use std::rc::Rc;

pub struct WarmStartIterateInitializer {
    opts: WarmStartOptions,
}

impl WarmStartIterateInitializer {
    pub fn new() -> Self {
        Self {
            opts: WarmStartOptions::default(),
        }
    }

    pub fn with_options(opts: WarmStartOptions) -> Self {
        Self { opts }
    }
}

impl Default for WarmStartIterateInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl IterateInitializer for WarmStartIterateInitializer {
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        // Two entry points share this initializer: the re-optimize path
        // (curr.x carries values from the prior solve) and the first
        // OptimizeTNLP call that opted into warm_start_init_point=yes
        // (curr.x is the application's placeholder seed — allocated but
        // never written). Detect the latter and rebuild `curr` from the
        // NLP's get_starting_x/y/z hooks before clamping.
        let needs_seed_from_nlp = {
            let borrow = data.borrow();
            match borrow.curr.as_ref() {
                None => return false,
                Some(c) => !is_initialized(&c.x),
            }
        };

        if needs_seed_from_nlp {
            seed_from_nlp(data, nlp, &self.opts);
        }

        if self.opts.mult_init_max > 0.0 {
            // Rebuild `curr` with clamped multipliers. Components are
            // shared via `Rc` with previous solves, so we make fresh
            // copies before mutating to avoid clobbering downstream
            // borrowers.
            let mut borrow = data.borrow_mut();
            let curr = borrow.curr.as_ref().unwrap();
            let cap = self.opts.mult_init_max;
            let new_curr = IteratesVector::new(
                Rc::clone(&curr.x),
                Rc::clone(&curr.s),
                clone_clamped(&curr.y_c, -cap, cap),
                clone_clamped(&curr.y_d, -cap, cap),
                clone_clamped(&curr.z_l, 0.0, cap),
                clone_clamped(&curr.z_u, 0.0, cap),
                clone_clamped(&curr.v_l, 0.0, cap),
                clone_clamped(&curr.v_u, 0.0, cap),
            );
            borrow.set_curr(new_curr);
        }

        if self.opts.target_mu > 0.0 {
            data.borrow_mut().curr_mu = self.opts.target_mu;
        }

        true
    }
}

/// Pull a fresh starting iterate from the NLP (which routes to
/// `TNLP::get_starting_point` with `init_x` / `init_lambda` /
/// `init_z` all true), push the primals and slacks into the bound
/// interior using warm-start-specific `bound_push`/`bound_frac`, and
/// install the result on `data.curr`. Mirrors steps 1-4 of
/// `DefaultIterateInitializer::set_initial_iterates`, but with
/// upstream's warm-start option block governing the push.
fn seed_from_nlp(
    data: &IpoptDataHandle,
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    opts: &WarmStartOptions,
) {
    let (n_x, n_s, n_yc, n_yd, n_zl, n_zu, n_vl, n_vu) = {
        let borrow = data.borrow();
        let c = borrow.curr.as_ref().unwrap();
        (
            c.x.dim(),
            c.s.dim(),
            c.y_c.dim(),
            c.y_d.dim(),
            c.z_l.dim(),
            c.z_u.dim(),
            c.v_l.dim(),
            c.v_u.dim(),
        )
    };

    let mut x = DenseVectorSpace::new(n_x).make_new_dense();
    nlp.borrow_mut().get_starting_x(&mut x);
    {
        let nlp_ref = nlp.borrow();
        push_x_into_interior(
            &mut x,
            &*nlp_ref.px_l(),
            nlp_ref.x_l(),
            &*nlp_ref.px_u(),
            nlp_ref.x_u(),
            opts.bound_push,
            opts.bound_frac,
        );
    }

    let mut s = DenseVectorSpace::new(n_s).make_new_dense();
    nlp.borrow_mut().eval_d(&x, &mut s);
    {
        let nlp_ref = nlp.borrow();
        push_x_into_interior(
            &mut s,
            &*nlp_ref.pd_l(),
            nlp_ref.d_l(),
            &*nlp_ref.pd_u(),
            nlp_ref.d_u(),
            opts.slack_bound_push,
            opts.slack_bound_frac,
        );
    }

    let mut y_c = DenseVectorSpace::new(n_yc).make_new_dense();
    let mut y_d = DenseVectorSpace::new(n_yd).make_new_dense();
    y_c.set(0.0);
    y_d.set(0.0);
    nlp.borrow_mut().get_starting_y(&mut y_c, &mut y_d);

    let mut z_l = DenseVectorSpace::new(n_zl).make_new_dense();
    let mut z_u = DenseVectorSpace::new(n_zu).make_new_dense();
    let mut v_l = DenseVectorSpace::new(n_vl).make_new_dense();
    let mut v_u = DenseVectorSpace::new(n_vu).make_new_dense();
    z_l.set(0.0);
    z_u.set(0.0);
    v_l.set(0.0);
    v_u.set(0.0);
    nlp.borrow_mut()
        .get_starting_z(&mut z_l, &mut z_u, &mut v_l, &mut v_u);

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
    data.borrow_mut().set_curr(iv);
}

fn is_initialized(v: &Rc<dyn Vector>) -> bool {
    if v.dim() == 0 {
        return true;
    }
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.is_initialized())
        .unwrap_or(true)
}

/// Clone `v` into a fresh owned vector and clamp every entry to
/// `[lo, hi]` componentwise. Empty vectors short-circuit. Vectors that
/// were never written to (the application's placeholder seed iterates
/// before any solve ran) collapse to a zero-initialized vector — `0`
/// is inside every well-formed warm-start clamp range, so this matches
/// upstream's behavior when a multiplier block has no carry-over
/// value.
fn clone_clamped(v: &Rc<dyn Vector>, lo: f64, hi: f64) -> Rc<dyn Vector> {
    let n = v.dim();
    if n == 0 {
        return Rc::clone(v);
    }
    let mut out = v.make_new();
    let initialized = v
        .as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.is_initialized())
        .unwrap_or(true);
    if initialized {
        out.copy(&**v);
    } else {
        out.set(0.0);
    }
    let mut cap_hi = v.make_new();
    cap_hi.set(hi);
    out.element_wise_min(&*cap_hi);
    let mut cap_lo = v.make_new();
    cap_lo.set(lo);
    out.element_wise_max(&*cap_lo);
    Rc::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    fn dense(n: i32, fill: f64) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(n);
        let mut v = space.make_new_dense();
        v.set(fill);
        Rc::new(v)
    }

    #[test]
    fn clamps_multipliers_to_cap() {
        let v = dense(3, 1e10);
        let out = clone_clamped(&v, 0.0, 1e6);
        assert_eq!(out.amax(), 1e6);
        let v2 = dense(3, -1e10);
        let out2 = clone_clamped(&v2, -1e6, 1e6);
        assert_eq!(out2.amax(), 1e6);
    }

    #[test]
    fn clamps_bound_mults_nonneg() {
        let v = dense(3, -5.0);
        let out = clone_clamped(&v, 0.0, 1e6);
        assert_eq!(out.amax(), 0.0);
    }

    #[test]
    fn empty_vector_short_circuits() {
        let v = dense(0, 0.0);
        let out = clone_clamped(&v, 0.0, 1.0);
        assert_eq!(out.dim(), 0);
    }

    #[test]
    fn in_range_values_pass_through_untouched() {
        let v = dense(3, 0.5);
        let out = clone_clamped(&v, 0.0, 1.0);
        assert!((out.max() - 0.5).abs() < 1e-15);
        assert!((out.min() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn uninitialized_source_collapses_to_zero() {
        // Application's placeholder seed iterate: vector allocated but
        // never written. `clone_clamped` must fall back to zero instead
        // of tripping the dense-vector "must be initialized" assert.
        let space = DenseVectorSpace::new(4);
        let v: Rc<dyn Vector> = Rc::new(space.make_new_dense());
        let out = clone_clamped(&v, 0.0, 1e6);
        assert_eq!(out.amax(), 0.0);
    }
}
