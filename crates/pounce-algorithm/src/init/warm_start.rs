//! Warm-start iterate initializer — port of
//! `IpWarmStartIterateInitializer.{hpp,cpp}`. Used when a previous
//! solve has left a trial point that should be reused.
//!
//! Trusts that the caller (typically `IpoptApplication::ReOptimizeTNLP`)
//! has populated `data.curr` with the warm-start iterate. Beyond that,
//! we honor two of the eight upstream `warm_start_*` options:
//!
//! * `warm_start_mult_init_max` — caps the magnitude of every
//!   multiplier block on `data.curr`. Equality multipliers (`y_c`,
//!   `y_d`) are clipped to `[-cap, +cap]`; bound multipliers (`z_l`,
//!   `z_u`, `v_l`, `v_u`) are clipped to `[0, cap]`. Mirrors the
//!   per-block clamps at `IpWarmStartIterateInitializer.cpp:148-181`
//!   (the `have_iterate` branch).
//! * `warm_start_target_mu` — when positive, overrides `data.curr_mu`
//!   at iter 0 so the barrier sub-problem starts at the user-requested
//!   value rather than `mu_init`.
//!
//! The remaining knobs (`bound_push`, `bound_frac`,
//! `slack_bound_push`, `slack_bound_frac`, `mult_bound_push`,
//! `entire_iterate`, `same_structure`) are stored on the initializer
//! but not yet consumed — full `push_variables` semantics land with
//! the rest of the warm-start path.

use crate::alg_builder::WarmStartOptions;
use crate::init::r#trait::IterateInitializer;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::aug_system_solver::AugSystemSolver;
use pounce_linalg::dense_vector::DenseVector;
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
        _nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        // The caller is expected to have placed the warm-start iterate
        // on `data.curr`; bail out early if that hasn't happened.
        let mut borrow = data.borrow_mut();
        if borrow.curr.is_none() {
            return false;
        }

        if self.opts.mult_init_max > 0.0 {
            // Rebuild `curr` with clamped multipliers. Components are
            // shared via `Rc` with previous solves, so we make fresh
            // copies before mutating to avoid clobbering downstream
            // borrowers.
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
            borrow.curr_mu = self.opts.target_mu;
        }

        true
    }
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
