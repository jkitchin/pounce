//! The homogeneous scalars `tau` and `kappa` of the last HSDE solve on this
//! thread (gh#990 item 9).
//!
//! The HSDE driver solves for `(x, y, z, tau, kappa)` and returns the
//! un-homogenized `(x, y, z) / tau`. On a solvable problem `tau` is positive
//! and `kappa` is near zero. On an infeasible one `tau -> 0`, `kappa > 0`, and
//! the returned certificate carries that `1/tau` scale -- `z ~ 4.5e10` on a
//! two-variable LP -- which is a *ray*, meaningful only up to positive scaling.
//! `tau` and `kappa` are what a caller needs to say how decisive the verdict
//! is (`kappa / tau`, or `tau / (tau + kappa)`).
//!
//! Carried out-of-band for the reason [`crate::sigma_verdict`] is: `QpSolution`
//! has over 200 construction sites and no `Default`. Clear before a solve, take
//! after it, on the same thread; a solve that never reached the HSDE driver (the
//! direct driver, the active-set engine) leaves the slot empty.

use std::cell::Cell;

/// `tau` and `kappa` at termination, and the iteration count of that run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HsdeScalars {
    pub tau: f64,
    pub kappa: f64,
    pub iters: usize,
}

thread_local! {
    static SLOT: Cell<Option<HsdeScalars>> = const { Cell::new(None) };
}

/// Forget any recorded scalars.
pub fn clear() {
    SLOT.with(|s| s.set(None));
}

/// Take the scalars of the last HSDE run on this thread, if one ran.
pub fn take() -> Option<HsdeScalars> {
    SLOT.with(|s| s.take())
}

pub(crate) fn record(tau: f64, kappa: f64, iters: usize) {
    SLOT.with(|s| s.set(Some(HsdeScalars { tau, kappa, iters })));
}

/// Read the slot without emptying it.
pub(crate) fn peek() -> Option<HsdeScalars> {
    SLOT.with(|s| s.get())
}

/// Overwrite the slot — `None` to say "the returned answer did not come from
/// an HSDE run". The recovery drivers in `ipm.rs` use this so that, whichever
/// candidate they return, the slot describes *that* candidate's run and never
/// a discarded one (gh#990).
pub(crate) fn set(v: Option<HsdeScalars>) {
    SLOT.with(|s| s.set(v));
}

/// Run `f` with the slot cleared first, and return its result together with
/// the scalars of the HSDE run (if any) that `f` itself performed last.
pub(crate) fn tracked<T>(f: impl FnOnce() -> T) -> (T, Option<HsdeScalars>) {
    clear();
    let r = f();
    (r, peek())
}
