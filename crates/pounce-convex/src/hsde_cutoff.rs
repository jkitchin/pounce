//! Early exit for the first, unequilibrated HSDE attempt of a QP solve
//! (gh#1042 R1d).
//!
//! [`crate::ipm`]'s gh #293 path re-solves a QP with Ruiz equilibration when
//! the first HSDE solve ends `IterationLimit` / `OptimalInaccurate` /
//! `NumericalFailure`. On `scaled_feasible_a` the first solve can be seen to
//! be hopeless long before its cap, and running it out cost 200 of the call's
//! 400 HSDE iterations for nothing: from iteration ~19 the homogeneous iterate
//! is frozen and only `τ` moves. `τ` creeps up by a constant `~4.6e-10` per
//! iteration (1.6e-9 at 19, 7.6e-8 at 199), so the un-homogenized point —
//! whose true complementarity is the frozen homogeneous product divided by
//! `τ` — improves as `1/τ`: `true_kkt_error` 7.2 → 0.149 at the cap, and still
//! 0.07 at `max_iter = 400`. The internal `μ` meanwhile sits at `~1e-37`, so the
//! centering has nothing left to work with.
//!
//! The detector (see `crate::hsde`'s main loop) recognises exactly that: over
//! the last [`WINDOW`] iterations `τ < `[`TAU_MAX`], `μ < tol²`, the product
//! `τ · true_kkt_error` held within [`FROZEN_SPREAD`] (the homogeneous point
//! is not moving) while `τ` grew by at least [`MIN_TAU_GROWTH`] (only `τ` is),
//! and even extrapolating `τ`'s window growth *geometrically* — optimistic
//! against the measured linear crawl — `true_kkt_error` cannot reach the
//! post-loop `OptimalInaccurate` band (`1e3·tol`) inside the remaining budget.
//! The solve then ends `IterationLimit` at once, and the gh #293 retry runs as
//! it would have at the cap.
//!
//! It is **armed only for that first attempt** ([`armed`], called by
//! `solve_qp_ipm_core`), and only for the first HSDE loop inside it (the
//! `σ` cost-normalization path disarms it, see [`disarm`]): the same pattern
//! *does* recover elsewhere. `scaled_feasible_a`'s own equilibrated retry sits
//! with `τ` flat at 9.784e-11 and `τ · true_kkt` within 4.4% over windows of
//! the regime for ~100 iterations, then re-inflates its slacks (`pres` 5e-6 →
//! 4.8e9 at 135–150) and certifies at 165. The frozen-product test does not
//! exclude it; `τ` not growing does (its largest window growth is 1.001,
//! against [`MIN_TAU_GROWTH`]; a flat `τ` also makes the projection
//! non-positive), and arming only the first attempt keeps the rule off it by
//! construction. A cut whose retry then
//! fails to certify is undone: `solve_qp_ipm_core` re-runs the first attempt
//! unarmed, so the reported status and iterate are exactly the uncut run's
//! and the detector's only effect on such a solve is the extra cost of the
//! cut iterations.
//!
//! The thresholds are set from the measured population: every HSDE loop of
//! the CLI fixture corpus (121 fixtures, with the probe that recorded `τ`,
//! `μ` and `true_kkt_error` per iteration) at `solver_selection=auto` (58
//! loops: 41 first attempts on the QP path, 1 gh #293 retry, 16 others —
//! SOCP-path and verify/re-solve loops), `qp-ipm` (44), `lp-ipm` (14), and
//! both again at `bound_relax_factor=0`. Only four loops ever enter the `τ`/
//! `μ` gate for a full window: `scaled_feasible_a`'s first attempt (window
//! spread ≤ 1.0009, fires at iteration 36 of 200, 48 of 400) and its retry
//! (spread ≥ 1.044, `τ` growth ≤ 1.001 — excluded), `feasible_x0_wide_scale`
//! (first attempt, `Optimal` at 26; spread ≥ 7.2e4) and `scaled_feasible_b`
//! (first attempt, `Optimal` at 36; spread ≥ 1.4e9). It fires on no loop
//! that ends `Optimal` or `OptimalInaccurate`. That corpus is small — the
//! retry it does fire on is one fixture — so `benchmarks/qp` is the
//! population that bounds it, not this one.

use std::cell::Cell;

/// Iterations over which the frozen-point pattern must hold.
pub(crate) const WINDOW: usize = 10;
/// `τ` must be below this on every iteration of the window: the embedding has
/// collapsed toward its certificate boundary (the plan's `τ ≪ 1`).
pub(crate) const TAU_MAX: f64 = 1e-6;
/// Largest/smallest `τ · true_kkt_error` over the window. Measured: ≤ 1.0009
/// on the loop the rule is for, ≥ 1.044 on the recovering retry, ≥ 7.2e4 on
/// every first attempt that converged while inside the `τ`/`μ` gate.
pub(crate) const FROZEN_SPREAD: f64 = 1.1;
/// `τ` must have grown by at least this factor over the window — it is the
/// one thing still moving. Measured: 1.96 at the fire on `scaled_feasible_a`'s
/// first attempt, ≤ 1.001 on its recovering retry.
pub(crate) const MIN_TAU_GROWTH: f64 = 1.01;
/// The weakest verdict the post-loop upgrade can grant is `OptimalInaccurate`
/// at `true_kkt_error < 1e3·tol`; the projection asks whether even that is in
/// reach.
pub(crate) const TARGET_FACTOR: f64 = 1e3;

/// What the detector saw when it fired.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HsdeCutoff {
    /// Iterations the first attempt had taken when it was cut.
    pub iter: usize,
    /// The attempt's `max_iter`.
    pub max_iter: usize,
    /// `τ` at the cut.
    pub tau: f64,
    /// `true_kkt_error` of the un-homogenized point at the cut.
    pub true_kkt: f64,
    /// Iterations the geometric `τ` projection needed to reach `1e3·tol`.
    pub projected_iters: f64,
}

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static LAST: Cell<Option<HsdeCutoff>> = const { Cell::new(None) };
}

/// Forget any recorded cut.
pub fn clear() {
    LAST.with(|s| s.set(None));
}

/// Take the last cut recorded on this thread, if the detector fired. It stays
/// recorded when the retry that followed was rejected and the first attempt
/// re-run uncut — it says the detector fired, not which answer was returned.
pub fn take() -> Option<HsdeCutoff> {
    LAST.with(|s| s.take())
}

/// Record a cut (called by the HSDE loop as it fires).
pub(crate) fn fired(c: HsdeCutoff) {
    LAST.with(|s| s.set(Some(c)));
    FIRED.with(|s| s.set(true));
}

/// Run `f` with the detector armed for the first HSDE loop it enters; returns
/// whether that loop was cut.
pub(crate) fn armed<T>(f: impl FnOnce() -> T) -> (T, bool) {
    FIRED.with(|s| s.set(false));
    ARMED.with(|a| a.set(true));
    let r = f();
    ARMED.with(|a| a.set(false));
    (r, FIRED.with(|s| s.replace(false)))
}

/// Consume the arming: the first HSDE loop to ask gets it, nested ones do not.
pub(crate) fn take_armed() -> bool {
    ARMED.with(|a| a.replace(false))
}

/// Drop the arming without using it (the `σ` cost-normalization path, whose
/// verdict guards — gh #414 — are not measured against this rule).
pub(crate) fn disarm() {
    ARMED.with(|a| a.set(false));
}
