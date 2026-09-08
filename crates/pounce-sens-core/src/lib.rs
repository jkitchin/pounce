//! Engine-agnostic core of POUNCE's sensitivity layer.
//!
//! This crate holds the parts of the sIPOPT port that do not know which
//! solver produced the KKT system they are reading. Everything here is
//! written against one trait — [`backsolver::SensBacksolver`], whose whole
//! required surface is `dim()` and `solve(rhs, lhs)` — so any engine that can
//! back-solve against its converged factor gets the parametric machinery for
//! free:
//!
//! * [`boundcheck`] — fix-relax refinement, path following through
//!   active-set breakpoints, and the directional derivative at a kink.
//! * [`rowlimit`] — an observer block that gives a limit written as a
//!   *constraint row* the primal coordinate `boundcheck` decides in,
//!   for engines whose KKT has no slack block (gh#929).
//! * [`sens_app`] — the sIPOPT `SensApplication` driver, the reduced-Hessian
//!   entry point, and the option registrations.
//! * [`p_calculator`], [`schur_data`], [`schur_driver`], [`step_calc`],
//!   [`reduced_hessian`] — the `P = K⁻¹A` / Schur-complement stack.
//!
//! Two consumers exist in tree. `pounce-sensitivity` implements the trait over
//! the NLP filter-IPM's KKT factor (`PdSensBacksolver`); `pounce-convex`
//! implements it over the convex active-set KKT. Neither depends on the other,
//! and this crate depends on neither — it needs only `pounce-common` and
//! `pounce-linalg`.
//!
//! # What deliberately did not move here
//!
//! Two parts of the NLP arm's sensitivity layer are genuinely engine-coupled
//! and stay in `pounce-sensitivity`, so this boundary is a decision rather
//! than an oversight:
//!
//! * **The corrector.** Its entry points take the concrete `PdSensBacksolver`
//!   and reach for `activity_handles()`, `offsets_public()`, `block_dims()`,
//!   `pack_natural()` and `corrector_sigma()` — none of which are on the
//!   trait, and several of which only mean anything for the filter-IPM's
//!   eight-block compound iterate.
//! * **Activity classification's plumbing.** It reads the filter-IPM's own
//!   iterate (`z_l`, `z_u`, `v_l`, `v_u`) out of an `IpoptData` handle. Its
//!   pure decision rule is portable and is expected to land here later; the
//!   plumbing around it is not.
//!
//! Generalizing either would mean abstracting `IpoptData` / `CalculatedQuantities`
//! access behind another trait, which is a larger project than this crate.
//!
//! # Provenance
//!
//! Port of upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete &
//! Biegler 2012, DOI [10.1007/s12532-012-0043-2]). The module names mirror
//! upstream's file names so the two can be read side by side.
//!
//! [10.1007/s12532-012-0043-2]: https://doi.org/10.1007/s12532-012-0043-2

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod activity_kernel;
pub mod backsolver;
pub mod boundcheck;
pub mod p_calculator;
pub mod reduced_hessian;
pub mod rowlimit;
pub mod schur_data;
pub mod schur_driver;
pub mod sens_app;
pub mod step_calc;

// Root re-exports. These are not a convenience: several moved modules refer to
// each other through the crate root (`&dyn crate::SchurData` appears in
// `schur_driver`'s public trait signature), so removing one of these breaks
// compilation rather than merely lengthening a path. `pounce-sensitivity`
// re-exports the same names, which is what keeps its published API unchanged
// across this extraction.
pub use backsolver::{DenseLuBacksolver, SensBacksolver};
pub use p_calculator::{IndexPCalculator, PCalculator};
pub use reduced_hessian::compute_reduced_hessian;
pub use schur_data::{IndexSchurData, SchurData};
pub use schur_driver::{DenseGenSchurDriver, SchurDriver};
pub use sens_app::{SensApplication, SensOptions, register_options};
pub use step_calc::{SensStepCalc, StdStepCalc, WithBacksolver};
