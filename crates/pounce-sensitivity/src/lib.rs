//! Sensitivity analysis for POUNCE — port of upstream Ipopt's `contrib/sIPOPT/`.
//!
//! # Status
//!
//! Phases A–C complete. Wired today:
//!
//! * [`schur_data::IndexSchurData`] + [`p_calculator::IndexPCalculator`]:
//!   row-selector representation of the perturbation matrix `B`.
//! * [`backsolver::DenseLuBacksolver`] + [`PdSensBacksolver`]: backsolves
//!   against the converged KKT factor (test / live IPM, respectively).
//! * [`schur_driver::DenseGenSchurDriver`]: dense Schur-complement
//!   factor `S = -B K⁻¹ Bᵀ` with parallel right-hand-side solves.
//! * [`step_calc::StdStepCalc`] + [`sens_app::SensApplication`]:
//!   high-level `parametric_step(Δp, dx)` and
//!   [`reduced_hessian::compute_reduced_hessian`] entry points.
//! * [`SensSolve`] / [`SensResult`]: one-call builder (covers the
//!   `on_converged` plumbing typically required to wire the above into
//!   an `IpoptApplication`).
//!
//! Verified against upstream sIPOPT 3.14.19's `parametric_cpp` golden
//! output to 1e-8 (see `tests/parametric_cpp.rs`); the standalone
//! `pounce_sens` AMPL driver in `pounce-cli` matches `sensitivity_amplsolver`'s
//! `_sens_sol` output on representative .nl problems.
//!
//! **Phase D progress** (per [pounce#7](https://github.com/jkitchin/pounce/issues/7)):
//!
//! * **Fixed-variable lifting** ✔ — `pounce_sens` handles `n_x != n_full`
//!   via the `IpoptNlp::full_x_to_var_x` / `var_x_to_full_x` /
//!   `full_g_to_c_block` trait methods (which delegate to
//!   `BoundClassification.x_not_fixed_map` / `c_map`).
//! * **Reduced-Hessian eigendecomposition** ✔ — pure-Rust cyclic Jacobi
//!   in [`pounce_linalg::symmetric_eigen`] (shared with the convex QP
//!   sensitivity path); surfaced via
//!   [`SensApplication::compute_reduced_hessian_eigen`],
//!   [`SensSolve::with_reduced_hessian_eigen`], the `pounce_sens
//!   --rh-eigendecomp` flag, and the Python `solve_with_sens(rh_eigendecomp=True)`
//!   kwarg.
//! * **`sens_boundcheck` bound refinement** ✔ —
//!   [`boundcheck::refine_step_onto_bounds`] repairs the active set the
//!   step implies, both halves of upstream's fix-relax: a coordinate
//!   the step carries past a bound is pinned AT that bound, and a bound
//!   multiplier the step drives negative is set to zero so the variable
//!   can leave. Either way the system is re-solved, so the other
//!   coordinates move with it. Surfaced via
//!   [`SensSolve::with_boundcheck`], `pounce_sens --sens-boundcheck`,
//!   the Python `solve_with_sens(sens_boundcheck=True)` kwarg, and
//!   `estimate(mode="fix_relax")` in pyomo-pounce, all four running the
//!   same refinement.
//!
//! # Algorithmic reference
//!
//! Pirnay, H., López-Negrete, R., and Biegler, L.T. (2012).
//! *Optimal sensitivity based on IPOPT.*
//! Mathematical Programming Computation, **4**(4), 307–331.
//! [DOI: 10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
//!
//! Verified 2026-05-14 via Crossref: title, authors (Hans Pirnay; Rodrigo
//! López-Negrete; Lorenz T. Biegler), MPC volume 4 issue 4 pp 307–331.
//!
//! # Upstream source mirror
//!
//! Port targets `ref/Ipopt/contrib/sIPOPT/src/` in this repo
//! (EPL-2.0, © Hans Pirnay 2009–2011 per the file headers). Each
//! public item in this crate documents the upstream symbol it mirrors
//! with file path and (where stable) line numbers.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod activity;
pub mod algorithm_backsolver;
pub mod backsolver;
pub mod boundcheck;
pub mod convenience;
pub mod diff_handoff;
pub mod options;
pub mod p_calculator;
pub mod reduced_hessian;
pub mod schur_data;
pub mod schur_driver;
pub mod sens_app;
pub mod solver;
pub mod step_calc;
mod vec_util;

pub use algorithm_backsolver::PdSensBacksolver;
pub use backsolver::{DenseLuBacksolver, SensBacksolver};
pub use convenience::{SensResult, SensSolve};
pub use diff_handoff::{DEFAULT_ACTIVE_TOL, DiffHandoff};
// Hoisted to pounce-linalg so the convex QP sensitivity path can share it;
// re-exported here to preserve `pounce_sensitivity::symmetric_eigen`.
pub use options::{DEFAULT_SENS_BOUND_EPS, SensOptionOverrides};
pub use p_calculator::{IndexPCalculator, PCalculator};
pub use pounce_linalg::symmetric_eigen;
pub use reduced_hessian::compute_reduced_hessian;
pub use schur_data::{IndexSchurData, SchurData};
pub use schur_driver::{DenseGenSchurDriver, SchurDriver};
pub use sens_app::{SensApplication, SensOptions, register_options};
pub use solver::{ConvergedState, Solver, SolverError};
pub use step_calc::{SensStepCalc, StdStepCalc, WithBacksolver};

/// Run a sensitivity-producing solve in the original TNLP coordinate system.
///
/// Presolve can reduce or reorder the KKT system, while sIPOPT pin indices and
/// reduced-Hessian coordinates are defined against the submitted TNLP. Keep
/// the public `presolve` option intact for callers, but bypass its generic
/// wrapper for this solve.
pub(crate) fn optimize_tnlp_for_sensitivity(
    app: &mut pounce_algorithm::IpoptApplication,
    tnlp: std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::TNLP>>,
) -> pounce_nlp::return_codes::ApplicationReturnStatus {
    let presolve_enabled = app
        .options()
        .get_bool_value("presolve", "")
        .ok()
        .map(|(value, _)| value)
        .unwrap_or(false);
    if presolve_enabled {
        tracing::warn!(
            target: "pounce::sensitivity",
            "disabling generic presolve for sensitivity / reduced-Hessian analysis; \
             its KKT coordinates must match the original TNLP"
        );
    }
    app.optimize_tnlp_without_presolve(tnlp)
}
