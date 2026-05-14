//! `SensApplication` — high-level entry point for sensitivity analysis.
//!
//! Port of upstream
//! [`SensApplication.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp).
//!
//! # Phase-D scope (this file)
//!
//! Ships the **skeleton** entry point and the option-key set that
//! upstream registers (so user-side `ipopt.opt` files port cleanly).
//! The orchestration runs the four-stage pipeline:
//!
//! 1. Construct an [`crate::IndexPCalculator`] from a converged
//!    [`crate::SensBacksolver`] + an `A` SchurData.
//! 2. Build a [`crate::DenseGenSchurDriver`] on top.
//! 3. Use [`crate::StdStepCalc`] to compute the sensitivity step
//!    or [`crate::reduced_hessian::compute_reduced_hessian`] to
//!    extract the reduced Hessian.
//!
//! Phase B.2 is the missing piece: a real [`crate::SensBacksolver`]
//! that wraps `pounce-algorithm::kkt::AugSystemSolver` so the
//! application can be driven by a live pounce IPM solve rather than
//! a synthetic dense LU.

use crate::backsolver::SensBacksolver;
use crate::p_calculator::IndexPCalculator;
use crate::reduced_hessian::compute_reduced_hessian;
use crate::schur_data::{IndexSchurData, SchurData};
use crate::schur_driver::{DenseGenSchurDriver, SchurDriver};
use crate::step_calc::{SensStepCalc, StdStepCalc};
use pounce_common::types::Number;

/// User-facing entry point for sensitivity analysis on a converged
/// pounce solve.
///
/// Mirrors `Ipopt::SensApplication` from
/// [`SensApplication.hpp:35-188`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.hpp).
///
/// Phase D ships only the **skeleton** — construction + Run-style
/// dispatch over the Phase B.1 numerical components. The option-table
/// integration with `pounce-algorithm::OptionsList` is via
/// [`register_options`] below; calling code reads option values from
/// the application's owning `OptionsList` and configures `SensOptions`
/// before invoking `SensApplication::run_*`.
pub struct SensApplication<B: SensBacksolver> {
    /// The `A` schur data for the perturbation rows. For
    /// reduced-Hessian use this also serves as the free-variable
    /// selector.
    a_data: IndexSchurData,
    /// Converged backsolver against `K`. Phase B.1 uses
    /// `DenseLuBacksolver` for tests; Phase B.2 wraps the real
    /// `pounce-algorithm` aug system solver.
    backsolver: B,
    /// Pre-resolved option values controlling the sensitivity run.
    options: SensOptions,
}

/// Numeric / boolean knobs that drive a `SensApplication`. The
/// fields' names + defaults mirror the option keys registered in
/// [`register_options`].
#[derive(Debug, Clone, Copy)]
pub struct SensOptions {
    /// Whether to compute the reduced Hessian. Mapped from
    /// `compute_red_hessian` (upstream
    /// [`SensApplication.cpp:73-75`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp)).
    pub compute_red_hessian: bool,
    /// Whether to run the sensitivity step. Mapped from `run_sens`
    /// ([`SensApplication.cpp:80-83`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp)).
    pub run_sens: bool,
    /// Number of parameter perturbations to step. Mapped from
    /// `n_sens_steps` (default 1, upstream
    /// [`SensApplication.cpp:60-62`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp)).
    pub n_sens_steps: i32,
    /// Objective scaling factor to apply when reporting the reduced
    /// Hessian. Default 1.0; pounce's IPM-side scaling lands in
    /// Phase B.2's algorithm-wrapper.
    pub obj_scal: Number,
}

impl Default for SensOptions {
    fn default() -> Self {
        Self {
            compute_red_hessian: false,
            run_sens: false,
            n_sens_steps: 1,
            obj_scal: 1.0,
        }
    }
}

impl<B: SensBacksolver> SensApplication<B> {
    /// Build a SensApplication from a converged backsolver, a
    /// parameter-row SchurData, and pre-resolved options.
    ///
    /// Equivalent of upstream's `SensApplication::Run` setup
    /// path ([`SensApplication.cpp:127-198`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp))
    /// but split so the construction is testable without the
    /// `IpoptApplication` plumbing.
    pub fn new(
        a_data: IndexSchurData,
        backsolver: B,
        options: SensOptions,
    ) -> Self {
        Self {
            a_data,
            backsolver,
            options,
        }
    }

    /// Compute the reduced Hessian into the caller-supplied
    /// row-/column-major buffer (column-major in pounce, matching
    /// `DenseGenSchurDriver`). Buffer length must be `n²` where
    /// `n = a_data.nrows()`. Mirrors the
    /// `compute_red_hessian=true` branch of upstream
    /// [`SensApplication::Run`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp).
    ///
    /// Returns `false` if the underlying schur computation fails.
    pub fn compute_reduced_hessian(&mut self, out: &mut [Number]) -> bool
    where
        B: Clone,
    {
        // We need ownership of the backsolver for the PCalculator;
        // clone here so the application can be re-used for follow-up
        // step computations. Phase B.2's real backsolver will be
        // cheap to clone (Arc handle to a factored matrix); the
        // synthetic DenseLuBacksolver is also cheap (Vec<f64>
        // arithmetic, no large reusable resources).
        let backsolver = self.backsolver.clone();
        let mut pcalc = IndexPCalculator::new(backsolver, self.a_data.clone());
        compute_reduced_hessian(&mut pcalc, &self.a_data, self.options.obj_scal, out)
    }

    /// Compute one sensitivity step: given `rhs_u` (length
    /// `a_data.nrows()`) the Schur-space parameter perturbation,
    /// produce `du` (Schur-space step) and `dx_full` (KKT-space
    /// step). Mirrors upstream's `run_sens=true` flow.
    ///
    /// `dx_full` must be the length of the backsolver's state
    /// dimension. `rhs_u.len() == du.len() == a_data.nrows()`.
    ///
    /// Returns `false` if the Schur driver factor fails or the inner
    /// backsolves fail.
    pub fn run_sens_step(
        &mut self,
        b_data: &IndexSchurData,
        rhs_u: &[Number],
        du: &mut [Number],
        dx_full: &mut [Number],
    ) -> bool
    where
        B: Clone,
    {
        let backsolver = self.backsolver.clone();
        let pcalc = IndexPCalculator::new(backsolver, self.a_data.clone());
        let mut driver = DenseGenSchurDriver::<_, B>::new(pcalc);
        if !driver.schur_build_and_factor(b_data) {
            return false;
        }
        let step = StdStepCalc::new(&driver, driver.pcalc());
        step.compute_step(rhs_u, du, dx_full)
    }

    /// Compute the **parametric** sensitivity step
    /// `Δw = K⁻¹ · Aᵀ · Δp` directly, without the Schur factor. This
    /// is the no-bound-check branch of upstream
    /// [`SensStdStepCalc::Step`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStdStepCalc.cpp)
    /// (lines 48–83): scatter the parameter perturbation onto the
    /// y_c / x slots picked by `a_data`, then run one backsolve
    /// against the converged KKT factor.
    ///
    /// `delta_p` has length `a_data.nrows()`; `dx_full` has length
    /// `backsolver.dim()`. Unlike [`Self::run_sens_step`] this method
    /// is the right one for the canonical "parametric Δx" use case —
    /// the Schur factor's only role in sIPOPT's std flow is the
    /// active-set bound-check refinement after a violating step, and
    /// that refinement is a follow-up (sens_boundcheck = yes).
    pub fn parametric_step(&self, delta_p: &[Number], dx_full: &mut [Number]) -> bool {
        let n_full = self.backsolver.dim();
        if dx_full.len() != n_full {
            return false;
        }
        if delta_p.len() != self.a_data.nrows() as usize {
            return false;
        }
        let mut rhs_full = vec![0.0; n_full];
        if self.a_data.trans_multiply(delta_p, &mut rhs_full).is_err() {
            return false;
        }
        self.backsolver.solve(&rhs_full, dx_full)
    }

    /// Borrow the resolved option set.
    pub fn options(&self) -> &SensOptions {
        &self.options
    }
}

/// Register sIPOPT's option keys against pounce's
/// `RegisteredOptions`. Mirrors upstream
/// [`SensApplication::RegisterOptions`](../../../ref/Ipopt/contrib/sIPOPT/src/SensApplication.cpp)
/// (lines 54–117).
///
/// The same key set is also registered by
/// `pounce-algorithm::upstream_options::register_sipopt_options`
/// (called transitively from `register_all_upstream_options`) so any
/// `IpoptApplication::new()` accepts these keys out of the box —
/// `pounce-cli` and `cutest_suite` recognize them without extra
/// wiring. This standalone copy stays for callers building a
/// `RegisteredOptions` without `pounce-algorithm`'s defaults (e.g.
/// integration tests inside `pounce-sensitivity` itself). Keep the
/// two blocks in lockstep when adding or renaming options.
pub fn register_options(
    r: &pounce_common::reg_options::RegisteredOptions,
) -> Result<(), pounce_common::exception::SolverException> {
    r.set_registering_category("sIPOPT");
    r.add_lower_bounded_integer_option(
        "n_sens_steps",
        "Number of sensitivity steps to perform per converged solve.",
        0,
        1,
        "Number of parameter perturbations to step through. Mirrors upstream `n_sens_steps` (SensApplication.cpp:60).",
    )?;
    r.add_bool_option(
        "compute_red_hessian",
        "Compute the reduced Hessian at the converged solution.",
        false,
        "When set, after the IPM converges pounce-sensitivity assembles `H_R = obj_scal · B K⁻¹ Bᵀ` with B selecting the free variables. Output is written to the user via the sIPOPT C ABI (Phase D follow-up). Mirrors upstream `compute_red_hessian` (SensApplication.cpp:73).",
    )?;
    r.add_bool_option(
        "run_sens",
        "Run the sensitivity step calc after convergence.",
        false,
        "When set, pounce-sensitivity computes a forward-sensitivity step for the parameter perturbation declared via TNLP suffixes. Mirrors upstream `run_sens` (SensApplication.cpp:80).",
    )?;
    r.add_bool_option(
        "sens_boundcheck",
        "Verify the sensitivity step does not violate bound multipliers.",
        false,
        "Mirrors upstream `sens_boundcheck` (SensApplication.cpp:63).",
    )?;
    r.add_lower_bounded_number_option(
        "sens_bound_eps",
        "Safety margin enforced when sens_boundcheck is set.",
        0.0,
        true,
        1.0e-3,
        "Mirrors upstream `sens_bound_eps` (SensApplication.cpp:67).",
    )?;
    r.add_lower_bounded_number_option(
        "sens_max_pdpert",
        "Maximum primal-dual perturbation accepted in the sensitivity step.",
        0.0,
        true,
        1.0e-3,
        "Mirrors upstream `sens_max_pdpert` (SensApplication.cpp:98).",
    )?;
    r.add_bool_option(
        "rh_eigendecomp",
        "Compute eigendecomposition of the reduced Hessian.",
        false,
        "Mirrors upstream `rh_eigendecomp` (SensApplication.cpp:106). Pounce ships the option key for ipopt.opt-compatibility; the eigendecomposition itself is a Phase-D follow-up.",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::DenseLuBacksolver;
    use crate::schur_data::IndexSchurData;

    /// End-to-end Phase-D smoke: build a SensApplication, ask for the
    /// reduced Hessian, verify it matches the same closed-form answer
    /// as the direct `compute_reduced_hessian` call in
    /// `reduced_hessian::tests`.
    #[test]
    fn sens_application_computes_reduced_hessian() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let opts = SensOptions {
            compute_red_hessian: true,
            obj_scal: 1.0,
            ..SensOptions::default()
        };
        let mut app = SensApplication::new(a, backsolver, opts);
        let mut hr = vec![0.0; 4];
        assert!(app.compute_reduced_hessian(&mut hr));
        // Same expected values as reduced_hessian::tests::reduced_hessian_matches_kinv_submatrix.
        assert!((hr[0] - 0.75).abs() < 1e-12);
        assert!((hr[1] - 0.25).abs() < 1e-12);
        assert!((hr[2] - 0.25).abs() < 1e-12);
        assert!((hr[3] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn sens_application_runs_step() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let b = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let opts = SensOptions { run_sens: true, ..SensOptions::default() };
        let mut app = SensApplication::new(a, backsolver, opts);
        let rhs_u = [1.0, 0.0];
        let mut du = [0.0; 2];
        let mut dx = [0.0; 3];
        assert!(app.run_sens_step(&b, &rhs_u, &mut du, &mut dx));
        // Same expected values as step_calc::tests::std_step_calc_runs_two_step_pipeline.
        assert!((du[0] - (-1.5)).abs() < 1e-10);
        assert!((du[1] - 0.5).abs() < 1e-10);
        assert!((dx[0] - (-1.0)).abs() < 1e-10);
        assert!((dx[1] - (-0.5)).abs() < 1e-10);
        assert!((dx[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn register_options_round_trips_through_options_list() {
        // The option keys should be registerable against a fresh
        // RegisteredOptions without collision; this exercises the
        // public register_options() path.
        let r = pounce_common::reg_options::RegisteredOptions::default();
        register_options(&r).expect("registration must succeed");
    }
}
