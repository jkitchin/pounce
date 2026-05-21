//! Restoration sub-algorithm builder — port of the resto half of
//! `Algorithm/IpAlgBuilder.cpp:BuildRestoIpoptAlgorithm` (and the
//! `"resto."` option prefix it pushes onto the inner solver).
//!
//! When the outer line search hands control to the restoration phase
//! (via [`crate::r#trait::RestorationPhase`]), a *nested* IPM has to
//! solve the relaxed feasibility problem
//!
//! ```text
//!   min  ρ · 1ᵀ(n_c + p_c + n_d + p_d) + ½ η(μ) · ||D_R (x − x_R)||²
//!   s.t. c(x) − n_c + p_c   = 0
//!        d(x) − n_d + p_d − s = 0
//! ```
//!
//! The nested solver reuses the regular-phase `AlgorithmBuilder` for
//! the inner-loop strategy slots (line search, KKT solver, mu update,
//! iteration output, ...), but with the option lookup namespace
//! shifted to the `"resto."` prefix and with the resto-specific
//! strategy substitutions applied:
//!
//! | Outer slot                  | Resto override                                              |
//! |-----------------------------|-------------------------------------------------------------|
//! | `IpoptNlp`                  | [`crate::resto_nlp::RestoIpoptNlp`]                         |
//! | `IterateInitializer`        | [`crate::init::RestoIterateInitializer`]                    |
//! | `ConvCheck` (filter LS)     | [`crate::conv_check::RestoFilterConvCheck`]                 |
//! | `ConvCheck` (penalty LS)    | [`crate::conv_check::RestoPenaltyConvCheck`]                |
//! | `IterationOutput`           | [`crate::output::RestoIterationOutput`]                     |
//! | `AugSystemSolver`           | [`crate::aug_resto_system_solver::AugRestoSystemSolver`]    |
//! | `RestorationPhase` (nested) | [`crate::resto_resto::RestoRestorationPhase`] (rare)        |
//!
//! All other slots — perturbation handler, PdFullSpaceSolver,
//! BacktrackingLineSearch, MuUpdate, EqMultCalculator, HessianUpdater
//! — are reused from the outer builder via the
//! [`RestoAlgorithmBuilder::outer_line_search`] selector below.
//!
//! Phase 9 follow-up ships the option-driven assembly surface and the
//! resto-side bundle struct. Wiring the bundle into a runnable nested
//! `IpoptAlgorithm` lands once the placeholder
//! `MinC1NormRestoration::perform_restoration` body is filled in (still
//! a [`crate::r#trait::RestorationPhase`] default-`false` stub). At
//! that point the line search picks up the bundle from
//! `RestoAlgorithmBuilder::build` and drives the nested loop.

use crate::conv_check::{RestoFilterConvCheck, RestoPenaltyConvCheck};
use crate::init::RestoIterateInitializer;
use crate::min_c_1nrm::MinC1NormRestoration;
use crate::output::{InfPrTag, PrintInfoString, RestoIterationOutput};
use crate::r#trait::{RestorationOutcome, RestorationPhase};
use crate::resto_nlp::RestoIpoptNlp;
use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::kkt::aug_system_solver::AugSystemSolver;
use std::cell::RefCell;
use std::rc::Rc;

/// Selects which outer-phase line-search method the resto sub-solver
/// is being called from. Determines which `RestoConvCheck` flavor the
/// bundle includes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterLineSearch {
    /// Outer phase uses `FilterLsAcceptor`. Nested solver wires
    /// [`RestoFilterConvCheck`].
    Filter,
    /// Outer phase uses the (CG-)`PenaltyLsAcceptor`. Nested solver
    /// wires [`RestoPenaltyConvCheck`].
    Penalty,
}

/// Either of the two convergence checkers the resto sub-builder may
/// emit, depending on `outer_line_search`. Boxed so callers can hold
/// the bundle without committing to a specific concrete type.
pub enum RestoConvCheckSlot {
    Filter(RestoFilterConvCheck),
    Penalty(RestoPenaltyConvCheck),
}

/// Bundle of resto-specific strategy objects produced by
/// [`RestoAlgorithmBuilder::build`]. The outer
/// `pounce_algorithm::AlgorithmBuilder` is invoked separately to wire
/// the inner-loop slots that the resto phase reuses unchanged
/// (line-search driver, KKT chain, mu update, ...).
pub struct RestoAlgorithmBundle {
    pub nlp: RestoIpoptNlp,
    pub init: RestoIterateInitializer,
    pub conv_check: RestoConvCheckSlot,
    pub iter_output: RestoIterationOutput,
    pub driver: MinC1NormRestoration,
}

/// Option-driven sub-builder. Mirrors the `resto.*` lookups upstream
/// performs in `BuildRestoIpoptAlgorithm` plus the resto-NLP-only
/// knobs (`rho`, `resto_proximity_weight`,
/// `evaluate_orig_obj_at_resto_trial`).
#[derive(Debug, Clone)]
pub struct RestoAlgorithmBuilder {
    /// `rho` — penalty on the slack 1-norm. Default from
    /// `IpRestoIpoptNLP.cpp:RegisterOptions`.
    pub rho: f64,
    /// `resto_proximity_weight` — proximity-weight option (η in the
    /// objective is `eta_factor * sqrt(mu)`).
    pub eta_factor: f64,
    /// Re-evaluate the original objective at the restoration trial
    /// point so failures surface early. Mirrors
    /// `evaluate_orig_obj_at_resto_trial` (default `true` upstream).
    pub evaluate_orig_obj_at_resto_trial: bool,
    /// `bound_mult_reset_threshold` and `constr_mult_reset_threshold`
    /// — propagated to the restoration driver.
    pub bound_mult_reset_threshold: f64,
    pub constr_mult_reset_threshold: f64,
    /// `expect_infeasible_problem` — propagated to the driver.
    pub expect_infeasible_problem: bool,
    /// `start_with_resto` — initial-step short-circuit knob.
    pub start_with_resto: bool,
    /// Which outer line-search method is in force; selects the
    /// resto conv check variant.
    pub outer_line_search: OuterLineSearch,
    /// `inf_pr_output` — `Internal` reports the resto problem's primal
    /// infeasibility, `Original` reports the orig NLP's unscaled
    /// constraint violation at the resto trial.
    pub inf_pr_output: InfPrTag,
    /// `print_info_string` — extra diagnostic column.
    pub print_info_string: PrintInfoString,
    /// `obj_max_inc` — forwarded to the resto-filter conv check's
    /// outer-iterate guard.
    pub obj_max_inc: f64,
}

impl Default for RestoAlgorithmBuilder {
    fn default() -> Self {
        Self {
            rho: 1e3,
            eta_factor: 1.0,
            evaluate_orig_obj_at_resto_trial: true,
            bound_mult_reset_threshold: 1e3,
            constr_mult_reset_threshold: 0.0,
            expect_infeasible_problem: false,
            start_with_resto: false,
            outer_line_search: OuterLineSearch::Filter,
            inf_pr_output: InfPrTag::Original,
            print_info_string: PrintInfoString::No,
            obj_max_inc: 5.0,
        }
    }
}

impl RestoAlgorithmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assemble the resto-side bundle. `n_orig`, `m_eq`, `m_ineq`, and
    /// `x_ref_vals` come from the *outer* IpoptNlp + the iterate at
    /// the moment the line search triggers restoration. Caller-side
    /// plumbing (`pounce_algorithm::IpoptAlgorithm` → resto-driver
    /// switch) supplies these when the trigger fires.
    pub fn build(
        &self,
        n_orig: pounce_common::types::Index,
        m_eq: pounce_common::types::Index,
        m_ineq: pounce_common::types::Index,
        x_ref_vals: &[f64],
    ) -> RestoAlgorithmBundle {
        let mut nlp =
            RestoIpoptNlp::new(n_orig, m_eq, m_ineq, x_ref_vals, self.rho, self.eta_factor);
        nlp.evaluate_orig_obj_at_resto_trial = self.evaluate_orig_obj_at_resto_trial;

        let init = RestoIterateInitializer::with_dims(n_orig, m_eq, m_ineq, x_ref_vals.to_vec())
            .with_rho(self.rho);

        let conv_check = match self.outer_line_search {
            OuterLineSearch::Filter => {
                let mut cc = RestoFilterConvCheck::new();
                cc.obj_max_inc = self.obj_max_inc;
                RestoConvCheckSlot::Filter(cc)
            }
            OuterLineSearch::Penalty => RestoConvCheckSlot::Penalty(RestoPenaltyConvCheck::new()),
        };

        let iter_output = RestoIterationOutput {
            print_info_string: self.print_info_string,
            inf_pr_output: self.inf_pr_output,
            ..RestoIterationOutput::default()
        };

        let driver = MinC1NormRestoration {
            bound_mult_reset_threshold: self.bound_mult_reset_threshold,
            constr_mult_reset_threshold: self.constr_mult_reset_threshold,
            expect_infeasible_problem: self.expect_infeasible_problem,
            start_with_resto: self.start_with_resto,
            ..MinC1NormRestoration::default()
        };

        RestoAlgorithmBundle {
            nlp,
            init,
            conv_check,
            iter_output,
            driver,
        }
    }
}

impl RestorationPhase for RestoAlgorithmBundle {
    /// Delegates to the bundle's [`MinC1NormRestoration`] driver. The
    /// driver's crate-default `inner_solver` short-circuits to
    /// [`RestorationOutcome::Failed`] until Phase 10 wires
    /// nested-IPM construction through `AlgBuilder`; tests can inject
    /// a synthetic `inner_solver` via
    /// [`MinC1NormRestoration::with_inner_solver`].
    fn perform_restoration(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
    ) -> RestorationOutcome {
        self.driver.perform_restoration(data, cq, nlp, aug_solver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_match_upstream() {
        let b = RestoAlgorithmBuilder::new();
        assert_eq!(b.rho, 1e3);
        assert_eq!(b.eta_factor, 1.0);
        assert!(b.evaluate_orig_obj_at_resto_trial);
        assert_eq!(b.bound_mult_reset_threshold, 1e3);
        assert_eq!(b.constr_mult_reset_threshold, 0.0);
        assert!(!b.expect_infeasible_problem);
        assert!(!b.start_with_resto);
        assert_eq!(b.outer_line_search, OuterLineSearch::Filter);
        assert_eq!(b.inf_pr_output, InfPrTag::Original);
        assert_eq!(b.print_info_string, PrintInfoString::No);
        assert_eq!(b.obj_max_inc, 5.0);
    }

    #[test]
    fn build_propagates_nlp_dims_and_rho_eta() {
        let b = RestoAlgorithmBuilder {
            rho: 2.5,
            eta_factor: 7.0,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(3, 2, 1, &[0.1, 0.2, 0.3]);
        assert_eq!(bundle.nlp.n_orig, 3);
        assert_eq!(bundle.nlp.m_eq, 2);
        assert_eq!(bundle.nlp.m_ineq, 1);
        assert_eq!(bundle.nlp.rho, 2.5);
        assert_eq!(bundle.nlp.eta_factor, 7.0);
    }

    #[test]
    fn build_propagates_evaluate_orig_obj_flag() {
        let b = RestoAlgorithmBuilder {
            evaluate_orig_obj_at_resto_trial: false,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(1, 0, 0, &[0.0]);
        assert!(!bundle.nlp.evaluate_orig_obj_at_resto_trial);
    }

    #[test]
    fn outer_filter_selects_filter_conv_check() {
        let b = RestoAlgorithmBuilder {
            outer_line_search: OuterLineSearch::Filter,
            obj_max_inc: 12.0,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(1, 0, 0, &[0.0]);
        match bundle.conv_check {
            RestoConvCheckSlot::Filter(cc) => {
                assert_eq!(cc.obj_max_inc, 12.0);
            }
            RestoConvCheckSlot::Penalty(_) => panic!("expected filter conv check"),
        }
    }

    #[test]
    fn outer_penalty_selects_penalty_conv_check() {
        let b = RestoAlgorithmBuilder {
            outer_line_search: OuterLineSearch::Penalty,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(1, 0, 0, &[0.0]);
        assert!(matches!(bundle.conv_check, RestoConvCheckSlot::Penalty(_)));
    }

    #[test]
    fn iter_output_carries_inf_pr_and_print_info_options() {
        let b = RestoAlgorithmBuilder {
            inf_pr_output: InfPrTag::Internal,
            print_info_string: PrintInfoString::Yes,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(1, 0, 0, &[0.0]);
        assert_eq!(bundle.iter_output.inf_pr_output, InfPrTag::Internal);
        assert_eq!(bundle.iter_output.print_info_string, PrintInfoString::Yes);
    }

    #[test]
    fn driver_picks_up_reset_thresholds_and_flags() {
        let b = RestoAlgorithmBuilder {
            bound_mult_reset_threshold: 2.5e2,
            constr_mult_reset_threshold: 7.0,
            expect_infeasible_problem: true,
            start_with_resto: true,
            ..RestoAlgorithmBuilder::default()
        };
        let bundle = b.build(1, 0, 0, &[0.0]);
        assert_eq!(bundle.driver.bound_mult_reset_threshold, 2.5e2);
        assert_eq!(bundle.driver.constr_mult_reset_threshold, 7.0);
        assert!(bundle.driver.expect_infeasible_problem);
        assert!(bundle.driver.start_with_resto);
    }

    #[test]
    fn bundle_driver_default_state_is_propagated_from_builder() {
        // The end-to-end call into `bundle.perform_restoration(...)`
        // requires real `IpoptData`/`IpoptCq`/`IpoptNlp`/`AugSystemSolver`
        // fixtures, which arrive when Phase 10 wires the nested-IPM
        // construction through `AlgBuilder`. Until then we pin the
        // builder→driver state propagation here and exercise the
        // failure-path arithmetic directly in `min_c_1nrm.rs`'s tests.
        let bundle = RestoAlgorithmBuilder::new().build(1, 0, 0, &[0.0]);
        assert_eq!(bundle.driver.bound_mult_reset_threshold, 1e3);
    }
}
