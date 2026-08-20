//! Resolving the registered sIPOPT option keys into the values this
//! crate actually acts on (gh#551 / gh#677).
//!
//! [`crate::sens_app::register_options`] (and its twin in
//! `pounce-algorithm::upstream_options`) registers upstream sIPOPT's
//! option names so an `ipopt.opt` written for sIPOPT parses unchanged.
//! Registering a name says nothing about reading it: until this module
//! existed, `compute_red_hessian=yes` in an options file was accepted,
//! stored, and then ignored — the reduced Hessian was reachable only
//! through the `--compute-red-hessian` CLI flag or the
//! [`crate::SensSolve`] builder. Same for `run_sens`, `rh_eigendecomp`,
//! `sens_boundcheck`, `sens_bound_eps` and `sens_max_pdpert`.
//!
//! # Every field is an `Option`, deliberately
//!
//! Each knob here has a registered default, and for two of them that
//! default does **not** describe what pounce does today:
//!
//! | option | registered default | pounce's behaviour before this module |
//! |---|---|---|
//! | `compute_red_hessian` | `no` | no reduced Hessian unless asked — **matches** |
//! | `rh_eigendecomp` | `no` | no eigendecomposition unless asked — **matches** |
//! | `sens_boundcheck` | `no` | no bound refinement unless asked — **matches** |
//! | `sens_bound_eps` | `1e-3` | `1e-3` (CLI `--sens-bound-eps`) — **matches** |
//! | `run_sens` | `no` | the step runs whenever the `.nl` declares the sIPOPT suffixes |
//! | `sens_max_pdpert` | `1e-3` | no cap at all: a step is returned however hard the KKT factor was regularized |
//!
//! For the last two, honouring the *registered* default would change
//! results for people who never set the option: `run_sens=no` would
//! switch off every suffix-driven sensitivity solve, and
//! `sens_max_pdpert=1e-3` would start refusing steps that are returned
//! today. So the reader reports only what the user **explicitly set**
//! (`found == true` on the options list) and leaves the pre-existing
//! effective default in place otherwise. The discrepancy is real and is
//! recorded here rather than papered over.
//!
//! (`n_sens_steps` is the seventh sIPOPT key. It is not here: pounce
//! implements the single `sens_state_1` perturbation tier only, so any
//! value above the registered default is refused by
//! `pounce_algorithm::unimplemented_options` rather than quietly
//! rounded down to one tier.)
//!
//! Upstream reference:
//! [`SensApplication::RegisterOptions`](https://github.com/coin-or/Ipopt/blob/master/contrib/sIPOPT/src/SensApplication.cpp).

use pounce_common::options_list::OptionsList;
use pounce_common::types::Number;

/// `sens_bound_eps`'s registered default — the margin by which a
/// coordinate has to leave its bound before `sens_boundcheck` pins it.
/// The CLI's `--sens-bound-eps` default is the same value, so reading
/// the option is behaviour-neutral for anyone who does not set it.
pub const DEFAULT_SENS_BOUND_EPS: Number = 1.0e-3;

/// What the caller explicitly asked for through the sIPOPT option
/// names. `None` means "not set" — never "set to the default" — so a
/// consumer can keep its own effective default where that differs from
/// the registered one (see the module docs).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SensOptionOverrides {
    /// `run_sens`. `Some(false)` suppresses a sensitivity step that
    /// would otherwise run (the suffix-driven CLI path); `Some(true)`
    /// asks for one and earns a warning when nothing declares the
    /// perturbation.
    pub run_sens: Option<bool>,
    /// `compute_red_hessian`.
    pub compute_red_hessian: Option<bool>,
    /// `rh_eigendecomp`. Implies `compute_red_hessian`, matching
    /// `--rh-eigendecomp` and
    /// [`crate::SensSolve::with_reduced_hessian_eigen`].
    pub rh_eigendecomp: Option<bool>,
    /// `sens_boundcheck`.
    pub sens_boundcheck: Option<bool>,
    /// `sens_bound_eps`. Only consulted when the bound refinement is
    /// on: on its own it enables nothing, so that an `ipopt.opt` that
    /// spells out the `1e-3` default does not silently switch the
    /// refinement on. (`--sens-bound-eps` *does* imply the flag — it
    /// is a direct request, not a written-out default.)
    pub sens_bound_eps: Option<Number>,
    /// `sens_max_pdpert`. The largest inertia-correction perturbation
    /// `(δ_x, δ_s, δ_c, δ_d)` the converged KKT factor may carry for
    /// its sensitivity outputs to be reported at all.
    pub sens_max_pdpert: Option<Number>,
}

/// `Some` only when the caller **set** the key; `None` for unset and for
/// a value that does not parse.
///
/// `key` first, and the `read_*` name, so the registered-but-unread scan
/// (`pounce-algorithm/tests/no_silent_options.rs`) can discover this as
/// an accessor and see the literal key at each call below. That scan is
/// what reports an option as a silent no-op, and a helper it cannot
/// recognise would hide these six behind it — the same shape that let
/// `limited_memory_initialization` look wired while it was not (#677).
fn read_yes(key: &str, options: &OptionsList) -> Option<bool> {
    match options.get_bool_value(key, "") {
        Ok((v, true)) => Some(v),
        _ => None,
    }
}

/// The numeric twin of [`read_yes`], with the same naming constraint.
fn read_num(key: &str, options: &OptionsList) -> Option<Number> {
    match options.get_numeric_value(key, "") {
        Ok((v, true)) => Some(v),
        _ => None,
    }
}

impl SensOptionOverrides {
    /// Read the sIPOPT keys off an options list. Unset keys — and keys
    /// whose value does not parse — come back `None`; a malformed value
    /// is already reported by the options layer when the solve reads
    /// it, and this reader must not be the thing that fails a solve.
    pub fn from_options_list(options: &OptionsList) -> Self {
        Self {
            run_sens: read_yes("run_sens", options),
            compute_red_hessian: read_yes("compute_red_hessian", options),
            rh_eigendecomp: read_yes("rh_eigendecomp", options),
            sens_boundcheck: read_yes("sens_boundcheck", options),
            sens_bound_eps: read_num("sens_bound_eps", options),
            sens_max_pdpert: read_num("sens_max_pdpert", options),
        }
    }

    /// True when the options ask for the reduced Hessian (directly, or
    /// through `rh_eigendecomp`, which needs it).
    pub fn wants_reduced_hessian(&self) -> bool {
        self.compute_red_hessian == Some(true) || self.rh_eigendecomp == Some(true)
    }

    /// True when the options ask for the reduced-Hessian
    /// eigendecomposition.
    pub fn wants_eigendecomp(&self) -> bool {
        self.rh_eigendecomp == Some(true)
    }

    /// True when the options explicitly switch the sensitivity step
    /// off. The suffix-driven path runs it by default, so only an
    /// explicit `run_sens=no` suppresses one.
    pub fn suppresses_sens_step(&self) -> bool {
        self.run_sens == Some(false)
    }

    /// The bound-refinement margin the options ask for, or `None` when
    /// they do not ask for the refinement at all. `sens_bound_eps`
    /// alone does not turn it on — see the field docs.
    pub fn boundcheck_eps(&self) -> Option<Number> {
        (self.sens_boundcheck == Some(true))
            .then(|| self.sens_bound_eps.unwrap_or(DEFAULT_SENS_BOUND_EPS))
    }

    /// The message `sens_max_pdpert` earns when the converged factor is
    /// perturbed past the cap the caller set, or `None` when it is not
    /// set or not exceeded.
    ///
    /// `perturbations` is `(δ_x, δ_s, δ_c, δ_d)` as
    /// [`crate::PdSensBacksolver::kkt_perturbations`] reports it: the
    /// inertia correction and Jacobian regularization baked into the
    /// factor the sensitivity step inverts. Nonzero entries mean the
    /// system solved is not the KKT system of the problem, so the step
    /// and the reduced Hessian describe a nearby, perturbed problem —
    /// which is what upstream's cap exists to catch.
    pub fn pdpert_refusal(&self, perturbations: &[Number; 4]) -> Option<String> {
        let limit = self.sens_max_pdpert?;
        let worst = perturbations
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        (worst > limit).then(|| {
            format!(
                "sensitivity skipped: the converged KKT factor carries a \
                 perturbation of {worst:.3e} (δ_x={:.3e}, δ_s={:.3e}, \
                 δ_c={:.3e}, δ_d={:.3e}), above the requested \
                 `sens_max_pdpert={limit:.3e}` — the factor the step would \
                 invert is not this problem's KKT matrix. Raise \
                 `sens_max_pdpert` to accept it anyway.",
                perturbations[0], perturbations[1], perturbations[2], perturbations[3],
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::reg_options::RegisteredOptions;

    fn list() -> OptionsList {
        let reg = RegisteredOptions::new();
        crate::sens_app::register_options(&reg).expect("register");
        OptionsList::with_registered(reg)
    }

    /// A pristine list asks for nothing: every field stays `None` so
    /// every consumer keeps the behaviour it had before this module.
    #[test]
    fn nothing_set_asks_for_nothing() {
        let o = SensOptionOverrides::from_options_list(&list());
        assert_eq!(o, SensOptionOverrides::default());
        assert!(!o.wants_reduced_hessian());
        assert!(!o.suppresses_sens_step());
        assert!(o.boundcheck_eps().is_none());
        assert!(o.pdpert_refusal(&[1e9, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn explicit_values_come_back() {
        let mut l = list();
        l.set_string_value("compute_red_hessian", "yes", true, false)
            .unwrap();
        l.set_string_value("rh_eigendecomp", "yes", true, false)
            .unwrap();
        l.set_string_value("run_sens", "no", true, false).unwrap();
        l.set_string_value("sens_boundcheck", "yes", true, false)
            .unwrap();
        l.set_numeric_value("sens_bound_eps", 1e-6, true, false)
            .unwrap();
        l.set_numeric_value("sens_max_pdpert", 1e-9, true, false)
            .unwrap();
        let o = SensOptionOverrides::from_options_list(&l);
        assert_eq!(o.compute_red_hessian, Some(true));
        assert!(o.wants_reduced_hessian() && o.wants_eigendecomp());
        assert!(o.suppresses_sens_step());
        assert_eq!(o.boundcheck_eps(), Some(1e-6));
        assert_eq!(o.sens_max_pdpert, Some(1e-9));
    }

    /// `rh_eigendecomp` needs the matrix it decomposes, so it implies
    /// the reduced Hessian on its own.
    #[test]
    fn eigendecomp_implies_the_reduced_hessian() {
        let mut l = list();
        l.set_string_value("rh_eigendecomp", "yes", true, false)
            .unwrap();
        let o = SensOptionOverrides::from_options_list(&l);
        assert!(o.wants_reduced_hessian());
    }

    /// The margin on its own must not switch the refinement on: a
    /// generated options file spells out `sens_bound_eps 1e-3`, and
    /// that asks for nothing.
    #[test]
    fn the_margin_alone_does_not_enable_the_refinement() {
        let mut l = list();
        l.set_numeric_value("sens_bound_eps", 1e-6, true, false)
            .unwrap();
        let o = SensOptionOverrides::from_options_list(&l);
        assert_eq!(o.sens_bound_eps, Some(1e-6));
        assert_eq!(o.boundcheck_eps(), None);
    }

    /// The cap compares against the largest perturbation of the four,
    /// and only fires when it is actually exceeded.
    #[test]
    fn the_perturbation_cap_fires_only_above_the_limit() {
        let mut l = list();
        l.set_numeric_value("sens_max_pdpert", 1e-6, true, false)
            .unwrap();
        let o = SensOptionOverrides::from_options_list(&l);
        assert!(o.pdpert_refusal(&[0.0, 0.0, 0.0, 0.0]).is_none());
        assert!(o.pdpert_refusal(&[1e-7, 1e-9, 0.0, 0.0]).is_none());
        let msg = o
            .pdpert_refusal(&[0.0, 0.0, 1e-4, 0.0])
            .expect("1e-4 is above the 1e-6 cap");
        assert!(msg.contains("sens_max_pdpert"), "{msg}");
    }
}
