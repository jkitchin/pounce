//! Typed materialization of the convex solver's [`OptionsList`] controls.
//!
//! Registration lives in `pounce-algorithm` because that registry is present
//! even when the optional `pounce-convex` dependency is not. Interpretation,
//! however, belongs beside the types being configured: callers should not
//! have to reproduce option names, conversions, or precedence rules.

use std::time::Duration;

use pounce_common::{OptionsList, SolverException, option_invalid};

use crate::QpOptions;

impl QpOptions {
    /// Build convex-IPM options from a registered [`OptionsList`].
    ///
    /// The bounds re-checked here duplicate the ones
    /// `pounce_algorithm::upstream_options` registers, so on any path that
    /// went through the registry they are unreachable — the registry rejects
    /// the value at *set* time. They exist for the caller who builds an
    /// `OptionsList` with no registry attached, where nothing else would.
    /// `pounce-cli`'s `convex_option_readers_match_the_registry` test pins the
    /// two sets of bounds together so the pair cannot drift.
    ///
    /// Shared NLP controls (`tol`, `max_iter`, and `max_wall_time`) are applied
    /// only when explicitly set, so their registry defaults do not replace the
    /// convex solver's independently tuned defaults. The same explicit-only
    /// rule preserves the precedence of `qp_tau` and `qp_tau_max`.
    pub fn try_from_options_list(options: &OptionsList) -> Result<Self, SolverException> {
        let mut parsed = Self::default();

        let (max_iter, explicitly_set) = options.get_integer_value("max_iter", "")?;
        if explicitly_set {
            parsed.max_iter = usize::try_from(max_iter)
                .map_err(|_| option_invalid("max_iter", "must be nonnegative and fit in usize"))?;
        }

        let (tol, explicitly_set) = options.get_numeric_value("tol", "")?;
        if explicitly_set {
            if tol <= 0.0 || tol.is_nan() {
                return Err(option_invalid("tol", "must be greater than zero"));
            }
            parsed.tol = tol;
        }

        let (wall_time, explicitly_set) = options.get_numeric_value("max_wall_time", "")?;
        if explicitly_set {
            if wall_time.is_nan() || wall_time < 0.0 {
                return Err(option_invalid("max_wall_time", "must be nonnegative"));
            }
            // Values outside Duration's honest range, including Ipopt's 1e20
            // sentinel and infinity, mean "no deadline" rather than an
            // arbitrary clamped deadline. Zero remains an immediate deadline.
            parsed.time_limit = Duration::try_from_secs_f64(wall_time).ok();
        }

        let (tau, explicitly_set) = options.get_numeric_value("qp_tau", "")?;
        if explicitly_set {
            if !(tau > 0.0 && tau < 1.0) {
                return Err(option_invalid(
                    "qp_tau",
                    "must lie strictly between zero and one",
                ));
            }
            parsed.tau = tau;
            // A raised floor lifts the default ceiling; an explicit
            // qp_tau_max is read afterwards and remains authoritative.
            parsed.tau_max = parsed.tau_max.max(tau);
        }

        let (tau_max, explicitly_set) = options.get_numeric_value("qp_tau_max", "")?;
        if explicitly_set {
            if !(tau_max > 0.0 && tau_max < 1.0) {
                return Err(option_invalid(
                    "qp_tau_max",
                    "must lie strictly between zero and one",
                ));
            }
            parsed.tau_max = tau_max;
        }

        let (reg, explicitly_set) = options.get_numeric_value("qp_reg", "")?;
        if explicitly_set {
            if reg < 0.0 || reg.is_nan() {
                return Err(option_invalid("qp_reg", "must be nonnegative"));
            }
            parsed.reg = reg;
        }

        let (correctors, explicitly_set) = options.get_integer_value("qp_gondzio_corr", "")?;
        if explicitly_set {
            if !(0..=10).contains(&correctors) {
                return Err(option_invalid(
                    "qp_gondzio_corr",
                    "must be between 0 and 10",
                ));
            }
            parsed.gondzio_max_corr = correctors as usize;
        }

        let (infeas_tol, explicitly_set) = options.get_numeric_value("qp_infeas_tol", "")?;
        if explicitly_set {
            if infeas_tol <= 0.0 || infeas_tol.is_nan() {
                return Err(option_invalid("qp_infeas_tol", "must be greater than zero"));
            }
            parsed.infeas_tol = infeas_tol;
        }

        // Two lookups rather than a bare `get_bool_value`, and deliberately
        // so: with no registry attached `get_string_value` answers "" /
        // not-found for an unset name, and `get_bool_value` would reject that
        // "" as a non-boolean instead of reporting it unset. Read the presence
        // flag first, and only decode a value that is actually there.
        let (_, explicitly_set) = options.get_string_value("qp_hsde", "")?;
        if explicitly_set {
            parsed.use_hsde = options.get_bool_value("qp_hsde", "")?.0;
        }
        let (_, explicitly_set) = options.get_string_value("qp_equilibrate", "")?;
        if explicitly_set {
            parsed.equilibrate = options.get_bool_value("qp_equilibrate", "")?.0;
        }
        let (_, explicitly_set) = options.get_string_value("qp_crossover", "")?;
        if explicitly_set {
            parsed.crossover = options.get_bool_value("qp_crossover", "")?.0;
        }

        Ok(parsed)
    }
}

/// Convex-path presolve settings materialized from an [`OptionsList`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvexPresolveOptions {
    /// Whether convex QP/conic presolve is enabled.
    pub enabled: bool,
}

impl Default for ConvexPresolveOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl ConvexPresolveOptions {
    /// Resolve the convex presolve switch and its NLP-path alias.
    ///
    /// An explicit `qp_presolve` is authoritative. Otherwise an explicit
    /// `presolve` setting is honored; when neither is set, convex presolve
    /// keeps its default-on behavior.
    pub fn try_from_options_list(options: &OptionsList) -> Result<Self, SolverException> {
        // Two lookups per name, for the reason given in
        // [`QpOptions::try_from_options_list`].
        let (_, qp_explicit) = options.get_string_value("qp_presolve", "")?;
        let qp_presolve = if qp_explicit {
            Some(options.get_bool_value("qp_presolve", "")?.0)
        } else {
            None
        };
        let (_, nlp_explicit) = options.get_string_value("presolve", "")?;
        let nlp_presolve = if nlp_explicit {
            Some(options.get_bool_value("presolve", "")?.0)
        } else {
            None
        };
        Ok(Self {
            enabled: qp_presolve.or(nlp_presolve).unwrap_or(true),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::Index;

    fn set_num(options: &mut OptionsList, name: &str, value: f64) {
        options.set_numeric_value(name, value, true, true).unwrap();
    }

    fn set_int(options: &mut OptionsList, name: &str, value: Index) {
        options.set_integer_value(name, value, true, true).unwrap();
    }

    fn set_str(options: &mut OptionsList, name: &str, value: &str) {
        options.set_string_value(name, value, true, true).unwrap();
    }

    #[test]
    fn qp_defaults_do_not_inherit_unset_registry_values() {
        let parsed = QpOptions::try_from_options_list(&OptionsList::new()).unwrap();
        let defaults = QpOptions::default();
        assert_eq!(parsed.time_limit, defaults.time_limit);
        assert_eq!(parsed.tol, defaults.tol);
        assert_eq!(parsed.max_iter, defaults.max_iter);
        assert_eq!(parsed.tau, defaults.tau);
        assert_eq!(parsed.tau_max, defaults.tau_max);
        assert_eq!(parsed.reg, defaults.reg);
        assert_eq!(parsed.infeas_tol, defaults.infeas_tol);
        assert_eq!(parsed.use_hsde, defaults.use_hsde);
        assert_eq!(parsed.equilibrate, defaults.equilibrate);
        assert_eq!(parsed.crossover, defaults.crossover);
        assert_eq!(parsed.gondzio_max_corr, defaults.gondzio_max_corr);
    }

    #[test]
    fn qp_explicit_controls_materialize_typed_values() {
        let mut options = OptionsList::new();
        set_int(&mut options, "max_iter", 0);
        set_num(&mut options, "tol", 2e-7);
        set_num(&mut options, "max_wall_time", 1.25);
        set_num(&mut options, "qp_tau", 0.9);
        set_num(&mut options, "qp_tau_max", 0.98);
        set_num(&mut options, "qp_reg", 2e-9);
        set_int(&mut options, "qp_gondzio_corr", 7);
        set_num(&mut options, "qp_infeas_tol", 3e-6);
        set_str(&mut options, "qp_hsde", "no");
        set_str(&mut options, "qp_equilibrate", "no");
        set_str(&mut options, "qp_crossover", "yes");

        let parsed = QpOptions::try_from_options_list(&options).unwrap();
        assert_eq!(parsed.max_iter, 0);
        assert_eq!(parsed.tol, 2e-7);
        assert_eq!(parsed.time_limit, Some(Duration::from_secs_f64(1.25)));
        assert_eq!(parsed.tau, 0.9);
        assert_eq!(parsed.tau_max, 0.98);
        assert_eq!(parsed.reg, 2e-9);
        assert_eq!(parsed.gondzio_max_corr, 7);
        assert_eq!(parsed.infeas_tol, 3e-6);
        assert!(!parsed.use_hsde);
        assert!(!parsed.equilibrate);
        assert!(parsed.crossover);
    }

    #[test]
    fn explicit_tau_max_wins_and_an_unset_ceiling_tracks_a_raised_floor() {
        let mut floor_only = OptionsList::new();
        set_num(&mut floor_only, "qp_tau", 1.0 - 5e-13);
        let parsed = QpOptions::try_from_options_list(&floor_only).unwrap();
        assert_eq!(parsed.tau_max, 1.0 - 5e-13);

        set_num(&mut floor_only, "qp_tau_max", 0.9);
        let parsed = QpOptions::try_from_options_list(&floor_only).unwrap();
        assert_eq!(parsed.tau_max, 0.9);
    }

    #[test]
    fn an_unrepresentable_wall_time_means_no_deadline() {
        let mut options = OptionsList::new();
        set_num(&mut options, "max_wall_time", 1e20);
        assert_eq!(
            QpOptions::try_from_options_list(&options)
                .unwrap()
                .time_limit,
            None
        );
    }

    #[test]
    fn presolve_alias_and_specific_precedence_are_typed() {
        let defaults = ConvexPresolveOptions::try_from_options_list(&OptionsList::new()).unwrap();
        assert!(defaults.enabled);

        let mut options = OptionsList::new();
        set_str(&mut options, "presolve", "no");
        assert!(
            !ConvexPresolveOptions::try_from_options_list(&options)
                .unwrap()
                .enabled
        );

        set_str(&mut options, "qp_presolve", "yes");
        assert!(
            ConvexPresolveOptions::try_from_options_list(&options)
                .unwrap()
                .enabled
        );

        set_str(&mut options, "presolve", "yes");
        set_str(&mut options, "qp_presolve", "no");
        assert!(
            !ConvexPresolveOptions::try_from_options_list(&options)
                .unwrap()
                .enabled
        );

        let mut alias_only = OptionsList::new();
        set_str(&mut alias_only, "presolve", "yes");
        assert!(
            ConvexPresolveOptions::try_from_options_list(&alias_only)
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn malformed_unregistered_values_are_rejected() {
        let mut options = OptionsList::new();
        set_num(&mut options, "qp_tau", 1.0);
        assert!(QpOptions::try_from_options_list(&options).is_err());

        let mut options = OptionsList::new();
        set_str(&mut options, "qp_hsde", "maybe");
        assert!(QpOptions::try_from_options_list(&options).is_err());
    }
}
