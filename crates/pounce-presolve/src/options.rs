//! Option-table integration for `pounce-presolve`.
//!
//! Mirrors the `presolve_*` option family naming used by CBC, GLPK,
//! and HiGHS so cross-solver muscle memory transfers. All keys are
//! registered into the standard `RegisteredOptions` registry; the
//! `IpoptApplication` reads them via the usual `OptionsList` path.

use pounce_common::exception::SolverException;
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_common::types::{Index, Number};

/// Resolved presolve options, materialised from an `OptionsList`.
///
/// Keeping this as a plain `Copy` struct makes Phase 0's no-op path
/// branch-free; later phases that grow more state can promote it to
/// a `Clone` struct without API churn.
#[derive(Debug, Clone, Copy)]
pub struct PresolveOptions {
    /// Master switch (`presolve`). When `false`, the wrapper is a
    /// no-op and `wrap_with_presolve` returns the inner TNLP.
    pub enabled: bool,
    /// `presolve_bound_tightening` — Phase 1.
    pub bound_tightening: bool,
    /// `presolve_redundant_constraint_removal` — Phase 2.
    pub redundant_constraint_removal: bool,
    /// `presolve_linear_eq_reduction` — Phase ≥2, off by default.
    pub linear_eq_reduction: bool,
    /// `presolve_licq_check` — Phase 3.
    pub licq_check: bool,
    /// `presolve_print_level` — 0 silent, 5 per-pass, 8 per-xform.
    pub print_level: Index,
    /// `presolve_max_passes` — fixed-point iteration cap.
    pub max_passes: Index,
    /// `presolve_licq_action` — what to do on degenerate equality
    /// rows. Phase 3 honours `auto_l1`; Phase 0 just stores it.
    pub licq_action: LicqAction,
    /// `presolve_warm_z_bounds` — Phase 4: hint bound-multiplier
    /// warm starts for variables whose bounds were tightened.
    pub warm_z_bounds: bool,
    /// `presolve_bound_mult_init_val` — value used for those hints.
    pub bound_mult_init_val: Number,
    /// `presolve_auxiliary` — master switch for the Phase-0
    /// auxiliary-equality preprocessing pass (issue #53). When
    /// `false` (the default), Phase 0 is a no-op even if the outer
    /// `enabled` master switch is on.
    pub auxiliary: bool,
    /// `presolve_auxiliary_tol` — residual tolerance for accepting a
    /// candidate block solve (full-space KKT residual after the
    /// reduction must stay within this).
    pub auxiliary_tol: Number,
    /// `presolve_auxiliary_max_block_dim` — the lightweight Newton
    /// solver only handles blocks up to this dimension. PR 11 lifts
    /// this by injecting an IPM-backed fallback.
    pub auxiliary_max_block_dim: Index,
    /// `presolve_auxiliary_wall_time_fraction` — fraction of the
    /// solver's overall wall-time budget Phase 0 is allowed to spend.
    pub auxiliary_wall_time_fraction: Number,
    /// `presolve_auxiliary_coupling` — which coupling classes are
    /// eligible for elimination.
    pub auxiliary_coupling: AuxiliaryCouplingPolicy,
    /// `presolve_auxiliary_diagnostics` — emit the diagnostics struct
    /// via the journalist.
    pub auxiliary_diagnostics: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicqAction {
    /// Just report — solver proceeds as usual.
    Warn,
    /// Set `l1_exact_penalty_barrier=yes` so the ℓ₁ wrapper handles
    /// the rank-deficient block. Interlocks with pounce#10.
    AutoL1,
}

/// Coupling-class gate for the auxiliary preprocessing pass. Mirrors
/// ripopt's defaults: `Safe` eliminates only pure-equality blocks;
/// `Aggressive` adds objective-coupled blocks as post-solve
/// candidates; `None` disables elimination entirely (Phase 0 still
/// runs and collects diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxiliaryCouplingPolicy {
    /// No elimination; useful for collecting diagnostics only.
    None,
    /// PureEquality blocks only.
    Safe,
    /// PureEquality + ObjectiveCoupled (post-solve only).
    Aggressive,
}

impl PresolveOptions {
    /// All-default settings, matching the registered defaults.
    pub fn defaults() -> Self {
        Self {
            enabled: false,
            bound_tightening: true,
            redundant_constraint_removal: true,
            linear_eq_reduction: false,
            licq_check: true,
            print_level: 0,
            max_passes: 3,
            licq_action: LicqAction::Warn,
            warm_z_bounds: true,
            bound_mult_init_val: 1.0,
            auxiliary: false,
            auxiliary_tol: 1e-8,
            auxiliary_max_block_dim: 8,
            auxiliary_wall_time_fraction: 0.1,
            auxiliary_coupling: AuxiliaryCouplingPolicy::Safe,
            auxiliary_diagnostics: false,
        }
    }

    /// Read every `presolve_*` key out of an `OptionsList`, falling
    /// back to registered defaults where unset.
    pub fn from_options_list(opts: &OptionsList) -> Result<Self, SolverException> {
        let enabled = opts.get_bool_value("presolve", "")?.0;
        let bound_tightening = opts.get_bool_value("presolve_bound_tightening", "")?.0;
        let redundant_constraint_removal = opts
            .get_bool_value("presolve_redundant_constraint_removal", "")?
            .0;
        let linear_eq_reduction = opts.get_bool_value("presolve_linear_eq_reduction", "")?.0;
        let licq_check = opts.get_bool_value("presolve_licq_check", "")?.0;
        let print_level = opts.get_integer_value("presolve_print_level", "")?.0;
        let max_passes = opts.get_integer_value("presolve_max_passes", "")?.0;
        let licq_action = match opts
            .get_string_value("presolve_licq_action", "")?
            .0
            .as_str()
        {
            "auto_l1" => LicqAction::AutoL1,
            // "warn" or anything else (including the registered default).
            _ => LicqAction::Warn,
        };
        let warm_z_bounds = opts.get_bool_value("presolve_warm_z_bounds", "")?.0;
        let bound_mult_init_val = opts
            .get_numeric_value("presolve_bound_mult_init_val", "")?
            .0;
        let auxiliary = opts.get_bool_value("presolve_auxiliary", "")?.0;
        let auxiliary_tol = opts.get_numeric_value("presolve_auxiliary_tol", "")?.0;
        let auxiliary_max_block_dim = opts
            .get_integer_value("presolve_auxiliary_max_block_dim", "")?
            .0;
        let auxiliary_wall_time_fraction = opts
            .get_numeric_value("presolve_auxiliary_wall_time_fraction", "")?
            .0;
        let auxiliary_coupling = match opts
            .get_string_value("presolve_auxiliary_coupling", "")?
            .0
            .as_str()
        {
            "none" => AuxiliaryCouplingPolicy::None,
            "aggressive" => AuxiliaryCouplingPolicy::Aggressive,
            // "safe" or anything else (including the registered default).
            _ => AuxiliaryCouplingPolicy::Safe,
        };
        let auxiliary_diagnostics = opts.get_bool_value("presolve_auxiliary_diagnostics", "")?.0;
        Ok(Self {
            enabled,
            bound_tightening,
            redundant_constraint_removal,
            linear_eq_reduction,
            licq_check,
            print_level,
            max_passes,
            licq_action,
            warm_z_bounds,
            bound_mult_init_val,
            auxiliary,
            auxiliary_tol,
            auxiliary_max_block_dim,
            auxiliary_wall_time_fraction,
            auxiliary_coupling,
            auxiliary_diagnostics,
        })
    }
}

/// Register every `presolve_*` key into `reg`. Called once from
/// `IpoptApplication::new` (after the upstream-options block).
pub fn register_options(reg: &RegisteredOptions) -> Result<(), SolverException> {
    reg.set_registering_category("NLP Presolve");

    reg.add_bool_option(
        "presolve",
        "Master switch for algorithmic NLP preprocessing.",
        false,
        "If yes, wraps the user TNLP with a presolve layer that may \
         tighten variable bounds, drop redundant constraints, and \
         detect rank-deficient equality blocks before the IPM starts. \
         Off by default; the per-pass options below are then dormant.",
    )?;

    reg.add_bool_option(
        "presolve_bound_tightening",
        "Tighten variable bounds via constraint propagation.",
        true,
        "When presolve is enabled, iteratively propagates each linear \
         constraint into implied bounds on its variables (Andersen's \
         LP presolve §2 adapted to NLP).",
    )?;

    reg.add_bool_option(
        "presolve_redundant_constraint_removal",
        "Drop constraints implied by current variable bounds.",
        true,
        "For each linear constraint, checks whether [lo, hi] is \
         implied by the box [x_l, x_u]; drops those that are.",
    )?;

    reg.add_bool_option(
        "presolve_linear_eq_reduction",
        "Eliminate variables via linear-equality rows.",
        false,
        "Reduces problem dimension by Gauss-eliminating against \
         linear equality rows. Off by default because it changes \
         the variable count and complicates sensitivity integration.",
    )?;

    reg.add_bool_option(
        "presolve_licq_check",
        "Detect rank-deficient equality blocks before the IPM starts.",
        true,
        "Probes rank(J_c) at the starting point via a sparse symbolic \
         factor. Interlocks with `presolve_licq_action` to (optionally) \
         auto-activate the ℓ₁-exact penalty-barrier wrapper.",
    )?;

    reg.add_string_option(
        "presolve_licq_action",
        "Action when presolve_licq_check reports rank deficiency.",
        "warn",
        &[
            ("warn", "Report on the journalist; do not modify the solve."),
            (
                "auto_l1",
                "Set l1_exact_penalty_barrier=yes so the ℓ₁ wrapper takes over.",
            ),
        ],
        "Only consulted when presolve_licq_check=yes and the verdict \
         is non-full-rank.",
    )?;

    reg.add_bounded_integer_option(
        "presolve_print_level",
        "Per-pass progress reporting for presolve.",
        0,
        8,
        0,
        "0 silent; 5 prints a one-line summary per pass; 8 prints \
         per-transformation detail (intended for debugging).",
    )?;

    reg.add_bounded_integer_option(
        "presolve_max_passes",
        "Maximum fixed-point iterations across presolve passes.",
        1,
        50,
        3,
        "Bound tightening (Phase 1) is iterated until no bound \
         changes or this cap is hit.",
    )?;

    reg.add_bool_option(
        "presolve_warm_z_bounds",
        "Warm-start bound multipliers for bounds tightened by presolve.",
        true,
        "When a variable's lower (upper) bound is moved by Phase 1 \
         tightening, that side is likely active at the optimum. With \
         this option on and `init_z=yes` requested, the wrapper sets \
         z_l (z_u) for those variables to `presolve_bound_mult_init_val` \
         instead of the global default.",
    )?;

    reg.add_lower_bounded_number_option(
        "presolve_bound_mult_init_val",
        "Value used when warm-starting bound multipliers from presolve.",
        0.0,
        true,
        1.0,
        "Only consulted when presolve_warm_z_bounds=yes.",
    )?;

    // --- Auxiliary-equality preprocessing (Phase 0, issue #53) ------------

    reg.add_bool_option(
        "presolve_auxiliary",
        "Master switch for auxiliary-equality preprocessing (Phase 0).",
        false,
        "Structural NLP preprocessing: incidence graph → bipartite \
         matching → Dulmage-Mendelsohn → block-triangular form → \
         per-block solve → postsolve multiplier recovery. Defaults \
         off; will land incrementally across pounce#53.",
    )?;

    reg.add_lower_bounded_number_option(
        "presolve_auxiliary_tol",
        "Residual tolerance for accepting an auxiliary block solve.",
        0.0,
        true,
        1e-8,
        "Full-space KKT residual after the candidate reduction must \
         stay within this; otherwise the reduction is rejected.",
    )?;

    reg.add_bounded_integer_option(
        "presolve_auxiliary_max_block_dim",
        "Largest block the lightweight Newton solver will attempt.",
        1,
        1024,
        8,
        "Blocks above this dimension fall through to the BlockSolver \
         trait (IPM-backed fallback lands in PR 11 of pounce#53).",
    )?;

    reg.add_bounded_number_option(
        "presolve_auxiliary_wall_time_fraction",
        "Fraction of overall wall-time budget Phase 0 may spend.",
        0.0,
        true,
        1.0,
        false,
        0.1,
        "When the solver enforces a wall-time limit, Phase 0 is given \
         this fraction of the budget; honoured by PR 8 onwards.",
    )?;

    reg.add_string_option(
        "presolve_auxiliary_coupling",
        "Coupling-class gate for auxiliary block elimination.",
        "safe",
        &[
            (
                "none",
                "Run Phase 0 for diagnostics only; eliminate nothing.",
            ),
            (
                "safe",
                "Eliminate only PureEquality blocks (matches ripopt default).",
            ),
            (
                "aggressive",
                "Also accept ObjectiveCoupled blocks as postsolve candidates.",
            ),
        ],
        "Only consulted when presolve_auxiliary=yes.",
    )?;

    reg.add_bool_option(
        "presolve_auxiliary_diagnostics",
        "Emit the auxiliary-preprocessing diagnostics summary.",
        false,
        "When yes, the diagnostics struct (block counts, timings, \
         rejection reasons) is published via the journalist after \
         Phase 0 runs.",
    )?;

    reg.set_registering_category("");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn reg_with_presolve() -> Rc<RegisteredOptions> {
        let reg = RegisteredOptions::default();
        register_options(&reg).unwrap();
        Rc::new(reg)
    }

    #[test]
    fn defaults_round_trip() {
        let reg = reg_with_presolve();
        let opts = OptionsList::with_registered(reg);
        let p = PresolveOptions::from_options_list(&opts).unwrap();
        assert!(!p.enabled);
        assert!(p.bound_tightening);
        assert!(p.redundant_constraint_removal);
        assert!(!p.linear_eq_reduction);
        assert!(p.licq_check);
        assert_eq!(p.licq_action, LicqAction::Warn);
        assert_eq!(p.print_level, 0);
        assert_eq!(p.max_passes, 3);
    }

    #[test]
    fn enabling_master_switch_round_trips() {
        let reg = reg_with_presolve();
        let mut opts = OptionsList::with_registered(reg);
        opts.set_string_value("presolve", "yes", true, false)
            .unwrap();
        opts.set_string_value("presolve_licq_action", "auto_l1", true, false)
            .unwrap();
        opts.set_integer_value("presolve_max_passes", 5, true, false)
            .unwrap();
        let p = PresolveOptions::from_options_list(&opts).unwrap();
        assert!(p.enabled);
        assert_eq!(p.licq_action, LicqAction::AutoL1);
        assert_eq!(p.max_passes, 5);
    }

    #[test]
    fn invalid_licq_action_rejected_at_set_time() {
        let reg = reg_with_presolve();
        let mut opts = OptionsList::with_registered(reg);
        // Registered enum option only accepts "warn" / "auto_l1".
        let err = opts
            .set_string_value("presolve_licq_action", "bogus", true, false)
            .err();
        assert!(err.is_some(), "invalid enum should be rejected at set time");
    }

    #[test]
    fn auxiliary_defaults_round_trip() {
        let reg = reg_with_presolve();
        let opts = OptionsList::with_registered(reg);
        let p = PresolveOptions::from_options_list(&opts).unwrap();
        assert!(!p.auxiliary);
        assert_eq!(p.auxiliary_tol, 1e-8);
        assert_eq!(p.auxiliary_max_block_dim, 8);
        assert_eq!(p.auxiliary_wall_time_fraction, 0.1);
        assert_eq!(p.auxiliary_coupling, AuxiliaryCouplingPolicy::Safe);
        assert!(!p.auxiliary_diagnostics);
    }

    #[test]
    fn auxiliary_master_switch_round_trips() {
        let reg = reg_with_presolve();
        let mut opts = OptionsList::with_registered(reg);
        opts.set_string_value("presolve_auxiliary", "yes", true, false)
            .unwrap();
        opts.set_string_value("presolve_auxiliary_coupling", "aggressive", true, false)
            .unwrap();
        opts.set_numeric_value("presolve_auxiliary_tol", 1e-10, true, false)
            .unwrap();
        opts.set_integer_value("presolve_auxiliary_max_block_dim", 16, true, false)
            .unwrap();
        opts.set_string_value("presolve_auxiliary_diagnostics", "yes", true, false)
            .unwrap();
        let p = PresolveOptions::from_options_list(&opts).unwrap();
        assert!(p.auxiliary);
        assert_eq!(p.auxiliary_coupling, AuxiliaryCouplingPolicy::Aggressive);
        assert_eq!(p.auxiliary_tol, 1e-10);
        assert_eq!(p.auxiliary_max_block_dim, 16);
        assert!(p.auxiliary_diagnostics);
    }

    #[test]
    fn invalid_auxiliary_coupling_rejected_at_set_time() {
        let reg = reg_with_presolve();
        let mut opts = OptionsList::with_registered(reg);
        let err = opts
            .set_string_value("presolve_auxiliary_coupling", "bogus", true, false)
            .err();
        assert!(err.is_some(), "invalid enum should be rejected at set time");
    }
}
