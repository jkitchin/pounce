//! Options registered for `ipopt.opt` compatibility whose *feature*
//! pounce does not implement (gh#483 follow-up, continuing #191).
//!
//! # Why these are refused rather than ignored
//!
//! `upstream_options.rs` is a faithful port of Ipopt's option registry:
//! every name Ipopt registers is registered here, so an `ipopt.opt`
//! written for Ipopt parses unchanged. That is a real compatibility
//! benefit — and it silently turned ~200 knobs into no-ops, because
//! registering an option says nothing about implementing it. Setting one
//! did exactly nothing and said exactly nothing.
//!
//! Issue #191 audited this class and fixed the half where the *feature*
//! runs and only the option's read site was missing. It explicitly
//! scoped out "feature genuinely unimplemented — expected no-ops". This
//! module closes that half: an option naming a feature pounce does not
//! have is now an error, not a shrug.
//!
//! # What is and is not in the table
//!
//! Membership was established per option, not guessed:
//!
//! 1. the option's name appears in **no** crate source outside the
//!    registry (whole-word — `penalty_max` does not count as present
//!    because `l1_penalty_max` exists), **and**
//! 2. the *feature* it configures is absent too.
//!
//! Both are needed. An option whose name is unread but whose feature
//! runs — the `limited_memory_*` tail, the corrector knobs — is a
//! missing read site, not a missing feature; refusing those would fail
//! solves whose current answers are already correct. They are
//! deliberately **not** here; wiring them is the other half of the work.
//!
//! A third shape turned up while wiring that half and belongs to
//! neither: an option for a *sub-capability* of a feature that does run.
//! `max_resto_iter` and `resto_failure_feasibility_threshold` are the
//! examples — restoration runs, but there is no iteration cap or failure
//! threshold to point a read site at, so honouring them means building
//! the capability, not adding a line. They are left out of this table
//! until that call is made, and so remain silent for now.
//!
//! The clearest case is the penalty line search. pounce implements
//! `IpPenaltyLSAcceptor` (`line_search_method=penalty`), so its knobs
//! (`nu_init`, `nu_inc`, `rho`, `eta_penalty`) are read sites to add.
//! Ipopt's *other* penalty acceptor — the CG-penalty / inexact-Newton
//! one — has no counterpart here at all, and the port registered its
//! whole option set. Those are refused.
//!
//! An entry leaves the table by being implemented. `option_file_name`
//! was here — refusing it was the cheap half of gh#518's "implement it
//! or fail loudly" — until gh#518 got the other half:
//! [`crate::application::IpoptApplication::initialize_with_option_file`]
//! reads the named file, so the option now configures something.
//!
//! # Backend knobs warn, they do not refuse
//!
//! The 111 `ma27_*` / `ma77_*` / `ma86_*` / `ma97_*` / `mumps_*` /
//! `pardiso_*` / `pardisomkl_*` / `spral_*` / `wsmp_*` options, plus
//! `pardisolib`, tune linear-solver backends pounce does not ship at
//! all: it factors the KKT system with `feral` or with MA57. They were
//! silent — #551 section 2 — and they are the one class where the
//! refusal above is the wrong instrument.
//!
//! Refusing them attacks the goal the registry exists to serve. The
//! rest of this table refuses an option whose *feature* the caller is
//! plainly asking for; `ma97_order` in a portable `ipopt.opt` is
//! usually not that. Such a file routinely carries settings for several
//! backends at once so that one file runs everywhere, and pounce would
//! reject it wholesale over knobs the run never touches — a hard error
//! for a user who is not using MA97 and never asked pounce to. That is
//! strictly worse than the silence: it breaks a working file instead of
//! under-serving one.
//!
//! But silence is what #677 cost, so the answer is the third
//! disposition this module already carries. The precedent is
//! [`UNEXPLOITED_HINTS`] — pinned by `a_caching_hint_warns_but_solves`
//! in `pounce-cli/tests/unimplemented_options.rs` — where the project
//! chose WARN over REFUSE for exactly this trade: ignoring the option
//! costs the caller nothing they cannot see, so blocking the solve
//! would take more from them than the silence did. Backend knobs are a
//! stronger case for it than the hints are: a hint changes the
//! evaluation count, whereas a knob for a backend that is not linked
//! could not have changed anything even in principle.
//!
//! Three properties follow from that, and each is pinned by a test:
//!
//! 1. **Only when explicitly set to a non-default.** The same gate as
//!    everything else here (below). A file spelling out `ma97_u 1e-8`
//!    asks for nothing, and a default run must stay completely silent —
//!    `a_default_run_is_silent`.
//! 2. **One line per backend family, not per option.** An MA97-tuned
//!    file sets a dozen `ma97_*` knobs; a dozen near-identical lines is
//!    noise a reader learns to skip, which is silence with extra steps.
//!    The warning names the backend, lists the options it saw, and says
//!    the rest of the family is inert too.
//! 3. **The solve runs and its answer is unaffected**, which the
//!    warning says in as many words — otherwise a warning naming a
//!    linear solver reads as "your factorization may be wrong".
//!
//! `hsllib` stays in the refusal table above rather than moving here,
//! and the line is deliberate: pounce *has* an HSL backend (MA57), so
//! `hsllib` is a caller trying to reach a solver pounce can actually
//! run, by a mechanism it does not have — the refusal tells them to
//! build with `--features ma57` instead of letting them believe MA57 is
//! loaded. `pardisolib` has no such other route (there is no Pardiso
//! here by any means), so it warns with the rest of its family.
//!
//! # The default gate
//!
//! Only an explicit value **different from the registered default** is
//! refused. `expect_infeasible_problem_ctol` left alone, or an
//! `ipopt.opt` that spells out defaults, must keep working: those ask
//! for nothing. Refusing them would break the very compatibility the
//! registry exists to provide.

use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::{DefaultValue, RegisteredOptions};

/// One unimplemented feature and the options that configure it.
pub struct UnimplementedFeature {
    /// Named in the error, e.g. "the CG-penalty / inexact-Newton line search".
    pub feature: &'static str,
    /// What the caller can do instead. Empty when there is nothing.
    pub advice: &'static str,
    /// The options that belong to it.
    pub options: &'static [&'static str],
    /// Issue tracking the missing feature, named in the error.
    pub issue: u32,
}

/// Feature groups pounce does not implement. Refused when set.
pub const UNIMPLEMENTED_FEATURES: &[UnimplementedFeature] = &[
    UnimplementedFeature {
        issue: 483,
        feature: "the Chen-Goldfarb (CG-penalty) / inexact-Newton line search \
                  — Ipopt's `CGPenaltyLSAcceptor`",
        advice: "pounce implements the filter line search (the default) and \
                 `line_search_method=penalty` (`IpPenaltyLSAcceptor`); tune \
                 those instead",
        options: &[
            "chi_cup",
            "chi_hat",
            "chi_tilde",
            "delta_y_max",
            "epsilon_c",
            "eta_min",
            "fast_des_fact",
            "gamma_hat",
            "gamma_tilde",
            "kappa_x_dis",
            "kappa_y_dis",
            "min_alpha_primal",
            "mult_diverg_feasibility_tol",
            "mult_diverg_y_tol",
            "never_use_fact_cgpen_direction",
            "never_use_piecewise_penalty_ls",
            "pen_des_fact",
            "pen_init_fac",
            "pen_theta_max_fact",
            "penalty_init_max",
            "penalty_init_min",
            "penalty_max",
            "penalty_update_compl_tol",
            "penalty_update_infeasibility_tol",
            "piecewisepenalty_gamma_infeasi",
            "piecewisepenalty_gamma_obj",
            "vartheta",
            "inexact_algorithm",
        ],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "derivative approximation by finite differences",
        advice: "supply `eval_grad_f` / `eval_jac_g` / `eval_h`, and check them \
                 with `derivative_test=first-order`",
        options: &[
            "gradient_approximation",
            "jacobian_approximation",
            "findiff_perturbation",
        ],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "linear-dependency detection on the equality constraints",
        advice: "pounce's presolve removes structurally redundant rows; see \
                 `presolve`",
        options: &[
            "dependency_detector",
            "dependency_detection_with_rhs",
            "ma28_pivtol",
        ],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "the per-iteration NaN/Inf check on derivative matrices",
        advice: "`derivative_test=first-order` checks the derivatives once, at \
                 the starting point",
        options: &["check_derivatives_for_naninf"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "least-square initialization of *all* dual variables \
                  (the first-order-optimality fit)",
        advice: "the equality multipliers are least-square initialized \
                 regardless (capped by `constr_mult_init_max`), and the bound \
                 multipliers take `bound_mult_init_val` — which is what \
                 `least_square_init_duals=no` asks for",
        options: &["least_square_init_duals"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "a selectable constraint-violation norm",
        advice: "pounce measures the violation in the 2-norm throughout",
        options: &["constraint_violation_norm_type"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "magic steps",
        advice: "",
        options: &["magic_steps"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "bound replacement on the original problem",
        advice: "",
        options: &["replace_bounds"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "the L-BFGS augmented-system and space variants",
        advice: "`hessian_approximation=limited-memory` uses the low-rank \
                 augmented system unconditionally",
        options: &["hessian_approximation_space", "limited_memory_aug_solver"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "skipping the finalize-solution callback",
        advice: "",
        options: &["skip_finalize_solution_call"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "the dynamic HSL loader",
        advice: "MA57 is linked at build time with `--features ma57`",
        options: &["hsllib"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "these output controls",
        advice: "use `print_level` (0 silences the solver) and `sb=yes` to \
                 suppress the banner",
        options: &["suppress_all_output", "debug_print_level"],
    },
    UnimplementedFeature {
        issue: 483,
        feature: "a randomly perturbed evaluation point for the derivative \
                  checker",
        advice: "pounce's checker tests at the (bound-projected) starting point, \
                 which is where the solve actually begins",
        options: &["point_perturbation_radius"],
    },
    UnimplementedFeature {
        issue: 606,
        feature: "reuse of a previously-solved iterate or problem structure \
                  through Ipopt's `TNLP::GetWarmStartIterate` surface",
        advice: "pounce's warm start goes through `TNLP::get_starting_point` \
                 with `warm_start_init_point=yes`, which carries the primal \
                 point and all three multiplier blocks; from Python, \
                 `pounce.WarmStart.from_info` packages it",
        options: &["warm_start_entire_iterate", "warm_start_same_structure"],
    },
];

/// One registered *value* of a string option that pounce does not
/// implement, even though the option itself is read and other values of
/// it work.
///
/// [`UNIMPLEMENTED_FEATURES`] refuses a whole option; this refuses one
/// mode of one. The registry keeps upstream's full value list so an
/// `ipopt.opt` written for Ipopt still parses — but a value that parses
/// and then quietly behaves as a *different* mode is the same lie the
/// module docstring is about, one level down.
pub struct UnimplementedValue {
    /// The option's registered name.
    pub option: &'static str,
    /// The value pounce does not implement.
    pub value: &'static str,
    /// What that value would mean, named in the error.
    pub feature: &'static str,
    /// What the caller can do instead. Empty when there is nothing.
    pub advice: &'static str,
}

/// String-option values pounce does not implement. Refused when set.
pub const UNIMPLEMENTED_VALUES: &[UnimplementedValue] = &[UnimplementedValue {
    option: "bound_mult_init_method",
    value: "mu-based",
    feature: "initializing each bound multiplier to mu_init divided by its \
              own slack",
    advice: "`bound_mult_init_method=constant` (the default) initializes them \
             all to `bound_mult_init_val`, which you can set",
}];

/// Options that *are* honored in the sense that matters — the answer is
/// unaffected — but whose performance hint pounce does not exploit.
/// These warn rather than fail: refusing them would stop a solve that
/// returns the right result today, only a little slower.
pub const UNEXPLOITED_HINTS: &[&str] = &[
    "grad_f_constant",
    "hessian_constant",
    "jac_c_constant",
    "jac_d_constant",
];

/// One linear-solver backend pounce does not implement, and the
/// registered options that tune it.
///
/// Separate from [`UnimplementedFeature`] because the disposition is
/// different: these *warn* and solve, they never refuse. See "Backend
/// knobs warn, they do not refuse" in the module header for why.
pub struct UnimplementedBackend {
    /// Named in the warning, e.g. "the HSL MA97 sparse symmetric linear
    /// solver".
    pub backend: &'static str,
    /// The registered prefix the family shares, quoted in the warning so
    /// the user learns the whole group is inert, not just the one option
    /// they happened to set.
    pub family: &'static str,
    /// Every registered option of this backend. Complete per family —
    /// `backend_families_are_complete` fails if the registry grows one
    /// that is missing here, which would hand it back its silence.
    pub options: &'static [&'static str],
}

/// Every linear-solver backend pounce does not implement, with the
/// options that tune it. Warned about when set; never refused.
pub const UNIMPLEMENTED_BACKENDS: &[UnimplementedBackend] = &[
    UnimplementedBackend {
        backend: "the HSL MA27 sparse symmetric linear solver",
        family: "ma27_*",
        options: &[
            "ma27_ignore_singularity",
            "ma27_la_init_factor",
            "ma27_liw_init_factor",
            "ma27_meminc_factor",
            "ma27_pivtol",
            "ma27_pivtolmax",
            "ma27_print_level",
            "ma27_skip_inertia_check",
        ],
    },
    UnimplementedBackend {
        backend: "the HSL MA77 out-of-core sparse symmetric linear solver",
        family: "ma77_*",
        options: &[
            "ma77_buffer_lpage",
            "ma77_buffer_npage",
            "ma77_file_size",
            "ma77_maxstore",
            "ma77_nemin",
            "ma77_order",
            "ma77_print_level",
            "ma77_small",
            "ma77_static",
            "ma77_u",
            "ma77_umax",
        ],
    },
    UnimplementedBackend {
        backend: "the HSL MA86 parallel sparse symmetric linear solver",
        family: "ma86_*",
        options: &[
            "ma86_nemin",
            "ma86_order",
            "ma86_print_level",
            "ma86_scaling",
            "ma86_small",
            "ma86_static",
            "ma86_u",
            "ma86_umax",
        ],
    },
    UnimplementedBackend {
        backend: "the HSL MA97 sparse symmetric linear solver",
        family: "ma97_*",
        options: &[
            "ma97_dump_matrix",
            "ma97_nemin",
            "ma97_order",
            "ma97_print_level",
            "ma97_scaling",
            "ma97_scaling1",
            "ma97_scaling2",
            "ma97_scaling3",
            "ma97_small",
            "ma97_solve_blas3",
            "ma97_switch1",
            "ma97_switch2",
            "ma97_switch3",
            "ma97_u",
            "ma97_umax",
        ],
    },
    UnimplementedBackend {
        backend: "the MUMPS sparse symmetric linear solver",
        family: "mumps_*",
        options: &[
            "mumps_dep_tol",
            "mumps_mem_percent",
            "mumps_mpi_communicator",
            "mumps_permuting_scaling",
            "mumps_pivot_order",
            "mumps_pivtol",
            "mumps_pivtolmax",
            "mumps_print_level",
            "mumps_scaling",
        ],
    },
    UnimplementedBackend {
        backend: "the Pardiso linear solver (pardiso-project.org)",
        family: "pardiso_*",
        options: &[
            "pardiso_iter_coarse_size",
            "pardiso_iter_dropping_factor",
            "pardiso_iter_dropping_schur",
            "pardiso_iter_inverse_norm_factor",
            "pardiso_iter_max_levels",
            "pardiso_iter_max_row_fill",
            "pardiso_iter_relative_tol",
            "pardiso_iterative",
            "pardiso_matching_strategy",
            "pardiso_max_droptol_corrections",
            "pardiso_max_iter",
            "pardiso_max_iterative_refinement_steps",
            "pardiso_msglvl",
            "pardiso_order",
            "pardiso_redo_symbolic_fact_only_if_inertia_wrong",
            "pardiso_repeated_perturbation_means_singular",
            "pardiso_skip_inertia_check",
            "pardisolib",
        ],
    },
    UnimplementedBackend {
        backend: "the Pardiso linear solver bundled with Intel MKL",
        family: "pardisomkl_*",
        options: &[
            "pardisomkl_matching_strategy",
            "pardisomkl_max_iterative_refinement_steps",
            "pardisomkl_msglvl",
            "pardisomkl_order",
            "pardisomkl_redo_symbolic_fact_only_if_inertia_wrong",
            "pardisomkl_repeated_perturbation_means_singular",
            "pardisomkl_skip_inertia_check",
        ],
    },
    UnimplementedBackend {
        backend: "the SPRAL SSIDS sparse symmetric linear solver",
        family: "spral_*",
        options: &[
            "spral_cpu_block_size",
            "spral_gpu_perf_coeff",
            "spral_ignore_numa",
            "spral_max_load_inbalance",
            "spral_min_gpu_work",
            "spral_nemin",
            "spral_order",
            "spral_pivot_method",
            "spral_print_level",
            "spral_scaling",
            "spral_scaling_1",
            "spral_scaling_2",
            "spral_scaling_3",
            "spral_small",
            "spral_small_subtree_threshold",
            "spral_switch_1",
            "spral_switch_2",
            "spral_switch_3",
            "spral_u",
            "spral_umax",
            "spral_use_gpu",
        ],
    },
    UnimplementedBackend {
        backend: "the WSMP sparse symmetric linear solver",
        family: "wsmp_*",
        options: &[
            "wsmp_inexact_droptol",
            "wsmp_inexact_fillin_limit",
            "wsmp_iterative",
            "wsmp_max_iter",
            "wsmp_no_pivoting",
            "wsmp_num_threads",
            "wsmp_ordering_option",
            "wsmp_ordering_option2",
            "wsmp_pivtol",
            "wsmp_pivtolmax",
            "wsmp_scaling",
            "wsmp_singularity_threshold",
            "wsmp_skip_inertia_check",
            "wsmp_write_matrix_iteration",
        ],
    },
];

/// Warnings for backend knobs the caller set. Never blocks a solve, and
/// emits at most one line per backend family: an `ipopt.opt` tuned for
/// MA97 sets a dozen `ma97_*` knobs at once, and a dozen near-identical
/// lines would be noise the reader learns to skip.
///
/// Same default gate as everything else here — an `ipopt.opt` that
/// spells out `ma97_u 1e-8` (the registered default) asks for nothing
/// and gets nothing said about it.
pub fn backend_warnings(options: &OptionsList, reg: &RegisteredOptions) -> Vec<String> {
    UNIMPLEMENTED_BACKENDS
        .iter()
        .filter_map(|group| {
            let set: Vec<&str> = group
                .options
                .iter()
                .copied()
                .filter(|name| set_to_a_non_default(options, reg, name))
                .collect();
            if set.is_empty() {
                return None;
            }
            let named = set
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let (verb, ignored, registered) = if set.len() == 1 {
                ("configures", "it is ignored".to_string(), "The name is")
            } else {
                (
                    "configure",
                    format!("those {} are ignored", set.len()),
                    "The names are",
                )
            };
            Some(format!(
                "pounce: warning: {named} {verb} {}, which pounce does not \
                 implement, so {ignored} — as is every other `{}` option. \
                 pounce factors the KKT system with `feral` (pure Rust, the \
                 default) or MA57 (`linear_solver=ma57`, in a `--features \
                 ma57` build); no setting written for another backend \
                 transfers to either. {registered} registered so an \
                 `ipopt.opt` written for Ipopt still parses unchanged — which \
                 is why this is a warning and not an error: the solve runs, \
                 and its result is unaffected. Tracking issue: \
                 https://github.com/jkitchin/pounce/issues/551",
                group.backend, group.family,
            ))
        })
        .collect()
}

/// An option set to something the registry says is not its default.
///
/// Both halves matter. `found` alone would fire on an `ipopt.opt` that
/// spells out a default; comparing values alone would fire on nothing,
/// since an unset option *reads back* as its default.
pub(crate) fn set_to_a_non_default(
    options: &OptionsList,
    reg: &RegisteredOptions,
    name: &str,
) -> bool {
    let Some(opt) = reg.get_option(name) else {
        return false;
    };
    match &opt.default {
        // Bools are registered as `yes`/`no` string options, so this arm
        // covers them too.
        DefaultValue::String(d) => {
            matches!(options.get_string_value(name, ""), Ok((v, true)) if !v.eq_ignore_ascii_case(d))
        }
        DefaultValue::Number(d) => {
            matches!(options.get_numeric_value(name, ""), Ok((v, true)) if v != *d)
        }
        DefaultValue::Integer(d) => {
            matches!(options.get_integer_value(name, ""), Ok((v, true)) if v != *d)
        }
        DefaultValue::None => false,
    }
}

/// The first unimplemented-feature option the caller set, with the
/// message it earns. `None` when nothing in the table was touched.
pub fn refusal(options: &OptionsList, reg: &RegisteredOptions) -> Option<String> {
    for group in UNIMPLEMENTED_FEATURES {
        for name in group.options {
            if set_to_a_non_default(options, reg, name) {
                let advice = if group.advice.is_empty() {
                    String::new()
                } else {
                    format!(" Instead: {}.", group.advice)
                };
                return Some(format!(
                    "pounce: `{name}` configures {}, which pounce does not \
                     implement. It is registered so an ipopt.opt written for \
                     Ipopt still parses, but setting it used to do nothing at \
                     all — silently — so it is refused instead.{advice} \
                     Remove it to run. Tracking issue: \
                     https://github.com/jkitchin/pounce/issues/{}",
                    group.feature, group.issue
                ));
            }
        }
    }
    None
}

/// The first unimplemented *value* the caller selected, with the message
/// it earns. `None` when every string option holds a mode pounce runs.
///
/// No default gate here, unlike [`refusal`]: a value that equals the
/// registered default is by construction the implemented one, so it
/// never reaches the table.
pub fn value_refusal(options: &OptionsList) -> Option<String> {
    for entry in UNIMPLEMENTED_VALUES {
        let selected = matches!(
            options.get_string_value(entry.option, ""),
            Ok((v, true)) if v.eq_ignore_ascii_case(entry.value)
        );
        if !selected {
            continue;
        }
        let advice = if entry.advice.is_empty() {
            String::new()
        } else {
            format!(" Instead: {}.", entry.advice)
        };
        return Some(format!(
            "pounce: `{}={}` selects {}, which pounce does not implement. The \
             value is registered so an ipopt.opt written for Ipopt still \
             parses; falling back to another mode would silently run a \
             different initialization than the one you asked for, so it is \
             refused instead.{advice} Tracking issue: \
             https://github.com/jkitchin/pounce/issues/604",
            entry.option, entry.value, entry.feature,
        ));
    }
    None
}

/// Warnings for hints pounce does not exploit. Never blocks a solve.
pub fn hint_warnings(options: &OptionsList, reg: &RegisteredOptions) -> Vec<String> {
    UNEXPLOITED_HINTS
        .iter()
        .filter(|name| set_to_a_non_default(options, reg, name))
        .map(|name| {
            format!(
                "pounce: warning: `{name}` is a caching hint pounce does not \
                 exploit — it re-evaluates each iteration regardless. Your \
                 answer is unaffected; only the evaluation count is. \
                 (gh#483)"
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn registry() -> std::rc::Rc<RegisteredOptions> {
        let r = RegisteredOptions::new();
        crate::upstream_options::register_all_upstream_options(&r).expect("register");
        r
    }

    /// A fresh options list over the shared registry, plus a handle on
    /// the registry itself for the default lookups.
    fn fixture() -> (OptionsList, std::rc::Rc<RegisteredOptions>) {
        let reg = registry();
        (OptionsList::with_registered(std::rc::Rc::clone(&reg)), reg)
    }

    /// Every name in the table must actually be registered — a typo
    /// would make its entry dead code that silently never fires, which
    /// is the exact failure mode this module exists to remove.
    #[test]
    fn every_listed_option_is_registered() {
        let (_, reg) = fixture();
        for group in UNIMPLEMENTED_FEATURES {
            for name in group.options {
                assert!(
                    reg.get_option(name).is_some(),
                    "`{name}` is in the refusal table but is not registered",
                );
            }
        }
        for name in UNEXPLOITED_HINTS {
            assert!(
                reg.get_option(name).is_some(),
                "`{name}` is in the hint table but is not registered",
            );
        }
        for group in UNIMPLEMENTED_BACKENDS {
            for name in group.options {
                assert!(
                    reg.get_option(name).is_some(),
                    "`{name}` is in the backend table but is not registered",
                );
            }
        }
    }

    /// The backend groups must cover their families *completely*. A
    /// `ma97_*` option registered later and not added here would be
    /// silent again — the exact defect this table removes — and it would
    /// slip past `no_silent_options.rs` too, which only asks whether a
    /// name is declared somewhere, not whether the family is whole.
    #[test]
    fn backend_families_are_complete() {
        let (_, reg) = fixture();
        for group in UNIMPLEMENTED_BACKENDS {
            let prefix = group
                .family
                .strip_suffix('*')
                .expect("family is a prefix glob, e.g. `ma97_*`");
            for opt in reg.registered_options_in_order() {
                if !opt.name.starts_with(prefix) {
                    continue;
                }
                assert!(
                    group.options.contains(&opt.name.as_str()),
                    "`{}` is registered and matches `{}` but is missing from \
                     the {} group, so setting it would still be silent",
                    opt.name,
                    group.family,
                    group.backend,
                );
            }
        }
    }

    /// No option may appear twice — once in two feature groups, or in
    /// both tables — or the message a user gets would depend on table
    /// order.
    #[test]
    fn the_tables_do_not_overlap() {
        let mut seen = BTreeSet::new();
        for name in UNIMPLEMENTED_FEATURES
            .iter()
            .flat_map(|g| g.options.iter())
            .chain(UNEXPLOITED_HINTS.iter())
            .chain(UNIMPLEMENTED_BACKENDS.iter().flat_map(|g| g.options.iter()))
        {
            assert!(seen.insert(*name), "`{name}` is listed twice");
        }
    }

    /// A pristine options list touches nothing.
    #[test]
    fn defaults_are_not_refused() {
        let (opts, reg) = fixture();
        assert_eq!(refusal(&opts, &reg), None);
        assert!(hint_warnings(&opts, &reg).is_empty());
        assert!(
            backend_warnings(&opts, &reg).is_empty(),
            "a default run must stay silent",
        );
    }

    /// A backend knob warns and solves — it never refuses. Refusing
    /// would fail a portable `ipopt.opt` over a backend the run does not
    /// use, which is the compatibility the registry exists to provide.
    #[test]
    fn a_backend_knob_warns_but_does_not_refuse() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("ma97_order", "metis", true, false)
            .unwrap();
        assert_eq!(
            refusal(&opts, &reg),
            None,
            "a backend knob must not block a solve",
        );
        let warnings = backend_warnings(&opts, &reg);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let w = &warnings[0];
        assert!(w.contains("warning:"), "{w}");
        assert!(w.contains("`ma97_order`"), "{w}");
        assert!(w.contains("MA97"), "the backend must be named: {w}");
        assert!(w.contains("`ma97_*`"), "the family must be named: {w}");
        // The user has to be told the answer is not at risk, or a
        // warning naming a linear solver reads as "your factorization
        // may be wrong".
        assert!(w.contains("result is unaffected"), "{w}");
        assert!(w.contains("551"), "{w}");
    }

    /// Explicitly writing a backend knob's registered default asks for
    /// nothing — the same gate the refusal table uses — so it must not
    /// even warn. A generated `ipopt.opt` spells defaults out.
    #[test]
    fn a_backend_knob_at_its_default_is_silent() {
        let (mut opts, reg) = fixture();
        // `ma97_order` defaults to "auto", `pardiso_msglvl` to 0.
        opts.set_string_value("ma97_order", "auto", true, false)
            .unwrap();
        opts.set_integer_value("pardiso_msglvl", 0, true, false)
            .unwrap();
        assert!(backend_warnings(&opts, &reg).is_empty());
    }

    /// One line per backend family, not per option: an MA97-tuned
    /// `ipopt.opt` sets a dozen `ma97_*` knobs at once, and a dozen
    /// near-identical lines is noise the reader learns to skip — silence
    /// with extra steps. The one line names every knob it saw.
    #[test]
    fn the_warning_is_grouped_by_backend_family() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("ma97_order", "metis", true, false)
            .unwrap();
        opts.set_numeric_value("ma97_u", 1e-4, true, false).unwrap();
        opts.set_string_value("ma97_scaling", "mc64", true, false)
            .unwrap();
        opts.set_integer_value("pardiso_msglvl", 1, true, false)
            .unwrap();

        let warnings = backend_warnings(&opts, &reg);
        assert_eq!(
            warnings.len(),
            2,
            "one per family, not per option: {warnings:?}"
        );
        let ma97 = warnings.iter().find(|w| w.contains("MA97")).expect("MA97");
        for name in ["`ma97_order`", "`ma97_u`", "`ma97_scaling`"] {
            assert!(ma97.contains(name), "{ma97}");
        }
        assert!(ma97.contains("those 3 are ignored"), "{ma97}");
        assert!(
            warnings.iter().any(|w| w.contains("Pardiso")),
            "{warnings:?}",
        );
    }

    /// `pardisolib` warns with its family rather than being refused like
    /// `hsllib`. The difference is that pounce *has* an HSL backend, so
    /// `hsllib` is a caller reaching for a solver pounce can run by a
    /// mechanism it lacks; there is no Pardiso here by any route.
    #[test]
    fn pardisolib_warns_with_the_pardiso_family() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("pardisolib", "libpardiso600.so", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None);
        let warnings = backend_warnings(&opts, &reg);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("`pardisolib`"), "{:?}", warnings[0]);
        assert!(warnings[0].contains("Pardiso"), "{:?}", warnings[0]);

        // …while `hsllib` keeps its refusal.
        let (mut opts, reg) = fixture();
        opts.set_string_value("hsllib", "libcoinhsl.so", true, false)
            .unwrap();
        assert!(refusal(&opts, &reg).is_some());
    }

    /// Explicitly writing a default is how a generated `ipopt.opt` looks;
    /// it asks for nothing and must not fail.
    #[test]
    fn explicitly_setting_the_default_is_not_refused() {
        let (mut opts, reg) = fixture();
        // `dependency_detector` defaults to "none"; `magic_steps` to "no".
        opts.set_string_value("dependency_detector", "none", true, false)
            .unwrap();
        opts.set_string_value("magic_steps", "no", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None);
    }

    /// …but asking for the feature is refused, by name, with a pointer.
    #[test]
    fn requesting_an_unimplemented_feature_is_refused() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("dependency_detector", "mumps", true, false)
            .unwrap();
        let msg = refusal(&opts, &reg).expect("must refuse");
        assert!(msg.contains("dependency_detector"), "{msg}");
        assert!(msg.contains("linear-dependency detection"), "{msg}");
        assert!(msg.contains("483"), "{msg}");
    }

    /// Numeric knobs of an absent feature are refused the same way.
    #[test]
    fn a_numeric_knob_of_an_absent_feature_is_refused() {
        let (mut opts, reg) = fixture();
        opts.set_numeric_value("penalty_init_max", 42.0, true, false)
            .unwrap();
        let msg = refusal(&opts, &reg).expect("must refuse");
        assert!(msg.contains("CG-penalty"), "{msg}");
    }

    /// Hints warn instead of failing: the answer is the same either way,
    /// so blocking the solve would cost the user more than the silence
    /// did.
    #[test]
    fn caching_hints_warn_but_do_not_refuse() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("hessian_constant", "yes", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None, "a hint must not block a solve");
        let warnings = hint_warnings(&opts, &reg);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("hessian_constant"));
    }

    /// `fast_step_computation` was in the refusal table for one commit,
    /// added by hand against the membership rule above. It fails here if
    /// it ever comes back: `PdSearchDirCalc` owns the flag and consumes
    /// it at two sites, so refusing it would fail a solve pounce can
    /// serve. Its read site is wired in `algorithm_builder_from_options`.
    #[test]
    fn fast_step_computation_is_wired_not_refused() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("fast_step_computation", "yes", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None);

        let mut app = crate::application::IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("fast_step_computation yes\n")
            .unwrap();
        assert!(
            app.algorithm_builder_from_options().fast_step_computation,
            "the option must reach the builder, or wiring it changed nothing",
        );
        // …and the default is still off.
        let mut app = crate::application::IpoptApplication::new();
        app.initialize().unwrap();
        assert!(!app.algorithm_builder_from_options().fast_step_computation);
    }

    /// `option_file_name` left the table when gh#518 implemented the
    /// feature it names. Refusing it *from the table* again would be a
    /// regression in the other direction: it now configures the run, so
    /// a user who sets it gets what they asked for rather than an error.
    #[test]
    fn option_file_name_is_implemented_not_in_the_table() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("option_file_name", "tiny.opt", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None);
    }

    /// …but leaving the table must not hand the option back its silence
    /// on the surfaces that still cannot honor it. Only
    /// `initialize_with_option_file` resolves it, and library callers
    /// (Python, the C interface, WASM) never call it — so there, setting
    /// the option is still refused, just by a different guard.
    #[test]
    fn option_file_name_is_refused_where_nothing_resolves_it() {
        let mut app = crate::application::IpoptApplication::new();
        app.initialize().unwrap();
        assert_eq!(app.unhonored_option_file_name(), None, "unset asks nothing");

        app.initialize_with_options_str("option_file_name tiny.opt\n")
            .unwrap();
        let msg = app
            .unhonored_option_file_name()
            .expect("a library caller cannot honor it");
        assert!(msg.contains("tiny.opt"), "{msg}");
        assert!(msg.contains("does not read options files"), "{msg}");
        assert!(msg.contains("518"), "{msg}");
    }

    /// The default gate applies here too: `option_file_name` defaults to
    /// `ipopt.opt`, so a caller replaying a full option dump sets that
    /// value while asking for nothing. Failing them would break the same
    /// compatibility the explicitly-set-default rule protects everywhere
    /// else.
    #[test]
    fn option_file_name_at_its_default_asks_nothing_of_a_library_caller() {
        let mut app = crate::application::IpoptApplication::new();
        app.initialize_with_options_str("option_file_name ipopt.opt\n")
            .unwrap();
        assert_eq!(app.unhonored_option_file_name(), None);
    }

    /// On the CLI's path the option *is* resolved, so the guard stays
    /// quiet — including when the resolver finds no file to read, which
    /// still means the option was honored (there was nothing to read).
    #[test]
    fn the_guard_is_quiet_once_the_option_file_path_has_run() {
        let dir = std::env::temp_dir().join(format!("pounce_gh518_lib_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.opt");
        std::fs::write(&path, "max_iter 5\n").unwrap();

        let mut app = crate::application::IpoptApplication::new();
        app.initialize_with_option_file(Some(&path)).unwrap();
        assert_eq!(app.unhonored_option_file_name(), None);
        assert_eq!(
            app.options().get_integer_value("max_iter", "").unwrap(),
            (5, true),
            "the file must actually have been read",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The restoration switches wired in gh#483 / #191 round 2. Each
    /// field was already consumed by `RestoAlgorithmBuilder`; only the
    /// read site was missing, so setting the option did nothing. The
    /// assertion that matters is that the value *reaches the builder* —
    /// a read site populating a field nobody consumes would be a fresh
    /// silent no-op, the very defect this work removes.
    #[test]
    fn the_restoration_switches_reach_the_builder() {
        for (key, default_on) in [
            ("evaluate_orig_obj_at_resto_trial", true),
            ("expect_infeasible_problem", false),
            ("start_with_resto", false),
        ] {
            let mut app = crate::application::IpoptApplication::new();
            app.initialize().unwrap();
            let resto = app.algorithm_builder_from_options().resto;
            let got = match key {
                "evaluate_orig_obj_at_resto_trial" => resto.evaluate_orig_obj_at_resto_trial,
                "expect_infeasible_problem" => resto.expect_infeasible_problem,
                _ => resto.start_with_resto,
            };
            assert_eq!(got, default_on, "{key}: default changed");

            // Flip it and check the flip lands.
            let flipped = if default_on { "no" } else { "yes" };
            let mut app = crate::application::IpoptApplication::new();
            app.initialize().unwrap();
            app.initialize_with_options_str(&format!("{key} {flipped}\n"))
                .unwrap();
            let resto = app.algorithm_builder_from_options().resto;
            let got = match key {
                "evaluate_orig_obj_at_resto_trial" => resto.evaluate_orig_obj_at_resto_trial,
                "expect_infeasible_problem" => resto.expect_infeasible_problem,
                _ => resto.start_with_resto,
            };
            assert_eq!(
                got, !default_on,
                "{key}={flipped} never reached the builder"
            );
        }
    }

    /// The L-BFGS σ clamp, wired in gh#483 / #191 round 2.
    /// `LimMemQuasiNewtonUpdater` consumes both bounds in
    /// `initial_hessian_scalar`; only the read sites were missing. Note
    /// the fields are named `init_val_{max,min}`, not after the options —
    /// which is why a grep for the option name found nothing and the
    /// consumer had to be located by hand.
    #[test]
    fn the_lbfgs_sigma_clamp_reaches_the_builder() {
        let mut app = crate::application::IpoptApplication::new();
        app.initialize().unwrap();
        let b = app.algorithm_builder_from_options();
        assert_eq!(b.limited_memory_init_val_max, 1e8, "default changed");
        assert_eq!(b.limited_memory_init_val_min, 1e-8, "default changed");

        let mut app = crate::application::IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "limited_memory_init_val_max 5e5\nlimited_memory_init_val_min 1e-3\n",
        )
        .unwrap();
        let b = app.algorithm_builder_from_options();
        assert_eq!(
            b.limited_memory_init_val_max, 5e5,
            "never reached the builder"
        );
        assert_eq!(
            b.limited_memory_init_val_min, 1e-3,
            "never reached the builder"
        );
    }

    /// Options whose *feature* runs and only whose read site is missing
    /// must stay out of the table — refusing them would fail solves that
    /// are correct today. This pins the boundary the triage drew.
    #[test]
    fn options_on_implemented_features_are_not_refused() {
        for (name, value) in [
            // restoration runs; these are missing read sites (#191 round 2)
            ("max_resto_iter", "17"),
            // the filter line search runs
            ("accept_after_max_steps", "3"),
            // L-BFGS runs
            ("limited_memory_max_skipping", "4"),
            // the Mehrotra corrector runs
            ("corrector_type", "affine"),
            // `PdSearchDirCalc` has the flag and consumes it; it was
            // briefly in the refusal table by hand, against the rule
            // above, which would have failed a solve it can serve.
            ("fast_step_computation", "yes"),
        ] {
            let (mut opts, reg) = fixture();
            // The table mixes string, integer and numeric options; try
            // each setter until one takes the value.
            let set = opts.set_string_value(name, value, true, false).is_ok()
                || value
                    .parse::<i32>()
                    .ok()
                    .is_some_and(|v| opts.set_integer_value(name, v, true, false).is_ok())
                || value
                    .parse::<f64>()
                    .ok()
                    .is_some_and(|v| opts.set_numeric_value(name, v, true, false).is_ok());
            assert!(set, "could not set `{name}` to `{value}`");
            assert_eq!(
                refusal(&opts, &reg),
                None,
                "`{name}` configures a feature pounce implements; it needs a \
                 read site, not a refusal",
            );
        }
    }
}
