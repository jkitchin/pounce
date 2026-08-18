//! Every registered option must either reach the algorithm or be
//! refused. Silence is a build failure (#677, #551).
//!
//! `upstream_options.rs` registers every name Ipopt registers so an
//! `ipopt.opt` written for Ipopt parses unchanged. Registering an option
//! says nothing about implementing it, so ~200 names became no-ops the
//! day they were registered — accepted, ignored, and silent about it.
//!
//! #677 is what that costs. `limited_memory_initialization` was
//! registered with Ipopt's `scalar1` default and read nowhere, so every
//! limited-memory solve used `scalar2` instead and no layer of testing
//! could see it: the unit tests check that each σ formula computes
//! correctly (true either way, they pin the formula and not the
//! selection); the registry declared the right default and nothing
//! compares the registry against behaviour; the option was absent from
//! the refusal table so setting it warned nothing; and the fixture
//! corpus never ran the limited-memory path at all. It took an outside
//! user diffing iteration logs against Ipopt on a 59,939-variable model
//! to find it.
//!
//! This test closes the class rather than the instance. It derives the
//! registered-but-unread set mechanically and fails if anything is in it
//! that is not explicitly accounted for. A new option added without a
//! read site cannot reach a release silently — it breaks the build with
//! its own name in the message.
//!
//! #551's two cautions are both load-bearing here and are why this is a
//! source scan rather than a hand-maintained list:
//!
//!   1. "Re-derive the classification mechanically; do not trust a
//!      hand-maintained list." A stale list is how `fast_step_computation`
//!      sat in the refusal table for a commit while its flag *was* being
//!      consumed, which would have failed solves pounce can serve.
//!   2. "Grepping the option name is not enough. Fields are often named
//!      differently — `init_val_max`, not `limited_memory_init_val_max`."
//!      So the scan keys on the registered *name as passed to an
//!      accessor*, never on a field name.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ is the parent of this crate")
        .to_path_buf()
}

fn read_all_rust_sources(root: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                // `target/` under a crate dir would be build output.
                if p.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    out.push((p, s));
                }
            }
        }
    }
    out
}

/// Names registered by `upstream_options.rs`, taken from the source
/// rather than by standing up a registry, so the two cannot drift.
fn registered_names(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (i, _) in src.match_indices("add_") {
        let rest = &src[i..];
        let Some(paren) = rest.find('(') else {
            continue;
        };
        let head = &rest[..paren];
        if !head.ends_with("_option") {
            continue;
        }
        // The registered name is the first string literal after `(`.
        let after = &rest[paren..];
        let Some(q1) = after.find('"') else { continue };
        let Some(q2) = after[q1 + 1..].find('"') else {
            continue;
        };
        let name = &after[q1 + 1..q1 + 1 + q2];
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Names that appear as the key argument of any options accessor —
/// `get_*_value("x")`, or the `read_num`/`read_int` helpers. This is the
/// "does it reach the algorithm" test, and it is deliberately keyed on
/// the registered name, never on a struct field name (#551 caution 2).
fn names_with_read_sites(sources: &[(PathBuf, String)]) -> BTreeSet<String> {
    // The base accessors on `OptionsList`, plus every local `read_*`
    // helper the sources define. Discovering the helpers instead of
    // listing them is the point: a hand-written list would go stale the
    // first time someone adds a `read_yes`-style shorthand, and the scan
    // would then report a wired option as silent. #551 caution 1 —
    // re-derive mechanically, never trust a maintained list.
    let mut accessors: BTreeSet<String> = [
        "get_string_value",
        "get_numeric_value",
        "get_integer_value",
        "get_bool_value",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    for (_, src) in sources {
        for pat in ["let read_", "fn read_"] {
            for (i, _) in src.match_indices(pat) {
                let rest = &src[i + pat.len()..];
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_lowercase() || *c == '_' || c.is_ascii_digit())
                    .collect();
                // Only helpers that take an option key, not e.g. `read_nl_file`.
                let after = &rest[name.len()..];
                if name.is_empty() {
                    continue;
                }
                if after.trim_start().starts_with("= |key")
                    || after.trim_start().starts_with("(key")
                {
                    accessors.insert(format!("read_{name}"));
                }
            }
        }
    }
    let accessors: Vec<&str> = accessors.iter().map(String::as_str).collect();
    let accessor_list: &[&str] = &accessors;
    let mut out = BTreeSet::new();
    for (path, src) in sources {
        // The registry itself does not count as a consumer.
        if path.file_name().is_some_and(|n| n == "upstream_options.rs") {
            continue;
        }
        for acc in accessor_list {
            for (i, _) in src.match_indices(acc) {
                // Allow whitespace/newlines between `(` and the literal,
                // since rustfmt wraps long accessor calls.
                let rest = &src[i + acc.len()..];
                let Some(paren) = rest.find('(') else {
                    continue;
                };
                if !rest[..paren].trim().is_empty() {
                    continue;
                }
                let after = &rest[paren + 1..];
                let lead: String = after.chars().take_while(|c| c.is_whitespace()).collect();
                let after = &after[lead.len()..];
                if !after.starts_with('"') {
                    continue;
                }
                let Some(end) = after[1..].find('"') else {
                    continue;
                };
                out.insert(after[1..1 + end].to_string());
            }
        }
    }
    out
}

/// Names the *warning* table declares as hints pounce does not exploit.
/// Setting one of these prints a warning naming the option and saying
/// what it does and does not cost, which is as far from silence as a
/// refusal is — and the established answer for this shape, because
/// ignoring a caching hint costs evaluations and never correctness, so
/// blocking the solve would be a worse trade than the silence was.
///
/// Parsed from the same source file as the refusal table rather than
/// listed here, for #551 caution 1: a hand-copied set of four names
/// would keep passing this test after someone deleted one from
/// `UNEXPLOITED_HINTS` and took its warning with it.
fn names_declared_unexploited_hints(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    // Anchored on the declaration, not on the name: the name also
    // appears in doc comments, and starting from one of those would
    // scan a block that is not the table.
    let Some(i) = src.find("UNEXPLOITED_HINTS: &[") else {
        return out;
    };
    let rest = &src[i..];
    // `= &[`, not the first `&[`: that one is the `&[&str]` in the type.
    let Some(open) = rest.find("= &[") else {
        return out;
    };
    let Some(end) = rest[open..].find(']') else {
        return out;
    };
    let block = &rest[open..open + end];
    let mut chunks = block.split('"');
    // Odd-indexed chunks are the literals.
    let _ = chunks.next();
    while let Some(name) = chunks.next() {
        out.insert(name.to_string());
        let _ = chunks.next();
    }
    out
}

/// Names the refusal table declares unimplemented. Setting one of these
/// is an error with an explanation, which is the opposite of silence.
fn names_declared_unimplemented(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (i, _) in src.match_indices("options: &[") {
        let rest = &src[i..];
        let Some(end) = rest.find(']') else { continue };
        let block = &rest[..end];
        let mut it = block.char_indices().peekable();
        while let Some((j, c)) = it.next() {
            if c != '"' {
                continue;
            }
            let after = &block[j + 1..];
            if let Some(k) = after.find('"') {
                out.insert(after[..k].to_string());
                for _ in 0..=k {
                    it.next();
                }
            }
        }
    }
    out
}

#[test]
fn every_registered_option_is_read_or_declared_unimplemented() {
    let crates = crates_dir();
    let registry_path = crates.join("pounce-algorithm/src/upstream_options.rs");
    let unimpl_path = crates.join("pounce-algorithm/src/unimplemented_options.rs");
    let registry_src = std::fs::read_to_string(&registry_path).expect("read upstream_options.rs");
    let unimpl_src = std::fs::read_to_string(&unimpl_path).expect("read unimplemented_options.rs");

    let sources = read_all_rust_sources(&crates);
    let registered = registered_names(&registry_src);
    let read = names_with_read_sites(&sources);
    let declared: BTreeSet<String> = names_declared_unimplemented(&unimpl_src)
        .into_iter()
        .chain(names_declared_unexploited_hints(&unimpl_src))
        .collect();

    // Sanity-check the scan itself before trusting its verdict. If the
    // accessor patterns ever stop matching — a rename, a new helper —
    // every option would look unread and the failure message would be
    // 200 lines of noise pointing at nothing.
    assert!(
        registered.len() > 300,
        "the registry scan found only {} options; the parser is broken, not the registry",
        registered.len(),
    );
    assert!(
        read.len() > 150,
        "the read-site scan found only {} options; the accessor list is stale, \
         not every option silently unread",
        read.len(),
    );
    for probe in [
        "mu_strategy",
        "tol",
        "max_iter",
        "limited_memory_initialization",
        // Wired since #190, but through a loop over an array of names
        // until #677 — the accessor saw a variable, not a literal, so a
        // wired option sat in the silent list. The probe keeps the
        // literal-key form from quietly regressing.
        "timing_statistics",
        // Read by `derivative_test_options`, whose numeric helper is
        // named `read_num` precisely so this scan can see it.
        "derivative_test_tol",
    ] {
        assert!(
            read.contains(probe),
            "`{probe}` has a read site in the source but the scan missed it — \
             the scan is wrong, so its verdict below cannot be trusted",
        );
    }
    for probe in ["hessian_constant", "limited_memory_max_skipping"] {
        assert!(
            declared.contains(probe),
            "`{probe}` is declared in `unimplemented_options.rs` (refused or \
             warned) but the declaration scan missed it — the scan is wrong, \
             so its verdict below cannot be trusted",
        );
    }

    let silent: Vec<&String> = registered
        .iter()
        .filter(|n| !read.contains(*n) && !declared.contains(*n))
        .collect();

    // The remainder of #551: registered, never read, and not declared.
    // Each is a silent no-op — it accepts a value, does nothing, and says
    // nothing. They are listed by name so this test states the debt
    // precisely instead of asserting a bare count, and so removing one
    // from the list is the visible act of fixing it.
    //
    // TO FIX ONE: give it a read site that reaches the algorithm AND a
    // test proving the option changes behaviour (#551: "a read site that
    // parses a value and discards it is the same silent no-op this whole
    // line of work exists to kill"), or add it to
    // `unimplemented_options.rs` with a message saying what is missing.
    // Then delete it from here. Never delete an entry without doing one
    // of those two things — this list is the debt, not the fix.
    // 111 per-backend knobs for linear solvers pounce does not implement
    // (pounce ships feral and MA57). These need a POLICY decision before a
    // fix, not a read site: refusing them would break the stated goal that
    // an `ipopt.opt` written for Ipopt parses unchanged, since such a file
    // may configure several backends. A warning on first use is the likely
    // answer. #551 section 2.
    const BACKEND_KNOBS: &[&str] = &[
        "ma27_ignore_singularity",
        "ma27_la_init_factor",
        "ma27_liw_init_factor",
        "ma27_meminc_factor",
        "ma27_pivtol",
        "ma27_pivtolmax",
        "ma27_print_level",
        "ma27_skip_inertia_check",
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
        "ma86_nemin",
        "ma86_order",
        "ma86_print_level",
        "ma86_scaling",
        "ma86_small",
        "ma86_static",
        "ma86_u",
        "ma86_umax",
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
        "mumps_dep_tol",
        "mumps_mem_percent",
        "mumps_mpi_communicator",
        "mumps_permuting_scaling",
        "mumps_pivot_order",
        "mumps_pivtol",
        "mumps_pivtolmax",
        "mumps_print_level",
        "mumps_scaling",
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
        "pardisomkl_matching_strategy",
        "pardisomkl_max_iterative_refinement_steps",
        "pardisomkl_msglvl",
        "pardisomkl_order",
        "pardisomkl_redo_symbolic_fact_only_if_inertia_wrong",
        "pardisomkl_repeated_perturbation_means_singular",
        "pardisomkl_skip_inertia_check",
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
    ];

    // #551 section 1 — feature runs, read site missing.
    const LINE_SEARCH: &[&str] = &[
        "accept_after_max_steps",
        "alpha_for_y_tol",
        "alpha_red_factor",
        "delta",
        "eta_penalty",
        "filter_margin_fact",
        "filter_max_margin",
        "nu_inc",
        "nu_init",
        "rho",
        "theta_min",
    ];

    // #551 section 1 — feature runs, read site missing.
    const CORRECTOR: &[&str] = &[
        "corrector_compl_avrg_red_fact",
        "corrector_type",
        "skip_corr_if_neg_curv",
        "skip_corr_in_monotone_mode",
    ];

    // #551 section 1 — feature runs, read site missing.
    const BARRIER_KKT: &[&str] = &[
        "fixed_mu_oracle",
        "neg_curv_test_reg",
        "neg_curv_test_tol",
        "s_max",
        "tau_min",
    ];

    // #551 section 1 — feature runs, read site missing.
    const RESTORATION: &[&str] = &[
        "expect_infeasible_problem_ctol",
        "expect_infeasible_problem_ytol",
        "limited_memory_special_for_resto",
        "max_resto_iter",
        "resto_failure_feasibility_threshold",
    ];

    // #551 section 1 — feature runs, read site missing.
    const SENSITIVITY: &[&str] = &[
        "compute_red_hessian",
        "n_sens_steps",
        "rh_eigendecomp",
        "run_sens",
        "sens_bound_eps",
        "sens_boundcheck",
        "sens_max_pdpert",
    ];

    let known_debt: BTreeSet<&str> = BACKEND_KNOBS
        .iter()
        .chain(LINE_SEARCH)
        .chain(CORRECTOR)
        .chain(BARRIER_KKT)
        .chain(RESTORATION)
        .chain(SENSITIVITY)
        .copied()
        .collect();

    let unexpected: Vec<&&String> = silent
        .iter()
        .filter(|n| !known_debt.contains(n.as_str()))
        .collect();
    assert!(
        unexpected.is_empty(),
        "these options are registered but neither read nor declared unimplemented, \
         so setting them does nothing and says nothing:\n{}\n\n\
         Give each a read site that reaches the algorithm plus a test proving it \
         changes behaviour, or declare it in `unimplemented_options.rs`. \
         Do NOT add it to `known_debt` to make this pass — that list is being \
         emptied, not extended.",
        unexpected
            .iter()
            .map(|n| format!("  - {n}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // The debt list must also not go stale in the other direction: an
    // option that has since been wired should be removed from it, so the
    // list keeps meaning "still silent" rather than decaying into
    // decoration.
    let fixed: Vec<&&str> = known_debt
        .iter()
        .filter(|n| read.contains(**n) || declared.contains(**n))
        .collect();
    assert!(
        fixed.is_empty(),
        "these options are listed as known silent no-ops but now have a read site \
         or a refusal — delete them from `known_debt`:\n{}",
        fixed
            .iter()
            .map(|n| format!("  - {n}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
