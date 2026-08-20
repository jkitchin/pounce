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
    let declared = names_declared_unimplemented(&unimpl_src);

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
        // until #551 — the accessor saw a variable, not a literal, so a
        // wired option sat in the silent list. Same story for the four
        // constant-derivative hints, which gh#588 Q6 made pounce
        // *exploit*, and for the two `derivative_test_*` knobs, whose
        // local helper was named `num` rather than `read_num`. These
        // probes keep the literal-key form from quietly regressing and
        // handing any of them its silence back.
        "timing_statistics",
        "derivative_test_perturbation",
        "derivative_test_tol",
        "grad_f_constant",
        "hessian_constant",
        "jac_c_constant",
        "jac_d_constant",
        // The sIPOPT keys, read in `pounce-sensitivity/src/options.rs`
        // through helpers deliberately named `read_yes` / `read_num`
        // with `key` first, so this scan can find them. That crate is
        // the furthest read site from here, which makes it the easiest
        // one to lose in a refactor.
        "run_sens",
        "compute_red_hessian",
        "sens_max_pdpert",
    ] {
        assert!(
            read.contains(probe),
            "`{probe}` has a read site in the source but the scan missed it — \
             the scan is wrong, so its verdict below cannot be trusted",
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
    // The 111 per-backend knobs — `ma27_*` through `wsmp_*`, plus
    // `pardisolib` — used to be listed here. They are now declared in
    // `unimplemented_options.rs` (`UNIMPLEMENTED_BACKENDS`), which is
    // why they no longer appear: setting one warns, naming the backend,
    // and solves. Warning rather than refusing was the policy call
    // #551 section 2 asked for — a portable `ipopt.opt` configures
    // several backends at once, so refusing would fail a file the
    // registry exists to accept. See that module's header. #551.

    // The line-search group is empty: every option that was in it is
    // now either read (`alpha_red_factor` on #678, then
    // `accept_after_max_steps`, `delta`, the four penalty-acceptor
    // knobs and the two adaptive-filter margin knobs here) or declared
    // unimplemented (`theta_min`, `alpha_for_y_tol`), so the group is
    // gone rather than kept as an empty decoration.

    // The barrier / KKT group is empty: `tau_min`, `s_max`,
    // `neg_curv_test_tol` and `neg_curv_test_reg` now have read sites,
    // and `fixed_mu_oracle` is refused (#551 / #677).

    // The corrector group is empty: `corrector_type` and its three
    // safeguards select `FilterLSAcceptor::TryCorrector`, which pounce
    // does not have, and are refused (#551).

    // The restoration group is empty: `max_resto_iter` has a read site
    // and the other four are refused (#551).

    // The sIPOPT group is empty: six of the seven now reach
    // `pounce_sensitivity::SensOptionOverrides`, and `n_sens_steps` is
    // refused above its default because pounce computes the single
    // `sens_state_1` perturbation tier (#551 / #677).

    // The NLP-hint and misc groups are empty, and both were scan false
    // positives rather than unwired options — which is worth stating,
    // because the fix was to the read site's *shape*, not to the
    // algorithm. All seven were wired and consumed; each reached its
    // accessor through a loop variable or a differently-named local
    // helper, so the scan saw no literal key and reported them silent.
    // The probes above keep the literal-key form from regressing.

    // The 111 per-backend knobs — `ma27_*` through `wsmp_*`, plus
    // `pardisolib` — used to be listed here. They are now declared in
    // `unimplemented_options.rs` (`UNIMPLEMENTED_BACKENDS`), which is
    // why they no longer appear: setting one warns, naming the backend,
    // and solves. Warning rather than refusing was the policy call
    // #551 section 2 asked for — a portable `ipopt.opt` configures
    // several backends at once, so refusing would fail a file the
    // registry exists to accept. See that module's header. #551.
    //
    // WITH THAT, THE DEBT IS EMPTY (#682). The set stays here, and
    // empty, rather than being deleted along with its last entry: the
    // assertion below is what tells the next person that adding a name
    // to it is not how you make this test pass.
    let known_debt: BTreeSet<&str> = BTreeSet::new();

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
