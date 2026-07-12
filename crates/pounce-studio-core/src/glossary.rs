//! Static glossary for solve-report column names, diagnose finding codes,
//! and citation metadata. Backs the `explain` and `citations` CLI tools.
//!
//! Ported from `studio/mcp/pounce_studio_mcp/glossary.py` so the
//! Rust-CLI / skill path returns the same definitions as the Python MCP
//! path. Entries are intentionally terse — Claude can elaborate at call
//! time; the job here is to surface canonical names, numeric ranges,
//! and the right paper keys.

use serde::Serialize;

/// One per-iter column entry returned by [`explain_column`].
#[derive(Debug, Clone, Serialize)]
pub struct ColumnEntry {
    pub definition: &'static str,
    pub typical_range: &'static str,
    pub what_abnormal_means: &'static str,
    pub see_also: &'static [&'static str],
}

/// One diagnose-finding entry returned by [`explain_finding`].
#[derive(Debug, Clone, Serialize)]
pub struct FindingEntry {
    pub severity: &'static str,
    pub meaning: &'static str,
    pub what_to_try: &'static str,
    pub see_also: &'static [&'static str],
}

/// One citation entry returned by [`citation_by_key`].
#[derive(Debug, Clone, Serialize)]
pub struct Citation {
    pub key: &'static str,
    pub entry_type: &'static str,
    pub title: &'static str,
    pub author: &'static str,
    pub year: &'static str,
    pub venue: &'static str,
    pub doi: &'static str,
}

/// Lookup result for [`explain`]: column entry, finding entry, or
/// fuzzy suggestions when the term is unknown.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Explanation {
    Column {
        term: String,
        #[serde(flatten)]
        entry: ColumnEntry,
    },
    Finding {
        term: String,
        #[serde(flatten)]
        entry: FindingEntry,
    },
    Unknown {
        term: String,
        suggestions: Vec<&'static str>,
        all_columns: Vec<&'static str>,
        all_findings: Vec<&'static str>,
    },
}

const COLUMNS: &[(&str, ColumnEntry)] = &[
    (
        "iter",
        ColumnEntry {
            definition: "Zero-based iteration index of the outer interior-point loop.",
            typical_range: "0 to a few hundred for well-scaled problems.",
            what_abnormal_means: "Hitting `max_iter` without converging usually points at scaling or degeneracy.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "objective",
        ColumnEntry {
            definition: "Current objective value f(x_k) at the iterate.",
            typical_range: "Problem-dependent. For well-scaled problems on the order of 1.",
            what_abnormal_means: "Wild swings (especially after restoration) can signal bad scaling.",
            see_also: &[],
        },
    ),
    (
        "inf_pr",
        ColumnEntry {
            definition: "Primal infeasibility: max-norm of constraint violation c(x_k).",
            typical_range: "Drops monotonically toward `tol` (default 1e-8) at convergence.",
            what_abnormal_means: "Stalling at large inf_pr → likely infeasible or restoration-stuck.",
            see_also: &["wachter2006", "byrd2010"],
        },
    ),
    (
        "inf_du",
        ColumnEntry {
            definition: "Dual infeasibility: max-norm of the gradient of the Lagrangian.",
            typical_range: "Drops toward `tol` alongside inf_pr; sometimes lags.",
            what_abnormal_means: "inf_du much larger than inf_pr → multipliers ill-conditioned or scaling bad.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "mu",
        ColumnEntry {
            definition: "Barrier parameter for the log-barrier homotopy.",
            typical_range: "Starts ~0.1, decreases toward 1e-9 as iterations progress.",
            what_abnormal_means: "Mu stuck at one value across many iterations → see finding `mu_stuck`.",
            see_also: &["wachter2006", "hinder2018"],
        },
    ),
    (
        "d_norm",
        ColumnEntry {
            definition: "Norm of the Newton search direction d_k.",
            typical_range: "Decreases as the iterate approaches the solution.",
            what_abnormal_means: "d_norm growing → search direction quality degrading; check regularization.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "regularization",
        ColumnEntry {
            definition: "Diagonal Hessian regularization δ added to make the KKT matrix have the correct inertia.",
            typical_range: "0 for convex problems; small positive values near saddle points.",
            what_abnormal_means: "Repeated large δ → Hessian is indefinite, problem is non-convex; finding `hessian_regularized` fires.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "alpha_dual",
        ColumnEntry {
            definition: "Step length applied to the dual variables (bound multipliers).",
            typical_range: "(0, 1]. Often 1.0 near the solution.",
            what_abnormal_means: "Persistently tiny alpha_dual → fraction-to-boundary biting; bounds may be active.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "alpha_primal",
        ColumnEntry {
            definition: "Step length applied to the primal variables x.",
            typical_range: "(0, 1]. Tiny values point at line-search difficulty.",
            what_abnormal_means: "Repeated tiny alpha_primal → finding `heavy_line_search` fires.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "alpha_primal_char",
        ColumnEntry {
            definition: "Single-character tag for what the line search did this iter: `f` filter-accepted, `h` Armijo, `r` restoration, `s` second-order correction, `R` restoration entry, `-` rejected.",
            typical_range: "Mostly `f` on a healthy solve.",
            what_abnormal_means: "Runs of `r` are restoration windows; consecutive `R` entries = `restoration_loop`.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "ls_trials",
        ColumnEntry {
            definition: "Number of line-search trials in this iteration.",
            typical_range: "1-3 for healthy solves.",
            what_abnormal_means: "Persistently high → curvature mismatch; consider regularization or restart.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "log10_mu",
        ColumnEntry {
            definition: "log10(mu); convenience for plotting the barrier homotopy.",
            typical_range: "Decreases from ~-1 to ~-9 over a typical solve.",
            what_abnormal_means: "Flat trace → mu_stuck.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "log10_inf_pr",
        ColumnEntry {
            definition: "log10(inf_pr); convenience for spotting stalls.",
            typical_range: "Monotone descent to ~-8.",
            what_abnormal_means: "Plateaus over many iters → `find_stalls` will flag the window.",
            see_also: &[],
        },
    ),
    (
        "log10_inf_du",
        ColumnEntry {
            definition: "log10(inf_du); convenience for spotting stalls.",
            typical_range: "Monotone descent to ~-8.",
            what_abnormal_means: "Plateaus → check Hessian regularization and step quality.",
            see_also: &[],
        },
    ),
    (
        "n_factors",
        ColumnEntry {
            definition: "Total successful symmetric factorisations across the solve.",
            typical_range: "≈ iteration_count for filter-line-search runs.",
            what_abnormal_means: "Much larger than iter count → repeated regularization retries.",
            see_also: &["n_pattern_reuse", "n_pattern_changes"],
        },
    ),
    (
        "n_pattern_reuse",
        ColumnEntry {
            definition: "Factors that reused the prior symbolic factorisation (sparsity pattern unchanged → cheap).",
            typical_range: "Should dominate n_factors after iter 1.",
            what_abnormal_means: "Low share → matrix structure changing per iter; analyse() runs repeatedly, hurting throughput.",
            see_also: &["n_pattern_changes"],
        },
    ),
    (
        "n_pattern_changes",
        ColumnEntry {
            definition: "Factors that required a fresh symbolic factorisation.",
            typical_range: "1 (the first factor) for a healthy solve.",
            what_abnormal_means: "> 1 → KKT structure shifting; check inertia-correction regularization policy or active-set churn.",
            see_also: &["n_pattern_reuse"],
        },
    ),
    (
        "max_fill_ratio",
        ColumnEntry {
            definition: "Max nnz(L) / nnz(A) observed across factors.",
            typical_range: "1–10 for well-ordered KKT systems.",
            what_abnormal_means: ">> 10 → AMD/METIS ordering struggled; expect memory + time spikes.",
            see_also: &["last_nnz_a", "last_nnz_l"],
        },
    ),
    (
        "min_abs_pivot",
        ColumnEntry {
            definition: "Smallest absolute pivot encountered during factorisation.",
            typical_range: "1e-8 .. 1e+6 depending on problem scaling.",
            what_abnormal_means: "Approaching working precision floor (~1e-16) → matrix near-singular; regularization is probably kicking in.",
            see_also: &["max_abs_pivot", "regularization"],
        },
    ),
    (
        "max_abs_pivot",
        ColumnEntry {
            definition: "Largest absolute pivot encountered during factorisation.",
            typical_range: "Within ~6 orders of magnitude of min_abs_pivot.",
            what_abnormal_means: "max/min >> 1e8 → catastrophic conditioning; consider nlp_scaling_method.",
            see_also: &["min_abs_pivot"],
        },
    ),
    (
        "last_inertia",
        ColumnEntry {
            definition: "(positive, negative, zero) eigenvalue counts of the final factorisation, from the LDLᵀ pivots.",
            typical_range: "(n, m, 0) at a converged primal-dual KKT system.",
            what_abnormal_means: "zero > 0 → singular; positive < n → indefinite, inertia correction failed.",
            see_also: &["regularization"],
        },
    ),
    (
        "last_nnz_a",
        ColumnEntry {
            definition: "nnz(A) at the final factorisation's input KKT matrix.",
            typical_range: "Problem-dependent.",
            what_abnormal_means: "n/a — informational.",
            see_also: &["last_nnz_l", "max_fill_ratio"],
        },
    ),
    (
        "last_nnz_l",
        ColumnEntry {
            definition: "nnz(L) at the final factorisation.",
            typical_range: "Problem-dependent.",
            what_abnormal_means: "n/a — informational; combine with last_nnz_a for fill.",
            see_also: &["last_nnz_a", "max_fill_ratio"],
        },
    ),
];

const FINDINGS: &[(&str, FindingEntry)] = &[
    (
        "converged",
        FindingEntry {
            severity: "info",
            meaning: "Solver reached the convergence tolerance on both primal and dual infeasibility.",
            what_to_try: "Nothing — this is the success path.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "max_iter_exceeded",
        FindingEntry {
            severity: "error",
            meaning: "Solver hit `max_iter` without satisfying tolerances.",
            what_to_try: "Inspect the convergence trace: is residual still decreasing (raise max_iter), stalled (loosen tol, improve scaling), or oscillating (regularize)?",
            see_also: &["wachter2006"],
        },
    ),
    (
        "restoration_used",
        FindingEntry {
            severity: "info",
            meaning: "Restoration phase was entered at least once during the solve.",
            what_to_try: "Often benign on hard problems. Check restoration-windows; a single short entry is fine, repeated entries suggest a deeper feasibility issue.",
            see_also: &["wachter2006", "byrd2010"],
        },
    ),
    (
        "mu_stuck",
        FindingEntry {
            severity: "warning",
            meaning: "Barrier parameter μ failed to decrease across a window of iterations. Usually a degenerate active set or a poorly-scaled barrier.",
            what_to_try: "Try `mu_strategy=adaptive`, tighten `bound_relax_factor`, or check that bound values are sensible (no infs masking effective bounds).",
            see_also: &["wachter2006", "hinder2018"],
        },
    ),
    (
        "heavy_line_search",
        FindingEntry {
            severity: "warning",
            meaning: "Line search needed many trials on average — search direction is low-quality.",
            what_to_try: "Often Hessian-related: enable second-order-correction, or investigate regularization values.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "hessian_regularized",
        FindingEntry {
            severity: "warning",
            meaning: "Hessian needed inertia-correction (added δ on the diagonal) frequently — the problem is non-convex or has near-singular Hessian.",
            what_to_try: "Consider tightening tolerances on `min_hessian_perturbation`, providing an analytic Hessian if you have one, or reformulating to convexify.",
            see_also: &["wachter2006"],
        },
    ),
    (
        "restoration_loop",
        FindingEntry {
            severity: "error",
            meaning: "Restoration phase was entered repeatedly and never exited cleanly.",
            what_to_try: "Strong signal of local infeasibility. Try a different start point, relax tight constraints, or run feasibility diagnostics.",
            see_also: &["wachter2006", "byrd2010", "leyffer2003"],
        },
    ),
    (
        "convergence_stall",
        FindingEntry {
            severity: "warning",
            meaning: "log10(inf_pr|inf_du) barely moved across a window — solver is grinding.",
            what_to_try: "Check `find-stalls` for the window, then `get-iterate` at its midpoint to inspect μ, alpha_primal, and regularization. Common cause: bad scaling.",
            see_also: &["wachter2006"],
        },
    ),
];

/// Topic → ordered list of citation keys, most-relevant first.
pub const TOPICS: &[(&str, &[&str])] = &[
    ("interior_point", &["wachter2006", "byrd1999", "hinder2018"]),
    ("filter_line_search", &["wachter2006"]),
    ("restoration", &["wachter2006", "byrd2010"]),
    ("regularization", &["wachter2006"]),
    ("trust_region", &["byrd2000", "waltz2006"]),
    ("inexact_step", &["curtis2010"]),
    ("infeasibility_detection", &["byrd2010", "leyffer2003"]),
    ("mu_strategy", &["wachter2006", "hinder2018"]),
    ("sensitivity", &["zavala2009"]),
    ("knitro", &["byrd2006"]),
];

/// Solve *feature* → ordered citation keys, for the `pounce --cite`
/// lister. Distinct from [`TOPICS`] (which is keyed by documentation
/// topic): these keys name things a *run* can actually do, so the CLI
/// can map "this solve used feature X" → "cite these papers".
///
/// `"core"` is the static set every pounce run should cite. The rest
/// are solve-aware extras; `"restoration"` is the only one wired in
/// v1 (the report schema records restoration activity). New entries
/// (adaptive-μ → `hinder2018`, etc.) slot in here as the report grows
/// a `features_used` block.
pub const SOLVE_FEATURES: &[(&str, &[&str])] = &[
    ("core", &["pounce2026", "wachter2006"]),
    // Restoration *entered* is often benign, so cite only the on-point
    // restoration/infeasibility-detection paper. `leyffer2003`
    // (penalty-IPM non-convergence) belongs to a restoration *loop* /
    // infeasibility signal, not a single restoration entry — kept out
    // of v1 until the report distinguishes the two.
    ("restoration", &["byrd2010"]),
];

const CITATIONS: &[Citation] = &[
    Citation {
        key: "pounce2026",
        entry_type: "software",
        title: "POUNCE: a pure-Rust port of the Ipopt interior-point NLP solver",
        author: "Kitchin, J. R.",
        year: "2026",
        venue: "Zenodo",
        doi: "10.5281/zenodo.20387011",
    },
    Citation {
        key: "wachter2006",
        entry_type: "article",
        title: "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming",
        author: "Wächter, A. and Biegler, L. T.",
        year: "2006",
        venue: "Mathematical Programming 106(1), 25–57",
        doi: "10.1007/s10107-004-0559-y",
    },
    Citation {
        key: "byrd1999",
        entry_type: "article",
        title: "An interior point algorithm for large-scale nonlinear programming",
        author: "Byrd, R. H., Hribar, M. E. and Nocedal, J.",
        year: "1999",
        venue: "SIAM Journal on Optimization 9(4), 877–900",
        doi: "10.1137/S1052623497325107",
    },
    Citation {
        key: "byrd2000",
        entry_type: "article",
        title: "A trust region method based on interior point techniques for nonlinear programming",
        author: "Byrd, R. H., Gilbert, J. C. and Nocedal, J.",
        year: "2000",
        venue: "Mathematical Programming 89(1), 149–185",
        doi: "10.1007/PL00011391",
    },
    Citation {
        key: "byrd2006",
        entry_type: "inbook",
        title: "Knitro: An integrated package for nonlinear optimization",
        author: "Byrd, R. H., Nocedal, J. and Waltz, R. A.",
        year: "2006",
        venue: "Large-Scale Nonlinear Optimization, Springer, 35–59",
        doi: "10.1007/0-387-30065-1_4",
    },
    Citation {
        key: "byrd2010",
        entry_type: "article",
        title: "Infeasibility detection and SQP methods for nonlinear optimization",
        author: "Byrd, R. H., Curtis, F. E. and Nocedal, J.",
        year: "2010",
        venue: "SIAM Journal on Optimization 20(5), 2281–2299",
        doi: "10.1137/080738222",
    },
    Citation {
        key: "waltz2006",
        entry_type: "article",
        title: "An interior algorithm for nonlinear optimization that combines line search and trust region steps",
        author: "Waltz, R. A., Morales, J. L., Nocedal, J. and Orban, D.",
        year: "2006",
        venue: "Mathematical Programming 107(3), 391–408",
        doi: "10.1007/s10107-004-0560-5",
    },
    Citation {
        key: "curtis2010",
        entry_type: "article",
        title: "An adaptive Gauss-Newton algorithm for training multilayer nonlinear filters that have embedded memory",
        author: "Curtis, F. E.",
        year: "2010",
        venue: "Mathematical Programming Computation 4(1), 27–62",
        doi: "10.1007/s12532-011-0033-9",
    },
    Citation {
        key: "leyffer2003",
        entry_type: "article",
        title: "Interior methods for mathematical programs with complementarity constraints",
        author: "Leyffer, S., López-Calva, G. and Nocedal, J.",
        year: "2003",
        venue: "Argonne National Laboratory technical report",
        doi: "",
    },
    Citation {
        key: "hinder2018",
        entry_type: "article",
        title: "One-phase: A new method for global linear and quadratic programming",
        author: "Hinder, O. and Ye, Y.",
        year: "2018",
        venue: "Optimization Online preprint",
        doi: "",
    },
    Citation {
        key: "zavala2009",
        entry_type: "article",
        title: "Real-time nonlinear optimization as a generalized equation",
        author: "Zavala, V. M. and Anitescu, M.",
        year: "2009",
        venue: "SIAM Journal on Control and Optimization 48(8), 5444–5467",
        doi: "10.1137/090762634",
    },
];

/// Look up a per-iter column entry by name.
pub fn column(name: &str) -> Option<&'static ColumnEntry> {
    COLUMNS.iter().find(|(k, _)| *k == name).map(|(_, v)| v)
}

/// Look up a diagnose-finding entry by code.
pub fn finding(code: &str) -> Option<&'static FindingEntry> {
    FINDINGS.iter().find(|(k, _)| *k == code).map(|(_, v)| v)
}

/// All known column names, in declaration order.
pub fn all_columns() -> Vec<&'static str> {
    COLUMNS.iter().map(|(k, _)| *k).collect()
}

/// All known finding codes, in declaration order.
pub fn all_findings() -> Vec<&'static str> {
    FINDINGS.iter().map(|(k, _)| *k).collect()
}

/// Look up a citation by bib key.
pub fn citation_by_key(key: &str) -> Option<&'static Citation> {
    CITATIONS.iter().find(|c| c.key == key)
}

/// Look up the list of citation keys associated with a topic.
pub fn topic_keys(topic: &str) -> Option<&'static [&'static str]> {
    TOPICS
        .iter()
        .find(|(t, _)| *t == topic)
        .map(|(_, keys)| *keys)
}

/// All known topics.
pub fn all_topics() -> Vec<&'static str> {
    TOPICS.iter().map(|(t, _)| *t).collect()
}

/// Citation keys for a solve *feature* (see [`SOLVE_FEATURES`]). Used
/// by the `pounce --cite` lister to turn "this run did X" into the
/// papers to cite.
pub fn solve_feature_keys(feature: &str) -> Option<&'static [&'static str]> {
    SOLVE_FEATURES
        .iter()
        .find(|(f, _)| *f == feature)
        .map(|(_, keys)| *keys)
}

/// All citations.
pub fn all_citations() -> &'static [Citation] {
    CITATIONS
}

/// Try to resolve `term` against the column and finding tables. Falls
/// back to fuzzy suggestions when neither matches.
pub fn explain(term: &str) -> Explanation {
    if let Some(entry) = column(term) {
        return Explanation::Column {
            term: term.to_string(),
            entry: entry.clone(),
        };
    }
    if let Some(entry) = finding(term) {
        return Explanation::Finding {
            term: term.to_string(),
            entry: entry.clone(),
        };
    }
    let mut pool = all_columns();
    pool.extend(all_findings());
    Explanation::Unknown {
        term: term.to_string(),
        suggestions: fuzzy_suggest(term, &pool, 3),
        all_columns: all_columns(),
        all_findings: all_findings(),
    }
}

fn fuzzy_suggest(term: &str, candidates: &[&'static str], limit: usize) -> Vec<&'static str> {
    let needle = term.to_ascii_lowercase();
    let mut scored: Vec<(u8, &'static str)> = Vec::new();
    for &c in candidates {
        let cl = c.to_ascii_lowercase();
        if cl == needle {
            scored.push((0, c));
        } else if cl.starts_with(&needle) || needle.starts_with(&cl) {
            scored.push((1, c));
        } else if cl.contains(&needle) || needle.contains(&cl) {
            scored.push((2, c));
        }
    }
    scored.sort_by_key(|(r, _)| *r);
    scored.into_iter().take(limit).map(|(_, c)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_lookup() {
        assert!(column("inf_pr").is_some());
        assert!(column("not_a_real_column").is_none());
    }

    #[test]
    fn finding_lookup() {
        let f = finding("mu_stuck").expect("mu_stuck should exist");
        assert_eq!(f.severity, "warning");
    }

    #[test]
    fn explain_finds_column() {
        let e = explain("inf_du");
        assert!(matches!(e, Explanation::Column { .. }));
    }

    #[test]
    fn explain_finds_finding() {
        let e = explain("restoration_loop");
        assert!(matches!(e, Explanation::Finding { .. }));
    }

    #[test]
    fn explain_fuzzy_unknown() {
        // "inf" is a prefix of several columns — should surface suggestions.
        let e = explain("inf");
        match e {
            Explanation::Unknown { suggestions, .. } => {
                assert!(
                    suggestions.iter().any(|s| s.starts_with("inf_")),
                    "got {suggestions:?}",
                );
            }
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn topic_lookup() {
        let keys = topic_keys("restoration").expect("restoration topic exists");
        assert!(keys.contains(&"wachter2006"));
    }

    #[test]
    fn citation_lookup() {
        let c = citation_by_key("wachter2006").expect("wachter2006 cited");
        assert!(c.title.contains("interior-point filter line-search"));
    }

    #[test]
    fn pounce_self_citation_present() {
        let c = citation_by_key("pounce2026").expect("pounce2026 cited");
        assert_eq!(c.doi, "10.5281/zenodo.20387011");
    }

    #[test]
    fn solve_feature_core_and_restoration() {
        let core = solve_feature_keys("core").expect("core feature exists");
        assert_eq!(core, &["pounce2026", "wachter2006"]);
        let resto = solve_feature_keys("restoration").expect("restoration feature exists");
        assert!(resto.contains(&"byrd2010"));
        assert!(solve_feature_keys("nonsense").is_none());
    }

    #[test]
    fn every_solve_feature_key_resolves() {
        for (_feature, keys) in SOLVE_FEATURES {
            for k in *keys {
                assert!(
                    citation_by_key(k).is_some(),
                    "SOLVE_FEATURES references unknown key {k:?}",
                );
            }
        }
    }
}
