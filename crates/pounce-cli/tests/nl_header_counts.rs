//! The `.nl` header's nonlinearity census, checked against the trees it
//! describes, over every `.nl` fixture in the repository (gh #588, phase Q2).
//!
//! [`NlCounts`](pounce_cli::nl_reader::NlCounts) records what the *writer*
//! claimed about the model. The design note for #588 proposed reading routing
//! decisions straight off it — "`classify_problem`'s LP fast path becomes an
//! O(1) header read". These tests are why that is not what shipped, and they
//! are the durable record of the measurement:
//!
//! | claim | holds on |
//! |---|---|
//! | the header carries a parseable census | 58 of 61 fixtures |
//! | variables that appear nonlinearly are inside the `nlvc + nlvo − nlvb` prefix | **58 of 58** |
//! | rows that are nonlinear are inside the `nlc` prefix | 55 of 58 |
//!
//! The row claim fails in both directions, and the two directions cost
//! different things:
//!
//! * **over-statement** (`lp_row_constant.nl`: `nlc = 1`, no row still
//!   nonlinear) is benign but not free — `parse_nl_text` folds a
//!   variable-free `C` body into the row bounds *after* AMPL took its census
//!   (`gh #492`), so a header-driven classifier would send a plain LP to the
//!   NLP path and undo that fix;
//! * **under-statement** (`parametric.nl`: `nlc = 0`, yet `C1` is
//!   `o2 v4 v0`) is a wrong answer — a classifier that skipped rows past
//!   `nlc` would see a convex quadratic objective over linear rows, route to
//!   the convex QP solver, and drop a bilinear constraint on the floor.
//!
//! So the classifier keeps walking the parsed rows, and the census is used
//! only where over-statement is the safe direction: the `POUNCE_DBG_CLASSIFY`
//! log, and `NlTnlp::get_number_of_nonlinear_variables`'s all-variables-are-
//! nonlinear shortcut.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use pounce_cli::nl_reader::{NlProblem, read_nl_file};

/// Fixtures whose header line 5 (`nlvc nlvo nlvb`) carries fewer than the
/// three documented fields, so no census is recorded. All three are
/// hand-written `.nl` text; AMPL emits all three fields.
const NO_CENSUS: &[&str] = &["convex_qp.nl", "infeasible_qp.nl", "nonconvex_qp.nl"];

/// Fixtures whose `nlc` does not bound the nonlinear rows, with what is
/// wrong. Pinned in both directions: a fixture here that starts conforming
/// fails the test just as loudly as a new violation, so the list cannot rot
/// into a mute allowlist.
const NONCONFORMING_ROWS: &[(&str, &str)] = &[
    (
        "linear_eq_aggregation.nl",
        "nlc=1, but the nonlinear row is row 1",
    ),
    (
        "linear_eq_aggregation_row_constant.nl",
        "nlc=1, but the nonlinear row is row 1",
    ),
    ("parametric.nl", "nlc=0, but row 1's body is `o2 v4 v0`"),
    (
        // `parametric.nl` byte for byte plus a `red_hessian` suffix
        // block, so it carries the same understated header rather than
        // a new finding of its own (gh#551 added it for the sIPOPT
        // option tests).
        "parametric_red_hessian.nl",
        "nlc=0, but row 1's body is `o2 v4 v0` — inherited from parametric.nl",
    ),
];

fn fixture_roots() -> Vec<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    vec![base.join("fixtures"), base.join("fixtures_issue_49")]
}

/// Every `.nl` under the fixture roots, recursively, sorted so a failure
/// message is stable.
fn all_fixtures() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "nl") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    for root in fixture_roots() {
        walk(&root, &mut out);
    }
    out.sort();
    out
}

fn base_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into()
}

/// Rows whose nonlinear part survived parsing.
fn nonlinear_rows(prob: &NlProblem) -> Vec<usize> {
    prob.con_nonlinear
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.is_trivially_zero())
        .map(|(i, _)| i)
        .collect()
}

/// Variables appearing in some nonlinear part — the same set
/// `NlTnlp::get_variables_linearity` publishes.
fn walked_nonlinear_vars(prob: &NlProblem) -> BTreeSet<usize> {
    let mut s = BTreeSet::new();
    prob.obj_nonlinear.collect_vars(&mut s);
    for row in &prob.con_nonlinear {
        row.collect_vars(&mut s);
    }
    s
}

#[test]
fn header_census_parses_for_every_conforming_fixture() {
    let fixtures = all_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected the fixture corpus, found {} files",
        fixtures.len()
    );
    for f in &fixtures {
        let Ok(prob) = read_nl_file(f) else { continue };
        let name = base_name(f);
        let expected_missing = NO_CENSUS.contains(&name.as_str());
        assert_eq!(
            prob.nl_counts.is_none(),
            expected_missing,
            "{}: census present = {}, expected missing = {expected_missing} \
             (NO_CENSUS is pinned in both directions)",
            f.display(),
            prob.nl_counts.is_some()
        );
    }
}

/// The property both shipped consumers rest on: every variable pounce will
/// differentiate lies inside the header's nonlinear-variable prefix.
#[test]
fn nonlinear_variables_are_inside_the_header_prefix() {
    let mut checked = 0;
    for f in all_fixtures() {
        let Ok(prob) = read_nl_file(&f) else { continue };
        let Some(c) = prob.nl_counts else { continue };
        checked += 1;
        let walked = walked_nonlinear_vars(&prob);
        let prefix = c.nonlinear_vars();
        if let Some(&worst) = walked.iter().next_back() {
            assert!(
                worst < prefix,
                "{}: variable {worst} appears nonlinearly but the header's \
                 prefix is only {prefix} wide (nlvc={} nlvo={} nlvb={}, n={})",
                f.display(),
                c.nl_vars_cons,
                c.nl_vars_objs,
                c.nl_vars_both,
                prob.n
            );
        }
    }
    assert!(checked >= 50, "only {checked} fixtures carried a census");
}

/// `nlvb` is counted inside both `nlvc` and `nlvo`, so the total is an
/// inclusion–exclusion. `eigena2` is the worked example: 100 variables
/// nonlinear in constraints, 110 in the objective, 100 in both — 110
/// distinct, which is every variable it has.
#[test]
fn overlapping_counts_are_not_summed() {
    let f = fixture_roots()[0].join("eigena2.nl");
    let prob = read_nl_file(&f).expect("read eigena2");
    let c = prob.nl_counts.expect("eigena2 census");
    assert_eq!(
        (c.nl_vars_cons, c.nl_vars_objs, c.nl_vars_both),
        (100, 110, 100)
    );
    assert_eq!(c.nonlinear_vars(), 110);
    assert_eq!(prob.n, 110);
    assert_eq!((c.nl_cons, c.nl_objs), (55, 1));
}

/// The `nlc` prefix claim, and the fixtures that break it. See the module
/// docs for why this is a recorded finding rather than a fixed bug: pounce
/// does not own these headers, and the classifier does not read them.
#[test]
fn nonlinear_rows_are_inside_the_nlc_prefix_except_where_pinned() {
    for f in all_fixtures() {
        let Ok(prob) = read_nl_file(&f) else { continue };
        let Some(c) = prob.nl_counts else { continue };
        let name = base_name(&f);
        let pinned = NONCONFORMING_ROWS.iter().find(|(n, _)| *n == name);
        let rows = nonlinear_rows(&prob);
        let conforms = rows.iter().all(|&i| i < c.nl_cons);
        match pinned {
            None => assert!(
                conforms,
                "{}: header nlc={} but nonlinear rows are {rows:?} — a new \
                 non-conforming header; nothing in pounce may route on `nlc`",
                f.display(),
                c.nl_cons
            ),
            Some((_, why)) => assert!(
                !conforms,
                "{}: pinned as non-conforming ({why}) but now conforms — \
                 drop it from NONCONFORMING_ROWS",
                f.display()
            ),
        }
    }
}

/// `parametric.nl` is the counterexample that decided the phase: its header
/// says there are no nonlinear constraints, and there is one. A classifier
/// that believed it would see a convex quadratic objective over linear rows
/// and route to the convex QP solver — silently dropping `x₄·x₀`.
#[test]
fn parametric_header_understates_and_the_classifier_ignores_it() {
    let f = fixture_roots()[0].join("parametric.nl");
    let prob = read_nl_file(&f).expect("read parametric");
    let c = prob.nl_counts.expect("parametric census");
    assert_eq!(c.nl_cons, 0, "header claims no nonlinear constraints");
    assert_eq!(
        nonlinear_rows(&prob),
        vec![1],
        "row 1 is nonlinear regardless"
    );
    assert_eq!(
        pounce_cli::dispatch::classify_problem(&prob),
        pounce_cli::dispatch::ProblemClass::Nlp,
        "the classifier must reach NLP from the trees, not LP/QP from the header"
    );
}
