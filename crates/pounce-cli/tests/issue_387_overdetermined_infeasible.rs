//! gh #387 — the CLI must report the AMPL infeasible band (200..299) for a
//! provably contradictory over-determined system, not the 5xx failure band.
//!
//! `x == 0.2` with `x == 0.8` over `x in [0, 1]` trips the too-few-degrees-
//! of-freedom gate (1 variable, 2 equality rows) before any iteration runs,
//! so the solve itself can never discover the contradiction; the DOF path now
//! consults presolve's bound-propagation certification first.
//!
//! This test goes through the full CLI stack on purpose: the CLI wraps the
//! `.nl` TNLP in `CountingTnlp`, which used to swallow
//! `get_constraints_linearity` (trait default `false`), silently disabling
//! linear-row propagation for anything stacked above it. An algorithm-level
//! test cannot see that hole — this one does.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// The Pyomo `inf_eq` model from `test_scale_invariance.py`, row-scaled by
/// `s`: `min x^2  s.t.  s*x == 0.2*s,  s*x == 0.8*s,  x in [0, 1]`.
fn inf_eq_nl(s: f64) -> String {
    format!(
        "g3 1 1 0\n \
         1 2 1 0 2\n \
         0 1 0 0 0 0\n \
         0 0\n \
         0 1 0\n \
         0 0 0 1\n \
         0 0 0 0 0\n \
         2 1\n \
         2 1\n \
         0 0 0 0 0\n\
         C0\nn0\n\
         C1\nn0\n\
         O0 0\no5\nv0\nn2\n\
         x1\n0 0.5\n\
         r\n4 {r1:e}\n4 {r2:e}\n\
         b\n0 0 1\n\
         k0\n\
         J0 1\n0 {s:e}\n\
         J1 1\n0 {s:e}\n\
         G0 1\n0 0\n",
        r1 = 0.2 * s,
        r2 = 0.8 * s,
        s = s,
    )
}

fn solve_result_num(text: &str) -> i32 {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("objno ") {
            if let Some(code) = rest.split_whitespace().nth(1) {
                return code.parse().expect("objno code parses");
            }
        }
    }
    panic!("no `objno` line in .sol:\n{text}");
}

/// Write `nl_text` to a temp file, solve it under `-AMPL` with default
/// options, and return the `.sol` text.
fn solve(tag: &str, nl_text: &str) -> String {
    solve_with(tag, nl_text, false)
}

/// `solve`, optionally with `presolve=yes` — the leg that reaches Phase 2's
/// redundant-row removal, which dropped both rows at `s = 1e-12` under an
/// absolute tolerance and reported the model solved (gh #1197).
fn solve_with(tag: &str, nl_text: &str, presolve: bool) -> String {
    let dir = std::env::temp_dir();
    let nl = dir.join(format!("pounce_issue387_{tag}.nl"));
    let sol = dir.join(format!("pounce_issue387_{tag}.sol"));
    std::fs::write(&nl, nl_text).expect("write .nl");
    let _ = std::fs::remove_file(&sol);

    // `solver_selection=nlp` pins the IPM route the issue was filed against
    // (and the scale-invariance harness runs); the default auto-route sends
    // this convex-QP-class model to `pounce-convex`, which detects the
    // infeasibility on its own.
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=nlp")
        .arg("print_level=0")
        .arg(if presolve {
            "presolve=yes"
        } else {
            "presolve=no"
        })
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");

    std::fs::read_to_string(&sol).expect("read .sol")
}

/// The headline regression: with all-default options (presolve off), the
/// contradiction must land in the AMPL infeasible band.
#[test]
fn contradictory_equalities_report_infeasible_band() {
    let text = solve("unit", &inf_eq_nl(1.0));
    let srn = solve_result_num(&text);
    assert!(
        (200..300).contains(&srn),
        "expected the AMPL infeasible band (200..299) for `x == 0.2, \
         x == 0.8`, got solve_result_num={srn}. The 5xx failure band \
         surfaces through Pyomo as an internal solver error, \
         indistinguishable from a solver bug:\n{text}"
    );
    assert!(
        text.contains("InfeasibleProblemDetected"),
        "expected the InfeasibleProblemDetected verdict:\n{text}"
    );
}

/// Multiplying every row by `s > 0` leaves the feasible set unchanged, so the
/// verdict must agree at every scale — including the sub-`1e-8` scalings this
/// sweep used to skip. Those were excluded because the witness gate's absolute
/// floor withdrew the proof there; gh#391 removed the floor on this path, where
/// the solve provably cannot run and so cannot disagree.
#[test]
fn verdict_is_scale_invariant_where_certifiable() {
    for presolve in [false, true] {
        for k in [-12, -9, -6, -3, 0, 3, 6, 9, 12] {
            let text = solve_with(
                &format!("k{k}_p{presolve}"),
                &inf_eq_nl(10.0_f64.powi(k)),
                presolve,
            );
            let srn = solve_result_num(&text);
            assert!(
                (200..300).contains(&srn),
                "row scale 1e{k}, presolve={presolve}: same empty feasible set, \
                 but solve_result_num={srn} left the infeasible band:\n{text}"
            );
        }
    }
}
