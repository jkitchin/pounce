//! The sentinel: the convex arm's answers, held against optima POUNCE did
//! not produce.
//!
//! Every other assertion about the `bound_relax_factor` widening compares one
//! POUNCE number against another POUNCE number, or against `ipopt_ma57.json`.
//! That is how the widening survived: Ipopt carries the same device, so
//! measuring against it could not see the error, and the one artifact that did
//! use independent ground truth — `benchmarks/qp_four_way.md`, scored on
//! Maros-Mészáros DOC 97/6 — was not what CI looked at. It had the answer:
//! the unrelaxed convex arm at **137/138 correct, 0 solved-but-wrong**, with
//! `LISWET1(re=2.5e-01)` listed among Ipopt-MA57's *wrong* objectives.
//!
//! So this file holds two numbers from outside this repository's solver
//! family, and it is written to go RED if the widening is ever made the
//! convex default again:
//!
//! * `issue745_netlib_problem.nl` — optimum **`0`**. HiGHS returns `0.0`
//!   exactly from both its simplex and its interior-point solver, on this
//!   fixture's `.nl` reconstructed as an LP and checked against the `.nl`
//!   evaluator at a random point. It is 46 variables and 12 rows, and a
//!   `1e-8` widening moves its objective by the ENTIRE answer, to `-1.6`:
//!   some bound here has a multiplier of order `1.6e8`, and the widening's
//!   error is `delta` times that multiplier, which nothing bounds.
//! * `convex_qp_qscfxm1.nl` — QSCFXM1, optimum **`1.68826916e+07`** (DOC 97/6;
//!   HiGHS agrees). Here the widening costs accuracy *and* iterations:
//!   `9.1e-09` on 131 iterations widened, against `1.8e-13` on 30 declared.
//!
//! Mutation check. Set the widening back on as the default in
//! `convex_bound_relax` and `the_declared_optimum_is_returned` goes red on
//! the first fixture by a factor of `1e8` in relative terms — not a
//! borderline miss. `the_widening_is_what_moves_it` is the other half: it
//! pins WHY, so a reader who finds this test red learns the mechanism rather
//! than just the number.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// HiGHS, simplex and interior-point, both exactly `0.0`.
const NETLIB_PROBLEM_OPT: f64 = 0.0;
/// Maros-Mészáros DOC 97/6.
const QSCFXM1_OPT: f64 = 1.68826916e+07;
/// Opting back in to the widening.
const RELAX: &str = "bound_relax_factor=1e-8";

fn objective(fixture: &str, opts: &[&str]) -> f64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "pounce_declared_sentinel_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");

    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir).arg("m.nl").arg("--no-sol");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The convex arm must be the one that took the model. It may hand off to
    // the NLP arm afterwards (gh #535); what would invalidate a sentinel is
    // the classifier never sending it here at all.
    assert!(
        stdout.contains("pounce-convex"),
        "{fixture} no longer routes to the convex arm, so this sentinel is \
         measuring something else:\n{stdout}"
    );
    stdout
        .lines()
        .find(|l| l.starts_with("Objective..."))
        .unwrap_or_else(|| panic!("no objective line in:\n{stdout}"))
        .split_whitespace()
        .nth(1)
        .expect("objective value")
        .parse()
        .expect("parse objective")
}

fn report(fixture: &str, opts: &[&str]) -> pounce_cli::solve_report::SolveReport {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "pounce_declared_sentinel_rep_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let json = dir.join("r.json");
    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir)
        .arg("m.nl")
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let _ = cmd.output().expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("read json report");
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

#[test]
fn the_declared_optimum_is_returned() {
    let p = objective("issue745_netlib_problem.nl", &[]);
    assert!(
        (p - NETLIB_PROBLEM_OPT).abs() < 1e-6,
        "issue745_netlib_problem: got {p:e}, want {NETLIB_PROBLEM_OPT:e} — the \
         optimum HiGHS returns from both its simplex and its interior-point \
         solver. If this reads about -1.6, the bound_relax_factor widening is \
         back on by default and the answer is the model Ipopt solves, not the \
         model the caller wrote."
    );

    let q = objective("convex_qp_qscfxm1.nl", &[]);
    let rel = (q - QSCFXM1_OPT).abs() / QSCFXM1_OPT.abs();
    assert!(
        rel < 1e-8,
        "convex_qp_qscfxm1: got {q:e}, want {QSCFXM1_OPT:e} (rel {rel:e}) — \
         the Maros-Meszaros DOC 97/6 optimum. Nobody in this repository chose \
         this number, which is what makes it a sentinel."
    );
}

/// The mechanism, pinned beside the number: it is the widening that moves
/// the answer, and by how much.
///
/// This half is deliberately an assertion that the widened answer is WRONG.
/// If a future change makes the widening harmless here, this test fails and
/// the reasoning above needs re-examining rather than the constant needing
/// updating.
#[test]
fn the_widening_is_what_moves_it() {
    let declared = objective("issue745_netlib_problem.nl", &[]);
    let widened = objective("issue745_netlib_problem.nl", &[RELAX]);
    assert!(
        (widened + 1.6).abs() < 1e-3,
        "the widened solve should land near -1.6 on this fixture; got \
         {widened:e}"
    );
    assert!(
        (widened - declared).abs() > 1.0,
        "a 1e-8 widening should move this 46-variable LP's objective by the \
         whole answer: declared {declared:e} vs widened {widened:e}"
    );
}

/// The declared model does not have to be slower to be right — and the route
/// that rescues a declining convex solve has to deliver the declared model,
/// not merely an answer.
///
/// `scaled_feasible_a.nl` is where the widening was doing its only real work
/// on this arm, and the work was not subtle: that model's rows carry `|b|` up
/// to `2.65e13`, and the row width is `min(factor, cap)·|b|` — relative, by
/// deliberate choice (gh #385, `orig_ipopt_nlp.rs::relax_bounds`) — so a
/// `1e-8` factor relaxes one row by **2.65e5 in absolute terms**. Converging
/// in 69 iterations on that is not a solver being clever; it is a solver
/// being handed a much easier problem.
///
/// On the model as declared the convex driver needs ~3596 iterations (20
/// orders of Jacobian spread) and says so — it emits the gh #293 scaling
/// warning and returns `IterationLimit`. gh #535's fallback, extended to
/// convex QP, hands it to the general path, which certifies it in ~20.
///
/// **The assertion is on `final_declared_constr_viol`, not the objective.**
/// An objective check cannot tell "rerouted" from "relaxed" — the widened
/// convex answer on shipped `main` also lands within `1e-6` of `0` here. Only
/// the declared-model measurement separates them, and before the fallback was
/// taught to pass `bound_relax_factor=0` it read **2679.85** on a solve
/// reporting `Solve_Succeeded`.
#[test]
fn a_declining_convex_solve_is_rerouted_rather_than_relaxed() {
    let r = report("scaled_feasible_a.nl", &[]);
    let obj = r.solution.objective;
    assert!(
        obj.abs() < 1e-6,
        "scaled_feasible_a should reach its optimum 0 at the default routing; \
         got {obj:e}"
    );
    // The point of the name: nothing was widened to get there.
    let d = r.statistics.final_declared_constr_viol;
    assert!(
        d.is_nan() || d < 1e-6,
        "the rerouted solve must answer the DECLARED model — \
         final_declared_constr_viol should be ~0 (no widening applied), got \
         {d:e}. If this reads about 2679, the fallback is handing the model \
         to the arm that still widens, and the route named \"rather than \
         relaxed\" is doing exactly that."
    );
    assert!(
        r.statistics.final_constr_viol < 1e-6,
        "and the solve it did produce must be feasible: {:e}",
        r.statistics.final_constr_viol
    );
}

/// Asking for the widening still reaches the rerouted solve, so the escape
/// hatch is not silently disabled on this path either.
#[test]
fn the_reroute_still_honours_an_explicit_widening() {
    let r = report("scaled_feasible_a.nl", &["bound_relax_factor=1e-8"]);
    let d = r.statistics.final_declared_constr_viol;
    assert!(
        d.is_finite() && d > 1e-6,
        "an explicit bound_relax_factor should still buy Ipopt's model on the \
         rerouted path, and be reported as a declared-model violation; got \
         {d:e}"
    );
}
