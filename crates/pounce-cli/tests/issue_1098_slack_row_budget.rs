//! gh #1098 — a fixture that reaches the row-budgeted branch of the slack
//! perturbation (`perturb_slack_max_rows`).
//!
//! The rule `δ_s = perturb_slack_fact · min(1, perturb_slack_max_rows / m_d)
//! · δ_x` branches on the number of inequality rows. Below the cap it is
//! upstream's `δ_s = δ_x`; above it the slack-block perturbation decays as
//! `1/m_d`. **Every other fixture in this corpus takes the first branch**:
//! the largest has 351 inequality rows (`deb7`) against a default cap of
//! 1000, and the Water suite tops out at 84. So without this file the fixture
//! sweep diffs empty across a change to the second branch *by construction*
//! — which is the CLAUDE.md rule that a test is only evidence about the
//! branch it reaches, and the reason this fixture exists rather than what it
//! measures.
//!
//! `manyineq_concave_qp` is the qcqp1500-1nc shape in miniature: 30
//! variables, 6 linear equalities, 1100 linear inequalities (60 of them the
//! box, written as rows so they count), a concave quadratic objective so the
//! inertia correction is needed on every iteration, and a reference point
//! that keeps the polytope strictly feasible. Most rows are inactive at the
//! vertex it converges to, which is the regime the budget acts on. At 1100
//! rows the effective factor is 1000/1100 ≈ 0.91 — deliberately just over
//! the cap, so the file stays small (116 KB; `deb7` is 152 KB) while the
//! branch is live. Measured when it was added: 29 iterations with the cap
//! off, 28 with it on, exact leg; the L-BFGS leg never raises `δ_x` and is
//! unchanged at 20. Those counts are recorded in the sweep, not asserted
//! here.
//!
//! **Provenance.** Generated, not hand-built:
//!
//! ```sh
//! scripts/gen-manyineq-nl.py 30 6 1100 4 \
//!     crates/pounce-cli/tests/fixtures/manyineq_concave_qp.nl neg
//! ```
//!
//! What this file asserts: the fixture has more inequality rows than the
//! default cap (so it is on the budgeted branch), it is solved on the NLP arm
//! (a nonconvex objective cannot route to the convex engines), and both
//! branches — the default and `perturb_slack_max_rows=0`, which is upstream
//! exactly at the default `perturb_slack_fact=1` — reach the same optimum.
//! It pins no iteration count; the sweep measures those.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

const FIXTURE: &str = "manyineq_concave_qp";

/// From the `.nl` header: 30 vars, 1106 constraints of which 6 are
/// equalities, so 1100 inequality rows.
const N: usize = 30;
const M: usize = 1106;
const M_INEQ: usize = 1100;

/// Must match `ftb_floor::DEFAULT_PERTURB_SLACK_MAX_ROWS`; the fixture is
/// only on the budgeted branch while `M_INEQ` exceeds it.
const DEFAULT_CAP: usize = 1000;

/// The objective both branches converge to (same vertex), measured on the
/// exact leg when the fixture was added. Reported to 1e-8 relative; the
/// assertion below allows 1e-6 so a tolerance-level drift does not fail it.
const OPTIMUM: f64 = -10.877_958_56;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{FIXTURE}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh1098_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

fn solve(opts: &[&str]) -> (SolveReport, String) {
    let tag = opts.join("_").replace(['=', '.', '-'], "_");
    let json = tmp_path(&tag, "json");
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture())
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {FIXTURE} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    (
        serde_json::from_str(&text).expect("parse SolveReport JSON"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn assert_at_optimum(r: &SolveReport, what: &str, stdout: &str) {
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE} ({what}): must solve; stdout=\n{stdout}"
    );
    let rel = ((r.solution.objective - OPTIMUM) / OPTIMUM).abs();
    assert!(
        rel < 1e-6,
        "{FIXTURE} ({what}): objective {} is {rel:.2e} from {OPTIMUM}; \
         stdout=\n{stdout}",
        r.solution.objective
    );
}

/// The fixture is on the budgeted branch, on the NLP arm, at its optimum.
#[test]
fn the_fixture_reaches_the_budgeted_branch_on_the_nlp_arm() {
    let (r, stdout) = solve(&[]);
    assert_at_optimum(&r, "default", &stdout);
    assert_eq!(
        (
            r.problem.n_variables as usize,
            r.problem.n_constraints as usize
        ),
        (N, M),
        "{FIXTURE}: dimensions changed; stdout=\n{stdout}"
    );
    assert!(
        M_INEQ > DEFAULT_CAP,
        "{FIXTURE}: {M_INEQ} inequality rows no longer exceed the default cap \
         {DEFAULT_CAP}; the fixture would be on the upstream branch like every \
         other one"
    );
    assert_eq!(
        r.solution.engine, "nlp",
        "{FIXTURE}: must be solved by the NLP arm — the budget lives in its \
         inertia correction; stdout=\n{stdout}"
    );
}

/// Both branches reach the same optimum: the budget changes the trajectory,
/// not the answer. `perturb_slack_max_rows=0` with the default
/// `perturb_slack_fact=1` is upstream Ipopt's `δ_s = δ_x` exactly.
#[test]
fn both_branches_reach_the_same_optimum() {
    let (cap_on, out_on) = solve(&[]);
    let (cap_off, out_off) = solve(&["perturb_slack_max_rows=0"]);
    assert_at_optimum(&cap_on, "default cap", &out_on);
    assert_at_optimum(&cap_off, "perturb_slack_max_rows=0", &out_off);
    // A cap far above the row count is the upstream branch too.
    let (cap_high, out_high) = solve(&["perturb_slack_max_rows=100000"]);
    assert_at_optimum(&cap_high, "perturb_slack_max_rows=100000", &out_high);
    assert_eq!(
        cap_high.statistics.iteration_count, cap_off.statistics.iteration_count,
        "{FIXTURE}: a cap above m_d must be indistinguishable from no cap"
    );
}
