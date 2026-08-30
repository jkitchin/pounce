//! The corpus's own instance of gh #848, at the CLI.
//!
//! `crates/pounce-cli/tests/fixtures/nonconvex_qp_ineq.nl` is
//! `min x₀x₁ s.t. x₀ + x₁ ≥ 2` over `[0, 4]²`. `P = [[0, 1], [1, 0]]` has
//! eigenvalues `[-1, 1]`, so the class is `NonconvexQp` and
//! `solver_selection=qp-active-set` is the only way to reach the active-set
//! engine with it — `auto` sends the class to the NLP filter-IPM.
//!
//! The engine settles on `(1, 1)` at `f = 1`, and that point is not merely
//! suboptimal, it is a **maximum along the active constraint**:
//! `f(1+t, 1−t) = 1 − t²`. `(0, 2)` is feasible at `f = 0`, and POUNCE's own
//! NLP arm returns `f ≈ 0` on the same file in the same binary. Before the fix
//! this came back `EXIT: Optimal Solution Found.` / `Solve_Succeeded`.
//!
//! This fixture was in the tree the whole time and no test looked at it through
//! this engine, which is why the defect shipped: `scripts/sweep-fixtures.sh`
//! runs at `solver_selection=auto`, and `auto` routes the class away from the
//! engine that had the defect. A leg is only evidence about the branch its
//! fixture reaches.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// `(stdout, stderr)` from a solve, with the `.sol` written to a temp path.
fn run(name: &str, extra: &[&str]) -> (String, String) {
    let sol = std::env::temp_dir().join(format!("pounce_848_{name}.sol"));
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn objective(stdout: &str) -> Option<f64> {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// The premise, from an engine that is not the one under test: the true
/// minimum is `0`, not `1`. `auto` routes this class to the NLP filter-IPM, so
/// the default invocation is already the independent oracle.
#[test]
fn the_nlp_arm_puts_the_optimum_at_zero() {
    let (out, _) = run("nonconvex_qp_ineq.nl", &["solver_selection=nlp"]);
    assert!(
        out.contains("EXIT: Optimal Solution Found."),
        "the NLP arm should solve this cleanly:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(
        obj.abs() < 1e-6,
        "the NLP arm should reach f = 0, got {obj:.6e}"
    );
}

/// The defect: the active-set engine must not certify `(1, 1)`.
#[test]
fn the_active_set_engine_does_not_certify_the_constrained_maximum() {
    let (out, err) = run("nonconvex_qp_ineq.nl", &["solver_selection=qp-active-set"]);
    let obj = objective(&out).expect("an objective line");
    let certified = out.contains("EXIT: Optimal Solution Found.");
    assert!(
        !(certified && obj > 1e-6),
        "certified f = {obj:.6e} as optimal, but f(1+t, 1-t) = 1 - t^2 makes \
         that point a maximum along the active constraint and (0, 2) is \
         feasible at f = 0:\n{out}"
    );
    // A refusal must explain itself: the shared console vocabulary renders
    // this status "INTERNAL ERROR: Unknown SolverReturn value.", which reads
    // like a crash for what is a deliberate and correct refusal.
    if !certified {
        assert!(
            err.contains("not a local minimum") && err.contains("solver_selection=nlp"),
            "the refusal should say what happened and where the answer is:\n{err}"
        );
    }
}

/// The sibling fixture, which the engine handles correctly, is unmoved. Without
/// this the fix could be "refuse every indefinite QP" and the test above would
/// still pass.
#[test]
fn the_unconstrained_nonconvex_fixture_is_unchanged() {
    let (out, _) = run("nonconvex_qp.nl", &["solver_selection=qp-active-set"]);
    assert!(
        out.contains("EXIT: Optimal Solution Found."),
        "this one was and remains solvable by the active-set engine:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(obj.abs() < 1e-6, "expected f ~ 0, got {obj:.6e}");
}
