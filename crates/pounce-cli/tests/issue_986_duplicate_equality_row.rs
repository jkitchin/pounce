//! gh #986 item 3: a convex QP with a *consistent duplicated equality row*
//! died at iteration 0 as "INTERNAL ERROR: Unknown SolverReturn value" under
//! `qp_presolve=no qp_reg=0` (the issue's 602-variable MPC). Two defects, both
//! fixed:
//!
//! * the seed factorization (`build_factorization`) is numeric as well as
//!   symbolic, and with no regularization the duplicated row leaves a zero
//!   pivot on a variable only the dependent rows touch. With `qp_reg <= 0` the
//!   seed now carries a `1e-8` floor on the `(x, x)` and equality blocks;
//!   every iteration still refactors with the caller's own regularization;
//! * a convex numerical failure reached the console as `InternalError`, a
//!   crash-shaped message; it now maps to `ErrorInStepComputation`.
//!
//! The model: `min x1^2 + x2^2  s.t.  x0 = 3,  2 x0 = 6,  -0.5 x0 + x1 + 0.1 x2 = 0`,
//! `-10 <= x <= 10` -- `x0` has no curvature, so with `qp_reg=0` nothing but the
//! (dependent) equality rows pins it.
//!
//! **This 3-variable model reproduces the defect** (gh#986 review): the issue
//! notes that *its* 3-variable QP with a duplicated row solved fine, but that
//! model gave every variable curvature. Measured by disabling the seed floor:
//! this one ends `Numerical failure (no verified KKT point) ... iters=0` /
//! `Error in step computation` on both the HSDE and the direct (`qp_hsde=no`)
//! route, and solves to `2.2277` in 9 / 6 iterations with it. (A
//! per-iteration δ_w/δ_c rescue added alongside the floor was never reached --
//! not here, not on the 602-variable MPC, not on two free-variable variants --
//! and was removed.)

use std::process::Command;

const NL: &str = "g3 1 1 0\n 3 3 1 0 3\n 0 1\n 0 0\n 0 3 0\n 0 0 0 1 0\n 0 0 0 0 0\n 5 0\n 0 0\n 0 0 0 0 0\nC0\nn0\nC1\nn0\nC2\nn0\nO0 0\no54\n3\no2\nn0.0\no5\nv0\nn2\no2\nn1.0\no5\nv1\nn2\no2\nn1.0\no5\nv2\nn2\nr\n4 3.0\n4 6.0\n4 0.0\nb\n0 -10.0 10.0\n0 -10.0 10.0\n0 -10.0 10.0\nk2\n3\n4\nJ0 1\n0 1.0\nJ1 1\n0 2.0\nJ2 3\n0 -0.5\n1 1.0\n2 0.1\n";

fn run(extra: &[&str]) -> String {
    let dir = std::env::temp_dir().join(format!("gh986_dup_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dup.nl");
    std::fs::write(&path, NL).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&path)
        .args([
            "--no-sol",
            "--no-options-file",
            "solver_selection=qp-ipm",
            "qp_presolve=no",
        ])
        .args(extra)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

const OPTIMUM: &str = "obj=2.2277";

#[test]
fn duplicated_equality_row_solves_with_no_regularization() {
    let text = run(&["qp_reg=0"]);
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
    assert!(text.contains(OPTIMUM), "{text}");
    assert!(!text.contains("INTERNAL ERROR"), "{text}");
}

#[test]
fn duplicated_equality_row_still_solves_at_default_regularization() {
    let text = run(&[]);
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
    assert!(text.contains(OPTIMUM), "{text}");
}
