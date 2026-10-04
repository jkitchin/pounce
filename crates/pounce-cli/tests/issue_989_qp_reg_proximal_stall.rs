//! gh #989 item 7: `qp_reg=1e-4` on the CSTR MPC QP with the control in J/min
//! (`data/mpc_cstr_jpermin_n40.nl`, the issue's model at N = 40 and the control
//! scaled by UA = 5e4) hit `Maximum iterations exceeded` at 200 iterations with
//! an objective of 497.8 against the optimum 149.4066. A static `qp_reg` is the
//! same absolute proximal weight on every column, and the control's reduced
//! curvature is ~1e-10 in those units, so the Newton step was damped to a
//! crawl. The HSDE driver now watches progress and decays a proximal-sized
//! `qp_reg` when the merit stops improving.
//!
//! The default `qp_reg` is below the guard's threshold and must keep taking
//! exactly the path it always did.

use std::process::Command;

fn run(extra: &[&str]) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/mpc_cstr_jpermin_n40.nl"
    );
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(path)
        .args(["--no-sol", "--no-options-file", "solver_selection=qp-ipm"])
        .args(extra)
        .output()
        .expect("run pounce");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

const OPTIMUM: &str = "obj=149.406564";

#[test]
fn proximal_qp_reg_does_not_stall_when_the_control_is_in_joules_per_minute() {
    let text = run(&["qp_reg=1e-4"]);
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
    assert!(text.contains(OPTIMUM), "{text}");
    assert!(
        text.contains("qp_reg is a proximal term"),
        "the guard must say it reduced the regularization: {text}"
    );
}

#[test]
fn default_qp_reg_is_untouched_by_the_guard() {
    let text = run(&[]);
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
    assert!(text.contains(OPTIMUM), "{text}");
    assert!(!text.contains("qp_reg is a proximal term"), "{text}");
}
