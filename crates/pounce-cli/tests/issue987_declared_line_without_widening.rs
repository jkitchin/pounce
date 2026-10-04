//! gh#987 (remaining item c): the red "Violation of the model as declared
//! (before the bound_relax_factor widening)" console line must not appear
//! when no widening moved the answer.
//!
//! Fixture: `min (x-3)^2  s.t.  1e6*x^2 = 1e6`. The row's gradient is `2e6`,
//! so gradient-based scaling divides it by `2e4`, and the scaled residual is
//! `2e4` times smaller than the user-unit one. At `bound_relax_factor = 0`
//! the line used to print, attributing that scale factor to a widening that
//! never happened (and equality rows are never widened at any setting).

use std::path::PathBuf;
use std::process::Command;

const SCALED_EQ: &str = "g3 1 1 0\n 1 1 1 0 1\n 1 1\n 0 0\n 1 1 1\n 0 0 0 1\n 0 0 0 0 0\n 1 1\n 0 0\n 0 0 0 0 0\nC0\no2\nn1000000\no5\nv0\nn2\nO0 0\no5\no0\nv0\nn-3\nn2\nr\n4 1000000\nb\n3\nx1\n0 2\nk0\nJ0 1\n0 0\nG0 1\n0 0\n";

fn solve(extra: &str) -> String {
    let mut nl = std::env::temp_dir();
    nl.push(format!(
        "pounce_987c_{}_{}.nl",
        std::process::id(),
        extra.replace(['=', '.', '-'], "_")
    ));
    std::fs::write(&nl, SCALED_EQ).unwrap();
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg(extra)
        .env_remove("CLICOLOR_FORCE")
        .output()
        .expect("spawn pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn no_widening_line_at_bound_relax_factor_zero_or_on_a_row_scale_gap() {
    for opt in ["bound_relax_factor=0", "bound_relax_factor=1e-8"] {
        let o = solve(opt);
        assert!(o.contains("EXIT: Optimal Solution Found"), "{opt}:\n{o}");
        assert!(
            !o.contains("Violation of the model as declared"),
            "{opt}: the gap here is the row scale factor, not a widening:\n{o}"
        );
    }
}
