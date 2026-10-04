//! gh #991 at the CLI: a convex QP infeasible by a small exact margin, solved
//! with `solver_selection=qp-active-set`, must exit `Infeasible_Problem_Detected`
//! / `solve_result_num = 200` — the verdict `qp-ipm` and `nlp` reach on the same
//! file — and not `Maximum_Iterations_Exceeded` / 400, which tells an AMPL,
//! Pyomo or GAMS driver to raise the limit and retry on a model no budget can
//! solve. Before the fix the engine stopped after 3 iterations with the budget
//! untouched, identically at `max_iter=100000`.
//!
//! Model: `min x₀² + x₀x₁ + x₁² − x₀ + 2x₁` s.t. `x₀ + 2x₁ ≤ 2`,
//! `x₀ + 2x₁ ≥ 2 + 1e-5`, both variables free. Farkas pair `y = (1, 1)`.

use std::process::Command;

const SLIVER: &str = "g3 1 1 0\n 2 2 1 0 0\n 0 1\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0 0 0 0\n 4 2\n 0 0\n 0 0 0 0 0\nC0\nn0\nC1\nn0\nO0 0\no54\n3\no5\nv0\nn2\no2\nv0\nv1\no5\nv1\nn2\nr\n1 2\n2 2.00001\nb\n3\n3\nk1\n2\nJ0 2\n0 1\n1 2\nJ1 2\n0 1\n1 2\nG0 2\n0 -1\n1 2\n";

fn solve(selection: &str, max_iter: &str) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!(
        "pounce_991_{}_{selection}_{max_iter}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let nl = dir.join("sliver.nl");
    std::fs::write(&nl, SLIVER).unwrap();
    let json = dir.join("report.json");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .arg(format!("solver_selection={selection}"))
        .arg(format!("max_iter={max_iter}"))
        .arg("--no-options-file")
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no JSON report ({e}); stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    v["solution"].clone()
}

#[test]
fn qp_active_set_reports_the_sliver_infeasible_at_any_budget() {
    for max_iter in ["3000", "100000"] {
        let s = solve("qp-active-set", max_iter);
        assert_eq!(s["engine"], "qp-active-set", "max_iter={max_iter}: {s}");
        assert_eq!(
            s["status_upstream"], "Infeasible_Problem_Detected",
            "max_iter={max_iter}: {s}"
        );
        assert_eq!(s["solve_result_num"], 200, "max_iter={max_iter}: {s}");
    }
}

/// The oracles: the other two arms already agree on this file.
#[test]
fn the_ipm_and_nlp_arms_agree() {
    for sel in ["qp-ipm", "nlp"] {
        let s = solve(sel, "3000");
        assert_eq!(
            s["status_upstream"], "Infeasible_Problem_Detected",
            "{sel}: {s}"
        );
        assert_eq!(s["solve_result_num"], 200, "{sel}: {s}");
    }
}
