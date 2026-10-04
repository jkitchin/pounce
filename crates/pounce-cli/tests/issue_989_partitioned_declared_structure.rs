//! gh #989 item 5: `hessian_approximation=partitioned` took 16 / 28 / 86
//! iterations against the exact Hessian's 12 / 16 / 18 as a Radau-collocated
//! batch reactor was refined. A per-constraint element was a dense `k x k`
//! block over its Jacobian row's support, but a collocation row is linear in
//! most of that support and nonlinear in small independent groups (a state and
//! the control at the same collocation point). One secant pair per iteration
//! cannot determine a dense block, and the rank spent on pairs that do not
//! couple shows up as spurious entries. With the model's *declared* Hessian
//! structure (`partitioned_structure=declared`, the default) each element is
//! split into the connected groups of that pattern and coordinates the pattern
//! never couples are dropped.
//!
//! `data/batch_reactor_radau_n25.nl` is the issue's model at 25 finite elements
//! (Radau IIA, 3 collocation points; 225 variables, 198 constraints). Measured
//! on it: exact 18, partitioned/jacobian 30, partitioned/declared 22.
//!
//! What this does NOT pin, because it is not fixed: `partitioned_update_type=bfgs`
//! still does not converge per-constraint (a PSD model of an indefinite
//! `d2c_j`, weighted by a multiplier of either sign), and the iteration count
//! still grows with the mesh at N = 400.

use std::process::Command;

fn iterations(extra: &[&str]) -> (u32, bool) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/batch_reactor_radau_n25.nl"
    );
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(path)
        .args(["--no-sol", "--no-options-file", "max_iter=300"])
        .args(extra)
        .output()
        .expect("run pounce");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let it = text
        .lines()
        .find(|l| l.starts_with("Number of Iterations"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no iteration count in:\n{text}"));
    (it, text.contains("EXIT: Optimal Solution Found"))
}

#[test]
fn declared_structure_beats_dense_rows_and_stays_near_the_exact_hessian() {
    let (exact, ok_e) = iterations(&["hessian_approximation=exact"]);
    let (jac, ok_j) = iterations(&[
        "hessian_approximation=partitioned",
        "partitioned_structure=jacobian",
    ]);
    let (decl, ok_d) = iterations(&["hessian_approximation=partitioned"]);
    assert!(ok_e && ok_j && ok_d, "all three must converge");
    assert!(
        decl < jac,
        "declared structure ({decl}) must beat dense rows ({jac}); exact {exact}"
    );
    assert!(
        decl * 10 <= exact * 15,
        "partitioned ({decl}) must stay within 1.5x of the exact Hessian ({exact})"
    );
}
