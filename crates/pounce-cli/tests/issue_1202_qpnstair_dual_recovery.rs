//! gh#1202: with `presolve=yes`, Vanderbei `qpnstair` reported the reduced
//! problem's duals, because postsolve's derived-bound dual recovery
//! (`pounce_presolve::dual_recovery`, M24) rolled back.
//!
//! The roll-back was the work budget (gh#1196's `MAX_WORK_FLOPS`), and the
//! budget was not too small: the proximal-point loop never stopped. Its
//! residual contracted ×0.44 per step and met the acceptance target
//! (`4.4e-5`) at step 15, then floored at `2e-9` from rounding by step 27.
//! The stop test, a relative step of `1e-15`, is not reachable there (the
//! step stalls near `4e-12` relative, against multipliers of `9e4`), so the
//! loop iterated noise for its remaining 170 steps, and the inner
//! active-set QP cycled on near-zero entries to its iteration cap. With the
//! budget lifted the recovery cost `3.6e10` flops, twelve times the budget.
//! It now stops on the residual, at the rounding floor `100·ε·‖A‖∞·‖w‖∞`:
//! step 25, `1.6e-8`, `3.4e8` flops.
//!
//! The duals this pins are a KKT multiplier set of the *original* model:
//! bound-projected stationarity and bound complementarity at
//! `presolve=no`'s level. They are not `presolve=no`'s duals: on 6 rows and
//! 50 variables the multipliers are not unique (both sets are stationary to
//! `2e-8` with the same signs and complementarity, so their difference is a
//! null vector), and the recovery keeps the reduced solve's split there.
//! Everywhere else the two agree to `1.4e-4` against magnitudes of `9e4`.
//!
//! The rolled-back `.sol` used to pass `pounce verify --require-optimal`.
//! That check gated only on the residual *with* the `.sol`'s bound
//! multipliers, and those balanced stationarity from sides that cannot carry
//! a multiplier. The largest is `z_U = 2.8e4` on `rgap1`, which has no upper
//! bound. `verify` now rejects that as a misplaced bound multiplier
//! (`verify_misplaced_bound_multipliers.rs`). The test also asserts the
//! bound-projected residual (`2.8e4` when rolled back) and bound
//! complementarity (`0.21` against `presolve=no`'s `2.2e-3`), which do not
//! depend on that check.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn temp(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pounce_1202_{}_{tag}.{ext}", std::process::id()))
}

/// Solve `qpnstair` with `presolve`, write the `.sol`; return stderr.
fn solve(presolve: &str, sol: &Path) -> String {
    let out = Command::new(pounce_exe())
        .arg(fixture("qpnstair.nl"))
        .arg(sol)
        .arg(format!("presolve={presolve}"))
        .arg("print_level=0")
        .env_remove("RUST_LOG")
        .env_remove("pounce_options")
        .arg("--no-options-file")
        .output()
        .expect("run pounce");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "presolve={presolve} failed: {stderr}\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    stderr
}

/// `pounce verify --require-optimal`'s receipt for `sol`.
fn verify(sol: &Path) -> serde_json::Value {
    let receipt = temp("receipt", "json");
    let out = Command::new(pounce_exe())
        .arg("verify")
        .arg(fixture("qpnstair.nl"))
        .arg(sol)
        .arg("--require-optimal")
        .arg("--json-output")
        .arg(&receipt)
        .output()
        .expect("run pounce verify");
    assert!(
        out.status.success(),
        "verify --require-optimal rejected {}: {}",
        sol.display(),
        String::from_utf8_lossy(&out.stdout)
    );
    let v = serde_json::from_str(&std::fs::read_to_string(&receipt).unwrap()).unwrap();
    let _ = std::fs::remove_file(&receipt);
    v
}

fn field(v: &serde_json::Value, name: &str) -> f64 {
    v["optimality"][name]
        .as_f64()
        .unwrap_or_else(|| panic!("receipt has no optimality.{name}: {v}"))
}

#[test]
fn qpnstair_presolve_duals_are_re_attributed_to_the_original_model() {
    let (with, without) = (temp("yes", "sol"), temp("no", "sol"));
    let stderr = solve("yes", &with);
    assert!(
        !stderr.contains("could not be re-attributed"),
        "the derived-bound dual recovery rolled back: {stderr}"
    );
    solve("no", &without);
    let (w, n) = (verify(&with), verify(&without));
    let _ = std::fs::remove_file(&with);
    let _ = std::fs::remove_file(&without);

    // Bound-projected stationarity: the constraint duals alone, with bound
    // multipliers inferred from activity. 2.8e4 when rolled back.
    let s = field(&w, "stationarity_residual");
    assert!(s <= 1e-6, "bound-projected stationarity {s:e}");
    // Bound complementarity: no multiplier mass on an inactive bound beyond
    // what presolve=no carries itself. 0.21 against 2.2e-3 when rolled back.
    let (cw, cn) = (
        field(&w, "bound_complementarity_residual"),
        field(&n, "bound_complementarity_residual"),
    );
    assert!(
        cw <= 2.0 * cn + 1e-6,
        "bound complementarity {cw:e} against presolve=no's {cn:e}"
    );
    let e = field(&w, "stationarity_residual_with_bound_multipliers");
    assert!(e <= 1e-6, "stationarity with bound multipliers {e:e}");
}
