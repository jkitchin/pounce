//! gh#1202: postsolve's derived-bound dual recovery on the two corpus models
//! where it rolled back, Vanderbei `qpnstair` and `deb7` (the second test,
//! documented on it).
//!
//! With `presolve=yes`, Vanderbei `qpnstair` reported the reduced
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
//! `presolve=no`'s level. On 6 rows and 50 variables the multipliers are not
//! unique: two valid sets were measured stationary to `2e-8` with the same
//! signs and complementarity, so their difference is a null vector. The
//! recovery first returned one differing from `presolve=no`'s by up to
//! `2.8e4` there. Since its proximal weight became small (each step lands on
//! the least-squares solution nearest its start), it returns `presolve=no`'s
//! to `5e-3`. Everywhere else the two agree to `1.4e-4` against magnitudes
//! of `9e4`. The test asserts validity, not the choice.
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

/// Solve `model` with `presolve`, write the `.sol`; return stderr.
fn solve(model: &str, presolve: &str, sol: &Path) -> String {
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
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

/// `pounce verify --require-optimal`'s receipt for `sol`, and whether it
/// verified.
fn receipt(model: &str, sol: &Path) -> (bool, serde_json::Value) {
    let receipt = temp(&format!("{model}_receipt"), "json");
    let out = Command::new(pounce_exe())
        .arg("verify")
        .arg(fixture(model))
        .arg(sol)
        .arg("--require-optimal")
        .arg("--json-output")
        .arg(&receipt)
        .output()
        .expect("run pounce verify");
    let v = serde_json::from_str(&std::fs::read_to_string(&receipt).unwrap()).unwrap();
    let _ = std::fs::remove_file(&receipt);
    (out.status.success(), v)
}

/// [`receipt`], which must verify.
fn verify(model: &str, sol: &Path) -> serde_json::Value {
    let (ok, v) = receipt(model, sol);
    assert!(
        ok,
        "verify --require-optimal rejected {}: {v}",
        sol.display()
    );
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
    let stderr = solve("qpnstair.nl", "yes", &with);
    assert!(
        !stderr.contains("could not be re-attributed"),
        "the derived-bound dual recovery rolled back: {stderr}"
    );
    solve("qpnstair.nl", "no", &without);
    let (w, n) = (
        verify("qpnstair.nl", &with),
        verify("qpnstair.nl", &without),
    );
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

/// deb7: the recovery rolled back twice over. Its component (684
/// multipliers) was past the dense solve's 600 cap (the recovery is now
/// sparse and uncapped); past the cap, the solve missed its
/// target (`2.4e-2` against `1.1e-3`). The reduced solve hands over a
/// cancelling pair `λ = ±1.16e10` on two equalities with the same gradient
/// at `x*`, and anchored there the iteration could not get below the
/// rounding that pair costs. Started from zero, it reaches the least-squares
/// optimum, `1.02e-4`, with multipliers of size `1.4e3` (scipy's bounded
/// least squares on the same system finds `1.03e-4`).
///
/// `verify --require-optimal` rejects deb7 whatever its duals, for its
/// declared integer columns, so this reads the receipt's fields instead of
/// its verdict. And `presolve=yes` lands on a different point from
/// `presolve=no` (same objective to `1e-7` relative, `x` differs by `0.18`),
/// so the duals are compared by the quantities a KKT point must satisfy,
/// not entry by entry.
#[test]
fn deb7_presolve_duals_are_re_attributed_to_the_original_model() {
    let (with, without) = (temp("deb7_yes", "sol"), temp("deb7_no", "sol"));
    let stderr = solve("deb7.nl", "yes", &with);
    assert!(
        !stderr.contains("could not be re-attributed"),
        "the derived-bound dual recovery rolled back: {stderr}"
    );
    solve("deb7.nl", "no", &without);
    let ((_, w), (_, n)) = (receipt("deb7.nl", &with), receipt("deb7.nl", &without));
    let _ = std::fs::remove_file(&with);
    let _ = std::fs::remove_file(&without);

    // No multiplier on a side that cannot carry one: 125 when rolled back.
    let m = field(&w, "misplaced_bound_multiplier");
    assert_eq!(m, 0.0, "misplaced bound multiplier {m:e}");
    // 452 when rolled back, against presolve=no's 3.3e-6.
    let (cw, cn) = (
        field(&w, "bound_complementarity_residual"),
        field(&n, "bound_complementarity_residual"),
    );
    assert!(
        cw <= 2.0 * cn + 1e-6,
        "bound complementarity {cw:e} against presolve=no's {cn:e}"
    );
    // Stationarity to the solver's own level at this point (1.1e-4).
    let e = field(&w, "stationarity_residual_with_bound_multipliers");
    assert!(e <= 2e-4, "stationarity with bound multipliers {e:e}");
    let s = field(&w, "stationarity_residual");
    assert!(s <= 2e-4, "bound-projected stationarity {s:e}");
}
