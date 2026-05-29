//! Regression test for pounce#24: the `pounce` binary must not panic
//! when `l1_fallback_on_restoration_failure=yes` or
//! `l1_exact_penalty_barrier=yes` triggers more than one inner-IPM
//! solve.
//!
//! Before the fix, `pounce-cli/src/main.rs` wired the restoration
//! phase via `app.set_restoration_factory(factory)` — a one-shot
//! closure that panicked with
//! `"restoration factory invoked more than once"` on the second call.
//! The fix routes through
//! `app.set_restoration_factory_provider(...)` (the multi-pass
//! provider from pounce#10 Phase 3), which mints a fresh factory per
//! inner solve.
//!
//! Two fixtures cover complementary code paths:
//!   * `parametric.nl` — converges on the standard solve. Proves the
//!     binary linked and ran end-to-end without the panic.
//!   * `builtin:infeasible-eq` — `min x0^2 + x1^2` subject to two
//!     contradictory equalities (`x0+x1=1`, `x0+x1=2`). The standard
//!     solve drives restoration; the ℓ₁ wrapper then performs a
//!     second inner solve, which is precisely the path that
//!     previously panicked with
//!     "restoration factory invoked more than once". This is the case
//!     the original regression test missed and that let pounce#24
//!     ship undetected.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

#[test]
fn l1_fallback_flag_does_not_panic_in_cli() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("l1_fallback_on_restoration_failure=yes")
        .output()
        .expect("spawn pounce");

    // Exit code 134 / 139 / negative signal = SIGABRT (panic on macOS
    // = signal 6 = exit 134) or other crash. We accept 0
    // (Solve_Succeeded) and 1 (any non-success terminal status).
    let code = output.status.code();
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
}

#[test]
fn l1_exact_penalty_barrier_flag_does_not_panic_in_cli() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("l1_exact_penalty_barrier=yes")
        .output()
        .expect("spawn pounce");

    let code = output.status.code();
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
}

/// Drives the second inner solve. The `infeasible-eq` builtin has two
/// contradictory equality constraints (`x0+x1=1` and `x0+x1=2`), so
/// the standard solve cannot achieve feasibility and the
/// `l1_fallback_on_restoration_failure` flag engages the ℓ₁ wrapper
/// for a second pass. Before pounce#24's fix, the second pass minted
/// a fresh restoration phase via the one-shot
/// `set_restoration_factory` path and panicked. The provider migration
/// (set_restoration_factory_provider) is what this test guards.
#[test]
fn l1_fallback_with_infeasible_problem_drives_second_inner_solve() {
    let output = Command::new(pounce_exe())
        .args(["--problem", "infeasible-eq"])
        .arg("l1_fallback_on_restoration_failure=yes")
        .output()
        .expect("spawn pounce");

    let code = output.status.code();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={stderr}",
        output.status,
    );
    // Confirm the wrapper fired: standard solve must drive
    // restoration (look for an `r` iteration in the log) so we know
    // the multi-pass codepath was actually exercised.
    assert!(
        stdout.lines().any(|l| l.contains('r')
            && l.split_whitespace()
                .next()
                .is_some_and(|t| t.ends_with('r'))),
        "expected a restoration iteration in stdout, fixture did not \
         exercise the multi-pass path:\n{stdout}",
    );
}

#[test]
fn l1_exact_penalty_barrier_with_infeasible_problem() {
    let output = Command::new(pounce_exe())
        .args(["--problem", "infeasible-eq"])
        .arg("l1_exact_penalty_barrier=yes")
        .output()
        .expect("spawn pounce");

    let code = output.status.code();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={stderr}",
        output.status,
    );
}
