//! gh #848 — `sqp_qp_certify_second_order` is a real switch, end to end.
//!
//! The second-order check that stops the active-set engine certifying a
//! saddle point lives in `pounce-qp`. It is on for every standalone QP solve
//! and **off** for the SQP's step subproblem, because those two callers ask
//! the engine different questions — see `QpOptions::sqp_subproblem`, and
//! gh #856 for what turning it on here needs first.
//!
//! Off-by-default makes the switch matter more, not less. It is a
//! *trajectory* change on every route that hands the engine an indefinite
//! Hessian, so it needs a user-reachable switch — and a switch nothing reads
//! is worse than none, because its documentation describes behaviour that does
//! not exist. That is gh #677 (`limited_memory_initialization` was registered
//! and never read) and the `sqp_qp_use_homotopy` no-op found while writing the
//! warm-start benchmark. Both were invisible to
//! `convex_option_readers_match_the_registry`, which pins that a value the
//! registry accepts is never rejected by a reader — a different claim from
//! "setting it changes the answer".
//!
//! So this file asserts the answer moves, on fixtures where the two
//! behaviours give *different objectives* rather than different iteration
//! counts. `pounce-qp`'s own
//! `issue848_second_order_certification::the_check_can_be_switched_off_and_then_the_saddle_comes_back`
//! pins the engine-level behaviour; what is only reachable from here is the
//! plumbing between the CLI option registry and that reader.
//!
//! ## Which fixture can carry which claim
//!
//! `nonconvex_qp.nl` is `min x₀·x₁ s.t. x₀ + x₁ = 2, 0 ≤ x ≤ 4` (gh #797). On
//! the feasible segment `f(x₀) = x₀(2 − x₀)` is concave, so the interior
//! stationary point `(1, 1)` is the constrained **maximum** at `obj = 1` and
//! the minimum `obj = 0` sits at either endpoint. With the check on, the arm
//! reaches the minimum — that claim is portable and is
//! `turning_it_on_stops_the_sqp_arm_certifying_the_constrained_maximum`.
//!
//! What is **not** portable is the *default's* answer on that fixture, and the
//! first version of this file asserted it. All three of `(1, 1)`, `(0, 2)` and
//! `(2, 0)` satisfy first-order KKT, so which one an active-set method returns
//! is decided by the working-set path and not by the specification — and the
//! model is exactly symmetric under swapping `x₀` and `x₁`, so that path turns
//! on a tie. Both architectures break it, and they break it differently:
//! macOS/arm64 returns the maximum `1` (20 of 20 runs, debug and release);
//! ubuntu-latest/x86-64 returns `0`, which is how the assertion failed in
//! CI run 33287729052. `nonconvex_two_escapes.py`'s own docstring names the
//! mechanism from the other side — "exactly the symmetry that makes
//! `nonconvex_qp.nl` converge onto its constrained maximum". A tie is not a
//! defect manifestation you can pin; the record of what the default costs
//! lives in prose here and on gh #856, not in an assertion that is true on
//! half the machines that run it.
//!
//! `nonconvex_two_escapes.nl` (gh #805) carries the switch-is-real claim
//! instead, because its default answer is reached by exact cancellation
//! rather than by a tie. The model is even in `x₁`, so `∂f/∂x₁` vanishes
//! identically on `x₁ = 0`, and `∂f/∂x₀` vanishes identically at `x₀ = 0` —
//! both in exact arithmetic and in floating point, where the products are
//! zero bit-for-bit. From the start `(0, 1)` the arm is driven onto `A =
//! (0, 0)`, `f = 0`, in one iteration and stops there: a genuine first-order
//! point, and the pre-#797 answer. With the check on it reaches the global
//! minimum at the corner, `f = −6752.25`. That is a gap of nearly four
//! orders of magnitude, so the test asserts the *strict improvement* rather
//! than either endpoint's exact value — a switch that stops being read fails
//! it, and a tie broken the other way on some future target does not.

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

fn objective(args: &[&str]) -> f64 {
    let out = Command::new(pounce_exe())
        .args(args)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let line = combined
        .lines()
        .find(|l| l.starts_with("Objective"))
        .unwrap_or_else(|| panic!("no objective line in:\n{combined}"));
    line.split_whitespace()
        .nth(1)
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("unparseable objective line {line:?}"))
}

#[test]
fn turning_it_on_stops_the_sqp_arm_certifying_the_constrained_maximum() {
    let f = fixture("nonconvex_qp.nl");
    let f = f.to_str().unwrap();
    let on = objective(&[
        f,
        "--no-sol",
        "algorithm=active-set-sqp",
        "sqp_qp_certify_second_order=yes",
    ]);
    assert!(
        on.abs() < 1e-6,
        "with the check on the SQP must reach the minimum 0, not the interior \
         stationary point 1; got {on}"
    );
}

/// The option is plumbed from the CLI registry through to the engine, and
/// setting it changes the answer — gh #677's lesson, asserted rather than
/// assumed.
///
/// This is the claim `nonconvex_qp.nl` cannot carry: see the module docs for
/// why its default answer is a tie that macOS and Linux break differently.
/// Here the default's stopping point is forced by exact cancellation, so what
/// separates the two settings is a real ~6752-wide improvement and not luck.
#[test]
fn the_switch_is_not_a_no_op_end_to_end() {
    let f = fixture("nonconvex_two_escapes.nl");
    let f = f.to_str().unwrap();
    let off = objective(&[f, "--no-sol", "algorithm=active-set-sqp"]);
    let on = objective(&[
        f,
        "--no-sol",
        "algorithm=active-set-sqp",
        "sqp_qp_certify_second_order=yes",
    ]);
    assert!(
        on < off - 1.0,
        "setting sqp_qp_certify_second_order=yes must change the answer, and \
         change it for the better: the default stops at the ridge point A \
         (f = 0) and the check must escape it. Got off={off}, on={on}. If \
         these are equal the option is being registered and not read, which \
         is gh #677's shape."
    );
    assert!(
        (on + 6752.25).abs() < 1e-3,
        "with the check on the arm should reach the global minimum at the \
         corner, f = -6752.25; got {on}. A different value means the escape \
         ladder changed — see nonconvex_two_escapes.py, which documents which \
         answer each rung gives."
    );
}
