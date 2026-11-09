//! gh#1147 Stage 2 — handle operations whose cost must not grow with the
//! model (plan §6.2 D1, §6.4 "Scale").
//!
//! Each test times the same `k` operations on a small and a large handle
//! and asserts the ratio, not an absolute time, so it does not depend on the
//! machine. A cost that is O(n + m) per call shows up as a ratio near the
//! size ratio (`SCALE`); the thresholds sit far below that and far above
//! timing noise.
//!
//! Sizes are kept small enough that an O(P)-per-call regression still
//! finishes in seconds in a debug build, so it fails rather than hangs.
//!
//! No solve runs: the callbacks exist only to satisfy `CreateIpoptProblem`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_cinterface::*;
use std::time::{Duration, Instant};

const INF: Number = 2e19;
/// Size ratio between the large and the small handle.
const SCALE: usize = 20;
/// Largest accepted large/small time ratio.
const MAX_RATIO: f64 = 2.5;

unsafe extern "C" fn ev_f(
    _: Index,
    _: *const Number,
    _: Bool,
    _: *mut Number,
    _: *mut std::ffi::c_void,
) -> Bool {
    0
}
unsafe extern "C" fn ev_grad_f(
    _: Index,
    _: *const Number,
    _: Bool,
    _: *mut Number,
    _: *mut std::ffi::c_void,
) -> Bool {
    0
}
unsafe extern "C" fn ev_g(
    _: Index,
    _: *const Number,
    _: Bool,
    _: Index,
    _: *mut Number,
    _: *mut std::ffi::c_void,
) -> Bool {
    0
}
unsafe extern "C" fn ev_jac_g(
    _: Index,
    _: *const Number,
    _: Bool,
    _: Index,
    _: Index,
    _: *mut Index,
    _: *mut Index,
    _: *mut Number,
    _: *mut std::ffi::c_void,
) -> Bool {
    0
}

/// `n` variables in `[0, 1]` and `n` rows, `g >= 0` (so every `(j, j)` is a
/// native one-sided + bound pair) or free (`free_rows`, as MOI `Complements`
/// needs).
fn create(n: usize, free_rows: bool) -> IpoptProblem {
    let x_l = vec![0.0; n];
    let x_u = vec![1.0; n];
    let g_l = vec![if free_rows { -INF } else { 0.0 }; n];
    let g_u = vec![INF; n];
    let p = unsafe {
        CreateIpoptProblem(
            n as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            n as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            0,
            0,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            None,
        )
    };
    assert!(!p.is_null());
    p
}

/// Best of three runs of `op` on a fresh handle of size `n`.
fn best_of_three(n: usize, free_rows: bool, op: impl Fn(IpoptProblem)) -> Duration {
    (0..3)
        .map(|_| {
            let p = create(n, free_rows);
            let t = Instant::now();
            op(p);
            let d = t.elapsed();
            unsafe { FreeIpoptProblem(p) };
            d
        })
        .min()
        .unwrap()
}

/// Per-operation time of `op(p, k)` (which performs `k` operations) at `k`
/// and at `k * SCALE`, on handles of size `n = k` and `n = k * SCALE`:
/// the ratio is ~1 when each operation is O(1) and ~SCALE when each is
/// O(P) or O(n).
fn assert_flat(what: &str, k: usize, free_rows: bool, op: impl Fn(IpoptProblem, usize) + Copy) {
    let small = best_of_three(k, free_rows, |p| op(p, k)).as_secs_f64() / k as f64;
    let kl = k * SCALE;
    let large = best_of_three(kl, free_rows, |p| op(p, kl)).as_secs_f64() / kl as f64;
    let ratio = large / small.max(1e-12);
    eprintln!(
        "{what}: {:.3} us/call at {k}, {:.3} us/call at {kl}, ratio {ratio:.1}",
        small * 1e6,
        large * 1e6
    );
    assert!(
        ratio < MAX_RATIO,
        "{what}: {:.2} us per call over {k} calls, {:.2} us over {kl} \
         (ratio {ratio:.1}, limit {MAX_RATIO}): the per-call cost grows with the model",
        small * 1e6,
        large * 1e6,
    );
}

#[test]
fn declaring_pairs_one_at_a_time_costs_o1_per_call() {
    // Before S2-I1 each call cloned and re-validated the whole stored set
    // with `validate_pairs`: O(P) per call, O(P^2) for P declarations.
    assert_flat("AddPounceComplementarityPair", 1000, false, |p, k| {
        for j in 0..k as Index {
            assert_eq!(
                unsafe { AddPounceComplementarityPair(p, j, j) },
                POUNCE_CC_OK
            );
        }
    });
}

#[test]
fn moi_complements_declarations_cost_o1_per_call() {
    // Before S2-I1 each call also scanned every declared pair and ledger
    // entry for duplicates: O(P) per call.
    assert_flat("AddPounceComplementsConstraint", 1000, true, |p, k| {
        for j in 0..k as Index {
            assert_eq!(
                unsafe { AddPounceComplementsConstraint(p, j, j) },
                POUNCE_COMPLEMENTS_PAIR_MIXED
            );
        }
    });
}

#[test]
fn bound_updates_cost_o1_per_entry() {
    // gh#1147 Stage 2 (plan §6.2 D1): an update touching a paired variable
    // and its row re-classifies that pair only. A setter that re-validated
    // the declared set, or allocated per variable or row, would grow with
    // n = m = P here. Each call is one variable and one row; half of them
    // touch a declared pair.
    assert_flat("SetPounceBounds", 1000, false, |p, k| {
        for j in (0..k as Index).step_by(2) {
            assert_eq!(
                unsafe { AddPounceComplementarityPair(p, j, j) },
                POUNCE_CC_OK
            );
        }
        for j in 0..k as Index {
            let (xi, xl, xu) = ([j], [0.0], [2.0]);
            let (gi, gl, gu) = ([j], [-0.5], [INF]);
            let code = unsafe {
                SetPounceBounds(
                    p,
                    1,
                    xi.as_ptr(),
                    xl.as_ptr(),
                    xu.as_ptr(),
                    1,
                    gi.as_ptr(),
                    gl.as_ptr(),
                    gu.as_ptr(),
                    0,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(code, POUNCE_BOUNDS_OK);
        }
    });
}
