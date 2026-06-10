//! Regression test for code-review 2026-06 item M18: per-call allocations
//! in the tape-AD gradient hot path.
//!
//! `Tape::gradient_seed` allocates a forward-value vector (`forward`) and an
//! adjoint vector (`reverse`) on every call. The `.nl` design deliberately
//! emits one tiny tape per summand — ~10⁶ on large models — so a single
//! `eval_jac_g` / `eval_grad_f` invokes the gradient sweep millions of
//! times, turning those two small allocations into millions of heap hits.
//! `Tape::gradient_seed_into` reuses two caller-supplied scratch arenas and
//! must allocate NOTHING per call.
//!
//! The test installs a counting global allocator and proves, on identical
//! tapes/inputs, that `gradient_seed_into` performs zero heap allocations
//! across many calls while `gradient_seed` allocates on essentially every
//! call — and that both compute the same gradient. A single test lives in
//! this file so no sibling test thread perturbs the global allocation
//! counter while the counting window is open.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pounce_nl::nl_tape::{Tape, TapeOp};

struct CountingAlloc;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Tape for f(x0, x1) = x0 * x1 + exp(x0) * x1 + x0 * x0.
/// Touches Mul, Exp, Add — enough op variety to exercise the reverse sweep.
fn sample_tape() -> Tape {
    Tape {
        ops: vec![
            TapeOp::Var(0),    // 0: x0
            TapeOp::Var(1),    // 1: x1
            TapeOp::Mul(0, 1), // 2: x0*x1
            TapeOp::Exp(0),    // 3: exp(x0)
            TapeOp::Mul(3, 1), // 4: exp(x0)*x1
            TapeOp::Add(2, 4), // 5: x0*x1 + exp(x0)*x1
            TapeOp::Mul(0, 0), // 6: x0^2
            TapeOp::Add(5, 6), // 7: total
        ],
    }
}

#[test]
fn gradient_seed_into_does_not_allocate_per_call() {
    let tape = sample_tape();
    let n = tape.ops.len();
    let x = [1.3_f64, -0.7_f64];

    // Pre-size every buffer OUTSIDE the counting window.
    let mut grad_into = vec![0.0_f64; 2];
    let mut grad_seed = vec![0.0_f64; 2];
    let mut vals = vec![0.0_f64; n];
    let mut adj = vec![0.0_f64; n];

    // Warm-up call (also fixes the reference gradient).
    grad_into.fill(0.0);
    tape.gradient_seed_into(&x, 1.0, &mut grad_into, &mut vals, &mut adj);
    let reference = grad_into.clone();

    // ---- gradient_seed_into: must allocate nothing across many calls ----
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..1000 {
        grad_into.fill(0.0); // reuses the Vec's buffer — no allocation
        tape.gradient_seed_into(&x, 1.0, &mut grad_into, &mut vals, &mut adj);
    }
    COUNTING.store(false, Ordering::Relaxed);
    let into_allocs = ALLOCS.load(Ordering::Relaxed);

    // ---- gradient_seed: the allocating baseline (forward + adj per call) ----
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..1000 {
        grad_seed.fill(0.0);
        tape.gradient_seed(&x, 1.0, &mut grad_seed);
    }
    COUNTING.store(false, Ordering::Relaxed);
    let seed_allocs = ALLOCS.load(Ordering::Relaxed);

    // Same numbers: the no-alloc path is a faithful refactor.
    assert_eq!(
        grad_into, grad_seed,
        "gradient_seed_into must match gradient_seed numerically"
    );
    assert_eq!(grad_into, reference, "result must be stable across calls");

    // The counting harness actually observes allocations: the old path
    // allocates on essentially every call (a forward Vec + an adjoint Vec).
    assert!(
        seed_allocs >= 1000,
        "baseline gradient_seed should allocate per call; saw {seed_allocs}"
    );

    // The fix: zero per-call allocations across 1000 calls.
    assert_eq!(
        into_allocs, 0,
        "gradient_seed_into must not allocate per call; saw {into_allocs}"
    );
}
