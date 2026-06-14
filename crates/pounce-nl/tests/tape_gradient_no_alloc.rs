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
//! call — and that both compute the same gradient.
//!
//! The counters are **thread-local**, not global: the `#[global_allocator]`
//! intercepts every allocation process-wide, but it only tallies those made
//! on the thread that opened the counting window. libtest runs background
//! threads of its own (output capture, timing, the test-runner thread) whose
//! allocations would otherwise race into the window and produce a sporadic
//! nonzero count on the no-alloc path — the original global-atomic version was
//! flaky for exactly this reason. The thread-locals use `const` initializers,
//! so reading them inside the allocator never allocates (no reentrancy) and
//! they carry no destructor to panic on at thread teardown.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use pounce_nl::nl_tape::{Tape, TapeOp};

struct CountingAlloc;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Count one allocation against the current thread, but only while its
/// counting window is open. `try_with` guards the rare case where the TLS is
/// touched during thread teardown.
fn tally() {
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

fn set_counting(on: bool) {
    COUNTING.with(|c| c.set(on));
}

fn reset_allocs() {
    ALLOCS.with(|n| n.set(0));
}

fn allocs() -> usize {
    ALLOCS.with(|n| n.get())
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        tally();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        tally();
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
    reset_allocs();
    set_counting(true);
    for _ in 0..1000 {
        grad_into.fill(0.0); // reuses the Vec's buffer — no allocation
        tape.gradient_seed_into(&x, 1.0, &mut grad_into, &mut vals, &mut adj);
    }
    set_counting(false);
    let into_allocs = allocs();

    // ---- gradient_seed: the allocating baseline (forward + adj per call) ----
    reset_allocs();
    set_counting(true);
    for _ in 0..1000 {
        grad_seed.fill(0.0);
        tape.gradient_seed(&x, 1.0, &mut grad_seed);
    }
    set_counting(false);
    let seed_allocs = allocs();

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
