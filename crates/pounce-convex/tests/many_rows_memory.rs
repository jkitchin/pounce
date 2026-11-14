//! Peak memory of a QP with many inequality rows over a few variables.
//!
//! ```text
//! min ½‖x‖² + cᵀx   s.t.  Gx ≤ h,   G: 6000 × 48, fully dense
//! ```
//!
//! Every row of `G` touches every variable, so in the KKT matrix each row's
//! multiplier is a leaf of one small parent front in feral's elimination
//! tree, and each leaf sends that parent a 48 × 48 contribution block. The
//! classic multifrontal scheme (`feral_early_assembly = off`) holds all 6,000
//! blocks until the parent is assembled: 56.4 MB packed, on every
//! factorization. `wide_parents` folds them into the parent's front as they
//! arrive. Measured on arm64 macOS, the solve's peak live heap above its
//! inputs is 128.6 MB under `off` and 75.9 MB under `wide_parents`, with the
//! same iterations and bit-identical `x`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::{EarlyAssembly, FeralConfig, FeralSolverInterface};
use pounce_linsol::SparseSymLinearSolverInterface;

/// Live and peak heap bytes, process-wide. This binary holds one test, so
/// nothing else allocates in any volume while it measures.
struct PeakAlloc;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(new_size);
        }
        p
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

const N: usize = 48;
const M: usize = 6000;

fn problem() -> QpProblem {
    let mut g = Vec::with_capacity(M * N);
    for r in 0..M {
        for j in 0..N {
            g.push(Triplet::new(
                r,
                j,
                ((r * 7 + j * 13) % 17) as f64 / 17.0 - 0.45,
            ));
        }
    }
    QpProblem {
        n: N,
        p_lower: (0..N).map(|j| Triplet::new(j, j, 1.0)).collect(),
        c: (0..N).map(|j| -1.0 - (j % 3) as f64).collect(),
        a: vec![],
        b: vec![],
        g,
        // Mostly slack rows, a few binding: the unconstrained optimum
        // x = −c violates rows whose right-hand side is small.
        h: (0..M).map(|r| 1.0 + (r % 11) as f64).collect(),
        lb: vec![],
        ub: vec![],
    }
}

/// Solve once with the given scheme; return the solution and the peak live
/// heap above what was live before the solve.
fn solve(qp: &QpProblem, early_assembly: EarlyAssembly) -> (pounce_convex::QpSolution, usize) {
    let cfg = FeralConfig {
        early_assembly,
        ..FeralConfig::default()
    };
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let sol = solve_qp_ipm(qp, &QpOptions::default(), move || {
        Box::new(FeralSolverInterface::with_config(cfg.clone()))
            as Box<dyn SparseSymLinearSolverInterface>
    });
    let above = PEAK.load(Ordering::Relaxed) - before;
    (sol, above)
}

#[test]
fn wide_parents_does_not_hold_every_contribution_block() {
    let qp = problem();
    let (off, off_peak) = solve(&qp, EarlyAssembly::Off);
    let (wide, wide_peak) = solve(&qp, EarlyAssembly::WideParents);

    assert_eq!(off.status, QpStatus::Optimal);
    assert_eq!(wide.status, QpStatus::Optimal);
    // Factors are bit-identical, so the whole trajectory is.
    assert_eq!(wide.iters, off.iters);
    assert_eq!(wide.obj.to_bits(), off.obj.to_bits());
    assert!(
        wide.x
            .iter()
            .zip(&off.x)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // Holding every leaf's block takes M · N(N+1)/2 f64s at the parent's
    // assembly; `wide_parents` must save most of that.
    let held_blocks = M * N * (N + 1) / 2 * std::mem::size_of::<f64>();
    assert!(
        wide_peak + held_blocks / 2 < off_peak,
        "peak above inputs: off {off_peak} B, wide_parents {wide_peak} B; blocks {held_blocks} B"
    );
}
