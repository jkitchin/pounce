//! Phase-separated benchmark: where the time goes, and what a refactorization
//! actually saves.
//!
//! POUNCE refactors matrices with an *identical* sparsity pattern hundreds of
//! times per solve and back-substitutes twice per iteration on top of that, so
//! a single "factorization took N ms" number is the wrong granularity. This
//! splits each backend's work into the phases a repeated-pattern workload
//! cares about:
//!
//! ```text
//!   conversion     triplet -> CSC (full build once, cached scatter after)
//!   equilibration  A_hat = D A D
//!   symbolic       fill-reducing ordering + supernodes  (once per pattern)
//!   numeric        the factorization proper
//!   materialize    L into CSC for the solve   (adapter overhead, see below)
//!   solve          one back-substitution
//! ```
//!
//! The RSLAB rows come from [`pounce_rslab::PhaseTimings`], measured inside
//! the adapter. FERAL does not expose an analyze/factor split through
//! `pounce-feral`, so its rows are the trait-level totals the cross-solver
//! harness records — first factor, refactor, solve — which is the comparison
//! that matters anyway: what the IPM pays.
//!
//! `materialize` is adapter overhead, not an RSLAB cost: the adapter reaches
//! for RSLAB's low-level entry points because the ergonomic `LdltSolver`
//! exposes no pivot magnitude, and pays for that by materializing `L` in CSC
//! on every factorization. It is reported separately so a reader can subtract
//! it when asking what RSLAB itself would cost.
//!
//! ```sh
//! cargo run -p pounce-rslab --release --example rslab_bench
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::time::Instant;

use pounce_common::types::{Index, Number};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_rslab::compare::{self, SymTriplet};
use pounce_rslab::{RslabConfig, RslabSolverInterface};

/// A synthetic KKT with POUNCE's shape: `[[W + Σ, Jᵀ], [J, 0]]` over a 2-D
/// grid, with the structurally zero (2,2) block and the explicit zeros a real
/// POUNCE triplet pattern carries.
///
/// The zero (2,2) block is the whole point — it is what forces 2×2 pivots and
/// what a backend without delayed pivoting trips over. `sigma` is the barrier
/// diagonal; small values push the (1,1) block towards singular, which is
/// where an IPM spends its last iterations.
fn synthetic_kkt(k: usize, sigma: f64) -> SymTriplet {
    let nx = k * k; // primal variables on a k x k grid
    let nc = k; // one constraint per row of the grid
    let n = nx + nc;
    let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let at = |i: usize, j: usize| i * k + j;

    // (1,1): 5-point Laplacian plus the barrier diagonal.
    for i in 0..k {
        for j in 0..k {
            let r = at(i, j);
            irn.push(r as Index + 1);
            jcn.push(r as Index + 1);
            vals.push(4.0 + sigma);
            if i > 0 {
                irn.push(r as Index + 1);
                jcn.push(at(i - 1, j) as Index + 1);
                vals.push(-1.0);
            }
            if j > 0 {
                irn.push(r as Index + 1);
                jcn.push(at(i, j - 1) as Index + 1);
                vals.push(-1.0);
            }
        }
    }
    // (2,1): each constraint couples one grid row.
    for i in 0..k {
        for j in 0..k {
            irn.push((nx + i) as Index + 1);
            jcn.push(at(i, j) as Index + 1);
            vals.push(1.0 + 0.1 * (j as f64));
        }
    }
    // (2,2): structurally present, numerically zero — as POUNCE emits it.
    for i in 0..nc {
        irn.push((nx + i) as Index + 1);
        jcn.push((nx + i) as Index + 1);
        vals.push(0.0);
    }
    debug_assert!(n > 0);
    SymTriplet {
        n: n as Index,
        irn,
        jcn,
        vals,
    }
}

/// The same shape with a **sparse** constraint Jacobian: one entry per row,
/// which is what a slack / simple-bound row contributes and what gives a KKT
/// its "arrow" signature.
///
/// This variant exists because the dense-row synthetic above factors fine in
/// RSLAB's exact mode at every size tried, and a corpus that is uniform in the
/// dimension a defect acts on reports nothing no matter how large its models
/// are. Whether *this* one trips the no-delayed-pivoting failure is a
/// measurement, and the bench prints it either way.
fn synthetic_kkt_sparse_j(k: usize, sigma: f64) -> SymTriplet {
    let nx = k * k;
    let nc = nx / 2;
    let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let at = |i: usize, j: usize| i * k + j;
    for i in 0..k {
        for j in 0..k {
            let r = at(i, j);
            irn.push(r as Index + 1);
            jcn.push(r as Index + 1);
            vals.push(4.0 + sigma);
            if i > 0 {
                irn.push(r as Index + 1);
                jcn.push(at(i - 1, j) as Index + 1);
                vals.push(-1.0);
            }
            if j > 0 {
                irn.push(r as Index + 1);
                jcn.push(at(i, j - 1) as Index + 1);
                vals.push(-1.0);
            }
        }
    }
    // One Jacobian entry per constraint row.
    for c in 0..nc {
        irn.push((nx + c) as Index + 1);
        jcn.push((2 * c) as Index + 1);
        vals.push(1.0);
        irn.push((nx + c) as Index + 1);
        jcn.push((nx + c) as Index + 1);
        vals.push(0.0);
    }
    SymTriplet {
        n: (nx + nc) as Index,
        irn,
        jcn,
        vals,
    }
}

/// Load a captured real KKT from `tests/fixtures/`.
fn load_fixture(name: &str) -> Option<SymTriplet> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.txt"));
    let text = std::fs::read_to_string(path).ok()?;
    let mut n = 0 as Index;
    let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("n") => n = f.next()?.parse().ok()?,
            Some("t") => {
                irn.push(f.next()?.parse().ok()?);
                jcn.push(f.next()?.parse().ok()?);
                vals.push(f.next()?.parse().ok()?);
            }
            _ => {}
        }
    }
    Some(SymTriplet { n, irn, jcn, vals })
}

/// Best of `reps` — wall-clock, so the minimum is the least contaminated
/// sample, not the mean.
fn best(reps: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// RSLAB's phase split on one matrix, with the refactorization measured
/// separately from the first factor.
fn bench_rslab(a: &SymTriplet, b: &[Number], cfg: RslabConfig, reps: usize) -> Option<String> {
    let mut s = RslabSolverInterface::with_config(cfg);
    s.initialize_structure(a.n, a.nnz() as Index, &a.irn, &a.jcn);
    s.values_array_mut().copy_from_slice(&a.vals);
    let mut rhs = b.to_vec();
    if s.multi_solve(true, &a.irn, &a.jcn, 1, &mut rhs, false, 0) != ESymSolverStatus::Success {
        return None;
    }
    let first = s.phase_timings();

    // Refactorization: same pattern, same values, analysis cached.
    let mut refac = first;
    for _ in 0..reps {
        s.values_array_mut().copy_from_slice(&a.vals);
        let mut r = b.to_vec();
        if s.multi_solve(true, &a.irn, &a.jcn, 1, &mut r, false, 0) != ESymSolverStatus::Success {
            return None;
        }
        let t = s.phase_timings();
        if t.factor_total_ms() < refac.factor_total_ms() {
            refac = t;
        }
    }

    // Back-substitution alone.
    let solve_ms = best(reps.max(3), || {
        let mut r = b.to_vec();
        let t = Instant::now();
        let _ = s.multi_solve(false, &a.irn, &a.jcn, 1, &mut r, false, 0);
        t.elapsed().as_secs_f64() * 1e3
    });

    Some(format!(
        "{:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} | {:>9.3} {:>9.3} {:>9.3} | {:>9.3}",
        first.conversion_ms,
        first.equilibration_ms,
        first.symbolic_ms,
        first.numeric_ms,
        first.materialize_ms,
        first.factor_total_ms(),
        refac.conversion_ms,
        refac.numeric_ms,
        refac.factor_total_ms(),
        solve_ms,
    ))
}

fn main() {
    let reps: usize = std::env::var("RSLAB_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let mut cases: Vec<(String, SymTriplet)> = Vec::new();
    // Small synthetic indefinite matrices.
    for k in [8usize, 24, 64] {
        cases.push((format!("synth k={k}"), synthetic_kkt(k, 1.0)));
    }
    // Medium, and near the end of an IPM (tiny barrier diagonal).
    cases.push(("synth k=64 sigma=1e-9".into(), synthetic_kkt(64, 1e-9)));
    // Sparse-Jacobian variant — the arrow-KKT shape.
    for k in [24usize, 64] {
        cases.push((
            format!("synth sparse-J k={k}"),
            synthetic_kkt_sparse_j(k, 1.0),
        ));
    }
    // Large: n = 160 000 + 400.
    cases.push(("synth k=400 (n=160400)".into(), synthetic_kkt(400, 1.0)));
    // Real POUNCE KKT systems.
    for name in ["airport_kkt_iter0", "eigena2_kkt_iter0"] {
        if let Some(m) = load_fixture(name) {
            cases.push((format!("real {name}"), m));
        }
    }

    println!("# Phase-separated timings, milliseconds, best of {reps}");
    println!("# RSLAB rows are pounce_rslab::PhaseTimings (measured inside the adapter).");
    println!("# `materialize` is adapter overhead: L into CSC because RSLAB's supernodal");
    println!("# SolvePlan is pub(crate). Subtract it to read RSLAB's own cost.\n");

    println!(
        "{:<26} {:<10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} | {:>9}",
        "case",
        "backend",
        "convert",
        "equil",
        "symbolic",
        "numeric",
        "material",
        "TOTAL",
        "re:conv",
        "re:num",
        "re:TOTAL",
        "solve"
    );

    for (name, a) in &cases {
        let b = a.sample_rhs();

        // FERAL, through the same trait, for the totals.
        let f = compare::run_feral(a, &b, pounce_feral::FeralConfig::default());
        println!(
            "{:<26} {:<10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9.3} | {:>9} {:>9} {:>9.3} | {:>9.3}",
            name,
            "feral",
            "-",
            "-",
            "-",
            "-",
            "-",
            f.factor_ms,
            "-",
            "-",
            f.refactor_ms.unwrap_or(f64::NAN),
            f.solve_ms.unwrap_or(f64::NAN),
        );

        for (tag, cfg) in [
            ("rslab", RslabConfig::default()),
            ("rslab-sp", RslabConfig::static_pivoting(1e-12)),
        ] {
            match bench_rslab(a, &b, cfg, reps) {
                Some(row) => println!("{name:<26} {tag:<10} {row}"),
                None => println!("{name:<26} {tag:<10} (factorization failed)"),
            }
        }
        println!();
    }

    // Format conversion on its own, since the prompt asks for it separately and
    // it is the one cost the adapter fully owns.
    println!("# Format conversion in isolation (triplet -> CSC + slot map), best of {reps}");
    println!(
        "{:<26} {:>10} {:>14} {:>14}",
        "case", "nnz", "first build ms", "cached refill ms"
    );
    for (name, a) in &cases {
        let b = a.sample_rhs();
        let mut s = RslabSolverInterface::new();
        s.initialize_structure(a.n, a.nnz() as Index, &a.irn, &a.jcn);
        s.values_array_mut().copy_from_slice(&a.vals);
        let mut rhs = b.clone();
        if s.multi_solve(true, &a.irn, &a.jcn, 1, &mut rhs, false, 0) != ESymSolverStatus::Success {
            // The conversion still ran and was still timed, even though the
            // factorization that followed it did not complete.
            println!(
                "{:<26} {:>10} {:>14.3} {:>14}",
                name,
                a.nnz(),
                s.phase_timings().conversion_ms,
                "(no refactor)"
            );
            continue;
        }
        let build = s.phase_timings().conversion_ms;
        let refill = best(reps, || {
            s.values_array_mut().copy_from_slice(&a.vals);
            let mut r = b.clone();
            let _ = s.multi_solve(true, &a.irn, &a.jcn, 1, &mut r, false, 0);
            s.phase_timings().conversion_ms
        });
        println!(
            "{:<26} {:>10} {:>14.3} {:>14.3}",
            name,
            a.nnz(),
            build,
            refill
        );
    }
}
