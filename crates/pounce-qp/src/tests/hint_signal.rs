//! gh #602 / #434 — is there a runtime signal for "is this working set worth
//! keeping"?
//!
//! Three separate decisions in this crate want the same missing thing:
//!
//! * #434: should a homotopy path be abandoned mid-trace?
//! * #602 step 1: when `solve_parametric` declines, is the previous working set
//!   better than a cold solve?
//! * #602 step 2: for a pair the path cannot model, is tracing it anyway better
//!   than the working-set hint? (declined for want of exactly this)
//!
//! #434 refuted the obvious *predictor* — `n_eq / n`, computable from the
//! problem data with no solve — and its conclusion was that the discriminator,
//! if one exists, has to be **measured rather than predicted**. This module is
//! the instrument for one candidate measurement:
//! [`ParametricActiveSetSolver::hint_pin_quality`], which pins the hint and
//! counts the rows and bounds outside it that the pinned point violates.
//!
//! The harness below is not an assertion, it is a table. Run it with
//!
//! ```text
//! cargo test -p pounce-qp --lib hint_signal -- --ignored --nocapture
//! ```
//!
//! and read the separation summary at the bottom. It is `#[ignore]`d because it
//! is a measurement, not a property: nothing here should fail CI, and a
//! threshold read off it is worth only as much as its validation on data it was
//! not derived from (`benchmarks/warmstart`).

use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::solver::{ParametricActiveSetSolver, QpSolver};
use crate::working_set::WorkingSet;
use pounce_common::Number;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

fn backend() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

fn pr(k: usize) -> Number {
    let s = ((k as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407))
        >> 33;
    ((s % 2000) as f64) / 1000.0 - 1.0
}

struct Data {
    n: usize,
    m: usize,
    h: SymTMatrix,
    a: GenTMatrix,
    g: Vec<Number>,
    bl: Vec<Number>,
    bu: Vec<Number>,
    xl: Vec<Number>,
    xu: Vec<Number>,
}

#[derive(Clone, Copy)]
struct Change {
    dh: Number,
    da: Number,
    dg: Number,
    db: Number,
    xu_cap: Option<Number>,
}

fn data(n: usize, m: usize, c: Change) -> Data {
    let (mut hi, mut hj, mut hv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        hi.push((i + 1) as i32);
        hj.push((i + 1) as i32);
        hv.push((1.0 + (i as f64) / (n as f64)) * (1.0 + c.dh * pr(i + 907)));
    }
    let hs = SymTMatrixSpace::new(n as i32, hi, hj);
    let mut h = SymTMatrix::new(hs);
    h.set_values(&hv);

    let (mut ai, mut aj, mut av) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..m {
        for k in 0..4 {
            ai.push((i + 1) as i32);
            aj.push(((i * 7 + k * 5) % n + 1) as i32);
            av.push((0.5 + pr(i * 13 + k).abs()) * (1.0 + c.da * pr(i * 31 + k + 7)));
        }
    }
    let asp = GenTMatrixSpace::new(m as i32, n as i32, ai, aj);
    let mut a = GenTMatrix::new(asp);
    a.set_values(&av);

    Data {
        n,
        m,
        h,
        a,
        g: (0..n)
            .map(|j| -2.0 - pr(j).abs() + c.dg * pr(j + 501))
            .collect(),
        bl: vec![NLP_LOWER_BOUND_INF; m],
        bu: (0..m)
            .map(|i| 1.0 + 0.5 * pr(i + 101).abs() + c.db * pr(i + 601))
            .collect(),
        xl: vec![0.0; n],
        xu: match c.xu_cap {
            Some(v) => (0..n)
                .map(|j| if j % 3 == 0 { v } else { 10.0 * v })
                .collect(),
            None => vec![NLP_UPPER_BOUND_INF; n],
        },
    }
}

fn qp(d: &Data) -> QpProblem<'_> {
    QpProblem {
        n: d.n,
        m: d.m,
        h: &d.h,
        g: &d.g,
        a: &d.a,
        bl: &d.bl,
        bu: &d.bu,
        xl: &d.xl,
        xu: &d.xu,
        hessian_inertia: HessianInertia::Psd,
    }
}

/// One measured pair: the signal, and the ground truth it would have to predict.
struct Sample {
    label: String,
    active: usize,
    violated: usize,
    /// Working-set changes actually taken by each route on the new problem.
    ws_cost: u32,
    cold_cost: u32,
}

impl Sample {
    /// The candidate signal: violated rows relative to the size of the hint.
    fn ratio(&self) -> f64 {
        self.violated as f64 / (self.active.max(1) as f64)
    }
    /// Ground truth: was keeping the working set the right call?
    fn ws_was_right(&self) -> bool {
        self.ws_cost <= self.cold_cost
    }
}

fn measure(label: String, prev: &Data, new: &Data, homotopy: bool, out: &mut Vec<Sample>) {
    // Which cold path the fallback would actually take depends on the caller's
    // options, and the two are very different opponents: `pounce-qp` defaults
    // `use_homotopy` off, `pounce-convex`'s driver turns it on. Both arms are
    // swept, because a rule that helps against one may be pointless against the
    // other.
    let opts = QpOptions {
        use_homotopy: homotopy,
        ..QpOptions::default()
    };
    let mut s = backend();
    let Ok(sol_prev) = s.solve(&qp(prev), None, &opts) else {
        return;
    };
    if sol_prev.status != crate::error::QpStatus::Optimal {
        return;
    }
    let hint: WorkingSet = sol_prev.working.reconciled_with(&qp(new), &opts);

    let Some(q) = backend().hint_pin_quality(&qp(new), &hint, &opts) else {
        // A hint too broken to measure. Recorded with a sentinel so the summary
        // can say how often that happens rather than silently dropping it.
        out.push(Sample {
            label,
            active: 0,
            violated: usize::MAX,
            ws_cost: 0,
            cold_cost: 0,
        });
        return;
    };

    let Ok(ws) = backend().solve_with_working_set(&qp(new), &hint, &opts) else {
        return;
    };
    let Ok(cold) = backend().solve(&qp(new), None, &opts) else {
        return;
    };
    out.push(Sample {
        label,
        active: q.active,
        violated: q.violated,
        ws_cost: ws.stats.n_working_set_changes,
        cold_cost: cold.stats.n_working_set_changes,
    });
}

#[test]
#[ignore = "measurement harness, not a property; see module docs"]
fn hint_signal_sweep() {
    for homotopy in [false, true] {
        println!(
            "\n################ cold arm = {} ################",
            if homotopy { "homotopy" } else { "conventional" }
        );
        sweep(homotopy);
    }
}

fn sweep(homotopy: bool) {
    const BASE: Change = Change {
        dh: 0.0,
        da: 0.0,
        dg: 0.15,
        db: 0.2,
        xu_cap: None,
    };
    let mut samples = Vec::new();

    for (n, m) in [(12, 8), (20, 14), (30, 20), (40, 28), (50, 34)] {
        let prev = data(
            n,
            m,
            Change {
                dg: 0.0,
                db: 0.0,
                ..BASE
            },
        );
        for dh in [0.0, 0.05, 0.5] {
            for da in [0.0, 0.05, 0.1, 0.3, 0.6, 1.0] {
                for (dg, db) in [(0.15, 0.2), (1.5, 1.0)] {
                    for xu_cap in [None, Some(0.5)] {
                        let c = Change {
                            dh,
                            da,
                            dg,
                            db,
                            xu_cap,
                        };
                        let cap = xu_cap.map_or("-".to_string(), |v| format!("{v}"));
                        measure(
                            format!("n{n} dh{dh} da{da} dg{dg} xu{cap}"),
                            &prev,
                            &data(n, m, c),
                            homotopy,
                            &mut samples,
                        );
                    }
                }
            }
        }
    }

    let broken = samples.iter().filter(|s| s.violated == usize::MAX).count();
    samples.retain(|s| s.violated != usize::MAX);
    println!("\n{} samples ({broken} unmeasurable pins)\n", samples.len());

    // Only the rows where the routes actually disagree are worth printing; the
    // rest are noise in a 360-row table.
    println!(
        "{:<34} {:>6} {:>6} {:>7} {:>8} {:>8}  {}",
        "case (routes differ)", "active", "viol", "ratio", "ws_cost", "cold", "ws better?"
    );
    for s in samples.iter().filter(|s| s.ws_cost != s.cold_cost) {
        println!(
            "{:<34} {:>6} {:>6} {:>7.3} {:>8} {:>8}  {}",
            s.label,
            s.active,
            s.violated,
            s.ratio(),
            s.ws_cost,
            s.cold_cost,
            if s.ws_was_right() { "yes" } else { "NO" }
        );
    }

    // Does any threshold on the ratio separate "keep the hint" from "go cold"?
    // Scored the way a rule would actually run: predict `keep` when the ratio is
    // at or below the threshold, and count where that prediction is wrong.
    println!("\nseparation: predict `keep the hint` when violated/active <= t");
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "t", "kept", "wrong", "cold", "wrong", "total_bad"
    );
    let total_right = samples.iter().filter(|s| s.ws_was_right()).count();
    println!(
        "  (baseline: always keep -> {} wrong of {})",
        samples.len() - total_right,
        samples.len()
    );
    for t in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 1e9] {
        let (mut kept, mut kept_wrong, mut cold, mut cold_wrong) = (0, 0, 0, 0);
        for s in &samples {
            if s.ratio() <= t {
                kept += 1;
                if !s.ws_was_right() {
                    kept_wrong += 1;
                }
            } else {
                cold += 1;
                if s.ws_was_right() {
                    cold_wrong += 1;
                }
            }
        }
        println!(
            "{t:>8} {kept:>8} {kept_wrong:>8} {cold:>8} {cold_wrong:>8} {:>10}",
            kept_wrong + cold_wrong
        );
    }

    // The ratio is one normalization; the raw count is the other, and a
    // negative result on one candidate says nothing about the other. Scored the
    // same way.
    println!("\nseparation: predict `keep the hint` when violated <= k");
    for k in [0usize, 1, 2, 3, 5, 8, 12, 20, usize::MAX] {
        let (mut kept_wrong, mut cold_wrong) = (0, 0);
        let mut spent: u32 = 0;
        for s in &samples {
            if s.violated <= k {
                spent += s.ws_cost;
                if !s.ws_was_right() {
                    kept_wrong += 1;
                }
            } else {
                spent += s.cold_cost;
                if s.ws_was_right() {
                    cold_wrong += 1;
                }
            }
        }
        println!(
            "  k={k:<12} wrong {:>4}   cost {spent}",
            kept_wrong + cold_wrong
        );
    }

    // A direct refutation is worth more than a threshold table: if two samples
    // carry the same signal and disagree on the answer, no rule reading only
    // that signal can be right about both.
    let mut by_signal: std::collections::HashMap<(usize, usize), Vec<&Sample>> =
        std::collections::HashMap::new();
    for s in &samples {
        by_signal.entry((s.active, s.violated)).or_default().push(s);
    }
    let mut collisions = 0;
    let mut shown = 0;
    for (_, group) in by_signal.iter() {
        if group.iter().any(|s| s.ws_was_right()) && group.iter().any(|s| !s.ws_was_right()) {
            collisions += 1;
            if shown < 3 {
                shown += 1;
                let yes = group.iter().find(|s| s.ws_was_right()).unwrap();
                let no = group.iter().find(|s| !s.ws_was_right()).unwrap();
                println!(
                    "\ncollision: active={} violated={} — same signal, opposite answers",
                    yes.active, yes.violated
                );
                println!(
                    "   {:<34} ws {:>4} cold {:>4}  keep",
                    yes.label, yes.ws_cost, yes.cold_cost
                );
                println!(
                    "   {:<34} ws {:>4} cold {:>4}  go cold",
                    no.label, no.ws_cost, no.cold_cost
                );
            }
        }
    }
    println!(
        "\n{collisions} of {} distinct (active, violated) signal values carry both answers",
        by_signal.len()
    );

    // The decision only matters where the two routes actually differ, so score
    // the *work* too: total working-set changes a rule would spend.
    println!("\ncost: total working-set changes spent under each policy");
    let always_ws: u32 = samples.iter().map(|s| s.ws_cost).sum();
    let always_cold: u32 = samples.iter().map(|s| s.cold_cost).sum();
    let oracle: u32 = samples.iter().map(|s| s.ws_cost.min(s.cold_cost)).sum();
    println!("  always keep hint : {always_ws}");
    println!("  always cold      : {always_cold}");
    println!("  oracle (best)    : {oracle}");
    for t in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0] {
        let spent: u32 = samples
            .iter()
            .map(|s| {
                if s.ratio() <= t {
                    s.ws_cost
                } else {
                    s.cold_cost
                }
            })
            .sum();
        println!("  rule t={t:<5}      : {spent}");
    }
}
