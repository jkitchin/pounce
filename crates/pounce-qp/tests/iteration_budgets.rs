//! `QpOptions::max_total_iter` and `QpOptions::max_stall_iters`, the two
//! bounds the SQP puts on its second-order-correction QP (gh#1195).
//!
//! `max_iter` bounds one pass of the pivot loop, and a solve can run many: an
//! infeasible start goes through the l1-elastic phase 1, and when that ends
//! with slack left, through penalty escalations and recovery solves, each with
//! `max_iter` of its own. `max_total_iter` bounds the iterations of all of
//! them together. `max_stall_iters` ends one pass early once its objective has
//! stopped improving.

use pounce_common::types::NLP_UPPER_BOUND_INF;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    AntiCyclingChoice, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution,
    QpSolver, QpStatus, QpWarmStart, WorkingSet,
};

/// `min cᵀx  s.t.  A x ≥ bl,  xl ≤ x ≤ xu` with `A` given densely.
fn solve_lp(
    a_dense: &[Vec<f64>],
    bl: &[f64],
    c: &[f64],
    xl: &[f64],
    xu: &[f64],
    warm_x: Option<&[f64]>,
    opts: &QpOptions,
) -> QpSolution {
    let (m, n) = (a_dense.len(), c.len());
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for (i, row) in a_dense.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                rows.push((i + 1) as i32);
                cols.push((j + 1) as i32);
                vals.push(v);
            }
        }
    }
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, rows, cols));
    a.set_values(&vals);
    let h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, Vec::new(), Vec::new()));
    let bu = vec![NLP_UPPER_BOUND_INF; m];
    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: c,
        a: &a,
        bl,
        bu: &bu,
        xl,
        xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    // A warm start at `warm_x` with nothing in the working set.
    let ws = warm_x.map(|x| QpWarmStart {
        x: x.to_vec(),
        lambda_g: vec![0.0; m],
        lambda_x: vec![0.0; n],
        working: WorkingSet::cold(n, m),
    });
    solver.solve(&qp, ws.as_ref(), opts).expect("solve")
}

// ---- max_total_iter ---------------------------------------------------------

const N: usize = 20;
const PASS: u32 = 20;

/// The LP of `working_set_change_budget.rs`: a sine pattern `S x ≥ S·1` plus
/// the contradictory pair `Σx ≥ 1`, `−Σx ≥ 0`, in `−10 ≤ x ≤ 10`. Infeasible,
/// and infeasible at the cold start, so the solve spans several elastic
/// passes.
fn infeasible_lp(opts: &QpOptions) -> QpSolution {
    let mut a = Vec::new();
    let mut bl = Vec::new();
    for i in 0..40 {
        let row: Vec<f64> = (0..N)
            .map(|j| {
                let v = ((i * 7 + j * 13) as f64).sin();
                if v.abs() > 0.3 { v } else { 0.0 }
            })
            .collect();
        bl.push(row.iter().sum());
        a.push(row);
    }
    a.push(vec![1.0; N]);
    bl.push(1.0);
    a.push(vec![-1.0; N]);
    bl.push(0.0);
    let c: Vec<f64> = (0..N).map(|j| 1.0 + ((j * 5) as f64).cos().abs()).collect();
    solve_lp(&a, &bl, &c, &[-10.0; N], &[10.0; N], None, opts)
}

/// Both pivot loops — the refactor path and the Schur-update path — carry
/// their own copy of each check, so every test runs on both.
const PATHS: [bool; 2] = [false, true];

fn pass_options(max_total_iter: Option<u32>, use_schur_updates: bool) -> QpOptions {
    QpOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: PASS,
        max_total_iter,
        use_schur_updates,
        ..QpOptions::default()
    }
}

#[test]
fn the_iteration_budget_caps_every_elastic_pass_together() {
    for schur in PATHS {
        iteration_budget_caps_every_elastic_pass_together(schur);
    }
}

fn iteration_budget_caps_every_elastic_pass_together(schur: bool) {
    // Precondition: with only the per-pass cap the passes add up to several
    // passes' worth. Each working-set change is made inside an iteration, so
    // the change count is a floor on the iteration count.
    let free = infeasible_lp(&pass_options(None, schur));
    assert!(
        free.stats.used_phase1 && free.stats.n_working_set_changes > 2 * PASS,
        "precondition (schur={schur}): the solve no longer spans several \
         elastic passes; pick an LP that does: {:?}",
        free.stats
    );

    let capped = infeasible_lp(&pass_options(Some(PASS), schur));
    // A pivot's rank repair can drop a few rows at once (see
    // `QpOptions::max_working_set_changes`), so allow a handful.
    assert!(
        capped.stats.n_working_set_changes <= PASS + 5,
        "schur={schur}: {:?}",
        capped.stats
    );
    // Cut off by the budget, which is not a verdict on the QP.
    assert_eq!(
        capped.status,
        QpStatus::MaxIter,
        "schur={schur}: {:?}",
        capped.stats
    );
}

#[test]
fn an_iteration_budget_the_solve_does_not_reach_changes_nothing() {
    for schur in PATHS {
        let free = infeasible_lp(&pass_options(None, schur));
        let roomy = infeasible_lp(&pass_options(Some(1_000_000), schur));
        assert_eq!(free.status, roomy.status, "schur={schur}");
        assert_eq!(free.x, roomy.x, "schur={schur}");
        assert_eq!(
            free.stats.n_working_set_changes, roomy.stats.n_working_set_changes,
            "schur={schur}"
        );
    }
}

// ---- max_stall_iters --------------------------------------------------------

/// Beale's degenerate LP (1955), written `A x ≥ bl`:
///
/// ```text
/// min  −¾x₁ + 20x₂ − ½x₃ + 6x₄
/// s.t. −¼x₁ +  8x₂ +  x₃ − 9x₄ ≥ 0
///      −½x₁ + 12x₂ + ½x₃ − 3x₄ ≥ 0
///                    −x₃       ≥ −1
///      x ≥ 0
/// ```
///
/// Solved warm from `x = 0` with an empty working set: a feasible, degenerate
/// vertex with four of its rows and bounds active at zero. The first steps
/// from there (inertia-shifted, an LP's Hessian being zero) are each blocked
/// almost at once, `α` between 1e-23 and 1e-18, so the working set grows while
/// the objective moves by round-off only — and on the fourth step, the
/// wrong way: −5.4e-10 → −3.3e-10. One drop later the solve reaches the
/// optimum, −5/4 at `x = (1, 0, 1, 0)`.
fn beale(opts: &QpOptions) -> QpSolution {
    let a = vec![
        vec![-0.25, 8.0, 1.0, -9.0],
        vec![-0.5, 12.0, 0.5, -3.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let inf = NLP_UPPER_BOUND_INF;
    solve_lp(
        &a,
        &[0.0, 0.0, -1.0],
        &[-0.75, 20.0, -0.5, 6.0],
        &[0.0; 4],
        &[inf; 4],
        Some(&[0.0; 4]),
        opts,
    )
}

fn stall_options(max_stall_iters: Option<u32>, use_schur_updates: bool) -> QpOptions {
    QpOptions {
        max_stall_iters,
        use_schur_updates,
        ..QpOptions::default()
    }
}

#[test]
fn beale_solves_without_a_stall_cap() {
    for schur in PATHS {
        let sol = beale(&stall_options(None, schur));
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "schur={schur}: {:?}",
            sol.stats
        );
        assert!(
            (sol.obj + 1.25).abs() < 1e-9,
            "schur={schur}: obj {}",
            sol.obj
        );
    }
}

#[test]
fn a_pass_that_stops_improving_ends_as_on_max_iter() {
    for schur in PATHS {
        a_pass_that_stops_improving_ends(schur);
    }
}

fn a_pass_that_stops_improving_ends(schur: bool) {
    let sol = beale(&stall_options(Some(1), schur));
    // The fourth step does not improve on the third; with a cap of 1 the
    // pass ends there, and is reported as a pass that ran out, not as a
    // verdict.
    assert_eq!(
        sol.status,
        QpStatus::MaxIter,
        "schur={schur}: {:?}",
        sol.stats
    );
    assert!(sol.obj > -1.25 + 1e-6, "schur={schur}: obj {}", sol.obj);
}

#[test]
fn a_stall_cap_the_solve_does_not_reach_changes_nothing() {
    for schur in PATHS {
        let free = beale(&stall_options(None, schur));
        let roomy = beale(&stall_options(Some(1_000), schur));
        assert_eq!(free.status, roomy.status, "schur={schur}");
        assert_eq!(free.x, roomy.x, "schur={schur}");
    }
}

#[test]
fn on_one_pass_a_total_budget_is_max_iter() {
    // Beale from its feasible vertex is a single pass, so a solve-wide budget
    // of `b` iterations must end exactly where a per-pass `max_iter = b` does:
    // every iteration within the budget runs to completion, and the solve
    // stops at the top of the first one past it.
    for schur in PATHS {
        let mut reached_optimal = false;
        for b in 1..=30 {
            let per_pass = beale(&QpOptions {
                max_iter: b,
                use_schur_updates: schur,
                ..QpOptions::default()
            });
            let total = beale(&QpOptions {
                max_total_iter: Some(b),
                use_schur_updates: schur,
                ..QpOptions::default()
            });
            assert_eq!(per_pass.status, total.status, "schur={schur} b={b}");
            // The same iterate, except that a solve cut off by a budget
            // reports it clamped into the variable box (`cut_off_solution`),
            // where a pass out of `max_iter` hands it back as it stood, a few
            // ulps outside `x ≥ 0`.
            let clamped: Vec<f64> = per_pass.x.iter().map(|v| v.max(0.0)).collect();
            let reported = if total.status == QpStatus::MaxIter {
                &clamped
            } else {
                &per_pass.x
            };
            assert_eq!(reported, &total.x, "schur={schur} b={b}");
            reached_optimal |= total.status == QpStatus::Optimal;
        }
        assert!(
            reached_optimal,
            "schur={schur}: 30 iterations no longer solve Beale"
        );
    }
}
