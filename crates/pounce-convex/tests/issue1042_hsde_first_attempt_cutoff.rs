//! gh#1042 R1d: the first, unequilibrated HSDE attempt of a QP solve is cut
//! short once it has provably frozen (see `pounce_convex::hsde_cutoff`), and
//! the gh #293 equilibrated retry runs sooner.
//!
//! The data are the `QpProblem`s the CLI hands the convex arm for
//! `crates/pounce-cli/tests/fixtures/scaled_feasible_a.nl` and
//! `scaled_feasible_b.nl` (dumped from `solve_qp_ipm` on `a18af517`, bounds
//! still unexpanded, objective constant as the CLI passes it).
//!
//! Measured on `a18af517` (x86_64): `scaled_feasible_a`'s first attempt ran
//! all 200 iterations to `true_kkt_error 0.148` (`IterationLimit`), then the
//! retry ran 200 more and certified — 400 HSDE iterations for a reported 200.
//! With the cut: 36 + 200 = 236, same retry, same answer.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, hsde_cutoff, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn diag2(n: usize) -> Vec<Triplet> {
    (0..n).map(|i| Triplet::new(i, i, 2.0)).collect()
}

/// `scaled_feasible_a`; caller objective = `obj + OBJ_CONSTANT_A`, optimum 0.
const OBJ_CONSTANT_A: f64 = 499999000015.74725;
fn scaled_feasible_a() -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: diag2(4),
        c: vec![7.6150526394493205, -1000000.0, -1.0, -999998.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -100000000.0),
            Triplet::new(0, 1, 100000000.0),
            Triplet::new(0, 2, 100000000.0),
            Triplet::new(0, 3, -100000000.0),
            Triplet::new(1, 1, -46903904.763146296),
            Triplet::new(1, 3, 100000000.0),
        ],
        h: vec![530752631.97246605, 26547947618426.85],
        lb: vec![-3.8075263202246603, 0.0, 0.0, -1.0],
        ub: vec![
            -3.8075263192246602,
            500057.6278671611,
            1.0,
            500056.1278671606,
        ],
    }
}

/// `scaled_feasible_b`; caller objective = `obj + OBJ_CONSTANT_B`, optimum 0.
const OBJ_CONSTANT_B: f64 = 500000000085.1919;
fn scaled_feasible_b() -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: diag2(4),
        c: vec![-1000000.0, 18.432787790440948, -1.0, -1000000.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -100000000.0),
            Triplet::new(0, 1, 100000000.0),
            Triplet::new(0, 2, 100000000.0),
            Triplet::new(0, 3, 100000000.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(1, 3, -1.0),
            Triplet::new(2, 0, 100000000.0),
            Triplet::new(2, 3, -100000000.0),
        ],
        h: vec![-871639389.5220464, -999989.7836061048, 1e-12],
        lb: vec![0.0, -9.716393895220474, 0.0, 0.0],
        ub: vec![1000000.0, -8.716393895220474, 1.0, 1000000.0],
    }
}

fn opts(obj_constant: f64) -> QpOptions {
    QpOptions {
        obj_constant,
        ..QpOptions::default()
    }
}

#[test]
fn the_frozen_first_attempt_is_cut_and_the_retry_certifies() {
    let prob = scaled_feasible_a();
    hsde_cutoff::clear();
    let sol = solve_qp_ipm(&prob, &opts(OBJ_CONSTANT_A), backend);
    let cut = hsde_cutoff::take();

    // The answer is the retry's, exactly as before the cut existed.
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "{:?} after {}",
        sol.status,
        sol.iters
    );
    let caller_obj = sol.obj + OBJ_CONSTANT_A;
    assert!(caller_obj.abs() <= 1e-3, "caller objective {caller_obj:e}");
    let kkt = sol.kkt_residuals(&prob).kkt_error();
    assert!(kkt < 1e-8, "kkt_error {kkt:e}");
    // The reported count is the retry's alone and does not move (200 on
    // `a18af517`, before and after).
    assert_eq!(sol.iters, 200, "reported (retry) iteration count moved");

    // The first attempt was cut — the mutation "detector off" fails here.
    let cut = cut.expect("the first attempt ran to its cap: the R1d detector did not fire");
    assert_eq!(cut.max_iter, 200);
    assert!(
        cut.iter <= 60,
        "the first attempt was cut only at iteration {} (measured 36)",
        cut.iter
    );
    assert!(cut.tau < 1e-6 && cut.true_kkt > 1e-5, "{cut:?}");
    assert!(
        cut.iter + sol.iters <= 260,
        "total HSDE iterations {} + {} > 260 (was 200 + 200)",
        cut.iter,
        sol.iters
    );
}

#[test]
fn a_first_attempt_that_converges_is_untouched() {
    // `scaled_feasible_b` is the measured control: its first attempt spends
    // windows inside the `τ < 1e-6`, `μ < tol²` gate and then converges at
    // 36 — the rule must leave it alone. A control, not a branch test: both
    // the frozen-product and the `τ`-growth clause exclude it (spread ≥ 1.4e9,
    // `τ` flat), so no single-clause mutation turns it red. The clause that
    // is load-bearing on a recovering loop is pinned in `ipm.rs`
    // (`hsde_cutoff_recovering_loop_tests`).
    let prob = scaled_feasible_b();
    hsde_cutoff::clear();
    let sol = solve_qp_ipm(&prob, &opts(OBJ_CONSTANT_B), backend);
    assert_eq!(
        hsde_cutoff::take(),
        None,
        "the detector fired on a converging attempt"
    );
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.iters, 36, "scaled_feasible_b's single attempt moved");
    assert!((sol.obj + OBJ_CONSTANT_B).abs() <= 1e-3);
}

#[test]
fn a_cut_whose_retry_is_rejected_reports_the_uncut_first_attempt() {
    // At `max_iter = 100` the cut still fires (even sooner: less budget to
    // reach the target), but the retry — which needs 165 iterations on this
    // model — cannot certify and is rejected. The first attempt's honest
    // answer then stands, and it must be the *uncut* run's: re-run, it
    // reports all 100 iterations, not the iteration it was cut at.
    // Mutation "no re-run after a rejected retry" fails on the count.
    let prob = scaled_feasible_a();
    let o = QpOptions {
        max_iter: 100,
        ..opts(OBJ_CONSTANT_A)
    };
    hsde_cutoff::clear();
    let sol = solve_qp_ipm(&prob, &o, backend);
    let cut = hsde_cutoff::take().expect("the detector did not fire at max_iter = 100");
    assert!(cut.iter < 100, "{cut:?}");
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "the retry was expected to be rejected"
    );
    assert_eq!(
        sol.iters, 100,
        "a rejected retry must leave the uncut first attempt's result (status {:?})",
        sol.status
    );
}
