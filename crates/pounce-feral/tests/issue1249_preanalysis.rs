//! gh#1249: FERAL's predicted fill, read before the numeric factor.
//!
//! The pre-analysis is only worth anything if it predicts the analysis the
//! solver goes on to compute, so the first test holds the two to equality on
//! every field FERAL's `WorkEstimate` feeds the factorization record. The
//! rest pin the three things the prediction can do to a factorization:
//! nothing (off, or under budget), refuse it (over budget, latched for the
//! pattern), and stop it between the phases (deadline).

use pounce_common::types::Index;
use pounce_feral::{FeralConfig, FeralSolverInterface, factor_deadline};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};

/// A 1-based lower-triangle pattern and values of an SPD matrix of order `n`
/// with fill: tridiagonal plus a dense last row (an arrow), so the factor is
/// not just the matrix.
fn arrow(n: usize) -> (Vec<Index>, Vec<Index>, Vec<f64>) {
    let (mut irn, mut jcn, mut val) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        irn.push(j as Index + 1);
        jcn.push(j as Index + 1);
        val.push(4.0 + n as f64);
        if j + 1 < n {
            irn.push(j as Index + 2);
            jcn.push(j as Index + 1);
            val.push(-1.0);
        }
        if j + 2 < n {
            irn.push(n as Index);
            jcn.push(j as Index + 1);
            val.push(0.5);
        }
    }
    (irn, jcn, val)
}

fn solve(s: &mut FeralSolverInterface, n: usize) -> ESymSolverStatus {
    let (irn, jcn, val) = arrow(n);
    assert_eq!(
        s.initialize_structure(n as Index, irn.len() as Index, &irn, &jcn),
        ESymSolverStatus::Success
    );
    s.values_array_mut().copy_from_slice(&val);
    let mut rhs = vec![1.0; n];
    s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0)
}

fn resolve_again(s: &mut FeralSolverInterface, n: usize) -> ESymSolverStatus {
    let (irn, jcn, val) = arrow(n);
    s.values_array_mut().copy_from_slice(&val);
    let mut rhs = vec![1.0; n];
    s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0)
}

fn reporting() -> FeralConfig {
    FeralConfig {
        report_symbolic: true,
        ..FeralConfig::default()
    }
}

#[test]
fn the_prediction_is_the_analysis_the_solver_computes() {
    let n = 400;
    let mut s = FeralSolverInterface::with_config(reporting());
    assert_eq!(solve(&mut s, n), ESymSolverStatus::Success);
    let rep = s.symbolic_report().expect("the pre-analysis ran").clone();
    let sum = s.summary();
    assert_eq!(rep.n, n);
    assert_eq!(Some(rep.work.peak_bytes), sum.last_peak_bytes);
    assert_eq!(Some(rep.work.n_supernodes), sum.last_n_supernodes);
    assert_eq!(Some(rep.work.max_front_rows), sum.last_max_front_rows);
    assert_eq!(Some(rep.work.factor_flops), sum.last_factor_flops);
    assert_eq!(Some(&rep.ordering), sum.last_ordering.as_ref());
    assert_eq!(Some(&rep.preprocess), sum.last_ordering_preprocess.as_ref());
    assert!(
        rep.work.factor_nnz >= rep.nnz_a,
        "nnz(L) holds A's lower triangle"
    );

    // The same pattern is not analysed again.
    assert_eq!(resolve_again(&mut s, n), ESymSolverStatus::Success);
    assert_eq!(s.symbolic_report(), Some(&rep));
}

#[test]
fn off_by_default_and_the_factor_is_unchanged_when_on() {
    let n = 300;
    let mut off = FeralSolverInterface::with_config(FeralConfig::default());
    let mut on = FeralSolverInterface::with_config(reporting());
    let (irn, jcn, val) = arrow(n);
    let mut x = Vec::new();
    for s in [&mut off, &mut on] {
        s.initialize_structure(n as Index, irn.len() as Index, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&val);
        let mut rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        x.push(rhs);
    }
    assert!(off.symbolic_report().is_none());
    assert!(on.symbolic_report().is_some());
    assert_eq!(x[0], x[1], "the pre-analysis must not change the solve");
    assert_eq!(off.summary().last_nnz_l, on.summary().last_nnz_l);
}

#[test]
fn over_budget_is_refused_for_the_whole_pattern() {
    // feral >= 0.25 counts the analysis and the workspace in the predicted
    // peak: n = 400 predicts ~235 KB and n = 4 ~9.4 KB, so 50 KB refuses the
    // first and admits the second with ~5x margin each way. (Under 0.24 the
    // budget was 1 000 bytes, which n = 4 no longer fits.)
    let n = 400;
    let mut s = FeralSolverInterface::with_config(FeralConfig {
        max_factor_bytes: Some(50_000),
        ..FeralConfig::default()
    });
    assert_eq!(solve(&mut s, n), ESymSolverStatus::FatalError);
    let rep = s
        .symbolic_report()
        .expect("a refusal still reports")
        .clone();
    assert!(rep.work.peak_bytes > 50_000);
    assert!(s.summary().last_nnz_l.is_none(), "no numeric factor ran");
    // Latched: a retry on the same pattern (e.g. after a perturbation) does
    // not go on into the factor the budget refused.
    assert_eq!(resolve_again(&mut s, n), ESymSolverStatus::FatalError);
    assert!(s.summary().last_nnz_l.is_none());
    // Within the same solve a new pattern is refused too (the refusal is
    // latched for the thread); once the next solve clears the latch, a new
    // pattern is judged afresh: this one fits.
    assert_eq!(solve(&mut s, 4), ESymSolverStatus::FatalError);
    pounce_feral::clear_budget_refusal();
    assert_eq!(solve(&mut s, 4), ESymSolverStatus::Success);
}

#[test]
fn a_generous_budget_changes_nothing() {
    let mut s = FeralSolverInterface::with_config(FeralConfig {
        max_factor_bytes: Some(u64::MAX),
        ..FeralConfig::default()
    });
    assert_eq!(solve(&mut s, 400), ESymSolverStatus::Success);
    assert!(s.symbolic_report().is_some());
}

#[test]
fn a_passed_deadline_stops_between_the_phases() {
    let n = 400;
    let mut s = FeralSolverInterface::with_config(reporting());
    {
        let _g = factor_deadline(Some(std::time::Instant::now()));
        assert_eq!(solve(&mut s, n), ESymSolverStatus::FatalError);
    }
    assert!(s.symbolic_report().is_some(), "the analysis was reported");
    assert!(s.summary().last_nnz_l.is_none(), "no numeric factor ran");
    // Outside the scope the same pattern factors, without a second analysis.
    assert_eq!(resolve_again(&mut s, n), ESymSolverStatus::Success);

    // Without the pre-analysis the deadline scope has no effect here: the
    // convex drivers check their own deadline before each factorization.
    let mut plain = FeralSolverInterface::with_config(FeralConfig::default());
    let _g = factor_deadline(Some(std::time::Instant::now()));
    assert_eq!(solve(&mut plain, n), ESymSolverStatus::Success);
}

/// gh#1265: the convex backend takes feral's default ordering at every size.
/// Until feral 0.25 it raced the orderings on a KKT of order >= 1M; feral's
/// `Auto` now picks what the race did on the models that needed it, faster.
#[test]
fn the_convex_backend_keeps_the_default_ordering_on_a_large_kkt() {
    use pounce_feral::OrderingMethod;
    if std::env::var_os("POUNCE_FERAL_ORDERING").is_some() {
        return;
    }
    let n = 1_000_000;
    let diag: Vec<Index> = (1..=n as Index).collect();
    let mut s = FeralSolverInterface::for_convex_ipm(1e-8, 0.0);
    s.initialize_structure(n as Index, n as Index, &diag, &diag);
    assert_eq!(s.ordering(), &OrderingMethod::Auto);
}

/// gh#1250 review: an over-budget refusal is terminal for the solve. While it
/// is latched, a *different* backend on the same thread (the convex fallback
/// driver, the NLP arm, restoration) refuses at once, without paying for its
/// own symbolic analysis; clearing the latch (a new top-level solve) lets the
/// next one through.
#[test]
fn a_refusal_is_terminal_until_cleared() {
    use pounce_feral::{budget_refusal, clear_budget_refusal};
    clear_budget_refusal();
    let budgeted = |bytes| {
        FeralSolverInterface::with_config(FeralConfig {
            max_factor_bytes: Some(bytes),
            ..FeralConfig::default()
        })
    };
    let mut first = budgeted(1_000);
    assert_eq!(solve(&mut first, 400), ESymSolverStatus::FatalError);
    let r = budget_refusal().expect("the refusal is latched");
    assert_eq!(r.budget, 1_000);
    assert!(r.peak_bytes > 1_000);

    // A second backend whose own pattern would fit: refused without analysis.
    let mut second = budgeted(u64::MAX);
    assert_eq!(solve(&mut second, 4), ESymSolverStatus::FatalError);
    assert!(
        second.symbolic_report().is_none(),
        "no analysis was paid for"
    );

    // A backend without a budget is not governed by the latch.
    let mut plain = FeralSolverInterface::with_config(FeralConfig::default());
    assert_eq!(solve(&mut plain, 4), ESymSolverStatus::Success);

    clear_budget_refusal();
    let mut third = budgeted(u64::MAX);
    assert_eq!(solve(&mut third, 4), ESymSolverStatus::Success);
    assert!(budget_refusal().is_none());
}
