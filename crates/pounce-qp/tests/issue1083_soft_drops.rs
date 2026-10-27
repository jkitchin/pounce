//! gh#1083 item 1: a drop on the refactor path refactors on the same pattern.
//!
//! Every working-set change on the refactor path costs a factorization, and on
//! the reporter's 26 072-dimension KKT about 55 % of one is FERAL's symbolic
//! analysis, paid afresh whenever the sparsity pattern changes. It changed on
//! every change. A dropped row now stays in the pattern as an inert sentinel
//! until the next add of a row the pattern does not hold, so a drop is a
//! values-only refactor and the backend's pattern-fingerprint cache reuses its
//! symbolic analysis (measured there: 0.59 s → 0.24 s per drop).
//!
//! The diagonal entries the sentinels need are added with the first one, so a
//! pattern with every row live is assembled exactly as before, and the first
//! drop after each rebuild is the one that pays the symbolic analysis: an
//! explicit zero on a live row's diagonal is not inert to FERAL (it cost a
//! fixture its elastic phase 1, see `assemble_soft_active_set_kkt`).
//!
//! Soft drops apply from `n + m ≥ SOFT_DROP_MIN_DIM` only, so the first two
//! tests pad the QP to that size with bounds that stay active.
//!
//! What is asserted is the pattern itself, recorded at every factorization by
//! a pass-through backend, because that is the property the saving rests on
//! and it is backend-independent; the time it saves is FERAL's to deliver.
//! The answers are held to the Schur path's, which assembles nothing of this.

use pounce_common::types::{Index, Number};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_qp::{
    BoundStatus, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution,
    QpSolver, QpStatus, QpWarmStart, SOFT_DROP_MIN_DIM, WorkingSet,
};
use std::sync::{Arc, Mutex};

type Pattern = (Vec<Index>, Vec<Index>);

/// FERAL, recording the triplet pattern of every factorization.
struct Recording {
    inner: FeralSolverInterface,
    current: Pattern,
    factored: Arc<Mutex<Vec<Pattern>>>,
    /// `Some(n)`: report `Singular` for every factorization whose pattern
    /// carries a diagonal entry past the `n` H rows, which only a pattern
    /// holding sentinels does.
    fail_sentinels: Option<usize>,
}

/// Whether `p` carries a diagonal entry past the first `n` rows.
fn has_sentinel_diagonal(p: &Pattern, n: usize) -> bool {
    p.0.iter()
        .zip(&p.1)
        .any(|(&i, &j)| i == j && i as usize > n)
}

impl SparseSymLinearSolverInterface for Recording {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        self.current = (ia.to_vec(), ja.to_vec());
        self.inner.initialize_structure(dim, nonzeros, ia, ja)
    }
    fn values_array_mut(&mut self) -> &mut [Number] {
        self.inner.values_array_mut()
    }
    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        if new_matrix {
            self.factored.lock().unwrap().push(self.current.clone());
            if self
                .fail_sentinels
                .is_some_and(|n| has_sentinel_diagonal(&self.current, n))
            {
                return ESymSolverStatus::Singular;
            }
        }
        self.inner.multi_solve(
            new_matrix,
            ia,
            ja,
            nrhs,
            rhs_vals,
            check_neg_evals,
            number_of_neg_evals,
        )
    }
    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }
    fn matrix_format(&self) -> EMatrixFormat {
        self.inner.matrix_format()
    }
}

/// `min ½‖x − c‖²` over `0 ≤ x ≤ 10` and `Σx ≤ cap`, warm-started at `x = 0`
/// with every bound at its lower end. The bounds with `c_i > 0` are wrong-sign
/// there and leave one at a time; those with `c_i < 0` stay, carrying `−c_i`.
/// With `cap` large nothing else ever blocks, so the run is drops only; with
/// `cap` binding, the general row is added part-way through.
fn solve(c: &[Number], cap: Number, schur: bool) -> (QpSolution, Vec<Pattern>) {
    solve_with(c, cap, schur, false)
}

/// [`solve`], with the backend failing every sentinel pattern when `fail`.
fn solve_with(c: &[Number], cap: Number, schur: bool, fail: bool) -> (QpSolution, Vec<Pattern>) {
    let n = c.len();
    let idx: Vec<Index> = (1..=n as Index).collect();
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(n as Index, idx.clone(), idx.clone()));
    h.set_values(&vec![1.0; n]);
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(1, n as Index, vec![1; n], idx));
    a.set_values(&vec![1.0; n]);
    let g: Vec<Number> = c.iter().map(|ci| -ci).collect();
    let (bl, bu) = (vec![-1e20], vec![cap]);
    let (xl, xu) = (vec![0.0; n], vec![10.0; n]);
    let qp = QpProblem {
        n,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut working = WorkingSet::cold(n, 1);
    working.bounds.fill(BoundStatus::AtLower);
    let ws = QpWarmStart {
        x: vec![0.0; n],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0; n],
        working,
    };
    let factored = Arc::new(Mutex::new(Vec::new()));
    let mut solver = ParametricActiveSetSolver::new(Box::new(Recording {
        inner: FeralSolverInterface::new(),
        current: (Vec::new(), Vec::new()),
        factored: Arc::clone(&factored),
        fail_sentinels: fail.then_some(n),
    }));
    let opts = QpOptions {
        use_schur_updates: schur,
        use_homotopy: false,
        ..QpOptions::default()
    };
    let sol = solver.solve(&qp, Some(&ws), &opts).expect("solve");
    let patterns = factored.lock().unwrap().clone();
    (sol, patterns)
}

/// How many factorizations were on a pattern the one before had not used.
fn pattern_changes(patterns: &[Pattern]) -> usize {
    patterns.windows(2).filter(|w| w[0] != w[1]).count()
}

fn assert_same_answer(soft: &QpSolution, schur: &QpSolution) {
    assert_eq!(soft.status, QpStatus::Optimal);
    assert_eq!(schur.status, QpStatus::Optimal);
    for (name, a, b) in [
        ("x", &soft.x, &schur.x),
        ("lambda_x", &soft.lambda_x, &schur.lambda_x),
        ("lambda_g", &soft.lambda_g, &schur.lambda_g),
    ] {
        for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (u - v).abs() <= 1e-9,
                "{name}[{i}]: refactor {u} vs Schur {v}"
            );
        }
    }
    assert_eq!(soft.working, schur.working);
}

const C: [Number; 7] = [5.0, 1.0, 4.0, -2.0, 3.0, -0.5, 2.0];

/// `C`, padded with `c_i = −1` (bounds that stay) to `n` variables, so the QP
/// (`n + 1` rows and columns) reaches [`SOFT_DROP_MIN_DIM`] when asked to.
fn padded(n: usize) -> Vec<Number> {
    let mut c = C.to_vec();
    c.resize(n.max(C.len()), -1.0);
    c
}

#[test]
fn drops_refactor_on_one_pattern() {
    let c = padded(SOFT_DROP_MIN_DIM);
    let (sol, patterns) = solve(&c, 1e3, false);
    let drops = C.iter().filter(|&&c| c > 0.0).count();
    assert_eq!(sol.stats.n_working_set_changes as usize, drops);
    // One factorization of the warm start, one per drop: the step after
    // each drop is unblocked, so its stationary check back-solves.
    assert_eq!(patterns.len(), drops + 1, "factorizations");
    // Before gh#1083 each of these was a pattern of its own (`drops`). The
    // first drop still moves it once: the diagonal entries the sentinels
    // need arrive with the first sentinel, so a pattern with every row live
    // is assembled exactly as it was.
    assert_eq!(pattern_changes(&patterns), 1);

    let (schur, _) = solve(&c, 1e3, true);
    assert_same_answer(&sol, &schur);
    for (i, &c) in c.iter().enumerate() {
        let (x, z) = if c > 0.0 { (c, 0.0) } else { (0.0, -c) };
        assert!((sol.x[i] - x).abs() <= 1e-12, "x[{i}] = {}", sol.x[i]);
        assert!((sol.lambda_x[i] - z).abs() <= 1e-12, "lambda_x[{i}]");
    }
}

/// `Σx ≤ 8` binds after the first two drops (5 + 4 > 8), so the run adds a row
/// the pattern does not hold. The pattern moves three times — at the first
/// drop, at that add, and at the first drop after it — where before gh#1083
/// it moved at every one of the five factorizations after the first. The
/// multiplier of the added row is read from its position in a pattern that
/// holds sentinels again by the end.
#[test]
fn an_add_starts_the_next_pattern() {
    let c = padded(SOFT_DROP_MIN_DIM);
    let (sol, patterns) = solve(&c, 8.0, false);
    let (schur, _) = solve(&c, 8.0, true);
    assert_same_answer(&sol, &schur);
    assert!(sol.working.constraints[0].is_active(), "the cap binds");
    assert!(sol.lambda_g[0].abs() > 1e-3, "and carries a multiplier");
    assert_eq!(patterns.len(), 6, "factorizations");
    assert_eq!(pattern_changes(&patterns), 3);
}

/// Below [`SOFT_DROP_MIN_DIM`] nothing changed: every drop is a pattern of its
/// own, as before gh#1083. The fixture corpus is all below it, and swept with
/// soft drops at every size it moved 12 fixture-legs, two to a worse verdict.
#[test]
fn small_qps_keep_the_compact_pattern() {
    let (sol, patterns) = solve(&C, 1e3, false);
    let drops = C.iter().filter(|&&c| c > 0.0).count();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(patterns.len(), drops + 1, "factorizations");
    assert_eq!(pattern_changes(&patterns), drops);
}

/// A sentinel pattern the backend will not factor costs one inertia ladder per
/// pass, not one per drop: the first failure sends the rest of the pass to the
/// compact path, and the answer is the compact path's.
#[test]
fn a_failing_sentinel_pattern_falls_back_once_per_pass() {
    let c = padded(SOFT_DROP_MIN_DIM);
    let (sol, patterns) = solve_with(&c, 1e3, false, true);
    let (schur, _) = solve(&c, 1e3, true);
    assert_same_answer(&sol, &schur);

    let n = c.len();
    let soft: Vec<bool> = patterns
        .iter()
        .map(|p| has_sentinel_diagonal(p, n))
        .collect();
    let attempts = soft.iter().filter(|&&s| s).count();
    let runs = soft.windows(2).filter(|w| !w[0] && w[1]).count();
    assert!(attempts >= 1, "the first drop tried a sentinel pattern");
    assert_eq!(runs, 1, "and no later drop did: {soft:?}");
    let ladder = 1 + QpOptions::default().inertia_max_shifts as usize;
    assert!(
        attempts <= ladder,
        "{attempts} attempts, one ladder is {ladder}"
    );
}
