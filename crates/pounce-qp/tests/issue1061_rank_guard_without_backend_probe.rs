//! gh#1061: the linear-independence guard must not depend on `linear_solver`.
//!
//! The guard had two implementations: the backend's sparse rank-reveal when it
//! offered one (FERAL), and dense modified Gram-Schmidt otherwise — one
//! length-`n` normal per active row, `O(k·n)` memory and `O(k²·n)` time. MA57
//! keeps the trait default `provides_degeneracy_detection() == false`, so a
//! user who picked MA57 for the interior point got the dense guard in every
//! SQP QP; on an NLP with `n ≈ 15 000` and `≈ 10 800` equalities a single pass
//! is gigabytes and `~10¹²` flops, and the first cold QP did not finish in
//! 30 minutes. The cold-solve equality prune (gh#971) was skipped outright on
//! such a backend for the same reason.
//!
//! `NoProbe` below is that backend shape: FERAL for every factorization, the
//! trait default for the rank-reveal. HSL is not linked here, so this is the
//! stand-in; what it pins is that `LinearSolver` answers the rank question
//! itself, whoever owns the KKT factor.

use std::rc::Rc;
use std::time::Duration;

use pounce_common::types::{Index, Number};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_qp::{
    HessianInertia, LinearSolver, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver,
    QpStatus,
};

const POS_INF: f64 = 1e20;
const NEG_INF: f64 = -1e20;

/// A KKT backend without a rank-reveal of its own, as MA57 is: every method
/// the trait requires forwards to FERAL, and `provides_degeneracy_detection` /
/// `determine_dependent_rows` keep the trait defaults (`false` / `FatalError`).
struct NoProbe(FeralSolverInterface);

impl SparseSymLinearSolverInterface for NoProbe {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        self.0.initialize_structure(dim, nonzeros, ia, ja)
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        self.0.values_array_mut()
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
        self.0.multi_solve(
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
        self.0.number_of_neg_evals()
    }

    fn increase_quality(&mut self) -> bool {
        self.0.increase_quality()
    }

    fn provides_inertia(&self) -> bool {
        self.0.provides_inertia()
    }

    fn matrix_format(&self) -> EMatrixFormat {
        self.0.matrix_format()
    }
}

fn no_probe() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(NoProbe(FeralSolverInterface::new()))
}

/// The wrapper answers the rank question for a backend that cannot. Before
/// the fix this was the trait default's `FatalError`, and the QP fell back to
/// dense Gram-Schmidt.
#[test]
fn linear_solver_rank_reveals_for_a_backend_without_a_probe() {
    let mut ls = LinearSolver::new(no_probe());
    assert!(ls.provides_degeneracy_detection());
    // J = [1 1 0; 0 1 1; 1 2 1]: row 3 = row 1 + row 2, rank 2.
    let irn = [1, 1, 2, 2, 3, 3, 3];
    let jcn = [1, 2, 2, 3, 1, 2, 3];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0];
    let mut deps = Vec::new();
    let st = ls.determine_dependent_rows(3, 3, &irn, &jcn, &vals, &mut deps);
    assert_eq!(st, ESymSolverStatus::Success);
    assert_eq!(deps.len(), 1, "exactly one dependent row, got {deps:?}");
}

/// A cold solve whose equality block is rank-deficient, at a size where the
/// dense guard is `k²·n = 2.7·10¹⁰` per pass — past what it may spend.
///
/// Measured on this fixture: 0.04 s with the sparse probe, 68 s through the
/// dense guard before the fix. The `time_limit` only bounds the solve around
/// the guard, not a pass of it — the guard's loops check no deadline — so it
/// caps a regression's damage to the suite rather than turning it into a
/// prompt `TimeLimit`.
///
/// `min ½‖x − t‖²` subject to `x_i − x_{i+p} = 0` for `i < p`, each row
/// repeated once scaled by 2 (rank `p` out of `2p` rows). The optimum pairs
/// the coordinates: `x_i = x_{i+p} = (t_i + t_{i+p}) / 2`.
#[test]
fn cold_solve_with_redundant_equalities_and_no_backend_probe() {
    let p = 1500;
    let n = 2 * p;
    let m = 2 * p;

    let diag: Vec<Index> = (1..=n as Index).collect();
    let h_space = SymTMatrixSpace::new(n as Index, diag.clone(), diag);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&vec![1.0; n]);

    let mut irn = Vec::with_capacity(4 * p);
    let mut jcn = Vec::with_capacity(4 * p);
    let mut vals = Vec::with_capacity(4 * p);
    for i in 0..p {
        for (row, scale) in [(i, 1.0), (p + i, 2.0)] {
            irn.extend([row as Index + 1, row as Index + 1]);
            jcn.extend([i as Index + 1, (i + p) as Index + 1]);
            vals.extend([scale, -scale]);
        }
    }
    let a_space = GenTMatrixSpace::new(m as Index, n as Index, irn, jcn);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&vals);

    let t: Vec<Number> = (0..n).map(|j| ((j * 7) % 13) as Number - 6.0).collect();
    let g: Vec<Number> = t.iter().map(|&tj| -tj).collect();
    let bl = vec![0.0; m];
    let bu = vec![0.0; m];
    let xl = vec![NEG_INF; n];
    let xu = vec![POS_INF; n];

    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = ParametricActiveSetSolver::new(no_probe());
    let opts = QpOptions {
        time_limit: Some(Duration::from_secs(20)),
        ..QpOptions::default()
    };
    let sol = solver
        .solve(&qp, None, &opts)
        .expect("rank-deficient equality QP must solve");
    assert_eq!(sol.status, QpStatus::Optimal);
    // The cold-solve equality prune ran — it was skipped outright on a
    // backend without a probe before the fix — and the counter says so.
    assert!(solver.rank_guard_calls() >= 1);
    for i in 0..p {
        let want = 0.5 * (t[i] + t[i + p]);
        assert!(
            (sol.x[i] - want).abs() < 1e-8 && (sol.x[i + p] - want).abs() < 1e-8,
            "pair {i}: got ({}, {}), want {want}",
            sol.x[i],
            sol.x[i + p]
        );
    }
}
