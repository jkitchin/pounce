//! gh#1206 B6 — the inertia-control ladders recognize a rank-deficient
//! working set by a `−δ_c` dual-shift probe instead of climbing every
//! `H`-block rung.
//!
//! `[H + δI, Aᵀ; A, 0]` is singular for every `δ` when `A`'s rows are
//! dependent, so the ladder used to spend `1 + inertia_max_shifts` (13)
//! factorizations before reporting the failure its callers answer with a
//! prune. Each fixture counts factorizations through `LinsolTimes`.

use crate::QpSolver;
use crate::kkt::assemble_active_set_kkt;
use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::schur::SchurState;
use crate::solver::ParametricActiveSetSolver;
use crate::working_set::{ConsStatus, WorkingSet};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

fn solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

fn n_factor(s: &ParametricActiveSetSolver) -> u64 {
    s.linsol_time_split().n_factor
}

/// `min ½‖x‖² − x₃` over `x₁ + x₂ + x₃ = 3`, `2x₁ + 2x₂ + 2x₃ = 6` (row 2 is
/// twice row 1: LICQ fails with both active), `0 ≤ x ≤ 10`. `h_diag` sets
/// `H`'s diagonal, so a zero entry adds an `H`-block singularity.
fn dependent_rows_qp<R>(h_diag: [f64; 3], f: impl FnOnce(&QpProblem) -> R) -> R {
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(3, vec![1, 2, 3], vec![1, 2, 3]));
    h.set_values(&h_diag);
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(
        2,
        3,
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 2, 3, 1, 2, 3],
    ));
    a.set_values(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let g = [0.0, 0.0, -1.0];
    let (bl, bu) = ([3.0, 6.0], [3.0, 6.0]);
    let (xl, xu) = ([0.0; 3], [10.0; 3]);
    f(&QpProblem {
        n: 3,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    })
}

fn no_probe() -> QpOptions {
    QpOptions {
        dual_shift_probe: 0.0,
        ..QpOptions::default()
    }
}

/// The refactor path's ladder: both rows pinned, the probe stops it after
/// the unshifted attempt, the first rung and the probe itself.
#[test]
fn the_ladder_reports_a_dependent_working_set_without_climbing() {
    dependent_rows_qp([1.0; 3], |qp| {
        let kkt = assemble_active_set_kkt(qp, &[0, 1], &[]);
        let mut s = solver();
        let mut rhs = vec![1.0; kkt.dim];
        let err = s
            .factorize_with_inertia_control(kkt.clone(), &mut rhs, 2, 3, &QpOptions::default())
            .unwrap_err();
        assert!(err.is_rank_deficient_working_set(), "{err}");
        assert!(err.is_recoverable_factorization_failure());
        assert_eq!(n_factor(&s), 3);
        assert!(
            !s.linsol.has_cached_factor(),
            "the probe's factor must not be kept"
        );

        // Without the probe the ladder climbs until FERAL stops flagging the
        // null direction: measured, it "succeeds" at δ = 1e6 after 9
        // factorizations — the masked factor `factor_pinned_primal`'s rank
        // guard exists to catch.
        let mut s = solver();
        let mut rhs = vec![1.0; kkt.dim];
        let out = s.factorize_with_inertia_control(kkt, &mut rhs, 2, 3, &no_probe());
        assert!(
            n_factor(&s) >= 8,
            "{} factorizations, {out:?}",
            n_factor(&s)
        );
        assert!(!matches!(out, Ok(d) if d < 1e4), "{out:?}");
    });
}

/// The Schur path's ladder (`SchurState::reset`): the same working set in
/// K_max, where the inactive bound slots carry `(p, p) = 1` sentinels the
/// probe must leave alone.
#[test]
fn the_schur_reset_reports_a_dependent_working_set_without_climbing() {
    dependent_rows_qp([1.0; 3], |qp| {
        let mut w = WorkingSet::cold(3, 2);
        w.constraints[0] = ConsStatus::Equality;
        w.constraints[1] = ConsStatus::Equality;
        let mut s = solver();
        let mut state = SchurState::new(3, 2);
        let err = state
            .reset(&mut s.linsol, qp, &w, 2, &QpOptions::default())
            .unwrap_err();
        assert!(err.is_rank_deficient_working_set(), "{err}");
        assert_eq!(n_factor(&s), 3);

        let mut s = solver();
        let mut state = SchurState::new(3, 2);
        let _ = state.reset(&mut s.linsol, qp, &w, 2, &no_probe());
        assert!(n_factor(&s) >= 8, "{} factorizations", n_factor(&s));
    });
}

/// The probe is not a reason to stop on a singularity the `H`-block shift
/// *does* remove: a structurally zero `H` column with only one row pinned
/// factors at the first rung, as before, and the probe never runs.
#[test]
fn an_h_block_singularity_still_takes_the_shift() {
    dependent_rows_qp([1.0, 0.0, 0.0], |qp| {
        let kkt = assemble_active_set_kkt(qp, &[0], &[]);
        let mut s = solver();
        let mut rhs = vec![1.0; kkt.dim];
        let delta = s
            .factorize_with_inertia_control(kkt, &mut rhs, 1, 3, &QpOptions::default())
            .unwrap();
        assert!(delta > 0.0);
        assert_eq!(n_factor(&s), 2, "unshifted, then the first rung");
    });
}

/// Both at once — an uncovered zero `H` column *and* dependent rows — is
/// still a rank deficiency: the probe keeps the first rung's `H` shift.
#[test]
fn a_dependent_working_set_over_a_singular_h_is_still_reported() {
    dependent_rows_qp([1.0, 0.0, 0.0], |qp| {
        let kkt = assemble_active_set_kkt(qp, &[0, 1], &[]);
        let mut s = solver();
        let mut rhs = vec![1.0; kkt.dim];
        let err = s
            .factorize_with_inertia_control(kkt, &mut rhs, 2, 3, &QpOptions::default())
            .unwrap_err();
        assert!(err.is_rank_deficient_working_set(), "{err}");
        assert_eq!(n_factor(&s), 3);
    });
}

/// End to end: a warm start that pins both rows is pruned and solved, to
/// the same point, for fewer factorizations than the climbing ladder paid.
#[test]
fn a_dependent_hint_is_pruned_for_fewer_factorizations() {
    dependent_rows_qp([1.0; 3], |qp| {
        let mut w = WorkingSet::cold(3, 2);
        w.constraints[0] = ConsStatus::Equality;
        w.constraints[1] = ConsStatus::Equality;
        let run = |opts: &QpOptions| {
            let mut s = solver();
            let sol = s.solve_with_working_set(qp, &w, opts).unwrap();
            (sol, n_factor(&s))
        };
        let (on, f_on) = run(&QpOptions::default());
        let (off, f_off) = run(&no_probe());
        assert_eq!(on.status, crate::QpStatus::Optimal);
        assert_eq!(off.status, crate::QpStatus::Optimal);
        for (a, b) in on.x.iter().zip(off.x.iter()) {
            assert!((a - b).abs() < 1e-9, "{:?} vs {:?}", on.x, off.x);
        }
        // x* = (2/3, 2/3, 5/3): ½‖x‖² − x₃ on the plane.
        assert!((on.x[2] - 5.0 / 3.0).abs() < 1e-9, "{:?}", on.x);
        assert!(
            // Measured: 4 against 10.
            f_on + 5 <= f_off,
            "{f_on} factorizations with the probe, {f_off} without"
        );
    });
}
