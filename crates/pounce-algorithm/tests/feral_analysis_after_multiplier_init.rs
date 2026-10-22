//! The IPM's first KKT factorization does its own symbolic analysis
//! (jkitchin/feral#224).
//!
//! The first matrix the main augmented-system solver factors is not a
//! KKT system: it is the least-squares multiplier initialization
//! (`W = 0`, `D_x = D_s = I`, `IpLeastSquareMults`). FERAL caches its
//! symbolic analysis per sparsity pattern, and that analysis is
//! value-dependent — the default `OrderingPreprocess::Auto` resolves to
//! `LdltCompress` on KKT-shaped patterns, an MC64 matching on the
//! values, and the default `scaling = Auto` pick is sticky per pattern.
//! So, left alone, every KKT system of the solve is factored under an
//! ordering chosen for the init matrix. On qcqp1000-2c, replaying the
//! solve's matrices with and without that analysis took 99 s against 56 s
//! of factorization (jkitchin/feral#224).
//!
//! The fix drops the analysis once the initializer has run, when — and
//! only when — the initializer's solves created it. These tests pin both
//! halves of that sentence through the real application and the real
//! FERAL backend, by recording every factorization the backend sees:
//!
//! * `the_first_kkt_factor_reanalyses_and_orders_by_its_own_values` —
//!   the first KKT factor runs a fresh analysis, and the ordering it
//!   factors under is the one a fresh FERAL solver picks for that very
//!   matrix, not the init matrix's. The fixture is chosen so the two
//!   orderings genuinely differ (asserted), or the second half would be
//!   vacuous.
//! * `an_analysis_the_init_only_reused_is_kept` — a backend shared
//!   across two solves (the pooled-backend shape, `batch.rs`) already
//!   holds an analysis when the second solve's initializer runs. That
//!   analysis was not built on the init matrix and must survive.
//!
//! Mutation-checked: deleting the `forget_analysis` call in
//! `IpoptAlgorithm::optimize_inner` fails the first test (one analysis,
//! init's ordering); dropping the `analysis_count` bracket around it
//! (forgetting unconditionally) fails the second.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::{
    EMatrixFormat, ESymSolverStatus, FactorPattern, SparseSymLinearSolverInterface,
};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Fixture: a chain-coupled least-squares objective with wildly spread
// weights, under sparse nonlinear equality constraints. Large enough
// (`dim ≥ 128`) and sparse enough that FERAL's structural predicate
// recommends the MC64 preprocess, and the Hessian diagonal spans six
// orders, so the KKT's matching differs from the init system's (where
// that block is the identity).
// ---------------------------------------------------------------------------

const N: usize = 160;
const M: usize = 80;

/// Weight of variable `i`: `~1e3` on the odd (squared) variables and
/// `~1e-3` on the even ones. In the init system every diagonal is `1`
/// and MC64 pairs each constraint row with its largest Jacobian entry,
/// the squared variable's `2·x_b`; in the KKT system that variable's
/// diagonal dominates and the matching pairs the row elsewhere.
fn w(i: usize) -> Number {
    let spread = 1.0 + 0.01 * i as Number;
    if i % 2 == 1 {
        1e3 * spread
    } else {
        1e-3 * spread
    }
}

/// Constraint `j` touches `a = 2j`, `b = 2j + 1` and `c = (2j + 3) mod N`:
/// `x_a + x_b² − x_c = 1`.
fn cols(j: usize) -> [usize; 3] {
    [2 * j, 2 * j + 1, (2 * j + 3) % N]
}

struct Chain;

impl TNLP for Chain {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as Index,
            m: M as Index,
            nnz_jac_g: (3 * M) as Index,
            // Diagonal + one sub-diagonal per chain link.
            nnz_h_lag: (N + N - 1) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-10.0);
        b.x_u.fill(10.0);
        b.g_l.fill(1.0);
        b.g_u.fill(1.0);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for (i, x) in sp.x.iter_mut().enumerate() {
            *x = 0.5 + 0.01 * i as Number;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..N {
            f += 0.5 * w(i) * (x[i] - 2.0).powi(2);
        }
        for i in 0..N - 1 {
            f += 0.1 * x[i] * x[i + 1];
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..N {
            g[i] = w(i) * (x[i] - 2.0);
        }
        for i in 0..N - 1 {
            g[i] += 0.1 * x[i + 1];
            g[i + 1] += 0.1 * x[i];
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (j, gj) in g.iter_mut().enumerate() {
            let [a, b, c] = cols(j);
            *gj = x[a] + x[b] * x[b] - x[c];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..M {
                    for (k, c) in cols(j).into_iter().enumerate() {
                        irow[3 * j + k] = j as Index;
                        jcol[3 * j + k] = c as Index;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values without x");
                for j in 0..M {
                    let [_, b, _] = cols(j);
                    values[3 * j] = 1.0;
                    values[3 * j + 1] = 2.0 * x[b];
                    values[3 * j + 2] = -1.0;
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..N {
                    irow[i] = i as Index;
                    jcol[i] = i as Index;
                }
                for i in 0..N - 1 {
                    irow[N + i] = (i + 1) as Index;
                    jcol[N + i] = i as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("values without lambda");
                for i in 0..N {
                    values[i] = obj_factor * w(i);
                }
                for (j, l) in lam.iter().enumerate() {
                    let [_, b, _] = cols(j);
                    values[b] += 2.0 * l;
                }
                for i in 0..N - 1 {
                    values[N + i] = obj_factor * 0.1;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

// ---------------------------------------------------------------------------
// A recording pass-through over the real FERAL backend.
// ---------------------------------------------------------------------------

/// One factorization, as the backend saw it.
#[derive(Clone)]
struct Factor {
    /// `analysis_count()` right after the factor: a rise over the
    /// previous record means this factor ran a fresh analysis.
    analyses: u64,
    /// Fill-reducing permutation it factored under.
    perm: Vec<usize>,
    /// The matrix, so a fresh solver can be asked what it would pick.
    vals: Vec<Number>,
}

#[derive(Default)]
struct Log {
    dim: Index,
    ia: Vec<Index>,
    ja: Vec<Index>,
    factors: Vec<Factor>,
}

/// Where a backend waits between solves: the second test hands the
/// *same* FERAL solver to two consecutive solves, the shape of the pooled
/// backend in `batch.rs`.
type Slot = Rc<RefCell<Option<FeralSolverInterface>>>;

/// Delegates every call to a FERAL backend taken from a [`Slot`] (parked
/// back on drop) and records each new factorization.
struct Recording {
    inner: Option<FeralSolverInterface>,
    slot: Slot,
    log: Rc<RefCell<Log>>,
}

impl Recording {
    fn get(&mut self) -> &mut FeralSolverInterface {
        self.inner.as_mut().expect("held until drop")
    }
    fn peek(&self) -> &FeralSolverInterface {
        self.inner.as_ref().expect("held until drop")
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        *self.slot.borrow_mut() = self.inner.take();
    }
}

impl SparseSymLinearSolverInterface for Recording {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        {
            let mut log = self.log.borrow_mut();
            log.dim = dim;
            log.ia = ia.to_vec();
            log.ja = ja.to_vec();
        }
        self.get().initialize_structure(dim, nonzeros, ia, ja)
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        self.get().values_array_mut()
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
        let vals = new_matrix.then(|| self.get().values_array_mut().to_vec());
        let status = self.get().multi_solve(
            new_matrix,
            ia,
            ja,
            nrhs,
            rhs_vals,
            check_neg_evals,
            number_of_neg_evals,
        );
        if let Some(vals) = vals {
            if let Some(p) = self.peek().factor_pattern(false) {
                let analyses = self.peek().analysis_count();
                self.log.borrow_mut().factors.push(Factor {
                    analyses,
                    perm: p.perm,
                    vals,
                });
            }
        }
        status
    }

    fn number_of_neg_evals(&self) -> Index {
        self.peek().number_of_neg_evals()
    }
    fn increase_quality(&mut self) -> bool {
        self.get().increase_quality()
    }
    fn provides_inertia(&self) -> bool {
        self.peek().provides_inertia()
    }
    fn singularity_certified(&self) -> bool {
        self.peek().singularity_certified()
    }
    fn analysis_count(&self) -> u64 {
        self.peek().analysis_count()
    }
    fn forget_analysis(&mut self) -> bool {
        self.get().forget_analysis()
    }
    fn multi_solve_matches_single_solve(&self, nrhs: usize) -> bool {
        self.peek().multi_solve_matches_single_solve(nrhs)
    }
    fn matrix_format(&self) -> EMatrixFormat {
        self.peek().matrix_format()
    }
    fn factor_pattern(&self, want_values: bool) -> Option<FactorPattern> {
        self.peek().factor_pattern(want_values)
    }
}

/// Solve the fixture once with the backend parked in `slot` (a fresh one
/// if it is empty), recording into `log`.
fn solve(slot: &Slot, log: &Rc<RefCell<Log>>) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let (s, l) = (Rc::clone(slot), Rc::clone(log));
    app.set_linear_backend_factory(Box::new(move |_choice| {
        let inner = s.borrow_mut().take().unwrap_or_default();
        Box::new(Recording {
            inner: Some(inner),
            slot: Rc::clone(&s),
            log: Rc::clone(&l),
        })
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Chain));
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must solve: {status:?}",
    );
}

/// The permutation a brand-new FERAL backend factors `vals` under.
fn fresh_perm(log: &Log, vals: &[Number]) -> Vec<usize> {
    let mut s = FeralSolverInterface::new();
    let nnz = log.ia.len() as Index;
    assert_eq!(
        s.initialize_structure(log.dim, nnz, &log.ia, &log.ja),
        ESymSolverStatus::Success
    );
    s.values_array_mut().copy_from_slice(vals);
    let mut rhs = vec![1.0; log.dim as usize];
    let st = s.multi_solve(true, &log.ia, &log.ja, 1, &mut rhs, false, 0);
    assert_ne!(st, ESymSolverStatus::FatalError);
    s.factor_pattern(false).expect("factored").perm
}

#[test]
fn the_first_kkt_factor_reanalyses_and_orders_by_its_own_values() {
    let slot: Slot = Rc::default();
    let log = Rc::new(RefCell::new(Log::default()));
    solve(&slot, &log);
    let log = log.borrow();
    assert!(
        log.dim as usize >= 128,
        "fixture too small for MC64 preprocess"
    );
    assert!(
        log.factors.len() >= 2,
        "need the init factor and a KKT factor"
    );

    let init = &log.factors[0];
    let kkt = &log.factors[1];
    assert_eq!(init.analyses, 1, "the init system creates the analysis");
    assert_eq!(
        kkt.analyses, 2,
        "the IPM's first KKT factor must run its own analysis, not reuse \
         the one built on the multiplier-init system",
    );

    // The fixture has to be one on which the analysis is value-dependent,
    // or the ordering comparison below proves nothing.
    let init_choice = fresh_perm(&log, &init.vals);
    let kkt_choice = fresh_perm(&log, &kkt.vals);
    assert_eq!(
        init.perm, init_choice,
        "sanity: init factored as a fresh solve"
    );
    assert_ne!(
        init_choice, kkt_choice,
        "fixture is not value-dependent: FERAL picks the same ordering for \
         the init system and the first KKT system",
    );
    assert_eq!(
        kkt.perm, kkt_choice,
        "the first KKT system must factor under the ordering FERAL picks for \
         it, not the init system's",
    );

    // And the rest of the solve keeps that analysis: one re-analysis, not
    // one per iteration.
    assert!(
        log.factors[1..].iter().all(|f| f.analyses == 2),
        "analysis counts after the first KKT: {:?}",
        log.factors.iter().map(|f| f.analyses).collect::<Vec<_>>(),
    );
}

#[test]
fn an_analysis_the_init_only_reused_is_kept() {
    let slot: Slot = Rc::default();
    let first = Rc::new(RefCell::new(Log::default()));
    solve(&slot, &first);
    let after_first = slot
        .borrow()
        .as_ref()
        .expect("the first solve parked its backend")
        .analysis_count();
    assert_eq!(after_first, 2, "first solve: init + first KKT");

    // Same backend, same pattern: the second solve's init factor reuses
    // the analysis the first solve's IPM built, so nothing is forgotten
    // and no factor of the second solve re-analyses.
    let second = Rc::new(RefCell::new(Log::default()));
    solve(&slot, &second);
    let counts: Vec<u64> = second.borrow().factors.iter().map(|f| f.analyses).collect();
    assert!(
        counts.iter().all(|&c| c == after_first),
        "an analysis the initializer did not create must be kept: {counts:?}",
    );
}
