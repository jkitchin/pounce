//! Cross-solver harness: run one symmetric system through every backend.
//!
//! Evaluation scaffolding, not part of the adapter. It lives in the crate
//! rather than in `tests/` so the integration tests, the example benchmark and
//! the real-KKT replay can all drive the same code — a harness that measured
//! FERAL one way in a test and RSLAB another way in a bench would answer no
//! question at all.
//!
//! Every backend is driven through [`pounce_linsol::SparseSymLinearSolverInterface`]
//! itself, in the same call order POUNCE's `TSymLinearSolver` uses, so what is
//! compared is what POUNCE would actually get.
//!
//! # Residuals
//!
//! Two are recorded, because they answer different questions:
//!
//! * [`SolveRecord::residual_ratio`] is **POUNCE's own** metric — Ipopt's
//!   `‖Ax - b‖_∞ / (min(‖Ax‖_∞, 10⁶·‖b‖_∞) + ‖b‖_∞)`, the quantity
//!   `PdFullSpaceSolver::compute_residual_ratio` computes and the iterative
//!   refinement loop steers on. A backend is acceptable to POUNCE if and only
//!   if this one is small.
//! * [`SolveRecord::rel_residual`] is `‖Ax - b‖₂ / (‖A‖_∞·‖x‖₂ + ‖b‖₂)`, the
//!   scale-robust normalized residual. `‖Ax - b‖ / ‖b‖` alone is reported by
//!   neither: on a badly scaled KKT — which is every KKT near convergence — it
//!   flatters a solver whose error sits in the large-magnitude rows.

use std::time::Instant;

use pounce_common::types::{Index, Number};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};

/// A symmetric matrix in the format POUNCE's backend trait takes: **1-based
/// triplets over the lower triangle**, duplicates allowed (they sum).
#[derive(Debug, Clone)]
pub struct SymTriplet {
    pub n: Index,
    pub irn: Vec<Index>,
    pub jcn: Vec<Index>,
    pub vals: Vec<Number>,
}

impl SymTriplet {
    /// Number of stored triplets (not of distinct nonzeros).
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// `A · x`, expanding the stored lower triangle to the full symmetric
    /// matrix. Duplicate triplets contribute additively, as they do in the
    /// factorization.
    pub fn mul(&self, x: &[Number]) -> Vec<Number> {
        let mut y = vec![0.0; self.n as usize];
        for k in 0..self.vals.len() {
            let i = (self.irn[k] - 1) as usize;
            let j = (self.jcn[k] - 1) as usize;
            let v = self.vals[k];
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
        y
    }

    /// `‖A‖_∞ = max_i Σ_j |A_ij|`, over the expanded symmetric matrix.
    pub fn norm_inf(&self) -> Number {
        let mut rows = vec![0.0; self.n as usize];
        for k in 0..self.vals.len() {
            let i = (self.irn[k] - 1) as usize;
            let j = (self.jcn[k] - 1) as usize;
            let v = self.vals[k].abs();
            rows[i] += v;
            if i != j {
                rows[j] += v;
            }
        }
        rows.into_iter().fold(0.0, f64::max)
    }

    /// A deterministic right-hand side, so two runs of the harness compare
    /// like with like. `sin`-based rather than constant, so a solver cannot
    /// look good by accident on a structured `b`.
    pub fn sample_rhs(&self) -> Vec<Number> {
        (0..self.n as usize)
            .map(|i| ((i as f64) * 0.7).sin() + 0.5)
            .collect()
    }
}

/// What one backend did with one system.
#[derive(Debug, Clone)]
pub struct SolveRecord {
    /// `"feral"`, `"rslab"`, `"ma57"`.
    pub solver: &'static str,
    /// Status of the factorization (`Success` means the factor completed).
    pub factor_status: ESymSolverStatus,
    /// Status of the back-substitution, if one was attempted.
    pub solve_status: Option<ESymSolverStatus>,
    /// `(positive, negative, zero)` where the backend reports a full triple.
    /// MA57 and the trait itself report only the negative count, so the other
    /// two entries are `None` there — see [`Self::negative_evals`].
    pub inertia: Option<(usize, usize, usize)>,
    /// The one inertia number every backend reports, and the only one POUNCE
    /// steers on.
    pub negative_evals: Option<Index>,
    /// Smallest accepted pivot magnitude (2×2 blocks by eigenvalue), in the
    /// backend's own scaled space.
    pub min_abs_pivot: Option<Number>,
    /// Largest accepted pivot magnitude, same space.
    pub max_abs_pivot: Option<Number>,
    /// Number of 2×2 Bunch-Kaufman blocks, where the backend reports it.
    pub two_by_two_pivots: Option<usize>,
    /// Statically perturbed pivots, where the backend reports it.
    pub perturbed_pivots: Option<usize>,
    /// Whether the backend considers its own inertia trustworthy. Only the
    /// RSLAB adapter answers this directly; for FERAL it is derived from the
    /// same floor comparison `FeralSolverInterface::factor` applies.
    pub inertia_reliable: Option<bool>,
    /// `nnz(A)` as the backend counted it.
    pub nnz_a: Option<usize>,
    /// `nnz(L)` of the factor.
    pub nnz_l: Option<usize>,
    /// `nnz(L) / nnz(A)`.
    pub fill_ratio: Option<f64>,
    /// Factorizations that reused the cached symbolic analysis, and those that
    /// had to build a fresh one. Both FERAL and the RSLAB adapter populate
    /// these, and they are the *structural* answer to "did the analysis get
    /// reused" — a timing comparison at these sizes is dominated by noise.
    pub n_pattern_reuse: u64,
    /// See [`Self::n_pattern_reuse`].
    pub n_pattern_changes: u64,
    /// Slots of `L` holding an exact zero, where the backend reports it.
    ///
    /// `nnz_l` is structural on every arm — it is what the factor occupies —
    /// but on a POUNCE KKT most of it carries no information, because the
    /// triplet pattern arrives full of explicit zeros (1768 of 2016 stored
    /// entries on `airport`'s first KKT). Only RSLAB reports the split, so a
    /// fill comparison against FERAL has to be read as structural-to-
    /// structural and nothing finer.
    pub explicit_zeros_in_l: Option<usize>,
    /// Wall time of `initialize_structure` — the symbolic-analysis *request*.
    /// Backends differ in how much they do here versus fold into the first
    /// numeric factor: both FERAL and the RSLAB adapter defer their ordering,
    /// so this row reads near zero for both and the analysis cost shows up in
    /// `factor_ms`.
    pub structure_ms: f64,
    /// Wall time of the first factorization (symbolic + numeric).
    pub factor_ms: f64,
    /// Wall time of a second factorization with the same pattern and new
    /// values — the refactorization the IPM actually pays, per iteration.
    pub refactor_ms: Option<f64>,
    /// Wall time of one single-RHS back-substitution against the stored factor.
    pub solve_ms: Option<f64>,
    /// POUNCE's own residual metric. See the module docs.
    pub residual_ratio: Option<f64>,
    /// `‖Ax - b‖₂ / (‖A‖_∞·‖x‖₂ + ‖b‖₂)`.
    pub rel_residual: Option<f64>,
}

impl SolveRecord {
    fn failed(
        solver: &'static str,
        status: ESymSolverStatus,
        structure_ms: f64,
        factor_ms: f64,
    ) -> Self {
        Self {
            solver,
            factor_status: status,
            solve_status: None,
            inertia: None,
            negative_evals: None,
            min_abs_pivot: None,
            max_abs_pivot: None,
            two_by_two_pivots: None,
            perturbed_pivots: None,
            inertia_reliable: None,
            nnz_a: None,
            nnz_l: None,
            fill_ratio: None,
            n_pattern_reuse: 0,
            n_pattern_changes: 0,
            explicit_zeros_in_l: None,
            structure_ms,
            factor_ms,
            refactor_ms: None,
            solve_ms: None,
            residual_ratio: None,
            rel_residual: None,
        }
    }

    /// `true` when the factor completed and the solve came back inside
    /// `tol` on POUNCE's own residual metric.
    pub fn is_acceptable(&self, tol: f64) -> bool {
        self.factor_status == ESymSolverStatus::Success
            && self.solve_status == Some(ESymSolverStatus::Success)
            && self.residual_ratio.is_some_and(|r| r <= tol)
    }

    /// One tab-separated line, for a bench that prints a table.
    pub fn row(&self) -> String {
        fn f(v: Option<f64>) -> String {
            v.map_or_else(|| "-".to_string(), |x| format!("{x:.3e}"))
        }
        fn u(v: Option<usize>) -> String {
            v.map_or_else(|| "-".to_string(), |x| x.to_string())
        }
        format!(
            "{:<6}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}",
            self.solver,
            self.factor_status,
            self.inertia.map_or_else(
                || format!("(-,{},-)", u(self.negative_evals.map(|v| v as usize))),
                |(p, n, z)| format!("({p},{n},{z})")
            ),
            u(self.two_by_two_pivots),
            u(self.perturbed_pivots),
            f(self.min_abs_pivot),
            u(self.nnz_l),
            self.factor_ms,
            self.refactor_ms.unwrap_or(f64::NAN),
            f(self.solve_ms.map(|v| v / 1e3)),
            f(self.fill_ratio),
            f(self.residual_ratio),
            f(self.rel_residual),
        )
    }
}

/// Column header matching [`SolveRecord::row`].
pub const ROW_HEADER: &str = "solver\tstatus\tinertia\t2x2\tpert\tmin|piv|\tnnz(L)\tfac_ms\trefac_ms\tsolve_s\tfill\tresid_ratio\trel_resid";

/// POUNCE's residual metric, as `PdFullSpaceSolver::compute_residual_ratio`
/// computes it: `‖r‖_∞ / (min(‖Ax‖_∞, 10⁶·‖b‖_∞) + ‖b‖_∞)`, falling back to
/// the bare `‖r‖_∞` when both norms vanish.
pub fn residual_ratio(a: &SymTriplet, x: &[Number], b: &[Number]) -> f64 {
    let ax = a.mul(x);
    let amax = |v: &[Number]| v.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
    let nrm_rhs = amax(b);
    let nrm_res = amax(&ax);
    let r: Vec<Number> = ax.iter().zip(b).map(|(p, q)| p - q).collect();
    let nrm_resid = amax(&r);
    if nrm_rhs + nrm_res == 0.0 {
        return nrm_resid;
    }
    let max_cond = 1e6;
    nrm_resid / (nrm_res.min(max_cond * nrm_rhs) + nrm_rhs)
}

/// `‖Ax - b‖₂ / (‖A‖_∞·‖x‖₂ + ‖b‖₂)`.
pub fn relative_residual(a: &SymTriplet, x: &[Number], b: &[Number]) -> f64 {
    let ax = a.mul(x);
    let two = |v: &[Number]| v.iter().map(|t| t * t).sum::<f64>().sqrt();
    let r: Vec<Number> = ax.iter().zip(b).map(|(p, q)| p - q).collect();
    let denom = a.norm_inf() * two(x) + two(b);
    if denom == 0.0 {
        two(&r)
    } else {
        two(&r) / denom
    }
}

/// Drive one backend through the full lifecycle and measure it.
///
/// The call sequence is the trait's own: `initialize_structure` →
/// `values_array_mut` → `multi_solve(new_matrix = true)` (factor + solve) →
/// `multi_solve(new_matrix = false)` (back-solve only) →
/// `multi_solve(new_matrix = true)` again with the values refilled (the
/// refactorization). That last step is the one the IPM pays every iteration
/// and the one a symbolic cache has to earn its keep on, so it is timed
/// separately from the first factor.
///
/// `probe` is handed the backend after the measurements so a caller can read
/// backend-specific diagnostics off it (the RSLAB adapter's `InertiaInfo`, the
/// FERAL backend's `LinearSolverSummary`) without this function knowing about
/// either.
pub fn run_backend<B: SparseSymLinearSolverInterface>(
    solver: &'static str,
    mut backend: B,
    a: &SymTriplet,
    b: &[Number],
    probe: impl FnOnce(&B, &mut SolveRecord),
) -> SolveRecord {
    let n = a.n;
    let nnz = a.nnz() as Index;

    let t = Instant::now();
    let st = backend.initialize_structure(n, nnz, &a.irn, &a.jcn);
    let structure_ms = t.elapsed().as_secs_f64() * 1e3;
    if st != ESymSolverStatus::Success {
        return SolveRecord::failed(solver, st, structure_ms, 0.0);
    }
    backend.values_array_mut().copy_from_slice(&a.vals);

    let mut x = b.to_vec();
    let t = Instant::now();
    let st = backend.multi_solve(true, &a.irn, &a.jcn, 1, &mut x, false, 0);
    let factor_ms = t.elapsed().as_secs_f64() * 1e3;
    if st != ESymSolverStatus::Success {
        return SolveRecord::failed(solver, st, structure_ms, factor_ms);
    }

    // Isolate the back-substitution against the factor just built.
    let mut x2 = b.to_vec();
    let t = Instant::now();
    let solve_st = backend.multi_solve(false, &a.irn, &a.jcn, 1, &mut x2, false, 0);
    let solve_ms = t.elapsed().as_secs_f64() * 1e3;

    // Refactorization: same pattern, same values. The values are identical on
    // purpose — the question here is what the symbolic cache saves, and a
    // different matrix would move the numeric cost too.
    backend.values_array_mut().copy_from_slice(&a.vals);
    let mut x3 = b.to_vec();
    let t = Instant::now();
    let refac_st = backend.multi_solve(true, &a.irn, &a.jcn, 1, &mut x3, false, 0);
    let refactor_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut rec = SolveRecord {
        solver,
        factor_status: ESymSolverStatus::Success,
        solve_status: Some(solve_st),
        inertia: None,
        negative_evals: backend
            .provides_inertia()
            .then(|| backend.number_of_neg_evals()),
        min_abs_pivot: None,
        max_abs_pivot: None,
        two_by_two_pivots: None,
        perturbed_pivots: None,
        inertia_reliable: None,
        nnz_a: None,
        nnz_l: None,
        fill_ratio: None,
        n_pattern_reuse: 0,
        n_pattern_changes: 0,
        explicit_zeros_in_l: None,
        structure_ms,
        factor_ms,
        refactor_ms: (refac_st == ESymSolverStatus::Success).then_some(refactor_ms),
        solve_ms: (solve_st == ESymSolverStatus::Success).then_some(solve_ms),
        residual_ratio: (solve_st == ESymSolverStatus::Success).then(|| residual_ratio(a, &x2, b)),
        rel_residual: (solve_st == ESymSolverStatus::Success).then(|| relative_residual(a, &x2, b)),
    };
    probe(&backend, &mut rec);
    rec
}

/// Fold a [`pounce_linsol::summary::LinearSolverSummary`] into a record. Both
/// the FERAL backend and the RSLAB adapter populate one, so this is the shared
/// half of the two probes below.
pub fn fold_summary(s: &pounce_linsol::summary::LinearSolverSummary, rec: &mut SolveRecord) {
    rec.inertia = s.last_inertia;
    rec.min_abs_pivot = s.min_abs_pivot;
    rec.max_abs_pivot = s.max_abs_pivot;
    rec.nnz_a = s.last_nnz_a;
    rec.nnz_l = s.last_nnz_l;
    rec.fill_ratio = s.max_fill_ratio;
    rec.n_pattern_reuse = s.n_pattern_reuse;
    rec.n_pattern_changes = s.n_pattern_changes;
}

/// Run the RSLAB adapter, reading its `InertiaInfo` for the fields no other
/// backend reports.
pub fn run_rslab(a: &SymTriplet, b: &[Number], cfg: crate::RslabConfig) -> SolveRecord {
    run_backend(
        "rslab",
        crate::RslabSolverInterface::with_config(cfg),
        a,
        b,
        |backend, rec| {
            fold_summary(&backend.summary(), rec);
            let info = backend.inertia_info();
            rec.two_by_two_pivots = Some(info.two_by_two_pivots);
            rec.perturbed_pivots = Some(info.perturbed_pivots);
            rec.inertia_reliable = Some(info.reliable);
            rec.explicit_zeros_in_l = Some(backend.explicit_zeros());
        },
    )
}

/// Run the FERAL backend — POUNCE's shipping default, and the thing RSLAB has
/// to beat to be worth a permanent dependency.
///
/// `inertia_reliable` is derived rather than read: `FeralSolverInterface` keeps
/// its gh#540 verdict internal, so the harness reapplies the same comparison
/// (`min |pivot| >= inertia_trust_floor(None, n)`) to the summary it does
/// publish. That is the *whole* of FERAL's reliability rule — the RSLAB
/// adapter's has two further disqualifiers — so a `true` here is a weaker
/// statement than a `true` from RSLAB, and the assessment says so.
pub fn run_feral(a: &SymTriplet, b: &[Number], cfg: pounce_feral::FeralConfig) -> SolveRecord {
    let n = a.n as usize;
    run_backend(
        "feral",
        pounce_feral::FeralSolverInterface::with_config(cfg),
        a,
        b,
        |backend, rec| {
            let s = backend.summary();
            fold_summary(&s, rec);
            let floor = pounce_feral::inertia_trust_floor(None, n);
            rec.inertia_reliable = s.min_abs_pivot.map(|m| m >= floor);
        },
    )
}

/// Run the HSL MA57 backend, when the `ma57` feature is on and libcoinhsl is
/// on the link path. `None` otherwise — the harness records the arm as
/// unavailable rather than quietly dropping it from the comparison.
///
/// `options` is **required**, not defaulted. `Ma57SolverInterface::new()`
/// hard-codes `Options::defaults()` and is what pounce gh#825 was: nine
/// `ma57_*` options registered, documented, accepted and silently discarded,
/// with every arm of every solve coming out identical to all seventeen digits.
/// A comparison harness is exactly where that would be invisible — the MA57
/// column would simply be MA57-with-defaults no matter what the reader had
/// tuned. Taking the options by value makes discarding them a thing a caller
/// has to write down.
///
/// MA57 keeps its factors inside opaque Fortran work arrays, so it reports
/// neither a pivot-magnitude extent nor a 2×2 count through this trait; only
/// `number_of_neg_evals` is comparable. Rows for it are therefore mostly `-`,
/// which is a fact about MA57's interface and not a gap in the harness.
#[cfg(feature = "ma57")]
pub fn run_ma57(a: &SymTriplet, b: &[Number], options: pounce_hsl::Options) -> Option<SolveRecord> {
    Some(run_backend(
        "ma57",
        pounce_hsl::Ma57SolverInterface::with_options(options),
        a,
        b,
        |_backend, _rec| {},
    ))
}

/// See the `ma57`-enabled sibling. The `options` parameter is kept so the two
/// signatures agree and a caller need not `cfg` around the call.
#[cfg(not(feature = "ma57"))]
pub fn run_ma57(_a: &SymTriplet, _b: &[Number], _options: Ma57Options) -> Option<SolveRecord> {
    None
}

/// `pounce_hsl::Options` when the `ma57` feature is on; a placeholder
/// otherwise, so [`run_ma57`]'s signature does not move with the feature.
#[cfg(feature = "ma57")]
pub type Ma57Options = pounce_hsl::Options;

/// See the `ma57`-enabled sibling.
#[cfg(not(feature = "ma57"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ma57Options;

impl Ma57Options {
    /// MA57's own library defaults.
    ///
    /// Named rather than implicit: a caller reaching for this is saying "I
    /// have no `ma57_*` options to honour", which is true of the harness and
    /// not true of POUNCE's solve path. See [`run_ma57`].
    #[cfg(not(feature = "ma57"))]
    pub fn library_defaults() -> Self {
        Self
    }
}

/// MA57's library defaults, for a caller with no `OptionsList` to read.
#[cfg(feature = "ma57")]
pub fn ma57_library_defaults() -> Ma57Options {
    pounce_hsl::Options::defaults()
}

/// See the `ma57`-enabled sibling.
#[cfg(not(feature = "ma57"))]
pub fn ma57_library_defaults() -> Ma57Options {
    Ma57Options
}

/// Every available backend on one system, in a stable order.
///
/// RSLAB appears twice, because one configuration of it does not answer the
/// question. `rslab` is the exact mode, the like-for-like comparison against
/// FERAL and MA57. `rslab-sp` is static pivoting at `1e-12·max|A|`, which is
/// the only mode in which RSLAB completes a POUNCE KKT factorization at all —
/// and which produces a preconditioner rather than a direct factor, so its
/// inertia is that of `A + E`. Dropping either arm would misreport RSLAB: the
/// first alone says it cannot do the job, the second alone hides what it cost.
pub fn run_all(a: &SymTriplet, b: &[Number]) -> Vec<SolveRecord> {
    let mut out = vec![
        run_feral(a, b, pounce_feral::FeralConfig::default()),
        run_rslab(a, b, crate::RslabConfig::default()),
        run_rslab_static_pivoting(a, b, 1e-12),
    ];
    // The harness has no `OptionsList` to read `ma57_*` from, so it asks for
    // MA57's library defaults by name. A caller who has tuned them calls
    // `run_ma57` directly with their own.
    out.extend(run_ma57(a, b, ma57_library_defaults()));
    out
}

/// The scaling-neutral control: FERAL and RSLAB with equilibration switched
/// off on both sides.
///
/// It exists because the two backends do not equilibrate the same way and
/// cannot be made to. FERAL's default is `ScalingStrategy::Auto` — MC64
/// matching on arrow-KKT-shaped matrices, iterative Knight-Ruiz otherwise —
/// and RSLAB offers neither through the surface the adapter can reach; its
/// `SolverSettings` default is the single-pass inf-norm step, which is what
/// `scaling::Equilibration::OnePassInfNorm` reproduces. A residual gap between
/// the two default configurations is therefore a gap between two *solvers*,
/// not between two factorization kernels, and reporting it as the latter would
/// be wrong. Running both unscaled removes the confound: what is left is the
/// pivoting and the elimination order.
pub fn run_scaling_neutral_pair(a: &SymTriplet, b: &[Number]) -> Vec<SolveRecord> {
    let mut feral = run_feral(
        a,
        b,
        pounce_feral::FeralConfig {
            scaling: pounce_feral::ScalingStrategy::Identity,
            ..pounce_feral::FeralConfig::default()
        },
    );
    feral.solver = "feral-ns";
    let mut rslab = run_rslab(
        a,
        b,
        crate::RslabConfig {
            equilibration: crate::scaling::Equilibration::Identity,
            ..crate::RslabConfig::default()
        },
    );
    rslab.solver = "rslab-ns";
    vec![feral, rslab]
}

/// RSLAB in static-pivoting mode. See [`crate::PivotPolicy::StaticPivotRelative`].
pub fn run_rslab_static_pivoting(a: &SymTriplet, b: &[Number], eps_rel: f64) -> SolveRecord {
    let mut rec = run_backend(
        "rslab-sp",
        crate::RslabSolverInterface::with_config(crate::RslabConfig::static_pivoting(eps_rel)),
        a,
        b,
        |backend, rec| {
            fold_summary(&backend.summary(), rec);
            let info = backend.inertia_info();
            rec.two_by_two_pivots = Some(info.two_by_two_pivots);
            rec.perturbed_pivots = Some(info.perturbed_pivots);
            rec.inertia_reliable = Some(info.reliable);
            rec.explicit_zeros_in_l = Some(backend.explicit_zeros());
        },
    );
    rec.solver = "rslab-sp";
    rec
}

/// Exact inertia of a **small** symmetric matrix, by dense cyclic-Jacobi
/// eigenvalues — an oracle independent of every LDLᵀ implementation under
/// test.
///
/// This is the outside number the comparison otherwise lacks. Every other
/// quantity in a [`SolveRecord`] is one factorization's opinion of itself; two
/// backends agreeing tells you they agree, not that either is right. Jacobi
/// converges for any real symmetric matrix and touches no pivot rule, so when
/// it disagrees with a backend the backend is wrong.
///
/// Returns `(positive, negative, zero, min|λ|, max|λ|)`, classifying against
/// `n·ε·max|λ|` — the backward-error bound for a symmetric eigensolve.
/// `None` above `max_n`, since the routine is `O(n³)` per sweep and dense.
pub fn dense_inertia_oracle(
    a: &SymTriplet,
    max_n: usize,
) -> Option<(usize, usize, usize, f64, f64)> {
    let n = a.n as usize;
    if n == 0 || n > max_n {
        return None;
    }
    let mut m = vec![0.0f64; n * n];
    for k in 0..a.vals.len() {
        let i = (a.irn[k] - 1) as usize;
        let j = (a.jcn[k] - 1) as usize;
        m[i * n + j] += a.vals[k];
        if i != j {
            m[j * n + i] += a.vals[k];
        }
    }
    let at = |m: &[f64], i: usize, j: usize| m[i * n + j];
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += at(&m, i, j) * at(&m, i, j);
            }
        }
        let scale = (0..n)
            .map(|i| at(&m, i, i).abs())
            .fold(0.0, f64::max)
            .max(1.0);
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = at(&m, p, q);
                if apq == 0.0 {
                    continue;
                }
                let theta = (at(&m, q, q) - at(&m, p, p)) / (2.0 * apq);
                let t = if theta >= 0.0 { 1.0 } else { -1.0 }
                    / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - sn * akq;
                    m[k * n + q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - sn * aqk;
                    m[q * n + k] = sn * apk + c * aqk;
                }
            }
        }
    }
    let ev: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    let max_abs = ev.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let min_abs = ev.iter().fold(f64::INFINITY, |acc, v| acc.min(v.abs()));
    let tol = max_abs * (n as f64) * f64::EPSILON;
    let pos = ev.iter().filter(|v| **v > tol).count();
    let neg = ev.iter().filter(|v| **v < -tol).count();
    Some((pos, neg, n - pos - neg, min_abs, max_abs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd_3x3() -> SymTriplet {
        // [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        SymTriplet {
            n: 3,
            irn: vec![1, 2, 2, 3, 3],
            jcn: vec![1, 1, 2, 2, 3],
            vals: vec![4.0, 1.0, 3.0, 1.0, 2.0],
        }
    }

    /// `mul` expands the stored lower triangle; a lower-triangle-only product
    /// would silently drop the upper half and make every residual look good.
    #[test]
    fn mul_expands_the_symmetric_matrix() {
        let a = spd_3x3();
        let y = a.mul(&[1.0, 1.0, 1.0]);
        assert_eq!(y, vec![5.0, 5.0, 3.0]);
    }

    #[test]
    fn norm_inf_is_the_expanded_row_sum() {
        // Row sums are 5, 5, 3.
        assert_eq!(spd_3x3().norm_inf(), 5.0);
    }

    /// Duplicate triplets contribute additively, matching `from_triplets`.
    #[test]
    fn duplicate_triplets_sum_in_the_product() {
        let a = SymTriplet {
            n: 1,
            irn: vec![1, 1],
            jcn: vec![1, 1],
            vals: vec![1.5, 0.5],
        };
        assert_eq!(a.mul(&[2.0]), vec![4.0]);
        assert_eq!(a.norm_inf(), 2.0);
    }

    /// The exact solution has both residuals at zero; a perturbed one does not.
    #[test]
    fn both_residual_metrics_separate_a_good_solve_from_a_bad_one() {
        let a = spd_3x3();
        let b = a.mul(&[1.0, 2.0, 3.0]);
        assert!(residual_ratio(&a, &[1.0, 2.0, 3.0], &b) < 1e-15);
        assert!(relative_residual(&a, &[1.0, 2.0, 3.0], &b) < 1e-15);
        let bad = [1.0, 2.0, 3.1];
        assert!(residual_ratio(&a, &bad, &b) > 1e-3);
        assert!(relative_residual(&a, &bad, &b) > 1e-4);
    }

    /// The harness reports every field it promises on a healthy system.
    #[test]
    fn the_rslab_record_is_fully_populated() {
        let a = spd_3x3();
        let b = a.sample_rhs();
        let rec = run_rslab(&a, &b, crate::RslabConfig::default());
        assert_eq!(rec.factor_status, ESymSolverStatus::Success);
        assert_eq!(rec.inertia, Some((3, 0, 0)));
        assert_eq!(rec.negative_evals, Some(0));
        assert_eq!(rec.two_by_two_pivots, Some(0));
        assert_eq!(rec.perturbed_pivots, Some(0));
        assert_eq!(rec.inertia_reliable, Some(true));
        assert!(rec.min_abs_pivot.is_some() && rec.nnz_l.is_some());
        assert!(rec.refactor_ms.is_some() && rec.solve_ms.is_some());
        assert!(rec.is_acceptable(1e-10), "{rec:?}");
    }
}
