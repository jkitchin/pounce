//! A FERAL symbolic analysis run on its own, before the numeric factor
//! (gh#1249).
//!
//! `feral::Solver::factor` runs the symbolic analysis and the numeric
//! multifrontal factorization in one call, and POUNCE records the factor's
//! size (`factor_nnz` on the `linear_solve` span) only after that call
//! succeeds. A KKT whose factor does not fit in memory therefore never
//! reports anything: on five large Mittelmann LPs the convex HSDE reached the
//! memory cap inside that call, and the logs could not say whether the
//! symbolic phase or the numeric factor was the problem, or how large a
//! factor the chosen ordering predicted.
//!
//! The pre-analysis runs [`feral::symbolic::symbolic_factorize_with_method`]
//! with the same ordering, preprocessing and values the solver will use, and
//! reads FERAL's [`WorkEstimate`] off the result before any numeric work.
//! It runs only for `report_symbolic` ([`crate::FeralConfig`]), and logs: one
//! line when the analysis starts, one with the prediction when it ends. A run
//! killed with only the first line logged died in the symbolic phase; one
//! killed after the second died in the numeric factor, whose predicted size the
//! second line gives. Between the two phases it checks the [`factor_deadline`]
//! in force, so a symbolic phase that used up the wall-clock budget does not go
//! on into a numeric factor nothing will wait for.
//!
//! The cost: feral's `Solver` has no way to adopt an analysis computed
//! outside it, so with the pre-analysis on, the symbolic phase runs twice
//! for each new sparsity pattern (once here, once inside `factor`). It is
//! off by default for that reason. It never changes the factor: the solver's
//! own analysis is untouched, and the pre-analysis result is dropped once
//! read.
//!
//! The two other things gh#1249 needed no longer go through here (gh#1265).
//! The byte budget (`max_factor_bytes`) is FERAL's own
//! (`Solver::with_max_factor_bytes`, feral >= 0.25), checked inside the
//! solver's single analysis, and a refusal is latched here
//! ([`budget_refusal`]). The [`factor_deadline`] reaches FERAL as a
//! `feral::Interrupt`, which stops an analysis — this one or the solver's —
//! at its next safe point instead of only between the phases.

use std::cell::Cell;
use std::time::Instant;

use feral::CscMatrix;
use feral::symbolic::{
    OrderingMethod, OrderingPreprocess, SupernodeParams, WorkEstimate,
    symbolic_factorize_with_method,
};

thread_local! {
    /// The deadline the FERAL pre-analysis honours on this thread, if any.
    static FACTOR_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    /// The over-budget refusal latched on this thread, if any. See
    /// [`budget_refusal`].
    static BUDGET_REFUSAL: Cell<Option<BudgetRefusal>> = const { Cell::new(None) };
}

/// A factorization refused because FERAL predicted more memory than
/// `max_factor_bytes` allows (gh#1249).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetRefusal {
    /// The predicted peak, FERAL's `WorkEstimate::peak_bytes`: the cached
    /// analysis, the solver's workspace and the numeric phase's high-water
    /// (feral >= 0.25; before that, `factor_bytes + peak_contrib_bytes`).
    ///
    /// When [`Self::stage`] is `"factor"`, FERAL refused on the factor's own
    /// bytes before computing the full peak, and this is that lower bound.
    pub peak_bytes: usize,
    /// The budget it exceeded.
    pub budget: u64,
    /// Which of FERAL's checks fired: `"factor"` (the factor alone is over)
    /// or `"peak"` (the full predicted peak is).
    pub stage: &'static str,
}

/// The over-budget refusal latched on this thread since the last
/// [`clear_budget_refusal`], if any.
///
/// A refusal is terminal for the solve, not for one factorization: while it is
/// latched, every backend on this thread that has a budget refuses at once,
/// without running another symbolic analysis. Without the latch a refusal fell
/// through to the next thing that factors — the convex driver's fallback
/// driver, the CLI's LP → NLP fallback, the NLP arm's restoration phase — and
/// each paid a fresh symbolic pass on its own pattern (gh#1250 review: on
/// `fhnw-binschedule1`, four analyses and a run past `max_wall_time`). Callers
/// that would reroute on a failure read this to report the refusal instead.
///
/// Cleared when a top-level solve starts: the convex drivers' outermost
/// deadline scope and the NLP application's `optimize_tnlp` both do.
pub fn budget_refusal() -> Option<BudgetRefusal> {
    BUDGET_REFUSAL.with(Cell::get)
}

/// Forget any latched [`budget_refusal`] on this thread.
pub fn clear_budget_refusal() {
    BUDGET_REFUSAL.with(|r| r.set(None));
}

pub(crate) fn latch_budget_refusal(r: BudgetRefusal) {
    BUDGET_REFUSAL.with(|c| c.set(Some(r)));
}

/// Restores the previous factor deadline when dropped. See
/// [`factor_deadline`].
#[must_use = "the deadline is lifted when the guard is dropped"]
pub struct FactorDeadline {
    prev: Option<Instant>,
}

impl Drop for FactorDeadline {
    fn drop(&mut self) {
        let prev = self.prev;
        FACTOR_DEADLINE.with(|d| d.set(prev));
    }
}

/// Make FERAL factorizations on this thread stop once `deadline` has passed,
/// until the returned guard is dropped: the symbolic analysis at its next safe
/// point (a `feral::Interrupt`, gh#1265), and with the pre-analysis on, the
/// numeric factor before it starts. A factorization already in its numeric
/// phase runs to completion. A deadline passes this way, not as an argument, because the factor
/// is reached through `SparseSymLinearSolverInterface::factor`, whose
/// signature every backend shares.
///
/// Scopes nest, and an inner one can only tighten: the effective deadline is
/// the earlier of the two. `None` leaves the enclosing deadline in force.
pub fn factor_deadline(deadline: Option<Instant>) -> FactorDeadline {
    let prev = FACTOR_DEADLINE.with(Cell::get);
    let eff = match (prev, deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    FACTOR_DEADLINE.with(|d| d.set(eff));
    FactorDeadline { prev }
}

/// The [`factor_deadline`] in force on this thread, if any.
pub(crate) fn current_factor_deadline() -> Option<Instant> {
    FACTOR_DEADLINE.with(Cell::get)
}

/// Whether the [`factor_deadline`] in force on this thread has passed.
pub fn factor_deadline_passed() -> bool {
    FACTOR_DEADLINE
        .with(Cell::get)
        .is_some_and(|d| Instant::now() >= d)
}

/// What the pre-analysis learned about one sparsity pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicReport {
    /// Order of the matrix.
    pub n: usize,
    /// Stored nonzeros of the (lower-triangle) matrix.
    pub nnz_a: usize,
    /// The ordering FERAL resolved to (`auto` resolved to a concrete method).
    pub ordering: String,
    /// The ordering preprocessing it resolved to.
    pub preprocess: String,
    /// Wall-clock seconds the symbolic analysis took.
    pub symbolic_secs: f64,
    /// FERAL's a-priori estimate of the numeric factorization.
    pub work: WorkEstimate,
}

/// What the pre-analysis tells the factorization to do.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Go on to the numeric factor.
    Proceed,
    /// The [`factor_deadline`] passed during the symbolic analysis, which
    /// completed.
    DeadlinePassed,
    /// The [`factor_deadline`] interrupted the symbolic analysis; nothing
    /// was learned about the pattern.
    Interrupted,
    /// FERAL refused the analysis on the `max_factor_bytes` budget in
    /// `params` (gh#1265), at `stage` (`"factor"` or `"peak"`).
    OverBudget {
        predicted_bytes: usize,
        budget_bytes: usize,
        stage: &'static str,
    },
    /// The symbolic analysis itself failed; the numeric factor would fail
    /// the same way and reports it.
    AnalysisFailed,
}

/// `bytes` as a short human-readable figure (`1.23 GB`).
pub fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1000.0 && u + 1 < UNITS.len() {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

pub(crate) fn preprocess_label(p: OrderingPreprocess) -> &'static str {
    match p {
        OrderingPreprocess::None => "none",
        OrderingPreprocess::LdltCompress => "ldlt_compress",
        OrderingPreprocess::Auto => "auto",
    }
}

/// Run the symbolic analysis `matrix` would get from a solver configured with
/// `params` and `ordering`, log it when `report` is set, and decide whether
/// the numeric factor should go ahead.
pub(crate) fn preanalyze(
    matrix: &CscMatrix,
    params: &SupernodeParams,
    ordering: &OrderingMethod,
    report: bool,
) -> (Option<SymbolicReport>, Verdict) {
    let n = matrix.n;
    let nnz_a = matrix.row_idx.len();
    if report {
        tracing::info!(
            target: "pounce::linsol",
            n, nnz_a,
            ordering = %crate::ordering_label(ordering),
            preprocess = preprocess_label(params.preprocess),
            "FERAL symbolic analysis starting: n = {n}, nnz(A) = {nnz_a}, ordering = {}, \
             preprocess = {}",
            crate::ordering_label(ordering),
            preprocess_label(params.preprocess),
        );
    }
    let t0 = Instant::now();
    let sym = match symbolic_factorize_with_method(matrix, params, ordering.clone()) {
        Ok(sym) => sym,
        Err(feral::FeralError::Interrupted { stage }) => {
            tracing::info!(
                target: "pounce::linsol",
                "FERAL symbolic analysis interrupted ({stage}): the wall-clock deadline passed"
            );
            return (None, Verdict::Interrupted);
        }
        Err(feral::FeralError::MemoryBudgetExceeded {
            predicted_bytes,
            budget_bytes,
            stage,
        }) => {
            return (
                None,
                Verdict::OverBudget {
                    predicted_bytes,
                    budget_bytes,
                    stage,
                },
            );
        }
        Err(e) => {
            tracing::warn!(
                target: "pounce::linsol",
                "FERAL symbolic pre-analysis failed ({e}); leaving the verdict to the factorization"
            );
            return (None, Verdict::AnalysisFailed);
        }
    };
    let symbolic_secs = t0.elapsed().as_secs_f64();
    let rep = SymbolicReport {
        n,
        nnz_a,
        ordering: crate::ordering_label(&sym.resolved_method),
        preprocess: preprocess_label(sym.resolved_preprocess).to_string(),
        symbolic_secs,
        work: sym.work_estimate(),
    };
    // The analysis holds O(nnz(A) + n_supernodes) of patterns and static row
    // lists; free it before the solver computes its own.
    drop(sym);
    let w = &rep.work;
    if report {
        tracing::info!(
            target: "pounce::linsol",
            n,
            ordering = %rep.ordering,
            preprocess = %rep.preprocess,
            symbolic_secs,
            factor_nnz = w.factor_nnz,
            factor_alloc_nnz = w.factor_alloc_nnz,
            factor_bytes = w.factor_bytes,
            factor_stored_bytes = w.factor_stored_bytes,
            peak_contrib_bytes = w.peak_contrib_bytes,
            analysis_bytes = w.analysis_bytes,
            workspace_bytes = w.workspace_bytes,
            numeric_peak_bytes = w.numeric_peak_bytes,
            peak_bytes = w.peak_bytes,
            max_front_rows = w.max_front_rows,
            n_supernodes = w.n_supernodes,
            factor_flops = w.factor_flops,
            "FERAL symbolic analysis done in {symbolic_secs:.2} s: ordering = {} ({}), \
             predicted nnz(L) = {} (allocated {}), factor {}, numeric phase {} + \
             analysis {} + workspace {} = peak {}, largest front {} rows, \
             {} supernodes, {:.3e} flops",
            rep.ordering,
            rep.preprocess,
            w.factor_nnz,
            w.factor_alloc_nnz,
            human_bytes(w.factor_stored_bytes),
            human_bytes(w.numeric_peak_bytes),
            human_bytes(w.analysis_bytes),
            human_bytes(w.workspace_bytes),
            human_bytes(w.peak_bytes),
            w.max_front_rows,
            w.n_supernodes,
            w.factor_flops,
        );
    }
    if factor_deadline_passed() {
        tracing::info!(
            target: "pounce::linsol",
            symbolic_secs,
            "FERAL numeric factorization not started: the wall-clock deadline passed \
             during the {symbolic_secs:.2} s symbolic analysis"
        );
        return (Some(rep), Verdict::DeadlinePassed);
    }
    (Some(rep), Verdict::Proceed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_picks_a_unit() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1_500), "1.50 KB");
        assert_eq!(human_bytes(38_600_000_000), "38.60 GB");
    }

    #[test]
    fn deadline_scopes_nest_and_only_tighten() {
        assert!(!factor_deadline_passed());
        let past = Instant::now();
        {
            let _g = factor_deadline(Some(past));
            assert!(factor_deadline_passed());
            {
                let _later = factor_deadline(Some(past + std::time::Duration::from_secs(3600)));
                assert!(factor_deadline_passed(), "an inner scope cannot loosen");
            }
            {
                let _none = factor_deadline(None);
                assert!(
                    factor_deadline_passed(),
                    "None keeps the enclosing deadline"
                );
            }
        }
        assert!(
            !factor_deadline_passed(),
            "the guard restores the previous state"
        );
    }
}
