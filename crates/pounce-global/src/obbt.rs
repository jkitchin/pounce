//! Optimization-based bound tightening (OBBT).
//!
//! FBBT tightens a box by interval propagation through each constraint; OBBT
//! tightens it by *optimizing*: minimize and maximize each variable over the
//! whole relaxation (the same polyhedral outer approximation used for the
//! lower bound), optionally with an incumbent-cutoff row `objective ≤ ub`. The
//! relaxation contains every truly feasible point, so its min/max of `x_i` are
//! valid new bounds — usually much tighter than FBBT's, at the cost of `2n` LP
//! solves per pass. This is the single biggest box-reducer in commercial
//! global solvers; it is gated by frequency in the driver because of that cost.

use crate::problem::GlobalProblem;
use crate::relax::{build_relaxation, Relaxation};
use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_linsol::SparseSymLinearSolverInterface;
use std::time::Instant;

/// The OBBT result for one variable: `(min xᵢ, max xᵢ)`, each `None` if that
/// solve was not optimal.
pub(crate) type VarRange = (Option<f64>, Option<f64>);

/// Largest relaxation-LP row count (`m_eq + m_ineq`, cutoff row included) for
/// which [`ObbtLp::Simplex`] is actually used. `pounce-simplex` keeps an
/// **explicit dense** basis inverse — `O(m²)` memory, `O(m²)` per pivot, `O(m³)`
/// per refactor — sized for the *small* post-FBBT OBBT LPs. Above this many
/// rows that dense engine would stall, so a pass that requested `Simplex`
/// silently falls back to the IPM sweep for that pass. Both engines return
/// valid bounds over the same polytope, so this trades only *engine*, never
/// correctness or tightness. Phase 6.2's sparse-factored basis is what lifts
/// the cap; until then this keeps a large relaxation from stalling the search.
/// Heuristic — tune against profiles.
pub const SIMPLEX_DENSE_MAX_ROWS: usize = 800;

/// Which LP engine OBBT uses for its `2n` min/max solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObbtLp {
    /// The HSDE interior-point method (`pounce-convex`). Each of the `2n` solves
    /// is a cold central-path walk. The long-standing default.
    #[default]
    Ipm,
    /// **Gated behind the off-by-default `simplex-obbt` feature, pending broader
    /// validation.** The warm-started revised simplex (`pounce-simplex`): one
    /// basis built per pass and warm-started across all `2n` objective flips.
    /// Serial within a pass (the basis is threaded sequentially), so `parallel`
    /// does not apply to it.
    ///
    /// History: an earlier dense explicit-inverse engine returned **wrong**
    /// certified optima on badly-scaled relaxation LPs (it reported an LP optimum
    /// that was not the true min/max, so OBBT tightened a box past the true bound
    /// and cut the global optimum). Phase 6.2 replaced that with a factored
    /// sparse LU (faer) plus geometric-mean equilibration, and two distinct
    /// scaling failure modes have since been found and fixed with regression
    /// guards: GLOBALLib `ex4_1_2` (`tests/ill_scaled_obbt.rs`) and a collapsed
    /// McCormick coefficient on a quartic child box
    /// (`tests/degenerate_mccormick_scaling.rs`). The 0-WRONG gate
    /// (`simplex_obbt_matches_ipm_certified_optimum`) now passes: simplex OBBT
    /// certifies the same optima as [`Ipm`] across the test spread.
    ///
    /// It stays gated because each fix so far has been a scaling-robustness
    /// patch, so a wider GLOBALLib cross-check against [`Ipm`] is wanted before
    /// graduating it to the default. Without the feature this variant is inert:
    /// [`tighten`] transparently runs the [`Ipm`] sweep instead (and
    /// `pounce-simplex` is not even linked).
    Simplex,
}

/// Outcome of an OBBT sweep over a node's box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObbtOutcome {
    /// The relaxation is infeasible over the box — the node can be pruned.
    Infeasible,
    /// Tightening completed (possibly stopping early on a no-improvement pass).
    /// `lo`/`hi` hold the tightened bounds.
    Done,
    /// The deadline fired mid-sweep. Any bounds applied so far are still valid,
    /// but the node was not fully bounded, so the driver treats it as
    /// timed-out (it remains a *live* node, not a pruned one).
    TimedOut,
}

/// `true` once the (optional) monotonic deadline has passed. A cheap `Instant`
/// compare — no float, no syscall beyond reading the monotonic clock.
pub(crate) fn past_deadline(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// Pick the `max_vars` variables with the widest current box side
/// (`hi[i] − lo[i]`) for a budgeted OBBT pass. Returns a length-`n` mask
/// (`true` = tighten this var), or `None` when `max_vars >= n` (the default —
/// tighten every variable, no allocation, no filtering). Widest-box selection
/// is cheap, deterministic, and targets the variables that most slow branching;
/// any subset is sound, since tightening fewer variables only relaxes the box
/// less than the full sweep. Ties broken by index (stable sort), so the choice
/// is reproducible run to run.
pub(crate) fn select_widest_vars(lo: &[f64], hi: &[f64], max_vars: usize) -> Option<Vec<bool>> {
    let n = lo.len();
    if max_vars >= n {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    // Descending by width; stable so equal widths keep index order.
    idx.sort_by(|&a, &b| {
        let wa = hi[a] - lo[a];
        let wb = hi[b] - lo[b];
        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut mask = vec![false; n];
    for &i in idx.iter().take(max_vars) {
        mask[i] = true;
    }
    Some(mask)
}

/// Tighten `[lo, hi]` in place by OBBT. `cutoff`, when set, adds the row
/// `objective ≤ cutoff` (the incumbent), which lets OBBT exploit that no
/// improving point exceeds the incumbent. Returns [`ObbtOutcome::Infeasible`]
/// if the relaxation is infeasible over the box (the node can then be pruned).
/// `parallel` runs each pass's `2n` min/max solves on a thread pool. They are
/// independent (all use the same pass-start relaxation), so the result is
/// identical to the serial sweep — only faster.
///
/// `deadline`, when set, bounds the sweep: it is polled at the top of each pass
/// and (in the serial sweep) before each variable's min/max pair. If it fires,
/// any bounds already applied are kept (every OBBT bound is independently
/// valid) and [`ObbtOutcome::TimedOut`] is returned so the driver can stop the
/// search without dropping the still-live node.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tighten<F>(
    prob: &GlobalProblem,
    lo: &mut [f64],
    hi: &mut [f64],
    cutoff: Option<f64>,
    passes: usize,
    parallel: bool,
    lp: ObbtLp,
    max_vars: usize,
    opts: &QpOptions,
    make_backend: &F,
    deadline: Option<Instant>,
    reuse_out: &mut Option<Relaxation>,
) -> ObbtOutcome
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    use rayon::prelude::*;

    let n = prob.n_vars;
    for _ in 0..passes {
        if past_deadline(deadline) {
            return ObbtOutcome::TimedOut;
        }
        // Budgeted sweep: when `max_vars < n`, tighten only the widest-box
        // variables this pass (`None` ⇒ all `n`, the default fast path). A mask
        // of length `n`; `targets[i] == false` variables are skipped (their
        // result is `(None, None)`), so the engine does `2·max_vars` LP solves
        // instead of `2n`. Tightening a subset is a strict subset of the full
        // sweep — bounds stay valid, so this is sound.
        let targets: Option<Vec<bool>> = select_widest_vars(lo, hi, max_vars);
        let relax = build_relaxation(prob, lo, hi, true);
        if relax.trivially_infeasible {
            return ObbtOutcome::Infeasible;
        }
        let mut qp = relax.qp;
        // Remember the base row counts so the cutoff cut can be peeled back off
        // for relaxation reuse (Phase 4.3): if this pass tightens nothing, this
        // relaxation is over the *final* box and `build_relaxation` would rebuild
        // it bit-for-bit, so the node lower-bound stage can reuse it instead.
        let base_g_len = qp.g.len();
        let base_h_len = qp.h.len();
        if let (Some(cut), Some(oc)) = (cutoff, relax.obj_col) {
            let row = qp.h.len();
            qp.g.push(Triplet::new(row, oc, 1.0));
            qp.h.push(cut);
        }

        // Engine selection. A `Simplex` request degrades to the IPM sweep when
        // either (a) the `simplex-obbt` feature is off — the default; the engine
        // is gated pending broader validation and is not even linked, so every
        // request runs the IPM path — or (b) the relaxation exceeds
        // `SIMPLEX_DENSE_MAX_ROWS`, where the simplex is slower than the IPM on
        // these larger systems. Both fall back to the same polytope's IPM sweep:
        // same valid bounds, just a different (sound, scalable) engine.
        let engine = if matches!(lp, ObbtLp::Simplex)
            && (cfg!(not(feature = "simplex-obbt"))
                || qp.m_eq() + qp.m_ineq() > SIMPLEX_DENSE_MAX_ROWS)
        {
            ObbtLp::Ipm
        } else {
            lp
        };

        // All `2n` min/max LPs share this pass's polytope (`A,b,G,h`, bounds) and
        // differ only in the objective `c`. Two engines compute the sweep
        // (selected by `engine`); both yield `(min x_i, max x_i)` per variable,
        // each `None` if that solve was not optimal. A pass may end early on the
        // deadline; partial results are still valid bounds, so we apply them and
        // then report `TimedOut`.
        let (results, timed_out): (Vec<VarRange>, bool) = match engine {
            // Warm-started revised simplex: one basis per pass, reused across all
            // `2n` objective flips (a few pivots each). Serial by construction —
            // the basis threads through the sweep — so `parallel` does not apply.
            // An IPM `QpFactorization` reuse was tried instead (Phase 1) and
            // measured ~13× *slower*: after FBBT the per-node LPs are small and
            // the direct IPM's fixed per-solve overhead dominated. Simplex
            // warm-start is the genuine lever for these same-polytope LPs.
            #[cfg(feature = "simplex-obbt")]
            ObbtLp::Simplex => crate::simplex_bridge::sweep(&qp, n, targets.as_deref(), deadline),
            // Without the feature, the selection above never yields `Simplex`;
            // this arm only satisfies exhaustiveness and is never reached.
            #[cfg(not(feature = "simplex-obbt"))]
            ObbtLp::Simplex => unreachable!(
                "ObbtLp::Simplex is downgraded to Ipm when the `simplex-obbt` feature is off"
            ),
            // Interior-point: `2n` independent cold HSDE solves, optionally on a
            // thread pool. `c` is zeroed in a representative `work`; each var sets
            // `c[i]=±1` then restores it.
            ObbtLp::Ipm => {
                let base = {
                    let mut b = qp.clone();
                    b.c.iter_mut().for_each(|c| *c = 0.0);
                    b
                };
                // `work` is per-worker scratch (one objective-mutated clone),
                // threaded in so the parallel path reuses it across variables.
                let solve_var = |work: &mut QpProblem, i: usize| -> (Option<f64>, Option<f64>) {
                    let mut mk = || make_backend();
                    work.c[i] = 1.0;
                    let smin = solve_qp_ipm(work, opts, &mut mk);
                    let min = (smin.status == QpStatus::Optimal).then_some(smin.obj);
                    work.c[i] = -1.0;
                    let smax = solve_qp_ipm(work, opts, &mut mk);
                    let max = (smax.status == QpStatus::Optimal).then_some(-smax.obj);
                    work.c[i] = 0.0;
                    (min, max)
                };
                let mut timed_out = false;
                let results: Vec<(Option<f64>, Option<f64>)> = if parallel {
                    // Per-worker scratch problem, built lazily by `map_init`.
                    // Workers already in flight finish; not-yet-started ones short
                    // out on the deadline.
                    let r = (0..n)
                        .into_par_iter()
                        .map_init(
                            || base.clone(),
                            |work, i| {
                                if past_deadline(deadline)
                                    || targets.as_ref().is_some_and(|t| !t[i])
                                {
                                    (None, None)
                                } else {
                                    solve_var(work, i)
                                }
                            },
                        )
                        .collect();
                    timed_out = past_deadline(deadline);
                    r
                } else {
                    let mut work = base.clone();
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        if past_deadline(deadline) {
                            timed_out = true;
                            break;
                        }
                        if targets.as_ref().is_some_and(|t| !t[i]) {
                            out.push((None, None));
                            continue;
                        }
                        out.push(solve_var(&mut work, i));
                    }
                    out
                };
                (results, timed_out)
            }
        };

        let mut improved = false;
        for (i, (min, max)) in results.into_iter().enumerate() {
            if let Some(v) = min {
                if v > lo[i] + 1e-9 {
                    lo[i] = v.min(hi[i]);
                    improved = true;
                }
            }
            if let Some(v) = max {
                if v < hi[i] - 1e-9 {
                    hi[i] = v.max(lo[i]);
                    improved = true;
                }
            }
            if lo[i] > hi[i] + 1e-9 {
                return ObbtOutcome::Infeasible; // box collapsed → infeasible
            }
        }
        if timed_out {
            return ObbtOutcome::TimedOut;
        }
        if !improved {
            // The box is final (this pass changed nothing), so `qp` minus the
            // appended cutoff cut is exactly `build_relaxation(prob, lo, hi,
            // true)` over the final box. Hand it back for the node lower-bound
            // stage to reuse instead of rebuilding it. Peel the cutoff cut off so
            // the reused relaxation matches a fresh build bit-for-bit; the
            // multilinear flag is `true` here, which the caller checks against
            // `opts.multilinear` before reusing.
            qp.g.truncate(base_g_len);
            qp.h.truncate(base_h_len);
            *reuse_out = Some(Relaxation {
                qp,
                atoms: relax.atoms,
                branch_terms: relax.branch_terms,
                obj_col: relax.obj_col,
                trivially_infeasible: false,
            });
            break;
        }
    }
    ObbtOutcome::Done
}
