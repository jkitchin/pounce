//! Per-solve counters and timers.
//!
//! Mirrors `Interfaces/IpSolveStatistics.{hpp,cpp}`. Values are
//! populated by `IpoptApplication` after a successful solve. This is
//! a Phase-3 skeleton — the cumulative timer bookkeeping is wired up
//! in Phase 7 once `IpoptAlg` is producing iterations.

use pounce_common::types::{Index, Number};

/// One row of per-iteration data — same numbers that
/// `IpoptAlgorithm` prints to stdout each iteration (the "iter
/// objective inf_pr inf_du lg(mu) ||d|| lg(rg) alpha_du alpha_pr ls"
/// line). Captured into [`SolveStatistics::iterations`] when a
/// JSON / programmatic consumer needs the trajectory rather than
/// just the final state.
///
/// Field semantics mirror upstream `IpOrigIterationOutput.cpp:152`
/// (`Snprintf` block) so a row in JSON round-trips back into the
/// same console table verbatim.
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IterRecord {
    /// Iteration index, starting at 0.
    pub iter: Index,
    /// Unscaled objective `f(x_k)` at the start of iter `k`.
    pub objective: Number,
    /// Primal infeasibility (max-norm of constraint violation).
    pub inf_pr: Number,
    /// Dual infeasibility (max-norm of grad-Lagrangian).
    pub inf_du: Number,
    /// Barrier parameter μ.
    pub mu: Number,
    /// `||d_xs||_∞` of the search step. `0.0` on iter 0 (no step yet).
    pub d_norm: Number,
    /// Hessian regularization `δ_w` applied this iter; `0.0` when
    /// no regularization was needed (printed as `-` in the console).
    pub regularization: Number,
    /// Dual step length.
    pub alpha_dual: Number,
    /// Primal step length.
    pub alpha_primal: Number,
    /// Single-character tag for the alpha-primal column (`f`, `h`,
    /// `r` for restoration etc.) — matches upstream's per-iter tag.
    pub alpha_primal_char: char,
    /// Number of backtracking line-search trials this iter.
    pub ls_trials: Index,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolveStatistics {
    pub iteration_count: Index,
    pub total_cpu_time_secs: Number,
    pub total_sys_time_secs: Number,
    pub total_wallclock_time_secs: Number,
    pub num_obj_evals: Index,
    pub num_constr_evals: Index,
    pub num_obj_grad_evals: Index,
    pub num_constr_jac_evals: Index,
    pub num_hess_evals: Index,
    pub final_objective: Number,
    pub final_scaled_objective: Number,
    pub final_dual_inf: Number,
    pub final_constr_viol: Number,
    pub final_compl: Number,
    pub final_kkt_error: Number,
    // Unscaled (user-original-space) counterparts of the four residuals
    // above. The `final_*` fields are max-norms in the internally-scaled
    // NLP space (objective × df, constraints × dc); these divide the
    // nlp_scaling back out so a consumer can verify a returned KKT
    // certificate in its own units. Equal to the scaled fields when no
    // nlp_scaling is active. `final_unscaled_kkt_error` is the plain
    // max-norm of the three (no s_d/s_c optimality scaling). (pounce#173)
    pub final_unscaled_dual_inf: Number,
    pub final_unscaled_constr_viol: Number,
    pub final_unscaled_compl: Number,
    pub final_unscaled_kkt_error: Number,
    /// Final barrier parameter μ at termination (the IPM's `curr_mu`
    /// after the last iterate). Lets a caller thread the converged
    /// barrier into a warm-started re-solve's `mu_init` /
    /// `warm_start_target_mu` for predictor–corrector path following
    /// (pounce#86). `0.0` on the barrier-free SQP path, where μ has
    /// no meaning.
    pub final_mu: Number,

    // ---- Restoration-phase audit counters (pounce#12). ----
    //
    // Populated by `IpoptApplication::optimize_constrained` after a
    // solve completes. All three are 0 when restoration never fires.
    //
    /// Number of times `IpoptAlgorithm::invoke_restoration` was
    /// entered during this solve.
    pub restoration_calls: Index,
    /// Cumulative inner-IPM iteration count across every restoration
    /// call (sum of `RestoSolveResult::iter_count`). Each restoration
    /// call's inner IPM runs to its own convergence; this is the
    /// total work the inner solver did.
    pub restoration_inner_iters: Index,
    /// Number of outer iterations that ran in restoration mode (the
    /// `r`-suffix iter lines visible in `print_level=5` output).
    /// Counts outer iters where the IPM was driving a restoration
    /// trial step rather than a normal Newton step.
    pub restoration_outer_iters: Index,
    /// Cumulative wall-clock seconds spent inside `perform_restoration`
    /// across all restoration calls. Useful for "what fraction of the
    /// solve was restoration?" without running with high print_level.
    pub restoration_wall_secs: Number,

    /// Per-iteration trajectory. Empty when the consumer doesn't ask
    /// for it (`iter_history_enabled = false` on the application or
    /// the binary's `--json-detail summary` mode). Populated in order
    /// by [`IpoptAlgorithm::iterate`] when enabled.
    pub iterations: Vec<IterRecord>,
}

/// The eight residual fields default to **NaN, not zero**.
///
/// They are populated by the convergence check at the end of a solve. A solve
/// that never gets that far -- rejected during setup (`Not_Enough_Degrees_Of_Freedom`,
/// `Invalid_Problem_Definition`), aborted, or caught by the batch panic
/// handler -- leaves them untouched, and a default of `0.0` there reads as
/// "converged perfectly" rather than "never computed".
///
/// That is not hypothetical. `pounce.minimize` upgrades a non-success status
/// to `success=True` when the final KKT error is within the acceptable
/// tolerance, which is right for a solve that stalled near a good point. With
/// a zero default it also fired for problems the solver had *refused*: an
/// over-determined NLP returned `Not_Enough_Degrees_Of_Freedom` together with
/// `success=True` and an `x` outside its own variable bounds. NaN makes the
/// existing `is_finite` guard on that path do what its comment already claims.
///
/// Consequences worth knowing:
///
/// * NaN compares false against everything, so any `residual <= tol` test now
///   fails closed for an uncomputed value. That is the intent.
/// * `serde_json` renders non-finite floats as `null`, so these fields appear
///   as `null` rather than `0.0` in a solve report for an aborted solve. See
///   `docs/src/schema/solve-report-v1.md`.
///
/// The two objective fields are in the set for the same reason, though the
/// stakes are lower: nothing *decides* anything from them, they are only
/// reported (console summary, studio markdown, the JSON report). But `0.0` is
/// a perfectly ordinary objective value, so a reader cannot tell a solve that
/// legitimately reached zero from one that never evaluated anything. One rule
/// -- uncomputed is NaN -- is easier to reason about than "residuals are NaN,
/// objectives are zero, and you have to remember which is which". Note they
/// are seeded best-effort from the current iterate whenever one exists, so
/// they are only NaN when the solve died before producing any point at all.
///
/// `final_mu` is deliberately *not* in this set: `0.0` is its documented value
/// on the barrier-free SQP path, where mu has no meaning.
impl Default for SolveStatistics {
    fn default() -> Self {
        Self {
            iteration_count: 0,
            total_cpu_time_secs: 0.0,
            total_sys_time_secs: 0.0,
            total_wallclock_time_secs: 0.0,
            num_obj_evals: 0,
            num_constr_evals: 0,
            num_obj_grad_evals: 0,
            num_constr_jac_evals: 0,
            num_hess_evals: 0,
            final_objective: Number::NAN,
            final_scaled_objective: Number::NAN,
            final_dual_inf: Number::NAN,
            final_constr_viol: Number::NAN,
            final_compl: Number::NAN,
            final_kkt_error: Number::NAN,
            final_unscaled_dual_inf: Number::NAN,
            final_unscaled_constr_viol: Number::NAN,
            final_unscaled_compl: Number::NAN,
            final_unscaled_kkt_error: Number::NAN,
            final_mu: 0.0,
            restoration_calls: 0,
            restoration_inner_iters: 0,
            restoration_outer_iters: 0,
            restoration_wall_secs: 0.0,
            iterations: Vec::new(),
        }
    }
}

impl SolveStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}
