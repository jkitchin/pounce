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
    /// How many bound and slack indices the barrier currently treats as
    /// active — the **phase profile** telemetry.
    ///
    /// An interior-point iterate never sits *on* a constraint, so
    /// "active" here is an inference, not a fact the solve established:
    /// index `i` is counted active when its bound multiplier exceeds its
    /// slack (`z_i > s_i`), which is the split of `s_i · z_i ≈ mu`
    /// between its two factors. A strongly active bound drives
    /// `s = O(mu)` against `z = O(1)`, an inactive one the reverse, so
    /// the comparison separates them by the barrier's own geometry.
    ///
    /// **The partition is frame-dependent and the count is therefore a
    /// within-run diagnostic only.** Under a per-variable rescaling the
    /// slack carries `d` and the multiplier `d⁻¹`, so the boundary moves
    /// while `s · z` does not; no test built from `(s_i, z_i, mu)` alone
    /// can be scale-free, because their only dimensionless combination
    /// is `s_i z_i / mu ≈ 1`. The comparison is taken in the solver's
    /// internal scaled frame — the one in which `kappa_sigma` and the
    /// fraction-to-boundary rule already compare these same quantities.
    /// Do not compare the count across models, across scalings, or
    /// against a scale-invariant classifier: `pounce-sensitivity`'s
    /// `classify_activity` normalizes `Sigma` by curvature instead and
    /// answers a different, more expensive question (one back-solve per
    /// index for the reduced form).
    ///
    /// **Iteration 0 of a cold start measures the initializer, not the
    /// problem.** The bound multipliers there are `bound_mult_init_val`
    /// — one constant for every index — rather than a dual estimate, so
    /// almost every index with a small starting slack classifies active
    /// and collapses on the first real dual update. On `laptime` at
    /// `N = 320` row 0 reads 22110 against 563 at the solution, and the
    /// change count on row 1 is that collapse rather than a measured
    /// move. Skip both rows when reading a cold run's profile; under a
    /// warm start that supplies bound duals the row is meaningful.
    ///
    /// `None` when the solver path does not measure it (the convex IPM
    /// report rows, and any record built by a consumer rather than the
    /// NLP iteration loop).
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub active_bounds: Option<Index>,
    /// How many indices changed activity class since the previous
    /// captured iteration — the number the phase profile is actually
    /// read off.
    ///
    /// Large while the solve is still deciding which constraints bind,
    /// falling to zero once the active set has settled. The split
    /// between "iterations spent finding the active set" and
    /// "iterations spent converging on it" is what decides whether a
    /// staged solve (coarse-to-fine, a cheap approach phase handed to a
    /// more accurate endgame) can pay at all.
    ///
    /// `None` on the first captured iteration — there is no predecessor
    /// to compare against, which is a different statement from "nothing
    /// changed" — and `None` on paths that do not measure it. Carries
    /// the same frame caveat as [`Self::active_bounds`]: a rescaling
    /// moves the boundary, so indices that sit near it can be counted
    /// differently, and the count is comparable only within one run.
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub active_set_changes: Option<Index>,
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
    /// Primal violation measured against the model **as declared**, before
    /// the `bound_relax_factor` widening the convex arm applies
    /// (`qp_extract::BoundRelax`, gh #744/#745).
    ///
    /// `final_constr_viol` measures the model the solver was HANDED, whose
    /// inequality rows and variable box are widened by
    /// `min(factor, cap)·|b|`. That is the right model for the convergence
    /// test, and it is what every acceptance gate reads — but it is not how
    /// far the returned point sits outside the model the caller wrote. On
    /// netlib `afiro` the point is `4.99e-06` outside a declared row
    /// `b = 500` (exactly `1e-8·500`) while `final_constr_viol` reads
    /// `8.68e-13`; on `25fv47` it is `1.97e-05` against `2.19e-11`.
    ///
    /// Reported so a caller can tell the two apart rather than reading the
    /// widened number as its own model's feasibility. `NaN` when the solve
    /// applied no widening (the two coincide) or on paths that do not
    /// compute it.
    pub final_declared_constr_viol: Number,
    /// How far the returned point sits outside the **declared** variable box
    /// — the box the caller wrote, before the `bound_relax_factor` widening.
    /// This is Ipopt's `Variable bound violation`, and it is the box half of
    /// [`Self::final_declared_constr_viol`] reported on its own, because once
    /// the two are maxed together a box violation cannot be told from a row
    /// violation.
    ///
    /// Variable bounds carry no scaling — POUNCE scales the objective and the
    /// constraint rows only — so there is no scaled/unscaled pair here; the
    /// one number is right in both columns.
    ///
    /// `NaN` on a path that does not compute it.
    pub final_declared_box_viol: Number,
    pub final_unscaled_dual_inf: Number,
    pub final_unscaled_constr_viol: Number,
    pub final_unscaled_compl: Number,
    pub final_unscaled_kkt_error: Number,
    /// `final_kkt_error` with each constraint row's residual counted only
    /// where it rises above what that row can represent in floating point —
    /// the aggregate the **strict** convergence gate actually tests (gh #528).
    /// Equal to `final_kkt_error` on every problem whose data is `O(1)`, and
    /// smaller only where a row is at its own resolution limit. Reported so a
    /// summary that ends `EXIT: Optimal Solution Found` beside an error above
    /// `tol` accounts for the gap rather than merely presenting it.
    pub final_kkt_error_above_noise: Number,
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
    /// Finite-difference Hessian census, when
    /// `hessian_approximation=finite-difference` actually built a pattern.
    /// All zero / `-1` on every other Hessian mode, which is how a caller
    /// tells "the mode did not run" from "it ran with an empty pattern".
    ///
    /// `fd_hessian_pattern_used` is the source the run **ended up with**,
    /// not the one requested: `0` declared, `1` jacobian, `-1` not run.
    /// `declared` silently falls back to `jacobian` when the TNLP declares
    /// no Hessian structure, and that fallback is the difference between
    /// 17 probe groups and 341 on `benchmarks/large_scale` `laptime`, so
    /// reporting the request would hide the number a reader is here for.
    pub fd_hessian_pattern_used: Index,
    /// Hessian nonzeros in the pattern that was coloured (lower triangle).
    pub fd_hessian_nnz: Index,
    /// Columns the colouring ran over, i.e. the problem's variable count.
    /// Present so the report is self-contained: `groups / n` is the
    /// compression, the fraction of a dense finite-difference scheme's
    /// probes this pattern costs, and without `n` a reader cannot form it.
    pub fd_hessian_n: Index,
    /// Probe groups per Hessian — the count of extra gradient/Jacobian
    /// evaluations each rebuild costs.
    pub fd_hessian_groups: Index,
    /// Widest row of the pattern; the quantity that decides whether the
    /// colouring can stay narrow under mesh refinement.
    pub fd_hessian_rho_max: Index,
    /// Whether a requested star colouring failed validation and CPR was
    /// substituted.
    pub fd_hessian_coloring_fell_back: bool,
    /// Whether the objective clique fell back to a conservative structural
    /// set because the model stated no objective linearity. This is the
    /// field that explains a surprising `fd_hessian_groups`: the clique is
    /// then `N`, or all `n`, and the probe count reflects that rather than
    /// the objective's true support.
    pub fd_hessian_objective_clique_widened: bool,
    pub restoration_calls: Index,
    /// Cumulative inner-IPM iterations across every restoration call —
    /// the number of `r`-suffix rows a `print_level=5` log would show.
    ///
    /// Each call contributes its sub-solve's own *length*: the inner
    /// counter is seeded from the outer's at entry (upstream
    /// `IpRestoMinC_1Nrm.cpp:181`), so the length is the terminating value
    /// minus the outer count at entry. Before gh #819 this summed the
    /// terminating values themselves — absolute positions in a shared
    /// numbering, not lengths — and recorded `0` for any call that failed,
    /// which is the case a reader is looking at this field to understand.
    pub restoration_inner_iters: Index,
    /// Number of *outer* iterations consumed by restoration: one per call,
    /// so this always equals `restoration_calls`.
    ///
    /// It is not the count of `r`-suffix rows — that is
    /// `restoration_inner_iters`, and reading this field as those rows is
    /// what the doc comment here said until gh #819. Restoration in POUNCE
    /// is a nested solve entered from a single outer iteration, not a mode
    /// the outer loop runs in, so there is no third number here to report.
    pub restoration_outer_iters: Index,
    /// Cumulative wall-clock seconds spent inside `perform_restoration`
    /// across all restoration calls. Useful for "what fraction of the
    /// solve was restoration?" without running with high print_level.
    pub restoration_wall_secs: Number,

    /// Successful linear-solver quality escalations over the whole solve
    /// — the main loop's and every restoration sub-solve's — i.e. the
    /// count of `q` flags in the info-string column (gh#857).
    ///
    /// An escalation is not an error and not, on its own, a problem: it
    /// is how the IPM answers a factorization that will not deliver.
    /// But with the FERAL backend it *reroutes the rest of the solve*,
    /// because that backend's ladder changes which pivots are taken and
    /// never steps back down, so a run that ends badly having escalated
    /// is a different animal from one that ends badly without. Before
    /// this counter the two were indistinguishable in a report, which is
    /// why gh#857's regression had to be found by instrumenting a build.
    ///
    /// `0` on the SQP and convex paths, which never escalate, and on any
    /// run whose backend declines to (`increase_quality` returning
    /// `false` is not counted — this counts escalations that *happened*,
    /// not escalations that were asked for).
    ///
    /// **On a laddered run this is the promoted solve's count, not the
    /// base solve's** — the same rule `iteration_count` follows, and the
    /// same trap. It is a sharp edge here because `feral_increase_quality_retry`
    /// promotes a re-solve that by construction escalated zero times, so
    /// a run whose base solve escalated twenty-five times reports `0`
    /// once the recovery lands. That is not a lost number: the ladder
    /// block records the base verdict alongside it, and
    /// `feral_increase_quality_retry=no` reproduces the base solve
    /// outright. The rung's own gate reads the base statistics inside the
    /// driver, before any promotion, so the gate is unaffected.
    pub quality_escalations: Index,

    /// gh#884. The solve observed the biactive dual-divergence signature:
    /// at one and the same iterate, a converged primal
    /// (`inf_pr <= dual_divergence_retry_primal_tol`), a scale-relative
    /// step at or below `dual_divergence_retry_step_tol`, and an
    /// *unscaled* dual infeasibility at or above
    /// `dual_divergence_retry_du_floor`.
    ///
    /// Reported whether or not a retry ran or promoted, so a caller can
    /// tell "the multipliers ran away on a settled iterate" from an exit
    /// that merely ran out of iterations.
    ///
    /// Unlike `quality_escalations` and `iteration_count`, which on a
    /// promoted run describe the promoted attempt alone, this accumulates
    /// across every attempt of one solve. That is deliberate: the reason
    /// a second solve happened at all is a fact about the *solve*, and a
    /// promoted run that reported `false` here would say the retry's
    /// answer came from nowhere.
    pub dual_divergence_signature: bool,
    /// gh#884. A dual-divergence retry ran *and* replaced the base
    /// attempt's answer. `false` both when no retry ran and when one ran
    /// and lost — in the latter case the returned point, status and
    /// residuals are the base attempt's.
    pub dual_divergence_retry_promoted: bool,

    // ---- Active-set SQP subproblem counters. ----
    //
    // Populated by `IpoptApplication::optimize_sqp_tnlp`; both stay 0
    // on the interior-point path, which has no QP subproblems.
    //
    /// Number of QP subproblems solved during this solve.
    pub sqp_qp_solves: Index,
    /// Active-set changes (adds + drops) summed over those QP
    /// subproblems. This is the measurement a working-set warm start
    /// is judged on: the outer iteration count can be identical
    /// between a cold and a warm solve while this differs by an order
    /// of magnitude, and on a QP-shaped NLP (one outer iteration by
    /// construction) it is the only thing that moves at all.
    pub sqp_qp_working_set_changes: Index,

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
            final_declared_constr_viol: Number::NAN,
            final_declared_box_viol: Number::NAN,
            final_unscaled_constr_viol: Number::NAN,
            final_unscaled_compl: Number::NAN,
            final_unscaled_kkt_error: Number::NAN,
            final_kkt_error_above_noise: Number::NAN,
            final_mu: 0.0,
            // -1 = the finite-difference updater never built a pattern,
            // which is every run that is not `hessian_approximation=
            // finite-difference`. Distinct from 0, a real pattern source.
            fd_hessian_pattern_used: -1,
            fd_hessian_nnz: 0,
            fd_hessian_n: 0,
            fd_hessian_groups: 0,
            fd_hessian_rho_max: 0,
            fd_hessian_coloring_fell_back: false,
            fd_hessian_objective_clique_widened: false,
            restoration_calls: 0,
            restoration_inner_iters: 0,
            restoration_outer_iters: 0,
            restoration_wall_secs: 0.0,
            quality_escalations: 0,
            dual_divergence_signature: false,
            dual_divergence_retry_promoted: false,
            sqp_qp_solves: 0,
            sqp_qp_working_set_changes: 0,
            iterations: Vec::new(),
        }
    }
}

impl SolveStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}
