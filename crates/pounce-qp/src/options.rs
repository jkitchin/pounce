//! Per-solve options. The defaults mirror the §7.1 option-registry
//! values from the design note so the SQP-side wiring can forward
//! `OptionsList` entries straight through without translation.

use pounce_common::Number;

/// Active-set QP algorithm variant. Phase 5a ships only the sparse
/// parametric active-set method; other entries are placeholders to
/// keep the option name `sqp_qp_solver` stable as future variants
/// (e.g., a dense Goldfarb-Idnani for tiny dense QPs) appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpAlgorithm {
    /// Sparse Schur-complement parametric active-set (§4.2,
    /// Kirches 2011 / Janka 2017). Default and only option in
    /// Phase 5a.
    #[default]
    ParametricActiveSet,
}

/// Anti-cycling rule. `Expand` is the SOTA default (§4.4,
/// Gill-Murray-Saunders-Wright 1989); `Bland` is a slower
/// guaranteed-finite fallback used in unit tests; `None` disables
/// anti-cycling and is for benchmarking only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AntiCyclingChoice {
    #[default]
    Expand,
    Bland,
    None,
}

#[derive(Debug, Clone)]
pub struct QpOptions {
    pub algorithm: QpAlgorithm,
    pub max_iter: u32,
    pub feas_tol: Number,
    pub opt_tol: Number,
    /// Maximum number of Schur-complement rank-1 updates before a
    /// fresh base-KKT refactorization. Default 50 per the design
    /// note §4.2 / §7.1; bound on the worst-case dense-Schur cost.
    pub max_schur_updates_before_refactor: u32,
    pub anti_cycling: AntiCyclingChoice,
    /// Elastic-mode penalty γ (§4.3). Default 1e6; large enough that
    /// the elastic slacks vanish at the solution of any feasible QP
    /// the SQP outer loop is likely to generate, small enough not to
    /// dominate the Hessian conditioning.
    pub elastic_gamma: Number,
    /// 0 = silent, 1 = per-solve summary, 2 = per-iteration trace,
    /// 3+ = per-pivot detail. Matches pounce's existing
    /// `print_level` convention.
    pub print_level: u8,

    /// §4.5 inertia control: when an LDLᵀ factor of the active-set
    /// KKT reports the wrong inertia or near-singularity, retry
    /// with `H ← H + δ·I` (only on the original-variable block).
    /// `inertia_shift_initial` is the first δ tried; subsequent
    /// retries multiply δ by `inertia_shift_factor`; the loop
    /// gives up after `inertia_max_shifts` attempts.
    ///
    /// Default values match the IPOPT-style perturbation handler
    /// in `pounce-algorithm/src/kkt/perturbation_handler.rs`.
    pub inertia_shift_initial: Number,
    pub inertia_shift_factor: Number,
    pub inertia_max_shifts: u32,

    /// Opt in to the §4.2 sparse Schur-complement update path in
    /// `solve_general`. When `true`, the inner loop uses a cached
    /// factor of the fixed-dim K_max matrix and absorbs working-
    /// set changes as Sherman-Morrison-Woodbury rank-2 updates,
    /// refactoring only when the Schur block reaches
    /// `max_schur_updates_before_refactor`. When `false`
    /// (default), each iteration assembles a fresh active-set
    /// KKT and factors from scratch — algorithmically correct,
    /// noticeably slower on large warm-started workloads.
    pub use_schur_updates: bool,

    /// §4.4 full EXPAND anti-cycling primal perturbation. Active
    /// only when `anti_cycling = Expand`. The Harris two-pass
    /// (c14) prevents cycling at non-degenerate vertices; these
    /// parameters add protection at truly degenerate (α = 0)
    /// vertices via a monotonically growing tolerance:
    ///
    /// - `expand_tol_initial` — starting τ at each reset.
    /// - `expand_tol_growth`  — per-iteration increment of τ.
    /// - `expand_tol_max`     — τ ceiling; on hitting it, snap
    ///   all active-bound primals exactly to their bounds and
    ///   reset τ to `expand_tol_initial`.
    ///
    /// Defaults are conservative — they ensure cycling protection
    /// kicks in only on pathological degeneracy. References:
    /// Gill-Murray-Saunders-Wright 1989 §4 (the EXPAND name and
    /// the τ-growth schedule); SNOPT defaults.
    pub expand_tol_initial: Number,
    pub expand_tol_growth: Number,
    pub expand_tol_max: Number,
}

impl Default for QpOptions {
    fn default() -> Self {
        Self {
            algorithm: QpAlgorithm::default(),
            max_iter: 200,
            feas_tol: 1e-9,
            opt_tol: 1e-9,
            max_schur_updates_before_refactor: 50,
            anti_cycling: AntiCyclingChoice::default(),
            elastic_gamma: 1e6,
            print_level: 0,
            inertia_shift_initial: 1e-8,
            inertia_shift_factor: 100.0,
            inertia_max_shifts: 12,
            use_schur_updates: false,
            expand_tol_initial: 1e-12,
            expand_tol_growth: 1e-11,
            expand_tol_max: 1e-7,
        }
    }
}
