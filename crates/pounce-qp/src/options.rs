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
        }
    }
}
