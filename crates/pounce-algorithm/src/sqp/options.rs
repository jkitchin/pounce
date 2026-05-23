//! SQP outer-loop options. Mirrors the spirit of
//! `crate::alg_builder::ConvCheckOptions` but only what the SQP
//! driver itself needs; QP-subproblem-specific options pass
//! through `pounce_qp::QpOptions` (which `SqpAlgorithm` owns).

use pounce_common::Number;

/// Choice of SQP globalization strategy. Defaults to filter
/// (Fletcher-Leyffer 2002) per the design note §4.1; l1-elastic
/// merit (Han-Powell with adaptive penalty) is the alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqpGlobalization {
    #[default]
    Filter,
    L1Elastic,
}

/// Hessian source for the SQP subproblem.
///
/// - `Exact`: use `nlp.eval_h(x, 1.0, λ_g, 0)` directly. Indefinite
///   on nonconvex problems; the QP subproblem solver handles
///   indefinite reduced Hessian via inertia control (§4.5).
/// - `DampedBfgs`: dense `n×n` Powell-damped BFGS, guaranteed PSD
///   (Powell 1978). See [`crate::sqp::bfgs::DampedBfgs`].
/// - `Lbfgs`: limited-memory BFGS with a circular `(s, y)` buffer
///   of `lbfgs_max_history` pairs (Nocedal-Wright §7.2; Byrd-
///   Nocedal-Schnabel 1994). See [`crate::sqp::lbfgs::LBfgs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqpHessianSource {
    #[default]
    Exact,
    DampedBfgs,
    Lbfgs,
}

#[derive(Debug, Clone)]
pub struct SqpOptions {
    pub globalization: SqpGlobalization,
    pub hessian: SqpHessianSource,

    /// KKT tolerance (max-norm) on the stationarity residual.
    pub tol: Number,
    /// Tolerance on constraint violation (max-norm).
    pub constr_viol_tol: Number,
    /// Tolerance on stationarity residual (max-norm).
    pub dual_inf_tol: Number,
    /// Outer-iteration cap.
    pub max_iter: u32,

    /// l1-merit penalty parameter ν. Used only when
    /// `globalization = L1Elastic`. Filter globalization ignores
    /// this. Default is a moderate starting value; the Han-Powell
    /// ν-adaptation in `l1_merit_line_search` grows ν to dominate
    /// `‖λ_qp‖_∞ + l1_penalty_safety` on every iteration.
    pub l1_penalty: Number,

    /// Additive safety margin in the Han-Powell ν update:
    /// `ν_new = max(ν, ‖λ_qp‖_∞ + l1_penalty_safety)`. Default 0.1
    /// per Nocedal-Wright §18.4. Only consulted when
    /// `globalization = L1Elastic`.
    pub l1_penalty_safety: Number,

    /// Upper clamp on ν. Prevents catastrophic Armijo failure on
    /// pathological problems where `‖λ_qp‖` momentarily spikes.
    /// Default 1e10 — large enough not to interfere with normal
    /// runs, small enough to keep the merit numerically stable.
    /// Only consulted when `globalization = L1Elastic`.
    pub l1_penalty_max: Number,

    /// Backtracking line-search reduction factor.
    pub bt_reduction: Number,
    /// Minimum step before declaring line-search failure.
    pub bt_min_alpha: Number,

    /// `print_level`: 0 = silent, 1 = per-iteration summary,
    /// 2+ = trace (planned).
    pub print_level: u8,

    /// Number of `(s, y)` history pairs retained when
    /// `hessian = Lbfgs`. Mirrors the upstream
    /// `limited_memory_max_history` default of 6 (Nocedal-Wright
    /// recommends 3–20). Ignored for `Exact` and `DampedBfgs`.
    pub lbfgs_max_history: u32,
}

impl Default for SqpOptions {
    fn default() -> Self {
        Self {
            globalization: SqpGlobalization::default(),
            hessian: SqpHessianSource::default(),
            tol: 1e-8,
            constr_viol_tol: 1e-6,
            dual_inf_tol: 1e-4,
            max_iter: 200,
            l1_penalty: 1.0,
            l1_penalty_safety: 0.1,
            l1_penalty_max: 1e10,
            bt_reduction: 0.5,
            bt_min_alpha: 1e-12,
            print_level: 0,
            lbfgs_max_history: 6,
        }
    }
}
