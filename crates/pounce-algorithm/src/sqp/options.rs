//! SQP outer-loop options. Mirrors the spirit of
//! `crate::alg_builder::ConvCheckOptions` but only what the SQP
//! driver itself needs; QP-subproblem-specific options pass
//! through `pounce_qp::QpOptions` (which `SqpAlgorithm` owns).

use pounce_common::Number;
use pounce_common::timing::Deadline;

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
/// - `QuasiNewtonAuto`: what `hessian_approximation=limited-memory`
///   selects — `DampedBfgs` up to [`SqpOptions::dense_hessian_max_n`]
///   variables, `Lbfgs` above it, decided per solve from the problem's
///   `n` (gh#1054). The dense update is the more robust of the two on
///   small models (issue #358) but needs `n²` storage and makes every QP
///   factorization dense, which a 15 000-variable model cannot afford.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqpHessianSource {
    #[default]
    Exact,
    DampedBfgs,
    Lbfgs,
    QuasiNewtonAuto,
}

impl SqpHessianSource {
    /// The concrete source a solve of `n` variables uses: resolves
    /// `QuasiNewtonAuto`, passes every other source through.
    pub fn resolve(self, n: usize, dense_hessian_max_n: usize) -> Self {
        match self {
            Self::QuasiNewtonAuto if n <= dense_hessian_max_n => Self::DampedBfgs,
            Self::QuasiNewtonAuto => Self::Lbfgs,
            other => other,
        }
    }

    /// Whether this source approximates the Hessian (and is therefore
    /// positive definite by construction) rather than evaluating it.
    pub fn is_quasi_newton(self) -> bool {
        !matches!(self, Self::Exact)
    }
}

/// Default for [`SqpOptions::qp_schur_min_dim`].
pub const DEFAULT_QP_SCHUR_MIN_DIM: usize = 500;

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

    /// Largest `n` at which a quasi-Newton Hessian is handed to the QP as a
    /// dense matrix (`sqp_dense_hessian_max_n`). Above it, `Lbfgs` poses the
    /// QP in its lifted low-rank form
    /// ([`crate::sqp::SqpQpData::build_low_rank`]) and `QuasiNewtonAuto`
    /// resolves to `Lbfgs` rather than `DampedBfgs`. At or below it nothing
    /// changes from before gh#1054. Default 1000.
    pub dense_hessian_max_n: usize,

    /// Size from which the step QP uses Schur-complement updates
    /// (gh#1054): when `Some(d)`, each solve sets
    /// `QpOptions::use_schur_updates` to `n + m ≥ d`; `None` leaves the QP
    /// options as given. The driver reads `sqp_qp_use_schur_updates` only
    /// when it is set explicitly, and then passes `None` here.
    ///
    /// Why size: the refactor path pays a full sparse factorization per
    /// working-set change, which is what made the cold first QP of a
    /// 15 000-variable model run for 30 minutes, while on small models the
    /// two paths cost the same and the Schur path is the less robust one
    /// (measured on the CLI fixture corpus under `algorithm=active-set-sqp`).
    /// See `docs/src/active-set-sqp.md` for the Maros-Mészáros comparison
    /// behind the default.
    pub qp_schur_min_dim: Option<usize>,

    /// Solve-wide time budget (`max_wall_time` / `max_cpu_time`), the
    /// same [`Deadline`] the interior-point arm checks. `None` means
    /// unlimited and is the default for a directly constructed driver.
    ///
    /// A shared handle rather than two durations, because the SQP is not
    /// always the first thing a solve runs: crossover hands it the
    /// interior-point phase's own deadline, and a duration measured from
    /// `optimize` would grant it a fresh budget (gh#1054).
    pub deadline: Option<Deadline>,
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
            dense_hessian_max_n: 1000,
            qp_schur_min_dim: Some(DEFAULT_QP_SCHUR_MIN_DIM),
            deadline: None,
        }
    }
}
