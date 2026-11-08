//! Matrix-form convex QP / conic entry (gh#1147 Stage 3): the `PounceQp*`
//! handle, its builders, [`PounceQpSolve`], the getters and the report writer.
//!
//! Not an Ipopt API, so every name carries the `Pounce` prefix. The standard
//! form is [`pounce_convex::QpProblem`]'s:
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b,  G x ≤ h  (or G x ⪯_K h with cones),  lb ≤ x ≤ ub
//! ```
//!
//! `P` is given as its **lower triangle** (`row ≥ col`) in triplets; `A` and
//! `G` as full triplets. Triplet indices are 0-based when the handle was
//! created with `index_style = 0` and 1-based with `index_style = 1` (the
//! `IpoptProblem` convention). Duplicate triplets are summed.
//!
//! The solve runs the pipeline Python's `solve_qp` / `solve_socp` run,
//! through the functions S3-I1 lifted into [`pounce_convex::pipeline`]
//! ([`EngineSwitches::check`] / [`EngineSwitches::apply`],
//! [`check_ipm_inertia`], [`solve_presolved`], [`demote_unverified_optimal`]),
//! so there is one implementation of it. What is *not* Python's is the PSD
//! verdict feeding [`check_ipm_inertia`]: Python decides it in numpy
//! (`python/pounce/qp.py::_psd_verdict`); this entry asks
//! [`pounce_convex::certify_psd_lower_triangle`], as the CLI's dispatch does,
//! at the CLI's tolerance band. See [`psd_inertia`].

use crate::{
    Bool, ExtensibleOut, FALSE, Index, Number, TRUE, ffi_guard, struct_size_of, write_extensible,
};
use pounce_convex::hsde_scalars::{self, HsdeScalars};
use pounce_convex::pipeline::{
    EngineSwitches, check_ipm_inertia, demote_unverified_optimal, solve_presolved,
};
use pounce_convex::{
    ActiveSetOverrides, ConeSpec, HessianInertia, NEG_INF, POS_INF, QpOptions, QpProblem,
    QpSolution, QpStatus, Triplet, certify_psd_lower_triangle, solve_qp_active_set_inertia,
    solve_qp_ipm, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use std::ffi::{CStr, c_char, c_int};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Codes (each mirrored by a `#define` in `pounce.h`, held both ways by
// `tests/header_abi.rs`)
// ---------------------------------------------------------------------------

/// [`PounceQpSolve`] statuses, one per [`QpStatus`] variant.
pub const POUNCE_QP_STATUS_OPTIMAL: c_int = 0;
pub const POUNCE_QP_STATUS_OPTIMAL_INACCURATE: c_int = 1;
pub const POUNCE_QP_STATUS_PRIMAL_INFEASIBLE: c_int = 2;
pub const POUNCE_QP_STATUS_DUAL_INFEASIBLE: c_int = 3;
pub const POUNCE_QP_STATUS_ITERATION_LIMIT: c_int = 4;
pub const POUNCE_QP_STATUS_TIME_LIMIT: c_int = 5;
pub const POUNCE_QP_STATUS_NUMERICAL_FAILURE: c_int = 6;

/// Negative returns of [`PounceQpSolve`] and [`PounceQpAddCone`].
pub const POUNCE_QP_ERR_NULL: c_int = -1;
pub const POUNCE_QP_ERR_ARGUMENT: c_int = -2;
pub const POUNCE_QP_ERR_INCOMPLETE: c_int = -3;
pub const POUNCE_QP_ERR_CONES: c_int = -4;
pub const POUNCE_QP_ERR_OPTIONS: c_int = -5;
pub const POUNCE_QP_ERR_INDEFINITE: c_int = -6;
pub const POUNCE_QP_ERR_INTERNAL: c_int = -7;

/// [`PounceQpAddCone`] kinds.
pub const POUNCE_CONE_NONNEG: c_int = 0;
pub const POUNCE_CONE_SOC: c_int = 1;
pub const POUNCE_CONE_EXP: c_int = 2;
pub const POUNCE_CONE_POW: c_int = 3;
pub const POUNCE_CONE_PSD: c_int = 4;

/// [`PounceQpCertificate::kind`].
pub const POUNCE_QP_CERT_NONE: c_int = 0;
pub const POUNCE_QP_CERT_PRIMAL_INFEASIBLE: c_int = 1;
pub const POUNCE_QP_CERT_DUAL_INFEASIBLE: c_int = 2;

/// The CLI's PSD band (`pounce-cli/src/dispatch.rs::PSD_TOL`): an eigenvalue
/// above `-PSD_TOL · min(‖P‖∞, 1)` counts as zero.
const PSD_TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// C structs
// ---------------------------------------------------------------------------

/// The final KKT residuals of the most recent solve — Python's
/// `QpResult.residuals`, field for field.
///
/// On the plain QP/LP path (no cones) the first four are read above their
/// finite-precision floor, in the objective's units (gh#984): the measurement
/// the `Optimal` verdict is judged on. The `_raw` four are the plain absolute
/// values. With cones both sets are the plain conic residuals.
///
/// Extensible: `struct_size` first, set by the caller to
/// `sizeof(PounceQpResiduals)` (`Default` leaves it 0, as C's `= {0}` does,
/// which the getter refuses); fields are only ever appended.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PounceQpResiduals {
    pub struct_size: Index,
    pub primal_infeasibility: Number,
    pub dual_infeasibility: Number,
    pub complementarity: Number,
    pub kkt_error: Number,
    pub primal_infeasibility_raw: Number,
    pub dual_infeasibility_raw: Number,
    pub complementarity_raw: Number,
    pub kkt_error_raw: Number,
}

/// Infeasibility certificate and HSDE scalars of the most recent solve —
/// Python's `QpResult.certificate_scale`, `tau` and `kappa`.
///
/// `kind` is `POUNCE_QP_CERT_PRIMAL_INFEASIBLE` when the status is primal
/// infeasible (the Farkas ray is in the `y`, `z`, `z_lb`, `z_ub` that
/// [`PounceQpGetSolution`] returns), `POUNCE_QP_CERT_DUAL_INFEASIBLE` when it
/// is dual infeasible (the recession ray is in `x`), and `POUNCE_QP_CERT_NONE`
/// otherwise. `scale` is the ray's inf-norm (divide by it for a unit ray), NaN
/// when `kind` is NONE. `tau` and `kappa` are the homogeneous embedding's at
/// termination, NaN when the answer came from a driver that records none (the
/// direct IPM, the active-set engine, the exponential/power-cone driver).
///
/// Extensible, like [`PounceQpResiduals`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PounceQpCertificate {
    pub struct_size: Index,
    pub kind: Index,
    pub scale: Number,
    pub tau: Number,
    pub kappa: Number,
}

// SAFETY: both are `#[repr(C)]` with `struct_size: Index` first.
unsafe impl ExtensibleOut for PounceQpResiduals {}
unsafe impl ExtensibleOut for PounceQpCertificate {}

// ---------------------------------------------------------------------------
// The handle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Ipm,
    ActiveSet,
}

/// Where the inertia of `P` handed to the pipeline comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InertiaChoice {
    /// Decide it with [`certify_psd_lower_triangle`] (the default).
    Auto,
    /// The caller's claim, used as given (Python's `hessian_inertia=`).
    Claim(HessianInertia),
}

#[derive(Debug, Clone, Copy)]
struct Settings {
    tol: Option<f64>,
    max_iter: Option<usize>,
    time_limit: Option<Duration>,
    tau: Option<f64>,
    tau_max: Option<f64>,
    method: Method,
    inertia: InertiaChoice,
    switches: EngineSwitches,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tol: None,
            max_iter: None,
            time_limit: None,
            tau: None,
            tau_max: None,
            method: Method::Ipm,
            inertia: InertiaChoice::Auto,
            switches: EngineSwitches::default(),
        }
    }
}

impl Settings {
    /// The [`QpOptions`] Python's `opts_tau` builds from the same inputs
    /// (`pounce-py/src/qp.rs`): a bare `tau` raises the ceiling with it, an
    /// explicit `tau_max` wins; then the engine switches are written on.
    fn qp_options(&self) -> QpOptions {
        let mut o = QpOptions {
            time_limit: self.time_limit,
            ..QpOptions::default()
        };
        if let Some(t) = self.tol {
            o.tol = t;
        }
        if let Some(m) = self.max_iter {
            o.max_iter = m;
        }
        if let Some(t) = self.tau {
            o.tau = t;
            o.tau_max = o.tau_max.max(t);
        }
        if let Some(t) = self.tau_max {
            o.tau_max = t;
        }
        self.switches.apply(&mut o);
        o
    }
}

/// What the most recent successful [`PounceQpSolve`] produced.
struct QpLast {
    sol: QpSolution,
    engine: &'static str,
    residuals: PounceQpResiduals,
    hsde: Option<HsdeScalars>,
    seconds: f64,
}

/// Owned state behind the opaque `PounceQp` handle. C sees only the pointer.
pub struct PounceQpInfo {
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    index_style: Index,
    p: Vec<Triplet>,
    c: Option<Vec<f64>>,
    a: Vec<Triplet>,
    b: Option<Vec<f64>>,
    g: Vec<Triplet>,
    h: Option<Vec<f64>>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    cones: Vec<ConeSpec>,
    settings: Settings,
    last: Option<QpLast>,
}

pub type PounceQp = *mut PounceQpInfo;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `ptr[0..len]`, or `None` when `ptr` is NULL with `len > 0`. A zero length
/// never dereferences.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        Some(&[])
    } else if ptr.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(ptr, len))
    }
}

impl PounceQpInfo {
    /// Read and validate `nnz` triplets of a `rows × n` matrix, shifting by the
    /// index style. `lower` additionally requires `row ≥ col`. `None` on any
    /// NULL array, out-of-range index or non-finite value.
    unsafe fn triplets(
        &self,
        nnz: Index,
        rows: usize,
        r: *const Index,
        c: *const Index,
        v: *const Number,
        lower: bool,
    ) -> Option<Vec<Triplet>> {
        let nnz = usize::try_from(nnz).ok()?;
        let (r, c, v) = (slice(r, nnz)?, slice(c, nnz)?, slice(v, nnz)?);
        let shift = self.index_style;
        let mut out = Vec::with_capacity(nnz);
        for k in 0..nnz {
            let row = usize::try_from(r[k].checked_sub(shift)?).ok()?;
            let col = usize::try_from(c[k].checked_sub(shift)?).ok()?;
            if row >= rows || col >= self.n || !v[k].is_finite() || (lower && row < col) {
                return None;
            }
            out.push(Triplet::new(row, col, v[k]));
        }
        Some(out)
    }

    /// `len` finite values, or `None` (NULL, or a NaN/infinite entry).
    unsafe fn finite_vec(ptr: *const Number, len: usize) -> Option<Vec<f64>> {
        let s = slice(ptr, len)?;
        s.iter().all(|x| x.is_finite()).then(|| s.to_vec())
    }

    fn problem(&self) -> QpProblem {
        QpProblem {
            n: self.n,
            p_lower: self.p.clone(),
            c: self.c.clone().unwrap_or_default(),
            a: self.a.clone(),
            b: self.b.clone().unwrap_or_default(),
            g: self.g.clone(),
            h: self.h.clone().unwrap_or_default(),
            lb: self.lb.clone(),
            ub: self.ub.clone(),
        }
    }
}

/// The inertia of `P` the pipeline is handed.
///
/// With the default `hessian_inertia = "auto"`, an empty `P` is `Psd`; any
/// other `P` is asked of [`certify_psd_lower_triangle`] at the CLI's band
/// (`1e-9 · min(‖P‖∞, 1)` with `‖P‖∞` the largest summed entry): certified is
/// `Psd`, not certified is `Indefinite`. "Not certified" includes a
/// factorization that could not decide — the CLI routes such a model to the
/// NLP arm, and this entry, which has no NLP arm, treats it the same
/// conservative way: refused under the IPM, driven as indefinite by the
/// active-set engine. An explicit claim is used as given.
///
/// `Err` only for an input the certificate refuses (duplicate entries whose
/// sum overflows); the builders have already refused everything else.
fn psd_inertia(qp: &PounceQpInfo) -> Result<HessianInertia, ()> {
    if let InertiaChoice::Claim(h) = qp.settings.inertia {
        return Ok(h);
    }
    if qp.p.is_empty() {
        return Ok(HessianInertia::Psd);
    }
    let mut summed = std::collections::HashMap::<(usize, usize), f64>::new();
    for t in &qp.p {
        *summed.entry((t.row, t.col)).or_default() += t.val;
    }
    let scale = summed.values().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = PSD_TOL * scale.min(1.0);
    match certify_psd_lower_triangle(qp.n, &qp.p, tol, backend) {
        Ok(true) => Ok(HessianInertia::Psd),
        Ok(false) => Ok(HessianInertia::Indefinite),
        Err(_) => Err(()),
    }
}

fn status_code(s: QpStatus) -> c_int {
    match s {
        QpStatus::Optimal => POUNCE_QP_STATUS_OPTIMAL,
        QpStatus::OptimalInaccurate => POUNCE_QP_STATUS_OPTIMAL_INACCURATE,
        QpStatus::PrimalInfeasible => POUNCE_QP_STATUS_PRIMAL_INFEASIBLE,
        QpStatus::DualInfeasible => POUNCE_QP_STATUS_DUAL_INFEASIBLE,
        QpStatus::IterationLimit => POUNCE_QP_STATUS_ITERATION_LIMIT,
        QpStatus::TimeLimit => POUNCE_QP_STATUS_TIME_LIMIT,
        QpStatus::NumericalFailure => POUNCE_QP_STATUS_NUMERICAL_FAILURE,
    }
}

/// The CLI's `qp_status_to_ars` (`pounce-cli/src/main.rs`), so a report from
/// this entry and the CLI's convex report share one status vocabulary. The
/// AMPL codes `status_to_solve_result_num` derives from these equal the
/// CLI's `convex_status_report` codes on every variant.
fn report_status(s: QpStatus) -> ApplicationReturnStatus {
    match s {
        QpStatus::Optimal => ApplicationReturnStatus::SolveSucceeded,
        QpStatus::OptimalInaccurate => ApplicationReturnStatus::SolvedToAcceptableLevel,
        QpStatus::PrimalInfeasible => ApplicationReturnStatus::InfeasibleProblemDetected,
        QpStatus::DualInfeasible => ApplicationReturnStatus::DivergingIterates,
        QpStatus::IterationLimit => ApplicationReturnStatus::MaximumIterationsExceeded,
        QpStatus::TimeLimit => ApplicationReturnStatus::MaximumWallTimeExceeded,
        QpStatus::NumericalFailure => ApplicationReturnStatus::ErrorInStepComputation,
    }
}

/// Python's residual block (`solution_dict_hsde` in `pounce-py/src/qp.rs`).
fn residuals(sol: &QpSolution, prob: &QpProblem, cones: &[ConeSpec]) -> PounceQpResiduals {
    let raw = sol.kkt_residuals_conic(prob, cones);
    let r = if cones.is_empty() {
        sol.kkt_residuals_above_floor(prob)
    } else {
        raw
    };
    PounceQpResiduals {
        struct_size: struct_size_of::<PounceQpResiduals>(),
        primal_infeasibility: r.primal_infeasibility,
        dual_infeasibility: r.dual_infeasibility,
        complementarity: r.complementarity,
        kkt_error: r.kkt_error(),
        primal_infeasibility_raw: raw.primal_infeasibility,
        dual_infeasibility_raw: raw.dual_infeasibility,
        complementarity_raw: raw.complementarity,
        kkt_error_raw: raw.kkt_error(),
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a QP handle with `n ≥ 1` variables, `m_eq ≥ 0` equality rows and
/// `m_ineq ≥ 0` inequality rows; `index_style` 0 (C) or 1 (Fortran) for every
/// triplet passed to it. NULL on any other argument. Release with
/// [`PounceFreeQp`].
///
/// # Safety
/// None beyond the C calling convention.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceCreateQp(
    n: Index,
    m_eq: Index,
    m_ineq: Index,
    index_style: Index,
) -> PounceQp {
    ffi_guard(std::ptr::null_mut(), || {
        let (Ok(n), Ok(m_eq), Ok(m_ineq)) = (
            usize::try_from(n),
            usize::try_from(m_eq),
            usize::try_from(m_ineq),
        ) else {
            return std::ptr::null_mut();
        };
        if n == 0 || !(index_style == 0 || index_style == 1) {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(PounceQpInfo {
            n,
            m_eq,
            m_ineq,
            index_style,
            p: Vec::new(),
            c: None,
            a: Vec::new(),
            b: (m_eq == 0).then(Vec::new),
            g: Vec::new(),
            h: (m_ineq == 0).then(Vec::new),
            lb: vec![NEG_INF; n],
            ub: vec![POS_INF; n],
            cones: Vec::new(),
            settings: Settings::default(),
            last: None,
        }))
    })
}

/// Free a handle from [`PounceCreateQp`]. NULL is a no-op.
///
/// # Safety
/// `qp` is NULL or a live handle from [`PounceCreateQp`], not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceFreeQp(qp: PounceQp) {
    if !qp.is_null() {
        drop(Box::from_raw(qp));
    }
}

// ---------------------------------------------------------------------------
// Builders: each validates everything first and changes nothing on FALSE.
// ---------------------------------------------------------------------------

/// Set `P` (lower triangle, `nnz_p` triplets) and `c` (length `n`, required).
/// Replaces any earlier objective. FALSE for a NULL handle, a NULL array, an
/// index out of range, an upper-triangle entry, or a non-finite value.
///
/// # Safety
/// `qp` is NULL or live; each array is NULL or holds the stated length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetObjective(
    qp: PounceQp,
    nnz_p: Index,
    p_row: *const Index,
    p_col: *const Index,
    p_val: *const Number,
    c: *const Number,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        let Some(p) = info.triplets(nnz_p, info.n, p_row, p_col, p_val, true) else {
            return FALSE;
        };
        let Some(c) = PounceQpInfo::finite_vec(c, info.n) else {
            return FALSE;
        };
        info.p = p;
        info.c = Some(c);
        TRUE
    })
}

/// Set `A` (`m_eq × n`, `nnz_a` triplets) and `b` (length `m_eq`). Required
/// before a solve when `m_eq > 0`. Same refusals as [`PounceQpSetObjective`]
/// (without the triangle rule).
///
/// # Safety
/// As [`PounceQpSetObjective`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetEqualities(
    qp: PounceQp,
    nnz_a: Index,
    a_row: *const Index,
    a_col: *const Index,
    a_val: *const Number,
    b: *const Number,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        let Some(a) = info.triplets(nnz_a, info.m_eq, a_row, a_col, a_val, false) else {
            return FALSE;
        };
        let Some(b) = PounceQpInfo::finite_vec(b, info.m_eq) else {
            return FALSE;
        };
        info.a = a;
        info.b = Some(b);
        TRUE
    })
}

/// Set `G` (`m_ineq × n`, `nnz_g` triplets) and `h` (length `m_ineq`).
/// Required before a solve when `m_ineq > 0`. Same refusals as
/// [`PounceQpSetEqualities`].
///
/// # Safety
/// As [`PounceQpSetObjective`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetInequalities(
    qp: PounceQp,
    nnz_g: Index,
    g_row: *const Index,
    g_col: *const Index,
    g_val: *const Number,
    h: *const Number,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        let Some(g) = info.triplets(nnz_g, info.m_ineq, g_row, g_col, g_val, false) else {
            return FALSE;
        };
        let Some(h) = PounceQpInfo::finite_vec(h, info.m_ineq) else {
            return FALSE;
        };
        info.g = g;
        info.h = Some(h);
        TRUE
    })
}

/// Set the variable box. `lb` / `ub` are length `n`, or NULL for "no bound on
/// that side". An entry `≤ -1e19` in `lb` or `≥ 1e19` in `ub` (the Ipopt
/// convention, infinities included) is no bound. FALSE for a NULL handle or a
/// NaN entry. Replaces any earlier box.
///
/// # Safety
/// `qp` is NULL or live; `lb` / `ub` are NULL or hold `n` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetBounds(
    qp: PounceQp,
    lb: *const Number,
    ub: *const Number,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        let n = info.n;
        let read = |p: *const Number, absent: f64, is_absent: fn(f64) -> bool| {
            if p.is_null() {
                return Some(vec![absent; n]);
            }
            let s = std::slice::from_raw_parts(p, n);
            s.iter()
                .map(|&v| {
                    if v.is_nan() {
                        None
                    } else if is_absent(v) {
                        Some(absent)
                    } else {
                        Some(v)
                    }
                })
                .collect()
        };
        let Some(l) = read(lb, NEG_INF, |v| v <= -1e19) else {
            return FALSE;
        };
        let Some(u) = read(ub, POS_INF, |v| v >= 1e19) else {
            return FALSE;
        };
        info.lb = l;
        info.ub = u;
        TRUE
    })
}

/// Append one cone to the partition of the `G` rows (in order). `dim` is the
/// row count for `POUNCE_CONE_NONNEG` (`≥ 0`) and `POUNCE_CONE_SOC` (`≥ 1`),
/// must be 3 for `POUNCE_CONE_EXP` and `POUNCE_CONE_POW`, and is the matrix
/// order `k ≥ 1` for `POUNCE_CONE_PSD` (spanning `k(k+1)/2` svec rows).
/// `alpha` is the power cone's exponent, in `(0, 1)`, and ignored otherwise.
/// Returns the cone's 0-based position, or `POUNCE_QP_ERR_NULL` /
/// `POUNCE_QP_ERR_ARGUMENT` (nothing added).
///
/// # Safety
/// `qp` is NULL or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpAddCone(
    qp: PounceQp,
    kind: c_int,
    dim: Index,
    alpha: Number,
) -> c_int {
    ffi_guard(POUNCE_QP_ERR_INTERNAL, || {
        let Some(info) = qp.as_mut() else {
            return POUNCE_QP_ERR_NULL;
        };
        let Ok(d) = usize::try_from(dim) else {
            return POUNCE_QP_ERR_ARGUMENT;
        };
        let spec = match kind {
            POUNCE_CONE_NONNEG => ConeSpec::Nonneg(d),
            POUNCE_CONE_SOC if d >= 1 => ConeSpec::SecondOrder(d),
            POUNCE_CONE_EXP if d == 3 => ConeSpec::Exponential,
            POUNCE_CONE_POW if d == 3 && alpha > 0.0 && alpha < 1.0 => ConeSpec::Power(alpha),
            POUNCE_CONE_PSD if d >= 1 && d.checked_mul(d + 1).is_some() => ConeSpec::Psd(d),
            _ => return POUNCE_QP_ERR_ARGUMENT,
        };
        let Ok(pos) = c_int::try_from(info.cones.len()) else {
            return POUNCE_QP_ERR_ARGUMENT;
        };
        info.cones.push(spec);
        pos
    })
}

/// Set a numeric option: `tol` (in `(0, 1)`), `max_iter` (a whole number
/// `≥ 1`), `time_limit` (seconds, `≥ 0`), `tau` / `tau_max` (in `(0, 1)`),
/// `qp_reg` (finite, `≥ 0`). FALSE for a NULL handle or key, an unknown key or
/// a value outside its range; nothing changes then.
///
/// # Safety
/// `qp` is NULL or live; `key` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetNumOption(
    qp: PounceQp,
    key: *const c_char,
    value: Number,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        if key.is_null() {
            return FALSE;
        }
        let Ok(key) = CStr::from_ptr(key).to_str() else {
            return FALSE;
        };
        let unit = value.is_finite() && value > 0.0 && value < 1.0;
        let s = &mut info.settings;
        match key {
            "tol" if unit => s.tol = Some(value),
            "max_iter" if value >= 1.0 && value.fract() == 0.0 && value <= 1e15 => {
                s.max_iter = Some(value as usize)
            }
            "time_limit" => match Duration::try_from_secs_f64(value) {
                Ok(d) => s.time_limit = Some(d),
                Err(_) => return FALSE,
            },
            "tau" if unit => s.tau = Some(value),
            "tau_max" if unit => s.tau_max = Some(value),
            "qp_reg" if value.is_finite() && value >= 0.0 => s.switches.reg = Some(value),
            _ => return FALSE,
        }
        TRUE
    })
}

/// Set a string option: `method` (`ipm` | `active-set`), `hessian_inertia`
/// (`auto` | `psd` | `indefinite` | `unknown`), and the switches `qp_presolve`,
/// `qp_hsde`, `qp_equilibrate`, `qp_crossover` (`yes` | `no`). FALSE for a
/// NULL handle, key or value, an unknown key or value; nothing changes then.
///
/// # Safety
/// `qp` is NULL or live; `key` and `value` are NULL or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSetStrOption(
    qp: PounceQp,
    key: *const c_char,
    value: *const c_char,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_mut() else {
            return FALSE;
        };
        if key.is_null() || value.is_null() {
            return FALSE;
        }
        let (Ok(key), Ok(value)) = (CStr::from_ptr(key).to_str(), CStr::from_ptr(value).to_str())
        else {
            return FALSE;
        };
        let yes_no = match value {
            "yes" => Some(true),
            "no" => Some(false),
            _ => None,
        };
        let s = &mut info.settings;
        match (key, value, yes_no) {
            ("method", "ipm", _) => s.method = Method::Ipm,
            ("method", "active-set", _) => s.method = Method::ActiveSet,
            ("hessian_inertia", "auto", _) => s.inertia = InertiaChoice::Auto,
            ("hessian_inertia", "psd", _) => s.inertia = InertiaChoice::Claim(HessianInertia::Psd),
            ("hessian_inertia", "indefinite", _) => {
                s.inertia = InertiaChoice::Claim(HessianInertia::Indefinite)
            }
            ("hessian_inertia", "unknown", _) => {
                s.inertia = InertiaChoice::Claim(HessianInertia::Unknown)
            }
            ("qp_presolve", _, Some(v)) => s.switches.presolve = Some(v),
            ("qp_hsde", _, Some(v)) => s.switches.hsde = Some(v),
            ("qp_equilibrate", _, Some(v)) => s.switches.equilibrate = Some(v),
            ("qp_crossover", _, Some(v)) => s.switches.crossover = Some(v),
            _ => return FALSE,
        }
        TRUE
    })
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

/// Solve the problem the handle holds. Returns a `POUNCE_QP_STATUS_*` (`≥ 0`)
/// and keeps the result for the getters, or a negative `POUNCE_QP_ERR_*`
/// before any engine runs (the previous result is discarded either way):
///
/// * `INCOMPLETE` — no objective, or `m_eq > 0` / `m_ineq > 0` rows never set.
/// * `CONES` — cones do not partition the `m_ineq` rows exactly, or a PSD
///   cone is mixed with an exponential/power one.
/// * `OPTIONS` — a switch the selected engine would drop
///   ([`EngineSwitches::check`]), or, with cones, `method=active-set` or any
///   option Python's `solve_socp` does not take (`tau`, `tau_max`, `qp_*`).
/// * `INDEFINITE` — `method=ipm` and `P` is not certified PSD (gh#112).
/// * `INTERNAL` — a panic caught at the boundary.
///
/// No cones: the `solve_qp` pipeline (IPM, or with `qp_presolve=yes` presolve
/// → IPM → postsolve → [`demote_unverified_optimal`]; or the active-set
/// engine, optionally presolved). Any cone: `solve_socp_ipm`, as `solve_socp`.
///
/// # Safety
/// `qp` is NULL or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpSolve(qp: PounceQp) -> c_int {
    let Some(info) = qp.as_mut() else {
        return POUNCE_QP_ERR_NULL;
    };
    info.last = None;
    ffi_guard(POUNCE_QP_ERR_INTERNAL, || solve(info))
}

fn solve(info: &mut PounceQpInfo) -> c_int {
    if info.c.is_none() || info.b.is_none() || info.h.is_none() {
        return POUNCE_QP_ERR_INCOMPLETE;
    }
    let s = info.settings;
    let conic = !info.cones.is_empty();
    if conic {
        let nonsym = info
            .cones
            .iter()
            .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)));
        let psd = info.cones.iter().any(|c| matches!(c, ConeSpec::Psd(_)));
        let rows = info
            .cones
            .iter()
            .try_fold(0usize, |acc, c| acc.checked_add(c.dim()));
        if (nonsym && psd) || rows != Some(info.m_ineq) {
            return POUNCE_QP_ERR_CONES;
        }
        if s.method != Method::Ipm
            || s.tau.is_some()
            || s.tau_max.is_some()
            || s.switches != EngineSwitches::default()
        {
            return POUNCE_QP_ERR_OPTIONS;
        }
    } else if s
        .switches
        .check(s.method == Method::ActiveSet, false)
        .is_err()
    {
        return POUNCE_QP_ERR_OPTIONS;
    }
    let Ok(inertia) = psd_inertia(info) else {
        return POUNCE_QP_ERR_ARGUMENT;
    };
    if s.method == Method::Ipm && check_ipm_inertia(inertia).is_err() {
        return POUNCE_QP_ERR_INDEFINITE;
    }

    let prob = info.problem();
    let o = s.qp_options();
    let presolve_on = s.switches.presolve.unwrap_or(false);
    let t0 = std::time::Instant::now();
    hsde_scalars::clear();
    pounce_convex::crossover::clear_report();
    let (sol, engine) = if conic {
        (solve_socp_ipm(&prob, &info.cones, &o, backend), "cvx-qcqp")
    } else if s.method == Method::Ipm {
        let sol = if presolve_on {
            let (sol, _) = solve_presolved(&prob, &o, |p, po| solve_qp_ipm(p, po, backend));
            demote_unverified_optimal(&prob, &o, sol)
        } else {
            solve_qp_ipm(&prob, &o, backend)
        };
        (sol, "cvx-qp")
    } else {
        let ov = ActiveSetOverrides::default();
        let sol = if presolve_on {
            solve_presolved(&prob, &o, |p, po| {
                solve_qp_active_set_inertia(p, po, &ov, inertia, &mut backend)
            })
            .0
        } else {
            solve_qp_active_set_inertia(&prob, &o, &ov, inertia, &mut backend)
        };
        (sol, "qp-active-set")
    };
    let seconds = t0.elapsed().as_secs_f64();
    let hsde = hsde_scalars::take();
    pounce_convex::crossover::clear_report();
    let code = status_code(sol.status);
    info.last = Some(QpLast {
        residuals: residuals(&sol, &prob, &info.cones),
        sol,
        engine,
        hsde,
        seconds,
    });
    code
}

// ---------------------------------------------------------------------------
// Getters: FALSE for a NULL handle and before any (successful) solve.
// ---------------------------------------------------------------------------

/// Copy the most recent solution out. Each output may be NULL to skip it;
/// lengths are `x`, `z_lb`, `z_ub`: `n`; `y`: `m_eq`; `z`: `m_ineq`. Signs
/// are `pounce_convex`'s: `Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0` with
/// `z, z_lb, z_ub ≥ 0` (cone-dual for conic `z`). `iters` is the engine's
/// iteration count. FALSE for a NULL handle or no solve result.
///
/// # Safety
/// `qp` is NULL or live; every non-NULL output holds its stated length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpGetSolution(
    qp: PounceQp,
    x: *mut Number,
    y: *mut Number,
    z: *mut Number,
    z_lb: *mut Number,
    z_ub: *mut Number,
    obj: *mut Number,
    iters: *mut Index,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(last) = qp.as_ref().and_then(|i| i.last.as_ref()) else {
            return FALSE;
        };
        let sol = &last.sol;
        for (dst, src) in [
            (x, &sol.x),
            (y, &sol.y),
            (z, &sol.z),
            (z_lb, &sol.z_lb),
            (z_ub, &sol.z_ub),
        ] {
            if !dst.is_null() {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            }
        }
        if !obj.is_null() {
            *obj = sol.obj;
        }
        if !iters.is_null() {
            *iters = Index::try_from(sol.iters).unwrap_or(Index::MAX);
        }
        TRUE
    })
}

/// Copy the most recent solve's KKT residuals into `*out`, whose
/// `struct_size` the caller sets to `sizeof(PounceQpResiduals)`. FALSE
/// (nothing written) for a NULL handle or `out`, a `struct_size` smaller
/// than this release's size, or no solve result.
///
/// # Safety
/// `qp` is NULL or live; `out` is NULL or points at a `PounceQpResiduals`
/// whose `struct_size` is readable, writable for this release's size when
/// `struct_size` is that size or more.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpGetResiduals(qp: PounceQp, out: *mut PounceQpResiduals) -> Bool {
    ffi_guard(FALSE, || {
        let Some(last) = qp.as_ref().and_then(|i| i.last.as_ref()) else {
            return FALSE;
        };
        if out.is_null() {
            return FALSE;
        }
        // Bounded write of this release's size (see `write_extensible`).
        Bool::from(write_extensible(out, &last.residuals))
    })
}

/// Copy the most recent solve's certificate and HSDE scalars into `*out`
/// (see `PounceQpCertificate`; the caller sets `struct_size` as for
/// [`PounceQpGetResiduals`]). TRUE after every solve, whatever its status;
/// FALSE (nothing written) for a NULL handle or `out`, a short
/// `struct_size`, or no solve result.
///
/// # Safety
/// As [`PounceQpGetResiduals`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpGetCertificate(
    qp: PounceQp,
    out: *mut PounceQpCertificate,
) -> Bool {
    ffi_guard(FALSE, || {
        let Some(last) = qp.as_ref().and_then(|i| i.last.as_ref()) else {
            return FALSE;
        };
        if out.is_null() {
            return FALSE;
        }
        let sol = &last.sol;
        let norm = |vs: &[&[f64]]| {
            vs.iter()
                .flat_map(|v| v.iter())
                .fold(0.0_f64, |m, x| m.max(x.abs()))
        };
        let (kind, scale) = match sol.status {
            QpStatus::PrimalInfeasible => (
                POUNCE_QP_CERT_PRIMAL_INFEASIBLE,
                norm(&[&sol.y, &sol.z, &sol.z_lb, &sol.z_ub]),
            ),
            QpStatus::DualInfeasible => (POUNCE_QP_CERT_DUAL_INFEASIBLE, norm(&[&sol.x])),
            _ => (POUNCE_QP_CERT_NONE, f64::NAN),
        };
        let cert = PounceQpCertificate {
            struct_size: struct_size_of::<PounceQpCertificate>(),
            kind,
            scale,
            tau: last.hsde.map_or(f64::NAN, |h| h.tau),
            kappa: last.hsde.map_or(f64::NAN, |h| h.kappa),
        };
        // Bounded write of this release's size (see `write_extensible`).
        Bool::from(write_extensible(out, &cert))
    })
}

/// Write the most recent solve as a `pounce.solve-report/v1` JSON file at
/// `detail` `"summary"` (also for NULL) or `"full"`. `solution.engine` is
/// `cvx-qp`, `qp-active-set` or (any cone) `cvx-qcqp`; the status is the
/// CLI's convex mapping; `solution.lambda` is `y` then `z`. FALSE for a NULL
/// handle or path, no solve result, an unknown detail, or an I/O error.
///
/// # Safety
/// `qp` is NULL or live; `path`, `detail` are NULL or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceQpWriteSolveReport(
    qp: PounceQp,
    path: *const c_char,
    detail: *const c_char,
) -> Bool {
    use pounce_solve_report::{
        InputDescriptor, ReportBuilder, ReportDetail, status_to_solve_result_num, write_report_file,
    };
    ffi_guard(FALSE, || {
        let Some(info) = qp.as_ref() else {
            return FALSE;
        };
        let Some(last) = info.last.as_ref() else {
            return FALSE;
        };
        if path.is_null() {
            return FALSE;
        }
        let Ok(path) = CStr::from_ptr(path).to_str() else {
            return FALSE;
        };
        let detail = if detail.is_null() {
            ReportDetail::Summary
        } else {
            match CStr::from_ptr(detail).to_str().map(ReportDetail::parse) {
                Ok(Ok(d)) => d,
                _ => return FALSE,
            }
        };
        let idx = |v: usize| Index::try_from(v).unwrap_or(Index::MAX);
        let sol = &last.sol;
        let status = report_status(sol.status);
        let mut b = ReportBuilder::new(detail, InputDescriptor::TnlpDirect);
        b.problem.n_variables = idx(info.n);
        b.problem.n_constraints = idx(info.m_eq + info.m_ineq);
        b.problem.n_objectives = 1;
        b.problem.nnz_jac_g = Some(idx(info.a.len() + info.g.len()));
        b.problem.nnz_h_lag = Some(idx(info.p.len()));
        b.solution.engine = last.engine.to_string();
        b.solution.status = status;
        b.solution.solve_result_num = status_to_solve_result_num(status);
        b.solution.objective = sol.obj;
        b.solution.x = sol.x.clone();
        b.solution.lambda = sol.y.iter().chain(&sol.z).copied().collect();
        b.stats.iteration_count = idx(sol.iters);
        b.stats.final_objective = sol.obj;
        b.stats.final_constr_viol = last.residuals.primal_infeasibility;
        b.stats.final_dual_inf = last.residuals.dual_infeasibility;
        b.stats.final_compl = last.residuals.complementarity;
        b.stats.final_kkt_error = last.residuals.kkt_error;
        b.stats.total_wallclock_time_secs = last.seconds;
        match write_report_file(std::path::Path::new(path), &b.finish()) {
            Ok(_) => TRUE,
            Err(_) => FALSE,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // error-path: PounceFreeQp
    #[test]
    fn free_null_is_a_no_op_and_create_refuses_bad_dimensions() {
        unsafe {
            PounceFreeQp(std::ptr::null_mut());
            assert!(PounceCreateQp(0, 0, 0, 0).is_null());
            assert!(PounceCreateQp(-1, 0, 0, 0).is_null());
            assert!(PounceCreateQp(2, -1, 0, 0).is_null());
            assert!(PounceCreateQp(2, 0, -1, 0).is_null());
            assert!(PounceCreateQp(2, 0, 0, 2).is_null());
            let qp = PounceCreateQp(2, 0, 0, 1);
            assert!(!qp.is_null());
            PounceFreeQp(qp);
        }
    }

    /// One fixture per branch of the status and report mappings: every
    /// variant, and the AMPL code equals the CLI's `convex_status_report`
    /// (0, 1, 200, 300, 400, 400, 500).
    #[test]
    fn status_mappings_cover_every_variant() {
        use pounce_solve_report::status_to_solve_result_num;
        let all = [
            (QpStatus::Optimal, 0, 0),
            (QpStatus::OptimalInaccurate, 1, 1),
            (QpStatus::PrimalInfeasible, 2, 200),
            (QpStatus::DualInfeasible, 3, 300),
            (QpStatus::IterationLimit, 4, 400),
            (QpStatus::TimeLimit, 5, 400),
            (QpStatus::NumericalFailure, 6, 500),
        ];
        for (s, code, srn) in all {
            assert_eq!(status_code(s), code, "{s:?}");
            assert_eq!(status_to_solve_result_num(report_status(s)), srn, "{s:?}");
        }
    }
}
