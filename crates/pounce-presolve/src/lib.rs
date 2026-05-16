//! Algorithmic NLP preprocessing exposed as a composable TNLP wrapper.
//!
//! Tracks pounce issue #20.
//!
//! * **Phase 0** — scaffolding, options table, no-op identity path.
//! * **Phase 1** — Andersen-style bound tightening against linear rows.
//! * **Phase 2** — redundant linear-constraint removal: rows whose
//!   activity interval is implied by the (possibly Phase-1-tightened)
//!   variable box are dropped from the problem the solver sees, then
//!   reinstated with `λ=0` when forwarding `eval_h` / `finalize_solution`
//!   to the inner TNLP.
//! * **Phase 3** — structural LICQ check on the surviving equality
//!   rows. Verdict is published via [`PresolveTnlp::licq_verdict`].
//! * **Phase 4** — bound-multiplier warm-start hints for variables
//!   whose bounds were tightened by Phase 1. Hints are emitted on
//!   `init_z` and exposed via [`PresolveTnlp::z_warm_starts`].
//! * **Phase 5** — sensitivity-aware passthrough: projects
//!   user-supplied constraint metadata and scaling through the row
//!   reduction on the way in, and expands outer→inner on the way out
//!   in `finalize_metadata`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::exception::SolverException;
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};

pub mod bound_tighten;
pub mod licq;
pub mod options;
pub mod redundant;

pub use bound_tighten::{tighten_bounds, LinearRow, TightenReport, INF_BOUND};
pub use licq::{licq_check, EqRow, LicqVerdict};
pub use options::{register_options, LicqAction, PresolveOptions};
pub use redundant::find_redundant_rows;

/// Errors that can arise while building a presolved TNLP.
#[derive(Debug)]
pub enum PresolveError {
    OptionsError(SolverException),
}

impl std::fmt::Display for PresolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OptionsError(e) => write!(f, "presolve options error: {e}"),
        }
    }
}

impl std::error::Error for PresolveError {}

impl From<SolverException> for PresolveError {
    fn from(e: SolverException) -> Self {
        Self::OptionsError(e)
    }
}

/// Top-level entry: returns a TNLP wrapping `inner` with whatever
/// presolve passes the option table has enabled. When the master
/// switch is off, returns `inner` unchanged.
pub fn wrap_with_presolve(
    inner: Rc<RefCell<dyn TNLP>>,
    opts: PresolveOptions,
) -> Result<Rc<RefCell<dyn TNLP>>, PresolveError> {
    if !opts.enabled {
        return Ok(inner);
    }
    Ok(Rc::new(RefCell::new(PresolveTnlp::new(inner, opts))))
}

/// Convenience: read the `presolve_*` keys out of an `OptionsList`
/// and call [`wrap_with_presolve`].
pub fn wrap_from_options(
    inner: Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
) -> Result<Rc<RefCell<dyn TNLP>>, PresolveError> {
    let opts = PresolveOptions::from_options_list(options)?;
    wrap_with_presolve(inner, opts)
}

/// Cached, reduced view of the problem after presolve passes have
/// run. Exposed for inspection from integration tests.
pub struct CachedBounds {
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    /// Reduced (post-row-drop) constraint lower bounds.
    pub g_l: Vec<Number>,
    /// Reduced constraint upper bounds.
    pub g_u: Vec<Number>,
}

/// TNLP wrapper that re-presents the inner problem after presolve.
pub struct PresolveTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    opts: PresolveOptions,

    /// `None` until init has run; afterwards `Some(state)`.
    state: Option<PresolveState>,
}

struct PresolveState {
    info_inner: NlpInfo,
    info_outer: NlpInfo,
    bounds: CachedBounds,

    /// Maps outer (reduced) row index → inner row index. Length
    /// equals `info_outer.m`.
    rows_kept: Vec<usize>,

    /// For each outer nnz, the position in the inner nnz array.
    jac_kept_idx: Vec<usize>,
    /// Cached outer (reduced + renumbered) Jacobian sparsity, served
    /// on `eval_jac_g(Structure)`.
    jac_irow_outer: Vec<Index>,
    jac_jcol_outer: Vec<Index>,

    /// Phase 1 report.
    tighten_report: TightenReport,
    /// Number of rows dropped by Phase 2.
    n_dropped_rows: Index,
    /// Phase 3 verdict (`None` if the LICQ check was disabled).
    licq_verdict: Option<LicqVerdict>,
    /// Phase 4: warm-start values for `z_l` per variable. Entry is
    /// 0.0 where presolve did not tighten the lower bound, else
    /// `bound_mult_init_val`. Same length as `bounds.x_l`.
    z_l_warm: Vec<Number>,
    /// Phase 4: warm-start values for `z_u` per variable.
    z_u_warm: Vec<Number>,

    /// Scratch buffers reused across eval_* calls.
    scratch_g: Vec<Number>,
    scratch_jac: Vec<Number>,
    scratch_lambda: Vec<Number>,
}

impl PresolveTnlp {
    /// Build a presolve wrapper directly. Prefer
    /// [`wrap_with_presolve`] in production code; this constructor
    /// is exposed so integration tests can keep a typed handle for
    /// accessors like [`Self::licq_verdict`].
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, opts: PresolveOptions) -> Self {
        Self {
            inner,
            opts,
            state: None,
        }
    }

    /// Phase 1 report (zeroed until init has run).
    pub fn tighten_report(&self) -> TightenReport {
        self.state
            .as_ref()
            .map(|s| s.tighten_report.clone())
            .unwrap_or_default()
    }

    /// Number of constraint rows dropped by Phase 2 (0 if presolve
    /// has not yet run or no rows are redundant).
    pub fn n_dropped_rows(&self) -> Index {
        self.state.as_ref().map(|s| s.n_dropped_rows).unwrap_or(0)
    }

    /// Cached reduced bounds, if presolve has run.
    pub fn cached_bounds(&self) -> Option<&CachedBounds> {
        self.state.as_ref().map(|s| &s.bounds)
    }

    /// Phase 3 verdict — `Some` only if the LICQ check was enabled
    /// and presolve has run.
    pub fn licq_verdict(&self) -> Option<&LicqVerdict> {
        self.state.as_ref().and_then(|s| s.licq_verdict.as_ref())
    }

    /// Phase 4 warm-start hints `(z_l, z_u)`. Each entry is 0.0 if
    /// no hint is set for that variable, else the configured
    /// `bound_mult_init_val`. `None` until init has run.
    pub fn z_warm_starts(&self) -> Option<(&[Number], &[Number])> {
        self.state.as_ref().map(|s| (&s.z_l_warm[..], &s.z_u_warm[..]))
    }

    /// Lazy initialization: pull inner dims, bounds, linearity tags,
    /// Jacobian, and starting point; run Phase 1 + Phase 2 passes;
    /// cache everything needed to translate later eval_* calls.
    fn ensure_init(&mut self) -> Option<&PresolveState> {
        if self.state.is_some() {
            return self.state.as_ref();
        }

        let info_inner = self.inner.borrow_mut().get_nlp_info()?;
        let n = info_inner.n as usize;
        let m_in = info_inner.m as usize;
        let nnz_in = info_inner.nnz_jac_g as usize;

        // Inner bounds.
        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        let mut g_l_inner = vec![0.0; m_in];
        let mut g_u_inner = vec![0.0; m_in];
        {
            let mut inner = self.inner.borrow_mut();
            if !inner.get_bounds_info(BoundsInfo {
                x_l: &mut x_l,
                x_u: &mut x_u,
                g_l: &mut g_l_inner,
                g_u: &mut g_u_inner,
            }) {
                return None;
            }
        }

        // Jacobian sparsity.
        let mut jac_irow_inner = vec![0 as Index; nnz_in];
        let mut jac_jcol_inner = vec![0 as Index; nnz_in];
        if nnz_in > 0 {
            let mut inner = self.inner.borrow_mut();
            if !inner.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut jac_irow_inner,
                    jcol: &mut jac_jcol_inner,
                },
            ) {
                return None;
            }
        }

        // Linearity tags (presolve is dormant without them).
        let mut linearity = vec![Linearity::NonLinear; m_in];
        let have_linearity = if m_in > 0 {
            self.inner
                .borrow_mut()
                .get_constraints_linearity(&mut linearity)
        } else {
            true
        };

        // Probe point for Jacobian values (linear rows have constant
        // Jacobians; this `x` is only needed because some inner
        // TNLPs assert on receipt).
        let mut x_probe = vec![0.0; n];
        let mut z_l_probe = vec![0.0; n];
        let mut z_u_probe = vec![0.0; n];
        let mut lambda_probe = vec![0.0; m_in];
        let started = self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x_probe,
            init_z: false,
            z_l: &mut z_l_probe,
            z_u: &mut z_u_probe,
            init_lambda: false,
            lambda: &mut lambda_probe,
        });
        if !started {
            return None;
        }

        // Jacobian values at the probe.
        let mut jac_values_inner = vec![0.0; nnz_in];
        if nnz_in > 0 {
            let ok = self.inner.borrow_mut().eval_jac_g(
                Some(&x_probe),
                true,
                SparsityRequest::Values {
                    values: &mut jac_values_inner,
                },
            );
            if !ok {
                return None;
            }
        }

        // Build LinearRow list from the inner Jacobian + linearity.
        let one_based = matches!(info_inner.index_style, IndexStyle::Fortran);
        let mut by_row: Vec<Vec<(Index, Number)>> = vec![Vec::new(); m_in];
        for k in 0..nnz_in {
            let i = if one_based {
                (jac_irow_inner[k] - 1) as usize
            } else {
                jac_irow_inner[k] as usize
            };
            let j = if one_based {
                jac_jcol_inner[k] - 1
            } else {
                jac_jcol_inner[k]
            };
            if i < m_in && (j as usize) < n {
                by_row[i].push((j, jac_values_inner[k]));
            }
        }
        let linear_row_map: Vec<Option<LinearRow>> = (0..m_in)
            .map(|i| {
                if have_linearity && matches!(linearity[i], Linearity::Linear) {
                    Some(LinearRow {
                        entries: by_row[i].clone(),
                        lo: g_l_inner[i],
                        hi: g_u_inner[i],
                    })
                } else {
                    None
                }
            })
            .collect();
        let linear_rows: Vec<LinearRow> =
            linear_row_map.iter().filter_map(|r| r.clone()).collect();

        // Snapshot inner bounds before Phase 1 mutates them; needed
        // for Phase 4 warm-start hints.
        let inner_x_l = x_l.clone();
        let inner_x_u = x_u.clone();

        // Phase 1: bound tightening using linear rows.
        let mut tighten_report = TightenReport::default();
        if self.opts.bound_tightening && !linear_rows.is_empty() {
            tighten_report =
                tighten_bounds(&linear_rows, &mut x_l, &mut x_u, self.opts.max_passes, 1e-12);
        }

        // Phase 4: any variable whose lower (upper) bound moved
        // strictly inward is a candidate for a bound-multiplier warm
        // start. Zero entries leave that bound's multiplier on the
        // global default (`bound_mult_init_val` from upstream).
        let warm_tol: Number = 1e-12;
        let (z_l_warm, z_u_warm) = if self.opts.warm_z_bounds {
            let v0 = self.opts.bound_mult_init_val;
            let mut zl = vec![0.0; n];
            let mut zu = vec![0.0; n];
            for i in 0..n {
                if x_l[i] > inner_x_l[i] + warm_tol {
                    zl[i] = v0;
                }
                if x_u[i] < inner_x_u[i] - warm_tol {
                    zu[i] = v0;
                }
            }
            (zl, zu)
        } else {
            (vec![0.0; n], vec![0.0; n])
        };

        // Phase 2: detect redundant linear rows in the (possibly
        // tightened) box. Non-linear rows are never dropped.
        let mut row_kept_inner: Vec<bool> = vec![true; m_in];
        let mut n_dropped_rows: Index = 0;
        if self.opts.redundant_constraint_removal {
            let redundant_mask = find_redundant_rows(&linear_rows, &x_l, &x_u, 1e-9);
            // redundant_mask aligns with `linear_rows`; map back to
            // inner row indices.
            let mut linear_iter = redundant_mask.iter();
            for (i, lr) in linear_row_map.iter().enumerate() {
                if lr.is_some() {
                    let is_red = *linear_iter.next().unwrap_or(&false);
                    if is_red {
                        row_kept_inner[i] = false;
                        n_dropped_rows += 1;
                    }
                }
            }
        }

        // Phase 3: structural LICQ check on the kept equality rows.
        let licq_verdict = if self.opts.licq_check {
            let eq_tol: Number = 1e-12;
            let mut eq_rows: Vec<EqRow> = Vec::new();
            for (i, &kept) in row_kept_inner.iter().enumerate() {
                if !kept {
                    continue;
                }
                if (g_u_inner[i] - g_l_inner[i]).abs() > eq_tol {
                    continue;
                }
                use std::collections::BTreeSet;
                let mut cols: BTreeSet<Index> = BTreeSet::new();
                for &(j, v) in &by_row[i] {
                    if v != 0.0 {
                        cols.insert(j);
                    }
                }
                eq_rows.push(EqRow {
                    cols: cols.into_iter().collect(),
                });
            }
            Some(licq_check(&eq_rows, info_inner.n))
        } else {
            None
        };

        // Build outer row mapping.
        let mut rows_kept: Vec<usize> = Vec::with_capacity(m_in);
        let mut row_inner_to_outer = vec![usize::MAX; m_in];
        for (i, &kept) in row_kept_inner.iter().enumerate() {
            if kept {
                row_inner_to_outer[i] = rows_kept.len();
                rows_kept.push(i);
            }
        }
        let m_out = rows_kept.len();

        // Build outer Jacobian sparsity: keep entries whose row is
        // kept, renumber rows.
        let mut jac_kept_idx = Vec::new();
        let mut jac_irow_outer = Vec::new();
        let mut jac_jcol_outer = Vec::new();
        for k in 0..nnz_in {
            let i_inner = if one_based {
                (jac_irow_inner[k] - 1) as usize
            } else {
                jac_irow_inner[k] as usize
            };
            if i_inner >= m_in {
                continue;
            }
            if !row_kept_inner[i_inner] {
                continue;
            }
            let outer = row_inner_to_outer[i_inner];
            let outer_row_index = if one_based {
                (outer as Index) + 1
            } else {
                outer as Index
            };
            jac_irow_outer.push(outer_row_index);
            jac_jcol_outer.push(jac_jcol_inner[k]);
            jac_kept_idx.push(k);
        }
        let nnz_out = jac_kept_idx.len();

        // Reduced g_l, g_u in outer ordering.
        let g_l: Vec<Number> = rows_kept.iter().map(|&i| g_l_inner[i]).collect();
        let g_u: Vec<Number> = rows_kept.iter().map(|&i| g_u_inner[i]).collect();

        let info_outer = NlpInfo {
            n: info_inner.n,
            m: m_out as Index,
            nnz_jac_g: nnz_out as Index,
            // Linear rows contribute zero to the Hessian, so dropping
            // them does not change `nnz_h_lag`. We carry the inner
            // sparsity through unchanged.
            nnz_h_lag: info_inner.nnz_h_lag,
            index_style: info_inner.index_style,
        };

        self.state = Some(PresolveState {
            info_inner,
            info_outer,
            bounds: CachedBounds { x_l, x_u, g_l, g_u },
            rows_kept,
            jac_kept_idx,
            jac_irow_outer,
            jac_jcol_outer,
            tighten_report,
            n_dropped_rows,
            licq_verdict,
            z_l_warm,
            z_u_warm,
            scratch_g: vec![0.0; m_in],
            scratch_jac: vec![0.0; nnz_in],
            scratch_lambda: vec![0.0; m_in],
        });
        self.state.as_ref()
    }
}

// Inside this impl every `.expect("inited")` is invariant-protected
// by the preceding `ensure_init` (which is the only way state ever
// becomes `Some`).
#[allow(clippy::expect_used)]
impl TNLP for PresolveTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let s = self.ensure_init()?;
        Some(s.info_outer)
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let Some(s) = self.ensure_init() else {
            return false;
        };
        b.x_l.copy_from_slice(&s.bounds.x_l);
        b.x_u.copy_from_slice(&s.bounds.x_u);
        b.g_l.copy_from_slice(&s.bounds.g_l);
        b.g_u.copy_from_slice(&s.bounds.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // n is unchanged by presolve; lambda warm-start is masked to
        // kept rows only (caller already sized `sp.lambda` to m_out).
        let Some(_) = self.ensure_init() else {
            return false;
        };
        // For now, ask the inner TNLP for its starting point in full
        // and project lambda down. Most users don't warm-start
        // duals, so this hits the no-op path.
        let m_in = self.state.as_ref().expect("inited").info_inner.m as usize;
        let mut z_l_full = vec![0.0; sp.z_l.len()];
        let mut z_u_full = vec![0.0; sp.z_u.len()];
        let mut lambda_full = vec![0.0; m_in];
        let ok = self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: sp.init_x,
            x: sp.x,
            init_z: sp.init_z,
            z_l: &mut z_l_full,
            z_u: &mut z_u_full,
            init_lambda: sp.init_lambda,
            lambda: &mut lambda_full,
        });
        if !ok {
            return false;
        }
        sp.z_l.copy_from_slice(&z_l_full);
        sp.z_u.copy_from_slice(&z_u_full);
        let s = self.state.as_ref().expect("inited");
        // Phase 4: overlay presolve hints onto any zero/unset
        // entries. User-supplied warm-start values always win.
        if sp.init_z && self.opts.warm_z_bounds {
            for (i, &hint) in s.z_l_warm.iter().enumerate() {
                if hint > 0.0 && sp.z_l[i] <= 0.0 {
                    sp.z_l[i] = hint;
                }
            }
            for (i, &hint) in s.z_u_warm.iter().enumerate() {
                if hint > 0.0 && sp.z_u[i] <= 0.0 {
                    sp.z_u[i] = hint;
                }
            }
        }
        for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
            sp.lambda[outer] = lambda_full[i_inner];
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        let s = self.state.as_mut().expect("inited");
        if !self
            .inner
            .borrow_mut()
            .eval_g(x, new_x, &mut s.scratch_g)
        {
            return false;
        }
        for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
            g[outer] = s.scratch_g[i_inner];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let s = self.state.as_ref().expect("inited");
                irow.copy_from_slice(&s.jac_irow_outer);
                jcol.copy_from_slice(&s.jac_jcol_outer);
                true
            }
            SparsityRequest::Values { values } => {
                let s = self.state.as_mut().expect("inited");
                if !self.inner.borrow_mut().eval_jac_g(
                    x,
                    new_x,
                    SparsityRequest::Values {
                        values: &mut s.scratch_jac,
                    },
                ) {
                    return false;
                }
                for (outer_k, &inner_k) in s.jac_kept_idx.iter().enumerate() {
                    values[outer_k] = s.scratch_jac[inner_k];
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        // Hessian sparsity is untouched: linear rows (the only ones
        // we drop) contribute zero. Forward `lambda` after expanding
        // outer → inner with zeros at dropped rows.
        let lambda_full_opt = if let Some(lam) = lambda {
            let s = self.state.as_mut().expect("inited");
            for v in s.scratch_lambda.iter_mut() {
                *v = 0.0;
            }
            for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
                s.scratch_lambda[i_inner] = lam[outer];
            }
            Some(&s.scratch_lambda[..])
        } else {
            None
        };
        // Re-borrow inner after dropping the state borrow.
        let lam_ref: Option<&[Number]> = lambda_full_opt;
        // SAFETY: `lam_ref` borrows from `self.state`'s scratch; the
        // call to `inner.borrow_mut()` does not touch `self.state`.
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lam_ref, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        let Some(_) = self.ensure_init() else {
            // Init failed earlier — best effort: just forward as-is.
            self.inner.borrow_mut().finalize_solution(sol, ip_data, ip_cq);
            return;
        };
        // Reconstruct inner-sized g and lambda.
        let (g_full, lambda_full) = {
            let s = self.state.as_mut().expect("inited");
            // Recompute g at sol.x — the solver gave us reduced g.
            self.inner
                .borrow_mut()
                .eval_g(sol.x, true, &mut s.scratch_g);
            for v in s.scratch_lambda.iter_mut() {
                *v = 0.0;
            }
            for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
                s.scratch_lambda[i_inner] = sol.lambda[outer];
            }
            (s.scratch_g.clone(), s.scratch_lambda.clone())
        };
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: sol.x,
                z_l: sol.z_l,
                z_u: sol.z_u,
                g: &g_full,
                lambda: &lambda_full,
                obj_value: sol.obj_value,
            },
            ip_data,
            ip_cq,
        );
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        // Variable count is unchanged by presolve, so var metadata
        // flows through. Constraint metadata is per-inner-row; if we
        // dropped rows, subset the per-row vectors to kept rows.
        let mut inner_var = MetaData::default();
        let mut inner_con = MetaData::default();
        if !self
            .inner
            .borrow_mut()
            .get_var_con_metadata(&mut inner_var, &mut inner_con)
        {
            return false;
        }
        *var = inner_var;
        let s = self.state.as_ref().expect("inited");
        let m_in = s.info_inner.m as usize;
        *con = project_con_metadata(&inner_con, &s.rows_kept, m_in);
        true
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        let s = self.state.as_ref().expect("inited");
        let m_in = s.info_inner.m as usize;
        // Allocate inner-sized g_scaling and forward.
        let mut inner_g = vec![1.0; m_in];
        let mut use_x = false;
        let mut use_g = false;
        let mut obj_scaling = 1.0;
        let inner_x_scaling_len = req.x_scaling.len();
        let mut inner_x = vec![1.0; inner_x_scaling_len];
        let ok = self.inner.borrow_mut().get_scaling_parameters(ScalingRequest {
            obj_scaling: &mut obj_scaling,
            use_x_scaling: &mut use_x,
            x_scaling: &mut inner_x,
            use_g_scaling: &mut use_g,
            g_scaling: &mut inner_g,
        });
        if !ok {
            return false;
        }
        *req.obj_scaling = obj_scaling;
        *req.use_x_scaling = use_x;
        *req.use_g_scaling = use_g;
        req.x_scaling.copy_from_slice(&inner_x);
        for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
            req.g_scaling[outer] = inner_g[i_inner];
        }
        true
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        let Some(_) = self.ensure_init() else {
            return false;
        };
        let m_in = self.state.as_ref().expect("inited").info_inner.m as usize;
        let mut full = vec![Linearity::NonLinear; m_in];
        if !self.inner.borrow_mut().get_constraints_linearity(&mut full) {
            return false;
        }
        let s = self.state.as_ref().expect("inited");
        for (outer, &i_inner) in s.rows_kept.iter().enumerate() {
            types[outer] = full[i_inner];
        }
        true
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    fn get_list_of_nonlinear_variables(&mut self, pos_nonlin_vars: &mut [Index]) -> bool {
        self.inner
            .borrow_mut()
            .get_list_of_nonlinear_variables(pos_nonlin_vars)
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        let Some(_) = self.ensure_init() else {
            self.inner.borrow_mut().finalize_metadata(var, con);
            return;
        };
        let s = self.state.as_ref().expect("inited");
        let m_in = s.info_inner.m as usize;
        let con_full = expand_con_metadata(con, &s.rows_kept, m_in);
        self.inner.borrow_mut().finalize_metadata(var, &con_full);
    }
}

/// Subset every per-row vector of `inner` to the rows in
/// `rows_kept`. Per-row is identified by length == `m_in`; other
/// vectors are passed through unchanged.
fn project_con_metadata(inner: &MetaData, rows_kept: &[usize], m_in: usize) -> MetaData {
    let mut out = MetaData::default();
    for (k, v) in &inner.strings {
        out.strings.insert(
            k.clone(),
            if v.len() == m_in {
                rows_kept.iter().map(|&i| v[i].clone()).collect()
            } else {
                v.clone()
            },
        );
    }
    for (k, v) in &inner.integers {
        out.integers.insert(
            k.clone(),
            if v.len() == m_in {
                rows_kept.iter().map(|&i| v[i]).collect()
            } else {
                v.clone()
            },
        );
    }
    for (k, v) in &inner.numerics {
        out.numerics.insert(
            k.clone(),
            if v.len() == m_in {
                rows_kept.iter().map(|&i| v[i]).collect()
            } else {
                v.clone()
            },
        );
    }
    out
}

/// Expand every per-(outer-row) vector back to `m_in` rows by
/// inserting empty / 0 / 0.0 defaults at dropped rows.
fn expand_con_metadata(outer: &MetaData, rows_kept: &[usize], m_in: usize) -> MetaData {
    let m_out = rows_kept.len();
    let mut full = MetaData::default();
    for (k, v) in &outer.strings {
        let mut buf: Vec<String> = vec![String::new(); m_in];
        if v.len() == m_out {
            for (outer_i, val) in v.iter().enumerate() {
                buf[rows_kept[outer_i]] = val.clone();
            }
            full.strings.insert(k.clone(), buf);
        } else {
            full.strings.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in &outer.integers {
        let mut buf: Vec<Index> = vec![0; m_in];
        if v.len() == m_out {
            for (outer_i, &val) in v.iter().enumerate() {
                buf[rows_kept[outer_i]] = val;
            }
            full.integers.insert(k.clone(), buf);
        } else {
            full.integers.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in &outer.numerics {
        let mut buf: Vec<Number> = vec![0.0; m_in];
        if v.len() == m_out {
            for (outer_i, &val) in v.iter().enumerate() {
                buf[rows_kept[outer_i]] = val;
            }
            full.numerics.insert(k.clone(), buf);
        } else {
            full.numerics.insert(k.clone(), v.clone());
        }
    }
    full
}

/// Re-export for callers that already imported
/// `pounce_presolve::register_options` directly.
pub fn register(reg: &RegisteredOptions) -> Result<(), SolverException> {
    register_options(reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe;
    impl TNLP for Probe {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, _b: BoundsInfo<'_>) -> bool {
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn disabled_returns_inner_unchanged() {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Probe));
        let opts = PresolveOptions {
            enabled: false,
            ..PresolveOptions::defaults()
        };
        let wrapped = wrap_with_presolve(Rc::clone(&inner), opts).unwrap();
        assert!(Rc::ptr_eq(&inner, &wrapped));
    }

    #[test]
    fn enabled_wraps_and_forwards() {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Probe));
        let opts = PresolveOptions {
            enabled: true,
            ..PresolveOptions::defaults()
        };
        let wrapped = wrap_with_presolve(Rc::clone(&inner), opts).unwrap();
        assert!(!Rc::ptr_eq(&inner, &wrapped));
        let info = wrapped.borrow_mut().get_nlp_info().unwrap();
        assert_eq!(info.n, 1);
        assert_eq!(info.m, 0);
    }

    #[test]
    fn register_options_roundtrip() {
        let reg = RegisteredOptions::default();
        register_options(&reg).unwrap();
        let opt = reg.get_option("presolve").expect("presolve registered");
        assert_eq!(opt.name, "presolve");
    }
}
