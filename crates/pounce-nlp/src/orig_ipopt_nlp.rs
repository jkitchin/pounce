//! `OrigIpoptNlp` — concrete `IpoptNlp` impl that wraps a [`TNLPAdapter`]
//! and an [`NlpScaling`] object. Port of
//! `Algorithm/IpOrigIpoptNLP.{hpp,cpp}` (Ipopt 3.14.19).
//!
//! # Design
//!
//! Upstream `OrigIpoptNLP` does four things:
//!
//! 1. Holds the equality / inequality-separated bound vectors
//!    (`x_L`, `x_U`, `d_L`, `d_U`) and the four expansion matrices
//!    (`Px_L`, `Px_U`, `Pd_L`, `Pd_U`).
//! 2. Routes `f / grad_f / c / d / jac_c / jac_d / h` evaluations down
//!    to the user's `NLP` (i.e. our `TNLP` via `TNLPAdapter`), splitting
//!    constraints into c/d and applying scaling.
//! 3. Caches each result keyed on the input vector tag (`CachedResults`
//!    upstream → [`pounce_common::cached::Cache`] here).
//! 4. Counts evaluations and forwards the unscaled solution to
//!    `TNLP::finalize_solution`.
//!
//! # Trait location
//!
//! The `Nlp` / `IpoptNlp` traits live in [`crate::ipopt_nlp`] and are
//! re-exported from `pounce_algorithm::ipopt_nlp` so the algorithm-side
//! code can keep its existing `crate::ipopt_nlp::IpoptNlp` import path.
//! We moved the traits down to `pounce-nlp` because the concrete impl
//! has to live alongside `TNLPAdapter` (its private dependency) and
//! `pounce-algorithm` already depends on `pounce-nlp` (the reverse
//! would cycle).
//!
//! # Phase scope
//!
//! Implemented for v1.0:
//! * `f / grad_f / c / d / jac_c / jac_d / h` with one-dependency
//!   tag-keyed caches.
//! * Bound vectors and 0/1 expansion matrices for `(x_L, x_U, d_L, d_U)`.
//! * Starting point retrieval + initial multiplier handling.
//! * Per-eval counters (used to populate `SolveStatistics`).
//! * `finalize_solution` plumbing.
//!
//! Deferred (phase numbers from
//! `we-are-going-to-polished-simon.md`):
//! * Phase 8: L-BFGS / SR1 quasi-Newton path
//!   (`hessian_approximation = limited-memory`). The `eval_h` path is
//!   wired here, but the `LowRankUpdateSymMatrix` h_space construction
//!   in `IpOrigIpoptNLP.cpp:251-278` is not.
//! * Phase 10: adaptive-mu's `objective_depends_on_mu` /
//!   `f(x, mu)` overload (CG-penalty objective).
//! * Bound relaxation (`bound_relax_factor`) and `honor_original_bounds`
//!   projection — these need `OptionsList` plumbing that lands later.
//! * `check_derivatives_for_naninf` — needs the journalist's NaN
//!   reporting, deferred with the option.
//! * Full `NLPScalingObject` integration (currently only `obj_scaling`
//!   is used; `apply_vector_scaling_*`, `apply_jac_*_scaling`,
//!   `apply_hessian_scaling` live behind future scaling-object API).
//! * Fixed-variable removal (`x_l == x_u`) — `TNLPAdapter` keeps fixed
//!   variables in `x_var` for now; the upstream
//!   `fixed_variable_treatment` knob lands when the option machinery
//!   does.

use crate::ipopt_nlp::{IpoptNlp, Nlp, SplitNames};
use crate::tnlp::{IDX_NAMES, MetaData, NlpInfo, ScalingRequest, SparsityRequest, StartingPoint};
use crate::tnlp_adapter::{BoundClassification, TNLPAdapter};
use pounce_common::cached::Cache;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::{
    DenseVector, DenseVectorSpace, ExpansionMatrix, ExpansionMatrixSpace, GenTMatrix,
    GenTMatrixSpace, Matrix, SymMatrix, SymTMatrix, SymTMatrixSpace, Vector,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// Opaque scaling-object handle. `OrigIpoptNlp` only consults this for
/// the *initial* objective scaling; the full per-row constraint /
/// Jacobian / Hessian scaling lives directly on `OrigIpoptNlp` (see
/// the `obj_scale_factor` / `c_scale` / `d_scale` fields and the
/// `determine_scaling_from_starting_point` method) so that the runtime
/// can compute gradient-based scaling without an upcall.
///
/// The trait is intentionally minimal and local to `pounce-nlp`: the
/// gradient-based scaling arithmetic lives on `OrigIpoptNlp` itself
/// (see `determine_scaling_from_starting_point`), so there is no
/// algorithm-layer scaling strategy object.
pub trait NlpScaling {
    /// Optional user-supplied multiplier on the objective scaling
    /// factor. Mirrors upstream's `obj_scaling_factor` option (default
    /// 1.0). Combined with the gradient-based factor in
    /// `OrigIpoptNlp::determine_scaling_from_starting_point`.
    fn obj_scaling(&self) -> Number {
        1.0
    }
}

/// No-op scaling — every factor is 1.0. Default for unit tests and
/// callers that have not configured a scaling strategy.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoScaling;
impl NlpScaling for NoScaling {}

/// Constant objective scaling: carries the user's `obj_scaling_factor`
/// option value into [`OrigIpoptNlp`]. A negative factor flips the
/// optimization direction (the IPM minimizes `factor·f`, i.e.
/// maximizes `f`), matching upstream Ipopt's documented semantics.
#[derive(Debug, Clone, Copy)]
pub struct ConstObjScaling(pub Number);
impl NlpScaling for ConstObjScaling {
    fn obj_scaling(&self) -> Number {
        self.0
    }
}

/// Selector for [`OrigIpoptNlp::determine_scaling_from_starting_point`].
/// Mirrors upstream's `nlp_scaling_method` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMethod {
    /// No automatic scaling beyond the constant `obj_scaling_factor`.
    None,
    /// Gradient-based per `Algorithm/IpGradientScaling.cpp`. Default.
    GradientBased,
    /// User-supplied scaling via [`crate::tnlp::TNLP::get_scaling_parameters`].
    /// Port of upstream's `nlp_scaling_method=user-scaling`. The TNLP
    /// fills `obj_scaling` and the per-constraint `g_scaling`; the
    /// per-variable `x_scaling` request is honored only insofar as
    /// `OrigIpoptNlp` currently models constraint+objective scaling
    /// (no variable-side rescale, matching the issue #61 design).
    UserScaling,
}

/// Concrete `IpoptNlp` over a `TNLPAdapter`. Mirrors upstream
/// `Ipopt::OrigIpoptNLP`.
pub struct OrigIpoptNlp {
    /// Backing TNLP (and its bound classification).
    adapter: Rc<RefCell<TNLPAdapter>>,
    /// Constant objective-scaling multiplier supplied by the user
    /// (mirrors upstream's `obj_scaling_factor` option). The
    /// gradient-based factor is multiplied into [`obj_scale_factor`]
    /// after [`Self::determine_scaling_from_starting_point`] runs.
    scaling: Rc<dyn NlpScaling>,

    // ----- gradient-based scaling state (port of `IpGradientScaling.cpp`) -----
    /// Effective objective scaling factor `df_` (1.0 when no scaling).
    obj_scale_factor: Cell<Number>,
    /// Per-row scaling for equality constraints (`dc_`). `None` ↔
    /// `IsValid(dc) == false` — i.e. row-max gradient is below the
    /// `nlp_scaling_max_gradient` cutoff so no scaling is applied.
    c_scale: RefCell<Option<Vec<Number>>>,
    /// Same as [`Self::c_scale`] but for inequality rows.
    d_scale: RefCell<Option<Vec<Number>>>,

    // ----- vector / matrix spaces (shared via Rc) -----
    x_space: Rc<DenseVectorSpace>,
    c_space: Rc<DenseVectorSpace>,
    d_space: Rc<DenseVectorSpace>,
    x_l_space: Rc<DenseVectorSpace>,
    x_u_space: Rc<DenseVectorSpace>,
    d_l_space: Rc<DenseVectorSpace>,
    d_u_space: Rc<DenseVectorSpace>,
    px_l_space: Rc<ExpansionMatrixSpace>,
    px_u_space: Rc<ExpansionMatrixSpace>,
    pd_l_space: Rc<ExpansionMatrixSpace>,
    pd_u_space: Rc<ExpansionMatrixSpace>,
    jac_c_space: Rc<GenTMatrixSpace>,
    jac_d_space: Rc<GenTMatrixSpace>,
    /// Hessian space; `None` when `eval_h` is not provided by the TNLP
    /// (the limited-memory quasi-Newton path lands in Phase 8).
    h_space: Option<Rc<SymTMatrixSpace>>,

    // ----- bound vectors (compressed-x sub-spaces) -----
    x_l: Rc<DenseVector>,
    x_u: Rc<DenseVector>,
    d_l: Rc<DenseVector>,
    d_u: Rc<DenseVector>,
    /// Constant equality right-hand side (upstream's `c_rhs`): for each
    /// equality row `i`, the bound `g_l[c_map[i]] == g_u[c_map[i]]`.
    /// Captured once at construction so [`Self::eval_c_internal`] forms the
    /// residual `g - c_rhs` without re-fetching all bounds (and the four
    /// full-size scratch allocations that requires) on every line-search
    /// trial. (Code review 2026-06 item M17.)
    c_rhs: Vec<Number>,

    // ----- expansion matrices (instances; spaces above) -----
    px_l: Rc<dyn Matrix>,
    px_u: Rc<dyn Matrix>,
    pd_l: Rc<dyn Matrix>,
    pd_u: Rc<dyn Matrix>,

    // ----- jacobian sparsity remap -----
    /// `jac_c_entry_in_g[k]` = position in the full TNLP jacobian's
    /// values array of the k-th equality-row entry.
    jac_c_entry_in_g: Vec<Index>,
    /// Same for inequality rows.
    jac_d_entry_in_g: Vec<Index>,
    /// Total nonzeros in the full (un-split) `eval_jac_g` triplet.
    nnz_jac_g_full: Index,

    // ----- hessian sparsity remap (fixed-var filtering) -----
    /// Total nonzeros the user's `eval_h` writes into. May exceed
    /// `h_space.nonzeros()` when fixed variables drop entries.
    nnz_h_lag_full: Index,
    /// `h_entry_in_full[k]` = position in the full TNLP hessian's
    /// values array of the k-th kept entry. Always has length
    /// `h_space.nonzeros()`; equals the identity `[0, 1, …, n-1]`
    /// when no fixed-var filtering dropped any entries.
    h_entry_in_full: Vec<Index>,

    // ----- caches (one entry; key = input vector tag) -----
    f_cache: RefCell<Cache<Number>>,
    grad_f_cache: RefCell<Cache<Rc<dyn Vector>>>,
    c_cache: RefCell<Cache<Rc<dyn Vector>>>,
    d_cache: RefCell<Cache<Rc<dyn Vector>>>,
    jac_c_cache: RefCell<Cache<Rc<dyn Matrix>>>,
    jac_d_cache: RefCell<Cache<Rc<dyn Matrix>>>,
    h_cache: RefCell<Cache<Rc<dyn SymMatrix>>>,
    /// Shared full-space buffers below the c/d split, so the dominant AD
    /// cost is paid once per iterate instead of twice. `eval_c`/`eval_d`
    /// both slice their rows out of one `eval_g` result (`full_g_cache`),
    /// and `eval_jac_c`/`eval_jac_d` both slice one `eval_jac_g` result
    /// (`full_jac_g_cache`). Keyed by the input vector's tag, like the
    /// per-subsystem caches; mirrors upstream's tagged `full_g_`/`jac_g_`
    /// buffers. (Code review 2026-06 item M16.)
    full_g_cache: RefCell<Cache<Rc<Vec<Number>>>>,
    full_jac_g_cache: RefCell<Cache<Rc<Vec<Number>>>>,

    // ----- evaluation counters -----
    f_evals: RefCell<Index>,
    grad_f_evals: RefCell<Index>,
    c_evals: RefCell<Index>,
    d_evals: RefCell<Index>,
    jac_c_evals: RefCell<Index>,
    jac_d_evals: RefCell<Index>,
    h_evals: RefCell<Index>,

    /// Cached `NlpInfo` (n, m, nnz_jac_g, nnz_h_lag, index_style) so we
    /// don't re-borrow the TNLP for dimension queries.
    info: NlpInfo,

    /// Shared per-subsystem timing accumulator. `None` until
    /// `IpoptApplication` installs the shared `Rc<TimingStatistics>` via
    /// [`Self::set_timing_stats`]; when `None`, all `eval_*` entry
    /// points skip the timing overhead.
    timing: RefCell<Option<Rc<TimingStatistics>>>,
}

impl std::fmt::Debug for OrigIpoptNlp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrigIpoptNlp")
            .field("info", &self.info)
            .field("f_evals", &*self.f_evals.borrow())
            .field("grad_f_evals", &*self.grad_f_evals.borrow())
            .field("c_evals", &*self.c_evals.borrow())
            .field("d_evals", &*self.d_evals.borrow())
            .field("jac_c_evals", &*self.jac_c_evals.borrow())
            .field("jac_d_evals", &*self.jac_d_evals.borrow())
            .field("h_evals", &*self.h_evals.borrow())
            .finish_non_exhaustive()
    }
}

impl OrigIpoptNlp {
    /// Construct an `OrigIpoptNlp` from a (already-classified) adapter
    /// and a scaling object. Mirrors
    /// `OrigIpoptNLP::OrigIpoptNLP` + `InitializeStructures`
    /// (`IpOrigIpoptNLP.cpp:22-457`) — the parts that don't need an
    /// `OptionsList` (those land with the option-machinery integration).
    pub fn new(
        adapter: Rc<RefCell<TNLPAdapter>>,
        scaling: Rc<dyn NlpScaling>,
    ) -> Result<Self, String> {
        // Snapshot dimensions / classification from the adapter.
        let (info, classification) = {
            let a = adapter.borrow();
            (*a.nlp_info(), a.classification().clone())
        };

        // ---- Vector spaces ----
        let n_x_var = classification.n_x_var();
        let x_space = DenseVectorSpace::new(n_x_var);
        let c_space = DenseVectorSpace::new(classification.n_c);
        let d_space = DenseVectorSpace::new(classification.n_d);
        let x_l_space = DenseVectorSpace::new(classification.n_x_l());
        let x_u_space = DenseVectorSpace::new(classification.n_x_u());
        let d_l_space = DenseVectorSpace::new(classification.n_d_l());
        let d_u_space = DenseVectorSpace::new(classification.n_d_u());

        // ---- Expansion matrix spaces (column-compressed → full x_var / d) ----
        let px_l_space =
            ExpansionMatrixSpace::new(n_x_var, classification.n_x_l(), &classification.x_l_map, 0);
        let px_u_space =
            ExpansionMatrixSpace::new(n_x_var, classification.n_x_u(), &classification.x_u_map, 0);
        let pd_l_space = ExpansionMatrixSpace::new(
            classification.n_d,
            classification.n_d_l(),
            &classification.d_l_map,
            0,
        );
        let pd_u_space = ExpansionMatrixSpace::new(
            classification.n_d,
            classification.n_d_u(),
            &classification.d_u_map,
            0,
        );
        let px_l: Rc<dyn Matrix> = Rc::new(ExpansionMatrix::new(Rc::clone(&px_l_space)));
        let px_u: Rc<dyn Matrix> = Rc::new(ExpansionMatrix::new(Rc::clone(&px_u_space)));
        let pd_l: Rc<dyn Matrix> = Rc::new(ExpansionMatrix::new(Rc::clone(&pd_l_space)));
        let pd_u: Rc<dyn Matrix> = Rc::new(ExpansionMatrix::new(Rc::clone(&pd_u_space)));

        // ---- Bound vectors. Pull the full `x_l/x_u/g_l/g_u` arrays from
        // the TNLP (the adapter discarded them after classification) and
        // pick out the entries pointed at by the `*_map` tables. -----
        let n_full_x = classification.n_full_x as usize;
        let n_full_g = classification.n_full_g as usize;
        let mut full_x_l = vec![0.0; n_full_x];
        let mut full_x_u = vec![0.0; n_full_x];
        let mut full_g_l = vec![0.0; n_full_g];
        let mut full_g_u = vec![0.0; n_full_g];
        {
            let a = adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            let ok = t.get_bounds_info(crate::tnlp::BoundsInfo {
                x_l: &mut full_x_l,
                x_u: &mut full_x_u,
                g_l: &mut full_g_l,
                g_u: &mut full_g_u,
            });
            if !ok {
                return Err("TNLP::get_bounds_info returned false on second call".into());
            }
        }

        let x_l = make_dense_from(&x_l_space, |i| {
            // x_l_map[i] is an index into x_var (== index into x_not_fixed_map).
            let var_idx = classification.x_l_map[i] as usize;
            let full_idx = classification.x_not_fixed_map[var_idx] as usize;
            full_x_l[full_idx]
        });
        let x_u = make_dense_from(&x_u_space, |i| {
            let var_idx = classification.x_u_map[i] as usize;
            let full_idx = classification.x_not_fixed_map[var_idx] as usize;
            full_x_u[full_idx]
        });
        let d_l = make_dense_from(&d_l_space, |i| {
            // d_l_map[i] is an index into d (== position in d_map).
            let d_idx = classification.d_l_map[i] as usize;
            let full_g_idx = classification.d_map[d_idx] as usize;
            full_g_l[full_g_idx]
        });
        let d_u = make_dense_from(&d_u_space, |i| {
            let d_idx = classification.d_u_map[i] as usize;
            let full_g_idx = classification.d_map[d_idx] as usize;
            full_g_u[full_g_idx]
        });

        // ---- Constant equality RHS (`c_rhs`). For an equality row the
        // bound satisfies `g_l == g_u`; capture it once here so the
        // line-search-hot `eval_c` subtracts a cached constant instead of
        // re-fetching every bound (and allocating four full-size scratch
        // vectors) per cache miss. (Code review 2026-06 item M17.) -----
        let c_rhs: Vec<Number> = classification
            .c_map
            .iter()
            .map(|&g_idx| full_g_l[g_idx as usize])
            .collect();

        // ---- Jacobian sparsity. Ask the TNLP for the full jacobian
        // structure (g rows × full-x cols), then split entries into
        // c-rows (equality) and d-rows (inequality). Within each split
        // the row index is remapped to the new dense indexing in
        // [0, n_c) / [0, n_d).
        //
        // We currently keep all original full-x columns (no fixed-var
        // removal); when fixed-var treatment lands, this is where the
        // column remap goes. -----
        let mut full_irow = vec![0 as Index; info.nnz_jac_g as usize];
        let mut full_jcol = vec![0 as Index; info.nnz_jac_g as usize];
        {
            let a = adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            let ok = t.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut full_irow,
                    jcol: &mut full_jcol,
                },
            );
            if !ok {
                return Err("TNLP::eval_jac_g(Structure) returned false".into());
            }
        }

        // Build the inverse maps: g-row → c-row (or d-row).
        let mut g_to_c = vec![-1 as Index; n_full_g];
        for (c_idx, &g_idx) in classification.c_map.iter().enumerate() {
            g_to_c[g_idx as usize] = c_idx as Index;
        }
        let mut g_to_d = vec![-1 as Index; n_full_g];
        for (d_idx, &g_idx) in classification.d_map.iter().enumerate() {
            g_to_d[g_idx as usize] = d_idx as Index;
        }

        let style_offset = match info.index_style {
            crate::tnlp::IndexStyle::C => 0 as Index,
            crate::tnlp::IndexStyle::Fortran => 1 as Index,
        };

        let mut jac_c_irow_1based = Vec::new();
        let mut jac_c_jcol_1based = Vec::new();
        let mut jac_c_entry_in_g = Vec::new();
        let mut jac_d_irow_1based = Vec::new();
        let mut jac_d_jcol_1based = Vec::new();
        let mut jac_d_entry_in_g = Vec::new();

        // `make_parameter`: drop Jacobian entries in fixed-variable
        // columns. Their contribution to f and g is constant under the
        // active-x search so they don't appear in the KKT.
        let full_to_var = &classification.full_to_var;
        for k in 0..info.nnz_jac_g as usize {
            let g_row_0 = (full_irow[k] - style_offset) as usize;
            let x_col_0 = (full_jcol[k] - style_offset) as usize;
            let var_col = full_to_var[x_col_0];
            if var_col < 0 {
                continue;
            }
            // Triplet output is 1-based (matches `GenTMatrix` convention).
            let col_1based = var_col + 1;
            let c_row = g_to_c[g_row_0];
            if c_row >= 0 {
                jac_c_irow_1based.push(c_row + 1);
                jac_c_jcol_1based.push(col_1based);
                jac_c_entry_in_g.push(k as Index);
            } else {
                let d_row = g_to_d[g_row_0];
                debug_assert!(d_row >= 0, "g row {g_row_0} is neither in c_map nor d_map");
                jac_d_irow_1based.push(d_row + 1);
                jac_d_jcol_1based.push(col_1based);
                jac_d_entry_in_g.push(k as Index);
            }
        }

        let jac_c_space = GenTMatrixSpace::new(
            classification.n_c,
            n_x_var,
            jac_c_irow_1based,
            jac_c_jcol_1based,
        );
        let jac_d_space = GenTMatrixSpace::new(
            classification.n_d,
            n_x_var,
            jac_d_irow_1based,
            jac_d_jcol_1based,
        );

        // ---- Hessian sparsity (optional). If the TNLP doesn't
        // implement `eval_h`, we leave `h_space = None`. The Phase-8
        // limited-memory quasi-Newton path will populate it from
        // `LowRankUpdateSymMatrixSpace` instead. -----
        let nnz_h_lag_full = info.nnz_h_lag;
        let mut h_entry_in_full: Vec<Index> = Vec::new();
        let h_space = if info.nnz_h_lag > 0 {
            let mut h_irow = vec![0 as Index; info.nnz_h_lag as usize];
            let mut h_jcol = vec![0 as Index; info.nnz_h_lag as usize];
            let supports_h = {
                let a = adapter.borrow();
                let mut t = a.tnlp().borrow_mut();
                t.eval_h(
                    None,
                    false,
                    1.0,
                    None,
                    false,
                    SparsityRequest::Structure {
                        irow: &mut h_irow,
                        jcol: &mut h_jcol,
                    },
                )
            };
            if supports_h {
                // `make_parameter`: drop Hessian entries where row OR
                // column is fixed (the second derivatives w.r.t. a
                // parameter are not needed in the active-x KKT). The
                // surviving entries are remapped from full-x indices
                // to var-x indices via `full_to_var`.
                let mut h_irow_1: Vec<Index> = Vec::with_capacity(h_irow.len());
                let mut h_jcol_1: Vec<Index> = Vec::with_capacity(h_jcol.len());
                for k in 0..h_irow.len() {
                    let i_full = (h_irow[k] - style_offset) as usize;
                    let j_full = (h_jcol[k] - style_offset) as usize;
                    let i_var = full_to_var[i_full];
                    let j_var = full_to_var[j_full];
                    if i_var < 0 || j_var < 0 {
                        continue;
                    }
                    h_irow_1.push(i_var + 1);
                    h_jcol_1.push(j_var + 1);
                    h_entry_in_full.push(k as Index);
                }
                Some(SymTMatrixSpace::new(n_x_var, h_irow_1, h_jcol_1))
            } else {
                // TODO(Phase 8): wire the L-BFGS / SR1 path here.
                None
            }
        } else {
            // LPs and other problems with structurally zero Hessian: build an
            // empty SymTMatrixSpace so eval_h_internal returns a zero matrix
            // rather than panicking down the L-BFGS error path.
            Some(SymTMatrixSpace::new(n_x_var, Vec::new(), Vec::new()))
        };

        // Honor the scaling object's constant factor from construction
        // (a negative `obj_scaling_factor` means maximize). Callers
        // that run `determine_scaling_from_starting_point` overwrite
        // this with the combined automatic·user factor; callers that
        // don't (e.g. the SQP path) still get the user's constant.
        let initial_obj_scal = scaling.obj_scaling();
        Ok(Self {
            adapter,
            scaling,
            obj_scale_factor: Cell::new(initial_obj_scal),
            c_scale: RefCell::new(None),
            d_scale: RefCell::new(None),
            x_space,
            c_space,
            d_space,
            x_l_space,
            x_u_space,
            d_l_space,
            d_u_space,
            px_l_space,
            px_u_space,
            pd_l_space,
            pd_u_space,
            jac_c_space,
            jac_d_space,
            h_space,
            x_l: Rc::new(x_l),
            x_u: Rc::new(x_u),
            d_l: Rc::new(d_l),
            d_u: Rc::new(d_u),
            c_rhs,
            px_l,
            px_u,
            pd_l,
            pd_u,
            jac_c_entry_in_g,
            jac_d_entry_in_g,
            nnz_jac_g_full: info.nnz_jac_g,
            nnz_h_lag_full,
            h_entry_in_full,
            f_cache: RefCell::new(Cache::new(1)),
            grad_f_cache: RefCell::new(Cache::new(1)),
            c_cache: RefCell::new(Cache::new(1)),
            d_cache: RefCell::new(Cache::new(1)),
            jac_c_cache: RefCell::new(Cache::new(1)),
            jac_d_cache: RefCell::new(Cache::new(1)),
            h_cache: RefCell::new(Cache::new(1)),
            full_g_cache: RefCell::new(Cache::new(1)),
            full_jac_g_cache: RefCell::new(Cache::new(1)),
            f_evals: RefCell::new(0),
            grad_f_evals: RefCell::new(0),
            c_evals: RefCell::new(0),
            d_evals: RefCell::new(0),
            jac_c_evals: RefCell::new(0),
            jac_d_evals: RefCell::new(0),
            h_evals: RefCell::new(0),
            info,
            timing: RefCell::new(None),
        })
    }

    /// Install the shared timing accumulator. `IpoptApplication` calls
    /// this once per solve so each `eval_*` entrypoint records into the
    /// same `TimingStatistics` instance the algorithm reports at the
    /// end of the run. Calling with `None` (or never calling) leaves
    /// timing disabled.
    pub fn set_timing_stats(&self, t: Rc<TimingStatistics>) {
        *self.timing.borrow_mut() = Some(t);
    }

    /// Run `f` with two timers active for the duration of the call:
    /// `pick(&timing)` (e.g. `eval_obj`) and `total_function_evaluation_time`.
    /// When no `TimingStatistics` is installed, the closure is invoked
    /// directly with no overhead.
    fn timed_eval<R, F>(&self, pick: fn(&TimingStatistics) -> &pounce_common::TimedTask, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let guard = self.timing.borrow();
        match guard.as_deref() {
            Some(t) => {
                let task = pick(t);
                task.start();
                t.total_function_evaluation_time.start();
                let r = f();
                t.total_function_evaluation_time.end();
                task.end();
                r
            }
            None => {
                drop(guard);
                f()
            }
        }
    }

    // ---- accessors used by the algorithm wiring layer ----

    pub fn nlp_info(&self) -> &NlpInfo {
        &self.info
    }
    pub fn classification_n_x_var(&self) -> Index {
        self.x_space.dim()
    }
    pub fn x_space(&self) -> &Rc<DenseVectorSpace> {
        &self.x_space
    }
    pub fn c_space(&self) -> &Rc<DenseVectorSpace> {
        &self.c_space
    }
    pub fn d_space(&self) -> &Rc<DenseVectorSpace> {
        &self.d_space
    }
    pub fn x_l_space(&self) -> &Rc<DenseVectorSpace> {
        &self.x_l_space
    }
    pub fn x_u_space(&self) -> &Rc<DenseVectorSpace> {
        &self.x_u_space
    }
    pub fn d_l_space(&self) -> &Rc<DenseVectorSpace> {
        &self.d_l_space
    }
    pub fn d_u_space(&self) -> &Rc<DenseVectorSpace> {
        &self.d_u_space
    }
    pub fn px_l_space(&self) -> &Rc<ExpansionMatrixSpace> {
        &self.px_l_space
    }
    pub fn px_u_space(&self) -> &Rc<ExpansionMatrixSpace> {
        &self.px_u_space
    }
    pub fn pd_l_space(&self) -> &Rc<ExpansionMatrixSpace> {
        &self.pd_l_space
    }
    pub fn pd_u_space(&self) -> &Rc<ExpansionMatrixSpace> {
        &self.pd_u_space
    }
    pub fn jac_c_space(&self) -> &Rc<GenTMatrixSpace> {
        &self.jac_c_space
    }
    pub fn jac_d_space(&self) -> &Rc<GenTMatrixSpace> {
        &self.jac_d_space
    }
    pub fn h_space(&self) -> Option<&Rc<SymTMatrixSpace>> {
        self.h_space.as_ref()
    }

    /// Effective objective scaling factor (`df_` upstream). 1.0 when
    /// no scaling has been determined.
    pub fn obj_scale_factor(&self) -> Number {
        self.obj_scale_factor.get()
    }

    /// Apply `bound_relax_factor` to the unscaled `x_L / x_U / d_L / d_U`
    /// in place. Mirrors `OrigIpoptNLP::relax_bounds`
    /// (`IpOrigIpoptNLP.cpp:343-358, 459-481`):
    ///
    /// `delta_i = min(constr_viol_tol, |relax| * max(|bound_i|, 1))`,
    /// then `x_L -= delta`, `x_U += delta`, `d_L -= delta`, `d_U += delta`.
    ///
    /// Must be called before `determine_scaling_from_starting_point`
    /// (which only reads bounds via cached evals — so order doesn't
    /// affect scaling — but the bounds themselves should be the
    /// post-relax values when they enter the algorithm).
    pub fn relax_bounds(&mut self, bound_relax_factor: Number, constr_viol_tol: Number) {
        if bound_relax_factor <= 0.0 {
            return;
        }
        let relax = bound_relax_factor.abs();
        let cap = constr_viol_tol;
        let apply = |v: &mut DenseVector, sign: Number| {
            let xs = v.values_mut();
            for x in xs.iter_mut() {
                let delta = (relax * x.abs().max(1.0)).min(cap);
                *x += sign * delta;
            }
        };
        // The bound `Rc`s are uniquely owned (nothing clones them — the same
        // invariant `adjust_variable_bounds` relies on), so `get_mut` must
        // succeed. A shared `Rc` here would silently skip the relaxation,
        // leaving bounds tighter than `bound_relax_factor` requires; that is
        // a programming error, so fail loudly to match `adjust_variable_bounds`
        // rather than no-op.
        apply(
            Rc::get_mut(&mut self.x_l).expect("relax_bounds: x_l is uniquely owned"),
            -1.0,
        );
        apply(
            Rc::get_mut(&mut self.x_u).expect("relax_bounds: x_u is uniquely owned"),
            1.0,
        );
        apply(
            Rc::get_mut(&mut self.d_l).expect("relax_bounds: d_l is uniquely owned"),
            -1.0,
        );
        apply(
            Rc::get_mut(&mut self.d_u).expect("relax_bounds: d_u is uniquely owned"),
            1.0,
        );
    }

    /// Determine objective + per-constraint scaling from the starting
    /// point, per `Algorithm/IpGradientScaling.cpp::DetermineScalingParametersImpl`
    /// (and now also `nlp_scaling_method=user-scaling`). Should be called
    /// once, after construction and before the algorithm enters its main
    /// loop.
    ///
    /// Arguments:
    /// * `method` — `None` / `GradientBased` / `UserScaling`.
    /// * `max_gradient` — `nlp_scaling_max_gradient` (cutoff above which
    ///   gradient-based scaling fires; default 100).
    /// * `min_value` — `nlp_scaling_min_value` (floor on computed scale
    ///   factors; default 1e-8).
    /// * `obj_target_gradient` — `nlp_scaling_obj_target_gradient`
    ///   (default 0; when `> 0`, fixes `df = obj_target_gradient /
    ///   max_grad_f` unconditionally, overriding the cutoff).
    /// * `constr_target_gradient` — `nlp_scaling_constr_target_gradient`
    ///   (default 0; when `> 0`, fixes per-row scale to
    ///   `constr_target_gradient / row_max` unconditionally).
    ///
    /// Cache state is invalidated so subsequent eval calls produce
    /// scaled values.
    pub fn determine_scaling_from_starting_point(
        &mut self,
        method: ScalingMethod,
        max_gradient: Number,
        min_value: Number,
        obj_target_gradient: Number,
        constr_target_gradient: Number,
    ) {
        // Always pull the user's `obj_scaling_factor` constant first;
        // it multiplies whatever the automatic scheme computes.
        let user_obj_factor = self.scaling.obj_scaling();
        if matches!(method, ScalingMethod::None) {
            self.obj_scale_factor.set(user_obj_factor);
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
            self.invalidate_eval_caches();
            return;
        }

        // ---- Get starting x_full (needed by both gradient + user paths) ----
        let cls = self.adapter.borrow().classification().clone();
        let n_full_x = cls.n_full_x as usize;
        let n_full_g = cls.n_full_g as usize;
        let mut full_x = vec![0.0; n_full_x];
        let mut full_z_l = vec![0.0; n_full_x];
        let mut full_z_u = vec![0.0; n_full_x];
        let mut full_lambda = vec![0.0; n_full_g];
        let starting_ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.get_starting_point(StartingPoint {
                init_x: true,
                x: &mut full_x,
                init_z: false,
                z_l: &mut full_z_l,
                z_u: &mut full_z_u,
                init_lambda: false,
                lambda: &mut full_lambda,
            })
        };
        if !starting_ok {
            // Fall back to no automatic scaling.
            self.obj_scale_factor.set(user_obj_factor);
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
            self.invalidate_eval_caches();
            return;
        }

        // Lift fixed variables (x_l == x_u) to their fixed value before
        // sampling the gradient / Jacobian. Fixed vars never enter the
        // algorithm's compressed x; every algorithm-side eval re-inserts
        // their fixed value via `lift_x_to_full`, so scaling must be
        // computed at that same point. Upstream achieves this implicitly:
        // `TNLPAdapter::GetStartingPoint` projects the start onto the
        // (relaxed) bounds, pinning fixed vars to their value. A raw `x0`
        // that leaves them elsewhere can shift the objective gradient by
        // orders of magnitude (pounce: flosp2hm — 41 fixed vars sitting at
        // x0=0 instead of their fixed value 1 made ‖∇f‖∞ read 40 instead of
        // 2.4e5, so obj_scale_factor stayed 1.0 and the solve stalled at
        // max-iter while IPOPT, scaling correctly, converged in 5 iters).
        for (i, &full_idx) in cls.x_fixed_map.iter().enumerate() {
            full_x[full_idx as usize] = cls.x_fixed_vals[i];
        }

        match method {
            ScalingMethod::None => unreachable!("handled above"),
            ScalingMethod::GradientBased => {
                self.scale_gradient_based(
                    &cls,
                    &full_x,
                    user_obj_factor,
                    max_gradient,
                    min_value,
                    obj_target_gradient,
                    constr_target_gradient,
                );
            }
            ScalingMethod::UserScaling => {
                let applied = self.scale_user_supplied(&cls, user_obj_factor, min_value);
                if !applied {
                    // TNLP declined to supply scaling — fall through to
                    // no automatic scaling (matches upstream's behavior
                    // when `get_scaling_parameters` returns false).
                    self.obj_scale_factor.set(user_obj_factor);
                    *self.c_scale.borrow_mut() = None;
                    *self.d_scale.borrow_mut() = None;
                }
            }
        }

        // Apply the d-row scaling to the d_l/d_u bound vectors so
        // feasibility checks compare like with like (gh#54).
        self.apply_d_scale_to_bounds();

        // Drop any cached eval results computed before the scales were
        // set (their values would be wrong now).
        self.invalidate_eval_caches();
    }

    /// Gradient-based pathway: compute `df_`, `dc_`, `dd_` from the
    /// objective gradient and constraint Jacobian at the starting point.
    fn scale_gradient_based(
        &self,
        cls: &BoundClassification,
        full_x: &[Number],
        user_obj_factor: Number,
        max_gradient: Number,
        min_value: Number,
        obj_target_gradient: Number,
        constr_target_gradient: Number,
    ) {
        let n_full_x = cls.n_full_x as usize;
        let n_full_g = cls.n_full_g as usize;

        // ---- Objective gradient scale ----
        let mut full_grad_f = vec![0.0; n_full_x];
        let grad_ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_grad_f(full_x, true, &mut full_grad_f)
        };
        let mut df = 1.0;
        if grad_ok {
            // Amax over the *compressed* x_var space (matches upstream
            // which scales the algorithm-side gradient).
            let mut max_grad_f: Number = 0.0;
            for &full_idx in cls.x_not_fixed_map.iter() {
                let v = full_grad_f[full_idx as usize].abs();
                if v > max_grad_f {
                    max_grad_f = v;
                }
            }
            if obj_target_gradient > 0.0 && max_grad_f > 0.0 {
                // Target overrides the cutoff (and the 1.0 clamp):
                // pin gradient ∞-norm to the requested value.
                df = obj_target_gradient / max_grad_f;
            } else if max_grad_f > max_gradient {
                df = max_gradient / max_grad_f;
            }
            if df < min_value {
                df = min_value;
            }
        }
        self.obj_scale_factor.set(df * user_obj_factor);

        // ---- Constraint Jacobian row-max scaling ----
        if cls.n_full_g == 0 {
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
            return;
        }
        // Evaluate full Jacobian once at x.
        let mut full_jac_vals = vec![0.0; self.nnz_jac_g_full as usize];
        let jac_ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_jac_g(
                Some(full_x),
                true,
                SparsityRequest::Values {
                    values: &mut full_jac_vals,
                },
            )
        };
        if !jac_ok {
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
            return;
        }
        // Recover row indices from the sparsity structure.
        let mut full_irow = vec![0 as Index; self.nnz_jac_g_full as usize];
        let mut full_jcol = vec![0 as Index; self.nnz_jac_g_full as usize];
        let _ = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut full_irow,
                    jcol: &mut full_jcol,
                },
            )
        };
        let style_offset: Index = match self.info.index_style {
            crate::tnlp::IndexStyle::C => 0,
            crate::tnlp::IndexStyle::Fortran => 1,
        };
        // Build inverse row maps to assign each entry to c or d.
        let mut g_to_c = vec![-1 as Index; n_full_g];
        for (c_idx, &g_idx) in cls.c_map.iter().enumerate() {
            g_to_c[g_idx as usize] = c_idx as Index;
        }
        let mut g_to_d = vec![-1 as Index; n_full_g];
        for (d_idx, &g_idx) in cls.d_map.iter().enumerate() {
            g_to_d[g_idx as usize] = d_idx as Index;
        }
        let n_c = cls.n_c as usize;
        let n_d = cls.n_d as usize;
        // Initialize row-max arrays to dbl_min as upstream does.
        let dbl_min = Number::MIN_POSITIVE;
        let mut c_row_max: Vec<Number> = vec![dbl_min; n_c];
        let mut d_row_max: Vec<Number> = vec![dbl_min; n_d];
        for k in 0..self.nnz_jac_g_full as usize {
            let g_row_0 = (full_irow[k] - style_offset) as usize;
            let v = full_jac_vals[k].abs();
            let cr = g_to_c[g_row_0];
            if cr >= 0 {
                let row = cr as usize;
                if v > c_row_max[row] {
                    c_row_max[row] = v;
                }
            } else {
                let dr = g_to_d[g_row_0];
                if dr >= 0 {
                    let row = dr as usize;
                    if v > d_row_max[row] {
                        d_row_max[row] = v;
                    }
                }
            }
        }

        let row_max_to_scale = |row_max: Number| -> Number {
            // With `constr_target_gradient` > 0 the user is asking for
            // a *fixed* gradient ∞-norm per row (overrides the cutoff
            // and the 1.0 clamp). Otherwise: scale only rows that
            // exceed the cutoff, never amplify (clamp at 1).
            let mut s = if constr_target_gradient > 0.0 {
                constr_target_gradient / row_max
            } else {
                let raw = max_gradient / row_max;
                if raw > 1.0 { 1.0 } else { raw }
            };
            if s < min_value {
                s = min_value;
            }
            s
        };
        let any_row_above = |rows: &[Number]| -> bool {
            constr_target_gradient > 0.0 || rows.iter().any(|&v| v > max_gradient)
        };

        if n_c > 0 && any_row_above(&c_row_max) {
            let dc: Vec<Number> = c_row_max.iter().map(|&v| row_max_to_scale(v)).collect();
            *self.c_scale.borrow_mut() = Some(dc);
        } else {
            *self.c_scale.borrow_mut() = None;
        }

        if n_d > 0 && any_row_above(&d_row_max) {
            let dd: Vec<Number> = d_row_max.iter().map(|&v| row_max_to_scale(v)).collect();
            *self.d_scale.borrow_mut() = Some(dd);
        } else {
            *self.d_scale.borrow_mut() = None;
        }
    }

    /// User-supplied scaling pathway: call `TNLP::get_scaling_parameters`
    /// and translate the user's `obj_scaling` and `g_scaling` arrays
    /// into the algorithm-side `obj_scale_factor`, `c_scale`, `d_scale`.
    /// Returns `true` if the TNLP supplied scaling (matches upstream's
    /// `GetScalingParameters` return-value contract).
    ///
    /// The `x_scaling` request channel is ignored: `OrigIpoptNlp` does
    /// not currently model per-variable rescaling (would require
    /// transforming `eval_grad_f`, `eval_jac_*`, and `eval_h` in
    /// concert), and issue #61's `nlp_scaling=user` design explicitly
    /// covers only `obj_scale` and `con_scale`.
    fn scale_user_supplied(
        &self,
        cls: &BoundClassification,
        user_obj_factor: Number,
        min_value: Number,
    ) -> bool {
        let n_full_x = cls.n_full_x as usize;
        let n_full_g = cls.n_full_g as usize;
        let mut obj_scaling: Number = 1.0;
        let mut use_x_scaling = false;
        let mut x_scaling = vec![1.0; n_full_x];
        let mut use_g_scaling = false;
        let mut g_scaling = vec![1.0; n_full_g];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.get_scaling_parameters(ScalingRequest {
                obj_scaling: &mut obj_scaling,
                use_x_scaling: &mut use_x_scaling,
                x_scaling: &mut x_scaling,
                use_g_scaling: &mut use_g_scaling,
                g_scaling: &mut g_scaling,
            })
        };
        if !ok {
            return false;
        }

        // Objective: user's obj_scaling combined with the constant
        // `obj_scaling_factor` (matches upstream's
        // `StandardScalingBase::DetermineScaling`).
        let mut df = obj_scaling;
        if df.abs() < min_value {
            // Defensively floor — a zero/near-zero obj scale would
            // make all duals divide-by-zero on the way out.
            df = df.signum().max(0.0).max(1.0) * min_value;
        }
        self.obj_scale_factor.set(df * user_obj_factor);

        // Constraint vector: split user g_scaling into c_scale / d_scale.
        if use_g_scaling && g_scaling.len() == n_full_g {
            let n_c = cls.n_c as usize;
            let n_d = cls.n_d as usize;
            let mut dc = vec![1.0; n_c];
            for (c_idx, &g_idx) in cls.c_map.iter().enumerate() {
                let s = g_scaling[g_idx as usize];
                dc[c_idx] = if s < min_value { min_value } else { s };
            }
            let mut dd = vec![1.0; n_d];
            for (d_idx, &g_idx) in cls.d_map.iter().enumerate() {
                let s = g_scaling[g_idx as usize];
                dd[d_idx] = if s < min_value { min_value } else { s };
            }
            // Only install the vectors when not all-ones (matches the
            // `Option::None ↔ identity` convention used elsewhere).
            let nontrivial_c = dc.iter().any(|&s| s != 1.0);
            *self.c_scale.borrow_mut() = if nontrivial_c && n_c > 0 {
                Some(dc)
            } else {
                None
            };
            let nontrivial_d = dd.iter().any(|&s| s != 1.0);
            *self.d_scale.borrow_mut() = if nontrivial_d && n_d > 0 {
                Some(dd)
            } else {
                None
            };
        } else {
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
        }
        // `use_x_scaling`: silently ignored (not modeled — see doc).
        let _ = use_x_scaling;
        true
    }

    /// Bring `d_l` / `d_u` into the scaled space so feasibility checks
    /// compare like with like (gh#54). Upstream's
    /// `OrigIpoptNLP::Initialize` does this via
    /// `Pd_L_->TransMultVector(scaling.apply_vec_d(...))`.
    fn apply_d_scale_to_bounds(&mut self) {
        let cls = self.adapter.borrow().classification().clone();
        if let Some(dd) = self.d_scale.borrow().as_ref() {
            if let Some(d_l) = Rc::get_mut(&mut self.d_l) {
                let xs = d_l.values_mut();
                for (i, slot) in xs.iter_mut().enumerate() {
                    let d_idx = cls.d_l_map[i] as usize;
                    *slot *= dd[d_idx];
                }
            }
            if let Some(d_u) = Rc::get_mut(&mut self.d_u) {
                let xs = d_u.values_mut();
                for (i, slot) in xs.iter_mut().enumerate() {
                    let d_idx = cls.d_u_map[i] as usize;
                    *slot *= dd[d_idx];
                }
            }
        }
    }

    fn invalidate_eval_caches(&self) {
        self.f_cache.borrow_mut().clear();
        self.grad_f_cache.borrow_mut().clear();
        self.c_cache.borrow_mut().clear();
        self.d_cache.borrow_mut().clear();
        self.jac_c_cache.borrow_mut().clear();
        self.jac_d_cache.borrow_mut().clear();
        self.h_cache.borrow_mut().clear();
    }

    pub fn f_evals(&self) -> Index {
        *self.f_evals.borrow()
    }
    pub fn grad_f_evals(&self) -> Index {
        *self.grad_f_evals.borrow()
    }
    pub fn c_evals(&self) -> Index {
        *self.c_evals.borrow()
    }
    pub fn d_evals(&self) -> Index {
        *self.d_evals.borrow()
    }
    pub fn jac_c_evals(&self) -> Index {
        *self.jac_c_evals.borrow()
    }
    pub fn jac_d_evals(&self) -> Index {
        *self.jac_d_evals.borrow()
    }
    pub fn h_evals(&self) -> Index {
        *self.h_evals.borrow()
    }

    /// Lift a compressed `x_var` (length `n_x_var`) up to the full TNLP
    /// `x` (length `n_full_x`), inserting `x_fixed_vals` at the
    /// `x_fixed_map` positions. Mirrors upstream
    /// `IpTNLPAdapter::ResortX` under `fixed_variable_treatment =
    /// make_parameter`.
    pub fn lift_x_to_full(&self, x: &dyn Vector) -> Vec<Number> {
        let Some(dx) = x.as_any().downcast_ref::<DenseVector>() else {
            panic!("OrigIpoptNlp expects DenseVector for x");
        };
        let a = self.adapter.borrow();
        let cls = a.classification();
        let mut full = vec![0.0; cls.n_full_x as usize];
        let vals = dx.expanded_values();
        for (var_idx, &full_idx) in cls.x_not_fixed_map.iter().enumerate() {
            full[full_idx as usize] = vals[var_idx];
        }
        for (i, &full_idx) in cls.x_fixed_map.iter().enumerate() {
            full[full_idx as usize] = cls.x_fixed_vals[i];
        }
        full
    }

    /// Lift the algorithm-side `(y_c, y_d)` multipliers to the user
    /// TNLP's `lambda` array (length `m_full = n_c + n_d`, indexed
    /// by original constraint-row order). Result matches the user's
    /// **unscaled-Lagrangian** convention `min f + λ·g(x)` —
    /// i.e. without the obj_factor that the algorithm threads through
    /// `eval_h`. Mirror of upstream
    /// `IpOrigIpoptNLP::FinalizeSolution`'s `mult_g` packing
    /// (`lambda_user = c_scale * y_c / obj_scale_factor`,
    /// `mu_user = d_scale * y_d / obj_scale_factor`). Used by
    /// `application.rs::finalize_via_orig_nlp` to populate the
    /// `Solution.lambda` slot — pounce#11.
    pub fn finalize_solution_lambda(&self, y_c: &dyn Vector, y_d: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let mut lambda = self.pack_lambda_for_user(y_c, y_d, &cls);
        let obj_scal = self.obj_scale_factor.get();
        if obj_scal != 0.0 && obj_scal != 1.0 {
            let inv = 1.0 / obj_scal;
            for v in lambda.iter_mut() {
                *v *= inv;
            }
        }
        lambda
    }

    /// Lift the algorithm-side compressed `z_l` (length `n_x_l`,
    /// indexed via `x_l_map`) to the user's full-x bound multiplier
    /// array (length `n_full_x`). Slots without a finite lower bound
    /// — including fixed variables — are reported as `0.0`. Sign and
    /// scale match upstream Ipopt: `z_l ≥ 0` for active lower
    /// bounds, divided by `obj_scale_factor` so the user sees the
    /// unscaled-Lagrangian dual.
    pub fn finalize_solution_z_l(&self, z_l: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let n_full_x = cls.n_full_x as usize;
        let mut full_z_l = vec![0.0; n_full_x];
        let n_x_l = self.x_l.dim() as usize;
        if n_x_l == 0 {
            return full_z_l;
        }
        let Some(dz) = z_l.as_any().downcast_ref::<DenseVector>() else {
            panic!("OrigIpoptNlp::finalize_solution_z_l expects DenseVector");
        };
        let vals = dz.expanded_values();
        let obj_scal = self.obj_scale_factor.get();
        let inv = if obj_scal == 0.0 { 1.0 } else { 1.0 / obj_scal };
        for i in 0..n_x_l {
            let var_idx = cls.x_l_map[i] as usize;
            let full_idx = cls.x_not_fixed_map[var_idx] as usize;
            full_z_l[full_idx] = vals[i] * inv;
        }
        full_z_l
    }

    /// Mirror of [`Self::finalize_solution_z_l`] for the upper-bound
    /// duals. Indexed via `x_u_map`.
    pub fn finalize_solution_z_u(&self, z_u: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let n_full_x = cls.n_full_x as usize;
        let mut full_z_u = vec![0.0; n_full_x];
        let n_x_u = self.x_u.dim() as usize;
        if n_x_u == 0 {
            return full_z_u;
        }
        let Some(dz) = z_u.as_any().downcast_ref::<DenseVector>() else {
            panic!("OrigIpoptNlp::finalize_solution_z_u expects DenseVector");
        };
        let vals = dz.expanded_values();
        let obj_scal = self.obj_scale_factor.get();
        let inv = if obj_scal == 0.0 { 1.0 } else { 1.0 / obj_scal };
        for i in 0..n_x_u {
            let var_idx = cls.x_u_map[i] as usize;
            let full_idx = cls.x_not_fixed_map[var_idx] as usize;
            full_z_u[full_idx] = vals[i] * inv;
        }
        full_z_u
    }

    /// Clone the user-provided multipliers (already in the
    /// algorithm's eq/ineq-split form) into a single `lambda` array of
    /// length `m_full = n_c + n_d` ordered by original g-index. Used
    /// by `eval_h` and `finalize_solution`.
    /// Pack the algorithm-side `(y_c, y_d)` multipliers into the user
    /// TNLP's `lambda` array (full-g indexed), applying c/d scale
    /// factors so the result is in the user's unscaled-constraint
    /// multiplier space (`lambda_user_i = c_scale_i * y_c_i`). Used
    /// when invoking the user's `eval_h`.
    pub fn pack_lambda_for_user(
        &self,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
        cls: &BoundClassification,
    ) -> Vec<Number> {
        let mut lambda = vec![0.0; cls.n_full_g as usize];
        if cls.n_c > 0 {
            let Some(dy) = y_c.as_any().downcast_ref::<DenseVector>() else {
                panic!("OrigIpoptNlp expects DenseVector for y_c");
            };
            let vals = dy.expanded_values();
            let cs = self.c_scale.borrow();
            for (i, &g_idx) in cls.c_map.iter().enumerate() {
                lambda[g_idx as usize] = match cs.as_ref() {
                    Some(v) => vals[i] * v[i],
                    None => vals[i],
                };
            }
        }
        if cls.n_d > 0 {
            let Some(dy) = y_d.as_any().downcast_ref::<DenseVector>() else {
                panic!("OrigIpoptNlp expects DenseVector for y_d");
            };
            let vals = dy.expanded_values();
            let ds = self.d_scale.borrow();
            for (i, &g_idx) in cls.d_map.iter().enumerate() {
                lambda[g_idx as usize] = match ds.as_ref() {
                    Some(v) => vals[i] * v[i],
                    None => vals[i],
                };
            }
        }
        lambda
    }

    // -------------------- Initialization --------------------

    /// Fill the algorithm's iterate slots with the TNLP's starting
    /// point. Mirrors the second half of upstream
    /// `InitializeStructures`. The caller passes already-allocated
    /// `DenseVector`s in the right spaces; we set them in place.
    ///
    /// Returns the four `init_*` flags so the caller can decide
    /// whether to overwrite zeros with the user's guess.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_starting_point(
        &mut self,
        x: &mut DenseVector,
        init_x: bool,
        y_c: &mut DenseVector,
        init_y_c: bool,
        y_d: &mut DenseVector,
        init_y_d: bool,
        z_l: &mut DenseVector,
        init_z_l: bool,
        z_u: &mut DenseVector,
        init_z_u: bool,
    ) -> bool {
        let n_full_x = self.adapter.borrow().classification().n_full_x as usize;
        let n_full_g = self.adapter.borrow().classification().n_full_g as usize;
        let n_x_l = self.x_l.dim() as usize;
        let n_x_u = self.x_u.dim() as usize;

        let mut full_x = vec![0.0; n_full_x];
        let mut full_z_l = vec![0.0; n_full_x];
        let mut full_z_u = vec![0.0; n_full_x];
        let mut full_lambda = vec![0.0; n_full_g];

        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.get_starting_point(StartingPoint {
                init_x,
                x: &mut full_x,
                init_z: init_z_l || init_z_u,
                z_l: &mut full_z_l,
                z_u: &mut full_z_u,
                init_lambda: init_y_c || init_y_d,
                lambda: &mut full_lambda,
            })
        };
        if !ok {
            return false;
        }

        let cls = self.adapter.borrow().classification().clone();
        let obj_scal = self.obj_scale_factor.get();
        let c_scale = self.c_scale.borrow();
        let d_scale = self.d_scale.borrow();

        // Compress full_x → x.
        if init_x {
            let xs = x.values_mut();
            for (var_idx, &full_idx) in cls.x_not_fixed_map.iter().enumerate() {
                xs[var_idx] = full_x[full_idx as usize];
            }
        }
        // Compress full_lambda → y_c, y_d. Upstream
        // (`IpOrigIpoptNLP.cpp:407-429`) divides the user multiplier
        // by the constraint scale (`unapply_vector_scaling_*`) and
        // multiplies by obj_scal so that the algorithm-side y_c sees
        // `(obj_scal / c_scale) * lambda_user`.
        if init_y_c && cls.n_c > 0 {
            let yc = y_c.values_mut();
            for (i, &g_idx) in cls.c_map.iter().enumerate() {
                let cs = c_scale.as_ref().map(|v| v[i]).unwrap_or(1.0);
                yc[i] = full_lambda[g_idx as usize] / cs * obj_scal;
            }
        }
        if init_y_d && cls.n_d > 0 {
            let yd = y_d.values_mut();
            for (i, &g_idx) in cls.d_map.iter().enumerate() {
                let ds = d_scale.as_ref().map(|v| v[i]).unwrap_or(1.0);
                yd[i] = full_lambda[g_idx as usize] / ds * obj_scal;
            }
        }
        // Compress full_z_l, full_z_u → z_l, z_u, indexed via x_l_map / x_u_map.
        if init_z_l && n_x_l > 0 {
            let zl = z_l.values_mut();
            for (i, slot) in zl.iter_mut().enumerate().take(n_x_l) {
                let var_idx = cls.x_l_map[i] as usize;
                let full_idx = cls.x_not_fixed_map[var_idx] as usize;
                *slot = full_z_l[full_idx] * obj_scal;
            }
        }
        if init_z_u && n_x_u > 0 {
            let zu = z_u.values_mut();
            for (i, slot) in zu.iter_mut().enumerate().take(n_x_u) {
                let var_idx = cls.x_u_map[i] as usize;
                let full_idx = cls.x_not_fixed_map[var_idx] as usize;
                *slot = full_z_u[full_idx] * obj_scal;
            }
        }
        true
    }

    // -------------------- Internal eval helpers --------------------

    fn eval_f_internal(&self, x: &dyn Vector) -> Number {
        if let Some(v) = self.f_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        *self.f_evals.borrow_mut() += 1;
        let full_x = self.lift_x_to_full(x);
        let unscaled = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            // A failed user eval (domain error, e.g. log of a negative) is
            // upstream Ipopt's `Eval_Error`. Return NaN so the line search's
            // non-finite-trial path backtracks the step, rather than aborting
            // — and a panic cannot unwind across the C FFI boundary anyway.
            t.eval_f(&full_x, true).unwrap_or(f64::NAN)
        };
        let scaled = unscaled * self.obj_scale_factor.get();
        self.f_cache.borrow_mut().add_1dep(scaled, x.as_tagged());
        scaled
    }

    fn eval_grad_f_internal(&self, x: &dyn Vector) -> Rc<dyn Vector> {
        if let Some(v) = self.grad_f_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        *self.grad_f_evals.borrow_mut() += 1;
        let full_x = self.lift_x_to_full(x);
        let mut full_g = vec![0.0; full_x.len()];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_grad_f(&full_x, true, &mut full_g)
        };
        // Eval failure → NaN-filled gradient, which propagates a non-finite
        // step the line search rejects (see `eval_f_internal`).
        if !ok {
            full_g.fill(f64::NAN);
        }
        // Compress full_g → grad in x_var-space, scale by obj_scal.
        let cls = self.adapter.borrow().classification().clone();
        let mut g_compressed = self.x_space.make_new_dense();
        let obj_scal = self.obj_scale_factor.get();
        {
            let gv = g_compressed.values_mut();
            for (var_idx, &full_idx) in cls.x_not_fixed_map.iter().enumerate() {
                gv[var_idx] = full_g[full_idx as usize] * obj_scal;
            }
        }
        let result: Rc<dyn Vector> = Rc::new(g_compressed);
        self.grad_f_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    /// Full-space constraint vector `g(x)` (length `n_full_g`), shared by
    /// `eval_c`/`eval_d` so the user `eval_g` runs once per iterate. On a
    /// cache hit no user evaluation occurs; on a failed eval the buffer is
    /// filled with NaN (so `theta_trial` goes non-finite and the line
    /// search backtracks), matching the per-subsystem paths.
    fn full_g(&self, x: &dyn Vector) -> Rc<Vec<Number>> {
        if let Some(v) = self.full_g_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        let n_full_g = self.adapter.borrow().classification().n_full_g as usize;
        let full_x = self.lift_x_to_full(x);
        let mut full_g = vec![0.0; n_full_g];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_g(&full_x, true, &mut full_g)
        };
        if !ok {
            full_g.fill(f64::NAN);
        }
        let result = Rc::new(full_g);
        self.full_g_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    /// Full-space Jacobian values (length `nnz_jac_g_full`, in the user's
    /// `eval_jac_g` order), shared by `eval_jac_c`/`eval_jac_d` so the user
    /// `eval_jac_g` runs once per iterate. Same NaN-on-failure contract as
    /// [`Self::full_g`].
    fn full_jac_g(&self, x: &dyn Vector) -> Rc<Vec<Number>> {
        if let Some(v) = self.full_jac_g_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        let mut full_vals = vec![0.0; self.nnz_jac_g_full as usize];
        let full_x = self.lift_x_to_full(x);
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_jac_g(
                Some(&full_x),
                true,
                SparsityRequest::Values {
                    values: &mut full_vals,
                },
            )
        };
        if !ok {
            full_vals.fill(f64::NAN);
        }
        let result = Rc::new(full_vals);
        self.full_jac_g_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    fn eval_c_internal(&self, x: &dyn Vector) -> Rc<dyn Vector> {
        let cls = self.adapter.borrow().classification().clone();
        if cls.n_c == 0 {
            // Empty constraint vector — still cache so the tag is stable.
            if let Some(v) = self.c_cache.borrow().get_1dep(x.as_tagged()) {
                return v;
            }
            let v = self.c_space.make_new_dense();
            let result: Rc<dyn Vector> = Rc::new(v);
            self.c_cache
                .borrow_mut()
                .add_1dep(Rc::clone(&result), x.as_tagged());
            return result;
        }
        if let Some(v) = self.c_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        *self.c_evals.borrow_mut() += 1;
        // Shared full-space `g(x)` — computed once per iterate and reused
        // by `eval_d` (and vice versa). NaN-on-failure handled in `full_g`,
        // so `theta_trial` goes non-finite and the line search backtracks
        // (see `eval_f_internal`).
        let full_g = self.full_g(x);
        let mut c = self.c_space.make_new_dense();
        // c_i = g(g_idx) - c_rhs[i]  (since g_l == g_u for equalities,
        // upstream subtracts the bound to make it a residual). Matches
        // `OrigIpoptNLP::c` which calls `nlp_->Eval_c` after the adapter
        // subtracted the bound — TNLPAdapter doesn't subtract yet, so we
        // do it here. The RHS is the constant `g_l[g_idx]`, captured once
        // at construction (`self.c_rhs`, M17) — no per-iterate bounds
        // fetch or full-size scratch allocations in the line-search path.
        {
            let cv = c.values_mut();
            let cs = self.c_scale.borrow();
            for (i, &g_idx) in cls.c_map.iter().enumerate() {
                let raw = full_g[g_idx as usize] - self.c_rhs[i];
                cv[i] = match cs.as_ref() {
                    Some(v) => raw * v[i],
                    None => raw,
                };
            }
        }
        let result: Rc<dyn Vector> = Rc::new(c);
        self.c_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    fn eval_d_internal(&self, x: &dyn Vector) -> Rc<dyn Vector> {
        let cls = self.adapter.borrow().classification().clone();
        if cls.n_d == 0 {
            if let Some(v) = self.d_cache.borrow().get_1dep(x.as_tagged()) {
                return v;
            }
            let v = self.d_space.make_new_dense();
            let result: Rc<dyn Vector> = Rc::new(v);
            self.d_cache
                .borrow_mut()
                .add_1dep(Rc::clone(&result), x.as_tagged());
            return result;
        }
        if let Some(v) = self.d_cache.borrow().get_1dep(x.as_tagged()) {
            return v;
        }
        *self.d_evals.borrow_mut() += 1;
        // Shared full-space `g(x)` — reused with `eval_c` (see `full_g`).
        let full_g = self.full_g(x);
        let mut d = self.d_space.make_new_dense();
        {
            let dv = d.values_mut();
            let ds = self.d_scale.borrow();
            for (i, &g_idx) in cls.d_map.iter().enumerate() {
                let raw = full_g[g_idx as usize];
                dv[i] = match ds.as_ref() {
                    Some(v) => raw * v[i],
                    None => raw,
                };
            }
        }
        let result: Rc<dyn Vector> = Rc::new(d);
        self.d_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    fn eval_jac_c_internal(&self, x: &dyn Vector) -> Rc<dyn Matrix> {
        if let Some(m) = self.jac_c_cache.borrow().get_1dep(x.as_tagged()) {
            return m;
        }
        *self.jac_c_evals.borrow_mut() += 1;
        // Shared full-space Jacobian — computed once per iterate and reused
        // by `eval_jac_d` (and vice versa). NaN-on-failure handled in
        // `full_jac_g`.
        let full_vals = self.full_jac_g(x);
        let mut jac_c = GenTMatrix::new(Rc::clone(&self.jac_c_space));
        {
            let cs = self.c_scale.borrow();
            let irows = self.jac_c_space.irows().to_vec();
            let vs = jac_c.values_mut();
            for (k, &src) in self.jac_c_entry_in_g.iter().enumerate() {
                let raw = full_vals[src as usize];
                vs[k] = match cs.as_ref() {
                    // irows are 1-based.
                    Some(v) => raw * v[(irows[k] - 1) as usize],
                    None => raw,
                };
            }
        }
        let result: Rc<dyn Matrix> = Rc::new(jac_c);
        self.jac_c_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    fn eval_jac_d_internal(&self, x: &dyn Vector) -> Rc<dyn Matrix> {
        if let Some(m) = self.jac_d_cache.borrow().get_1dep(x.as_tagged()) {
            return m;
        }
        *self.jac_d_evals.borrow_mut() += 1;
        // Shared full-space Jacobian — reused with `eval_jac_c` (see
        // `full_jac_g`).
        let full_vals = self.full_jac_g(x);
        let mut jac_d = GenTMatrix::new(Rc::clone(&self.jac_d_space));
        {
            let ds = self.d_scale.borrow();
            let irows = self.jac_d_space.irows().to_vec();
            let vs = jac_d.values_mut();
            for (k, &src) in self.jac_d_entry_in_g.iter().enumerate() {
                let raw = full_vals[src as usize];
                vs[k] = match ds.as_ref() {
                    Some(v) => raw * v[(irows[k] - 1) as usize],
                    None => raw,
                };
            }
        }
        let result: Rc<dyn Matrix> = Rc::new(jac_d);
        self.jac_d_cache
            .borrow_mut()
            .add_1dep(Rc::clone(&result), x.as_tagged());
        result
    }

    fn eval_h_internal(
        &self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        // h_cache key: (x, y_c, y_d) tags + obj_factor scalar dep, as
        // upstream `IpOrigIpoptNLP.cpp:786`.
        if let Some(m) = self.h_cache.borrow().get(
            &[x.as_tagged(), y_c.as_tagged(), y_d.as_tagged()],
            &[obj_factor],
        ) {
            return m;
        }
        *self.h_evals.borrow_mut() += 1;
        let Some(h_space) = self.h_space.as_ref() else {
            panic!(
                "OrigIpoptNlp::eval_h called but the TNLP did not provide \
                 eval_h sparsity. The L-BFGS path lands in Phase 8."
            );
        };
        let cls = self.adapter.borrow().classification().clone();
        let full_x = self.lift_x_to_full(x);
        // Upstream `IpOrigIpoptNLP.cpp:792-794` passes the user TNLP's
        // `eval_h` the multipliers in the user's unscaled-constraint
        // space, i.e. `lambda_user = c_scale * y_c` (and same for d).
        // The obj_factor is also scaled (`scaled_obj_factor = obj_scale
        // * obj_factor`). Together this gives the user-space Hessian
        // contribution that's already in the algorithm's scaled space
        // (no extra Hessian-side scaling because we don't scale x).
        let full_lambda = self.pack_lambda_for_user(y_c, y_d, &cls);
        let scaled_obj_factor = obj_factor * self.obj_scale_factor.get();

        // The user TNLP writes `nnz_h_lag_full` values; the kept
        // (var-x ⊗ var-x) subset has `h_space.nonzeros()` entries
        // selected via `h_entry_in_full`. They differ when fixed
        // variables drop entries.
        let mut full_vals = vec![0.0; self.nnz_h_lag_full as usize];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_h(
                Some(&full_x),
                true,
                scaled_obj_factor,
                Some(&full_lambda),
                true,
                SparsityRequest::Values {
                    values: &mut full_vals,
                },
            )
        };
        if !ok {
            full_vals.fill(f64::NAN);
        }
        let mut h = SymTMatrix::new(Rc::clone(h_space));
        let kept = h_space.nonzeros() as usize;
        let h_vals = h.values_mut();
        // `h_entry_in_full` always has length `kept` (identity when no
        // fixed-var filtering, sparse selection otherwise).
        debug_assert_eq!(kept, self.h_entry_in_full.len());
        for (k, &src) in self.h_entry_in_full.iter().enumerate() {
            h_vals[k] = full_vals[src as usize];
        }
        let result: Rc<dyn SymMatrix> = Rc::new(h);
        self.h_cache.borrow_mut().add(
            Rc::clone(&result),
            &[x.as_tagged(), y_c.as_tagged(), y_d.as_tagged()],
            &[obj_factor],
        );
        result
    }
}

// ---- helpers ----

fn make_dense_from(
    space: &Rc<DenseVectorSpace>,
    mut f: impl FnMut(usize) -> Number,
) -> DenseVector {
    let mut v = space.make_new_dense();
    let dim = space.dim() as usize;
    if dim > 0 {
        let vs = v.values_mut();
        for (i, slot) in vs.iter_mut().enumerate().take(dim) {
            *slot = f(i);
        }
    }
    v
}

// -------------------- Trait impls --------------------

impl Nlp for OrigIpoptNlp {
    fn n(&self) -> Index {
        self.x_space.dim()
    }
    fn m_eq(&self) -> Index {
        self.c_space.dim()
    }
    fn m_ineq(&self) -> Index {
        self.d_space.dim()
    }

    fn eval_f(&mut self, x: &dyn Vector) -> Number {
        self.timed_eval(|t| &t.eval_obj, || self.eval_f_internal(x))
    }
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
        let result = self.timed_eval(|t| &t.eval_grad_obj, || self.eval_grad_f_internal(x));
        g.copy(&*result);
    }
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
        let result = self.timed_eval(|t| &t.eval_constr, || self.eval_c_internal(x));
        c.copy(&*result);
    }
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
        let result = self.timed_eval(|t| &t.eval_constr, || self.eval_d_internal(x));
        d.copy(&*result);
    }
    fn eval_jac_c(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        self.timed_eval(|t| &t.eval_constr_jac, || self.eval_jac_c_internal(x))
    }
    fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        self.timed_eval(|t| &t.eval_constr_jac, || self.eval_jac_d_internal(x))
    }
    fn eval_h(
        &mut self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        self.timed_eval(
            |t| &t.eval_lag_hess,
            || self.eval_h_internal(x, obj_factor, y_c, y_d),
        )
    }
}

impl IpoptNlp for OrigIpoptNlp {
    fn x_l(&self) -> &dyn Vector {
        &*self.x_l
    }
    fn x_u(&self) -> &dyn Vector {
        &*self.x_u
    }
    fn d_l(&self) -> &dyn Vector {
        &*self.d_l
    }
    fn d_u(&self) -> &dyn Vector {
        &*self.d_u
    }
    fn px_l(&self) -> Rc<dyn Matrix> {
        Rc::clone(&self.px_l)
    }
    fn px_u(&self) -> Rc<dyn Matrix> {
        Rc::clone(&self.px_u)
    }
    fn pd_l(&self) -> Rc<dyn Matrix> {
        Rc::clone(&self.pd_l)
    }
    fn pd_u(&self) -> Rc<dyn Matrix> {
        Rc::clone(&self.pd_u)
    }

    /// Install moved bounds from the safe-slack mechanism. Mirrors
    /// `OrigIpoptNLP::AdjustVariableBounds` (`IpOrigIpoptNLP.cpp:990`):
    /// upstream simply swaps in the new bound vectors. We copy the values
    /// into the existing `Rc<DenseVector>` storage (falling back to a
    /// fresh allocation if the bound is somehow shared), which leaves the
    /// `Px_* / Pd_*` expansion matrices — keyed on the bound *spaces*,
    /// not values — untouched.
    fn adjust_variable_bounds(
        &mut self,
        new_x_l: &dyn Vector,
        new_x_u: &dyn Vector,
        new_d_l: &dyn Vector,
        new_d_u: &dyn Vector,
    ) {
        // The bound `Rc`s are uniquely owned (nothing clones them — same
        // invariant `relax_bounds` relies on), so `get_mut` always
        // succeeds and we copy the moved values into the existing storage.
        fn install(slot: &mut Rc<DenseVector>, new: &dyn Vector) {
            Rc::get_mut(slot)
                .expect("adjust_variable_bounds: bound vector is uniquely owned")
                .copy(new);
        }
        install(&mut self.x_l, new_x_l);
        install(&mut self.x_u, new_x_u);
        install(&mut self.d_l, new_d_l);
        install(&mut self.d_u, new_d_u);
    }

    fn obj_scaling_factor(&self) -> Number {
        self.obj_scale_factor.get()
    }

    fn c_scale_vec(&self) -> Option<Vec<Number>> {
        self.c_scale.borrow().clone()
    }

    fn d_scale_vec(&self) -> Option<Vec<Number>> {
        self.d_scale.borrow().clone()
    }

    /// Project the underlying TNLP's `idx_names` metadata into the
    /// algorithm's split space. Variable names are gathered through the
    /// fixed-variable map (`x_not_fixed_map`), equality names through the
    /// c-block map (`c_map`), and inequality names through the d-block map
    /// (`d_map`) — exactly the permutations the adapter applied when it
    /// split the problem, so a residual at split index `k` is labeled with
    /// the equation the user actually wrote.
    ///
    /// Returns `None` when the TNLP exposes no names (e.g. presolve, which
    /// renumbers rows, declines `get_var_con_metadata`) so callers fall
    /// back to index labels rather than mislabeling permuted rows. This is
    /// the seam that turns "row 3" into `mass_balance` per Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>).
    fn split_space_names(&self) -> Option<SplitNames> {
        let a = self.adapter.borrow();
        let cls = a.classification();

        let mut var_meta = MetaData::default();
        let mut con_meta = MetaData::default();
        if !a
            .tnlp()
            .borrow_mut()
            .get_var_con_metadata(&mut var_meta, &mut con_meta)
        {
            return None;
        }

        // Full-space (original TNLP order) name pools. Either may be
        // absent — a model can name variables but not constraints, etc.
        let var_full = var_meta.strings.get(IDX_NAMES);
        let con_full = con_meta.strings.get(IDX_NAMES);
        if var_full.is_none() && con_full.is_none() {
            return None;
        }

        // Look a full-space name up safely; `None` for out-of-range or
        // empty entries so we degrade to an index label per slot.
        let pick = |pool: Option<&Vec<String>>, full_idx: Index| -> Option<String> {
            pool.and_then(|v| v.get(full_idx as usize))
                .filter(|s| !s.is_empty())
                .cloned()
        };

        let x_var = cls
            .x_not_fixed_map
            .iter()
            .map(|&full_idx| pick(var_full, full_idx))
            .collect();
        let eq = cls
            .c_map
            .iter()
            .map(|&full_idx| pick(con_full, full_idx))
            .collect();
        let ineq = cls
            .d_map
            .iter()
            .map(|&full_idx| pick(con_full, full_idx))
            .collect();

        let names = SplitNames { x_var, eq, ineq };
        names.any_present().then_some(names)
    }

    /// Populate `x` (length `n_x_var`) from the TNLP's starting point,
    /// compressed via `x_not_fixed_map`. Mirrors the `init_x` arm of
    /// upstream `IpOrigIpoptNLP::GetStartingPoint`.
    fn get_starting_x(&mut self, x: &mut dyn Vector) -> bool {
        let cls = self.adapter.borrow().classification().clone();
        let n_full_x = cls.n_full_x as usize;
        let n_full_g = cls.n_full_g as usize;
        let mut full_x = vec![0.0; n_full_x];
        let mut full_z_l = vec![0.0; n_full_x];
        let mut full_z_u = vec![0.0; n_full_x];
        let mut full_lambda = vec![0.0; n_full_g];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.get_starting_point(StartingPoint {
                init_x: true,
                x: &mut full_x,
                init_z: false,
                z_l: &mut full_z_l,
                z_u: &mut full_z_u,
                init_lambda: false,
                lambda: &mut full_lambda,
            })
        };
        if !ok {
            return false;
        }
        let Some(dx) = x.as_any_mut().downcast_mut::<DenseVector>() else {
            return false;
        };
        let xs = dx.values_mut();
        for (var_idx, &full_idx) in cls.x_not_fixed_map.iter().enumerate() {
            xs[var_idx] = full_x[full_idx as usize];
        }
        true
    }

    fn lift_x_to_full(&self, x: &dyn Vector) -> Vec<Number> {
        OrigIpoptNlp::lift_x_to_full(self, x)
    }

    fn n_full_x(&self) -> Index {
        self.adapter.borrow().classification().n_full_x
    }

    fn n_full_g(&self) -> Index {
        self.adapter.borrow().classification().n_full_g
    }

    fn pack_lambda_for_user(&self, y_c: &dyn Vector, y_d: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        OrigIpoptNlp::pack_lambda_for_user(self, y_c, y_d, &cls)
    }

    fn pack_g_for_user(&self, c: &dyn Vector, d: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let mut g = vec![0.0; cls.n_full_g as usize];
        if cls.n_c > 0 {
            let Some(dc) = c.as_any().downcast_ref::<DenseVector>() else {
                panic!("OrigIpoptNlp expects DenseVector for c");
            };
            let cs = self.c_scale.borrow();
            for (i, &g_idx) in cls.c_map.iter().enumerate() {
                let v = dc.expanded_values()[i];
                g[g_idx as usize] = match cs.as_ref() {
                    Some(s) => v / s[i],
                    None => v,
                };
            }
        }
        if cls.n_d > 0 {
            let Some(dd) = d.as_any().downcast_ref::<DenseVector>() else {
                panic!("OrigIpoptNlp expects DenseVector for d");
            };
            let ds = self.d_scale.borrow();
            for (i, &g_idx) in cls.d_map.iter().enumerate() {
                let v = dd.expanded_values()[i];
                g[g_idx as usize] = match ds.as_ref() {
                    Some(s) => v / s[i],
                    None => v,
                };
            }
        }
        g
    }

    fn pack_z_l_for_user(&self, z_l: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let mut full = vec![0.0; cls.n_full_x as usize];
        if z_l.dim() == 0 {
            return full;
        }
        let Some(dz) = z_l.as_any().downcast_ref::<DenseVector>() else {
            panic!("OrigIpoptNlp expects DenseVector for z_l");
        };
        let vals = dz.expanded_values();
        for (k, &var_idx) in cls.x_l_map.iter().enumerate() {
            let full_idx = cls.x_not_fixed_map[var_idx as usize] as usize;
            full[full_idx] = vals[k];
        }
        full
    }

    fn pack_z_u_for_user(&self, z_u: &dyn Vector) -> Vec<Number> {
        let cls = self.adapter.borrow().classification().clone();
        let mut full = vec![0.0; cls.n_full_x as usize];
        if z_u.dim() == 0 {
            return full;
        }
        let Some(dz) = z_u.as_any().downcast_ref::<DenseVector>() else {
            panic!("OrigIpoptNlp expects DenseVector for z_u");
        };
        let vals = dz.expanded_values();
        for (k, &var_idx) in cls.x_u_map.iter().enumerate() {
            let full_idx = cls.x_not_fixed_map[var_idx as usize] as usize;
            full[full_idx] = vals[k];
        }
        full
    }

    fn finalize_solution_lambda(&self, y_c: &dyn Vector, y_d: &dyn Vector) -> Vec<Number> {
        OrigIpoptNlp::finalize_solution_lambda(self, y_c, y_d)
    }

    fn finalize_solution_z_l(&self, z_l: &dyn Vector) -> Vec<Number> {
        OrigIpoptNlp::finalize_solution_z_l(self, z_l)
    }

    fn finalize_solution_z_u(&self, z_u: &dyn Vector) -> Vec<Number> {
        OrigIpoptNlp::finalize_solution_z_u(self, z_u)
    }

    fn full_x_to_var_x(&self, full_idx: Index) -> Option<Index> {
        let cls = self.adapter.borrow();
        let cls = cls.classification();
        let f = full_idx as usize;
        if f >= cls.full_to_var.len() {
            return None;
        }
        let v = cls.full_to_var[f];
        if v < 0 { None } else { Some(v) }
    }

    fn full_g_to_c_block(&self, full_idx: Index) -> Option<Index> {
        let cls = self.adapter.borrow();
        let cls = cls.classification();
        cls.c_map
            .iter()
            .position(|&g_idx| g_idx == full_idx)
            .map(|p| p as Index)
    }

    fn var_x_to_full_x(&self, var_idx: Index) -> Index {
        let cls = self.adapter.borrow();
        let cls = cls.classification();
        cls.x_not_fixed_map[var_idx as usize]
    }
}

// -------------------- Tests --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::{
        BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest,
        StartingPoint, TNLP,
    };

    /// HS071: min x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    /// s.t.   x[0]*x[1]*x[2]*x[3] >= 25                (inequality)
    ///        x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2 == 40  (equality)
    ///        1 <= x[i] <= 5
    #[derive(Default)]
    struct Hs071 {
        eval_f_calls: usize,
        eval_grad_f_calls: usize,
        eval_g_calls: usize,
        eval_jac_g_value_calls: usize,
        eval_h_value_calls: usize,
        get_bounds_info_calls: usize,
    }

    impl TNLP for Hs071 {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 4,
                m: 2,
                nnz_jac_g: 8,
                nnz_h_lag: 10,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.get_bounds_info_calls += 1;
            b.x_l.copy_from_slice(&[1.0; 4]);
            b.x_u.copy_from_slice(&[5.0; 4]);
            // Constraint 0: 25 <= g0 (inequality, finite lower only)
            // Constraint 1: g1 == 40                (equality)
            b.g_l.copy_from_slice(&[25.0, 40.0]);
            b.g_u.copy_from_slice(&[2.0e19, 40.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            self.eval_f_calls += 1;
            Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            self.eval_grad_f_calls += 1;
            // df/dx0 = x3*(2x0 + x1 + x2)
            // df/dx1 = x0*x3
            // df/dx2 = x0*x3 + 1
            // df/dx3 = x0*(x0 + x1 + x2)
            g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
            g[1] = x[0] * x[3];
            g[2] = x[0] * x[3] + 1.0;
            g[3] = x[0] * (x[0] + x[1] + x[2]);
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            self.eval_g_calls += 1;
            // g0 = x0*x1*x2*x3 (>=25)
            // g1 = x0^2 + x1^2 + x2^2 + x3^2 (==40)
            g[0] = x[0] * x[1] * x[2] * x[3];
            g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            true
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    // Dense 2x4: row-major (g0 over x0..x3, then g1 over x0..x3).
                    irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                    jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
                }
                SparsityRequest::Values { values } => {
                    self.eval_jac_g_value_calls += 1;
                    let x = x.expect("eval_jac_g(Values) without x");
                    // d g0 / d x_j
                    values[0] = x[1] * x[2] * x[3];
                    values[1] = x[0] * x[2] * x[3];
                    values[2] = x[0] * x[1] * x[3];
                    values[3] = x[0] * x[1] * x[2];
                    // d g1 / d x_j
                    values[4] = 2.0 * x[0];
                    values[5] = 2.0 * x[1];
                    values[6] = 2.0 * x[2];
                    values[7] = 2.0 * x[3];
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            _new_x: bool,
            obj_factor: Number,
            lambda: Option<&[Number]>,
            _new_lambda: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            // Dense lower triangle of 4x4 = 10 entries:
            // (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
                    jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
                }
                SparsityRequest::Values { values } => {
                    self.eval_h_value_calls += 1;
                    let x = x.expect("eval_h(Values) without x");
                    let lam = lambda.expect("eval_h(Values) without lambda");
                    let of = obj_factor;
                    // Hessian of objective:
                    //   d2f/dx0^2 = 2*x3
                    //   d2f/dx0dx1 = x3,  d2f/dx0dx2 = x3,
                    //   d2f/dx0dx3 = 2*x0+x1+x2
                    //   d2f/dx1dx3 = x0,  d2f/dx2dx3 = x0
                    // Hessian of g0 = x0*x1*x2*x3:
                    //   d2/dx0dx1 = x2*x3, d2/dx0dx2 = x1*x3, d2/dx0dx3 = x1*x2
                    //   d2/dx1dx2 = x0*x3, d2/dx1dx3 = x0*x2, d2/dx2dx3 = x0*x1
                    // Hessian of g1 = sum x_i^2: 2*I.
                    let l0 = lam[0];
                    let l1 = lam[1];
                    values[0] = of * (2.0 * x[3]) + l1 * 2.0; // (0,0)
                    values[1] = of * x[3] + l0 * (x[2] * x[3]); // (1,0)
                    values[2] = l1 * 2.0; // (1,1)
                    values[3] = of * x[3] + l0 * (x[1] * x[3]); // (2,0)
                    values[4] = l0 * (x[0] * x[3]); // (2,1)
                    values[5] = l1 * 2.0; // (2,2)
                    values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]); // (3,0)
                    values[7] = of * x[0] + l0 * (x[0] * x[2]); // (3,1)
                    values[8] = of * x[0] + l0 * (x[0] * x[1]); // (3,2)
                    values[9] = l1 * 2.0; // (3,3)
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn build_orig_nlp() -> (Rc<RefCell<TNLPAdapter>>, OrigIpoptNlp) {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071::default()));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        (adapter, nlp)
    }

    fn dense_x(values: &[Number], space: &Rc<DenseVectorSpace>) -> DenseVector {
        let mut v = space.make_new_dense();
        v.values_mut().copy_from_slice(values);
        v
    }

    #[test]
    fn dimensions_match_classification() {
        let (_, nlp) = build_orig_nlp();
        // HS071: 4 vars (none fixed), 1 equality, 1 inequality.
        assert_eq!(nlp.n(), 4);
        assert_eq!(nlp.m_eq(), 1);
        assert_eq!(nlp.m_ineq(), 1);
        // 4 entries of jac_g go to c-row (g1), 4 go to d-row (g0).
        assert_eq!(nlp.jac_c_space().nonzeros(), 4);
        assert_eq!(nlp.jac_d_space().nonzeros(), 4);
        // Hessian sparsity comes through.
        assert_eq!(nlp.h_space().unwrap().nonzeros(), 10);
        // Bounds: all 4 x's bounded both sides; 1 ineq with finite lower only.
        assert_eq!(nlp.x_l().dim(), 4);
        assert_eq!(nlp.x_u().dim(), 4);
        assert_eq!(nlp.d_l().dim(), 1);
        assert_eq!(nlp.d_u().dim(), 0);
    }

    #[test]
    fn eval_f_at_starting_point() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        // f = 1*1*(1+5+5) + 5 = 11 + 5 = 16
        assert_eq!(nlp.eval_f(&x), 16.0);
        assert_eq!(nlp.f_evals(), 1);
    }

    #[test]
    fn eval_grad_f_at_starting_point() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut g = nlp.x_space().make_new_dense();
        nlp.eval_grad_f(&x, &mut g);
        // df/dx0 = 1*(2 + 5 + 5) = 12
        // df/dx1 = 1*1 = 1
        // df/dx2 = 1*1 + 1 = 2
        // df/dx3 = 1*(1 + 5 + 5) = 11
        assert_eq!(g.values(), &[12.0, 1.0, 2.0, 11.0]);
        assert_eq!(nlp.grad_f_evals(), 1);
    }

    #[test]
    fn eval_c_returns_equality_residual() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut c = nlp.c_space().make_new_dense();
        nlp.eval_c(&x, &mut c);
        // g1 = 1 + 25 + 25 + 1 = 52; residual = 52 - 40 = 12.
        assert_eq!(c.values(), &[12.0]);
        assert_eq!(nlp.c_evals(), 1);
    }

    #[test]
    fn eval_d_returns_inequality_value_unshifted() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut d = nlp.d_space().make_new_dense();
        nlp.eval_d(&x, &mut d);
        // g0 = 1*5*5*1 = 25.
        assert_eq!(d.values(), &[25.0]);
        assert_eq!(nlp.d_evals(), 1);
    }

    #[test]
    fn cache_returns_without_re_eval() {
        let (_, mut nlp) = build_orig_nlp();
        let mut x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let f1 = nlp.eval_f(&x);
        let f2 = nlp.eval_f(&x);
        assert_eq!(f1, f2);
        assert_eq!(nlp.f_evals(), 1, "second call must be served from cache");
        // Bumping x's tag (i.e. mutating it) should invalidate the cache.
        x.values_mut()[0] = 1.0; // values_mut bumps the cache.
        let _ = nlp.eval_f(&x);
        assert_eq!(nlp.f_evals(), 2);
    }

    #[test]
    fn jac_c_picks_only_equality_rows() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let m = nlp.eval_jac_c(&x);
        let g = m
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("jac_c is a GenTMatrix");
        // Equality is g1: dg1/dxj = 2*x_j.
        assert_eq!(g.values(), &[2.0, 10.0, 10.0, 2.0]);
        // 1-based row should all be 1 (the single equality row).
        assert_eq!(g.irows(), &[1, 1, 1, 1]);
        assert_eq!(g.jcols(), &[1, 2, 3, 4]);
    }

    #[test]
    fn jac_d_picks_only_inequality_rows() {
        let (_, mut nlp) = build_orig_nlp();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let m = nlp.eval_jac_d(&x);
        let g = m
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("jac_d is a GenTMatrix");
        // Inequality is g0: d/dxj of x0*x1*x2*x3 at (1,5,5,1).
        // d/dx0 = 5*5*1 = 25, d/dx1 = 1*5*1 = 5, d/dx2 = 1*5*1 = 5, d/dx3 = 1*5*5 = 25.
        assert_eq!(g.values(), &[25.0, 5.0, 5.0, 25.0]);
    }

    /// Build an `OrigIpoptNlp` over `Hs071` while retaining a typed handle
    /// to the underlying TNLP, so a test can read its `eval_g_calls` /
    /// `eval_jac_g_value_calls` counters (the adapter only exposes a
    /// `dyn TNLP`). Both `Rc`s alias the same allocation.
    fn build_orig_nlp_counting() -> (Rc<RefCell<Hs071>>, OrigIpoptNlp) {
        let concrete = Rc::new(RefCell::new(Hs071::default()));
        let tnlp: Rc<RefCell<dyn TNLP>> = concrete.clone();
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        (concrete, nlp)
    }

    #[test]
    fn eval_c_and_eval_d_share_one_eval_g_per_iterate() {
        // Code review 2026-06 item M16: `eval_c` and `eval_d` must slice
        // their rows out of ONE shared `g(x)`, not call the user `eval_g`
        // twice. Before the fix this asserted 2.
        let (tnlp, mut nlp) = build_orig_nlp_counting();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut c = nlp.c_space().make_new_dense();
        let mut d = nlp.d_space().make_new_dense();
        nlp.eval_c(&x, &mut c);
        nlp.eval_d(&x, &mut d);
        assert_eq!(
            tnlp.borrow().eval_g_calls,
            1,
            "eval_c + eval_d at one iterate must share a single user eval_g"
        );
        // Per-subsystem counters still report one c and one d evaluation.
        assert_eq!(nlp.c_evals(), 1);
        assert_eq!(nlp.d_evals(), 1);
        // Values stay correct: c = g1 - 40 = 52 - 40 = 12, d = g0 = 25.
        assert_eq!(c.values(), &[12.0]);
        assert_eq!(d.values(), &[25.0]);

        // A genuinely new iterate (x mutated → tag bumped) costs exactly
        // one more eval_g shared across both subsystems.
        let mut x2 = x;
        x2.values_mut()[0] = 2.0;
        nlp.eval_c(&x2, &mut c);
        nlp.eval_d(&x2, &mut d);
        assert_eq!(
            tnlp.borrow().eval_g_calls,
            2,
            "a new iterate triggers exactly one more shared eval_g"
        );
    }

    #[test]
    fn eval_c_does_not_refetch_bounds_per_iterate() {
        // Code review 2026-06 item M17: the constant equality RHS is the
        // bound `g_l == g_u`, captured once at construction. `eval_c` must
        // NOT call the user's `get_bounds_info` on every (cache-missing)
        // iterate just to subtract that RHS. Before the fix each fresh
        // iterate re-fetched all bounds (and allocated four full-size
        // scratch vectors); this asserted the call count climbed with the
        // iterate count.
        let (tnlp, mut nlp) = build_orig_nlp_counting();
        // Construction fetches the bounds (once for classification, once in
        // `OrigIpoptNlp::new`). Snapshot whatever that baseline is.
        let baseline = tnlp.borrow().get_bounds_info_calls;

        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut c = nlp.c_space().make_new_dense();
        nlp.eval_c(&x, &mut c);
        // RHS is correct: c = g1 - 40 = (1+25+25+1) - 40 = 12.
        assert_eq!(c.values(), &[12.0]);

        // Several genuinely new iterates, each a cache miss.
        let mut x2 = x;
        for k in 0..5 {
            x2.values_mut()[0] = 2.0 + k as Number;
            nlp.eval_c(&x2, &mut c);
        }

        assert_eq!(
            tnlp.borrow().get_bounds_info_calls,
            baseline,
            "eval_c must reuse the captured c_rhs, not re-fetch bounds per iterate"
        );
    }

    #[test]
    fn eval_jac_c_and_eval_jac_d_share_one_eval_jac_g_per_iterate() {
        // Code review 2026-06 item M16: the full Jacobian is evaluated once
        // per iterate and sliced into jac_c / jac_d. Before the fix this
        // asserted 2.
        let (tnlp, mut nlp) = build_orig_nlp_counting();
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let _ = nlp.eval_jac_c(&x);
        let _ = nlp.eval_jac_d(&x);
        assert_eq!(
            tnlp.borrow().eval_jac_g_value_calls,
            1,
            "eval_jac_c + eval_jac_d at one iterate must share a single eval_jac_g"
        );
        assert_eq!(nlp.jac_c_evals(), 1);
        assert_eq!(nlp.jac_d_evals(), 1);
    }

    #[test]
    fn starting_point_is_compressed_into_x_var() {
        let (_, mut nlp) = build_orig_nlp();
        let mut x = nlp.x_space().make_new_dense();
        let mut yc = nlp.c_space().make_new_dense();
        let mut yd = nlp.d_space().make_new_dense();
        let mut zl = nlp.x_l_space().make_new_dense();
        let mut zu = nlp.x_u_space().make_new_dense();
        let ok = nlp.initialize_starting_point(
            &mut x, true, &mut yc, false, &mut yd, false, &mut zl, false, &mut zu, false,
        );
        assert!(ok);
        assert_eq!(x.values(), &[1.0, 5.0, 5.0, 1.0]);
    }

    /// Two-variable TNLP with `x[0]` fixed at 7.0 (`x_l == x_u`) and
    /// one equality on `x[1]`. Exercises the index-mapping methods on
    /// the `IpoptNlp` trait that are used by `pounce_sens` to support
    /// `.nl` files with fixed variables.
    struct OneFixedOneFree;
    impl TNLP for OneFixedOneFree {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 1,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 7.0;
            b.x_u[0] = 7.0; // fixed
            b.x_l[1] = -1.0e19;
            b.x_u[1] = 1.0e19;
            b.g_l[0] = 0.0;
            b.g_u[0] = 0.0; // equality
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 7.0;
            sp.x[1] = 0.5;
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(x[1])
        }
        fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 0.0;
            g[1] = 1.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[1];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, m: SparsityRequest<'_>) -> bool {
            match m {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 1;
                }
                SparsityRequest::Values { values } => values[0] = 1.0,
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn ipopt_nlp_index_mapping_methods_handle_fixed_var() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedOneFree));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();

        // Sanity: classification trimmed x[0] (fixed) from var-x space.
        assert_eq!(nlp.n_full_x(), 2);
        assert_eq!(nlp.n(), 1);

        // full_x_to_var_x: x[0] is fixed → None; x[1] → var idx 0.
        let nlp_dyn: &dyn crate::ipopt_nlp::IpoptNlp = &nlp;
        assert_eq!(nlp_dyn.full_x_to_var_x(0), None);
        assert_eq!(nlp_dyn.full_x_to_var_x(1), Some(0));

        // var_x_to_full_x: var 0 → full 1.
        assert_eq!(nlp_dyn.var_x_to_full_x(0), 1);

        // full_g_to_c_block: the one g is an equality → c-block 0.
        assert_eq!(nlp_dyn.full_g_to_c_block(0), Some(0));

        // lift_x_to_full inflates a compressed [v_0] back to [7.0, v_0].
        let mut x_var = nlp.x_space().make_new_dense();
        x_var.values_mut()[0] = 0.5;
        let lifted = nlp_dyn.lift_x_to_full(&x_var);
        assert_eq!(lifted, vec![7.0, 0.5]);
    }

    /// `OneFixedOneFree` plus `idx_names` metadata — used to check the
    /// split-space name projection threads names through the fixed-var
    /// and c/d-split permutations.
    struct NamedFixedOneFree;
    impl TNLP for NamedFixedOneFree {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            OneFixedOneFree.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            OneFixedOneFree.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            OneFixedOneFree.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            OneFixedOneFree.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            OneFixedOneFree.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            OneFixedOneFree.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, m: SparsityRequest<'_>) -> bool {
            OneFixedOneFree.eval_jac_g(x, n, m)
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
        fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
            var.strings.insert(
                IDX_NAMES.to_string(),
                vec!["fixed_x".to_string(), "free_x".to_string()],
            );
            con.strings
                .insert(IDX_NAMES.to_string(), vec!["balance".to_string()]);
            true
        }
    }

    #[test]
    fn split_space_names_threads_through_fixed_var_and_cd_split() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(NamedFixedOneFree));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();

        let names = nlp.split_space_names().expect("names present");
        // x[0] (fixed) dropped; var-x 0 is full-x 1 = "free_x".
        assert_eq!(names.x_var, vec![Some("free_x".to_string())]);
        // The single g is an equality → c-block 0 = "balance".
        assert_eq!(names.eq, vec![Some("balance".to_string())]);
        // No inequalities.
        assert!(names.ineq.is_empty());
        assert!(names.any_present());
    }

    #[test]
    fn split_space_names_none_when_tnlp_declines() {
        // OneFixedOneFree does not implement get_var_con_metadata.
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedOneFree));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        assert!(nlp.split_space_names().is_none());
    }

    /// Regression: a TNLP with `x[0]` fixed and `nnz_h_lag = 1` whose
    /// only Hessian entry is (0,0). After fixed-var filtering kept = 0
    /// but `nnz_h_lag_full = 1`, which used to hit the broken
    /// `h_entry_in_full.is_empty()` fast path and panic in
    /// `copy_from_slice`.
    struct FixedOnlyHess;
    impl TNLP for FixedOnlyHess {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 1,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 7.0;
            b.x_u[0] = 7.0; // fixed
            b.x_l[1] = -1.0e19;
            b.x_u[1] = 1.0e19;
            b.g_l[0] = 0.0;
            b.g_u[0] = 0.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 7.0;
            sp.x[1] = 0.5;
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(0.5 * x[0] * x[0] + x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[0];
            g[1] = 1.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[1];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, m: SparsityRequest<'_>) -> bool {
            match m {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 1;
                }
                SparsityRequest::Values { values } => values[0] = 1.0,
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            obj_factor: Number,
            _: Option<&[Number]>,
            _: bool,
            m: SparsityRequest<'_>,
        ) -> bool {
            match m {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = obj_factor,
            }
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    /// One variable with a single one-sided inequality whose Jacobian
    /// magnitude trips `nlp_scaling_max_gradient` (default 100): coeff
    /// 1000, bound `lo = 4e6`. After gradient-based scaling the
    /// `d_scale` for this row is `100/1000 = 0.1`, so the algorithm
    /// sees `d(x) = 0.1 * 1000 * x`. The bound must be scaled to
    /// `0.1 * 4e6 = 4e5` to match — otherwise the algorithm reads a
    /// 10x-too-large lower bound and reports phantom infeasibility
    /// (gh#54).
    struct OneIneqLargeOffset;
    impl TNLP for OneIneqLargeOffset {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 1,
                nnz_jac_g: 1,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -1.0e19;
            b.x_u[0] = 1.0e19;
            b.g_l[0] = 4.0e6;
            b.g_u[0] = 2.0e19;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 5000.0;
            true
        }
        fn eval_f(&mut self, _: &[Number], _: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 0.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 1000.0 * x[0];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, m: SparsityRequest<'_>) -> bool {
            match m {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = 1000.0,
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn gradient_based_scaling_scales_d_l_and_d_u() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneIneqLargeOffset));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();

        // Pre-scaling: d_l carries the raw user bound 4e6.
        assert_eq!(nlp.d_l().dim(), 1);
        let pre = nlp
            .d_l()
            .as_any()
            .downcast_ref::<DenseVector>()
            .unwrap()
            .values()[0];
        assert_eq!(pre, 4.0e6);

        nlp.determine_scaling_from_starting_point(
            ScalingMethod::GradientBased,
            100.0,
            1e-8,
            0.0,
            0.0,
        );

        // d_scale = 100 / 1000 = 0.1; bound must scale in step.
        let post = nlp
            .d_l()
            .as_any()
            .downcast_ref::<DenseVector>()
            .unwrap()
            .values()[0];
        assert!(
            (post - 4.0e5).abs() < 1e-9,
            "d_l should be scaled by d_scale=0.1; got {}",
            post
        );

        // And d(x) at the starting point must agree with the scaled
        // bound: d(5000) = 0.1 * 1000 * 5000 = 5e5 > 4e5, so feasible.
        let x = dense_x(&[5000.0], nlp.x_space());
        let mut d = nlp.d_space().make_new_dense();
        nlp.eval_d(&x, &mut d);
        assert!(
            (d.values()[0] - 5.0e5).abs() < 1e-6,
            "scaled d(x) mismatch; got {}",
            d.values()[0]
        );
        assert!(
            d.values()[0] >= post,
            "starting point must be feasible in scaled space"
        );
    }

    /// Same fixture as [`OneIneqLargeOffset`] but with a non-zero
    /// objective gradient (10), so we can verify that
    /// `nlp_scaling_obj_target_gradient` pins the scaled gradient
    /// ∞-norm exactly to the requested value (independent of the
    /// `max_gradient` cutoff).
    struct OneIneqWithObj;
    impl TNLP for OneIneqWithObj {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 1,
                nnz_jac_g: 1,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -1.0e19;
            b.x_u[0] = 1.0e19;
            b.g_l[0] = 4.0e6;
            b.g_u[0] = 2.0e19;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 5000.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(10.0 * x[0])
        }
        fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 10.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 1000.0 * x[0];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, m: SparsityRequest<'_>) -> bool {
            match m {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = 1000.0,
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn obj_target_gradient_pins_obj_scale() {
        // grad_f = [10], so the default gradient-based path (max_grad=100,
        // 10 < cutoff) does NOT scale the objective: df = 1.
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneIneqWithObj));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        nlp.determine_scaling_from_starting_point(
            ScalingMethod::GradientBased,
            100.0,
            1e-8,
            0.0, // no target → use cutoff path
            0.0,
        );
        assert!(
            (nlp.obj_scale_factor() - 1.0).abs() < 1e-12,
            "no-target path leaves df=1 when grad < cutoff; got {}",
            nlp.obj_scale_factor()
        );

        // With obj_target_gradient = 1.0 the scaled gradient ∞-norm
        // must be exactly 1, i.e. df = 1.0 / 10.0 = 0.1.
        let tnlp2: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneIneqWithObj));
        let adapter2 = Rc::new(RefCell::new(TNLPAdapter::new(tnlp2).unwrap()));
        let mut nlp2 = OrigIpoptNlp::new(Rc::clone(&adapter2), Rc::new(NoScaling)).unwrap();
        nlp2.determine_scaling_from_starting_point(
            ScalingMethod::GradientBased,
            100.0,
            1e-8,
            1.0,
            0.0,
        );
        assert!(
            (nlp2.obj_scale_factor() - 0.1).abs() < 1e-12,
            "target_gradient=1, max_grad_f=10 → df=0.1; got {}",
            nlp2.obj_scale_factor()
        );
    }

    /// Regression (flosp2hm): gradient-based scaling must sample the
    /// objective gradient at the point the algorithm actually operates
    /// on — i.e. with fixed variables (`x_l == x_u`) lifted to their
    /// fixed value — not at the raw `x0` returned by `get_starting_point`.
    /// Here `x[1]` is fixed at 1000 but the starting point places it at 0,
    /// and the only free-variable gradient is `df/dx0 = x[1]`. Sampling at
    /// the raw `x0` gives `max_grad_f = 0` (df stays 1.0, no scaling);
    /// lifting `x[1]→1000` gives `max_grad_f = 1000`, so df = 100/1000 = 0.1.
    /// Pre-fix this left df=1 and stalled flosp2hm at max-iter.
    struct FixedVarShiftsObjGrad;
    impl TNLP for FixedVarShiftsObjGrad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -1.0e19;
            b.x_u[0] = 1.0e19;
            b.x_l[1] = 1000.0;
            b.x_u[1] = 1000.0; // fixed at 1000
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 1.0;
            sp.x[1] = 0.0; // deliberately NOT the fixed value
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(x[0] * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[1];
            g[1] = x[0];
            true
        }
        fn eval_g(&mut self, _: &[Number], _: bool, _: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, _: SparsityRequest<'_>) -> bool {
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn gradient_scaling_lifts_fixed_vars_to_their_value() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(FixedVarShiftsObjGrad));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();

        // Sanity: x[1] is fixed out of the var-x space, fixed value 1000.
        assert_eq!(nlp.n_full_x(), 2);
        assert_eq!(nlp.n(), 1);

        nlp.determine_scaling_from_starting_point(
            ScalingMethod::GradientBased,
            100.0,
            1e-8,
            0.0,
            0.0,
        );

        // Lifted gradient ∞-norm over free vars is |df/dx0| = x[1] = 1000,
        // so df = 100/1000 = 0.1. Sampling at the raw x0 (x[1]=0) would
        // give 0 and leave df=1.0 (the pre-fix bug).
        assert!(
            (nlp.obj_scale_factor() - 0.1).abs() < 1e-12,
            "fixed var must be lifted before scaling; expected df=0.1, got {}",
            nlp.obj_scale_factor()
        );
    }

    #[test]
    fn constr_target_gradient_overrides_cutoff_and_clamp() {
        // Jacobian row max = 1000. Default gradient-based: cutoff 100
        // fires, dc = min(1, 100/1000) = 0.1. With
        // constr_target_gradient = 50 → dc = 50/1000 = 0.05 (no clamp
        // at 1, no cutoff check).
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneIneqLargeOffset));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        nlp.determine_scaling_from_starting_point(
            ScalingMethod::GradientBased,
            100.0,
            1e-8,
            0.0,
            50.0,
        );
        let x = dense_x(&[5000.0], nlp.x_space());
        let mut d = nlp.d_space().make_new_dense();
        nlp.eval_d(&x, &mut d);
        // scaled d(x) = 0.05 * 1000 * 5000 = 2.5e5.
        assert!(
            (d.values()[0] - 2.5e5).abs() < 1e-6,
            "constr target=50 → dd=0.05; scaled d(5000)=2.5e5, got {}",
            d.values()[0]
        );
    }

    /// User-supplied TNLP that returns a per-constraint scaling vector
    /// via `get_scaling_parameters`. Constraint 0 is the equality (g1);
    /// constraint 1 is the inequality (g0). We reuse the HS071 fixture
    /// so the c/d split is well-defined.
    struct Hs071UserScaled;
    impl TNLP for Hs071UserScaled {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Hs071::default().get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            Hs071::default().get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            Hs071::default().get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
            Hs071::default().eval_f(x, new_x)
        }
        fn eval_grad_f(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            Hs071::default().eval_grad_f(x, new_x, g)
        }
        fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            Hs071::default().eval_g(x, new_x, g)
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            Hs071::default().eval_jac_g(x, new_x, mode)
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
            Hs071::default().eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
        }
        fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
            *req.obj_scaling = 2.0;
            *req.use_x_scaling = false;
            *req.use_g_scaling = true;
            // HS071 g layout: g[0] = inequality, g[1] = equality.
            req.g_scaling[0] = 0.5;
            req.g_scaling[1] = 0.25;
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn user_scaling_dispatch_applies_obj_and_g_scaling() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071UserScaled));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        nlp.determine_scaling_from_starting_point(
            ScalingMethod::UserScaling,
            100.0,
            1e-8,
            0.0,
            0.0,
        );

        // Objective scaling: 2.0 (no automatic floor needed since
        // user supplied a normal-sized factor).
        assert!(
            (nlp.obj_scale_factor() - 2.0).abs() < 1e-12,
            "user obj_scaling=2.0 should be installed; got {}",
            nlp.obj_scale_factor()
        );

        // Equality row (g1) gets g_scaling[1] = 0.25 → c-scaled
        // residual is 0.25× the unscaled one. Compute c at the
        // starting point.
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut c = nlp.c_space().make_new_dense();
        nlp.eval_c(&x, &mut c);
        // Unscaled: g1 = 1+25+25+1 = 52, residual = 52-40 = 12.
        // Scaled: 0.25 * 12 = 3.0.
        assert!(
            (c.values()[0] - 3.0).abs() < 1e-9,
            "user g_scaling=0.25 on equality → c=3.0; got {}",
            c.values()[0]
        );

        // Inequality row (g0) gets g_scaling[0] = 0.5 → d = 0.5 *
        // 1*5*5*1 = 12.5.
        let mut d = nlp.d_space().make_new_dense();
        nlp.eval_d(&x, &mut d);
        assert!(
            (d.values()[0] - 12.5).abs() < 1e-9,
            "user g_scaling=0.5 on inequality → d=12.5; got {}",
            d.values()[0]
        );

        // And d_l must have been brought along: the user lower bound
        // on g0 is 25 (HS071); scaled by 0.5 → 12.5.
        let post_d_l = nlp
            .d_l()
            .as_any()
            .downcast_ref::<DenseVector>()
            .unwrap()
            .values()[0];
        assert!(
            (post_d_l - 12.5).abs() < 1e-9,
            "d_l scaled in step: got {}",
            post_d_l
        );
    }

    /// TNLP whose `get_scaling_parameters` returns false — selecting
    /// `UserScaling` must fall back to no automatic scaling (matches
    /// upstream behavior).
    struct Hs071DeclinesScaling;
    impl TNLP for Hs071DeclinesScaling {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Hs071::default().get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            Hs071::default().get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            Hs071::default().get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
            Hs071::default().eval_f(x, new_x)
        }
        fn eval_grad_f(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            Hs071::default().eval_grad_f(x, new_x, g)
        }
        fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            Hs071::default().eval_g(x, new_x, g)
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            Hs071::default().eval_jac_g(x, new_x, mode)
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
            Hs071::default().eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn user_scaling_falls_back_when_tnlp_declines() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071DeclinesScaling));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();
        nlp.determine_scaling_from_starting_point(
            ScalingMethod::UserScaling,
            100.0,
            1e-8,
            0.0,
            0.0,
        );
        // No automatic scaling installed: obj_scale_factor = 1.0, c/d
        // unscaled.
        assert!((nlp.obj_scale_factor() - 1.0).abs() < 1e-12);
        let x = dense_x(&[1.0, 5.0, 5.0, 1.0], nlp.x_space());
        let mut c = nlp.c_space().make_new_dense();
        nlp.eval_c(&x, &mut c);
        assert_eq!(c.values(), &[12.0], "unscaled equality residual");
    }

    #[test]
    fn eval_h_with_all_entries_on_fixed_var_does_not_panic() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(FixedOnlyHess));
        let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
        let mut nlp = OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)).unwrap();

        // After filtering, the kept Hessian over var-x has 0 nonzeros,
        // while the user's full Hessian has 1.
        assert_eq!(nlp.h_space().unwrap().nonzeros(), 0);

        let x = dense_x(&[0.5], &nlp.x_space().clone());
        let yc = dense_x(&[0.0], &nlp.c_space().clone());
        let yd = nlp.d_space().make_new_dense();
        let h = nlp.eval_h(&x, 1.0, &yc, &yd);
        assert_eq!(h.n_rows(), 1);
    }

    #[test]
    fn relax_bounds_widens_uniquely_owned_bounds() {
        // Baseline: with uniquely-owned bound Rcs (the normal post-construction
        // state) relax_bounds loosens x_l downward and x_u upward.
        let (_adapter, mut nlp) = build_orig_nlp();
        let x_l_before = nlp.x_l.values().to_vec();
        let x_u_before = nlp.x_u.values().to_vec();
        nlp.relax_bounds(1e-2, 1.0);
        for (b, a) in x_l_before.iter().zip(nlp.x_l.values()) {
            assert!(a < b, "x_l should relax downward: {a} !< {b}");
        }
        for (b, a) in x_u_before.iter().zip(nlp.x_u.values()) {
            assert!(a > b, "x_u should relax upward: {a} !> {b}");
        }
    }

    #[test]
    #[should_panic(expected = "x_l is uniquely owned")]
    fn relax_bounds_panics_on_shared_bound_rc() {
        // Code review L33: a shared bound Rc used to make relax_bounds silently
        // skip the relaxation, leaving bounds tighter than bound_relax_factor
        // requires. The unique-ownership invariant is now enforced loudly,
        // matching adjust_variable_bounds' `expect`.
        let (_adapter, mut nlp) = build_orig_nlp();
        let _shared = Rc::clone(&nlp.x_l); // bump strong_count so get_mut fails
        nlp.relax_bounds(1e-2, 1.0);
    }
}
