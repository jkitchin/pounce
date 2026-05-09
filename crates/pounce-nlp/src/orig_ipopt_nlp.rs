//! `OrigIpoptNlp` — concrete `IpoptNlp` impl that wraps a [`TNLPAdapter`]
//! and an `NlpScalingObject`. Port of
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

use crate::ipopt_nlp::{IpoptNlp, Nlp};
use crate::tnlp::{NlpInfo, SparsityRequest, StartingPoint};
use crate::tnlp_adapter::{BoundClassification, TNLPAdapter};
use pounce_common::cached::Cache;
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
/// We deliberately keep the trait local instead of pulling in
/// `pounce_algorithm::scaling::NlpScalingObject` to avoid the
/// `pounce-nlp → pounce-algorithm` dependency cycle.
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

/// Selector for [`OrigIpoptNlp::determine_scaling_from_starting_point`].
/// Mirrors upstream's `nlp_scaling_method` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMethod {
    /// No automatic scaling beyond the constant `obj_scaling_factor`.
    None,
    /// Gradient-based per `Algorithm/IpGradientScaling.cpp`. Default.
    GradientBased,
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

    // ----- caches (one entry; key = input vector tag) -----
    f_cache: RefCell<Cache<Number>>,
    grad_f_cache: RefCell<Cache<Rc<dyn Vector>>>,
    c_cache: RefCell<Cache<Rc<dyn Vector>>>,
    d_cache: RefCell<Cache<Rc<dyn Vector>>>,
    jac_c_cache: RefCell<Cache<Rc<dyn Matrix>>>,
    jac_d_cache: RefCell<Cache<Rc<dyn Matrix>>>,
    h_cache: RefCell<Cache<Rc<dyn SymMatrix>>>,

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
        let px_l_space = ExpansionMatrixSpace::new(
            n_x_var,
            classification.n_x_l(),
            &classification.x_l_map,
            0,
        );
        let px_u_space = ExpansionMatrixSpace::new(
            n_x_var,
            classification.n_x_u(),
            &classification.x_u_map,
            0,
        );
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

        for k in 0..info.nnz_jac_g as usize {
            let g_row_0 = (full_irow[k] - style_offset) as usize;
            let x_col_0 = (full_jcol[k] - style_offset) as usize;
            // Triplet output is 1-based (matches `GenTMatrix` convention).
            let col_1based = (x_col_0 + 1) as Index;
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
                // Convert to 1-based (irrespective of TNLP index style).
                let h_irow_1: Vec<Index> = h_irow
                    .iter()
                    .map(|&v| v - style_offset + 1)
                    .collect();
                let h_jcol_1: Vec<Index> = h_jcol
                    .iter()
                    .map(|&v| v - style_offset + 1)
                    .collect();
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

        Ok(Self {
            adapter,
            scaling,
            obj_scale_factor: Cell::new(1.0),
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
            px_l,
            px_u,
            pd_l,
            pd_u,
            jac_c_entry_in_g,
            jac_d_entry_in_g,
            nnz_jac_g_full: info.nnz_jac_g,
            f_cache: RefCell::new(Cache::new(1)),
            grad_f_cache: RefCell::new(Cache::new(1)),
            c_cache: RefCell::new(Cache::new(1)),
            d_cache: RefCell::new(Cache::new(1)),
            jac_c_cache: RefCell::new(Cache::new(1)),
            jac_d_cache: RefCell::new(Cache::new(1)),
            h_cache: RefCell::new(Cache::new(1)),
            f_evals: RefCell::new(0),
            grad_f_evals: RefCell::new(0),
            c_evals: RefCell::new(0),
            d_evals: RefCell::new(0),
            jac_c_evals: RefCell::new(0),
            jac_d_evals: RefCell::new(0),
            h_evals: RefCell::new(0),
            info,
        })
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
        if let Some(x_l) = Rc::get_mut(&mut self.x_l) {
            apply(x_l, -1.0);
        }
        if let Some(x_u) = Rc::get_mut(&mut self.x_u) {
            apply(x_u, 1.0);
        }
        if let Some(d_l) = Rc::get_mut(&mut self.d_l) {
            apply(d_l, -1.0);
        }
        if let Some(d_u) = Rc::get_mut(&mut self.d_u) {
            apply(d_u, 1.0);
        }
    }

    /// Gradient-based determination of `df_`, `dc_`, `dd_` per
    /// `Algorithm/IpGradientScaling.cpp::DetermineScalingParametersImpl`.
    /// Should be called once, after construction and before the
    /// algorithm enters its main loop.
    ///
    /// `max_gradient` is `nlp_scaling_max_gradient` (cutoff above which
    /// scaling is applied; default 100). `min_value` is
    /// `nlp_scaling_min_value` (floor on computed scale factors; default
    /// 1e-8). Cache state is invalidated so subsequent eval calls
    /// produce scaled values.
    pub fn determine_scaling_from_starting_point(
        &self,
        method: ScalingMethod,
        max_gradient: Number,
        min_value: Number,
    ) {
        // Always pull the user's `obj_scaling_factor` constant first;
        // it multiplies whatever the gradient-based scheme computes.
        let user_obj_factor = self.scaling.obj_scaling();
        if matches!(method, ScalingMethod::None) {
            self.obj_scale_factor.set(user_obj_factor);
            *self.c_scale.borrow_mut() = None;
            *self.d_scale.borrow_mut() = None;
            self.invalidate_eval_caches();
            return;
        }

        // ---- Get starting x_full -----
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

        // ---- Objective gradient scale ----
        let mut full_grad_f = vec![0.0; n_full_x];
        let grad_ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_grad_f(&full_x, true, &mut full_grad_f)
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
            if max_grad_f > max_gradient {
                df = max_gradient / max_grad_f;
            }
            if df < min_value {
                df = min_value;
            }
        }
        self.obj_scale_factor.set(df * user_obj_factor);

        // ---- Constraint Jacobian row-max scaling ----
        if cls.n_full_g > 0 {
            // Evaluate full Jacobian once at x.
            let mut full_jac_vals = vec![0.0; self.nnz_jac_g_full as usize];
            let jac_ok = {
                let a = self.adapter.borrow();
                let mut t = a.tnlp().borrow_mut();
                t.eval_jac_g(
                    Some(&full_x),
                    true,
                    SparsityRequest::Values {
                        values: &mut full_jac_vals,
                    },
                )
            };
            if jac_ok {
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

                // c-scale: only populated if any row exceeds the cutoff.
                if n_c > 0 {
                    let mut any_above = false;
                    for &v in c_row_max.iter() {
                        if v > max_gradient {
                            any_above = true;
                            break;
                        }
                    }
                    if any_above {
                        let mut dc = vec![0.0; n_c];
                        for i in 0..n_c {
                            // dc = max_gradient / row_max, capped at 1
                            // (ElementWiseMin with Set(1.0) upstream).
                            let mut s = max_gradient / c_row_max[i];
                            if s > 1.0 {
                                s = 1.0;
                            }
                            if s < min_value {
                                s = min_value;
                            }
                            dc[i] = s;
                        }
                        *self.c_scale.borrow_mut() = Some(dc);
                    } else {
                        *self.c_scale.borrow_mut() = None;
                    }
                } else {
                    *self.c_scale.borrow_mut() = None;
                }

                if n_d > 0 {
                    let mut any_above = false;
                    for &v in d_row_max.iter() {
                        if v > max_gradient {
                            any_above = true;
                            break;
                        }
                    }
                    if any_above {
                        let mut dd = vec![0.0; n_d];
                        for i in 0..n_d {
                            let mut s = max_gradient / d_row_max[i];
                            if s > 1.0 {
                                s = 1.0;
                            }
                            if s < min_value {
                                s = min_value;
                            }
                            dd[i] = s;
                        }
                        *self.d_scale.borrow_mut() = Some(dd);
                    } else {
                        *self.d_scale.borrow_mut() = None;
                    }
                } else {
                    *self.d_scale.borrow_mut() = None;
                }
            }
        }

        // Drop any cached eval results computed before the scales were
        // set (their values would be wrong now).
        self.invalidate_eval_caches();
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
    /// `x` (length `n_full_x`). Fixed-variable removal will live here
    /// once Phase-3 introduces it; today `x_not_fixed_map` is identity
    /// for non-fixed problems so this is essentially a copy.
    fn lift_x_to_full(&self, x: &dyn Vector) -> Vec<Number> {
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
        full
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
    fn pack_lambda_for_user(
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
            match t.eval_f(&full_x, true) {
                Some(v) => v,
                None => panic!("TNLP::eval_f returned None during OrigIpoptNlp::eval_f"),
            }
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
        assert!(ok, "TNLP::eval_grad_f returned false");
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
        let full_x = self.lift_x_to_full(x);
        let mut full_g = vec![0.0; cls.n_full_g as usize];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_g(&full_x, true, &mut full_g)
        };
        assert!(ok, "TNLP::eval_g returned false");
        let mut c = self.c_space.make_new_dense();
        // c_i = g(g_idx) - g_l(g_idx)  (since g_l == g_u for equalities,
        // upstream subtracts the bound to make it a residual). Matches
        // `OrigIpoptNLP::c` which calls `nlp_->Eval_c` after the adapter
        // subtracted the bound — TNLPAdapter doesn't subtract yet, so we
        // do it here.
        let n_full_g = cls.n_full_g as usize;
        let mut full_g_l = vec![0.0; n_full_g];
        let mut full_g_u = vec![0.0; n_full_g];
        {
            let mut tmp_x_l = vec![0.0; cls.n_full_x as usize];
            let mut tmp_x_u = vec![0.0; cls.n_full_x as usize];
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.get_bounds_info(crate::tnlp::BoundsInfo {
                x_l: &mut tmp_x_l,
                x_u: &mut tmp_x_u,
                g_l: &mut full_g_l,
                g_u: &mut full_g_u,
            });
        }
        {
            let cv = c.values_mut();
            let cs = self.c_scale.borrow();
            for (i, &g_idx) in cls.c_map.iter().enumerate() {
                let raw = full_g[g_idx as usize] - full_g_l[g_idx as usize];
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
        let full_x = self.lift_x_to_full(x);
        let mut full_g = vec![0.0; cls.n_full_g as usize];
        let ok = {
            let a = self.adapter.borrow();
            let mut t = a.tnlp().borrow_mut();
            t.eval_g(&full_x, true, &mut full_g)
        };
        assert!(ok, "TNLP::eval_g returned false");
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
        assert!(ok, "TNLP::eval_jac_g(Values) returned false");
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
        assert!(ok, "TNLP::eval_jac_g(Values) returned false");
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

        let mut full_vals = vec![0.0; h_space.nonzeros() as usize];
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
        assert!(ok, "TNLP::eval_h(Values) returned false");
        let mut h = SymTMatrix::new(Rc::clone(h_space));
        h.values_mut().copy_from_slice(&full_vals);
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

fn make_dense_from(space: &Rc<DenseVectorSpace>, mut f: impl FnMut(usize) -> Number) -> DenseVector {
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
        self.eval_f_internal(x)
    }
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
        let result = self.eval_grad_f_internal(x);
        g.copy(&*result);
    }
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
        let result = self.eval_c_internal(x);
        c.copy(&*result);
    }
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
        let result = self.eval_d_internal(x);
        d.copy(&*result);
    }
    fn eval_jac_c(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        self.eval_jac_c_internal(x)
    }
    fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix> {
        self.eval_jac_d_internal(x)
    }
    fn eval_h(
        &mut self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        self.eval_h_internal(x, obj_factor, y_c, y_d)
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
                    values[0] = of * (2.0 * x[3]) + l1 * 2.0;            // (0,0)
                    values[1] = of * x[3] + l0 * (x[2] * x[3]);          // (1,0)
                    values[2] = l1 * 2.0;                                // (1,1)
                    values[3] = of * x[3] + l0 * (x[1] * x[3]);          // (2,0)
                    values[4] = l0 * (x[0] * x[3]);                      // (2,1)
                    values[5] = l1 * 2.0;                                // (2,2)
                    values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]); // (3,0)
                    values[7] = of * x[0] + l0 * (x[0] * x[2]);          // (3,1)
                    values[8] = of * x[0] + l0 * (x[0] * x[1]);          // (3,2)
                    values[9] = l1 * 2.0;                                // (3,3)
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
}
