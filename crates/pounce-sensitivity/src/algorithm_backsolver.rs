//! `PdSensBacksolver` — `SensBacksolver` adapter over the converged
//! `PdFullSpaceSolver` from `pounce-algorithm`.
//!
//! This is the Phase B.2 piece tracked in
//! [pounce#16](https://github.com/jkitchin/pounce/issues/16): it lets
//! `pounce-sensitivity` drive backsolves against the real converged
//! KKT factor, replacing the synthetic [`crate::DenseLuBacksolver`]
//! used by Phase B.1 unit tests.
//!
//! # Use
//!
//! 1. Register an `on_converged` callback on `IpoptApplication` via
//!    [`pounce_algorithm::application::IpoptApplication::set_on_converged`].
//! 2. Inside the callback, build a `PdSensBacksolver` from the four
//!    handles passed in (`data`, `cq`, `nlp`, `&mut pd_solver`).
//! 3. Hand it to [`crate::SensApplication`] / a `SensStepCalc` /
//!    [`crate::compute_reduced_hessian`] like any other
//!    [`SensBacksolver`].
//!
//! Upstream `SensSimpleBacksolver`
//! ([`ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp))
//! is the analogous wrapper around `IpoptCalculatedQuantities` +
//! `PDSystemSolver` upstream.
//!
//! # Flat-slice ↔ `IteratesVector` mapping
//!
//! The full primal-dual state of pounce's IPM is the eight-block
//! compound `(x, s, λ_c, λ_d, z_l, z_u, v_l, v_u)` (see
//! [`pounce_algorithm::iterates_vector::IteratesVector`]). This
//! adapter packs / unpacks the flat slices that
//! [`crate::SensBacksolver`] takes as the concatenation
//! `x || s || λ_c || λ_d || z_l || z_u || v_l || v_u`, mirroring
//! upstream's `CompoundVector` layout (`IpCompoundVector.hpp`).
//!
//! # Reference
//!
//! Pirnay, H.; López-Negrete, R.; Biegler, L. T. (2012). *Optimal
//! sensitivity based on IPOPT*. Mathematical Programming Computation,
//! **4**(4), 307–331. DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
//! Verified via Crossref on 2026-05-13.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::iterates_vector::{IteratesVector, IteratesVectorMut};
use pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_nlp::ipopt_nlp::IpoptNlp;

use crate::backsolver::SensBacksolver;

/// Adapter from `PdFullSpaceSolver` to [`SensBacksolver`]. Holds
/// owning clones of the four pieces of the algorithm's converged
/// state, plus the 8-block iterate template used to allocate fresh
/// RHS / LHS vectors.
///
/// The PD solver lives behind an `Rc<RefCell<…>>` because
/// [`SensBacksolver::solve`] is `&self` but the upstream signature
/// for `PdFullSpaceSolver::solve` is `&mut self` (it caches the
/// last-solve dependency tags and the augsys-improved flag). The
/// `RefCell` is single-thread-only, single-borrow, exactly matching
/// the call pattern from `pounce-sensitivity`'s pipeline.
///
/// Owning (rather than borrowing) the four handles is what lets a
/// `PdSensBacksolver` outlive the `on_converged` callback frame —
/// required by the public `Solver` session API in `pounce-algorithm`,
/// which retains the backsolver for repeated `parametric_step` /
/// `kkt_solve` / `compute_reduced_hessian` calls after the IPM has
/// returned. The data, cq, and nlp handles are already
/// `Rc<RefCell<…>>` cheap-clone handles upstream, so this carries no
/// allocation overhead.
#[derive(Clone)]
pub struct PdSensBacksolver {
    /// Shared, interior-mutable handle to the converged PD solver.
    /// Cloned from `PdSearchDirCalc::pd_solver_rc()` at construction.
    pd: Rc<RefCell<PdFullSpaceSolver>>,
    data: IpoptDataHandle,
    cq: IpoptCqHandle,
    nlp: Rc<RefCell<dyn IpoptNlp>>,
    /// Block dimensions in `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    dims: [usize; 8],
    /// 8-block prototype used to mint fresh vectors with the same
    /// `VectorSpace`s as the converged iterate; cloned from
    /// `data.borrow().curr`.
    template: IteratesVector,
    /// Natural-units row/column scaling pair (pounce#128). The IPM's
    /// KKT factor is held in the NLP's internally **scaled** space
    /// (objective factor `df`, per-row constraint factors `dc` / `dd`
    /// from `nlp_scaling_method`; scaled multipliers `ỹ = (df/dc)·y`,
    /// `z̃ = df·z`, `ṽ = (df/dd)·v`, scaled slack `s̃ = dd·s`). The
    /// scaled 8-block primal-dual system is the two-sided diagonal
    /// scaling `K̃ = E K F` of the natural-units system, with
    /// per-block entries
    ///
    /// ```text
    ///        x      s        y_c      y_d     z_l/z_u   v_l/v_u
    /// E  =   df     df/dd_i  dc_i     dd_i    df        df
    /// F  =   1      1/dd_i   dc_i/df  dd_i/df 1/df      dd_r(j)/df
    /// ```
    ///
    /// (`dd_r(j)` = the d-row scaling of the j-th finite d-bound,
    /// through the `pd_l` / `pd_u` expansion). Hence
    /// `K⁻¹ = F K̃⁻¹ E`: scale the RHS by `E`, back-solve against the
    /// held factor, scale the result by `F`. Unlike a symmetric
    /// congruence this needs no square root, so it covers a negative
    /// `obj_scaling_factor` (maximization) and covers the z/v
    /// bound-multiplier rows exactly (those rows admit no symmetric
    /// diagonal: `K̃_{z,x} = df·Z·Pᵀ` but `K̃_{z,z} = X − x_L` is
    /// unscaled). `None` ⇔ scaling inactive, identity.
    conj: Option<Rc<ConjPair>>,
}

/// Left/right diagonal pair for the natural-units back-solve; see the
/// `conj` field doc on [`PdSensBacksolver`]. Both vectors are
/// flat-KKT-length, in the `x‖s‖y_c‖y_d‖z_l‖z_u‖v_l‖v_u` packing.
struct ConjPair {
    /// `E`: multiplied into the RHS before the scaled-space solve.
    e: Vec<Number>,
    /// `F`: multiplied into the solution after the scaled-space solve.
    f: Vec<Number>,
}

impl PdSensBacksolver {
    /// Construct from the four handles handed in by the `on_converged`
    /// callback. Errors if `data` has no `curr` (i.e. the algorithm
    /// never reached an iterate — should not happen on
    /// `SolveSucceeded`) or the NLP reports scaling data inconsistent
    /// with the converged iterate (see [`Self::natural_units_conj`]).
    pub fn new(
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd: Rc<RefCell<PdFullSpaceSolver>>,
    ) -> Result<Self, String> {
        let curr = data
            .borrow()
            .curr
            .clone()
            .ok_or_else(|| "no current iterate at convergence".to_string())?;
        let dims = [
            curr.x.dim() as usize,
            curr.s.dim() as usize,
            curr.y_c.dim() as usize,
            curr.y_d.dim() as usize,
            curr.z_l.dim() as usize,
            curr.z_u.dim() as usize,
            curr.v_l.dim() as usize,
            curr.v_u.dim() as usize,
        ];
        let conj = Self::natural_units_conj(nlp, &dims)?;
        Ok(Self {
            pd,
            data: Rc::clone(data),
            cq: Rc::clone(cq),
            nlp: Rc::clone(nlp),
            dims,
            template: curr,
            conj,
        })
    }

    /// Build the natural-units scaling pair `(E, F)` from the NLP's
    /// effective scaling (see the field doc on [`Self::conj`]).
    /// Returns `Ok(None)` when no scaling is active. Errors when the
    /// NLP reports scaling data inconsistent with the converged
    /// iterate's block dimensions (would silently corrupt every
    /// back-solve) or a zero/non-finite `df`.
    fn natural_units_conj(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Result<Option<Rc<ConjPair>>, String> {
        let nlp_ref = nlp.borrow();
        let df = nlp_ref.obj_scaling_factor();
        let dc = nlp_ref.c_scale_vec();
        let dd = nlp_ref.d_scale_vec();
        if df == 1.0 && dc.is_none() && dd.is_none() {
            return Ok(None);
        }
        // df may be negative (obj_scaling_factor < 0 means maximize);
        // the two-sided scaling needs no square root, only df ≠ 0.
        if !df.is_finite() || df == 0.0 {
            return Err(format!("invalid obj_scaling_factor {df}"));
        }
        if let Some(v) = &dc {
            if v.len() != dims[2] {
                return Err(format!("c_scale length {} != y_c dim {}", v.len(), dims[2]));
            }
        }
        if let Some(v) = &dd {
            if v.len() != dims[3] || dims[1] != dims[3] {
                return Err(format!(
                    "d_scale length {} != y_d dim {} (s dim {})",
                    v.len(),
                    dims[3],
                    dims[1]
                ));
            }
        }
        // Per-entry d-row scale for the compressed v_l / v_u blocks:
        // entry j of v_l covers the d row pd_l.expanded_pos[j].
        let v_row_scale = |pm: Rc<dyn pounce_linalg::matrix::Matrix>,
                           n_v: usize,
                           which: &str|
         -> Result<Vec<Number>, String> {
            let Some(dd) = &dd else {
                return Ok(vec![1.0; n_v]);
            };
            if n_v == 0 {
                return Ok(Vec::new());
            }
            let Some(em) = pm
                .as_any()
                .downcast_ref::<pounce_linalg::expansion_matrix::ExpansionMatrix>()
            else {
                return Err(format!("{which} is not an ExpansionMatrix"));
            };
            let pos = em.expanded_pos_indices();
            if pos.len() != n_v {
                return Err(format!(
                    "{which} expansion length {} != {} block dim {}",
                    pos.len(),
                    which,
                    n_v
                ));
            }
            pos.iter()
                .map(|&r| {
                    dd.get(r as usize).copied().ok_or_else(|| {
                        format!(
                            "{which} expansion row {r} out of d_scale range {}",
                            dd.len()
                        )
                    })
                })
                .collect()
        };
        let vl_dd = v_row_scale(nlp_ref.pd_l(), dims[6], "pd_l")?;
        let vu_dd = v_row_scale(nlp_ref.pd_u(), dims[7], "pd_u")?;
        drop(nlp_ref);

        let total: usize = dims.iter().sum();
        let mut e = Vec::with_capacity(total);
        let mut f = Vec::with_capacity(total);
        // x block: E = df, F = 1.
        e.extend(std::iter::repeat_n(df, dims[0]));
        f.extend(std::iter::repeat_n(1.0, dims[0]));
        // s block: E = df/dd_i, F = 1/dd_i (slacks live in scaled d-space).
        match &dd {
            Some(v) => {
                e.extend(v.iter().map(|&ddi| df / ddi));
                f.extend(v.iter().map(|&ddi| 1.0 / ddi));
            }
            None => {
                e.extend(std::iter::repeat_n(df, dims[1]));
                f.extend(std::iter::repeat_n(1.0, dims[1]));
            }
        }
        // y_c block: E = dc_i, F = dc_i/df.
        match &dc {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&dci| dci / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[2]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[2]));
            }
        }
        // y_d block: E = dd_i, F = dd_i/df.
        match &dd {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&ddi| ddi / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[3]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[3]));
            }
        }
        // z_l / z_u blocks: E = df, F = 1/df (z̃ = df·z; bounds on x
        // are unscaled so the slack diagonal X − x_L is shared by both
        // systems).
        e.extend(std::iter::repeat_n(df, dims[4] + dims[5]));
        f.extend(std::iter::repeat_n(1.0 / df, dims[4] + dims[5]));
        // v_l / v_u blocks: E = df, F = dd_r/df (ṽ = (df/dd)·v and the
        // slack diagonal s̃ − d̃_l = dd·(s − d_l) carries the d-row
        // scale).
        e.extend(std::iter::repeat_n(df, dims[6] + dims[7]));
        f.extend(vl_dd.iter().map(|&ddr| ddr / df));
        f.extend(vu_dd.iter().map(|&ddr| ddr / df));
        Ok(Some(Rc::new(ConjPair { e, f })))
    }

    /// Effective objective scaling factor `df` of the converged NLP
    /// (1.0 when no scaling is active).
    pub fn obj_scaling_factor(&self) -> Number {
        self.nlp.borrow().obj_scaling_factor()
    }

    /// Effective NLP scaling at convergence:
    /// `(obj_scaling_factor, c_scale, d_scale)`. The vectors are
    /// `None` when the corresponding block carries no row scaling.
    pub fn nlp_scaling(&self) -> (Number, Option<Vec<Number>>, Option<Vec<Number>>) {
        let n = self.nlp.borrow();
        (n.obj_scaling_factor(), n.c_scale_vec(), n.d_scale_vec())
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor (the IPM's `current_perturbation`
    /// state at convergence). All zero ⇔ the final factorization was
    /// unregularized and the natural-units back-solves invert the
    /// exact KKT matrix. Nonzero ⇔ the factor carries a (scaled-space)
    /// regularization, so sensitivity outputs — covariance in
    /// particular — are perturbed and no longer exactly
    /// scaling-invariant; consumers should check this before trusting
    /// `-inv(reduced_hessian)` on ill-conditioned problems
    /// (pounce#128 follow-up).
    pub fn kkt_perturbations(&self) -> [Number; 4] {
        let p = self.data.borrow().perturbations;
        [p.delta_x, p.delta_s, p.delta_c, p.delta_d]
    }

    /// Map user-facing 0-based `g(x)` indices of parameter-pin
    /// equality constraints to flat KKT rows **and** the pin rows'
    /// `dc_i` scaling factors, in one pass. The KKT row of pin `i` is
    /// `n_x + n_s + c_block_idx`, i.e. the matching `y_c` slot, found
    /// through `IpoptNlp::full_g_to_c_block` so the c/d split's row
    /// permutation is honored (pounce#128 follow-up: the previous
    /// direct `n_x + n_s + g_idx` mapping silently picked wrong rows
    /// when inequalities preceded the pins). The scales are 1.0 when
    /// no constraint scaling is active; they relate the natural and
    /// solver-space reduced Hessians via
    /// `H̃_ij = (df / (dc_i·dc_j)) · H_ij`. Errors when a pin index
    /// is out of range or refers to an inequality row.
    pub fn pin_rows_and_c_scales(
        &self,
        pin_g_indices: &[Index],
    ) -> Result<(Vec<Index>, Vec<Number>), String> {
        let y_c_offset = (self.dims[0] + self.dims[1]) as Index;
        let nlp = self.nlp.borrow();
        let dc = nlp.c_scale_vec();
        let n_full_g = nlp.n_full_g();
        let mut rows = Vec::with_capacity(pin_g_indices.len());
        let mut scales = Vec::with_capacity(pin_g_indices.len());
        for &gi in pin_g_indices {
            // n_full_g() defaults to 0 for IpoptNlp impls that don't
            // report it; only range-check when it's meaningful.
            if gi < 0 || (n_full_g > 0 && gi >= n_full_g) {
                return Err(format!(
                    "pin constraint index {gi} out of range [0, m={n_full_g})"
                ));
            }
            let Some(ci) = nlp.full_g_to_c_block(gi) else {
                return Err(format!(
                    "pin constraint index {gi} is an inequality (not an equality row); \
                     parameter pins must be exact equalities"
                ));
            };
            rows.push(y_c_offset + ci);
            scales.push(dc.as_ref().map(|v| v[ci as usize]).unwrap_or(1.0));
        }
        Ok((rows, scales))
    }

    /// KKT-row half of [`Self::pin_rows_and_c_scales`].
    pub fn map_pin_g_to_kkt_rows(&self, pin_g_indices: &[Index]) -> Result<Vec<Index>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.0)
    }

    /// Scaling half of [`Self::pin_rows_and_c_scales`].
    pub fn pin_c_scales(&self, pin_g_indices: &[Index]) -> Result<Vec<Number>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.1)
    }

    /// Block dimensions of the compound KKT vector at convergence, in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order. Sum equals
    /// [`SensBacksolver::dim`]. Useful when a caller needs to compute
    /// the flat offset of a non-x block (e.g. `n_x + n_s` for the
    /// start of the equality-multiplier `y_c` block).
    pub fn block_dims(&self) -> [usize; 8] {
        self.dims
    }

    /// Cumulative block offsets: `offset(i)` is the start index of
    /// block `i` in the flat slice.
    fn offsets(&self) -> [usize; 9] {
        let mut o = [0usize; 9];
        for i in 0..8 {
            o[i + 1] = o[i] + self.dims[i];
        }
        o
    }

    /// Pack a flat slice into a freshly-allocated `IteratesVectorMut`
    /// shaped like the converged iterate.
    fn pack(&self, flat: &[Number]) -> Result<IteratesVectorMut, ()> {
        let mut out = self.template.make_new_zeroed();
        let off = self.offsets();
        let blocks: [&mut Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &mut out.x,
            &mut out.s,
            &mut out.y_c,
            &mut out.y_d,
            &mut out.z_l,
            &mut out.z_u,
            &mut out.v_l,
            &mut out.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let slice = &flat[off[i]..off[i + 1]];
            let dv = blk.as_any_mut().downcast_mut::<DenseVector>().ok_or(())?;
            dv.set_values(slice);
        }
        Ok(out)
    }

    /// Read an `IteratesVectorMut` into a flat slice. Uses
    /// [`DenseVector::expanded_values`] rather than `values()` so
    /// blocks that the IPM left in homogeneous-scalar form (typical
    /// for empty z_l/z_u/v_l/v_u when the TNLP has no bounds) are
    /// materialized rather than panicking.
    fn unpack(&self, iv: &IteratesVectorMut, out: &mut [Number]) -> Result<(), ()> {
        let off = self.offsets();
        let blocks: [&Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &iv.x, &iv.s, &iv.y_c, &iv.y_d, &iv.z_l, &iv.z_u, &iv.v_l, &iv.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let dst = &mut out[off[i]..off[i + 1]];
            if dst.is_empty() {
                continue;
            }
            let dv = (**blk).as_any().downcast_ref::<DenseVector>().ok_or(())?;
            let ev = dv.expanded_values();
            dst.copy_from_slice(&ev);
        }
        Ok(())
    }
}

impl PdSensBacksolver {
    /// Batched-RHS back-solve over the held factor. `rhs_flat` and
    /// `lhs_flat` are row-major `(n_rhs, dim)` buffers. Equivalent to
    /// looping [`SensBacksolver::solve`] over each row but reuses one
    /// frozen `IteratesVector` for the RHS and one `IteratesVectorMut`
    /// for the result across all `n_rhs` calls into
    /// [`PdFullSpaceSolver::solve`]. The pack step writes into the
    /// existing `DenseVector` storage via `Rc::get_mut` +
    /// `set_values`, and the unpack step reads it back via `values()`
    /// /`scalar()` — skipping the per-call 8-block `make_new_zeroed`
    /// (Box alloc) in `pack` and the per-block `expanded_values()` Vec
    /// alloc in `unpack` that otherwise dominate the held-factor
    /// back-solve cost under `jax.jacrev` over a JaxProblem solve
    /// (pounce#77 follow-up).
    ///
    /// The matrix and perturbation state inside `PdFullSpaceSolver`
    /// are unchanged across calls, so each iteration hits the cached
    /// fast path in `solve_once` (`uptodate && !pretend_singular`).
    ///
    /// Like [`SensBacksolver::solve`], results are in **natural
    /// (unscaled) units** — see [`Self::solve_many_scaled_space`] for
    /// the raw solver-space back-solve.
    pub fn solve_many(&self, rhs_flat: &[Number], lhs_flat: &mut [Number], n_rhs: usize) -> bool {
        match &self.conj {
            None => self.solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs),
            Some(c) => {
                let total = self.dim();
                if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
                    return false;
                }
                let mut rhs_scaled = rhs_flat.to_vec();
                for row in rhs_scaled.chunks_mut(total) {
                    for (r, &ei) in row.iter_mut().zip(c.e.iter()) {
                        *r *= ei;
                    }
                }
                if !self.solve_many_scaled_space(&rhs_scaled, lhs_flat, n_rhs) {
                    return false;
                }
                for row in lhs_flat.chunks_mut(total) {
                    for (l, &fi) in row.iter_mut().zip(c.f.iter()) {
                        *l *= fi;
                    }
                }
                true
            }
        }
    }

    /// Batched-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). Same
    /// buffer contract as [`Self::solve_many`].
    pub fn solve_many_scaled_space(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> bool {
        let total = self.dim();
        if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
            return false;
        }
        if n_rhs == 0 {
            return true;
        }
        let off = self.offsets();

        // Tier 1: fully-inline flat-slice path. `PdFullSpaceSolver::
        // solve_many_cached_flat` downcasts the slack / z / v vectors to
        // `DenseVector` and the bound-expansion matrices to
        // `ExpansionMatrix` once at the top, then runs Phase 1 / Phase 3
        // as raw scatter-add / divide loops on flat slices with no dyn
        // dispatch in the per-RHS inner loops. Returns `None` if a
        // downcast fails (homogeneous-on-non-empty block, unusual matrix
        // type) — we fall to Tier 2.
        {
            let mut pd_ref = self.pd.borrow_mut();
            let fast_flat = pd_ref.solve_many_cached_flat(
                &self.data, &self.cq, &self.nlp, n_rhs, rhs_flat, lhs_flat, self.dims,
            );
            match fast_flat {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to Tier 2 */ }
            }
        }

        // Tier 2: closure-based cached-factor path. Same single
        // back-substitution through the linsol, but Phase 1 / Phase 3
        // go through `dyn Vector` / `dyn Matrix` ops on a per-RHS
        // `IteratesVectorMut`. Slower than Tier 1 but correct for
        // homogeneous DenseVectors and non-`ExpansionMatrix` bound
        // expansions.
        {
            let mut pd_ref = self.pd.borrow_mut();
            let fast = pd_ref.solve_many_cached(
                &self.data,
                &self.cq,
                &self.nlp,
                n_rhs,
                |k, iv| {
                    let row = &rhs_flat[k * total..(k + 1) * total];
                    let _ = write_rhs_box(&mut iv.x, &row[off[0]..off[1]])
                        && write_rhs_box(&mut iv.s, &row[off[1]..off[2]])
                        && write_rhs_box(&mut iv.y_c, &row[off[2]..off[3]])
                        && write_rhs_box(&mut iv.y_d, &row[off[3]..off[4]])
                        && write_rhs_box(&mut iv.z_l, &row[off[4]..off[5]])
                        && write_rhs_box(&mut iv.z_u, &row[off[5]..off[6]])
                        && write_rhs_box(&mut iv.v_l, &row[off[6]..off[7]])
                        && write_rhs_box(&mut iv.v_u, &row[off[7]..off[8]]);
                },
                |k, iv| {
                    let row = &mut lhs_flat[k * total..(k + 1) * total];
                    let _ = read_res_block(&*iv.x, &mut row[off[0]..off[1]])
                        && read_res_block(&*iv.s, &mut row[off[1]..off[2]])
                        && read_res_block(&*iv.y_c, &mut row[off[2]..off[3]])
                        && read_res_block(&*iv.y_d, &mut row[off[3]..off[4]])
                        && read_res_block(&*iv.z_l, &mut row[off[4]..off[5]])
                        && read_res_block(&*iv.z_u, &mut row[off[5]..off[6]])
                        && read_res_block(&*iv.v_l, &mut row[off[6]..off[7]])
                        && read_res_block(&*iv.v_u, &mut row[off[7]..off[8]]);
                },
            );
            match fast {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to per-RHS loop */ }
            }
        }

        // Per-RHS fallback: reuse one frozen rhs and one mut sol across
        // all n_rhs `solve` calls.
        let rhs_mut0 = self.template.make_new_zeroed();
        let mut rhs_iv = rhs_mut0.freeze();
        let mut res_iv = self.template.make_new_zeroed();

        let mut pd_ref = self.pd.borrow_mut();
        for k in 0..n_rhs {
            let rhs_row = &rhs_flat[k * total..(k + 1) * total];
            let lhs_row = &mut lhs_flat[k * total..(k + 1) * total];

            if !write_rhs_block(&mut rhs_iv.x, &rhs_row[off[0]..off[1]])
                || !write_rhs_block(&mut rhs_iv.s, &rhs_row[off[1]..off[2]])
                || !write_rhs_block(&mut rhs_iv.y_c, &rhs_row[off[2]..off[3]])
                || !write_rhs_block(&mut rhs_iv.y_d, &rhs_row[off[3]..off[4]])
                || !write_rhs_block(&mut rhs_iv.z_l, &rhs_row[off[4]..off[5]])
                || !write_rhs_block(&mut rhs_iv.z_u, &rhs_row[off[5]..off[6]])
                || !write_rhs_block(&mut rhs_iv.v_l, &rhs_row[off[6]..off[7]])
                || !write_rhs_block(&mut rhs_iv.v_u, &rhs_row[off[7]..off[8]])
            {
                return false;
            }

            let ok = pd_ref.solve(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
            );
            if !ok {
                return false;
            }

            if !read_res_block(&*res_iv.x, &mut lhs_row[off[0]..off[1]])
                || !read_res_block(&*res_iv.s, &mut lhs_row[off[1]..off[2]])
                || !read_res_block(&*res_iv.y_c, &mut lhs_row[off[2]..off[3]])
                || !read_res_block(&*res_iv.y_d, &mut lhs_row[off[3]..off[4]])
                || !read_res_block(&*res_iv.z_l, &mut lhs_row[off[4]..off[5]])
                || !read_res_block(&*res_iv.z_u, &mut lhs_row[off[5]..off[6]])
                || !read_res_block(&*res_iv.v_l, &mut lhs_row[off[6]..off[7]])
                || !read_res_block(&*res_iv.v_u, &mut lhs_row[off[7]..off[8]])
            {
                return false;
            }
        }
        true
    }
}

/// Write `slice` into the `DenseVector` behind `b` in place. Used by
/// the fast path's `write_rhs` closure, where the new
/// `PdFullSpaceSolver::solve_many_cached` API hands back an
/// `IteratesVectorMut` (Box-backed blocks).
fn write_rhs_box(b: &mut Box<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(dv) = b.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Write `slice` into the `DenseVector` behind `rc` in place. Returns
/// `false` if the Rc is unexpectedly shared (would indicate a bug in
/// `PdFullSpaceSolver::solve`'s borrow discipline — it should never
/// `Rc::clone` from the rhs vector) or if the block is not a
/// `DenseVector`.
fn write_rhs_block(rc: &mut Rc<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(v) = Rc::get_mut(rc) else {
        return false;
    };
    let Some(dv) = v.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Read the `DenseVector` behind `blk` into `dst`. Handles the
/// homogeneous case (empty z/v blocks for a TNLP with no bounds) by
/// broadcasting the scalar rather than calling `expanded_values()`,
/// which would allocate a fresh `Vec<Number>` every call.
fn read_res_block(blk: &dyn pounce_linalg::vector::Vector, dst: &mut [Number]) -> bool {
    if dst.is_empty() {
        return true;
    }
    let Some(dv) = blk.as_any().downcast_ref::<DenseVector>() else {
        return false;
    };
    if dv.is_homogeneous() {
        let s = dv.scalar();
        for x in dst.iter_mut() {
            *x = s;
        }
    } else {
        dst.copy_from_slice(dv.values());
    }
    true
}

impl PdSensBacksolver {
    /// Single-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). This
    /// is the value [`SensBacksolver::solve`] returned before
    /// pounce#128; kept for callers that want the raw factor.
    pub fn solve_scaled_space(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        let total = self.dim();
        if rhs.len() != total || lhs.len() != total {
            return false;
        }
        // Pack rhs into block form.
        let rhs_mut = match self.pack(rhs) {
            Ok(v) => v,
            Err(()) => return false,
        };
        let rhs_iv = rhs_mut.freeze();
        // Fresh result slot, zeroed.
        let mut res_iv = self.template.make_new_zeroed();

        // K · lhs = rhs   ⇒   solve(α=1, β=0, rhs, res) writes
        // res = K⁻¹ · rhs.
        //
        // `allow_inexact=true` mirrors upstream sIPOPT's
        // `SensSimpleBacksolver`: skip `PdFullSpaceSolver`'s iterative-
        // refinement loop and accept the first back-solve against the
        // held factor. The IPM-level refinement (`min_refinement_steps
        // = 1`, residual_ratio_max = 1e-10`) is there to clean up
        // numerical noise during forward IPM steps; for the held-factor
        // back-solve used by sens / JaxProblem bwd, it ~doubles the
        // per-call cost and produces gains that are below `tol`. Under
        // `jax.jacrev` over a JaxProblem solve this dominates the wall
        // time at moderate `n+m` (pounce#77 follow-up).
        let ok = {
            let mut pd_ref = self.pd.borrow_mut();
            pd_ref.solve(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
            )
        };
        if !ok {
            return false;
        }
        self.unpack(&res_iv, lhs).is_ok()
    }
}

impl SensBacksolver for PdSensBacksolver {
    fn dim(&self) -> usize {
        self.dims.iter().sum()
    }

    /// Solve `K · lhs = rhs` against the converged factor, in
    /// **natural (unscaled) units** (pounce#128): when the NLP carries
    /// active scaling (`nlp_scaling_method`, `obj_scaling_factor`,
    /// user scaling) the RHS is pre-multiplied by `E` and the result
    /// post-multiplied by `F` (see the `conj` field doc), so
    /// `lhs = K_natural⁻¹ rhs` for **all eight blocks** — including
    /// the z/v bound-multiplier rows. Use
    /// [`Self::solve_scaled_space`] for the raw factor.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        match &self.conj {
            None => self.solve_scaled_space(rhs, lhs),
            Some(c) => {
                let total = self.dim();
                if rhs.len() != total || lhs.len() != total {
                    return false;
                }
                let rhs_scaled: Vec<Number> =
                    rhs.iter().zip(c.e.iter()).map(|(&r, &ei)| r * ei).collect();
                if !self.solve_scaled_space(&rhs_scaled, lhs) {
                    return false;
                }
                for (l, &fi) in lhs.iter_mut().zip(c.f.iter()) {
                    *l *= fi;
                }
                true
            }
        }
    }
}
