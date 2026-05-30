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
use pounce_common::types::Number;
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
}

impl PdSensBacksolver {
    /// Construct from the four handles handed in by the `on_converged`
    /// callback. Returns `Err(())` if `data` has no `curr` (i.e. the
    /// algorithm never reached an iterate — should not happen on
    /// `SolveSucceeded`).
    pub fn new(
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd: Rc<RefCell<PdFullSpaceSolver>>,
    ) -> Result<Self, ()> {
        let curr = data.borrow().curr.clone().ok_or(())?;
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
        Ok(Self {
            pd,
            data: Rc::clone(data),
            cq: Rc::clone(cq),
            nlp: Rc::clone(nlp),
            dims,
            template: curr,
        })
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
    pub fn solve_many(&self, rhs_flat: &[Number], lhs_flat: &mut [Number], n_rhs: usize) -> bool {
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
                &self.data,
                &self.cq,
                &self.nlp,
                n_rhs,
                rhs_flat,
                lhs_flat,
                self.dims,
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
fn write_rhs_box(
    b: &mut Box<dyn pounce_linalg::vector::Vector>,
    slice: &[Number],
) -> bool {
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
fn write_rhs_block(
    rc: &mut Rc<dyn pounce_linalg::vector::Vector>,
    slice: &[Number],
) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(v) = Rc::get_mut(rc) else { return false };
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

impl SensBacksolver for PdSensBacksolver {
    fn dim(&self) -> usize {
        self.dims.iter().sum()
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
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
