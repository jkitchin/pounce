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

/// Adapter from `PdFullSpaceSolver` to [`SensBacksolver`]. Borrows
/// the four pieces of the algorithm's converged state, plus the
/// 8-block iterate template used to allocate fresh RHS / LHS vectors.
///
/// The PD solver lives behind a [`RefCell`] because
/// [`SensBacksolver::solve`] is `&self` but the upstream signature
/// for `PdFullSpaceSolver::solve` is `&mut self` (it caches the
/// last-solve dependency tags and the augsys-improved flag). The
/// `RefCell` is single-thread-only, single-borrow, exactly matching
/// the call pattern from `pounce-sensitivity`'s pipeline.
#[derive(Clone)]
pub struct PdSensBacksolver<'a> {
    /// Shared, interior-mutable handle to the converged PD solver.
    /// `Rc<RefCell<…>>` (rather than a bare `RefCell<&mut …>`) so the
    /// adapter is `Clone` — [`SensApplication`] internally clones the
    /// backsolver to hand it to `IndexPCalculator`, and the
    /// reduced-Hessian path may run multiple Schur builds against the
    /// same factor.
    pd: Rc<RefCell<&'a mut PdFullSpaceSolver>>,
    data: &'a IpoptDataHandle,
    cq: &'a IpoptCqHandle,
    nlp: &'a Rc<RefCell<dyn IpoptNlp>>,
    /// Block dimensions in `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    dims: [usize; 8],
    /// 8-block prototype used to mint fresh vectors with the same
    /// `VectorSpace`s as the converged iterate; cloned from
    /// `data.borrow().curr`.
    template: IteratesVector,
}

impl<'a> PdSensBacksolver<'a> {
    /// Construct from the four handles handed in by the `on_converged`
    /// callback. Returns `Err(())` if `data` has no `curr` (i.e. the
    /// algorithm never reached an iterate — should not happen on
    /// `SolveSucceeded`).
    pub fn new(
        data: &'a IpoptDataHandle,
        cq: &'a IpoptCqHandle,
        nlp: &'a Rc<RefCell<dyn IpoptNlp>>,
        pd: &'a mut PdFullSpaceSolver,
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
            pd: Rc::new(RefCell::new(pd)),
            data,
            cq,
            nlp,
            dims,
            template: curr,
        })
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
            &mut out.x, &mut out.s, &mut out.y_c, &mut out.y_d,
            &mut out.z_l, &mut out.z_u, &mut out.v_l, &mut out.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let slice = &flat[off[i]..off[i + 1]];
            let dv = blk
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .ok_or(())?;
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

impl<'a> SensBacksolver for PdSensBacksolver<'a> {
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
        let ok = {
            let mut pd_ref = self.pd.borrow_mut();
            pd_ref.solve(
                self.data,
                self.cq,
                self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                false,
                false,
            )
        };
        if !ok {
            return false;
        }
        self.unpack(&res_iv, lhs).is_ok()
    }
}
