//! PD search-direction calculator — port of
//! `Algorithm/IpPDSearchDirCalc.{hpp,cpp}`.
//!
//! Builds the right-hand side from the current iterate's KKT
//! residuals (gradient of Lagrangian, constraint values, relaxed
//! complementarities), optionally adds a Mehrotra corrector, then
//! calls `PdFullSpaceSolver::solve` to produce the search direction
//! `delta`.
//!
//! Two RHS modes:
//! * standard: z-blocks are the relaxed complementarities
//!   `s_L · z_L − μ`, …
//! * Mehrotra: z-blocks include the second-order term
//!   `(P_L^T Δx_aff) · Δz_aff_L + (s_L · z_L − μ)`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::{IteratesVector, IteratesVectorMut};
use crate::kkt::pd_full_space_solver::PdFullSpaceSolver;
use crate::kkt::search_dir_calc::SearchDirCalculator;
use pounce_common::types::Number;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;

pub struct PdSearchDirCalc {
    /// Owned via `Rc<RefCell<…>>` so external callers (e.g. the
    /// post-converged sensitivity callback) can retain a cloned handle
    /// past the IPM call. During the IPM loop refcount is 1 and every
    /// internal call goes through `borrow_mut`; the runtime borrow
    /// check costs are negligible relative to the linear solve.
    pd_solver: Rc<RefCell<PdFullSpaceSolver>>,
    /// Skip the residual check on the search direction. Mirrors
    /// `fast_step_computation` (default false).
    pub fast_step_computation: bool,
    /// Mehrotra-style predictor-corrector step. Mirrors
    /// `mehrotra_algorithm` (default false in v1.0; flipped on by
    /// the adaptive-mu wiring in Phase 10).
    pub mehrotra_algorithm: bool,
}

impl PdSearchDirCalc {
    pub fn new(pd_solver: PdFullSpaceSolver) -> Self {
        Self {
            pd_solver: Rc::new(RefCell::new(pd_solver)),
            fast_step_computation: false,
            mehrotra_algorithm: false,
        }
    }

    /// Clone the shared handle to the PD solver. Used by the
    /// post-converged sensitivity callback to retain a factor handle
    /// past the IPM call.
    pub fn pd_solver_rc(&self) -> Rc<RefCell<PdFullSpaceSolver>> {
        Rc::clone(&self.pd_solver)
    }

    /// Borrow the PD solver mutably. Caller is responsible for not
    /// holding two mutable borrows at once (single-thread, single-
    /// borrow access pattern — matches every existing call site).
    pub fn pd_solver_mut(&self) -> RefMut<'_, PdFullSpaceSolver> {
        self.pd_solver.borrow_mut()
    }

    /// Compute the search direction and write it back into
    /// `data.delta`. Returns `false` if the underlying linear solve
    /// fails. Mirrors `PDSearchDirCalculator::ComputeSearchDirection`.
    pub fn compute_search_direction(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
    ) -> bool {
        let improve_solution = data.borrow().delta.is_some();

        if improve_solution && self.fast_step_computation {
            return true;
        }

        let curr = {
            let d = data.borrow();
            d.curr
                .clone()
                .unwrap_or_else(|| panic!("PdSearchDirCalc: IpoptData::curr is unset"))
        };

        // Build RHS.
        let mut rhs = curr.make_new_zeroed();
        {
            let cq_ref = cq.borrow();
            rhs.x.copy(&*cq_ref.curr_grad_lag_with_damping_x());
            rhs.s.copy(&*cq_ref.curr_grad_lag_with_damping_s());
            rhs.y_c.copy(&*cq_ref.curr_c());
            rhs.y_d.copy(&*cq_ref.curr_d_minus_s());
        }

        let nbounds = {
            let n = nlp.borrow();
            n.x_l().dim() + n.x_u().dim() + n.d_l().dim() + n.d_u().dim()
        };

        if nbounds > 0 && self.mehrotra_algorithm {
            let delta_aff = {
                let d = data.borrow();
                d.delta_aff
                    .clone()
                    .unwrap_or_else(|| panic!("PdSearchDirCalc: delta_aff missing for Mehrotra"))
            };
            self.fill_mehrotra_z_blocks(&delta_aff, cq, nlp, &mut rhs);
        } else {
            let cq_ref = cq.borrow();
            rhs.z_l.copy(&*cq_ref.curr_relaxed_compl_x_l());
            rhs.z_u.copy(&*cq_ref.curr_relaxed_compl_x_u());
            rhs.v_l.copy(&*cq_ref.curr_relaxed_compl_s_l());
            rhs.v_u.copy(&*cq_ref.curr_relaxed_compl_s_u());
        }

        let frozen_rhs = rhs.freeze();

        // Allocate the search direction. If we are improving an
        // existing one, seed it with `−delta` (per upstream).
        let mut delta = frozen_rhs.make_new_zeroed();
        if improve_solution {
            let prev = {
                let d = data.borrow();
                let Some(p) = d.delta.clone() else {
                    unreachable!("PdSearchDirCalc: delta cleared between is_some() and clone()")
                };
                p
            };
            delta.add_one_vector(-1.0, &prev, 0.0);
        }

        let allow_inexact = self.fast_step_computation;
        let ok = self.pd_solver.borrow_mut().solve(
            data,
            cq,
            nlp,
            -1.0,
            0.0,
            &frozen_rhs,
            &mut delta,
            allow_inexact,
            improve_solution,
        );

        if ok {
            data.borrow_mut().set_delta(delta.freeze());
        }
        ok
    }

    /// Affine (predictor) step — port of upstream's
    /// `IpAdaptiveMuUpdate::ComputeMuMehrotra` predictor solve. Builds
    /// the same RHS as [`Self::compute_search_direction`] except the
    /// z-blocks use the *unrelaxed* complementarity `s · z`
    /// (μ-target = 0) so the resulting step targets the affine-scaling
    /// system. The solution is stored in `data.delta_aff` for
    /// consumption by the Probing / Quality-Function oracles.
    ///
    /// Returns `false` if the linear solve fails.
    pub fn compute_affine_step(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
    ) -> bool {
        let curr = {
            let d = data.borrow();
            d.curr
                .clone()
                .unwrap_or_else(|| panic!("PdSearchDirCalc: IpoptData::curr is unset"))
        };

        let mut rhs = curr.make_new_zeroed();
        {
            let cq_ref = cq.borrow();
            // Upstream `IpQualityFunctionMuOracle.cpp:193-200` uses the
            // *plain* `curr_grad_lag_{x,s}` here, NOT the damped variant.
            // The `μ·κ_d·(P_L − P_U)` damping enters the main-step RHS
            // only — for the affine (predictor) RHS upstream wants the
            // gradient at μ=0.
            rhs.x.copy(&*cq_ref.curr_grad_lag_x());
            rhs.s.copy(&*cq_ref.curr_grad_lag_s());
            rhs.y_c.copy(&*cq_ref.curr_c());
            rhs.y_d.copy(&*cq_ref.curr_d_minus_s());
            // Affine RHS: complementarity blocks use `s·z` (μ=0),
            // not `s·z − μ`.
            rhs.z_l.copy(&*cq_ref.curr_compl_x_l());
            rhs.z_u.copy(&*cq_ref.curr_compl_x_u());
            rhs.v_l.copy(&*cq_ref.curr_compl_s_l());
            rhs.v_u.copy(&*cq_ref.curr_compl_s_u());
        }

        let frozen_rhs = rhs.freeze();
        let mut delta_aff = frozen_rhs.make_new_zeroed();

        // Upstream `IpQualityFunctionMuOracle.cpp:208` passes
        // `allow_inexact = true`. Pounce keeps full iterative
        // refinement here (allow_inexact=false): an earlier attempt to
        // set this to `true` regressed TRO3X3 from Solve_Succeeded to
        // Infeasible_Problem_Detected, because pounce's IR-driven
        // `increase_quality()` cascade produces materially different
        // steps than upstream's single-shot MA57. Leaving as-is until
        // the MA57 backend lands in Phase 4.
        let ok = self.pd_solver.borrow_mut().solve(
            data,
            cq,
            nlp,
            -1.0,
            0.0,
            &frozen_rhs,
            &mut delta_aff,
            false,
            false,
        );

        if ok {
            data.borrow_mut().set_delta_aff(delta_aff.freeze());
        }
        ok
    }

    /// Pure centering step — port of upstream
    /// `IpQualityFunctionMuOracle.cpp::CalculateMu` lines 218-247. RHS
    /// is `(0, 0, 0, 0, μ̄·1, μ̄·1, μ̄·1, μ̄·1)` with μ̄ = `curr_avrg_compl`.
    /// Solution stored on `data.delta_cen` for the quality-function
    /// oracle's σ-bracket search.
    ///
    /// Returns `false` if the linear solve fails.
    pub fn compute_centering_step(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
    ) -> bool {
        let curr = {
            let d = data.borrow();
            d.curr
                .clone()
                .unwrap_or_else(|| panic!("PdSearchDirCalc: IpoptData::curr is unset"))
        };
        let avrg_compl = cq.borrow().curr_avrg_compl();

        let mut rhs = curr.make_new_zeroed();
        // x/s blocks: -avrg_compl · grad_kappa_times_damping_{x,s}, per
        // upstream IpQualityFunctionMuOracle.cpp:229-230. With kappa_d=0
        // (the default) these are zero, but kappa_d=1e-5 (default) makes
        // them nonzero on damped components and the centering direction
        // depends on them.
        {
            let cq_ref = cq.borrow();
            rhs.x
                .add_one_vector(-avrg_compl, &*cq_ref.grad_kappa_times_damping_x(), 0.0);
            rhs.s
                .add_one_vector(-avrg_compl, &*cq_ref.grad_kappa_times_damping_s(), 0.0);
        }
        rhs.y_c.set(0.0);
        rhs.y_d.set(0.0);
        rhs.z_l.set(avrg_compl);
        rhs.z_u.set(avrg_compl);
        rhs.v_l.set(avrg_compl);
        rhs.v_u.set(avrg_compl);

        let frozen_rhs = rhs.freeze();
        let mut delta_cen = frozen_rhs.make_new_zeroed();

        // Upstream `IpQualityFunctionMuOracle.cpp:243` passes
        // `allow_inexact = true`. Same caveat as `compute_affine_step`
        // — flipping this on regresses TRO3X3 because the FERAL-backed
        // IR cascade differs from MA57's single-shot. Defer until MA57.
        let ok = self.pd_solver.borrow_mut().solve(
            data,
            cq,
            nlp,
            1.0,
            0.0,
            &frozen_rhs,
            &mut delta_cen,
            false,
            false,
        );

        if ok {
            data.borrow_mut().set_delta_cen(delta_cen.freeze());
        }
        ok
    }

    /// Solve the second-order-correction (SOC) linear system used by
    /// the filter line search to recover full-step acceptability when
    /// the Newton step grows the constraint violation. Mirrors the RHS
    /// assembly + `pd_solver_->Solve(-1.0, 0.0, ...)` block in upstream
    /// `IpFilterLSAcceptor.cpp:577-608`.
    ///
    /// The caller supplies the SOC right-hand sides for the equality and
    /// inequality blocks (`c_soc`, `dms_soc`); this method assembles the
    /// remaining six blocks using the current iterate's KKT residuals
    /// and returns the resulting `delta_soc`. `soc_method = 0` matches
    /// upstream's default (gradient blocks unscaled); `soc_method = 1`
    /// scales the gradient blocks by `alpha_primal_soc` to reuse a
    /// previously-tried correction.
    pub fn compute_soc_step(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        c_soc: &dyn pounce_linalg::Vector,
        dms_soc: &dyn pounce_linalg::Vector,
        alpha_primal_soc: Number,
        soc_method: i32,
    ) -> Option<IteratesVector> {
        let curr = {
            let d = data.borrow();
            d.curr
                .clone()
                .unwrap_or_else(|| panic!("PdSearchDirCalc::compute_soc_step: curr is unset"))
        };
        let mut rhs = curr.make_new_zeroed();
        {
            let cq_ref = cq.borrow();
            rhs.x.copy(&*cq_ref.curr_grad_lag_with_damping_x());
            rhs.s.copy(&*cq_ref.curr_grad_lag_with_damping_s());
            if soc_method == 1 {
                rhs.x.scal(alpha_primal_soc);
                rhs.s.scal(alpha_primal_soc);
            }
            rhs.y_c.copy(c_soc);
            rhs.y_d.copy(dms_soc);
            rhs.z_l.copy(&*cq_ref.curr_relaxed_compl_x_l());
            rhs.z_u.copy(&*cq_ref.curr_relaxed_compl_x_u());
            rhs.v_l.copy(&*cq_ref.curr_relaxed_compl_s_l());
            rhs.v_u.copy(&*cq_ref.curr_relaxed_compl_s_u());
        }
        let frozen_rhs = rhs.freeze();
        let mut delta_soc = frozen_rhs.make_new_zeroed();
        let ok = self.pd_solver.borrow_mut().solve(
            data,
            cq,
            nlp,
            -1.0,
            0.0,
            &frozen_rhs,
            &mut delta_soc,
            false,
            false,
        );
        if ok {
            Some(delta_soc.freeze())
        } else {
            None
        }
    }

    /// Mehrotra z-block:
    ///   tmp_zL =  P_L^T · Δx_aff;  tmp_zL ⊙= Δz_aff_L;  tmp_zL += relaxed_compl_x_L
    /// Symmetric for the U / s blocks.
    fn fill_mehrotra_z_blocks(
        &self,
        delta_aff: &IteratesVector,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        rhs: &mut IteratesVectorMut,
    ) {
        let n = nlp.borrow();
        let cq_ref = cq.borrow();

        // z_L
        n.px_l()
            .trans_mult_vector(1.0, &*delta_aff.x, 0.0, &mut *rhs.z_l);
        rhs.z_l.element_wise_multiply(&*delta_aff.z_l);
        rhs.z_l.axpy(1.0, &*cq_ref.curr_relaxed_compl_x_l());

        // z_U
        n.px_u()
            .trans_mult_vector(-1.0, &*delta_aff.x, 0.0, &mut *rhs.z_u);
        rhs.z_u.element_wise_multiply(&*delta_aff.z_u);
        rhs.z_u.axpy(1.0, &*cq_ref.curr_relaxed_compl_x_u());

        // v_L
        n.pd_l()
            .trans_mult_vector(1.0, &*delta_aff.s, 0.0, &mut *rhs.v_l);
        rhs.v_l.element_wise_multiply(&*delta_aff.v_l);
        rhs.v_l.axpy(1.0, &*cq_ref.curr_relaxed_compl_s_l());

        // v_U
        n.pd_u()
            .trans_mult_vector(-1.0, &*delta_aff.s, 0.0, &mut *rhs.v_u);
        rhs.v_u.element_wise_multiply(&*delta_aff.v_u);
        rhs.v_u.axpy(1.0, &*cq_ref.curr_relaxed_compl_s_u());
    }
}

impl SearchDirCalculator for PdSearchDirCalc {}

// --- per-element helpers retained from the Phase-6 stub for
// downstream callers (CG-penalty path, restoration RHS unit tests).
// Not used by `compute_search_direction` itself.

pub fn mehrotra_corrector_lower(
    delta_aff_x_lo: Number,
    delta_aff_z: Number,
    relaxed_compl: Number,
) -> Number {
    delta_aff_x_lo * delta_aff_z + relaxed_compl
}

pub fn mehrotra_corrector_upper(
    delta_aff_x_up: Number,
    delta_aff_z: Number,
    relaxed_compl: Number,
) -> Number {
    -delta_aff_x_up * delta_aff_z + relaxed_compl
}

pub fn relaxed_complementarity(x: Number, z: Number, mu: Number) -> Number {
    x * z - mu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_compl_at_central_path_is_zero() {
        assert_eq!(relaxed_complementarity(2.0, 0.5, 1.0), 0.0);
    }

    #[test]
    fn mehrotra_lower_combines_linearly() {
        assert_eq!(mehrotra_corrector_lower(1.0, 2.0, 0.5), 2.5);
    }

    #[test]
    fn mehrotra_upper_negates_dx() {
        assert_eq!(mehrotra_corrector_upper(1.0, 2.0, 0.5), -1.5);
    }
}
