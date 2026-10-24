//! pounce's own converged iterate behind [`BarrierIterate`], the trait the
//! corrector and the activity classification in `pounce-sens-core` read.
//!
//! Everything here reads `IpoptData`, the calculated quantities and the NLP
//! of the solve the backsolver holds, in the scaled frame the algorithm
//! iterated in. The arithmetic of both analyses lives in `pounce-sens-core`.

use std::rc::Rc;

use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_common::types::{Index, Number};
use pounce_linalg::Matrix;
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use pounce_sens_core::barrier::{
    ActivityData, BarrierIterate, BoundSide, EngineScaling, KktOperator, ModelMaps,
};

use crate::algorithm_backsolver::{CorrectorSigma, PdSensBacksolver};
use crate::backsolver::SensBacksolver;
use crate::vec_util::dense_to_vec;

impl BarrierIterate for PdSensBacksolver {
    fn block_dims(&self) -> [usize; 8] {
        PdSensBacksolver::block_dims(self)
    }

    fn offsets(&self) -> [usize; 9] {
        self.offsets_public()
    }

    fn barrier_mu(&self) -> Number {
        PdSensBacksolver::barrier_mu(self)
    }

    fn converged_iterate(&self, out: &mut [Number]) -> Result<(), String> {
        self.curr_flat(out)
            .map_err(|_| "the solve holds no current iterate of that length".to_string())
    }

    fn scaled_rhs_factor(&self) -> Option<&[Number]> {
        PdSensBacksolver::scaled_rhs_factor(self)
    }

    /// Installs `point` as the trial iterate and reads the residual off the
    /// calculated quantities there, so scaling, fixed variables and the bound
    /// expansions are handled exactly as the solve handles them. Setting a
    /// trial point leaves `curr` alone.
    fn barrier_residual(&self, point: &[Number], out: &mut [Number]) -> Result<(), String> {
        let dim = self.dim();
        if point.len() != dim || out.len() != dim {
            return Err(format!(
                "barrier residual: point of length {} and out of length {} for a KKT system of {dim}",
                point.len(),
                out.len()
            ));
        }
        let iv = self
            .pack_public(point)
            .map_err(|_| "corrector: pack failed".to_string())?;
        let (data, cq, _) = self.activity_handles();
        data.borrow_mut().set_trial(iv.freeze());

        let off = self.offsets_public();
        let cqb = cq.borrow();
        let blocks: [Rc<dyn pounce_linalg::Vector>; 8] = [
            cqb.trial_grad_lag_x(),
            cqb.trial_grad_lag_s(),
            cqb.trial_c(),
            cqb.trial_d_minus_s(),
            cqb.trial_compl_x_l(),
            cqb.trial_compl_x_u(),
            cqb.trial_compl_s_l(),
            cqb.trial_compl_s_u(),
        ];
        for (i, v) in blocks.into_iter().enumerate() {
            let vals = dense_to_vec(&*v);
            let (a, b) = (off[i], off[i + 1]);
            if vals.len() != b - a {
                return Err(format!(
                    "corrector: block {i} is {} long, expected {}",
                    vals.len(),
                    b - a
                ));
            }
            out[a..b].copy_from_slice(&vals);
        }
        Ok(())
    }

    /// Swaps the predicted point in as the current iterate for the life of
    /// the returned operator, so every block of the operator evaluates there,
    /// and builds the barrier diagonal after the swap. The stored base-point
    /// diagonal, which the ceiling (gh#737) or crossover (gh#654) would
    /// otherwise freeze, is never consulted: the frame rule and the ceiling
    /// are re-derived at the predicted point. See `corrector_sigma`.
    ///
    /// The solve takes its Hessian from `IpoptData::w`, the matrix the
    /// algorithm last stored, so moving the iterate alone moves only the
    /// Jacobians and slacks. Under an exact-Hessian solve the matrix is
    /// re-evaluated at the predicted point, with the step's own multipliers.
    /// A limited-memory solve keeps its quasi-Newton matrix, since there is
    /// no exact Hessian to evaluate anywhere.
    fn corrector_operator<'a>(
        &'a self,
        predicted: &[Number],
        pinned: &[(usize, Number)],
        exact_hessian: bool,
    ) -> Result<Box<dyn KktOperator + 'a>, String> {
        let (data_h, cq_h, _) = self.activity_handles();
        let predicted = self
            .pack_natural(predicted)
            .ok_or_else(|| "corrector: packing the predicted point failed".to_string())?;
        let mut restore = RestoreCurr {
            data: data_h,
            saved: data_h.borrow().curr.clone(),
            saved_w: None,
            restore_w: false,
        };
        data_h.borrow_mut().set_curr(predicted);
        if exact_hessian {
            restore.saved_w = data_h.borrow().w.clone();
            restore.restore_w = true;
            let w = cq_h.borrow().curr_exact_hessian();
            data_h.borrow_mut().w = Some(w);
        }
        let op = self
            .corrector_sigma(pinned)
            .ok_or_else(|| "corrector: operator diagonal unavailable".to_string())?;
        Ok(Box::new(PdCorrectorOperator {
            bs: self,
            op,
            _restore: restore,
        }))
    }

    fn activity_data(&self) -> Result<ActivityData, String> {
        let (data, cq, nlp) = self.activity_handles();

        // Scoped borrows: the calculated-quantity getters below re-borrow
        // the NLP (mutably, for lazy evaluation) and the data, so nothing
        // here may hold either across one of those calls.
        let (z_l, z_u, v_l, v_u) = {
            let d = data.borrow();
            let curr = d
                .curr
                .as_ref()
                .ok_or("activity: the converged state has no iterate")?;
            (
                dense_to_vec(curr.z_l.as_ref()),
                dense_to_vec(curr.z_u.as_ref()),
                dense_to_vec(curr.v_l.as_ref()),
                dense_to_vec(curr.v_u.as_ref()),
            )
        };
        let (px_l, px_u, pd_l, pd_u) = {
            let nl = nlp.borrow();
            (
                positions(&nl.px_l())?,
                positions(&nl.px_u())?,
                positions(&nl.pd_l())?,
                positions(&nl.pd_u())?,
            )
        };
        let (s_x_l, s_x_u, s_d_l, s_d_u, hessian, jac_c, jac_d) = {
            let c = cq.borrow();
            (
                dense_to_vec(c.curr_slack_x_l().as_ref()),
                dense_to_vec(c.curr_slack_x_u().as_ref()),
                dense_to_vec(c.curr_slack_s_l().as_ref()),
                dense_to_vec(c.curr_slack_s_u().as_ref()),
                c.curr_exact_hessian(),
                c.curr_jac_c(),
                c.curr_jac_d(),
            )
        };
        let sigma_x = dense_to_vec(self.barrier_sigma_x().as_ref());
        let sigma_s = dense_to_vec(self.barrier_sigma_s().as_ref());

        let side =
            |pos: Vec<usize>, mult: Vec<Number>, slack: Vec<Number>| BoundSide { pos, mult, slack };
        Ok(ActivityData {
            x_l: side(px_l, z_l, s_x_l),
            x_u: side(px_u, z_u, s_x_u),
            d_l: side(pd_l, v_l, s_d_l),
            d_u: side(pd_u, v_u, s_d_u),
            sigma_x,
            sigma_s,
            hessian,
            jac_c,
            jac_d,
        })
    }

    /// The NLP's own maps. The inequality positions are its
    /// `full_g_to_d_block`, the exact complement of `full_g_to_c_block`:
    /// one ascending scan over the model's rows assigns each to the next
    /// position of its block.
    fn model_maps(&self) -> Result<ModelMaps, String> {
        let (_, _, nlp) = self.activity_handles();
        let nl = nlp.borrow();
        let n_x = self.block_dims()[0];
        let n_full_g = nl.n_full_g();
        let block = |p: Option<Index>| p.map(|p| p as usize);
        Ok(ModelMaps {
            n_full_x: nl.n_full_x() as usize,
            var_x_to_full_x: (0..n_x)
                .map(|i| nl.var_x_to_full_x(i as Index) as usize)
                .collect(),
            full_g_to_c: (0..n_full_g)
                .map(|g| block(nl.full_g_to_c_block(g)))
                .collect(),
            full_g_to_d: (0..n_full_g)
                .map(|g| block(nl.full_g_to_d_block(g)))
                .collect(),
        })
    }

    fn scaling(&self) -> EngineScaling {
        let (obj, c_scale, d_scale) = self.nlp_scaling();
        EngineScaling {
            objective: obj,
            variables: self.variable_scaling().map(|v| v.to_vec()),
            equality_rows: c_scale,
            inequality_rows: d_scale,
        }
    }

    /// The bounds the NLP holds, relaxed by `bound_relax_factor` when the
    /// solve relaxed them, expanded from its compressed bound vectors.
    fn primal_bounds(&self) -> Result<(Vec<Number>, Vec<Number>), String> {
        let (_, _, nlp) = self.activity_handles();
        let nl = nlp.borrow();
        let [n_x, n_s, ..] = self.block_dims();
        let (mut lo, mut hi) = pounce_sens_core::boundcheck::expand_bounds(
            n_x,
            &nl.px_l(),
            &nl.px_u(),
            nl.x_l(),
            nl.x_u(),
        );
        let (s_lo, s_hi) = pounce_sens_core::boundcheck::expand_bounds(
            n_s,
            &nl.pd_l(),
            &nl.pd_u(),
            nl.d_l(),
            nl.d_u(),
        );
        lo.extend_from_slice(&s_lo);
        hi.extend_from_slice(&s_hi);
        Ok((lo, hi))
    }

    fn kkt_perturbations(&self) -> Option<[Number; 4]> {
        Some(PdSensBacksolver::kkt_perturbations(self))
    }

    /// The released diagonals built once and held, so every solve passes
    /// the same objects and the factorization cache, which keys on the
    /// diagonal's tag, factors the released system once.
    fn released_operator<'a>(
        &'a self,
        released: &[usize],
    ) -> Result<Box<dyn KktOperator + 'a>, String> {
        let (sigma, sigma_s) = self
            .released_sigmas(released)
            .ok_or_else(|| "released sigma unavailable".to_string())?;
        Ok(Box::new(PdReleasedOperator {
            bs: self,
            released: released.to_vec(),
            sigma,
            sigma_s,
        }))
    }
}

/// The operator [`PdSensBacksolver`] hands a calculation that solves
/// against one released system many times.
struct PdReleasedOperator<'a> {
    bs: &'a PdSensBacksolver,
    released: Vec<usize>,
    sigma: Rc<dyn pounce_linalg::Vector>,
    sigma_s: Option<Rc<dyn pounce_linalg::Vector>>,
}

impl KktOperator for PdReleasedOperator<'_> {
    /// [`SensBacksolver::solve_released`] against the held diagonals: it
    /// does nothing to the right-hand side.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.bs.solve_released_prebuilt(
            &self.released,
            Rc::clone(&self.sigma),
            self.sigma_s.as_ref().map(Rc::clone),
            None,
            rhs,
            lhs,
            false,
        )
    }
}

/// The entries a bound expansion places its compressed vector at.
fn positions(m: &Rc<dyn Matrix>) -> Result<Vec<usize>, String> {
    m.as_any()
        .downcast_ref::<ExpansionMatrix>()
        .map(|e| {
            e.expanded_pos_indices()
                .iter()
                .map(|&i| i as usize)
                .collect()
        })
        .ok_or_else(|| "activity: a bound projection is not an expansion matrix".to_string())
}

/// The operator [`PdSensBacksolver`] hands the corrector: a solve against
/// the held factorization with the predicted point's diagonal, and the
/// iterate swap that diagonal was built under, undone on drop.
struct PdCorrectorOperator<'a> {
    bs: &'a PdSensBacksolver,
    op: CorrectorSigma,
    _restore: RestoreCurr<'a>,
}

impl KktOperator for PdCorrectorOperator<'_> {
    /// The same `Rc` every call: the factorization cache keys on its tag,
    /// so the predicted point's operator is factored once and every later
    /// solve is a back-solve.
    ///
    /// The ratios go with the diagonals: the operator eliminates each bound
    /// row into a diagonal the gh#737 ceiling may have softened, so the row
    /// is read back through the same cap (gh#828). A no-op wherever the
    /// ceiling did not bind.
    ///
    /// No bound is released here. A bound the corrector released has a
    /// zero multiplier at the point the diagonal was built at, and a zero
    /// right-hand side, so it adds nothing to the system.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.bs.solve_released_prebuilt(
            &[],
            Rc::clone(&self.op.sigma_x),
            Some(Rc::clone(&self.op.sigma_s)),
            Some((&self.op.ratio_x, &self.op.ratio_s)),
            rhs,
            lhs,
            false,
        )
    }
}

/// Puts back the current iterate, and the stored Hessian when it was
/// replaced, that [`BarrierIterate::corrector_operator`] swapped out.
struct RestoreCurr<'a> {
    data: &'a IpoptDataHandle,
    saved: Option<IteratesVector>,
    saved_w: Option<Rc<dyn pounce_linalg::SymMatrix>>,
    restore_w: bool,
}

impl Drop for RestoreCurr<'_> {
    fn drop(&mut self) {
        if let Some(c) = self.saved.take() {
            self.data.borrow_mut().set_curr(c);
        }
        if self.restore_w {
            self.data.borrow_mut().w = self.saved_w.take();
        }
    }
}
