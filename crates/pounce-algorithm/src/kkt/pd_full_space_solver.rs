//! Full-space PD system solver — port of
//! `Algorithm/IpPDFullSpaceSolver.{hpp,cpp}`.
//!
//! Iterative refinement on the FULL 8-block primal-dual KKT system,
//! driving the augmented-system solver repeatedly. See
//! `KKT_SYSTEM.md` §5 for the refinement-quit criteria. The outer
//! loop alternates between back-solves and quality escalation
//! (`AugSystemSolver::increase_quality()` and `pretend_singular`).

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::{IteratesVector, IteratesVectorMut};
use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use crate::kkt::pd_system_solver::PdSystemSolver;
use crate::kkt::perturbation_handler::PdPerturbationHandler;
use pounce_common::tagged::Tag;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use pounce_linalg::{Matrix, SymMatrix, Vector};
use pounce_linsol::ESymSolverStatus;
use std::cell::RefCell;
use std::rc::Rc;

pub struct PdFullSpaceSolver {
    aug_solver: Box<dyn AugSystemSolver>,
    perturb: Rc<RefCell<PdPerturbationHandler>>,
    pub min_refinement_steps: Index,
    pub max_refinement_steps: Index,
    pub residual_ratio_max: Number,
    pub residual_ratio_singular: Number,
    pub residual_improvement_factor: Number,
    /// Negative-curvature test tolerance (`neg_curv_test_tol_`). Zero
    /// disables the heuristic; matches upstream's `RegisterOptions`
    /// default. The non-zero branch is not exercised in v1.0.
    pub neg_curv_test_tol: Number,
    /// Mirrors `augsys_improved_`. Set by quality-escalation; cleared
    /// each time the cached aug-system data changes.
    augsys_improved: bool,
    /// Mirrors upstream's `dummy_cache_` hit/miss. `false` ⇒ the next
    /// `solve_once` is operating on a *new* augmented matrix and must
    /// run the `ConsiderNewSystem` + perturbation-escalation path;
    /// `true` ⇒ the matrix is identical to the previous successful
    /// `solve_once`, so we can reuse `CurrentPerturbation` and just do
    /// a single back-solve (the iterative-refinement / quality-retry
    /// re-call path). Reset to `false` at the start of every outer
    /// `solve()` invocation since each outer iter delivers a fresh
    /// matrix from the algorithm's perspective.
    matrix_considered: bool,
    /// Tags of the 13 dependencies (W, J_c, J_d, z_L, z_U, v_L, v_U,
    /// slack_x_L, slack_x_U, slack_s_L, slack_s_U, sigma_x, sigma_s)
    /// at the time `matrix_considered` was last set to `true`. Mirrors
    /// upstream's `dummy_cache_` keyed on the same 13 `TaggedObject`s
    /// (`IpPDFullSpaceSolver.cpp:430-448`). Reset to `None` whenever
    /// any tag changes.
    last_dep_tags: Option<[Tag; 13]>,
    last_status: Option<ESymSolverStatus>,
}

impl PdFullSpaceSolver {
    pub fn new(
        aug_solver: Box<dyn AugSystemSolver>,
        perturb: Rc<RefCell<PdPerturbationHandler>>,
    ) -> Self {
        Self {
            aug_solver,
            perturb,
            // Defaults from `IpPDFullSpaceSolver.cpp:RegisterOptions`.
            min_refinement_steps: 1,
            max_refinement_steps: 10,
            residual_ratio_max: 1e-10,
            residual_ratio_singular: 1e-5,
            residual_improvement_factor: 0.999_999_999,
            neg_curv_test_tol: 0.0,
            augsys_improved: false,
            matrix_considered: false,
            last_dep_tags: None,
            last_status: None,
        }
    }

    pub fn aug_solver(&self) -> &dyn AugSystemSolver {
        &*self.aug_solver
    }

    pub fn aug_solver_mut(&mut self) -> &mut dyn AugSystemSolver {
        &mut *self.aug_solver
    }

    /// Replace the underlying [`AugSystemSolver`] by passing the
    /// existing one through the supplied wrapper closure. Used by the
    /// restoration phase to decorate the inner `StdAugSystemSolver`
    /// with `AugRestoSystemSolver` (which performs the 8-block →
    /// 4-block Schur reduction before delegating).
    pub fn wrap_aug_solver<F>(&mut self, wrap: F)
    where
        F: FnOnce(Box<dyn AugSystemSolver>) -> Box<dyn AugSystemSolver>,
    {
        // Take the inner aug solver out via a temporary noop, wrap it,
        // and slot the wrapped one back in. The placeholder is never
        // observed externally because we replace it before returning.
        let noop: Box<dyn AugSystemSolver> = Box::new(NoopAugSolver);
        let inner = std::mem::replace(&mut self.aug_solver, noop);
        self.aug_solver = wrap(inner);
    }

    /// Solve the full PD system. `res = α · M⁻¹ · rhs + β · res_in`,
    /// matching `IpPDFullSpaceSolver::Solve`. Returns `true` on
    /// success. The iterate fields used to assemble the system are
    /// pulled from `data` (`W`, `curr`) and `cq` (jacobians, slacks,
    /// sigmas).
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        alpha: Number,
        beta: Number,
        rhs: &IteratesVector,
        res: &mut IteratesVectorMut,
        allow_inexact: bool,
        improve_solution: bool,
    ) -> bool {
        debug_assert!(!allow_inexact || !improve_solution);
        debug_assert!(!improve_solution || beta == 0.0);

        // Snapshot the incoming `res` if β ≠ 0 (we add it back at the
        // end via `res = α · sol + β · copy_res`).
        let copy_res: Option<IteratesVector> = if beta != 0.0 {
            Some(snapshot_mut(res))
        } else {
            None
        };

        // Pull all blocks once. None of these change during the
        // refinement / escalation loop, so collecting them here
        // matches upstream's structure (lines 168-189).
        let w = data
            .borrow()
            .w
            .clone()
            .unwrap_or_else(|| panic!("PdFullSpaceSolver::solve: IpoptData::w is unset"));
        let cq_ref = cq.borrow();
        let j_c = cq_ref.curr_jac_c();
        let j_d = cq_ref.curr_jac_d();
        let sigma_x = cq_ref.curr_sigma_x();
        let sigma_s = cq_ref.curr_sigma_s();
        let slack_x_l = cq_ref.curr_slack_x_l();
        let slack_x_u = cq_ref.curr_slack_x_u();
        let slack_s_l = cq_ref.curr_slack_s_l();
        let slack_s_u = cq_ref.curr_slack_s_u();
        drop(cq_ref);

        let nlp_ref = nlp.borrow();
        let px_l = nlp_ref.px_l();
        let px_u = nlp_ref.px_u();
        let pd_l = nlp_ref.pd_l();
        let pd_u = nlp_ref.pd_u();
        drop(nlp_ref);

        let curr = {
            let d = data.borrow();
            d.curr
                .clone()
                .unwrap_or_else(|| panic!("PdFullSpaceSolver::solve: IpoptData::curr is unset"))
        };

        let blocks = SolveBlocks {
            w: &*w,
            j_c: &*j_c,
            j_d: &*j_d,
            px_l: &*px_l,
            px_u: &*px_u,
            pd_l: &*pd_l,
            pd_u: &*pd_u,
            z_l: &*curr.z_l,
            z_u: &*curr.z_u,
            v_l: &*curr.v_l,
            v_u: &*curr.v_u,
            slack_x_l: &*slack_x_l,
            slack_x_u: &*slack_x_u,
            slack_s_l: &*slack_s_l,
            slack_s_u: &*slack_s_u,
            sigma_x: &*sigma_x,
            sigma_s: &*sigma_s,
        };

        // Mirror upstream's `dummy_cache_` lookup
        // (`IpPDFullSpaceSolver.cpp:430-450`): if all 13 dependency tags
        // are unchanged since the last successful `solve()`, the matrix
        // is "uptodate" — keep `matrix_considered = true` so the
        // perturbation handler is NOT re-entered, and reuse the
        // existing `augsys_improved_` state. On a cache miss, reset
        // both flags.
        let cur_tags: [Tag; 13] = [
            blocks.w.as_tagged().get_tag(),
            blocks.j_c.as_tagged().get_tag(),
            blocks.j_d.as_tagged().get_tag(),
            blocks.z_l.as_tagged().get_tag(),
            blocks.z_u.as_tagged().get_tag(),
            blocks.v_l.as_tagged().get_tag(),
            blocks.v_u.as_tagged().get_tag(),
            blocks.slack_x_l.as_tagged().get_tag(),
            blocks.slack_x_u.as_tagged().get_tag(),
            blocks.slack_s_l.as_tagged().get_tag(),
            blocks.slack_s_u.as_tagged().get_tag(),
            blocks.sigma_x.as_tagged().get_tag(),
            blocks.sigma_s.as_tagged().get_tag(),
        ];
        let uptodate = self.last_dep_tags.map_or(false, |prev| prev == cur_tags);
        if !uptodate {
            if std::env::var_os("POUNCE_DBG_PD_TAGS").is_some() {
                if let Some(prev) = self.last_dep_tags {
                    let names = [
                        "w",
                        "j_c",
                        "j_d",
                        "z_l",
                        "z_u",
                        "v_l",
                        "v_u",
                        "slack_x_l",
                        "slack_x_u",
                        "slack_s_l",
                        "slack_s_u",
                        "sigma_x",
                        "sigma_s",
                    ];
                    let mut diffs = String::new();
                    for i in 0..13 {
                        if prev[i] != cur_tags[i] {
                            diffs.push_str(&format!(
                                " {}({:?}→{:?})",
                                names[i], prev[i], cur_tags[i]
                            ));
                        }
                    }
                    eprintln!("[PN_PD_TAGS] cache_miss diffs:{}", diffs);
                } else {
                    eprintln!("[PN_PD_TAGS] cache_miss first_solve");
                }
            }
            self.last_dep_tags = Some(cur_tags);
            self.matrix_considered = false;
            self.augsys_improved = false;
        }

        let mut done = false;
        let mut resolve_with_better_quality = false;
        let mut pretend_singular = false;
        let mut pretend_singular_last_time = false;
        let mut improve = improve_solution;

        while !done {
            let solve_ok = if improve {
                true
            } else {
                let ok = self.solve_once(
                    data,
                    &blocks,
                    1.0,
                    0.0,
                    rhs,
                    res,
                    resolve_with_better_quality,
                    pretend_singular,
                );
                resolve_with_better_quality = false;
                pretend_singular = false;
                ok
            };
            improve = false;

            if !solve_ok {
                return false;
            }

            if allow_inexact {
                break;
            }

            // Initial residual.
            let mut resid = res.fresh_zeroed();
            self.compute_residuals(data, &blocks, rhs, res, &mut resid);
            let mut residual_ratio = self.compute_residual_ratio(rhs, res, &resid);
            let mut residual_ratio_old = residual_ratio;

            let mut num_iter_ref: Index = 0;
            let mut quit_refinement = false;

            while !quit_refinement
                && (num_iter_ref < self.min_refinement_steps
                    || residual_ratio > self.residual_ratio_max)
            {
                let frozen_resid = resid.freeze();
                let solve_ok = self.solve_once(
                    data,
                    &blocks,
                    -1.0,
                    1.0,
                    &frozen_resid,
                    res,
                    resolve_with_better_quality,
                    false,
                );
                resid = thaw(frozen_resid);
                if !solve_ok {
                    return false;
                }

                self.compute_residuals(data, &blocks, rhs, res, &mut resid);
                residual_ratio = self.compute_residual_ratio(rhs, res, &resid);
                num_iter_ref += 1;

                if residual_ratio > self.residual_ratio_max
                    && num_iter_ref > self.min_refinement_steps
                    && (num_iter_ref > self.max_refinement_steps
                        || residual_ratio > self.residual_improvement_factor * residual_ratio_old)
                {
                    quit_refinement = true;
                    resolve_with_better_quality = false;

                    if !pretend_singular_last_time {
                        if !self.augsys_improved {
                            self.augsys_improved = self.aug_solver.increase_quality();
                            if self.augsys_improved {
                                data.borrow_mut().append_info_string("q");
                                resolve_with_better_quality = true;
                            } else {
                                pretend_singular = true;
                            }
                        } else {
                            pretend_singular = true;
                        }
                        pretend_singular_last_time = pretend_singular;
                        if pretend_singular {
                            if residual_ratio < self.residual_ratio_singular {
                                pretend_singular = false;
                                data.borrow_mut().append_info_string("S");
                            } else {
                                data.borrow_mut().append_info_string("s");
                            }
                        }
                    } else {
                        pretend_singular = false;
                    }
                }

                residual_ratio_old = residual_ratio;
            }

            done = !resolve_with_better_quality && !pretend_singular;
        }

        // Final assembly: res = α · res + β · copy_res.
        if alpha != 0.0 {
            res.scal(alpha);
        }
        if let Some(copy_res) = copy_res {
            res.axpy(beta, &copy_res);
        }

        self.last_status = Some(ESymSolverStatus::Success);
        true
    }

    /// Batched back-substitution against the cached KKT factor for
    /// `n_rhs` right-hand sides, sharing one
    /// `pounce_linsol::TSymLinearSolver::multi_solve` call with
    /// `nrhs > 1`. Each column k pulls its RHS through `write_rhs(k,
    /// &mut iv)` and emits its solution through `write_lhs(k, &iv)` —
    /// closures over the caller's flat / strided buffer keep the
    /// rhs/sol `IteratesVectorMut` scratch out of the API surface.
    ///
    /// Returns:
    /// - `Some(true)`  — fast path executed against the cached factor.
    /// - `Some(false)` — fast path was attempted but the linsol
    ///   reported a back-solve failure.
    /// - `None`        — fast path not taken. Either the matrix tags
    ///   differ from the last successful [`Self::solve`] (cache miss),
    ///   the matrix has not been considered yet, or the underlying
    ///   `AugSystemSolver` does not implement
    ///   [`AugSystemSolver::try_resolve_many_flat`]. The caller should
    ///   fall back to looping [`Self::solve`].
    ///
    /// Used by `pounce_sensitivity::PdSensBacksolver::solve_many` for
    /// the JaxProblem `jacrev` backward path, where every cotangent
    /// re-solves against the same converged factor (pounce#77 follow-up).
    pub fn solve_many_cached<F1, F2>(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        n_rhs: usize,
        mut write_rhs: F1,
        mut write_lhs: F2,
    ) -> Option<bool>
    where
        F1: FnMut(usize, &mut IteratesVectorMut),
        F2: FnMut(usize, &IteratesVectorMut),
    {
        if n_rhs == 0 {
            return Some(true);
        }

        // Pull all blocks (same shape as `solve()`).
        let w = data.borrow().w.clone()?;
        let cq_ref = cq.borrow();
        let j_c = cq_ref.curr_jac_c();
        let j_d = cq_ref.curr_jac_d();
        let sigma_x = cq_ref.curr_sigma_x();
        let sigma_s = cq_ref.curr_sigma_s();
        let slack_x_l = cq_ref.curr_slack_x_l();
        let slack_x_u = cq_ref.curr_slack_x_u();
        let slack_s_l = cq_ref.curr_slack_s_l();
        let slack_s_u = cq_ref.curr_slack_s_u();
        drop(cq_ref);

        let nlp_ref = nlp.borrow();
        let px_l = nlp_ref.px_l();
        let px_u = nlp_ref.px_u();
        let pd_l = nlp_ref.pd_l();
        let pd_u = nlp_ref.pd_u();
        drop(nlp_ref);

        let curr = data.borrow().curr.clone()?;

        let blocks = SolveBlocks {
            w: &*w,
            j_c: &*j_c,
            j_d: &*j_d,
            px_l: &*px_l,
            px_u: &*px_u,
            pd_l: &*pd_l,
            pd_u: &*pd_u,
            z_l: &*curr.z_l,
            z_u: &*curr.z_u,
            v_l: &*curr.v_l,
            v_u: &*curr.v_u,
            slack_x_l: &*slack_x_l,
            slack_x_u: &*slack_x_u,
            slack_s_l: &*slack_s_l,
            slack_s_u: &*slack_s_u,
            sigma_x: &*sigma_x,
            sigma_s: &*sigma_s,
        };

        // Cache-tag check (same 13 tags as `solve()`). If the matrix
        // has changed since the last successful solve, or we never
        // marked it as considered, bail and let the caller take the
        // per-RHS path.
        let cur_tags: [Tag; 13] = [
            blocks.w.as_tagged().get_tag(),
            blocks.j_c.as_tagged().get_tag(),
            blocks.j_d.as_tagged().get_tag(),
            blocks.z_l.as_tagged().get_tag(),
            blocks.z_u.as_tagged().get_tag(),
            blocks.v_l.as_tagged().get_tag(),
            blocks.v_u.as_tagged().get_tag(),
            blocks.slack_x_l.as_tagged().get_tag(),
            blocks.slack_x_u.as_tagged().get_tag(),
            blocks.slack_s_l.as_tagged().get_tag(),
            blocks.slack_s_u.as_tagged().get_tag(),
            blocks.sigma_x.as_tagged().get_tag(),
            blocks.sigma_s.as_tagged().get_tag(),
        ];
        if !self.matrix_considered
            || !self.last_dep_tags.map_or(false, |prev| prev == cur_tags)
        {
            return None;
        }

        // Coeffs reuse the perturbation stashed by the most recent
        // `solve_once`. `current_perturbation()` returns the same
        // values that solve_once wrote into `data.perturbations`.
        let d = self.perturb.borrow().current_perturbation();
        let coeffs = AugSysCoeffs {
            w: Some(blocks.w),
            w_factor: 1.0,
            d_x: Some(blocks.sigma_x),
            delta_x: d.delta_x,
            d_s: Some(blocks.sigma_s),
            delta_s: d.delta_s,
            j_c: blocks.j_c,
            d_c: None,
            delta_c: d.delta_c,
            j_d: blocks.j_d,
            d_d: None,
            delta_d: d.delta_d,
        };

        let n_x = curr.x.dim() as usize;
        let n_s = curr.s.dim() as usize;
        let n_y_c = curr.y_c.dim() as usize;
        let n_y_d = curr.y_d.dim() as usize;
        let aug_dim = n_x + n_s + n_y_c + n_y_d;

        // Scratch — one set of Box allocs, reused across every column.
        let mut rhs_iv = curr.make_new_zeroed();
        let mut sol_iv = curr.make_new_zeroed();
        let mut aug_rhs_x_box: Box<dyn Vector> = curr.x.make_new();
        let mut aug_rhs_s_box: Box<dyn Vector> = curr.s.make_new();

        // Column-major `(aug_dim, n_rhs)` packed buffer — single
        // allocation that the linsol's `multi_solve` writes solutions
        // back into in place.
        let mut aug_packed = vec![0.0 as Number; aug_dim * n_rhs];

        // Phase 1: populate aug_packed column-by-column. The aug-system
        // RHS is `[aug_rhs_x | aug_rhs_s | rhs.y_c | rhs.y_d]`, where
        //   aug_rhs_x = rhs.x + Px_L·S_xL⁻¹·z_L − Px_U·S_xU⁻¹·z_U
        //   aug_rhs_s = rhs.s + Pd_L·S_sL⁻¹·v_L − Pd_U·S_sU⁻¹·v_U
        // matching `solve_once`'s aug-RHS build.
        for k in 0..n_rhs {
            write_rhs(k, &mut rhs_iv);

            aug_rhs_x_box.copy(&*rhs_iv.x);
            blocks
                .px_l
                .add_m_sinv_z(1.0, blocks.slack_x_l, &*rhs_iv.z_l, &mut *aug_rhs_x_box);
            blocks
                .px_u
                .add_m_sinv_z(-1.0, blocks.slack_x_u, &*rhs_iv.z_u, &mut *aug_rhs_x_box);

            aug_rhs_s_box.copy(&*rhs_iv.s);
            blocks
                .pd_l
                .add_m_sinv_z(1.0, blocks.slack_s_l, &*rhs_iv.v_l, &mut *aug_rhs_s_box);
            blocks
                .pd_u
                .add_m_sinv_z(-1.0, blocks.slack_s_u, &*rhs_iv.v_u, &mut *aug_rhs_s_box);

            let col = &mut aug_packed[k * aug_dim..(k + 1) * aug_dim];
            copy_vector_to_slice(&*aug_rhs_x_box, &mut col[..n_x]);
            copy_vector_to_slice(&*aug_rhs_s_box, &mut col[n_x..n_x + n_s]);
            copy_vector_to_slice(
                &*rhs_iv.y_c,
                &mut col[n_x + n_s..n_x + n_s + n_y_c],
            );
            copy_vector_to_slice(&*rhs_iv.y_d, &mut col[n_x + n_s + n_y_c..]);
        }

        // Phase 2: single batched back-substitution.
        let status = self
            .aug_solver
            .try_resolve_many_flat(&coeffs, &mut aug_packed, n_rhs)?;
        if status != ESymSolverStatus::Success {
            self.last_status = Some(status);
            return Some(false);
        }
        self.last_status = Some(status);

        // Phase 3: unpack each column into `sol_iv`, run the bound-
        // multiplier expansion, hand the result to the caller. We have
        // to re-invoke `write_rhs` because expand_bound_multipliers
        // reads `rhs.z_l/z_u/v_l/v_u` and we re-used `rhs_iv` across
        // all columns in phase 1.
        for k in 0..n_rhs {
            write_rhs(k, &mut rhs_iv);

            let col = &aug_packed[k * aug_dim..(k + 1) * aug_dim];
            set_vector_from_slice(&mut *sol_iv.x, &col[..n_x]);
            set_vector_from_slice(&mut *sol_iv.s, &col[n_x..n_x + n_s]);
            set_vector_from_slice(
                &mut *sol_iv.y_c,
                &col[n_x + n_s..n_x + n_s + n_y_c],
            );
            set_vector_from_slice(&mut *sol_iv.y_d, &col[n_x + n_s + n_y_c..]);

            // Inline expand_bound_multipliers — that helper takes
            // `&IteratesVector` (Rc-backed) but our `rhs_iv` is
            // `IteratesVectorMut` (Box-backed). The four
            // `sinv_blrm_zmt_dbr` calls work on `&dyn Vector` either
            // way.
            blocks.px_l.sinv_blrm_zmt_dbr(
                -1.0,
                blocks.slack_x_l,
                &*rhs_iv.z_l,
                blocks.z_l,
                &*sol_iv.x,
                &mut *sol_iv.z_l,
            );
            blocks.px_u.sinv_blrm_zmt_dbr(
                1.0,
                blocks.slack_x_u,
                &*rhs_iv.z_u,
                blocks.z_u,
                &*sol_iv.x,
                &mut *sol_iv.z_u,
            );
            blocks.pd_l.sinv_blrm_zmt_dbr(
                -1.0,
                blocks.slack_s_l,
                &*rhs_iv.v_l,
                blocks.v_l,
                &*sol_iv.s,
                &mut *sol_iv.v_l,
            );
            blocks.pd_u.sinv_blrm_zmt_dbr(
                1.0,
                blocks.slack_s_u,
                &*rhs_iv.v_u,
                blocks.v_u,
                &*sol_iv.s,
                &mut *sol_iv.v_u,
            );

            write_lhs(k, &sol_iv);
        }

        Some(true)
    }

    /// Flat-slice cached-factor multi-RHS path. Same cache-check
    /// semantics as [`Self::solve_many_cached`] but operates on
    /// row-major `(n_rhs, total)` flat buffers without going through
    /// `IteratesVectorMut` or any `dyn Vector` / `dyn Matrix` dispatch
    /// in the per-RHS inner loops — the eight source blocks
    /// (`slack_{x,s}_{l,u}`, `z_{l,u}`, `v_{l,u}`) get downcast to
    /// `DenseVector` once at the top, the four bound-expansion matrices
    /// (`px_l`, `px_u`, `pd_l`, `pd_u`) get downcast to
    /// `ExpansionMatrix` once, and Phase 1 / Phase 3 then run as raw
    /// `&[Number]` / `&mut [Number]` arithmetic on the flat buffers.
    ///
    /// `total` is the sum of the eight `block_dims` entries (in the
    /// same `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order that
    /// `IteratesVector` uses); `rhs_flat.len() == lhs_flat.len() ==
    /// n_rhs * total`.
    ///
    /// Returns `None` (caller should fall back to
    /// [`Self::solve_many_cached`]) when:
    /// - the cache check fails (matrix tags differ),
    /// - any block source vector is not a `DenseVector` or is
    ///   homogeneous (uniform-scalar) on a non-empty block,
    /// - any bound-expansion matrix is not an `ExpansionMatrix`,
    /// - the underlying `AugSystemSolver` doesn't implement
    ///   [`AugSystemSolver::try_resolve_many_flat`].
    ///
    /// Returns `Some(true)` on success, `Some(false)` on linsol back-
    /// solve failure.
    ///
    /// Used by `pounce_sensitivity::PdSensBacksolver::solve_many` as
    /// the fastest tier of the JaxProblem `jacrev` backward path
    /// (pounce#77 follow-up).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_many_cached_flat(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        n_rhs: usize,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        block_dims: [usize; 8],
    ) -> Option<bool> {
        if n_rhs == 0 {
            return Some(true);
        }
        let total: usize = block_dims.iter().sum();
        if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
            return Some(false);
        }
        let mut off = [0usize; 9];
        for i in 0..8 {
            off[i + 1] = off[i] + block_dims[i];
        }
        let n_x = block_dims[0];
        let n_s = block_dims[1];
        let n_y_c = block_dims[2];
        let n_y_d = block_dims[3];

        // Pull all blocks (same shape as `solve()`).
        let w = data.borrow().w.clone()?;
        let cq_ref = cq.borrow();
        let j_c = cq_ref.curr_jac_c();
        let j_d = cq_ref.curr_jac_d();
        let sigma_x = cq_ref.curr_sigma_x();
        let sigma_s = cq_ref.curr_sigma_s();
        let slack_x_l = cq_ref.curr_slack_x_l();
        let slack_x_u = cq_ref.curr_slack_x_u();
        let slack_s_l = cq_ref.curr_slack_s_l();
        let slack_s_u = cq_ref.curr_slack_s_u();
        drop(cq_ref);

        let nlp_ref = nlp.borrow();
        let px_l = nlp_ref.px_l();
        let px_u = nlp_ref.px_u();
        let pd_l = nlp_ref.pd_l();
        let pd_u = nlp_ref.pd_u();
        drop(nlp_ref);

        let curr = data.borrow().curr.clone()?;

        // Cache-tag check (same 13 tags as `solve()`).
        let cur_tags: [Tag; 13] = [
            w.as_tagged().get_tag(),
            j_c.as_tagged().get_tag(),
            j_d.as_tagged().get_tag(),
            curr.z_l.as_tagged().get_tag(),
            curr.z_u.as_tagged().get_tag(),
            curr.v_l.as_tagged().get_tag(),
            curr.v_u.as_tagged().get_tag(),
            slack_x_l.as_tagged().get_tag(),
            slack_x_u.as_tagged().get_tag(),
            slack_s_l.as_tagged().get_tag(),
            slack_s_u.as_tagged().get_tag(),
            sigma_x.as_tagged().get_tag(),
            sigma_s.as_tagged().get_tag(),
        ];
        if !self.matrix_considered
            || !self.last_dep_tags.map_or(false, |prev| prev == cur_tags)
        {
            return None;
        }

        // Concrete downcasts. Bail to closure-based fallback on any
        // type mismatch (homogeneous-on-non-empty included — the math
        // below assumes a real `[Number]` slice for slack / z / v).
        let slack_x_l_d = dense_slice_or_none(&*slack_x_l, block_dims[4])?;
        let slack_x_u_d = dense_slice_or_none(&*slack_x_u, block_dims[5])?;
        let slack_s_l_d = dense_slice_or_none(&*slack_s_l, block_dims[6])?;
        let slack_s_u_d = dense_slice_or_none(&*slack_s_u, block_dims[7])?;
        let blocks_z_l_d = dense_slice_or_none(&*curr.z_l, block_dims[4])?;
        let blocks_z_u_d = dense_slice_or_none(&*curr.z_u, block_dims[5])?;
        let blocks_v_l_d = dense_slice_or_none(&*curr.v_l, block_dims[6])?;
        let blocks_v_u_d = dense_slice_or_none(&*curr.v_u, block_dims[7])?;

        let exp_x_l = exp_pos_or_none(&*px_l)?;
        let exp_x_u = exp_pos_or_none(&*px_u)?;
        let exp_s_l = exp_pos_or_none(&*pd_l)?;
        let exp_s_u = exp_pos_or_none(&*pd_u)?;

        // Coeffs reuse the perturbation stashed by the most recent
        // `solve_once`.
        let d = self.perturb.borrow().current_perturbation();
        let coeffs = AugSysCoeffs {
            w: Some(&*w),
            w_factor: 1.0,
            d_x: Some(&*sigma_x),
            delta_x: d.delta_x,
            d_s: Some(&*sigma_s),
            delta_s: d.delta_s,
            j_c: &*j_c,
            d_c: None,
            delta_c: d.delta_c,
            j_d: &*j_d,
            d_d: None,
            delta_d: d.delta_d,
        };

        let aug_dim = n_x + n_s + n_y_c + n_y_d;
        // Column-major `(aug_dim, n_rhs)` packed buffer, single alloc.
        let mut aug_packed = vec![0.0 as Number; aug_dim * n_rhs];

        // ---------------- Phase 1 ----------------
        // For each k: build the aug-system RHS into column k of
        // aug_packed, all inline against raw slices.
        for k in 0..n_rhs {
            let r_base = k * total;
            let rhs_x = &rhs_flat[r_base + off[0]..r_base + off[1]];
            let rhs_s = &rhs_flat[r_base + off[1]..r_base + off[2]];
            let rhs_y_c = &rhs_flat[r_base + off[2]..r_base + off[3]];
            let rhs_y_d = &rhs_flat[r_base + off[3]..r_base + off[4]];
            let rhs_z_l = &rhs_flat[r_base + off[4]..r_base + off[5]];
            let rhs_z_u = &rhs_flat[r_base + off[5]..r_base + off[6]];
            let rhs_v_l = &rhs_flat[r_base + off[6]..r_base + off[7]];
            let rhs_v_u = &rhs_flat[r_base + off[7]..r_base + off[8]];

            let aug_col = &mut aug_packed[k * aug_dim..(k + 1) * aug_dim];
            let (aug_x, rest) = aug_col.split_at_mut(n_x);
            let (aug_s, rest) = rest.split_at_mut(n_s);
            let (aug_y_c, aug_y_d) = rest.split_at_mut(n_y_c);

            // aug_x = rhs_x + Px_L · S_xL⁻¹ · z_L − Px_U · S_xU⁻¹ · z_U
            aug_x.copy_from_slice(rhs_x);
            scatter_add_div(aug_x, exp_x_l, rhs_z_l, slack_x_l_d, 1.0);
            scatter_add_div(aug_x, exp_x_u, rhs_z_u, slack_x_u_d, -1.0);
            // aug_s = rhs_s + Pd_L · S_sL⁻¹ · v_L − Pd_U · S_sU⁻¹ · v_U
            aug_s.copy_from_slice(rhs_s);
            scatter_add_div(aug_s, exp_s_l, rhs_v_l, slack_s_l_d, 1.0);
            scatter_add_div(aug_s, exp_s_u, rhs_v_u, slack_s_u_d, -1.0);
            aug_y_c.copy_from_slice(rhs_y_c);
            aug_y_d.copy_from_slice(rhs_y_d);
        }

        // ---------------- Phase 2 ----------------
        let status = self
            .aug_solver
            .try_resolve_many_flat(&coeffs, &mut aug_packed, n_rhs)?;
        if status != ESymSolverStatus::Success {
            self.last_status = Some(status);
            return Some(false);
        }
        self.last_status = Some(status);

        // ---------------- Phase 3 ----------------
        // For each k: copy sol_x/s/y_c/y_d into lhs_flat, then build
        // sol_z_l/z_u/v_l/v_u from the bound-multiplier expansion.
        for k in 0..n_rhs {
            let r_base = k * total;
            let rhs_z_l = &rhs_flat[r_base + off[4]..r_base + off[5]];
            let rhs_z_u = &rhs_flat[r_base + off[5]..r_base + off[6]];
            let rhs_v_l = &rhs_flat[r_base + off[6]..r_base + off[7]];
            let rhs_v_u = &rhs_flat[r_base + off[7]..r_base + off[8]];

            let aug_col = &aug_packed[k * aug_dim..(k + 1) * aug_dim];
            let sol_x = &aug_col[..n_x];
            let sol_s = &aug_col[n_x..n_x + n_s];
            let sol_y_c = &aug_col[n_x + n_s..n_x + n_s + n_y_c];
            let sol_y_d = &aug_col[n_x + n_s + n_y_c..];

            let l_base = k * total;
            let (lhs_xs, lhs_zv) = lhs_flat[l_base..l_base + total].split_at_mut(off[4]);
            let (lhs_x, rest) = lhs_xs.split_at_mut(n_x);
            let (lhs_s, rest) = rest.split_at_mut(n_s);
            let (lhs_y_c, lhs_y_d) = rest.split_at_mut(n_y_c);
            lhs_x.copy_from_slice(sol_x);
            lhs_s.copy_from_slice(sol_s);
            lhs_y_c.copy_from_slice(sol_y_c);
            lhs_y_d.copy_from_slice(sol_y_d);

            let (lhs_z_l, rest) = lhs_zv.split_at_mut(block_dims[4]);
            let (lhs_z_u, rest) = rest.split_at_mut(block_dims[5]);
            let (lhs_v_l, lhs_v_u) = rest.split_at_mut(block_dims[6]);

            // sol_z_l[i] = (rhs_z_l[i] − z_l[i] · sol_x[exp_x_l[i]]) / slack_x_l[i]
            expand_bound_mult(lhs_z_l, rhs_z_l, blocks_z_l_d, sol_x, exp_x_l, slack_x_l_d, -1.0);
            // sol_z_u[i] = (rhs_z_u[i] + z_u[i] · sol_x[exp_x_u[i]]) / slack_x_u[i]
            expand_bound_mult(lhs_z_u, rhs_z_u, blocks_z_u_d, sol_x, exp_x_u, slack_x_u_d, 1.0);
            expand_bound_mult(lhs_v_l, rhs_v_l, blocks_v_l_d, sol_s, exp_s_l, slack_s_l_d, -1.0);
            expand_bound_mult(lhs_v_u, rhs_v_u, blocks_v_u_d, sol_s, exp_s_u, slack_s_u_d, 1.0);
        }

        Some(true)
    }

    /// One outer back-solve through the augmented system, including
    /// the `Px_L · S_xL⁻¹ · z_L` lifts on the RHS and the bound-
    /// multiplier expansion on the solution side. Mirrors
    /// `IpPDFullSpaceSolver::SolveOnce`.
    #[allow(clippy::too_many_arguments)]
    fn solve_once(
        &mut self,
        data: &IpoptDataHandle,
        b: &SolveBlocks<'_>,
        alpha: Number,
        beta: Number,
        rhs: &IteratesVector,
        res: &mut IteratesVectorMut,
        _resolve_with_better_quality: bool,
        mut pretend_singular: bool,
    ) -> bool {
        // Build aug-system primal RHS:
        //   augRhs_x = rhs.x + Px_L · S_xL⁻¹ · z_L − Px_U · S_xU⁻¹ · z_U
        let mut aug_rhs_x = rhs.x.make_new_copy();
        b.px_l
            .add_m_sinv_z(1.0, b.slack_x_l, &*rhs.z_l, &mut *aug_rhs_x);
        b.px_u
            .add_m_sinv_z(-1.0, b.slack_x_u, &*rhs.z_u, &mut *aug_rhs_x);

        let mut aug_rhs_s = rhs.s.make_new_copy();
        b.pd_l
            .add_m_sinv_z(1.0, b.slack_s_l, &*rhs.v_l, &mut *aug_rhs_s);
        b.pd_u
            .add_m_sinv_z(-1.0, b.slack_s_u, &*rhs.v_u, &mut *aug_rhs_s);

        // Solution slot for the aug-system (dx, ds, dy_c, dy_d).
        let mut sol = res.fresh_zeroed();

        // Number of negative eigenvalues we expect.
        let num_neg_evals = rhs.y_c.dim() + rhs.y_d.dim();

        let curr_mu = data.borrow().curr_mu;

        // Upstream's `IpPDFullSpaceSolver::SolveOnce` (cpp:457-482)
        // splits on `(uptodate && !pretend_singular)`: if the matrix is
        // unchanged since the last `SolveOnce` and we are not faking a
        // singularity, reuse the existing perturbation, do a single
        // back-solve with `check_inertia=false`, and return. Iterative
        // refinement and the post-`IncreaseQuality` retry both land
        // here. Calling `ConsiderNewSystem` again on a same-matrix
        // re-solve would corrupt the perturbation handler's
        // `delta_x_last` bookkeeping.
        if self.matrix_considered && !pretend_singular {
            let d = self.perturb.borrow().current_perturbation();
            let coeffs = AugSysCoeffs {
                w: Some(b.w),
                w_factor: 1.0,
                d_x: Some(b.sigma_x),
                delta_x: d.delta_x,
                d_s: Some(b.sigma_s),
                delta_s: d.delta_s,
                j_c: b.j_c,
                d_c: None,
                delta_c: d.delta_c,
                j_d: b.j_d,
                d_d: None,
                delta_d: d.delta_d,
            };
            let aug_rhs = AugSysRhs {
                rhs_x: &*aug_rhs_x,
                rhs_s: &*aug_rhs_s,
                rhs_c: &*rhs.y_c,
                rhs_d: &*rhs.y_d,
            };
            let mut aug_sol = AugSysSol {
                sol_x: &mut *sol.x,
                sol_s: &mut *sol.s,
                sol_c: &mut *sol.y_c,
                sol_d: &mut *sol.y_d,
            };
            // Same matrix, same perturbations, inertia already known —
            // use the cached factor and avoid the per-call refactor
            // that otherwise dominates MA57 wall-time on long iter-ref
            // loops (cont5_2_4_l drops 97s → ~30s).
            let retval = self.aug_solver.resolve(&coeffs, &aug_rhs, &mut aug_sol);
            if retval != ESymSolverStatus::Success {
                return false;
            }
            // Stash perturbations on data, expand bound multipliers,
            // assemble final res, and return — skipping the
            // escalation loop entirely (matches upstream's `if(uptodate
            // && !pretend_singular)` branch in IpPDFullSpaceSolver.cpp).
            {
                let mut dm = data.borrow_mut();
                dm.perturbations.delta_x = d.delta_x;
                dm.perturbations.delta_s = d.delta_s;
                dm.perturbations.delta_c = d.delta_c;
                dm.perturbations.delta_d = d.delta_d;
            }
            expand_bound_multipliers(b, rhs, &mut sol);
            let frozen_sol = sol.freeze();
            res.add_one_vector(alpha, &frozen_sol, beta);
            return true;
        }

        let mut deltas = self
            .perturb
            .borrow_mut()
            .consider_new_system(curr_mu, Some(data));
        let Some(mut d) = deltas.take() else {
            return false;
        };

        let mut count = 0_i32;
        let mut retval;
        loop {
            if pretend_singular {
                retval = ESymSolverStatus::Singular;
                pretend_singular = false;
            } else {
                count += 1;
                let check_inertia = self.neg_curv_test_tol <= 0.0;
                let coeffs = AugSysCoeffs {
                    w: Some(b.w),
                    w_factor: 1.0,
                    d_x: Some(b.sigma_x),
                    delta_x: d.delta_x,
                    d_s: Some(b.sigma_s),
                    delta_s: d.delta_s,
                    j_c: b.j_c,
                    d_c: None,
                    delta_c: d.delta_c,
                    j_d: b.j_d,
                    d_d: None,
                    delta_d: d.delta_d,
                };
                let aug_rhs = AugSysRhs {
                    rhs_x: &*aug_rhs_x,
                    rhs_s: &*aug_rhs_s,
                    rhs_c: &*rhs.y_c,
                    rhs_d: &*rhs.y_d,
                };
                let mut aug_sol = AugSysSol {
                    sol_x: &mut *sol.x,
                    sol_s: &mut *sol.s,
                    sol_c: &mut *sol.y_c,
                    sol_d: &mut *sol.y_d,
                };
                retval = self.aug_solver.solve(
                    &coeffs,
                    &aug_rhs,
                    &mut aug_sol,
                    check_inertia,
                    num_neg_evals,
                );
            }

            if retval == ESymSolverStatus::FatalError {
                return false;
            }

            if retval == ESymSolverStatus::Singular && (rhs.y_c.dim() + rhs.y_d.dim() > 0) {
                let curr_mu = data.borrow().curr_mu;
                let next = self
                    .perturb
                    .borrow_mut()
                    .perturb_for_singular(curr_mu, Some(data));
                let Some(nd) = next else { return false };
                d = nd;
            } else if retval == ESymSolverStatus::WrongInertia
                && self.aug_solver.number_of_neg_evals() < num_neg_evals
            {
                let mut assume_singular = true;
                if !self.augsys_improved {
                    self.augsys_improved = self.aug_solver.increase_quality();
                    if self.augsys_improved {
                        data.borrow_mut().append_info_string("q");
                        assume_singular = false;
                    }
                }
                if assume_singular {
                    let curr_mu = data.borrow().curr_mu;
                    let next = self
                        .perturb
                        .borrow_mut()
                        .perturb_for_singular(curr_mu, Some(data));
                    let Some(nd) = next else { return false };
                    d = nd;
                    data.borrow_mut().append_info_string("a");
                }
            } else if retval == ESymSolverStatus::WrongInertia
                || retval == ESymSolverStatus::Singular
            {
                let curr_mu = data.borrow().curr_mu;
                let next = self
                    .perturb
                    .borrow_mut()
                    .perturb_for_wrong_inertia(curr_mu, Some(data));
                let Some(nd) = next else { return false };
                d = nd;
            }

            if retval == ESymSolverStatus::Success {
                break;
            }
        }
        let _ = count;

        // Stash the perturbation on data — upstream calls
        // `IpData().setPDPert(...)` here.
        {
            let mut dm = data.borrow_mut();
            dm.perturbations.delta_x = d.delta_x;
            dm.perturbations.delta_s = d.delta_s;
            dm.perturbations.delta_c = d.delta_c;
            dm.perturbations.delta_d = d.delta_d;
        }

        // Mark this matrix as "considered" so subsequent `solve_once`
        // re-calls within the same outer `solve()` (iterative refinement
        // / quality retry) take the single-solve path above.
        self.matrix_considered = true;

        expand_bound_multipliers(b, rhs, &mut sol);

        // res = α · sol + β · res
        let frozen_sol = sol.freeze();
        res.add_one_vector(alpha, &frozen_sol, beta);
        true
    }

    /// `resid = M · res − rhs` per `ComputeResiduals`. Skips terms
    /// whose perturbation is exactly zero.
    fn compute_residuals(
        &self,
        _data: &IpoptDataHandle,
        b: &SolveBlocks<'_>,
        rhs: &IteratesVector,
        res: &IteratesVectorMut,
        resid: &mut IteratesVectorMut,
    ) {
        let d = self.perturb.borrow().current_perturbation();

        // x: W·res.x + J_c^T·res.y_c + J_d^T·res.y_d
        //    − Px_L·res.z_L + Px_U·res.z_U + δ_x·res.x − rhs.x
        b.w.mult_vector(1.0, &*res.x, 0.0, &mut *resid.x);
        b.j_c.trans_mult_vector(1.0, &*res.y_c, 1.0, &mut *resid.x);
        b.j_d.trans_mult_vector(1.0, &*res.y_d, 1.0, &mut *resid.x);
        b.px_l.mult_vector(-1.0, &*res.z_l, 1.0, &mut *resid.x);
        b.px_u.mult_vector(1.0, &*res.z_u, 1.0, &mut *resid.x);
        // resid.x += δ_x·res.x − rhs.x
        resid
            .x
            .add_two_vectors(d.delta_x, &*res.x, -1.0, &*rhs.x, 1.0);

        // s: Pd_U·res.v_U − Pd_L·res.v_L − res.y_d − rhs.s + δ_s·res.s
        b.pd_u.mult_vector(1.0, &*res.v_u, 0.0, &mut *resid.s);
        b.pd_l.mult_vector(-1.0, &*res.v_l, 1.0, &mut *resid.s);
        resid.s.add_two_vectors(-1.0, &*res.y_d, -1.0, &*rhs.s, 1.0);
        if d.delta_s != 0.0 {
            resid.s.axpy(d.delta_s, &*res.s);
        }

        // c: J_c·res.x − δ_c·res.y_c − rhs.y_c
        b.j_c.mult_vector(1.0, &*res.x, 0.0, &mut *resid.y_c);
        resid
            .y_c
            .add_two_vectors(-d.delta_c, &*res.y_c, -1.0, &*rhs.y_c, 1.0);

        // d: J_d·res.x − res.s − rhs.y_d − δ_d·res.y_d
        b.j_d.mult_vector(1.0, &*res.x, 0.0, &mut *resid.y_d);
        resid
            .y_d
            .add_two_vectors(-1.0, &*res.s, -1.0, &*rhs.y_d, 1.0);
        if d.delta_d != 0.0 {
            resid.y_d.axpy(-d.delta_d, &*res.y_d);
        }

        // zL: res.z_L · slack_x_L + (Px_L^T·res.x) · z_L − rhs.z_L
        resid.z_l.copy(&*res.z_l);
        resid.z_l.element_wise_multiply(b.slack_x_l);
        let mut tmp_zl = b.z_l.make_new();
        b.px_l.trans_mult_vector(1.0, &*res.x, 0.0, &mut *tmp_zl);
        tmp_zl.element_wise_multiply(b.z_l);
        resid
            .z_l
            .add_two_vectors(1.0, &*tmp_zl, -1.0, &*rhs.z_l, 1.0);

        // zU: res.z_U · slack_x_U − (Px_U^T·res.x) · z_U − rhs.z_U
        resid.z_u.copy(&*res.z_u);
        resid.z_u.element_wise_multiply(b.slack_x_u);
        let mut tmp_zu = b.z_u.make_new();
        b.px_u.trans_mult_vector(1.0, &*res.x, 0.0, &mut *tmp_zu);
        tmp_zu.element_wise_multiply(b.z_u);
        resid
            .z_u
            .add_two_vectors(-1.0, &*tmp_zu, -1.0, &*rhs.z_u, 1.0);

        // vL: res.v_L · slack_s_L + (Pd_L^T·res.s) · v_L − rhs.v_L
        resid.v_l.copy(&*res.v_l);
        resid.v_l.element_wise_multiply(b.slack_s_l);
        let mut tmp_vl = b.v_l.make_new();
        b.pd_l.trans_mult_vector(1.0, &*res.s, 0.0, &mut *tmp_vl);
        tmp_vl.element_wise_multiply(b.v_l);
        resid
            .v_l
            .add_two_vectors(1.0, &*tmp_vl, -1.0, &*rhs.v_l, 1.0);

        // vU: res.v_U · slack_s_U − (Pd_U^T·res.s) · v_U − rhs.v_U
        resid.v_u.copy(&*res.v_u);
        resid.v_u.element_wise_multiply(b.slack_s_u);
        let mut tmp_vu = b.v_u.make_new();
        b.pd_u.trans_mult_vector(1.0, &*res.s, 0.0, &mut *tmp_vu);
        tmp_vu.element_wise_multiply(b.v_u);
        resid
            .v_u
            .add_two_vectors(-1.0, &*tmp_vu, -1.0, &*rhs.v_u, 1.0);
    }

    /// `nrm_resid / (min(nrm_res, max_cond·nrm_rhs) + nrm_rhs)`, with
    /// `max_cond = 1e6`. Mirrors `ComputeResidualRatio`.
    fn compute_residual_ratio(
        &self,
        rhs: &IteratesVector,
        res: &IteratesVectorMut,
        resid: &IteratesVectorMut,
    ) -> Number {
        let nrm_rhs = rhs.amax();
        let nrm_res = res.amax();
        let nrm_resid = resid.amax();
        if nrm_rhs + nrm_res == 0.0 {
            nrm_resid
        } else {
            let max_cond = 1e6;
            nrm_resid / (nrm_res.min(max_cond * nrm_rhs) + nrm_rhs)
        }
    }
}

impl PdSystemSolver for PdFullSpaceSolver {
    fn solve_status(&self) -> ESymSolverStatus {
        self.last_status.unwrap_or(ESymSolverStatus::FatalError)
    }
}

/// Bag of borrowed blocks used by both `solve_once` and
/// `compute_residuals` — keeps argument lists tractable.
struct SolveBlocks<'a> {
    w: &'a dyn SymMatrix,
    j_c: &'a dyn Matrix,
    j_d: &'a dyn Matrix,
    px_l: &'a dyn Matrix,
    px_u: &'a dyn Matrix,
    pd_l: &'a dyn Matrix,
    pd_u: &'a dyn Matrix,
    z_l: &'a dyn Vector,
    z_u: &'a dyn Vector,
    v_l: &'a dyn Vector,
    v_u: &'a dyn Vector,
    slack_x_l: &'a dyn Vector,
    slack_x_u: &'a dyn Vector,
    slack_s_l: &'a dyn Vector,
    slack_s_u: &'a dyn Vector,
    sigma_x: &'a dyn Vector,
    sigma_s: &'a dyn Vector,
}

/// Helper trait extension on `IteratesVectorMut` for fresh zeroed
/// allocations matching the same shape — the shape lives implicitly
/// in the existing components' `dim()`.
trait FreshZeroed {
    fn fresh_zeroed(&self) -> IteratesVectorMut;
}

impl FreshZeroed for IteratesVectorMut {
    fn fresh_zeroed(&self) -> IteratesVectorMut {
        IteratesVectorMut {
            x: self.x.make_new(),
            s: self.s.make_new(),
            y_c: self.y_c.make_new(),
            y_d: self.y_d.make_new(),
            z_l: self.z_l.make_new(),
            z_u: self.z_u.make_new(),
            v_l: self.v_l.make_new(),
            v_u: self.v_u.make_new(),
        }
    }
}

/// Snapshot a mutable iterate into a frozen, shareable copy without
/// consuming it. Used to remember `res_in` when β ≠ 0.
fn snapshot_mut(m: &IteratesVectorMut) -> IteratesVector {
    let mut out = m.fresh_zeroed();
    out.x.copy(&*m.x);
    out.s.copy(&*m.s);
    out.y_c.copy(&*m.y_c);
    out.y_d.copy(&*m.y_d);
    out.z_l.copy(&*m.z_l);
    out.z_u.copy(&*m.z_u);
    out.v_l.copy(&*m.v_l);
    out.v_u.copy(&*m.v_u);
    out.freeze()
}

/// Convert a frozen `IteratesVector` back to a mutable owned form.
/// Allocates fresh storage and copies; the iterative-refinement loop
/// re-freezes/thaws once per iteration, so a single per-component
/// copy is acceptable.
/// Expand the four bound-multiplier blocks of `sol` from the just-
/// computed primal-step blocks (`sol.x`, `sol.s`):
///
/// ```text
/// sol.z_L = S_xL⁻¹ · (rhs.z_L − z_L · (Px_L^T · sol.x))
/// sol.z_U = S_xU⁻¹ · (rhs.z_U + z_U · (Px_U^T · sol.x))
/// sol.v_L = S_sL⁻¹ · (rhs.v_L − v_L · (Pd_L^T · sol.s))
/// sol.v_U = S_sU⁻¹ · (rhs.v_U + v_U · (Pd_U^T · sol.s))
/// ```
///
/// Encoded via `SinvBlrmZMTdBr` with `α = ±1`. Mirrors the bound-
/// multiplier expansion at the bottom of upstream's
/// `IpPDFullSpaceSolver::SolveOnce`.
fn expand_bound_multipliers(
    b: &SolveBlocks<'_>,
    rhs: &IteratesVector,
    sol: &mut IteratesVectorMut,
) {
    b.px_l
        .sinv_blrm_zmt_dbr(-1.0, b.slack_x_l, &*rhs.z_l, b.z_l, &*sol.x, &mut *sol.z_l);
    b.px_u
        .sinv_blrm_zmt_dbr(1.0, b.slack_x_u, &*rhs.z_u, b.z_u, &*sol.x, &mut *sol.z_u);
    b.pd_l
        .sinv_blrm_zmt_dbr(-1.0, b.slack_s_l, &*rhs.v_l, b.v_l, &*sol.s, &mut *sol.v_l);
    b.pd_u
        .sinv_blrm_zmt_dbr(1.0, b.slack_s_u, &*rhs.v_u, b.v_u, &*sol.s, &mut *sol.v_u);
}

/// Copy a `DenseVector`'s materialized values into `dst`. Used by
/// `solve_many_cached` to pack the aug-system RHS into a column of the
/// flat `aug_packed` buffer.
fn copy_vector_to_slice(src: &dyn Vector, dst: &mut [Number]) {
    if dst.is_empty() {
        return;
    }
    let dv = src
        .as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
        .expect("solve_many_cached requires DenseVector blocks");
    if dv.is_homogeneous() {
        let v = dv.scalar();
        dst.iter_mut().for_each(|x| *x = v);
    } else {
        dst.copy_from_slice(dv.values());
    }
}

/// Inverse of [`copy_vector_to_slice`]: write `src` into a
/// `DenseVector` in place.
fn set_vector_from_slice(dst: &mut dyn Vector, src: &[Number]) {
    if src.is_empty() {
        return;
    }
    let dv = dst
        .as_any_mut()
        .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
        .expect("solve_many_cached requires DenseVector blocks");
    dv.set_values(src);
}

/// Downcast a `dyn Vector` block to its concrete `DenseVector` slice.
/// Returns `None` if the block is not a `DenseVector`, or if the block
/// is non-empty but stored as a homogeneous scalar (the
/// `solve_many_cached_flat` fast path needs a real slice for its
/// inline scatter loops; the closure-based fallback handles
/// homogeneous-on-non-empty correctly via `add_m_sinv_z` / `sinv_blrm_zmt_dbr`).
fn dense_slice_or_none(v: &dyn Vector, expected_dim: usize) -> Option<&[Number]> {
    if expected_dim == 0 {
        // An empty block doesn't need a slice — the scatter loops below
        // simply don't iterate when exp_pos is empty. Return an empty
        // slice so the caller can pass it through unconditionally.
        return Some(&[]);
    }
    let dv = v.as_any().downcast_ref::<DenseVector>()?;
    if dv.is_homogeneous() {
        return None;
    }
    Some(dv.values())
}

/// Downcast a `dyn Matrix` to its concrete `ExpansionMatrix`'s
/// expanded-position index slice. Returns `None` if the matrix is not
/// an `ExpansionMatrix`.
fn exp_pos_or_none(m: &dyn Matrix) -> Option<&[Index]> {
    let em = m.as_any().downcast_ref::<ExpansionMatrix>()?;
    Some(em.expanded_pos_indices())
}

/// Phase-1 inner kernel: `out[exp_pos[i]] += alpha · src[i] / denom[i]`.
/// Hot loop in `solve_many_cached_flat`. Specialised on `alpha = ±1`
/// to skip the multiply.
#[inline]
fn scatter_add_div(
    out: &mut [Number],
    exp_pos: &[Index],
    src: &[Number],
    denom: &[Number],
    alpha: Number,
) {
    if exp_pos.is_empty() {
        return;
    }
    debug_assert_eq!(src.len(), exp_pos.len());
    debug_assert_eq!(denom.len(), exp_pos.len());
    if alpha == 1.0 {
        for i in 0..exp_pos.len() {
            out[exp_pos[i] as usize] += src[i] / denom[i];
        }
    } else if alpha == -1.0 {
        for i in 0..exp_pos.len() {
            out[exp_pos[i] as usize] -= src[i] / denom[i];
        }
    } else {
        for i in 0..exp_pos.len() {
            out[exp_pos[i] as usize] += alpha * src[i] / denom[i];
        }
    }
}

/// Phase-3 inner kernel: bound-multiplier expansion,
/// `out[i] = (r[i] + alpha · z[i] · sol[exp_pos[i]]) / s[i]`.
/// Mirrors `ExpansionMatrix::sinv_blrm_zmt_dbr_impl` (the non-
/// homogeneous specialisation) inlined against raw slices.
#[inline]
#[allow(clippy::too_many_arguments)]
fn expand_bound_mult(
    out: &mut [Number],
    r: &[Number],
    z: &[Number],
    sol: &[Number],
    exp_pos: &[Index],
    s: &[Number],
    alpha: Number,
) {
    if exp_pos.is_empty() {
        return;
    }
    debug_assert_eq!(out.len(), exp_pos.len());
    debug_assert_eq!(r.len(), exp_pos.len());
    debug_assert_eq!(z.len(), exp_pos.len());
    debug_assert_eq!(s.len(), exp_pos.len());
    if alpha == 1.0 {
        for i in 0..exp_pos.len() {
            out[i] = (r[i] + z[i] * sol[exp_pos[i] as usize]) / s[i];
        }
    } else if alpha == -1.0 {
        for i in 0..exp_pos.len() {
            out[i] = (r[i] - z[i] * sol[exp_pos[i] as usize]) / s[i];
        }
    } else {
        for i in 0..exp_pos.len() {
            out[i] = (r[i] + alpha * z[i] * sol[exp_pos[i] as usize]) / s[i];
        }
    }
}

fn thaw(iv: IteratesVector) -> IteratesVectorMut {
    fn one(v: Rc<dyn Vector>) -> Box<dyn Vector> {
        let mut b = v.make_new();
        b.copy(&*v);
        b
    }
    IteratesVectorMut {
        x: one(iv.x),
        s: one(iv.s),
        y_c: one(iv.y_c),
        y_d: one(iv.y_d),
        z_l: one(iv.z_l),
        z_u: one(iv.z_u),
        v_l: one(iv.v_l),
        v_u: one(iv.v_u),
    }
}

/// Internal placeholder used only inside [`PdFullSpaceSolver::wrap_aug_solver`]
/// to satisfy `std::mem::replace`'s requirement for a value of the same
/// type while the real boxed solver is being moved through the wrapper
/// closure. None of the trait methods are ever invoked.
struct NoopAugSolver;

impl AugSystemSolver for NoopAugSolver {
    fn provides_inertia(&self) -> bool {
        unreachable!("NoopAugSolver is a transient placeholder")
    }
    fn number_of_neg_evals(&self) -> Index {
        unreachable!("NoopAugSolver is a transient placeholder")
    }
    fn increase_quality(&mut self) -> bool {
        unreachable!("NoopAugSolver is a transient placeholder")
    }
    fn last_solve_status(&self) -> ESymSolverStatus {
        unreachable!("NoopAugSolver is a transient placeholder")
    }
    fn solve(
        &mut self,
        _coeffs: &AugSysCoeffs<'_>,
        _rhs: &AugSysRhs<'_>,
        _sol: &mut AugSysSol<'_>,
        _check_neg_evals: bool,
        _num_neg_evals: Index,
    ) -> ESymSolverStatus {
        unreachable!("NoopAugSolver is a transient placeholder")
    }
}
