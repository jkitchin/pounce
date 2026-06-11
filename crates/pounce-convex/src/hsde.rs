//! Homogeneous self-dual embedding (HSDE) driver for the convex IPM.
//!
//! This is the foundation for Clarabel cone parity (see
//! `dev-notes/hsde.md` and `dev-notes/clarabel-parity.md`). It reformulates
//! the interior-point iteration as a *single self-dual system* in the
//! embedded variables `(x, y, z, s, τ, κ)`, so that
//!
//! - a self-starting iterate handles primal- and dual-infeasible problems
//!   uniformly (no infeasible start), and
//! - infeasibility/unboundedness falls out of the embedding (`τ → 0`,
//!   `κ > 0`) rather than from a bolt-on certificate watch.
//!
//! **The per-cone math and the KKT factorization are reused verbatim.** The
//! embedding borders the existing symmetric `(x, y, z)` block `M`
//! (assembled by [`crate::ipm::KktStructure`], with each cone's NT scaling
//! `W²` from [`Cone::kkt_block`]) by the scalar `τ`, and solves it with
//! **two** back-solves through the *same* factorization plus a scalar Schur
//! complement (the SCS/ECOS scheme): `M p = (−c, b, h)` (the constant
//! direction) and `M q = residual`, combined with `Δτ` from the τ/κ row.
//!
//! ## Scope (Phases H2–H3)
//!
//! This driver implements the embedding over a product of nonnegative-orthant
//! and second-order cones — it solves LPs, QPs, and SOCPs (the full current
//! problem class). The **quadratic objective** (`P ⪰ 0`) is handled via
//! Clarabel's QP embedding: the τ-row gains the `xᵀPx/τ` coupling, so its
//! gradient becomes `g̃ = (c + (2/τ)Px, b, h)` and its scalar Schur
//! complement a `−xᵀPx/τ²` term. Crucially, `P` already sits in `M`'s
//! `(x, x)` block and in the dual residual `ρ_x`, so the two M-solves, the
//! cone elimination, and the step are *identical* to the linear case — only
//! the τ-row scalar is new (and reduces to the linear case at `P = 0`).
//!
//! The switch-over to make HSDE the default (Phase H4) still follows; for
//! now `solve_qp_ipm`/`solve_socp_ipm` remain the production path and this
//! module is validated to reproduce their optima and certificates.

use crate::cones::{CompositeCone, Cone};
use crate::debug::{fire, ConvexDebugState};
use crate::ipm::{
    build_factorization, build_rhs, detect_infeasibility_cone, dot, inf_norm, split_step, QpOptions,
};
use crate::qp::{QpIterate, QpProblem, QpSolution, QpStatus};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
use pounce_linsol::SparseSymLinearSolverInterface;

/// Fraction-to-boundary step for a positive scalar ray `v + α dv > 0`,
/// scaled by `tau` and capped at 1 (the scalar analogue of `Cone::max_step`
/// for the homogenizing variables `τ`, `κ`).
fn ray_step(v: f64, dv: f64, tau: f64) -> f64 {
    if dv < 0.0 {
        (tau * (-v / dv)).min(1.0)
    } else {
        1.0
    }
}

/// Solve `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ⪯_K h` via the homogeneous
/// self-dual embedding, returning the un-homogenized solution. `P = 0` is an
/// LP/SOCP; `P ⪰ 0` a QP (the τ-row picks up the `xᵀPx/τ` coupling).
///
/// `cone` is the product cone `K` over the `m_ineq` inequality rows (built
/// by the caller exactly as for [`crate::ipm::solve_socp_ipm`]). Variable
/// bounds must already be expanded into `cone` rows by the caller.
pub(crate) fn solve_conic_hsde<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    mut make_backend: F,
    mut hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let degree = cone.degree();

    let (kkt, mut fact) = match build_factorization(prob, cone, opts, &mut make_backend) {
        Ok(pair) => pair,
        Err(()) => return failed(prob),
    };

    // Constant border data: −b, −h (so `build_rhs` yields the `(−c, b, h)`
    // right-hand side of the constant direction `p`).
    let neg_b: Vec<f64> = prob.b.iter().map(|v| -v).collect();
    let neg_h: Vec<f64> = prob.h.iter().map(|v| -v).collect();
    let zeros_m = vec![0.0; m_ineq];

    // Self-dual start: x = y = 0, s = z = e (cone identity), τ = κ = 1.
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m_eq];
    let mut e = vec![0.0; m_ineq];
    cone.identity(&mut e);
    let mut s = e.clone();
    let mut z = e;
    let mut tau = 1.0_f64;
    let mut kappa = 1.0_f64;

    // Residual + work buffers.
    let mut rho_x = vec![0.0; n];
    let mut rho_y = vec![0.0; m_eq];
    let mut rho_z = vec![0.0; m_ineq];
    let mut px_vec = vec![0.0; n]; // P x (quadratic-objective coupling)
    let mut r_c = vec![0.0; m_ineq];
    let mut comp = vec![0.0; m_ineq];
    let mut kkt_vals = kkt.values.clone();
    let mut rhs = vec![0.0; kkt.dim];

    // Direction buffers: p = constant direction, (dx,dy,dz) = the running
    // step, with affine slack/dual kept for the Mehrotra corrector.
    let mut p_x = vec![0.0; n];
    let mut p_y = vec![0.0; m_eq];
    let mut p_z = vec![0.0; m_ineq];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];

    let mut status = QpStatus::IterationLimit;
    let mut iters = 0;
    // Opt-in per-iteration convergence trace (mirrors the direct path's
    // `collect_iterates`): one record per stepping iteration plus a terminal
    // record at the converged iterate (α = 0).
    let mut trace: Vec<QpIterate> = Vec::new();

    for it in 0..opts.max_iter {
        iters = it;

        // --- quadratic-objective coupling: Px and xᵀPx (zero for an LP) ---
        for v in px_vec.iter_mut() {
            *v = 0.0;
        }
        prob.p_mul(&x, &mut px_vec);
        let xpx = dot(&x, &px_vec);

        // --- homogeneous residuals ---
        // ρ_x = P x + Aᵀy + Gᵀz + c·τ
        for (r, (&ci, &pxi)) in rho_x.iter_mut().zip(prob.c.iter().zip(&px_vec)) {
            *r = ci * tau + pxi;
        }
        prob.at_mul(&y, &mut rho_x);
        prob.gt_mul(&z, &mut rho_x);
        // ρ_y = A x − b·τ
        for (r, &bi) in rho_y.iter_mut().zip(&prob.b) {
            *r = -bi * tau;
        }
        prob.a_mul(&x, &mut rho_y);
        // ρ_z = G x + s − h·τ
        for i in 0..m_ineq {
            rho_z[i] = s[i] - prob.h[i] * tau;
        }
        prob.g_mul(&x, &mut rho_z);
        // ρ_τ = κ + cᵀx + bᵀy + hᵀz + xᵀPx/τ
        let ctx = dot(&prob.c, &x);
        let bty = dot(&prob.b, &y);
        let htz = dot(&prob.h, &z);
        let rho_tau = kappa + ctx + bty + htz + xpx / tau;

        let sz = dot(&s, &z);
        let mu = (sz + tau * kappa) / (degree as f64 + 1.0);

        // --- convergence (un-homogenized residuals; divide out τ) ---
        // Gap = x̂ᵀPx̂ + cᵀx̂ + bᵀŷ + hᵀẑ = (xᵀPx/τ + cᵀx + bᵀy + hᵀz)/τ.
        let pres = inf_norm(&rho_y).max(inf_norm(&rho_z)) / tau;
        let dres = inf_norm(&rho_x) / tau;
        let gap = (xpx / tau + ctx + bty + htz).abs() / tau;
        let res = pres.max(dres).max(gap);
        // "Acceptable level": near the cone boundary the scaling/factorization
        // can break down a hair short of `tol`. If the unregularized KKT
        // residuals are already tiny (within `~1e3·tol`) when that happens, the
        // current iterate *is* essentially optimal — accept it rather than
        // reporting a spurious `NumericalFailure`. This mirrors the
        // non-symmetric HSDE driver (`hsde_nonsym.rs`), which already does this;
        // the two drivers were inconsistent (the symmetric one discarded usable
        // SOC/orthant iterates that the non-symmetric one would have accepted).
        let near_opt = res < 1e3 * opts.tol;
        // Un-homogenized objective `½x̂ᵀPx̂ + cᵀx̂` (x̂ = x/τ) — what the
        // trace and debugger report.
        let obj_hat = 0.5 * xpx / (tau * tau) + ctx / tau;

        // Debugger checkpoint: top of iteration. Blocks expose the
        // homogeneous iterate `(x, s, y, z, τ, κ)`; the objective is the
        // un-homogenized `½x̂ᵀPx̂ + cᵀx̂` with `x̂ = x/τ` (what the user reads).
        if hook.is_some() {
            let mut st = ConvexDebugState {
                cp: Checkpoint::IterStart,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (0.0, 0.0),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        if pres < opts.tol && dres < opts.tol && gap < opts.tol {
            status = QpStatus::Optimal;
            // Terminal record at the converged iterate (no step taken).
            if opts.collect_iterates {
                trace.push(QpIterate {
                    iter: it,
                    objective: obj_hat,
                    primal_infeasibility: pres,
                    dual_infeasibility: dres,
                    mu,
                    alpha_primal: 0.0,
                    alpha_dual: 0.0,
                });
            }
            break;
        }

        // --- infeasibility (the embedding drives the iterate onto the
        // Farkas/recession ray as τ → 0; the same verified relative checks
        // as the direct driver apply to the homogeneous (x, y, z)). ---
        if tau < 1e-2 * kappa.max(1.0) {
            if let Some(st) = detect_infeasibility_cone(prob, &x, &y, &z, opts, cone) {
                status = st;
                break;
            }
        }

        // --- refactor M with the current cone scaling ---
        kkt.update_blocks(cone, &s, &z, opts.reg, &mut kkt_vals);
        if fact.refactor(&kkt_vals).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }

        // --- constant direction p: M p = (−c, b, h) ---
        build_rhs(&prob.c, &neg_b, &neg_h, &zeros_m, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut p_x, &mut p_y, &mut p_z);
        // τ-row gradient g̃ = (c + (2/τ)Px, b, h) and the scalar Schur
        // denominator g̃ᵀp − κ/τ − xᵀPx/τ² (the last two terms are the τ/κ
        // ray and the quadratic coupling; both vanish for an LP).
        let two_over_tau = 2.0 / tau;
        let gtp = dot(&prob.c, &p_x)
            + two_over_tau * dot(&px_vec, &p_x)
            + dot(&prob.b, &p_y)
            + dot(&prob.h, &p_z);
        let denom = gtp - kappa / tau - xpx / (tau * tau);

        // === Predictor (affine, σ = 0) ===
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
        build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        let gtq = dot(&prob.c, &dx)
            + two_over_tau * dot(&px_vec, &dx)
            + dot(&prob.b, &dy)
            + dot(&prob.h, &dz);
        // Δτ = [−ρ_τ − g̃ᵀq − (σμ − τκ)/τ] / denom; predictor σμ = 0,
        // so −(0 − τκ)/τ = +κ.
        let dtau_aff = (-rho_tau - gtq + kappa) / denom;
        // Full affine directions dw = q + Δτ·p (only dz needed downstream).
        for i in 0..m_ineq {
            dz_aff[i] = dz[i] + dtau_aff * p_z[i];
        }
        let dkappa_aff = (-tau * kappa - kappa * dtau_aff) / tau;
        cone.recover_ds(&s, &z, &r_c, &dz_aff, &mut ds_aff);

        // Affine step length over the cone and the τ/κ rays.
        let mut alpha_aff =
            ray_step(tau, dtau_aff, opts.tau).min(ray_step(kappa, dkappa_aff, opts.tau));
        if m_ineq > 0 {
            alpha_aff = alpha_aff
                .min(cone.max_step(&s, &ds_aff, opts.tau))
                .min(cone.max_step(&z, &dz_aff, opts.tau));
        }
        // μ_aff and Mehrotra centering σ = (μ_aff/μ)³.
        let mut dot_aff = (tau + alpha_aff * dtau_aff) * (kappa + alpha_aff * dkappa_aff);
        for i in 0..m_ineq {
            dot_aff += (s[i] + alpha_aff * ds_aff[i]) * (z[i] + alpha_aff * dz_aff[i]);
        }
        let mu_aff = dot_aff / (degree as f64 + 1.0);
        let sigma = if mu > 0.0 { (mu_aff / mu).powi(3) } else { 0.0 };
        let sigma_mu = sigma * mu;

        // === Corrector (centered target + second-order term) ===
        cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
        build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        let gtq = dot(&prob.c, &dx)
            + two_over_tau * dot(&px_vec, &dx)
            + dot(&prob.b, &dy)
            + dot(&prob.h, &dz);
        // τκ corrector residual: τκ + Δτ_aff·Δκ_aff (target σμ).
        let r_tk = tau * kappa + dtau_aff * dkappa_aff;
        let dtau = (-rho_tau - gtq - (sigma_mu - r_tk) / tau) / denom;
        // Combine: dw = q + Δτ·p.
        for i in 0..n {
            dx[i] += dtau * p_x[i];
        }
        for i in 0..m_eq {
            dy[i] += dtau * p_y[i];
        }
        for i in 0..m_ineq {
            dz[i] += dtau * p_z[i];
        }
        let dkappa = (sigma_mu - r_tk - kappa * dtau) / tau;
        cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

        // Single fraction-to-boundary step (HSDE is primal/dual-symmetric).
        let mut alpha = ray_step(tau, dtau, opts.tau).min(ray_step(kappa, dkappa, opts.tau));
        if m_ineq > 0 {
            alpha = alpha
                .min(cone.max_step(&s, &ds, opts.tau))
                .min(cone.max_step(&z, &dz, opts.tau));
        }

        // Debugger checkpoint: the combined Newton direction and the single
        // symmetric step length are known but not yet applied (α reported
        // in both the primal and dual slots).
        // Stepping record: the residuals/μ/objective at the start of this
        // iteration, paired with the symmetric step length just computed.
        if opts.collect_iterates {
            trace.push(QpIterate {
                iter: it,
                objective: obj_hat,
                primal_infeasibility: pres,
                dual_infeasibility: dres,
                mu,
                alpha_primal: alpha,
                alpha_dual: alpha,
            });
        }

        if hook.is_some() {
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterSearchDirection,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (alpha, alpha),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_eq {
            y[i] += alpha * dy[i];
        }
        for i in 0..m_ineq {
            s[i] += alpha * ds[i];
            z[i] += alpha * dz[i];
        }
        tau += alpha * dtau;
        kappa += alpha * dkappa;

        // Debugger checkpoint: the new homogeneous iterate is in place.
        if hook.is_some() {
            // Recompute the objective at the *new* point (`x`, `τ` just moved).
            let mut pxn = vec![0.0; n];
            prob.p_mul(&x, &mut pxn);
            let obj_hat = 0.5 * dot(&x, &pxn) / (tau * tau) + dot(&prob.c, &x) / tau;
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterStep,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (alpha, alpha),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }
    }

    // Un-homogenize: divide by τ to recover the original-space solution.
    let inv = if tau.abs() > 0.0 { 1.0 / tau } else { 1.0 };
    let mut x: Vec<f64> = x.iter().map(|v| v * inv).collect();
    let mut y: Vec<f64> = y.iter().map(|v| v * inv).collect();
    let mut z: Vec<f64> = z.iter().map(|v| v * inv).collect();
    // Objective ½xᵀPx + cᵀx.
    let mut px = vec![0.0; n];
    prob.p_mul(&x, &mut px);
    let obj = 0.5 * dot(&x, &px) + dot(&prob.c, &x);

    // Debugger post-mortem at the recovered (un-homogenized) solution. `s`
    // stays in its homogeneous scaling; `dx`/… are the last step.
    if hook.is_some() {
        let status_str = format!("{status:?}");
        let mut st = ConvexDebugState {
            cp: Checkpoint::Terminated,
            iter: iters as i32,
            mu: 0.0,
            pinf: 0.0,
            dinf: 0.0,
            res: 0.0,
            obj,
            alpha: (0.0, 0.0),
            x: &mut x,
            s: &mut s,
            y: &mut y,
            z: &mut z,
            dx: &dx,
            dy: &dy,
            dz: &dz,
            ds: &ds,
            tau: Some(&mut tau),
            kappa: Some(&mut kappa),
            status: Some(&status_str),
        };
        let _ = fire(&mut hook, &mut st);
    }

    QpSolution {
        status,
        x,
        y,
        z,
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj,
        iters,
        iterates: trace,
    }
}

fn failed(prob: &QpProblem) -> QpSolution {
    QpSolution {
        status: QpStatus::NumericalFailure,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![1.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cones::ConeSpec;
    use crate::ipm::{solve_qp_ipm, solve_socp_ipm};
    use crate::qp::{QpProblem, Triplet};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn opts() -> QpOptions {
        QpOptions {
            max_iter: 200,
            ..QpOptions::default()
        }
    }

    /// Solve the same (P=0) problem with the HSDE driver and the direct
    /// driver; assert both converge and agree on the primal.
    fn assert_agrees(prob: &QpProblem, specs: &[ConeSpec], tol: f64) -> QpSolution {
        let cone = CompositeCone::from_specs(specs);
        let hsde = solve_conic_hsde(prob, &cone, &opts(), backend, None);
        let direct = solve_socp_ipm(prob, specs, &opts(), backend);
        assert_eq!(hsde.status, QpStatus::Optimal, "HSDE not optimal");
        assert_eq!(direct.status, QpStatus::Optimal, "direct not optimal");
        assert_eq!(hsde.x.len(), direct.x.len());
        for i in 0..hsde.x.len() {
            assert!(
                (hsde.x[i] - direct.x[i]).abs() < tol,
                "x[{i}] HSDE {} vs direct {}",
                hsde.x[i],
                direct.x[i]
            );
        }
        hsde
    }

    /// LP with one inequality and a known vertex optimum.
    /// min −x0 − x1 s.t. x0+x1 ≤ 1, x ≥ 0  → obj −1 on the face x0+x1=1.
    #[test]
    fn lp_inequality_matches_direct() {
        // rows: x0+x1 ≤ 1 ; −x0 ≤ 0 ; −x1 ≤ 0  (all nonneg slacks)
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(2, 1, -1.0),
            ],
            h: vec![1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::Nonneg(3)], 1e-6);
        assert!((sol.obj + 1.0).abs() < 1e-6, "obj {}", sol.obj);
        assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// LP with an equality constraint: min cᵀx s.t. 1ᵀx = 1, x ≥ 0.
    /// min x0 + 2x1 s.t. x0+x1=1, x≥0  → x=(1,0), obj 1.
    #[test]
    fn lp_equality_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 2.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 1, -1.0)],
            h: vec![0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::Nonneg(2)], 1e-6);
        assert!((sol.obj - 1.0).abs() < 1e-5, "obj {}", sol.obj);
        assert!(sol.x[0] > 0.99 && sol.x[1] < 1e-4, "x {:?}", sol.x);
    }

    /// SOCP norm minimization: min t s.t. (t, x−a) ∈ SOC(3).
    /// With G=−I, h=(0,−a0,−a1): optimum t=0, x=a.
    #[test]
    fn socp_norm_min_matches_direct() {
        let a = [2.0_f64, -1.0];
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![1.0, 0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(2, 2, -1.0),
            ],
            h: vec![0.0, -a[0], -a[1]],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::SecondOrder(3)], 1e-5);
        assert!(sol.x[0].abs() < 1e-5, "t {}", sol.x[0]);
        assert!((sol.x[1] - a[0]).abs() < 1e-5 && (sol.x[2] - a[1]).abs() < 1e-5);
    }

    /// Mixed cone: a nonneg row and a second-order block together.
    /// min −x1 s.t. x1 ≤ 0.5 (nonneg), ‖x‖ ≤ 1 (soc (1,x0,x1)).
    #[test]
    fn socp_mixed_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![0.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 1, 1.0),  // nonneg: 0.5 − x1 ≥ 0
                Triplet::new(2, 0, -1.0), // soc s1 = x0
                Triplet::new(3, 1, -1.0), // soc s2 = x1
            ],
            h: vec![0.5, 1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        assert_agrees(
            &prob,
            &[ConeSpec::Nonneg(1), ConeSpec::SecondOrder(3)],
            1e-5,
        );
    }

    /// Equality-constrained QP with a closed-form optimum:
    /// min ½‖x‖² − pᵀx s.t. 1ᵀx = 1  →  x = p + (1 − Σp)/n.
    #[test]
    fn qp_equality_closed_form() {
        let p = [0.2_f64, 0.5, 0.1];
        let n = 3;
        let prob = QpProblem {
            n,
            p_lower: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 2, 1.0),
            ],
            c: vec![-p[0], -p[1], -p[2]],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            b: vec![1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[], 1e-6);
        let shift = (1.0 - p.iter().sum::<f64>()) / n as f64;
        for i in 0..n {
            assert!((sol.x[i] - (p[i] + shift)).abs() < 1e-6, "x {:?}", sol.x);
        }
    }

    /// Inequality QP with a known optimum:
    /// min ‖x‖² − 3x0 − 4x1 s.t. x0+x1 ≤ 1, x ≥ 0  →  x = (0.25, 0.75).
    #[test]
    fn qp_inequality_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(2, 1, -1.0),
            ],
            h: vec![1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::Nonneg(3)], 1e-6);
        assert!((sol.x[0] - 0.25).abs() < 1e-5 && (sol.x[1] - 0.75).abs() < 1e-5);
        assert!((sol.obj + 3.125).abs() < 1e-5, "obj {}", sol.obj);
    }

    /// Quadratic objective *and* a second-order cone together (P in the
    /// (x,x) block, SOC scaling in the (z,z) block):
    /// min ‖x‖² − 3x0 − 4x1 s.t. ‖x‖ ≤ 1  (slack (1, x0, x1) ∈ SOC).
    #[test]
    fn qp_with_soc_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(1, 0, -1.0), Triplet::new(2, 1, -1.0)],
            h: vec![1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::SecondOrder(3)], 1e-5);
        // Constraint active: the optimum lies on the unit ball.
        assert!(
            (sol.x[0].hypot(sol.x[1]) - 1.0).abs() < 1e-5,
            "x {:?}",
            sol.x
        );
    }

    /// Primal-infeasible LP: x ≥ 2 and x ≤ 1.
    #[test]
    fn detects_primal_infeasible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 0, 1.0)],
            h: vec![-2.0, 1.0], // −x ≤ −2 (x≥2) ; x ≤ 1
            lb: vec![],
            ub: vec![],
        };
        let cone = CompositeCone::from_specs(&[ConeSpec::Nonneg(2)]);
        let sol = solve_conic_hsde(&prob, &cone, &opts(), backend, None);
        assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    }

    /// The `use_hsde` flag routes a bound-constrained QP through the
    /// embedding via the *public* entry point (exercising bound expansion
    /// into cone rows and the z_lb/z_ub split on the way back). The result
    /// must match the default driver.
    #[test]
    fn flag_routes_through_public_entry_with_bounds() {
        // min ‖x‖² − 3x0 − 4x1 s.t. x0+x1 ≤ 1, 0 ≤ x ≤ 1.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        };
        let direct = solve_qp_ipm(&prob, &opts(), backend);
        let hsde_opts = QpOptions {
            use_hsde: true,
            ..opts()
        };
        let hsde = solve_qp_ipm(&prob, &hsde_opts, backend);
        assert_eq!(direct.status, QpStatus::Optimal);
        assert_eq!(hsde.status, QpStatus::Optimal);
        for i in 0..2 {
            assert!(
                (direct.x[i] - hsde.x[i]).abs() < 1e-5,
                "x[{i}] direct {} vs hsde {}",
                direct.x[i],
                hsde.x[i]
            );
            // Bound multipliers must survive the round-trip split.
            assert!((direct.z_lb[i] - hsde.z_lb[i]).abs() < 1e-5);
            assert!((direct.z_ub[i] - hsde.z_ub[i]).abs() < 1e-5);
        }
        assert!((direct.x[0] - 0.25).abs() < 1e-5 && (direct.x[1] - 0.75).abs() < 1e-5);
    }

    /// Dual-infeasible / unbounded LP: min −x s.t. x ≥ 0 (no upper bound).
    #[test]
    fn detects_dual_infeasible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0],
            lb: vec![],
            ub: vec![],
        };
        let cone = CompositeCone::from_specs(&[ConeSpec::Nonneg(1)]);
        let sol = solve_conic_hsde(&prob, &cone, &opts(), backend, None);
        assert_eq!(sol.status, QpStatus::DualInfeasible);
    }

    /// SDP `max λ s.t. M − λI ⪰ 0` ⇒ `λ = λ_min(M)`. Diagonal `M = diag(2,5)`
    /// (λ_min = 2): the PSD slack `s = svec(M − λI)` exercises the dense
    /// `(z,z)` block on a diagonal matrix. Solved through the public conic
    /// entry `solve_socp_ipm` with a `Psd(2)` cone.
    #[test]
    fn psd_min_eigenvalue_diagonal() {
        // x = (λ); minimize −λ. G·x places λ on the diagonal svec entries
        // (positions 0 and 2 for a 2×2), h = svec(M), s = svec(M − λI) ⪰ 0.
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(2, 0, 1.0)],
            h: vec![2.0, 0.0, 5.0], // svec(diag(2,5))
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(2)], &opts(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!((sol.x[0] - 2.0).abs() < 1e-5, "λ = {}", sol.x[0]);
        assert!((sol.obj + 2.0).abs() < 1e-5, "obj = {}", sol.obj);
    }

    /// Same SDP with a **non-diagonal** `M = [[2,1],[1,2]]` (λ_min = 1), so
    /// the PSD slack has a nonzero off-diagonal — exercising the off-diagonal
    /// entries of the dense `W ⊗ₛ W` scaling block.
    #[test]
    fn psd_min_eigenvalue_offdiagonal() {
        let r2 = std::f64::consts::SQRT_2;
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(2, 0, 1.0)],
            h: vec![2.0, r2, 2.0], // svec([[2,1],[1,2]])
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(2)], &opts(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "λ = {}", sol.x[0]);
        assert!((sol.obj + 1.0).abs() < 1e-5, "obj = {}", sol.obj);
    }

    /// A block-diagonal PSD cone (4×4 = two 2×2 blocks, no cross coupling)
    /// decomposes into two `Psd(2)` cones, dropping the structurally-zero
    /// cross rows. svec(4×4) indices: diag at k∈{0,4,7,9}; the within-block
    /// off-diagonals (1,0)=k1 and (3,2)=k8 are present; the cross entries
    /// k∈{2,3,5,6} are absent.
    #[test]
    fn psd_decompose_splits_block_diagonal() {
        use crate::ipm::decompose_psd;
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            lb: vec![],
            ub: vec![],
        };
        let (_p2, cones2, row_map) = decompose_psd(&prob, &[ConeSpec::Psd(4)]);
        assert_eq!(cones2, vec![ConeSpec::Psd(2), ConeSpec::Psd(2)]);
        assert_eq!(row_map, vec![0, 1, 4, 7, 8, 9]); // cross rows 2,3,5,6 dropped
    }

    /// A genuinely coupled PSD cone (a cross entry present) stays one block.
    #[test]
    fn psd_decompose_keeps_coupled() {
        use crate::ipm::decompose_psd;
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            // k=2 is the cross entry (2,0); making it present couples the blocks.
            h: vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            g: vec![],
            lb: vec![],
            ub: vec![],
        };
        let (_p2, cones2, _) = decompose_psd(&prob, &[ConeSpec::Psd(4)]);
        assert_eq!(cones2, vec![ConeSpec::Psd(4)]);
    }

    /// End-to-end: a block-diagonal SDP declared as a single `Psd(4)` cone
    /// solves correctly through the auto-decomposition. `max λ s.t. M−λI⪰0`
    /// with `M = blkdiag([[2,1],[1,2]], [[4,1],[1,4]])` has
    /// `λ_min(M) = min(1, 3) = 1`. The decomposed cross rows get dual 0.
    #[test]
    fn psd_block_diagonal_solves_end_to_end() {
        let r2 = std::f64::consts::SQRT_2;
        // G column = svec(I₄): diagonal entries k ∈ {0,4,7,9}.
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(4, 0, 1.0),
                Triplet::new(7, 0, 1.0),
                Triplet::new(9, 0, 1.0),
            ],
            // svec(M): (0,0)=2,(1,0)=√2,(1,1)=2 | (2,2)=4,(3,2)=√2,(3,3)=4.
            h: vec![2.0, r2, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, r2, 4.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(4)], &opts(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "λ = {}", sol.x[0]);
        assert!((sol.obj + 1.0).abs() < 1e-5, "obj = {}", sol.obj);
        // z is returned in the original 10-row layout (dropped rows = 0).
        assert_eq!(sol.z.len(), 10);
        for &k in &[2usize, 3, 5, 6] {
            assert_eq!(sol.z[k], 0.0, "dropped cross row {k} should have dual 0");
        }
    }

    /// Connected **sparse** PSD cone: chordal range-space decomposition.
    /// `max λ s.t. M − λI ⪰ 0` with tridiagonal `M` (path 0–1–2, so the
    /// (2,0) entry is structurally zero). The pattern is chordal with
    /// overlapping cliques {0,1},{1,2}, so `solve_socp_ipm` rewrites it via
    /// clique blocks + consistency equalities. The optimum (`λ = λ_min(M)`)
    /// and objective must match a direct **dense** `Psd(3)` solve (the primal
    /// is unique; the PSD dual is not, so only x/obj are compared).
    #[test]
    fn psd_chordal_matches_dense_on_path_sdp() {
        let r2 = std::f64::consts::SQRT_2;
        // svec(M), M tridiagonal diag 2, off 0.5: (2,0)=k2 is structurally 0.
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(5, 0, 1.0),
            ],
            h: vec![2.0, 0.5 * r2, 0.0, 2.0, 0.5 * r2, 2.0],
            lb: vec![],
            ub: vec![],
        };
        // Dense reference: the HSDE driver on a single Psd(3) (no decomposition).
        let dense = solve_conic_hsde(
            &prob,
            &CompositeCone::from_specs(&[ConeSpec::Psd(3)]),
            &opts(),
            backend,
            None,
        );
        // solve_socp_ipm auto-applies the chordal decomposition.
        let decomp = solve_socp_ipm(&prob, &[ConeSpec::Psd(3)], &opts(), backend);
        assert_eq!(dense.status, QpStatus::Optimal, "dense {:?}", dense.status);
        assert_eq!(
            decomp.status,
            QpStatus::Optimal,
            "decomp {:?}",
            decomp.status
        );
        assert!(
            (dense.x[0] - decomp.x[0]).abs() < 1e-5,
            "λ: dense {} vs decomp {}",
            dense.x[0],
            decomp.x[0]
        );
        assert!(
            (dense.obj - decomp.obj).abs() < 1e-5,
            "obj: dense {} vs decomp {}",
            dense.obj,
            decomp.obj
        );
        assert_eq!(decomp.z.len(), 6, "dual returned in original svec layout");
    }
}
