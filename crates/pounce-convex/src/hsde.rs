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
use crate::debug::{ConvexDebugState, fire};
use crate::ipm::{
    QpOptions, build_factorization, build_rhs, detect_infeasibility_cone, dot, inf_norm, split_step,
};
use crate::qp::{QpIterate, QpProblem, QpSolution, QpStatus};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
use pounce_common::types::Index;
use pounce_linsol::{Factorization, FactorizationError, SparseSymLinearSolverInterface};

/// Strictly-positive floor for the adaptive static regularization δ (see the
/// `update_blocks` call in [`solve_conic_hsde`]). δ must stay above this so the
/// reduced KKT remains quasi-definite for a stable LDLᵀ inertia; the adaptive
/// rule only ever shrinks δ *toward* this floor on large-dual problems, never
/// below it. Sits ~4 orders under the configured default (`QpOptions::reg`,
/// 1e-10) — enough room for the LISWET cluster (‖ŷ‖ ~ 1e4 ⇒ δ ~ 1e-12) without
/// reaching machine-noise territory.
const REG_MIN: f64 = 1e-14;

/// Maximum iterative-refinement passes per KKT back-solve, and the relative
/// residual at which refinement stops. Each HSDE step solves the same factored
/// KKT three times (constant direction, predictor, corrector); near a
/// degenerate optimum that KKT is ill-conditioned and a single LDLᵀ back-solve
/// loses several digits, biasing the Newton direction and collapsing the
/// fraction-to-boundary step (the NETLIB GEN stall). Refining each solve
/// against the assembled matrix recovers those digits — Clarabel, QDLDL, and
/// SCS all refine every solve by default. Passes are few (refinement converges
/// in 1–2 steps when it helps) and a well-conditioned solve exits after a
/// single residual check, so the overhead on easy problems is one matvec.
const IR_MAX_PASSES: usize = 5;
const IR_RELTOL: f64 = 1e-12;

/// Dynamic-regularization schedule (Ipopt-style inertia/regularization
/// correction; Clarabel's dynamic KKT regularization). When the factorization
/// is singular, or the constant-direction solve cannot be refined below
/// [`DYN_REG_RES_TOL`] (the KKT is numerically singular on the degenerate
/// face), raise the `(z,z)` regularization δ by [`DYN_REG_FACTOR`] — up to
/// [`DYN_REG_MAX`] — and refactor. δ is reset to its small per-iteration base
/// every iteration, so a single hard iterate never inflates δ for the rest of
/// the solve; well-conditioned iterations keep the tight static δ and never
/// enter the bump loop. The bump count is shared with the δ_c inertia
/// escalation below and bounded by [`INERTIA_MAX_TRIES`].
const DYN_REG_FACTOR: f64 = 10.0;
const DYN_REG_MAX: f64 = 1e-7;
const DYN_REG_RES_TOL: f64 = 1e-8;

/// Inertia-based perturbation escalation (Ipopt's `perturb_for_singular` /
/// `perturb_for_wrong_inertia`, adapted to the HSDE KKT). A rank-deficient
/// equality Jacobian — gen/gen1's redundant rows, non-unique duals — factors
/// "successfully" but with **too few negative eigenvalues**: an indefinite
/// (saddle) direction whose step never reduces the primal residual, so the
/// solve plateaus at `δ·‖ŷ‖ ~ 9e-5` and runs to `max_iter`. The `(z,z)`
/// dynamic bump above keys only on a *singular* factor or an un-refinable
/// constant-direction solve; it misses wrong inertia because nothing checks
/// it. Here we read the factor's negative-eigenvalue count and, when it falls
/// short of the `m_eq + m_ineq` a correct KKT requires, escalate the
/// equality-block regularization `δ_c` — from [`DELTA_C_INIT`], by
/// [`DELTA_C_FACTOR`], up to [`DELTA_C_MAX`] — and refactor until the inertia
/// is right, so the Newton step is a genuine descent direction. `δ_c` resets
/// to its small μ-scaled base every iteration; the regularized step leaves the
/// new primal residual at `δ_c·‖dy‖`, and as `μ → 0` both `δ_c` and `dy`
/// shrink, so that residual drives to zero (Ipopt's convergence argument on
/// degenerate equality systems).
const DELTA_C_INIT: f64 = 1e-8;
const DELTA_C_FACTOR: f64 = 10.0;
const DELTA_C_MAX: f64 = 1e-1;
const INERTIA_MAX_TRIES: usize = 20;

/// Gondzio multiple centrality correctors (Gondzio 1996, "Multiple centrality
/// corrections in a primal–dual method for linear programming"). After the
/// Mehrotra corrector, up to [`GONDZIO_MAX_CORR`] additional corrections are
/// computed through the *same* factorization (each is one extra back-solve):
/// a trial step enlarged by [`GONDZIO_DELTA`] is formed, and any
/// complementarity product it would push outside the centered box
/// `[β_lo·μ, β_hi·μ]` is corrected back toward the box. Each corrector is
/// accepted only if it lengthens the fraction-to-boundary step by at least
/// `GONDZIO_GAMMA·GONDZIO_DELTA`; otherwise correction stops. Lengthening the
/// step is exactly the documented purpose of the scheme — and it is what
/// breaks the degenerate-face step collapse (σ→1 centering freezing μ) on the
/// NETLIB GEN family, where the Mehrotra corrector alone stalls. β_lo = 0.1,
/// β_hi = 10 is Gondzio's recommended symmetric box.
const GONDZIO_MAX_CORR: usize = 3;
const GONDZIO_DELTA: f64 = 0.1;
const GONDZIO_GAMMA: f64 = 0.1;
const GONDZIO_BETA_LO: f64 = 0.1;
const GONDZIO_BETA_HI: f64 = 10.0;

/// Centering fallback for a collapsing step (gh #218).
///
/// Mehrotra's centering parameter `σ = (μ_aff/μ)³` is inferred from how far the
/// *affine* (σ = 0) direction could travel. On a degenerate face that inference
/// inverts: the affine direction looks excellent while actually pointing almost
/// straight out of the cone, so σ comes back near zero — almost no centering —
/// exactly when centering is the only thing that helps. The step length then
/// collapses geometrically and the solve freezes with `μ` still far from zero.
///
/// Observed on gh #218's order-4 moment SDP: `σ` pinned at 0.0218 while the
/// step fell 4.0e-1 → 2.1e-2 → 9.7e-4 → 4.8e-5 → … → 1e-281, throttled by the
/// PSD slack block alone (`z`, `τ`, `κ` all stayed healthy), with `μ` stuck at
/// 2.6e-3 and the residuals frozen at 4.2e-4. Nothing was near convergence; the
/// iterate had simply run into the boundary and could not get off it.
///
/// So when the corrector's step comes back below `CENTERING_MIN_STEP`, redo it
/// with a larger σ — a more centered target pulls the direction back inside the
/// cone, where a usable step exists. Each retry costs one back-solve through
/// the factorization already computed. The ladder is bounded by
/// `CENTERING_MAX_TRIES`, and the last (most centered) attempt is kept if none
/// clears the bar, since a near-pure centering direction is the one most likely
/// to admit a step.
///
/// This is the PSD-cone counterpart of what the Gondzio correctors below do on
/// the orthant — they lengthen the step by re-centering — and it is deliberately
/// step-length-triggered so that a healthy iteration never pays for it.
const CENTERING_MIN_STEP: f64 = 1e-2;
const CENTERING_MAX_TRIES: usize = 3;
const CENTERING_FACTOR: f64 = 10.0;
const CENTERING_SIGMA_FLOOR: f64 = 0.1;
const CENTERING_SIGMA_MAX: f64 = 0.9;

/// HSDE infeasibility-ray discriminant: is the homogenizing pair `(τ, κ)` on
/// the Farkas/recession ray (`κ ≫ τ`), as opposed to converging to a solution
/// (`τ → τ* > 0`) or merely degenerating on a feasible-but-ill-scaled problem
/// (`τ, κ → 0` together)?
///
/// The defining signature of infeasibility is the *ratio* `κ/τ → ∞`: at a
/// genuine certificate `τ → 0` while `κ → κ* > 0`. This must NOT be a bare
/// `τ < ε` floor — a feasible large-‖x*‖ QP (e.g. POWELL20) can drive `τ → 0`
/// too, but there `κ → 0` *alongside* `τ` (no ray supports `κ > 0`), so `κ/τ`
/// stays bounded and the gate stays shut. The threshold here is `κ/τ > 100`.
///
/// A prior `κ.max(1.0)` floor degraded this to `τ < 1e-2`, firing on the
/// collapsed-`τ` iterate of a feasible problem and declaring it infeasible.
fn on_infeasibility_ray(tau: f64, kappa: f64) -> bool {
    tau < 1e-2 * kappa
}

/// May the scale-relative stopping test *relax* the absolute one for a problem
/// of natural scale `max_scale` (the largest of the dual/primal/gap term norms)
/// at tolerance `tol`?
///
/// Only once `tol`-level *absolute* KKT accuracy is genuinely unreachable: the
/// finite-precision floor on an absolute residual is `≈ max_scale·ε`, so when
/// `max_scale·ε > tol` no iterate can drive the absolute residual under `tol`
/// and the scale-relative residual is the only valid optimality certificate.
/// Below that crossover (`max_scale ≲ tol/ε ≈ 4.5e7` at `tol = 1e-8`) the tight
/// absolute test must govern — relaxing it there lets a moderately-scaled
/// reduced problem stop a step early, and the bound-tightening presolve's dual
/// re-attribution then amplifies that slack into a large stationarity violation
/// in the original problem. Tying the crossover to `tol` keeps it self-consistent
/// under a caller-tightened/loosened tolerance.
fn relative_stop_permitted(max_scale: f64, tol: f64) -> bool {
    max_scale * f64::EPSILON > tol
}

/// The KKT error of the **un-homogenized** iterate `(x, y, z)/τ`, measured
/// directly against the original problem and this solve's own cones.
///
/// This is the definition of optimality applied to the point a caller would
/// actually receive: cone feasibility of `h − Gx̂`, stationarity
/// `‖Px̂ + c + Aᵀŷ + Gᵀẑ‖∞`, and per-cone-block complementarity. It is a
/// *stronger* certificate than the driver's homogeneous residual, which also
/// carries the consistency of the internal slack `s` (`Gx + s − hτ`) — a
/// quantity that is pure bookkeeping (`s` is never returned) and that goes
/// numerically noisy once `μ` reaches ~1e-16 and the NT scaling's condition
/// number blows up. On a feasible convex QCQP the surrogate stalled at ~1e-7
/// and drifted as high as 1e-4 while this error sat at 1e-14 (pounce#209).
///
/// Returns `None` when `τ ≤ 0` (an infeasibility ray, where un-homogenizing is
/// meaningless) or when `ẑ` has left the dual cone — without `ẑ ∈ K*` the
/// residuals below are not a KKT certificate, however small they are.
fn true_kkt_error(
    prob: &QpProblem,
    cone: &CompositeCone,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    tau: f64,
) -> Option<f64> {
    if !(tau > 0.0) {
        return None;
    }
    let inv = 1.0 / tau;
    let z_hat: Vec<f64> = z.iter().map(|v| v * inv).collect();
    // `in_dual_cone` is scale-free in the sense that matters here: the cones
    // are self-dual, so this asks whether `ẑ` is (nearly) in `K` itself.
    if !cone.in_dual_cone(&z_hat, 1e-9) {
        return None;
    }
    let candidate = QpSolution {
        status: QpStatus::Optimal,
        x: x.iter().map(|v| v * inv).collect(),
        y: y.iter().map(|v| v * inv).collect(),
        z: z_hat,
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    Some(
        candidate
            .kkt_residuals_conic(prob, &cone.specs())
            .kkt_error(),
    )
}

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

/// Symmetric matvec `y ← M x` for the lower-triangle KKT triplets
/// (`airn`/`ajcn` are 1-based with `row ≥ col`). Each strictly-lower entry
/// contributes to both `y[i]` and `y[j]`; the diagonal once. Used to form the
/// residual `rhs − M u` that drives iterative refinement.
fn kkt_matvec(airn: &[Index], ajcn: &[Index], vals: &[f64], x: &[f64], y: &mut [f64]) {
    for v in y.iter_mut() {
        *v = 0.0;
    }
    for k in 0..vals.len() {
        let i = (airn[k] - 1) as usize;
        let j = (ajcn[k] - 1) as usize;
        let v = vals[k];
        y[i] += v * x[j];
        if i != j {
            y[j] += v * x[i];
        }
    }
}

/// Solve `M u = rhs` against the cached factor (overwriting `rhs` with `u`),
/// applying iterative refinement against the assembled KKT triplets
/// `(airn, ajcn, vals)`. Recovers digits lost to factorization round-off on
/// the ill-conditioned KKT near a degenerate optimum (see [`IR_MAX_PASSES`]).
///
/// Refinement runs against the *factored* (regularized) matrix `vals`: the
/// static δ is below `tol` at the default, so the regularized and true systems
/// coincide to working precision there; when dynamic regularization raises δ
/// the resulting bias is accepted in exchange for a usable (stable) step.
/// `b`/`r`/`d` are caller-owned scratch of length `dim`. Returns the final
/// relative residual `‖rhs − M u‖∞ / (1 + ‖rhs‖∞)`, which the caller uses as
/// the dynamic-regularization trigger.
#[allow(clippy::too_many_arguments)]
fn solve_refined(
    fact: &mut Factorization,
    airn: &[Index],
    ajcn: &[Index],
    vals: &[f64],
    rhs: &mut [f64],
    b: &mut [f64],
    r: &mut [f64],
    d: &mut [f64],
) -> Result<f64, FactorizationError> {
    b.copy_from_slice(rhs);
    fact.solve_one(rhs)?;
    let bnorm = 1.0 + inf_norm(b);
    let mut res = f64::INFINITY;
    for _ in 0..IR_MAX_PASSES {
        kkt_matvec(airn, ajcn, vals, rhs, r);
        for k in 0..r.len() {
            r[k] = b[k] - r[k];
        }
        let new_res = inf_norm(r) / bnorm;
        // Stop once converged, or once refinement stops making progress — the
        // latter means the factor can no longer recover accuracy (a
        // near-singular KKT on the degenerate face), which is the signal the
        // dynamic-regularization loop keys on via the returned residual.
        if new_res <= IR_RELTOL || new_res >= res {
            res = new_res;
            break;
        }
        res = new_res;
        d.copy_from_slice(r);
        fact.solve_one(d)?;
        for k in 0..rhs.len() {
            rhs[k] += d[k];
        }
    }
    Ok(res)
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
    // Zero linear-residual blocks for the Gondzio corrector solves, whose only
    // non-zero right-hand side is the re-centered complementarity term.
    let zeros_n = vec![0.0; n];
    let zeros_meq = vec![0.0; m_eq];

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
    // Iterative-refinement scratch (residual b, residual-of-residual r, and
    // the correction d), each one full KKT right-hand side.
    let mut ir_b = vec![0.0; kkt.dim];
    let mut ir_r = vec![0.0; kkt.dim];
    let mut ir_d = vec![0.0; kkt.dim];

    // Scratch + constants for the scale-relative convergence normalizers
    // (see the stopping test below). Dual side: ‖Aᵀŷ‖, ‖Gᵀẑ‖; primal side:
    // ‖Ax̂‖, ‖Gx̂‖. The RHS/cost data norms are constant and double as the
    // `1+` absolute floors that preserve the old behavior on well-scaled data.
    let mut nrm_aty = vec![0.0; n];
    let mut nrm_gtz = vec![0.0; n];
    let mut nrm_ax = vec![0.0; m_eq];
    let mut nrm_gx = vec![0.0; m_ineq];
    let norm_b = inf_norm(&prob.b);
    let norm_h = inf_norm(&prob.h);
    let norm_c = inf_norm(&prob.c);

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
    // Gondzio centrality-corrector buffers: one extra direction (cdx, cdy,
    // cdz, cds) plus two combined-step scratch vectors. Allocated once.
    let mut cdx = vec![0.0; n];
    let mut cdy = vec![0.0; m_eq];
    let mut cdz = vec![0.0; m_ineq];
    let mut cds = vec![0.0; m_ineq];
    let mut step_s = vec![0.0; m_ineq];
    let mut step_z = vec![0.0; m_ineq];

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
        // These are the *absolute* residuals (what the trace/debugger report).
        let pres = inf_norm(&rho_y).max(inf_norm(&rho_z)) / tau;
        let dres = inf_norm(&rho_x) / tau;
        let gap = (xpx / tau + ctx + bty + htz).abs() / tau;
        // Un-homogenized objective `½x̂ᵀPx̂ + cᵀx̂` (x̂ = x/τ).
        let obj_hat = 0.5 * xpx / (tau * tau) + ctx / tau;

        // --- scale-relative residuals (Clarabel-style) ---
        // A pure absolute test `res < tol` is unreachable for large-data
        // problems: a feasible QP whose data norms are large (POWELL20:
        // ‖Px̂‖ ~ 7.5e3, objective ~5e10) floors its absolute KKT residuals far
        // above `tol` (primal ~3e-5, gap ~4e2) while the *relative* residuals
        // are ~1e-9 — genuinely optimal. Normalize each residual by the natural
        // scale of its own terms. Data-only (`‖c‖`/`‖b‖`/`‖h‖`) normalizers are
        // *insufficient* for a QP — the dual residual's dominant scale is the
        // Hessian-gradient term ‖Px̂‖, which only the matvec norms capture.
        // `px_vec` already holds `Px` (computed above) ⇒ ‖Px̂‖ = ‖px_vec‖/τ.
        //
        // These relative residuals do NOT simply replace the absolute test:
        // the relative test is *gated* on the problem scale below (it only
        // relaxes the absolute test once `tol`-level absolute accuracy is
        // unreachable). The `1.0 +` floor alone is not enough — a moderately
        // scaled problem (scale ~30) has a relative residual ~30× looser than
        // its absolute one, and the bound-tightening presolve's dual
        // re-attribution amplifies that slack into a large stationarity
        // violation in the *original* problem (the reduced solve stops a step
        // early, before the final quadratic plunge to ~machine-ε accuracy).
        for v in nrm_aty.iter_mut() {
            *v = 0.0;
        }
        prob.at_mul(&y, &mut nrm_aty);
        for v in nrm_gtz.iter_mut() {
            *v = 0.0;
        }
        prob.gt_mul(&z, &mut nrm_gtz);
        for v in nrm_ax.iter_mut() {
            *v = 0.0;
        }
        prob.a_mul(&x, &mut nrm_ax);
        for v in nrm_gx.iter_mut() {
            *v = 0.0;
        }
        prob.g_mul(&x, &mut nrm_gx);
        let scale_d = (inf_norm(&px_vec)
            .max(inf_norm(&nrm_aty))
            .max(inf_norm(&nrm_gtz))
            / tau)
            .max(norm_c);
        let scale_p = (inf_norm(&nrm_ax).max(inf_norm(&nrm_gx)).max(inf_norm(&s)) / tau)
            .max(norm_b)
            .max(norm_h);
        let d_obj = -0.5 * xpx / (tau * tau) - bty / tau - htz / tau;
        let scale_g = obj_hat.abs().max(d_obj.abs());
        let pres_rel = pres / (1.0 + scale_p);
        let dres_rel = dres / (1.0 + scale_d);
        let gap_rel = gap / (1.0 + scale_g);
        // Gate (see [`relative_stop_permitted`]): the scale-relative test only
        // *relaxes* the absolute one once the problem's natural scale is large
        // enough that `tol`-level absolute accuracy is below the
        // finite-precision floor. Empirically this cleanly separates the two
        // regimes seen across the QP set — well-/moderately-scaled problems
        // (scale ≲ 1e3, incl. every bound-tightening presolve instance at
        // scale ≲ 30) reach the tight absolute test, while the large-data
        // cluster (POWELL20/BOYD/QFORPLAN/QSHELL at scale 7e9–4e12) can only be
        // certified relatively.
        let large_scale = relative_stop_permitted(scale_d.max(scale_p).max(scale_g), opts.tol);
        // `res` (used only for the `near_opt` salvage check below) tracks
        // whichever test governs this iterate.
        let res = if large_scale {
            pres_rel.max(dres_rel).max(gap_rel)
        } else {
            pres.max(dres).max(gap)
        };
        // (`res` also feeds the debugger checkpoints below. The
        // reduced-accuracy salvage that used to read it here now runs *after*
        // the loop, against the true KKT residual — see `SALVAGE`.)

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

        // Breakdown: a non-finite iterate carries no information, and every
        // test below is a comparison against it. Stop and say so (gh #222).
        if !crate::ipm::all_finite(&[&x, &s, &y, &z]) || !tau.is_finite() || !kappa.is_finite() {
            status = QpStatus::NumericalFailure;
            break;
        }

        // Absolute test always governs; the scale-relative test only relaxes
        // it for genuinely large-data problems (`large_scale`, gated above).
        let converged = (pres < opts.tol && dres < opts.tol && gap < opts.tol)
            || (large_scale && pres_rel < opts.tol && dres_rel < opts.tol && gap_rel < opts.tol);
        if converged {
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
        //
        // The trigger is the ratio discriminant κ ≫ τ — see
        // [`on_infeasibility_ray`]. Concretely POWELL20 (‖x*‖ ~ 1e7): once τ
        // collapses far enough that the homogeneous Farkas residual
        // ‖Aᵀy+Gᵀz‖ = τ·‖Px̂+c‖ slips under `FARKAS_RESID_TOL` (~iter 33), κ has
        // collapsed with it (κ/τ ~ 1e-9), so the gate stays shut.
        if on_infeasibility_ray(tau, kappa) {
            if let Some(st) = detect_infeasibility_cone(prob, &x, &y, &z, opts, cone) {
                status = st;
                break;
            }
        }

        // --- adaptive static regularization δ ---
        // The static δ added to the KKT diagonal biases the *step*, flooring
        // the un-homogenized residual at ~δ·‖(x̂,ŷ,ẑ)‖ (x̂ = x/τ, …) — see the
        // `δ·‖dy‖` floor noted on `QpOptions::reg`. On large-dual problems
        // (the LISWET cluster: ‖ŷ‖ ~ 1e4) a fixed δ = 1e-10 floors that
        // residual at ~1e-6 ≫ `tol`, so the solve plateaus and runs to
        // `max_iter`. Scale δ down by the current iterate norm so the floor
        // tracks ~`tol` regardless of dual scale, but never below `REG_MIN`
        // (keeps the reduced KKT quasi-definite). Well-scaled problems
        // (‖iterate‖ ~ 1) are untouched: `tol/scale ≥ opts.reg`, so the `min`
        // keeps the configured δ.
        let iterate_scale = (inf_norm(&x).max(inf_norm(&y)).max(inf_norm(&z)) / tau).max(1.0);
        let base_reg = (opts.tol / iterate_scale).min(opts.reg).max(REG_MIN);

        // --- refactor M with the current cone scaling + dynamic regularization ---
        // Factor at the small static δ; if the factorization is singular, or
        // the constant-direction solve cannot be refined below DYN_REG_RES_TOL
        // (the KKT is numerically singular on the degenerate face), raise δ on
        // the (z,z) block and refactor — bounded, and reset to `base_reg` next
        // iteration. The constant direction `p: M p = (−c, b, h)` doubles as
        // the conditioning probe: it is solved here inside the loop so its
        // refinement residual gates the bump.
        let mut reg_eff = base_reg;
        // δ_c on the (y,y) equality-multiplier block. `build` freezes (y,y) at
        // the static `opts.reg`; we override it each iteration, starting from a
        // moderate μ-scaled base (Ipopt's `1e-8·μ^0.25`) and escalating on
        // wrong inertia / singularity (see `DELTA_C_INIT` & co.). The (z,z)
        // slack block keeps `reg_eff`, whose iterate-norm scaling is correct
        // for full-rank large-dual problems (LISWET).
        let mut delta_c = crate::ipm::adaptive_eq_reg(mu, opts.reg);
        // Correct KKT inertia has one negative eigenvalue per equality and per
        // inequality row; the SOC auxiliary variables contribute positives
        // only. Too few negatives ⇒ the factor is an indefinite saddle.
        let expected_neg = (m_eq + m_ineq) as Index;
        let mut tries = 0usize;
        let mut factored = false;
        loop {
            kkt.update_blocks(cone, &s, &z, reg_eff, &mut kkt_vals);
            kkt.update_eq_reg(delta_c, &mut kkt_vals);

            // Escalation budget (tries + per-block ceilings) and the bump that
            // raises δ_c (equality/Jacobian reg) and the (z,z) reg together.
            // `macro` would be cleaner, but inlined to keep the borrow trivial.
            let budget =
                tries < INERTIA_MAX_TRIES && (delta_c < DELTA_C_MAX || reg_eff < DYN_REG_MAX);

            match fact.refactor(&kkt_vals) {
                Err(_) => {
                    // Singular factorization: rank-deficient KKT. Escalate.
                    if budget {
                        delta_c = (delta_c.max(DELTA_C_INIT) * DELTA_C_FACTOR).min(DELTA_C_MAX);
                        reg_eff = (reg_eff * DYN_REG_FACTOR).min(DYN_REG_MAX);
                        tries += 1;
                        continue;
                    }
                    break; // factored stays false → breakdown below
                }
                Ok(()) => {
                    // Inertia check: a rank-deficient equality Jacobian factors
                    // without error but with too few negative eigenvalues — a
                    // saddle direction. Escalate δ_c until the inertia is right.
                    let wrong_inertia =
                        matches!(fact.number_of_neg_evals(), Some(n) if n < expected_neg);
                    if wrong_inertia && budget {
                        delta_c = (delta_c.max(DELTA_C_INIT) * DELTA_C_FACTOR).min(DELTA_C_MAX);
                        reg_eff = (reg_eff * DYN_REG_FACTOR).min(DYN_REG_MAX);
                        tries += 1;
                        continue;
                    }
                }
            }

            build_rhs(&prob.c, &neg_b, &neg_h, &zeros_m, n, m_eq, m_ineq, &mut rhs);
            let res_p = match solve_refined(
                &mut fact, &kkt.airn, &kkt.ajcn, &kkt_vals, &mut rhs, &mut ir_b, &mut ir_r,
                &mut ir_d,
            ) {
                Ok(res) => res,
                Err(_) => break, // factored stays false → breakdown below
            };
            // An un-refinable solve means the *scaling* has gone ill-conditioned,
            // so bump the `(z,z)` dynamic regularization — and deliberately NOT
            // `δ_c`. `δ_c` regularizes the equality block; escalating it here
            // treats a cone-conditioning symptom as an equality-Jacobian rank
            // defect, and it is actively harmful. The KKT is inherently
            // ill-conditioned in the μ→0 endgame (the NT scaling's condition
            // number blows up by design), so this branch fires there on healthy
            // solves and used to ratchet `δ_c` all the way to `DELTA_C_MAX`
            // = 1e-1. That biases the equality residual by ~`δ_c·‖dy‖`, which
            // floors `pres` permanently.
            //
            // gh #218 order 3: at the iteration before the escalation the solve
            // stood at `pres` 8.6e-9 ✓, `dres` 1.2e-10 ✓, `gap` 1.7e-8 — one
            // step from converging. The escalation pushed `pres` to 2.7e-8 and
            // it never recovered, ending `OptimalInaccurate` instead of
            // `Optimal`. Inertia was correct at every single try throughout, so
            // nothing here was a rank defect.
            if res_p > DYN_REG_RES_TOL && tries < INERTIA_MAX_TRIES && reg_eff < DYN_REG_MAX {
                reg_eff = (reg_eff * DYN_REG_FACTOR).min(DYN_REG_MAX);
                tries += 1;
                continue;
            }
            factored = true;
            break;
        }
        if !factored {
            status = QpStatus::NumericalFailure;
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
        if solve_refined(
            &mut fact, &kkt.airn, &kkt.ajcn, &kkt_vals, &mut rhs, &mut ir_b, &mut ir_r, &mut ir_d,
        )
        .is_err()
        {
            status = QpStatus::NumericalFailure;
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
        let mut sigma = if mu > 0.0 { (mu_aff / mu).powi(3) } else { 0.0 };

        // === Corrector (centered target + second-order term) ===
        // Recomputed under an escalating σ when the resulting step collapses —
        // see `CENTERING_MIN_STEP`. Each retry is one extra back-solve through
        // the factorization already in hand, and a healthy iteration (the
        // overwhelming majority) clears the threshold on the first pass and
        // never enters the ladder.
        let mut dtau = 0.0;
        let mut dkappa = 0.0;
        let mut alpha = 0.0;
        let mut centering_tries = 0usize;
        let mut solve_failed = false;
        loop {
            let sigma_mu = sigma * mu;
            cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
            cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
            build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
            if solve_refined(
                &mut fact, &kkt.airn, &kkt.ajcn, &kkt_vals, &mut rhs, &mut ir_b, &mut ir_r,
                &mut ir_d,
            )
            .is_err()
            {
                solve_failed = true;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            let gtq = dot(&prob.c, &dx)
                + two_over_tau * dot(&px_vec, &dx)
                + dot(&prob.b, &dy)
                + dot(&prob.h, &dz);
            // τκ corrector residual: τκ + Δτ_aff·Δκ_aff (target σμ).
            let r_tk = tau * kappa + dtau_aff * dkappa_aff;
            dtau = (-rho_tau - gtq - (sigma_mu - r_tk) / tau) / denom;
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
            dkappa = (sigma_mu - r_tk - kappa * dtau) / tau;
            cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

            // Single fraction-to-boundary step (HSDE is primal/dual-symmetric).
            // A *separate* primal/dual step is unsound here: τ couples both
            // residuals (ρ_x carries `cτ` + `Px`, ρ_y carries `bτ`), so stepping
            // the primal block (x, s, τ) and dual block (y, z, κ) by different
            // amounts leaves a dual-infeasibility residual ∝ (α_p − α_d) — on the
            // degenerate NETLIB GEN family (α_p ≫ α_d) that blows ρ_x up from
            // ~1e-8 to ~5e-2. The symmetric step keeps the embedding's clean
            // (1−α) residual decrease.
            alpha = ray_step(tau, dtau, opts.tau).min(ray_step(kappa, dkappa, opts.tau));
            if m_ineq > 0 {
                alpha = alpha
                    .min(cone.max_step(&s, &ds, opts.tau))
                    .min(cone.max_step(&z, &dz, opts.tau));
            }

            if alpha >= CENTERING_MIN_STEP
                || centering_tries >= CENTERING_MAX_TRIES
                || sigma >= CENTERING_SIGMA_MAX
            {
                break;
            }
            centering_tries += 1;
            sigma = (sigma * CENTERING_FACTOR).clamp(CENTERING_SIGMA_FLOOR, CENTERING_SIGMA_MAX);
        }
        if solve_failed {
            status = QpStatus::NumericalFailure;
            break;
        }

        // === Gondzio multiple centrality correctors ===
        // Restricted to the pure nonnegative orthant: the complementarity
        // product s∘z is then elementwise, so we can box-project it directly.
        // Each pass forms a trial step enlarged by GONDZIO_DELTA, projects the
        // resulting complementarity products into the centrality band
        // [β_lo·μ, β_hi·μ], solves a *corrector* system through the existing
        // factor (zero linear residual, only the re-centered complementarity
        // RHS), and accepts the extra direction only if it lengthens the step
        // by at least GONDZIO_GAMMA·GONDZIO_DELTA — otherwise the loop stops.
        if cone.is_orthant() && m_ineq > 0 && mu > 0.0 {
            let lo = GONDZIO_BETA_LO * mu;
            let hi = GONDZIO_BETA_HI * mu;
            for _ in 0..GONDZIO_MAX_CORR {
                let a_trial = (alpha + GONDZIO_DELTA).min(1.0);
                // Cone complementarity targets: project the trial products into
                // the band; r_c holds the deviation ṽ − t so that recover_ds
                // yields a correction with z∘cds + s∘cdz = t − ṽ.
                let mut active = false;
                for i in 0..m_ineq {
                    let v = (s[i] + a_trial * ds[i]) * (z[i] + a_trial * dz[i]);
                    let t = v.clamp(lo, hi);
                    r_c[i] = v - t;
                    if r_c[i] != 0.0 {
                        active = true;
                    }
                }
                // τ/κ complementarity target (same band).
                let v_tk = (tau + a_trial * dtau) * (kappa + a_trial * dkappa);
                let t_tk = v_tk.clamp(lo, hi);
                if !active && (v_tk - t_tk) == 0.0 {
                    break;
                }
                // Corrector system: zero linear residual, complementarity RHS
                // only. Solve through the existing factor with refinement.
                cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
                build_rhs(
                    &zeros_n, &zeros_meq, &zeros_m, &comp, n, m_eq, m_ineq, &mut rhs,
                );
                if solve_refined(
                    &mut fact, &kkt.airn, &kkt.ajcn, &kkt_vals, &mut rhs, &mut ir_b, &mut ir_r,
                    &mut ir_d,
                )
                .is_err()
                {
                    break;
                }
                split_step(&rhs, n, m_eq, m_ineq, &mut cdx, &mut cdy, &mut cdz);
                // τ-row Schur solve for the corrector (rho_tau = 0).
                let gtq_c = dot(&prob.c, &cdx)
                    + two_over_tau * dot(&px_vec, &cdx)
                    + dot(&prob.b, &cdy)
                    + dot(&prob.h, &cdz);
                let dtau_c = (-gtq_c - (t_tk - v_tk) / tau) / denom;
                for i in 0..n {
                    cdx[i] += dtau_c * p_x[i];
                }
                for i in 0..m_eq {
                    cdy[i] += dtau_c * p_y[i];
                }
                for i in 0..m_ineq {
                    cdz[i] += dtau_c * p_z[i];
                }
                let dkappa_c = (t_tk - v_tk - kappa * dtau_c) / tau;
                cone.recover_ds(&s, &z, &r_c, &cdz, &mut cds);
                // Trial enlarged step and its fraction-to-boundary length.
                for i in 0..m_ineq {
                    step_s[i] = ds[i] + cds[i];
                    step_z[i] = dz[i] + cdz[i];
                }
                let a_new = ray_step(tau, dtau + dtau_c, opts.tau)
                    .min(ray_step(kappa, dkappa + dkappa_c, opts.tau))
                    .min(cone.max_step(&s, &step_s, opts.tau))
                    .min(cone.max_step(&z, &step_z, opts.tau));
                if a_new >= alpha + GONDZIO_GAMMA * GONDZIO_DELTA {
                    for i in 0..n {
                        dx[i] += cdx[i];
                    }
                    for i in 0..m_eq {
                        dy[i] += cdy[i];
                    }
                    for i in 0..m_ineq {
                        dz[i] += cdz[i];
                        ds[i] += cds[i];
                    }
                    dtau += dtau_c;
                    dkappa += dkappa_c;
                    alpha = a_new;
                } else {
                    break;
                }
            }
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

    // VERDICT. The loop's convergence test reads the *homogeneous* residuals,
    // which carry the internal slack's consistency `Gx + s − hτ` alongside the
    // real KKT quantities. That term is bookkeeping — `s` is never returned —
    // and it floors out once `μ` reaches ~1e-16 and the NT scaling's condition
    // number explodes. So a solve can be optimal in every sense the caller can
    // observe while the surrogate refuses to certify it: on a feasible convex
    // QCQP `pres` bottomed at 1e-8, wandered back up to 1e-4, and the solve
    // ground on to a factorization breakdown while the iterate itself was
    // accurate to 1e-14 — reported as `InternalError` / exit 1 (pounce#209).
    //
    // So when the loop ends *without* a verdict of its own, ask the question
    // where it is actually defined: the true KKT error of the un-homogenized
    // point being returned, against this solve's own cones (see
    // [`true_kkt_error`], which is strictly stronger than the surrogate — it
    // also requires `ẑ ∈ K*`). Below `tol` that point satisfies the optimality
    // conditions and `Optimal` is the honest verdict; within `~1e3·tol` it is
    // usable at reduced accuracy (`OptimalInaccurate`, deliberately kept
    // distinguishable from a clean convergence — code review 2026-06 item M20);
    // beyond that the breakdown stands.
    //
    // This deliberately runs *after* the loop rather than as an extra
    // convergence test inside it. Breaking early on the certificate would stop
    // some solves sooner and hand back a less-polished iterate: on a degenerate
    // tangency (`min −x₁ s.t. ‖x‖ ≤ 1, x₀ ≥ 0`) the map from KKT residual to
    // `x`-error is a square root, so a residual of `tol` still leaves `x₀` off
    // by ~1e-4. Judging only at the end changes the verdict and never the
    // iterates. Costs a handful of matvecs, once, and only on a solve that
    // would otherwise have no answer.
    if matches!(
        status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit
    ) {
        // `x`/`y`/`z` are already un-homogenized above, hence `τ = 1`. Strictly
        // an upgrade: a point that fails both bands leaves the loop's own
        // verdict (breakdown *or* iteration limit) untouched.
        status = match true_kkt_error(prob, cone, &x, &y, &z, 1.0) {
            Some(e) if e < opts.tol => QpStatus::Optimal,
            Some(e) if e < 1e3 * opts.tol => QpStatus::OptimalInaccurate,
            _ => status,
        };
    }

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

    // Never hand back a success verdict without a usable solution (gh #222).
    let status = crate::ipm::demote_unusable(status, &x, obj);
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
        // Trivial dual: `z = 0` (the cone apex) is valid in every dual cone,
        // unlike the all-ones vector, which is not a member of an SOC of
        // dimension ≥ 3. Matches `ipm::failed_solution`.
        z: vec![0.0; prob.m_ineq()],
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

    /// The infeasibility-ray discriminant must key off the *ratio* κ/τ, not a
    /// bare `τ < ε` floor. Pins the exact `(τ, κ)` iterates POWELL20 produces:
    /// the early iterates where κ genuinely dominates (κ/τ ≫ 100) are on the
    /// ray; the collapsed-τ late iterate (κ/τ ~ 1e-9) — where the prior
    /// `κ.max(1.0)` floor wrongly fired and declared the feasible QP
    /// infeasible — must NOT be. (See `on_infeasibility_ray`.)
    #[test]
    fn infeasibility_ray_discriminant_is_ratio_not_floor() {
        // POWELL20 late iterate (it=33): τ=9.95e-15, κ=1.15e-23 ⇒ κ/τ ~ 1e-9.
        // Feasible-degenerate: both collapsing, κ does NOT dominate. The old
        // `τ < 1e-2·κ.max(1.0)` = `τ < 1e-2` floor fired here — the bug.
        assert!(
            !on_infeasibility_ray(9.95e-15, 1.15e-23),
            "feasible collapsed-τ iterate (κ ≪ τ) must not be flagged as a ray"
        );
        // POWELL20 early iterate (it=9): τ=1.1e-11, κ=0.637 ⇒ κ/τ ~ 6e10.
        // κ genuinely dominates — on the ray (the FARKAS_RESID_TOL residual
        // gate then rejects the still-too-large certificate, as it should).
        assert!(
            on_infeasibility_ray(1.1e-11, 0.637),
            "κ ≫ τ is the infeasibility-ray signature and must be flagged"
        );
        // A genuine small-certificate infeasible iterate: κ/τ = 1e6 > 100.
        assert!(on_infeasibility_ray(1e-9, 1e-3));
        // Boundary: κ/τ exactly 100 is not yet dominant (strict `>` after the
        // 1e-2 factor); κ/τ = 1000 is.
        assert!(!on_infeasibility_ray(1.0, 100.0));
        assert!(on_infeasibility_ray(1.0, 1000.0));
        // A converged feasible solve (τ → τ* > 0, κ → 0) is never a ray.
        assert!(!on_infeasibility_ray(0.8, 1e-12));
    }

    /// The scale-relative stopping test may only relax the absolute one once
    /// `tol`-level absolute accuracy is unreachable (`max_scale·ε > tol`). Pins
    /// the two regimes measured across the QP set: every bound-tightening
    /// presolve instance (scale ≲ 30, where postsolve dual re-attribution would
    /// otherwise amplify a relaxed reduced-problem residual) stays on the tight
    /// absolute test; the large-data cluster (POWELL20/QFORPLAN/QSHELL/BOYD at
    /// scale 7e9–4e12, whose absolute residuals floor far above `tol`) gets the
    /// relative certificate. (See `relative_stop_permitted`.)
    #[test]
    fn relative_stop_gated_on_unreachable_absolute_accuracy() {
        let tol = 1e-8;
        // Presolve-regime scales (measured at the stopping iterate): the gate
        // must stay shut so the absolute test governs.
        assert!(
            !relative_stop_permitted(32.0, tol),
            "scale 32 must use absolute"
        );
        assert!(
            !relative_stop_permitted(2.489e3, tol),
            "LISWET1 scale (reaches absolute tol) must use absolute"
        );
        // Large-data cluster scales (measured): the gate must open.
        assert!(
            relative_stop_permitted(7.457e9, tol),
            "QFORPLAN scale (smallest of the cluster) must permit relative"
        );
        assert!(relative_stop_permitted(5.209e10, tol), "POWELL20 scale");
        assert!(relative_stop_permitted(3.750e12, tol), "BOYD1 scale");
        // Crossover is `tol/ε`: just below stays absolute, just above goes
        // relative — and it tracks `tol` (a tighter tol opens the gate sooner).
        let crossover = tol / f64::EPSILON;
        assert!(!relative_stop_permitted(0.5 * crossover, tol));
        assert!(relative_stop_permitted(2.0 * crossover, tol));
        assert!(
            relative_stop_permitted(0.5 * crossover, 1e-10),
            "tighter tol lowers the crossover"
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
