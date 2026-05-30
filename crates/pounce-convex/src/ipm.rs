//! Primal-dual interior-point driver for convex QP.
//!
//! Infeasible-start primal-dual path-following with **Mehrotra
//! predictor-corrector** (adaptive centering σ = (μ_aff/μ)³ plus the
//! second-order `Δs∘Δz` term) and fraction-to-boundary step control.
//! Predictor and corrector share one factorization per iteration. The
//! homogeneous self-dual embedding (for clean infeasibility detection
//! and a self-starting iterate) is the remaining Phase 3 piece and slots
//! into this same scaffolding.
//!
//! On bound/inequality-constrained convex QPs this reaches the solution
//! in materially fewer interior-point iterations than routing the same
//! problem through the NLP filter-IPM — see
//! `crates/pounce-cli/tests/qp_vs_nlp_iterations.rs` (≈41% fewer at
//! n=50), the check behind the plan's 30–50% claim.
//!
//! ## Method
//!
//! For the standard-form QP (see [`crate::qp`]) with slacks `s ≥ 0` on
//! the inequalities (`Gx + s = h`) and multipliers `y` (equality),
//! `z ≥ 0` (inequality), the KKT conditions are
//!
//! ```text
//!   P x + c + Aᵀ y + Gᵀ z = 0      (stationarity, r_d)
//!   A x − b              = 0       (r_p)
//!   G x + s − h          = 0       (r_g)
//!   s ∘ z                = 0       (complementarity)
//! ```
//!
//! Each iteration solves the symmetric indefinite Newton system
//!
//! ```text
//!   ⎡ P+δI   Aᵀ      Gᵀ        ⎤ ⎡dx⎤   ⎡ −r_d            ⎤
//!   ⎢ A      −δI     0         ⎥ ⎢dy⎥ = ⎢ −r_p            ⎥
//!   ⎣ G      0    −(S⊘Z)−δI    ⎦ ⎣dz⎦   ⎣ −r_g + r_c ⊘ z  ⎦
//! ```
//!
//! (with `ds` recovered from `dz`) through the shared
//! [`pounce_linsol::Factorization`]. The tiny static regularization `δ`
//! makes the system quasi-definite so the LDLᵀ has a well-defined
//! inertia; because convergence is tested on the *unregularized*
//! residuals, the fixed point is the true QP solution — `δ` only
//! perturbs the search direction.
//!
//! The cone-specific pieces (`μ`, the `S⊘Z` scaling diagonal, the
//! complementarity residual, `ds` recovery, and the fraction-to-boundary
//! step) all route through the [`Cone`](crate::cones::Cone) trait so
//! that Phases 4–6 extend rather than rewrite this driver.

use crate::cones::{Cone, NonnegCone};
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// Options for the QP interior-point solve.
#[derive(Debug, Clone, Copy)]
pub struct QpOptions {
    /// Convergence tolerance on the max KKT residual and duality measure.
    pub tol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Fraction-to-boundary parameter τ ∈ (0, 1). (The centering
    /// parameter σ is computed adaptively by the Mehrotra predictor;
    /// it is not an option.)
    pub tau: f64,
    /// Static KKT regularization δ.
    pub reg: f64,
    /// Relative tolerance for accepting an infeasibility/unboundedness
    /// certificate. A certificate is declared only when its defining
    /// inequalities hold to this tolerance *relative to the certificate's
    /// own magnitude*, so the status is always backed by a verified
    /// proof — there are no false positives, only (rarely) an
    /// `IterationLimit` fallback when no certificate is verifiable.
    pub infeas_tol: f64,
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            tol: 1e-8,
            max_iter: 200,
            tau: 0.95,
            reg: 1e-8,
            infeas_tol: 1e-7,
        }
    }
}

/// Solve a convex QP with the bare primal-dual IPM, using `backend` for
/// the augmented-system factorization. `make_backend` is called once per
/// iteration (the KKT pattern is rebuilt each step in this first
/// increment; constant-pattern symbolic reuse is a documented follow-up,
/// see `dev-notes/performance-engineering.md`).
pub fn solve_qp_ipm<F>(prob: &QpProblem, opts: &QpOptions, mut make_backend: F) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let dim = n + m_eq + m_ineq;

    let cone = NonnegCone::new(m_ineq);

    // Infeasible-start iterate: x = 0, y = 0, s = z = 1. Strictly
    // interior for (s, z); primal/dual residuals are driven to zero.
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m_eq];
    let mut z = vec![1.0; m_ineq];
    let mut s = vec![1.0; m_ineq];

    // Scratch.
    let mut r_d = vec![0.0; n];
    let mut r_p = vec![0.0; m_eq];
    let mut r_g = vec![0.0; m_ineq];
    let mut r_c = vec![0.0; m_ineq];
    let mut scaling = vec![0.0; m_ineq];
    let mut rhs = vec![0.0; dim];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];

    // Build the fixed KKT pattern and the factorization *once*. The
    // pattern never changes across iterations — only the (z,z) scaling
    // diagonal — so each iteration recomputes O(m_ineq) values and
    // `refactor`s (numeric-only, reusing the symbolic factor / ordering)
    // instead of paying repeated symbolic analysis. This is what keeps
    // the per-iteration cost tracking the sparse factor rather than
    // blowing up on large sparse QPs.
    let kkt = KktStructure::build(prob, opts.reg);
    let mut kkt_vals = kkt.values.clone();
    cone.scaling_diag(&s, &z, &mut scaling);
    kkt.update_scaling(&scaling, opts.reg, &mut kkt_vals);
    let mut fact = match Factorization::new(
        dim as Index,
        kkt.airn.clone(),
        kkt.ajcn.clone(),
        kkt_vals.clone(),
        make_backend(),
    ) {
        Ok(f) => f,
        Err(_) => {
            return failed_solution(prob, x, y, z, 0);
        }
    };

    let mut iters = 0;
    let mut status = QpStatus::IterationLimit;

    for it in 0..opts.max_iter {
        iters = it;

        // --- residuals (unregularized; this is the convergence test) ---
        // r_d = P x + c + Aᵀ y + Gᵀ z
        r_d.iter_mut().zip(&prob.c).for_each(|(r, c)| *r = *c);
        prob.p_mul_add(&x, &mut r_d);
        prob.at_mul_add(&y, &mut r_d);
        prob.gt_mul_add(&z, &mut r_d);
        // r_p = A x − b
        r_p.iter_mut().zip(&prob.b).for_each(|(r, b)| *r = -*b);
        prob.a_mul_add(&x, &mut r_p);
        // r_g = G x + s − h
        for i in 0..m_ineq {
            r_g[i] = s[i] - prob.h[i];
        }
        prob.g_mul_add(&x, &mut r_g);

        let mu = cone.mu(&s, &z);
        let res = inf_norm(&r_d)
            .max(inf_norm(&r_p))
            .max(inf_norm(&r_g))
            .max(mu);
        if res < opts.tol {
            status = QpStatus::Optimal;
            break;
        }

        // Verified infeasibility / unboundedness detection. Checked
        // (not assumed), so a positive result is a proof and a false
        // positive is impossible; this is the HSDE benefit without the
        // homogeneous-embedding rewrite. Cheap (a few matvecs).
        if let Some(infeas) = detect_infeasibility(prob, &x, &y, &z, opts) {
            status = infeas;
            break;
        }

        // --- update only the (z,z) scaling diagonal and refactor
        // (numeric-only; the symbolic factor / ordering is reused). The
        // one factorization then backs both the predictor and corrector
        // solves this iteration. ---
        cone.scaling_diag(&s, &z, &mut scaling);
        kkt.update_scaling(&scaling, opts.reg, &mut kkt_vals);
        if fact.refactor(&kkt_vals).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }

        // === Predictor (affine-scaling) step: σ = 0 ===
        // r_c = s∘z (affine target).
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        build_rhs(&r_d, &r_p, &r_g, &r_c, &z, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        cone.recover_ds(&s, &z, &r_c, &dz, &mut ds_aff);
        dz_aff.copy_from_slice(&dz);

        // Affine step lengths and the predicted duality measure μ_aff.
        let (alpha_p_aff, alpha_d_aff) =
            step_lengths(&cone, &s, &ds_aff, &z, &dz_aff, opts.tau, m_ineq);
        let sigma = if m_ineq == 0 {
            0.0
        } else {
            // μ_aff = ⟨s + αp ds_aff, z + αd dz_aff⟩ / m
            let mut dot = 0.0;
            for i in 0..m_ineq {
                dot += (s[i] + alpha_p_aff * ds_aff[i]) * (z[i] + alpha_d_aff * dz_aff[i]);
            }
            let mu_aff = dot / m_ineq as f64;
            // Mehrotra's heuristic centering parameter σ = (μ_aff/μ)³.
            (mu_aff / mu).powi(3)
        };

        // === Corrector step: centered target + second-order term ===
        if m_ineq == 0 {
            // No cone: predictor is already the full Newton step.
            for i in 0..n {
                x[i] += dx[i];
            }
            for i in 0..m_eq {
                y[i] += dy[i];
            }
        } else {
            let sigma_mu = sigma * mu;
            cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
            build_rhs(&r_d, &r_p, &r_g, &r_c, &z, n, m_eq, m_ineq, &mut rhs);
            if fact.solve_one(&mut rhs).is_err() {
                status = QpStatus::NumericalFailure;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

            let (alpha_p, alpha_d) = step_lengths(&cone, &s, &ds, &z, &dz, opts.tau, m_ineq);
            for i in 0..n {
                x[i] += alpha_p * dx[i];
            }
            for i in 0..m_eq {
                y[i] += alpha_d * dy[i];
            }
            for i in 0..m_ineq {
                s[i] += alpha_p * ds[i];
                z[i] += alpha_d * dz[i];
            }
        }
    }

    // Objective ½ xᵀP x + cᵀx.
    let mut px = vec![0.0; n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }

    QpSolution {
        status,
        x,
        y,
        z,
        obj,
        iters,
    }
}

/// Build a `NumericalFailure` solution from the current iterate (used
/// when the *initial* factorization fails before the loop starts).
fn failed_solution(
    prob: &QpProblem,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    iters: usize,
) -> QpSolution {
    let mut px = vec![0.0; prob.n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..prob.n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }
    QpSolution {
        status: QpStatus::NumericalFailure,
        x,
        y,
        z,
        obj,
        iters,
    }
}

/// Build the Newton RHS `[−r_d; −r_p; −r_g + r_c ⊘ z]` for a given
/// complementarity residual `r_c` (predictor or corrector).
#[allow(clippy::too_many_arguments)]
fn build_rhs(
    r_d: &[f64],
    r_p: &[f64],
    r_g: &[f64],
    r_c: &[f64],
    z: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    rhs: &mut [f64],
) {
    for i in 0..n {
        rhs[i] = -r_d[i];
    }
    for i in 0..m_eq {
        rhs[n + i] = -r_p[i];
    }
    for i in 0..m_ineq {
        rhs[n + m_eq + i] = -r_g[i] + r_c[i] / z[i];
    }
}

/// Copy the solved RHS into the (dx, dy, dz) step components.
fn split_step(
    rhs: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    dz: &mut [f64],
) {
    dx.copy_from_slice(&rhs[0..n]);
    dy.copy_from_slice(&rhs[n..n + m_eq]);
    dz.copy_from_slice(&rhs[n + m_eq..n + m_eq + m_ineq]);
}

/// Separate fraction-to-boundary step lengths for the primal slack `s`
/// (via `ds`) and dual `z` (via `dz`). Returns `(alpha_primal,
/// alpha_dual)`; both are 1 when there is no cone.
fn step_lengths(
    cone: &NonnegCone,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    tau: f64,
    m_ineq: usize,
) -> (f64, f64) {
    if m_ineq == 0 {
        return (1.0, 1.0);
    }
    (cone.max_step(s, ds, tau), cone.max_step(z, dz, tau))
}

/// Bench-only re-export of the KKT assembly so the `scaling` example can
/// time it in isolation. Not part of the public solving API.
#[doc(hidden)]
pub fn assemble_kkt_for_bench(
    prob: &QpProblem,
    scaling: &[f64],
    reg: f64,
    _dim: usize,
) -> (Vec<Index>, Vec<Index>, Vec<Number>) {
    let kkt = KktStructure::build(prob, reg);
    let mut vals = kkt.values.clone();
    kkt.update_scaling(scaling, reg, &mut vals);
    (kkt.airn, kkt.ajcn, vals)
}

/// Fixed-pattern KKT structure for the QP augmented system.
///
/// The KKT *sparsity pattern* is identical across all IPM iterations —
/// only the `(z, z)` diagonal (the cone scaling block) changes from step
/// to step. This struct captures the pattern (`airn`/`ajcn`, 1-based
/// lower triangle) and the constant part of the values once, plus the
/// positions of the scaling-dependent diagonal entries, so each
/// iteration recomputes only `O(m_ineq)` values and the solver can
/// `refactor` (numeric-only, reusing the symbolic factor / fill-reducing
/// ordering) instead of rebuilding the factorization from scratch. This
/// is the constant-pattern symbolic reuse called for in
/// `dev-notes/performance-engineering.md`; without it the per-iteration
/// cost is dominated by repeated symbolic analysis on large sparse QPs.
struct KktStructure {
    airn: Vec<Index>,
    ajcn: Vec<Index>,
    /// Constant values (everything except the scaling block; the
    /// `(z, z)` diagonal entries hold their `-reg` term here).
    values: Vec<Number>,
    /// `z_diag_pos[i]` = index into `values` of inequality `i`'s
    /// `(z, z)` diagonal entry.
    z_diag_pos: Vec<usize>,
}

impl KktStructure {
    /// Build the pattern and constant values once. The `(z, z)` diagonal
    /// entries are seeded with `-reg`; [`Self::update_scaling`] adds the
    /// per-iteration `-scaling[i]` on top.
    fn build(prob: &QpProblem, reg: f64) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let m_ineq = prob.m_ineq();
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            *entries.entry((r, c)).or_insert(0.0) += v;
        };

        // (x,x): P + δI.
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
        }
        for i in 0..n {
            add(i, i, reg);
        }
        // (y,x): A; (y,y): −δI.
        for t in &prob.a {
            add(n + t.row, t.col, t.val);
        }
        for i in 0..m_eq {
            add(n + i, n + i, -reg);
        }
        // (z,x): G; (z,z): seed −δI (scaling added per iteration).
        for t in &prob.g {
            add(n + m_eq + t.row, t.col, t.val);
        }
        for i in 0..m_ineq {
            add(n + m_eq + i, n + m_eq + i, -reg);
        }

        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        // Map (z,z) diagonal coordinates → output position.
        let mut coord_to_pos: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (pos, ((r, c), v)) in entries.into_iter().enumerate() {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
            coord_to_pos.insert((r, c), pos);
        }
        let z_diag_pos: Vec<usize> = (0..m_ineq)
            .map(|i| coord_to_pos[&(n + m_eq + i, n + m_eq + i)])
            .collect();

        KktStructure {
            airn,
            ajcn,
            values,
            z_diag_pos,
        }
    }

    /// Write the per-iteration scaling into `out` (which must start as a
    /// copy of `self.values`): sets each `(z, z)` diagonal entry to
    /// `-scaling[i] - reg`.
    fn update_scaling(&self, scaling: &[f64], reg: f64, out: &mut [Number]) {
        for (i, &pos) in self.z_diag_pos.iter().enumerate() {
            out[pos] = -scaling[i] - reg;
        }
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Check the current iterate for a *verified* infeasibility certificate.
///
/// Returns `Some(PrimalInfeasible | DualInfeasible)` **only** when the
/// certificate's defining (in)equalities hold to `opts.infeas_tol`
/// relative to the certificate's own magnitude. Because the certificate
/// is checked, not assumed, a positive result is a genuine proof and a
/// false positive is impossible; an unverifiable iterate returns `None`
/// and the solve keeps going (ultimately `IterationLimit`).
///
/// This recovers HSDE's headline benefit — clean infeasible/unbounded
/// status instead of silently exhausting the iteration budget — without
/// the homogeneous embedding's full rewrite of the iteration. When the
/// problem is primal-infeasible the IPM's dual iterate `(y, z)` diverges
/// along a Farkas ray, so its normalization satisfies the primal
/// certificate; when the problem is unbounded the primal iterate `x`
/// diverges along a recession direction satisfying the dual certificate.
///
/// Certificates (for `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ≤ h`):
/// - **Primal infeasible:** `(y, z ≥ 0)` with `Aᵀy + Gᵀz ≈ 0` and
///   `bᵀy + hᵀz < 0` (Farkas). `z ≥ 0` is maintained by the IPM.
/// - **Dual infeasible / unbounded:** direction `d` (= `x`) with
///   `Pd ≈ 0, Ad ≈ 0, Gd ≤ 0, cᵀd < 0`.
fn detect_infeasibility(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    opts: &QpOptions,
) -> Option<QpStatus> {
    let n = prob.n;
    let ctol = opts.infeas_tol;

    // --- Primal infeasibility (Farkas certificate) ---
    let dual_norm = inf_norm(y).max(inf_norm(z));
    if dual_norm > 0.0 {
        let mut resid = vec![0.0; n]; // Aᵀy + Gᵀz
        prob.at_mul(y, &mut resid);
        prob.gt_mul(z, &mut resid);
        let cert = dot(&prob.b, y) + dot(&prob.h, z); // bᵀy + hᵀz
        let z_ok = z.iter().all(|&zi| zi >= -ctol * dual_norm);
        if cert < -ctol * dual_norm && inf_norm(&resid) <= ctol * dual_norm && z_ok {
            return Some(QpStatus::PrimalInfeasible);
        }
    }

    // --- Dual infeasibility / unboundedness (recession direction d = x) ---
    let x_norm = inf_norm(x);
    if x_norm > 0.0 {
        let mut pd = vec![0.0; n];
        prob.p_mul(x, &mut pd);
        let mut ad = vec![0.0; prob.m_eq()];
        prob.a_mul(x, &mut ad);
        let mut gd = vec![0.0; prob.m_ineq()];
        prob.g_mul(x, &mut gd);
        let cd = dot(&prob.c, x);
        let gd_max = gd.iter().fold(0.0_f64, |m, &v| m.max(v));
        if cd < -ctol * x_norm
            && inf_norm(&pd) <= ctol * x_norm
            && inf_norm(&ad) <= ctol * x_norm
            && gd_max <= ctol * x_norm
        {
            return Some(QpStatus::DualInfeasible);
        }
    }

    None
}
