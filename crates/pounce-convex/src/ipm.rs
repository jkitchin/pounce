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

use crate::cones::{CompositeCone, Cone, ConeBlock, ConeSpec};
use crate::qp::{QpIterate, QpProblem, QpSolution, QpStatus};
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
    /// Use the homogeneous self-dual embedding driver ([`crate::hsde`])
    /// instead of the default infeasible-start primal–dual method. The HSDE
    /// driver self-starts, produces infeasibility certificates natively, and
    /// is the substrate for the non-symmetric cones (exp/power). It does not
    /// (yet) exploit warm starts or reuse an external factorization, so the
    /// default path keeps those advantages for symmetric cones; this opts a
    /// single solve into the embedding. Default `false`.
    pub use_hsde: bool,
    /// Collect a per-iteration convergence trace into
    /// [`crate::QpSolution::iterates`]. Off by default so a normal solve has
    /// no recording overhead; turn on when a solve report or benchmark
    /// harness wants the per-iteration history. Default `false`.
    pub collect_iterates: bool,
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            tol: 1e-8,
            max_iter: 200,
            tau: 0.95,
            reg: 1e-8,
            infeas_tol: 1e-7,
            use_hsde: false,
            collect_iterates: false,
        }
    }
}

/// Solve a convex QP, honoring any per-variable bounds (`lb`/`ub`).
///
/// Variable bounds are a first-class part of [`QpProblem`] so presolve
/// can reason about boxes; the solver itself expands the *finite* bounds
/// into internal inequality rows, runs the bounds-agnostic Mehrotra core
/// ([`solve_qp_core`]), and splits the returned inequality multipliers
/// back into the original `z` and the bound multipliers `z_lb`/`z_ub`.
/// The iteration math is unchanged by the presence of bounds.
pub fn solve_qp_ipm<F>(prob: &QpProblem, opts: &QpOptions, make_backend: F) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return solve_qp_core(prob, &cone, opts, None, make_backend);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Solve a convex QP starting from a warm point (typically a previous
/// solution of a nearby problem). See [`QpWarmStart`] for the centering
/// strategy and when warm starting helps.
///
/// Identical to [`solve_qp_ipm`] except the interior-point iteration is
/// seeded from `warm` instead of the cold default. The *solution* is
/// independent of the start (the IPM converges to the same KKT point); a
/// good warm start only reduces the iteration count.
pub fn solve_qp_ipm_warm<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    warm: &QpWarmStart,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let w = WarmStart {
            x: warm.x.clone(),
            y: warm.y.clone(),
            z: warm.z.clone(),
        };
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return solve_qp_core(prob, &cone, opts, Some(&w), make_backend);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let w = WarmStart {
        x: warm.x.clone(),
        y: warm.y.clone(),
        z: merge_bound_duals(prob, &bound_rows, warm),
    };
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, opts, Some(&w), make_backend);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Solve a standard-form **SOCP** (or mixed LP/QP + second-order cones):
/// `min ½xᵀPx+cᵀx s.t. Ax=b, Gx ⪯_K h`, where the inequality block `Gx ≤ h`
/// is partitioned into the cones `K` described by `cones` (in row order;
/// each `s = h − Gx` block must lie in its cone). `cones` must cover the
/// `m_ineq` rows. Variable bounds (`lb`/`ub`) are appended as a trailing
/// nonnegative block.
pub fn solve_socp_ipm<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // The cones must partition the inequality rows exactly; otherwise the
    // cone vectors and the `m_ineq` slack disagree and the driver would read
    // out of bounds (an exp/power cone is always 3 rows). Fail cleanly here.
    if !cone_dims_cover(cones, prob.m_ineq()) {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    // Non-symmetric cones (exponential / power) route to the dedicated HSDE
    // driver; self-scaled cones (orthant / SOC / PSD) stay on the symmetric
    // path below. Mixing the two families in one problem is not supported.
    let has_nonsym = cones
        .iter()
        .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)));
    let has_psd = cones.iter().any(|c| matches!(c, ConeSpec::Psd(_)));
    if has_nonsym && has_psd {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    if has_nonsym {
        return solve_nonsym(prob, cones, opts, make_backend);
    }
    if !prob.has_bounds() {
        let cone = CompositeCone::from_specs(cones);
        return solve_qp_core(prob, &cone, opts, None, make_backend);
    }
    // Bounds expand into a trailing nonnegative block after the user cones.
    let (expanded, bound_rows) = expand_bounds(prob);
    let mut specs = cones.to_vec();
    specs.push(ConeSpec::Nonneg(bound_rows.len()));
    let cone = CompositeCone::from_specs(&specs);
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Warm-started [`solve_socp_ipm`]: seed the iteration from `warm` (a nearby
/// SOCP's solution). The warm `(s, z)` are projected into each cone's
/// interior (orthant positivity / SOC `λ_min` floor); the solution is
/// start-independent, so warm starting only reduces the iteration count.
/// `prob` must be bound-free (use `G`/`h` rows for all constraints).
pub fn solve_socp_ipm_warm<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    warm: &QpWarmStart,
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    assert!(
        !prob.has_bounds(),
        "solve_socp_ipm_warm: encode bounds as G/h rows (bound expansion + warm not combined)"
    );
    if !cone_dims_cover(cones, prob.m_ineq()) {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    let cone = CompositeCone::from_specs(cones);
    let w = WarmStart {
        x: warm.x.clone(),
        y: warm.y.clone(),
        z: warm.z.clone(),
    };
    solve_qp_core(prob, &cone, opts, Some(&w), make_backend)
}

/// Route a problem whose cone product contains an **exponential** cone to the
/// non-symmetric HSDE driver ([`crate::hsde_nonsym`]). Orthant, second-order,
/// exponential, and power blocks are all supported (a second-order cone may be
/// mixed with a non-symmetric one). Variable bounds expand into a trailing
/// orthant block exactly as in the symmetric path.
fn solve_nonsym<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    use crate::hsde_nonsym::{solve_conic_hsde_nonsym, NsBlock};

    fn blocks_of(cones: &[ConeSpec], extra_orthant: usize) -> Vec<NsBlock> {
        let mut blocks = Vec::with_capacity(cones.len() + 1);
        for c in cones {
            match c {
                ConeSpec::Nonneg(n) => blocks.push(NsBlock::Orthant(*n)),
                ConeSpec::SecondOrder(m) => blocks.push(NsBlock::SecondOrder(*m)),
                ConeSpec::Exponential => blocks.push(NsBlock::exp()),
                ConeSpec::Power(a) => blocks.push(NsBlock::power(*a)),
                // PSD is self-scaled and runs on the symmetric driver; the
                // PSD-with-exp/power mix is rejected upstream in
                // `solve_socp_ipm`, so this arm is never reached.
                ConeSpec::Psd(_) => {
                    unreachable!("PSD cone routes to the symmetric driver, not hsde_nonsym")
                }
            }
        }
        if extra_orthant > 0 {
            blocks.push(NsBlock::Orthant(extra_orthant));
        }
        blocks
    }

    if !prob.has_bounds() {
        let blocks = blocks_of(cones, 0);
        return solve_conic_hsde_nonsym(prob, &blocks, opts, make_backend);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let blocks = blocks_of(cones, bound_rows.len());
    let sol = solve_conic_hsde_nonsym(&expanded, &blocks, opts, make_backend);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Expand a problem's finite variable bounds into extra `G` rows
/// (`x_i ≤ ub_i` and `−x_i ≤ −lb_i`), returning the bounds-free expanded
/// problem and the `(row, var, is_upper)` provenance of each appended row
/// so the bound multipliers can be split back out.
fn expand_bounds(prob: &QpProblem) -> (QpProblem, Vec<(usize, usize, bool)>) {
    let mut g = prob.g.clone();
    let mut h = prob.h.clone();
    let mut bound_rows: Vec<(usize, usize, bool)> = Vec::new();
    for i in 0..prob.n {
        let ub = prob.ub_of(i);
        if ub < crate::qp::BOUND_INF {
            let r = h.len();
            g.push(crate::qp::Triplet::new(r, i, 1.0));
            h.push(ub);
            bound_rows.push((r, i, true));
        }
        let lb = prob.lb_of(i);
        if lb > -crate::qp::BOUND_INF {
            let r = h.len();
            g.push(crate::qp::Triplet::new(r, i, -1.0));
            h.push(-lb);
            bound_rows.push((r, i, false));
        }
    }
    let expanded = QpProblem {
        n: prob.n,
        p_lower: prob.p_lower.clone(),
        c: prob.c.clone(),
        a: prob.a.clone(),
        b: prob.b.clone(),
        g,
        h,
        lb: Vec::new(),
        ub: Vec::new(),
    };
    (expanded, bound_rows)
}

/// A warm-start iterate: a previous primal/dual solution to seed the
/// interior-point iteration for a *nearby* problem (same structure, mildly
/// perturbed `c`/`b`/`h`/bounds). Its fields mirror [`QpSolution`], so the
/// idiomatic use is to feed back the prior solve's solution.
///
/// ## Why warm starting an IPM needs care
///
/// Unlike active-set/simplex methods, a primal-dual interior-point method
/// converges *to* the complementarity boundary (`s∘z → 0`). A converged
/// warm point therefore lies essentially **on** that boundary — the worst
/// place to restart, since the IPM needs a well-centered interior iterate.
/// Seeding `(x, s, z)` verbatim typically stalls.
///
/// [`solve_qp_ipm_warm`] handles this with a Mehrotra-style recentering
/// ([`init_iterate`]): it keeps the warm primal `x` (whose slack pattern
/// `h − Gx` encodes the active set) but pushes the slacks `s` and
/// multipliers `z` back into the interior with a **scale-aware floor**, so
/// the start is genuinely interior and centered while still benefiting
/// from the warm `x`. The benefit is real but bounded — it is largest when
/// the active set is stable across the perturbation, and modest or absent
/// when it changes substantially (a known property of IPM warm starts).
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    /// Primal iterate (length `n`).
    pub x: Vec<f64>,
    /// Equality multipliers (length `m_eq`).
    pub y: Vec<f64>,
    /// Inequality multipliers for the original `G` rows (length `m_ineq`).
    pub z: Vec<f64>,
    /// Lower-bound multipliers (length `n`).
    pub z_lb: Vec<f64>,
    /// Upper-bound multipliers (length `n`).
    pub z_ub: Vec<f64>,
}

impl QpWarmStart {
    /// Build a warm start from a previous [`QpSolution`].
    pub fn from_solution(sol: &QpSolution) -> Self {
        QpWarmStart {
            x: sol.x.clone(),
            y: sol.y.clone(),
            z: sol.z.clone(),
            z_lb: sol.z_lb.clone(),
            z_ub: sol.z_ub.clone(),
        }
    }
}

/// Internal warm start expressed in the *expanded* space (variable bounds
/// already folded into the inequality block, so `z` covers `G`-rows then
/// the appended bound rows).
struct WarmStart {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

/// Build the expanded-space `z` for a warm start: the original `G`-row
/// multipliers followed by each appended bound row's `z_lb`/`z_ub` value,
/// in the same append order as [`expand_bounds`]. Inverse of
/// [`split_bound_duals`]'s `z` handling.
fn merge_bound_duals(
    prob: &QpProblem,
    bound_rows: &[(usize, usize, bool)],
    warm: &QpWarmStart,
) -> Vec<f64> {
    let base_m = prob.m_ineq();
    let mut z = vec![0.0; base_m + bound_rows.len()];
    let copy = base_m.min(warm.z.len());
    z[..copy].copy_from_slice(&warm.z[..copy]);
    for &(r, var, is_upper) in bound_rows {
        let v = if is_upper {
            warm.z_ub.get(var).copied().unwrap_or(0.0)
        } else {
            warm.z_lb.get(var).copied().unwrap_or(0.0)
        };
        if r < z.len() {
            z[r] = v;
        }
    }
    z
}

/// Move the appended bound rows' multipliers from the expanded solution's
/// `z` into `z_lb`/`z_ub`, and trim `z` back to the original rows.
fn split_bound_duals(
    prob: &QpProblem,
    bound_rows: &[(usize, usize, bool)],
    mut sol: QpSolution,
) -> QpSolution {
    let base_m = prob.m_ineq();
    let mut z = vec![0.0; base_m];
    z.copy_from_slice(&sol.z[..base_m]);
    let mut z_lb = vec![0.0; prob.n];
    let mut z_ub = vec![0.0; prob.n];
    for &(r, var, is_upper) in bound_rows {
        if is_upper {
            z_ub[var] = sol.z[r];
        } else {
            z_lb[var] = sol.z[r];
        }
    }
    sol.z = z;
    sol.z_lb = z_lb;
    sol.z_ub = z_ub;
    sol
}

/// Bounds-agnostic Mehrotra predictor-corrector core. `prob.lb`/`ub` are
/// ignored here; the public [`solve_qp_ipm`] handles bound expansion.
fn solve_qp_core<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    warm: Option<&WarmStart>,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Opt-in homogeneous self-dual embedding driver. It builds its own
    // factorization and self-starts, so it bypasses the warm-start /
    // factor-reuse plumbing below (warm is ignored — it cannot change the
    // solution, only the iteration count, which HSDE does not exploit yet).
    if opts.use_hsde {
        return crate::hsde::solve_conic_hsde(prob, cone, opts, make_backend);
    }

    // Build the fixed KKT pattern and an initial factorization, then run
    // the iteration. The pattern is constant across iterations (only the
    // cone scaling block changes), so the loop `refactor`s rather than
    // re-analyzing. Build-once / solve-many across *instances* with the
    // same pattern is exposed via [`QpFactorization`].
    let (kkt, mut fact) = match build_factorization(prob, cone, opts, &mut make_backend) {
        Ok(pair) => pair,
        Err(()) => {
            let n = prob.n;
            return failed_solution(
                prob,
                vec![0.0; n],
                vec![0.0; prob.m_eq()],
                vec![1.0; prob.m_ineq()],
                0,
            );
        }
    };
    run_ipm(prob, cone, opts, &kkt, &mut fact, warm)
}

/// Build the constant KKT pattern for `prob` and a `Factorization` over
/// it (seeded with the initial scaling). Shared by the single-shot path
/// and the reusable [`QpFactorization`] handle. `Err(())` ⇒ the initial
/// factorization failed.
pub(crate) fn build_factorization<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    make_backend: &mut F,
) -> Result<(KktStructure, Factorization), ()>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Seed the scaling at the cone identity (s = z = e ⇒ block = I).
    let mut e = vec![0.0; prob.m_ineq()];
    cone.identity(&mut e);

    let kkt = KktStructure::build(prob, cone, opts.reg);
    let dim = kkt.dim; // base rows + per-SOC auxiliary variables
    let mut kkt_vals = kkt.values.clone();
    kkt.update_blocks(cone, &e, &e, opts.reg, &mut kkt_vals);
    let fact = Factorization::new(
        dim as Index,
        kkt.airn.clone(),
        kkt.ajcn.clone(),
        kkt_vals,
        make_backend(),
    )
    .map_err(|_| ())?;
    Ok((kkt, fact))
}

/// Build the starting iterate `(x, y, z, s)` for [`run_ipm`].
///
/// With no warm start (`warm = None`) this is the cold default
/// `x = 0, y = 0, z = 1, s = 1` — a perfectly centered interior point
/// (`s∘z = 1`) — preserving the established cold-start behavior exactly.
///
/// With a warm start it applies a **Mehrotra-style recentering** seeded
/// from the warm point (Mehrotra 1992, §7, adapted for warm starting):
///
/// 1. Keep the warm primal `x` and equality multipliers `y`.
/// 2. Take the implied slacks `s̃ = h − Gx` (their signs encode which
///    inequalities the warm `x` makes active/violated) and the warm `z`.
/// 3. Shift both into the strict interior by `δ = max(−1.5·min(·), floor)`.
///    The `floor` is **adaptive**: it is the warm point's KKT residual `ρ`
///    on *this* problem, clamped to `[1e-9·scale, 0.1·scale]` with
///    `scale = max(1, ‖s̃‖∞, ‖z‖∞)`. A converged warm point sits on the
///    complementarity boundary (`s̃ᵢ` or `zᵢ ≈ 0`), so a floor is required
///    to keep the restart interior — but a *fixed* floor overwrites the
///    warm dual structure and degrades to a primal-only warm start.
///    Sizing the floor to `ρ` keeps `s`/`z` near their warm (correctly
///    structured) values when the problem is nearby (small `ρ`), so the
///    IPM exploits the warm duals — and softens toward the conservative
///    `0.1·scale` when the active set has moved (large `ρ`). This both
///    deepens the benefit on nearby problems and keeps it from ever doing
///    worse than a centered start.
/// 4. A final centering shift `½(s·z)/Σz`, `½(s·z)/Σs` balances `s` and
///    `z` (Mehrotra's second step).
///
/// The returned iterate always satisfies `s > 0, z > 0`. If `warm`'s
/// dimensions don't match the (expanded) problem it is ignored and the
/// cold start is used, so a stale warm start can never corrupt a solve.
fn init_iterate(
    prob: &QpProblem,
    cone: &CompositeCone,
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    warm: Option<&WarmStart>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Cold start at the cone identity e (orthant: all ones; SOC: (1,0,…)),
    // a perfectly centered interior point (s∘z = e).
    let cold = || {
        let mut e = vec![0.0; m_ineq];
        cone.identity(&mut e);
        (vec![0.0; n], vec![0.0; m_eq], e.clone(), e)
    };
    // A matching primal `x` is enough to warm start; `y`/`z` fall back to
    // the cold values when they don't match (so a primal-only warm start —
    // e.g. feeding back just the previous primal — is supported).
    let w = match warm {
        Some(w) if w.x.len() == n => w,
        _ => return cold(),
    };

    let x = w.x.clone();
    let y = if w.y.len() == m_eq {
        w.y.clone()
    } else {
        vec![0.0; m_eq]
    };
    let mut z = if w.z.len() == m_ineq {
        w.z.clone()
    } else {
        let mut e = vec![0.0; m_ineq];
        cone.identity(&mut e);
        e
    };

    // No cone: x/y are the whole iterate, s/z are empty.
    if m_ineq == 0 {
        return (x, y, z, Vec::new());
    }

    // Implied slacks s̃ = h − Gx.
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(&x, &mut gx);
    let mut s: Vec<f64> = (0..m_ineq).map(|i| prob.h[i] - gx[i]).collect();

    let scale = 1.0_f64.max(inf_norm(&s)).max(inf_norm(&z));

    // Adaptive interior floor sized to the warm point's KKT residual ρ on
    // *this* problem. ρ measures how far the warm point is from satisfying
    // the new KKT system: a small ρ (nearby problem, stable active set)
    // lets the slacks/multipliers stay near their warm — correctly
    // structured — values, so the IPM exploits the warm duals and needs
    // few steps; a large ρ (the active set moved, so the warm point is
    // badly infeasible) softens the floor toward the conservative cold
    // level `0.1·scale`. This self-corrects: warm starting never does
    // worse than a centered start, and gains the most when it can.
    let floor = {
        let mut rd = prob.c.clone();
        prob.p_mul_add(&x, &mut rd);
        prob.at_mul_add(&y, &mut rd);
        prob.gt_mul_add(&z, &mut rd);
        let mut rp: Vec<f64> = prob.b.iter().map(|b| -b).collect();
        prob.a_mul_add(&x, &mut rp);
        // Inequality infeasibility of the warm point: max(0, Gx − h) = −s̃.
        let viol = s.iter().fold(0.0_f64, |m, &si| m.max((-si).max(0.0)));
        let rho = inf_norm(&rd).max(inf_norm(&rp)).max(viol);
        rho.clamp(1e-9 * scale, 0.1 * scale)
    };
    // Project (s, z) into the strict interior of each cone block and
    // rebalance (orthant: positivity + Mehrotra; SOC: lift λ_min).
    cone.recenter_warm(&mut s, &mut z, floor);
    (x, y, z, s)
}

/// Run the Mehrotra predictor-corrector iteration for `prob` given an
/// already-built KKT pattern (`kkt`) and a live `Factorization` (`fact`)
/// over that pattern. The factorization is re-numeric-factored each
/// iteration (symbolic reuse); when `fact` is reused across instances
/// with the *same pattern*, the AMD ordering / symbolic factor is reused
/// across instances too.
fn run_ipm(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    kkt: &KktStructure,
    fact: &mut Factorization,
    warm: Option<&WarmStart>,
) -> QpSolution {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    let (mut x, mut y, mut z, mut s) = init_iterate(prob, cone, n, m_eq, m_ineq, warm);

    let mut r_d = vec![0.0; n];
    let mut r_p = vec![0.0; m_eq];
    let mut r_g = vec![0.0; m_ineq];
    let mut r_c = vec![0.0; m_ineq];
    let mut rhs_term = vec![0.0; m_ineq];
    // The KKT system carries one auxiliary variable per second-order cone;
    // the rhs is sized to it (auxiliary rows are zero).
    let mut rhs = vec![0.0; kkt.dim];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];
    let mut kkt_vals = kkt.values.clone();

    let mut iters = 0;
    let mut status = QpStatus::IterationLimit;
    let mut iterates: Vec<QpIterate> = Vec::new();

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
        let pinf = inf_norm(&r_p).max(inf_norm(&r_g));
        let dinf = inf_norm(&r_d);
        let res = dinf.max(pinf).max(mu);
        if res < opts.tol {
            status = QpStatus::Optimal;
            break;
        }
        // Per-iteration objective, only when a trace is being collected.
        let obj_it = if opts.collect_iterates {
            let mut px = vec![0.0; n];
            prob.p_mul_add(&x, &mut px);
            (0..n).map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i]).sum()
        } else {
            0.0
        };

        // Verified infeasibility / unboundedness detection. Checked
        // (not assumed), so a positive result is a proof and a false
        // positive is impossible; this is the HSDE benefit without the
        // homogeneous-embedding rewrite. Cheap (a few matvecs).
        if let Some(infeas) = detect_infeasibility(prob, &x, &y, &z, opts) {
            status = infeas;
            break;
        }

        // --- update the cone scaling block(s) and refactor (numeric-only;
        // the symbolic factor / ordering is reused). The one factorization
        // then backs both the predictor and corrector solves. ---
        kkt.update_blocks(cone, &s, &z, opts.reg, &mut kkt_vals);
        if fact.refactor(&kkt_vals).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }

        // === Predictor (affine-scaling) step: σ = 0 ===
        // r_c = s∘z (affine target).
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut rhs_term);
        build_rhs(&r_d, &r_p, &r_g, &rhs_term, n, m_eq, m_ineq, &mut rhs);
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
        // Step lengths taken this iteration (full step when there is no cone).
        let (mut step_p, mut step_d) = (1.0_f64, 1.0_f64);
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
            cone.rhs_comp_term(&s, &z, &r_c, &mut rhs_term);
            build_rhs(&r_d, &r_p, &r_g, &rhs_term, n, m_eq, m_ineq, &mut rhs);
            if fact.solve_one(&mut rhs).is_err() {
                status = QpStatus::NumericalFailure;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

            let (alpha_p, alpha_d) = step_lengths(&cone, &s, &ds, &z, &dz, opts.tau, m_ineq);
            step_p = alpha_p;
            step_d = alpha_d;
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

        if opts.collect_iterates {
            iterates.push(QpIterate {
                iter: it,
                objective: obj_it,
                primal_infeasibility: pinf,
                dual_infeasibility: dinf,
                mu,
                alpha_primal: step_p,
                alpha_dual: step_d,
            });
        }
    }

    // Objective ½ xᵀP x + cᵀx.
    let mut px = vec![0.0; n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }

    let nn = n;
    QpSolution {
        status,
        x,
        y,
        z,
        z_lb: vec![0.0; nn],
        z_ub: vec![0.0; nn],
        obj,
        iters,
        iterates,
    }
}

/// A reusable convex-QP factorization: build the KKT symbolic factor
/// (AMD ordering) **once** for a fixed problem *structure*, then solve
/// many instances that share that structure, paying the symbolic
/// analysis only on construction. This is the build-once / solve-many
/// handle (cf. the JAX `JaxProblem` from pounce#75) at the convex-QP
/// level.
///
/// "Same structure" means: same `n`, same `A`/`G`/`P` sparsity pattern,
/// and the same *set* of finite variable bounds (so the bound-expanded
/// KKT pattern is identical). Only the numeric data — `c`, `b`, `h`, and
/// the bound *values* — may change between solves. A solve whose problem
/// does not match the captured structure returns
/// [`QpStatus::NumericalFailure`] rather than silently producing a wrong
/// answer; use the one-shot [`solve_qp_ipm`] for heterogeneous problems.
pub struct QpFactorization {
    fact: Factorization,
    opts: QpOptions,
    /// The (orthant) inequality cone of the expanded problem; reused for
    /// the KKT pattern check and the per-solve scaling.
    cone: CompositeCone,
    /// Captured structure fingerprint for the per-solve compatibility
    /// check (same `n` and same expanded KKT pattern).
    n: usize,
    airn: Vec<Index>,
    ajcn: Vec<Index>,
}

impl QpFactorization {
    /// Build the reusable factor from a representative `base` problem.
    /// Returns `None` if the initial factorization fails (e.g. a
    /// structurally singular KKT system).
    pub fn build<F>(base: &QpProblem, opts: &QpOptions, mut make_backend: F) -> Option<Self>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        let expanded = if base.has_bounds() {
            expand_bounds(base).0
        } else {
            base.clone()
        };
        let cone = CompositeCone::single_nonneg(expanded.m_ineq());
        let (kkt, fact) = build_factorization(&expanded, &cone, opts, &mut make_backend).ok()?;
        Some(QpFactorization {
            airn: kkt.airn,
            ajcn: kkt.ajcn,
            n: base.n,
            fact,
            cone,
            opts: *opts,
        })
    }

    /// Solve `prob`, reusing the captured symbolic factor. `prob` must
    /// share the captured structure (see the type docs); otherwise a
    /// `NumericalFailure` solution is returned.
    pub fn solve(&mut self, prob: &QpProblem) -> QpSolution {
        self.solve_inner(prob, None)
    }

    /// Solve `prob` reusing the captured symbolic factor **and** warm
    /// starting from `warm` (a nearby problem's solution). Combines the
    /// two reuse axes: the symbolic factorization is paid once at `build`,
    /// and the interior-point iteration is seeded from the warm point (see
    /// [`QpWarmStart`]). Same structure requirement as [`Self::solve`].
    pub fn solve_warm(&mut self, prob: &QpProblem, warm: &QpWarmStart) -> QpSolution {
        let (expanded_z, _) = if prob.has_bounds() {
            // `merge_bound_duals` needs the bound-row provenance.
            let (_, bound_rows) = expand_bounds(prob);
            (merge_bound_duals(prob, &bound_rows, warm), ())
        } else {
            (warm.z.clone(), ())
        };
        let w = WarmStart {
            x: warm.x.clone(),
            y: warm.y.clone(),
            z: expanded_z,
        };
        self.solve_inner(prob, Some(&w))
    }

    fn solve_inner(&mut self, prob: &QpProblem, warm: Option<&WarmStart>) -> QpSolution {
        let (expanded, bound_rows) = if prob.has_bounds() {
            expand_bounds(prob)
        } else {
            (prob.clone(), Vec::new())
        };
        // Rebuild this instance's pattern and require it to match the
        // captured one exactly (same nnz, same row/col indices).
        let kkt = KktStructure::build(&expanded, &self.cone, self.opts.reg);
        if prob.n != self.n || kkt.airn != self.airn || kkt.ajcn != self.ajcn {
            return failed_solution(
                prob,
                vec![0.0; prob.n],
                vec![0.0; prob.m_eq()],
                vec![1.0; prob.m_ineq()],
                0,
            );
        }
        // Reuse the live factorization (it carries the symbolic analysis;
        // `run_ipm` refactors numerically per iteration). The same factor
        // object is reused across solves, so the AMD ordering / symbolic
        // factor is paid once at `build`.
        let sol = run_ipm(
            &expanded,
            &self.cone,
            &self.opts,
            &kkt,
            &mut self.fact,
            warm,
        );
        split_bound_duals(prob, &bound_rows, sol)
    }
}

/// Whether the cone specs partition exactly `m_ineq` inequality rows — the
/// invariant the conic drivers assume (each `s = h − Gx` block sits in one
/// cone, with an exp/power cone occupying exactly 3 rows). A mismatch is a
/// caller error that would otherwise index past the slack vector.
fn cone_dims_cover(cones: &[ConeSpec], m_ineq: usize) -> bool {
    cones.iter().map(|c| c.dim()).sum::<usize>() == m_ineq
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
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj,
        iters,
        iterates: Vec::new(),
    }
}

/// Build the Newton RHS `[−r_d; −r_p; −r_g + r_c ⊘ z]` for a given
/// complementarity residual `r_c` (predictor or corrector).
#[allow(clippy::too_many_arguments)]
/// Assemble the reduced KKT right-hand side `[-r_d; -r_p; -r_g + comp_term]`.
/// `comp_term` is the cone's contribution at the `(z)` rows (the orthant's
/// is `r_c ⊘ z`), computed by the caller via [`Cone::rhs_comp_term`] so the
/// block is cone-specific rather than baked in here.
pub(crate) fn build_rhs(
    r_d: &[f64],
    r_p: &[f64],
    r_g: &[f64],
    comp_term: &[f64],
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
        rhs[n + m_eq + i] = -r_g[i] + comp_term[i];
    }
    // Auxiliary-variable rows (per second-order cone, appended after the
    // base rows) have zero right-hand side; re-zero them since `solve_one`
    // overwrote the buffer with the previous step.
    for v in rhs.iter_mut().skip(n + m_eq + m_ineq) {
        *v = 0.0;
    }
}

/// Copy the solved RHS into the (dx, dy, dz) step components.
pub(crate) fn split_step(
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
    cone: &CompositeCone,
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
    let cone = CompositeCone::single_nonneg(prob.m_ineq());
    let kkt = KktStructure::build(prob, &cone, reg);
    let mut vals = kkt.values.clone();
    // Orthant block s/z = scaling at z = 1.
    let ones = vec![1.0; prob.m_ineq()];
    kkt.update_blocks(&cone, scaling, &ones, reg, &mut vals);
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
/// Value-array positions of one cone's `(z, z)` scaling block, aligned with
/// the cone's [`CompositeCone::blocks`] order.
enum ZBlockPos {
    /// One value position per row (orthant diagonal).
    Diagonal(Vec<usize>),
    /// A second-order cone in **diagonal + rank-1** form, represented with
    /// one auxiliary variable `ξ`: the `(z,z)` diagonal entries, the
    /// coupling column `(z_i, ξ) = u_i`, and the `(ξ,ξ) = +1` entry. Its
    /// Schur complement reproduces the dense block `diag(d) + uuᵀ`, keeping
    /// the factorization sparse (ECOS/Clarabel sparse-SOC trick).
    DiagRank1 {
        diag_pos: Vec<usize>,
        u_pos: Vec<usize>,
        aux_pos: usize,
    },
    /// A fully dense symmetric block (the PSD cone's `W ⊗ₛ W`): the
    /// value-array positions of its lower triangle, row-major
    /// `[(0,0),(1,0),(1,1),…]`, aligned with [`ConeBlock::DenseLower`].
    Dense { pos: Vec<usize> },
}

/// How a cone block enters the `(z,z)` position of the KKT system.
#[derive(Clone, Copy, PartialEq)]
enum BlockShape {
    /// Orthant: one diagonal entry per row.
    Diagonal,
    /// Second-order cone: diagonal + rank-1 via an auxiliary variable.
    DiagRank1,
    /// PSD cone: a fully dense symmetric lower-triangle block.
    Dense,
}

pub(crate) struct KktStructure {
    pub(crate) airn: Vec<Index>,
    pub(crate) ajcn: Vec<Index>,
    /// Constant values (everything except the scaling block; the `(z, z)`
    /// diagonal entries hold their `-reg` term here).
    pub(crate) values: Vec<Number>,
    /// Total KKT dimension, including the per-SOC auxiliary variables.
    pub(crate) dim: usize,
    /// Per-cone `(z, z)` block positions, in `cone.blocks()` order.
    z_blocks: Vec<ZBlockPos>,
}

impl KktStructure {
    /// Build the pattern and constant values once for `prob`'s inequality
    /// cone `cone`. Each cone block contributes either a diagonal entry per
    /// row (orthant) or a dense lower-triangle block (SOC) at its `(z, z)`
    /// position; all seeded with `-reg` on the diagonal. The pattern is
    /// constant across iterations — only the scaling values change — so the
    /// solver `refactor`s rather than re-analyzing.
    pub(crate) fn build(prob: &QpProblem, cone: &CompositeCone, reg: f64) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
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
        // (z,x): G.
        for t in &prob.g {
            add(n + m_eq + t.row, t.col, t.val);
        }
        // (z,z): per cone block, seeded with −δI. SOC blocks get an
        // auxiliary variable (appended after the base rows) carrying the
        // rank-1 term. The scaling values are written by `update_blocks`.
        let base_dim = n + m_eq + prob.m_ineq();
        let shapes = block_shapes(cone);
        let mut aux = base_dim; // next auxiliary-variable index
        for ((off, k), shape) in cone.blocks().iter().zip(&shapes) {
            let d = k.dim();
            let zbase = n + m_eq + off;
            for i in 0..d {
                add(zbase + i, zbase + i, -reg); // diagonal (filled per iter)
            }
            match shape {
                BlockShape::Diagonal => {}
                BlockShape::DiagRank1 => {
                    // Aux: coupling (z_i, ξ) = u_i and (ξ, ξ) = +1.
                    for i in 0..d {
                        add(aux, zbase + i, 0.0);
                    }
                    add(aux, aux, 1.0);
                    aux += 1;
                }
                BlockShape::Dense => {
                    // Reserve the strict lower triangle of the (z,z) block;
                    // the diagonal was already added above.
                    for i in 0..d {
                        for j in 0..i {
                            add(zbase + i, zbase + j, 0.0);
                        }
                    }
                }
            }
        }
        let dim = aux;

        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut coord_to_pos: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (pos, ((r, c), v)) in entries.into_iter().enumerate() {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
            coord_to_pos.insert((r, c), pos);
        }

        // Record each cone block's positions in `blocks()` order.
        let mut z_blocks = Vec::with_capacity(cone.blocks().len());
        let mut aux = base_dim;
        for ((off, k), shape) in cone.blocks().iter().zip(&shapes) {
            let d = k.dim();
            let zbase = n + m_eq + off;
            match shape {
                BlockShape::Diagonal => {
                    let diag_pos = (0..d)
                        .map(|i| coord_to_pos[&(zbase + i, zbase + i)])
                        .collect();
                    z_blocks.push(ZBlockPos::Diagonal(diag_pos));
                }
                BlockShape::DiagRank1 => {
                    let diag_pos = (0..d)
                        .map(|i| coord_to_pos[&(zbase + i, zbase + i)])
                        .collect();
                    let u_pos = (0..d).map(|i| coord_to_pos[&(aux, zbase + i)]).collect();
                    let aux_pos = coord_to_pos[&(aux, aux)];
                    z_blocks.push(ZBlockPos::DiagRank1 {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    });
                    aux += 1;
                }
                BlockShape::Dense => {
                    // Lower triangle, row-major — matching ConeBlock::DenseLower.
                    let mut pos = Vec::with_capacity(d * (d + 1) / 2);
                    for i in 0..d {
                        for j in 0..=i {
                            pos.push(coord_to_pos[&(zbase + i, zbase + j)]);
                        }
                    }
                    z_blocks.push(ZBlockPos::Dense { pos });
                }
            }
        }

        KktStructure {
            airn,
            ajcn,
            values,
            dim,
            z_blocks,
        }
    }

    /// Write the per-iteration cone scaling into `out` (a copy of
    /// `self.values`): each block's `(z, z)` entries become `-(block) -
    /// reg·I`, from the cone's [`Cone::kkt_block`].
    pub(crate) fn update_blocks(
        &self,
        cone: &CompositeCone,
        s: &[f64],
        z: &[f64],
        reg: f64,
        out: &mut [Number],
    ) {
        for ((off, k), zb) in cone.blocks().iter().zip(&self.z_blocks) {
            let d = k.dim();
            let block = k.kkt_block(&s[*off..off + d], &z[*off..off + d]);
            match (zb, block) {
                (ZBlockPos::Diagonal(pos), ConeBlock::Diagonal(vals)) => {
                    for (i, &p) in pos.iter().enumerate() {
                        out[p] = -vals[i] - reg;
                    }
                }
                (
                    ZBlockPos::DiagRank1 {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    },
                    ConeBlock::DiagPlusRank1 { diag, u },
                ) => {
                    // (z,z) block = −(diag(d) + uuᵀ) − reg, with the rank-1
                    // carried by the aux variable ξ: diagonal −dᵢ − reg, the
                    // coupling (z_i, ξ) = uᵢ, and (ξ, ξ) = +1. Its Schur
                    // complement is −diag(d) − reg − uuᵀ = −(W²) − reg.
                    for i in 0..d {
                        out[diag_pos[i]] = -diag[i] - reg;
                        out[u_pos[i]] = u[i];
                    }
                    out[*aux_pos] = 1.0;
                }
                (ZBlockPos::Dense { pos }, ConeBlock::DenseLower { dim: _, lower }) => {
                    // (z,z) block = −H − reg·I, H = W⊗ₛW dense. Lower triangle
                    // row-major; reg only on the diagonal (i == j).
                    let mut idx = 0;
                    for i in 0..d {
                        for j in 0..=i {
                            out[pos[idx]] = -lower[idx] - if i == j { reg } else { 0.0 };
                            idx += 1;
                        }
                    }
                }
                _ => unreachable!("cone block shape changed between build and update"),
            }
        }
    }
}

/// How each cone block enters the `(z,z)` position — diagonal (orthant),
/// diag-plus-rank-1 (SOC), or fully dense (PSD) — probed via `kkt_block` at
/// the cone identity.
fn block_shapes(cone: &CompositeCone) -> Vec<BlockShape> {
    cone.blocks()
        .iter()
        .map(|(_, k)| {
            let d = k.dim();
            let mut e = vec![0.0; d];
            k.identity(&mut e);
            match k.kkt_block(&e, &e) {
                ConeBlock::Diagonal(_) => BlockShape::Diagonal,
                ConeBlock::DiagPlusRank1 { .. } => BlockShape::DiagRank1,
                ConeBlock::DenseLower { .. } => BlockShape::Dense,
            }
        })
        .collect()
}

pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
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
pub(crate) fn detect_infeasibility(
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
