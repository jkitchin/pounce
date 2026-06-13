//! LP crossover: purify a near-optimal interior-point iterate to an *exact*
//! optimal vertex using the active-set QP engine ([`pounce_qp`]).
//!
//! # Why
//!
//! A pure interior-point method cannot certify a **degenerate** LP vertex to
//! `tol`: where strict complementarity fails the fraction-to-boundary step
//! collapses (`α → 1e-4`), `μ` freezes, and the primal residual plateaus above
//! the tolerance, so the solve grinds to its iteration cap with the *correct*
//! objective but no convergence certificate (NETLIB GEN family). This is a
//! fundamental limit — it is exactly why every commercial LP solver pairs an
//! IPM with a **crossover** that pivots the near-optimal interior point to an
//! exact optimal vertex basis (Andersen & Ye 1996, "Combining interior-point
//! and pivoting algorithms for linear programming"; Megiddo 1991).
//!
//! # How
//!
//! pounce already has a production, fully-autonomous active-set QP engine
//! ([`pounce_qp::ParametricActiveSetSolver`]) with an internal add/drop pivot
//! loop and Bland anti-cycling. Crossover is therefore a *bridge*, not a new
//! solver: translate the convex standard-form LP into pounce-qp's two-sided
//! form, seed the working set from the interior iterate's active set, let the
//! engine purify to the exact vertex, then sign-transform the multipliers back.
//!
//! # Representation bridge
//!
//! Convex standard form (this crate, [`crate::qp::QpProblem`]):
//! `min ½xᵀPx + cᵀx  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub`, with equality dual
//! `y` (free), inequality dual `z ≥ 0`, and bound duals `z_lb, z_ub ≥ 0`;
//! stationarity `Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0`.
//!
//! pounce-qp form ([`pounce_qp::QpProblem`]): `min ½xᵀHx + gᵀx`
//! `s.t.  bl ≤ A_qp x ≤ bu,  xl ≤ x ≤ xu`, Lagrangian
//! `L = ½xᵀHx + gᵀx + λ_gᵀ(A_qp x − β) + λ_satᵀ(x − β_bnd)` ⇒ stationarity
//! `Hx + g + A_qpᵀλ_g + λ_sat = 0`, with the user-facing bound multiplier
//! `lambda_x = z_l − z_u = −λ_sat` (`solver.rs` §5/§6 convention).
//!
//! Mapping `A_qp = [A_eq ; G]`, `g = c`, `H = P`:
//! - **eq rows** `bl = bu = b`            → `y[k]   = lambda_g[k]`
//! - **ineq rows** `bl = −∞, bu = h`      → `z[i]   = lambda_g[m_eq + i]`
//!   (`≥ 0` at the optimum: an active `≤` row is `AtUpper`, whose `lambda_g`
//!   the drop test in `solver.rs:810` keeps non-negative — *no* sign flip)
//! - **native variable bounds** `xl=lb, xu=ub` → `z_lb[i] = max(0,  lambda_x[i])`,
//!   `z_ub[i] = max(0, −lambda_x[i])` (from `−z_lb + z_ub = −lambda_x`).
//!
//! Mapping the convex variable bounds to pounce-qp's *native* `xl/xu` (rather
//! than expanding them into general rows) keeps the bridge correct for library
//! callers that carry bounds in `lb/ub`; the CLI `.nl` path already expands
//! bounds into `G` rows, so for the GEN target `lb/ub` are empty and the bound
//! mapping is a no-op.
//!
//! # Never-regress
//!
//! Crossover is a strict refinement: the purified vertex replaces the interior
//! iterate only when it is at least as good a KKT point of the *original*
//! problem (its `kkt_error()` does not exceed the interior iterate's, within a
//! tiny slack). Because an LP KKT point is globally optimal, a pounce-qp
//! `Optimal` vertex has the correct objective by construction; the KKT-error
//! gate is what guarantees we never return something worse — on any
//! `Err`/non-`Optimal`/regressing outcome we return the original solution
//! unchanged.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{
    AntiCyclingChoice, ConsStatus, HessianInertia, ParametricActiveSetSolver,
    QpOptions as ActiveSetOptions, QpProblem as ActiveSetProblem, QpSolver,
    QpStatus as ActiveSetStatus, WorkingSet,
};

use crate::ipm::QpOptions;
use crate::qp::{QpProblem, QpSolution, QpStatus};

/// Inequality dual above which the IPM iterate is taken to mark a row active
/// (warm-start hint only; the active-set engine pivots to fix any over- or
/// under-identification, so this threshold need not be sharp).
const ACTIVE_Z_TOL: f64 = 1e-7;
/// Slack below which the IPM iterate is taken to mark a row binding.
const ACTIVE_S_TOL: f64 = 1e-7;
/// Slack on the never-regress KKT-error comparison.
const REGRESS_SLACK: f64 = 1e-9;
/// Iteration cap for the purifying active-set solve (generous; the pivot loop
/// terminates well inside this on the LPs crossover targets).
const CROSSOVER_MAX_ITER: u32 = 1000;

/// Clamp a convex lower-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_lower(lb: f64) -> f64 {
    if lb <= NLP_LOWER_BOUND_INF {
        NLP_LOWER_BOUND_INF
    } else {
        lb
    }
}

/// Clamp a convex upper-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_upper(ub: f64) -> f64 {
    if ub >= NLP_UPPER_BOUND_INF {
        NLP_UPPER_BOUND_INF
    } else {
        ub
    }
}

/// Run the LP-crossover phase. Returns a purified exact-vertex solution when it
/// is a strict improvement (never-regress), otherwise the original `sol`.
///
/// `make_backend` supplies the sparse symmetric linear-solver backend for the
/// active-set engine — the same factory the IPM uses, so crossover inherits the
/// caller's solver choice (FERAL in tree).
pub fn maybe_crossover<F>(
    prob: &QpProblem,
    sol: QpSolution,
    opts: &QpOptions,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // ---- Gate: pure LP, plausibly-optimal status, at least one constraint ----
    if !opts.crossover || !prob.p_lower.is_empty() {
        return sol;
    }
    if !matches!(
        sol.status,
        QpStatus::Optimal | QpStatus::OptimalInaccurate | QpStatus::IterationLimit
    ) {
        return sol;
    }
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let m = m_eq + m_ineq;
    if m == 0 {
        // No constraints: the IPM already returns the (bound-clamped)
        // unconstrained optimum; there is no vertex to pivot to.
        return sol;
    }

    // ---- Primary path: revised-simplex LU-basis crossover ----
    // The architecturally-correct engine ([`crate::simplex`]) pivots one
    // variable at a time on an unsymmetric LU basis, walking straight through
    // degeneracy with Bland's rule — so it resolves the highly-degenerate NETLIB
    // GEN vertices (issue #133) that the active-set bridge below stalls on. On
    // any breakdown it returns `None` and we fall through to the bridge.
    if let Some(v) = crate::simplex::crossover_simplex(prob, &sol, opts) {
        let candidate = QpSolution {
            status: QpStatus::Optimal,
            x: v.x,
            y: v.y,
            z: v.z,
            z_lb: v.z_lb,
            z_ub: v.z_ub,
            obj: v.obj,
            iters: sol.iters,
            iterates: sol.iterates.clone(),
        };
        let cand_err = candidate.kkt_residuals(prob).kkt_error();
        let orig_err = sol.kkt_residuals(prob).kkt_error();
        if cand_err.is_finite() && cand_err <= orig_err.max(0.0) + REGRESS_SLACK {
            return candidate;
        }
    }

    // ---- Translate to pounce-qp form (owned locals outlive the borrow) ----
    // Hessian: pure LP ⇒ empty symmetric triplet of dimension n.
    let h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, Vec::new(), Vec::new()));

    // Jacobian A_qp = [A_eq ; G], 1-based indices.
    let nnz = prob.a.len() + prob.g.len();
    let mut irows = Vec::with_capacity(nnz);
    let mut jcols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for t in &prob.a {
        irows.push((t.row + 1) as i32);
        jcols.push((t.col + 1) as i32);
        vals.push(t.val);
    }
    for t in &prob.g {
        irows.push((m_eq + t.row + 1) as i32);
        jcols.push((t.col + 1) as i32);
        vals.push(t.val);
    }
    let mut a_qp = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, irows, jcols));
    a_qp.set_values(&vals);

    // Row bounds: eq rows bl=bu=b; ineq rows bl=−∞, bu=h.
    let mut bl = Vec::with_capacity(m);
    let mut bu = Vec::with_capacity(m);
    for &bk in &prob.b {
        bl.push(bk);
        bu.push(bk);
    }
    for &hi in &prob.h {
        bl.push(NLP_LOWER_BOUND_INF);
        bu.push(to_qp_upper(hi));
    }

    // Native variable bounds (no-op when the convex problem has none).
    let mut xl = Vec::with_capacity(n);
    let mut xu = Vec::with_capacity(n);
    for i in 0..n {
        xl.push(to_qp_lower(prob.lb_of(i)));
        xu.push(to_qp_upper(prob.ub_of(i)));
    }

    let g_lin = prob.c.clone();
    let qp = ActiveSetProblem {
        n,
        m,
        h: &h,
        g: &g_lin,
        a: &a_qp,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    // ---- Working-set warm-start hint from the interior iterate ----
    // slacks s_i = h_i − (Gx)_i; a row is hinted active when its IPM dual is
    // positive and its slack is ~0. Equality rows are always active.
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(&sol.x, &mut gx);
    let mut constraints = Vec::with_capacity(m);
    for _ in 0..m_eq {
        constraints.push(ConsStatus::Equality);
    }
    for i in 0..m_ineq {
        let slack = prob.h[i] - gx[i];
        if sol.z[i] > ACTIVE_Z_TOL && slack.abs() <= ACTIVE_S_TOL {
            constraints.push(ConsStatus::AtUpper);
        } else {
            constraints.push(ConsStatus::Inactive);
        }
    }
    // Bounds left Inactive in the hint: the engine pivots them in as needed.
    let working = WorkingSet {
        bounds: vec![pounce_qp::BoundStatus::Inactive; n],
        constraints,
    };

    // ---- Solve (Bland keeps the degenerate pivot loop finite) ----
    let qopts = ActiveSetOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: CROSSOVER_MAX_ITER,
        ..ActiveSetOptions::default()
    };
    // Warm-start hint first: `solve_with_working_set` factorizes the hinted
    // active set up front to recover a primal, which is fast when the hint is
    // good. But at a *degenerate* optimum the IPM's active set has linearly-
    // dependent binding rows (more than `n` of them — the very degeneracy
    // crossover exists to resolve), so that initial factorization can be
    // singular and the §4.5 inertia control cannot rescue a rank-deficient
    // constraint block. Fall back to a fresh COLD solve, whose
    // `cold_general_initial` only factorizes the (smaller) equality block and
    // routes an infeasible start through l1-elastic recovery — robust to the
    // redundant rows that defeat the warm path.
    let mut warm_solver = ParametricActiveSetSolver::new(make_backend());
    let warm = warm_solver.solve_with_working_set(&qp, &working, &qopts);
    let qsol = match warm {
        Ok(q) if q.status == ActiveSetStatus::Optimal => q,
        _ => {
            let mut cold_solver = ParametricActiveSetSolver::new(make_backend());
            let cold = cold_solver.solve(&qp, None, &qopts);
            match cold {
                Ok(q) if q.status == ActiveSetStatus::Optimal => q,
                _ => return sol,
            }
        }
    };

    // ---- Back-translate (sign transform — see module docs) ----
    let mut y = vec![0.0; m_eq];
    y.copy_from_slice(&qsol.lambda_g[..m_eq]);
    let mut z = vec![0.0; m_ineq];
    for i in 0..m_ineq {
        z[i] = qsol.lambda_g[m_eq + i].max(0.0);
    }
    let mut z_lb = vec![0.0; n];
    let mut z_ub = vec![0.0; n];
    for i in 0..n {
        z_lb[i] = qsol.lambda_x[i].max(0.0);
        z_ub[i] = (-qsol.lambda_x[i]).max(0.0);
    }
    // Objective in convex coordinates (½xᵀPx + cᵀx; the Px term is 0 for an LP
    // but evaluated generally so the recomputation can't silently drift).
    let mut px = vec![0.0; n];
    prob.p_mul(&qsol.x, &mut px);
    let obj = (0..n).map(|i| (0.5 * px[i] + prob.c[i]) * qsol.x[i]).sum();

    let candidate = QpSolution {
        status: QpStatus::Optimal,
        x: qsol.x,
        y,
        z,
        z_lb,
        z_ub,
        obj,
        iters: sol.iters,
        iterates: sol.iterates.clone(),
    };

    // ---- Never-regress ----
    // The candidate is an exact KKT point of the LP (pounce-qp `Optimal`), so by
    // LP convexity its objective is globally optimal. The KKT-error gate is the
    // robust realization of "never return something worse": if the purified
    // vertex's residuals against the ORIGINAL problem exceed the interior
    // iterate's (a sign translation / feasibility surprise), keep the original.
    let cand_err = candidate.kkt_residuals(prob).kkt_error();
    let orig_err = sol.kkt_residuals(prob).kkt_error();
    if cand_err.is_finite() && cand_err <= orig_err.max(0.0) + REGRESS_SLACK {
        candidate
    } else {
        sol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::{solve_qp_ipm, QpOptions};
    use crate::qp::{QpProblem, Triplet};
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// LP used across the sign/round-trip tests:
    ///   min −2x₁ − x₂
    ///   s.t. x₁+x₂≤6 (r0), x₁≤4 (r1), x₂≤4 (r2), −x₁≤0 (r3), −x₂≤0 (r4)
    /// Unique optimum x* = (4, 2), f* = −10. Rows 0 and 1 bind. Stationarity
    /// c + z₀(1,1) + z₁(1,0) = 0 ⇒ z₀ = z₁ = 1 (pins the dual sign exactly).
    fn unique_vertex_lp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(2, 1, 1.0),
                Triplet::new(3, 0, -1.0),
                Triplet::new(4, 1, -1.0),
            ],
            h: vec![6.0, 4.0, 4.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        }
    }

    fn opts_on() -> QpOptions {
        // Crossover is off by default (opt-in); these tests exercise the
        // crossover path, so enable it explicitly.
        QpOptions {
            crossover: true,
            ..QpOptions::default()
        }
    }

    fn opts_off() -> QpOptions {
        QpOptions {
            crossover: false,
            ..QpOptions::default()
        }
    }

    /// Translation round-trip and the **sign transform** (R3): purifying the
    /// interior iterate lands the exact vertex with vanishing KKT error,
    /// `z ≥ 0`, and the analytically-pinned duals `z₀ = z₁ = 1`. A flipped
    /// sign would clamp these duals to 0, blow up the stationarity residual,
    /// and the never-regress gate would return the (non-vertex) interior point
    /// instead — so a pass here proves the sign is right.
    #[test]
    fn sign_transform_and_round_trip() {
        let prob = unique_vertex_lp();
        // Interior iterate (crossover disabled), then purify explicitly.
        let interior = solve_qp_ipm(&prob, &opts_off(), backend);
        let mut mk = backend;
        let cand = maybe_crossover(&prob, interior, &opts_on(), &mut mk);

        assert_eq!(cand.status, QpStatus::Optimal);
        assert!((cand.x[0] - 4.0).abs() < 1e-8, "x0 = {}", cand.x[0]);
        assert!((cand.x[1] - 2.0).abs() < 1e-8, "x1 = {}", cand.x[1]);
        assert!((cand.obj + 10.0).abs() < 1e-8, "obj = {}", cand.obj);
        assert!(
            cand.z.iter().all(|&zi| zi >= -1e-12),
            "z must be ≥ 0: {:?}",
            cand.z
        );
        assert!((cand.z[0] - 1.0).abs() < 1e-7, "z0 = {} (sign!)", cand.z[0]);
        assert!((cand.z[1] - 1.0).abs() < 1e-7, "z1 = {} (sign!)", cand.z[1]);
        for i in 2..5 {
            assert!(cand.z[i].abs() < 1e-7, "z{i} = {} should be 0", cand.z[i]);
        }
        assert!(
            cand.kkt_residuals(&prob).kkt_error() < 1e-9,
            "kkt_error = {}",
            cand.kkt_residuals(&prob).kkt_error()
        );
    }

    /// Degenerate LP: the optimum x* = (4, 2) is pinned by THREE binding rows
    /// (more than the 2 variables) — the loss-of-unique-basis case that defeats
    /// a pure IPM. Crossover must still reach the exact vertex.
    #[test]
    fn degenerate_lp_purifies_to_exact_vertex() {
        let mut prob = unique_vertex_lp();
        // Add 2x₁ + x₂ ≤ 10, binding & redundant at (4, 2).
        prob.g.push(Triplet::new(5, 0, 2.0));
        prob.g.push(Triplet::new(5, 1, 1.0));
        prob.h.push(10.0);

        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 4.0).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 2.0).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 10.0).abs() < 1e-8, "obj = {}", sol.obj);
        assert!(
            sol.z.iter().all(|&zi| zi >= -1e-12),
            "z must be ≥ 0: {:?}",
            sol.z
        );
        assert!(
            sol.kkt_residuals(&prob).kkt_error() < 1e-9,
            "kkt_error = {}",
            sol.kkt_residuals(&prob).kkt_error()
        );
    }

    /// Never-regress: crossover on is never *worse* than crossover off — the
    /// purified result's KKT error and objective do not regress. Checked on the
    /// degenerate LP, where crossover is supposed to help most.
    #[test]
    fn never_regress_vs_crossover_off() {
        let mut prob = unique_vertex_lp();
        prob.g.push(Triplet::new(5, 0, 2.0));
        prob.g.push(Triplet::new(5, 1, 1.0));
        prob.h.push(10.0);

        let off = solve_qp_ipm(&prob, &opts_off(), backend);
        let on = solve_qp_ipm(&prob, &opts_on(), backend);
        let off_err = off.kkt_residuals(&prob).kkt_error();
        let on_err = on.kkt_residuals(&prob).kkt_error();
        assert!(
            on_err <= off_err + REGRESS_SLACK,
            "crossover regressed KKT error: on {on_err} > off {off_err}"
        );
        // Objective must not get worse (this is a minimization).
        assert!(
            on.obj <= off.obj + 1e-7 * (1.0 + off.obj.abs()),
            "crossover regressed objective: on {} > off {}",
            on.obj,
            off.obj
        );
    }

    /// Gate: a genuine QP (`P ≠ 0`) is left untouched — crossover is a no-op,
    /// so the interior optimum (which is generally NOT a vertex) is returned as
    /// is. `min ½‖x‖² s.t. x₁+x₂ ≥ 1` ⇒ x* = (0.5, 0.5).
    #[test]
    fn gate_skips_genuine_qp() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            // −x₁ − x₂ ≤ −1  (i.e. x₁ + x₂ ≥ 1)
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 0.5).abs() < 1e-6, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 0.5).abs() < 1e-6, "x1 = {}", sol.x[1]);
    }

    /// Crossover handles native variable bounds (`lb/ub`) correctly: the bound
    /// duals come back through `lambda_x`. `min −x₁ − x₂ s.t. x₁+x₂≤3, 0≤x≤2`
    /// ⇒ x* = (1, 2) or (2, 1) — pick the unique vertex with c = (−2, −1):
    /// x* = (2, 1) (x₁ at its upper bound, x₁+x₂≤3 binding), f* = −5.
    #[test]
    fn handles_native_variable_bounds() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![3.0],
            lb: vec![0.0, 0.0],
            ub: vec![2.0, 2.0],
        };
        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 5.0).abs() < 1e-8, "obj = {}", sol.obj);
        assert!(
            sol.kkt_residuals(&prob).kkt_error() < 1e-8,
            "kkt_error = {}",
            sol.kkt_residuals(&prob).kkt_error()
        );
    }
}
