//! Ruiz equilibration for the convex LP/QP interior-point method.
//!
//! The direct primal–dual IPM ([`crate::ipm::solve_qp_ipm`]) factorizes the
//! KKT system of the **raw** problem data. On a badly-scaled LP/QP — large
//! dynamic range across the rows of `A`/`G`, the columns (variables), or the
//! objective — that system is ill-conditioned, the Newton steps are wild, the
//! iterates blow up, and the cone-scaling block `S⁻¹Z` eventually drives the
//! KKT matrix singular, surfacing as a `NumericalFailure`. (The NLP solver and
//! Ipopt/MA57 avoid this because they equilibrate the problem first.)
//!
//! This module supplies the missing piece for the orthant (LP/QP) path: a few
//! sweeps of **Ruiz scaling** on the symmetric augmented matrix
//!
//! ```text
//!     K = | P   Aᵀ  Gᵀ |
//!         | A   0   0  |
//!         | G   0   0  |
//! ```
//!
//! followed by a scalar **cost scaling** σ that brings the objective gradient
//! to O(1). Each Ruiz sweep rescales every row/column of `K` by the inverse
//! square root of its current ∞-norm; because `K` is symmetric the row and
//! column scalings coincide, yielding one scale vector split into a per-column
//! (variable) scaling `Dc`, per-equality-row `R_A`, and per-inequality-row
//! `R_G`.
//!
//! Equilibration is a *change of variables*, so the recovered optimum is the
//! same KKT point — only the conditioning of the iteration changes. The
//! substitution is `x = Dc x̂`, giving the scaled data
//!
//! ```text
//!   P̂ = σ·Dc P Dc,   ĉ = σ·Dc c,
//!   Â = R_A A Dc,     b̂ = R_A b,
//!   Ĝ = R_G G Dc,     ĥ = R_G h,
//!   lb̂ = Dc⁻¹ lb,     ûb = Dc⁻¹ ub,
//! ```
//!
//! and the dual unscaling (derived in [`Scaling::unscale_solution`])
//!
//! ```text
//!   x   = Dc x̂,                 y    = R_A ŷ / σ,        z = R_G ẑ / σ,
//!   z_lb = ẑ_lb /(σ·Dc),        z_ub = ẑ_ub /(σ·Dc).
//! ```
//!
//! **Scope.** This is valid only for the **nonnegative orthant** (the LP/QP
//! inequalities and the expanded variable bounds): per-row scaling of `G`
//! preserves `z ≥ 0`. It must NOT be applied to second-order / exponential /
//! power cones, whose rows must scale uniformly to preserve the cone — hence
//! it is wired only into [`crate::ipm::solve_qp_ipm`] and skipped under the
//! HSDE/conic drivers.

use crate::qp::{QpProblem, QpSolution, Triplet, BOUND_INF, NEG_INF, POS_INF};
use crate::QpWarmStart;

/// Number of Ruiz sweeps. Ruiz converges geometrically; a handful of passes
/// brings the row/column ∞-norms to within a few percent of 1, which is all
/// the conditioning improvement the IPM needs. More passes cost
/// `O(nnz)` each for negligible further gain.
const RUIZ_SWEEPS: usize = 10;

/// Clamp on the scalar cost-scaling factor σ, so a degenerate objective
/// (tiny or huge gradient) cannot itself create an extreme scaling.
const SIGMA_LO: f64 = 1e-8;
const SIGMA_HI: f64 = 1e8;

/// The diagonal scaling recovered by [`equilibrate`], retained so a scaled
/// solution can be mapped back to the original problem's variables and duals.
pub(crate) struct Scaling {
    /// Per-variable (column) scaling `Dc`; `x = Dc x̂`.
    dcol: Vec<f64>,
    /// Per-equality-row scaling `R_A`.
    drow_a: Vec<f64>,
    /// Per-inequality-row scaling `R_G`.
    drow_g: Vec<f64>,
    /// Scalar objective (cost) scaling σ > 0.
    sigma: f64,
}

/// Ruiz-equilibrate `prob`, returning the scaled problem and the [`Scaling`]
/// needed to undo it. The scaled problem has the same dimensions, sparsity
/// pattern, and bound structure as the original; only the numeric data is
/// rescaled. A solution of the scaled problem maps back via
/// [`Scaling::unscale_solution`].
pub(crate) fn equilibrate(prob: &QpProblem) -> (QpProblem, Scaling) {
    let n = prob.n;
    let me = prob.m_eq();
    let mi = prob.m_ineq();
    let dim = n + me + mi;

    // Cumulative symmetric scaling for each row/column of the augmented K.
    // Index layout: [0, n) variables, [n, n+me) equality rows,
    // [n+me, n+me+mi) inequality rows.
    let mut s = vec![1.0f64; dim];
    let mut rownorm = vec![0.0f64; dim];

    for _ in 0..RUIZ_SWEEPS {
        rownorm.iter_mut().for_each(|v| *v = 0.0);
        // P (lower triangle): symmetric var–var entries.
        for t in &prob.p_lower {
            let v = (s[t.row] * t.val * s[t.col]).abs();
            if v > rownorm[t.row] {
                rownorm[t.row] = v;
            }
            if t.row != t.col && v > rownorm[t.col] {
                rownorm[t.col] = v;
            }
        }
        // A entry (r, c) sits at K(n+r, c) and its transpose K(c, n+r).
        for t in &prob.a {
            let (ri, ci) = (n + t.row, t.col);
            let v = (s[ri] * t.val * s[ci]).abs();
            if v > rownorm[ri] {
                rownorm[ri] = v;
            }
            if v > rownorm[ci] {
                rownorm[ci] = v;
            }
        }
        // G entry (r, c) sits at K(n+me+r, c) and its transpose.
        for t in &prob.g {
            let (ri, ci) = (n + me + t.row, t.col);
            let v = (s[ri] * t.val * s[ci]).abs();
            if v > rownorm[ri] {
                rownorm[ri] = v;
            }
            if v > rownorm[ci] {
                rownorm[ci] = v;
            }
        }
        // Ruiz update: s_i ← s_i / sqrt(‖row_i‖∞). An all-zero row (e.g. an
        // empty column) is left unscaled.
        for i in 0..dim {
            if rownorm[i] > 0.0 {
                s[i] /= rownorm[i].sqrt();
            }
        }
    }

    let dcol = s[..n].to_vec();
    let drow_a = s[n..n + me].to_vec();
    let drow_g = s[n + me..].to_vec();

    // Apply the column/row scalings to the data: P̂₀ = Dc P Dc, ĉ₀ = Dc c,
    // Â = R_A A Dc, b̂ = R_A b, Ĝ = R_G G Dc, ĥ = R_G h.
    let mut p_lower: Vec<Triplet> = prob
        .p_lower
        .iter()
        .map(|t| Triplet::new(t.row, t.col, t.val * dcol[t.row] * dcol[t.col]))
        .collect();
    let mut c: Vec<f64> = prob
        .c
        .iter()
        .enumerate()
        .map(|(i, &ci)| ci * dcol[i])
        .collect();
    let a: Vec<Triplet> = prob
        .a
        .iter()
        .map(|t| Triplet::new(t.row, t.col, t.val * drow_a[t.row] * dcol[t.col]))
        .collect();
    let b: Vec<f64> = prob
        .b
        .iter()
        .enumerate()
        .map(|(r, &br)| br * drow_a[r])
        .collect();
    let g: Vec<Triplet> = prob
        .g
        .iter()
        .map(|t| Triplet::new(t.row, t.col, t.val * drow_g[t.row] * dcol[t.col]))
        .collect();
    let h: Vec<f64> = prob
        .h
        .iter()
        .enumerate()
        .map(|(r, &hr)| hr * drow_g[r])
        .collect();
    let lb = scale_bounds(&prob.lb, &dcol, NEG_INF);
    let ub = scale_bounds(&prob.ub, &dcol, POS_INF);

    // Cost scaling σ, applied to the objective **only for a pure LP**
    // (empty/zero `P`). Rationale: the Ruiz pass above already normalizes the
    // `P` block of the augmented matrix to O(1), so for a QP the objective is
    // *already* commensurate with the constraint blocks — and because σ must
    // scale `P` and `c` together to preserve the minimizer, applying σ < 1 to a
    // QP would shrink the Hessian below the constraint scale, degrading the
    // scaled problem's strong convexity, diverging the dual iterates, and
    // tripping the direct path's Farkas detector with a false `PrimalInfeasible`.
    //
    // An LP has no `P` block for Ruiz to anchor the objective scale against, so
    // a large linear term `c` (e.g. NETLIB `nl`, ‖c‖ ~ 1e6) survives
    // equilibration, drives huge Newton steps, and pushes the cone-scaling block
    // until the KKT factorization goes singular. Here σ = 1/max|ĉ| is both
    // necessary and harmless (no Hessian to unbalance).
    let is_lp = p_lower.iter().all(|t| t.val == 0.0);
    let cmax = c.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let sigma = if is_lp && cmax > 0.0 {
        (1.0 / cmax).clamp(SIGMA_LO, SIGMA_HI)
    } else {
        1.0
    };
    if sigma != 1.0 {
        // (`p_lower` is empty here, but scale it for completeness/robustness.)
        p_lower.iter_mut().for_each(|t| t.val *= sigma);
        c.iter_mut().for_each(|v| *v *= sigma);
    }

    let scaled = QpProblem {
        n,
        p_lower,
        c,
        a,
        b,
        g,
        h,
        lb,
        ub,
    };
    (
        scaled,
        Scaling {
            dcol,
            drow_a,
            drow_g,
            sigma,
        },
    )
}

/// Scale a bound vector by `1/dcol` (since `x̂ = Dc⁻¹ x`), preserving the
/// ±∞ sentinels and the "no bounds" empty-vector convention. `dcol > 0`, so
/// the sign and finiteness of each bound are preserved.
fn scale_bounds(bnd: &[f64], dcol: &[f64], inf: f64) -> Vec<f64> {
    if bnd.is_empty() {
        return Vec::new();
    }
    bnd.iter()
        .enumerate()
        .map(|(i, &v)| {
            if v.abs() >= BOUND_INF {
                inf
            } else {
                v / dcol[i]
            }
        })
        .collect()
}

impl Scaling {
    /// Map a solution of the scaled problem back to the original problem's
    /// variables and duals, in place. `orig` is the unscaled problem, used to
    /// recompute the objective `½xᵀPx + cᵀx` directly at the recovered `x`
    /// (cheaper and more robust than dividing the scaled objective by σ).
    pub(crate) fn unscale_solution(&self, orig: &QpProblem, sol: &mut QpSolution) {
        for (xi, &d) in sol.x.iter_mut().zip(&self.dcol) {
            *xi *= d;
        }
        for (yi, &d) in sol.y.iter_mut().zip(&self.drow_a) {
            *yi *= d / self.sigma;
        }
        for (zi, &d) in sol.z.iter_mut().zip(&self.drow_g) {
            *zi *= d / self.sigma;
        }
        for (zi, &d) in sol.z_lb.iter_mut().zip(&self.dcol) {
            *zi /= self.sigma * d;
        }
        for (zi, &d) in sol.z_ub.iter_mut().zip(&self.dcol) {
            *zi /= self.sigma * d;
        }
        // Recompute the objective at the unscaled primal point.
        let mut px = vec![0.0; orig.n];
        orig.p_mul(&sol.x, &mut px);
        let mut obj = 0.0;
        for ((&xi, &pxi), &ci) in sol.x.iter().zip(&px).zip(&orig.c) {
            obj += 0.5 * xi * pxi + ci * xi;
        }
        sol.obj = obj;
    }

    /// Map a warm-start point given in the **original** problem's coordinates
    /// into the scaled problem's coordinates — the exact inverse of
    /// [`Scaling::unscale_solution`]'s primal/dual maps:
    ///
    /// ```text
    ///   x̂ = Dc⁻¹ x,   ŷ = σ y / R_A,        ẑ = σ z / R_G,
    ///   ẑ_lb = σ·Dc·z_lb,                    ẑ_ub = σ·Dc·z_ub.
    /// ```
    ///
    /// Used so the equilibrated warm path seeds the scaled solve with a point
    /// equivalent to the caller's warm start, preserving the warm-start benefit.
    pub(crate) fn scale_warm_start(&self, warm: &QpWarmStart) -> QpWarmStart {
        QpWarmStart {
            x: warm
                .x
                .iter()
                .zip(&self.dcol)
                .map(|(&xi, &d)| xi / d)
                .collect(),
            y: warm
                .y
                .iter()
                .zip(&self.drow_a)
                .map(|(&yi, &d)| yi * self.sigma / d)
                .collect(),
            z: warm
                .z
                .iter()
                .zip(&self.drow_g)
                .map(|(&zi, &d)| zi * self.sigma / d)
                .collect(),
            z_lb: warm
                .z_lb
                .iter()
                .zip(&self.dcol)
                .map(|(&zi, &d)| zi * self.sigma * d)
                .collect(),
            z_ub: warm
                .z_ub
                .iter()
                .zip(&self.dcol)
                .map(|(&zi, &d)| zi * self.sigma * d)
                .collect(),
        }
    }
}
