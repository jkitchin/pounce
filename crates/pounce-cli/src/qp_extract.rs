//! Extract a `pounce_convex::QpProblem` (standard form) from a parsed
//! `.nl` problem, for the LP/QP dispatch path (Phase 2).
//!
//! The classifier (`crate::dispatch`) has already decided the problem is
//! an LP or convex QP; this module marshals the parsed `NlProblem` into
//! the standard form the convex IPM consumes:
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equalities)
//!             G x ≤ h          (inequalities, incl. finite var bounds)
//! ```
//!
//! Mapping from the `.nl` representation:
//! - **Objective.** `P` is the Hessian of the (degree-≤2) objective —
//!   recovered with the same `analyze_quadratic` the classifier uses, so
//!   `P` here is exactly the matrix whose definiteness was tested. `c`
//!   is the objective's linear part. A `maximize` objective is negated
//!   into a minimization.
//! - **Constraints.** Each row has a linear part and bounds `g_l ≤ row ≤
//!   g_u`. An equality (`g_l == g_u`) becomes a row of `A`; a one- or
//!   two-sided inequality becomes one or two rows of `G` (`row ≤ g_u`
//!   and/or `−row ≤ −g_l`).
//! - **Variable bounds.** Finite `x_l`/`x_u` become `G` rows
//!   (`−x_i ≤ −x_l`, `x_i ≤ x_u`); the `.nl` "infinity" sentinel
//!   (`|v| ≥ 1e19`) is treated as no bound.

use crate::dispatch::analyze_quadratic_full;
use crate::nl_reader::NlProblem;
use pounce_convex::{ConeSpec, QpProblem, Triplet};

/// The `.nl` infinity sentinel: AMPL writes ±1e20-ish for "no bound";
/// upstream Ipopt treats anything with magnitude ≥ 1e19 as infinite.
const NL_INF: f64 = 1e19;

fn is_finite_bound(v: f64) -> bool {
    v.abs() < NL_INF
}

/// Convert a classified LP/convex-QP `NlProblem` into `QpProblem`
/// standard form. Returns `None` if the objective is not actually a
/// degree-≤2 polynomial (should not happen for a problem the classifier
/// routed here, but the conversion is total and falls back gracefully).
pub fn extract_qp(prob: &NlProblem) -> Option<QpProblem> {
    Some(extract_qp_with_map(prob)?.0) // drops con_map + reporting constant
}

/// Where each `.nl` constraint's rows landed in the standard-form QP, so
/// the QP's multipliers can be mapped back to a per-`.nl`-constraint
/// dual for the `.sol`. One entry per original constraint, in order.
#[derive(Debug, Clone)]
pub enum ConRowMap {
    /// Equality constraint → row `a_row` of `A` (multiplier `y[a_row]`).
    Eq { a_row: usize },
    /// Inequality / range constraint → up to two rows of `G`: the
    /// `row ≤ g_u` upper bound and/or the `−row ≤ −g_l` lower bound
    /// (multipliers `z[..]`, each ≥ 0).
    Ineq {
        upper: Option<usize>,
        lower: Option<usize>,
    },
}

/// Extract the QP, the constraint→row provenance map, and the objective
/// constant folded into the nonlinear tree (see below), together.
///
/// The third return value is the **degree-0 term of the nonlinear
/// objective** (e.g. the `+9` of `(x₀−3)²` that AMPL/Pyomo emit inside the
/// nonlinear tree rather than in `NlProblem::obj_constant`). The QP itself
/// ignores it — it does not move the minimizer — but the caller must add
/// it to the *reported* objective so the convex solve agrees with the NLP
/// path. It is returned in the problem's natural (user) sense, *not*
/// multiplied by the maximize/minimize `sign`.
pub fn extract_qp_with_map(prob: &NlProblem) -> Option<(QpProblem, Vec<ConRowMap>, f64)> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective Hessian P (lower triangle) + nonlinear-tree linear part
    //     + nonlinear-tree constant (degree-0 term, for reporting only) ---
    let (hess, obj_nl_linear, obj_nl_constant) = analyze_quadratic_full(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        // analyze_quadratic returns (i ≤ j) upper-ish keys; store as
        // lower triangle (row ≥ col) for the solver.
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }

    // --- objective linear term c ---
    // Two disjoint sources, exactly as the NLP path's eval_f sums them:
    // the `.nl` linear section (`obj_linear`) and the degree-1 terms AMPL
    // kept inside the nonlinear objective tree (e.g. the `−6·x₀` of
    // `(x₀−3)²`). Dropping the latter silently solves the wrong objective.
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }
    for (var, coef) in &obj_nl_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A x = b, inequalities → G x ≤ h ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut con_map: Vec<ConRowMap> = Vec::with_capacity(prob.con_linear.len());

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];

        // Combine the `.nl` linear section with any degree-≤1 terms AMPL
        // folded into the (here empty-Hessian) nonlinear tree — the
        // classifier admits constraint rows whose nonlinear expression
        // reduces to degree ≤ 1 (`dispatch.rs`), e.g. cancelled
        // quadratics or defined variables, and those linear/constant
        // terms live in `con_nonlinear`, not `con_linear`. Dropping them
        // silently solves the wrong constraint. The folded constant
        // shifts the bounds: `g_l ≤ row + k ≤ g_u  ⇔  g_l−k ≤ row ≤ g_u−k`.
        // This mirrors the SOCP extractor's linear-constraint handling.
        let (nl_lin, const_shift) = analyze_quadratic_full(&prob.con_nonlinear[row], n)
            .map(|(_, l, k)| (l, k))
            .unwrap_or_default();
        let mut coef = vec![0.0; n];
        for (var, v) in lin {
            coef[*var] += *v;
        }
        for (var, v) in &nl_lin {
            coef[*var] += *v;
        }
        let nonzeros = || coef.iter().enumerate().filter(|(_, v)| **v != 0.0);

        if lo == hi && is_finite_bound(lo) {
            // Equality row.
            let eq_row = next_row(&b);
            for (var, v) in nonzeros() {
                a.push(Triplet::new(eq_row, var, *v));
            }
            b.push(lo - const_shift);
            con_map.push(ConRowMap::Eq { a_row: eq_row });
        } else {
            // Upper bound: row ≤ hi.
            let upper = if is_finite_bound(hi) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, *v));
                }
                h.push(hi - const_shift);
                Some(gr)
            } else {
                None
            };
            // Lower bound: row ≥ lo  ⇔  −row ≤ −lo.
            let lower = if is_finite_bound(lo) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, -*v));
                }
                h.push(-(lo - const_shift));
                Some(gr)
            } else {
                None
            };
            con_map.push(ConRowMap::Ineq { upper, lower });
        }
    }

    // --- variable bounds as G rows (not part of the constraint map) ---
    for i in 0..n {
        let xl = prob.x_l[i];
        let xu = prob.x_u[i];
        if is_finite_bound(xu) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, 1.0)); // x_i ≤ xu
            h.push(xu);
        }
        if is_finite_bound(xl) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, -1.0)); // −x_i ≤ −xl
            h.push(-xl);
        }
    }

    Some((
        QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            // Variable bounds are currently emitted as `G` rows (see the
            // bound-handling above), so the explicit box is left empty.
            lb: Vec::new(),
            ub: Vec::new(),
        },
        con_map,
        obj_nl_constant,
    ))
}

/// Map the QP solver's multipliers `(y, z)` back to a per-`.nl`-
/// constraint dual vector (length `prob.m`), in the AMPL `.sol`
/// convention used by POUNCE's NLP path.
///
/// The QP solver enforces stationarity `∇f + Aᵀy + Gᵀz = 0` with
/// `z ≥ 0`, where each inequality `.nl` row contributes a `row ≤ g_u`
/// (`+row`) and/or `−row ≤ −g_l` (`−row`) `G` row. The per-constraint
/// `.nl`/Ipopt multiplier `λ` is recovered as:
/// - equality: `λ = sign · y[a_row]`;
/// - inequality: `λ = sign · (z_upper − z_lower)` — at most one of the
///   two bound rows is active at a solution.
///
/// The inequality sign (`z_upper − z_lower`, *not* `z_lower − z_upper`)
/// is fixed to match POUNCE's NLP path, which is the reference for what
/// a POUNCE `.sol` carries; this is verified empirically against the NLP
/// solve in the crate tests. `sign` undoes the maximize→minimize
/// negation so the reported dual is in the user's original sense.
pub fn recover_duals(prob: &NlProblem, con_map: &[ConRowMap], y: &[f64], z: &[f64]) -> Vec<f64> {
    let sign = if prob.minimize { 1.0 } else { -1.0 };
    con_map
        .iter()
        .map(|m| match m {
            ConRowMap::Eq { a_row } => sign * y[*a_row],
            ConRowMap::Ineq { upper, lower } => {
                let zu = upper.map(|r| z[r]).unwrap_or(0.0);
                let zl = lower.map(|r| z[r]).unwrap_or(0.0);
                sign * (zu - zl)
            }
        })
        .collect()
}

/// The next 0-based row index for a constraint block keyed by its RHS
/// vector's current length.
fn next_row(rhs: &[f64]) -> usize {
    rhs.len()
}

// ===========================================================================
// QCQP → SOCP extraction
// ===========================================================================

/// Where each `.nl` constraint landed in the standard-form **conic** program,
/// so the cone multipliers can be mapped back to a per-`.nl`-constraint dual.
/// One entry per original constraint, in order. (Analogue of [`ConRowMap`] for
/// the SOCP path produced by [`extract_socp_with_map`].)
#[derive(Debug, Clone)]
pub enum ConSocpMap {
    /// Linear equality → row `a_row` of `A` (multiplier `y[a_row]`).
    Eq { a_row: usize },
    /// Linear inequality / range → up to two rows of the nonnegative `G`
    /// block (`row ≤ g_u` and/or `−row ≤ −g_l`), multipliers `z[..] ≥ 0`.
    Ineq {
        upper: Option<usize>,
        lower: Option<usize>,
    },
    /// Convex quadratic inequality `g(x) ≤ g_u`, reformulated to one
    /// second-order cone. The first two cone rows both carry the linear
    /// coefficient vector `a = ∇(linear part)`, so the original constraint
    /// multiplier is recovered as `z[r0] + z[r1]` (see
    /// [`recover_socp_duals`]).
    Quad { z_row0: usize, z_row1: usize },
}

/// A deferred second-order-cone block, built after the nonnegative `G` rows
/// are known so the cones partition `G` in row order (nonneg block first,
/// then the SOCs).
struct SocBlock {
    /// Index in `con_map` of the originating constraint, to patch with the
    /// final cone-row indices once they are assigned.
    con_idx: usize,
    /// Linear coefficient vector `a` of the constraint (length `n`).
    a: Vec<f64>,
    /// `b_eff = (nonlinear constant) − g_u`, the constraint's degree-0 term
    /// after moving the upper bound to the right: `½xᵀQx + aᵀx + b_eff ≤ 0`.
    b_eff: f64,
    /// Rows of the factor `F` (each length `n`) with `FᵀF = Q`; `‖Fx‖² = xᵀQx`.
    f_rows: Vec<Vec<f64>>,
}

/// Convert a classified **convex QCQP** `NlProblem` into the conic standard
/// form the SOCP IPM consumes:
///
/// ```text
/// minimize    ½ xᵀP x + cᵀx
/// subject to  A x = b
///             h − G x  ∈  K        (K = nonneg orthant × second-order cones)
/// ```
///
/// Returns `(QpProblem, con_map, obj_nl_constant, cones)`:
/// - the objective `P`/`c` exactly as the LP/QP path builds them;
/// - linear equalities in `A`/`b`; linear inequalities and finite variable
///   bounds as a leading **nonnegative** `G` block; and each convex quadratic
///   inequality `g(x) ≤ g_u` as one **second-order cone** block appended
///   after it (so `cones` covers the `G` rows in order);
/// - `con_map` mapping each original constraint to its rows for dual recovery;
/// - `obj_nl_constant`, the objective's folded degree-0 term (added back to the
///   reported value, exactly as in [`extract_qp_with_map`]).
///
/// `None` if the objective is not degree-≤2 (should not happen for a problem
/// the classifier routed here). The reformulation of a convex quadratic
/// `½xᵀQx + aᵀx + b_eff ≤ 0` (with `Q = FᵀF ⪰ 0`) is the standard rotated→
/// standard SOC: writing `s = −(aᵀx + b_eff)`, the cone slack
/// `(s+1, s−1, √2·Fx)` lies in the second-order cone iff `‖Fx‖² ≤ 2s`, i.e.
/// iff the original constraint holds.
pub fn extract_socp_with_map(
    prob: &NlProblem,
) -> Option<(QpProblem, Vec<ConSocpMap>, f64, Vec<ConeSpec>)> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective P (lower triangle) + folded linear / constant terms ---
    let (hess, obj_nl_linear, obj_nl_constant) = analyze_quadratic_full(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }
    for (var, coef) in &obj_nl_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A; linear ineqs → nonneg G block;
    //     convex quadratics → deferred SOC blocks (added after the nonneg
    //     rows so the cones partition G in row order) ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut con_map: Vec<ConSocpMap> = Vec::with_capacity(prob.m);
    let mut soc_blocks: Vec<SocBlock> = Vec::new();

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];
        let nl = &prob.con_nonlinear[row];
        let quad = analyze_quadratic_full(nl, n);
        let is_quadratic = matches!(&quad, Some((hmap, _, _)) if !hmap.is_empty());

        if is_quadratic {
            // Convex quadratic inequality `g(x) ≤ g_u` (the classifier
            // guarantees an upper-only bound with PSD Hessian). Build the
            // factor F (FᵀF = Q) and defer the SOC rows.
            let (hmap, nl_lin, nl_const) = quad.expect("checked above");
            // Full linear coefficient vector a = linear-section + folded
            // nonlinear-tree linear part.
            let mut a_vec = vec![0.0; n];
            for (var, coef) in lin {
                a_vec[*var] += *coef;
            }
            for (var, coef) in &nl_lin {
                a_vec[*var] += *coef;
            }
            let dense = dense_symmetric(&hmap, n);
            let f_rows = psd_outer_factor(dense, n);
            let con_idx = con_map.len();
            con_map.push(ConSocpMap::Quad {
                z_row0: 0,
                z_row1: 0,
            }); // patched in the SOC pass below
            soc_blocks.push(SocBlock {
                con_idx,
                a: a_vec,
                b_eff: nl_const - hi,
                f_rows,
            });
            continue;
        }

        // Linear constraint. Combine the `.nl` linear section with any
        // degree-≤1 terms AMPL folded into the (here empty-Hessian)
        // nonlinear tree, and shift the bounds by the folded constant.
        let (nl_lin, const_shift) = quad.map(|(_, l, k)| (l, k)).unwrap_or_default();
        let mut coef = vec![0.0; n];
        for (var, v) in lin {
            coef[*var] += *v;
        }
        for (var, v) in &nl_lin {
            coef[*var] += *v;
        }
        let nonzeros = || coef.iter().enumerate().filter(|(_, v)| **v != 0.0);
        if lo == hi && is_finite_bound(lo) {
            let eq_row = next_row(&b);
            for (var, v) in nonzeros() {
                a.push(Triplet::new(eq_row, var, *v));
            }
            b.push(lo - const_shift);
            con_map.push(ConSocpMap::Eq { a_row: eq_row });
        } else {
            let upper = if is_finite_bound(hi) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, *v));
                }
                h.push(hi - const_shift);
                Some(gr)
            } else {
                None
            };
            let lower = if is_finite_bound(lo) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, -*v));
                }
                h.push(-(lo - const_shift));
                Some(gr)
            } else {
                None
            };
            con_map.push(ConSocpMap::Ineq { upper, lower });
        }
    }

    // --- variable bounds as nonnegative G rows (not in the constraint map) ---
    for i in 0..n {
        let xl = prob.x_l[i];
        let xu = prob.x_u[i];
        if is_finite_bound(xu) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, 1.0));
            h.push(xu);
        }
        if is_finite_bound(xl) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, -1.0));
            h.push(-xl);
        }
    }

    // The nonnegative block is every G row built so far. The cones list must
    // cover G in row order: this orthant block, then one SOC per quadratic.
    let num_nonneg = h.len();
    let mut cones: Vec<ConeSpec> = Vec::with_capacity(1 + soc_blocks.len());
    if num_nonneg > 0 {
        cones.push(ConeSpec::Nonneg(num_nonneg));
    }

    // --- emit the deferred second-order cones ---
    for blk in soc_blocks {
        let r = blk.f_rows.len();
        let dim = r + 2;
        let row0 = next_row(&h);
        // s0 = (1 − b_eff) − aᵀx  →  G row = a, h = 1 − b_eff.
        for (var, &coef) in blk.a.iter().enumerate() {
            if coef != 0.0 {
                g.push(Triplet::new(row0, var, coef));
            }
        }
        h.push(1.0 - blk.b_eff);
        let row1 = next_row(&h);
        // s1 = −(1 + b_eff) − aᵀx  →  G row = a, h = −(1 + b_eff).
        for (var, &coef) in blk.a.iter().enumerate() {
            if coef != 0.0 {
                g.push(Triplet::new(row1, var, coef));
            }
        }
        h.push(-(1.0 + blk.b_eff));
        // s_{2+k} = √2·(Fx)_k  →  G row = −√2·F_k, h = 0.
        let sqrt2 = std::f64::consts::SQRT_2;
        for f in &blk.f_rows {
            let gr = next_row(&h);
            for (var, &fv) in f.iter().enumerate() {
                if fv != 0.0 {
                    g.push(Triplet::new(gr, var, -sqrt2 * fv));
                }
            }
            h.push(0.0);
        }
        cones.push(ConeSpec::SecondOrder(dim));
        con_map[blk.con_idx] = ConSocpMap::Quad {
            z_row0: row0,
            z_row1: row1,
        };
    }

    Some((
        QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            lb: Vec::new(),
            ub: Vec::new(),
        },
        con_map,
        obj_nl_constant,
        cones,
    ))
}

/// Map the SOCP solver's multipliers `(y, z)` back to a per-`.nl`-constraint
/// dual vector (length `prob.m`), in POUNCE's NLP-path `.sol` convention.
///
/// Linear rows reuse the QP-path recovery (`y[a_row]` for an equality;
/// `z_upper − z_lower` for an inequality). For a convex quadratic
/// `g(x) ≤ g_u` reformulated to a second-order cone, the constraint
/// multiplier is recovered as the sum of the two cone duals on the rows
/// carrying the linear coefficient vector `a`: `λ = z[r0] + z[r1]`. (At a
/// KKT point stationarity reads `λ(∇g) = (z[r0]+z[r1])·a + …`, so this sum is
/// the original multiplier; the cone's remaining rows reconstruct the `Qx`
/// part.) `sign` undoes the maximize→minimize negation.
pub fn recover_socp_duals(
    prob: &NlProblem,
    con_map: &[ConSocpMap],
    y: &[f64],
    z: &[f64],
) -> Vec<f64> {
    let sign = if prob.minimize { 1.0 } else { -1.0 };
    con_map
        .iter()
        .map(|m| match m {
            ConSocpMap::Eq { a_row } => sign * y[*a_row],
            ConSocpMap::Ineq { upper, lower } => {
                let zu = upper.map(|r| z[r]).unwrap_or(0.0);
                let zl = lower.map(|r| z[r]).unwrap_or(0.0);
                sign * (zu - zl)
            }
            ConSocpMap::Quad { z_row0, z_row1 } => sign * (z[*z_row0] + z[*z_row1]),
        })
        .collect()
}

/// Build a dense symmetric `n×n` matrix from a [`QuadHessian`]-style map of
/// `(i ≤ j) → Hessian entry` (diagonal entries are the full `∂²/∂xᵢ²`, so
/// `½xᵀHx` reproduces the quadratic form). Off-diagonals are mirrored.
fn dense_symmetric(hmap: &std::collections::BTreeMap<(usize, usize), f64>, n: usize) -> Vec<f64> {
    let mut dense = vec![0.0; n * n];
    for (&(i, j), &v) in hmap {
        dense[i * n + j] = v;
        dense[j * n + i] = v;
    }
    dense
}

/// Symmetric **pivoted (rank-revealing) Cholesky** of a PSD matrix `H`
/// (row-major `n×n`, consumed as scratch), returning the factor rows `f_k`
/// (each length `n`) such that `Σ_k f_k f_kᵀ = H` — equivalently `FᵀF = H`
/// with `F` the matrix whose rows are the `f_k`. The number of rows is the
/// numerical rank, so a rank-deficient `Q` (e.g. `Q = vvᵀ`) yields the
/// minimal cone. Complete diagonal pivoting keeps the factorization stable
/// on the indefinite-looking-but-PSD matrices finite precision can produce.
fn psd_outer_factor(mut a: Vec<f64>, n: usize) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    // Tolerance relative to the largest initial diagonal: pivots at or below
    // this are treated as the zero eigenvalues of the PSD matrix.
    let max_diag = (0..n).map(|i| a[i * n + i]).fold(0.0_f64, f64::max);
    let tol = 1e-12 * max_diag.max(1.0);
    for _ in 0..n {
        // Largest remaining diagonal pivot.
        let mut p = 0usize;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            let d = a[i * n + i];
            if d > best {
                best = d;
                p = i;
            }
        }
        if best <= tol {
            break;
        }
        let d = best.sqrt();
        // f = column p of the residual, scaled by 1/d.
        let mut f = vec![0.0; n];
        for i in 0..n {
            f[i] = a[i * n + p] / d;
        }
        // Rank-1 downdate: A ← A − f fᵀ.
        for i in 0..n {
            let fi = f[i];
            if fi == 0.0 {
                continue;
            }
            for j in 0..n {
                a[i * n + j] -= fi * f[j];
            }
        }
        rows.push(f);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr};
    use pounce_convex::{solve_qp_ipm, solve_socp_ipm, QpOptions, QpStatus};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn pow2(var: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(var)),
            Box::new(Expr::Const(2.0)),
        )
    }

    /// min −x0 − x1  s.t.  x0² + x1² ≤ 1  → x* = (1/√2, 1/√2), f* = −√2.
    /// Exercises the QCQP→SOCP reformulation end-to-end: a rank-2 ball
    /// constraint becomes one second-order cone, no nonnegative block.
    #[test]
    fn extract_and_solve_socp_ball() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Binary(
                BinOp::Add,
                Box::new(pow2(0)),
                Box::new(pow2(1)),
            )],
            con_linear: vec![vec![]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![-2e19],
            g_u: vec![1.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(obj_const, 0.0);
        // No linear inequalities / bounds → no nonneg block; one SOC of
        // dimension rank(Q)+2 = 2+2 = 4.
        assert_eq!(cones, vec![ConeSpec::SecondOrder(4)]);
        assert_eq!(qp.m_ineq(), 4);

        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((sol.x[0] - inv_sqrt2).abs() < 1e-5, "x0={}", sol.x[0]);
        assert!((sol.x[1] - inv_sqrt2).abs() < 1e-5, "x1={}", sol.x[1]);
        assert!(
            (sol.obj - (-2.0_f64.sqrt())).abs() < 1e-5,
            "obj={}",
            sol.obj
        );

        // Analytic multiplier: c + λ·2x = 0 ⇒ λ = 1/(2x0) = √2/2 ≈ 0.7071,
        // positive (active upper bound), matching the `.sol` sign convention.
        let lambda = recover_socp_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(
            (lambda[0] - 0.5 * 2.0_f64.sqrt()).abs() < 1e-3,
            "ball constraint dual={}",
            lambda[0]
        );
    }

    /// min x0  s.t.  (x0−3)² ≤ 1  → feasible x0 ∈ [2, 4], optimum x0 = 2.
    /// The constraint's linear (`−6x0`) and constant (`+9`) terms are folded
    /// into the nonlinear tree; the reformulation must recover `b_eff = 9 − 1`
    /// so the cone encodes `x0² − 6x0 + 8 ≤ 0`, not `x0² ≤ 1`.
    #[test]
    fn extract_and_solve_socp_folds_constraint_constant() {
        let con = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]],
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![-2e19],
            g_u: vec![1.0],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, _con_map, obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(obj_const, 0.0);
        assert_eq!(cones, vec![ConeSpec::SecondOrder(3)]); // rank 1 + 2.

        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-5, "x0={}", sol.x[0]);
    }

    /// `psd_outer_factor` recovers a rank-1 `Q = vvᵀ` with a single factor row
    /// (minimal cone), and reconstructs `Q` exactly.
    #[test]
    fn psd_outer_factor_is_rank_revealing() {
        // Q = [[1,2],[2,4]] = v vᵀ with v = (1,2): rank 1.
        let q = vec![1.0, 2.0, 2.0, 4.0];
        let rows = psd_outer_factor(q.clone(), 2);
        assert_eq!(rows.len(), 1, "rank-1 Q must give one factor row");
        // Reconstruct Σ f fᵀ and compare to Q.
        let mut recon = vec![0.0; 4];
        for f in &rows {
            for i in 0..2 {
                for j in 0..2 {
                    recon[i * 2 + j] += f[i] * f[j];
                }
            }
        }
        for k in 0..4 {
            assert!((recon[k] - q[k]).abs() < 1e-9, "recon[{k}]={}", recon[k]);
        }
    }

    /// min (x0)^2 + (x1)^2 s.t. x0 + x1 = 2, no var bounds → (1,1), f*=2.
    #[test]
    fn extract_and_solve_equality_qp() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Binary(BinOp::Add, Box::new(pow2(0)), Box::new(pow2(1))),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![2.0],
            g_u: vec![2.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // No constant anywhere in this objective.
        assert_eq!(obj_const, 0.0);
        // P = 2I → two diagonal entries.
        assert_eq!(qp.p_lower.len(), 2);
        assert_eq!(qp.m_eq(), 1);
        assert_eq!(qp.m_ineq(), 0);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
        assert!((sol.obj - 2.0).abs() < 1e-6, "obj={}", sol.obj);

        // KKT for the equality: ∇f + y·∇g = 0 → 2x_i + y = 0 at x=1 → y=−2.
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(
            (lambda[0] - (-2.0)).abs() < 1e-5,
            "equality dual={}",
            lambda[0]
        );
    }

    /// Regression for the dropped-linear-term bug: the objective `(x0-3)²`
    /// lives entirely in the nonlinear tree, so its linear part (`−6·x0`)
    /// must be folded into `c`. Without it the solve minimizes `x0²`
    /// (optimum 0) instead of `(x0-3)²` (optimum 3).
    #[test]
    fn extract_keeps_linear_term_from_nonlinear_tree() {
        // (x0 - 3)^2 = x0^2 - 6 x0 + 9, all in obj_nonlinear.
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(qp.c.len(), 1);
        assert!(
            (qp.c[0] - (-6.0)).abs() < 1e-12,
            "c[0]={} — linear term from the nonlinear tree was dropped",
            qp.c[0]
        );
        // P = 2 (one diagonal entry).
        assert_eq!(qp.p_lower.len(), 1);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            (sol.x[0] - 3.0).abs() < 1e-6,
            "x0={} (expected 3)",
            sol.x[0]
        );
    }

    /// Inequality dual sign/magnitude. min x0² s.t. x0 ≥ 1 (a one-sided
    /// inequality g_l=1, g_u=+inf). Optimum x0=1, active. The expected
    /// dual −2.0 is the value POUNCE's *NLP* path writes for this exact
    /// problem (verified by running `solver_selection=nlp` on the same
    /// `.nl`); recover_duals must match that reference convention.
    #[test]
    fn inequality_dual_recovered() {
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0)]], // g(x) = x0
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![1.0], // x0 ≥ 1
            g_u: vec![2e19],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // This model puts its constant in the `obj_constant` field, not the
        // nonlinear tree, so the tree constant is 0 here.
        assert_eq!(obj_const, 0.0);
        // One inequality row (the lower bound row −x0 ≤ −1); no upper.
        assert_eq!(qp.m_ineq(), 1);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert!((lambda[0] - (-2.0)).abs() < 1e-5, "ineq dual={}", lambda[0]);
    }

    /// Regression (M11): a *constraint* whose linear and constant
    /// terms are folded into the nonlinear tree (not the `con_linear`
    /// section) must still reach `A`/`G`. AMPL/Pyomo emit this shape for
    /// rows the classifier admits as degree-≤1 (cancelled quadratics,
    /// defined variables): the whole `x0 − 3` lives in `con_nonlinear`
    /// and `con_linear[0]` is empty.
    ///
    ///     min x0   s.t.   x0 − 3 ≥ 0     (body in the nonlinear tree)
    ///
    /// True optimum: x0 = 3. The QP extractor used to build `A`/`G` from
    /// `con_linear` only — dropping the folded `+x0` *and* the `−3`
    /// shift, leaving a vacuous `0 ≤ 0` row, so `min x0` came out
    /// unbounded (or otherwise wrong) on the convex path.
    #[test]
    fn constraint_linear_terms_folded_in_tree_are_recovered() {
        // con body = x0 − 3, entirely in the nonlinear tree.
        let con = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]], // the `+x0` lives in the TREE
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![0.0], // x0 − 3 ≥ 0
            g_u: vec![2e19],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, _obj_const) = extract_qp_with_map(&prob).expect("extract");
        // One inequality row: −x0 ≤ −3 (the lower bound, constant-shifted).
        assert_eq!(qp.m_ineq(), 1);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 3.0).abs() < 1e-5, "x0={}", sol.x[0]);
        // Dual is recoverable and finite (the row carries a real coef now).
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(lambda[0].is_finite(), "dual={}", lambda[0]);
    }

    /// Regression: a constant folded into the *nonlinear objective tree*
    /// (not the `obj_constant` field) must still reach the reported
    /// objective. This is the real `.nl` shape AMPL/Pyomo emit for
    /// `min (x0-3)^2` — the whole `x0^2 - 6 x0 + 9` lives in the nonlinear
    /// tree and `obj_constant == 0`. The convex path used to drop the `+9`
    /// and report an objective 9 too small (cf. HS35 in the benchmark
    /// comparison). The minimizer is x0 = 1 (upper bound binds), where the
    /// true objective is (1-3)^2 = 4.
    #[test]
    fn tree_embedded_objective_constant_is_recovered() {
        // (x0 - 3)^2 as a single nonlinear tree: Pow(Sub(x0, 3), 2).
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0, // the +9 is in the TREE, not here
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, _con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // The degree-0 term of (x0-3)^2 is +9, recovered from the tree.
        assert!((obj_const - 9.0).abs() < 1e-12, "tree constant={obj_const}");
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        // Reported objective = (½xᵀPx + cᵀx) + obj_const must equal the true
        // (1-3)^2 = 4, not the constant-dropped −5.
        let reported = sol.obj + obj_const;
        assert!((reported - 4.0).abs() < 1e-5, "reported obj={reported}");
    }

    /// Bound-constrained: min (x0-3)^2 = x0^2 - 6 x0 + 9, 0 ≤ x0 ≤ 1.
    /// Optimum x0 = 1 (upper bound binds). Here the constant 9 is carried
    /// in the `obj_constant` field (not the tree), so the extracted tree
    /// constant is 0 (asserted inside).
    #[test]
    fn extract_and_solve_bounded_qp() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![(0, -6.0)],
            obj_constant: 9.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // Two var-bound rows (x0 ≤ 1, −x0 ≤ 0).
        assert_eq!(qp.m_ineq(), 2);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }

    /// LP: min −x0 − x1, 0 ≤ x ≤ 1 → (1,1).
    #[test]
    fn extract_and_solve_lp() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0, 0.0],
            x_u: vec![1.0, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0, 0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert!(qp.p_lower.is_empty(), "LP has no Hessian");
        assert_eq!(qp.m_ineq(), 4); // 2 vars × (upper + lower)
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// maximize x0 s.t. 0 ≤ x0 ≤ 5 → x0 = 5. Tests sign flip on a
    /// maximize objective.
    #[test]
    fn extract_maximize_negates() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: false,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![5.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // minimize −x0.
        assert_eq!(qp.c[0], -1.0);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 5.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }
}
