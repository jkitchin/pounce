//! Curvature-based scaling for quadratically-constrained models (gh #703).
//!
//! # What this is for
//!
//! POUNCE's default `nlp_scaling_method=gradient-based` is a **point
//! sample**: it reads `∇f` and the Jacobian once, at x₀, and sets row `i`'s
//! factor from `‖∇gᵢ(x₀)‖_∞`. That is a fine estimator of a row's magnitude
//! when the row's derivative at x₀ is representative of its derivative
//! elsewhere, and no estimator at all when it is not.
//!
//! A row written `½xᵀQᵢx ≤ bᵢ` about the origin has `∇gᵢ(0) = 0`. Started
//! from `x₀ = 0` — the default for a model with free variables and no
//! initial guess, which is how AMPL emits the Mittelmann `qcqp*` family —
//! the sample reads nothing and the row is assigned factor **1.0** however
//! far `Qᵢ` and `bᵢ` disagree in magnitude. No cutoff reaches it: `100/0`
//! and `1e-6/0` both clamp to 1. Measured across POUNCE's own CLI fixture
//! corpus, that is 196 of 196 quadratic rows left unscaled.
//!
//! This module computes the factors from the model's **coefficients**
//! instead of from a derivative sample, so a row's scale does not depend on
//! where the modeller happened to start. It is the scheme worked out in
//! `dev-notes/quadratic-structure-exploitation.md` §8, in two stages.
//!
//! # Stage 1 — one `D` for the whole family
//!
//! Ruiz per `Qᵢ` individually is wrong: each would demand its own column
//! scaling and there is only one `x`. The matrix that gets factored is
//! `H(λ) = Q₀ + Σλᵢ Qᵢ`, so `D` must equilibrate all of them **jointly**.
//! §8's device is a pair of λ-independent magnitude surrogates — the
//! ∞-norm envelope of every `H(λ)` that can arise —
//!
//! ```text
//!   P̂[j,k] = max( |Q₀[j,k]| , maxᵢ |Qᵢ[j,k]| )
//!   Ĵ[i,j] = max( |aᵢ[j]|   , maxₖ |Qᵢ[j,k]| )
//! ```
//!
//! which are then Ruiz-equilibrated as the symmetric augmented matrix
//! `K̂ = [[P̂, Ĵᵀ], [Ĵ, 0]]`, exactly as
//! `pounce_convex::equilibrate` sweeps the LP/QP one. `D` is the variable
//! block of the result.
//!
//! **Extension beyond §8, stated.** §8 was written for a driver whose only
//! rows are quadratic. A model on the NLP path has linear rows too, and they
//! constrain the same `x`; leaving them out of `Ĵ` would balance the
//! quadratic rows by unbalancing the linear ones. So every constraint row
//! contributes a row to `Ĵ` — a linear row through `|aᵢ[j]|` alone, which is
//! what the formula reduces to when `Qᵢ = 0`.
//!
//! # Stage 2 — the per-row scale
//!
//! After `D` is fixed,
//!
//! ```text
//!   eᵢ = 1 / max( ‖D Qᵢ D‖_∞ , ‖D aᵢ‖_∞ , |bᵢ| )
//! ```
//!
//! This is the term that carries the measured win. On a QCQP shaped like
//! `qcqp1000-2c` the right-hand sides run ~50× above `‖Qᵢ‖_∞`, so the `max`
//! *is* the right-hand side — and an unnormalized `bᵢ` biases the slack
//! `sᵢ = −gᵢ(x)` and with it the `−sᵢ/λᵢ` diagonal of the KKT system.
//!
//! # What is deliberately **not** scaled
//!
//! The objective. `pounce_convex::equilibrate` learned this the hard way and
//! its comment is the authority: the Ruiz pass already normalizes the `P`
//! block against the constraint blocks, and a `σ < 1` applied to a problem
//! that *has* a Hessian shrinks it below the constraint scale, degrades the
//! scaled problem's strong convexity and diverges the dual iterates. Every
//! model this method accepts has a quadratic objective or none, so `σ` would
//! never be the LP case where that module found it necessary. `obj_scaling`
//! is left at 1.0.
//!
//! # Scope, and why it declines
//!
//! Both surrogates are `λ`-independent only because each `Qᵢ` is a
//! *constant* matrix. A row with a genuine nonlinearity has no such `Qᵢ`,
//! the envelope does not exist, and building `D` from that row's linear
//! section alone would silently equilibrate against a fiction. So
//! [`curvature_scaling`] **declines** (returns `None`) on any model whose
//! objective or rows are not all degree ≤ 2, and the caller turns that into
//! an error naming the option rather than falling back to something else —
//! a scaling option that is accepted and then quietly not applied is gh #483
//! all over again.

use std::collections::BTreeMap;

use crate::nl_reader::NlProblem;
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// Ruiz sweeps over `K̂`. Matches `pounce_convex::equilibrate`'s
/// `RUIZ_SWEEPS`: Ruiz converges geometrically and a handful of passes
/// brings the row/column ∞-norms within a few percent of 1.
const RUIZ_SWEEPS: usize = 10;

/// Clamp on every emitted factor, matching the `[1e-8, 1e8]` bracket
/// `pounce_convex::equilibrate` puts on `σ`, so degenerate data cannot
/// itself create an extreme scaling. The lower end coincides with
/// `nlp_scaling_min_value`'s default.
const SCALE_LO: Number = 1e-8;
const SCALE_HI: Number = 1e8;

/// Point-free coefficient magnitudes of one quadratic constraint row.
///
/// `curvature` is `‖Q‖_∞`, the largest absolute row sum of the row's
/// Hessian — Gershgorin's bound on `λ_max(Q)`, and the exact quantity
/// stage 2's `eᵢ` is built from. It is an upper bound on the curvature,
/// not the curvature, so a mismatch measured against it **understates**
/// the one measured against `λ_max`.
///
/// Defined in `pounce-nlp` alongside the starting-point preflight that
/// consumes it, so a frontend can hold the census without depending on the
/// `.nl` reader. [`quad_row_coefs`] below is the `.nl`-specific producer,
/// and stays here: reading these coefficients needs the model's linear
/// section and nonlinear tree, which a bare TNLP does not expose.
pub use pounce_nlp::diagnostics::preflight::QuadRowCoef;

/// Read every constraint row's quadratic coefficients out of an
/// [`NlProblem`].
///
/// Uses [`crate::nl_reader::NlBody::analyze_quadratic_full`] — the same
/// read-out the LP/QP dispatch classifies with — so a row counted here is
/// a row the recognizer agrees is quadratic. Rows it refuses (a genuine
/// nonlinearity, or a quadratic whose recognition lost a term) are simply
/// absent, which is why a caller reporting this census must report the
/// count alongside `m` rather than implying it covers the model.
///
/// `O(nnz)` in the stored Hessian entries and no evaluation: this is a
/// property of the file, not of a point.
pub fn quad_row_coefs(prob: &NlProblem) -> Vec<QuadRowCoef> {
    let mut out = Vec::new();
    for i in 0..prob.m {
        let Some((hess, nl_lin, nl_const)) = prob.con_nonlinear[i].analyze_quadratic_full() else {
            continue;
        };
        if hess.is_empty() {
            continue; // degree ≤ 1: a linear row, not this census's business
        }
        // `hess` is the upper triangle (i ≤ j) of a symmetric matrix, so an
        // off-diagonal entry contributes its magnitude to two row sums.
        let mut row_sum: BTreeMap<usize, Number> = BTreeMap::new();
        for (&(r, c), v) in &hess {
            let a = v.abs();
            *row_sum.entry(r).or_insert(0.0) += a;
            if r != c {
                *row_sum.entry(c).or_insert(0.0) += a;
            }
        }
        let curvature = row_sum.values().fold(0.0_f64, |m, &v| m.max(v));
        let linear = row_linear(prob, i, &nl_lin)
            .values()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        out.push(QuadRowCoef {
            index: i,
            curvature,
            linear,
            rhs: row_rhs(prob, i, nl_const),
        });
    }
    out
}

/// A row's full linear part: the `.nl` linear section plus the degree-1
/// terms AMPL folded into the nonlinear tree. They can land on the same
/// variable, so they are accumulated rather than concatenated.
fn row_linear(prob: &NlProblem, i: usize, nl_lin: &[(usize, Number)]) -> BTreeMap<usize, Number> {
    let mut lin: BTreeMap<usize, Number> = BTreeMap::new();
    for (var, coef) in &prob.con_linear[i] {
        *lin.entry(*var).or_insert(0.0) += *coef;
    }
    for (var, coef) in nl_lin {
        *lin.entry(*var).or_insert(0.0) += *coef;
    }
    lin
}

/// `|bᵢ|` — the finite bound the row is written against, shifted by the
/// constant the writer folded into the tree (the row is
/// `½xᵀQx + aᵀx ≤ g_u − c`). A range row reports the larger magnitude,
/// since one scale has to serve both sides.
fn row_rhs(prob: &NlProblem, i: usize, nl_const: Number) -> Number {
    let (lo, hi) = (prob.g_l[i], prob.g_u[i]);
    let mut rhs = 0.0_f64;
    if lower_bound_present(lo) {
        rhs = rhs.max((lo - nl_const).abs());
    }
    if upper_bound_present(hi) {
        rhs = rhs.max((hi - nl_const).abs());
    }
    rhs
}

/// The factors [`curvature_scaling`] produces, in the conventions
/// [`crate::nl_reader::NlTnlp::get_scaling_parameters`] hands back.
#[derive(Debug, Clone)]
pub struct CurvatureScaling {
    /// Per-variable factors in **`ScalingTnlp`'s** convention, `x̃ = d ⊙ x`
    /// — so `d = D⁻¹` for §8's `x = D x̂`. Emitting `D` here instead would
    /// apply the scaling backwards, which is the one sign error in this
    /// module that no test of `D` alone would catch; `x_factors_invert_d`
    /// pins it.
    pub x: Vec<Number>,
    /// Per-row factors `eᵢ`, multiplying the row exactly as
    /// gradient-based scaling's `c_scale` / `d_scale` do.
    pub g: Vec<Number>,
    /// Whether the model this was computed from actually carries a nonzero
    /// second-order coefficient — in the objective's `P` or in some row's
    /// `Qᵢ`.
    ///
    /// A model of degree ≤ 2 need not have any: an LP is degree ≤ 2 with
    /// every `Q` empty, and the scheme still returns factors for it, but
    /// with `‖D Qᵢ D‖_∞ = 0` throughout, stage 2 collapses to
    /// `eᵢ = 1/max(‖D aᵢ‖_∞, |bᵢ|)` and stage 1's `K̂` loses its `P̂`
    /// block. What is left is plain Ruiz equilibration of `[A b]` — a
    /// perfectly good scaling, but not one that read any curvature, because
    /// there was none to read. The caller needs to know the difference: it
    /// is the whole justification for spending the convex fast path on this
    /// option (gh #703, gh#483).
    pub quadratic: bool,
}

/// One row's degree-≤2 read-out, as [`curvature_scaling`] needs it: the
/// Hessian's stored triangle, the accumulated linear part, and `|bᵢ|`.
struct QuadRow {
    hess: BTreeMap<(usize, usize), Number>,
    lin: BTreeMap<usize, Number>,
    rhs: Number,
}

/// Compute §8's two-stage scaling for `prob`.
///
/// Returns `None` when the model is not one this scheme is defined for —
/// the objective or some row is not degree ≤ 2, so no constant `Qᵢ` exists
/// and the magnitude envelope of `H(λ)` is not λ-independent. See the
/// module docs on why that is a refusal and not a fallback.
pub fn curvature_scaling(prob: &NlProblem) -> Option<CurvatureScaling> {
    let n = prob.n;
    let m = prob.m;

    // ---- read the model's constant structure, or decline ----
    let (obj_hess, _obj_lin, _obj_const) = prob.obj_nonlinear.analyze_quadratic_full()?;
    let mut rows: Vec<QuadRow> = Vec::with_capacity(m);
    for i in 0..m {
        let (hess, nl_lin, nl_const) = prob.con_nonlinear[i].analyze_quadratic_full()?;
        rows.push(QuadRow {
            hess,
            lin: row_linear(prob, i, &nl_lin),
            rhs: row_rhs(prob, i, nl_const),
        });
    }

    // ---- stage 1: the two λ-independent magnitude surrogates ----
    //
    // `P̂` is sparse over the union of the objective's and every row's
    // Hessian support. Materializing it dense would be `n²` on a model
    // whose whole point is that `n` is four digits.
    let mut p_hat: BTreeMap<(usize, usize), Number> = BTreeMap::new();
    let bump = |map: &mut BTreeMap<(usize, usize), Number>, key, v: Number| {
        let slot = map.entry(key).or_insert(0.0);
        if v > *slot {
            *slot = v;
        }
    };
    for (&k, v) in &obj_hess {
        bump(&mut p_hat, k, v.abs());
    }
    // `Ĵ[i, j] = max(|aᵢ[j]|, maxₖ |Qᵢ[j, k]|)`, sparse per row over the
    // row's own support.
    let mut j_hat: Vec<BTreeMap<usize, Number>> = Vec::with_capacity(m);
    for QuadRow { hess, lin, .. } in &rows {
        for (&k, v) in hess {
            bump(&mut p_hat, k, v.abs());
        }
        let mut row: BTreeMap<usize, Number> = BTreeMap::new();
        for (&(r, c), v) in hess {
            let a = v.abs();
            let slot = row.entry(r).or_insert(0.0);
            if a > *slot {
                *slot = a;
            }
            let slot = row.entry(c).or_insert(0.0);
            if a > *slot {
                *slot = a;
            }
        }
        for (&j, v) in lin {
            let a = v.abs();
            let slot = row.entry(j).or_insert(0.0);
            if a > *slot {
                *slot = a;
            }
        }
        j_hat.push(row);
    }

    // ---- Ruiz on `K̂ = [[P̂, Ĵᵀ], [Ĵ, 0]]` ----
    //
    // `K̂` is symmetric, so one scale vector serves rows and columns; the
    // layout is [0, n) variables then [n, n+m) rows, matching
    // `pounce_convex::equilibrate`.
    let dim = n + m;
    let mut s = vec![1.0_f64; dim];
    let mut rownorm = vec![0.0_f64; dim];
    for _ in 0..RUIZ_SWEEPS {
        rownorm.iter_mut().for_each(|v| *v = 0.0);
        for (&(r, c), v) in &p_hat {
            let x = (s[r] * v * s[c]).abs();
            if x > rownorm[r] {
                rownorm[r] = x;
            }
            if r != c && x > rownorm[c] {
                rownorm[c] = x;
            }
        }
        for (i, row) in j_hat.iter().enumerate() {
            let ri = n + i;
            for (&j, v) in row {
                let x = (s[ri] * v * s[j]).abs();
                if x > rownorm[ri] {
                    rownorm[ri] = x;
                }
                if x > rownorm[j] {
                    rownorm[j] = x;
                }
            }
        }
        // Ruiz update. An all-zero row — an empty column, or a row with no
        // entries at all — is left unscaled rather than divided by zero.
        for i in 0..dim {
            if rownorm[i] > 0.0 {
                s[i] /= rownorm[i].sqrt();
            }
        }
    }
    let d: Vec<Number> = s[..n].iter().map(|v| v.clamp(SCALE_LO, SCALE_HI)).collect();

    // ---- stage 2: eᵢ = 1 / max(‖D Qᵢ D‖_∞, ‖D aᵢ‖_∞, |bᵢ|) ----
    let mut g = vec![1.0_f64; m];
    for (i, QuadRow { hess, lin, rhs }) in rows.iter().enumerate() {
        // `‖D Qᵢ D‖_∞` is the largest absolute row sum of the *full*
        // symmetric `D Qᵢ D`; `hess` holds one triangle, so an
        // off-diagonal entry lands in two row sums.
        let mut row_sum: BTreeMap<usize, Number> = BTreeMap::new();
        for (&(r, c), v) in hess {
            let scaled = (d[r] * v * d[c]).abs();
            *row_sum.entry(r).or_insert(0.0) += scaled;
            if r != c {
                *row_sum.entry(c).or_insert(0.0) += scaled;
            }
        }
        let q_norm = row_sum.values().fold(0.0_f64, |m, &v| m.max(v));
        let a_norm = lin
            .iter()
            .fold(0.0_f64, |m, (&j, v)| m.max((v * d[j]).abs()));
        let scale = q_norm.max(a_norm).max(*rhs);
        // An empty row constrains nothing and normalizing it is a division
        // by zero; leave it alone.
        if scale > 0.0 {
            g[i] = (1.0 / scale).clamp(SCALE_LO, SCALE_HI);
        }
    }

    // `ScalingTnlp` substitutes `x̃ = d ⊙ x` while §8 writes `x = D x̂`, so
    // the factor handed over is the reciprocal. See `CurvatureScaling::x`.
    Some(CurvatureScaling {
        x: d.iter().map(|v| 1.0 / v).collect(),
        g,
        // Nonzero, not merely present: `analyze_quadratic_full` reports the
        // support it found, and a stored explicit zero is not curvature.
        quadratic: obj_hess
            .values()
            .chain(rows.iter().flat_map(|r| r.hess.values()))
            .any(|v| *v != 0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::parse_nl_text;

    /// `min ½(2x₀² + 2e-8·x₁²) − x₀ − x₁  s.t.  ½(4x₀² + 2e-8·x₁²) ≤ 1e5`,
    /// both variables free.
    ///
    /// **The spread is in the objective too, and that is the point.**
    /// Stage 1 equilibrates the ∞-norm envelope of `Q₀ + Σλᵢ Qᵢ`, so a
    /// tiny coefficient in one row that the objective's own entry masks is
    /// correctly left to stage 2's `eᵢ` — a column rescale would be wrong
    /// there, because the matrix that actually gets factored is *not*
    /// ill-scaled in that direction. Putting the spread in both makes the
    /// envelope itself span eight orders, which is when `D` is the right
    /// tool and the reciprocal convention below can be read off it.
    const SPREAD_NL: &str = "\
g3 0 1 0
 2 1 1 0 0
 1 1
 0 0
 2 2 2
 0 0 0 1
 0 0 0 0 0
 2 2
 0 0
 0 0 0 0 0
b
3
3
r
1 100000
C0
o54
2
o2
n0.5
o2
o2
n4.0
v0
v0
o2
n0.5
o2
o2
n2e-8
v1
v1
O0 0
o54
2
o2
n0.5
o2
o2
n2.0
v0
v0
o2
n0.5
o2
o2
n2e-8
v1
v1
k1
1
J0 2
0 0
1 0
G0 2
0 -1.0
1 -1.0
";

    /// `min x₀  s.t.  exp(x₀) ≤ 2` — degree > 2, so no constant `Q`.
    const NONLINEAR_NL: &str = "\
g3 0 1 0
 1 1 1 0 0
 1 0
 0 0
 1 1 1
 0 0 0 1
 0 0 0 0 0
 1 1
 0 0
 0 0 0 0 0
b
3
r
1 2
C0
o44
v0
O0 0
n0
k0
J0 1
0 0
G0 1
0 1.0
";

    #[test]
    fn a_genuine_nonlinearity_is_declined_not_approximated() {
        let prob = parse_nl_text(NONLINEAR_NL).expect("parse");
        assert!(
            curvature_scaling(&prob).is_none(),
            "exp(x) has no constant Hessian; approximating it from the \
             linear section would equilibrate against a fiction"
        );
    }

    #[test]
    fn a_quadratic_model_is_accepted() {
        let prob = parse_nl_text(SPREAD_NL).expect("parse");
        let sc = curvature_scaling(&prob).expect("degree ≤ 2 everywhere");
        assert_eq!(sc.x.len(), 2);
        assert_eq!(sc.g.len(), 1);
        assert!(sc.x.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(sc.g.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    /// The factor handed back is `D⁻¹`, not `D`: `ScalingTnlp` substitutes
    /// `x̃ = d ⊙ x` while §8 writes `x = D x̂`. Emitting `D` would apply
    /// the whole scheme backwards — doubling the imbalance instead of
    /// removing it — and every test of `D`'s *magnitudes* would still pass.
    ///
    /// `x₁`'s curvature is 2e-8 against `x₀`'s 4 throughout the pencil, so
    /// `x₁` must be ~10⁴ times larger than `x₀` for either form to be
    /// O(1); the scaled variable `x̃₁ = d₁·x₁` is only O(1) if `d₁ ≪ d₀`.
    #[test]
    fn x_factors_invert_d() {
        let prob = parse_nl_text(SPREAD_NL).expect("parse");
        let sc = curvature_scaling(&prob).expect("quadratic");
        assert!(
            sc.x[1] < sc.x[0],
            "the small-coefficient variable must be shrunk, not grown: \
             d = {:?}",
            sc.x
        );
        // Eight orders in the coefficients is four in the variable, and
        // the ratio should be that big rather than a rounding away from 1.
        assert!(
            sc.x[0] / sc.x[1] > 1e2,
            "expected a large ratio, got {:?}",
            sc.x
        );
    }

    /// Stage 2's whole content: after the row is scaled, the largest of
    /// its three magnitudes is 1. Checked against the same three terms the
    /// formula is built from, recomputed here from the model rather than
    /// carried over from the implementation.
    #[test]
    fn the_row_scale_normalizes_the_row() {
        let prob = parse_nl_text(SPREAD_NL).expect("parse");
        let sc = curvature_scaling(&prob).expect("quadratic");
        // Recover D from the emitted reciprocal.
        let d: Vec<f64> = sc.x.iter().map(|v| 1.0 / v).collect();
        let (hess, nl_lin, nl_const) = prob.con_nonlinear[0]
            .analyze_quadratic_full()
            .expect("quadratic row");
        let mut row_sum: BTreeMap<usize, f64> = BTreeMap::new();
        for (&(r, c), v) in &hess {
            let s = (d[r] * v * d[c]).abs();
            *row_sum.entry(r).or_insert(0.0) += s;
            if r != c {
                *row_sum.entry(c).or_insert(0.0) += s;
            }
        }
        let q = row_sum.values().fold(0.0_f64, |m, &v| m.max(v));
        let a = row_linear(&prob, 0, &nl_lin)
            .iter()
            .fold(0.0_f64, |m, (&j, v)| m.max((v * d[j]).abs()));
        let b = row_rhs(&prob, 0, nl_const);
        let scaled_max = q.max(a).max(b) * sc.g[0];
        assert!(
            (scaled_max - 1.0).abs() < 1e-12,
            "scaled row magnitude should be exactly 1, got {scaled_max}"
        );
    }

    /// `quad_row_coefs` reads the coefficients, not a point: `Q = diag(4,
    /// 2e-8)` gives `‖Q‖_∞ = 4`, the row carries no linear part, and the
    /// right-hand side is the one written in the file.
    #[test]
    fn quad_row_coefs_reads_the_file() {
        let prob = parse_nl_text(SPREAD_NL).expect("parse");
        let rows = quad_row_coefs(&prob);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.index, 0);
        assert!((r.curvature - 4.0).abs() < 1e-12);
        assert_eq!(r.linear, 0.0);
        assert!((r.rhs - 1.0e5).abs() < 1e-9);
    }

    /// gh #703 / gh#483: the `quadratic` flag distinguishes a model the
    /// scheme *read curvature from* from one that merely satisfies its
    /// degree-≤2 precondition. An LP is degree ≤ 2 with every `Qᵢ` empty,
    /// so `curvature_scaling` returns factors — good ones, but the ones
    /// plain Ruiz equilibration of `[A b]` would give, because there was no
    /// second-order coefficient anywhere to read. The CLI spends the convex
    /// fast path on this option only when the answer here is `true`; see
    /// `decline_convex_for_curvature_scaling` in `pounce-cli`.
    #[test]
    fn quadratic_is_false_exactly_when_no_second_order_coefficient_exists() {
        // A pure LP: `min x0 + x1` s.t. `x0 + x1 >= 1`, `x0, x1 >= 0`.
        let lp = "\
g3 0 1 0
 2 1 1 0 0
 0 0
 0 0
 0 0 0
 0 0 0 1
 0 0 0 0 0
 2 2
 0 0
 0 0 0 0 0
C0
n0
O0 0
n0
x2
0 0
1 0
r
2 1
b
2 0
2 0
k1
2
J0 2
0 1
1 1
G0 2
0 1
1 1
";
        let prob = crate::nl_reader::parse_nl_text(lp).expect("parse LP");
        let sc = curvature_scaling(&prob).expect("an LP is degree <= 2");
        assert!(
            !sc.quadratic,
            "an LP has no `Q` to read; factors {:?} / {:?}",
            sc.x, sc.g
        );

        // The same model with a quadratic objective bolted on is the other
        // side of the same test: identical rows, one nonzero second-order
        // coefficient, and the flag flips.
        let qp = lp.replace(
            "O0 0
n0",
            "O0 0
o5
v0
n2",
        );
        let prob = crate::nl_reader::parse_nl_text(&qp).expect("parse QP");
        let sc = curvature_scaling(&prob).expect("x0^2 is degree 2");
        assert!(sc.quadratic, "`x0^2` is a second-order coefficient");
    }
}
