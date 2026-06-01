//! αBB convex underestimators.
//!
//! For any twice-differentiable `f` on a box, `L(x) = f(x) − α·Σᵢ (xᵢ−lᵢ)(uᵢ−xᵢ)`
//! is a *convex* underestimator once `α ≥ max(0, −½·λ_min(∇²f))` over the box
//! (the added term is ≥ 0 on the box, so `L ≤ f`, and its Hessian `+2α·I`
//! convexifies `f`). αBB needs no factorable structure — it underestimates the
//! expression as a whole — so it complements the term-wise McCormick/envelope
//! relaxation, and the two together are tighter than either.
//!
//! The spectral shift `α` is computed *rigorously*: a second-order interval
//! forward sweep over the tape encloses every Hessian entry `∂²f/∂xᵢ∂xⱼ` across
//! the box, and a scaled-Gershgorin bound turns that into a valid `λ_min`
//! lower bound. The convex `L` is then linearized at sample points into tangent-
//! plane cuts on the expression's LP column — each a valid global
//! underestimator of `f`, so adding them only tightens the bound.

// The interval-Hessian sweep is matrix math over `0..n` indices on several
// parallel arrays (`g[i]`, `h[i][j]`, `child.g[i]`); explicit indexing reads
// far clearer here than zipped iterators.
#![allow(clippy::needless_range_loop)]

use crate::ad::gradient;
use crate::expr::eval;
use pounce_nlp::{FbbtOp, FbbtTape};
use pounce_presolve::fbbt::Interval;

/// Per-slot second-order interval AD payload: value, gradient, and Hessian
/// (the latter two over the `n` problem variables), all as interval enclosures.
struct Jet {
    v: Interval,
    g: Vec<Interval>,
    h: Vec<Vec<Interval>>,
}

fn zero_mat(n: usize) -> Vec<Vec<Interval>> {
    vec![vec![Interval::point(0.0); n]; n]
}

/// Chain rule for a unary atom `φ(child)` with first/second derivatives
/// `dphi`, `d2phi` (interval extensions evaluated at the child's value).
fn unary(child: &Jet, val: Interval, dphi: Interval, d2phi: Interval, n: usize) -> Jet {
    let mut g = vec![Interval::point(0.0); n];
    let mut h = zero_mat(n);
    for i in 0..n {
        g[i] = dphi.mul(child.g[i]);
    }
    for i in 0..n {
        for j in 0..n {
            // φ''·g_i·g_j + φ'·H_ij
            h[i][j] = d2phi
                .mul(child.g[i])
                .mul(child.g[j])
                .add(dphi.mul(child.h[i][j]));
        }
    }
    Jet { v: val, g, h }
}

/// Interval Hessian of the tape's root over the box `[lo, hi]`, or `None` if a
/// domain is invalid (e.g. `ln`/`÷` over an interval touching 0) or any entry
/// is unbounded — αBB then declines.
fn interval_hessian(tape: &FbbtTape, lo: &[f64], hi: &[f64]) -> Option<Vec<Vec<Interval>>> {
    let n = lo.len();
    let mut jets: Vec<Jet> = Vec::with_capacity(tape.ops.len());
    let recip = |x: Interval| -> Option<(Interval, Interval, Interval)> {
        // 1/x and its first two derivatives; None if x straddles 0.
        if x.contains_zero() {
            return None;
        }
        let inv = Interval::point(1.0).div(x);
        let d1 = Interval::point(-1.0).div(x.mul(x));
        let d2 = Interval::point(2.0).div(x.mul(x).mul(x));
        Some((inv, d1, d2))
    };

    for op in &tape.ops {
        let jet = match *op {
            FbbtOp::Const(c) => Jet {
                v: Interval::point(c),
                g: vec![Interval::point(0.0); n],
                h: zero_mat(n),
            },
            FbbtOp::Var(k) => {
                let mut g = vec![Interval::point(0.0); n];
                g[k] = Interval::point(1.0);
                Jet {
                    v: Interval::new(lo[k], hi[k]),
                    g,
                    h: zero_mat(n),
                }
            }
            FbbtOp::Add(a, b) | FbbtOp::Sub(a, b) => {
                let (ja, jb) = (&jets[a], &jets[b]);
                let sub = matches!(*op, FbbtOp::Sub(_, _));
                let mut g = vec![Interval::point(0.0); n];
                let mut h = zero_mat(n);
                for i in 0..n {
                    g[i] = if sub {
                        ja.g[i].sub(jb.g[i])
                    } else {
                        ja.g[i].add(jb.g[i])
                    };
                    for j in 0..n {
                        h[i][j] = if sub {
                            ja.h[i][j].sub(jb.h[i][j])
                        } else {
                            ja.h[i][j].add(jb.h[i][j])
                        };
                    }
                }
                let v = if sub { ja.v.sub(jb.v) } else { ja.v.add(jb.v) };
                Jet { v, g, h }
            }
            FbbtOp::Neg(a) => {
                let ja = &jets[a];
                let mut g = vec![Interval::point(0.0); n];
                let mut h = zero_mat(n);
                for i in 0..n {
                    g[i] = ja.g[i].neg();
                    for j in 0..n {
                        h[i][j] = ja.h[i][j].neg();
                    }
                }
                Jet {
                    v: ja.v.neg(),
                    g,
                    h,
                }
            }
            FbbtOp::Mul(a, b) => mul_jet(&jets[a], &jets[b], n),
            FbbtOp::Div(a, b) => {
                let (inv, d1, d2) = recip(jets[b].v)?;
                let rb = unary(&jets[b], inv, d1, d2, n); // 1/b with its derivatives
                mul_jet(&jets[a], &rb, n)
            }
            FbbtOp::PowInt(a, p) => {
                let c = &jets[a];
                let pf = p as f64;
                let val = c.v.pow_uint(p);
                let d1 = Interval::point(pf).mul(pow_i(c.v, p as i32 - 1)?);
                let d2 = Interval::point(pf * (pf - 1.0)).mul(pow_i(c.v, p as i32 - 2)?);
                unary(c, val, d1, d2, n)
            }
            FbbtOp::Exp(a) => {
                let c = &jets[a];
                let e = c.v.exp();
                unary(c, e, e, e, n)
            }
            FbbtOp::Ln(a) => {
                let c = &jets[a];
                if c.v.lo <= 0.0 {
                    return None;
                }
                let d1 = Interval::point(1.0).div(c.v);
                let d2 = Interval::point(-1.0).div(c.v.mul(c.v));
                unary(c, c.v.ln(), d1, d2, n)
            }
            FbbtOp::Sqrt(a) => {
                let c = &jets[a];
                if c.v.lo <= 0.0 {
                    return None;
                }
                let s = c.v.sqrt();
                let d1 = Interval::point(0.5).div(s);
                let d2 = Interval::point(-0.25).div(s.mul(c.v));
                unary(c, s, d1, d2, n)
            }
            FbbtOp::Sin(a) => {
                let c = &jets[a];
                unary(c, c.v.sin(), c.v.cos(), c.v.sin().neg(), n)
            }
            FbbtOp::Cos(a) => {
                let c = &jets[a];
                unary(c, c.v.cos(), c.v.sin().neg(), c.v.cos().neg(), n)
            }
            FbbtOp::Abs(_) | FbbtOp::Opaque => return None, // not twice-differentiable
        };
        jets.push(jet);
    }

    let root = jets.last()?;
    // Reject if any Hessian entry is non-finite (α would be unbounded).
    for row in &root.h {
        for e in row {
            if !e.lo.is_finite() || !e.hi.is_finite() {
                return None;
            }
        }
    }
    Some(root.h.clone())
}

fn mul_jet(ja: &Jet, jb: &Jet, n: usize) -> Jet {
    let mut g = vec![Interval::point(0.0); n];
    let mut h = zero_mat(n);
    for i in 0..n {
        g[i] = ja.g[i].mul(jb.v).add(ja.v.mul(jb.g[i]));
    }
    for i in 0..n {
        for j in 0..n {
            // H_ij·b + g_i^a g_j^b + g_i^b g_j^a + a·H_ij^b
            h[i][j] = ja.h[i][j]
                .mul(jb.v)
                .add(ja.g[i].mul(jb.g[j]))
                .add(jb.g[i].mul(ja.g[j]))
                .add(ja.v.mul(jb.h[i][j]));
        }
    }
    Jet {
        v: ja.v.mul(jb.v),
        g,
        h,
    }
}

/// `x^p` for a possibly-negative integer exponent, as an interval; `None` if it
/// requires dividing by an interval that straddles 0.
fn pow_i(x: Interval, p: i32) -> Option<Interval> {
    if p >= 0 {
        Some(x.pow_uint(p as u32))
    } else {
        if x.contains_zero() {
            return None;
        }
        Some(Interval::point(1.0).div(x.pow_uint((-p) as u32)))
    }
}

/// Scaled-Gershgorin lower bound on `λ_min` of the interval Hessian.
fn lambda_min_lb(h: &[Vec<Interval>]) -> f64 {
    let n = h.len();
    let mut lo = f64::INFINITY;
    for i in 0..n {
        let radius: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| h[i][j].lo.abs().max(h[i][j].hi.abs()))
            .sum();
        lo = lo.min(h[i][i].lo - radius);
    }
    lo
}

/// αBB tangent-plane underestimator cuts of `tape` on the box, as
/// `(row terms over [obj_col] ∪ variable columns, rhs)` in `≤` form. Returns
/// empty if αBB declines (non-smooth atom, invalid domain, unbounded Hessian).
pub(crate) fn objective_cuts(
    tape: &FbbtTape,
    lo: &[f64],
    hi: &[f64],
    obj_col: usize,
    samples: usize,
) -> Vec<(Vec<(usize, f64)>, f64)> {
    let n = lo.len();
    if samples == 0 || n == 0 {
        return Vec::new();
    }
    let Some(hess) = interval_hessian(tape, lo, hi) else {
        return Vec::new();
    };
    let lam = lambda_min_lb(&hess);
    if !lam.is_finite() {
        return Vec::new();
    }
    let alpha = (-0.5 * lam).max(0.0);

    // Sample points: box center, then a deterministic spread toward the corners.
    let mut pts: Vec<Vec<f64>> = Vec::new();
    pts.push((0..n).map(|i| 0.5 * (lo[i] + hi[i])).collect());
    for s in 1..samples {
        let frac = s as f64 / samples as f64;
        pts.push((0..n).map(|i| lo[i] + frac * (hi[i] - lo[i])).collect());
    }

    let mut cuts = Vec::new();
    for x0 in pts {
        let f0 = eval(tape, &x0);
        if !f0.is_finite() {
            continue;
        }
        let gf = gradient(tape, &x0, n);
        // L(x0) and ∇L(x0): the bilinear correction is −α Σ (x_i−l_i)(u_i−x_i).
        let mut l0 = f0;
        let mut grad_l = gf.clone();
        for i in 0..n {
            l0 -= alpha * (x0[i] - lo[i]) * (hi[i] - x0[i]);
            grad_l[i] = gf[i] - alpha * (lo[i] + hi[i] - 2.0 * x0[i]);
        }
        if !l0.is_finite() || grad_l.iter().any(|g| !g.is_finite()) {
            continue;
        }
        // w_obj ≥ L(x0) + ∇L·(x − x0)  ⇔  Σ ∇L_i·x_i − w_obj ≤ Σ ∇L_i·x0_i − L(x0)
        let mut terms = vec![(obj_col, -1.0)];
        let mut rhs = -l0;
        for i in 0..n {
            terms.push((i, grad_l[i]));
            rhs += grad_l[i] * x0[i];
        }
        cuts.push((terms, rhs));
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::var;

    #[test]
    fn alpha_zero_for_convex() {
        // x² + y²: Hessian 2I, λ_min = 2 ⇒ α = 0 (already convex).
        let tape = (var(0).powi(2) + var(1).powi(2)).to_tape();
        let h = interval_hessian(&tape, &[-1.0, -1.0], &[1.0, 1.0]).unwrap();
        assert!(lambda_min_lb(&h) > 1.0, "λ_min lb = {}", lambda_min_lb(&h));
    }

    #[test]
    fn nonconvex_gets_positive_alpha() {
        // x·y: Hessian [[0,1],[1,0]], λ_min = −1 ⇒ α = 0.5.
        let tape = (var(0) * var(1)).to_tape();
        let h = interval_hessian(&tape, &[-1.0, -1.0], &[1.0, 1.0]).unwrap();
        let lam = lambda_min_lb(&h);
        assert!(lam < 0.0, "λ_min lb = {lam}");
        assert!(((-0.5 * lam) - 0.5).abs() < 1e-9, "α = {}", -0.5 * lam);
    }

    #[test]
    fn cuts_underestimate_objective() {
        // Validate every cut lies ≤ f on a grid: f = x·y on [−1,1]².
        let tape = (var(0) * var(1)).to_tape();
        let lo = [-1.0, -1.0];
        let hi = [1.0, 1.0];
        // Treat obj_col as a notional column index 2 (vars are 0,1).
        let cuts = objective_cuts(&tape, &lo, &hi, 2, 3);
        assert!(!cuts.is_empty());
        for gx in -4..=4 {
            for gy in -4..=4 {
                let x = gx as f64 / 4.0;
                let y = gy as f64 / 4.0;
                let f = x * y;
                for (terms, rhs) in &cuts {
                    // Σ terms·col ≤ rhs must hold with w_obj = f (the true value),
                    // i.e. the plane underestimates f.
                    let mut lhs = 0.0;
                    for &(c, coeff) in terms {
                        lhs += coeff
                            * if c == 0 {
                                x
                            } else if c == 1 {
                                y
                            } else {
                                f
                            };
                    }
                    assert!(lhs <= *rhs + 1e-9, "cut violated at ({x},{y})");
                }
            }
        }
    }
}
