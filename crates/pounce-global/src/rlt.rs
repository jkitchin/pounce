//! Reformulation–Linearization Technique (RLT), level 1.
//!
//! For a linear constraint `αᵀx ≤ β` and a variable bound factor `x_k − l_k ≥ 0`
//! (or `u_k − x_k ≥ 0`), the product `(β − αᵀx)(x_k − l_k) ≥ 0` is valid. It is
//! quadratic, but replacing each `x_i·x_k` with an auxiliary variable `p_{ik}`
//! (itself bounded by the four McCormick inequalities of that product) turns it
//! into a *linear* cut coupling the variables far more tightly than the bound
//! factors alone. This is what lets RLT-based global solvers close bilinearly
//! coupled problems (pooling, QCQP) fast.
//!
//! We detect the affine constraints, multiply each by every variable's two
//! bound factors, and append the linearized cuts (sharing one `p_{ik}` column
//! across all cuts that need it). Nonlinear constraints and pure box problems
//! contribute nothing, so RLT is a no-op there.

use crate::problem::GlobalProblem;
use pounce_convex::{QpProblem, Triplet};
use pounce_nlp::{FbbtOp, FbbtTape};
use std::collections::{BTreeMap, HashMap};

const INF: f64 = 1e20;

/// If `tape` is affine in the `n` variables, return `(coeffs, constant)` with
/// `value = coeffs·x + constant`; otherwise `None`.
fn affine_form(tape: &FbbtTape, n: usize) -> Option<(Vec<f64>, f64)> {
    let is_const = |f: &(Vec<f64>, f64)| f.0.iter().all(|&c| c == 0.0);
    let mut forms: Vec<(Vec<f64>, f64)> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let form = match *op {
            FbbtOp::Const(c) => (vec![0.0; n], c),
            FbbtOp::Var(i) => {
                let mut a = vec![0.0; n];
                a[i] = 1.0;
                (a, 0.0)
            }
            FbbtOp::Add(a, b) => {
                let (fa, fb) = (&forms[a], &forms[b]);
                (
                    fa.0.iter().zip(&fb.0).map(|(x, y)| x + y).collect(),
                    fa.1 + fb.1,
                )
            }
            FbbtOp::Sub(a, b) => {
                let (fa, fb) = (&forms[a], &forms[b]);
                (
                    fa.0.iter().zip(&fb.0).map(|(x, y)| x - y).collect(),
                    fa.1 - fb.1,
                )
            }
            FbbtOp::Neg(a) => (forms[a].0.iter().map(|x| -x).collect(), -forms[a].1),
            FbbtOp::Mul(a, b) => {
                let (fa, fb) = (&forms[a], &forms[b]);
                if is_const(fa) {
                    (fb.0.iter().map(|x| x * fa.1).collect(), fb.1 * fa.1)
                } else if is_const(fb) {
                    (fa.0.iter().map(|x| x * fb.1).collect(), fa.1 * fb.1)
                } else {
                    return None; // genuine bilinear term
                }
            }
            FbbtOp::Div(a, b) => {
                let fb = &forms[b];
                if is_const(fb) && fb.1 != 0.0 {
                    (
                        forms[a].0.iter().map(|x| x / fb.1).collect(),
                        forms[a].1 / fb.1,
                    )
                } else {
                    return None;
                }
            }
            FbbtOp::PowInt(a, 0) => (vec![0.0; n], forms[a].1.powi(0).max(1.0)),
            FbbtOp::PowInt(a, 1) => forms[a].clone(),
            _ => return None, // any higher power / transcendental ⇒ nonlinear
        };
        forms.push(form);
    }
    forms.pop()
}

/// Append RLT product columns + McCormick + bound-factor cuts to `qp`.
pub(crate) fn augment(qp: &mut QpProblem, prob: &GlobalProblem, lo: &[f64], hi: &[f64]) {
    let n = prob.n_vars;

    // Affine constraints → linear inequalities `αᵀx ≤ β`.
    let mut lins: Vec<(Vec<f64>, f64)> = Vec::new();
    for con in &prob.constraints {
        if let Some((a, c)) = affine_form(&con.tape, n) {
            if con.hi < INF {
                lins.push((a.clone(), con.hi - c));
            }
            if con.lo > -INF {
                lins.push((a.iter().map(|v| -v).collect(), -(con.lo - c)));
            }
        }
    }
    if lins.is_empty() {
        return;
    }

    // Product columns p_{ik} = x_i·x_k (i ≤ k), bounded by McCormick, created on
    // demand and shared across cuts.
    let mut prod: HashMap<(usize, usize), usize> = HashMap::new();

    for (alpha, beta) in &lins {
        for k in 0..n {
            // Two bound factors give two cuts; build each as Σ coeff·col ≤ rhs.
            // A: (β − αᵀx)(x_k − l_k) ≥ 0
            //    Σ αᵢ p_{ik} − β x_k − l_k Σ αᵢ xᵢ ≤ −β l_k
            // B: (β − αᵀx)(u_k − x_k) ≥ 0
            //    β x_k + u_k Σ αᵢ xᵢ − Σ αᵢ p_{ik} ≤ β u_k
            let mut a_row: BTreeMap<usize, f64> = BTreeMap::new();
            let mut b_row: BTreeMap<usize, f64> = BTreeMap::new();
            for (i, &ai) in alpha.iter().enumerate() {
                if ai == 0.0 {
                    continue;
                }
                let p = product_col(qp, &mut prod, lo, hi, i, k);
                *a_row.entry(p).or_insert(0.0) += ai;
                *a_row.entry(i).or_insert(0.0) -= lo[k] * ai;
                *b_row.entry(p).or_insert(0.0) -= ai;
                *b_row.entry(i).or_insert(0.0) += hi[k] * ai;
            }
            *a_row.entry(k).or_insert(0.0) -= beta;
            *b_row.entry(k).or_insert(0.0) += beta;
            push_row(qp, &a_row, -beta * lo[k]);
            push_row(qp, &b_row, beta * hi[k]);
        }
    }
}

/// Fetch (or create with McCormick bounds) the column for `x_i·x_k`.
fn product_col(
    qp: &mut QpProblem,
    prod: &mut HashMap<(usize, usize), usize>,
    lo: &[f64],
    hi: &[f64],
    i: usize,
    k: usize,
) -> usize {
    let key = (i.min(k), i.max(k));
    if let Some(&c) = prod.get(&key) {
        return c;
    }
    let (u, v) = key; // p = x_u · x_v
    let (ul, uu, vl, vu) = (lo[u], hi[u], lo[v], hi[v]);
    let cands = [ul * vl, ul * vu, uu * vl, uu * vu];
    let pl = cands.iter().copied().fold(f64::INFINITY, f64::min);
    let pu = cands.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let col = qp.n;
    qp.n += 1;
    qp.c.push(0.0);
    qp.lb.push(pl.clamp(-INF, INF));
    qp.ub.push(pu.clamp(-INF, INF));
    prod.insert(key, col);

    // McCormick of p = x_u·x_v (skip if a factor range is non-finite).
    if [ul, uu, vl, vu].iter().all(|x| x.is_finite()) {
        let cuts = [
            (vec![(col, -1.0), (v, ul), (u, vl)], ul * vl),
            (vec![(col, -1.0), (v, uu), (u, vu)], uu * vu),
            (vec![(col, 1.0), (v, -uu), (u, -vl)], -uu * vl),
            (vec![(col, 1.0), (v, -ul), (u, -vu)], -ul * vu),
        ];
        for (terms, rhs) in cuts {
            let mut row: BTreeMap<usize, f64> = BTreeMap::new();
            for (c, val) in terms {
                *row.entry(c).or_insert(0.0) += val;
            }
            push_row(qp, &row, rhs);
        }
    }
    col
}

fn push_row(qp: &mut QpProblem, row: &BTreeMap<usize, f64>, rhs: f64) {
    let r = qp.h.len();
    let mut any = false;
    for (&c, &v) in row {
        if v != 0.0 {
            qp.g.push(Triplet::new(r, c, v));
            any = true;
        }
    }
    if any {
        qp.h.push(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{con, var};

    #[test]
    fn detects_affine() {
        let e = 2.0 * var(0) - 3.0 * var(1) + con(5.0);
        let (a, c) = affine_form(&e.to_tape(), 2).unwrap();
        assert_eq!(a, vec![2.0, -3.0]);
        assert_eq!(c, 5.0);
    }

    #[test]
    fn rejects_nonlinear() {
        assert!(affine_form(&(var(0) * var(1)).to_tape(), 2).is_none());
        assert!(affine_form(&var(0).powi(2).to_tape(), 1).is_none());
    }
}
