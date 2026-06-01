//! Reverse-mode automatic differentiation of an [`FbbtTape`].
//!
//! A forward sweep records every slot's value; a reverse sweep propagates
//! adjoints back to the variable leaves. Used to feed exact first derivatives
//! (objective gradient, constraint Jacobian) to the local NLP solve that
//! produces upper bounds; the Lagrangian Hessian is then finite-differenced
//! from these gradients in [`crate::nlp`].

use pounce_nlp::{FbbtOp, FbbtTape};

/// Forward sweep: value at every tape slot (`out[k]` = slot `k`'s value).
pub(crate) fn forward_vals(tape: &FbbtTape, x: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let r = match *op {
            FbbtOp::Const(c) => c,
            FbbtOp::Var(i) => x[i],
            FbbtOp::Add(a, b) => v[a] + v[b],
            FbbtOp::Sub(a, b) => v[a] - v[b],
            FbbtOp::Mul(a, b) => v[a] * v[b],
            FbbtOp::Div(a, b) => v[a] / v[b],
            FbbtOp::PowInt(a, n) => v[a].powi(n as i32),
            FbbtOp::Neg(a) => -v[a],
            FbbtOp::Sqrt(a) => v[a].sqrt(),
            FbbtOp::Exp(a) => v[a].exp(),
            FbbtOp::Ln(a) => v[a].ln(),
            FbbtOp::Abs(a) => v[a].abs(),
            FbbtOp::Sin(a) => v[a].sin(),
            FbbtOp::Cos(a) => v[a].cos(),
            FbbtOp::Opaque => f64::NAN,
        };
        v.push(r);
    }
    v
}

/// Accumulate `seed · ∂(tape)/∂x_i` into `grad[i]` for every variable `i` the
/// tape references. `grad` is **not** zeroed (gradients can be chained).
pub(crate) fn accumulate_gradient(tape: &FbbtTape, x: &[f64], seed: f64, grad: &mut [f64]) {
    if tape.ops.is_empty() || seed == 0.0 {
        return;
    }
    let v = forward_vals(tape, x);
    let mut adj = vec![0.0; tape.ops.len()];
    if let Some(last) = adj.last_mut() {
        *last = seed;
    }

    for k in (0..tape.ops.len()).rev() {
        let a = adj[k];
        if a == 0.0 {
            continue;
        }
        match tape.ops[k] {
            FbbtOp::Const(_) => {}
            FbbtOp::Var(i) => grad[i] += a,
            FbbtOp::Add(p, q) => {
                adj[p] += a;
                adj[q] += a;
            }
            FbbtOp::Sub(p, q) => {
                adj[p] += a;
                adj[q] -= a;
            }
            FbbtOp::Mul(p, q) => {
                adj[p] += a * v[q];
                adj[q] += a * v[p];
            }
            FbbtOp::Div(p, q) => {
                adj[p] += a / v[q];
                adj[q] -= a * v[p] / (v[q] * v[q]);
            }
            FbbtOp::PowInt(p, n) => {
                if n >= 1 {
                    adj[p] += a * n as f64 * v[p].powi(n as i32 - 1);
                }
            }
            FbbtOp::Neg(p) => adj[p] -= a,
            FbbtOp::Sqrt(p) => adj[p] += a * 0.5 / v[p].max(1e-300).sqrt(),
            FbbtOp::Exp(p) => adj[p] += a * v[k], // v[k] = exp(v[p])
            FbbtOp::Ln(p) => adj[p] += a / v[p],
            FbbtOp::Abs(p) => adj[p] += a * v[p].signum(),
            FbbtOp::Sin(p) => adj[p] += a * v[p].cos(),
            FbbtOp::Cos(p) => adj[p] -= a * v[p].sin(),
            FbbtOp::Opaque => {}
        }
    }
}

/// Gradient of `tape` at `x` into a fresh length-`n` vector.
pub(crate) fn gradient(tape: &FbbtTape, x: &[f64], n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n];
    accumulate_gradient(tape, x, 1.0, &mut g);
    g
}

/// Variables (0-based) that `tape` references, ascending and deduplicated.
pub(crate) fn referenced_vars(tape: &FbbtTape) -> Vec<usize> {
    let mut vs: Vec<usize> = tape
        .ops
        .iter()
        .filter_map(|op| match op {
            FbbtOp::Var(i) => Some(*i),
            _ => None,
        })
        .collect();
    vs.sort_unstable();
    vs.dedup();
    vs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{con, var};

    #[test]
    fn gradient_of_product_and_power() {
        // f = x0² · x1 + 3·x1.  ∇f = (2 x0 x1, x0² + 3).
        let f = (var(0).powi(2) * var(1)) + con(3.0) * var(1);
        let tape = f.to_tape();
        let g = gradient(&tape, &[2.0, 5.0], 2);
        assert!((g[0] - 2.0 * 2.0 * 5.0).abs() < 1e-9, "{g:?}");
        assert!((g[1] - (4.0 + 3.0)).abs() < 1e-9, "{g:?}");
    }

    #[test]
    fn gradient_of_transcendental() {
        // f = exp(x0) + ln(x1).  ∇f = (exp(x0), 1/x1).
        let f = var(0).exp() + var(1).ln();
        let tape = f.to_tape();
        let g = gradient(&tape, &[0.5, 4.0], 2);
        assert!((g[0] - 0.5_f64.exp()).abs() < 1e-9, "{g:?}");
        assert!((g[1] - 0.25).abs() < 1e-9, "{g:?}");
    }

    #[test]
    fn referenced_vars_dedup() {
        let f = var(2) * var(0) + var(2);
        assert_eq!(referenced_vars(&f.to_tape()), vec![0, 2]);
    }
}
