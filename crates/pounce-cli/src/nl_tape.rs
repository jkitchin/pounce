//! Flat-tape reverse-mode AD for `.nl` expression trees.
//!
//! Replaces the FD-based Hessian path with a port of the tape AD used
//! in `ripopt::nl::autodiff`. The tape is a `Vec<TapeOp>` where each op
//! refers to its operands by tape-slot index; forward evaluation runs
//! through the slice once filling a parallel `Vec<f64>` of values, and
//! reverse-mode adjoints walk the same buffer backwards.
//!
//! Sparse Hessians are computed by forward-over-reverse: for each
//! variable `j` that the tape depends on, run a forward tangent sweep
//! seeded with `e_j`, then a second-order reverse sweep that produces
//! column `j` of the Hessian. The caller supplies a `(row, col) -> nnz
//! position` map (lower triangle, row >= col), and contributions are
//! accumulated in place — the outer loop in `eval_h` calls the same
//! map for the objective and every active constraint, so every
//! Lagrangian term lands in the right slot.
//!
//! Common subexpressions are tape-emitted **once**: when the recursive
//! builder hits `Expr::Cse(rc)` it keys on the `Rc` pointer identity,
//! emitting the body the first time and returning the cached
//! result-slot index on subsequent references. The forward pass then
//! computes each CSE once and the reverse pass folds adjoints from
//! every reference into a single slot — exact chain-rule behaviour.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::nl_reader::{BinOp, Expr, UnaryOp};

/// One operation in the flattened tape. Operand fields are tape-slot
/// indices into the same tape; `Var(i)` references problem variable
/// index `i` (read from the input `x` slice during forward).
#[derive(Debug, Clone)]
pub enum TapeOp {
    Const(f64),
    Var(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    Neg(usize),
    Abs(usize),
    Sqrt(usize),
    Exp(usize),
    Log(usize),
    Log10(usize),
    Sin(usize),
    Cos(usize),
}

/// A flattened expression tape. The result of evaluation is the value
/// at slot `ops.len() - 1` (i.e. the last op).
#[derive(Debug, Clone)]
pub struct Tape {
    pub ops: Vec<TapeOp>,
}

impl Tape {
    /// Build a tape from an `Expr` tree. CSE bodies (`Expr::Cse(rc)`)
    /// are cached by `Rc` pointer identity so each body is emitted
    /// once even when referenced many times.
    pub fn build(expr: &Expr) -> Self {
        let mut ops = Vec::new();
        let mut cache: HashMap<*const Expr, usize> = HashMap::new();
        build_recursive(expr, &mut ops, &mut cache);
        Tape { ops }
    }

    /// Forward sweep: returns `vals[i] = value of tape slot i`. The
    /// scalar tape result is `vals[ops.len() - 1]`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut vals: Vec<f64> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match op {
                TapeOp::Const(c) => *c,
                TapeOp::Var(i) => x[*i],
                TapeOp::Add(a, b) => vals[*a] + vals[*b],
                TapeOp::Sub(a, b) => vals[*a] - vals[*b],
                TapeOp::Mul(a, b) => vals[*a] * vals[*b],
                TapeOp::Div(a, b) => vals[*a] / vals[*b],
                TapeOp::Pow(a, b) => vals[*a].powf(vals[*b]),
                TapeOp::Neg(a) => -vals[*a],
                TapeOp::Abs(a) => vals[*a].abs(),
                TapeOp::Sqrt(a) => vals[*a].sqrt(),
                TapeOp::Exp(a) => vals[*a].exp(),
                TapeOp::Log(a) => vals[*a].ln(),
                TapeOp::Log10(a) => vals[*a].log10(),
                TapeOp::Sin(a) => vals[*a].sin(),
                TapeOp::Cos(a) => vals[*a].cos(),
            };
            vals.push(v);
        }
        vals
    }

    pub fn eval(&self, x: &[f64]) -> f64 {
        let vals = self.forward(x);
        *vals.last().unwrap_or(&0.0)
    }

    /// Reverse-mode AD: accumulate `seed * df/dx_i` into `grad[i]` for
    /// every problem variable `i` referenced by the tape. `grad` is
    /// **not** zeroed by this routine — the caller can chain multiple
    /// gradient accumulations into the same buffer.
    pub fn gradient_seed(&self, x: &[f64], seed: f64, grad: &mut [f64]) {
        if seed == 0.0 || self.ops.is_empty() {
            return;
        }
        let vals = self.forward(x);
        self.reverse(&vals, seed, grad);
    }

    fn reverse(&self, vals: &[f64], seed: f64, grad: &mut [f64]) {
        let n = self.ops.len();
        let mut adj = vec![0.0f64; n];
        adj[n - 1] = seed;

        for i in (0..n).rev() {
            let a = adj[i];
            if a == 0.0 {
                continue;
            }
            match &self.ops[i] {
                TapeOp::Const(_) => {}
                TapeOp::Var(j) => {
                    grad[*j] += a;
                }
                TapeOp::Add(l, r) => {
                    adj[*l] += a;
                    adj[*r] += a;
                }
                TapeOp::Sub(l, r) => {
                    adj[*l] += a;
                    adj[*r] -= a;
                }
                TapeOp::Mul(l, r) => {
                    adj[*l] += a * vals[*r];
                    adj[*r] += a * vals[*l];
                }
                TapeOp::Div(l, r) => {
                    let rv = vals[*r];
                    adj[*l] += a / rv;
                    adj[*r] -= a * vals[*l] / (rv * rv);
                }
                TapeOp::Pow(l, r) => {
                    let lv = vals[*l];
                    let rv = vals[*r];
                    if rv != 0.0 {
                        adj[*l] += a * rv * lv.powf(rv - 1.0);
                    }
                    if lv > 0.0 {
                        adj[*r] += a * vals[i] * lv.ln();
                    }
                }
                TapeOp::Neg(j) => {
                    adj[*j] -= a;
                }
                TapeOp::Abs(j) => {
                    if vals[*j] >= 0.0 {
                        adj[*j] += a;
                    } else {
                        adj[*j] -= a;
                    }
                }
                TapeOp::Sqrt(j) => {
                    let sv = vals[i];
                    if sv > 0.0 {
                        adj[*j] += a * 0.5 / sv;
                    }
                }
                TapeOp::Exp(j) => {
                    adj[*j] += a * vals[i];
                }
                TapeOp::Log(j) => {
                    adj[*j] += a / vals[*j];
                }
                TapeOp::Log10(j) => {
                    adj[*j] += a / (vals[*j] * std::f64::consts::LN_10);
                }
                TapeOp::Sin(j) => {
                    adj[*j] += a * vals[*j].cos();
                }
                TapeOp::Cos(j) => {
                    adj[*j] -= a * vals[*j].sin();
                }
            }
        }
    }

    /// Sorted distinct problem-variable indices that the tape depends on.
    pub fn variables(&self) -> Vec<usize> {
        let mut s: BTreeSet<usize> = BTreeSet::new();
        for op in &self.ops {
            if let TapeOp::Var(j) = op {
                s.insert(*j);
            }
        }
        s.into_iter().collect()
    }

    /// Forward tangent sweep: `dot[i] = d(slot_i) / dx_{seed_var}`.
    /// Caller-supplied `dot` buffer is overwritten in full; no zeroing
    /// needed beforehand because every slot is written before it is
    /// read (the loop walks forward and only reads earlier slots).
    fn forward_tangent(&self, vals: &[f64], seed_var: usize, dot: &mut [f64]) {
        let n = self.ops.len();
        debug_assert_eq!(dot.len(), n);
        for i in 0..n {
            dot[i] = match &self.ops[i] {
                TapeOp::Const(_) => 0.0,
                TapeOp::Var(k) => {
                    if *k == seed_var {
                        1.0
                    } else {
                        0.0
                    }
                }
                TapeOp::Add(a, b) => dot[*a] + dot[*b],
                TapeOp::Sub(a, b) => dot[*a] - dot[*b],
                TapeOp::Mul(a, b) => dot[*a] * vals[*b] + vals[*a] * dot[*b],
                TapeOp::Div(a, b) => {
                    let vb = vals[*b];
                    (dot[*a] * vb - vals[*a] * dot[*b]) / (vb * vb)
                }
                TapeOp::Pow(a, b) => {
                    let u = vals[*a];
                    let r = vals[*b];
                    let du = dot[*a];
                    let dr = dot[*b];
                    let mut result = 0.0;
                    if r != 0.0 && u != 0.0 {
                        result += r * u.powf(r - 1.0) * du;
                    }
                    if u > 0.0 {
                        result += vals[i] * u.ln() * dr;
                    }
                    result
                }
                TapeOp::Neg(a) => -dot[*a],
                TapeOp::Abs(a) => {
                    if vals[*a] >= 0.0 {
                        dot[*a]
                    } else {
                        -dot[*a]
                    }
                }
                TapeOp::Sqrt(a) => {
                    let sv = vals[i];
                    if sv > 0.0 {
                        dot[*a] * 0.5 / sv
                    } else {
                        0.0
                    }
                }
                TapeOp::Exp(a) => dot[*a] * vals[i],
                TapeOp::Log(a) => dot[*a] / vals[*a],
                TapeOp::Log10(a) => dot[*a] / (vals[*a] * std::f64::consts::LN_10),
                TapeOp::Sin(a) => dot[*a] * vals[*a].cos(),
                TapeOp::Cos(a) => -dot[*a] * vals[*a].sin(),
            };
        }
    }

    /// Forward-over-reverse Hessian: for each variable `j` the tape
    /// depends on, accumulate `weight * (d²f / dx_i dx_j)` into
    /// `values[hess_map[(i, j)]]` for every `(i, j)` lower-triangle
    /// pair in the map. The same routine is used for the objective
    /// (with `weight = obj_factor`) and each active constraint (with
    /// `weight = lambda[k]`); contributions sum into the shared map.
    pub fn hessian_accumulate(
        &self,
        x: &[f64],
        weight: f64,
        hess_map: &HashMap<(usize, usize), usize>,
        values: &mut [f64],
    ) {
        let n = self.ops.len();
        if n == 0 || weight == 0.0 {
            return;
        }
        let v = self.forward(x);
        let var_indices = self.variables();

        // Hoist scratch allocations out of the per-variable loop —
        // each was costing O(n) per j on every hessian_accumulate
        // call, which dominated runtime on large tapes (the dense-
        // Hessian Mittelmann problems). `forward_tangent` fully
        // overwrites `dot`, so no reset is needed there. `adj` and
        // `adj_dot` are mutated additively, so we zero them per j.
        let mut dot = vec![0.0f64; n];
        let mut adj = vec![0.0f64; n];
        let mut adj_dot = vec![0.0f64; n];
        for &j in &var_indices {
            self.forward_tangent(&v, j, &mut dot);

            // adj[i] = standard adjoint (∂f/∂slot_i)
            // adj_dot[i] = derivative of adj[i] w.r.t. x_j = ∂²f/(∂slot_i ∂x_j)
            adj.fill(0.0);
            adj_dot.fill(0.0);
            adj[n - 1] = 1.0;

            for i in (0..n).rev() {
                let w = adj[i];
                let wd = adj_dot[i];
                if w == 0.0 && wd == 0.0 {
                    continue;
                }
                match &self.ops[i] {
                    TapeOp::Const(_) => {}
                    TapeOp::Var(k) => {
                        // Lower-triangle: only emit when row k >= col j
                        // so an off-diagonal pair appears once.
                        if wd != 0.0 && *k >= j {
                            if let Some(&pos) = hess_map.get(&(*k, j)) {
                                values[pos] += weight * wd;
                            }
                        }
                    }
                    TapeOp::Add(a, b) => {
                        adj[*a] += w;
                        adj[*b] += w;
                        adj_dot[*a] += wd;
                        adj_dot[*b] += wd;
                    }
                    TapeOp::Sub(a, b) => {
                        adj[*a] += w;
                        adj[*b] -= w;
                        adj_dot[*a] += wd;
                        adj_dot[*b] -= wd;
                    }
                    TapeOp::Mul(a, b) => {
                        adj[*a] += w * v[*b];
                        adj[*b] += w * v[*a];
                        adj_dot[*a] += wd * v[*b] + w * dot[*b];
                        adj_dot[*b] += wd * v[*a] + w * dot[*a];
                    }
                    TapeOp::Div(a, b) => {
                        let vb = v[*b];
                        let vb2 = vb * vb;
                        let vb3 = vb2 * vb;
                        adj[*a] += w / vb;
                        adj_dot[*a] += wd / vb + w * (-dot[*b] / vb2);
                        adj[*b] += w * (-v[*a] / vb2);
                        adj_dot[*b] += wd * (-v[*a] / vb2)
                            + w * (-dot[*a] / vb2 + 2.0 * v[*a] * dot[*b] / vb3);
                    }
                    TapeOp::Pow(a, b) => {
                        let u = v[*a];
                        let r = v[*b];
                        let du = dot[*a];
                        let dr = dot[*b];
                        if r != 0.0 {
                            if u != 0.0 {
                                let p_a = r * u.powf(r - 1.0);
                                adj[*a] += w * p_a;
                                let mut dp_a = dr * u.powf(r - 1.0);
                                if u > 0.0 {
                                    dp_a += r
                                        * u.powf(r - 1.0)
                                        * ((r - 1.0) * du / u + dr * u.ln());
                                } else {
                                    dp_a += r * (r - 1.0) * u.powf(r - 2.0) * du;
                                }
                                adj_dot[*a] += wd * p_a + w * dp_a;
                            } else if r >= 2.0 {
                                let p_a = 0.0;
                                adj[*a] += w * p_a;
                                let dp_a = if r == 2.0 {
                                    2.0 * du
                                } else {
                                    r * (r - 1.0) * (0.0_f64).powf(r - 2.0) * du
                                };
                                adj_dot[*a] += wd * p_a + w * dp_a;
                            }
                        }
                        if u > 0.0 {
                            let ln_u = u.ln();
                            let p_b = v[i] * ln_u;
                            adj[*b] += w * p_b;
                            let dur = v[i] * (r * du / u + dr * ln_u);
                            let dp_b = dur * ln_u + v[i] * du / u;
                            adj_dot[*b] += wd * p_b + w * dp_b;
                        }
                    }
                    TapeOp::Neg(a) => {
                        adj[*a] -= w;
                        adj_dot[*a] -= wd;
                    }
                    TapeOp::Abs(a) => {
                        let s = if v[*a] >= 0.0 { 1.0 } else { -1.0 };
                        adj[*a] += w * s;
                        adj_dot[*a] += wd * s;
                    }
                    TapeOp::Sqrt(a) => {
                        let sv = v[i];
                        if sv > 0.0 {
                            let fp = 0.5 / sv;
                            let fpp = -0.25 / (v[*a] * sv);
                            adj[*a] += w * fp;
                            adj_dot[*a] += wd * fp + w * fpp * dot[*a];
                        }
                    }
                    TapeOp::Exp(a) => {
                        let ev = v[i];
                        adj[*a] += w * ev;
                        adj_dot[*a] += wd * ev + w * ev * dot[*a];
                    }
                    TapeOp::Log(a) => {
                        let u = v[*a];
                        adj[*a] += w / u;
                        adj_dot[*a] += wd / u + w * (-1.0 / (u * u)) * dot[*a];
                    }
                    TapeOp::Log10(a) => {
                        let u = v[*a];
                        let c = std::f64::consts::LN_10;
                        adj[*a] += w / (u * c);
                        adj_dot[*a] += wd / (u * c) + w * (-1.0 / (u * u * c)) * dot[*a];
                    }
                    TapeOp::Sin(a) => {
                        let u = v[*a];
                        let cu = u.cos();
                        adj[*a] += w * cu;
                        adj_dot[*a] += wd * cu + w * (-u.sin()) * dot[*a];
                    }
                    TapeOp::Cos(a) => {
                        let u = v[*a];
                        let su = u.sin();
                        adj[*a] -= w * su;
                        adj_dot[*a] += wd * (-su) + w * (-u.cos()) * dot[*a];
                    }
                }
            }
        }
    }

    /// Structural Hessian sparsity (lower triangle, row >= col).
    /// Propagates per-slot variable-dependence sets forward; each
    /// nonlinear op emits the cross/self products of its operand sets.
    /// Linear ops contribute no second-derivative pairs.
    pub fn hessian_sparsity(&self) -> BTreeSet<(usize, usize)> {
        let n = self.ops.len();
        let mut var_sets: Vec<BTreeSet<usize>> = Vec::with_capacity(n);
        let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();

        let emit_cross =
            |s1: &BTreeSet<usize>, s2: &BTreeSet<usize>, pairs: &mut BTreeSet<(usize, usize)>| {
                for &v1 in s1 {
                    for &v2 in s2 {
                        let (r, c) = if v1 >= v2 { (v1, v2) } else { (v2, v1) };
                        pairs.insert((r, c));
                    }
                }
            };
        let emit_self = |s: &BTreeSet<usize>, pairs: &mut BTreeSet<(usize, usize)>| {
            let vars: Vec<usize> = s.iter().copied().collect();
            for (ai, &vi) in vars.iter().enumerate() {
                for &vj in &vars[..=ai] {
                    let (r, c) = if vi >= vj { (vi, vj) } else { (vj, vi) };
                    pairs.insert((r, c));
                }
            }
        };

        for op in &self.ops {
            let vset = match op {
                TapeOp::Const(_) => BTreeSet::new(),
                TapeOp::Var(j) => {
                    let mut s = BTreeSet::new();
                    s.insert(*j);
                    s
                }
                TapeOp::Add(a, b) | TapeOp::Sub(a, b) => {
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Neg(a) | TapeOp::Abs(a) => var_sets[*a].clone(),
                TapeOp::Mul(a, b) => {
                    emit_cross(&var_sets[*a], &var_sets[*b], &mut pairs);
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Div(a, b) => {
                    emit_cross(&var_sets[*a], &var_sets[*b], &mut pairs);
                    emit_self(&var_sets[*b], &mut pairs);
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Pow(a, b) => {
                    let combined: BTreeSet<usize> =
                        var_sets[*a].union(&var_sets[*b]).copied().collect();
                    emit_self(&combined, &mut pairs);
                    combined
                }
                TapeOp::Sqrt(a)
                | TapeOp::Exp(a)
                | TapeOp::Log(a)
                | TapeOp::Log10(a)
                | TapeOp::Sin(a)
                | TapeOp::Cos(a) => {
                    emit_self(&var_sets[*a], &mut pairs);
                    var_sets[*a].clone()
                }
            };
            var_sets.push(vset);
        }
        pairs
    }
}

fn build_recursive(
    expr: &Expr,
    ops: &mut Vec<TapeOp>,
    cache: &mut HashMap<*const Expr, usize>,
) -> usize {
    match expr {
        Expr::Const(c) => {
            let idx = ops.len();
            ops.push(TapeOp::Const(*c));
            idx
        }
        Expr::Var(i) => {
            let idx = ops.len();
            ops.push(TapeOp::Var(*i));
            idx
        }
        Expr::Binary(op, a, b) => {
            // Pow(x, const) is the dominant libm/dispatch cost in
            // transcendental-heavy AMPL tapes (henon, lane_emden, …):
            // `powf` itself is ~30–50 cycles AND the reverse-mode arm
            // for `Pow` carries an extra `ln(x)` branch. Rewriting
            // small integer / half-integer exponents into mul/sqrt
            // chains drops these calls entirely and reroutes the AD
            // through the much cheaper `Mul`/`Sqrt` arms.
            if let BinOp::Pow = op {
                if let Some(c) = peek_const(b) {
                    if let Some(idx) = try_emit_const_pow(a, c, ops, cache) {
                        return idx;
                    }
                }
            }
            let l = build_recursive(a, ops, cache);
            let r = build_recursive(b, ops, cache);
            let idx = ops.len();
            ops.push(match op {
                BinOp::Add => TapeOp::Add(l, r),
                BinOp::Sub => TapeOp::Sub(l, r),
                BinOp::Mul => TapeOp::Mul(l, r),
                BinOp::Div => TapeOp::Div(l, r),
                BinOp::Pow => TapeOp::Pow(l, r),
            });
            idx
        }
        Expr::Unary(op, a) => {
            let v = build_recursive(a, ops, cache);
            let idx = ops.len();
            ops.push(match op {
                UnaryOp::Neg => TapeOp::Neg(v),
                UnaryOp::Sqrt => TapeOp::Sqrt(v),
                UnaryOp::Log => TapeOp::Log(v),
                UnaryOp::Log10 => TapeOp::Log10(v),
                UnaryOp::Exp => TapeOp::Exp(v),
                UnaryOp::Abs => TapeOp::Abs(v),
                UnaryOp::Sin => TapeOp::Sin(v),
                UnaryOp::Cos => TapeOp::Cos(v),
            });
            idx
        }
        Expr::Sum(args) => {
            if args.is_empty() {
                let idx = ops.len();
                ops.push(TapeOp::Const(0.0));
                return idx;
            }
            let mut acc = build_recursive(&args[0], ops, cache);
            for a in &args[1..] {
                let next = build_recursive(a, ops, cache);
                let idx = ops.len();
                ops.push(TapeOp::Add(acc, next));
                acc = idx;
            }
            acc
        }
        Expr::Cse(body) => {
            // Cache by Rc identity so each shared body is emitted into
            // the tape exactly once and every reference resolves to the
            // same result-slot index. Forward computes the body once;
            // reverse-mode adjoint sums contributions from every ref
            // into that shared slot — exact chain rule for shared
            // sub-expressions.
            let key = Rc::as_ptr(body) as *const Expr;
            if let Some(&idx) = cache.get(&key) {
                idx
            } else {
                let idx = build_recursive(body, ops, cache);
                cache.insert(key, idx);
                idx
            }
        }
    }
}

/// Resolve `e` to a literal constant if it is one (transparently
/// peering through `Cse` wrappers, which AMPL emits around shared
/// constants in CSE-heavy problems).
fn peek_const(e: &Expr) -> Option<f64> {
    match e {
        Expr::Const(c) => Some(*c),
        Expr::Cse(body) => peek_const(body),
        _ => None,
    }
}

/// Try to rewrite `base ^ exponent_const` into cheaper ops. Returns
/// the result tape-slot on success; `None` means "fall through to
/// generic Pow." Handles the cases that account for the bulk of
/// AMPL-emitted Pow nodes: integer exponents up to ±8 and the
/// `Sqrt`/passthrough/one specials. Half-integer exponents (e.g.
/// `^1.5`) and larger integers are left to generic `Pow` since the
/// resulting mul chain grows the tape faster than it saves work.
fn try_emit_const_pow(
    base_expr: &Expr,
    c: f64,
    ops: &mut Vec<TapeOp>,
    cache: &mut HashMap<*const Expr, usize>,
) -> Option<usize> {
    if c == 0.0 {
        let idx = ops.len();
        ops.push(TapeOp::Const(1.0));
        return Some(idx);
    }
    if c == 1.0 {
        return Some(build_recursive(base_expr, ops, cache));
    }
    if c == 0.5 {
        let b = build_recursive(base_expr, ops, cache);
        let idx = ops.len();
        ops.push(TapeOp::Sqrt(b));
        return Some(idx);
    }
    // Integer exponents: bounded so a bad tape can't blow up the
    // op count. 8 covers everything AMPL typically emits for
    // polynomial models; beyond that the binary-expansion mul
    // chain (≥4 ops) starts to lose to a single `powf`.
    if c.is_finite() && c.fract() == 0.0 && c.abs() <= 8.0 {
        let n = c.abs() as u32;
        if n == 0 {
            // Already handled above, but guard.
            let idx = ops.len();
            ops.push(TapeOp::Const(1.0));
            return Some(idx);
        }
        let b = build_recursive(base_expr, ops, cache);
        let pos = emit_int_pow(b, n, ops);
        if c < 0.0 {
            // x^-n = 1 / x^n. Saves the powf and its ln branch in
            // reverse mode; cost is one Div in their place.
            let one_idx = ops.len();
            ops.push(TapeOp::Const(1.0));
            let idx = ops.len();
            ops.push(TapeOp::Div(one_idx, pos));
            return Some(idx);
        }
        return Some(pos);
    }
    None
}

/// Emit `base^n` for `n >= 1` as a binary-expansion mul chain.
/// Worst-case op count is `2·floor(log2(n))` — i.e. 1 op for n=2, 2
/// for n=3/4, 3 for n=5..8.
fn emit_int_pow(base: usize, n: u32, ops: &mut Vec<TapeOp>) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return base;
    }
    let half = emit_int_pow(base, n / 2, ops);
    let squared = ops.len();
    ops.push(TapeOp::Mul(half, half));
    if n % 2 == 1 {
        let idx = ops.len();
        ops.push(TapeOp::Mul(squared, base));
        idx
    } else {
        squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cnst(c: f64) -> Expr {
        Expr::Const(c)
    }
    fn var(i: usize) -> Expr {
        Expr::Var(i)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn pow(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Pow, Box::new(a), Box::new(b))
    }
    fn div(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Div, Box::new(a), Box::new(b))
    }
    fn unary(op: UnaryOp, a: Expr) -> Expr {
        Expr::Unary(op, Box::new(a))
    }

    #[test]
    fn polynomial_eval_and_grad() {
        // f = 3*x0^2 + 2*x1
        let e = add(
            mul(cnst(3.0), pow(var(0), cnst(2.0))),
            mul(cnst(2.0), var(1)),
        );
        let t = Tape::build(&e);
        assert!((t.eval(&[2.0, 3.0]) - 18.0).abs() < 1e-12);
        let mut g = vec![0.0; 2];
        t.gradient_seed(&[2.0, 3.0], 1.0, &mut g);
        // df/dx0 = 6*x0 = 12, df/dx1 = 2
        assert!((g[0] - 12.0).abs() < 1e-12);
        assert!((g[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cse_shared_body_evaluated_once() {
        // body = x0 + x1, shared via Rc. f = body^2 + body.
        let body = Rc::new(add(var(0), var(1)));
        let e = add(
            pow(Expr::Cse(body.clone()), cnst(2.0)),
            Expr::Cse(body.clone()),
        );
        let t = Tape::build(&e);
        // body should appear once in the tape: count Add(Var(0),Var(1)) ops
        let n_body_adds = t
            .ops
            .iter()
            .filter(|op| matches!(op, TapeOp::Add(a, b) if {
                matches!(t.ops[*a], TapeOp::Var(0)) && matches!(t.ops[*b], TapeOp::Var(1))
            }))
            .count();
        assert_eq!(n_body_adds, 1, "CSE body should be emitted exactly once");

        // f(1, 2) = 9 + 3 = 12
        assert!((t.eval(&[1.0, 2.0]) - 12.0).abs() < 1e-12);
        let mut g = vec![0.0; 2];
        t.gradient_seed(&[1.0, 2.0], 1.0, &mut g);
        // df/dx0 = 2*(x0+x1) + 1 = 7, same for x1
        assert!((g[0] - 7.0).abs() < 1e-12);
        assert!((g[1] - 7.0).abs() < 1e-12);
    }

    fn fd_check(tape: &Tape, x: &[f64], n: usize, tol: f64) {
        let vars = tape.variables();
        let mut hess_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut pairs = Vec::new();
        for (ai, &vi) in vars.iter().enumerate() {
            for &vj in &vars[..=ai] {
                let (r, c) = if vi >= vj { (vi, vj) } else { (vj, vi) };
                hess_map.entry((r, c)).or_insert_with(|| {
                    let p = pairs.len();
                    pairs.push((r, c));
                    p
                });
            }
        }
        let nnz = pairs.len();
        let mut ad = vec![0.0; nnz];
        tape.hessian_accumulate(x, 1.0, &hess_map, &mut ad);

        let mut fd = vec![0.0; nnz];
        let mut xp = x.to_vec();
        let mut gp = vec![0.0; n];
        let mut gm = vec![0.0; n];
        for &j in &vars {
            let h = (1e-7_f64).max(x[j].abs() * 1e-7);
            xp[j] = x[j] + h;
            gp.iter_mut().for_each(|v| *v = 0.0);
            tape.gradient_seed(&xp, 1.0, &mut gp);
            xp[j] = x[j] - h;
            gm.iter_mut().for_each(|v| *v = 0.0);
            tape.gradient_seed(&xp, 1.0, &mut gm);
            xp[j] = x[j];
            for &i in &vars {
                if i >= j {
                    if let Some(&pos) = hess_map.get(&(i, j)) {
                        fd[pos] = (gp[i] - gm[i]) / (2.0 * h);
                    }
                }
            }
        }
        for (k, &(r, c)) in pairs.iter().enumerate() {
            let scale = fd[k].abs().max(1.0);
            assert!(
                (ad[k] - fd[k]).abs() / scale < tol,
                "H[{},{}]: AD={:.6e} FD={:.6e}",
                r,
                c,
                ad[k],
                fd[k]
            );
        }
    }

    #[test]
    fn hessian_quadratic_matches_fd() {
        // f = 3 x0^2 + 2 x0 x1 + x1^2
        let e = add(
            add(
                mul(cnst(3.0), pow(var(0), cnst(2.0))),
                mul(cnst(2.0), mul(var(0), var(1))),
            ),
            pow(var(1), cnst(2.0)),
        );
        let t = Tape::build(&e);
        fd_check(&t, &[2.0, 3.0], 2, 1e-5);
    }

    #[test]
    fn hessian_transcendental_matches_fd() {
        // f = exp(x0) + sin(x1) + log(x0) + sqrt(x1) + x0*x1
        let e = Expr::Sum(vec![
            unary(UnaryOp::Exp, var(0)),
            unary(UnaryOp::Sin, var(1)),
            unary(UnaryOp::Log, var(0)),
            unary(UnaryOp::Sqrt, var(1)),
            mul(var(0), var(1)),
        ]);
        let t = Tape::build(&e);
        fd_check(&t, &[1.5, 2.0], 2, 1e-5);
    }

    #[test]
    fn hessian_division_matches_fd() {
        // f = x0/x1 + cos(x0)
        let e = add(div(var(0), var(1)), unary(UnaryOp::Cos, var(0)));
        let t = Tape::build(&e);
        fd_check(&t, &[0.5, 1.2], 2, 1e-5);
    }

    #[test]
    fn hessian_sparsity_separable() {
        // f = sin(x0) + x1*x2; couplings: (0,0) from sin, (2,1) from x1*x2
        let e = add(unary(UnaryOp::Sin, var(0)), mul(var(1), var(2)));
        let t = Tape::build(&e);
        let s = t.hessian_sparsity();
        assert!(s.contains(&(0, 0)));
        assert!(s.contains(&(2, 1)));
        assert!(!s.contains(&(1, 0)));
        assert!(!s.contains(&(2, 0)));
    }

    fn count_op<F: Fn(&TapeOp) -> bool>(t: &Tape, pred: F) -> usize {
        t.ops.iter().filter(|o| pred(o)).count()
    }

    #[test]
    fn pow_zero_const_folds_to_one() {
        // x^0 → 1 (no Pow, no reference to x in the tape)
        let e = pow(var(0), cnst(0.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Var(_))), 0);
        assert!((t.eval(&[7.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pow_one_passes_through() {
        // x^1 → x (no Pow, no Const introduced for the exponent)
        let e = pow(var(0), cnst(1.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Const(_))), 0);
        assert!((t.eval(&[3.5]) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn pow_half_lowers_to_sqrt() {
        let e = pow(var(0), cnst(0.5));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Sqrt(_))), 1);
        assert!((t.eval(&[16.0]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn pow_two_lowers_to_single_mul() {
        let e = pow(var(0), cnst(2.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Mul(..))), 1);
        assert!((t.eval(&[3.0]) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn pow_three_lowers_to_two_muls() {
        let e = pow(var(0), cnst(3.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Mul(..))), 2);
        assert!((t.eval(&[2.0]) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn pow_eight_lowers_to_three_muls() {
        // Binary expansion: x → x² → x⁴ → x⁸ (3 squarings)
        let e = pow(var(0), cnst(8.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Mul(..))), 3);
        assert!((t.eval(&[2.0]) - 256.0).abs() < 1e-12);
    }

    #[test]
    fn pow_negative_two_lowers_to_div() {
        // x^-2 → 1 / (x*x)
        let e = pow(var(0), cnst(-2.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Div(..))), 1);
        assert!((t.eval(&[4.0]) - (1.0 / 16.0)).abs() < 1e-12);
    }

    #[test]
    fn pow_large_const_stays_generic() {
        // x^9 stays as Pow — beyond the cutoff, generic is cheaper.
        let e = pow(var(0), cnst(9.0));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 1);
    }

    #[test]
    fn pow_non_integer_const_stays_generic() {
        // x^1.5 stays as Pow until half-integer handling is added.
        let e = pow(var(0), cnst(1.5));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 1);
    }

    #[test]
    fn pow_const_through_cse_const() {
        // Exponent wrapped in Cse — peek_const should still see it.
        let two = Rc::new(cnst(2.0));
        let e = Expr::Binary(BinOp::Pow, Box::new(var(0)), Box::new(Expr::Cse(two)));
        let t = Tape::build(&e);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Pow(..))), 0);
        assert_eq!(count_op(&t, |o| matches!(o, TapeOp::Mul(..))), 1);
    }

    #[test]
    fn hessian_pow_three_matches_fd() {
        // f = 5 * x0^3 + x0 * x1 — exercises the lowered cubic + cross term.
        let e = add(mul(cnst(5.0), pow(var(0), cnst(3.0))), mul(var(0), var(1)));
        let t = Tape::build(&e);
        fd_check(&t, &[1.7, 0.8], 2, 1e-5);
    }

    #[test]
    fn hessian_pow_negative_matches_fd() {
        // f = 1/x0^2 + x1^2 — exercises lowered x^-2 and x^2.
        let e = add(pow(var(0), cnst(-2.0)), pow(var(1), cnst(2.0)));
        let t = Tape::build(&e);
        fd_check(&t, &[1.3, 2.4], 2, 1e-5);
    }

    #[test]
    fn hessian_pow_half_matches_fd() {
        // f = sqrt(x0) + x0*x1 (via Pow(_, 0.5) → Sqrt)
        let e = add(pow(var(0), cnst(0.5)), mul(var(0), var(1)));
        let t = Tape::build(&e);
        fd_check(&t, &[2.5, 1.1], 2, 1e-5);
    }

    #[test]
    fn hessian_sparsity_through_cse() {
        // body = x0+x1 (CSE). f = body^2 + body.
        // d²/dx² of body^2 couples (0,0), (1,0), (1,1).
        let body = Rc::new(add(var(0), var(1)));
        let e = add(
            pow(Expr::Cse(body.clone()), cnst(2.0)),
            Expr::Cse(body.clone()),
        );
        let t = Tape::build(&e);
        let s = t.hessian_sparsity();
        assert!(s.contains(&(0, 0)));
        assert!(s.contains(&(1, 0)));
        assert!(s.contains(&(1, 1)));
        assert_eq!(s.len(), 3);
    }
}
