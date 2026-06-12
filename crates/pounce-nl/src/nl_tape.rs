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
//! builder hits `Expr::Cse(rc)` it keys on the `Arc` pointer identity,
//! emitting the body the first time and returning the cached
//! result-slot index on subsequent references. The forward pass then
//! computes each CSE once and the reverse pass folds adjoints from
//! every reference into a single slot — exact chain-rule behaviour.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::nl_external::{EvalResult, ExternalArg, ExternalLibrary, ExternalResolver};
use super::nl_reader::{BinOp, CmpOp, Expr, FuncallArg, UnaryOp};

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
    Tan(usize),
    Atan(usize),
    Acos(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
    Asin(usize),
    Acosh(usize),
    Asinh(usize),
    Atanh(usize),
    /// Two-argument arctangent `atan2(vals[a], vals[b])` (operands are
    /// `(y, x)`, matching AMPL's `atan2(y, x)` / `.nl` opcode o48).
    Atan2(usize, usize),
    /// Pairwise minimum `min(vals[a], vals[b])`. Piecewise linear: the
    /// value/tangent/adjoint route through whichever operand is smaller
    /// (ties pick the first), and the second derivative is identically
    /// zero. n-ary AMPL `min` (opcode o11) folds to a chain of these.
    Min(usize, usize),
    /// Pairwise maximum `max(vals[a], vals[b])` — the `Min` mirror;
    /// n-ary AMPL `max` (opcode o12) folds to a chain of these.
    Max(usize, usize),
    /// Relational comparison `vals[a] OP vals[b]` → `1.0`/`0.0`.
    /// Piecewise constant, so its derivative is identically zero — the
    /// AD passes treat it as a constant w.r.t. its operands.
    Cmp(CmpOp, usize, usize),
    /// Logical AND: `1.0` iff both operands are nonzero. Zero derivative.
    And(usize, usize),
    /// Logical OR: `1.0` iff either operand is nonzero. Zero derivative.
    Or(usize, usize),
    /// Logical NOT: `1.0` iff the operand is zero. Zero derivative.
    Not(usize),
    /// `if-then-else`: operands `(cond, then, else)`. The value is
    /// `vals[then]` when `vals[cond] != 0` else `vals[else]`, and the
    /// value/tangent/adjoint all route through the active branch only.
    /// The condition contributes no derivative (the branch switch is a
    /// non-smooth event the AD ignores).
    Select(usize, usize, usize),
    /// AMPL imported (external) function call. The payload (library
    /// handle, name, and argument list) is boxed so this rare variant
    /// does not inflate `size_of::<TapeOp>()`: without the box the
    /// `Arc`+`String`+`Vec` make every op ~64 bytes, which on a
    /// summand-split objective with millions of tiny tapes (e.g.
    /// `sensors`) costs gigabytes. Boxing drops the common arithmetic
    /// ops back to the size of the next-largest variant.
    Funcall(Box<FuncallData>),
}

/// Boxed payload of [`TapeOp::Funcall`]. The library is kept alive by
/// the `Arc`; `name` is the registered function name; `args` carries
/// positional arguments where real-valued args reference earlier tape
/// slots and string args are inline literals.
#[derive(Debug, Clone)]
pub struct FuncallData {
    pub lib: Arc<ExternalLibrary>,
    pub name: String,
    pub args: Vec<TapeFuncallArg>,
}

/// One argument of a `TapeOp::Funcall`. Real arguments are tape-slot indices
/// (their values come from the running `vals[]` during forward); string
/// arguments are owned literals (AMPL `h<len>:<chars>` tokens).
#[derive(Debug, Clone)]
pub enum TapeFuncallArg {
    Tape(usize),
    Str(String),
}

/// Evaluate a relational opcode on two scalar values, returning the
/// boolean truth (callers map it to `1.0`/`0.0`).
#[inline]
fn cmp_holds(op: CmpOp, a: f64, b: f64) -> bool {
    match op {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Eq => a == b,
        CmpOp::Ge => a >= b,
        CmpOp::Gt => a > b,
        CmpOp::Ne => a != b,
    }
}

fn funcall_to_ext_args<'a>(args: &'a [TapeFuncallArg], vals: &[f64]) -> Vec<ExternalArg<'a>> {
    args.iter()
        .map(|a| match a {
            TapeFuncallArg::Tape(idx) => ExternalArg::Real(vals[*idx]),
            TapeFuncallArg::Str(s) => ExternalArg::Str(s.as_str()),
        })
        .collect()
}

/// Evaluate an external (AMPL imported) function, poisoning the result with
/// `NaN` instead of panicking when the library reports an error.
///
/// An external eval fails on user-controllable conditions — most commonly an
/// out-of-domain property evaluation (e.g. an IDAES Helmholtz thermo call
/// outside its valid pressure/temperature range). We mirror the tape's own
/// arithmetic domain-error semantics (`log(-1) → NaN`): hand back NaN so the
/// IPM sees a failed evaluation and the line search backs off, rather than
/// raising an uncatchable panic across the pyo3 boundary on the `read_nl`
/// surface. The NaN derivative/Hessian vectors are sized by the full argument
/// count — an upper bound on the real-arg count a successful eval returns — so
/// every downstream index into them stays in range.
fn ext_eval_or_nan(
    lib: &ExternalLibrary,
    name: &str,
    call_args: &[ExternalArg<'_>],
    n_args: usize,
    want_derivs: bool,
    want_hes: bool,
) -> EvalResult {
    lib.eval(name, call_args, want_derivs, want_hes)
        .unwrap_or_else(|_| EvalResult {
            value: f64::NAN,
            derivs: want_derivs.then(|| vec![f64::NAN; n_args]),
            hessian: want_hes.then(|| vec![f64::NAN; n_args * (n_args + 1) / 2]),
        })
}

/// A flattened expression tape. The result of evaluation is the value
/// at slot `ops.len() - 1` (i.e. the last op).
#[derive(Debug, Clone)]
pub struct Tape {
    pub ops: Vec<TapeOp>,
}

impl Tape {
    /// Build a tape from an `Expr` tree (no AMPL external functions). CSE
    /// bodies (`Expr::Cse(rc)`) are cached by `Arc` pointer identity so each
    /// body is emitted once even when referenced many times.
    pub fn build(expr: &Expr) -> Self {
        Self::build_with_externals(expr, &ExternalResolver::default())
    }

    /// Build a tape from an `Expr` tree, resolving any `Expr::Funcall`
    /// nodes through `resolver`. Panics if the expression references a
    /// funcall id that is not in the resolver — `NlProblem::resolve_externals`
    /// must populate the resolver before tape construction.
    pub fn build_with_externals(expr: &Expr, resolver: &ExternalResolver) -> Self {
        let mut ops = Vec::new();
        let mut cache: HashMap<*const Expr, usize> = HashMap::new();
        build_recursive(expr, &mut ops, &mut cache, resolver);
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
                TapeOp::Tan(a) => vals[*a].tan(),
                TapeOp::Atan(a) => vals[*a].atan(),
                TapeOp::Acos(a) => vals[*a].acos(),
                TapeOp::Sinh(a) => vals[*a].sinh(),
                TapeOp::Cosh(a) => vals[*a].cosh(),
                TapeOp::Tanh(a) => vals[*a].tanh(),
                TapeOp::Asin(a) => vals[*a].asin(),
                TapeOp::Acosh(a) => vals[*a].acosh(),
                TapeOp::Asinh(a) => vals[*a].asinh(),
                TapeOp::Atanh(a) => vals[*a].atanh(),
                TapeOp::Atan2(a, b) => vals[*a].atan2(vals[*b]),
                TapeOp::Min(a, b) => vals[*a].min(vals[*b]),
                TapeOp::Max(a, b) => vals[*a].max(vals[*b]),
                TapeOp::Cmp(op, a, b) => f64::from(cmp_holds(*op, vals[*a], vals[*b])),
                TapeOp::And(a, b) => f64::from(vals[*a] != 0.0 && vals[*b] != 0.0),
                TapeOp::Or(a, b) => f64::from(vals[*a] != 0.0 || vals[*b] != 0.0),
                TapeOp::Not(a) => f64::from(vals[*a] == 0.0),
                TapeOp::Select(c, t, e) => {
                    if vals[*c] != 0.0 {
                        vals[*t]
                    } else {
                        vals[*e]
                    }
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, &vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), false, false);
                    res.value
                }
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

    /// Reverse-mode AD reusing two caller-supplied scratch buffers
    /// (`vals` from [`forward_into`], and an `adj` arena ≥
    /// `self.ops.len()`) instead of allocating a forward-value vector and
    /// an adjoint vector per call like [`gradient_seed`]. The `.nl` design
    /// emits one tiny tape per summand — ~10⁶ on large models — so a single
    /// `eval_jac_g` / `eval_grad_f` drives this millions of times and the
    /// per-call allocation dominated. `grad` is accumulated into (not
    /// zeroed); `adj` may be passed dirty (it is zeroed at the touched
    /// slots internally).
    ///
    /// [`forward_into`]: Tape::forward_into
    pub fn gradient_seed_into(
        &self,
        x: &[f64],
        seed: f64,
        grad: &mut [f64],
        vals: &mut [f64],
        adj: &mut [f64],
    ) {
        if seed == 0.0 || self.ops.is_empty() {
            return;
        }
        debug_assert!(vals.len() >= self.ops.len());
        self.forward_into(x, vals);
        self.reverse_into(vals, seed, grad, adj);
    }

    fn reverse(&self, vals: &[f64], seed: f64, grad: &mut [f64]) {
        let n = self.ops.len();
        let mut adj = vec![0.0f64; n];
        self.reverse_into(vals, seed, grad, &mut adj);
    }

    /// Reverse adjoint sweep into a caller-supplied `adj` scratch buffer
    /// (length ≥ `self.ops.len()`), the allocation-free core of [`reverse`].
    /// `adj` is zeroed over `[0, n)` internally, so a dirty arena is fine;
    /// `grad` is accumulated into (not zeroed).
    fn reverse_into(&self, vals: &[f64], seed: f64, grad: &mut [f64], adj: &mut [f64]) {
        let n = self.ops.len();
        debug_assert!(adj.len() >= n);
        adj[..n].fill(0.0);
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
                TapeOp::Tan(j) => {
                    let t = vals[i];
                    adj[*j] += a * (1.0 + t * t);
                }
                TapeOp::Atan(j) => {
                    let u = vals[*j];
                    adj[*j] += a / (1.0 + u * u);
                }
                TapeOp::Acos(j) => {
                    let u = vals[*j];
                    adj[*j] -= a / (1.0 - u * u).sqrt();
                }
                TapeOp::Sinh(j) => {
                    adj[*j] += a * vals[*j].cosh();
                }
                TapeOp::Cosh(j) => {
                    adj[*j] += a * vals[*j].sinh();
                }
                TapeOp::Tanh(j) => {
                    let t = vals[i];
                    adj[*j] += a * (1.0 - t * t);
                }
                TapeOp::Asin(j) => {
                    let u = vals[*j];
                    adj[*j] += a / (1.0 - u * u).sqrt();
                }
                TapeOp::Acosh(j) => {
                    let u = vals[*j];
                    adj[*j] += a / (u * u - 1.0).sqrt();
                }
                TapeOp::Asinh(j) => {
                    let u = vals[*j];
                    adj[*j] += a / (u * u + 1.0).sqrt();
                }
                TapeOp::Atanh(j) => {
                    let u = vals[*j];
                    adj[*j] += a / (1.0 - u * u);
                }
                TapeOp::Atan2(l, r) => {
                    let y = vals[*l];
                    let x = vals[*r];
                    let d = y * y + x * x;
                    adj[*l] += a * (x / d);
                    adj[*r] += a * (-y / d);
                }
                // min/max are piecewise linear: the adjoint flows to the
                // selected operand only (ties pick the first, a valid
                // subgradient choice).
                TapeOp::Min(l, r) => {
                    if vals[*l] <= vals[*r] {
                        adj[*l] += a;
                    } else {
                        adj[*r] += a;
                    }
                }
                TapeOp::Max(l, r) => {
                    if vals[*l] >= vals[*r] {
                        adj[*l] += a;
                    } else {
                        adj[*r] += a;
                    }
                }
                // Comparisons and logical connectives are piecewise
                // constant: zero derivative, so no adjoint propagates.
                TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => {}
                // if-then-else: the adjoint flows entirely into the
                // active branch; the condition gets none.
                TapeOp::Select(c, t, e) => {
                    if vals[*c] != 0.0 {
                        adj[*t] += a;
                    } else {
                        adj[*e] += a;
                    }
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), true, false);
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let mut k = 0usize;
                    for arg in args {
                        if let TapeFuncallArg::Tape(idx) = arg {
                            adj[*idx] += a * derivs[k];
                            k += 1;
                        }
                    }
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
                    // Match the reverse-mode gradient's guard (`rv != 0.0` only): at base
                    // u == 0 the slope is still well defined for r >= 1 (and a
                    // genuine ±inf for r < 1), so it must not be silently dropped,
                    // or the forward tangent disagrees with the reverse gradient.
                    if r != 0.0 {
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
                TapeOp::Tan(a) => {
                    let t = vals[i];
                    dot[*a] * (1.0 + t * t)
                }
                TapeOp::Atan(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 + u * u)
                }
                TapeOp::Acos(a) => {
                    let u = vals[*a];
                    -dot[*a] / (1.0 - u * u).sqrt()
                }
                TapeOp::Sinh(a) => dot[*a] * vals[*a].cosh(),
                TapeOp::Cosh(a) => dot[*a] * vals[*a].sinh(),
                TapeOp::Tanh(a) => {
                    let t = vals[i];
                    dot[*a] * (1.0 - t * t)
                }
                TapeOp::Asin(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 - u * u).sqrt()
                }
                TapeOp::Acosh(a) => {
                    let u = vals[*a];
                    dot[*a] / (u * u - 1.0).sqrt()
                }
                TapeOp::Asinh(a) => {
                    let u = vals[*a];
                    dot[*a] / (u * u + 1.0).sqrt()
                }
                TapeOp::Atanh(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 - u * u)
                }
                TapeOp::Atan2(a, b) => {
                    let y = vals[*a];
                    let x = vals[*b];
                    let d = y * y + x * x;
                    (x * dot[*a] - y * dot[*b]) / d
                }
                // min/max: the tangent follows the selected operand.
                TapeOp::Min(a, b) => {
                    if vals[*a] <= vals[*b] {
                        dot[*a]
                    } else {
                        dot[*b]
                    }
                }
                TapeOp::Max(a, b) => {
                    if vals[*a] >= vals[*b] {
                        dot[*a]
                    } else {
                        dot[*b]
                    }
                }
                TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => 0.0,
                TapeOp::Select(c, t, e) => {
                    if vals[*c] != 0.0 {
                        dot[*t]
                    } else {
                        dot[*e]
                    }
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), true, false);
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let mut acc = 0.0;
                    let mut k = 0usize;
                    for arg in args {
                        if let TapeFuncallArg::Tape(idx) = arg {
                            acc += derivs[k] * dot[*idx];
                            k += 1;
                        }
                    }
                    acc
                }
            };
        }
    }

    /// Forward sweep into a caller-supplied buffer. Avoids the
    /// per-call allocation of `forward()` so hot paths can reuse
    /// one scratch arena across many tapes.
    pub fn forward_into(&self, x: &[f64], vals: &mut [f64]) {
        let n = self.ops.len();
        debug_assert!(vals.len() >= n);
        for i in 0..n {
            vals[i] = match &self.ops[i] {
                TapeOp::Const(c) => *c,
                TapeOp::Var(j) => x[*j],
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
                TapeOp::Tan(a) => vals[*a].tan(),
                TapeOp::Atan(a) => vals[*a].atan(),
                TapeOp::Acos(a) => vals[*a].acos(),
                TapeOp::Sinh(a) => vals[*a].sinh(),
                TapeOp::Cosh(a) => vals[*a].cosh(),
                TapeOp::Tanh(a) => vals[*a].tanh(),
                TapeOp::Asin(a) => vals[*a].asin(),
                TapeOp::Acosh(a) => vals[*a].acosh(),
                TapeOp::Asinh(a) => vals[*a].asinh(),
                TapeOp::Atanh(a) => vals[*a].atanh(),
                TapeOp::Atan2(a, b) => vals[*a].atan2(vals[*b]),
                TapeOp::Min(a, b) => vals[*a].min(vals[*b]),
                TapeOp::Max(a, b) => vals[*a].max(vals[*b]),
                TapeOp::Cmp(op, a, b) => f64::from(cmp_holds(*op, vals[*a], vals[*b])),
                TapeOp::And(a, b) => f64::from(vals[*a] != 0.0 && vals[*b] != 0.0),
                TapeOp::Or(a, b) => f64::from(vals[*a] != 0.0 || vals[*b] != 0.0),
                TapeOp::Not(a) => f64::from(vals[*a] == 0.0),
                TapeOp::Select(c, t, e) => {
                    if vals[*c] != 0.0 {
                        vals[*t]
                    } else {
                        vals[*e]
                    }
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, &*vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), false, false);
                    res.value
                }
            };
        }
    }

    /// Directional Hessian-vector product: emits
    /// `weight * (∇²f · seed)[k]` into `out[k]` for every problem
    /// variable `k` the tape references. Caller supplies the
    /// forward-pass result `vals` (use [`forward_into`]) plus three
    /// scratch buffers (`dot`, `adj`, `adj_dot`), each at least
    /// `self.ops.len()` long. `out` must be at least one past the
    /// largest variable index in the tape; the routine reads
    /// `seed[k]` for each `Var(k)` and writes `out[k] += weight *
    /// (Hess · seed)[k]`.
    ///
    /// This is one forward-over-reverse AD pass — O(n_ops) work —
    /// regardless of how many variables the tape depends on, which
    /// is what makes Hessian coloring efficient: a single
    /// directional pass recovers a whole color group of columns.
    ///
    /// [`forward_into`]: Tape::forward_into
    pub fn hessian_directional(
        &self,
        vals: &[f64],
        seed: &[f64],
        weight: f64,
        out: &mut [f64],
        dot: &mut [f64],
        adj: &mut [f64],
        adj_dot: &mut [f64],
    ) {
        let n = self.ops.len();
        if n == 0 || weight == 0.0 {
            return;
        }
        debug_assert!(vals.len() >= n);
        debug_assert!(dot.len() >= n);
        debug_assert!(adj.len() >= n);
        debug_assert!(adj_dot.len() >= n);

        // Forward tangent: dot[i] = (∂vals[i] / ∂x · seed). At
        // Var(k) the seed entry feeds in; the rest of the chain
        // rule matches `forward_tangent` exactly.
        for i in 0..n {
            dot[i] = match &self.ops[i] {
                TapeOp::Const(_) => 0.0,
                TapeOp::Var(k) => seed[*k],
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
                    // Match the reverse-mode gradient's guard (`rv != 0.0` only): at base
                    // u == 0 the slope is still well defined for r >= 1 (and a
                    // genuine ±inf for r < 1), so it must not be silently dropped,
                    // or the forward tangent disagrees with the reverse gradient.
                    if r != 0.0 {
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
                TapeOp::Exp(a) => vals[i] * dot[*a],
                TapeOp::Log(a) => dot[*a] / vals[*a],
                TapeOp::Log10(a) => dot[*a] / (vals[*a] * std::f64::consts::LN_10),
                TapeOp::Sin(a) => vals[*a].cos() * dot[*a],
                TapeOp::Cos(a) => -vals[*a].sin() * dot[*a],
                TapeOp::Tan(a) => {
                    let t = vals[i];
                    (1.0 + t * t) * dot[*a]
                }
                TapeOp::Atan(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 + u * u)
                }
                TapeOp::Acos(a) => {
                    let u = vals[*a];
                    -dot[*a] / (1.0 - u * u).sqrt()
                }
                TapeOp::Sinh(a) => dot[*a] * vals[*a].cosh(),
                TapeOp::Cosh(a) => dot[*a] * vals[*a].sinh(),
                TapeOp::Tanh(a) => {
                    let t = vals[i];
                    (1.0 - t * t) * dot[*a]
                }
                TapeOp::Asin(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 - u * u).sqrt()
                }
                TapeOp::Acosh(a) => {
                    let u = vals[*a];
                    dot[*a] / (u * u - 1.0).sqrt()
                }
                TapeOp::Asinh(a) => {
                    let u = vals[*a];
                    dot[*a] / (u * u + 1.0).sqrt()
                }
                TapeOp::Atanh(a) => {
                    let u = vals[*a];
                    dot[*a] / (1.0 - u * u)
                }
                TapeOp::Atan2(a, b) => {
                    let y = vals[*a];
                    let x = vals[*b];
                    let d = y * y + x * x;
                    (x * dot[*a] - y * dot[*b]) / d
                }
                // min/max: the tangent follows the selected operand.
                TapeOp::Min(a, b) => {
                    if vals[*a] <= vals[*b] {
                        dot[*a]
                    } else {
                        dot[*b]
                    }
                }
                TapeOp::Max(a, b) => {
                    if vals[*a] >= vals[*b] {
                        dot[*a]
                    } else {
                        dot[*b]
                    }
                }
                TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => 0.0,
                TapeOp::Select(c, t, e) => {
                    if vals[*c] != 0.0 {
                        dot[*t]
                    } else {
                        dot[*e]
                    }
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), true, false);
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let mut acc = 0.0;
                    let mut k = 0usize;
                    for arg in args {
                        if let TapeFuncallArg::Tape(idx) = arg {
                            acc += derivs[k] * dot[*idx];
                            k += 1;
                        }
                    }
                    acc
                }
            };
        }

        // Reverse over tangent. adj[i] = ∂f/∂vals[i],
        // adj_dot[i] = derivative of adj[i] along `seed`
        // direction = (Hess · seed) projected onto slot i.
        for slot in adj.iter_mut().take(n) {
            *slot = 0.0;
        }
        for slot in adj_dot.iter_mut().take(n) {
            *slot = 0.0;
        }
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
                    if wd != 0.0 {
                        out[*k] += weight * wd;
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
                    adj[*a] += w * vals[*b];
                    adj[*b] += w * vals[*a];
                    adj_dot[*a] += wd * vals[*b] + w * dot[*b];
                    adj_dot[*b] += wd * vals[*a] + w * dot[*a];
                }
                TapeOp::Div(a, b) => {
                    let vb = vals[*b];
                    let vb2 = vb * vb;
                    let vb3 = vb2 * vb;
                    adj[*a] += w / vb;
                    adj_dot[*a] += wd / vb + w * (-dot[*b] / vb2);
                    adj[*b] += w * (-vals[*a] / vb2);
                    adj_dot[*b] += wd * (-vals[*a] / vb2)
                        + w * (-dot[*a] / vb2 + 2.0 * vals[*a] * dot[*b] / vb3);
                }
                TapeOp::Pow(a, b) => {
                    let u = vals[*a];
                    let r = vals[*b];
                    let du = dot[*a];
                    let dr = dot[*b];
                    if r != 0.0 {
                        if u != 0.0 {
                            let p_a = r * u.powf(r - 1.0);
                            adj[*a] += w * p_a;
                            let mut dp_a = dr * u.powf(r - 1.0);
                            if u > 0.0 {
                                dp_a += r * u.powf(r - 1.0) * ((r - 1.0) * du / u + dr * u.ln());
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
                        let p_b = vals[i] * ln_u;
                        adj[*b] += w * p_b;
                        let dur = vals[i] * (r * du / u + dr * ln_u);
                        let dp_b = dur * ln_u + vals[i] * du / u;
                        adj_dot[*b] += wd * p_b + w * dp_b;
                    }
                }
                TapeOp::Neg(a) => {
                    adj[*a] -= w;
                    adj_dot[*a] -= wd;
                }
                TapeOp::Abs(a) => {
                    let s = if vals[*a] >= 0.0 { 1.0 } else { -1.0 };
                    adj[*a] += w * s;
                    adj_dot[*a] += wd * s;
                }
                TapeOp::Sqrt(a) => {
                    let sv = vals[i];
                    if sv > 0.0 {
                        let fp = 0.5 / sv;
                        let fpp = -0.25 / (vals[*a] * sv);
                        adj[*a] += w * fp;
                        adj_dot[*a] += wd * fp + w * fpp * dot[*a];
                    }
                }
                TapeOp::Exp(a) => {
                    let ev = vals[i];
                    adj[*a] += w * ev;
                    adj_dot[*a] += wd * ev + w * ev * dot[*a];
                }
                TapeOp::Log(a) => {
                    let u = vals[*a];
                    adj[*a] += w / u;
                    adj_dot[*a] += wd / u + w * (-1.0 / (u * u)) * dot[*a];
                }
                TapeOp::Log10(a) => {
                    let u = vals[*a];
                    let c = std::f64::consts::LN_10;
                    adj[*a] += w / (u * c);
                    adj_dot[*a] += wd / (u * c) + w * (-1.0 / (u * u * c)) * dot[*a];
                }
                TapeOp::Sin(a) => {
                    let u = vals[*a];
                    let cu = u.cos();
                    adj[*a] += w * cu;
                    adj_dot[*a] += wd * cu + w * (-u.sin()) * dot[*a];
                }
                TapeOp::Cos(a) => {
                    let u = vals[*a];
                    let su = u.sin();
                    adj[*a] -= w * su;
                    adj_dot[*a] += wd * (-su) + w * (-u.cos()) * dot[*a];
                }
                TapeOp::Tan(a) => {
                    let t = vals[i];
                    let gp = 1.0 + t * t;
                    let gpp = 2.0 * t * gp;
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Atan(a) => {
                    let u = vals[*a];
                    let d = 1.0 + u * u;
                    let gp = 1.0 / d;
                    let gpp = -2.0 * u / (d * d);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Acos(a) => {
                    let u = vals[*a];
                    let s = 1.0 - u * u;
                    let r = s.sqrt();
                    let gp = -1.0 / r;
                    let gpp = -u / (s * r);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Sinh(a) => {
                    let u = vals[*a];
                    let gp = u.cosh();
                    let gpp = u.sinh();
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Cosh(a) => {
                    let u = vals[*a];
                    let gp = u.sinh();
                    let gpp = u.cosh();
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Tanh(a) => {
                    let t = vals[i];
                    let gp = 1.0 - t * t;
                    let gpp = -2.0 * t * gp;
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Asin(a) => {
                    let u = vals[*a];
                    let s = 1.0 - u * u;
                    let r = s.sqrt();
                    let gp = 1.0 / r;
                    let gpp = u / (s * r);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Acosh(a) => {
                    let u = vals[*a];
                    let s = u * u - 1.0;
                    let r = s.sqrt();
                    let gp = 1.0 / r;
                    let gpp = -u / (s * r);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Asinh(a) => {
                    let u = vals[*a];
                    let s = u * u + 1.0;
                    let r = s.sqrt();
                    let gp = 1.0 / r;
                    let gpp = -u / (s * r);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Atanh(a) => {
                    let u = vals[*a];
                    let d = 1.0 - u * u;
                    let gp = 1.0 / d;
                    let gpp = 2.0 * u / (d * d);
                    adj[*a] += w * gp;
                    adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                }
                TapeOp::Atan2(a, b) => {
                    let y = vals[*a];
                    let x = vals[*b];
                    let d = y * y + x * x;
                    let d2 = d * d;
                    let fa = x / d;
                    let fb = -y / d;
                    let faa = -2.0 * y * x / d2;
                    let fab = (y * y - x * x) / d2;
                    let fbb = 2.0 * y * x / d2;
                    adj[*a] += w * fa;
                    adj[*b] += w * fb;
                    adj_dot[*a] += wd * fa + w * (faa * dot[*a] + fab * dot[*b]);
                    adj_dot[*b] += wd * fb + w * (fab * dot[*a] + fbb * dot[*b]);
                }
                // min/max are piecewise linear (zero second derivative):
                // route the adjoint and its tangent into the selected
                // operand, exactly like the active branch of a Select.
                TapeOp::Min(a, b) => {
                    let br = if vals[*a] <= vals[*b] { *a } else { *b };
                    adj[br] += w;
                    adj_dot[br] += wd;
                }
                TapeOp::Max(a, b) => {
                    let br = if vals[*a] >= vals[*b] { *a } else { *b };
                    adj[br] += w;
                    adj_dot[br] += wd;
                }
                // Zero derivative: no first- or second-order adjoint.
                TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => {}
                // Route both the adjoint and its tangent into the
                // active branch; the condition contributes nothing.
                TapeOp::Select(c, t, e) => {
                    let br = if vals[*c] != 0.0 { *t } else { *e };
                    adj[br] += w;
                    adj_dot[br] += wd;
                }
                TapeOp::Funcall(fc) => {
                    let FuncallData { lib, name, args } = fc.as_ref();
                    let call_args = funcall_to_ext_args(args, vals);
                    let res = ext_eval_or_nan(lib, name, &call_args, args.len(), true, true);
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let hes = res.hessian.expect("want_hes=true returns hessian");
                    let real_tape: Vec<usize> = args
                        .iter()
                        .filter_map(|a| match a {
                            TapeFuncallArg::Tape(t) => Some(*t),
                            TapeFuncallArg::Str(_) => None,
                        })
                        .collect();
                    for (k, &tk) in real_tape.iter().enumerate() {
                        adj[tk] += w * derivs[k];
                        let mut second_term = 0.0;
                        for (l, &tl) in real_tape.iter().enumerate() {
                            let (lo, hi) = if k <= l { (k, l) } else { (l, k) };
                            let h_kl = hes[lo + hi * (hi + 1) / 2];
                            second_term += h_kl * dot[tl];
                        }
                        adj_dot[tk] += wd * derivs[k] + w * second_term;
                    }
                }
            }
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
                                    dp_a +=
                                        r * u.powf(r - 1.0) * ((r - 1.0) * du / u + dr * u.ln());
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
                    TapeOp::Tan(a) => {
                        let t = v[i];
                        let gp = 1.0 + t * t;
                        let gpp = 2.0 * t * gp;
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Atan(a) => {
                        let u = v[*a];
                        let d = 1.0 + u * u;
                        let gp = 1.0 / d;
                        let gpp = -2.0 * u / (d * d);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Acos(a) => {
                        let u = v[*a];
                        let s = 1.0 - u * u;
                        let r = s.sqrt();
                        let gp = -1.0 / r;
                        let gpp = -u / (s * r);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Sinh(a) => {
                        let u = v[*a];
                        let gp = u.cosh();
                        let gpp = u.sinh();
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Cosh(a) => {
                        let u = v[*a];
                        let gp = u.sinh();
                        let gpp = u.cosh();
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Tanh(a) => {
                        let t = v[i];
                        let gp = 1.0 - t * t;
                        let gpp = -2.0 * t * gp;
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Asin(a) => {
                        let u = v[*a];
                        let s = 1.0 - u * u;
                        let r = s.sqrt();
                        let gp = 1.0 / r;
                        let gpp = u / (s * r);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Acosh(a) => {
                        let u = v[*a];
                        let s = u * u - 1.0;
                        let r = s.sqrt();
                        let gp = 1.0 / r;
                        let gpp = -u / (s * r);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Asinh(a) => {
                        let u = v[*a];
                        let s = u * u + 1.0;
                        let r = s.sqrt();
                        let gp = 1.0 / r;
                        let gpp = -u / (s * r);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Atanh(a) => {
                        let u = v[*a];
                        let d = 1.0 - u * u;
                        let gp = 1.0 / d;
                        let gpp = 2.0 * u / (d * d);
                        adj[*a] += w * gp;
                        adj_dot[*a] += wd * gp + w * gpp * dot[*a];
                    }
                    TapeOp::Atan2(a, b) => {
                        let y = v[*a];
                        let x = v[*b];
                        let d = y * y + x * x;
                        let d2 = d * d;
                        let fa = x / d;
                        let fb = -y / d;
                        let faa = -2.0 * y * x / d2;
                        let fab = (y * y - x * x) / d2;
                        let fbb = 2.0 * y * x / d2;
                        adj[*a] += w * fa;
                        adj[*b] += w * fb;
                        adj_dot[*a] += wd * fa + w * (faa * dot[*a] + fab * dot[*b]);
                        adj_dot[*b] += wd * fb + w * (fab * dot[*a] + fbb * dot[*b]);
                    }
                    // min/max are piecewise linear (zero second
                    // derivative): route adjoint and its tangent into
                    // the selected operand, like an active Select branch.
                    TapeOp::Min(a, b) => {
                        let br = if v[*a] <= v[*b] { *a } else { *b };
                        adj[br] += w;
                        adj_dot[br] += wd;
                    }
                    TapeOp::Max(a, b) => {
                        let br = if v[*a] >= v[*b] { *a } else { *b };
                        adj[br] += w;
                        adj_dot[br] += wd;
                    }
                    // Zero derivative: no first- or second-order adjoint.
                    TapeOp::Cmp(_, _, _)
                    | TapeOp::And(_, _)
                    | TapeOp::Or(_, _)
                    | TapeOp::Not(_) => {}
                    // Route adjoint and its tangent into the active
                    // branch only; the condition contributes nothing.
                    TapeOp::Select(c, t, e) => {
                        let br = if v[*c] != 0.0 { *t } else { *e };
                        adj[br] += w;
                        adj_dot[br] += wd;
                    }
                    TapeOp::Funcall(fc) => {
                        let FuncallData { lib, name, args } = fc.as_ref();
                        let call_args = funcall_to_ext_args(args, &v);
                        let res = ext_eval_or_nan(lib, name, &call_args, args.len(), true, true);
                        let derivs = res.derivs.expect("want_derivs=true returns derivs");
                        let hes = res.hessian.expect("want_hes=true returns hessian");
                        let real_tape: Vec<usize> = args
                            .iter()
                            .filter_map(|a| match a {
                                TapeFuncallArg::Tape(t) => Some(*t),
                                TapeFuncallArg::Str(_) => None,
                            })
                            .collect();
                        for (k, &tk) in real_tape.iter().enumerate() {
                            adj[tk] += w * derivs[k];
                            let mut second_term = 0.0;
                            for (l, &tl) in real_tape.iter().enumerate() {
                                let (lo, hi) = if k <= l { (k, l) } else { (l, k) };
                                let h_kl = hes[lo + hi * (hi + 1) / 2];
                                second_term += h_kl * dot[tl];
                            }
                            adj_dot[tk] += wd * derivs[k] + w * second_term;
                        }
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
                | TapeOp::Cos(a)
                | TapeOp::Tan(a)
                | TapeOp::Atan(a)
                | TapeOp::Acos(a)
                | TapeOp::Sinh(a)
                | TapeOp::Cosh(a)
                | TapeOp::Tanh(a)
                | TapeOp::Asin(a)
                | TapeOp::Acosh(a)
                | TapeOp::Asinh(a)
                | TapeOp::Atanh(a) => {
                    emit_self(&var_sets[*a], &mut pairs);
                    var_sets[*a].clone()
                }
                // atan2(y, x) is nonlinear in both operands with a full
                // 2×2 second-derivative block; the structural superset is
                // every self/cross pair within the combined operand set.
                TapeOp::Atan2(a, b) => {
                    let combined: BTreeSet<usize> =
                        var_sets[*a].union(&var_sets[*b]).copied().collect();
                    emit_self(&combined, &mut pairs);
                    combined
                }
                // Comparisons / logical connectives are piecewise
                // constant: identically-zero derivative, so they
                // introduce no second-derivative pairs and carry no
                // variable dependence downstream (their result is a
                // constant as far as AD is concerned).
                TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => {
                    BTreeSet::new()
                }
                // Select passes through the active branch's value with
                // unit derivative, so it emits no pairs of its own; its
                // dependence set is the union of *both* branches
                // (either may become active as x varies — conservative
                // and correct for a structural superset). The condition
                // contributes no derivative and is excluded.
                TapeOp::Select(_c, t, e) => var_sets[*t].union(&var_sets[*e]).copied().collect(),
                // min/max are piecewise linear: the active operand passes
                // through with unit derivative, so the second derivative is
                // identically zero (no pairs). Their dependence set is the
                // union of both operands (either may become active as x
                // varies — conservative and correct for a structural
                // superset), mirroring Select.
                TapeOp::Min(a, b) | TapeOp::Max(a, b) => {
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Funcall(fc) => {
                    let args = &fc.args;
                    let mut combined: BTreeSet<usize> = BTreeSet::new();
                    for arg in args {
                        if let TapeFuncallArg::Tape(t) = arg {
                            for &vv in &var_sets[*t] {
                                combined.insert(vv);
                            }
                        }
                    }
                    emit_self(&combined, &mut pairs);
                    combined
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
    resolver: &ExternalResolver,
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
                    if let Some(idx) = try_emit_const_pow(a, c, ops, cache, resolver) {
                        return idx;
                    }
                }
            }
            let l = build_recursive(a, ops, cache, resolver);
            let r = build_recursive(b, ops, cache, resolver);
            let idx = ops.len();
            ops.push(match op {
                BinOp::Add => TapeOp::Add(l, r),
                BinOp::Sub => TapeOp::Sub(l, r),
                BinOp::Mul => TapeOp::Mul(l, r),
                BinOp::Div => TapeOp::Div(l, r),
                BinOp::Pow => TapeOp::Pow(l, r),
                BinOp::Atan2 => TapeOp::Atan2(l, r),
            });
            idx
        }
        Expr::Unary(op, a) => {
            let v = build_recursive(a, ops, cache, resolver);
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
                UnaryOp::Tan => TapeOp::Tan(v),
                UnaryOp::Atan => TapeOp::Atan(v),
                UnaryOp::Acos => TapeOp::Acos(v),
                UnaryOp::Sinh => TapeOp::Sinh(v),
                UnaryOp::Cosh => TapeOp::Cosh(v),
                UnaryOp::Tanh => TapeOp::Tanh(v),
                UnaryOp::Asin => TapeOp::Asin(v),
                UnaryOp::Acosh => TapeOp::Acosh(v),
                UnaryOp::Asinh => TapeOp::Asinh(v),
                UnaryOp::Atanh => TapeOp::Atanh(v),
            });
            idx
        }
        Expr::Sum(args) => {
            if args.is_empty() {
                let idx = ops.len();
                ops.push(TapeOp::Const(0.0));
                return idx;
            }
            let mut acc = build_recursive(&args[0], ops, cache, resolver);
            for a in &args[1..] {
                let next = build_recursive(a, ops, cache, resolver);
                let idx = ops.len();
                ops.push(TapeOp::Add(acc, next));
                acc = idx;
            }
            acc
        }
        // n-ary min/max fold to a left-associative chain of binary
        // Min/Max TapeOps. The chain reproduces the list extremum, and
        // the binary Min/Max AD arms route the (sub)gradient to the
        // active operand at each step — equivalent to selecting the one
        // active operand of the whole list. An empty list cannot arise
        // from a well-formed `.nl` MINLIST/MAXLIST (count >= 1); guard
        // with a 0 constant for safety rather than panicking.
        Expr::MinList(args) | Expr::MaxList(args) => {
            let is_min = matches!(expr, Expr::MinList(_));
            if args.is_empty() {
                let idx = ops.len();
                ops.push(TapeOp::Const(0.0));
                return idx;
            }
            let mut acc = build_recursive(&args[0], ops, cache, resolver);
            for a in &args[1..] {
                let next = build_recursive(a, ops, cache, resolver);
                let idx = ops.len();
                ops.push(if is_min {
                    TapeOp::Min(acc, next)
                } else {
                    TapeOp::Max(acc, next)
                });
                acc = idx;
            }
            acc
        }
        Expr::Cse(body) => {
            // Cache by Arc identity so each shared body is emitted into
            // the tape exactly once and every reference resolves to the
            // same result-slot index. Forward computes the body once;
            // reverse-mode adjoint sums contributions from every ref
            // into that shared slot — exact chain rule for shared
            // sub-expressions.
            let key = Arc::as_ptr(body) as *const Expr;
            if let Some(&idx) = cache.get(&key) {
                idx
            } else {
                let idx = build_recursive(body, ops, cache, resolver);
                cache.insert(key, idx);
                idx
            }
        }
        Expr::Compare(op, a, b) => {
            let l = build_recursive(a, ops, cache, resolver);
            let r = build_recursive(b, ops, cache, resolver);
            let idx = ops.len();
            ops.push(TapeOp::Cmp(*op, l, r));
            idx
        }
        Expr::And(a, b) => {
            let l = build_recursive(a, ops, cache, resolver);
            let r = build_recursive(b, ops, cache, resolver);
            let idx = ops.len();
            ops.push(TapeOp::And(l, r));
            idx
        }
        Expr::Or(a, b) => {
            let l = build_recursive(a, ops, cache, resolver);
            let r = build_recursive(b, ops, cache, resolver);
            let idx = ops.len();
            ops.push(TapeOp::Or(l, r));
            idx
        }
        Expr::Not(a) => {
            let v = build_recursive(a, ops, cache, resolver);
            let idx = ops.len();
            ops.push(TapeOp::Not(v));
            idx
        }
        Expr::Cond { cond, then_, else_ } => {
            let c = build_recursive(cond, ops, cache, resolver);
            let t = build_recursive(then_, ops, cache, resolver);
            let e = build_recursive(else_, ops, cache, resolver);
            let idx = ops.len();
            ops.push(TapeOp::Select(c, t, e));
            idx
        }
        Expr::Funcall { id, args } => {
            let (lib, name) = resolver
                .funcs_by_id
                .get(id)
                .unwrap_or_else(|| panic!("unresolved AMPL funcall id {id}"));
            let tape_args: Vec<TapeFuncallArg> = args
                .iter()
                .map(|a| match a {
                    FuncallArg::Real(e) => {
                        TapeFuncallArg::Tape(build_recursive(e, ops, cache, resolver))
                    }
                    FuncallArg::Str(s) => TapeFuncallArg::Str(s.clone()),
                })
                .collect();
            let idx = ops.len();
            ops.push(TapeOp::Funcall(Box::new(FuncallData {
                lib: Arc::clone(lib),
                name: name.clone(),
                args: tape_args,
            })));
            idx
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
    resolver: &ExternalResolver,
) -> Option<usize> {
    if c == 0.0 {
        let idx = ops.len();
        ops.push(TapeOp::Const(1.0));
        return Some(idx);
    }
    if c == 1.0 {
        return Some(build_recursive(base_expr, ops, cache, resolver));
    }
    if c == 0.5 {
        let b = build_recursive(base_expr, ops, cache, resolver);
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
        let b = build_recursive(base_expr, ops, cache, resolver);
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

// ============================================================
// HybridTape: per-summand local tapes + shared CSE prelude.
//
// Partial separability — the .nl Sum/Add structure — gets each
// summand its own local Vec<SummandOp>. CSE bodies (V-segments
// in .nl) that appear in two or more summands are promoted into
// a single shared `prelude: Vec<TapeOp>`; per-summand references
// to a promoted CSE are SummandOp::Shared(prelude_slot).
//
// This is strictly better than either extreme:
//   - per-summand Tape (no cross-summand sharing): re-inlines
//     every shared CSE, blows up tape size when many constraints
//     share a stencil derivative (Mittelmann *120 problems).
//   - GlobalTape (single shared Vec<TapeOp> for everything):
//     per-root reverse sweeps scatter across a many-MB buffer,
//     thrashing cache when no CSE is actually shared (lane_emden
//     120: each constraint owns its own ops → 50% regression
//     vs per-summand tapes).
//
// Forward: prelude once, then each summand's local pass.
// Reverse / forward-over-reverse: per-summand sweep over local
// reach (which propagates adjoints into prelude_adj at Shared
// boundaries), then a small reverse pass over the summand's
// prelude_reach to fold those into grad / Hessian.
// ============================================================

/// One slot in a per-summand local tape.
#[derive(Debug, Clone)]
pub enum SummandOp {
    /// Local op — operand indices reference other slots in the
    /// same per-summand vector.
    Local(TapeOp),
    /// Pull a value from the shared prelude at slot `usize`. No
    /// downstream cost beyond the lookup; adjoints flowing into
    /// this slot accumulate into the prelude adjoint buffer.
    Shared(usize),
}

#[derive(Debug, Clone)]
pub struct Summand {
    pub ops: Vec<SummandOp>,
    /// Local slot holding the summand's final value.
    pub root_slot: usize,
    /// Local slots reachable from `root_slot`, ascending (topo).
    pub local_reach: Vec<usize>,
    /// Prelude slots reachable from the summand's Shared refs,
    /// ascending (topo in prelude's operand DAG).
    pub prelude_reach: Vec<usize>,
    /// Variables touched by Var ops inside `local_reach`.
    pub local_vars: Vec<usize>,
    /// Variables touched by Var ops inside `prelude_reach`.
    pub prelude_vars: Vec<usize>,
    /// `local_vars ∪ prelude_vars`, sorted. Hessian j-loop set.
    pub all_vars: Vec<usize>,
}

#[derive(Debug)]
pub struct HybridTape {
    /// Shared CSE bodies. Slot indices in `SummandOp::Shared`
    /// point here; this Vec is built bottom-up by `build_recursive`,
    /// so operand indices are always less than the consumer's
    /// index (topo in ascending order).
    pub prelude: Vec<TapeOp>,
    pub summands: Vec<Summand>,
}

impl HybridTape {
    /// Build hybrid tape from a list of root expressions. CSE
    /// bodies referenced from ≥ 2 roots are promoted into the
    /// shared prelude; CSEs touched by only one root are inlined
    /// into that summand's local ops.
    pub fn build_multi(exprs: &[Expr]) -> Self {
        // Pass 1: per-Cse-pointer count of how many roots reference
        // it (each root contributes at most 1 to the count). The
        // ≥2 threshold means a CSE is shared across summands.
        let mut cse_count: HashMap<*const Expr, usize> = HashMap::new();
        for e in exprs {
            let mut seen_in_root: HashSet<*const Expr> = HashSet::new();
            count_cse_appearances(e, &mut seen_in_root, &mut cse_count);
        }

        // Pass 2: build prelude + each summand. The summand builder
        // hits the prelude path lazily — only when it encounters a
        // promoted Cse — so the prelude grows only with bodies that
        // are actually referenced from multiple summands.
        let mut prelude: Vec<TapeOp> = Vec::new();
        let mut prelude_map: HashMap<*const Expr, usize> = HashMap::new();
        let mut summands: Vec<Summand> = Vec::with_capacity(exprs.len());
        for e in exprs {
            let mut local: Vec<SummandOp> = Vec::new();
            let mut local_cache: HashMap<*const Expr, usize> = HashMap::new();
            let root_slot = build_into_summand(
                e,
                &mut local,
                &mut local_cache,
                &mut prelude,
                &mut prelude_map,
                &cse_count,
            );
            summands.push(Summand {
                ops: local,
                root_slot,
                local_reach: Vec::new(),
                prelude_reach: Vec::new(),
                local_vars: Vec::new(),
                prelude_vars: Vec::new(),
                all_vars: Vec::new(),
            });
        }

        // Pass 3: per-summand reach / vars. Prelude reach uses an
        // epoch-tagged shared visited buffer so total cost stays
        // O(Σ |prelude_reach_i|) rather than O(n_summands × |prelude|).
        let mut p_visited: Vec<u32> = vec![0; prelude.len()];
        let mut p_epoch: u32 = 0;
        let mut p_stack: Vec<usize> = Vec::new();
        for s in &mut summands {
            let (local_reach, shared_refs) = compute_local_reach(&s.ops, s.root_slot);
            s.local_reach = local_reach;

            let mut lv: BTreeSet<usize> = BTreeSet::new();
            for &i in &s.local_reach {
                if let SummandOp::Local(TapeOp::Var(j)) = &s.ops[i] {
                    lv.insert(*j);
                }
            }
            s.local_vars = lv.iter().copied().collect();

            if !shared_refs.is_empty() {
                p_epoch += 1;
                let mut preach: Vec<usize> = Vec::new();
                for &start in &shared_refs {
                    bfs_prelude(
                        &prelude,
                        start,
                        &mut p_visited,
                        p_epoch,
                        &mut p_stack,
                        &mut preach,
                    );
                }
                preach.sort_unstable();
                s.prelude_vars = vars_in(&prelude, &preach);
                s.prelude_reach = preach;
            }

            let mut av: BTreeSet<usize> = lv;
            for &v in &s.prelude_vars {
                av.insert(v);
            }
            s.all_vars = av.into_iter().collect();
        }

        HybridTape { prelude, summands }
    }

    pub fn n_prelude_ops(&self) -> usize {
        self.prelude.len()
    }
    pub fn n_summands(&self) -> usize {
        self.summands.len()
    }
    pub fn max_summand_ops(&self) -> usize {
        self.summands.iter().map(|s| s.ops.len()).max().unwrap_or(0)
    }
    pub fn total_local_ops(&self) -> usize {
        self.summands.iter().map(|s| s.ops.len()).sum()
    }

    /// Forward sweep over the shared prelude. `prelude_vals` must
    /// have length `n_prelude_ops`.
    pub fn forward_prelude(&self, x: &[f64], prelude_vals: &mut [f64]) {
        debug_assert_eq!(prelude_vals.len(), self.prelude.len());
        for i in 0..self.prelude.len() {
            prelude_vals[i] = fwd_step(&self.prelude[i], x, prelude_vals);
        }
    }

    /// Forward sweep over one summand. `local_vals` must hold at
    /// least `s.ops.len()` entries.
    pub fn forward_summand(
        &self,
        s: &Summand,
        x: &[f64],
        prelude_vals: &[f64],
        local_vals: &mut [f64],
    ) {
        debug_assert!(local_vals.len() >= s.ops.len());
        for i in 0..s.ops.len() {
            local_vals[i] = match &s.ops[i] {
                SummandOp::Local(op) => fwd_step(op, x, local_vals),
                SummandOp::Shared(k) => prelude_vals[*k],
            };
        }
    }

    /// Value at the summand root after `forward_summand`.
    #[inline]
    pub fn root_value(&self, s: &Summand, local_vals: &[f64]) -> f64 {
        local_vals[s.root_slot]
    }

    /// Reverse-mode gradient for one summand. Walks `local_reach`
    /// in reverse — propagating adjoints into `prelude_adj` at
    /// Shared boundaries — and then walks `prelude_reach` in
    /// reverse to land contributions in `grad`. Scratch arrays
    /// `local_adj` and `prelude_adj` are zeroed only at the slots
    /// actually touched.
    #[allow(clippy::too_many_arguments)]
    pub fn gradient_summand(
        &self,
        s: &Summand,
        prelude_vals: &[f64],
        local_vals: &[f64],
        seed: f64,
        grad: &mut [f64],
        local_adj: &mut [f64],
        prelude_adj: &mut [f64],
    ) {
        if seed == 0.0 || s.local_reach.is_empty() {
            return;
        }
        for &i in &s.local_reach {
            local_adj[i] = 0.0;
        }
        for &i in &s.prelude_reach {
            prelude_adj[i] = 0.0;
        }
        local_adj[s.root_slot] = seed;
        for &i in s.local_reach.iter().rev() {
            let a = local_adj[i];
            if a == 0.0 {
                continue;
            }
            match &s.ops[i] {
                SummandOp::Local(op) => rev_step(op, i, local_vals, local_adj, a, grad),
                SummandOp::Shared(k) => {
                    prelude_adj[*k] += a;
                }
            }
        }
        for &i in s.prelude_reach.iter().rev() {
            let a = prelude_adj[i];
            if a == 0.0 {
                continue;
            }
            rev_step(&self.prelude[i], i, prelude_vals, prelude_adj, a, grad);
        }
    }

    /// Forward-over-reverse Hessian for one summand with multiplier
    /// `weight`. Iterates over `s.all_vars`; for each seed variable
    /// j: (1) forward tangent through prelude_reach then local_reach,
    /// (2) reverse over local (folding adj/adj_dot into prelude at
    /// Shared boundaries), (3) reverse over prelude_reach. All
    /// scratch buffers are zeroed only at the touched slots inside
    /// the per-j loop.
    #[allow(clippy::too_many_arguments)]
    pub fn hessian_summand(
        &self,
        s: &Summand,
        prelude_vals: &[f64],
        local_vals: &[f64],
        weight: f64,
        hess_map: &HashMap<(usize, usize), usize>,
        values: &mut [f64],
        local_dot: &mut [f64],
        local_adj: &mut [f64],
        local_adj_dot: &mut [f64],
        prelude_dot: &mut [f64],
        prelude_adj: &mut [f64],
        prelude_adj_dot: &mut [f64],
    ) {
        if weight == 0.0 || s.local_reach.is_empty() {
            return;
        }
        for &j in &s.all_vars {
            for &i in &s.local_reach {
                local_dot[i] = 0.0;
                local_adj[i] = 0.0;
                local_adj_dot[i] = 0.0;
            }
            for &i in &s.prelude_reach {
                prelude_dot[i] = 0.0;
                prelude_adj[i] = 0.0;
                prelude_adj_dot[i] = 0.0;
            }
            for &i in &s.prelude_reach {
                prelude_dot[i] = fwd_tan_step(&self.prelude[i], j, prelude_vals, prelude_dot, i);
            }
            for &i in &s.local_reach {
                local_dot[i] = match &s.ops[i] {
                    SummandOp::Local(op) => fwd_tan_step(op, j, local_vals, local_dot, i),
                    SummandOp::Shared(k) => prelude_dot[*k],
                };
            }
            local_adj[s.root_slot] = 1.0;
            for &i in s.local_reach.iter().rev() {
                let w = local_adj[i];
                let wd = local_adj_dot[i];
                if w == 0.0 && wd == 0.0 {
                    continue;
                }
                match &s.ops[i] {
                    SummandOp::Local(op) => {
                        ror_step(
                            op,
                            i,
                            j,
                            local_vals,
                            local_dot,
                            local_adj,
                            local_adj_dot,
                            w,
                            wd,
                            weight,
                            hess_map,
                            values,
                        );
                    }
                    SummandOp::Shared(k) => {
                        prelude_adj[*k] += w;
                        prelude_adj_dot[*k] += wd;
                    }
                }
            }
            for &i in s.prelude_reach.iter().rev() {
                let w = prelude_adj[i];
                let wd = prelude_adj_dot[i];
                if w == 0.0 && wd == 0.0 {
                    continue;
                }
                ror_step(
                    &self.prelude[i],
                    i,
                    j,
                    prelude_vals,
                    prelude_dot,
                    prelude_adj,
                    prelude_adj_dot,
                    w,
                    wd,
                    weight,
                    hess_map,
                    values,
                );
            }
        }
    }

    /// Structural Hessian sparsity over the whole hybrid tape:
    /// every pair the prelude or any summand can produce.
    pub fn hessian_sparsity_all(&self) -> BTreeSet<(usize, usize)> {
        let mut pairs = hessian_sparsity_impl(&self.prelude);

        // Per-prelude-slot var-set, reused across summands as the
        // var-set carrier for Shared refs.
        let prelude_var_sets = compute_var_sets(&self.prelude);

        for s in &self.summands {
            summand_sparsity(&s.ops, &prelude_var_sets, &mut pairs);
        }
        pairs
    }
}

/// Pass-1 helper: per-root walk that increments `counts[ptr]` the
/// first time a Cse pointer is encountered in this root. Recursing
/// into the body is gated on the first visit to avoid quadratic
/// blowup on heavily shared CSE DAGs.
/// True when `expr` (or any subexpression) is an AMPL external function
/// call. The hybrid summand path rejects funcalls outright, but the
/// *promoted*-CSE branch emits a shared CSE body via `build_recursive`
/// with an **empty** `ExternalResolver::default()` — it has no resolver
/// of its own. Without this pre-scan a funcall buried in a promoted CSE
/// would reach `build_recursive`'s `Expr::Funcall` arm and panic with the
/// misleading `unresolved AMPL funcall id <n>` message, instead of the
/// clear "not supported on the hybrid path" message the non-promoted
/// summand path raises. Pre-scanning makes both paths report the same
/// reason. (Funcalls are unsupported on the hybrid path regardless of
/// whether the id would resolve, so this never rejects a buildable tape.)
fn cse_contains_funcall(expr: &Expr) -> bool {
    match expr {
        Expr::Funcall { .. } => true,
        Expr::Const(_) | Expr::Var(_) => false,
        Expr::Binary(_, a, b) => cse_contains_funcall(a) || cse_contains_funcall(b),
        Expr::Unary(_, a) => cse_contains_funcall(a),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            args.iter().any(cse_contains_funcall)
        }
        Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            cse_contains_funcall(a) || cse_contains_funcall(b)
        }
        Expr::Not(a) => cse_contains_funcall(a),
        Expr::Cond { cond, then_, else_ } => {
            cse_contains_funcall(cond) || cse_contains_funcall(then_) || cse_contains_funcall(else_)
        }
        Expr::Cse(body) => cse_contains_funcall(body),
    }
}

fn count_cse_appearances(
    e: &Expr,
    seen_in_root: &mut HashSet<*const Expr>,
    counts: &mut HashMap<*const Expr, usize>,
) {
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) => {
            count_cse_appearances(a, seen_in_root, counts);
            count_cse_appearances(b, seen_in_root, counts);
        }
        Expr::Unary(_, a) => count_cse_appearances(a, seen_in_root, counts),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                count_cse_appearances(a, seen_in_root, counts);
            }
        }
        Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            count_cse_appearances(a, seen_in_root, counts);
            count_cse_appearances(b, seen_in_root, counts);
        }
        Expr::Not(a) => count_cse_appearances(a, seen_in_root, counts),
        Expr::Cond { cond, then_, else_ } => {
            count_cse_appearances(cond, seen_in_root, counts);
            count_cse_appearances(then_, seen_in_root, counts);
            count_cse_appearances(else_, seen_in_root, counts);
        }
        Expr::Cse(body) => {
            let key = Arc::as_ptr(body) as *const Expr;
            if seen_in_root.insert(key) {
                *counts.entry(key).or_insert(0) += 1;
                count_cse_appearances(body, seen_in_root, counts);
            }
        }
        Expr::Funcall { args, .. } => {
            for arg in args {
                if let FuncallArg::Real(e) = arg {
                    count_cse_appearances(e, seen_in_root, counts);
                }
            }
        }
    }
}

/// Recursive summand builder. CSEs that meet the promotion bar
/// (≥ 2 roots reference them per `cse_count`) get a single prelude
/// emission via `build_recursive`; the summand records a Shared op
/// pointing at the prelude slot. Non-promoted CSEs are inlined
/// into the summand with intra-summand Arc-pointer dedup.
fn build_into_summand(
    expr: &Expr,
    local: &mut Vec<SummandOp>,
    local_cache: &mut HashMap<*const Expr, usize>,
    prelude: &mut Vec<TapeOp>,
    prelude_map: &mut HashMap<*const Expr, usize>,
    cse_count: &HashMap<*const Expr, usize>,
) -> usize {
    match expr {
        Expr::Const(c) => {
            let i = local.len();
            local.push(SummandOp::Local(TapeOp::Const(*c)));
            i
        }
        Expr::Var(j) => {
            let i = local.len();
            local.push(SummandOp::Local(TapeOp::Var(*j)));
            i
        }
        Expr::Binary(op, a, b) => {
            if let BinOp::Pow = op {
                if let Some(c) = peek_const(b) {
                    if let Some(i) = try_emit_const_pow_summand(
                        a,
                        c,
                        local,
                        local_cache,
                        prelude,
                        prelude_map,
                        cse_count,
                    ) {
                        return i;
                    }
                }
            }
            let l = build_into_summand(a, local, local_cache, prelude, prelude_map, cse_count);
            let r = build_into_summand(b, local, local_cache, prelude, prelude_map, cse_count);
            let i = local.len();
            local.push(SummandOp::Local(match op {
                BinOp::Add => TapeOp::Add(l, r),
                BinOp::Sub => TapeOp::Sub(l, r),
                BinOp::Mul => TapeOp::Mul(l, r),
                BinOp::Div => TapeOp::Div(l, r),
                BinOp::Pow => TapeOp::Pow(l, r),
                BinOp::Atan2 => TapeOp::Atan2(l, r),
            }));
            i
        }
        Expr::Unary(op, a) => {
            let v = build_into_summand(a, local, local_cache, prelude, prelude_map, cse_count);
            let i = local.len();
            local.push(SummandOp::Local(match op {
                UnaryOp::Neg => TapeOp::Neg(v),
                UnaryOp::Sqrt => TapeOp::Sqrt(v),
                UnaryOp::Log => TapeOp::Log(v),
                UnaryOp::Log10 => TapeOp::Log10(v),
                UnaryOp::Exp => TapeOp::Exp(v),
                UnaryOp::Abs => TapeOp::Abs(v),
                UnaryOp::Sin => TapeOp::Sin(v),
                UnaryOp::Cos => TapeOp::Cos(v),
                UnaryOp::Tan => TapeOp::Tan(v),
                UnaryOp::Atan => TapeOp::Atan(v),
                UnaryOp::Acos => TapeOp::Acos(v),
                UnaryOp::Sinh => TapeOp::Sinh(v),
                UnaryOp::Cosh => TapeOp::Cosh(v),
                UnaryOp::Tanh => TapeOp::Tanh(v),
                UnaryOp::Asin => TapeOp::Asin(v),
                UnaryOp::Acosh => TapeOp::Acosh(v),
                UnaryOp::Asinh => TapeOp::Asinh(v),
                UnaryOp::Atanh => TapeOp::Atanh(v),
            }));
            i
        }
        Expr::Sum(args) => {
            if args.is_empty() {
                let i = local.len();
                local.push(SummandOp::Local(TapeOp::Const(0.0)));
                return i;
            }
            let mut acc = build_into_summand(
                &args[0],
                local,
                local_cache,
                prelude,
                prelude_map,
                cse_count,
            );
            for a in &args[1..] {
                let nxt =
                    build_into_summand(a, local, local_cache, prelude, prelude_map, cse_count);
                let i = local.len();
                local.push(SummandOp::Local(TapeOp::Add(acc, nxt)));
                acc = i;
            }
            acc
        }
        Expr::Cse(body) => {
            let key = Arc::as_ptr(body) as *const Expr;
            if let Some(&li) = local_cache.get(&key) {
                return li;
            }
            let promoted = cse_count.get(&key).copied().unwrap_or(0) >= 2;
            if promoted {
                // `build_recursive` below runs with an empty resolver, so a
                // funcall hidden inside the promoted body would panic with the
                // misleading "unresolved AMPL funcall id" message rather than
                // the clear hybrid-unsupported message the non-promoted summand
                // path (and the `Expr::Funcall` arm at the bottom) raises.
                // Reject it up front so both CSE paths report the same reason.
                if cse_contains_funcall(body) {
                    panic!(
                        "HybridTape: AMPL external function calls are not supported on the \
                         hybrid (partial-separability) tape path. Build with \
                         Tape::build_with_externals instead."
                    );
                }
                // Build (or reuse) the prelude slot for this CSE.
                // `build_recursive(expr, ...)` hits the Cse arm,
                // emits the body once into prelude, and caches it
                // in `prelude_map` keyed by this Arc pointer.
                let pslot =
                    build_recursive(expr, prelude, prelude_map, &ExternalResolver::default());
                let li = local.len();
                local.push(SummandOp::Shared(pslot));
                local_cache.insert(key, li);
                li
            } else {
                let li =
                    build_into_summand(body, local, local_cache, prelude, prelude_map, cse_count);
                local_cache.insert(key, li);
                li
            }
        }
        Expr::Compare(_, _, _)
        | Expr::And(_, _)
        | Expr::Or(_, _)
        | Expr::Not(_)
        | Expr::Cond { .. }
        | Expr::MinList(_)
        | Expr::MaxList(_) => {
            panic!(
                "HybridTape: conditional / logical / min-max opcodes (comparisons, \
                 AND/OR/NOT, if-then-else, min/max lists) are not supported on the \
                 hybrid (partial-separability) tape path. Build with \
                 Tape::build_with_externals instead."
            );
        }
        Expr::Funcall { .. } => {
            panic!(
                "HybridTape: AMPL external function calls are not supported on the \
                 hybrid (partial-separability) tape path. Build with Tape::build_with_externals \
                 instead."
            );
        }
    }
}

/// Pow-lowering specialised for summand builds. Mirrors
/// `try_emit_const_pow` but with summand-flavoured emission.
fn try_emit_const_pow_summand(
    base_expr: &Expr,
    c: f64,
    local: &mut Vec<SummandOp>,
    local_cache: &mut HashMap<*const Expr, usize>,
    prelude: &mut Vec<TapeOp>,
    prelude_map: &mut HashMap<*const Expr, usize>,
    cse_count: &HashMap<*const Expr, usize>,
) -> Option<usize> {
    if c == 0.0 {
        let i = local.len();
        local.push(SummandOp::Local(TapeOp::Const(1.0)));
        return Some(i);
    }
    if c == 1.0 {
        return Some(build_into_summand(
            base_expr,
            local,
            local_cache,
            prelude,
            prelude_map,
            cse_count,
        ));
    }
    if c == 0.5 {
        let b = build_into_summand(
            base_expr,
            local,
            local_cache,
            prelude,
            prelude_map,
            cse_count,
        );
        let i = local.len();
        local.push(SummandOp::Local(TapeOp::Sqrt(b)));
        return Some(i);
    }
    if c.is_finite() && c.fract() == 0.0 && c.abs() <= 8.0 {
        let n = c.abs() as u32;
        if n == 0 {
            let i = local.len();
            local.push(SummandOp::Local(TapeOp::Const(1.0)));
            return Some(i);
        }
        let b = build_into_summand(
            base_expr,
            local,
            local_cache,
            prelude,
            prelude_map,
            cse_count,
        );
        let pos = emit_int_pow_summand(b, n, local);
        if c < 0.0 {
            let one_idx = local.len();
            local.push(SummandOp::Local(TapeOp::Const(1.0)));
            let i = local.len();
            local.push(SummandOp::Local(TapeOp::Div(one_idx, pos)));
            return Some(i);
        }
        return Some(pos);
    }
    None
}

fn emit_int_pow_summand(base: usize, n: u32, local: &mut Vec<SummandOp>) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return base;
    }
    let half = emit_int_pow_summand(base, n / 2, local);
    let squared = local.len();
    local.push(SummandOp::Local(TapeOp::Mul(half, half)));
    if n % 2 == 1 {
        let i = local.len();
        local.push(SummandOp::Local(TapeOp::Mul(squared, base)));
        i
    } else {
        squared
    }
}

/// Walk a summand's local op DAG from `root`, returning the
/// reachable local slots (sorted ascending) plus the distinct
/// prelude slots referenced by any Shared op along the way.
fn compute_local_reach(ops: &[SummandOp], root: usize) -> (Vec<usize>, Vec<usize>) {
    let mut visited = vec![false; ops.len()];
    let mut reach: Vec<usize> = Vec::new();
    let mut shared: BTreeSet<usize> = BTreeSet::new();
    let mut stack: Vec<usize> = Vec::with_capacity(16);
    visited[root] = true;
    reach.push(root);
    stack.push(root);
    while let Some(s) = stack.pop() {
        match &ops[s] {
            SummandOp::Local(op) => {
                let (a, b) = op_operands(op);
                if let Some(a) = a {
                    if !visited[a] {
                        visited[a] = true;
                        reach.push(a);
                        stack.push(a);
                    }
                }
                if let Some(b) = b {
                    if !visited[b] {
                        visited[b] = true;
                        reach.push(b);
                        stack.push(b);
                    }
                }
            }
            SummandOp::Shared(k) => {
                shared.insert(*k);
            }
        }
    }
    reach.sort_unstable();
    (reach, shared.into_iter().collect())
}

/// Epoch-tagged BFS over the prelude operand DAG, accumulating
/// reachable slots into `out`. Caller is responsible for sorting
/// `out` after a batch of starts has been processed.
fn bfs_prelude(
    prelude: &[TapeOp],
    start: usize,
    visited: &mut [u32],
    cur: u32,
    stack: &mut Vec<usize>,
    out: &mut Vec<usize>,
) {
    if visited[start] == cur {
        return;
    }
    visited[start] = cur;
    out.push(start);
    stack.push(start);
    while let Some(s) = stack.pop() {
        let (a, b) = op_operands(&prelude[s]);
        if let Some(a) = a {
            if visited[a] != cur {
                visited[a] = cur;
                out.push(a);
                stack.push(a);
            }
        }
        if let Some(b) = b {
            if visited[b] != cur {
                visited[b] = cur;
                out.push(b);
                stack.push(b);
            }
        }
    }
}

/// Per-op var-set for the prelude — every slot's transitive
/// variable footprint. Used by `summand_sparsity` to expand
/// `SummandOp::Shared(k)` into its var-set carrier.
fn compute_var_sets(ops: &[TapeOp]) -> Vec<BTreeSet<usize>> {
    let mut out: Vec<BTreeSet<usize>> = Vec::with_capacity(ops.len());
    for op in ops {
        let vs: BTreeSet<usize> = match op {
            TapeOp::Const(_) => BTreeSet::new(),
            TapeOp::Var(j) => {
                let mut s = BTreeSet::new();
                s.insert(*j);
                s
            }
            TapeOp::Add(a, b)
            | TapeOp::Sub(a, b)
            | TapeOp::Mul(a, b)
            | TapeOp::Div(a, b)
            | TapeOp::Pow(a, b)
            | TapeOp::Atan2(a, b) => out[*a].union(&out[*b]).copied().collect(),
            TapeOp::Neg(a)
            | TapeOp::Abs(a)
            | TapeOp::Sqrt(a)
            | TapeOp::Exp(a)
            | TapeOp::Log(a)
            | TapeOp::Log10(a)
            | TapeOp::Sin(a)
            | TapeOp::Cos(a)
            | TapeOp::Tan(a)
            | TapeOp::Atan(a)
            | TapeOp::Acos(a)
            | TapeOp::Sinh(a)
            | TapeOp::Cosh(a)
            | TapeOp::Tanh(a)
            | TapeOp::Asin(a)
            | TapeOp::Acosh(a)
            | TapeOp::Asinh(a)
            | TapeOp::Atanh(a) => out[*a].clone(),
            TapeOp::Cmp(_, _, _)
            | TapeOp::And(_, _)
            | TapeOp::Or(_, _)
            | TapeOp::Not(_)
            | TapeOp::Select(_, _, _)
            | TapeOp::Min(_, _)
            | TapeOp::Max(_, _) => unreachable!(
                "HybridTape prelude cannot contain conditional / logical / min-max \
                 TapeOps; build_into_summand panics on those Expr variants."
            ),
            TapeOp::Funcall(_) => unreachable!(
                "HybridTape prelude cannot contain TapeOp::Funcall; \
                 build_into_summand panics on Expr::Funcall."
            ),
        };
        out.push(vs);
    }
    out
}

/// Per-op Hessian-sparsity propagation over a summand's mixed
/// SummandOp slice. Shared refs contribute their prelude var-set
/// but do not themselves emit pairs (those came from
/// `hessian_sparsity_impl(&prelude)`).
fn summand_sparsity(
    ops: &[SummandOp],
    prelude_var_sets: &[BTreeSet<usize>],
    pairs: &mut BTreeSet<(usize, usize)>,
) {
    let mut var_sets: Vec<BTreeSet<usize>> = Vec::with_capacity(ops.len());
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
    for so in ops {
        let vset: BTreeSet<usize> = match so {
            SummandOp::Shared(k) => prelude_var_sets[*k].clone(),
            SummandOp::Local(op) => match op {
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
                    emit_cross(&var_sets[*a], &var_sets[*b], pairs);
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Div(a, b) => {
                    emit_cross(&var_sets[*a], &var_sets[*b], pairs);
                    emit_self(&var_sets[*b], pairs);
                    var_sets[*a].union(&var_sets[*b]).copied().collect()
                }
                TapeOp::Pow(a, b) | TapeOp::Atan2(a, b) => {
                    let combined: BTreeSet<usize> =
                        var_sets[*a].union(&var_sets[*b]).copied().collect();
                    emit_self(&combined, pairs);
                    combined
                }
                TapeOp::Sqrt(a)
                | TapeOp::Exp(a)
                | TapeOp::Log(a)
                | TapeOp::Log10(a)
                | TapeOp::Sin(a)
                | TapeOp::Cos(a)
                | TapeOp::Tan(a)
                | TapeOp::Atan(a)
                | TapeOp::Acos(a)
                | TapeOp::Sinh(a)
                | TapeOp::Cosh(a)
                | TapeOp::Tanh(a)
                | TapeOp::Asin(a)
                | TapeOp::Acosh(a)
                | TapeOp::Asinh(a)
                | TapeOp::Atanh(a) => {
                    emit_self(&var_sets[*a], pairs);
                    var_sets[*a].clone()
                }
                TapeOp::Cmp(_, _, _)
                | TapeOp::And(_, _)
                | TapeOp::Or(_, _)
                | TapeOp::Not(_)
                | TapeOp::Select(_, _, _)
                | TapeOp::Min(_, _)
                | TapeOp::Max(_, _) => unreachable!(
                    "HybridTape summand cannot contain conditional / logical / min-max \
                     TapeOps; build_into_summand panics on those Expr variants."
                ),
                TapeOp::Funcall(_) => unreachable!(
                    "HybridTape summand cannot contain TapeOp::Funcall; \
                     build_into_summand panics on Expr::Funcall."
                ),
            },
        };
        var_sets.push(vset);
    }
}

/// Operand indices of a `TapeOp`, normalized into a fixed-length
/// array so callers don't need to re-match every site.
#[inline]
fn op_operands(op: &TapeOp) -> (Option<usize>, Option<usize>) {
    match op {
        TapeOp::Const(_) | TapeOp::Var(_) => (None, None),
        TapeOp::Add(a, b)
        | TapeOp::Sub(a, b)
        | TapeOp::Mul(a, b)
        | TapeOp::Div(a, b)
        | TapeOp::Pow(a, b)
        | TapeOp::Atan2(a, b) => (Some(*a), Some(*b)),
        TapeOp::Neg(a)
        | TapeOp::Abs(a)
        | TapeOp::Sqrt(a)
        | TapeOp::Exp(a)
        | TapeOp::Log(a)
        | TapeOp::Log10(a)
        | TapeOp::Sin(a)
        | TapeOp::Cos(a)
        | TapeOp::Tan(a)
        | TapeOp::Atan(a)
        | TapeOp::Acos(a)
        | TapeOp::Sinh(a)
        | TapeOp::Cosh(a)
        | TapeOp::Tanh(a)
        | TapeOp::Asin(a)
        | TapeOp::Acosh(a)
        | TapeOp::Asinh(a)
        | TapeOp::Atanh(a) => (Some(*a), None),
        // Conditional / logical TapeOps never reach the HybridTape
        // operand-walk (build_into_summand rejects them). Cmp/And/Or
        // have two operands; Not has one; Select's three can't be
        // expressed in this two-slot shape, so it would be a bug to
        // see it here.
        TapeOp::Cmp(_, a, b) | TapeOp::And(a, b) | TapeOp::Or(a, b) => (Some(*a), Some(*b)),
        TapeOp::Not(a) => (Some(*a), None),
        TapeOp::Select(_, _, _) => unreachable!(
            "op_operands: TapeOp::Select has three operands and is unsupported on \
             the HybridTape path"
        ),
        TapeOp::Min(_, _) | TapeOp::Max(_, _) => unreachable!(
            "op_operands: TapeOp::Min/Max are unsupported on the HybridTape path \
             (build_into_summand rejects min/max lists)"
        ),
        TapeOp::Funcall(_) => (None, None),
    }
}

fn vars_in(ops: &[TapeOp], reach: &[usize]) -> Vec<usize> {
    let mut s: BTreeSet<usize> = BTreeSet::new();
    for &i in reach {
        if let TapeOp::Var(j) = &ops[i] {
            s.insert(*j);
        }
    }
    s.into_iter().collect()
}

// ----- Free-function AD step kernels used by GlobalTape -----

#[inline]
fn fwd_step(op: &TapeOp, x: &[f64], vals: &[f64]) -> f64 {
    match op {
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
        TapeOp::Tan(a) => vals[*a].tan(),
        TapeOp::Atan(a) => vals[*a].atan(),
        TapeOp::Acos(a) => vals[*a].acos(),
        TapeOp::Sinh(a) => vals[*a].sinh(),
        TapeOp::Cosh(a) => vals[*a].cosh(),
        TapeOp::Tanh(a) => vals[*a].tanh(),
        TapeOp::Asin(a) => vals[*a].asin(),
        TapeOp::Acosh(a) => vals[*a].acosh(),
        TapeOp::Asinh(a) => vals[*a].asinh(),
        TapeOp::Atanh(a) => vals[*a].atanh(),
        TapeOp::Atan2(a, b) => vals[*a].atan2(vals[*b]),
        TapeOp::Cmp(_, _, _)
        | TapeOp::And(_, _)
        | TapeOp::Or(_, _)
        | TapeOp::Not(_)
        | TapeOp::Select(_, _, _)
        | TapeOp::Min(_, _)
        | TapeOp::Max(_, _) => panic!(
            "GlobalTape free-function kernels do not implement conditional / logical \
             / min-max TapeOps; use the Tape (build_with_externals) interpreter path \
             instead."
        ),
        TapeOp::Funcall(fc) => {
            let FuncallData { lib, name, args } = fc.as_ref();
            let call_args = funcall_to_ext_args(args, vals);
            let res = lib
                .eval(name, &call_args, false, false)
                .unwrap_or_else(|e| panic!("external function '{name}' eval failed: {e}"));
            res.value
        }
    }
}

#[inline]
fn rev_step(op: &TapeOp, i: usize, vals: &[f64], adj: &mut [f64], a: f64, grad: &mut [f64]) {
    match op {
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
        TapeOp::Tan(j) => {
            let t = vals[i];
            adj[*j] += a * (1.0 + t * t);
        }
        TapeOp::Atan(j) => {
            let u = vals[*j];
            adj[*j] += a / (1.0 + u * u);
        }
        TapeOp::Acos(j) => {
            let u = vals[*j];
            adj[*j] -= a / (1.0 - u * u).sqrt();
        }
        TapeOp::Sinh(j) => {
            adj[*j] += a * vals[*j].cosh();
        }
        TapeOp::Cosh(j) => {
            adj[*j] += a * vals[*j].sinh();
        }
        TapeOp::Tanh(j) => {
            let t = vals[i];
            adj[*j] += a * (1.0 - t * t);
        }
        TapeOp::Asin(j) => {
            let u = vals[*j];
            adj[*j] += a / (1.0 - u * u).sqrt();
        }
        TapeOp::Acosh(j) => {
            let u = vals[*j];
            adj[*j] += a / (u * u - 1.0).sqrt();
        }
        TapeOp::Asinh(j) => {
            let u = vals[*j];
            adj[*j] += a / (u * u + 1.0).sqrt();
        }
        TapeOp::Atanh(j) => {
            let u = vals[*j];
            adj[*j] += a / (1.0 - u * u);
        }
        TapeOp::Atan2(l, r) => {
            let y = vals[*l];
            let x = vals[*r];
            let d = y * y + x * x;
            adj[*l] += a * (x / d);
            adj[*r] += a * (-y / d);
        }
        TapeOp::Cmp(_, _, _)
        | TapeOp::And(_, _)
        | TapeOp::Or(_, _)
        | TapeOp::Not(_)
        | TapeOp::Select(_, _, _)
        | TapeOp::Min(_, _)
        | TapeOp::Max(_, _) => panic!(
            "GlobalTape free-function kernels do not implement conditional / logical \
             / min-max TapeOps; use the Tape (build_with_externals) interpreter path \
             instead."
        ),
        TapeOp::Funcall(fc) => {
            let FuncallData { lib, name, args } = fc.as_ref();
            let call_args = funcall_to_ext_args(args, vals);
            let res = lib
                .eval(name, &call_args, true, false)
                .unwrap_or_else(|e| panic!("external function '{name}' reverse eval failed: {e}"));
            let derivs = res.derivs.expect("want_derivs=true returns derivs");
            let mut k = 0usize;
            for arg in args {
                if let TapeFuncallArg::Tape(idx) = arg {
                    adj[*idx] += a * derivs[k];
                    k += 1;
                }
            }
            let _ = i;
            let _ = grad;
        }
    }
}

#[inline]
fn fwd_tan_step(op: &TapeOp, seed_var: usize, vals: &[f64], dot: &[f64], i: usize) -> f64 {
    match op {
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
            // Match the reverse-mode gradient's guard (`rv != 0.0` only): at base
            // u == 0 the slope is still well defined for r >= 1 (and a genuine
            // ±inf for r < 1), so it must not be silently dropped, or the forward
            // tangent disagrees with the reverse gradient.
            if r != 0.0 {
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
        TapeOp::Tan(a) => {
            let t = vals[i];
            dot[*a] * (1.0 + t * t)
        }
        TapeOp::Atan(a) => {
            let u = vals[*a];
            dot[*a] / (1.0 + u * u)
        }
        TapeOp::Acos(a) => {
            let u = vals[*a];
            -dot[*a] / (1.0 - u * u).sqrt()
        }
        TapeOp::Sinh(a) => dot[*a] * vals[*a].cosh(),
        TapeOp::Cosh(a) => dot[*a] * vals[*a].sinh(),
        TapeOp::Tanh(a) => {
            let t = vals[i];
            dot[*a] * (1.0 - t * t)
        }
        TapeOp::Asin(a) => {
            let u = vals[*a];
            dot[*a] / (1.0 - u * u).sqrt()
        }
        TapeOp::Acosh(a) => {
            let u = vals[*a];
            dot[*a] / (u * u - 1.0).sqrt()
        }
        TapeOp::Asinh(a) => {
            let u = vals[*a];
            dot[*a] / (u * u + 1.0).sqrt()
        }
        TapeOp::Atanh(a) => {
            let u = vals[*a];
            dot[*a] / (1.0 - u * u)
        }
        TapeOp::Atan2(a, b) => {
            let y = vals[*a];
            let x = vals[*b];
            let d = y * y + x * x;
            (x * dot[*a] - y * dot[*b]) / d
        }
        TapeOp::Cmp(_, _, _)
        | TapeOp::And(_, _)
        | TapeOp::Or(_, _)
        | TapeOp::Not(_)
        | TapeOp::Select(_, _, _)
        | TapeOp::Min(_, _)
        | TapeOp::Max(_, _) => panic!(
            "GlobalTape free-function kernels do not implement conditional / logical \
             / min-max TapeOps; use the Tape (build_with_externals) interpreter path \
             instead."
        ),
        TapeOp::Funcall(fc) => {
            let FuncallData { lib, name, args } = fc.as_ref();
            let call_args = funcall_to_ext_args(args, vals);
            let res = lib
                .eval(name, &call_args, true, false)
                .unwrap_or_else(|e| panic!("external function '{name}' tangent eval failed: {e}"));
            let derivs = res.derivs.expect("want_derivs=true returns derivs");
            let mut acc = 0.0;
            let mut k = 0usize;
            for arg in args {
                if let TapeFuncallArg::Tape(idx) = arg {
                    acc += derivs[k] * dot[*idx];
                    k += 1;
                }
            }
            let _ = seed_var;
            acc
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ror_step(
    op: &TapeOp,
    i: usize,
    seed_var: usize,
    vals: &[f64],
    dot: &[f64],
    adj: &mut [f64],
    adj_dot: &mut [f64],
    w: f64,
    wd: f64,
    weight: f64,
    hess_map: &HashMap<(usize, usize), usize>,
    values: &mut [f64],
) {
    match op {
        TapeOp::Const(_) => {}
        TapeOp::Var(k) => {
            if wd != 0.0 && *k >= seed_var {
                if let Some(&pos) = hess_map.get(&(*k, seed_var)) {
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
            adj[*a] += w * vals[*b];
            adj[*b] += w * vals[*a];
            adj_dot[*a] += wd * vals[*b] + w * dot[*b];
            adj_dot[*b] += wd * vals[*a] + w * dot[*a];
        }
        TapeOp::Div(a, b) => {
            let vb = vals[*b];
            let vb2 = vb * vb;
            let vb3 = vb2 * vb;
            adj[*a] += w / vb;
            adj_dot[*a] += wd / vb + w * (-dot[*b] / vb2);
            adj[*b] += w * (-vals[*a] / vb2);
            adj_dot[*b] +=
                wd * (-vals[*a] / vb2) + w * (-dot[*a] / vb2 + 2.0 * vals[*a] * dot[*b] / vb3);
        }
        TapeOp::Pow(a, b) => {
            let u = vals[*a];
            let r = vals[*b];
            let du = dot[*a];
            let dr = dot[*b];
            if r != 0.0 {
                if u != 0.0 {
                    let p_a = r * u.powf(r - 1.0);
                    adj[*a] += w * p_a;
                    let mut dp_a = dr * u.powf(r - 1.0);
                    if u > 0.0 {
                        dp_a += r * u.powf(r - 1.0) * ((r - 1.0) * du / u + dr * u.ln());
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
                let p_b = vals[i] * ln_u;
                adj[*b] += w * p_b;
                let dur = vals[i] * (r * du / u + dr * ln_u);
                let dp_b = dur * ln_u + vals[i] * du / u;
                adj_dot[*b] += wd * p_b + w * dp_b;
            }
        }
        TapeOp::Neg(a) => {
            adj[*a] -= w;
            adj_dot[*a] -= wd;
        }
        TapeOp::Abs(a) => {
            let s = if vals[*a] >= 0.0 { 1.0 } else { -1.0 };
            adj[*a] += w * s;
            adj_dot[*a] += wd * s;
        }
        TapeOp::Sqrt(a) => {
            let sv = vals[i];
            if sv > 0.0 {
                let fp = 0.5 / sv;
                let fpp = -0.25 / (vals[*a] * sv);
                adj[*a] += w * fp;
                adj_dot[*a] += wd * fp + w * fpp * dot[*a];
            }
        }
        TapeOp::Exp(a) => {
            let ev = vals[i];
            adj[*a] += w * ev;
            adj_dot[*a] += wd * ev + w * ev * dot[*a];
        }
        TapeOp::Log(a) => {
            let u = vals[*a];
            adj[*a] += w / u;
            adj_dot[*a] += wd / u + w * (-1.0 / (u * u)) * dot[*a];
        }
        TapeOp::Log10(a) => {
            let u = vals[*a];
            let c = std::f64::consts::LN_10;
            adj[*a] += w / (u * c);
            adj_dot[*a] += wd / (u * c) + w * (-1.0 / (u * u * c)) * dot[*a];
        }
        TapeOp::Sin(a) => {
            let u = vals[*a];
            let cu = u.cos();
            adj[*a] += w * cu;
            adj_dot[*a] += wd * cu + w * (-u.sin()) * dot[*a];
        }
        TapeOp::Cos(a) => {
            let u = vals[*a];
            let su = u.sin();
            adj[*a] -= w * su;
            adj_dot[*a] += wd * (-su) + w * (-u.cos()) * dot[*a];
        }
        TapeOp::Tan(a) => {
            let t = vals[i];
            let gp = 1.0 + t * t;
            let gpp = 2.0 * t * gp;
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Atan(a) => {
            let u = vals[*a];
            let d = 1.0 + u * u;
            let gp = 1.0 / d;
            let gpp = -2.0 * u / (d * d);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Acos(a) => {
            let u = vals[*a];
            let s = 1.0 - u * u;
            let r = s.sqrt();
            let gp = -1.0 / r;
            let gpp = -u / (s * r);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Sinh(a) => {
            let u = vals[*a];
            let gp = u.cosh();
            let gpp = vals[i]; // sinh(u)
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Cosh(a) => {
            let u = vals[*a];
            let gp = u.sinh();
            let gpp = vals[i]; // cosh(u)
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Tanh(a) => {
            let t = vals[i];
            let gp = 1.0 - t * t;
            let gpp = -2.0 * t * gp;
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Asin(a) => {
            let u = vals[*a];
            let s = 1.0 - u * u;
            let r = s.sqrt();
            let gp = 1.0 / r;
            let gpp = u / (s * r);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Acosh(a) => {
            let u = vals[*a];
            let s = u * u - 1.0;
            let r = s.sqrt();
            let gp = 1.0 / r;
            let gpp = -u / (s * r);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Asinh(a) => {
            let u = vals[*a];
            let s = u * u + 1.0;
            let r = s.sqrt();
            let gp = 1.0 / r;
            let gpp = -u / (s * r);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Atanh(a) => {
            let u = vals[*a];
            let d = 1.0 - u * u;
            let gp = 1.0 / d;
            let gpp = 2.0 * u / (d * d);
            adj[*a] += w * gp;
            adj_dot[*a] += wd * gp + w * gpp * dot[*a];
        }
        TapeOp::Atan2(a, b) => {
            let y = vals[*a];
            let x = vals[*b];
            let d = y * y + x * x;
            let d2 = d * d;
            let fa = x / d;
            let fb = -y / d;
            let faa = -2.0 * x * y / d2;
            let fab = (y * y - x * x) / d2;
            let fbb = 2.0 * x * y / d2;
            adj[*a] += w * fa;
            adj[*b] += w * fb;
            adj_dot[*a] += wd * fa + w * (faa * dot[*a] + fab * dot[*b]);
            adj_dot[*b] += wd * fb + w * (fab * dot[*a] + fbb * dot[*b]);
        }
        TapeOp::Cmp(_, _, _)
        | TapeOp::And(_, _)
        | TapeOp::Or(_, _)
        | TapeOp::Not(_)
        | TapeOp::Select(_, _, _)
        | TapeOp::Min(_, _)
        | TapeOp::Max(_, _) => panic!(
            "GlobalTape free-function kernels do not implement conditional / logical \
             / min-max TapeOps; use the Tape (build_with_externals) interpreter path \
             instead."
        ),
        TapeOp::Funcall(fc) => {
            let FuncallData { lib, name, args } = fc.as_ref();
            let call_args = funcall_to_ext_args(args, vals);
            let res = lib.eval(name, &call_args, true, true).unwrap_or_else(|e| {
                panic!("external function '{name}' 2nd-order eval failed: {e}")
            });
            let derivs = res.derivs.expect("want_derivs=true returns derivs");
            let hes = res.hessian.expect("want_hes=true returns hessian");
            let real_tape: Vec<usize> = args
                .iter()
                .filter_map(|a| match a {
                    TapeFuncallArg::Tape(t) => Some(*t),
                    TapeFuncallArg::Str(_) => None,
                })
                .collect();
            for (k, &tk) in real_tape.iter().enumerate() {
                adj[tk] += w * derivs[k];
                let mut second_term = 0.0;
                for (l, &tl) in real_tape.iter().enumerate() {
                    let (lo, hi) = if k <= l { (k, l) } else { (l, k) };
                    let h_kl = hes[lo + hi * (hi + 1) / 2];
                    second_term += h_kl * dot[tl];
                }
                adj_dot[tk] += wd * derivs[k] + w * second_term;
            }
            let _ = seed_var;
            let _ = hess_map;
            let _ = values;
            let _ = weight;
            let _ = i;
        }
    }
}

/// Per-op Hessian-sparsity propagation. Same algorithm as
/// `Tape::hessian_sparsity` but as a free function so `GlobalTape`
/// can call it over its shared `ops` slice.
fn hessian_sparsity_impl(ops: &[TapeOp]) -> BTreeSet<(usize, usize)> {
    let n = ops.len();
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

    for op in ops {
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
            TapeOp::Pow(a, b) | TapeOp::Atan2(a, b) => {
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
            | TapeOp::Cos(a)
            | TapeOp::Tan(a)
            | TapeOp::Atan(a)
            | TapeOp::Acos(a)
            | TapeOp::Sinh(a)
            | TapeOp::Cosh(a)
            | TapeOp::Tanh(a)
            | TapeOp::Asin(a)
            | TapeOp::Acosh(a)
            | TapeOp::Asinh(a)
            | TapeOp::Atanh(a) => {
                emit_self(&var_sets[*a], &mut pairs);
                var_sets[*a].clone()
            }
            TapeOp::Funcall(fc) => {
                let args = &fc.args;
                let mut combined: BTreeSet<usize> = BTreeSet::new();
                for arg in args {
                    if let TapeFuncallArg::Tape(t) = arg {
                        for &vv in &var_sets[*t] {
                            combined.insert(vv);
                        }
                    }
                }
                emit_self(&combined, &mut pairs);
                combined
            }
            TapeOp::Cmp(_, _, _) | TapeOp::And(_, _) | TapeOp::Or(_, _) | TapeOp::Not(_) => {
                // Comparisons / logical ops have identically-zero derivative, so
                // they contribute no Hessian structure.
                BTreeSet::new()
            }
            TapeOp::Select(_, t, e) => {
                // Either branch may be active; the structural superset is the
                // union of both branches' variable sets.
                var_sets[*t].union(&var_sets[*e]).copied().collect()
            }
            TapeOp::Min(a, b) | TapeOp::Max(a, b) => {
                // min/max are piecewise linear: zero second derivative (no
                // pairs); dependence set is the union of both operands.
                var_sets[*a].union(&var_sets[*b]).copied().collect()
            }
        };
        var_sets.push(vset);
    }
    pairs
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
    fn cmp(op: CmpOp, a: Expr, b: Expr) -> Expr {
        Expr::Compare(op, Box::new(a), Box::new(b))
    }
    fn cond(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::Cond {
            cond: Box::new(c),
            then_: Box::new(t),
            else_: Box::new(e),
        }
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
        // body = x0 + x1, shared via Arc. f = body^2 + body.
        let body = Arc::new(add(var(0), var(1)));
        let e = add(
            pow(Expr::Cse(body.clone()), cnst(2.0)),
            Expr::Cse(body.clone()),
        );
        let t = Tape::build(&e);
        // body should appear once in the tape: count Add(Var(0),Var(1)) ops
        let n_body_adds = t
            .ops
            .iter()
            .filter(|op| {
                matches!(op, TapeOp::Add(a, b) if {
                    matches!(t.ops[*a], TapeOp::Var(0)) && matches!(t.ops[*b], TapeOp::Var(1))
                })
            })
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
    fn inverse_trig_grad_and_hessian_match_fd() {
        // f = tan(x0) + atan(x1) + acos(x2) + x0*x1
        // Point chosen so every op is in its smooth domain:
        // tan away from pi/2, acos arg in (-1, 1).
        let e = Expr::Sum(vec![
            unary(UnaryOp::Tan, var(0)),
            unary(UnaryOp::Atan, var(1)),
            unary(UnaryOp::Acos, var(2)),
            mul(var(0), var(1)),
        ]);
        let t = Tape::build(&e);
        let x = [0.5, 1.3, 0.3];

        // Gradient vs central finite difference of the value. This
        // pins the first derivatives independently of the Hessian
        // (fd_check only ties the Hessian to the AD gradient).
        let mut g = vec![0.0; 3];
        t.gradient_seed(&x, 1.0, &mut g);
        for j in 0..3 {
            let h = (1e-7_f64).max(x[j].abs() * 1e-7);
            let mut xp = x;
            let mut xm = x;
            xp[j] += h;
            xm[j] -= h;
            let fd = (t.eval(&xp) - t.eval(&xm)) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (g[j] - fd).abs() / scale < 1e-5,
                "grad[{j}]: AD={:.6e} FD={:.6e}",
                g[j],
                fd
            );
        }

        // Hessian (forward-over-reverse) vs FD of the gradient.
        fd_check(&t, &x, 3, 1e-5);
    }

    /// Shared helper: check AD gradient vs central FD of the value at
    /// `x`, then the Hessian via `fd_check`.
    fn grad_and_hess_match_fd(e: &Expr, x: &[f64], tol: f64) {
        let n = x.len();
        let t = Tape::build(e);
        let mut g = vec![0.0; n];
        t.gradient_seed(x, 1.0, &mut g);
        for j in 0..n {
            let h = (1e-7_f64).max(x[j].abs() * 1e-7);
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let fd = (t.eval(&xp) - t.eval(&xm)) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (g[j] - fd).abs() / scale < tol,
                "grad[{j}]: AD={:.6e} FD={:.6e}",
                g[j],
                fd
            );
        }
        fd_check(&t, x, n, tol);
    }

    #[test]
    fn hyperbolic_grad_and_hessian_match_fd() {
        // f = sinh(x0) + cosh(x1) + tanh(x2) + asinh(x3) + x0*x1 + x2*x3
        // sinh/cosh/tanh/asinh are smooth on all of R.
        let e = Expr::Sum(vec![
            unary(UnaryOp::Sinh, var(0)),
            unary(UnaryOp::Cosh, var(1)),
            unary(UnaryOp::Tanh, var(2)),
            unary(UnaryOp::Asinh, var(3)),
            mul(var(0), var(1)),
            mul(var(2), var(3)),
        ]);
        grad_and_hess_match_fd(&e, &[0.5, 0.7, 0.3, 1.1], 1e-5);
    }

    #[test]
    fn restricted_inverse_grad_and_hessian_match_fd() {
        // f = asin(x0) + acosh(x1) + atanh(x2) + x0*x2
        // Point chosen in each op's smooth domain:
        // asin/atanh need |arg| < 1; acosh needs arg > 1.
        let e = Expr::Sum(vec![
            unary(UnaryOp::Asin, var(0)),
            unary(UnaryOp::Acosh, var(1)),
            unary(UnaryOp::Atanh, var(2)),
            mul(var(0), var(2)),
        ]);
        grad_and_hess_match_fd(&e, &[0.4, 1.8, 0.3], 1e-5);
    }

    #[test]
    fn atan2_grad_and_hessian_match_fd() {
        // f = atan2(x0, x1) + x0*x1, away from the origin.
        let atan2 = |a: Expr, b: Expr| Expr::Binary(BinOp::Atan2, Box::new(a), Box::new(b));
        let e = Expr::Sum(vec![atan2(var(0), var(1)), mul(var(0), var(1))]);
        grad_and_hess_match_fd(&e, &[1.2, 0.7], 1e-5);
    }

    #[test]
    fn minmax_grad_and_hessian_match_fd() {
        // f = min(x0, x1, x2) + max(x1, x2) + x0*x2
        // Point chosen so each list has a UNIQUE strictly-active
        // operand, so the subgradient equals the FD slope (the ±h
        // probes never cross a kink):
        //   min(0.5, 3.0, 2.0) = 0.5  -> active x0
        //   max(3.0, 2.0)      = 3.0  -> active x1
        let e = Expr::Sum(vec![
            Expr::MinList(vec![var(0), var(1), var(2)]),
            Expr::MaxList(vec![var(1), var(2)]),
            mul(var(0), var(2)),
        ]);
        grad_and_hess_match_fd(&e, &[0.5, 3.0, 2.0], 1e-5);
    }

    #[test]
    fn minmax_value_and_active_operand() {
        // Spot-check the value and that the gradient routes entirely
        // through the active operand (zero second derivative).
        let e = Expr::Sum(vec![
            Expr::MinList(vec![var(0), var(1)]),
            Expr::MaxList(vec![var(0), var(1)]),
        ]);
        let t = Tape::build(&e);
        // min(x0,x1) + max(x0,x1) == x0 + x1 for any inputs.
        let x = [1.3, -0.4];
        assert!((t.eval(&x) - (x[0] + x[1])).abs() < 1e-12);
        let mut g = vec![0.0; 2];
        t.gradient_seed(&x, 1.0, &mut g);
        // min active = x1 (smaller), max active = x0 (larger):
        // d/dx0 = 1 (from max), d/dx1 = 1 (from min).
        assert!((g[0] - 1.0).abs() < 1e-12, "g0={}", g[0]);
        assert!((g[1] - 1.0).abs() < 1e-12, "g1={}", g[1]);
    }

    #[test]
    fn hessian_division_matches_fd() {
        // f = x0/x1 + cos(x0)
        let e = add(div(var(0), var(1)), unary(UnaryOp::Cos, var(0)));
        let t = Tape::build(&e);
        fd_check(&t, &[0.5, 1.2], 2, 1e-5);
    }

    #[test]
    fn conditional_value_grad_hessian_active_branch() {
        // f = if x0 >= 1 then x0*x1 else x1^2
        // The if-then-else differentiates only the active branch; the
        // condition (a comparison) contributes no derivative.
        let e = cond(
            cmp(CmpOp::Ge, var(0), cnst(1.0)),
            mul(var(0), var(1)),
            pow(var(1), cnst(2.0)),
        );
        let t = Tape::build(&e);

        // x0 = 2 (>= 1) -> "then" branch x0*x1 is active.
        let x = [2.0, 5.0];
        assert!((t.eval(&x) - 10.0).abs() < 1e-12);
        let mut g = vec![0.0; 2];
        t.gradient_seed(&x, 1.0, &mut g);
        // d(x0*x1) = (x1, x0) = (5, 2)
        assert!((g[0] - 5.0).abs() < 1e-10);
        assert!((g[1] - 2.0).abs() < 1e-10);
        // H[0,1] = 1, diagonals 0. (Stay clear of the x0 = 1 kink.)
        fd_check(&t, &x, 2, 1e-5);

        // x0 = 0 (< 1) -> "else" branch x1^2 is active; x0 drops out.
        let x2 = [0.0, 5.0];
        assert!((t.eval(&x2) - 25.0).abs() < 1e-12);
        let mut g2 = vec![0.0; 2];
        t.gradient_seed(&x2, 1.0, &mut g2);
        assert!(g2[0].abs() < 1e-10);
        assert!((g2[1] - 10.0).abs() < 1e-10);
        fd_check(&t, &x2, 2, 1e-5);
    }

    #[test]
    fn comparison_and_logical_have_zero_derivative() {
        // f = (x0 < x1) + (x0 > 0 && x1 > 0) + !(x0 == x1)
        // Every term is piecewise-constant in the variables, so the
        // gradient must be identically zero away from the kinks.
        let lt = cmp(CmpOp::Lt, var(0), var(1));
        let and = Expr::And(
            Box::new(cmp(CmpOp::Gt, var(0), cnst(0.0))),
            Box::new(cmp(CmpOp::Gt, var(1), cnst(0.0))),
        );
        let notc = Expr::Not(Box::new(cmp(CmpOp::Eq, var(0), var(1))));
        let e = add(add(lt, and), notc);
        let t = Tape::build(&e);

        let x = [1.0, 2.0];
        // 1 (1<2) + 1 (both > 0) + 1 (1 != 2) = 3
        assert!((t.eval(&x) - 3.0).abs() < 1e-12);
        let mut g = vec![0.0; 2];
        t.gradient_seed(&x, 1.0, &mut g);
        assert!(g[0].abs() < 1e-12, "d/dx0 should be 0, got {}", g[0]);
        assert!(g[1].abs() < 1e-12, "d/dx1 should be 0, got {}", g[1]);
    }

    #[test]
    fn logical_or_value() {
        // f = (x0 > 0 || x1 > 0)
        let e = Expr::Or(
            Box::new(cmp(CmpOp::Gt, var(0), cnst(0.0))),
            Box::new(cmp(CmpOp::Gt, var(1), cnst(0.0))),
        );
        let t = Tape::build(&e);
        assert!((t.eval(&[-1.0, 3.0]) - 1.0).abs() < 1e-12);
        assert!((t.eval(&[-1.0, -3.0]) - 0.0).abs() < 1e-12);
    }

    /// `hessian_directional` (one forward-over-reverse pass with
    /// a seed vector) recovers `H · e_j` for each unit-vector seed,
    /// matching column `j` of the dense Hessian computed by
    /// `hessian_accumulate`.
    fn directional_matches_accumulate(tape: &Tape, x: &[f64], n: usize) {
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

        let nops = tape.ops.len();
        let mut vals = vec![0.0; nops];
        tape.forward_into(x, &mut vals);
        let mut dot = vec![0.0; nops];
        let mut adj = vec![0.0; nops];
        let mut adj_dot = vec![0.0; nops];

        for &j in &vars {
            let mut seed = vec![0.0; n];
            seed[j] = 1.0;
            let mut col = vec![0.0; n];
            tape.hessian_directional(
                &vals,
                &seed,
                1.0,
                &mut col,
                &mut dot,
                &mut adj,
                &mut adj_dot,
            );
            for &i in &vars {
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                let expect = ad[hess_map[&(r, c)]];
                assert!(
                    (col[i] - expect).abs() < 1e-10,
                    "directional H[{i},{j}] = {} vs accumulate {}",
                    col[i],
                    expect
                );
            }
        }
    }

    #[test]
    fn directional_quadratic_matches_accumulate() {
        // f = 3 x0^2 + 2 x0 x1 + x1^2
        let e = add(
            add(
                mul(cnst(3.0), pow(var(0), cnst(2.0))),
                mul(mul(cnst(2.0), var(0)), var(1)),
            ),
            pow(var(1), cnst(2.0)),
        );
        let t = Tape::build(&e);
        directional_matches_accumulate(&t, &[0.5, -0.3], 2);
    }

    #[test]
    fn directional_transcendental_matches_accumulate() {
        let e = Expr::Sum(vec![
            unary(UnaryOp::Exp, var(0)),
            unary(UnaryOp::Sin, var(1)),
            unary(UnaryOp::Log, var(0)),
            unary(UnaryOp::Sqrt, var(1)),
            mul(var(0), var(1)),
        ]);
        let t = Tape::build(&e);
        directional_matches_accumulate(&t, &[1.5, 2.0], 2);
    }

    #[test]
    fn directional_with_division_matches_accumulate() {
        let e = add(div(var(0), var(1)), unary(UnaryOp::Cos, var(0)));
        let t = Tape::build(&e);
        directional_matches_accumulate(&t, &[0.5, 1.2], 2);
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
        let two = Arc::new(cnst(2.0));
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
        let body = Arc::new(add(var(0), var(1)));
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

    #[test]
    fn pow_forward_tangent_matches_reverse_gradient_at_base_zero() {
        // Code review L29: `Pow` first-order tangent disagreed with the
        // reverse-mode gradient at base 0. f = x0 ^ x1 keeps a genuine `Pow`
        // op (variable exponent is not lowered to a Mul/Sqrt chain). At the
        // `.nl` default start x0 = 0, the base derivative d/dx0 (x0^1) = 1 is
        // well defined; reverse mode has always computed it, but the forward
        // tangent used to guard on `u != 0` and drop it, so Jacobian-vector
        // products silently disagreed with the gradient at x = 0. After the
        // fix both arms must agree.
        let e = pow(var(0), var(1));
        let t = Tape::build(&e);
        // Guard: the op must survive as a real Pow (not lowered away), else
        // this test would no longer exercise the fixed branch.
        assert!(
            t.ops.iter().any(|op| matches!(op, TapeOp::Pow(_, _))),
            "expected a Pow op in the tape; got {:?}",
            t.ops
        );
        let x = [0.0, 1.0];
        let n = t.ops.len();

        // Reverse-mode gradient w.r.t. x0.
        let mut grad = vec![0.0; 2];
        t.gradient_seed(&x, 1.0, &mut grad);

        // Forward tangent seeded on x0: dot[output] = df/dx0.
        let vals = t.forward(&x);
        let mut dot = vec![0.0; n];
        t.forward_tangent(&vals, 0, &mut dot);
        let fwd_dfx0 = dot[n - 1];

        assert!(
            (grad[0] - 1.0).abs() < 1e-12,
            "reverse gradient df/dx0 at base 0 should be 1, got {}",
            grad[0]
        );
        assert!(
            (fwd_dfx0 - grad[0]).abs() < 1e-12,
            "forward tangent df/dx0 = {fwd_dfx0} must match reverse gradient {} at base 0",
            grad[0]
        );
    }

    #[test]
    #[should_panic(expected = "external function calls are not supported on the")]
    fn hybrid_promoted_cse_with_funcall_reports_clear_message() {
        // Code review L34: `HybridTape::build_multi` builds a promoted CSE
        // (one shared across ≥2 summands) via `build_recursive` with an empty
        // resolver. A funcall inside that promoted body used to panic with the
        // misleading "unresolved AMPL funcall id 0" — implying a resolution
        // failure — instead of the real reason: funcalls are unsupported on
        // the hybrid path. Here the funcall body is shared across two roots so
        // it is promoted; assert the clear hybrid-unsupported message fires.
        let body = Arc::new(Expr::Funcall {
            id: 0,
            args: vec![FuncallArg::Real(var(0))],
        });
        let exprs = vec![
            add(Expr::Cse(body.clone()), cnst(1.0)),
            add(Expr::Cse(body.clone()), cnst(2.0)),
        ];
        HybridTape::build_multi(&exprs);
    }
}
