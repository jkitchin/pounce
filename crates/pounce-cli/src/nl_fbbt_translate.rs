//! Translate parsed `.nl` constraint expressions into an
//! [`FbbtTape`] for the presolve FBBT pass (issue [#62]).
//!
//! The `Expr` tree pounce reads from a `.nl` file uses a richer
//! operator set than FBBT supports (extern function calls,
//! variable-exponent powers, AMPL `log10`, n-ary sums) and embeds
//! common subexpressions via `Rc` sharing. This module flattens
//! the tree into a tape where:
//!
//! * Every `Expr::Cse(rc)` is emitted **once** and re-referenced by
//!   slot index on every subsequent occurrence — matching the
//!   per-Rc-pointer caching strategy `nl_tape::Tape::build` already
//!   uses for AD tapes.
//! * Operators FBBT can reason about become the corresponding
//!   [`FbbtOp`] variants.
//! * Anything else collapses to [`FbbtOp::Opaque`], which forward /
//!   reverse interval passes treat as "no information here." A single
//!   unsupported sub-expression doesn't poison the whole constraint —
//!   intervals just stop tightening through that subtree.
//!
//! The full constraint expression on row `i` is `con_nonlinear[i] +
//! Σ_k coef_k · x_{var_k}`. The linear part is folded in after the
//! nonlinear translation, so the resulting tape has *one* root
//! representing the entire constraint.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62

use std::collections::HashMap;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};

use crate::nl_reader::{BinOp, Expr, UnaryOp};

/// Result of translating one `Expr` into a tape.
struct Builder {
    ops: Vec<FbbtOp>,
    /// CSE cache: `Rc::as_ptr` → tape slot of the body.
    cse_cache: HashMap<*const Expr, usize>,
}

impl Builder {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            cse_cache: HashMap::new(),
        }
    }

    fn emit(&mut self, op: FbbtOp) -> usize {
        let idx = self.ops.len();
        self.ops.push(op);
        idx
    }

    /// Recursively translate `expr` and return its slot index in
    /// `self.ops`.
    fn translate(&mut self, expr: &Expr) -> usize {
        match expr {
            Expr::Const(v) => self.emit(FbbtOp::Const(*v)),
            Expr::Var(i) => self.emit(FbbtOp::Var(*i)),
            Expr::Cse(rc) => {
                let key = Rc::as_ptr(rc);
                if let Some(&slot) = self.cse_cache.get(&key) {
                    return slot;
                }
                let slot = self.translate(rc.as_ref());
                self.cse_cache.insert(key, slot);
                slot
            }
            Expr::Binary(op, lhs, rhs) => {
                let a = self.translate(lhs);
                let b = self.translate(rhs);
                match op {
                    BinOp::Add => self.emit(FbbtOp::Add(a, b)),
                    BinOp::Sub => self.emit(FbbtOp::Sub(a, b)),
                    BinOp::Mul => self.emit(FbbtOp::Mul(a, b)),
                    BinOp::Div => self.emit(FbbtOp::Div(a, b)),
                    BinOp::Pow => {
                        // FBBT only handles integer exponent pinned at
                        // compile time. If the right-hand side is an
                        // `Expr::Const(c)` with `c` a small non-negative
                        // integer, emit `PowInt`; otherwise bail.
                        let exp_const = const_value(rhs).and_then(integer_exponent);
                        if let Some(n) = exp_const {
                            self.emit(FbbtOp::PowInt(a, n))
                        } else {
                            self.emit(FbbtOp::Opaque)
                        }
                    }
                    // atan2 has no interval-arithmetic support in FbbtOp —
                    // treat as opaque (no bound propagation through it).
                    BinOp::Atan2 => {
                        let _ = (a, b);
                        self.emit(FbbtOp::Opaque)
                    }
                }
            }
            Expr::Unary(op, x) => {
                let a = self.translate(x);
                match op {
                    UnaryOp::Neg => self.emit(FbbtOp::Neg(a)),
                    UnaryOp::Sqrt => self.emit(FbbtOp::Sqrt(a)),
                    UnaryOp::Log => self.emit(FbbtOp::Ln(a)),
                    UnaryOp::Exp => self.emit(FbbtOp::Exp(a)),
                    UnaryOp::Abs => self.emit(FbbtOp::Abs(a)),
                    UnaryOp::Sin => self.emit(FbbtOp::Sin(a)),
                    UnaryOp::Cos => self.emit(FbbtOp::Cos(a)),
                    // log10 = ln / ln(10) — translate as (Ln(x) /
                    // Const(ln 10)) so we don't drop info.
                    UnaryOp::Log10 => {
                        let ln = self.emit(FbbtOp::Ln(a));
                        let denom = self.emit(FbbtOp::Const(std::f64::consts::LN_10));
                        self.emit(FbbtOp::Div(ln, denom))
                    }
                    // tan / atan / acos have no interval-arithmetic
                    // support in FbbtOp yet — treat them as opaque so
                    // bound tightening simply doesn't propagate through
                    // them (correct, just not tightened) rather than
                    // emitting a wrong interval. `a` is left as a dead
                    // sub-tree in the tape, which the evaluator ignores.
                    UnaryOp::Tan
                    | UnaryOp::Atan
                    | UnaryOp::Acos
                    | UnaryOp::Sinh
                    | UnaryOp::Cosh
                    | UnaryOp::Tanh
                    | UnaryOp::Asin
                    | UnaryOp::Acosh
                    | UnaryOp::Asinh
                    | UnaryOp::Atanh => {
                        let _ = a;
                        self.emit(FbbtOp::Opaque)
                    }
                }
            }
            Expr::Sum(parts) => {
                // Left-fold the n-ary sum into binary Adds.
                if parts.is_empty() {
                    return self.emit(FbbtOp::Const(0.0));
                }
                let mut acc = self.translate(&parts[0]);
                for p in &parts[1..] {
                    let next = self.translate(p);
                    acc = self.emit(FbbtOp::Add(acc, next));
                }
                acc
            }
            // Comparisons, logical connectives, and if-then-else have
            // no interval-arithmetic support in FbbtOp. Translating
            // their operands and emitting `Opaque` keeps bound
            // tightening sound (it simply doesn't propagate through
            // them) without asserting a wrong interval. The operand
            // sub-trees are left dead in the tape; the evaluator
            // ignores them.
            Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
                let _ = (self.translate(a), self.translate(b));
                self.emit(FbbtOp::Opaque)
            }
            Expr::Not(a) => {
                let _ = self.translate(a);
                self.emit(FbbtOp::Opaque)
            }
            Expr::Cond { cond, then_, else_ } => {
                let _ = (
                    self.translate(cond),
                    self.translate(then_),
                    self.translate(else_),
                );
                self.emit(FbbtOp::Opaque)
            }
            // n-ary min/max have no FbbtOp interval form yet. Translate
            // the operands (so they remain well-formed sub-trees) and
            // emit Opaque, keeping bound tightening sound without
            // asserting a wrong interval.
            Expr::MinList(args) | Expr::MaxList(args) => {
                for a in args {
                    let _ = self.translate(a);
                }
                self.emit(FbbtOp::Opaque)
            }
            Expr::Funcall { .. } => {
                // External / imported functions are opaque to FBBT.
                self.emit(FbbtOp::Opaque)
            }
        }
    }
}

/// Borrow the constant payload of an `Expr::Const`, or follow one
/// layer of `Cse` to find a constant. Returns `None` for any other
/// shape — including expressions that are *value-equivalent* to a
/// constant but not syntactically one.
fn const_value(expr: &Expr) -> Option<Number> {
    match expr {
        Expr::Const(v) => Some(*v),
        Expr::Cse(rc) => const_value(rc.as_ref()),
        _ => None,
    }
}

/// Coerce a `Number` to a non-negative integer suitable for
/// [`FbbtOp::PowInt`]. Caps at 64 — beyond that, interval arithmetic
/// quickly hits the floating-point overflow band and produces
/// uninformative bounds anyway.
fn integer_exponent(v: Number) -> Option<u32> {
    if !v.is_finite() {
        return None;
    }
    if v < 0.0 || v > 64.0 {
        return None;
    }
    let rounded = v.round();
    if (v - rounded).abs() > 1e-9 {
        return None;
    }
    Some(rounded as u32)
}

/// Translate the nonlinear part of constraint `i` together with its
/// linear coefficients into a single tape. Returns `None` if neither
/// part contributes anything (no nonlinear expression *and* no
/// linear coefficients) — there's nothing for FBBT to tighten
/// against.
///
/// `nonlinear` is the `Expr` from `NlProblem::con_nonlinear[i]`;
/// `linear` is the corresponding `con_linear[i]` slice. Variable
/// indices in `linear` are 0-based and refer to the same `Var(j)`
/// slots in `nonlinear`.
pub fn translate_constraint(nonlinear: &Expr, linear: &[(usize, Number)]) -> Option<FbbtTape> {
    let nonlinear_trivial = matches!(nonlinear, Expr::Const(c) if *c == 0.0);
    if nonlinear_trivial && linear.is_empty() {
        return None;
    }

    let mut b = Builder::new();
    let mut root = if nonlinear_trivial {
        // Skip emitting a zero placeholder if we have linear terms;
        // the linear fold will start from the first linear term.
        None
    } else {
        Some(b.translate(nonlinear))
    };

    for &(var_idx, coef) in linear {
        let v_slot = b.emit(FbbtOp::Var(var_idx));
        let c_slot = b.emit(FbbtOp::Const(coef));
        let term = b.emit(FbbtOp::Mul(v_slot, c_slot));
        root = Some(match root {
            None => term,
            Some(prev) => b.emit(FbbtOp::Add(prev, term)),
        });
    }

    // The builder's last emit is always the root after the linear
    // fold; if both contributions were trivial we returned above.
    debug_assert!(root.is_some());
    Some(FbbtTape { ops: b.ops })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_linear_translates_to_sum_of_terms() {
        // 3 * x0 + (-2) * x1
        let nonlinear = Expr::Const(0.0);
        let linear = vec![(0usize, 3.0), (1usize, -2.0)];
        let tape = translate_constraint(&nonlinear, &linear).unwrap();
        // ops: Var(0), Const(3), Mul(0,1), Var(1), Const(-2), Mul(3,4), Add(2,5).
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Add(2, 5))));
        // Forward pass for x0 ∈ [0,1], x1 ∈ [0,1]: result ∈ [-2, 3].
    }

    #[test]
    fn purely_zero_constraint_returns_none() {
        let nonlinear = Expr::Const(0.0);
        assert!(translate_constraint(&nonlinear, &[]).is_none());
    }

    #[test]
    fn unary_translations_cover_all_supported_ops() {
        let inner = Box::new(Expr::Var(0));
        let cases = [
            (UnaryOp::Neg, FbbtOp::Neg(0)),
            (UnaryOp::Sqrt, FbbtOp::Sqrt(0)),
            (UnaryOp::Log, FbbtOp::Ln(0)),
            (UnaryOp::Exp, FbbtOp::Exp(0)),
            (UnaryOp::Abs, FbbtOp::Abs(0)),
            (UnaryOp::Sin, FbbtOp::Sin(0)),
            (UnaryOp::Cos, FbbtOp::Cos(0)),
        ];
        for (op, expected) in cases {
            let e = Expr::Unary(op, inner.clone());
            let tape = translate_constraint(&e, &[]).unwrap();
            assert_eq!(tape.ops.last().unwrap(), &expected);
        }
    }

    #[test]
    fn inverse_trig_translate_to_opaque() {
        // tan/atan/acos have no interval-arithmetic FbbtOp yet, so the
        // translator emits Opaque (FBBT won't tighten through them).
        for op in [UnaryOp::Tan, UnaryOp::Atan, UnaryOp::Acos] {
            let e = Expr::Unary(op, Box::new(Expr::Var(0)));
            let tape = translate_constraint(&e, &[]).unwrap();
            assert_eq!(tape.ops.last().unwrap(), &FbbtOp::Opaque, "{op:?}");
        }
    }

    #[test]
    fn log10_decomposes_into_ln_div() {
        let e = Expr::Unary(UnaryOp::Log10, Box::new(Expr::Var(0)));
        let tape = translate_constraint(&e, &[]).unwrap();
        // ops: Var(0), Ln(0), Const(ln 10), Div(1, 2).
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Div(1, 2))));
    }

    #[test]
    fn pow_with_const_int_rhs_uses_powint() {
        // x^3
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        let tape = translate_constraint(&e, &[]).unwrap();
        // ops: Var(0), Const(3), PowInt(0, 3).
        assert!(matches!(tape.ops.last(), Some(FbbtOp::PowInt(0, 3))));
    }

    #[test]
    fn pow_with_variable_rhs_is_opaque() {
        // x^y
        let e = Expr::Binary(BinOp::Pow, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let tape = translate_constraint(&e, &[]).unwrap();
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Opaque)));
    }

    #[test]
    fn pow_with_fractional_const_is_opaque() {
        // x^1.5
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.5)),
        );
        let tape = translate_constraint(&e, &[]).unwrap();
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Opaque)));
    }

    #[test]
    fn cse_shared_body_emitted_once() {
        // body = x + 1; (body * 2) + body
        let body = Rc::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        ));
        let two_body = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Cse(Rc::clone(&body))),
            Box::new(Expr::Const(2.0)),
        );
        let total = Expr::Binary(BinOp::Add, Box::new(two_body), Box::new(Expr::Cse(body)));
        let tape = translate_constraint(&total, &[]).unwrap();
        // The body should appear only once: count Var(0)s.
        let n_var0 = tape
            .ops
            .iter()
            .filter(|op| matches!(op, FbbtOp::Var(0)))
            .count();
        assert_eq!(n_var0, 1, "CSE body must be emitted once: {:?}", tape.ops);
    }

    #[test]
    fn sum_node_folds_to_binary_adds() {
        let s = Expr::Sum(vec![Expr::Var(0), Expr::Var(1), Expr::Var(2)]);
        let tape = translate_constraint(&s, &[]).unwrap();
        // Var(0), Var(1), Add(0,1), Var(2), Add(2,3).
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Add(2, 3))));
    }

    #[test]
    fn empty_sum_folds_to_zero_constant() {
        let s = Expr::Sum(vec![]);
        let tape = translate_constraint(&s, &[]).unwrap();
        // Const(0) — and since linear is empty too, the whole tape
        // is just that one slot.
        assert_eq!(tape.ops.len(), 1);
        assert!(matches!(tape.ops[0], FbbtOp::Const(c) if c == 0.0));
    }

    #[test]
    fn funcall_collapses_to_opaque() {
        let e = Expr::Funcall {
            id: 0,
            args: vec![],
        };
        let tape = translate_constraint(&e, &[]).unwrap();
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Opaque)));
    }

    #[test]
    fn nonlinear_plus_linear_combines() {
        // x0^2 + 3*x1 + 5*x2 (where x0^2 is nonlinear and 3*x1, 5*x2 are linear)
        let nonlinear = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let linear = vec![(1usize, 3.0), (2usize, 5.0)];
        let tape = translate_constraint(&nonlinear, &linear).unwrap();
        // Last op must be Add (folding linear in).
        assert!(matches!(tape.ops.last(), Some(FbbtOp::Add(_, _))));
        assert!(tape.first_invalid_slot().is_none());
    }

    #[test]
    fn translated_tape_is_well_formed() {
        // A messy expression mixing CSEs, unary, binary, sums.
        let body = Rc::new(Expr::Unary(UnaryOp::Exp, Box::new(Expr::Var(0))));
        let e = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Cse(Rc::clone(&body))),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Cse(body)),
                Box::new(Expr::Const(3.0)),
            )),
        );
        let tape = translate_constraint(&e, &[(1, 0.5)]).unwrap();
        assert!(tape.first_invalid_slot().is_none());
    }
}
