//! Constraint expression DAGs for FBBT (issue [#62]).
//!
//! The [`ExpressionProvider`] trait gives a presolve pass access to
//! per-constraint expression trees in a form independent of the
//! TNLP source. The shape is a *tape* (a `Vec<FbbtOp>` where operand
//! fields are indices into earlier slots of the same vector), which:
//!
//! * Folds common subexpressions naturally (each unique subexpression
//!   gets a single slot, even if used in many places).
//! * Lets the forward interval pass be a single linear scan.
//! * Lets the reverse interval pass (commit 3 of #62) also be a
//!   single linear scan.
//!
//! ## Operator set
//!
//! Only the operators FBBT's interval arithmetic can soundly handle
//! appear in [`FbbtOp`]. Anything else — extern function calls,
//! integer powers above some safe cap, AMPL's `log10` reduction, the
//! `Sum` n-ary node — is conveyed as [`FbbtOp::Opaque`], and the
//! forward / reverse rules treat that slot as
//! [`crate::expression_provider::FbbtOp::Opaque`] (= `ENTIRE`, no
//! tightening). Providers MUST emit `Opaque` rather than panicking
//! on unsupported sub-expressions; a presolve pass can degrade
//! gracefully across mixed-feature problems that way.
//!
//! ## Implementation status
//!
//! The trait has a default `constraint_expression` implementation
//! that returns `None`, meaning "I don't expose any expressions."
//! That makes the trait safe to require on existing TNLPs (e.g.
//! `PyTnlp`, `CCallbackTnlp`) — they decline and FBBT silently
//! becomes a no-op for those problems.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62

use pounce_common::types::Number;

/// One node in an FBBT-friendly expression tape. Operand fields are
/// indices into the same tape's `ops` vector, and must reference
/// strictly earlier slots (`< self`). The result of the whole tape is
/// the value computed at slot `ops.len() - 1` (the last entry).
#[derive(Debug, Clone, PartialEq)]
pub enum FbbtOp {
    /// Numeric constant.
    Const(Number),
    /// Reference to problem variable `i` (0-based).
    Var(usize),

    // -- Binary --
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    /// `x^n` for non-negative integer `n`. Variable-exponent power is
    /// emitted as [`FbbtOp::Opaque`] (interval `Pow` with a
    /// non-constant exponent is non-trivial and rare in practice;
    /// pounce defers it).
    PowInt(usize, u32),

    // -- Unary --
    Neg(usize),
    Sqrt(usize),
    Exp(usize),
    Ln(usize),
    Abs(usize),
    Sin(usize),
    Cos(usize),

    /// "Don't reason about this slot." The forward pass assigns
    /// `ENTIRE` to it; the reverse pass declines to push information
    /// through it. Providers emit this for operators FBBT doesn't
    /// support (extern function calls, AMPL log10 / sum, non-integer
    /// or variable-exponent powers).
    Opaque,
}

impl FbbtOp {
    /// Whether the op reads any predecessor slots. Used by validators.
    pub fn operand_indices(&self) -> ArrayVec2 {
        match *self {
            FbbtOp::Const(_) | FbbtOp::Var(_) | FbbtOp::Opaque => ArrayVec2::new(),
            FbbtOp::Neg(a)
            | FbbtOp::Sqrt(a)
            | FbbtOp::Exp(a)
            | FbbtOp::Ln(a)
            | FbbtOp::Abs(a)
            | FbbtOp::Sin(a)
            | FbbtOp::Cos(a)
            | FbbtOp::PowInt(a, _) => ArrayVec2::one(a),
            FbbtOp::Add(a, b) | FbbtOp::Sub(a, b) | FbbtOp::Mul(a, b) | FbbtOp::Div(a, b) => {
                ArrayVec2::two(a, b)
            }
        }
    }
}

/// Tiny stack-allocated 0..=2-element vector — every `FbbtOp` has at
/// most two operands. Used so [`FbbtOp::operand_indices`] doesn't
/// allocate.
#[derive(Debug, Clone, Copy)]
pub struct ArrayVec2 {
    data: [usize; 2],
    len: u8,
}

impl ArrayVec2 {
    pub fn new() -> Self {
        Self {
            data: [0, 0],
            len: 0,
        }
    }
    pub fn one(a: usize) -> Self {
        Self {
            data: [a, 0],
            len: 1,
        }
    }
    pub fn two(a: usize, b: usize) -> Self {
        Self {
            data: [a, b],
            len: 2,
        }
    }
    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len as usize]
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for ArrayVec2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Flat expression tape. The root is the **last** slot
/// (`ops[ops.len() - 1]`).
#[derive(Debug, Clone, Default)]
pub struct FbbtTape {
    pub ops: Vec<FbbtOp>,
}

impl FbbtTape {
    /// Empty tape — semantically equivalent to "no information".
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Validate that every operand index is strictly less than its
    /// owner's slot index. Returns the slot of the first offender or
    /// `None` if the tape is well-formed.
    pub fn first_invalid_slot(&self) -> Option<usize> {
        for (i, op) in self.ops.iter().enumerate() {
            for &operand in op.operand_indices().as_slice() {
                if operand >= i {
                    return Some(i);
                }
            }
        }
        None
    }
}

/// Optional capability hooked onto a TNLP: per-constraint expression
/// trees in tape form, used by presolve passes like FBBT.
///
/// Default impls return `None` / empty so existing TNLPs without
/// structural expression access (callback-based bridges, AD-tape-only
/// problems) compile against the trait but contribute nothing —
/// downstream passes degrade silently.
pub trait ExpressionProvider {
    /// Expression tape for constraint `i` (0-based, in the same order
    /// as the TNLP's `eval_g`). Returns `None` to indicate "no
    /// structural information" — a constraint expressed only via the
    /// numerical `eval_g` callback. Bounds for the constraint
    /// (`g_l[i]`, `g_u[i]`) are read separately via the parent TNLP's
    /// `get_bounds_info`.
    fn constraint_expression(&self, _i: usize) -> Option<FbbtTape> {
        None
    }

    /// Expression tape for the objective. Optional in the same sense
    /// as [`Self::constraint_expression`]; FBBT does not use the
    /// objective today, but a future OBBT-style pass might.
    fn objective_expression(&self) -> Option<FbbtTape> {
        None
    }

    /// Human-readable name for variable `i` (0-based, original problem
    /// order), if the model carries one. `None` ⇒ the caller should fall
    /// back to an index label like `x[i]`.
    ///
    /// Names turn index-level diagnostics into model-level ones — e.g.
    /// reporting that a near-singular Jacobian row is the `mass_balance`
    /// equation rather than "row 3". Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>) call out that gap as a key
    /// roadblock for debugging equation-oriented models; this method is
    /// the seam the debugger reads names through.
    fn variable_name(&self, _i: usize) -> Option<&str> {
        None
    }

    /// Human-readable name for constraint `i` (0-based, original problem
    /// order), if the model carries one. `None` ⇒ fall back to an index
    /// label like `c[i]`. See [`Self::variable_name`].
    fn constraint_name(&self, _i: usize) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_tape(c: Number) -> FbbtTape {
        FbbtTape {
            ops: vec![FbbtOp::Const(c)],
        }
    }

    #[test]
    fn operand_indices_match_op_arity() {
        assert!(FbbtOp::Const(1.0).operand_indices().is_empty());
        assert!(FbbtOp::Var(0).operand_indices().is_empty());
        assert!(FbbtOp::Opaque.operand_indices().is_empty());
        assert_eq!(FbbtOp::Neg(3).operand_indices().as_slice(), &[3]);
        assert_eq!(FbbtOp::PowInt(2, 4).operand_indices().as_slice(), &[2]);
        assert_eq!(FbbtOp::Add(1, 2).operand_indices().as_slice(), &[1, 2]);
    }

    #[test]
    fn validate_well_formed_tape() {
        // (x0 + x1) * 2
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),
                FbbtOp::Var(1),
                FbbtOp::Add(0, 1),
                FbbtOp::Const(2.0),
                FbbtOp::Mul(2, 3),
            ],
        };
        assert_eq!(tape.first_invalid_slot(), None);
    }

    #[test]
    fn validate_catches_forward_reference() {
        // Slot 0 references slot 1 → invalid.
        let tape = FbbtTape {
            ops: vec![FbbtOp::Neg(1), FbbtOp::Const(0.0)],
        };
        assert_eq!(tape.first_invalid_slot(), Some(0));
    }

    #[test]
    fn validate_catches_self_reference() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Neg(0)],
        };
        assert_eq!(tape.first_invalid_slot(), Some(0));
    }

    #[test]
    fn default_trait_returns_none() {
        struct NoExpr;
        impl ExpressionProvider for NoExpr {}
        let p = NoExpr;
        assert!(p.constraint_expression(0).is_none());
        assert!(p.objective_expression().is_none());
    }

    /// A trivial provider that returns the same constant for every
    /// constraint — checks the trait can be implemented and used.
    #[test]
    fn custom_provider_returns_tape() {
        struct Always(Number);
        impl ExpressionProvider for Always {
            fn constraint_expression(&self, _i: usize) -> Option<FbbtTape> {
                Some(const_tape(self.0))
            }
        }
        let p = Always(3.5);
        let t = p.constraint_expression(7).unwrap();
        assert_eq!(t.ops, vec![FbbtOp::Const(3.5)]);
    }
}
