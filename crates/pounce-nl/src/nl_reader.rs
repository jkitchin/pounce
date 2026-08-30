//! Minimal AMPL `.nl` ASCII-format reader.
//!
//! Implements the `g`-header text dialect for problems whose constraint
//! and objective expressions are restricted to a polynomial-friendly
//! subset of opcodes. This is **not** a full `.nl` reader — it is the
//! smallest piece that lets `pounce --nl-file foo.nl` solve a real
//! AMPL-emitted unconstrained problem.
//!
//! Supported:
//! * Text header (`g…`).
//! * Constraint and objective expression segments using opcodes
//!   `o0` (add), `o1` (sub), `o2` (mul), `o3` (div), `o5` (pow),
//!   `o16` (unary minus), `o39` (sqrt), `o42` (log10), `o43` (log),
//!   `o44` (exp), `o15` (abs), `o41` (sin), `o46` (cos), `o38` (tan),
//!   `o49` (atan), `o53` (acos), plus
//!   `n<num>` constants and `v<idx>` variables.
//! * Linear-Jacobian (`J`) and linear-objective (`G`) segments.
//! * Variable bounds (`b`) and constraint bounds (`r`).
//! * Optional initial primal (`x`) segment and initial dual (`d`)
//!   segment. Both are parsed (into `x0` / `lambda0`) and returned by
//!   `get_starting_point`; the duals feed a `warm_start_init_point` solve.
//! * Multiple objectives (we use only the first; per AMPL convention).
//!
//! Not supported (will return an error explaining what's missing):
//! * Network / piecewise-linear constructs.
//! * Complementarity rows.
//! * Binary-format `.nl` files (`b…` header).
//!
//! References:
//! * <https://ampl.com/REFS/hooking2.pdf> — "Hooking Your Solver to
//!   AMPL" (David M. Gay), the canonical `.nl` spec.
//! * `ref/Ipopt/test/mytoy.nl` — annotated example used for the unit
//!   tests in this module.

use crate::nl_quadratic::{
    FactoredQuadratic, Quad2, QuadForm, QuadHessian, is_expanded_quadratic, is_trivially_zero,
    quad_form_readout, recognize_expr, recognize_factored_quadratic,
};
use crate::nl_tape::{HybridTape, Tape, hybrid_supported};
use pounce_common::types::{Index, Number, lower_bound_present, upper_bound_present};
use pounce_nlp::constant_derivatives::{DerivativeProof, DerivativeProofs};
use pounce_nlp::quadratic::{QuadraticStructure, SquareTerm};
use pounce_nlp::tnlp::{
    BoundsInfo, IDX_NAMES, IndexStyle, IpoptCq, IpoptData, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Expr {
    /// Numeric constant.
    Const(Number),
    /// Variable reference (0-based index into `x`).
    Var(usize),
    /// Binary op: `args = [lhs, rhs]`.
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Unary op.
    Unary(UnaryOp, Box<Expr>),
    /// n-ary sum (opcode `o54` — variadic; we may emit it from `o0`
    /// folding optimization, but the parser treats `o0` as binary).
    Sum(Vec<Expr>),
    /// Reference to a common subexpression (`.nl` `V` segment). The
    /// payload is a shared body; many references to the same CSE share
    /// one `Arc`, so the parsed problem is a DAG. Walking through `Cse`
    /// is mathematically equivalent to inlining the body at each
    /// occurrence (every reference is an independent occurrence in the
    /// chain rule), so eval/grad/collect_vars just recurse into the
    /// inner `Expr`. The pointer is atomically refcounted (`Arc`, not
    /// `Rc`) so a parsed problem — and the `NlTnlp` built from it —
    /// is `Send` and can move to a rayon worker for batched solving
    /// (pounce#126); sharing is still read-only after parse.
    Cse(Arc<Expr>),
    /// AMPL imported (external) function call. `id` matches an entry in
    /// `NlProblem.imported_funcs`; resolution to a live shared library
    /// happens when the tape is built (see `nl_external::ExternalResolver`).
    Funcall { id: usize, args: Vec<FuncallArg> },
    /// Relational comparison (`o22`/`o23`/`o24`/`o28`/`o29`/`o30`).
    /// Evaluates to `1.0` when the comparison holds, else `0.0`. The
    /// result is piecewise-constant, so it has zero derivative
    /// everywhere (the kink at equality is ignored — standard
    /// subgradient-free treatment, matching ASL).
    Compare(CmpOp, Box<Expr>, Box<Expr>),
    /// Logical AND (`o21`). `1.0` iff both operands are nonzero.
    /// Zero derivative (piecewise constant).
    And(Box<Expr>, Box<Expr>),
    /// Logical OR (`o20`). `1.0` iff either operand is nonzero.
    /// Zero derivative (piecewise constant).
    Or(Box<Expr>, Box<Expr>),
    /// Logical NOT (`o34`). `1.0` iff the operand is zero.
    /// Zero derivative (piecewise constant).
    Not(Box<Expr>),
    /// `if-then-else` (`o35` OPIFnl). Evaluates `cond`; when it is
    /// nonzero the value and all derivatives flow through `then_`,
    /// otherwise through `else_`. The branch switch is a non-smooth
    /// event the derivative ignores (it differentiates only the
    /// active branch), exactly as ASL/IPOPT does for `if`.
    Cond {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },
    /// n-ary minimum (`o11` MINLIST). Value is the smallest operand.
    /// Piecewise linear: the derivative flows through whichever operand
    /// is currently smallest (a subgradient; ties resolve to the first
    /// such operand), and the second derivative is identically zero —
    /// the standard AD treatment for min/max, matching ASL/IPOPT.
    MinList(Vec<Expr>),
    /// n-ary maximum (`o12` MAXLIST). Value is the largest operand;
    /// derivative routing mirrors [`Expr::MinList`].
    MaxList(Vec<Expr>),
}

/// Relational operator carried by [`Expr::Compare`]. The variants map
/// 1:1 onto AMPL opcodes `o22 LT`, `o23 LE`, `o24 EQ`, `o28 GE`,
/// `o29 GT`, `o30 NE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
    Ne,
}

/// One positional argument to an AMPL imported function call. AMPL splits
/// arguments into reals (carried by `ra[]`) and strings (carried by `sa[]`);
/// `FuncallArg` mirrors that split. Real args are arbitrary expressions.
#[derive(Debug, Clone)]
pub enum FuncallArg {
    Real(Expr),
    Str(String),
}

/// An AMPL imported (external) function declaration from a top-level
/// `F<id> <type> <nargs> <name>` segment.
#[derive(Debug, Clone)]
pub struct ImportedFunc {
    pub id: usize,
    /// 0 = real-valued, 1 = string-args (per AMPL's funcadd ABI).
    pub kind: usize,
    /// Declared arg count. >=0 exact arity; <=-1 means at least `-(nargs+1)`.
    pub nargs: i64,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// Two-argument arctangent `atan2(a, b)` with operands `(y, x)`.
    Atan2,
    /// `a·ln(a/b)` — GAMS `centropy`. No `.nl` opcode; in-memory `Expr` only.
    ///
    /// Fused for the same reason as [`UnaryOp::XLogX`], plus one of its own:
    /// `∂²/∂b²` is `a/b²`, and `b²` overflows for `|b| > 1.3e154` while
    /// `a/b²` itself stays comfortably in range. The fused rule evaluates it
    /// as `q/b` with `q = a/b` and never squares anything.
    CEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Log,
    Exp,
    Abs,
    Sin,
    Cos,
    Log10,
    Tan,
    Atan,
    Acos,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acosh,
    Asinh,
    Atanh,
    /// Gauss error function. No `.nl` opcode maps here — AMPL has no `erf`,
    /// so the parser never emits it — but the in-memory builder (issue #469)
    /// does, which is the whole point: a frontend that constructs an `Expr`
    /// directly is not limited to what `.nl` can spell.
    Erf,
    /// `a·ln(a)` — GAMS `entropy`. Like [`UnaryOp::Erf`], no `.nl` opcode maps
    /// here; it is reachable only from an in-memory `Expr`.
    ///
    /// Fused rather than lowered to `Mul(a, Log(a))` because the chain rule
    /// *cannot* produce its second derivative. `(a·ln a)'' = 1/a` is finite
    /// wherever `a > 0` — at `a = 1e-299` it is `1e299` — but every
    /// decomposition routes through `ln''(a) = -1/a² = -1e598`, which exceeds
    /// `f64::MAX`. A composite that is in range, built from a factor that is
    /// not, is unreachable by any chain rule however carefully written, so the
    /// fusion is a correctness requirement rather than an optimization.
    XLogX,
}

/// The `.nl` header's nonlinearity census — lines 3 and 5 of Gay's header
/// table (*Hooking Your Solver to AMPL*, §D and Table 1).
///
/// AMPL has already done this analysis when it writes the file, so these are
/// facts about the model that cost nothing to keep and would otherwise have
/// to be recovered by walking every expression tree in it.
///
/// ```text
///  55 1        # nonlinear constraints, objectives          -> nl_cons, nl_objs
///  100 110 100 # nonlinear vars in constraints, objectives, both
/// ```
///
/// Two properties of the format make these usable rather than merely
/// informative, and both are asserted against the fixture corpus in
/// `crates/pounce-cli/tests/nl_header_counts.rs`:
///
/// 1. **Nonlinear rows come first.** Constraints `0..nl_cons` are the ones
///    with a nonlinear body; objectives `0..nl_objs` likewise.
/// 2. **Nonlinear variables come first.** The `.nl` variable order is
///    "nonlinear in both, then constraints-only, then objectives-only, then
///    everything linear", so the variables that appear nonlinearly occupy a
///    prefix of length [`NlCounts::nonlinear_vars`].
///
/// The counts are what the *writer* asserted, which is not always what the
/// parsed trees say: `parse_nl_text` folds a variable-free `C` body into the
/// row bounds (`gh #492`), so a row counted in `nl_cons` can arrive here with
/// a nonlinear part of `Const(0.0)`. The discrepancy is one-directional —
/// the header only ever over-states nonlinearity relative to the trees — and
/// every consumer below is written to be sound under exactly that direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlCounts {
    /// `nlc`: constraints with a nonlinear body.
    pub nl_cons: usize,
    /// `nlo`: objectives with a nonlinear body.
    pub nl_objs: usize,
    /// `nlvc`: variables appearing nonlinearly in constraints. **Includes**
    /// the [`Self::nl_vars_both`] variables that are also nonlinear in an
    /// objective.
    pub nl_vars_cons: usize,
    /// `nlvo`: variables appearing nonlinearly in objectives. Also includes
    /// [`Self::nl_vars_both`].
    pub nl_vars_objs: usize,
    /// `nlvb`: variables appearing nonlinearly in *both* a constraint and an
    /// objective. Counted a second and third time in the two fields above,
    /// which is why the total is an inclusion–exclusion and not a sum.
    pub nl_vars_both: usize,
}

impl NlCounts {
    /// Number of distinct variables that appear nonlinearly anywhere:
    /// `nlvc + nlvo − nlvb`, because `nlvb` is double-counted by the other
    /// two. Saturating, so a malformed header cannot underflow.
    ///
    /// This is *not* `max(nlvc, nlvo)`: for `min x₀² s.t. x₁² ≤ 1` the counts
    /// are `nlvc = nlvo = 1`, `nlvb = 0` and there are two nonlinear
    /// variables, not one.
    pub fn nonlinear_vars(&self) -> usize {
        self.nl_vars_cons
            .saturating_add(self.nl_vars_objs)
            .saturating_sub(self.nl_vars_both)
    }
}

/// The nonlinear body of the objective, or of one constraint row, as the
/// parser left it.
///
/// Before gh #588 Q5 this was always an [`Expr`], and for most bodies it
/// still is. The second variant exists because the `Expr` DAG is what sets
/// peak RSS on a quadratic model — 2.32 M nodes for a ten-row
/// `qcqp500-3c` — and every consumer of a *recognized* body wants the
/// degree-≤2 coefficients rather than the tree they would have to walk to
/// recover them. So the parser recognizes those bodies from the token
/// stream and never builds the tree at all.
///
/// The distinction is deliberately impossible to ignore. Nine consumers
/// read these bodies (§5.3 of `dev-notes/quadratic-structure-exploitation.md`
/// enumerates them) and several would read a *missing* tree as "this row is
/// linear" — a silent wrong answer. An enum makes every one of them a
/// compile error until it says which reading it wants.
#[derive(Debug, Clone)]
pub enum NlBody {
    /// The expression tree.
    Tree(Expr),
    /// A degree-2 form recognized while the token stream was consumed. The
    /// tree was never built; [`NlProblem::con_expr`] rebuilds it on demand,
    /// byte for byte, by re-parsing the same bytes with the same parser.
    Quad(Box<QuadBody>),
}

/// A body the parser recognized as an already-expanded quadratic.
#[derive(Debug, Clone)]
pub struct QuadBody {
    /// The recognized form. Bit-for-bit what
    /// [`crate::nl_quadratic::recognize_expr`] returns for the tree these
    /// same bytes parse to — asserted directly, over the whole corpus, by
    /// `pounce-cli/tests/quad_parse_differential.rs`.
    pub form: Quad2,
    /// Every variable the body's token stream mentions, ascending and
    /// deduplicated — exactly the set [`collect_vars`] reports for the tree,
    /// **including** variables whose coefficient cancelled to zero (which
    /// `form` necessarily drops). Structural consumers want this one; the
    /// linearity contract is over-stating-is-safe, and a support taken from
    /// `form` would under-state.
    pub vars: Vec<u32>,
    /// Byte range of this body's token stream inside [`NlProblem::src`].
    pub src: std::ops::Range<usize>,
    /// Nesting depth of the tree these tokens would have built, on the same
    /// convention as a leaf counting 1.
    ///
    /// Recorded because the streaming recognizer is iterative and the tree
    /// parser is not: a body deep enough to overflow the parser's stack now
    /// *loads*, and the depth guard that used to be enforced implicitly by
    /// the parse failing has to be enforced by something. `pounce-py`'s
    /// `checked_depth` reads it (pounce #472). Rebuilding such a body with
    /// [`NlProblem::con_expr`] would still recurse, which is the same
    /// ceiling Q3 recorded and Q5 does not lift.
    pub depth: u32,
}

impl NlBody {
    /// The identity zero — "this row has no nonlinear part". A recognized
    /// body is degree 2 by construction, so it is never trivially zero;
    /// the question still has to be asked through here rather than by
    /// matching on a tree that may not exist.
    pub fn is_trivially_zero(&self) -> bool {
        match self {
            NlBody::Tree(e) => matches!(e, Expr::Const(c) if *c == 0.0),
            NlBody::Quad(_) => false,
        }
    }

    /// The recognized degree-2 form, when the parser produced one.
    pub fn quad(&self) -> Option<&Quad2> {
        match self {
            NlBody::Tree(_) => None,
            NlBody::Quad(q) => Some(&q.form),
        }
    }

    /// The tree, when there is one resident. `None` for a recognized body —
    /// use [`NlProblem::con_expr`] / [`NlProblem::obj_expr`] to rebuild it.
    pub fn tree(&self) -> Option<&Expr> {
        match self {
            NlBody::Tree(e) => Some(e),
            NlBody::Quad(_) => None,
        }
    }

    /// This body as a degree-≤2 Hessian, or `None` if it is not provably
    /// quadratic — [`crate::nl_quadratic::analyze_quadratic`] for a tree,
    /// and the form the parser already proved for a recognized body. The
    /// two answers are the same by construction and asserted to be so bit
    /// for bit; this is the accessor that keeps the corpus from being
    /// re-recognized once per consumer.
    ///
    /// `None` also when a term was dropped getting to the form — see
    /// [`Self::analyze_quadratic_full`].
    pub fn analyze_quadratic(&self) -> Option<QuadHessian> {
        self.analyze_quadratic_full().map(|(h, _, _)| h)
    }

    /// [`Self::analyze_quadratic`] with the linear and constant parts.
    ///
    /// ## A form that dropped a term is not this body
    ///
    /// Everything downstream of here *reads coefficients out*: the
    /// classifier decides a problem class from the Hessian, and
    /// `qp_extract` builds `P`, `c`, `A` and `G` from all three parts. So
    /// this accessor owes its callers a form that is the whole body, and
    /// a recognized form is only that when nothing was dropped reaching
    /// it. `2⁵³·x₀² + x₀² − 2⁵³·x₀²` folds to `x₀²` and stores nothing;
    /// `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)` underflows the same way (gh #683).
    ///
    /// Handing that form out is what routed the reproduction in gh #685
    /// to the **LP** fast path: with the row's only quadratic term gone
    /// the classifier saw a linear row, `qp_extract` folded an empty
    /// linear part into `G`, and the constraint left the model
    /// altogether — `min −x₀` subject to a vanished row walks `x₀` to its
    /// `10⁶` bound and reports `Optimal`. A wrong answer, on the default
    /// route, with no option set (gh #685 part 2).
    ///
    /// The gate is [`Quad2::lost_terms`] and not an emptiness test, for
    /// the reason spelled out on [`Self::admitted_quad_form`]: partial
    /// cancellation leaves a non-empty map that is still short a term.
    /// Refusing costs reach only — the row falls back to the AD tape,
    /// and the model to the NLP path, which solves it soundly.
    ///
    /// `lost_terms` is the *inexact fold* and not the drop it leads to
    /// (gh #687), so the reach given up here is only the reach that has
    /// to be. `x − x` cancels exactly — nothing was lost, the form is
    /// the body, and it is still handed out; `2⁵³·x + x − 2⁵³·x` loses
    /// the `x` at `fl(2⁵³ + 1)`, and that is what this refuses.
    ///
    /// Use [`Self::quad_terms_dropped`] to tell the two `None`s apart.
    pub fn analyze_quadratic_full(&self) -> Option<QuadForm> {
        match self {
            NlBody::Tree(e) => {
                let form = recognize_expr(e)?;
                (!form.lost_terms()).then(|| quad_form_readout(&form))
            }
            NlBody::Quad(q) => (!q.form.lost_terms()).then(|| quad_form_readout(&q.form)),
        }
    }

    /// Whether the recognizer reached a degree-≤2 form for this body but
    /// lost at least one term getting there — the case
    /// [`Self::analyze_quadratic_full`] refuses.
    ///
    /// `false` both for a body that recognized cleanly and for one that
    /// did not recognize at all, so this separates the two reasons that
    /// accessor answers `None`; it is not a nonlinearity test. Meant for
    /// the *refusal* path (the classifier naming its reason), not the hot
    /// one: on a tree it re-runs the recognizer.
    pub fn quad_terms_dropped(&self) -> bool {
        match self {
            NlBody::Tree(e) => recognize_expr(e).is_some_and(|f| f.lost_terms()),
            NlBody::Quad(q) => q.form.lost_terms(),
        }
    }

    /// The form the constant-structure evaluator is allowed to use — i.e.
    /// [`Self::analyze_quadratic_full`] behind the *exactness* gate.
    ///
    /// A recognized body has already passed that gate: the parser admits
    /// only a flat sum of monomials, which is the same rule
    /// [`crate::nl_quadratic::is_expanded_quadratic`] applies to a tree.
    /// Reading a factored form out of stored coefficients cancels — five
    /// digits on `(x − 500000)²` — so the gate is on both arms or on
    /// neither (gh #588, Q4).
    ///
    /// A body this refuses for that reason is not out of reach, only out of
    /// *this* representation: [`Self::admitted_factored_form`] serves it by
    /// keeping the squares factored (gh #673), and only a body neither can
    /// express keeps its tape.
    ///
    /// ## The second gate: a term that was *lost* is a term that is missing
    ///
    /// `is_expanded_quadratic` is a gate on the *shape* the coefficients
    /// were derived from. It says nothing about whether the derivation kept
    /// them. A flat sum of monomials passes it and still folds to a form
    /// with an entry missing, because the fold is floating-point addition:
    /// `2⁵³·x₀² + x₀² − 2⁵³·x₀²` is `x₀²` and stores nothing, and
    /// `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)` underflows the same way (gh #683).
    ///
    /// Evaluating **that** form is not a five-digit cancellation, it is a
    /// missing term. At `x₀ = 3` the row reads `0` where its own tape reads
    /// `16`, and `∂g/∂x` reads `[0, 0]` where the tape reads `[8, 0]` — so
    /// the `≤` the row sits under stops constraining anything at all. In the
    /// reproduction (`issue_685_cancelled_quadratic_evaluation`) the solve
    /// then walks the objective variable to its `-10⁶` floor and reports
    /// `Optimal`, where the same bytes down the tape stop at `-0.281`. On
    /// the default route, with no option set. So the form is admitted only
    /// when [`Quad2::lost_terms`] is clear (gh #685 part 1).
    ///
    /// It has to be that flag and not an emptiness test. Partial
    /// cancellation is the same defect wearing a different face:
    /// `2⁵³·x₀² + x₀² − 2⁵³·x₀² + x₁²` keeps `x₁²`, so the map is not empty
    /// and [`Self::provably_affine`] answers `Some(false)` quite correctly —
    /// while the read-out is still short an entire `x₀²`. A gate that looked
    /// at emptiness would pass this and stay wrong.
    ///
    /// And it is the *loss*, not the drop. `x₀² − x₀²` folds through
    /// `fl(1) + fl(−1) = 0` with nothing rounded away, so its read-out is
    /// the whole body and it keeps this fast path; gating on the drop
    /// refused it alongside the absorbing row above, for arithmetic that
    /// lost nothing (gh #687).
    ///
    /// The cost is reach, not correctness: a row that lost a term goes back
    /// to the AD tape, which is where it was before Q4.
    pub fn admitted_quad_form(&self) -> Option<QuadForm> {
        match self {
            NlBody::Tree(e) => {
                if is_trivially_zero(e) || !is_expanded_quadratic(e) {
                    return None;
                }
                let form = recognize_expr(e)?;
                (!form.lost_terms()).then(|| quad_form_readout(&form))
            }
            NlBody::Quad(q) => (!q.form.lost_terms()).then(|| quad_form_readout(&q.form)),
        }
    }

    /// The **factored** form the constant-structure evaluator may use when
    /// [`Self::admitted_quad_form`] refuses (gh #673).
    ///
    /// That accessor's gate is `is_expanded_quadratic`, and what it refuses
    /// is a body whose read-out would be an algebraic *expansion* of what
    /// the writer wrote — `(x − 500000)²` read back as
    /// `x² − 10⁶x + 2.5·10¹¹`, five digits gone. The refusal was never
    /// about the body being unsuitable for constant-structure evaluation;
    /// it was about the *representation*. So a body written as a sum of
    /// squared residuals — which is every least-squares model, and 41 of
    /// `airport.nl`'s 42 rows — is served here instead, by keeping the
    /// writer's own grouping and squaring it at evaluation time exactly as
    /// the tape does. See
    /// [`recognize_factored_quadratic`](crate::nl_quadratic::recognize_factored_quadratic)
    /// for what is admitted and why it costs no accuracy.
    ///
    /// Answers `None` for a body the parser recognized: a
    /// [`NlBody::Quad`] is an already-flat sum of monomials by
    /// construction, so it has no factoring left to keep and
    /// [`Self::admitted_quad_form`] has already served it.
    ///
    /// ## This arm answers for the bodies refused on *shape*, and only those
    ///
    /// [`Self::admitted_quad_form`] says `None` for two different reasons,
    /// and only one of them is this arm's. A body refused because its shape
    /// is factored is what this serves. A body refused because a term went
    /// **missing** in the fold ([`Quad2::lost_terms`], gh #685) keeps its
    /// tape, and the explicit `is_expanded_quadratic` test here is what
    /// keeps it there: those bodies are flat sums of monomials, so the
    /// square-shaped ones among them — `2⁵³x₀² + x₀² − 2⁵³x₀²` is three —
    /// would otherwise be admitted here by the back door.
    ///
    /// What breaks when they are is worth stating, because it is *not* that
    /// the fast path becomes less accurate. Measured with this test
    /// removed: the row whose tape answers `16.0` at `x₀ = 3` is answered
    /// `9.0` by the factored arm — and `9.0` is the mathematically right
    /// value of `x₀²`, which the compensated outer sum (gh #702) recovers
    /// and the tape's naive fold does not. End to end the reproduction
    /// moves from `−1.812` to `−2.236`, which is `−√5`, the true optimum of
    /// the model those bytes describe.
    ///
    /// It is still a defect, for the reason this file's own doc comment
    /// gives: the tape is the reference because it is what the row means
    /// *to this solver*, not because it is exact. Two routes over the same
    /// bytes that answer `9` and `16` are a `POUNCE_DBG_NO_QUAD`-shaped
    /// divergence whichever one is closer to the algebra. (The Hessian
    /// **pattern** diverges too — `Σ 2wₖbₖbₖᵀ` folds that row's `(0, 0)` to
    /// exactly `0.0`, `2⁵⁴ + 2` tying back to `2⁵⁴`, and a zero entry is
    /// not stored where the tape declares one: `nnz_h` 2 → 1.)
    ///
    /// Pinned by
    /// `a_row_that_dropped_a_term_is_not_admitted_as_a_factored_form_either`.
    ///
    /// Callers must still try [`Self::admitted_quad_form`] first: both can
    /// answer for the same body (a bare `x²` is a monomial and a square),
    /// and the expanded arm is the cheaper evaluation — a matvec over a
    /// merged row rather than one squaring per term.
    pub fn admitted_factored_form(&self) -> Option<FactoredQuadratic> {
        match self {
            NlBody::Tree(e) => {
                if is_trivially_zero(e) || is_expanded_quadratic(e) {
                    return None;
                }
                recognize_factored_quadratic(e)
            }
            NlBody::Quad(_) => None,
        }
    }

    /// Whether this body is provably **affine** — degree ≤ 1 — as a
    /// three-valued answer: `Some(true)` proved affine, `Some(false)`
    /// proved to have a nonzero second derivative, `None` no proof
    /// either way.
    ///
    /// This is the degree question *without* the exactness gate
    /// [`Self::admitted_quad_form`] applies, and the difference is the
    /// point (gh #588, Q6). That gate exists because reading a value out
    /// of stored coefficients cancels for a factored form; nothing is
    /// read out here. The answer is used to decide whether a derivative
    /// may be **reused** across iterates — a question about the
    /// *degree* of the body, which a factored `(x − a)²` answers just as
    /// well as an expanded one.
    ///
    /// `None` is not evidence of nonlinearity: the recognizer refuses
    /// `2·(x + 1)`, which is affine. Consumers must treat it as "not
    /// established".
    ///
    /// ## What the exactness argument above still does not buy
    ///
    /// It is true that nothing is *evaluated* from the coefficients here.
    /// What the argument missed is that the **degree answer is itself
    /// computed by the coefficient arithmetic**: the recognizer sums a
    /// row's quadratic coefficients in floating point and drops the ones
    /// that reach exactly zero, so `2⁵³·x² + x² − 2⁵³·x²` — and
    /// `(10⁻²⁰⁰·x)·(10⁻²⁰⁰·x)`, by underflow — folded to an empty
    /// quadratic map and were reported *proved affine*. Q6's consumer then
    /// froze those rows' Jacobians for the whole solve (gh #683).
    ///
    /// So an empty quadratic map is a proof of degree ≤ 1 only when no term
    /// went missing getting there, which is what
    /// [`Quad2::lost_terms`](crate::nl_quadratic::Quad2::lost_terms)
    /// records. When one did, this answers `None` — the state the contract
    /// already reserved for "not established", which is why the fix needs
    /// nothing of its consumer.
    ///
    /// A term that cancelled **exactly** did not go missing, and gh #687 is
    /// where that stopped costing a proof: `x₀² − x₀²` is degree 0 by an
    /// add that rounded nothing, its tape holds `∂g/∂x` at zero for every
    /// `x`, and answering `None` for it gave up a whole solve of frozen
    /// Jacobian to be safe from arithmetic that never happened.
    ///
    /// Deliberately answers from the term maps rather than from
    /// [`Self::analyze_quadratic_full`]'s triplets: on `qssp180` that is
    /// 65 341 recognized rows, and materializing a `QuadHessian` per row
    /// to ask whether it is empty is the allocation-per-object cost Q3
    /// removed from the recognizer in the first place.
    pub fn provably_affine(&self) -> Option<bool> {
        match self {
            NlBody::Tree(e) => {
                if is_trivially_zero(e) {
                    return Some(true);
                }
                affine_from_form(&recognize_expr(e)?)
            }
            NlBody::Quad(q) => affine_from_form(&q.form),
        }
    }

    /// Add this body's structural variable support to `out`.
    pub fn collect_vars(&self, out: &mut BTreeSet<usize>) {
        match self {
            NlBody::Tree(e) => collect_vars(e, out),
            NlBody::Quad(q) => out.extend(q.vars.iter().map(|&v| v as usize)),
        }
    }
}

/// Give one body a constant-structure form, if either representation will
/// take it, and return its id.
///
/// The order is the contract [`NlBody::admitted_factored_form`] states:
/// the expanded read-out first, because it is the cheaper evaluation and
/// because it is what the `qcqp*` family — the target of gh #588's Q4 —
/// arrives as; the factored one only for the bodies that gate refuses
/// (gh #673). A body neither admits keeps its AD tape, which is where every
/// body was before Q4.
fn push_body_form(quad: &mut QuadraticStructure, body: &NlBody) -> Option<u32> {
    if let Some((h, lin, c)) = body.admitted_quad_form() {
        return Some(quad.push_form(&h, &lin, c));
    }
    let fq = body.admitted_factored_form()?;
    let terms: Vec<SquareTerm<'_>> = fq
        .squares
        .iter()
        .map(|t| SquareTerm {
            weight: t.weight,
            coefs: &t.coefs,
            constant: t.constant,
        })
        .collect();
    // `None` here is the gh #685 gate on this arm: the body is factored,
    // but assembling `Σ 2wₖbₖbₖᵀ` dropped an entry the tape will declare.
    // Falls through to the tape, like any other refusal.
    quad.push_factored_form(&terms, &fq.linear, fq.constant)
}

/// The degree read-out [`NlBody::provably_affine`] makes of a recognized
/// form, in one place so both arms answer it the same way.
///
/// A stored quadratic coefficient is a witness that the body is degree 2.
/// An *absent* one is only a witness of the opposite when nothing went
/// missing on the way — see [`Quad2::lost_terms`], gh #683 and gh #687.
fn affine_from_form(q: &Quad2) -> Option<bool> {
    if q.quadratic().is_empty() {
        (!q.lost_terms()).then_some(true)
    } else {
        Some(false)
    }
}

impl From<Expr> for NlBody {
    fn from(e: Expr) -> Self {
        NlBody::Tree(e)
    }
}

/// Parsed `.nl` problem in the form needed by `NlTnlp`.
#[derive(Debug, Clone)]
pub struct NlProblem {
    pub n: usize,
    pub m: usize,
    pub num_obj: usize,
    pub minimize: bool,
    pub obj_nonlinear: NlBody,
    pub obj_linear: Vec<(usize, Number)>,
    pub obj_constant: Number,
    /// Per-constraint nonlinear part (length m).
    pub con_nonlinear: Vec<NlBody>,
    /// Per-constraint linear part (length m), each a list of (var, coef).
    pub con_linear: Vec<Vec<(usize, Number)>>,
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    pub g_l: Vec<Number>,
    pub g_u: Vec<Number>,
    pub x0: Vec<Number>,
    pub lambda0: Vec<Number>,
    /// AMPL suffix dictionaries. Variable / constraint / objective
    /// suffixes are stored as dense vectors (length n / m / num_obj)
    /// with the sparse `.nl` `S`-segment entries scattered in, default
    /// zero. The integer / real split matches the `S`-segment header's
    /// kind bit (`0x4` ⇒ real, else integer). See
    /// <https://ampl.com/REFS/hooking2.pdf> §6 and the upstream `.nl`
    /// reader in `ref/Ipopt/src/Apps/AmplSolver/AmplTNLP.cpp`.
    pub suffixes: NlSuffixes,
    /// The model's own AMPL option words, taken verbatim from `.nl`
    /// header line 0 (`g<count> <opt0> <opt1> ...`). A solver echoes
    /// these back in the `.sol` `Options` block rather than interpreting
    /// them — see [`crate::sol_writer::format_sol_with_options`]. Empty
    /// for problems not built from a `.nl` file.
    pub ampl_options: Vec<i64>,
    /// The header's nonlinearity census, when the problem came from a `.nl`
    /// file whose header parsed cleanly. `None` for a model built in memory
    /// ([`NlProblem::from_expressions`]) — there is no header to read, and
    /// inventing one would let a consumer trust a count nobody computed.
    pub nl_counts: Option<NlCounts>,
    /// AMPL imported (external) functions declared via top-level `F` segments.
    /// Empty unless the `.nl` file calls compiled-C user functions (typically
    /// emitted by IDAES property packages — see issue #49).
    pub imported_funcs: Vec<ImportedFunc>,
    /// Variable names from the sibling `.col` file, index-aligned to `x`
    /// (one name per line, column order). Empty when no `.col` file was
    /// found — AMPL only emits it under `option auxfiles rc;`.
    ///
    /// Carrying names lets diagnostics report `flow_balance` / `T_reactor`
    /// instead of `c[3]` / `x[132]`. Lee et al. (2024) identify the gap
    /// between detecting an issue and tracing it to a *named* equation as a
    /// central roadblock for equation-oriented model debugging; threading
    /// names through to the solver/debugger is the prerequisite for closing
    /// it. See <https://doi.org/10.69997/sct.147875>.
    pub var_names: Vec<String>,
    /// Constraint names from the sibling `.row` file, index-aligned to `g`
    /// (one name per line, row order). Empty when no `.row` file was found.
    /// See [`NlProblem::var_names`] for why names are captured.
    pub con_names: Vec<String>,
    /// The `.nl` text this problem was parsed from, kept only when some
    /// body was recognized as a quadratic and therefore has no tree of its
    /// own. It is what makes [`NlProblem::con_expr`] able to hand back the
    /// *exact* tree rather than a re-derived one: same bytes, same parser,
    /// same `Expr`. `None` for [`NlProblem::from_expressions`] models and
    /// for a parse that recognized nothing.
    ///
    /// The text is resident during parsing either way, so keeping it costs
    /// nothing at the peak this is all in aid of — see
    /// `dev-notes/quadratic-structure-exploitation.md` §0f.
    pub src: Option<Arc<String>>,
    /// The `V`-segment common subexpressions, by CSE-local index. Held so a
    /// re-parse resolves `v<i>` (`i >= n`) to the *same* `Arc` the original
    /// parse did — `HybridTape::build_multi` keys sharing on pointer
    /// identity, so a rebuilt body that allocated fresh bodies would look
    /// unshared.
    pub cse_bodies: Vec<Arc<Expr>>,
}

/// The pieces of a model built in memory, as handed to
/// [`NlProblem::from_expressions`].
///
/// Everything is expressed as [`Expr`] trees — there is no linear/nonlinear
/// split to fill in, because the AD tape treats a linear term exactly like
/// any other subexpression (`.nl`'s `J`/`G` segments are a file-format
/// optimization, not an evaluator requirement). `n` is taken from the length
/// of `x_l`; `m` from the length of `constraints`.
///
/// One cost to that simplification, in *metadata* rather than values: with
/// `con_linear` empty, `get_constraints_linearity` tags a row `Linear` only
/// when its expression is literally `Const(0.0)`, so a genuinely linear row
/// built here reports `NonLinear`. Presolve consumes that tag, and the
/// direction is the safe one — it loses tightening it could have done, and
/// never asserts linearity that does not hold — but a frontend that cares
/// about presolve strength on linear rows should know the tag is
/// pessimistic on this path.
#[derive(Debug, Clone)]
pub struct NlProblemParts {
    /// `true` to minimize `objective`, `false` to maximize it. Matches
    /// [`NlProblem::minimize`]: the evaluator negates a maximize objective
    /// so callers always see the minimization form.
    pub minimize: bool,
    /// Objective expression.
    pub objective: Expr,
    /// Constant offset added to the objective.
    pub obj_constant: Number,
    /// One expression per constraint row; row `i` is bounded by
    /// `g_l[i] <= constraints[i](x) <= g_u[i]`.
    pub constraints: Vec<Expr>,
    /// Variable bounds and starting point, each length `n`. Use `±1e19`
    /// for "unbounded", the same sentinel the `.nl` reader emits.
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    pub x0: Vec<Number>,
    /// Constraint bounds, each length `m`. `g_l[i] == g_u[i]` is an
    /// equality row.
    pub g_l: Vec<Number>,
    pub g_u: Vec<Number>,
    /// Optional names, index-aligned to `x` / `g`. Empty is fine — every
    /// consumer falls back to indices (see [`NlProblem::var_names`]).
    pub var_names: Vec<String>,
    pub con_names: Vec<String>,
}

impl NlProblem {
    /// Assemble a problem from expression trees, with no `.nl` file
    /// anywhere in the loop (issue #469).
    ///
    /// A modeling frontend that already has its own expression DAG should
    /// come in here rather than serialize to `.nl` and re-parse: the round
    /// trip is not only slower, it is *lossy*, because `.nl` writers
    /// routinely refuse operators this tape supports natively (`atan2`,
    /// `min`/`max`, and — with no `.nl` opcode at all — [`UnaryOp::Erf`]).
    ///
    /// The result is an ordinary [`NlProblem`], so it feeds
    /// [`NlTnlp::try_new`] and gets exactly the evaluators a parsed model
    /// does: objective, gradient, constraints, Jacobian + structure,
    /// Lagrangian Hessian + structure, and
    /// [`NlTnlp::hessian_vector_product`].
    ///
    /// Errors on a length mismatch or on a `Var(i)` index at or beyond `n`
    /// — the latter would otherwise be an out-of-bounds read in the tape's
    /// forward sweep, so it must be caught while it is still a diagnosable
    /// user error.
    pub fn from_expressions(parts: NlProblemParts) -> Result<NlProblem, String> {
        let NlProblemParts {
            minimize,
            objective,
            obj_constant,
            constraints,
            x_l,
            x_u,
            x0,
            g_l,
            g_u,
            var_names,
            con_names,
        } = parts;

        let n = x_l.len();
        let m = constraints.len();
        let check = |name: &str, got: usize, want: usize| -> Result<(), String> {
            if got == want {
                Ok(())
            } else {
                Err(format!(
                    "from_expressions: {name} has length {got}, expected {want}"
                ))
            }
        };
        check("x_u", x_u.len(), n)?;
        check("x0", x0.len(), n)?;
        check("g_l", g_l.len(), m)?;
        check("g_u", g_u.len(), m)?;
        if !var_names.is_empty() {
            check("var_names", var_names.len(), n)?;
        }
        if !con_names.is_empty() {
            check("con_names", con_names.len(), m)?;
        }

        // Numeric validation, the same screen the `.nl` reader applies
        // (gh #847). `lower_bound_present` / `upper_bound_present` are
        // `is_finite() && ...`, so a caller that builds a model here and hands
        // in a non-finite bound gets it silently dropped -- read as "no bound
        // declared" -- exactly as a file containing `1e400` was. `-inf` on a
        // lower side and `+inf` on an upper one are the sentinel said a
        // different way and are normalized; everything else, `NaN` included,
        // is refused.
        let mut x_l = x_l;
        let mut x_u = x_u;
        let mut g_l = g_l;
        let mut g_u = g_u;
        for (i, v) in x_l.iter_mut().enumerate() {
            *v = finite_bound_or_err(&format!("x_l[{i}]"), *v, true)
                .map_err(|e| format!("from_expressions: {e}"))?;
        }
        for (i, v) in x_u.iter_mut().enumerate() {
            *v = finite_bound_or_err(&format!("x_u[{i}]"), *v, false)
                .map_err(|e| format!("from_expressions: {e}"))?;
        }
        for (i, v) in g_l.iter_mut().enumerate() {
            *v = finite_bound_or_err(&format!("g_l[{i}]"), *v, true)
                .map_err(|e| format!("from_expressions: {e}"))?;
        }
        for (i, v) in g_u.iter_mut().enumerate() {
            *v = finite_bound_or_err(&format!("g_u[{i}]"), *v, false)
                .map_err(|e| format!("from_expressions: {e}"))?;
        }
        for (i, v) in x0.iter().enumerate() {
            finite_or_err(&format!("x0[{i}]"), *v).map_err(|e| format!("from_expressions: {e}"))?;
        }
        finite_or_err("obj_constant", obj_constant)
            .map_err(|e| format!("from_expressions: {e}"))?;

        // Structural validation. Memoized on `Cse` pointer identity so a
        // heavily-shared DAG costs O(nodes) rather than O(inlined tree).
        let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
        validate_expr(&objective, n, &mut seen).map_err(|e| format!("objective {e}"))?;
        for (i, c) in constraints.iter().enumerate() {
            validate_expr(c, n, &mut seen).map_err(|e| format!("constraint {i} {e}"))?;
        }

        Ok(NlProblem {
            n,
            m,
            num_obj: 1,
            minimize,
            obj_nonlinear: NlBody::Tree(objective),
            obj_linear: Vec::new(),
            obj_constant,
            con_nonlinear: constraints.into_iter().map(NlBody::Tree).collect(),
            con_linear: vec![Vec::new(); m],
            x_l,
            x_u,
            g_l,
            g_u,
            x0,
            lambda0: vec![0.0; m],
            suffixes: NlSuffixes::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            // No header was read, so there is no census to report. Consumers
            // fall back to walking the trees, which is what they would have
            // to do here anyway.
            nl_counts: None,
            var_names,
            con_names,
            // Built from trees, so every body has one and there is nothing
            // to rebuild from.
            src: None,
            cse_bodies: Vec::new(),
        })
    }

    /// The objective's nonlinear body as an [`Expr`].
    ///
    /// Borrowed when the tree is resident, rebuilt when the parser
    /// recognized the body and skipped building it. The rebuild re-parses
    /// the body's own bytes with the same parser that produced the rest of
    /// the model, so the result is the tree a non-recognizing parse would
    /// have produced — structurally identical, coefficient bit patterns
    /// included, and sharing the same `Cse` allocations.
    ///
    /// It is not free: it allocates the nodes the recognizer avoided. Reach
    /// for [`NlBody::quad`] first if the coefficients are what you want.
    pub fn obj_expr(&self) -> std::borrow::Cow<'_, Expr> {
        self.body_expr(&self.obj_nonlinear, "objective")
    }

    /// Row `k`'s nonlinear body as an [`Expr`]. See [`Self::obj_expr`].
    ///
    /// # Panics
    ///
    /// Panics if `k >= m`, or if re-parsing a recognized body fails — the
    /// latter cannot happen for a problem this crate produced (the bytes
    /// parsed once already, by this code) and means the `NlProblem` was
    /// assembled by hand with a `src` that does not match its bodies.
    pub fn con_expr(&self, k: usize) -> std::borrow::Cow<'_, Expr> {
        self.body_expr(&self.con_nonlinear[k], "constraint")
    }

    fn body_expr<'a>(&'a self, body: &'a NlBody, what: &str) -> std::borrow::Cow<'a, Expr> {
        match body {
            NlBody::Tree(e) => std::borrow::Cow::Borrowed(e),
            NlBody::Quad(q) => {
                let src = self
                    .src
                    .as_deref()
                    .unwrap_or_else(|| panic!("{what} body was recognized but no source is kept"));
                std::borrow::Cow::Owned(
                    parse_body_fragment(&src[q.src.clone()], self.n, &self.cse_bodies)
                        .unwrap_or_else(|e| panic!("re-parsing a recognized {what} body: {e}")),
                )
            }
        }
    }
}

/// Structural check on an expression bound for [`NlProblem::from_expressions`]:
/// every `Var(i)` must satisfy `i < n`, and no `Expr::Funcall` may appear.
///
/// Both are things the tape cannot recover from later. An out-of-range
/// `Var` is an out-of-bounds read in the forward sweep. A `Funcall` is
/// worse-looking than it is fatal: `from_expressions` has nowhere to put
/// the `F`-segment declarations an AMPL imported function needs
/// (`NlProblemParts` has no field for them, and the built problem's
/// `imported_funcs` is necessarily empty), so *any* funcall on this path is
/// unresolvable. Accepting it would surface as "AMPLFUNC is not set" —
/// advice the user cannot act on, because setting `AMPLFUNC` just moves the
/// failure to "funcall id N has no F<N> declaration". Rejecting it here
/// says the true thing: this door does not carry external functions; go
/// through `read_nl` / `parse_nl_text` for a model that needs them.
///
/// `seen` memoizes `Cse` bodies by pointer identity across calls, so
/// passing one set through a whole problem keeps the walk linear in
/// distinct DAG nodes rather than exponential in sharing depth. Skipping a
/// repeat visit is sound because the caller aborts on the first violation:
/// reaching a body a second time proves the first visit found none.
fn validate_expr(
    e: &Expr,
    n: usize,
    seen: &mut std::collections::HashSet<*const Expr>,
) -> Result<(), String> {
    match e {
        Expr::Const(_) => Ok(()),
        Expr::Var(i) => {
            if *i < n {
                Ok(())
            } else {
                Err(format!("references Var({i}) but n = {n}"))
            }
        }
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            validate_expr(a, n, seen)?;
            validate_expr(b, n, seen)
        }
        Expr::Unary(_, a) | Expr::Not(a) => validate_expr(a, n, seen),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                validate_expr(a, n, seen)?;
            }
            Ok(())
        }
        Expr::Cond { cond, then_, else_ } => {
            validate_expr(cond, n, seen)?;
            validate_expr(then_, n, seen)?;
            validate_expr(else_, n, seen)
        }
        Expr::Cse(body) => {
            if seen.insert(Arc::as_ptr(body)) {
                validate_expr(body, n, seen)
            } else {
                Ok(())
            }
        }
        Expr::Funcall { id, .. } => Err(format!(
            "references AMPL imported function id {id}, which this path cannot \
             resolve: a problem built from expressions has no F-segment \
             declarations to bind it to. Load such a model with read_nl or \
             parse_nl_text instead."
        )),
    }
}

/// Suffix data parsed out of `S`-segments. Sparse entries are scattered
/// into dense vectors at problem load time so callers can index by
/// variable / constraint number directly. Empty maps when the `.nl`
/// file declared no suffixes.
#[derive(Debug, Clone, Default)]
pub struct NlSuffixes {
    /// Variable-level integer suffixes (kind = 0). Each vector has
    /// length `n_full` (problem variables).
    pub var_int: BTreeMap<String, Vec<Index>>,
    /// Constraint-level integer suffixes (kind = 1). Length `m_full`.
    pub con_int: BTreeMap<String, Vec<Index>>,
    /// Objective-level integer suffixes (kind = 2). Length `num_obj`.
    pub obj_int: BTreeMap<String, Vec<Index>>,
    /// Problem-level integer suffixes (kind = 3). Single value per name.
    pub problem_int: BTreeMap<String, Index>,
    /// Variable-level real suffixes (kind = 4). Length `n_full`.
    pub var_real: BTreeMap<String, Vec<Number>>,
    /// Constraint-level real suffixes (kind = 5). Length `m_full`.
    pub con_real: BTreeMap<String, Vec<Number>>,
    /// Objective-level real suffixes (kind = 6). Length `num_obj`.
    pub obj_real: BTreeMap<String, Vec<Number>>,
    /// Problem-level real suffixes (kind = 7). Single value per name.
    pub problem_real: BTreeMap<String, Number>,
}

/// Parse an `.nl` file from disk.
///
/// After parsing the `.nl` body, this also looks for AMPL's optional
/// sibling name files — `stub.col` (variable names) and `stub.row`
/// (constraint names), emitted only when the modeler sets
/// `option auxfiles rc;`. When present and well-formed they populate
/// [`NlProblem::var_names`] / [`NlProblem::con_names`]; when absent or
/// malformed the names stay empty and every downstream consumer falls
/// back to indices. Names are a diagnostic nicety, never load-blocking
/// (cf. Lee et al. 2024, <https://doi.org/10.69997/sct.147875>).
pub fn read_nl_file(path: &Path) -> Result<NlProblem, String> {
    // AMPL invokes a solver with an extensionless *stub* — e.g.
    // `pounce mymodel -AMPL` — and expects `mymodel.nl` to be read (and
    // the `.col`/`.row`/`.sol` siblings named off the same stem). If the
    // path as given is missing but appending `.nl` names an existing file,
    // resolve to that. This only ever *adds* a fallback: an existing path
    // is read verbatim, so nothing changes for callers that already pass a
    // full `.nl` path (Pyomo, `--nl-file`, the second-positional form).
    let resolved = if path.exists() {
        path.to_path_buf()
    } else {
        let with_nl = append_extension(path, "nl");
        if with_nl.exists() {
            with_nl
        } else {
            path.to_path_buf()
        }
    };
    let txt = std::fs::read_to_string(&resolved)
        .map_err(|e| format!("could not read {}: {}", resolved.display(), e))?;
    // By value: a recognized body keeps a byte range into this text rather
    // than a tree, so it is moved into the problem instead of copied.
    let mut prob = parse_nl_string(txt, std::env::var("POUNCE_DBG_NO_QUAD").is_err())?;
    prob.var_names = read_name_file(&resolved.with_extension("col"), prob.n);
    prob.con_names = read_name_file(&resolved.with_extension("row"), prob.m);
    Ok(prob)
}

/// Append `.ext` to `path`'s full file name (AMPL stub convention:
/// `mymodel` → `mymodel.nl`), as opposed to [`Path::with_extension`],
/// which would *replace* an existing extension. A stub that itself
/// contains a dot (`my.model` → `my.model.nl`) is therefore handled the
/// way AMPL names it.
fn append_extension(path: &Path, ext: &str) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".");
    name.push(ext);
    std::path::PathBuf::from(name)
}

/// Read an AMPL name file (`.col` / `.row`): one name per line, in index
/// order. Returns the first `expected` names, or an empty vector when the
/// file is missing, unreadable, or has fewer than `expected` lines.
///
/// Returning empty (rather than erroring) on any mismatch is deliberate:
/// names are an optional diagnostic aid, so a missing or truncated file
/// must never block a solve. The `.take(expected)` also drops AMPL's
/// convention of appending the objective name after the constraint names
/// in `.row`, keeping the result aligned 1:1 with `g`.
fn read_name_file(path: &Path, expected: usize) -> Vec<String> {
    let Ok(txt) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let names: Vec<String> = txt.lines().take(expected).map(str::to_owned).collect();
    if names.len() == expected {
        names
    } else {
        Vec::new()
    }
}

/// The value of a constraint's nonlinear part when that part is a
/// constant, else `None`. Drives the constant-row-body fold in
/// [`parse_nl_text`].
///
/// "Constant" is decided by *evaluation*, not by syntax: a literal
/// `Expr::Const` is the common case, but `o0 n1 n2` is just as constant
/// and is folded too.
///
/// Declines in two cases, both of which would make the fold unsound:
/// * The expression calls an AMPL imported function. Its value depends on
///   a shared library resolved much later (`nl_external::ExternalResolver`),
///   so it is not a parse-time constant even with constant arguments — and
///   [`eval_expr`] panics on `Funcall` rather than guess.
/// * The value is not finite (`n0 / n0`, `log(-1)`, an overflow). Pushing
///   a NaN or infinity into a bound would corrupt a row that is merely
///   infeasible; leaving the expression in place keeps it a solver-time
///   fact.
fn row_constant_value(e: &Expr) -> Option<Number> {
    // The identity zero the parser preallocates for every untouched row is
    // by far the most common input; settle it without walking anything.
    if let Expr::Const(c) = e {
        return c.is_finite().then_some(*c);
    }
    let mut vars: BTreeSet<usize> = BTreeSet::new();
    collect_vars(e, &mut vars);
    if !vars.is_empty() {
        return None;
    }
    let mut funcs: BTreeSet<usize> = BTreeSet::new();
    crate::nl_external::collect_funcall_ids(e, &mut funcs);
    if !funcs.is_empty() {
        return None;
    }
    // Variable-free, so no `Expr::Var` can index into the (empty) point.
    let v = eval_expr(e, &[]);
    v.is_finite().then_some(v)
}

/// Parse `.nl` text content. Public so tests can use string literals.
///
/// Degree-2 bodies are recognized as the token stream is read, so their
/// `Expr` trees are never built (gh #588, Q5); `POUNCE_DBG_NO_QUAD=1`
/// turns that off and restores the pre-Q5 parse exactly, which is what the
/// A/B reference and gh #540's guard test need. See
/// [`parse_nl_text_with_quadratic`] for the same switch as a parameter.
pub fn parse_nl_text(txt: &str) -> Result<NlProblem, String> {
    parse_nl_text_with_quadratic(txt, std::env::var("POUNCE_DBG_NO_QUAD").is_err())
}

/// [`parse_nl_text`] with parse-time quadratic recognition explicitly on
/// or off.
///
/// The env var the plain entry point reads is process-global, which is
/// exactly wrong for a differential test that has to parse the *same* text
/// both ways in one process and compare the results — so the knob is a
/// parameter here, mirroring [`NlTnlp::try_new_with_quadratic`].
pub fn parse_nl_text_with_quadratic(txt: &str, use_quadratic: bool) -> Result<NlProblem, String> {
    parse_nl_string(txt.to_string(), use_quadratic)
}

/// [`parse_nl_text_with_quadratic`], taking the text by value.
///
/// A recognized body keeps a byte range into the source rather than a tree,
/// so the source outlives the parse. Taking ownership means the file
/// [`read_nl_file`] already read is *moved* into the problem instead of
/// copied: the text is resident during parsing either way, so this costs
/// nothing at the peak the phase is about.
pub fn parse_nl_string(txt: String, use_quadratic: bool) -> Result<NlProblem, String> {
    let src = Arc::new(txt);
    let mut p = Parser::new(&src, use_quadratic);
    p.parse_header()?;
    let n = p.n;
    let m = p.m;
    let num_obj = p.num_obj;

    let mut con_nonlinear: Vec<NlBody> = (0..m).map(|_| NlBody::Tree(Expr::Const(0.0))).collect();
    let mut obj_nonlinear = NlBody::Tree(Expr::Const(0.0));
    let mut minimize = true;
    let mut obj_linear: Vec<(usize, Number)> = Vec::new();
    let mut con_linear: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    let mut x_l = vec![-1e19; n];
    let mut x_u = vec![1e19; n];
    let mut g_l = vec![-1e19; m];
    let mut g_u = vec![1e19; m];
    let mut x0 = vec![0.0; n];
    let mut lambda0 = vec![0.0; m];
    let mut suffixes = NlSuffixes::default();
    let mut imported_funcs: Vec<ImportedFunc> = Vec::new();
    // Segment presence, for the truncation check after the loop (gh#785).
    let mut saw_r = false;
    let mut saw_b = false;

    while let Some(line) = p.peek_segment_line() {
        let tag = line
            .trim_start()
            .chars()
            .next()
            .ok_or("unexpected blank segment header")?;
        match tag {
            'C' => {
                let (_hdr, rest) = p.eat_segment_header()?;
                let _ = rest;
                let idx = parse_segment_index(_hdr, 'C')?;
                if idx >= m {
                    return Err(format!("C{idx} out of range; m={m}"));
                }
                con_nonlinear[idx] = p.parse_body()?;
            }
            'O' => {
                let (hdr, _rest) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(format!("malformed O-segment header: {hdr}"));
                }
                let idx = parse_segment_index(parts[0], 'O')?;
                let kind: i32 = parts[1].parse().map_err(|e| format!("O kind: {e}"))?;
                if idx == 0 {
                    minimize = kind == 0;
                    obj_nonlinear = p.parse_body()?;
                } else {
                    // Extra objectives are read but ignored.
                    let _ = p.parse_expr()?;
                }
            }
            'r' => {
                p.eat_segment_header()?;
                saw_r = true;
                for i in 0..m {
                    let line = p.next_data_line()?;
                    let (lo, hi) = parse_bound_line(line)?;
                    g_l[i] = lo;
                    g_u[i] = hi;
                }
            }
            'b' => {
                p.eat_segment_header()?;
                saw_b = true;
                for i in 0..n {
                    let line = p.next_data_line()?;
                    let (lo, hi) = parse_bound_line(line)?;
                    x_l[i] = lo;
                    x_u[i] = hi;
                }
            }
            'k' => {
                // Column counts in the Jacobian; we don't need their
                // values for evaluation (the J segments give explicit
                // lists), but we must consume exactly as many data lines
                // as follow or the segment stream desyncs. The `.nl`
                // format writes that line count in the header itself
                // (`k<count>`), and the standard value is `n-1`. Read the
                // declared count rather than assuming it: a file with a
                // nonstandard count would otherwise leave us reading the
                // wrong number of lines, swallowing a later segment header
                // (or stopping short) and failing with a confusing,
                // far-removed error. Validate against the expected `n-1`
                // so a mismatch surfaces here, clearly, at its source.
                let (hdr, _) = p.eat_segment_header()?;
                let declared = parse_segment_index(hdr, 'k')?;
                let expected = if n == 0 { 0 } else { n - 1 };
                if declared != expected {
                    return Err(format!(
                        "k-segment declares {declared} column-count lines but \
                         the standard count for n={n} variables is {expected}"
                    ));
                }
                for _ in 0..declared {
                    p.next_data_line()?;
                }
            }
            'J' => {
                let (hdr, _) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(format!("malformed J-segment header: {hdr}"));
                }
                let row = parse_segment_index(parts[0], 'J')?;
                let nz: usize = parts[1].parse().map_err(|e| format!("J nz: {e}"))?;
                if row >= m {
                    return Err(format!("J{row} out of range"));
                }
                for _ in 0..nz {
                    let line = p.next_data_line()?;
                    let (var, coef) = parse_var_coef(line)?;
                    // Validate the column index here: an out-of-range `var`
                    // would otherwise be stored and panic as a slice OOB
                    // (`x[var]`) during constraint evaluation. Mirror the
                    // clean parse error used for the row index above.
                    if var >= n {
                        return Err(format!(
                            "J{row} entry variable index {var} out of range (n={n})"
                        ));
                    }
                    con_linear[row].push((var, coef));
                }
            }
            'G' => {
                let (hdr, _) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(format!("malformed G-segment header: {hdr}"));
                }
                let idx = parse_segment_index(parts[0], 'G')?;
                let nz: usize = parts[1].parse().map_err(|e| format!("G nz: {e}"))?;
                let mut acc = Vec::with_capacity(nz);
                for _ in 0..nz {
                    let line = p.next_data_line()?;
                    let (var, coef) = parse_var_coef(line)?;
                    // Same as J: reject an out-of-range gradient column index
                    // up front rather than letting it panic on `x[var]` later.
                    if var >= n {
                        return Err(format!(
                            "G{idx} entry variable index {var} out of range (n={n})"
                        ));
                    }
                    acc.push((var, coef));
                }
                if idx == 0 {
                    obj_linear = acc;
                }
            }
            'x' => {
                let (hdr, _) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                let nx: usize = parts
                    .first()
                    .and_then(|s| s.trim_start_matches('x').parse().ok())
                    .ok_or_else(|| format!("malformed x-segment header: {hdr}"))?;
                for _ in 0..nx {
                    let line = p.next_data_line()?;
                    let (idx, val) = parse_var_coef(line)?;
                    // Reject out-of-range indices as a parse error, matching
                    // J/G strictness, rather than silently dropping the entry
                    // (which hides a corrupt initial-primal segment).
                    if idx >= n {
                        return Err(format!(
                            "x-segment variable index {idx} out of range (n={n})"
                        ));
                    }
                    x0[idx] = val;
                }
            }
            'd' => {
                let (hdr, _) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                let nd: usize = parts
                    .first()
                    .and_then(|s| s.trim_start_matches('d').parse().ok())
                    .ok_or_else(|| format!("malformed d-segment header: {hdr}"))?;
                for _ in 0..nd {
                    let line = p.next_data_line()?;
                    let (idx, val) = parse_var_coef(line)?;
                    // Reject out-of-range indices as a parse error, matching
                    // J/G strictness, rather than silently dropping the entry
                    // (which hides a corrupt initial-dual segment).
                    if idx >= m {
                        return Err(format!(
                            "d-segment constraint index {idx} out of range (m={m})"
                        ));
                    }
                    lambda0[idx] = val;
                }
            }
            'V' => p.parse_v_segment()?,
            'S' => {
                parse_suffix_segment(&mut p, n, m, num_obj, &mut suffixes)?;
            }
            'F' => {
                // AMPL imported (external) function declaration:
                // `F<k> <type> <nargs> <name>`.
                let (hdr, _rest) = p.eat_segment_header()?;
                let parts: Vec<&str> = hdr.split_whitespace().collect();
                if parts.is_empty() {
                    return Err(format!("malformed F-segment header: '{hdr}'"));
                }
                let id = parse_segment_index(parts[0], 'F')?;
                let kind: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let nargs: i64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                let name = parts.get(3).copied().unwrap_or("").to_string();
                imported_funcs.push(ImportedFunc {
                    id,
                    kind,
                    nargs,
                    name,
                });
            }
            other => return Err(format!("unknown .nl segment tag '{other}'")),
        }
    }

    // A `.nl` file that ends early is not a smaller model — it is a corrupt
    // one, and every segment the truncation ate is a piece of the problem
    // that silently reverts to a default: a dropped `r` leaves every row at
    // ±1e19 (i.e. unconstrained), a dropped `b` leaves every variable free,
    // a dropped `J` leaves every row's linear part empty. The segment loop
    // cannot tell those defaults from a legitimately absent segment, so the
    // truncated model solves, and reports `SolveSucceeded` with a
    // confidently wrong objective — which is strictly worse than the parse
    // error every other malformed input already gets (gh#785). The header
    // declares enough to tell the two apart; check it here, once, against
    // what the segments actually delivered.
    //
    // `r` and `b` are unconditional in the format: AMPL writes both whenever
    // the model has rows / columns at all, free rows and free variables
    // included — those get the `3` ("no bounds") code, not an omitted line.
    if m > 0 && !saw_r {
        return Err(format!(
            "missing `r` (constraint-bounds) segment for a model declaring {m} \
             constraint(s): the .nl file is truncated or corrupt"
        ));
    }
    if n > 0 && !saw_b {
        return Err(format!(
            "missing `b` (variable-bounds) segment for a model declaring {n} \
             variable(s): the .nl file is truncated or corrupt"
        ));
    }
    // The `J` segments must deliver exactly the Jacobian nonzero count the
    // header declares — `nzc` is the sum of their lengths by construction,
    // nonlinear-only columns included (they are written with a zero
    // coefficient, not omitted). This is the check that catches a truncation
    // landing *after* `r` and `b`, where the bounds are all present and only
    // the coefficients are gone.
    if let Some(declared) = p.declared_jac_nnz {
        let parsed: usize = con_linear.iter().map(Vec::len).sum();
        if parsed != declared {
            return Err(format!(
                "header declares {declared} Jacobian nonzero(s) but the J \
                 segments supply {parsed}: the .nl file is truncated or corrupt"
            ));
        }
    }

    // Normalize constant row bodies into the row bounds, so that
    // "the nonlinear part is the identity zero" and "this row's body is
    // Σaⱼxⱼ" mean the same thing for every downstream consumer.
    //
    // A `C<i>` segment holding a variable-free expression (`n3.0`, or
    // anything that evaluates to a constant such as `o0 n1 n2`) is an
    // affine row — but it arrives with a non-zero `con_nonlinear[i]`,
    // which every consumer reads as "nonlinear": the linearity predicate
    // (`get_constraints_linearity`), which is what makes presolve's
    // linear-equality reduction decline the row; the CLI problem
    // classifier (`is_trivially_zero` in `dispatch.rs`, whose fallback
    // polynomial walk absorbs a bare literal but not a constant it has to
    // compute — so an otherwise plain LP carrying a `sqrt(9)` classified
    // NLP and never reached the convex path); and the FBBT translator.
    // Folding here fixes all of them at once, with no per-consumer audit.
    //
    // The shift is exact and invisible from outside: the body drops by `c`
    // and each bound drops by `c` with it, so the feasible set, the active
    // set, and the duals are unchanged (`gh #492`). This is the same
    // normalization `qp_extract::analyze_quadratic_full` already performs
    // ad hoc via `const_shift`, promoted to the parse boundary.
    for i in 0..m {
        // A recognized body is degree 2, so it is not a constant row and
        // this fold has nothing to do with it.
        let Some(tree) = con_nonlinear[i].tree() else {
            continue;
        };
        let Some(c) = row_constant_value(tree) else {
            continue;
        };
        // Presence is directional (gh #401): shifting an *absent* bound
        // would turn the ±1e19 sentinel into a real bound for `c < 0`
        // (lower) or `c > 0` (upper), inventing a constraint. Leave the
        // sentinels alone.
        if lower_bound_present(g_l[i]) {
            g_l[i] -= c;
        }
        if upper_bound_present(g_u[i]) {
            g_u[i] -= c;
        }
        con_nonlinear[i] = NlBody::Tree(Expr::Const(0.0));
    }

    // The source is kept only when something in it has no tree of its own.
    // A model with nothing recognized is byte-for-byte the pre-Q5 problem,
    // extra field included.
    let any_recognized =
        obj_nonlinear.quad().is_some() || con_nonlinear.iter().any(|b| b.quad().is_some());
    let kept_src = any_recognized.then(|| Arc::clone(&src));

    Ok(NlProblem {
        n,
        m,
        num_obj,
        minimize,
        obj_nonlinear,
        obj_linear,
        obj_constant: 0.0,
        con_nonlinear,
        con_linear,
        x_l,
        x_u,
        g_l,
        g_u,
        x0,
        lambda0,
        suffixes,
        ampl_options: p.ampl_options.clone(),
        nl_counts: p.nl_counts,
        imported_funcs,
        // `.nl` text carries no names; `read_nl_file` fills these from the
        // sibling `.col`/`.row` files when present.
        var_names: Vec::new(),
        con_names: Vec::new(),
        src: kept_src,
        cse_bodies: p.cses.clone(),
    })
}

/// Parse a single `S`-segment. Format (Gay 2005, "Hooking Your Solver
/// to AMPL", §6, and `ref/Ipopt/src/Apps/AmplSolver/AmplTNLP.cpp`):
///
/// ```text
/// S<kind> <nentries> <suffix_name>
/// <idx> <value>      ... nentries lines
/// ```
///
/// `<kind>` is a 3-bit encoding:
/// * Bits 0-1 select the suffix target: 0 = variables, 1 = constraints,
///   2 = objectives, 3 = problem-level.
/// * Bit 2 (`0x4`) selects the value type: 0 = integer, 1 = real.
///
/// Sparse entries scatter into a freshly-allocated dense vector (zero
/// default), sized for the target dimension. Problem-level suffixes
/// (kind = 3 / 7) carry a single value.
fn parse_suffix_segment(
    p: &mut Parser,
    n: usize,
    m: usize,
    num_obj: usize,
    out: &mut NlSuffixes,
) -> Result<(), String> {
    let (hdr, _) = p.eat_segment_header()?;
    let parts: Vec<&str> = hdr.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(format!(
            "malformed S-segment header: '{hdr}' (expected `S<kind> <n> <name>`)"
        ));
    }
    let kind_str = parts[0].trim_start_matches('S');
    let kind: u32 = kind_str
        .parse()
        .map_err(|e| format!("S kind '{kind_str}': {e}"))?;
    let nentries: usize = parts[1].parse().map_err(|e| format!("S nentries: {e}"))?;
    let name = parts[2].to_string();

    let is_real = (kind & 0x4) != 0;
    let target = kind & 0x3;
    let target_dim = match target {
        0 => n,
        1 => m,
        2 => num_obj,
        3 => 0, // problem-level — entries are single-valued (idx=0)
        _ => unreachable!("kind & 0x3 is in 0..=3"),
    };

    // Pre-allocate dense buffers (default zero). Problem-level kinds
    // (3 / 7) hold a single scalar — we still read the (idx, value)
    // pairs but only the value field is meaningful.
    let mut int_buf: Vec<Index> = if !is_real && target != 3 {
        vec![0; target_dim]
    } else {
        Vec::new()
    };
    let mut real_buf: Vec<Number> = if is_real && target != 3 {
        vec![0.0; target_dim]
    } else {
        Vec::new()
    };
    let mut problem_int: Index = 0;
    let mut problem_real: Number = 0.0;

    for _ in 0..nentries {
        let line = p.next_data_line()?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!(
                "malformed S-segment entry '{line}' (expected `<idx> <value>`)"
            ));
        }
        let idx: usize = parts[0]
            .parse()
            .map_err(|e| format!("S entry idx '{}': {e}", parts[0]))?;
        if target != 3 && idx >= target_dim {
            return Err(format!(
                "S-suffix '{name}' index {idx} out of range for target dim {target_dim}"
            ));
        }
        if is_real {
            let v: Number = parts[1]
                .parse()
                .map_err(|e| format!("S real entry value '{}': {e}", parts[1]))?;
            if target == 3 {
                problem_real = v;
            } else {
                real_buf[idx] = v;
            }
        } else {
            let v: Index = parts[1]
                .parse()
                .map_err(|e| format!("S int entry value '{}': {e}", parts[1]))?;
            if target == 3 {
                problem_int = v;
            } else {
                int_buf[idx] = v;
            }
        }
    }

    match (target, is_real) {
        (0, false) => {
            out.var_int.insert(name, int_buf);
        }
        (1, false) => {
            out.con_int.insert(name, int_buf);
        }
        (2, false) => {
            out.obj_int.insert(name, int_buf);
        }
        (3, false) => {
            out.problem_int.insert(name, problem_int);
        }
        (0, true) => {
            out.var_real.insert(name, real_buf);
        }
        (1, true) => {
            out.con_real.insert(name, real_buf);
        }
        (2, true) => {
            out.obj_real.insert(name, real_buf);
        }
        (3, true) => {
            out.problem_real.insert(name, problem_real);
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn parse_segment_index(s: &str, tag: char) -> Result<usize, String> {
    let trimmed = s.trim_start_matches(tag);
    trimmed
        .parse()
        .map_err(|e| format!("malformed {tag}-segment index '{s}': {e}"))
}

// `parse_bound_line` and `parse_var_coef` run once per `r` / `b` / `J` /
// `G` / `x` / `d` data line, so between them they see every Jacobian and
// gradient nonzero in the file. Both walk the whitespace iterator
// directly instead of collecting a `Vec<&str>` first — that collect was
// a heap allocation per line on top of the one the reader used to make
// handing the line over.
/// Refuse a non-finite number read out of a `.nl` file (gh #847).
///
/// `str::parse::<f64>()` accepts `inf`, `-inf` and `nan`, and it also *returns*
/// `inf` for any literal that overflows the type — `1e400` is a plausible thing
/// for a model generator to write. Nothing downstream treats such a value as an
/// error, and in one place it is actively misread: `lower_bound_present` /
/// `upper_bound_present` are `is_finite() && ...`, so a non-finite bound is
/// indistinguishable from a bound that was never declared, and is silently
/// dropped. On a model that a lower bound of `1e300` makes infeasible, the same
/// bound written `1e400` returned `EXIT: Optimal Solution Found.` with exit code
/// 0. A `nan` is worse: it propagates into the answer, and the solve reports
/// `Objective: nan` under `Solve_Succeeded`.
///
/// Ipopt refuses this input ("Invalid number"), POUNCE's own NLP arm refuses it,
/// and `pounce.solve_qp` refuses a non-finite bound with a bespoke `ValueError`.
/// There is no reading on which `Optimal` is the intended answer, so the reader
/// refuses it too.
fn finite_or_err(what: &str, v: Number) -> Result<Number, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!(
            "invalid number: {what} is {v}, which is not finite"
        ))
    }
}

/// The same screen for a *bound* slot, where one non-finite value has an
/// unambiguous meaning and is normalized instead of refused.
///
/// `.nl` states "no bound" with a bound **kind** (1 = upper only, 2 = lower
/// only, 3 = free), so a non-finite number in a bound slot is a corrupt value
/// rather than a notation — with one exception per side. `-inf` in a *lower*
/// slot and `+inf` in an *upper* slot say precisely what the `±1e19` sentinel
/// says, and a writer that emits them means it, so they map to the sentinel.
///
/// Everything else is refused, and the asymmetry is the whole point: `+inf` as
/// a *lower* bound is the gh #847 case. It is not "unbounded below" — it is an
/// empty box, and reading it as "absent" is what turned an infeasible model
/// into an `Optimal` one. `NaN` is refused on either side, having no meaning at
/// all.
fn finite_bound_or_err(what: &str, v: Number, lower: bool) -> Result<Number, String> {
    if v.is_finite() {
        return Ok(v);
    }
    if lower && v == Number::NEG_INFINITY {
        return Ok(-1e19);
    }
    if !lower && v == Number::INFINITY {
        return Ok(1e19);
    }
    Err(format!(
        "invalid number: {what} is {v}, which is not finite (a `.nl` file \
         states an absent bound with a bound kind of 1, 2 or 3, not with a \
         non-finite value)"
    ))
}

fn parse_bound_line(line: &str) -> Result<(Number, Number), String> {
    let mut parts = line.split_whitespace();
    let kind: i32 = parts
        .next()
        .ok_or("empty bound line")?
        .parse()
        .map_err(|e| format!("bound kind: {e}"))?;
    let lo;
    let hi;
    match kind {
        0 => {
            // 0  lo  hi
            let (l, h) = (parts.next(), parts.next());
            let (Some(l), Some(h)) = (l, h) else {
                return Err(format!("bound kind 0 needs 2 values: '{line}'"));
            };
            lo = finite_bound_or_err("lo", l.parse().map_err(|e| format!("lo: {e}"))?, true)?;
            hi = finite_bound_or_err("hi", h.parse().map_err(|e| format!("hi: {e}"))?, false)?;
        }
        1 => {
            // 1  hi
            let Some(h) = parts.next() else {
                return Err(format!("bound kind 1 needs 1 value: '{line}'"));
            };
            lo = -1e19;
            hi = finite_bound_or_err("hi", h.parse().map_err(|e| format!("hi: {e}"))?, false)?;
        }
        2 => {
            // 2  lo
            let Some(l) = parts.next() else {
                return Err(format!("bound kind 2 needs 1 value: '{line}'"));
            };
            lo = finite_bound_or_err("lo", l.parse().map_err(|e| format!("lo: {e}"))?, true)?;
            hi = 1e19;
        }
        3 => {
            // 3  (free)
            lo = -1e19;
            hi = 1e19;
        }
        4 => {
            // 4  eq
            let Some(v) = parts.next() else {
                return Err(format!("bound kind 4 needs 1 value: '{line}'"));
            };
            // An equality has no "absent" side, so neither infinity is a
            // notation here and both are refused.
            let v: Number = finite_or_err("eq bound", v.parse().map_err(|e| format!("eq: {e}"))?)?;
            lo = v;
            hi = v;
        }
        5 => return Err("complementarity (kind 5) bounds are not supported".into()),
        other => return Err(format!("unknown bound kind {other}")),
    }
    Ok((lo, hi))
}

fn parse_var_coef(line: &str) -> Result<(usize, Number), String> {
    let mut parts = line.split_whitespace();
    let (Some(v), Some(c)) = (parts.next(), parts.next()) else {
        return Err(format!("malformed var/coef line: '{line}'"));
    };
    let v: usize = v.parse().map_err(|e| format!("var idx: {e}"))?;
    let c: Number = finite_or_err("coefficient", c.parse().map_err(|e| format!("coef: {e}"))?)?;
    Ok((v, c))
}

/// Build an [`NlCounts`] from `.nl` header lines 3 (`nlc nlo`) and 5
/// (`nlvc nlvo nlvb`).
///
/// Returns `None` unless both lines carry the full complement of
/// non-negative integers, so a truncated or non-conforming header reads as
/// "unknown" rather than as zeros — "no nonlinear variables" is a claim, and
/// a header that failed to parse has not made it.
fn parse_nl_counts(line3: &str, line5: &str) -> Option<NlCounts> {
    let nums = |line: &str, want: usize| -> Option<Vec<usize>> {
        let v: Vec<usize> = line
            .split_whitespace()
            .take(want)
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;
        (v.len() == want).then_some(v)
    };
    let cons_objs = nums(line3, 2)?;
    let vars = nums(line5, 3)?;
    Some(NlCounts {
        nl_cons: cons_objs[0],
        nl_objs: cons_objs[1],
        nl_vars_cons: vars[0],
        nl_vars_objs: vars[1],
        nl_vars_both: vars[2],
    })
}

struct Parser<'a> {
    lines: Vec<&'a str>,
    /// Start address and length of the source text. A recognized body
    /// records the byte *range* it consumed so it can be re-parsed later
    /// (see [`QuadBody::src`]), and every line borrows from this buffer, so
    /// one subtraction per line recovers its offset.
    ///
    /// Deliberately not a per-line offset table: on a 119 MB generated
    /// `qcqp500-3c` the file is ~17 M lines, and a `Vec<usize>` beside the
    /// existing `Vec<&str>` would add 136 MB to the peak this phase exists
    /// to reduce.
    txt_base: usize,
    txt_len: usize,
    pos: usize,
    n: usize,
    m: usize,
    num_obj: usize,
    /// Number of AMPL imported (external) functions declared in the header.
    n_funcs: usize,
    /// Header lines 3 and 5, when both parsed. See [`NlCounts`].
    nl_counts: Option<NlCounts>,
    /// `nzc` from header line 8: the number of Jacobian nonzeros the file
    /// *declares*. [`parse_nl_string`] cross-checks it against the number the
    /// `J` segments actually deliver, which is how a file truncated before
    /// them is told from a model that genuinely has none (gh#785). `None`
    /// when the header does not carry it in the documented shape.
    declared_jac_nnz: Option<usize>,
    ampl_options: Vec<i64>,
    /// Common subexpressions (`V` segments). Index in this vec is the
    /// CSE-local index, i.e. the global `.nl` index minus `n`.
    cses: Vec<Arc<Expr>>,
    /// Recognize degree-2 bodies from the token stream instead of building
    /// their trees (gh #588, Q5). Off restores the pre-Q5 parse exactly.
    quad_enabled: bool,
    /// Per-CSE answers the streaming recognizer needs about a `V` body it
    /// may be handed a reference to, all keyed by CSE-local index: the
    /// body's own degree-≤2 form, whether it is legal on a sum spine,
    /// whether it is legal *inside* a monomial, and its variable support.
    ///
    /// These are computed from the built `V` body with the same functions
    /// `NlTnlp` would apply to a whole row, so a reference costs a lookup
    /// and cannot drift from what the tree walk would have said. `V` bodies
    /// keep their trees regardless — they are shared, so the memory is
    /// amortized over every reference, and dropping them would mean
    /// rebuilding them for the rows that are *not* recognized.
    cse_quad: Vec<Option<Quad2>>,
    cse_sum_ok: Vec<bool>,
    cse_mono_ok: Vec<bool>,
    cse_vars: Vec<Vec<u32>>,
    cse_depth: Vec<u32>,
}

/// One pending operator on the streaming recognizer's stack.
struct QFrame {
    op: QOp,
    /// Operands still to be read.
    remaining: usize,
    /// Read this frame's operands in monomial mode — no `+`/`-` may appear
    /// below it. See [`crate::nl_quadratic::is_expanded_quadratic`].
    mono: bool,
}

/// The operators the streaming recognizer accepts. One variant per shape
/// that [`crate::nl_quadratic::recognize_expr`] handles; everything else
/// makes it bail.
enum QOp {
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    /// `o5`/`o81`/`o83`: base and exponent both read.
    Pow,
    /// `o82`: square, exponent implicit.
    Square,
    Sum(usize),
}

impl<'a> Parser<'a> {
    fn new(txt: &'a str, quad_enabled: bool) -> Self {
        let lines: Vec<&str> = txt.lines().collect();
        Self {
            lines,
            txt_base: txt.as_ptr() as usize,
            txt_len: txt.len(),
            pos: 0,
            n: 0,
            m: 0,
            num_obj: 0,
            n_funcs: 0,
            nl_counts: None,
            declared_jac_nnz: None,
            ampl_options: Vec::new(),
            cses: Vec::new(),
            quad_enabled,
            cse_quad: Vec::new(),
            cse_sum_ok: Vec::new(),
            cse_mono_ok: Vec::new(),
            cse_vars: Vec::new(),
            cse_depth: Vec::new(),
        }
    }

    /// Byte offset in the source text of the start of line `line`, or the
    /// end of the text when the cursor has run off it.
    ///
    /// Pointer arithmetic against the same allocation, never a deref: every
    /// line in `lines` borrows from the source buffer, so the difference is
    /// its offset.
    fn byte_at(&self, line: usize) -> usize {
        match self.lines.get(line) {
            Some(l) => l.as_ptr() as usize - self.txt_base,
            None => self.txt_len,
        }
    }

    fn next_line(&mut self) -> Option<&'a str> {
        while self.pos < self.lines.len() {
            let l = self.lines[self.pos];
            self.pos += 1;
            // Strip comment after '#' for header / data lines (but
            // leave the segment-tag tokens untouched — they are the
            // first token on the line).
            let trimmed = strip_comment(l).trim();
            if !trimmed.is_empty() {
                return Some(l);
            }
        }
        None
    }

    /// Next non-blank line, comment stripped and trimmed.
    ///
    /// Borrows from the source text rather than allocating: the result
    /// is `&'a str`, tied to the `.nl` buffer and not to `self`, so it
    /// outlives the `&mut self` this took. A large `.nl` is mostly data
    /// lines — 620k of them for a 20k-variable model — and returning an
    /// owned `String` put one heap allocation on every single one.
    fn next_data_line(&mut self) -> Result<&'a str, String> {
        while self.pos < self.lines.len() {
            let l = self.lines[self.pos];
            self.pos += 1;
            let trimmed = strip_comment(l).trim();
            if !trimmed.is_empty() {
                return Ok(trimmed);
            }
        }
        Err("unexpected end of file in data line".to_string())
    }

    fn parse_header(&mut self) -> Result<(), String> {
        let line0 = self.next_line().ok_or("empty .nl file")?;
        let trimmed = strip_comment(line0).trim();
        let first = trimmed.chars().next().ok_or("empty header line")?;
        if first != 'g' {
            return Err(format!(
                "only ASCII (g-) .nl files supported; got header '{trimmed}'"
            ));
        }
        // Line 0 is `g<count> <opt0> <opt1> ...`: the digits glued to the
        // `g` say how many AMPL option words follow on the same line.
        // A solver echoes them back in the `.sol` `Options` block, so
        // keep them verbatim. Malformed or truncated option lists are not
        // fatal — the writer falls back to a generic block.
        let mut words = trimmed.split_whitespace();
        let n_opts: usize = words.next().and_then(|w| w[1..].parse().ok()).unwrap_or(0);
        let opts: Vec<i64> = words.filter_map(|w| w.parse().ok()).collect();
        if opts.len() >= n_opts {
            self.ampl_options = opts[..n_opts].to_vec();
        }

        // Header line 2: n_vars n_cons n_objs ranges eqns
        let l2 = self.next_data_line()?;
        let nums: Vec<&str> = l2.split_whitespace().collect();
        if nums.len() < 3 {
            return Err(format!("malformed line 2: '{l2}'"));
        }
        self.n = nums[0].parse().map_err(|e| format!("n: {e}"))?;
        self.m = nums[1].parse().map_err(|e| format!("m: {e}"))?;
        self.num_obj = nums[2].parse().map_err(|e| format!("num_obj: {e}"))?;

        // Header line 3: `nlc nlo`. Line 4: the network-constraint census,
        // which pounce has no use for. Line 5: `nlvc nlvo nlvb`. Together
        // these are the model's nonlinearity census — see [`NlCounts`].
        //
        // A header that does not carry them in the documented shape leaves
        // `nl_counts` at `None` rather than at a guess: every consumer has a
        // walk-the-trees fallback, and a fabricated count is worse than an
        // absent one. The rest of the header stays tolerant in the same way
        // the `nfunc` read below is.
        let l3 = self.next_data_line()?;
        let _l4_network = self.next_data_line()?;
        let l5 = self.next_data_line()?;
        self.nl_counts = parse_nl_counts(l3, l5);
        // Line 5 (0-indexed from `g`-header): `nwv nfunc arith flags`
        let l6 = self.next_data_line()?;
        let nums5: Vec<&str> = l6.split_whitespace().collect();
        self.n_funcs = nums5.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        // Lines 6..10 are metadata we mostly don't need. Line 7 is the
        // discrete-variable census; line 8 is `nzc nzo` — the Jacobian and
        // objective-gradient nonzero counts, kept because `nzc` is what
        // catches a file truncated before its `J` segments (gh#785); lines
        // 9 and 10 are the maximum name lengths and the common-expression
        // census.
        let _l7_discrete = self.next_data_line()?;
        let l8 = self.next_data_line()?;
        // Tolerant like the `nfunc` read above: a header that does not
        // carry the count in the documented shape leaves it `None`, and
        // the cross-check that reads it is skipped rather than fabricated.
        self.declared_jac_nnz = l8.split_whitespace().next().and_then(|s| s.parse().ok());
        let _l9_name_lens = self.next_data_line()?;
        let _l10_common_exprs = self.next_data_line()?;
        Ok(())
    }

    fn peek_segment_line(&mut self) -> Option<&'a str> {
        let saved = self.pos;
        let l = self.next_line()?;
        self.pos = saved;
        Some(l)
    }

    /// Eat the next non-blank line as a segment header. Returns the
    /// whole header (after stripping comments) and the comment text.
    fn eat_segment_header(&mut self) -> Result<(&'a str, &'a str), String> {
        let raw = self
            .next_line()
            .ok_or_else(|| "expected segment header".to_string())?;
        let (hdr, comment) = split_comment(raw);
        Ok((hdr.trim(), comment.trim()))
    }

    /// Parse one `C`/`O` body, recognizing it from the token stream when
    /// that is possible instead of building its tree (gh #588, Q5).
    ///
    /// The cursor is the whole mechanism. `Parser` is line-based, so a
    /// failed recognition costs one assignment to rewind — the same trick
    /// `parse_funcall_arg` already uses to route a Hollerith literal — and
    /// the fallback then produces exactly the tree this file has always
    /// produced. Nothing downstream can tell which arm ran except by
    /// measuring memory.
    ///
    /// Only **degree-2** forms are kept. A body that recognizes as constant
    /// or linear is re-parsed as a tree and stored as one: the constant-row
    /// fold (gh #492), the linearity contract and the classifier's LP fast
    /// path all key on the identity zero and on trees, they are cheap
    /// already, and the memory that motivates this phase is entirely in
    /// degree-2 rows. Refusing to touch them keeps the change confined to
    /// the rows it is for.
    fn parse_body(&mut self) -> Result<NlBody, String> {
        if !self.quad_enabled {
            return Ok(NlBody::Tree(self.parse_expr()?));
        }
        let saved = self.pos;
        if let Some((form, vars, depth)) = self.parse_expr_quadratic() {
            if !form.quadratic().is_empty() {
                let src = self.byte_at(saved)..self.byte_at(self.pos);
                return Ok(NlBody::Quad(Box::new(QuadBody {
                    form,
                    vars,
                    src,
                    depth,
                })));
            }
        }
        self.pos = saved;
        Ok(NlBody::Tree(self.parse_expr()?))
    }

    /// Read one expression off the token stream as a degree-≤2 form,
    /// **without building it**, or give up.
    ///
    /// This is the same computation as
    /// `is_expanded_quadratic(e) && recognize_expr(e)` run on the tree these
    /// tokens parse to — the accuracy gate and the algebra, interleaved so
    /// neither needs the tree. That equivalence is the phase's correctness
    /// claim and is asserted directly, bit for bit, over every body of every
    /// `.nl` file in the repository by
    /// `pounce-cli/tests/quad_parse_differential.rs`.
    ///
    /// Two things it must reproduce and not merely approximate:
    ///
    /// * **The exactness rule.** `is_expanded_quadratic` admits a body only
    ///   when reading it as `½xᵀHx + aᵀx + c` repeats the additions the
    ///   writer already wrote — expanding a *factored* form cancels, which
    ///   is what took `airport.nl` from 16 to 300 iterations in Q4. Here
    ///   that rule is structural rather than a separate pass: `+`/`-` are
    ///   the spine, everything under a `*`, `/` or `^` is read in monomial
    ///   mode, and a `+` seen in monomial mode gives up.
    /// * **The association.** `recognize_expr` folds an `o54` sumlist from
    ///   the *first* operand to the last, which is also the order the AD
    ///   tape sums in, so this folds the same way — operands arrive here in
    ///   file order and are folded as they arrive. Summation order is not
    ///   observable on distinct monomials and is exactly observable on
    ///   repeated ones.
    ///
    /// On `None` the cursor is left wherever it stopped; the caller rewinds.
    fn parse_expr_quadratic(&mut self) -> Option<(Quad2, Vec<u32>, u32)> {
        let mut frames: Vec<QFrame> = Vec::new();
        let mut vals: Vec<Quad2> = Vec::new();
        let mut vars: Vec<u32> = Vec::new();
        // Deepest node of the tree these tokens describe. A leaf sits under
        // `frames.len()` operators, and a CSE reference carries its body's
        // depth under it — the same convention `pounce-py`'s `expr_depth`
        // uses, because that is who reads the answer.
        let mut depth: u32 = 0;
        let note_leaf = |frames: &[QFrame], depth: &mut u32, below: u32| {
            let d = u32::try_from(frames.len()).unwrap_or(u32::MAX);
            *depth = (*depth).max(d.saturating_add(1).saturating_add(below));
        };

        loop {
            let mono = frames.last().is_some_and(|f| f.mono);
            // A `^` base must be atomic — `Const`, `Var` or a `Neg` — or the
            // body is a factored form dressed as a monomial. `is_monomial`
            // applies the same test to `Pow`'s left operand.
            let pow_base = matches!(
                frames.last(),
                Some(QFrame {
                    op: QOp::Pow,
                    remaining: 2,
                    ..
                }) | Some(QFrame {
                    op: QOp::Square,
                    remaining: 1,
                    ..
                })
            );

            let raw = self.next_line()?;
            let tok = strip_comment(raw).trim();
            let first = tok.chars().next()?;
            match first {
                'n' => {
                    // A constant base is atomic; `is_monomial` accepts it.
                    // A non-finite literal bails out of the fast path so the
                    // general parser reaches it and reports the error rather
                    // than folding it into a quadratic (gh #847).
                    let v: Number = tok[1..]
                        .trim()
                        .parse()
                        .ok()
                        .filter(|v: &Number| v.is_finite())?;
                    note_leaf(&frames, &mut depth, 0);
                    vals.push(Quad2::of_constant(v));
                }
                'v' => {
                    let i: usize = tok[1..].trim().parse().ok()?;
                    if i < self.n {
                        note_leaf(&frames, &mut depth, 0);
                        vars.push(u32::try_from(i).ok()?);
                        vals.push(Quad2::of_var(i));
                    } else {
                        // A CSE reference lowers to `Expr::Cse`, which is
                        // *not* one of the shapes `is_monomial` accepts as a
                        // power's base.
                        if pow_base {
                            return None;
                        }
                        let local = i.checked_sub(self.n)?;
                        let ok = if mono {
                            *self.cse_mono_ok.get(local)?
                        } else {
                            *self.cse_sum_ok.get(local)?
                        };
                        if !ok {
                            return None;
                        }
                        let form = self.cse_quad.get(local)?.clone()?;
                        note_leaf(&frames, &mut depth, *self.cse_depth.get(local)?);
                        vars.extend_from_slice(self.cse_vars.get(local)?);
                        vals.push(form);
                    }
                }
                'o' => {
                    let code: i32 = tok[1..].trim().parse().ok()?;
                    if pow_base && code != 16 {
                        return None;
                    }
                    let (op, arity, child_mono) = match code {
                        // The sum spine. Inside a monomial there is no such
                        // thing, and the body is a factored form.
                        0 if !mono => (QOp::Add, 2, false),
                        1 if !mono => (QOp::Sub, 2, false),
                        16 => (QOp::Neg, 1, mono),
                        54 if !mono => {
                            let count_line = self.next_data_line().ok()?;
                            let count: usize =
                                count_line.split_whitespace().next()?.parse().ok()?;
                            (QOp::Sum(count), count, false)
                        }
                        // Monomial operators: everything below them is read
                        // in monomial mode.
                        2 => (QOp::Mul, 2, true),
                        3 => (QOp::Div, 2, true),
                        5 | 81 | 83 => (QOp::Pow, 2, true),
                        82 => (QOp::Square, 1, true),
                        // Transcendentals, comparisons, conditionals,
                        // min/max lists, and `+`/`-` under a monomial.
                        _ => return None,
                    };
                    if arity == 0 {
                        // An empty `o54` is the empty sum: zero.
                        note_leaf(&frames, &mut depth, 0);
                        vals.push(Quad2::default());
                    } else {
                        frames.push(QFrame {
                            op,
                            remaining: arity,
                            mono: child_mono,
                        });
                        continue;
                    }
                }
                // `f` (imported function call), `h`, and anything else.
                _ => return None,
            }

            // A value has just been produced. Close every frame it completes.
            while let Some(f) = frames.last_mut() {
                f.remaining -= 1;
                if f.remaining > 0 {
                    break;
                }
                let f = frames.pop()?;
                let combined = apply_quad_op(f.op, &mut vals)?;
                vals.push(combined);
            }
            if frames.is_empty() {
                break;
            }
        }

        if vals.len() != 1 {
            return None;
        }
        vars.sort_unstable();
        vars.dedup();
        Some((vals.pop()?, vars, depth))
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let raw = self
            .next_line()
            .ok_or_else(|| "expected expression token".to_string())?;
        // Borrowed, not owned: this runs once per node of every
        // expression tree in the file, so an owned `String` here is one
        // heap allocation per tape op in the whole model.
        let tok = strip_comment(raw).trim();
        if tok.is_empty() {
            return Err("empty expression token".into());
        }
        let first = tok.chars().next().ok_or("empty expression token")?;
        match first {
            'n' => {
                let v: Number = tok[1..]
                    .trim()
                    .parse()
                    .map_err(|e| format!("n value: {e}"))?;
                // A `nan` literal in the objective body reached the answer
                // itself: `Objective: nan` under `Solve_Succeeded` and exit
                // code 0 (gh #847).
                Ok(Expr::Const(finite_or_err("numeric literal", v)?))
            }
            'v' => {
                let i: usize = tok[1..]
                    .trim()
                    .parse()
                    .map_err(|e| format!("v index: {e}"))?;
                Ok(self.var_or_cse(i)?)
            }
            'o' => {
                let code: i32 = tok[1..]
                    .trim()
                    .parse()
                    .map_err(|e| format!("opcode: {e}"))?;
                self.parse_opcode(code)
            }
            'f' => {
                // AMPL imported (external) function call: `f<id> <nargs>`
                // followed by nargs child expressions (or string literals).
                let rest = &tok[1..];
                let mut parts = rest.split_whitespace();
                let id_str = parts
                    .next()
                    .ok_or_else(|| format!("missing function id in '{tok}'"))?;
                let nargs_str = parts
                    .next()
                    .ok_or_else(|| format!("missing nargs in '{tok}'"))?;
                let id: usize = id_str
                    .parse()
                    .map_err(|e| format!("bad function id '{id_str}': {e}"))?;
                let nargs: usize = nargs_str
                    .parse()
                    .map_err(|e| format!("bad funcall nargs '{nargs_str}': {e}"))?;
                let mut args: Vec<FuncallArg> = Vec::with_capacity(nargs);
                for _ in 0..nargs {
                    args.push(self.parse_funcall_arg()?);
                }
                Ok(Expr::Funcall { id, args })
            }
            't' | 'u' => Err(format!("unsupported expression token '{tok}'")),
            other => Err(format!(
                "unexpected expression token start '{other}': '{tok}'"
            )),
        }
    }

    /// Parse one argument to an AMPL imported function. An argument
    /// is either a normal expression (real-valued) or a string literal
    /// in the form `h<len>:<chars>`. AMPL emits string args only when the
    /// function was declared `FUNCADD_STRING_ARGS` (e.g. component name
    /// or a parameters-directory path for IDAES Helmholtz functions).
    fn parse_funcall_arg(&mut self) -> Result<FuncallArg, String> {
        // Peek the next non-blank line so we can route `h...` differently.
        let saved = self.pos;
        let raw = self
            .next_line()
            .ok_or_else(|| "expected funcall argument".to_string())?;
        // A string arg is a Hollerith literal `h<len>:<chars>` where the
        // chars are *exactly* `<len>` bytes and may legitimately contain
        // '#'. We must NOT strip a trailing comment before extracting the
        // content (that would truncate e.g. a path like `a#b`), and we
        // honor the declared length rather than splitting loosely on ':'.
        // Detect the form from the leading non-blank char of the raw line;
        // no expression opcode (`o`/`v`/`n`/`f`) begins with 'h'.
        let lead = raw.trim_start();
        if let Some(after_h) = lead.strip_prefix('h') {
            let colon = after_h
                .find(':')
                .ok_or_else(|| format!("malformed Hollerith string arg (no ':'): {lead:?}"))?;
            let len: usize = after_h[..colon]
                .trim()
                .parse()
                .map_err(|e| format!("Hollerith length in {lead:?}: {e}"))?;
            let chars = &after_h[colon + 1..];
            if chars.len() < len {
                return Err(format!(
                    "Hollerith string shorter than declared length {len}: {chars:?}"
                ));
            }
            // Take exactly `len` bytes; anything past it (trailing
            // whitespace, a real comment) is not part of the string.
            if !chars.is_char_boundary(len) {
                return Err(format!(
                    "Hollerith length {len} splits a multibyte char in {chars:?}"
                ));
            }
            Ok(FuncallArg::Str(chars[..len].to_string()))
        } else {
            // Rewind: parse_expr re-consumes the line we just peeked.
            self.pos = saved;
            Ok(FuncallArg::Real(self.parse_expr()?))
        }
    }

    fn parse_opcode(&mut self, code: i32) -> Result<Expr, String> {
        match code {
            0 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Add, Box::new(a), Box::new(b)))
            }
            1 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b)))
            }
            2 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b)))
            }
            3 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Div, Box::new(a), Box::new(b)))
            }
            5 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Pow, Box::new(a), Box::new(b)))
            }
            15 => Ok(Expr::Unary(UnaryOp::Abs, Box::new(self.parse_expr()?))),
            16 => Ok(Expr::Unary(UnaryOp::Neg, Box::new(self.parse_expr()?))),
            39 => Ok(Expr::Unary(UnaryOp::Sqrt, Box::new(self.parse_expr()?))),
            41 => Ok(Expr::Unary(UnaryOp::Sin, Box::new(self.parse_expr()?))),
            42 => Ok(Expr::Unary(UnaryOp::Log10, Box::new(self.parse_expr()?))),
            43 => Ok(Expr::Unary(UnaryOp::Log, Box::new(self.parse_expr()?))),
            44 => Ok(Expr::Unary(UnaryOp::Exp, Box::new(self.parse_expr()?))),
            46 => Ok(Expr::Unary(UnaryOp::Cos, Box::new(self.parse_expr()?))),
            38 => Ok(Expr::Unary(UnaryOp::Tan, Box::new(self.parse_expr()?))),
            49 => Ok(Expr::Unary(UnaryOp::Atan, Box::new(self.parse_expr()?))),
            53 => Ok(Expr::Unary(UnaryOp::Acos, Box::new(self.parse_expr()?))),
            40 => Ok(Expr::Unary(UnaryOp::Sinh, Box::new(self.parse_expr()?))),
            45 => Ok(Expr::Unary(UnaryOp::Cosh, Box::new(self.parse_expr()?))),
            37 => Ok(Expr::Unary(UnaryOp::Tanh, Box::new(self.parse_expr()?))),
            51 => Ok(Expr::Unary(UnaryOp::Asin, Box::new(self.parse_expr()?))),
            52 => Ok(Expr::Unary(UnaryOp::Acosh, Box::new(self.parse_expr()?))),
            50 => Ok(Expr::Unary(UnaryOp::Asinh, Box::new(self.parse_expr()?))),
            47 => Ok(Expr::Unary(UnaryOp::Atanh, Box::new(self.parse_expr()?))),
            // atan2(y, x): binary, operand order `y` then `x`.
            48 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Atan2, Box::new(a), Box::new(b)))
            }
            // Relational comparisons (binary). Operand order is
            // `left OP right`.
            22 => self.parse_compare(CmpOp::Lt),
            23 => self.parse_compare(CmpOp::Le),
            24 => self.parse_compare(CmpOp::Eq),
            28 => self.parse_compare(CmpOp::Ge),
            29 => self.parse_compare(CmpOp::Gt),
            30 => self.parse_compare(CmpOp::Ne),
            // Logical connectives.
            20 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::Or(Box::new(a), Box::new(b)))
            }
            21 => {
                let a = self.parse_expr()?;
                let b = self.parse_expr()?;
                Ok(Expr::And(Box::new(a), Box::new(b)))
            }
            34 => Ok(Expr::Not(Box::new(self.parse_expr()?))),
            // if-then-else: condition, then-value, else-value.
            35 => {
                let cond = self.parse_expr()?;
                let then_ = self.parse_expr()?;
                let else_ = self.parse_expr()?;
                Ok(Expr::Cond {
                    cond: Box::new(cond),
                    then_: Box::new(then_),
                    else_: Box::new(else_),
                })
            }
            54 => {
                // Variadic sum: next data line gives the count.
                let count_line = self.next_data_line()?;
                let count: usize = count_line
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| "missing variadic count".to_string())?
                    .parse()
                    .map_err(|e| format!("variadic count: {e}"))?;
                let mut args = Vec::with_capacity(count);
                for _ in 0..count {
                    args.push(self.parse_expr()?);
                }
                Ok(Expr::Sum(args))
            }
            // Variadic min (o11 MINLIST) / max (o12 MAXLIST): like o54,
            // a count data line followed by that many operands.
            11 | 12 => {
                let count_line = self.next_data_line()?;
                let count: usize = count_line
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| "missing min/max list count".to_string())?
                    .parse()
                    .map_err(|e| format!("min/max list count: {e}"))?;
                let mut args = Vec::with_capacity(count);
                for _ in 0..count {
                    args.push(self.parse_expr()?);
                }
                if code == 11 {
                    Ok(Expr::MinList(args))
                } else {
                    Ok(Expr::MaxList(args))
                }
            }
            // AMPL power specializations (ASL `opcode.hd` 81/82/83). AMPL
            // emits these in place of the general `o5` (OPPOW) as a hint that
            // one operand is constant. The distinction exists because an
            // integer / half-integer constant power is evaluated by a
            // mul/sqrt chain that stays real for a negative base, whereas the
            // general `pow` (via `exp(c·ln x)`) returns NaN there. Structurally
            // they read exactly like `o5`, so they lower to the same `Pow` AST
            // and reuse the existing constant-power tape lowering (see
            // `nl_tape::try_emit_const_pow`). Arity/operand order confirmed
            // against the ASL reader and the `ampl/mp` opcode table:
            // POW_CONST_EXP / POW_CONST_BASE are binary `base, exp`; POW2 is
            // unary with an implicit exponent of 2.
            //
            // o81 OP1POW: `base ^ (const exponent)` — binary, operands
            // `base` then `exp` (the exponent is a numeric node here).
            81 => {
                let base = self.parse_expr()?;
                let exp = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exp)))
            }
            // o82 OP2POW: square — unary, single operand; exponent 2 implicit.
            82 => {
                let base = self.parse_expr()?;
                Ok(Expr::Binary(
                    BinOp::Pow,
                    Box::new(base),
                    Box::new(Expr::Const(2.0)),
                ))
            }
            // o83 OPCPOW: `(const base) ^ exponent` — binary, operands `base`
            // (the numeric node) then `exp`.
            83 => {
                let base = self.parse_expr()?;
                let exp = self.parse_expr()?;
                Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exp)))
            }
            other => Err(format!("unsupported opcode o{other}")),
        }
    }

    /// Parse the two operands of a relational opcode into an
    /// [`Expr::Compare`]. Operand order is `left OP right`.
    fn parse_compare(&mut self, op: CmpOp) -> Result<Expr, String> {
        let a = self.parse_expr()?;
        let b = self.parse_expr()?;
        Ok(Expr::Compare(op, Box::new(a), Box::new(b)))
    }

    /// Resolve a `v<i>` token into either a plain variable reference
    /// (`i < n`) or a shared CSE reference (`i >= n`).
    fn var_or_cse(&self, i: usize) -> Result<Expr, String> {
        if i < self.n {
            Ok(Expr::Var(i))
        } else {
            let local = i - self.n;
            self.cses
                .get(local)
                .map(|rc| Expr::Cse(rc.clone()))
                .ok_or_else(|| {
                    format!(
                        "v{i} references CSE {local} but only {} have been defined",
                        self.cses.len()
                    )
                })
        }
    }

    /// Parse a `V<k> <nlin> <type>` common-subexpression segment. The
    /// CSE evaluates to `nonlinear_expr + sum_i coef_i * v_{var_i}`.
    /// CSEs are numbered starting at `n` and must appear in order.
    fn parse_v_segment(&mut self) -> Result<(), String> {
        let (hdr, _) = self.eat_segment_header()?;
        let parts: Vec<&str> = hdr.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("malformed V-segment header: {hdr}"));
        }
        let cse_idx = parse_segment_index(parts[0], 'V')?;
        let nlin: usize = parts[1].parse().map_err(|e| format!("V nlin: {e}"))?;
        // parts[2] (type) is ignored; values >0 just mark special-purpose CSEs.
        let mut linear: Vec<(usize, Number)> = Vec::with_capacity(nlin);
        for _ in 0..nlin {
            let line = self.next_data_line()?;
            let (var, coef) = parse_var_coef(line)?;
            linear.push((var, coef));
        }
        let nonlin = self.parse_expr()?;
        // Build `nonlin + sum coef_i * v_{var_i}`. Linear terms can
        // reference earlier CSEs as well as plain variables.
        let mut combined = nonlin;
        for (var, coef) in linear {
            let v_expr = self.var_or_cse(var)?;
            let term = if coef == 1.0 {
                v_expr
            } else {
                Expr::Binary(BinOp::Mul, Box::new(Expr::Const(coef)), Box::new(v_expr))
            };
            combined = Expr::Binary(BinOp::Add, Box::new(combined), Box::new(term));
        }
        if cse_idx < self.n {
            return Err(format!("V{cse_idx} below n={}", self.n));
        }
        let local = cse_idx - self.n;
        if local != self.cses.len() {
            return Err(format!(
                "V-segment index V{cse_idx} out of order; expected V{}",
                self.n + self.cses.len()
            ));
        }
        // What a reference to this body would mean to the streaming
        // recognizer, answered once here rather than per reference. The
        // `V` tree exists at this point, so these are the *same* functions
        // `NlTnlp` applies to a whole row — the parse-time recognizer never
        // gets a second opinion about a CSE.
        if self.quad_enabled {
            self.cse_quad
                .push(crate::nl_quadratic::recognize_expr(&combined));
            self.cse_sum_ok
                .push(crate::nl_quadratic::is_expanded_quadratic(&combined));
            self.cse_mono_ok
                .push(crate::nl_quadratic::is_monomial_expr(&combined));
            let mut vars: BTreeSet<usize> = BTreeSet::new();
            collect_vars(&combined, &mut vars);
            self.cse_vars
                .push(vars.into_iter().map(|v| v as u32).collect());
            self.cse_depth.push(expr_tree_depth(&combined));
        }
        self.cses.push(Arc::new(combined));
        Ok(())
    }
}

/// Combine the operands of one recognized operator, mirroring
/// [`crate::nl_quadratic::recognize_expr`]'s `Apply` arm term for term.
///
/// Every difference from that function is a difference in the coefficients
/// this parser stores, so there is nothing here that is "equivalent but
/// tidier": the division scales by the reciprocal because that is what the
/// tree walk does, and the sumlist folds back to front for the same reason.
/// Nesting depth of a `V`-segment body, leaf = 1, a `Cse` reference
/// counting one level above its body — the convention `pounce-py`'s
/// `expr_depth` uses, since that is the guard the answer feeds.
///
/// Recursive, and safe to be: this only ever runs on a tree
/// [`Parser::parse_expr`] has just built *recursively* on this same stack,
/// so a frame that fits the parser fits this.
fn expr_tree_depth(e: &Expr) -> u32 {
    let deepest = |kids: &mut dyn Iterator<Item = &Expr>| {
        kids.fold(0u32, |acc, k| acc.max(expr_tree_depth(k)))
    };
    1 + match e {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            deepest(&mut [&**a, &**b].into_iter())
        }
        Expr::Unary(_, a) | Expr::Not(a) => expr_tree_depth(a),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => deepest(&mut args.iter()),
        Expr::Cond { cond, then_, else_ } => {
            deepest(&mut [&**cond, &**then_, &**else_].into_iter())
        }
        Expr::Funcall { args, .. } => deepest(&mut args.iter().filter_map(|a| match a {
            FuncallArg::Real(inner) => Some(inner),
            FuncallArg::Str(_) => None,
        })),
        Expr::Cse(body) => expr_tree_depth(body),
    }
}

fn apply_quad_op(op: QOp, vals: &mut Vec<Quad2>) -> Option<Quad2> {
    let pop2 = |vals: &mut Vec<Quad2>| -> Option<(Quad2, Quad2)> {
        let b = vals.pop()?;
        let a = vals.pop()?;
        Some((a, b))
    };
    Some(match op {
        QOp::Sum(n) => {
            let at = vals.len().checked_sub(n)?;
            let mut acc = Quad2::default();
            // Operands arrive in file order, so `drain` already yields them
            // front to back — the order `recognize_expr` folds in, and the
            // order the AD tape sums in. Do not reverse this: floating-point
            // addition is not associative, and on repeated monomials the two
            // orders do not agree bit for bit.
            for p in vals.drain(at..) {
                acc = Quad2::add(acc, p);
            }
            acc
        }
        QOp::Neg => vals.pop()?.neg(),
        QOp::Add => {
            let (a, b) = pop2(vals)?;
            Quad2::add(a, b)
        }
        QOp::Sub => {
            let (a, b) = pop2(vals)?;
            Quad2::add(a, b.neg())
        }
        QOp::Mul => {
            let (a, b) = pop2(vals)?;
            a.mul(&b)?
        }
        QOp::Div => {
            let (a, b) = pop2(vals)?;
            let d = b.as_constant()?;
            if d == 0.0 {
                return None;
            }
            let mut out = a.div_by_constant(d);
            out.absorb_flags(&b);
            out
        }
        QOp::Pow => {
            let (a, b) = pop2(vals)?;
            let exp = b.as_constant()?;
            let mut out = if exp == 0.0 {
                Quad2::of_constant(1.0)
            } else if exp == 1.0 {
                a
            } else if exp == 2.0 {
                a.mul(&a)?
            } else {
                return None;
            };
            // The exponent is read out of a form with `as_constant`, which
            // leaves its flags behind; `Div` above does the same.
            out.absorb_flags(&b);
            out
        }
        // `o82` is `Pow(base, Const(2.0))` with the exponent left implicit,
        // so it takes the `exp == 2.0` branch above.
        QOp::Square => {
            let a = vals.pop()?;
            a.mul(&a)?
        }
    })
}

/// Re-parse one recognized body from the bytes it was recognized from.
///
/// The point is that this is the *same* function the original parse ran, on
/// the *same* bytes, with the same `n` and the same `V`-segment bodies — so
/// the tree it returns is the tree that parse would have built, down to the
/// `Arc` identities that `HybridTape::build_multi` keys CSE sharing on.
/// Deriving an equivalent tree from the stored coefficients instead would
/// be a different tree, evaluated by a different tape, and this phase would
/// stop being invisible from outside.
fn parse_body_fragment(txt: &str, n: usize, cses: &[Arc<Expr>]) -> Result<Expr, String> {
    let mut p = Parser::new(txt, false);
    p.n = n;
    p.cses = cses.to_vec();
    p.parse_expr()
}

fn strip_comment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn split_comment(s: &str) -> (&str, &str) {
    match s.find('#') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    }
}

// --------------------------------------------------------------------
// Expression evaluation and gradient (tree walkers, kept for tests).
// The hot paths in `NlTnlp` use the flat `Tape` AD in `nl_tape.rs`
// instead — see `Tape::gradient_seed` / `Tape::hessian_accumulate`.
// --------------------------------------------------------------------

/// Forward-mode value evaluation.
pub fn eval_expr(e: &Expr, x: &[Number]) -> Number {
    match e {
        Expr::Const(c) => *c,
        Expr::Var(i) => x[*i],
        Expr::Binary(op, a, b) => {
            let va = eval_expr(a, x);
            let vb = eval_expr(b, x);
            match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Div => va / vb,
                BinOp::Pow => va.powf(vb),
                BinOp::Atan2 => va.atan2(vb),
                BinOp::CEntropy => crate::nl_tape::centropy(va, vb),
            }
        }
        Expr::Unary(op, a) => {
            let va = eval_expr(a, x);
            match op {
                UnaryOp::Neg => -va,
                UnaryOp::Sqrt => va.sqrt(),
                UnaryOp::Log => va.ln(),
                UnaryOp::Log10 => va.log10(),
                UnaryOp::Exp => va.exp(),
                UnaryOp::Abs => va.abs(),
                UnaryOp::Sin => va.sin(),
                UnaryOp::Cos => va.cos(),
                UnaryOp::Tan => va.tan(),
                UnaryOp::Atan => va.atan(),
                UnaryOp::Acos => va.acos(),
                UnaryOp::Sinh => va.sinh(),
                UnaryOp::Cosh => va.cosh(),
                UnaryOp::Tanh => va.tanh(),
                UnaryOp::Asin => va.asin(),
                UnaryOp::Acosh => va.acosh(),
                UnaryOp::Asinh => va.asinh(),
                UnaryOp::Atanh => va.atanh(),
                UnaryOp::Erf => crate::nl_tape::erf(va),
                UnaryOp::XLogX => crate::nl_tape::xlogx(va),
            }
        }
        Expr::Sum(args) => args.iter().map(|a| eval_expr(a, x)).sum(),
        Expr::MinList(args) => args
            .iter()
            .map(|a| eval_expr(a, x))
            .fold(Number::INFINITY, Number::min),
        Expr::MaxList(args) => args
            .iter()
            .map(|a| eval_expr(a, x))
            .fold(Number::NEG_INFINITY, Number::max),
        Expr::Compare(op, a, b) => {
            let va = eval_expr(a, x);
            let vb = eval_expr(b, x);
            let truth = match op {
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Eq => va == vb,
                CmpOp::Ge => va >= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ne => va != vb,
            };
            if truth { 1.0 } else { 0.0 }
        }
        Expr::And(a, b) => {
            if eval_expr(a, x) != 0.0 && eval_expr(b, x) != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Expr::Or(a, b) => {
            if eval_expr(a, x) != 0.0 || eval_expr(b, x) != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Expr::Not(a) => {
            if eval_expr(a, x) == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Expr::Cond { cond, then_, else_ } => {
            if eval_expr(cond, x) != 0.0 {
                eval_expr(then_, x)
            } else {
                eval_expr(else_, x)
            }
        }
        Expr::Cse(body) => eval_expr(body, x),
        Expr::Funcall { .. } => panic!(
            "eval_expr: AMPL imported function called without an external resolver; \
             evaluate through the tape AD path (Tape::build_with_externals) instead"
        ),
    }
}

/// Index of the active operand of an n-ary min (`want_min = true`) or
/// max (`want_min = false`) list at point `x`: the smallest / largest
/// value, with ties resolved to the first such operand (the
/// conventional subgradient choice). Returns `None` for an empty list.
fn argmin_argmax(args: &[Expr], x: &[Number], want_min: bool) -> Option<usize> {
    let mut best: Option<(usize, Number)> = None;
    for (i, a) in args.iter().enumerate() {
        let v = eval_expr(a, x);
        match best {
            None => best = Some((i, v)),
            Some((_, bv)) => {
                // Strict comparison keeps the FIRST extremal operand on
                // ties, matching the subgradient convention used by Abs
                // and Select elsewhere in the tape.
                if (want_min && v < bv) || (!want_min && v > bv) {
                    best = Some((i, v));
                }
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Reverse-mode gradient: accumulates `seed * d(expr)/dx_i` into `grad`.
pub fn grad_expr(e: &Expr, x: &[Number], seed: Number, grad: &mut [Number]) {
    match e {
        Expr::Const(_) => {}
        Expr::Var(i) => grad[*i] += seed,
        Expr::Binary(op, a, b) => {
            let va = eval_expr(a, x);
            let vb = eval_expr(b, x);
            match op {
                BinOp::Add => {
                    grad_expr(a, x, seed, grad);
                    grad_expr(b, x, seed, grad);
                }
                BinOp::Sub => {
                    grad_expr(a, x, seed, grad);
                    grad_expr(b, x, -seed, grad);
                }
                BinOp::Mul => {
                    grad_expr(a, x, seed * vb, grad);
                    grad_expr(b, x, seed * va, grad);
                }
                BinOp::Div => {
                    grad_expr(a, x, seed / vb, grad);
                    grad_expr(b, x, -seed * va / (vb * vb), grad);
                }
                BinOp::Pow => {
                    // d/da: b * a^(b-1)
                    let dpa = vb * va.powf(vb - 1.0);
                    grad_expr(a, x, seed * dpa, grad);
                    // d/db: a^b * ln(a) (only valid for a>0; simple branch)
                    if va > 0.0 {
                        let dpb = va.powf(vb) * va.ln();
                        grad_expr(b, x, seed * dpb, grad);
                    }
                }
                BinOp::Atan2 => {
                    // atan2(y=a, x=b): d/dy = x/(x²+y²), d/dx = -y/(x²+y²)
                    let d = va * va + vb * vb;
                    grad_expr(a, x, seed * vb / d, grad);
                    grad_expr(b, x, -seed * va / d, grad);
                }
                BinOp::CEntropy => {
                    grad_expr(a, x, seed * crate::nl_tape::centropy_da(va, vb), grad);
                    grad_expr(b, x, seed * crate::nl_tape::centropy_db(va, vb), grad);
                }
            }
        }
        Expr::Unary(op, a) => {
            let va = eval_expr(a, x);
            let d = match op {
                UnaryOp::Neg => -1.0,
                UnaryOp::Sqrt => 0.5 / va.sqrt(),
                UnaryOp::Log => 1.0 / va,
                UnaryOp::Log10 => 1.0 / (va * std::f64::consts::LN_10),
                UnaryOp::Exp => va.exp(),
                UnaryOp::Abs => {
                    if va > 0.0 {
                        1.0
                    } else if va < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                UnaryOp::Sin => va.cos(),
                UnaryOp::Cos => -va.sin(),
                UnaryOp::Tan => {
                    let t = va.tan();
                    1.0 + t * t
                }
                UnaryOp::Atan => 1.0 / (1.0 + va * va),
                UnaryOp::Acos => -1.0 / (1.0 - va * va).sqrt(),
                UnaryOp::Sinh => va.cosh(),
                UnaryOp::Cosh => va.sinh(),
                UnaryOp::Tanh => {
                    let t = va.tanh();
                    1.0 - t * t
                }
                UnaryOp::Asin => 1.0 / (1.0 - va * va).sqrt(),
                UnaryOp::Acosh => 1.0 / (va * va - 1.0).sqrt(),
                UnaryOp::Asinh => 1.0 / (va * va + 1.0).sqrt(),
                UnaryOp::Atanh => 1.0 / (1.0 - va * va),
                UnaryOp::Erf => crate::nl_tape::erf_d1(va),
                UnaryOp::XLogX => crate::nl_tape::xlogx_d1(va),
            };
            grad_expr(a, x, seed * d, grad);
        }
        Expr::Sum(args) => {
            for arg in args {
                grad_expr(arg, x, seed, grad);
            }
        }
        // min/max are piecewise linear: the seed flows only through the
        // currently-active (smallest / largest) operand — a subgradient.
        // Ties resolve to the first such operand. Empty list: no operand,
        // no derivative (matches the ±inf eval fold).
        Expr::MinList(args) => {
            if let Some(k) = argmin_argmax(args, x, true) {
                grad_expr(&args[k], x, seed, grad);
            }
        }
        Expr::MaxList(args) => {
            if let Some(k) = argmin_argmax(args, x, false) {
                grad_expr(&args[k], x, seed, grad);
            }
        }
        // Comparisons and logical connectives are piecewise constant:
        // zero derivative, so no seed propagates into their operands.
        Expr::Compare(_, _, _) | Expr::And(_, _) | Expr::Or(_, _) | Expr::Not(_) => {}
        // if-then-else: differentiate only the active branch. The
        // branch-switch discontinuity contributes no derivative.
        Expr::Cond { cond, then_, else_ } => {
            if eval_expr(cond, x) != 0.0 {
                grad_expr(then_, x, seed, grad);
            } else {
                grad_expr(else_, x, seed, grad);
            }
        }
        Expr::Cse(body) => grad_expr(body, x, seed, grad),
        Expr::Funcall { .. } => {
            panic!("grad_expr: AMPL imported function called without an external resolver")
        }
    }
}

/// Walk `e` and insert every `Var(i)` index into `out`.
///
/// Shared `Cse` bodies are visited once per call, memoized on `Arc` pointer
/// identity. Without that this is Θ(2^depth) on a DAG that shares
/// subexpressions — each reference re-walks the whole body — and presolve
/// calls this on every solve (`get_variables_linearity`). Skipping a
/// repeat visit cannot change the answer: `out` is a set, and a second
/// walk of the same body inserts exactly the indices the first already did.
pub fn collect_vars(e: &Expr, out: &mut BTreeSet<usize>) {
    // `HashSet::new` does not allocate until the first insert, so an
    // expression with no CSEs pays nothing for the memo.
    let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
    collect_vars_memo(e, out, &mut seen);
}

fn collect_vars_memo(
    e: &Expr,
    out: &mut BTreeSet<usize>,
    seen: &mut std::collections::HashSet<*const Expr>,
) {
    match e {
        Expr::Const(_) => {}
        Expr::Var(i) => {
            out.insert(*i);
        }
        Expr::Binary(_, a, b) => {
            collect_vars_memo(a, out, seen);
            collect_vars_memo(b, out, seen);
        }
        Expr::Unary(_, a) => collect_vars_memo(a, out, seen),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                collect_vars_memo(a, out, seen);
            }
        }
        // Collect from every child, including the condition: even
        // though the comparison/branch-test contributes no derivative,
        // the variables it reads are genuinely "used" by the problem,
        // and being conservative here only ever adds structural zeros
        // to the Jacobian/Hessian (never drops a real nonzero).
        Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            collect_vars_memo(a, out, seen);
            collect_vars_memo(b, out, seen);
        }
        Expr::Not(a) => collect_vars_memo(a, out, seen),
        Expr::Cond { cond, then_, else_ } => {
            collect_vars_memo(cond, out, seen);
            collect_vars_memo(then_, out, seen);
            collect_vars_memo(else_, out, seen);
        }
        Expr::Cse(body) => {
            if seen.insert(Arc::as_ptr(body)) {
                collect_vars_memo(body, out, seen);
            }
        }
        Expr::Funcall { args, .. } => {
            for a in args {
                if let FuncallArg::Real(e) = a {
                    collect_vars_memo(e, out, seen);
                }
            }
        }
    }
}

// --------------------------------------------------------------------
// TNLP wrapper — backed by `Tape` reverse-mode AD for value, gradient,
// Jacobian, and Hessian. Built once at construction; every solve-time
// callback is a tape sweep, no expression-tree recursion.
// --------------------------------------------------------------------

/// Per-color decoding instruction for `eval_h` Hessian-coloring.
/// After a directional Hessian-vector product `compressed = H · s_c`,
/// the entry at row `row` came uniquely from column `col` (because
/// no two columns of color `c` share any nonzero row), so we
/// scatter `compressed[row]` into `values[hess_idx]`.
#[derive(Debug, Clone)]
struct ColorWrite {
    row: u32,
    hess_idx: u32,
}

/// Constraint-block [`HybridTape`]: one local op list per summand plus a
/// **shared prelude** holding every CSE body referenced by two or more
/// summands, evaluated once per sweep.
///
/// `con_tapes` builds an independent flat `Tape` per summand, so a `.nl`
/// defined variable (`V` segment) referenced from many rows is re-emitted —
/// and re-evaluated — once per reference. That is invisible on most models
/// but quadratic-ish in the wrong shape: on Mittelmann's `robot_a`
/// (n = 1001, m = 52013, 12003 defined variables each feeding 13 rows) the
/// flat tapes total 3.6M ops per `eval_g` against 894k for the shared
/// prelude — 4.0x the arithmetic, paid ~10x per iteration inside the line
/// search. See pounce#476.
///
/// `eval_g` reads this unconditionally; `eval_jac_g` and `eval_h` read it
/// above their respective op-ratio gates ([`HYBRID_JAC_MIN_OP_RATIO`],
/// [`HYBRID_HESS_MIN_OP_RATIO`]) — the hybrid traversal carries per-op
/// overhead the flat tapes do not, so each derivative order has to earn
/// its switch.
#[derive(Debug, Clone)]
struct ConHybrid {
    tape: HybridTape,
    /// `row_start[i]..row_start[i + 1]` are the summands of constraint `i`.
    /// Length `m + 1`.
    row_start: Vec<usize>,
    /// Prelude forward values, sized to `tape.n_prelude_ops()`.
    prelude_vals: Vec<f64>,
    /// Per-summand local forward values, sized to `tape.max_summand_ops()`.
    local_vals: Vec<f64>,
    /// Reverse-mode adjoint arenas for `eval_jac_g`, sized like the two
    /// value arenas above. `gradient_summand` zeroes only the slots a
    /// summand actually reaches, so these are allocated once and reused.
    local_adj: Vec<f64>,
    prelude_adj: Vec<f64>,
    /// Whether `eval_jac_g` should take the shared-CSE path too, or stay
    /// on the flat per-summand tapes. See [`HYBRID_JAC_MIN_OP_RATIO`] —
    /// unlike `eval_g`, the hybrid Jacobian is not a free win.
    use_for_jac: bool,
    /// Whether `eval_h` routes the constraint block through the shared
    /// prelude (issue #557). See [`HYBRID_HESS_MIN_OP_RATIO`].
    use_for_hess: bool,
    // ---- eval_h (shared-CSE Hessian, issue #557) state. Populated in
    // `try_new` after the Hessian coloring exists; cheap enough (a few
    // index tables plus one f64 per local op) to build whenever the
    // hybrid tape is, so tests can flip `use_for_hess` on directly. ----
    /// Forward values of every summand, packed at
    /// `local_off[si]..local_off[si + 1]`. One forward pass per `eval_h`
    /// fills it; every color then reuses the values, mirroring the flat
    /// path's forward-once-per-tape structure.
    local_vals_all: Vec<f64>,
    /// Prefix offsets into `local_vals_all`, length `n_summands + 1`.
    local_off: Vec<usize>,
    /// Constraint row of each summand (the inverse of `row_start`), for
    /// the `λ[row]` weight lookup.
    summand_row: Vec<u32>,
    /// Per color: the summands whose variables fall in that color — the
    /// hybrid analogue of `con_tape_colors`, inverted so `eval_h` walks
    /// exactly the live (color, summand) pairs.
    hess_color_summands: Vec<Vec<u32>>,
    /// Per color: the prelude slots that color's summands actually reach,
    /// ascending — `hess_color_reach[hess_color_reach_off[c]..off[c + 1]]`.
    /// Both prelude sweeps run once per color, so walking the whole
    /// prelude each time would cost `n_colors × |prelude|` where the
    /// op-ratio gate assumes `|prelude|`; iterating the union of the
    /// color's `prelude_reach` sets makes the cost proportional to what
    /// is used, so the gate does not need an `n_colors` term. Stored as
    /// a flat `u32` CSR rather than `Vec<Vec<_>>`: the total is
    /// `Σ_c |reach_c|`, which is proportional to the work it drives, and
    /// this struct is the one #552 made O(n²) by holding per-color dense
    /// arrays.
    hess_color_reach: Vec<u32>,
    hess_color_reach_off: Vec<usize>,
    /// Per-color prelude tangent, sized to `tape.n_prelude_ops()`.
    prelude_dot: Vec<f64>,
    /// First-/second-order prelude adjoint accumulators. NOT shared
    /// with `eval_jac_g`'s `prelude_adj`: these two carry an all-zero-
    /// between-colors invariant (`prelude_reverse_directional`'s
    /// consume-and-zero contract), while `gradient_summand` zeroes only
    /// the slots it is about to use and leaves them dirty afterwards —
    /// sharing the buffer would seed a later `eval_h` with a stale
    /// Jacobian adjoint.
    hess_prelude_adj: Vec<f64>,
    prelude_adj_dot: Vec<f64>,
    /// Local tangent / second-order adjoint arenas, sized to
    /// `tape.max_summand_ops()`.
    local_dot: Vec<f64>,
    local_adj_dot: Vec<f64>,
}

/// Flat-to-shared op-count ratio above which `eval_jac_g` switches to the
/// shared-CSE prelude.
///
/// `eval_g` takes the hybrid path unconditionally because it only needs
/// *values*: the prelude is swept once for the whole constraint block and
/// the saving is the full op-count ratio. The Jacobian is different. Each
/// row needs its own gradient, so only the forward sweep can be shared —
/// the reverse sweep still walks each summand's `prelude_reach`
/// separately, and it pays a per-op cost the flat tape does not: a nested
/// `SummandOp` dispatch and an indirected walk over a reach list instead
/// of a straight loop over a contiguous `Vec<TapeOp>`.
///
/// So the hybrid Jacobian wins only when the shared bodies are large
/// enough for the halved forward sweep to outweigh that overhead.
/// Measured on chain models at CSE redundancy 40, varying the body size
/// (`eval_jac_g`, flat → hybrid):
///
/// | op ratio | 1.94 | 2.20 | 2.84 | 3.53 | 4.20 | 5.16 | 6.35 | 8.00 |
/// |---|---|---|---|---|---|---|---|---|
/// | speedup | 0.77× | 0.63× | 0.88× | 1.21× | 1.21× | 1.18× | 1.50× | 1.32× |
///
/// The crossover sits near 3; this gate is set at 4 to keep a margin, so
/// a model that does not clearly benefit stays on the flat path. For
/// reference `robot_a` (#476) measures 4.03×.
const HYBRID_JAC_MIN_OP_RATIO: f64 = 4.0;

/// Flat-to-shared op-count ratio above which `eval_h` routes the constraint
/// block through the shared-CSE prelude (issue #557).
///
/// The Hessian shares **both** second-order sweeps of the prelude, not just
/// the forward one: the coloring hands every summand of a color the same
/// seed vector, so the prelude forward tangent runs once per color, and —
/// because reverse-over-tangent is linear in its adjoint seeds — the
/// `λ_k`-weighted boundary adjoints of all summands accumulate into one
/// unit-weight prelude reverse sweep per color. That is why its crossover
/// sits *below* the Jacobian's ([`HYBRID_JAC_MIN_OP_RATIO`], set at 4): the
/// Jacobian can only share the forward half, and per-row gradients forbid
/// batching its reverse sweeps at all.
///
/// Measured on chain models at CSE redundancy 40 (m = 20,000, 500 shared
/// bodies), varying the body size — the same protocol as the Jacobian
/// gate's table (`eval_h`, flat → hybrid). **Median of 5 interleaved
/// flat/hybrid pairs per point**, with the observed range, because
/// single runs on a shared machine are not reproducible to the precision
/// a threshold decision needs — one sample below spans 0.36×–1.14× at a
/// single ratio:
///
/// | op ratio | 1.94 | 2.54 | 3.12 | 3.69 | 4.24 | 5.29 | 6.76 | 8.53 |
/// |---|---|---|---|---|---|---|---|---|
/// | median speedup | 1.00× | 1.04× | 1.18× | 1.23× | 1.31× | 1.44× | 1.36× | 1.49× |
/// | min–max | 0.91–1.11 | 0.36–1.14 | 1.10–1.33 | 1.16–1.30 | 1.23–1.32 | 1.19–1.54 | 1.27–1.50 | 1.36–1.65 |
///
/// The gate sits at 3.0: that is the lowest ratio where **every** sample
/// wins (by ≥ 10%), whereas at 1.94 and 2.54 the median is within noise of
/// break-even and individual runs lose. Setting it there costs only a
/// marginal forgone gain — below the gate `eval_h` stays on the flat path
/// bit-identically, so a gate placed too high is merely conservative while
/// one placed too low risks a real regression. For reference `robot_a`
/// (#476) measures 4.03×.
const HYBRID_HESS_MIN_OP_RATIO: f64 = 3.0;

// `Clone` supports the batched-solve path (pounce#126): one parsed
// model is cloned per batch instance (tapes are flat `Vec`s of ops, so
// the clone is cheap relative to a solve) and each clone gets its own
// bound / starting-point overrides via [`NlTnlp::variant`].
#[derive(Debug, Clone)]
pub struct NlTnlp {
    prob: NlProblem,
    /// Per-summand objective tapes (one `Tape` per top-level
    /// summand after `split_top_sums`).
    obj_tapes: Vec<Tape>,
    /// Per-constraint, per-summand tapes. Length `m`; row `i` holds
    /// one `Tape` per summand of constraint `i`.
    con_tapes: Vec<Vec<Tape>>,
    /// Constraint-block tape with a **shared** CSE prelude, used by
    /// `eval_g` when the model benefits (see [`ConHybrid`]). `None` keeps
    /// `eval_g` on the per-summand `con_tapes` above.
    con_hybrid: Option<ConHybrid>,
    /// The degree-≤2 objective and rows, evaluated from their constant
    /// matrices instead of from a tape (gh #588, Q4). A row with a form here
    /// has an **empty** `con_tapes` entry, so every loop over the tapes
    /// naturally contributes nothing for it and only the sites that consult
    /// `quad` add it back: `eval_f`, `eval_grad_f`, `eval_g`, `eval_jac_g`,
    /// `eval_h`, and `hessian_vector_products` — the last of which the design
    /// note's list of five omitted, and which is the one where an omission
    /// would be silent. Empty for a model with nothing recognized, in which
    /// case everything below behaves bit for bit as it did before this
    /// existed.
    quad: QuadraticStructure,
    /// Which entries of `h_irow`/`h_jcol` some *tape* can contribute to.
    /// Empty when `quad` is — the pattern is then all tape. The coloring is
    /// built over this subset alone, which is what stops one dense quadratic
    /// block from forcing the color count to `n` for the benefit of tapes
    /// that no longer exist.
    h_tape_mask: Vec<bool>,
    /// Lower-triangle Hessian sparsity (row >= col), one entry per
    /// structurally nonzero second derivative in the Lagrangian.
    h_irow: Vec<i32>,
    h_jcol: Vec<i32>,
    /// Per-row sorted variable indices for the constraint Jacobian.
    jac_cols: Vec<Vec<usize>>,
    jac_nnz: usize,
    /// Per-color seed vector: `seeds[c][k] = 1.0` iff variable `k`
    /// is in color `c`, else `0.0`. Each color is a set of
    /// variables whose Hessian columns have pairwise-disjoint
    /// nonzero rows; one directional H·s product per color
    /// recovers all those columns simultaneously. Dense for
    /// O(1) lookup in the per-op forward tangent.
    seeds: Vec<Vec<f64>>,
    /// Per-color decoding table: for each `(row, hess_idx)` entry,
    /// scatter `compressed_c[row] -> values[hess_idx]` after the
    /// per-color directional product.
    decoding: Vec<Vec<ColorWrite>>,
    /// For each objective tape: the distinct colors of vars it
    /// references. Lets us skip tape × color pairs where the tape
    /// has zero overlap with the color's seed.
    obj_tape_colors: Vec<Vec<u32>>,
    /// Same as `obj_tape_colors` but per constraint × summand.
    con_tape_colors: Vec<Vec<Vec<u32>>>,
    /// Color of each variable's Hessian column, `u32::MAX` for a column
    /// that needs no pass of its own. Kept so
    /// [`NlTnlp::veto_ill_conditioned_peels`] can find a peeled column's
    /// pass in `compressed`.
    var_color: Vec<u32>,
    /// Columns peeled out of the conflict structure and given singleton
    /// colors. Bounded by `MAX_PEELED_COLS`, usually empty.
    peeled_cols: Vec<u32>,
    final_x: Option<Vec<Number>>,
    final_obj: Number,
    /// Converged constraint multipliers (length `m`, original `.nl` row
    /// order, user convention), captured from the same `finalize_solution`
    /// call as `final_x`. Kept so a frontend can write the `.sol` dual
    /// block without re-deriving it from the algorithm's internal `y_c` /
    /// `y_d` split and scaling.
    final_lambda: Option<Vec<Number>>,
    /// Converged bound multipliers (length `n` each, Ipopt's internal
    /// convention `z_l, z_u >= 0`), captured with `final_x`. Written as the
    /// `ipopt_zL_out` / `ipopt_zU_out` `.sol` suffixes, which is what Pyomo
    /// reads for reduced costs.
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
    /// Per-row Jacobian accumulator (length n).
    scratch_row_grad: Vec<f64>,
    /// Scratch buffers for `Tape::hessian_directional` (each sized
    /// to `max_tape_n`).
    vals_scratch: Vec<f64>,
    dot_scratch: Vec<f64>,
    adj_scratch: Vec<f64>,
    adj_dot_scratch: Vec<f64>,
    /// Per-color compressed Hessian-vector results, sized to
    /// `prob.n`. Reused across `eval_h` calls but allocated once.
    compressed: Vec<Vec<f64>>,
    /// Per-direction "carries signal" mask for `hessian_vector_products`,
    /// kept here rather than allocated per call. This crate holds an
    /// explicit no-per-call-allocation line on the tape sweeps (see
    /// `tests/tape_gradient_no_alloc.rs`), and the headline Newton-Krylov
    /// use is `k = 1`, where a fresh `Vec` would be pure overhead on every
    /// Krylov iteration.
    hvp_live: Vec<bool>,
    /// Model-derived scaling factors (gh #703), computed on demand by
    /// [`NlTnlp::enable_curvature_scaling`] and served through
    /// [`TNLP::get_scaling_parameters`]. `None` until asked for, which is
    /// the state every solve that does not select `curvature-based` stays
    /// in — computing them costs a pass over every stored Hessian entry
    /// and nothing else reads them.
    curvature_scaling: Option<crate::nl_scaling::CurvatureScaling>,
}

// ---------------------------------------------------------------------
// Human-readable equation rendering (`print equation` in the debugger).
//
// Turns a parsed constraint back into infix text using the model's
// variable / constraint names, so the debugger can show the actual
// equation a user wrote — `T_reactor*flow - 300 = 0` — instead of a
// bare row index. This is the "print the specific equation, with
// names" capability Lee et al. (2024, <https://doi.org/10.69997/sct.147875>)
// argue makes equation-oriented model diagnostics actionable.
//
// The renderer is intentionally separate from the evaluation `Tape`:
// tapes are lossy for display (CSEs flattened, externals opaque),
// whereas the `Expr` DAG is the faithful source the `.nl` parser built.
// ---------------------------------------------------------------------

/// Binding strength for parenthesization. Higher binds tighter.
const P_ADD: u8 = 10;
const P_MUL: u8 = 20;
const P_NEG: u8 = 30;
const P_POW: u8 = 40;
const P_ATOM: u8 = 100;

/// Format a numeric literal compactly: integers without a trailing `.0`,
/// everything else via the shortest round-tripping `f64` form.
fn fmt_num(x: Number) -> String {
    if x.is_finite() && x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// Display label for variable `i`: its `.col` name when present, else
/// `x[i]`.
fn var_label(i: usize, var_names: &[String]) -> String {
    match var_names.get(i) {
        Some(s) if !s.is_empty() => s.clone(),
        _ => format!("x[{i}]"),
    }
}

/// Precedence of an expression's top operator (for child wrapping).
fn expr_prec(e: &Expr) -> u8 {
    match e {
        Expr::Binary(BinOp::Add, ..) | Expr::Binary(BinOp::Sub, ..) | Expr::Sum(_) => P_ADD,
        Expr::Binary(BinOp::Mul, ..) | Expr::Binary(BinOp::Div, ..) => P_MUL,
        Expr::Unary(UnaryOp::Neg, _) => P_NEG,
        Expr::Binary(BinOp::Pow, ..) => P_POW,
        Expr::Cse(inner) => expr_prec(inner),
        // Everything else renders as an atom / `f(...)` form.
        _ => P_ATOM,
    }
}

/// Render an expression as infix text, using `var_names` for variable
/// labels where available (`x[i]` otherwise).
///
/// The debugger reaches the renderer through the constraint/objective
/// walkers; this is the bare entry point for a caller that holds an [`Expr`]
/// directly — notably the Python `NlExpr.__repr__` (issue #469), where being
/// able to *see* the expression you just built is most of the debugging
/// story.
///
/// `Cse` bodies are inlined at every occurrence, so the output of a
/// heavily-shared DAG can be far larger than the DAG itself. Callers
/// rendering user-built expressions should bound the input first.
pub fn render_expression(e: &Expr, var_names: &[String]) -> String {
    render_expr(e, var_names, &[])
}

/// Render `e`, wrapping in parentheses iff its precedence is looser than
/// `min_prec`.
fn render_prec(e: &Expr, min_prec: u8, vn: &[String], funcs: &[ImportedFunc]) -> String {
    let s = render_expr(e, vn, funcs);
    if expr_prec(e) < min_prec {
        format!("({s})")
    } else {
        s
    }
}

fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Log => "log",
        UnaryOp::Exp => "exp",
        UnaryOp::Abs => "abs",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Log10 => "log10",
        UnaryOp::Tan => "tan",
        UnaryOp::Atan => "atan",
        UnaryOp::Acos => "acos",
        UnaryOp::Sinh => "sinh",
        UnaryOp::Cosh => "cosh",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Asin => "asin",
        UnaryOp::Acosh => "acosh",
        UnaryOp::Asinh => "asinh",
        UnaryOp::Atanh => "atanh",
        UnaryOp::Erf => "erf",
        // Spelled as the operation, not as GAMS `entropy` (which is -x·ln x):
        // the rendered text is read by humans debugging a model and must not
        // imply a sign the op does not have.
        UnaryOp::XLogX => "xlogx",
    }
}

fn cmp_sym(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ge => ">=",
        CmpOp::Gt => ">",
        CmpOp::Ne => "!=",
    }
}

/// Append an additive sub-term with a tidy sign: a rendered term that
/// begins with `-` is folded into a ` - ` separator, so `a + -b` reads as
/// `a - b`. The identity `a + (-b …) = a - b …` keeps this exact even when
/// the term is itself a sum. The first term is emitted verbatim.
fn push_additive(out: &mut String, rendered: &str, first: bool) {
    if first {
        out.push_str(rendered);
    } else if let Some(rest) = rendered.strip_prefix('-') {
        out.push_str(" - ");
        out.push_str(rest);
    } else {
        out.push_str(" + ");
        out.push_str(rendered);
    }
}

/// Render an [`Expr`] DAG to infix text using model names.
fn render_expr(e: &Expr, vn: &[String], funcs: &[ImportedFunc]) -> String {
    match e {
        Expr::Const(c) => fmt_num(*c),
        Expr::Var(i) => var_label(*i, vn),
        Expr::Binary(op, l, r) => match op {
            BinOp::Add => {
                let mut s = render_prec(l, P_ADD, vn, funcs);
                push_additive(&mut s, &render_prec(r, P_ADD, vn, funcs), false);
                s
            }
            // Right operand at P_ADD+1 so `a - (b - c)` keeps its parens.
            BinOp::Sub => format!(
                "{} - {}",
                render_prec(l, P_ADD, vn, funcs),
                render_prec(r, P_ADD + 1, vn, funcs)
            ),
            BinOp::Mul => format!(
                "{}*{}",
                render_prec(l, P_MUL, vn, funcs),
                render_prec(r, P_MUL, vn, funcs)
            ),
            BinOp::Div => format!(
                "{}/{}",
                render_prec(l, P_MUL, vn, funcs),
                render_prec(r, P_MUL + 1, vn, funcs)
            ),
            // Pow is right-associative: tighten the left operand instead.
            BinOp::Pow => format!(
                "{}^{}",
                render_prec(l, P_POW + 1, vn, funcs),
                render_prec(r, P_POW, vn, funcs)
            ),
            BinOp::Atan2 => format!(
                "atan2({}, {})",
                render_expr(l, vn, funcs),
                render_expr(r, vn, funcs)
            ),
            BinOp::CEntropy => format!(
                "centropy({}, {})",
                render_expr(l, vn, funcs),
                render_expr(r, vn, funcs)
            ),
        },
        Expr::Unary(UnaryOp::Neg, a) => format!("-{}", render_prec(a, P_NEG, vn, funcs)),
        Expr::Unary(op, a) => format!("{}({})", unary_name(*op), render_expr(a, vn, funcs)),
        Expr::Sum(xs) => {
            if xs.is_empty() {
                "0".to_string()
            } else {
                let mut s = String::new();
                for (k, x) in xs.iter().enumerate() {
                    push_additive(&mut s, &render_prec(x, P_ADD, vn, funcs), k == 0);
                }
                s
            }
        }
        Expr::Cse(inner) => render_expr(inner, vn, funcs),
        Expr::Funcall { id, args } => {
            let name = funcs
                .iter()
                .find(|f| f.id == *id)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("extern#{id}"));
            let parts: Vec<String> = args
                .iter()
                .map(|a| match a {
                    FuncallArg::Real(x) => render_expr(x, vn, funcs),
                    FuncallArg::Str(s) => format!("{s:?}"),
                })
                .collect();
            format!("{name}({})", parts.join(", "))
        }
        Expr::Compare(op, a, b) => format!(
            "({} {} {})",
            render_expr(a, vn, funcs),
            cmp_sym(*op),
            render_expr(b, vn, funcs)
        ),
        Expr::And(a, b) => format!(
            "({} && {})",
            render_expr(a, vn, funcs),
            render_expr(b, vn, funcs)
        ),
        Expr::Or(a, b) => format!(
            "({} || {})",
            render_expr(a, vn, funcs),
            render_expr(b, vn, funcs)
        ),
        Expr::Not(a) => format!("!({})", render_expr(a, vn, funcs)),
        Expr::Cond { cond, then_, else_ } => format!(
            "if({}, {}, {})",
            render_expr(cond, vn, funcs),
            render_expr(then_, vn, funcs),
            render_expr(else_, vn, funcs)
        ),
        Expr::MinList(xs) => format!(
            "min({})",
            xs.iter()
                .map(|x| render_expr(x, vn, funcs))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::MaxList(xs) => format!(
            "max({})",
            xs.iter()
                .map(|x| render_expr(x, vn, funcs))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Render the affine `Σ cᵢ·xᵢ` part with tidy signs (`a - 2*b`, not
/// `a + -2*b`). Returns `""` when there are no linear terms.
fn render_linear(linear: &[(usize, Number)], vn: &[String]) -> String {
    let mut out = String::new();
    // The `.nl` linear part carries an entry for every variable in the
    // row's Jacobian, including a 0 coefficient for variables that appear
    // only *nonlinearly* (they're rendered in the nonlinear part). Skip
    // those zeros so the equation reads as written, not as a sparsity map.
    let mut first = true;
    for (var, coef) in linear {
        if *coef == 0.0 {
            continue;
        }
        let neg = *coef < 0.0;
        let mag = coef.abs();
        let term = if mag == 1.0 {
            var_label(*var, vn)
        } else {
            format!("{}*{}", fmt_num(mag), var_label(*var, vn))
        };
        if first {
            if neg {
                out.push('-');
            }
            out.push_str(&term);
            first = false;
        } else {
            out.push_str(if neg { " - " } else { " + " });
            out.push_str(&term);
        }
    }
    out
}

/// Render the constraint body (linear + nonlinear parts combined).
fn render_body(linear: &[(usize, Number)], nonlinear: &Expr, prob: &NlProblem) -> String {
    let mut s = render_linear(linear, &prob.var_names);
    let nl_is_zero = matches!(nonlinear, Expr::Const(c) if *c == 0.0);
    if !nl_is_zero {
        let nl = render_prec(nonlinear, P_ADD, &prob.var_names, &prob.imported_funcs);
        if s.is_empty() {
            s = nl;
        } else {
            push_additive(&mut s, &nl, false);
        }
    }
    if s.is_empty() {
        s = "0".to_string();
    }
    s
}

/// Render constraint `k` as a full relation, e.g. `mass_in - mass_out = 0`
/// or `0 <= T_reactor <= 500`. Bounds outside ±1e19 are treated as
/// infinite (AMPL's convention), matching [`TNLPAdapter`]'s classifier.
pub fn render_constraint_equation(prob: &NlProblem, k: usize) -> String {
    // Diagnostic path: a recognized body has no tree, so this is one of
    // the places that pays to rebuild one. It runs once per rendered row.
    let body = render_body(&prob.con_linear[k], &prob.con_expr(k), prob);
    let lo = prob.g_l[k];
    let hi = prob.g_u[k];
    const INF: Number = 1.0e19;
    let has_lo = lo > -INF;
    let has_hi = hi < INF;
    match (has_lo, has_hi) {
        (true, true) if lo == hi => format!("{body} = {}", fmt_num(lo)),
        (true, true) => format!("{} <= {body} <= {}", fmt_num(lo), fmt_num(hi)),
        (true, false) => format!("{body} >= {}", fmt_num(lo)),
        (false, true) => format!("{body} <= {}", fmt_num(hi)),
        (false, false) => format!("{body}  (free)"),
    }
}

/// Render every constraint to text, index-aligned to `g` (original `.nl`
/// row order). Used to build the debugger's static equation book.
pub fn render_all_constraint_equations(prob: &NlProblem) -> Vec<String> {
    (0..prob.m)
        .map(|k| render_constraint_equation(prob, k))
        .collect()
}

/// Structural sparsity of the constraint Jacobian as flat 0-based
/// triplets `(irow, jcol)`: one pair per variable that constraint `k`
/// structurally depends on — the union of its linear support and the
/// `Var(i)` indices appearing anywhere in its nonlinear tree
/// ([`collect_vars`]). Sorted and deduplicated within each row.
///
/// This is the input to the debugger's Dulmage–Mendelsohn
/// structural-rank check (`diagnose`), which names the over-determined
/// (candidate redundant / inconsistent) equations and under-determined
/// variables. Naming the dependent rows — rather than reporting
/// "equations 3, 15, …" — is the roadblock Lee et al. (2024) flag for
/// equation-oriented model debugging. See
/// <https://doi.org/10.69997/sct.147875>.
pub fn constraint_jacobian_sparsity(prob: &NlProblem) -> (Vec<Index>, Vec<Index>) {
    let mut irow: Vec<Index> = Vec::new();
    let mut jcol: Vec<Index> = Vec::new();
    let mut support: BTreeSet<usize> = BTreeSet::new();
    for k in 0..prob.m {
        support.clear();
        for &(j, _coef) in &prob.con_linear[k] {
            support.insert(j);
        }
        prob.con_nonlinear[k].collect_vars(&mut support);
        for &j in &support {
            irow.push(k as Index);
            jcol.push(j as Index);
        }
    }
    (irow, jcol)
}

/// Flatten an additive expression tree into independent summand
/// expressions, each of which becomes its own Hessian tape.
///
/// This is the linchpin of the colored-AD Hessian: `eval_h` walks
/// each summand tape once *per color the summand touches*, so the
/// cost is `Σ_summand (tape_len · colors_touched)`. Keeping summands
/// small (few variables → few colors) is what makes a sparse Hessian
/// cheap. A single fused tape spanning all `n` variables, by
/// contrast, is walked once per color → `O(n · tape_len)`, which on a
/// dense `n`-variable objective is `O(n³)` (observed: 47 s on the
/// 1000-var `sensors`, whose objective is `-(Σ 10⁶ pairwise terms)`).
///
/// We therefore descend through the *affine* envelope of the sum, not
/// just `+`/`Sum`:
///
///   * `Neg(x)`            → split `x`, negate each summand
///   * `Sub(l, r)`         → split `l`; split `r`, negate each summand
///   * `c * x` / `x * c`   → split `x`, scale each summand by `c`
///   * `x / c`             → split `x`, scale each summand by `1/c`
///
/// so that an objective like `-(Σ …)` or `0.5·(Σ …)` (the usual
/// least-squares / max-entropy shapes) still decomposes to its leaf
/// terms instead of collapsing into one giant tape. The carried
/// `factor` is materialised onto each leaf only when it differs from
/// `1` (as `Neg` for `-1`, else a `Const·term` multiply), so the math
/// is unchanged and the per-summand op count grows by at most one.
fn split_top_sums(expr: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    fn push_leaf(e: &Expr, factor: f64, out: &mut Vec<Expr>) {
        if factor == 1.0 {
            out.push(e.clone());
        } else if factor == -1.0 {
            out.push(Expr::Unary(UnaryOp::Neg, Box::new(e.clone())));
        } else {
            out.push(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(factor)),
                Box::new(e.clone()),
            ));
        }
    }
    fn go(e: &Expr, factor: f64, out: &mut Vec<Expr>) {
        match e {
            Expr::Sum(terms) => {
                for t in terms {
                    go(t, factor, out);
                }
            }
            Expr::Binary(BinOp::Add, l, r) => {
                go(l, factor, out);
                go(r, factor, out);
            }
            Expr::Binary(BinOp::Sub, l, r) => {
                go(l, factor, out);
                go(r, -factor, out);
            }
            Expr::Unary(UnaryOp::Neg, x) => {
                go(x, -factor, out);
            }
            // Affine scaling: distribute a constant coefficient into
            // the summands so a leading `c·(Σ …)` still splits.
            Expr::Binary(BinOp::Mul, l, r) => match (l.as_ref(), r.as_ref()) {
                (Expr::Const(c), _) => go(r, factor * c, out),
                (_, Expr::Const(c)) => go(l, factor * c, out),
                _ => push_leaf(e, factor, out),
            },
            Expr::Binary(BinOp::Div, l, r) => match r.as_ref() {
                Expr::Const(c) if *c != 0.0 => go(l, factor / c, out),
                _ => push_leaf(e, factor, out),
            },
            _ => push_leaf(e, factor, out),
        }
    }
    go(expr, 1.0, &mut out);
    if out.is_empty() {
        out.push(Expr::Const(0.0));
    }
    out
}

/// Greedy column coloring of a symmetric sparsity pattern stored
/// as lower-triangle pairs.
///
/// Builds the column-intersection graph: columns `c1` and `c2` are
/// adjacent iff there exists a row `r` with `H[r, c1] != 0` and
/// `H[r, c2] != 0`. A distance-1 greedy coloring on this graph
/// satisfies the direct-recovery condition for symmetric Hessians
/// (Coleman-Moré): for any color, the columns it contains have
/// pairwise disjoint row supports, so a single H·s product
/// recovers them all unambiguously.
///
/// Returns `(var_color, n_colors)` where `var_color[k]` is the
/// color assigned to variable `k`, or `u32::MAX` for variables
/// not in any Hessian pair (they contribute nothing and don't
/// need a color).
/// A column is treated as **dense** — and peeled out of the coloring —
/// once its nonzero-row count exceeds `DENSE_COL_FACTOR` times the
/// average, but never below `DENSE_COL_MIN`. Both guards matter: the
/// factor keeps uniformly-dense Hessians (where every column looks like
/// every other) on the plain coloring path, and the absolute floor stops
/// a very sparse average from declaring a 10-entry column "dense".
const DENSE_COL_FACTOR: usize = 16;
const DENSE_COL_MIN: usize = 32;
/// Largest relative error a peeled column may inflict on the smallest
/// entry recovered from its pass before
/// [`NlTnlp::veto_ill_conditioned_peels`] un-peels it.
///
/// The ratio is a worst-case bound — the pass's roundoff floor over the
/// smallest entry read out of it — and it is a *loose* one, by an amount
/// that varies per column: `rocket_12800` bounds at 2e-9 and measures 3e-14
/// against an uncompressed reference, while `orthregd` bounds at 3e-8 and
/// measures 4e-16. The cut is therefore calibrated on the corpus, not
/// derived, and the corpus leaves only a narrow gap to sit in: over the 56
/// models that peel anything, the highest bound on a column that recovers
/// its entries to machine precision is 2.9e-8 (`orthregd`), and the lowest
/// bound on one of `cho_parmest`'s harmful columns is 8.3e-8 — a factor of
/// 2.8 apart, with `cho_parmest`'s worst running to 5e-2.
///
/// 1e-8 sits below both, which deliberately buys correctness with speed:
/// the errors are asymmetric, since a false veto costs one model a coloring
/// (measured: five sub-second `orth*` models pay 2-3.5x, worst case +0.32s)
/// while a missed veto costs a solve its certificate. Five of the 56 take a
/// veto they do not need; none takes a wrong answer. Raising this constant
/// to buy those five back would put the cut inside a 2.8x window measured
/// on two model families, which is not a margin worth trading a certificate
/// for.
const PEEL_MAX_REL_ERR: f64 = 1e-8;
/// Hard cap on how many columns get peeled, applied on top of the
/// pay-for-itself rule in [`select_peeled_cols`].
const MAX_PEELED_COLS: usize = 256;

/// Lower bound on the color count that results from peeling `peeled`.
///
/// Peeling costs one color per peeled column. On what remains, any row
/// with `d` surviving entries makes those `d` columns pairwise
/// conflicting, so the greedy walk needs at least `d` colors. Hence
/// `|peeled| + max surviving row degree` is a lower bound on the total,
/// computable in one O(nnz) pass — no coloring required.
///
/// The bound is what makes the choice decidable at all: the plain
/// coloring cannot be run as a comparison baseline, because on the
/// one-dense-row case the plain walk is itself O(n^2) — precisely the
/// blowup peeling exists to avoid.
fn peel_color_bound(n: usize, lower_pairs: &[(usize, usize)], peeled: &[bool]) -> usize {
    let mut deg = vec![0usize; n];
    for &(i, j) in lower_pairs {
        if peeled[i] || peeled[j] {
            continue;
        }
        deg[j] += 1;
        if i != j {
            deg[i] += 1;
        }
    }
    let n_peeled = peeled.iter().filter(|&&p| p).count();
    n_peeled + deg.iter().copied().max().unwrap_or(0)
}

/// Choose which of the candidate dense columns to actually peel.
///
/// Evaluates [`peel_color_bound`] for peeling nothing and for peeling the
/// top `k` candidates by degree, over a doubling ladder of `k` up to
/// [`MAX_PEELED_COLS`], and keeps the best. Ties go to the smaller `k`,
/// so peeling has to earn its colors.
///
/// **Why not just truncate an over-long candidate list.** Cutting the
/// candidates down to `MAX_PEELED_COLS` is not a damage bound: the
/// columns that miss the cut stay in the conflict structure, so the base
/// color count is untouched and the singleton colors are pure addition.
/// On disjoint 50x50 blocks scattered through a 200k-variable Hessian
/// that colors to `50 + 256` where a plain walk needs 50 — a 6x
/// regression in exactly the quantity peeling exists to reduce. The
/// bound above sees it: peeling `k` of several thousand equal-degree
/// columns leaves the surviving max degree unchanged, so every `k > 0`
/// scores strictly worse than peeling nothing.
///
/// **Why not a simple degree rule.** "Peel only columns denser than some
/// fraction of `n`" would refuse three rows of degree 10,000 in a
/// 200,000-variable model, where peeling three columns takes the
/// coloring from >= 10,000 down to a handful. The win depends on what
/// peeling leaves behind, not on the peeled column's degree alone.
fn select_peeled_cols(
    n: usize,
    lower_pairs: &[(usize, usize)],
    deg: &[usize],
    mut candidates: Vec<usize>,
) -> Vec<usize> {
    if candidates.is_empty() {
        return candidates;
    }
    // Worst offenders first; they remove the most conflict per color spent.
    candidates.sort_unstable_by(|&a, &b| deg[b].cmp(&deg[a]).then(a.cmp(&b)));
    candidates.truncate(MAX_PEELED_COLS);

    let mut mask = vec![false; n];
    let mut best_k = 0usize;
    // Peeling nothing: the bound is just the largest row degree.
    let mut best_bound = peel_color_bound(n, lower_pairs, &mask);

    // Doubling ladder 1, 2, 4, ... capped at the candidate count, so the
    // cost is O(nnz log MAX_PEELED_COLS) rather than O(nnz) per k. The
    // mask only ever gains entries, so each step just marks the new slice.
    let mut marked = 0usize;
    let mut k = 1usize;
    loop {
        let k_now = k.min(candidates.len());
        for &j in &candidates[marked..k_now] {
            mask[j] = true;
        }
        marked = k_now;
        let bound = peel_color_bound(n, lower_pairs, &mask);
        if bound < best_bound {
            best_bound = bound;
            best_k = k_now;
        }
        if k_now == candidates.len() {
            break;
        }
        k *= 2;
    }

    candidates.truncate(best_k);
    candidates
}

/// Greedy distance-1 coloring of the Hessian's column-intersection
/// graph, with **dense columns peeled out**.
///
/// Returns `(var_color, n_colors, peeled)`. `var_color[j] == u32::MAX`
/// marks a column that needs no directional product of its own: either
/// it has no Hessian entries at all, or every entry it has is recovered
/// from a peeled column's pass (see below).
///
/// # Why peeling
///
/// The plain coloring rule is "two columns may share a color when they
/// have no common nonzero row". A single **dense row** — one variable
/// multiplying a sum over all the others, a total-cost variable, a
/// shared design parameter — puts a nonzero in *every* column at that
/// row, so every pair of columns conflicts and the greedy walk hands out
/// `n` colors for a Hessian that may have only ~3n nonzeros. Since
/// `NlTnlp` holds `n_colors × n` dense `seeds` and `compressed` arrays,
/// that turns into O(n²) memory (6.4 GB at n = 20,000) and O(n²) work
/// per `eval_h`, on a problem whose Hessian is perfectly sparse.
///
/// Peeling exploits the Hessian's symmetry. Give a dense column `d` its
/// own singleton color: one directional product with seed `e_d` recovers
/// the whole of column `d` exactly. Every pair `(d, j)` — row `d`,
/// column `j` — is then already known, because `H[d, j] == H[j, d]` sits
/// at row `j` of that same pass. So row `d` no longer constrains any
/// other column's color and is dropped from the conflict structure, and
/// the remaining columns colour on their genuine sparsity. On the
/// one-dense-row case above this takes `n_colors` from `n` to a handful.
///
/// # What peeling costs, and `peel_veto`
///
/// Recovering `H[d, j]` from column `d`'s pass is exact in real
/// arithmetic but not in floating point: the pass is accumulated at the
/// scale of the whole dense column, so every entry read out of it carries
/// an absolute roundoff floor of about `eps * ||H(:, d)||`, where the
/// ordinary path — column `j`'s own pass — would have left a floor of
/// about `eps * |H[d, j]|`. The two agree to the last bit whenever the
/// dense column is well scaled, and they do on every peel-firing model in
/// the benchmark corpus but one. Where a peeled column spans a wide
/// dynamic range, though, that floor swamps its small entries: a column
/// holding both 2.8e5 and 5.6e-4 loses about nine digits on the latter.
///
/// Structure cannot see this — it is a property of the values — so
/// [`NlTnlp::veto_ill_conditioned_peels`] probes the peeled columns once
/// and passes the offenders back here in `peel_veto`, which bars them
/// from being peeled again. A vetoed column is colored normally, its row
/// returns to the conflict structure, and its entries go back to the
/// accurate path.
fn greedy_hessian_coloring(
    n: usize,
    lower_pairs: &[(usize, usize)],
    peel_veto: &[bool],
) -> (Vec<u32>, usize, Vec<bool>) {
    if n == 0 {
        return (Vec::new(), 0, Vec::new());
    }

    // Column degrees in the FULL (symmetric) Hessian: pair (i, j) with
    // i >= j contributes row i to column j and row j to column i; a
    // diagonal contributes once.
    let mut deg = vec![0usize; n];
    for &(i, j) in lower_pairs {
        deg[j] += 1;
        if i != j {
            deg[i] += 1;
        }
    }

    // Pick the dense columns to peel.
    let total: usize = deg.iter().sum();
    let threshold = DENSE_COL_MIN.max(DENSE_COL_FACTOR.saturating_mul(total / n));
    let mut peeled = vec![false; n];
    let candidates: Vec<usize> = (0..n)
        .filter(|&j| deg[j] > threshold && !peel_veto.get(j).copied().unwrap_or(false))
        .collect();
    let dense = select_peeled_cols(n, lower_pairs, &deg, candidates);
    for &j in &dense {
        peeled[j] = true;
    }

    // Conflict structure over the *non-peeled* columns only. Pairs with
    // a peeled endpoint are recovered from that endpoint's own pass, so
    // they neither need a color nor constrain one.
    let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut row_cols: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(i, j) in lower_pairs {
        if peeled[i] || peeled[j] {
            continue;
        }
        col_rows[j].push(i as u32);
        row_cols[i].push(j as u32);
        if i != j {
            col_rows[i].push(j as u32);
            row_cols[j].push(i as u32);
        }
    }

    let mut var_color = vec![u32::MAX; n];
    let mut forbidden = vec![u32::MAX; n + 1];
    let mut n_colors: u32 = 0;

    for j in 0..n {
        // Peeled columns are colored below; a column with no surviving
        // Hessian entries needs no color at all.
        if peeled[j] || col_rows[j].is_empty() {
            continue;
        }
        // Mark colors used by any column sharing a row with `j`.
        // Row-of-col -> col-in-row visit pattern collects all
        // distance-1 neighbors in the column-intersection graph.
        for &r in &col_rows[j] {
            for &c in &row_cols[r as usize] {
                if c as usize == j {
                    continue;
                }
                let cc = var_color[c as usize];
                if cc != u32::MAX {
                    forbidden[cc as usize] = j as u32;
                }
            }
        }
        // First color not stamped with `j as u32`.
        let mut chosen: u32 = 0;
        while (chosen as usize) < forbidden.len() && forbidden[chosen as usize] == j as u32 {
            chosen += 1;
        }
        var_color[j] = chosen;
        if chosen + 1 > n_colors {
            n_colors = chosen + 1;
        }
    }

    // One singleton color per peeled column, appended after the shared
    // ones so the non-peeled numbering is untouched.
    for &j in &dense {
        var_color[j] = n_colors;
        n_colors += 1;
    }

    (var_color, n_colors as usize, peeled)
}

/// Everything downstream of the Hessian coloring: seed vectors, the
/// per-color decode table, the per-tape color sets, and the shared-CSE
/// per-color summand / prelude-reach tables.
///
/// Split out of [`NlTnlp::new`] because
/// [`NlTnlp::veto_ill_conditioned_peels`] may have to build it a second
/// time, with a peel veto in hand, once it has seen real Hessian values.
fn build_color_tables(
    n: usize,
    m: usize,
    lower_pairs: &[(usize, usize)],
    tape_mask: &[bool],
    peel_veto: &[bool],
    obj_tapes: &[Tape],
    con_tapes: &[Vec<Tape>],
    con_hybrid: Option<&mut ConHybrid>,
) -> ColorTables {
    // Hessian column coloring. The chromatic number of the
    // column-intersection graph bounds how many directional
    // Hessian-vector products we need per `eval_h` call —
    // typically O(stencil) for PDE-mesh problems.
    //
    // Only the entries a *tape* can write take part. `tape_mask` is empty
    // when every entry is a tape entry (no quadratic structure), which is
    // the pre-#588 behaviour bit for bit; otherwise the quadratic forms'
    // entries are excluded, since they are scattered from their stored
    // values and never read out of a directional product. A variable that
    // appears only in quadratic rows then has no color at all
    // (`u32::MAX`) — `greedy_hessian_coloring` already declines to color a
    // column with no surviving entries.
    let colored_pairs: Vec<(usize, usize)>;
    let color_input: &[(usize, usize)] = if tape_mask.is_empty() {
        lower_pairs
    } else {
        colored_pairs = lower_pairs
            .iter()
            .zip(tape_mask)
            .filter_map(|(p, &t)| t.then_some(*p))
            .collect();
        &colored_pairs
    };
    let (var_color, n_colors, peeled) = greedy_hessian_coloring(n, color_input, peel_veto);

    // Per-color seed vectors (dense for O(1) Var lookup in
    // `Tape::hessian_directional`).
    let mut seeds: Vec<Vec<f64>> = vec![vec![0.0; n]; n_colors];
    for (k, &c) in var_color.iter().enumerate() {
        if c != u32::MAX {
            seeds[c as usize][k] = 1.0;
        }
    }

    // Per-color decoding table. For each lower-tri pair (i, j)
    // with i >= j, the entry belongs to column j's color: after
    // computing compressed_{c_j} = (H · s_{c_j}), the value at
    // row i is exactly H[i, j] (coloring guarantees no other
    // column in c_j has a nonzero at row i).
    // Built straight from `lower_pairs`, which is sorted, so each
    // color's table is in ascending `hess_idx` order and the decode
    // scatter walks `values` forward instead of hopping (the old
    // build drained a `HashMap`, whose iteration order is arbitrary).
    let mut decoding: Vec<Vec<ColorWrite>> = vec![Vec::new(); n_colors];
    for (idx, &(i, j)) in lower_pairs.iter().enumerate() {
        // An entry no tape writes has no pass to be decoded out of; the
        // quadratic scatter already put its value there.
        if !tape_mask.is_empty() && !tape_mask[idx] {
            continue;
        }
        // Which directional product recovers H[i, j]? Column `j`'s,
        // read at row `i` — except when `i` is a peeled column and
        // `j` is not: then `j` may have no color of its own, and the
        // entry is already in column `i`'s pass at row `j`, since
        // H[i, j] == H[j, i].
        let (c, row) = if peeled[i] && !peeled[j] {
            (var_color[i], j)
        } else {
            (var_color[j], i)
        };
        debug_assert!(
            c != u32::MAX,
            "Hessian pair ({i}, {j}) at index {idx} has no color"
        );
        decoding[c as usize].push(ColorWrite {
            row: row as u32,
            hess_idx: idx as u32,
        });
    }

    // Per-tape distinct color set: for each tape, the colors
    // its variables fall into. `eval_h` loops over only these
    // (tape, color) pairs instead of n_tapes × n_colors.
    let tape_colors = |t: &Tape| -> Vec<u32> {
        let mut s: Vec<u32> = t
            .variables()
            .into_iter()
            .map(|v| var_color[v])
            .filter(|&c| c != u32::MAX)
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    };
    let obj_tape_colors: Vec<Vec<u32>> = obj_tapes.iter().map(tape_colors).collect();
    let con_tape_colors: Vec<Vec<Vec<u32>>> = con_tapes
        .iter()
        .map(|row| row.iter().map(tape_colors).collect())
        .collect();

    // Shared-CSE Hessian tables (issue #557): the per-color summand
    // lists, row lookup, and packed forward-value arena `eval_h`'s
    // hybrid path walks. Built whenever the hybrid tape is — not just
    // above the gate — so flipping `use_for_hess` on (tests, the
    // force env var) needs no extra setup; the cost is one f64 per
    // local op plus small index tables.
    if let Some(h) = con_hybrid {
        let n_sum = h.tape.n_summands();
        let mut local_off: Vec<usize> = Vec::with_capacity(n_sum + 1);
        let mut acc = 0usize;
        for s in &h.tape.summands {
            local_off.push(acc);
            acc += s.ops.len();
        }
        local_off.push(acc);
        h.local_vals_all = vec![0.0; acc];
        h.local_off = local_off;

        let mut summand_row = vec![0u32; n_sum];
        for i in 0..m {
            for si in h.row_start[i]..h.row_start[i + 1] {
                summand_row[si] = i as u32;
            }
        }
        h.summand_row = summand_row;

        // A summand's variable set (`all_vars`) equals its flat tape's,
        // so this is `con_tape_colors` inverted to color-major order —
        // the loop `eval_h` actually runs.
        let mut by_color: Vec<Vec<u32>> = vec![Vec::new(); n_colors];
        for (si, s) in h.tape.summands.iter().enumerate() {
            let mut cs: Vec<u32> = s
                .all_vars
                .iter()
                .map(|&v| var_color[v])
                .filter(|&c| c != u32::MAX)
                .collect();
            cs.sort_unstable();
            cs.dedup();
            for c in cs {
                by_color[c as usize].push(si as u32);
            }
        }
        // Per-color prelude reach: the union of `prelude_reach` over the
        // color's summands, ascending. A union of operand-closed
        // ascending sets is itself operand-closed and ascending, which is
        // exactly what the two prelude sweeps require. Deduped with an
        // epoch-tagged buffer so the build costs
        // `Σ_c Σ_{s ∈ c} |prelude_reach_s|` — the same order as the work
        // it saves — rather than `n_colors × |prelude|`.
        let np = h.tape.n_prelude_ops();
        let mut seen: Vec<u32> = vec![0; np];
        let mut epoch: u32 = 0;
        let mut reach: Vec<u32> = Vec::new();
        let mut reach_off: Vec<usize> = Vec::with_capacity(n_colors + 1);
        for list in &by_color {
            reach_off.push(reach.len());
            epoch += 1;
            let start = reach.len();
            for &si in list {
                for &p in &h.tape.summands[si as usize].prelude_reach {
                    if seen[p] != epoch {
                        seen[p] = epoch;
                        reach.push(p as u32);
                    }
                }
            }
            reach[start..].sort_unstable();
        }
        reach_off.push(reach.len());
        h.hess_color_reach = reach;
        h.hess_color_reach_off = reach_off;

        h.hess_color_summands = by_color;
        h.prelude_dot = vec![0.0; h.tape.n_prelude_ops()];
        h.hess_prelude_adj = vec![0.0; h.tape.n_prelude_ops()];
        h.prelude_adj_dot = vec![0.0; h.tape.n_prelude_ops()];
        h.local_dot = vec![0.0; h.tape.max_summand_ops()];
        h.local_adj_dot = vec![0.0; h.tape.max_summand_ops()];
    }

    ColorTables {
        var_color,
        n_colors,
        peeled_cols: peeled
            .iter()
            .enumerate()
            .filter(|(_, p)| **p)
            .map(|(j, _)| j as u32)
            .collect(),
        seeds,
        decoding,
        obj_tape_colors,
        con_tape_colors,
    }
}

/// The color-dependent half of an [`NlTnlp`], as built by
/// [`build_color_tables`].
struct ColorTables {
    var_color: Vec<u32>,
    n_colors: usize,
    /// Columns given a singleton color and dropped from the conflict
    /// structure. Small by construction (`MAX_PEELED_COLS`).
    peeled_cols: Vec<u32>,
    seeds: Vec<Vec<f64>>,
    decoding: Vec<Vec<ColorWrite>>,
    obj_tape_colors: Vec<Vec<u32>>,
    con_tape_colors: Vec<Vec<Vec<u32>>>,
}

impl NlTnlp {
    /// Build the TNLP, panicking if AMPL external-function resolution fails.
    ///
    /// Kept for the many infallible call sites (CLI, tests) that operate on
    /// `.nl` models known to need no external libraries. Surfaces that can be
    /// handed an arbitrary user model — notably the Python `read_nl` binding —
    /// must call [`Self::try_new`] instead so a missing `$AMPLFUNC` library
    /// becomes a catchable error rather than an uncatchable panic across the
    /// pyo3 boundary.
    pub fn new(prob: NlProblem) -> Self {
        Self::try_new(prob)
            .unwrap_or_else(|e| panic!("failed to resolve AMPL external functions: {e}"))
    }

    /// Build the TNLP, returning an error (instead of panicking) when AMPL
    /// imported functions named by the model can't be resolved — e.g.
    /// `$AMPLFUNC` is unset, a named library is missing/unloadable, or a
    /// referenced function id isn't registered by any loaded library.
    pub fn try_new(prob: NlProblem) -> Result<Self, String> {
        // `POUNCE_DBG_NO_QUAD=1` forces the AD tape for every row and the
        // objective — the A/B reference for the constant-structure path,
        // mirroring `POUNCE_DBG_NO_HYBRID`. Diagnostic only: it is how the
        // fast path's derivatives are checked against the ones they replace
        // on a real model, and how a suspected fast-path bug is bisected
        // against a reference that computes the same numbers a different way.
        Self::try_new_with_quadratic(prob, std::env::var("POUNCE_DBG_NO_QUAD").is_err())
    }

    /// [`Self::try_new`] with the constant-structure fast path (gh #588, Q4)
    /// explicitly on or off.
    ///
    /// The env var `try_new` reads is process-global, which is exactly wrong
    /// for the differential test that has to build the *same* model both ways
    /// and compare the derivatives — so the knob is a parameter here and the
    /// env var only chooses its default.
    pub fn try_new_with_quadratic(prob: NlProblem, use_quadratic: bool) -> Result<Self, String> {
        // Resolve any AMPL imported (external) functions. Walk every
        // nonlinear expression to collect the funcall ids actually
        // referenced; load the libraries named in $AMPLFUNC and bind
        // each id to its (library, registered-name) pair so the tape
        // builder can emit live `TapeOp::Funcall` ops.
        let mut referenced: BTreeSet<usize> = BTreeSet::new();
        // Only trees can carry a `Funcall`: the recognizer refuses one, so
        // a recognized body provably contains none.
        for body in std::iter::once(&prob.obj_nonlinear).chain(prob.con_nonlinear.iter()) {
            if let Some(e) = body.tree() {
                super::nl_external::collect_funcall_ids(e, &mut referenced);
            }
        }
        let resolver = if referenced.is_empty() {
            super::nl_external::ExternalResolver::default()
        } else {
            super::nl_external::ExternalResolver::build_for_problem(
                &prob.imported_funcs,
                &referenced,
            )?
        };

        // Recognize the degree-≤2 objective and rows *before* anything is
        // taped, because the win is not in evaluating the tape faster — it
        // is in never building it. On `qcqp500-3c` the ten quadratic rows
        // are 2.32 M monomials, one `Tape` each; recognizing them first
        // means those 2.32 M tapes, their color lists and their per-tape
        // sparsity sets are never allocated.
        //
        // A row that is trivially zero is left alone: it has no nonlinear
        // part to replace, and routing it through a (necessarily empty)
        // quadratic form would touch every model in the corpus to save
        // nothing.
        let mut quad = QuadraticStructure::new(prob.m);
        if use_quadratic {
            // `is_expanded_quadratic` is the accuracy gate, not an
            // optimization: it admits only forms whose read-out repeats the
            // additions the `.nl` writer already wrote. See its docs — and
            // note it is checked *before* recognition, so a factored form
            // costs one cheap structural walk rather than a full expansion.
            // Recognition is the parser's job now (gh #588, Q5) for the
            // bodies it could reach; `admitted_quad_form` reads its answer
            // back, and still walks a tree for the bodies that kept one —
            // a `from_expressions` model, a factored row rewound by the
            // parser, or any body at all under `POUNCE_DBG_NO_QUAD`.
            //
            // A form that gate refuses is offered the *factored* read-out
            // before it falls back to a tape (gh #673): the gate is a
            // verdict on the expansion, not on the body, and a sum of
            // squared residuals keeps its structure by keeping its squares.
            // `push_body_form` owns that order.
            if let Some(f) = push_body_form(&mut quad, &prob.obj_nonlinear) {
                quad.assign_objective(f);
            }
            for k in 0..prob.m {
                if let Some(f) = push_body_form(&mut quad, &prob.con_nonlinear[k]) {
                    quad.assign_row(k, f);
                }
            }
        }

        // Flatten objective and each constraint into independent
        // summands. Each summand becomes its own `Tape` (CSE bodies
        // are deduplicated within a tape via Rc identity in
        // `Tape::build`; bodies shared across summands are
        // duplicated, which we accept as a simplicity tradeoff).
        let obj_tapes: Vec<Tape> = if quad.objective_form().is_some() {
            Vec::new()
        } else {
            split_top_sums(&prob.obj_expr())
                .iter()
                .map(|e| Tape::build_with_externals(e, &resolver))
                .collect()
        };

        let mut con_tapes: Vec<Vec<Tape>> = Vec::with_capacity(prob.m);
        let mut con_roots: Vec<Expr> = Vec::new();
        let mut row_start: Vec<usize> = Vec::with_capacity(prob.m + 1);
        for k in 0..prob.m {
            row_start.push(con_roots.len());
            // A row with a quadratic form contributes no tape and no
            // hybrid-tape root, so `con_tapes[k]` is empty and its
            // `row_start` range is empty too.
            if quad.row_form(k).is_some() {
                con_tapes.push(Vec::new());
                continue;
            }
            let summands = split_top_sums(&prob.con_expr(k));
            con_tapes.push(
                summands
                    .iter()
                    .map(|e| Tape::build_with_externals(e, &resolver))
                    .collect(),
            );
            // Move (not clone) the split summands into the root list: their
            // `Expr::Cse` payloads are `Arc`s, and `build_multi` keys CSE
            // sharing on `Arc` pointer identity, so the roots must be the
            // same allocations the parse produced.
            con_roots.extend(summands);
        }
        row_start.push(con_roots.len());

        // Shared-CSE constraint tape for `eval_g` (pounce#476). Worth
        // building only when some CSE body is actually referenced from two
        // or more summands — otherwise the prelude comes out empty and the
        // hybrid tape is the flat tape plus an indirection. `hybrid_supported`
        // gates the opcodes `build_multi` would panic on.
        // `POUNCE_DBG_NO_HYBRID=1` forces the flat per-summand tapes for the
        // whole constraint block. Diagnostic only: it is how the
        // flat-versus-shared trade in `HYBRID_JAC_MIN_OP_RATIO` is measured
        // on a real model, and how a suspected hybrid-path bug is bisected
        // against a reference that computes the same derivatives a
        // different way.
        let mut con_hybrid = if std::env::var("POUNCE_DBG_NO_HYBRID").is_ok() {
            None
        } else if hybrid_supported(&con_roots) {
            let tape = HybridTape::build_multi(&con_roots);
            (tape.n_prelude_ops() > 0).then(|| {
                let flat_ops: usize = con_tapes.iter().flatten().map(|t| t.ops.len()).sum();
                let shared_ops = tape.n_prelude_ops() + tape.total_local_ops();
                // `POUNCE_DBG_FORCE_HYBRID_HESS=1` turns the Hessian gate on
                // regardless of the op ratio. Diagnostic only — it is how the
                // crossover in `HYBRID_HESS_MIN_OP_RATIO` is measured
                // (same-binary A/B against `POUNCE_DBG_NO_HYBRID=1`) on
                // models that sit below the gate.
                let force_hess = std::env::var("POUNCE_DBG_FORCE_HYBRID_HESS").is_ok();
                ConHybrid {
                    prelude_vals: vec![0.0; tape.n_prelude_ops()],
                    local_vals: vec![0.0; tape.max_summand_ops()],
                    local_adj: vec![0.0; tape.max_summand_ops()],
                    prelude_adj: vec![0.0; tape.n_prelude_ops()],
                    use_for_jac: flat_ops as f64
                        >= HYBRID_JAC_MIN_OP_RATIO * shared_ops.max(1) as f64,
                    use_for_hess: force_hess
                        || flat_ops as f64 >= HYBRID_HESS_MIN_OP_RATIO * shared_ops.max(1) as f64,
                    local_vals_all: Vec::new(),
                    local_off: Vec::new(),
                    summand_row: Vec::new(),
                    hess_color_summands: Vec::new(),
                    hess_color_reach: Vec::new(),
                    hess_color_reach_off: Vec::new(),
                    prelude_dot: Vec::new(),
                    hess_prelude_adj: Vec::new(),
                    prelude_adj_dot: Vec::new(),
                    local_dot: Vec::new(),
                    local_adj_dot: Vec::new(),
                    row_start,
                    tape,
                }
            })
        } else {
            None
        };
        drop(con_roots);

        // Hessian-of-Lagrangian sparsity: union of each tape's own
        // structural Hessian sparsity.
        // One flat `Vec`, sorted and deduped once, rather than a global
        // `BTreeSet` fed a single insert at a time across every summand
        // in the model: sort+dedup walks contiguous memory where the tree
        // chased a pointer and allocated a node per entry. The result is
        // exactly the ascending order the rest of this function wants, so
        // it doubles as `lower_pairs` instead of being copied into it.
        let mut tape_pairs: Vec<(usize, usize)> = Vec::new();
        for t in &obj_tapes {
            tape_pairs.extend(t.hessian_sparsity());
        }
        for row in &con_tapes {
            for t in row {
                tape_pairs.extend(t.hessian_sparsity());
            }
        }
        tape_pairs.sort_unstable();
        tape_pairs.dedup();

        // The assembled pattern is the tapes' union *plus* the quadratic
        // forms'. They are kept apart because the coloring only ever needs
        // the tape half — a quadratic block's entries are scattered
        // directly, not recovered from a directional product — and coloring
        // a dense quadratic block would put the color count back at `n` to
        // pay for products nobody runs (`qcqp500-3c`: 500 colors → 0).
        let mut lower_pairs = tape_pairs.clone();
        if !quad.is_empty() {
            for f in quad
                .objective_form()
                .into_iter()
                .chain((0..prob.m).filter_map(|i| quad.row_form(i)))
            {
                lower_pairs.extend(
                    quad.lower_triangle(f)
                        .map(|(r, c, _)| (r as usize, c as usize)),
                );
            }
            lower_pairs.sort_unstable();
            lower_pairs.dedup();
        }

        // Which assembled entries a tape can write. Both sides are sorted
        // and `tape_pairs ⊆ lower_pairs`, so this is a merge walk, and the
        // mask is left empty (meaning "all of them") when nothing was
        // recognized.
        let h_tape_mask: Vec<bool> = if quad.is_empty() {
            Vec::new()
        } else {
            let mut mask = vec![false; lower_pairs.len()];
            let mut t = 0usize;
            for (idx, pair) in lower_pairs.iter().enumerate() {
                if t < tape_pairs.len() && tape_pairs[t] == *pair {
                    mask[idx] = true;
                    t += 1;
                }
            }
            debug_assert_eq!(t, tape_pairs.len(), "every tape pair is in the union");
            mask
        };
        drop(tape_pairs);

        let mut h_irow = Vec::with_capacity(lower_pairs.len());
        let mut h_jcol = Vec::with_capacity(lower_pairs.len());
        for &(hi, lo) in &lower_pairs {
            h_irow.push(hi as i32);
            h_jcol.push(lo as i32);
        }

        // Bind each form's lower-triangle entries to their index in the
        // assembled pattern. `lower_pairs` is sorted, so the lookup is a
        // binary search — done once here, never again on the hot path.
        if !quad.is_empty() {
            quad.bind_slots(|r, c| {
                lower_pairs
                    .binary_search(&(r as usize, c as usize))
                    .unwrap_or_else(|_| {
                        unreachable!("quadratic entry ({r}, {c}) missing from the union pattern")
                    })
            });
        }

        // Hessian column coloring and everything keyed off it. The
        // chromatic number of the column-intersection graph bounds how
        // many directional Hessian-vector products we need per `eval_h`
        // call — typically O(stencil) for PDE-mesh problems.
        let ColorTables {
            var_color,
            n_colors,
            peeled_cols,
            seeds,
            decoding,
            obj_tape_colors,
            con_tape_colors,
        } = build_color_tables(
            prob.n,
            prob.m,
            &lower_pairs,
            &h_tape_mask,
            &vec![false; prob.n],
            &obj_tapes,
            &con_tapes,
            con_hybrid.as_mut(),
        );

        // Per-row Jacobian sparsity = union of tape vars plus
        // linear-segment vars.
        let mut jac_cols: Vec<Vec<usize>> = Vec::with_capacity(prob.m);
        let mut jac_nnz = 0;
        for (i, row_tapes) in con_tapes.iter().enumerate() {
            let mut cols: Vec<usize> = Vec::with_capacity(prob.con_linear[i].len());
            for t in row_tapes {
                cols.extend(t.variables());
            }
            // A quadratic row has no tape, so its Jacobian support comes
            // from the form: `Hx + a` is nonzero exactly on the union of the
            // Hessian's rows and the folded linear part.
            if let Some(f) = quad.row_form(i) {
                cols.extend(quad.gradient_support(f).iter().map(|&v| v as usize));
            }
            cols.extend(prob.con_linear[i].iter().map(|(v, _)| *v));
            cols.sort_unstable();
            cols.dedup();
            cols.shrink_to_fit();
            jac_nnz += cols.len();
            jac_cols.push(cols);
        }

        let mut max_tape_n: usize = 0;
        for t in &obj_tapes {
            max_tape_n = max_tape_n.max(t.ops.len());
        }
        for row in &con_tapes {
            for t in row {
                max_tape_n = max_tape_n.max(t.ops.len());
            }
        }

        if std::env::var("POUNCE_DBG_TAPE_STATS").is_ok() {
            let n_obj = obj_tapes.len();
            let n_con: usize = con_tapes.iter().map(|r| r.len()).sum();
            let total = n_obj + n_con;
            let mut sum_ops: usize = 0;
            for t in &obj_tapes {
                sum_ops += t.ops.len();
            }
            for row in &con_tapes {
                for t in row {
                    sum_ops += t.ops.len();
                }
            }
            let t = total.max(1);
            let nnz_h = h_irow.len();
            let avg_decode =
                decoding.iter().map(|d| d.len()).sum::<usize>() as f64 / n_colors.max(1) as f64;
            eprintln!(
                "[tape stats] summands={total} (obj={n_obj} con={n_con}) \
                 total_ops={sum_ops} avg_ops={:.1} max_ops={max_tape_n} \
                 n_colors={n_colors} avg_decode_per_color={avg_decode:.1} nnz_h={nnz_h}",
                sum_ops as f64 / t as f64,
            );
            // Flat vs shared-CSE op counts for the constraint block. The
            // ratio is how much duplicated CSE work `eval_g`'s hybrid path
            // avoids, and the ceiling on what routing the Jacobian /
            // Hessian through the same prelude could save.
            match &con_hybrid {
                Some(h) => {
                    let flat: usize = con_tapes.iter().flatten().map(|t| t.ops.len()).sum();
                    let prelude = h.tape.n_prelude_ops();
                    let local = h.tape.total_local_ops();
                    eprintln!(
                        "[hybrid stats] con flat_ops={flat} prelude_ops={prelude} \
                         local_ops={local} shared_total={} flat/shared={:.2}x \
                         jac_gate={} hess_gate={}",
                        prelude + local,
                        flat as f64 / (prelude + local).max(1) as f64,
                        if h.use_for_jac { "on" } else { "off" },
                        if h.use_for_hess { "on" } else { "off" },
                    );
                }
                None => eprintln!("[hybrid stats] con hybrid not built (no shared CSE bodies)"),
            }
            // What the constant-structure path took off the tape builder.
            // `forms` counts the objective too, which is why it can exceed
            // the row count by one.
            let quad_rows = (0..prob.m).filter(|&i| quad.row_form(i).is_some()).count();
            // How many of those forms the *parser* produced, i.e. how many
            // never cost an `Expr` (gh #588, Q5). The rest were recognized
            // from a tree that was built: a `from_expressions` model, or a
            // body the parser rewound because it was not degree 2.
            let parsed = std::iter::once(&prob.obj_nonlinear)
                .chain(prob.con_nonlinear.iter())
                .filter(|b| b.quad().is_some())
                .count();
            eprintln!(
                "[quad stats] forms={} (parse-time {parsed}) rows={quad_rows}/{} obj={} \
                 stored_h_entries={} colored_pairs={}/{}",
                quad.len(),
                prob.m,
                if quad.objective_form().is_some() {
                    "quadratic"
                } else {
                    "taped"
                },
                quad.stored_entries(),
                if h_tape_mask.is_empty() {
                    lower_pairs.len()
                } else {
                    h_tape_mask.iter().filter(|&&t| t).count()
                },
                lower_pairs.len(),
            );
        }

        let compressed: Vec<Vec<f64>> = vec![vec![0.0; prob.n]; n_colors];

        let mut me = Self {
            prob,
            obj_tapes,
            con_tapes,
            con_hybrid,
            quad,
            h_tape_mask,
            h_irow,
            h_jcol,
            jac_cols,
            jac_nnz,
            seeds,
            decoding,
            obj_tape_colors,
            con_tape_colors,
            var_color,
            peeled_cols,
            final_x: None,
            final_obj: 0.0,
            final_lambda: None,
            final_z_l: None,
            final_z_u: None,
            scratch_row_grad: Vec::new(),
            vals_scratch: vec![0.0; max_tape_n],
            dot_scratch: vec![0.0; max_tape_n],
            adj_scratch: vec![0.0; max_tape_n],
            adj_dot_scratch: vec![0.0; max_tape_n],
            compressed,
            hvp_live: Vec::new(),
            curvature_scaling: None,
        };
        me.veto_ill_conditioned_peels();
        Ok(me)
    }

    /// Un-peel any dense column whose own pass is too ill-scaled to read
    /// its small entries out of, and re-color if that changes anything.
    ///
    /// `greedy_hessian_coloring` picks the peel set from structure alone,
    /// which is the right call for the memory and the color count but
    /// blind to the one thing that can go wrong: an entry recovered from
    /// column `d`'s pass inherits that pass's roundoff floor, about
    /// `eps * ||H(:, d)||`, rather than its own much smaller one. A
    /// well-scaled dense column loses nothing to that — the recovered
    /// entries come back bit-identical to the uncompressed reference on
    /// every peel-firing model in the benchmark corpus but one. A column
    /// spanning many orders of magnitude, though, hands its small entries
    /// a relative error of `eps * ||H(:, d)|| / |H[d, j]|`, which on
    /// `cho_parmest` (a 12-parameter kinetic fit whose peeled columns
    /// hold both 2.8e5 and 5.6e-4) reaches 1e-5. The primal solution
    /// survives that, but the multipliers come out of the KKT system the
    /// Hessian sits in, so `inf_du` picks up a jitter floor near 1e-6 and
    /// the solve stalls short of `Optimal` on a problem it used to
    /// certify.
    ///
    /// Nothing structural distinguishes the two cases, so measure it: one
    /// Hessian evaluation at `x0` with unit multipliers leaves each peeled
    /// column's exact pass sitting in `compressed`, and a column whose
    /// worst recovered entry would lose more than
    /// `PEEL_MAX_REL_ERR` is vetoed and colored the ordinary way. Costs
    /// one `eval_h` — at the peeled color count, so cheap — and only for
    /// the ~3% of models that peel anything at all.
    fn veto_ill_conditioned_peels(&mut self) {
        if self.peeled_cols.is_empty() {
            return;
        }

        let mut values = vec![0.0; self.h_irow.len()];
        let lambda = vec![1.0; self.prob.m];
        let x0 = self.prob.x0.clone();
        if !self.eval_h(
            Some(&x0),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        ) {
            return;
        }

        let dbg = std::env::var("POUNCE_DBG_TAPE_STATS").is_ok();
        // A column whose whole pass is negligible against the Hessian as a
        // whole cannot move the KKT system no matter how badly its own
        // entries are rounded, and columns that are identically zero at
        // `x0` would otherwise veto on a ratio of pure noise.
        let h_scale = self
            .compressed
            .iter()
            .flat_map(|c| c.iter())
            .fold(0.0f64, |a, &v| a.max(v.abs()));
        let mut peel_veto = vec![false; self.prob.n];
        let mut vetoed = 0usize;
        for &d in &self.peeled_cols {
            let c = self.var_color[d as usize];
            if c == u32::MAX {
                continue;
            }
            let pass = &self.compressed[c as usize];
            // The floor the pass was accumulated at, against the smallest
            // entry actually read out of it.
            let scale = pass.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            // Entries at or below the pass's own roundoff floor carry no
            // information to lose: `eps * scale` is the noise the pass was
            // accumulated at, so such an entry is already indistinguishable
            // from zero whether or not the column is peeled. Including them
            // would divide by that noise -- `orthregd` holds entries of
            // 8e-15 in a pass of norm 6e5, and bounds at 1e4 while measuring
            // 4e-16 against an uncompressed reference.
            let noise = f64::EPSILON * scale;
            let smallest = self.decoding[c as usize]
                .iter()
                .map(|w| pass[w.row as usize].abs())
                .filter(|v| *v > noise)
                .fold(f64::INFINITY, f64::min);
            if !smallest.is_finite() || smallest == 0.0 || scale == 0.0 {
                continue;
            }
            if scale <= h_scale * f64::EPSILON {
                continue;
            }
            let rel_err = f64::EPSILON * scale / smallest;
            if dbg {
                eprintln!(
                    "[peel probe] col={d} color={c} ||pass||={scale:.3e} \
                     min_entry={smallest:.3e} rel_err={rel_err:.3e}{}",
                    if rel_err > PEEL_MAX_REL_ERR {
                        " VETO"
                    } else {
                        ""
                    }
                );
            }
            if rel_err > PEEL_MAX_REL_ERR {
                peel_veto[d as usize] = true;
                vetoed += 1;
            }
        }

        if vetoed == 0 {
            return;
        }
        if dbg {
            eprintln!(
                "[peel probe] vetoing {vetoed}/{} peeled columns; re-coloring",
                self.peeled_cols.len()
            );
        }
        self.recolor(&peel_veto);
    }

    /// Rebuild the coloring and everything keyed off it, barring
    /// `peel_veto` from the peel set.
    fn recolor(&mut self, peel_veto: &[bool]) {
        let lower_pairs: Vec<(usize, usize)> = self
            .h_irow
            .iter()
            .zip(&self.h_jcol)
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();
        let ColorTables {
            var_color,
            n_colors,
            peeled_cols,
            seeds,
            decoding,
            obj_tape_colors,
            con_tape_colors,
        } = build_color_tables(
            self.prob.n,
            self.prob.m,
            &lower_pairs,
            &self.h_tape_mask,
            peel_veto,
            &self.obj_tapes,
            &self.con_tapes,
            self.con_hybrid.as_mut(),
        );
        self.var_color = var_color;
        self.peeled_cols = peeled_cols;
        self.seeds = seeds;
        self.decoding = decoding;
        self.obj_tape_colors = obj_tape_colors;
        self.con_tape_colors = con_tape_colors;
        self.compressed = vec![vec![0.0; self.prob.n]; n_colors];
    }

    pub fn final_x(&self) -> Option<&[Number]> {
        self.final_x.as_deref()
    }

    pub fn final_obj(&self) -> Number {
        self.final_obj
    }

    /// Converged constraint multipliers from the last solve, in original
    /// `.nl` row order. `None` before a solve finishes. See
    /// [`Self::final_x`] for the primal counterpart.
    pub fn final_lambda(&self) -> Option<&[Number]> {
        self.final_lambda.as_deref()
    }

    /// Converged lower / upper bound multipliers from the last solve, in
    /// original `.nl` variable order and Ipopt's internal convention (both
    /// `>= 0`). `None` before a solve finishes.
    pub fn final_bound_multipliers(&self) -> Option<(&[Number], &[Number])> {
        Some((self.final_z_l.as_deref()?, self.final_z_u.as_deref()?))
    }

    /// The parsed problem this TNLP evaluates (bounds, starting point,
    /// names, suffixes). Read-only; per-instance overrides go through
    /// [`Self::variant`].
    pub fn problem(&self) -> &NlProblem {
        &self.prob
    }

    /// Opt this model in to **curvature-based** scaling (gh #703): compute
    /// the per-variable and per-row factors of
    /// [`crate::nl_scaling::curvature_scaling`] and serve them from
    /// [`TNLP::get_scaling_parameters`], so `nlp_scaling_method` reaches
    /// them through the channel it already has for user factors.
    ///
    /// Returns `false` when the model is not one the scheme is defined for
    /// — some row or the objective is not degree ≤ 2, so no constant `Qᵢ`
    /// exists. The caller must surface that as an error rather than solving
    /// unscaled: an accepted scaling option that is then quietly not applied
    /// is exactly the gh #483 failure.
    ///
    /// Costs one pass over every stored Hessian entry plus `RUIZ_SWEEPS`
    /// passes over the magnitude surrogates, and nothing at all for a solve
    /// that never calls it.
    pub fn enable_curvature_scaling(&mut self) -> bool {
        match crate::nl_scaling::curvature_scaling(&self.prob) {
            Some(sc) => {
                self.curvature_scaling = Some(sc);
                true
            }
            None => false,
        }
    }

    /// Whether [`Self::enable_curvature_scaling`] has been called and
    /// succeeded.
    pub fn curvature_scaling_enabled(&self) -> bool {
        self.curvature_scaling.is_some()
    }

    /// Whether the enabled curvature scaling actually read any curvature —
    /// see [`crate::nl_scaling::CurvatureScaling::quadratic`]. `false` when
    /// scaling is not enabled, and `false` for a degree-≤2 model whose every
    /// `Q` is empty (an LP), where the scheme degenerates to plain Ruiz
    /// equilibration of `[A b]`.
    pub fn curvature_scaling_read_curvature(&self) -> bool {
        self.curvature_scaling
            .as_ref()
            .is_some_and(|sc| sc.quadratic)
    }

    /// Is constraint row `i` evaluated from a constant quadratic form
    /// rather than from an AD tape (gh #588, Q4)?
    ///
    /// Structural, so it answers before any evaluation. Exposed for the
    /// differential test, which has to know which models exercise the fast
    /// path at all, and for `POUNCE_DBG_TAPE_STATS`.
    pub fn quadratic_row(&self, i: usize) -> bool {
        self.quad.row_form(i).is_some()
    }

    /// As [`Self::quadratic_row`], for the objective.
    pub fn quadratic_objective(&self) -> bool {
        self.quad.objective_form().is_some()
    }

    /// Structural set of variables that appear in *some* nonlinear part —
    /// the union of `collect_vars` over the objective and every constraint
    /// row. Shared by `get_variables_linearity` and the nonlinear-variable
    /// list below so the two can never disagree.
    fn nonlinear_var_set(&self) -> BTreeSet<usize> {
        let mut nonlinear: BTreeSet<usize> = BTreeSet::new();
        self.prob.obj_nonlinear.collect_vars(&mut nonlinear);
        for row in &self.prob.con_nonlinear {
            row.collect_vars(&mut nonlinear);
        }
        nonlinear
    }

    /// The nonlinear-variable list published through the TNLP contract
    /// (`get_number_of_nonlinear_variables` / `get_list_of_nonlinear_variables`),
    /// ascending, in the C index style [`NlTnlp`] reports.
    ///
    /// The contract's asymmetry decides how this is computed. A consumer
    /// treats every variable *absent* from the list as linear — Ipopt's
    /// limited-memory Hessian skips the quasi-Newton update in that
    /// subspace — so naming too few variables is a wrong answer, while
    /// naming too many merely costs work. Hence:
    ///
    /// * When the `.nl` header says **every** variable is nonlinear, publish
    ///   that and skip the walk. This is the maximally conservative answer,
    ///   so it is sound whatever the header's provenance, and it is the
    ///   common case for the models this matters on (`eigena2`: 110 of 110).
    /// * Otherwise walk the trees. The header's prefix
    ///   (`nlvc + nlvo − nlvb` — see [`NlCounts`]) would also be an O(1)
    ///   answer, but it would be one that *trusts* the writer to have
    ///   ordered the variables as the format requires, and the walk is
    ///   already paid once per solve by `get_variables_linearity`. The
    ///   walk is also the only option for a model built through
    ///   [`NlProblem::from_expressions`], which has no header at all.
    ///
    /// The two disagree only in the safe direction, which
    /// `crates/pounce-cli/tests/nl_header_counts.rs` asserts over the
    /// fixture corpus: the walked set always sits inside the header's
    /// prefix, because `parse_nl_text` folds constant `C` bodies away
    /// (`gh #492`) and AMPL's own census predates that fold.
    fn nonlinear_variables(&self) -> Vec<Index> {
        if let Some(c) = self.prob.nl_counts
            && c.nonlinear_vars() >= self.prob.n
        {
            return (0..self.prob.n as Index).collect();
        }
        self.nonlinear_var_set()
            .into_iter()
            .map(|i| i as Index)
            .collect()
    }

    /// Mutable access to that same problem, for a caller that owns this
    /// TNLP outright.
    ///
    /// The tapes were built from the expressions in [`Self::problem`] and
    /// are not rebuilt, so editing an expression here does **not** change
    /// what this TNLP evaluates. It exists for teardown: the Python
    /// binding takes the expression trees out through here so a deeply
    /// nested one is dropped on a stack chosen for it rather than
    /// recursively on whatever thread collected the object (pounce#472).
    pub fn problem_mut(&mut self) -> &mut NlProblem {
        &mut self.prob
    }

    /// Hessian-vector product of the Lagrangian:
    /// `out = (obj_factor·∇²f(x) + Σ_i λ_i·∇²g_i(x)) · v`.
    ///
    /// This is the matrix-free counterpart of `eval_h`. `eval_h` runs one
    /// [`Tape::hessian_directional`] pass *per color* and then decodes the
    /// compressed columns into the sparse lower triangle; here the seed is
    /// the caller's `v` directly, so it is a single forward-over-reverse
    /// pass per tape — O(tape ops), independent of `n` and of the coloring's
    /// chromatic number. That is what makes it usable on models where
    /// materializing `∇²L` is impractical (issue #469): a Newton–Krylov /
    /// truncated-CG step only ever needs `∇²L · v`.
    ///
    /// Sign convention matches `eval_h` and the rest of this evaluator: a
    /// `maximize` model's objective is negated so the returned operator is
    /// the one that minimizing solves. `lambda` is `None` for the objective
    /// block alone.
    ///
    /// `out` is overwritten (not accumulated into). Errors on any length
    /// mismatch rather than panicking, since the Python binding hands this
    /// arbitrary user arrays.
    pub fn hessian_vector_product(
        &mut self,
        x: &[Number],
        v: &[Number],
        obj_factor: Number,
        lambda: Option<&[Number]>,
        out: &mut [Number],
    ) -> Result<(), String> {
        self.hessian_vector_products(x, v, 1, obj_factor, lambda, out)
    }

    /// Block form of [`Self::hessian_vector_product`]: `k` directions at
    /// once, `out[:, c] = ∇²L · v[:, c]`.
    ///
    /// `v` and `out` are `n × k` in **column-major** order — direction `c`
    /// occupies `v[c*n .. (c+1)*n]`. `out` is overwritten.
    ///
    /// Worth having as its own entry point rather than a loop over the
    /// single-vector call: the forward sweep depends only on `x`, so a block
    /// runs it *once per tape* and reuses `vals` across all `k` directions,
    /// where `k` separate calls would redo it `k` times. Only the
    /// forward-tangent + reverse-over-tangent passes are per-direction. That
    /// is the shape a block-Krylov solve, a directional-derivative probe, or
    /// a densify-the-Hessian loop wants.
    ///
    /// An all-zero direction is skipped, so passing a sparse block whose
    /// columns are mostly empty costs only the columns that carry signal.
    /// (The sparsity that dominates is the model's own: each tape touches
    /// only its own variables, and `hessian_directional` is O(tape ops), not
    /// O(n).)
    pub fn hessian_vector_products(
        &mut self,
        x: &[Number],
        v: &[Number],
        k: usize,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        out: &mut [Number],
    ) -> Result<(), String> {
        let (n, m) = (self.prob.n, self.prob.m);
        let check = |name: &str, got: usize, want: usize| -> Result<(), String> {
            if got == want {
                Ok(())
            } else {
                Err(format!(
                    "hessian_vector_product: {name} has length {got}, expected {want}"
                ))
            }
        };
        check("x", x.len(), n)?;
        check("v", v.len(), n * k)?;
        check("out", out.len(), n * k)?;
        if let Some(lam) = lambda {
            check("lambda", lam.len(), m)?;
        }

        out.fill(0.0);
        if k == 0 || n == 0 {
            return Ok(());
        }

        // Which directions carry signal. Computed once, not per (tape,
        // direction) pair — with many small summand tapes the scan would
        // otherwise dominate the work it is meant to save. Reuses the
        // persistent mask so a Krylov loop allocates nothing per iteration.
        self.hvp_live.clear();
        self.hvp_live
            .extend((0..k).map(|c| v[c * n..(c + 1) * n].iter().any(|&s| s != 0.0)));
        if !self.hvp_live.iter().any(|&l| l) {
            return Ok(());
        }

        let obj_seed = if self.prob.minimize {
            obj_factor
        } else {
            -obj_factor
        };
        // The constant blocks. `H · v` for a quadratic form is a matvec
        // against stored values, so unlike a tape it needs no forward sweep
        // and no `x` at all. Easy to forget — a matrix-free solve that
        // silently dropped the quadratic part of `∇²L` would still converge,
        // just to a different point by a different route, which is the
        // failure mode this series is most exposed to.
        if !self.quad.is_empty() {
            if obj_seed != 0.0 {
                if let Some(f) = self.quad.objective_form() {
                    for (c, out_col) in out.chunks_mut(n).enumerate() {
                        if self.hvp_live[c] {
                            self.quad.add_hessian_vector(
                                f,
                                &v[c * n..(c + 1) * n],
                                obj_seed,
                                out_col,
                            );
                        }
                    }
                }
            }
            if let Some(lam) = lambda {
                for (i, &w) in lam.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let Some(f) = self.quad.row_form(i) else {
                        continue;
                    };
                    for (c, out_col) in out.chunks_mut(n).enumerate() {
                        if self.hvp_live[c] {
                            self.quad
                                .add_hessian_vector(f, &v[c * n..(c + 1) * n], w, out_col);
                        }
                    }
                }
            }
        }

        if obj_seed != 0.0 {
            for t in &self.obj_tapes {
                if t.ops.is_empty() {
                    continue;
                }
                // Once per tape, not once per direction — the whole point
                // of the block form.
                t.forward_into(x, &mut self.vals_scratch);
                for (c, out_col) in out.chunks_mut(n).enumerate() {
                    if !self.hvp_live[c] {
                        continue;
                    }
                    t.hessian_directional(
                        &self.vals_scratch,
                        &v[c * n..(c + 1) * n],
                        obj_seed,
                        out_col,
                        &mut self.dot_scratch,
                        &mut self.adj_scratch,
                        &mut self.adj_dot_scratch,
                    );
                }
            }
        }

        if let Some(lam) = lambda {
            // `lam.len() == m` was checked above, so `con_tapes[k]` is in
            // range for every k.
            for (i, &w) in lam.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                for t in &self.con_tapes[i] {
                    if t.ops.is_empty() {
                        continue;
                    }
                    t.forward_into(x, &mut self.vals_scratch);
                    for (c, out_col) in out.chunks_mut(n).enumerate() {
                        if !self.hvp_live[c] {
                            continue;
                        }
                        t.hessian_directional(
                            &self.vals_scratch,
                            &v[c * n..(c + 1) * n],
                            w,
                            out_col,
                            &mut self.dot_scratch,
                            &mut self.adj_scratch,
                            &mut self.adj_dot_scratch,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Clone this TNLP with per-instance overrides applied — the
    /// "one structure, many bound / starting-point variations" case of
    /// batched NLP solving (pounce#126). The AD tapes, sparsity, and
    /// coloring are reused via `Clone` (they depend only on the model
    /// structure, which a variation cannot change); only the values in
    /// `prob.x0` / `prob.x_l` / `prob.x_u` / `prob.g_l` / `prob.g_u`
    /// are replaced. Any stale `final_x` from a previous solve of
    /// `self` is cleared on the clone.
    ///
    /// Errors when an override's length does not match the model
    /// (`n` for `x0`/`x_l`/`x_u`, `m` for `g_l`/`g_u`).
    pub fn variant(&self, v: &NlVariation) -> Result<Self, String> {
        let check = |name: &str, got: usize, want: usize| -> Result<(), String> {
            if got == want {
                Ok(())
            } else {
                Err(format!(
                    "NlVariation.{name} has length {got}, expected {want}"
                ))
            }
        };
        let mut out = self.clone();
        out.final_x = None;
        out.final_obj = 0.0;
        out.final_lambda = None;
        out.final_z_l = None;
        out.final_z_u = None;
        if let Some(x0) = &v.x0 {
            check("x0", x0.len(), self.prob.n)?;
            out.prob.x0.clone_from(x0);
        }
        if let Some(x_l) = &v.x_l {
            check("x_l", x_l.len(), self.prob.n)?;
            out.prob.x_l.clone_from(x_l);
        }
        if let Some(x_u) = &v.x_u {
            check("x_u", x_u.len(), self.prob.n)?;
            out.prob.x_u.clone_from(x_u);
        }
        if let Some(g_l) = &v.g_l {
            check("g_l", g_l.len(), self.prob.m)?;
            out.prob.g_l.clone_from(g_l);
        }
        if let Some(g_u) = &v.g_u {
            check("g_u", g_u.len(), self.prob.m)?;
            out.prob.g_u.clone_from(g_u);
        }
        Ok(out)
    }

    /// Build one [`NlTnlp`] per variation, sharing this instance's
    /// structure (see [`Self::variant`]). Returns instances in input
    /// order; errors on the first length-mismatched variation.
    pub fn variants(&self, vs: &[NlVariation]) -> Result<Vec<Self>, String> {
        vs.iter().map(|v| self.variant(v)).collect()
    }
}

/// Per-instance overrides for building a family of related NLP
/// instances from one parsed `.nl` model (pounce#126): same structure
/// and tapes, different starting point and/or bounds — parametric
/// sweeps, multi-start, or branch-and-bound node relaxations where
/// each node only tightens variable bounds. `None` keeps the base
/// model's value.
#[derive(Debug, Clone, Default)]
pub struct NlVariation {
    pub x0: Option<Vec<Number>>,
    pub x_l: Option<Vec<Number>>,
    pub x_u: Option<Vec<Number>>,
    pub g_l: Option<Vec<Number>>,
    pub g_u: Option<Vec<Number>>,
}

impl pounce_nlp::expression_provider::ExpressionProvider for NlTnlp {
    /// Per-`.nl`-row constraint expression tape, with the linear
    /// part folded in. Returns `None` for constraints that contribute
    /// neither a nonlinear expression nor any linear coefficients
    /// (so FBBT skips them — there's nothing to tighten).
    fn constraint_expression(&self, i: usize) -> Option<pounce_nlp::FbbtTape> {
        if i >= self.prob.con_nonlinear.len() {
            return None;
        }
        let nonlinear = self.prob.con_expr(i);
        let linear = self
            .prob
            .con_linear
            .get(i)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        // FBBT needs the tree, so a recognized body is rebuilt here. It is
        // rebuilt once per row per presolve pass and dropped again, so the
        // DAG never all exists at once — which is the property the phase is
        // actually about. Translating the stored coefficients instead would
        // propagate bounds through a different association and change the
        // tightening; that is not a trade this phase makes.
        crate::nl_fbbt_translate::translate_constraint(&nonlinear, linear)
    }

    /// Variable name from the sibling `.col` file, if one was loaded.
    /// Index is original `.nl` column order.
    fn variable_name(&self, i: usize) -> Option<&str> {
        self.prob.var_names.get(i).map(String::as_str)
    }

    /// Constraint name from the sibling `.row` file, if one was loaded.
    /// Index is original `.nl` row order.
    fn constraint_name(&self, i: usize) -> Option<&str> {
        self.prob.con_names.get(i).map(String::as_str)
    }
}

impl TNLP for NlTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.prob.n as Index,
            m: self.prob.m as Index,
            nnz_jac_g: self.jac_nnz as Index,
            nnz_h_lag: self.h_irow.len() as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.prob.x_l);
        b.x_u.copy_from_slice(&self.prob.x_u);
        if !self.prob.g_l.is_empty() {
            b.g_l.copy_from_slice(&self.prob.g_l);
            b.g_u.copy_from_slice(&self.prob.g_u);
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.prob.x0);
        // The `.nl` `d` segment supplies initial constraint multipliers
        // (`lambda0`). Honor a warm-start request — `init_lambda` is set by
        // the engine when `warm_start_init_point yes` — by handing them
        // back; `OrigIpoptNlp::get_starting_point` then compresses them into
        // the algorithm-side y_c / y_d. Without this the warm start silently
        // began from zero multipliers, discarding the parsed duals. (Code
        // review 2026-06 item M19.) The `.nl` `d` segment carries no bound
        // multipliers, so `z_l`/`z_u` are left to the engine's defaults.
        if sp.init_lambda {
            sp.lambda.copy_from_slice(&self.prob.lambda0);
        }
        true
    }

    /// Hand the `.nl` file's `scaling_factor` suffixes to the engine's
    /// `nlp_scaling_method=user-scaling` pathway — the AMPL/ASL channel
    /// Ipopt reads in `AmplTNLP::GetScalingParameters`, and the one a
    /// Pyomo `Suffix(direction=Suffix.EXPORT)` named `scaling_factor`
    /// writes into. Before gh#483 nothing implemented this callback for
    /// `.nl` input, so a tagged model reached the solver with the option
    /// accepted and *no* scaling applied, silently.
    ///
    /// Returns `false` (engine falls back to no scaling) when the file
    /// declares no `scaling_factor` suffix at all — the same "user
    /// supplied nothing" answer as the default `TNLP` impl.
    ///
    /// AMPL suffix vectors default to **0** for components the model did
    /// not tag, and 0 is not a usable scale factor. A zero entry is
    /// therefore read as "not tagged" and becomes 1.0, which is what
    /// "unlisted components are unscaled" means. Per-variable factors
    /// are passed straight through: `OrigIpoptNlp` does not model them
    /// and refuses the solve with a message rather than dropping them.
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        const NAME: &str = "scaling_factor";
        let sfx = &self.prob.suffixes;
        let obj = sfx.obj_real.get(NAME);
        let var = sfx.var_real.get(NAME);
        let con = sfx.con_real.get(NAME);
        let computed = self.curvature_scaling.as_ref();
        if obj.is_none() && var.is_none() && con.is_none() && computed.is_none() {
            return false;
        }
        // Curvature-based factors (gh #703) are the *base*; a
        // `scaling_factor` suffix the model actually carries overrides
        // them component by component, because an explicit factor from the
        // modeller beats one inferred from the coefficients. A model with
        // no suffixes is the ordinary case and gets the computed vectors
        // whole.
        if let Some(sc) = computed {
            // The length guards are a `copy_from_slice` panic guard, not a
            // policy: `curvature_scaling` sizes both vectors from the same
            // `NlProblem` this callback is answering for, so a mismatch is a
            // bug upstream, not a model the scheme declines. Declining is the
            // *worst* available response to it — `use_*_scaling` stays false,
            // the engine reads that as "user supplied nothing", and the run
            // proceeds unscaled with the option accepted, which is gh #483
            // again. Assert it in debug builds so a mismatch is found here,
            // where it is one line, instead of as a slow solve later.
            debug_assert_eq!(
                sc.x.len(),
                req.x_scaling.len(),
                "curvature x-scaling sized {} for a {}-variable request",
                sc.x.len(),
                req.x_scaling.len()
            );
            debug_assert_eq!(
                sc.g.len(),
                req.g_scaling.len(),
                "curvature g-scaling sized {} for a {}-row request",
                sc.g.len(),
                req.g_scaling.len()
            );
            if sc.x.len() == req.x_scaling.len() {
                req.x_scaling.copy_from_slice(&sc.x);
                *req.use_x_scaling = true;
            }
            if sc.g.len() == req.g_scaling.len() {
                req.g_scaling.copy_from_slice(&sc.g);
                *req.use_g_scaling = true;
            }
        }
        // Objective 0 is the one `NlTnlp` evaluates (extra `O` segments
        // are parsed and ignored), so its entry is the objective scale.
        *req.obj_scaling = obj
            .and_then(|v| v.first().copied())
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        // A zero entry is AMPL's "untagged" default, not a scale factor.
        // An untagged component therefore falls back to the base: the
        // curvature factor when one was computed, and an explicit 1.0
        // otherwise — explicit because the callback's contract is to fill
        // the buffer, not to assume the caller pre-filled it with ones.
        if let Some(v) = var.filter(|v| v.len() == req.x_scaling.len()) {
            for (slot, &s) in req.x_scaling.iter_mut().zip(v) {
                if s != 0.0 {
                    *slot = s;
                } else if computed.is_none() {
                    *slot = 1.0;
                }
            }
            *req.use_x_scaling = true;
        } else if computed.is_none() {
            *req.use_x_scaling = false;
        }
        if let Some(g) = con.filter(|g| g.len() == req.g_scaling.len()) {
            for (slot, &s) in req.g_scaling.iter_mut().zip(g) {
                if s != 0.0 {
                    *slot = s;
                } else if computed.is_none() {
                    *slot = 1.0;
                }
            }
            *req.use_g_scaling = true;
        } else if computed.is_none() {
            *req.use_g_scaling = false;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        // Reuse the shared forward-value arena (sized to `max_tape_n`) so
        // each summand sweep allocates nothing — see `Tape::eval_into`.
        let (obj_tapes, vals) = (&self.obj_tapes, &mut self.vals_scratch);
        let mut nl: Number = 0.0;
        for t in obj_tapes {
            nl += t.eval_into(x, vals);
        }
        // A recognized objective has no tapes, so the loop above added
        // nothing and the form supplies the whole nonlinear part — including
        // the linear and constant terms AMPL folded into that tree, which is
        // why they are *not* also in `obj_linear` / `obj_constant`.
        if let Some(f) = self.quad.objective_form() {
            nl += self.quad.value(f, x);
        }
        let lin: Number = self.prob.obj_linear.iter().map(|(i, c)| c * x[*i]).sum();
        let v = self.prob.obj_constant + nl + lin;
        let signed = if self.prob.minimize { v } else { -v };
        Some(signed)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad.fill(0.0);
        // Reuse the forward-value / adjoint scratch arenas (sized to
        // `max_tape_n`) so each summand tape's reverse-AD sweep allocates
        // nothing — see `Tape::gradient_seed_into` (M18).
        for t in &self.obj_tapes {
            t.gradient_seed_into(x, 1.0, grad, &mut self.vals_scratch, &mut self.adj_scratch);
        }
        if let Some(f) = self.quad.objective_form() {
            self.quad.add_gradient(f, x, 1.0, grad);
        }
        for (i, c) in &self.prob.obj_linear {
            grad[*i] += c;
        }
        if !self.prob.minimize {
            for g in grad.iter_mut() {
                *g = -*g;
            }
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        // Constraint values are the line search's inner loop: on a
        // constraint-heavy model (m >> n) this runs ~10x per iteration over
        // every summand tape in the problem. Reuse the shared forward-value
        // arena so the sweep allocates nothing — the per-summand `Vec` the
        // allocating `Tape::eval` used to build was ~20% of `eval_g` on
        // Mittelmann's `robot_a` (52013 rows / 148037 summands). See
        // `Tape::eval_into`.
        let m = self.prob.m;
        let con_linear = &self.prob.con_linear;
        let quad = &self.quad;
        if let Some(h) = &mut self.con_hybrid {
            // Shared CSE bodies once for the whole constraint block, then one
            // local sweep per summand (pounce#476).
            let ConHybrid {
                tape,
                row_start,
                prelude_vals,
                local_vals,
                ..
            } = h;
            tape.forward_prelude(x, prelude_vals);
            for i in 0..m {
                let mut nl: Number = 0.0;
                for s in &tape.summands[row_start[i]..row_start[i + 1]] {
                    tape.forward_summand(s, x, prelude_vals, local_vals);
                    nl += tape.root_value(s, local_vals);
                }
                if let Some(f) = quad.row_form(i) {
                    nl += quad.value(f, x);
                }
                let lin: Number = con_linear[i].iter().map(|(j, c)| c * x[*j]).sum();
                g[i] = nl + lin;
            }
            return true;
        }
        let (con_tapes, vals) = (&self.con_tapes, &mut self.vals_scratch);
        for i in 0..m {
            let mut nl: Number = 0.0;
            for t in &con_tapes[i] {
                nl += t.eval_into(x, vals);
            }
            // A quadratic row's summand range is empty above, so this is its
            // whole nonlinear part: one matvec against a constant matrix in
            // place of a walk over every monomial's tape.
            if let Some(f) = quad.row_form(i) {
                nl += quad.value(f, x);
            }
            let lin: Number = con_linear[i].iter().map(|(j, c)| c * x[*j]).sum();
            g[i] = nl + lin;
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..self.prob.m {
                    for &j in &self.jac_cols[i] {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let n = self.prob.n;
                if self.scratch_row_grad.len() < n {
                    self.scratch_row_grad.resize(n, 0.0);
                }
                let Self {
                    prob,
                    con_tapes,
                    con_hybrid,
                    quad,
                    jac_cols,
                    scratch_row_grad,
                    vals_scratch,
                    adj_scratch,
                    ..
                } = self;
                let xs = x.unwrap_or(&prob.x0);
                let mut k = 0;
                // Shared-CSE path: the forward sweep over the CSE bodies runs
                // once for the whole constraint block instead of once per
                // referencing summand. The reverse sweep cannot be shared —
                // each row needs its own gradient — so a summand still walks
                // its own `prelude_reach` backwards.
                if let Some(h) = con_hybrid.as_mut().filter(|h| h.use_for_jac) {
                    let ConHybrid {
                        tape,
                        row_start,
                        prelude_vals,
                        local_vals,
                        local_adj,
                        prelude_adj,
                        ..
                    } = h;
                    tape.forward_prelude(xs, prelude_vals);
                    for i in 0..prob.m {
                        for &j in &jac_cols[i] {
                            scratch_row_grad[j] = 0.0;
                        }
                        for s in &tape.summands[row_start[i]..row_start[i + 1]] {
                            tape.forward_summand(s, xs, prelude_vals, local_vals);
                            tape.gradient_summand(
                                s,
                                prelude_vals,
                                local_vals,
                                1.0,
                                scratch_row_grad,
                                local_adj,
                                prelude_adj,
                            );
                        }
                        if let Some(f) = quad.row_form(i) {
                            quad.add_gradient(f, xs, 1.0, scratch_row_grad);
                        }
                        for &(v, c) in &prob.con_linear[i] {
                            scratch_row_grad[v] += c;
                        }
                        for &j in &jac_cols[i] {
                            values[k] = scratch_row_grad[j];
                            k += 1;
                        }
                    }
                    return true;
                }
                for i in 0..prob.m {
                    for &j in &jac_cols[i] {
                        scratch_row_grad[j] = 0.0;
                    }
                    for t in &con_tapes[i] {
                        // Allocation-free reverse-AD per summand tape (M18):
                        // reuse the shared forward/adjoint scratch arenas.
                        t.gradient_seed_into(xs, 1.0, scratch_row_grad, vals_scratch, adj_scratch);
                    }
                    // `Hx + a` for a quadratic row: one matvec over the
                    // row's support, in place of a reverse sweep per
                    // monomial. `jac_cols[i]` already covers the form's
                    // gradient support, so the scatter lands inside the
                    // window zeroed above.
                    if let Some(f) = quad.row_form(i) {
                        quad.add_gradient(f, xs, 1.0, scratch_row_grad);
                    }
                    for &(v, c) in &prob.con_linear[i] {
                        scratch_row_grad[v] += c;
                    }
                    for &j in &jac_cols[i] {
                        values[k] = scratch_row_grad[j];
                        k += 1;
                    }
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&self.h_irow);
                jcol.copy_from_slice(&self.h_jcol);
                true
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap_or(&self.prob.x0);
                values.fill(0.0);

                let obj_seed = if self.prob.minimize {
                    obj_factor
                } else {
                    -obj_factor
                };
                // Coloring path. For each (tape, weight) we do
                // one forward pass into `vals_scratch`, then one
                // forward-tangent+reverse-over-tangent per color
                // touched by that tape. Each pass accumulates a
                // weighted contribution of (H_tape · seed_c) into
                // `compressed[c]`. After all tapes done, we
                // decode each color's compressed vector into the
                // sparse `values` array.
                for buf in &mut self.compressed {
                    buf.fill(0.0);
                }

                // The constant blocks first: no forward sweep, no
                // directional product, no decode — the multipliers are the
                // only thing that changed since the model was read, so this
                // is one `values[slot] += w · h` pass per live form. Skipped
                // wholesale on a model with nothing recognized, so such a
                // model does not pay an `O(m)` scan for a structure that is
                // empty.
                if !self.quad.is_empty() {
                    if obj_seed != 0.0 {
                        if let Some(f) = self.quad.objective_form() {
                            self.quad.accumulate_hessian(f, obj_seed, values);
                        }
                    }
                    if let Some(lam) = lambda {
                        for i in 0..self.prob.m {
                            let w = lam[i];
                            if w == 0.0 {
                                continue;
                            }
                            if let Some(f) = self.quad.row_form(i) {
                                self.quad.accumulate_hessian(f, w, values);
                            }
                        }
                    }
                }

                if obj_seed != 0.0 {
                    for (ti, t) in self.obj_tapes.iter().enumerate() {
                        if t.ops.is_empty() {
                            continue;
                        }
                        t.forward_into(x, &mut self.vals_scratch);
                        for &c in &self.obj_tape_colors[ti] {
                            t.hessian_directional(
                                &self.vals_scratch,
                                &self.seeds[c as usize],
                                obj_seed,
                                &mut self.compressed[c as usize],
                                &mut self.dot_scratch,
                                &mut self.adj_scratch,
                                &mut self.adj_dot_scratch,
                            );
                        }
                    }
                }

                match (lambda, self.con_hybrid.as_mut()) {
                    // Shared-CSE path (issue #557). Per color the prelude's
                    // second-order work runs ONCE for the whole constraint
                    // block: one forward tangent, then — because
                    // reverse-over-tangent is linear in its adjoint seeds —
                    // one unit-weight reverse sweep over the λ-weighted
                    // adjoints accumulated by every summand of that color.
                    // The flat path below repeats both sweeps over the
                    // inlined CSE body once per referencing summand.
                    (Some(lam), Some(h)) if h.use_for_hess => {
                        let ConHybrid {
                            tape,
                            prelude_vals,
                            local_vals_all,
                            local_off,
                            summand_row,
                            hess_color_summands,
                            hess_color_reach,
                            hess_color_reach_off,
                            prelude_dot,
                            hess_prelude_adj,
                            prelude_adj_dot,
                            local_dot,
                            local_adj,
                            local_adj_dot,
                            ..
                        } = h;
                        // Forward once (values are color-independent):
                        // prelude for the block, then each summand of a row
                        // with a live multiplier into its packed slice.
                        tape.forward_prelude(x, prelude_vals);
                        for (si, s) in tape.summands.iter().enumerate() {
                            if lam[summand_row[si] as usize] == 0.0 {
                                continue;
                            }
                            tape.forward_summand(
                                s,
                                x,
                                prelude_vals,
                                &mut local_vals_all[local_off[si]..local_off[si + 1]],
                            );
                        }
                        for (c, list) in hess_color_summands.iter().enumerate() {
                            if !list
                                .iter()
                                .any(|&si| lam[summand_row[si as usize] as usize] != 0.0)
                            {
                                continue;
                            }
                            let seed = &self.seeds[c];
                            let out = &mut self.compressed[c];
                            let creach = &hess_color_reach
                                [hess_color_reach_off[c]..hess_color_reach_off[c + 1]];
                            tape.prelude_tangent(prelude_vals, seed, creach, prelude_dot);
                            for &si in list {
                                let si = si as usize;
                                let w = lam[summand_row[si] as usize];
                                if w == 0.0 {
                                    continue;
                                }
                                tape.hessian_summand_directional(
                                    &tape.summands[si],
                                    &local_vals_all[local_off[si]..local_off[si + 1]],
                                    prelude_dot,
                                    seed,
                                    w,
                                    out,
                                    local_dot,
                                    local_adj,
                                    local_adj_dot,
                                    hess_prelude_adj,
                                    prelude_adj_dot,
                                );
                            }
                            tape.prelude_reverse_directional(
                                prelude_vals,
                                prelude_dot,
                                creach,
                                out,
                                hess_prelude_adj,
                                prelude_adj_dot,
                            );
                        }
                    }
                    (Some(lam), _) => {
                        for k in 0..self.prob.m {
                            let w = lam[k];
                            if w == 0.0 {
                                continue;
                            }
                            for (ti, t) in self.con_tapes[k].iter().enumerate() {
                                if t.ops.is_empty() {
                                    continue;
                                }
                                t.forward_into(x, &mut self.vals_scratch);
                                for &c in &self.con_tape_colors[k][ti] {
                                    t.hessian_directional(
                                        &self.vals_scratch,
                                        &self.seeds[c as usize],
                                        w,
                                        &mut self.compressed[c as usize],
                                        &mut self.dot_scratch,
                                        &mut self.adj_scratch,
                                        &mut self.adj_dot_scratch,
                                    );
                                }
                            }
                        }
                    }
                    (None, _) => {}
                }

                // Decode each color's compressed Hessian-vector
                // result into the lower-triangle `values` array.
                for (c, table) in self.decoding.iter().enumerate() {
                    let comp = &self.compressed[c];
                    for w in table {
                        values[w.hess_idx as usize] += comp[w.row as usize];
                    }
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some(sol.x.to_vec());
        self.final_obj = sol.obj_value;
        self.final_lambda = Some(sol.lambda.to_vec());
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
    }

    /// Publish the `.col` / `.row` names (captured at load time) under the
    /// conventional `idx_names` metadata key, in original `.nl` order. The
    /// adapter permutes these into split space (see
    /// `OrigIpoptNlp::split_space_names`) so the debugger can report a
    /// near-singular Jacobian row as the `mass_balance` equation rather
    /// than "row 3" — the model-vs-index gap Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>) flag for equation-oriented
    /// model debugging. Declines (returns false) when the model shipped no
    /// name files so callers fall back to index labels.
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        let mut any = false;
        if !self.prob.var_names.is_empty() {
            var.strings
                .insert(IDX_NAMES.to_string(), self.prob.var_names.clone());
            any = true;
        }
        if !self.prob.con_names.is_empty() {
            con.strings
                .insert(IDX_NAMES.to_string(), self.prob.con_names.clone());
            any = true;
        }
        any
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        // A row is linear iff its nonlinear-part expression is the
        // identity zero — either left over from initial allocation ("no
        // `C<idx>` segment touched this row") or installed by the
        // constant-row-body fold in `parse_nl_text`, which shifts a
        // variable-free `C<idx>` body into the row bounds precisely so
        // that this test is a genuine linearity test and not just an
        // identity check (`gh #492`).
        for (i, t) in types.iter_mut().enumerate() {
            *t = if self.prob.con_nonlinear[i].is_trivially_zero() {
                Linearity::Linear
            } else {
                Linearity::NonLinear
            };
        }
        true
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        // Global linearity, per the upstream TNLP contract: a variable is
        // NonLinear iff it appears in the nonlinear part of the objective
        // or of any constraint; otherwise Linear. The parsed `.nl` splits
        // every row into a linear part (J/G coefficient list) and a
        // nonlinear expression, so the set of nonlinear variables is
        // exactly the structural union of `collect_vars` over
        // `obj_nonlinear` and every `con_nonlinear` row. A variable touched
        // only by a linear part — or not referenced at all — is Linear.
        //
        let nonlinear = self.nonlinear_var_set();
        for (i, t) in types.iter_mut().enumerate() {
            *t = if nonlinear.contains(&i) {
                Linearity::NonLinear
            } else {
                Linearity::Linear
            };
        }
        true
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        // Objective-scoped variant of `get_variables_linearity`: only
        // `obj_nonlinear` contributes. This is what engages the presolve
        // auxiliary-elimination safeguard (pounce-presolve H11): a variable
        // that is nonlinear in the objective but happens to have a zero
        // gradient at the single probe point (e.g. `f = (x - x0)^2`
        // warm-started at `x0`) is kept in the objective support instead of
        // being mis-classified objective-free and eliminated. A variable
        // that is nonlinear only in *constraints* stays `Linear` here, so
        // the guard does not block legitimate eliminations of
        // objective-free equality blocks (the gas-network case).
        let mut nonlinear: BTreeSet<usize> = BTreeSet::new();
        self.prob.obj_nonlinear.collect_vars(&mut nonlinear);
        for (i, t) in types.iter_mut().enumerate() {
            *t = if nonlinear.contains(&i) {
                Linearity::NonLinear
            } else {
                Linearity::Linear
            };
        }
        true
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.nonlinear_variables().len() as Index
    }

    fn get_list_of_nonlinear_variables(&mut self, pos_nonlin_vars: &mut [Index]) -> bool {
        let list = self.nonlinear_variables();
        if pos_nonlin_vars.len() < list.len() {
            return false;
        }
        pos_nonlin_vars[..list.len()].copy_from_slice(&list);
        true
    }

    fn derivative_proofs(&mut self) -> DerivativeProofs {
        // Degree is the whole argument (gh #588, Q6). A body proved
        // degree ≤ 1 has a constant gradient and no second derivative; a
        // body proved degree 2 has a nonzero second derivative, hence a
        // gradient that moves. A body the recognizer refuses is
        // `Unknown` — and `Unknown` must stay `Unknown`, because the
        // refusal is structural, not a finding of nonlinearity.
        fn proof(affine: Option<bool>) -> DerivativeProof {
            match affine {
                Some(true) => DerivativeProof::Constant,
                Some(false) => DerivativeProof::Varying,
                None => DerivativeProof::Unknown,
            }
        }
        let obj_affine = self.prob.obj_nonlinear.provably_affine();
        let jac: Vec<DerivativeProof> = self
            .prob
            .con_nonlinear
            .iter()
            .map(|b| proof(b.provably_affine()))
            .collect();

        // `∇²L = σ·∇²f + Σᵢ λᵢ·∇²gᵢ`. One row proved genuinely quadratic
        // makes `∇²L` a non-constant function of `λ` — this is the QCQP
        // case, and it is exactly the assertion Ipopt would honour and
        // get wrong (§4 Lever 2). Otherwise every row must be *proved*
        // affine and the objective *proved* degree ≤ 2, at which point
        // `∇²L = σ·∇²f`, constant for a given `σ`; the caller keys its
        // reuse on `σ` because the restoration phase passes a different
        // one.
        let hessian = if jac.iter().any(|&p| p == DerivativeProof::Varying) {
            DerivativeProof::Varying
        } else if obj_affine.is_some() && jac.iter().all(|&p| p == DerivativeProof::Constant) {
            DerivativeProof::Constant
        } else {
            DerivativeProof::Unknown
        };

        DerivativeProofs {
            grad_f: proof(obj_affine),
            hessian,
            jac,
        }
    }
}

/// Convenience: read an `.nl` file and build a TNLP-compatible Rc.
pub fn load_nl_as_tnlp(path: &Path) -> Result<Rc<RefCell<dyn TNLP>>, String> {
    let prob = read_nl_file(path)?;
    Ok(Rc::new(RefCell::new(NlTnlp::new(prob))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time guarantee for the batched-solve path (pounce#126):
    /// a parsed problem and the TNLP built from it must be movable to a
    /// rayon worker. Regresses if anyone reintroduces an `Rc` (or other
    /// `!Send` state) into the `Expr` DAG / tape pipeline.
    #[test]
    fn nl_problem_and_tnlp_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NlProblem>();
        assert_send::<NlTnlp>();
        assert_send::<Expr>();
    }

    /// `variant()` patches starting point / bounds on a clone and
    /// validates override lengths; the base instance is untouched.
    #[test]
    fn variant_overrides_bounds_and_x0() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let base = NlTnlp::new(p);
        let var = base
            .variant(&NlVariation {
                x0: Some(vec![3.0, 4.0]),
                x_l: Some(vec![-1.0, -2.0]),
                x_u: Some(vec![5.0, 6.0]),
                ..Default::default()
            })
            .expect("variant");
        let mut var = var;
        let (mut x_l, mut x_u) = ([0.0; 2], [0.0; 2]);
        let (mut g_l, mut g_u) = ([0.0; 0], [0.0; 0]);
        assert!(var.get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        }));
        assert_eq!(x_l, [-1.0, -2.0]);
        assert_eq!(x_u, [5.0, 6.0]);
        let mut x = [0.0; 2];
        let (mut zl, mut zu, mut lam) = ([0.0; 2], [0.0; 2], [0.0; 0]);
        assert!(var.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl,
            z_u: &mut zu,
            init_lambda: false,
            lambda: &mut lam,
        }));
        assert_eq!(x, [3.0, 4.0]);
        // Base keeps its parsed (free) bounds.
        assert!(base.problem().x_l[0] < -1.0e18);
        // Length mismatch is an error, not a panic.
        assert!(
            base.variant(&NlVariation {
                x0: Some(vec![1.0]),
                ..Default::default()
            })
            .is_err()
        );
    }

    /// `min (x0 - 1)^2 + (x1 - 2)^2` written in `.nl` ASCII form.
    /// Header values:
    ///   line 2: n=2 m=0 num_obj=1 0 0
    ///   line 3: 0 1   (1 nonlinear objective)
    ///   line 4: 0 0
    ///   line 5: 0 2 0 (nonlinear vars in obj=2)
    ///   line 6: 0 0 0 1
    ///   line 7: 0 0 0 0 0
    ///   line 8: 0 0   (no Jacobian nonzeros, no linear obj)
    ///   line 9: 0 0
    ///   line 10: 0 0 0 0 0
    /// Then `O0 0` followed by an expression tree:
    /// `(x0 - 1)^2 + (x1 - 2)^2` =
    ///   o0
    ///     o5 (o1 v0 n1) n2
    ///     o5 (o1 v1 n2) n2
    /// Then `b` segment: free for both.
    const SIMPLE: &str = "g3 0 1 0
2 0 1 0 0
0 1
0 0
0 2 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 0 0 0 0
O0 0
o0
o5
o1
v0
n1
n2
o5
o1
v1
n2
n2
b
3
3
";

    #[test]
    fn parses_simple_quadratic() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        assert_eq!(p.n, 2);
        assert_eq!(p.m, 0);
        assert_eq!(p.num_obj, 1);
        // f(0,0) = 1 + 4 = 5
        let f = eval_expr(&p.obj_expr(), &[0.0, 0.0]);
        assert!((f - 5.0).abs() < 1e-12);
        // f(1,2) = 0
        let f = eval_expr(&p.obj_expr(), &[1.0, 2.0]);
        assert!(f.abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_analytic() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let x = [0.5, 1.0];
        let mut g = [0.0_f64; 2];
        grad_expr(&p.obj_expr(), &x, 1.0, &mut g);
        // d/dx0 = 2*(x0-1) = -1.0
        // d/dx1 = 2*(x1-2) = -2.0
        assert!((g[0] - (-1.0)).abs() < 1e-12);
        assert!((g[1] - (-2.0)).abs() < 1e-12);
    }

    /// F3 (H11 dormant): `NlTnlp` must answer `get_variables_linearity`
    /// with global semantics so the presolve auxiliary-elimination
    /// safeguard actually engages. Pre-fix the default trait stub returned
    /// `false` and left the slice untouched, so a variable that is
    /// nonlinear in the objective but zero-gradient at the probe point
    /// could be wrongly eliminated.
    ///
    /// Problem: `min (x0 - 1)^2 + 3*x1`. x0 appears in the nonlinear part
    /// of the objective (NonLinear); x1 appears only in the linear part
    /// (Linear).
    #[test]
    fn variables_linearity_tags_obj_nonlinear_vs_linear_vars() {
        // (x0 - 1)^2
        let obj_nl = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            src: None,
            cse_bodies: Vec::new(),
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: NlBody::Tree(obj_nl),
            obj_linear: vec![(1, 3.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![f64::NEG_INFINITY; 2],
            x_u: vec![f64::INFINITY; 2],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0; 2],
            lambda0: vec![],
            suffixes: NlSuffixes::default(),
            imported_funcs: vec![],
            ampl_options: vec![],
            nl_counts: None,
            var_names: vec![],
            con_names: vec![],
        };
        let mut tnlp = NlTnlp::new(prob);
        let mut types = vec![Linearity::Linear; 2];
        let ok = tnlp.get_variables_linearity(&mut types);
        // Pre-fix: default stub returns false (slice untouched).
        assert!(
            ok,
            "get_variables_linearity must report it filled the slice"
        );
        assert!(
            matches!(types[0], Linearity::NonLinear),
            "x0 is nonlinear in the objective"
        );
        assert!(
            matches!(types[1], Linearity::Linear),
            "x1 appears only in the linear part"
        );
    }

    /// Objective-scoped linearity must NOT inherit constraint
    /// nonlinearity. `min 3*x1 s.t. x0^2 = 4`: x0 is nonlinear globally
    /// (constraint tape) but linear w.r.t. the objective, so the presolve
    /// H11 guard must not treat it as objective-coupled — that was the CI
    /// regression where every gas-network variable (nonlinear in the flow
    /// equations, absent from the linear objective) blocked Phase-0
    /// elimination.
    #[test]
    fn objective_variables_linearity_ignores_constraint_nonlinearity() {
        // x0^2
        let con_nl = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            src: None,
            cse_bodies: Vec::new(),
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: NlBody::Tree(Expr::Const(0.0)),
            obj_linear: vec![(1, 3.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![NlBody::Tree(con_nl)],
            con_linear: vec![vec![]],
            x_l: vec![f64::NEG_INFINITY; 2],
            x_u: vec![f64::INFINITY; 2],
            g_l: vec![4.0],
            g_u: vec![4.0],
            x0: vec![0.0; 2],
            lambda0: vec![0.0],
            suffixes: NlSuffixes::default(),
            imported_funcs: vec![],
            ampl_options: vec![],
            nl_counts: None,
            var_names: vec![],
            con_names: vec![],
        };
        let mut tnlp = NlTnlp::new(prob);

        let mut global = vec![Linearity::Linear; 2];
        assert!(tnlp.get_variables_linearity(&mut global));
        assert!(
            matches!(global[0], Linearity::NonLinear),
            "global tags see x0's constraint nonlinearity"
        );

        let mut obj = vec![Linearity::NonLinear; 2];
        assert!(tnlp.get_objective_variables_linearity(&mut obj));
        assert!(
            matches!(obj[0], Linearity::Linear),
            "x0 is linear w.r.t. the objective despite the nonlinear constraint"
        );
        assert!(
            matches!(obj[1], Linearity::Linear),
            "x1 is linear everywhere"
        );
    }

    /// Header lines 3 and 5 land in [`NlCounts`], with the field order the
    /// format documents: `nlc nlo` then `nlvc nlvo nlvb`. `SIMPLE` is
    /// `min (x0-1)^2 + (x1-2)^2`, so one nonlinear objective, no nonlinear
    /// constraints, and both variables nonlinear in the objective only.
    #[test]
    fn header_census_is_parsed() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let c = p.nl_counts.expect("SIMPLE has a well-formed header");
        assert_eq!(c.nl_cons, 0);
        assert_eq!(c.nl_objs, 1);
        assert_eq!((c.nl_vars_cons, c.nl_vars_objs, c.nl_vars_both), (0, 2, 0));
        assert_eq!(c.nonlinear_vars(), 2);
    }

    /// `nlvb` is inside both `nlvc` and `nlvo`, so the total is
    /// `nlvc + nlvo − nlvb`. The two degenerate directions matter as much
    /// as the overlapping one: disjoint sets add, and `max` would be wrong
    /// for them.
    #[test]
    fn nonlinear_var_total_uses_inclusion_exclusion() {
        let c = |vc, vo, vb| NlCounts {
            nl_cons: 0,
            nl_objs: 0,
            nl_vars_cons: vc,
            nl_vars_objs: vo,
            nl_vars_both: vb,
        };
        // Fully shared: the same 5 variables in both.
        assert_eq!(c(5, 5, 5).nonlinear_vars(), 5);
        // Disjoint: `min x0^2 s.t. x1^2 <= 1` is two nonlinear variables,
        // not the `max(nlvc, nlvo) = 1` a naive reading gives.
        assert_eq!(c(1, 1, 0).nonlinear_vars(), 2);
        // Partial overlap: 4 + 3 - 2.
        assert_eq!(c(4, 3, 2).nonlinear_vars(), 5);
        // Nonsense header: saturates instead of underflowing.
        assert_eq!(c(1, 1, 9).nonlinear_vars(), 0);
    }

    /// A header that does not carry the documented fields records no
    /// census at all rather than a guess of zero — "no nonlinear
    /// variables" is a claim, and a truncated header has not made it.
    #[test]
    fn short_header_line_records_no_census() {
        // Line 5 with two fields instead of `nlvc nlvo nlvb`.
        let txt = SIMPLE.replacen("0 2 0\n", "0 2\n", 1);
        assert_ne!(txt, SIMPLE, "the substitution must have applied");
        let p = parse_nl_text(&txt).expect("parse");
        assert!(p.nl_counts.is_none());
    }

    /// `get_number_of_nonlinear_variables` used to be the trait default,
    /// `-1` ("assume everything is nonlinear"). It now answers from the
    /// trees, and `get_list_of_nonlinear_variables` agrees with it.
    ///
    /// `min (x0 - 1)^2 + 3*x1`: x0 is nonlinear, x1 is not.
    #[test]
    fn nonlinear_variable_list_excludes_linear_columns() {
        let obj_nl = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let parts = NlProblemParts {
            minimize: true,
            objective: obj_nl,
            obj_constant: 0.0,
            constraints: vec![],
            x_l: vec![-1e19; 2],
            x_u: vec![1e19; 2],
            x0: vec![0.0; 2],
            g_l: vec![],
            g_u: vec![],
            var_names: vec![],
            con_names: vec![],
        };
        let prob = NlProblem::from_expressions(parts).expect("build");
        assert!(
            prob.nl_counts.is_none(),
            "a model built in memory has no header to read"
        );
        let mut tnlp = NlTnlp::new(prob);
        assert_eq!(tnlp.get_number_of_nonlinear_variables(), 1);
        let mut list = [-1 as Index; 2];
        assert!(tnlp.get_list_of_nonlinear_variables(&mut list));
        assert_eq!(list[0], 0);
    }

    /// When the header says every variable is nonlinear the walk is
    /// skipped, and the answer is the same one the walk would give for
    /// `SIMPLE` (both variables appear in the objective's nonlinear part).
    #[test]
    fn all_nonlinear_header_short_circuits_to_n() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        assert_eq!(p.nl_counts.expect("census").nonlinear_vars(), p.n);
        let mut tnlp = NlTnlp::new(p);
        assert_eq!(tnlp.get_number_of_nonlinear_variables(), 2);
        let mut list = [-1 as Index; 2];
        assert!(tnlp.get_list_of_nonlinear_variables(&mut list));
        assert_eq!(list, [0, 1]);
    }

    /// The list must not be written when the caller's slice is too small —
    /// the contract's `false` return, not a panic.
    #[test]
    fn nonlinear_variable_list_declines_a_short_slice() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let mut tnlp = NlTnlp::new(p);
        let mut list = [-1 as Index; 1];
        assert!(!tnlp.get_list_of_nonlinear_variables(&mut list));
        assert_eq!(list, [-1]);
    }

    /// `min x0^2 + x1^2  s.t.  x0 + x1 = 1`.
    /// One equality constraint with a purely linear Jacobian — exercises
    /// the constrained path (`eval_g`, `eval_jac_g`, `r`-segment bound
    /// kind 4).
    ///
    /// Header layout:
    ///   line 1: g3 0 1 0
    ///   line 2: 2 1 1 0 0   (n=2, m=1, num_obj=1)
    ///   line 3: 0 1         (1 nonlinear obj, 0 nonlinear cons)
    ///   line 4: 0 0
    ///   line 5: 0 2 0       (nonlinear vars in obj=2)
    ///   line 6: 0 0 0 1
    ///   line 7: 0 0 0 0 0
    ///   line 8: 2 0         (Jacobian nnz=2, no linear obj)
    ///   line 9: 0 0
    ///   line 10: 0 0 0 0 0
    /// Then C0 = const 0 (no nonlinear part), O0 = x0^2 + x1^2,
    /// r-segment kind 4 (eq) value 1, b-segment free, k-segment, J-row.
    const EQ_LIN: &str = "g3 0 1 0
2 1 1 0 0
0 1
0 0
0 2 0
0 0 0 1
0 0 0 0 0
2 0
0 0
0 0 0 0 0
C0
n0
O0 0
o0
o5
v0
n2
o5
v1
n2
r
4 1
b
3
3
k1
2
J0 2
0 1
1 1
";

    #[test]
    fn parses_constrained_problem() {
        let p = parse_nl_text(EQ_LIN).expect("parse");
        assert_eq!(p.n, 2);
        assert_eq!(p.m, 1);
        // r-segment kind 4 (equality with rhs=1).
        assert!((p.g_l[0] - 1.0).abs() < 1e-12);
        assert!((p.g_u[0] - 1.0).abs() < 1e-12);
        // J-row 0: x0 (coef 1), x1 (coef 1).
        assert_eq!(p.con_linear[0], vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn malformed_j_variable_index_is_parse_error_not_panic() {
        // Code review L32: a J-segment entry's variable (column) index was
        // pushed into con_linear unchecked, so an out-of-range index (here 5
        // with n=2) flowed through to a slice OOB panic (`x[*j]`) during
        // constraint evaluation. It must instead surface as a clean parse
        // error, consistent with the existing `J<row> out of range` check.
        let bad = EQ_LIN.replace("J0 2\n0 1\n1 1\n", "J0 2\n0 1\n5 1\n");
        assert_ne!(bad, EQ_LIN, "fixture substitution must apply");
        let err = parse_nl_text(&bad).expect_err("out-of-range J var must error");
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    #[test]
    fn out_of_range_x_segment_index_is_parse_error() {
        // Same strictness for the initial-primal `x` segment: an index past
        // `n` used to be silently dropped; now it is a parse error, so the
        // four index-bearing segments (J/G/x/d) behave consistently.
        let bad = format!("{EQ_LIN}x1\n5 0.5\n");
        let err = parse_nl_text(&bad).expect_err("out-of-range x index must error");
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    // ---------------------------------------------------------------
    // gh #492 — a constant `C<i>` body folds into the row bounds.
    //
    // `EQ_LIN` is `x0 + x1 = 1` with an empty `C0` (`n0`) and the row
    // bound in the `r` segment (`4 1`). Rewriting `C0` gives a family of
    // constant-body rows to fold.
    // ---------------------------------------------------------------

    /// Linearity is a *linearity* test, not "did a `C` segment touch this
    /// row". A row whose body is the bare constant `3` is affine.
    #[test]
    fn a_constant_row_body_folds_into_both_bounds_and_reads_linear() {
        // `x0 + x1 + 3 = 1`  ⇔  `x0 + x1 = -2`.
        let nl = EQ_LIN.replace("C0\nn0\n", "C0\nn3\n");
        assert_ne!(nl, EQ_LIN, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse");

        assert!(
            p.con_nonlinear[0].is_trivially_zero(),
            "the constant body must be replaced by the identity zero, got {:?}",
            p.con_nonlinear[0]
        );
        assert!((p.g_l[0] - (-2.0)).abs() < 1e-12, "g_l = {}", p.g_l[0]);
        assert!((p.g_u[0] - (-2.0)).abs() < 1e-12, "g_u = {}", p.g_u[0]);

        let mut lin = [Linearity::NonLinear];
        let mut t = NlTnlp::new(p);
        assert!(t.get_constraints_linearity(&mut lin));
        assert_eq!(lin[0], Linearity::Linear);
    }

    /// The shift must be exact, not merely "linear now": the folded model
    /// and the hand-folded one must be the same problem, row body for row
    /// body and bound for bound. That is what keeps feasibility, the
    /// active set, and the duals unchanged.
    #[test]
    fn folding_a_row_constant_gives_the_hand_folded_problem() {
        // `x0 + x1 + 3 = 1` against `x0 + x1 = -2`, written directly.
        let offset = EQ_LIN.replace("C0\nn0\n", "C0\nn3\n");
        let folded = EQ_LIN.replace("r\n4 1\n", "r\n4 -2\n");
        assert_ne!(offset, EQ_LIN);
        assert_ne!(folded, EQ_LIN);

        let a = parse_nl_text(&offset).expect("parse offset form");
        let b = parse_nl_text(&folded).expect("parse folded form");
        assert_eq!(a.g_l, b.g_l);
        assert_eq!(a.g_u, b.g_u);
        assert_eq!(a.con_linear, b.con_linear);

        // And the row *values* agree pointwise, which is the property the
        // duals ride on: `g(x) - g_l` is the same residual either way.
        let mut ga = [0.0];
        let mut gb = [0.0];
        let x = [0.75, -1.25];
        assert!(NlTnlp::new(a).eval_g(&x, true, &mut ga));
        assert!(NlTnlp::new(b).eval_g(&x, true, &mut gb));
        assert!((ga[0] - gb[0]).abs() < 1e-12, "{ga:?} vs {gb:?}");
    }

    /// The fold is by *evaluation*, not by syntax: `o0 n1 n2` is as
    /// constant as `n3`, and AMPL emits such trees when a expression
    /// collapses without being re-simplified.
    #[test]
    fn a_row_body_that_evaluates_to_a_constant_folds_too() {
        // `C0` = `1 + 2`.
        let nl = EQ_LIN.replace("C0\nn0\n", "C0\no0\nn1\nn2\n");
        assert_ne!(nl, EQ_LIN, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse");
        assert!(p.con_nonlinear[0].is_trivially_zero());
        assert!((p.g_l[0] - (-2.0)).abs() < 1e-12, "g_l = {}", p.g_l[0]);
        assert!((p.g_u[0] - (-2.0)).abs() < 1e-12, "g_u = {}", p.g_u[0]);
    }

    /// Bound presence is directional (gh #401): the ±1e19 sentinels mean
    /// "absent", not "a very large number". Shifting one turns it into a
    /// real bound and invents a constraint that is not in the model.
    ///
    /// The constants here are deliberately huge. An everyday `3` is
    /// absorbed by the sentinel's own ULP (2048 at 1e19), so a missing
    /// presence guard would go unnoticed at ordinary magnitudes and then
    /// bite on a model that scales its rows. The guard is what makes the
    /// sentinel untouchable at *any* magnitude.
    #[test]
    fn folding_a_row_constant_leaves_the_absent_bound_sentinel_alone() {
        // `x0 + x1 - 1e18 <= 1`: upper-bounded row (`r` kind 1), no lower
        // bound. Shifting the lower sentinel would leave `-9e18`, a real
        // bound, so the row would gain a floor the model never stated.
        let nl = EQ_LIN
            .replace("C0\nn0\n", "C0\nn-1e18\n")
            .replace("r\n4 1\n", "r\n1 1\n");
        let p = parse_nl_text(&nl).expect("parse");
        assert!((p.g_u[0] - 1.0e18).abs() < 1024.0, "g_u = {}", p.g_u[0]);
        assert!(
            !lower_bound_present(p.g_l[0]),
            "the absent-lower sentinel became a real bound: {}",
            p.g_l[0]
        );

        // The mirror case: a positive constant on a lower-bounded row is
        // the one that would pull the *upper* sentinel below 1e19.
        let nl = EQ_LIN
            .replace("C0\nn0\n", "C0\nn1e18\n")
            .replace("r\n4 1\n", "r\n2 1\n");
        let p = parse_nl_text(&nl).expect("parse");
        assert!((p.g_l[0] + 1.0e18).abs() < 1024.0, "g_l = {}", p.g_l[0]);
        assert!(
            !upper_bound_present(p.g_u[0]),
            "the absent-upper sentinel became a real bound: {}",
            p.g_u[0]
        );
    }

    /// A row body that mentions a variable is not a constant, however
    /// simple it looks. Folding it would delete the term.
    #[test]
    fn a_row_body_with_a_variable_is_not_folded() {
        let nl = EQ_LIN.replace("C0\nn0\n", "C0\no5\nv0\nn2\n"); // x0²
        assert_ne!(nl, EQ_LIN, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse");
        assert!(
            !matches!(p.con_nonlinear[0].tree(), Some(Expr::Const(_))),
            "a row in x0 was folded away: {:?}",
            p.con_nonlinear[0]
        );
        assert!((p.g_l[0] - 1.0).abs() < 1e-12, "bounds moved: {}", p.g_l[0]);
        assert!((p.g_u[0] - 1.0).abs() < 1e-12, "bounds moved: {}", p.g_u[0]);
    }

    /// A variable-free body whose value is not finite is left in place. It
    /// makes the row infeasible (or ill-posed) and that is the solver's
    /// verdict to report; pushing a NaN into `g_l`/`g_u` would instead
    /// corrupt the bound pair and take every downstream presence test with
    /// it.
    #[test]
    fn a_non_finite_constant_row_body_is_not_folded() {
        // `C0` = `log(-1)` = NaN.
        let nl = EQ_LIN.replace("C0\nn0\n", "C0\no43\nn-1\n");
        assert_ne!(nl, EQ_LIN, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse");
        assert!(
            !p.con_nonlinear[0].is_trivially_zero(),
            "a NaN body was folded into the bounds"
        );
        assert!(p.g_l[0].is_finite() && p.g_u[0].is_finite());
        assert!((p.g_l[0] - 1.0).abs() < 1e-12);
    }

    /// An imported-function call is not a parse-time constant even with
    /// constant arguments — it is resolved to a shared library much later
    /// (`nl_external::ExternalResolver`), and `eval_expr` panics on it
    /// rather than guess. The fold must decline before it evaluates.
    #[test]
    fn a_constant_argument_funcall_row_body_is_not_folded() {
        // Declare one imported function and call it with a literal.
        let nl = EQ_LIN.replace("C0\nn0\n", "F0 1 1 myfunc\nC0\nf0 1\nn2.0\n");
        assert_ne!(nl, EQ_LIN, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse");
        assert!(
            matches!(p.con_nonlinear[0].tree(), Some(Expr::Funcall { .. })),
            "expected the funcall to survive the fold, got {:?}",
            p.con_nonlinear[0]
        );
        assert!((p.g_l[0] - 1.0).abs() < 1e-12, "bounds moved: {}", p.g_l[0]);
    }

    #[test]
    fn k_segment_nonstandard_count_is_parse_error_at_source() {
        // Code review L35: the `k` (Jacobian column-count) segment header
        // declares how many count lines follow — `k<count>` — and the
        // standard value is n-1. The parser used to *assume* n-1 and ignore
        // the header, so a file declaring a different count read the wrong
        // number of data lines, desynced the segment stream, and failed far
        // downstream with a confusing error (or silently mis-parsed). With
        // the declared count now read and validated, a nonstandard count is
        // a clear parse error at its source. Here EQ_LIN has n=2 (expected
        // count 1); rewrite its `k1` + one count line to `k0`.
        let bad = EQ_LIN.replace("k1\n2\n", "k0\n");
        assert_ne!(bad, EQ_LIN, "fixture substitution must apply");
        let err = parse_nl_text(&bad).expect_err("nonstandard k count must error");
        assert!(
            err.contains("k-segment declares"),
            "expected a clear k-segment count error, got: {err}"
        );
    }

    #[test]
    fn get_starting_point_returns_nl_initial_duals() {
        // Code review 2026-06 item M19: the `.nl` `d` segment supplies
        // initial constraint multipliers. They are parsed into `lambda0`,
        // but `get_starting_point` previously ignored them — so a
        // `warm_start_init_point yes` solve silently began from zero duals.
        // `get_starting_point` must hand the parsed duals back when the
        // engine requests them (`init_lambda`), and leave the buffer
        // untouched when it does not.
        let nl = format!("{EQ_LIN}\nd1\n0 2.5\n");
        let p = parse_nl_text(&nl).expect("parse");
        assert_eq!(p.lambda0, vec![2.5], "the `d` segment fills lambda0");

        let mut t = NlTnlp::new(p);
        let info = t.get_nlp_info().unwrap();
        let (n, m) = (info.n as usize, info.m as usize);

        // Warm-start request: init_lambda = true → the parsed `.nl` duals
        // must be returned (pre-fix this stayed zero).
        let mut x = vec![0.0; n];
        let mut z_l = vec![0.0; n];
        let mut z_u = vec![0.0; n];
        let mut lambda = vec![0.0; m];
        assert!(t.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: true,
            lambda: &mut lambda,
        }));
        assert_eq!(
            lambda,
            vec![2.5],
            "a warm start must use the `.nl` initial duals, not zero"
        );

        // No warm-start request: the multiplier buffer is left alone (the
        // engine owns its default), so honoring the flag does not clobber it.
        let mut lambda_untouched = vec![7.0; m];
        assert!(t.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: false,
            lambda: &mut lambda_untouched,
        }));
        assert_eq!(
            lambda_untouched,
            vec![7.0],
            "without init_lambda the multiplier buffer must be untouched"
        );
    }

    /// `.nl` text for `minimize sum_j (x_j - 1)^2 + x_0 * sum_j x_j`,
    /// unconstrained, `n` variables. The trailing product puts a nonzero
    /// in row 0 of *every* Hessian column, which is the shape that used
    /// to force the greedy coloring to hand out one color per variable.
    fn dense_row_objective_nl(n: usize) -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} 0 1 0 0 0\n"));
        s.push_str(" 0 1\n 0 0\n");
        s.push_str(&format!(" {n} {n} {n}\n"));
        s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
        s.push_str(&format!(" 0 {n}\n"));
        s.push_str(" 0 0\n 0 0 0 0 0\n");
        // objective
        s.push_str("O0 0\n");
        s.push_str(&format!("o54\n{}\n", n + 1));
        for j in 0..n {
            s.push_str(&format!("o5\no1\nv{j}\nn1.0\nn2\n"));
        }
        s.push_str(&format!("o2\nv0\no54\n{n}\n"));
        for j in 0..n {
            s.push_str(&format!("v{j}\n"));
        }
        // start, bounds, gradient
        s.push_str(&format!("x{n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.5\n"));
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        s.push_str(&format!("G0 {n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.0\n"));
        }
        s
    }

    /// Same shape as [`dense_row_objective_nl`], but the coupling term
    /// carries per-variable weights: `sum_j (x_j - 1)^2 + x_0 * sum_j w_j
    /// x_j`, so `H[j, 0] == w_j`. Spreading `w` over `span` orders of
    /// magnitude makes the dense column ill-scaled — the case where
    /// reading its entries out of its own pass costs real digits.
    fn weighted_dense_row_objective_nl(n: usize, span: f64) -> String {
        let w = |j: usize| 10_f64.powf(span / 2.0 - span * j as f64 / (n - 1) as f64);
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} 0 1 0 0 0\n"));
        s.push_str(" 0 1\n 0 0\n");
        s.push_str(&format!(" {n} {n} {n}\n"));
        s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
        s.push_str(&format!(" 0 {n}\n"));
        s.push_str(" 0 0\n 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str(&format!("o54\n{}\n", n + 1));
        for j in 0..n {
            s.push_str(&format!("o5\no1\nv{j}\nn1.0\nn2\n"));
        }
        s.push_str(&format!("o2\nv0\no54\n{n}\n"));
        for j in 0..n {
            s.push_str(&format!("o2\nn{:.17e}\nv{j}\n", w(j)));
        }
        s.push_str(&format!("x{n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.5\n"));
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        s.push_str(&format!("G0 {n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.0\n"));
        }
        s
    }

    /// Locate a model in the benchmark corpus, or `None` if the corpus is
    /// not on this machine.
    ///
    /// The corpus is ~2 GB and deliberately outside the checkout (see
    /// `POUNCE_BENCH_DATA`), so tests that need it have to degrade to a
    /// no-op rather than fail. That is a real limitation — a check that
    /// silently does nothing is how the gap below went unnoticed in the
    /// first place — so anything using this must also be covered by a
    /// synthetic case that always runs.
    fn bench_model(rel: &str) -> Option<std::path::PathBuf> {
        let root = std::env::var("POUNCE_BENCH_DATA").ok()?;
        let p = std::path::PathBuf::from(root).join(rel);
        p.is_file().then_some(p)
    }

    /// The corpus check the dense-column optimization never got.
    ///
    /// `cho_parmest` is the model whose certificate the optimization cost,
    /// and it could not have caught it: the model is 4.3 MB and lives in
    /// the benchmark data set, not in the repository, so validating
    /// against the in-repo `.nl` fixtures said nothing about it. This test
    /// closes that by checking the corpus directly wherever the corpus
    /// exists, which is every machine and CI job that runs the benchmarks.
    ///
    /// The assertion is the one that matters and the one that was never
    /// made: whatever the guard leaves peeled must decode to the same
    /// Hessian as peeling nothing. Against the pre-fix code this fails —
    /// 48,931 of the 96,000 entries disagreed, to 5.75e-12 relative.
    #[test]
    fn cho_parmest_decodes_to_its_unpeeled_reference() {
        let Some(path) = bench_model("cho/nl_export_results/cho_parmest.nl") else {
            eprintln!("POUNCE_BENCH_DATA/cho not present — skipping corpus check");
            return;
        };
        let p = read_nl_file(&path).expect("read cho_parmest");
        let n = p.n;
        let mut t = NlTnlp::new(p);
        // The guard vetoes 7 of cho's 12 peeled columns, and putting those
        // seven dense rows back into the conflict graph costs the coloring
        // outright: it goes from 17 colors to 9010, and no column clears the
        // density threshold afterwards, so the model ends up fully unpeeled.
        // That is the price of the certificate on this model, and it is worth
        // knowing rather than assuming the other five survive.
        assert!(t.peeled_cols.is_empty());

        let info = t.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0_i32; nnz], vec![0_i32; nnz]);
        assert!(t.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        // Evaluate away from `x0` with non-uniform multipliers. The damage
        // is point-dependent, and `x0` is close to where it is least
        // visible: only 5 entries move there, against 48,931 here. Note the
        // guard's own probe runs at `x0` — it fires anyway because it tests
        // a bound on what a pass *could* lose, not the damage it happens to
        // commit at one point. That is the property that makes it robust,
        // and this asymmetry is worth keeping in front of anyone who
        // retunes it.
        let x: Vec<f64> = t
            .prob
            .x0
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.01 * (i % 7) as f64 + 0.001)
            .collect();
        let lambda: Vec<f64> = (0..t.prob.m).map(|i| 0.5 + 0.01 * (i % 5) as f64).collect();

        let mut got = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values { values: &mut got }
        ));

        t.recolor(&vec![true; n]);
        assert!(t.peeled_cols.is_empty());
        let mut want = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values { values: &mut want }
        ));

        let scale = want.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let mut worst = 0.0_f64;
        let mut at = 0usize;
        for k in 0..nnz {
            let rel = (got[k] - want[k]).abs() / want[k].abs().max(f64::MIN_POSITIVE);
            if rel > worst {
                worst = rel;
                at = k;
            }
        }
        assert!(
            worst <= 1e-13,
            "H[{},{}] decoded {:e}, unpeeled reference {:e} — relative error \
             {worst:e} (||H||inf = {scale:e}). A peeled column is being read \
             out of a pass it cannot be read out of.",
            irow[at],
            jcol[at],
            got[at],
            want[at]
        );

        // The check above passes trivially on correct code, because the
        // guard leaves cho fully unpeeled and so compares a configuration
        // against itself. It only has teeth against a regression. So prove
        // the guard is doing necessary work rather than assuming it: put
        // the peels back the way the pre-fix reader had them and confirm
        // the Hessian really does come apart.
        t.recolor(&vec![false; n]);
        assert!(
            !t.peeled_cols.is_empty(),
            "restoring the unvetoed coloring must peel again"
        );
        let mut unguarded = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values {
                values: &mut unguarded
            }
        ));
        let mut bad = 0usize;
        let mut worst_unguarded = 0.0_f64;
        for k in 0..nnz {
            let rel = (unguarded[k] - want[k]).abs() / want[k].abs().max(f64::MIN_POSITIVE);
            if rel > 1e-13 {
                bad += 1;
            }
            worst_unguarded = worst_unguarded.max(rel);
        }
        // Two different counts get quoted about this model and they measure
        // different things: 48,931 of the 96,000 entries differ from the
        // uncompressed reference *at all*, down to the last bit, while 88
        // exceed 1e-13 relative. The second is the one worth asserting on.
        assert!(
            bad >= 50 && worst_unguarded > 1e-11,
            "peeling cho_parmest unguarded should damage the entries the guard \
             exists to protect (measured: 88 entries past 1e-13 relative, \
             worst 9.9e-11); got {bad} entries, worst {worst_unguarded:e}. If \
             this fires, the model or the corpus changed and the guard's \
             calibration should be re-derived rather than the bound relaxed."
        );
        eprintln!("unguarded peeling damages {bad}/{nnz} entries, worst {worst_unguarded:e}");
    }

    /// Same weighted coupling as [`weighted_dense_row_objective_nl`], but
    /// the dense variable is `x_{n-1}` instead of `x_0`:
    /// `sum_j (x_j - 1)^2 + x_{n-1} * sum_{j < n-1} w_j x_j`.
    ///
    /// The index matters, and it is the whole reason this helper exists.
    /// With the dense variable at 0 every coupling entry is stored as
    /// `(i, 0)` — the *column* is the peeled one — and the decode reads it
    /// out of column 0's own pass whether or not peeling is on, so the two
    /// paths are bit-identical and no test built on that shape can tell
    /// them apart. Putting the dense variable last stores them as
    /// `(n-1, j)`, where the *row* is the peeled column: peeled, they come
    /// back from column `n-1`'s pass by symmetry, carrying that pass's
    /// roundoff floor; unpeeled, they come back from column `j`'s own pass.
    /// That is the category every one of `cho_parmest`'s 48,931 damaged
    /// entries fell into.
    fn weighted_dense_last_col_nl(n: usize, span: f64) -> String {
        let w = |j: usize| 10_f64.powf(span / 2.0 - span * j as f64 / (n - 2) as f64);
        let d = n - 1;
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} 0 1 0 0 0\n"));
        s.push_str(" 0 1\n 0 0\n");
        s.push_str(&format!(" {n} {n} {n}\n"));
        s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
        s.push_str(&format!(" 0 {n}\n"));
        s.push_str(" 0 0\n 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str(&format!("o54\n{}\n", n + 1));
        for j in 0..n {
            s.push_str(&format!("o5\no1\nv{j}\nn1.0\nn2\n"));
        }
        s.push_str(&format!("o2\nv{d}\no54\n{}\n", n - 1));
        for j in 0..d {
            s.push_str(&format!("o2\nn{:.17e}\nv{j}\n", w(j)));
        }
        s.push_str(&format!("x{n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.5\n"));
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        s.push_str(&format!("G0 {n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.0\n"));
        }
        s
    }

    /// A well-scaled dense column is still peeled, and the entries read
    /// out of its pass are exact — the property the peel guard must not
    /// cost us.
    #[test]
    fn a_well_scaled_dense_column_is_still_peeled() {
        let n = 200;
        let p = parse_nl_text(&weighted_dense_row_objective_nl(n, 0.0)).expect("parse");
        let t = NlTnlp::new(p);
        assert_eq!(
            t.peeled_cols,
            vec![0],
            "a dense column that costs no accuracy must stay peeled"
        );
        assert!(
            t.seeds.len() <= 4,
            "peeling should keep the color count at O(1), got {}",
            t.seeds.len()
        );
    }

    /// An ill-scaled dense column must be un-peeled and colored the
    /// ordinary way, so its small entries come back to full precision.
    ///
    /// This test covers the *guard*, not the damage. It asserts that a
    /// column spanning 12 orders is un-peeled and that its entries are then
    /// exact — and it is worth being precise that it does not, and cannot,
    /// show what peeling would have cost, because on this model peeling
    /// costs nothing: force the peel through and every entry still comes
    /// back bit-identical to its analytic weight. The Hessian here is
    /// constant and each entry is a single product, so there is no
    /// accumulation for a large-magnitude pass to pollute.
    ///
    /// Reproducing the actual digit loss takes a model whose entries are
    /// summed through shared intermediates, which is why the demonstration
    /// lives in `cho_parmest_decodes_to_its_unpeeled_reference` against the
    /// real model rather than a synthetic one.
    #[test]
    fn an_ill_scaled_dense_column_is_not_peeled_and_stays_exact() {
        let n = 200;
        let span = 12.0;
        let w = |j: usize| 10_f64.powf(span / 2.0 - span * j as f64 / (n - 1) as f64);
        let p = parse_nl_text(&weighted_dense_row_objective_nl(n, span)).expect("parse");
        let mut t = NlTnlp::new(p);

        assert!(
            t.peeled_cols.is_empty(),
            "a dense column spanning {span} orders must not be peeled; got {:?}",
            t.peeled_cols
        );

        let info = t.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0_i32; nnz], vec![0_i32; nnz]);
        assert!(t.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        let x: Vec<f64> = (0..n).map(|j| 0.1 * j as f64).collect();
        let mut vals = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            1.0,
            None,
            true,
            SparsityRequest::Values { values: &mut vals }
        ));

        // Every coupling entry must be its weight to full relative
        // precision. Peeled, the smallest of them came back with a
        // relative error near 1e-4.
        let mut checked = 0;
        for k in 0..nnz {
            let (i, j) = (irow[k] as usize, jcol[k] as usize);
            if i == j {
                continue;
            }
            assert_eq!(j, 0, "unexpected off-diagonal ({i}, {j})");
            checked += 1;
            let want = w(i);
            assert!(
                (vals[k] - want).abs() <= 1e-13 * want,
                "H[{i},0] = {:e}, want {want:e} (relative error {:e})",
                vals[k],
                (vals[k] - want).abs() / want
            );
        }
        assert_eq!(checked, n - 1);
    }

    /// Whatever survives the peel guard must decode to the same Hessian
    /// that not peeling at all produces — every entry, not just the
    /// objective.
    ///
    /// This exists because the original dense-column optimization was
    /// validated by running the repository's `.nl` fixtures and comparing
    /// objective value and exit status, which was doubly blind: not one of
    /// the 60 fixtures has a Hessian row dense enough to peel anything, so
    /// the decode path under test never executed, and even had it executed,
    /// the defect cost digits in the multipliers while leaving the
    /// objective intact. So the instrument has to be the assembled Hessian
    /// and the input has to actually peel — hence `peeled_any` below, which
    /// fails if the sweep ever goes vacuous the way the fixture suite
    /// silently did.
    ///
    /// What this catches is a *decode* fault: an entry recovered from the
    /// wrong pass, which is wrong by O(1). It would not have caught the
    /// `cho_parmest` stall, because these synthetic models lose no
    /// precision under peeling at all (see
    /// `an_ill_scaled_dense_column_is_not_peeled_and_stays_exact`). The
    /// precision half is covered against the real model in
    /// `cho_parmest_decodes_to_its_unpeeled_reference`.
    #[test]
    fn a_peeled_decode_matches_an_unpeeled_reference() {
        // `last = true` puts the dense variable at `n-1`, so the coupling
        // entries are stored with the peeled column as their *row* — the
        // only shape in which peeling and not peeling read an entry out of
        // different passes, and so the only shape that can detect a
        // difference at all. See `weighted_dense_last_col_nl`.
        let cases = [
            (200, 0.0, false),
            (200, 2.0, false),
            (200, 0.0, true),
            (200, 1.0, true),
            (200, 2.0, true),
            (400, 3.0, true),
            (600, 0.0, true),
        ];
        let mut peeled_any = false;
        let mut worst = 0.0_f64;

        for &(n, span, last) in &cases {
            let text = if last {
                weighted_dense_last_col_nl(n, span)
            } else {
                weighted_dense_row_objective_nl(n, span)
            };
            let p = parse_nl_text(&text).expect("parse");
            let mut t = NlTnlp::new(p);
            let peeled = t.peeled_cols.clone();
            peeled_any |= !peeled.is_empty();

            let info = t.get_nlp_info().unwrap();
            let nnz = info.nnz_h_lag as usize;
            let (mut irow, mut jcol) = (vec![0_i32; nnz], vec![0_i32; nnz]);
            assert!(t.eval_h(
                None,
                true,
                1.0,
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut irow,
                    jcol: &mut jcol
                }
            ));
            let x: Vec<f64> = (0..n).map(|j| 0.25 + 0.05 * (j % 13) as f64).collect();

            let mut got = vec![0.0_f64; nnz];
            assert!(t.eval_h(
                Some(&x),
                true,
                1.0,
                None,
                true,
                SparsityRequest::Values { values: &mut got }
            ));

            // Same object, same tapes, same point — only the coloring
            // differs, so any disagreement is the decode path.
            t.recolor(&vec![true; n]);
            assert!(
                t.peeled_cols.is_empty(),
                "a fully vetoed model must peel nothing"
            );
            let mut want = vec![0.0_f64; nnz];
            assert!(t.eval_h(
                Some(&x),
                true,
                1.0,
                None,
                true,
                SparsityRequest::Values { values: &mut want }
            ));

            for k in 0..nnz {
                let scale = want[k].abs().max(f64::MIN_POSITIVE);
                let rel = (got[k] - want[k]).abs() / scale;
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-13,
                    "n={n} span={span} last={last} peeled={peeled:?}: H[{},{}] decoded {:e}, \
                     unpeeled reference {:e} (relative error {rel:e})",
                    irow[k],
                    jcol[k],
                    got[k],
                    want[k]
                );
            }
        }

        assert!(
            peeled_any,
            "no case peeled anything, so this test proved nothing about the \
             decode path — the exact way the fixture suite missed the bug"
        );
        assert!(worst < 1e-13, "worst relative disagreement {worst:e}");
    }

    /// `peel_veto` bars a column from the peel set, and the row it puts
    /// back into the conflict structure costs the colors it used to save.
    #[test]
    fn a_vetoed_column_is_colored_the_ordinary_way() {
        let n = 300;
        // One dense row plus a diagonal: column 0 touches every row.
        let mut pairs: Vec<(usize, usize)> = (0..n).map(|j| (j, j)).collect();
        pairs.extend((1..n).map(|i| (i, 0)));
        pairs.sort_unstable();

        let (_, colors_peeled, peeled) = greedy_hessian_coloring(n, &pairs, &vec![false; n]);
        assert!(peeled[0], "the dense column should peel by default");
        assert!(
            colors_peeled <= 4,
            "peeling should collapse the count, got {colors_peeled}"
        );

        let mut veto = vec![false; n];
        veto[0] = true;
        let (_, colors_vetoed, peeled) = greedy_hessian_coloring(n, &pairs, &veto);
        assert!(!peeled[0], "a vetoed column must not be peeled");
        assert!(
            colors_vetoed > colors_peeled,
            "un-peeling restores row 0's conflicts, so colors must rise: \
             {colors_vetoed} vs {colors_peeled}"
        );
    }

    /// A single dense Hessian row must not blow the coloring up to one
    /// color per variable. It used to: every column shares row 0, so no
    /// two columns could be colored alike, and `seeds` / `compressed`
    /// — both `n_colors × n` dense — became O(n²) memory and O(n²) work
    /// per `eval_h` on a Hessian holding only ~2n nonzeros.
    #[test]
    fn a_dense_hessian_row_does_not_explode_the_coloring() {
        let n = 200;
        let p = parse_nl_text(&dense_row_objective_nl(n)).expect("parse");
        let t = NlTnlp::new(p);
        // 2n - 1 entries: the diagonal (j, j) for every j, plus (j, 0)
        // for j >= 1 from the coupling term.
        assert_eq!(t.h_irow.len(), 2 * n - 1);
        assert!(
            t.seeds.len() <= 4,
            "one dense row should cost one extra color, not n; got {} colors for n={n}",
            t.seeds.len()
        );
        assert_eq!(t.seeds.len(), t.compressed.len());
    }

    /// Peeling changes *which* directional product recovers an entry, so
    /// the recovered Hessian must still be exactly right — including the
    /// entries read out of a peeled column's pass by symmetry.
    #[test]
    fn peeled_dense_column_still_recovers_the_exact_hessian() {
        let n = 200;
        let p = parse_nl_text(&dense_row_objective_nl(n)).expect("parse");
        let mut t = NlTnlp::new(p);
        let info = t.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;

        let mut irow = vec![0_i32; nnz];
        let mut jcol = vec![0_i32; nnz];
        assert!(t.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));

        // f = sum_j (x_j - 1)^2 + x_0 * sum_j x_j
        //   H[0,0] = 2 + 2 = 4;  H[j,j] = 2 (j >= 1);  H[j,0] = 1 (j >= 1).
        let x: Vec<f64> = (0..n).map(|j| 0.1 * j as f64).collect();
        let obj_factor = 2.5;
        let mut vals = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            obj_factor,
            None,
            true,
            SparsityRequest::Values { values: &mut vals }
        ));

        let mut seen_diag = 0;
        let mut seen_coupling = 0;
        for k in 0..nnz {
            let (i, j) = (irow[k] as usize, jcol[k] as usize);
            let want = if i == 0 && j == 0 {
                4.0
            } else if i == j {
                seen_diag += 1;
                2.0
            } else {
                assert_eq!(j, 0, "unexpected off-diagonal ({i}, {j})");
                seen_coupling += 1;
                1.0
            } * obj_factor;
            assert!(
                (vals[k] - want).abs() < 1e-12,
                "H[{i},{j}] = {}, want {want}",
                vals[k]
            );
        }
        assert_eq!(seen_diag, n - 1);
        assert_eq!(seen_coupling, n - 1);
    }

    /// The peeling threshold must leave ordinary sparse models on
    /// exactly the coloring they had before: a banded Hessian colors by
    /// its bandwidth, with nothing peeled.
    #[test]
    fn a_sparse_hessian_is_colored_by_its_bandwidth_not_peeled() {
        let n = 400;
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|j| {
                let mut v = vec![(j, j)];
                if j + 1 < n {
                    v.push((j + 1, j));
                }
                v
            })
            .collect();
        let (var_color, n_colors, peeled) = greedy_hessian_coloring(n, &pairs, &vec![false; n]);
        assert!(!peeled.iter().any(|&p| p), "nothing in a band is dense");
        assert!(
            n_colors <= 3,
            "a tridiagonal Hessian needs a handful of colors, got {n_colors}"
        );
        assert!(var_color.iter().all(|&c| c != u32::MAX));
    }

    /// Build a Hessian of `blocks` disjoint dense `size`x`size` blocks
    /// scattered through an otherwise diagonal `n`-variable problem.
    /// A plain coloring needs exactly `size` colors no matter how many
    /// blocks there are, because the blocks share no rows.
    fn disjoint_blocks(n: usize, blocks: usize, size: usize) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = (0..n).map(|j| (j, j)).collect();
        let stride = n / blocks;
        for b in 0..blocks {
            let base = b * stride;
            for i in 0..size {
                for j in 0..=i {
                    if i != j {
                        pairs.push((base + i, base + j));
                    }
                }
            }
        }
        pairs
    }

    /// Many medium-degree columns must not be peeled.
    ///
    /// Regression: selecting candidates by "degree > 16x average" and then
    /// *truncating* the list to `MAX_PEELED_COLS` is not a damage bound.
    /// The columns that miss the cut stay in the conflict structure, so the
    /// base color count is untouched and the singleton colors are pure
    /// addition — these two patterns colored to 306 and 290 against a plain
    /// walk's 50 and 34.
    #[test]
    fn thousands_of_medium_degree_cols_are_not_peeled() {
        for (n, blocks, size) in [(200_000, 100, 50), (200_000, 300, 34)] {
            let pairs = disjoint_blocks(n, blocks, size);
            let (_, n_colors, peeled) = greedy_hessian_coloring(n, &pairs, &vec![false; n]);
            let n_peeled = peeled.iter().filter(|&&p| p).count();
            assert_eq!(
                n_peeled, 0,
                "degree-{size} columns do not pay for a singleton color \
                 (n={n}, blocks={blocks}), yet {n_peeled} were peeled"
            );
            assert!(
                n_colors <= size + 1,
                "disjoint {size}x{size} blocks color by block size regardless \
                 of block count; got {n_colors} for n={n}, blocks={blocks}"
            );
        }
    }

    /// The pay-for-itself rule must not throw away the case peeling exists
    /// for: a handful of genuinely dense rows still get peeled, and still
    /// collapse the coloring.
    #[test]
    fn a_few_truly_dense_rows_are_still_peeled() {
        let n = 5_000;
        let dense_rows = 4;
        let mut pairs: Vec<(usize, usize)> = (0..n).map(|j| (j, j)).collect();
        for d in 0..dense_rows {
            for j in 0..n {
                if j != d {
                    pairs.push((j.max(d), j.min(d)));
                }
            }
        }
        let (_, n_colors, peeled) = greedy_hessian_coloring(n, &pairs, &vec![false; n]);
        let n_peeled = peeled.iter().filter(|&&p| p).count();
        assert_eq!(n_peeled, dense_rows, "every full row should peel");
        assert!(
            n_colors <= dense_rows + 2,
            "peeling {dense_rows} full rows should leave a diagonal remainder, \
             got {n_colors} colors"
        );
    }

    /// The cap still binds, and when it does the kept columns are the
    /// highest-degree ones.
    #[test]
    fn peeling_is_capped_and_keeps_the_worst_offenders() {
        let n = 20_000;
        let dense_rows = 400;
        let mut pairs: Vec<(usize, usize)> = (0..n).map(|j| (j, j)).collect();
        for d in 0..dense_rows {
            // Row d touches the first (n - d) columns, so degree strictly
            // decreases with d and the ordering is unambiguous.
            for j in 0..(n - d) {
                if j != d {
                    pairs.push((j.max(d), j.min(d)));
                }
            }
        }
        let (_, _, peeled) = greedy_hessian_coloring(n, &pairs, &vec![false; n]);
        let n_peeled = peeled.iter().filter(|&&p| p).count();
        assert_eq!(n_peeled, MAX_PEELED_COLS, "cap binds at {MAX_PEELED_COLS}");
        assert!(
            (0..MAX_PEELED_COLS).all(|d| peeled[d]),
            "the {MAX_PEELED_COLS} densest rows are the ones kept"
        );
    }

    /// Header line 0 is `g<count> <opt0> ...`; the option words are the
    /// model's own and a solver echoes them into the `.sol` `Options`
    /// block. `EQ_LIN`'s header is `g3 0 1 0`, so three words follow.
    #[test]
    fn header_option_words_are_kept_verbatim() {
        let p = parse_nl_text(EQ_LIN).expect("parse");
        assert_eq!(p.ampl_options, vec![0, 1, 0]);
    }

    /// A count that does not match the words present must not be
    /// guessed at — the writer falls back rather than emit a wrong block.
    #[test]
    fn a_truncated_option_list_is_dropped_not_padded() {
        let text = EQ_LIN.replacen("g3 0 1 0", "g9 0 1 0", 1);
        let p = parse_nl_text(&text).expect("parse");
        assert!(
            p.ampl_options.is_empty(),
            "9 declared but 3 present: {:?}",
            p.ampl_options
        );
    }

    #[test]
    fn constrained_tnlp_eval_g_jac_h() {
        let p = parse_nl_text(EQ_LIN).expect("parse");
        let mut t = NlTnlp::new(p);
        let info = t.get_nlp_info().unwrap();
        assert_eq!(info.m, 1);
        assert_eq!(info.nnz_jac_g, 2);

        // g(0.3, 0.4) = 0.3 + 0.4 = 0.7
        let mut g = [0.0_f64; 1];
        assert!(t.eval_g(&[0.3, 0.4], true, &mut g));
        assert!((g[0] - 0.7).abs() < 1e-12);

        // Jacobian structure: row 0, cols [0, 1].
        let mut irow = [0_i32; 2];
        let mut jcol = [0_i32; 2];
        assert!(t.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        assert_eq!(irow, [0, 0]);
        assert_eq!(jcol, [0, 1]);

        // Jacobian values: both 1.0.
        let mut vals = [0.0_f64; 2];
        assert!(t.eval_jac_g(
            Some(&[0.3, 0.4]),
            true,
            SparsityRequest::Values { values: &mut vals }
        ));
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);

        // Hessian of L = (x0^2 + x1^2) + λ*(x0 + x1 - 1) is diag(2,2);
        // λ contributes nothing because the constraint is linear, and
        // x0^2 + x1^2 is separable so there's no (1,0) entry in the
        // structural sparsity. nnz_h_lag = 2: (0,0) and (1,1).
        assert_eq!(info.nnz_h_lag, 2);
        let mut hirow = [0_i32; 2];
        let mut hjcol = [0_i32; 2];
        assert!(t.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut hirow,
                jcol: &mut hjcol
            }
        ));
        assert_eq!(hirow, [0, 1]);
        assert_eq!(hjcol, [0, 1]);
        let mut hvals = [0.0_f64; 2];
        assert!(t.eval_h(
            Some(&[0.3, 0.4]),
            true,
            1.0,
            Some(&[0.5]),
            true,
            SparsityRequest::Values { values: &mut hvals }
        ));
        assert!((hvals[0] - 2.0).abs() < 1e-12);
        assert!((hvals[1] - 2.0).abs() < 1e-12);
    }

    /// `min (x0 + x1)^2 + (x0 + x1)` with the shared sum `(x0 + x1)`
    /// encoded as common-subexpression `V2`. Header line 10 declares
    /// one obj-only CSE; expression tree references `v2` twice.
    const CSE_OBJ: &str = "g3 0 1 0
2 0 1 0 0
0 1
0 0
0 2 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 1 0 0 0
V2 0 0
o0
v0
v1
O0 0
o0
o5
v2
n2
v2
b
3
3
";

    #[test]
    fn parses_v_segment_cse() {
        let p = parse_nl_text(CSE_OBJ).expect("parse");
        assert_eq!(p.n, 2);
        // f(1,2) = 9 + 3 = 12
        let f = eval_expr(&p.obj_expr(), &[1.0, 2.0]);
        assert!((f - 12.0).abs() < 1e-12, "got {f}");
        // d/dx0 = 2*(x0+x1) + 1 = 7 at (1,2). Same for x1.
        let mut g = [0.0_f64; 2];
        grad_expr(&p.obj_expr(), &[1.0, 2.0], 1.0, &mut g);
        assert!((g[0] - 7.0).abs() < 1e-12, "g[0]={}", g[0]);
        assert!((g[1] - 7.0).abs() < 1e-12, "g[1]={}", g[1]);
        // collect_vars reaches into the CSE body and finds {0, 1}.
        let mut vs = BTreeSet::new();
        p.obj_nonlinear.collect_vars(&mut vs);
        assert_eq!(vs.into_iter().collect::<Vec<_>>(), vec![0, 1]);
    }

    /// `min (x0 - 1)^2` with three suffix segments attached: an
    /// integer constraint-suffix (target=1, kind=1), an integer var-
    /// suffix (target=0, kind=0), and a real var-suffix (target=0,
    /// kind=4). The .nl format is `S<kind> <nentries> <name>` then
    /// `<idx> <value>` lines.
    const WITH_SUFFIXES: &str = "g3 0 1 0
1 0 1 0 0
0 1
0 0
0 1 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 0 0 0 0
O0 0
o5
o1
v0
n1
n2
b
3
S0 1 sens_state_1
0 7
S4 1 sens_state_value_1
0 4.5
";

    #[test]
    fn parses_var_int_and_var_real_suffixes() {
        let p = parse_nl_text(WITH_SUFFIXES).expect("parse");
        // Integer var-suffix: dense length 1, slot 0 = 7.
        let v = p.suffixes.var_int.get("sens_state_1").expect("var_int");
        assert_eq!(v.as_slice(), &[7]);
        // Real var-suffix: dense length 1, slot 0 = 4.5.
        let r = p
            .suffixes
            .var_real
            .get("sens_state_value_1")
            .expect("var_real");
        assert_eq!(r.len(), 1);
        assert!((r[0] - 4.5).abs() < 1e-12);
        // Other suffix slots stay empty.
        assert!(p.suffixes.con_int.is_empty());
        assert!(p.suffixes.con_real.is_empty());
    }

    /// Two-variable + two-constraint problem with a constraint-level
    /// integer suffix (kind=1). Sparse entries scatter to dense length 2.
    const WITH_CON_SUFFIX: &str = "g3 0 1 0
2 2 1 0 0
0 0
0 0
0 2 0
0 0 0 1
0 0 0 0 0
4 0
0 0
0 0 0 0 0 0
C0
n0
C1
n0
O0 0
n0
r
4 0.0
4 0.0
b
3
3
k1
0
J0 2
0 1
1 1
J1 2
0 1
1 -1
S1 2 sens_init_constr
0 1
1 2
";

    #[test]
    fn parses_con_int_suffix() {
        let p = parse_nl_text(WITH_CON_SUFFIX).expect("parse");
        let s = p.suffixes.con_int.get("sens_init_constr").expect("con_int");
        // Sparse {0:1, 1:2} → dense [1, 2] at length m=2.
        assert_eq!(s.as_slice(), &[1, 2]);
    }

    // ---- gh#785: a truncated file is rejected, not silently defaulted ----
    //
    // `WITH_CON_SUFFIX` is a complete, well-formed file, so cutting it at a
    // segment boundary is exactly the failure the issue reports: an
    // interrupted write. Each cut loses a different first segment and each
    // is caught by a different check, so all three are asserted — and the
    // untruncated text parsing cleanly (`parses_con_int_suffix` above) is
    // what keeps these from passing against a parser that rejects
    // everything.

    /// Everything from `at` (a segment header line) onward is gone.
    fn truncate_before(txt: &str, at: &str) -> String {
        let cut = txt
            .find(at)
            .unwrap_or_else(|| panic!("fixture has no {at:?} segment"));
        txt[..cut].to_string()
    }

    #[test]
    fn truncation_before_the_row_bounds_is_a_parse_error() {
        let err = parse_nl_text(&truncate_before(WITH_CON_SUFFIX, "\nr\n"))
            .expect_err("truncated file must not parse");
        assert!(
            err.contains("`r` (constraint-bounds) segment"),
            "error should name the missing segment: {err}"
        );
    }

    #[test]
    fn truncation_before_the_variable_bounds_is_a_parse_error() {
        let err = parse_nl_text(&truncate_before(WITH_CON_SUFFIX, "\nb\n"))
            .expect_err("truncated file must not parse");
        assert!(
            err.contains("`b` (variable-bounds) segment"),
            "error should name the missing segment: {err}"
        );
    }

    /// The cut that leaves every bound in place and takes only the
    /// coefficients. Neither presence check can see it; the declared-vs-
    /// parsed nonzero count is the whole of the evidence.
    #[test]
    fn truncation_before_the_jacobian_is_a_parse_error() {
        let err = parse_nl_text(&truncate_before(WITH_CON_SUFFIX, "\nk1\n"))
            .expect_err("truncated file must not parse");
        assert!(
            err.contains("declares 4 Jacobian nonzero(s) but the J segments supply 0"),
            "error should report the mismatch: {err}"
        );
    }

    /// The mismatch is an equality, not a floor: a file supplying *more*
    /// entries than it declares is as corrupt as one supplying fewer, and a
    /// `>=` check would wave it through.
    #[test]
    fn more_jacobian_entries_than_declared_is_also_a_parse_error() {
        let extra = WITH_CON_SUFFIX.replace("J1 2\n0 1\n1 -1\n", "J1 2\n0 1\n1 -1\nJ0 1\n0 5\n");
        let err = parse_nl_text(&extra).expect_err("over-full file must not parse");
        assert!(
            err.contains("declares 4 Jacobian nonzero(s) but the J segments supply 5"),
            "error should report the mismatch: {err}"
        );
    }

    /// Fill a `ScalingRequest` from `tnlp` sized for this fixture
    /// (n = m = 2) and hand back everything the engine would see.
    fn scaling_of(tnlp: &mut NlTnlp) -> (bool, Number, bool, Vec<Number>, bool, Vec<Number>) {
        let mut obj = 1.0;
        let mut use_x = false;
        let mut x = vec![0.0; 2];
        let mut use_g = false;
        let mut g = vec![0.0; 2];
        let ok = tnlp.get_scaling_parameters(ScalingRequest {
            obj_scaling: &mut obj,
            use_x_scaling: &mut use_x,
            x_scaling: &mut x,
            use_g_scaling: &mut use_g,
            g_scaling: &mut g,
        });
        (ok, obj, use_x, x, use_g, g)
    }

    /// gh#483: a `.nl` carrying Pyomo/AMPL `scaling_factor` suffixes on
    /// the objective (`S6`) and one constraint (`S5`) reaches the
    /// engine's `user-scaling` pathway. The untagged second row is
    /// unscaled — its AMPL suffix default is 0, which is not a usable
    /// scale factor and reads as "not tagged".
    #[test]
    fn scaling_factor_suffix_feeds_obj_and_constraint_scaling() {
        let nl = WITH_CON_SUFFIX.to_string()
            + "S5 1 scaling_factor\n0 10.0\nS6 1 scaling_factor\n0 100.0\n";
        let p = parse_nl_text(&nl).expect("parse");
        let mut tnlp = NlTnlp::new(p);
        let (ok, obj, use_x, _x, use_g, g) = scaling_of(&mut tnlp);
        assert!(ok, "a tagged model must supply scaling");
        assert!((obj - 100.0).abs() < 1e-12, "obj_scaling={obj}");
        assert!(use_g);
        assert_eq!(g, vec![10.0, 1.0]);
        assert!(!use_x, "no variable suffix was declared");
    }

    /// Variable-level `scaling_factor` entries are passed through, not
    /// dropped on the floor: `OrigIpoptNlp` does not model them and
    /// refuses the solve, which is the whole point of gh#483.
    #[test]
    fn scaling_factor_suffix_forwards_variable_factors() {
        let nl = WITH_CON_SUFFIX.to_string() + "S4 1 scaling_factor\n1 3.0\n";
        let p = parse_nl_text(&nl).expect("parse");
        let mut tnlp = NlTnlp::new(p);
        let (ok, _obj, use_x, x, _use_g, _g) = scaling_of(&mut tnlp);
        assert!(ok);
        assert!(use_x, "variable factors must reach the engine");
        assert_eq!(x, vec![1.0, 3.0]);
    }

    /// gh #703: with curvature-based scaling switched on, the computed
    /// factors are the base and a `scaling_factor` suffix the model
    /// actually carries wins **component by component** — an explicit
    /// factor from the modeller beats one inferred from the coefficients,
    /// and the components they did not tag keep the inferred one rather
    /// than snapping back to 1.
    #[test]
    fn a_user_suffix_overrides_the_computed_factor_component_wise() {
        let baseline = {
            let p = parse_nl_text(WITH_CON_SUFFIX).expect("parse");
            let mut t = NlTnlp::new(p);
            assert!(t.enable_curvature_scaling(), "an LP is degree ≤ 2");
            scaling_of(&mut t).5
        };
        assert!(
            baseline.iter().all(|v| *v > 0.0),
            "curvature scaling should produce usable row factors, got {baseline:?}"
        );

        // Tag row 0 only. AMPL's untagged default is 0, which must read as
        // "not tagged" and leave row 1 on the computed factor.
        let nl = WITH_CON_SUFFIX.to_string() + "S5 1 scaling_factor\n0 7.0\n";
        let p = parse_nl_text(&nl).expect("parse");
        let mut t = NlTnlp::new(p);
        assert!(t.enable_curvature_scaling());
        let (ok, _obj, use_x, _x, use_g, g) = scaling_of(&mut t);
        assert!(ok);
        assert!(use_g && use_x);
        assert_eq!(g[0], 7.0, "the tagged row takes the user's factor");
        assert_eq!(
            g[1], baseline[1],
            "the untagged row keeps the computed one, not 1.0"
        );
    }

    /// No `scaling_factor` suffix ⇒ "the user supplied nothing", the
    /// same answer the default `TNLP` impl gives, so `user-scaling`
    /// falls back to no scaling instead of a bogus all-zero vector.
    #[test]
    fn no_scaling_factor_suffix_declines() {
        let p = parse_nl_text(WITH_CON_SUFFIX).expect("parse");
        let mut tnlp = NlTnlp::new(p);
        let (ok, ..) = scaling_of(&mut tnlp);
        assert!(!ok);
    }

    #[test]
    fn rejects_suffix_with_out_of_range_index() {
        let bad = WITH_CON_SUFFIX.replace("1 2\n", "5 2\n"); // m=2, idx=5 invalid
        let err = parse_nl_text(&bad).expect_err("must reject");
        assert!(
            err.contains("out of range"),
            "expected out-of-range error, got: {err}"
        );
    }

    #[test]
    fn tnlp_round_trip_solves() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let mut tnlp = NlTnlp::new(p);
        let info = tnlp.get_nlp_info().unwrap();
        assert_eq!(info.n, 2);
        assert_eq!(info.m, 0);
        let f0 = tnlp.eval_f(&[0.0, 0.0], true).unwrap();
        assert!((f0 - 5.0).abs() < 1e-12);
        let mut g = [0.0_f64; 2];
        tnlp.eval_grad_f(&[0.0, 0.0], true, &mut g);
        // d/dx0 at x=0: 2*(0-1) = -2; d/dx1: 2*(0-2) = -4
        assert!((g[0] - (-2.0)).abs() < 1e-12);
        assert!((g[1] - (-4.0)).abs() < 1e-12);
    }

    // ---- Sibling `.col` / `.row` name-file capture --------------------
    //
    // Names let diagnostics name the offending equation instead of "row 3"
    // (Lee et al. 2024, https://doi.org/10.69997/sct.147875). These cover
    // the read path and the documented fallback-to-empty behavior.

    use pounce_nlp::expression_provider::ExpressionProvider;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Unique scratch dir for one test (no `tempfile` dev-dep available).
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let seq = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "pounce_nlnames_{}_{}_{}",
            std::process::id(),
            tag,
            seq
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn read_name_file_reads_in_order() {
        let dir = scratch_dir("col_order");
        let p = dir.join("m.col");
        std::fs::write(&p, "x_in\nT_reactor\nflow\n").unwrap();
        assert_eq!(read_name_file(&p, 3), vec!["x_in", "T_reactor", "flow"]);
    }

    #[test]
    fn read_name_file_truncates_extra_lines() {
        // `.row` conventionally appends the objective name after the m
        // constraint names; `.take(expected)` must drop it so names stay
        // 1:1 with `g`.
        let dir = scratch_dir("row_obj");
        let p = dir.join("m.row");
        std::fs::write(&p, "mass_balance\nenergy_balance\nobj\n").unwrap();
        assert_eq!(
            read_name_file(&p, 2),
            vec!["mass_balance", "energy_balance"]
        );
    }

    #[test]
    fn read_name_file_empty_on_short_or_missing() {
        let dir = scratch_dir("short");
        let short = dir.join("m.col");
        std::fs::write(&short, "only_one\n").unwrap();
        // Fewer lines than expected ⇒ empty (never a partial mapping).
        assert!(read_name_file(&short, 3).is_empty());
        // Missing file ⇒ empty, no error.
        assert!(read_name_file(&dir.join("absent.col"), 2).is_empty());
    }

    #[test]
    fn read_nl_file_captures_sibling_names() {
        // SIMPLE is n=2, m=0. Drop a `.col` next to it and confirm the
        // names ride through onto the TNLP's ExpressionProvider.
        let dir = scratch_dir("sibling");
        let nl = dir.join("m.nl");
        std::fs::write(&nl, SIMPLE).unwrap();
        std::fs::write(dir.join("m.col"), "alpha\nbeta\n").unwrap();

        let prob = read_nl_file(&nl).expect("parse + name capture");
        assert_eq!(prob.var_names, vec!["alpha", "beta"]);
        assert!(prob.con_names.is_empty()); // no `.row` written, m=0 anyway

        let tnlp = NlTnlp::new(prob);
        assert_eq!(tnlp.variable_name(0), Some("alpha"));
        assert_eq!(tnlp.variable_name(1), Some("beta"));
        assert_eq!(tnlp.variable_name(2), None); // out of range ⇒ index fallback
    }

    #[test]
    fn read_nl_file_without_names_yields_empty() {
        let dir = scratch_dir("noname");
        let nl = dir.join("m.nl");
        std::fs::write(&nl, SIMPLE).unwrap();
        let prob = read_nl_file(&nl).expect("parse");
        assert!(prob.var_names.is_empty());
        assert!(prob.con_names.is_empty());
        let tnlp = NlTnlp::new(prob);
        assert_eq!(tnlp.variable_name(0), None);
    }

    #[test]
    fn read_nl_file_resolves_extensionless_ampl_stub() {
        // AMPL invokes `pounce mystub -AMPL`, passing the stub *without*
        // the `.nl` extension; the solver must read `mystub.nl`. Code
        // review 2026-06 item M15.
        let dir = scratch_dir("stub");
        std::fs::write(dir.join("mystub.nl"), SIMPLE).unwrap();
        // Pass the extensionless stub — the file `mystub` does not exist.
        let stub = dir.join("mystub");
        assert!(!stub.exists(), "stub must be extensionless / absent");
        let prob = read_nl_file(&stub).expect("stub should resolve to mystub.nl");
        assert_eq!(prob.n, 2);
        assert_eq!(prob.m, 0);

        // Sibling name files are still found off the resolved stem.
        std::fs::write(dir.join("mystub.col"), "alpha\nbeta\n").unwrap();
        let prob = read_nl_file(&stub).expect("stub resolves, names ride along");
        assert_eq!(prob.var_names, vec!["alpha", "beta"]);
    }

    #[test]
    fn read_nl_file_prefers_exact_path_over_nl_sibling() {
        // An existing path is read verbatim — the `.nl` fallback only
        // kicks in when the literal path is missing, so a caller passing a
        // real file is never silently redirected to a `<file>.nl` sibling.
        let dir = scratch_dir("exact");
        // `data` exists and IS a valid .nl; `data.nl` is deliberate garbage.
        std::fs::write(dir.join("data"), SIMPLE).unwrap();
        std::fs::write(dir.join("data.nl"), "not an nl file").unwrap();
        let prob = read_nl_file(&dir.join("data")).expect("exact path wins");
        assert_eq!(prob.n, 2);
    }

    #[test]
    fn append_extension_appends_rather_than_replaces() {
        use std::path::Path;
        assert_eq!(
            append_extension(Path::new("mystub"), "nl"),
            Path::new("mystub.nl")
        );
        // A stub that itself contains a dot keeps its stem (AMPL names it
        // `my.model.nl`, not `my.nl`).
        assert_eq!(
            append_extension(Path::new("my.model"), "nl"),
            Path::new("my.model.nl")
        );
    }

    // ---- equation rendering (`print equation`) ----

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn render_uses_variable_names_when_present() {
        let e = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        assert_eq!(render_expr(&e, &names(&["T", "flow"]), &[]), "T*flow");
        // Falls back to x[i] when names are absent.
        assert_eq!(render_expr(&e, &[], &[]), "x[0]*x[1]");
    }

    #[test]
    fn render_parenthesizes_by_precedence() {
        // (x0 + x1) * x2  must keep the parens around the sum.
        let sum = Expr::Binary(BinOp::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let e = Expr::Binary(BinOp::Mul, Box::new(sum), Box::new(Expr::Var(2)));
        assert_eq!(render_expr(&e, &[], &[]), "(x[0] + x[1])*x[2]");

        // x0 + x1 * x2  needs no parens (mul binds tighter).
        let mul = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(1)), Box::new(Expr::Var(2)));
        let e2 = Expr::Binary(BinOp::Add, Box::new(Expr::Var(0)), Box::new(mul));
        assert_eq!(render_expr(&e2, &[], &[]), "x[0] + x[1]*x[2]");
    }

    #[test]
    fn render_subtraction_right_assoc_parens() {
        // x0 - (x1 - x2) keeps the parens; x0 - x1 - x2 does not.
        let inner = Expr::Binary(BinOp::Sub, Box::new(Expr::Var(1)), Box::new(Expr::Var(2)));
        let e = Expr::Binary(BinOp::Sub, Box::new(Expr::Var(0)), Box::new(inner));
        assert_eq!(render_expr(&e, &[], &[]), "x[0] - (x[1] - x[2])");
    }

    #[test]
    fn render_functions_and_pow() {
        let sq = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let e = Expr::Unary(UnaryOp::Exp, Box::new(sq));
        assert_eq!(render_expr(&e, &names(&["q"]), &[]), "exp(q^2)");
    }

    #[test]
    fn render_linear_signs_are_tidy() {
        // 1*a - 2*b + c  (coef +1 omits the multiplier).
        let lin = vec![(0usize, 1.0), (1, -2.0), (2, 1.0)];
        assert_eq!(render_linear(&lin, &names(&["a", "b", "c"])), "a - 2*b + c");
    }

    #[test]
    fn render_linear_skips_zero_coefficients() {
        // A 0 coefficient (a variable present only in the nonlinear part)
        // is dropped, not rendered as `0*x`.
        let lin = vec![(0usize, 1.0), (1, 0.0), (2, -3.0)];
        assert_eq!(render_linear(&lin, &names(&["a", "b", "c"])), "a - 3*c");
        // Leading term zero ⇒ the first emitted term still has no ` + `.
        let lin = vec![(0usize, 0.0), (1, 2.0)];
        assert_eq!(render_linear(&lin, &names(&["a", "b"])), "2*b");
    }

    #[test]
    fn render_sum_folds_negative_terms() {
        // Σ(a², -b⁴, -c) reads `a^2 - b^4 - c`, not `a^2 + -b^4 + -c`.
        let sq = |i| {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(i)),
                Box::new(Expr::Const(2.0)),
            )
        };
        let neg = |i| {
            Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(-1.0)),
                Box::new(Expr::Var(i)),
            )
        };
        let e = Expr::Sum(vec![
            sq(0),
            neg(1),
            Expr::Unary(UnaryOp::Neg, Box::new(Expr::Var(2))),
        ]);
        assert_eq!(
            render_expr(&e, &names(&["a", "b", "c"]), &[]),
            "a^2 - 1*b - c"
        );
    }

    #[test]
    fn render_constraint_equation_forms() {
        // Build a 2-constraint problem by hand: an equality and a range.
        let mut prob = parse_nl_text(SIMPLE).unwrap();
        // Overwrite to a known small shape: 1 var, 2 cons.
        prob.n = 2;
        prob.m = 2;
        prob.var_names = names(&["mass_in", "mass_out"]);
        prob.con_names = names(&["balance", "window"]);
        prob.con_linear = vec![
            vec![(0, 1.0), (1, -1.0)], // mass_in - mass_out
            vec![(0, 1.0)],            // mass_in
        ];
        prob.con_nonlinear = vec![
            NlBody::Tree(Expr::Const(0.0)),
            NlBody::Tree(Expr::Const(0.0)),
        ];
        prob.g_l = vec![0.0, 0.0];
        prob.g_u = vec![0.0, 500.0];

        assert_eq!(
            render_constraint_equation(&prob, 0),
            "mass_in - mass_out = 0"
        );
        assert_eq!(render_constraint_equation(&prob, 1), "0 <= mass_in <= 500");

        let all = render_all_constraint_equations(&prob);
        assert_eq!(all.len(), 2);
        assert_eq!(all[1], "0 <= mass_in <= 500");
    }

    #[test]
    fn constraint_jacobian_sparsity_unions_linear_and_nonlinear() {
        let mut prob = parse_nl_text(SIMPLE).unwrap();
        prob.n = 3;
        prob.m = 2;
        // Row 0: linear in x1, nonlinear in x0 and x2 → support {0,1,2}.
        // Row 1: linear in x2 only → support {2}.
        prob.con_linear = vec![vec![(1, 4.0)], vec![(2, 1.0)]];
        prob.con_nonlinear = vec![
            NlBody::Tree(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(2)),
            )),
            NlBody::Tree(Expr::Const(0.0)),
        ];
        prob.g_l = vec![0.0, 0.0];
        prob.g_u = vec![0.0, 0.0];

        let (irow, jcol) = constraint_jacobian_sparsity(&prob);
        // Sorted, deduped per row: row 0 → cols 0,1,2; row 1 → col 2.
        assert_eq!(irow, vec![0, 0, 0, 1]);
        assert_eq!(jcol, vec![0, 1, 2, 2]);
    }

    #[test]
    fn funcall_string_arg_with_hash_is_not_truncated() {
        // Code review L31: an AMPL string argument is a Hollerith literal
        // `h<len>:<chars>` whose content is exactly <len> bytes and may
        // legitimately contain '#' (e.g. a parameters-directory path). The
        // old parser ran strip_comment() over the line first, truncating
        // the content at the '#'. Here `h3:a#b` must round-trip to "a#b".
        let mut p = Parser::new("h3:a#b\n", false);
        match p.parse_funcall_arg().expect("parse hollerith arg") {
            FuncallArg::Str(s) => assert_eq!(s, "a#b"),
            other => panic!("expected Str, got {other:?}"),
        }
    }

    #[test]
    fn funcall_string_arg_honors_declared_length() {
        // The declared `<len>` is authoritative: exactly that many bytes
        // after the ':' form the string; trailing content (here a real
        // ` # comment`) is not part of it.
        let mut p = Parser::new("h3:abc # trailing comment\n", false);
        match p.parse_funcall_arg().expect("parse hollerith arg") {
            FuncallArg::Str(s) => assert_eq!(s, "abc"),
            other => panic!("expected Str, got {other:?}"),
        }
    }

    // --- AMPL power specializations (opcodes o81/o82/o83) --------------------
    //
    // AMPL emits these in place of the general `o5` (OPPOW) when one operand
    // is constant. They must parse to the same `Pow` AST as `o5` so the tape's
    // negative-base-safe constant-power lowering applies. The eval points below
    // are chosen to pin down BOTH the arity and the operand order: a swapped
    // `base`/`exp` (or treating `o82` as a different unary op) gives a
    // different number at these points, so each assertion is discriminating.

    /// Parse a single expression `expr_src` with `n` variables in scope,
    /// driving the real `parse_opcode` path through `parse_expr`.
    fn parse_one_expr(n: usize, expr_src: &str) -> Expr {
        let mut p = Parser::new(expr_src, false);
        p.n = n;
        p.parse_expr().expect("parse expression")
    }

    #[test]
    fn opcode_o82_square_is_unary_pow_of_two() {
        // o82 OP2POW: `x^2`, unary — one operand, implicit exponent 2.
        let e = parse_one_expr(1, "o82\nv0\n");
        match &e {
            Expr::Binary(BinOp::Pow, base, exp) => {
                assert!(matches!(**base, Expr::Var(0)));
                match **exp {
                    Expr::Const(c) => assert!((c - 2.0).abs() < 1e-12, "exp const = {c}"),
                    ref other => panic!("o82 exponent must be Const(2.0), got {other:?}"),
                }
            }
            other => panic!("o82 must parse to Pow(base, 2), got {other:?}"),
        }
        // value: 3^2 = 9, and — the whole point of o82 — a NEGATIVE base stays
        // real: (-3)^2 = 9 (general `exp(2·ln x)` would be NaN here).
        assert!((eval_expr(&e, &[3.0]) - 9.0).abs() < 1e-12);
        assert!((eval_expr(&e, &[-3.0]) - 9.0).abs() < 1e-12);
        // gradient d/dx x^2 = 2x: 6 at x=3, -6 at x=-3 (real on both sides).
        let mut g = [0.0_f64; 1];
        grad_expr(&e, &[3.0], 1.0, &mut g);
        assert!((g[0] - 6.0).abs() < 1e-9, "grad at 3 = {}", g[0]);
        g[0] = 0.0;
        grad_expr(&e, &[-3.0], 1.0, &mut g);
        assert!((g[0] + 6.0).abs() < 1e-9, "grad at -3 = {}", g[0]);
    }

    #[test]
    fn opcode_o81_const_exponent_is_base_pow_const() {
        // o81 OP1POW: `base ^ const`, binary, operands `base` then `exp`.
        let e = parse_one_expr(1, "o81\nv0\nn3\n");
        match &e {
            Expr::Binary(BinOp::Pow, base, exp) => {
                assert!(matches!(**base, Expr::Var(0)), "base must be the variable");
                match **exp {
                    Expr::Const(c) => assert!((c - 3.0).abs() < 1e-12, "exp const = {c}"),
                    ref other => panic!("o81 exponent must be Const(3.0), got {other:?}"),
                }
            }
            other => panic!("o81 must parse to Pow(var, const), got {other:?}"),
        }
        // x^3 at x=2 is 8, NOT 3^2=9 — pins operand order (base^exp, not exp^base).
        assert!((eval_expr(&e, &[2.0]) - 8.0).abs() < 1e-12);
        // NEGATIVE base, odd integer exponent: (-2)^3 = -8. This is exactly the
        // case the general `pow` (exp(3·ln x)) cannot do — it returns NaN.
        assert!((eval_expr(&e, &[-2.0]) + 8.0).abs() < 1e-12);
        // gradient d/dx x^3 = 3x^2 = 12 at x=2.
        let mut g = [0.0_f64; 1];
        grad_expr(&e, &[2.0], 1.0, &mut g);
        assert!((g[0] - 12.0).abs() < 1e-9, "grad at 2 = {}", g[0]);
    }

    #[test]
    fn opcode_o83_const_base_is_const_pow_exp() {
        // o83 OPCPOW: `const ^ exp`, binary, operands `base` (the const) then `exp`.
        let e = parse_one_expr(1, "o83\nn2\nv0\n");
        match &e {
            Expr::Binary(BinOp::Pow, base, exp) => {
                match **base {
                    Expr::Const(c) => assert!((c - 2.0).abs() < 1e-12, "base const = {c}"),
                    ref other => panic!("o83 base must be Const(2.0), got {other:?}"),
                }
                assert!(
                    matches!(**exp, Expr::Var(0)),
                    "exponent must be the variable"
                );
            }
            other => panic!("o83 must parse to Pow(const, var), got {other:?}"),
        }
        // 2^x at x=3 is 8, NOT x^2=9 at x=3 — pins operand order (const^exp).
        assert!((eval_expr(&e, &[3.0]) - 8.0).abs() < 1e-12);
        assert!((eval_expr(&e, &[0.0]) - 1.0).abs() < 1e-12);
        // gradient d/dx 2^x = 2^x · ln 2; at x=3 that is 8·ln2.
        let mut g = [0.0_f64; 1];
        grad_expr(&e, &[3.0], 1.0, &mut g);
        assert!(
            (g[0] - 8.0 * 2.0_f64.ln()).abs() < 1e-9,
            "grad at 3 = {} (want {})",
            g[0],
            8.0 * 2.0_f64.ln()
        );
    }

    #[test]
    fn power_specializations_agree_with_general_o5() {
        // Where both are defined, o81/o82/o83 must be numerically identical to
        // the general `o5` pow on the same operands — they are only routing
        // hints, not different math.
        let o5_sq = parse_one_expr(1, "o5\nv0\nn2\n"); // x^2
        let o82 = parse_one_expr(1, "o82\nv0\n");
        let o5_cube = parse_one_expr(1, "o5\nv0\nn3\n"); // x^3
        let o81 = parse_one_expr(1, "o81\nv0\nn3\n");
        let o5_exp = parse_one_expr(1, "o5\nn2\nv0\n"); // 2^x
        let o83 = parse_one_expr(1, "o83\nn2\nv0\n");
        for &x in &[-2.0_f64, -0.5, 0.0, 1.0, 2.5, 4.0] {
            assert!((eval_expr(&o82, &[x]) - eval_expr(&o5_sq, &[x])).abs() < 1e-12);
            assert!((eval_expr(&o81, &[x]) - eval_expr(&o5_cube, &[x])).abs() < 1e-12);
            // 2^x is real for all x; compare across the same points.
            assert!((eval_expr(&o83, &[x]) - eval_expr(&o5_exp, &[x])).abs() < 1e-12);
        }
    }

    #[test]
    fn power_opcodes_round_trip_through_parse_nl_text() {
        // End-to-end through the public entry point: `min x0^2 + x1^2` written
        // with o82 (square) parses and evaluates like its o5 twin. Reuses the
        // SIMPLE header (n=2, m=0, both vars nonlinear in the objective).
        let nl = SIMPLE.replace(
            "o0\no5\no1\nv0\nn1\nn2\no5\no1\nv1\nn2\nn2\n",
            "o0\no82\nv0\no82\nv1\n",
        );
        assert_ne!(nl, SIMPLE, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse o82 objective");
        // f(3,4) = 9 + 16 = 25; both bases negative still real: f(-3,-4)=25.
        assert!((eval_expr(&p.obj_expr(), &[3.0, 4.0]) - 25.0).abs() < 1e-12);
        assert!((eval_expr(&p.obj_expr(), &[-3.0, -4.0]) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn power_opcode_o81_evaluates_through_the_tape_at_negative_base() {
        // Full production path: parse o81 -> build the tape -> eval_f/eval_grad_f.
        // `min x0^3 + x1^3` lowers each cube to an integer-power mul chain
        // (the negative-base-safe path) rather than a generic `powf`. The check
        // at a NEGATIVE base is the one that would break if o81 wrongly routed
        // through `exp(c·ln x)`: (-2)^3 must be -8, not NaN.
        let nl = SIMPLE.replace(
            "o0\no5\no1\nv0\nn1\nn2\no5\no1\nv1\nn2\nn2\n",
            "o0\no81\nv0\nn3\no81\nv1\nn3\n",
        );
        assert_ne!(nl, SIMPLE, "fixture substitution must apply");
        let p = parse_nl_text(&nl).expect("parse o81 objective");
        let mut tnlp = NlTnlp::new(p);
        tnlp.get_nlp_info().unwrap();
        // f(-2, 1) = (-2)^3 + 1^3 = -8 + 1 = -7 (real, not NaN).
        let f = tnlp.eval_f(&[-2.0, 1.0], true).unwrap();
        assert!((f + 7.0).abs() < 1e-12, "f(-2,1) = {f}");
        // grad = (3 x0^2, 3 x1^2) = (12, 3) at (-2, 1).
        let mut g = [0.0_f64; 2];
        assert!(tnlp.eval_grad_f(&[-2.0, 1.0], true, &mut g));
        assert!((g[0] - 12.0).abs() < 1e-9, "df/dx0 = {}", g[0]);
        assert!((g[1] - 3.0).abs() < 1e-9, "df/dx1 = {}", g[1]);
    }

    // ---- Shared-CSE constraint tape (issue #476) ----------------------

    /// Three constraints over one `V` segment (a `.nl` *defined variable*),
    /// `V3 = 2*x0 + 3*x1`, referenced by all three:
    ///   C0: V3^2        C1: V3^3 + x2        C2: V3*x2
    /// `{BODY2}` is a substitution point so a variant can drop an opcode the
    /// hybrid path rejects into C2.
    const SHARED_CSE: &str = "g3 1 1 0
 3 3 1 0 0
 3 0
 0 0
 3 0 0
 0 0 0 1
 0 0 0 0 0
 8 3
 0 0
 0 1 0 0 0
V3 2 0
0 2.0
1 3.0
n0
C0
o5
v3
n2
C1
o0
o5
v3
n3
v2
C2
{BODY2}
O0 0
n0
r
2 0
2 0
2 0
b
3
3
3
k2
3
6
J0 2
0 0
1 0
J1 3
0 0
1 0
2 0
J2 3
0 0
1 0
2 0
G0 3
0 1.0
1 1.0
2 1.0
";

    fn shared_cse_nl(body2: &str) -> String {
        SHARED_CSE.replace("{BODY2}", body2)
    }

    /// `eval_jac_g`'s shared-CSE path must return exactly what the flat
    /// per-summand tapes return — it is a different traversal of the same
    /// DAG, not a different derivative. Forced on here regardless of
    /// `HYBRID_JAC_MIN_OP_RATIO` so the path is covered independently of
    /// the size heuristic that decides when to use it.
    #[test]
    fn shared_cse_jacobian_matches_flat_tape_bit_for_bit() {
        let nl = shared_cse_nl("o2\nv3\nv2");
        let p = parse_nl_text(&nl).expect("parse shared-CSE model");

        let mut hybrid = NlTnlp::new(p.clone());
        let info = hybrid.get_nlp_info().unwrap();
        let nnz = info.nnz_jac_g as usize;
        hybrid
            .con_hybrid
            .as_mut()
            .expect("CSE shared by 3 constraints must build the hybrid tape")
            .use_for_jac = true;

        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        for x in [[1.0, 1.0, 1.0], [-2.0, 0.5, 3.0], [0.0, -1.5, -0.25]] {
            let mut jh = vec![0.0_f64; nnz];
            let mut jf = vec![0.0_f64; nnz];
            assert!(hybrid.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jh }));
            assert!(flat.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jf }));
            assert_eq!(
                jh, jf,
                "hybrid Jacobian differs from the flat tape at {x:?}"
            );

            // V3 = 2 x0 + 3 x1; rows are V3^2, V3^3 + x2, V3 * x2.
            let s = 2.0 * x[0] + 3.0 * x[1];
            let want = [
                4.0 * s,
                6.0 * s,
                6.0 * s * s,
                9.0 * s * s,
                1.0,
                2.0 * x[2],
                3.0 * x[2],
                s,
            ];
            assert_eq!(nnz, want.len());
            for k in 0..nnz {
                assert!(
                    (jh[k] - want[k]).abs() < 1e-9,
                    "entry {k} at {x:?}: got {}, want {}",
                    jh[k],
                    want[k]
                );
            }
        }
    }

    /// `.nl` text for `m` constraints `S * x_{i+2} >= 0`, all sharing one
    /// CSE `S = body(x0, x1)` (the caller passes the body's expression
    /// text, which must reference exactly `v0` and `v1`). A deep shared
    /// body over few local ops per row is the regime the op-ratio gates
    /// are meant to catch: no repository fixture has a CSE shared across
    /// constraints at all, so without this the gate-on paths have no
    /// natural coverage.
    fn shared_body_chain_nl(m: usize, body: &str) -> String {
        let n = m + 2;
        let nzc = 3 * m;
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} {m} 1 0 0 0\n"));
        s.push_str(&format!(" {m} 0\n 0 0\n"));
        s.push_str(&format!(" {n} 0 0\n"));
        s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
        s.push_str(&format!(" {nzc} {n}\n"));
        s.push_str(" 0 0\n 0 1 0 0 0\n");
        // The shared body.
        s.push_str(&format!("V{n} 0 0\n"));
        s.push_str(body);
        // Rows: S * x_{i+2}.
        for i in 0..m {
            s.push_str(&format!("C{i}\no2\nv{n}\nv{}\n", i + 2));
        }
        s.push_str("O0 0\nn0\n");
        s.push_str(&format!("x{n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} {}\n", 0.3 + 0.05 * j as f64));
        }
        s.push_str("r\n");
        for _ in 0..m {
            s.push_str("2 0\n");
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        // Column counts: cols 0 and 1 appear in every row, col i+2 in one.
        s.push_str(&format!("k{}\n", n - 1));
        let mut acc = 0;
        for j in 0..n - 1 {
            acc += if j < 2 { m } else { 1 };
            s.push_str(&format!("{acc}\n"));
        }
        for i in 0..m {
            s.push_str(&format!("J{i} 3\n0 0.0\n1 0.0\n{} 0.0\n", i + 2));
        }
        s.push_str(&format!("G0 {n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.0\n"));
        }
        s
    }

    /// [`shared_body_chain_nl`] with `S = exp^depth(0.01 * (x0 + x1))`.
    /// The forward value saturates to `inf` past depth ≈ 5, which the
    /// Jacobian tests tolerate (`inf == inf`); Hessian tests need the
    /// bounded variant below instead, whose second-order terms would
    /// otherwise mix `inf`s of both signs into `NaN`.
    fn deep_shared_cse_nl(m: usize, depth: usize) -> String {
        let mut body = String::new();
        for _ in 0..depth {
            body.push_str("o44\n");
        }
        body.push_str("o2\nn0.01\no0\nv0\nv1\n");
        shared_body_chain_nl(m, &body)
    }

    /// [`shared_body_chain_nl`] with `S = (log ∘ exp)^pairs(2 + 0.01 *
    /// (x0 + x1))` — mathematically the identity chain, so the value
    /// stays bounded (no `inf`/`NaN` at any reasonable `x`) while every
    /// stage still carries nonzero curvature (`exp'' ≠ 0`, `log'' ≠ 0`)
    /// through both second-order sweeps. `2 * pairs` body ops drive the
    /// flat/shared op ratio as high as the Hessian gate tests need.
    fn bounded_deep_shared_cse_nl(m: usize, pairs: usize) -> String {
        let mut body = String::new();
        for _ in 0..pairs {
            body.push_str("o43\no44\n");
        }
        body.push_str("o0\nn2\no2\nn0.01\no0\nv0\nv1\n");
        shared_body_chain_nl(m, &body)
    }

    /// With a deep shared body the gate turns itself on, and the path it
    /// turns on must still agree with the flat tapes exactly.
    #[test]
    fn a_deep_shared_body_turns_the_jacobian_gate_on_and_still_agrees() {
        let p = parse_nl_text(&deep_shared_cse_nl(16, 40)).expect("parse");

        let mut hybrid = NlTnlp::new(p.clone());
        let info = hybrid.get_nlp_info().unwrap();
        let nnz = info.nnz_jac_g as usize;
        assert!(
            hybrid
                .con_hybrid
                .as_ref()
                .expect("shared CSE must build the hybrid tape")
                .use_for_jac,
            "a 40-deep body shared by 16 rows is well past the op-ratio gate"
        );

        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        for scale in [1.0_f64, -0.7, 2.5] {
            let x: Vec<f64> = (0..info.n as usize)
                .map(|j| scale * (0.2 + 0.03 * j as f64))
                .collect();
            let mut jh = vec![0.0_f64; nnz];
            let mut jf = vec![0.0_f64; nnz];
            assert!(hybrid.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jh }));
            assert!(flat.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jf }));
            assert_eq!(jh, jf, "gate-on Jacobian differs from the flat tape");
            assert!(jh.iter().any(|v| *v != 0.0), "all-zero Jacobian is no test");
        }
    }

    /// The Jacobian gate is off for a model whose shared bodies are small,
    /// because there the hybrid traversal's per-op overhead outweighs the
    /// halved forward sweep. `eval_g` still takes the hybrid path — that
    /// one is a win at any ratio.
    #[test]
    fn a_small_shared_body_leaves_the_jacobian_on_the_flat_path() {
        let p = parse_nl_text(&shared_cse_nl("o2\nv3\nv2")).expect("parse");
        let mut t = NlTnlp::new(p);
        t.get_nlp_info().unwrap();
        let h = t.con_hybrid.as_ref().expect("hybrid built for eval_g");
        assert!(
            !h.use_for_jac,
            "a 3-row model with a 2-term CSE is far below the op-ratio gate"
        );
    }

    /// `eval_h`'s shared-CSE path (issue #557) against the flat tapes on a
    /// polynomial model with dyadic inputs. Folding `λ_k` into the boundary
    /// adjoints and running one prelude sweep reassociates floating-point
    /// products, so the two paths agree only to rounding on general inputs —
    /// but here every operation in both traversals is exact (dyadic values,
    /// small-integer coefficients, polynomial ops), so the results must be
    /// bit-identical, pinning the arithmetic itself and not just its
    /// magnitude. Forced on regardless of `HYBRID_HESS_MIN_OP_RATIO` so the
    /// path is covered independently of the size heuristic.
    #[test]
    fn shared_cse_hessian_matches_flat_tape_bit_for_bit() {
        let nl = shared_cse_nl("o2\nv3\nv2");
        let p = parse_nl_text(&nl).expect("parse shared-CSE model");

        let mut hybrid = NlTnlp::new(p.clone());
        let info = hybrid.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;
        hybrid
            .con_hybrid
            .as_mut()
            .expect("CSE shared by 3 constraints must build the hybrid tape")
            .use_for_hess = true;

        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        let pairs: Vec<(usize, usize)> = hybrid
            .h_irow
            .iter()
            .zip(&hybrid.h_jcol)
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();

        // Two multiplier sets: one all-live, one with a dead row so the
        // λ == 0 skip is exercised on the hybrid path too.
        for lam in [[0.5, -1.25, 2.0], [0.0, 1.0, 0.5]] {
            for x in [[1.0, 1.0, 1.0], [-2.0, 0.5, 3.0], [0.0, -1.5, -0.25]] {
                let mut hh = vec![0.0_f64; nnz];
                let mut hf = vec![0.0_f64; nnz];
                assert!(hybrid.eval_h(
                    Some(&x),
                    true,
                    1.0,
                    Some(&lam),
                    true,
                    SparsityRequest::Values { values: &mut hh }
                ));
                assert!(flat.eval_h(
                    Some(&x),
                    true,
                    1.0,
                    Some(&lam),
                    true,
                    SparsityRequest::Values { values: &mut hf }
                ));
                assert_eq!(
                    hh, hf,
                    "hybrid Hessian differs from the flat tape at {x:?}, λ = {lam:?}"
                );

                // Analytic cross-check. V3 = 2 x0 + 3 x1 =: s with gradient
                // dV = (2, 3, 0); the rows are V3², V3³ + x2, V3·x2 and the
                // objective is constant, so the Lagrangian Hessian is
                //   (2 λ0 + 6 s λ1) · dV dVᵀ + λ2 · (dV e2ᵀ + e2 dVᵀ).
                let s = 2.0 * x[0] + 3.0 * x[1];
                let q = 2.0 * lam[0] + 6.0 * s * lam[1];
                let dv = [2.0, 3.0, 0.0];
                for (k, &(i, j)) in pairs.iter().enumerate() {
                    let mut want = q * dv[i] * dv[j];
                    if i == 2 {
                        want += lam[2] * dv[j];
                    }
                    if j == 2 {
                        want += lam[2] * dv[i];
                    }
                    assert!(
                        (hh[k] - want).abs() < 1e-9,
                        "entry ({i}, {j}) at {x:?}, λ = {lam:?}: got {}, want {want}",
                        hh[k]
                    );
                }
            }
        }
    }

    /// A deep (but bounded — see `bounded_deep_shared_cse_nl`) shared body
    /// turns the Hessian gate on by itself, and the path it turns on must
    /// agree with the flat tapes. Not bitwise here: the shared prelude
    /// reverse sweep runs once over the `λ_k`-folded adjoints of all
    /// summands where the flat path runs per summand, and that
    /// reassociation moves transcendental results by rounding — so the bar
    /// is a relative few-ULP band, with the exact-arithmetic case pinned
    /// bitwise by `shared_cse_hessian_matches_flat_tape_bit_for_bit`.
    #[test]
    fn a_deep_shared_body_turns_the_hessian_gate_on_and_still_agrees() {
        let m = 16;
        let p = parse_nl_text(&bounded_deep_shared_cse_nl(m, 20)).expect("parse");

        let mut hybrid = NlTnlp::new(p.clone());
        let info = hybrid.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;
        assert!(
            hybrid
                .con_hybrid
                .as_ref()
                .expect("shared CSE must build the hybrid tape")
                .use_for_hess,
            "a 40-op body shared by 16 rows is well past the op-ratio gate"
        );

        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        let lam: Vec<f64> = (0..m).map(|k| 0.25 + 0.125 * k as f64).collect();
        for scale in [1.0_f64, -0.7, 2.5] {
            let x: Vec<f64> = (0..info.n as usize)
                .map(|j| scale * (0.2 + 0.03 * j as f64))
                .collect();
            // Run the hybrid Jacobian first, the order a real solve
            // iteration uses. Its `gradient_summand` sweeps leave the
            // *Jacobian's* prelude adjoint arena dirty; the Hessian's
            // accumulators must be its own buffers with the all-zero-
            // between-colors invariant, or this seeds `eval_h` with a
            // stale row gradient (caught here).
            let mut jac = vec![0.0_f64; info.nnz_jac_g as usize];
            assert!(hybrid.eval_jac_g(
                Some(&x),
                true,
                SparsityRequest::Values { values: &mut jac }
            ));
            let mut hh = vec![0.0_f64; nnz];
            let mut hf = vec![0.0_f64; nnz];
            assert!(hybrid.eval_h(
                Some(&x),
                true,
                1.0,
                Some(&lam),
                true,
                SparsityRequest::Values { values: &mut hh }
            ));
            assert!(flat.eval_h(
                Some(&x),
                true,
                1.0,
                Some(&lam),
                true,
                SparsityRequest::Values { values: &mut hf }
            ));
            for k in 0..nnz {
                assert!(
                    hh[k].is_finite() && hf[k].is_finite(),
                    "non-finite Hessian entry {k} defeats the comparison"
                );
                let tol = 1e-12 * hf[k].abs().max(1.0);
                assert!(
                    (hh[k] - hf[k]).abs() <= tol,
                    "gate-on Hessian entry {k} at scale {scale}: hybrid {} vs flat {}",
                    hh[k],
                    hf[k]
                );
            }
            assert!(hh.iter().any(|v| *v != 0.0), "all-zero Hessian is no test");
        }
    }

    /// `.nl` text for two independent shared-CSE blocks of different widths:
    /// block A's body is `sin(x0 + … + x_{wide-1})`, block B's is
    /// `sin(x_wide + … )` over `narrow` variables, each feeding `rows` rows
    /// of the form `body * x_r`.
    ///
    /// The width difference is the point. A body summing `w` variables gives
    /// its rows a dense `w × w` Hessian block, so those `w` columns pairwise
    /// conflict and the coloring must spend `w` colors on them; the narrower
    /// block reuses the low colors. The surplus colors therefore belong to
    /// block A alone, and a per-color prelude walk over *both* bodies would
    /// be doing work for a body that color cannot reach.
    fn two_block_shared_cse_nl(wide: usize, narrow: usize, rows: usize) -> String {
        let nvars = wide + narrow;
        let m = 2 * rows;
        let n = nvars + m;
        // Block A rows touch `wide` body vars + 1 row var; block B rows
        // touch `narrow` + 1.
        let nzc = rows * (wide + 1) + rows * (narrow + 1);
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} {m} 1 0 0 0\n"));
        s.push_str(&format!(" {m} 0\n 0 0\n"));
        s.push_str(&format!(" {n} 0 0\n"));
        s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
        s.push_str(&format!(" {nzc} {n}\n"));
        s.push_str(" 0 0\n 0 2 0 0 0\n");
        // Two bodies: V{n} over the first `wide` vars, V{n+1} over the next
        // `narrow`. `sin` of a left-nested sum: k terms need k-1 adds.
        for (b, (base, count)) in [(0, wide), (wide, narrow)].iter().enumerate() {
            s.push_str(&format!("V{} 0 0\n", n + b));
            s.push_str("o41\n");
            for _ in 0..count - 1 {
                s.push_str("o0\n");
            }
            for j in 0..*count {
                s.push_str(&format!("v{}\n", base + j));
            }
        }
        // Rows: body_b * x_{rowvar}.
        for i in 0..m {
            let b = i / rows;
            s.push_str(&format!("C{i}\no2\nv{}\nv{}\n", n + b, nvars + i));
        }
        s.push_str("O0 0\nn0\n");
        s.push_str(&format!("x{n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} {}\n", 0.2 + 0.01 * j as f64));
        }
        s.push_str("r\n");
        for _ in 0..m {
            s.push_str("2 0\n");
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        // Cumulative Jacobian column counts for the first n-1 columns.
        s.push_str(&format!("k{}\n", n - 1));
        let mut acc = 0;
        for j in 0..n - 1 {
            acc += if j < nvars { rows } else { 1 };
            s.push_str(&format!("{acc}\n"));
        }
        for i in 0..m {
            let (base, count) = if i < rows { (0, wide) } else { (wide, narrow) };
            s.push_str(&format!("J{i} {}\n", count + 1));
            let mut cols: Vec<usize> = (base..base + count).collect();
            cols.push(nvars + i);
            cols.sort_unstable();
            for c in cols {
                s.push_str(&format!("{c} 0.0\n"));
            }
        }
        s.push_str(&format!("G0 {n}\n"));
        for j in 0..n {
            s.push_str(&format!("{j} 0.0\n"));
        }
        s
    }

    /// Both prelude sweeps run once per color, so iterating the whole prelude
    /// each time would cost `n_colors × |prelude|` where the op-ratio gate
    /// assumes `|prelude|` — a cost the gate cannot see (PR #559 review).
    /// `eval_h` instead walks the union of that color's summands'
    /// `prelude_reach`.
    ///
    /// What this can and cannot pin is worth being exact about, because the
    /// change is pure performance: walking the whole prelude per color
    /// computes the *same* Hessian, so no assertion on output values can
    /// detect it, and a timing assertion would be flaky. So this asserts the
    /// two things that are checkable — that the table the sweeps iterate is
    /// strictly smaller than the naive `n_colors × |prelude|` walk on a model
    /// where colors genuinely reach different bodies, and that each reach list
    /// is ascending and operand-closed, the invariants that make the narrowed
    /// walk safe — plus agreement with the flat tapes under narrowing.
    #[test]
    fn per_color_prelude_reach_skips_bodies_the_color_cannot_touch() {
        let p = parse_nl_text(&two_block_shared_cse_nl(6, 2, 3)).expect("parse");
        let mut hybrid = NlTnlp::new(p.clone());
        let info = hybrid.get_nlp_info().unwrap();
        let nnz = info.nnz_h_lag as usize;
        let m = info.m as usize;

        {
            let h = hybrid
                .con_hybrid
                .as_mut()
                .expect("two shared CSE bodies must build the hybrid tape");
            h.use_for_hess = true;

            let np = h.tape.n_prelude_ops();
            let n_colors = h.hess_color_reach_off.len() - 1;
            let total: usize = h.hess_color_reach.len();
            assert!(np > 0 && n_colors > 1, "np={np} n_colors={n_colors}");
            assert!(
                total < n_colors * np,
                "per-color reach must be strictly smaller than walking the whole \
                 prelude per color: Σ|reach_c| = {total}, n_colors × |prelude| = {}",
                n_colors * np
            );
            // Every reach list must be ascending and operand-closed, which is
            // what makes the narrowed walk safe.
            for c in 0..n_colors {
                let r =
                    &h.hess_color_reach[h.hess_color_reach_off[c]..h.hess_color_reach_off[c + 1]];
                assert!(
                    r.windows(2).all(|w| w[0] < w[1]),
                    "color {c} reach is not strictly ascending"
                );
                let member: std::collections::HashSet<u32> = r.iter().copied().collect();
                for &i in r {
                    let (a, b) = crate::nl_tape::op_operands(&h.tape.prelude[i as usize]);
                    for opnd in [a, b].into_iter().flatten() {
                        assert!(
                            member.contains(&(opnd as u32)),
                            "color {c}: slot {i}'s operand {opnd} is missing from its reach"
                        );
                    }
                }
            }
        }

        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        let lam: Vec<f64> = (0..m).map(|k| 0.3 + 0.2 * k as f64).collect();
        for scale in [1.0_f64, -0.6] {
            let x: Vec<f64> = (0..info.n as usize)
                .map(|j| scale * (0.15 + 0.02 * j as f64))
                .collect();
            let mut hh = vec![0.0_f64; nnz];
            let mut hf = vec![0.0_f64; nnz];
            assert!(hybrid.eval_h(
                Some(&x),
                true,
                1.0,
                Some(&lam),
                true,
                SparsityRequest::Values { values: &mut hh }
            ));
            assert!(flat.eval_h(
                Some(&x),
                true,
                1.0,
                Some(&lam),
                true,
                SparsityRequest::Values { values: &mut hf }
            ));
            for k in 0..nnz {
                let tol = 1e-12 * hf[k].abs().max(1.0);
                assert!(
                    (hh[k] - hf[k]).abs() <= tol,
                    "narrowed-reach Hessian entry {k} at scale {scale}: \
                     hybrid {} vs flat {}",
                    hh[k],
                    hf[k]
                );
            }
            assert!(hh.iter().any(|v| *v != 0.0), "all-zero Hessian is no test");
        }
    }

    /// The Hessian gate is off for a model whose shared bodies are small —
    /// below the ratio where the shared prelude sweeps pay for the hybrid
    /// traversal's per-op overhead — leaving `eval_h` on the flat path
    /// (which the gate keeps bit-identical for such models by definition).
    #[test]
    fn a_small_shared_body_leaves_the_hessian_on_the_flat_path() {
        let p = parse_nl_text(&shared_cse_nl("o2\nv3\nv2")).expect("parse");
        let mut t = NlTnlp::new(p);
        t.get_nlp_info().unwrap();
        let h = t.con_hybrid.as_ref().expect("hybrid built for eval_g");
        assert!(
            !h.use_for_hess,
            "a 3-row model with a 2-term CSE is below the Hessian op-ratio gate"
        );
    }

    /// A CSE referenced from several constraints is evaluated once per
    /// `eval_g` via the shared prelude instead of once per reference. The
    /// values must be bit-identical to the flat per-summand `Tape` path,
    /// which is what makes the optimization safe to apply unconditionally.
    #[test]
    fn shared_cse_constraint_tape_matches_flat_tape_bit_for_bit() {
        let nl = shared_cse_nl("o2\nv3\nv2");
        let p = parse_nl_text(&nl).expect("parse shared-CSE model");
        let mut hybrid = NlTnlp::new(p.clone());
        hybrid.get_nlp_info().unwrap();
        let h = hybrid
            .con_hybrid
            .as_ref()
            .expect("CSE shared by 3 constraints must take the hybrid path");
        assert!(
            h.tape.n_prelude_ops() > 0,
            "shared CSE body must land in the prelude"
        );

        // Same model with the hybrid path switched off: the reference.
        let mut flat = NlTnlp::new(p);
        flat.get_nlp_info().unwrap();
        flat.con_hybrid = None;

        for x in [[1.0, 1.0, 1.0], [-2.0, 0.5, 3.0], [0.0, -1.5, -0.25]] {
            let mut gh = [0.0_f64; 3];
            let mut gf = [0.0_f64; 3];
            assert!(hybrid.eval_g(&x, true, &mut gh));
            assert!(flat.eval_g(&x, true, &mut gf));
            // V3 = 2 x0 + 3 x1.
            let s = 2.0 * x[0] + 3.0 * x[1];
            let want = [s * s, s * s * s + x[2], s * x[2]];
            for i in 0..3 {
                assert_eq!(gh[i], gf[i], "row {i} differs from the flat tape at {x:?}");
                assert!(
                    (gh[i] - want[i]).abs() < 1e-9,
                    "row {i}: got {}, want {}",
                    gh[i],
                    want[i]
                );
            }
        }
    }

    /// `HybridTape::build_multi` *panics* on comparisons, AND/OR/NOT,
    /// if-then-else, min/max lists and external funcalls, so `eval_g` may only
    /// take that path after `hybrid_supported` clears the model. Here a
    /// min-list in one constraint has to disable it for the whole block —
    /// falling back, not panicking.
    #[test]
    fn unsupported_opcode_falls_back_to_the_flat_tape() {
        let nl = shared_cse_nl("o11\n2\nv3\nv2");
        let p = parse_nl_text(&nl).expect("parse min-list model");
        let mut tnlp = NlTnlp::new(p);
        tnlp.get_nlp_info().unwrap();
        assert!(
            tnlp.con_hybrid.is_none(),
            "a min-list anywhere in the constraint block must disable the hybrid path"
        );
        let mut g = [0.0_f64; 3];
        assert!(tnlp.eval_g(&[-2.0, 0.5, 3.0], true, &mut g));
        let s = 2.0 * -2.0 + 3.0 * 0.5; // -2.5
        assert!((g[0] - s * s).abs() < 1e-9);
        assert!((g[2] - s.min(3.0)).abs() < 1e-9, "min(V3, x2) = {}", g[2]);
    }

    // ---- In-memory construction + HVP (issue #469) --------------------

    fn v(i: usize) -> Expr {
        Expr::Var(i)
    }

    fn c(x: Number) -> Expr {
        Expr::Const(x)
    }

    fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }

    fn un(op: UnaryOp, a: Expr) -> Expr {
        Expr::Unary(op, Box::new(a))
    }

    /// `NlProblemParts` for an `n`-variable, unbounded model.
    fn parts(n: usize, objective: Expr, constraints: Vec<Expr>) -> NlProblemParts {
        let m = constraints.len();
        NlProblemParts {
            minimize: true,
            objective,
            obj_constant: 0.0,
            constraints,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![0.0; n],
            g_l: vec![-1e19; m],
            g_u: vec![1e19; m],
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// A model built from expressions evaluates exactly like a parsed one:
    /// objective, gradient, constraints, and Jacobian all come from the
    /// same tape, with no `.nl` text in the loop.
    #[test]
    fn from_expressions_builds_evaluable_problem() {
        // min (1-x0)^2 + 100*(x1 - x0^2)^2   s.t.  x0^2 + x1^2 <= 2
        let rosen = bin(
            BinOp::Add,
            bin(BinOp::Pow, bin(BinOp::Sub, c(1.0), v(0)), c(2.0)),
            bin(
                BinOp::Mul,
                c(100.0),
                bin(
                    BinOp::Pow,
                    bin(BinOp::Sub, v(1), bin(BinOp::Pow, v(0), c(2.0))),
                    c(2.0),
                ),
            ),
        );
        let circle = bin(
            BinOp::Add,
            bin(BinOp::Pow, v(0), c(2.0)),
            bin(BinOp::Pow, v(1), c(2.0)),
        );

        let mut p = parts(2, rosen, vec![circle]);
        p.g_l = vec![0.0];
        p.g_u = vec![2.0];
        p.x0 = vec![-1.2, 1.0];
        p.var_names = names(&["x", "y"]);
        p.con_names = names(&["circle"]);

        let prob = NlProblem::from_expressions(p).expect("build");
        assert_eq!((prob.n, prob.m), (2, 1));
        assert_eq!(prob.var_names, names(&["x", "y"]));

        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        // f(-1.2, 1) = (2.2)^2 + 100*(1 - 1.44)^2 = 4.84 + 19.36 = 24.2
        let f = t.eval_f(&[-1.2, 1.0], true).unwrap();
        assert!((f - 24.2).abs() < 1e-10, "f = {f}");

        // ∇f = (-2(1-x0) - 400 x0 (x1 - x0^2), 200 (x1 - x0^2))
        //    = (4.4 + 480*(-0.44)... ) — computed below rather than
        //      transcribed, so the check is the formula, not an editor.
        let (x0, x1) = (-1.2, 1.0);
        let want = [
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0),
        ];
        let mut g = [0.0_f64; 2];
        assert!(t.eval_grad_f(&[x0, x1], true, &mut g));
        for j in 0..2 {
            assert!((g[j] - want[j]).abs() < 1e-8, "g[{j}] = {} ", g[j]);
        }

        // g(x) = x0^2 + x1^2 = 2.44
        let mut gv = [0.0_f64; 1];
        assert!(t.eval_g(&[x0, x1], true, &mut gv));
        assert!((gv[0] - 2.44).abs() < 1e-10, "g = {}", gv[0]);
    }

    /// `min`/`max`, `atan2`, and `erf` all reach the evaluator through
    /// this path. None of the three survives a `.nl` round trip in a
    /// typical frontend — `atan2` has no two-argument funcall path,
    /// `min`/`max` force a DNLP model type, and AMPL has no `erf` opcode
    /// at all — which is the reason the in-memory door exists.
    #[test]
    fn from_expressions_carries_ops_nl_cannot_express() {
        let obj = Expr::Sum(vec![
            bin(BinOp::Atan2, v(0), v(1)),
            Expr::MinList(vec![v(0), v(1)]),
            Expr::MaxList(vec![v(0), v(1)]),
            un(UnaryOp::Erf, v(0)),
        ]);
        let prob = NlProblem::from_expressions(parts(2, obj, Vec::new())).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        let x: [Number; 2] = [0.8, 1.5];
        // atan2 + min + max + erf; min+max == x0+x1 for any pair.
        let want = x[0].atan2(x[1]) + x[0] + x[1] + crate::nl_tape::erf(x[0]);
        let f = t.eval_f(&x, true).unwrap();
        assert!((f - want).abs() < 1e-12, "f = {f}, want {want}");
    }

    /// A `Var` index past `n` would be an out-of-bounds read in the
    /// tape's forward sweep. It has to be caught at construction, while
    /// it is still a diagnosable user error.
    #[test]
    fn from_expressions_rejects_out_of_range_var() {
        let err = NlProblem::from_expressions(parts(2, v(5), Vec::new()))
            .expect_err("Var(5) with n = 2 must be rejected");
        assert!(err.contains("Var(5)"), "{err}");

        let err = NlProblem::from_expressions(parts(2, c(0.0), vec![v(2)]))
            .expect_err("constraint Var(2) with n = 2 must be rejected");
        assert!(err.contains("constraint 0"), "{err}");

        // Length mismatches are errors too, not panics.
        let mut p = parts(2, c(0.0), Vec::new());
        p.x0 = vec![0.0; 3];
        let err = NlProblem::from_expressions(p).expect_err("x0 length must be checked");
        assert!(err.contains("x0"), "{err}");
    }

    /// An out-of-range `Var` must be caught wherever it hides, not just at
    /// the top level — inside a `Cse` body, a nested `Cse`, a `Cond`
    /// branch, and a funcall argument all reach the same forward sweep.
    #[test]
    fn from_expressions_finds_out_of_range_vars_in_every_position() {
        let inner_cse = Arc::new(v(7));
        let cases: Vec<(&str, Expr)> = vec![
            ("bare", v(7)),
            ("cse", Expr::Cse(Arc::new(v(7)))),
            ("nested cse", Expr::Cse(Arc::new(Expr::Cse(inner_cse)))),
            (
                "cond branch",
                Expr::Cond {
                    cond: Box::new(c(1.0)),
                    then_: Box::new(v(7)),
                    else_: Box::new(c(0.0)),
                },
            ),
            ("min list", Expr::MinList(vec![c(0.0), v(7)])),
            (
                "sum",
                Expr::Sum(vec![c(0.0), bin(BinOp::Mul, c(2.0), v(7))]),
            ),
        ];
        for (label, e) in cases {
            let err = NlProblem::from_expressions(parts(2, e, Vec::new()))
                .err()
                .unwrap_or_else(|| panic!("{label}: Var(7) with n = 2 should be rejected"));
            assert!(err.contains("Var(7)"), "{label}: {err}");
        }
    }

    /// A balanced share-DAG: each level is one `Cse` whose body
    /// references the level below twice. Depth `d` is `d` distinct nodes
    /// but `2^d` paths, so any walk that re-enters a shared body per
    /// occurrence is Θ(2^d).
    fn share_dag(depth: usize) -> Expr {
        let mut e = v(0);
        for _ in 0..depth {
            let shared = Arc::new(e);
            e = bin(
                BinOp::Add,
                Expr::Cse(Arc::clone(&shared)),
                Expr::Cse(shared),
            );
        }
        e
    }

    /// The walks over a shared DAG must be memoized, not exponential.
    ///
    /// At depth 30 an unmemoized walk is ~10^9 node visits — this test
    /// does not "fail" so much as never finish, which is exactly the
    /// signal. `from_expressions` is the door that makes such a DAG
    /// trivially constructible, but the blowup was reachable through
    /// `collect_vars` (which presolve calls on every solve, via
    /// `get_variables_linearity`) and `collect_funcall_ids` (which
    /// `NlTnlp::try_new` runs over every row).
    #[test]
    fn shared_dag_walks_are_memoized_not_exponential() {
        const DEPTH: usize = 30;
        let e = share_dag(DEPTH);

        let mut vars = BTreeSet::new();
        collect_vars(&e, &mut vars);
        assert_eq!(vars.iter().copied().collect::<Vec<_>>(), vec![0]);

        let mut ids = BTreeSet::new();
        super::super::nl_external::collect_funcall_ids(&e, &mut ids);
        assert!(ids.is_empty());

        // The whole build path, end to end: validation, tape construction,
        // and the linearity metadata presolve consumes.
        let prob = NlProblem::from_expressions(parts(1, e, Vec::new())).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();
        let mut lin = vec![Linearity::Linear; 1];
        assert!(t.get_variables_linearity(&mut lin));
    }

    /// `from_expressions` cannot carry AMPL imported functions — there is
    /// nowhere to put the `F`-segment declarations that bind a funcall id
    /// to a library — so a `Funcall` must be refused up front. Accepting it
    /// produces "AMPLFUNC is not set", which the user cannot act on:
    /// setting `AMPLFUNC` only moves the failure to "no F<id> declaration".
    #[test]
    fn from_expressions_rejects_imported_function_calls() {
        let call = Expr::Funcall {
            id: 0,
            args: vec![FuncallArg::Real(v(0))],
        };
        let err = NlProblem::from_expressions(parts(1, call.clone(), Vec::new()))
            .expect_err("a Funcall must be rejected, not deferred to AMPLFUNC");
        assert!(err.contains("imported function"), "{err}");
        assert!(
            err.contains("read_nl") || err.contains("parse_nl_text"),
            "the error must point at the paths that do support externals: {err}"
        );

        // Also when buried in a constraint, behind a Cse.
        let buried = Expr::Cse(Arc::new(Expr::Sum(vec![c(1.0), call])));
        let err = NlProblem::from_expressions(parts(1, c(0.0), vec![buried]))
            .expect_err("a buried Funcall must be rejected too");
        assert!(err.contains("constraint 0"), "{err}");
    }

    /// The matrix-free HVP must reproduce `eval_h`'s Hessian exactly —
    /// same tapes, same weights, one seed instead of a color sweep. The
    /// objective and both constraints are chosen with cross terms so the
    /// off-diagonal blocks actually carry signal.
    #[test]
    fn hessian_vector_product_matches_dense_hessian() {
        let obj = Expr::Sum(vec![
            bin(BinOp::Mul, v(0), bin(BinOp::Mul, v(1), v(2))),
            un(UnaryOp::Exp, bin(BinOp::Mul, v(0), v(1))),
            un(UnaryOp::Erf, v(2)),
        ]);
        let cons = vec![
            bin(
                BinOp::Add,
                bin(BinOp::Pow, v(0), c(2.0)),
                un(UnaryOp::Sin, v(2)),
            ),
            bin(BinOp::Mul, v(1), v(2)),
        ];
        let prob = NlProblem::from_expressions(parts(3, obj, cons)).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        let info = t.get_nlp_info().unwrap();

        let x = [0.3, -0.7, 1.1];
        let lam = [0.5, -1.25];
        let obj_factor = 2.0;

        // Dense Hessian from the sparse lower triangle.
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0_i32; nnz], vec![0_i32; nnz]);
        assert!(t.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        let mut hvals = vec![0.0_f64; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            obj_factor,
            Some(&lam),
            true,
            SparsityRequest::Values { values: &mut hvals }
        ));
        let mut dense = [[0.0_f64; 3]; 3];
        for k in 0..nnz {
            let (i, j) = (irow[k] as usize, jcol[k] as usize);
            dense[i][j] += hvals[k];
            if i != j {
                dense[j][i] += hvals[k];
            }
        }

        // Each unit seed recovers a column; a mixed seed catches an HVP
        // that only happens to be right on the basis vectors.
        let seeds: [[Number; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.4, -1.3, 2.0],
        ];
        let mut out = vec![0.0; 3];
        for s in &seeds {
            t.hessian_vector_product(&x, s, obj_factor, Some(&lam), &mut out)
                .expect("hvp");
            for i in 0..3 {
                let want: Number = (0..3).map(|j| dense[i][j] * s[j]).sum();
                assert!(
                    (out[i] - want).abs() < 1e-9,
                    "seed {s:?} row {i}: hvp={:.9e} dense={want:.9e}",
                    out[i]
                );
            }
        }
    }

    /// `lam = None` is the objective block alone, and `out` is
    /// overwritten (not accumulated) so a reused buffer is safe.
    #[test]
    fn hessian_vector_product_defaults_and_validation() {
        // f = x0^2 + 3 x0 x1  ->  ∇²f = [[2, 3], [3, 0]]
        let obj = bin(
            BinOp::Add,
            bin(BinOp::Pow, v(0), c(2.0)),
            bin(BinOp::Mul, c(3.0), bin(BinOp::Mul, v(0), v(1))),
        );
        let prob = NlProblem::from_expressions(parts(2, obj, Vec::new())).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        let mut out = vec![7.0, -7.0]; // dirty buffer
        t.hessian_vector_product(&[0.5, 2.0], &[1.0, 1.0], 1.0, None, &mut out)
            .expect("hvp");
        assert!((out[0] - 5.0).abs() < 1e-12, "out = {out:?}");
        assert!((out[1] - 3.0).abs() < 1e-12, "out = {out:?}");

        // obj_factor scales linearly.
        t.hessian_vector_product(&[0.5, 2.0], &[1.0, 1.0], -2.0, None, &mut out)
            .expect("hvp");
        assert!((out[0] + 10.0).abs() < 1e-12, "out = {out:?}");

        // Length mismatches are errors, not panics or silent truncation.
        let mut short = vec![0.0; 1];
        assert!(
            t.hessian_vector_product(&[0.5, 2.0], &[1.0, 1.0], 1.0, None, &mut short)
                .is_err()
        );
        assert!(
            t.hessian_vector_product(&[0.5], &[1.0, 1.0], 1.0, None, &mut out)
                .is_err()
        );
        assert!(
            t.hessian_vector_product(&[0.5, 2.0], &[1.0], 1.0, None, &mut out)
                .is_err()
        );
    }

    /// A chain objective `Σ (x_i·x_{i+1})² + exp(x_i)` has a tridiagonal
    /// Hessian — the sparse shape an IPM actually meets. The block HVP has
    /// to reproduce it column for column, including the structural zeros:
    /// a bug that leaked coupling between non-adjacent variables would
    /// show up here and nowhere in a small dense test.
    #[test]
    fn hessian_vector_products_on_a_sparse_hessian() {
        const N: usize = 8;
        let mut terms = Vec::new();
        for i in 0..N - 1 {
            terms.push(bin(BinOp::Pow, bin(BinOp::Mul, v(i), v(i + 1)), c(2.0)));
        }
        for i in 0..N {
            terms.push(un(UnaryOp::Exp, v(i)));
        }
        let prob =
            NlProblem::from_expressions(parts(N, Expr::Sum(terms), Vec::new())).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        let info = t.get_nlp_info().unwrap();

        // Tridiagonal lower triangle: N diagonal + (N-1) sub-diagonal.
        assert_eq!(
            info.nnz_h_lag as usize,
            2 * N - 1,
            "chain objective should give a tridiagonal Hessian, not a dense one"
        );

        let x: Vec<Number> = (0..N).map(|i| 0.2 + 0.1 * i as Number).collect();

        // Densify the sparse triangle for the reference.
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0_i32; nnz], vec![0_i32; nnz]);
        assert!(t.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        let mut hvals = vec![0.0; nnz];
        assert!(t.eval_h(
            Some(&x),
            true,
            1.0,
            None,
            true,
            SparsityRequest::Values { values: &mut hvals }
        ));
        let mut dense = vec![vec![0.0; N]; N];
        for k in 0..nnz {
            let (i, j) = (irow[k] as usize, jcol[k] as usize);
            dense[i][j] += hvals[k];
            if i != j {
                dense[j][i] += hvals[k];
            }
        }

        // One block of N unit seeds recovers the whole matrix — the
        // "densify via HVPs" path, in one call.
        let mut seeds = vec![0.0; N * N];
        for cc in 0..N {
            seeds[cc * N + cc] = 1.0;
        }
        let mut out = vec![0.0; N * N];
        t.hessian_vector_products(&x, &seeds, N, 1.0, None, &mut out)
            .expect("block hvp");
        for cc in 0..N {
            for i in 0..N {
                assert!(
                    (out[cc * N + i] - dense[i][cc]).abs() < 1e-9,
                    "H[{i},{cc}]: block={:.9e} sparse={:.9e}",
                    out[cc * N + i],
                    dense[i][cc]
                );
            }
        }
    }

    /// The block form must agree with `k` separate single-vector calls
    /// (it shares one forward sweep across directions, so a bug there
    /// would show as a per-direction discrepancy), and must skip an
    /// all-zero direction without disturbing its neighbours.
    #[test]
    fn hessian_vector_products_match_repeated_single_calls() {
        let obj = Expr::Sum(vec![
            un(UnaryOp::Exp, bin(BinOp::Mul, v(0), v(1))),
            bin(BinOp::Pow, v(2), c(4.0)),
            bin(BinOp::Mul, v(0), v(2)),
        ]);
        let cons = vec![bin(BinOp::Mul, v(1), v(2))];
        let prob = NlProblem::from_expressions(parts(3, obj, cons)).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        let x = [0.4, -0.6, 1.3];
        let lam = [0.75];
        let cols: [[Number; 3]; 4] = [
            [1.0, 2.0, -3.0],
            [0.0, 0.0, 0.0], // the skipped direction
            [0.5, 0.0, 0.0],
            [-1.0, 1.0, 1.0],
        ];

        let mut block = vec![0.0; 3 * cols.len()];
        let flat: Vec<Number> = cols.iter().flat_map(|c| c.iter().copied()).collect();
        t.hessian_vector_products(&x, &flat, cols.len(), 1.0, Some(&lam), &mut block)
            .expect("block hvp");

        for (c, col) in cols.iter().enumerate() {
            let mut single = vec![0.0; 3];
            t.hessian_vector_product(&x, col, 1.0, Some(&lam), &mut single)
                .expect("single hvp");
            for i in 0..3 {
                assert!(
                    (block[c * 3 + i] - single[i]).abs() < 1e-12,
                    "direction {c} row {i}: block={:.12e} single={:.12e}",
                    block[c * 3 + i],
                    single[i]
                );
            }
        }
        // The zero direction really is zero, not stale scratch.
        assert!(block[3..6].iter().all(|&z| z == 0.0), "{block:?}");
    }

    /// `k = 0` is a legal empty block, and the length checks scale with
    /// `k` rather than assuming a single direction.
    #[test]
    fn hessian_vector_products_validate_block_shape() {
        let prob = NlProblem::from_expressions(parts(2, bin(BinOp::Pow, v(0), c(2.0)), Vec::new()))
            .expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        let mut empty: Vec<Number> = Vec::new();
        assert!(
            t.hessian_vector_products(&[1.0, 1.0], &[], 0, 1.0, None, &mut empty)
                .is_ok()
        );

        // v sized for one direction while k says two.
        let mut out = vec![0.0; 4];
        assert!(
            t.hessian_vector_products(&[1.0, 1.0], &[1.0, 1.0], 2, 1.0, None, &mut out)
                .is_err()
        );
        // out sized for one direction while k says two.
        let mut short = vec![0.0; 2];
        assert!(
            t.hessian_vector_products(&[1.0, 1.0], &[1.0; 4], 2, 1.0, None, &mut short)
                .is_err()
        );
    }

    /// A `maximize` model's objective is negated by the evaluator, and
    /// the HVP has to agree with `eval_h` about that — otherwise a
    /// Hessian-free step would climb where the sparse path descends.
    #[test]
    fn hessian_vector_product_respects_maximize_sign() {
        let obj = bin(BinOp::Pow, v(0), c(2.0));
        let mut p = parts(1, obj, Vec::new());
        p.minimize = false;
        let prob = NlProblem::from_expressions(p).expect("build");
        let mut t = NlTnlp::try_new(prob).expect("tnlp");
        t.get_nlp_info().unwrap();

        // max x0^2 is minimized as -x0^2, so ∇² = -2.
        let mut out = vec![0.0; 1];
        t.hessian_vector_product(&[1.0], &[1.0], 1.0, None, &mut out)
            .expect("hvp");
        assert!((out[0] + 2.0).abs() < 1e-12, "out = {out:?}");
    }
}
