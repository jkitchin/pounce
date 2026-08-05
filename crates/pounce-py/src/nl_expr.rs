//! In-memory model construction for Python (issue #469).
//!
//! [`PyNlExpr`] is a thin handle on [`pounce_nl::nl_reader::Expr`], the same
//! expression DAG the `.nl` parser produces, with Python operators wired to
//! its nodes. [`build_nl_problem`] assembles a set of those expressions into
//! an [`crate::PyNlProblem`] — so a modeling frontend can go straight from
//! its own DAG to pounce's AD tape, with no `.nl` file on disk and no
//! parser in the middle:
//!
//! ```python
//! import pounce
//! x = pounce.NlExpr.vars(2)
//! rosen = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
//! p = pounce.build_nl_problem(
//!     n=2,
//!     objective=rosen,
//!     constraints=[x[0] ** 2 + x[1] ** 2],
//!     g_l=[0.0], g_u=[1.0],
//! )
//! p.gradient([0.5, 0.5])          # same evaluator `read_nl` returns
//! ```
//!
//! Going through `.nl` is not merely slower, it is lossy: `.nl` writers
//! commonly refuse `atan2` (no two-argument funcall path) and `min`/`max`
//! (they force a DNLP model type), and AMPL has no `erf` opcode at all —
//! yet the tape supports all four. Built here, they survive.
//!
//! [`PyNlExpr::eval`] and [`PyNlExpr::gradient`] expose
//! `pounce_nl::nl_tape::Tape::build` on a single scalar expression, which is
//! mostly useful for checking a subexpression in isolation before wiring it
//! into a model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use std::collections::HashMap;
use std::sync::Arc;

use pounce_common::types::Number;
use pounce_nl::nl_reader::{
    BinOp, CmpOp, Expr, FuncallArg, NlProblem, NlProblemParts, NlTnlp, UnaryOp, render_expression,
};
use pounce_nl::nl_tape::Tape;

use crate::nl_problem::PyNlProblem;

/// The `.nl` reader's "unbounded" sentinel. Bounds at or beyond this
/// magnitude are treated as absent by the solver, so it is the right
/// default for an omitted bound vector.
const INF: Number = 1e19;

/// Deepest expression any Python entry point will hand on: what
/// [`PyNlExpr`] will build, and what `read_nl` / `parse_nl_text` will
/// accept from a file (pounce#472).
///
/// Everything that consumes an `Expr` — `Tape::build`, [`materialize`],
/// the problem assembler, the derived `Drop` — recurses once per level, as
/// does the `.nl` parser that produces one. Overrunning the stack is a
/// hard crash, not a Python exception: the interpreter dies with SIGSEGV
/// and no traceback. Two things together keep that unreachable, and the
/// limit is the second of them, not the first:
///
/// * every such walk runs on a worker thread with a [`WORKER_STACK`]-byte
///   stack (see [`on_deep_stack`]), so what is survivable stops depending
///   on the calling thread — 8 MB on a macOS/Linux main thread, 1 MB on
///   Windows, less on a `threading.Thread`;
/// * this limit then keeps the depth well inside what that stack holds.
///
/// 10 000 is the arithmetic: the deepest frames measured are ~300 bytes in
/// a release build and ~2 KB in a debug build, so 64 MB covers ~200 000
/// levels released and ~32 000 in debug — a 3x margin at this limit even
/// in the worst configuration. It is also comfortably past what the parser
/// managed before it was guarded (~3 000), so no `.nl` file that loaded
/// before is refused now.
///
/// Both surfaces share the number deliberately. A cap on one door and none
/// on the other is what the issue reported second: `NlExpr` refusing depth
/// 1 001 while `parse_nl_text` accepted 3 000 and crashed on 4 000.
pub(crate) const MAX_DEPTH: u32 = 10_000;

/// Depth up to which recursive walks run directly on the calling thread.
///
/// At the ~300-byte release frame this is ~150 KB, safe on a 1 MB Windows
/// main thread with a Python call stack above it. Past it, the walk moves
/// to the worker.
pub(crate) const INLINE_DEPTH: u32 = 512;

/// Stack for the worker thread. Reserved, not committed: only the pages
/// actually touched cost anything. See [`MAX_DEPTH`] for the margin this
/// buys.
const WORKER_STACK: usize = 64 << 20;

/// Run `f` on a worker thread with a stack sized for recursion over a
/// deep expression, whatever the caller's stack is.
///
/// For work whose depth is not known ahead of time — parsing a `.nl` file
/// is exactly that, since the depth is what the file says — rather than
/// [`on_deep_stack`], which skips the worker for shallow input.
pub(crate) fn on_worker_stack<T, F>(f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn_scoped(scope, f);
        let handle = match spawned {
            Ok(h) => h,
            // Nothing safe is left to do: running the walk here is the
            // segfault this function exists to prevent. Panic instead, so
            // pyo3 raises it as a Python exception.
            Err(e) => panic!("pounce: cannot spawn the expression worker thread: {e}"),
        };
        match handle.join() {
            Ok(v) => v,
            // Carry a panic across the join so pyo3 still turns it into a
            // Python exception rather than losing it here.
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// [`on_worker_stack`], skipped when `depth` says the walk fits anywhere.
///
/// The spawn costs on the order of 100 µs. Nothing per-operator goes
/// through here — building an expression does not walk it — so that is
/// paid once per query, per assembled problem, and per deep teardown.
pub(crate) fn on_deep_stack<T, F>(depth: u32, f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if depth <= INLINE_DEPTH {
        return f();
    }
    on_worker_stack(f)
}

/// Nesting depth of an already-built expression, for input that did not
/// come through [`PyNlExpr`]'s O(1) accounting — a parsed `.nl` model.
///
/// Recursive, so it must itself run inside [`on_worker_stack`]. Memoized
/// on shared `Cse` bodies: a body has one depth however many references
/// reach it, and without the memo a sharing DAG costs one visit per path
/// rather than per node.
pub(crate) fn expr_depth(e: &Expr, memo: &mut HashMap<*const Expr, u32>) -> u32 {
    let below = |kids: &mut dyn Iterator<Item = &Expr>, memo: &mut HashMap<*const Expr, u32>| {
        kids.fold(0, |acc, k| acc.max(expr_depth(k, memo)))
    };
    1 + match e {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            below(&mut [&**a, &**b].into_iter(), memo)
        }
        Expr::Unary(_, a) | Expr::Not(a) => expr_depth(a, memo),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            below(&mut args.iter(), memo)
        }
        Expr::Cond { cond, then_, else_ } => {
            below(&mut [&**cond, &**then_, &**else_].into_iter(), memo)
        }
        Expr::Funcall { args, .. } => below(
            &mut args.iter().filter_map(|a| match a {
                FuncallArg::Real(inner) => Some(inner),
                FuncallArg::Str(_) => None,
            }),
            memo,
        ),
        Expr::Cse(body) => {
            let key = Arc::as_ptr(body);
            if let Some(&d) = memo.get(&key) {
                d - 1
            } else {
                let d = expr_depth(body, memo);
                memo.insert(key, d + 1);
                d
            }
        }
    }
}

/// A node in an expression DAG, and the building block of
/// [`build_nl_problem`].
///
/// **Cheap to use as an operand.** The payload is behind an `Arc`, and an
/// operator references it through an [`Expr::Cse`] node rather than
/// copying it, so `a + b` costs the same whether the operands are two
/// variables or two half-million-node models. That is what keeps
/// accumulation in a Python loop linear:
///
/// ```python
/// e = pounce.NlExpr.const_(0.0)
/// for t in terms:                 # O(len(terms)) — each `+` is O(1), but
///     e = e + t                   # nests one level deeper per term
///
/// e = pounce.NlExpr.sum(terms)    # O(len(terms)) and one level, whatever
/// ```                             # the term count
///
/// `sum` is still the right spelling for a many-term sum: one flat n-ary
/// node tapes better than a chain, and it does not spend a level of
/// nesting per term, so [`MAX_DEPTH`] never binds on a wide model however
/// data-driven its width.
///
/// Sharing is invisible in the result. A subexpression reached more than
/// once stays a `Cse` — a shared body, which the tape emits once and whose
/// adjoint sums the contributions from every reference, i.e. exactly what
/// duplicating it would compute — and one reached a single time is inlined
/// when the model is assembled, so the problem sees the same plain tree it
/// would have seen from a `.nl` file.
#[pyclass(module = "pounce", name = "NlExpr")]
#[derive(Clone)]
pub struct PyNlExpr {
    /// Shared so that using this expression as an operand is O(1). Every
    /// consumer treats it as immutable.
    pub(crate) inner: Arc<Expr>,
    /// Nesting depth of `inner`, maintained incrementally (O(1) per
    /// operation) so the [`MAX_DEPTH`] check never has to walk the tree.
    /// Counts this expression's own operators; the `Cse` node each operand
    /// reference adds can double that in the tree a walk actually sees,
    /// which is part of why `MAX_DEPTH` leaves the worker stack so much
    /// room.
    depth: u32,
}

/// Dropping the last handle on a deep chain releases it one `Arc` at a
/// time, recursing once per level — and a drop is the one thing the caller
/// cannot decline, since the interpreter runs it whenever the object is
/// collected. Hand the teardown to the worker thread (pounce#472).
impl Drop for PyNlExpr {
    fn drop(&mut self) {
        if self.depth <= INLINE_DEPTH || Arc::strong_count(&self.inner) > 1 {
            return;
        }
        let doomed = std::mem::replace(&mut self.inner, Arc::new(Expr::Const(0.0)));
        on_deep_stack(self.depth, move || drop(doomed));
    }
}

/// An operand decoded from Python: the expression plus its depth, so the
/// consumer can compute its own depth without a walk.
///
/// The expression is either a leaf or a `Cse` reference to an existing
/// [`PyNlExpr`]'s payload, so building one, moving it, and dropping it are
/// all O(1) however large the operand is.
struct Operand {
    expr: Expr,
    depth: u32,
}

impl PyNlExpr {
    /// Wrap an expression whose depth is already known to be in range.
    /// Only for leaves (`Var` / `Const`), where depth is 1 by definition.
    fn leaf(inner: Expr) -> PyNlExpr {
        PyNlExpr {
            inner: Arc::new(inner),
            depth: 1,
        }
    }

    /// Wrap a node built over operands of depth `child_depth`, rejecting
    /// it if that puts the result past [`MAX_DEPTH`].
    ///
    /// The `PyNlExpr` is formed *before* the check so that a rejected node
    /// is torn down by the `Drop` above rather than recursively here.
    fn nested(inner: Expr, child_depth: u32) -> PyResult<PyNlExpr> {
        let depth = child_depth.saturating_add(1);
        let built = PyNlExpr {
            inner: Arc::new(inner),
            depth,
        };
        if depth > MAX_DEPTH {
            return Err(PyValueError::new_err(format!(
                "NlExpr: expression nesting would reach depth {depth}, past the \
                 limit of {MAX_DEPTH}. Deeper trees overflow the stack when the \
                 expression is taped, walked, or freed — a hard crash rather than \
                 an exception — so they are refused here. If you are accumulating \
                 terms in a loop (`e = e + t`), use NlExpr.sum([...]) instead: it \
                 builds one n-ary node of depth 1, whatever the term count."
            )));
        }
        Ok(built)
    }

    /// This expression as an operand of a new node: a `Cse` reference to
    /// the payload, not a copy of it.
    ///
    /// Leaves are passed through as themselves — a `Cse` around a `Var`
    /// costs a pointer chase and a memo lookup in every walk and buys
    /// nothing, the leaf already being the smallest thing there is.
    fn operand(&self) -> Operand {
        let expr = match &*self.inner {
            leaf @ (Expr::Const(_) | Expr::Var(_)) => leaf.clone(),
            _ => Expr::Cse(Arc::clone(&self.inner)),
        };
        Operand {
            expr,
            depth: self.depth,
        }
    }
}

/// Accept an `NlExpr` or any Python float/int as an expression operand, so
/// `2 * x[0]` and `x[0] ** 2` read the way a modeler expects.
fn coerce(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Operand> {
    if let Ok(e) = v.extract::<PyRef<'_, PyNlExpr>>() {
        return Ok(e.operand());
    }
    // Reject strings explicitly: Python would happily `float("nan")` some
    // of them via `extract`, and silently turning "1e-3" into a constant
    // hides a typo rather than reporting it.
    if v.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: expected NlExpr or a number, got str"
        )));
    }
    match v.extract::<Number>() {
        Ok(c) => Ok(Operand {
            expr: Expr::Const(c),
            depth: 1,
        }),
        Err(_) => Err(PyTypeError::new_err(format!(
            "{what}: expected NlExpr or a number, got {}",
            v.get_type().name()?
        ))),
    }
}

fn binary(op: BinOp, a: Operand, b: Operand) -> PyResult<PyNlExpr> {
    PyNlExpr::nested(
        Expr::Binary(op, Box::new(a.expr), Box::new(b.expr)),
        a.depth.max(b.depth),
    )
}

fn unary(op: UnaryOp, a: Operand) -> PyResult<PyNlExpr> {
    PyNlExpr::nested(Expr::Unary(op, Box::new(a.expr)), a.depth)
}

/// Collect a Python iterable of operands, returning them with the deepest
/// operand's depth.
fn coerce_all(items: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<Expr>, u32)> {
    let mut out = Vec::new();
    let mut depth = 0;
    for item in items.iter()? {
        let o = coerce(&item?, what)?;
        depth = depth.max(o.depth);
        out.push(o.expr);
    }
    Ok((out, depth))
}

/// Number of distinct DAG nodes in `e`, stopping as soon as the count
/// exceeds `cap`. Used to keep `__repr__` from rendering a model-sized
/// expression (the renderer inlines shared bodies, so its output can be
/// exponentially larger than the DAG).
fn node_count_capped(e: &Expr, cap: usize, acc: &mut usize) {
    if *acc > cap {
        return;
    }
    *acc += 1;
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            node_count_capped(a, cap, acc);
            node_count_capped(b, cap, acc);
        }
        Expr::Unary(_, a) | Expr::Not(a) => node_count_capped(a, cap, acc),
        Expr::Cse(body) => node_count_capped(body, cap, acc),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                node_count_capped(a, cap, acc);
            }
        }
        Expr::Cond { cond, then_, else_ } => {
            node_count_capped(cond, cap, acc);
            node_count_capped(then_, cap, acc);
            node_count_capped(else_, cap, acc);
        }
        Expr::Funcall { args, .. } => {
            for a in args {
                if let pounce_nl::nl_reader::FuncallArg::Real(inner) = a {
                    node_count_capped(inner, cap, acc);
                }
            }
        }
    }
}

/// Map a Python comparison spelling onto a [`CmpOp`].
fn parse_cmp(op: &str) -> PyResult<CmpOp> {
    Ok(match op {
        "<" | "lt" => CmpOp::Lt,
        "<=" | "le" => CmpOp::Le,
        "==" | "eq" => CmpOp::Eq,
        ">=" | "ge" => CmpOp::Ge,
        ">" | "gt" => CmpOp::Gt,
        "!=" | "ne" => CmpOp::Ne,
        other => {
            return Err(PyValueError::new_err(format!(
                "compare: unknown operator {other:?}; expected one of \
                 '<', '<=', '==', '>=', '>', '!='"
            )));
        }
    })
}

#[pymethods]
impl PyNlExpr {
    /// Reference to problem variable `index` (0-based).
    #[staticmethod]
    fn var(index: usize) -> PyNlExpr {
        PyNlExpr::leaf(Expr::Var(index))
    }

    /// `[var(0), var(1), ..., var(n-1)]` — the usual first line of a model.
    #[staticmethod]
    fn vars(n: usize) -> Vec<PyNlExpr> {
        (0..n).map(PyNlExpr::var).collect()
    }

    /// A numeric literal. Rarely needed explicitly: plain Python numbers
    /// are accepted anywhere an `NlExpr` is.
    #[staticmethod]
    fn const_(value: Number) -> PyNlExpr {
        PyNlExpr::leaf(Expr::Const(value))
    }

    /// `sum(args)` as a single n-ary node — cheaper to build and to tape
    /// than a left-leaning chain of `+`.
    #[staticmethod]
    fn sum(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "sum")?;
        PyNlExpr::nested(Expr::Sum(items), depth)
    }

    /// `atan2(y, x)`, the two-argument arctangent. Has no `.nl` writer
    /// path in most frontends, which is one of the reasons this module
    /// exists.
    #[staticmethod]
    fn atan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Atan2, coerce(y, "atan2: y")?, coerce(x, "atan2: x")?)
    }

    /// n-ary minimum. Piecewise linear: the derivative follows whichever
    /// operand is currently smallest (ties pick the first) and the second
    /// derivative is identically zero — the standard AD treatment.
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn min(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "min")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("min: needs at least one operand"));
        }
        PyNlExpr::nested(Expr::MinList(items), depth)
    }

    /// n-ary maximum; mirrors [`PyNlExpr::min`].
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn max(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "max")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("max: needs at least one operand"));
        }
        PyNlExpr::nested(Expr::MaxList(items), depth)
    }

    /// Relational test `a <op> b`, evaluating to `1.0` or `0.0`. `op` is
    /// one of `'<' '<=' '==' '>=' '>' '!='`.
    ///
    /// Spelled as a function rather than as Python's `<` operator because
    /// overloading comparison would break every ordinary use of an
    /// `NlExpr` in a container. The result is piecewise constant, hence
    /// zero-derivative; pair it with [`PyNlExpr::select`].
    #[staticmethod]
    fn compare(op: &str, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let op = parse_cmp(op)?;
        let (a, b) = (coerce(a, "compare: a")?, coerce(b, "compare: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::Compare(op, Box::new(a.expr), Box::new(b.expr)), depth)
    }

    /// `then_ if cond else else_`. The value and all derivatives flow
    /// through the active branch only; the branch switch itself is a
    /// non-smooth event the AD ignores, exactly as ASL/Ipopt treat `if`.
    #[staticmethod]
    fn select(
        cond: &Bound<'_, PyAny>,
        then_: &Bound<'_, PyAny>,
        else_: &Bound<'_, PyAny>,
    ) -> PyResult<PyNlExpr> {
        let cond = coerce(cond, "select: cond")?;
        let then_ = coerce(then_, "select: then_")?;
        let else_ = coerce(else_, "select: else_")?;
        let depth = cond.depth.max(then_.depth).max(else_.depth);
        PyNlExpr::nested(
            Expr::Cond {
                cond: Box::new(cond.expr),
                then_: Box::new(then_.expr),
                else_: Box::new(else_.expr),
            },
            depth,
        )
    }

    /// Logical AND: `1.0` iff both operands are nonzero. Zero derivative.
    #[staticmethod]
    fn logical_and(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (a, b) = (coerce(a, "logical_and: a")?, coerce(b, "logical_and: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::And(Box::new(a.expr), Box::new(b.expr)), depth)
    }

    /// Logical OR: `1.0` iff either operand is nonzero. Zero derivative.
    #[staticmethod]
    fn logical_or(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (a, b) = (coerce(a, "logical_or: a")?, coerce(b, "logical_or: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::Or(Box::new(a.expr), Box::new(b.expr)), depth)
    }

    /// Logical NOT: `1.0` iff the operand is zero. Zero derivative.
    #[staticmethod]
    fn logical_not(a: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let a = coerce(a, "logical_not: a")?;
        let depth = a.depth;
        PyNlExpr::nested(Expr::Not(Box::new(a.expr)), depth)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Add, self.operand(), coerce(other, "+")?)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Add, coerce(other, "+")?, self.operand())
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Sub, self.operand(), coerce(other, "-")?)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Sub, coerce(other, "-")?, self.operand())
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Mul, self.operand(), coerce(other, "*")?)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Mul, coerce(other, "*")?, self.operand())
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Div, self.operand(), coerce(other, "/")?)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Div, coerce(other, "/")?, self.operand())
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNlExpr> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyValueError::new_err(
                "NlExpr ** exp % mod: three-argument pow is not supported",
            ));
        }
        binary(BinOp::Pow, self.operand(), coerce(other, "**")?)
    }

    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNlExpr> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyValueError::new_err(
                "base ** NlExpr % mod: three-argument pow is not supported",
            ));
        }
        binary(BinOp::Pow, coerce(other, "**")?, self.operand())
    }

    fn __neg__(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Neg, self.operand())
    }

    fn __pos__(&self) -> PyNlExpr {
        self.clone()
    }

    /// Copy support. `Expr` is a plain tree, so a copy is a deep copy
    /// either way — `__copy__` and `__deepcopy__` are the same operation,
    /// and both are what a frontend cloning its own DAG reaches for.
    /// (Pickling is *not* supported: an `NlExpr` has no serialized form,
    /// and pickling one raises `TypeError`. Rebuild from the model source
    /// instead.)
    fn __copy__(&self) -> PyNlExpr {
        self.clone()
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<&Bound<'_, PyAny>>) -> PyNlExpr {
        self.clone()
    }

    /// Nesting depth of this expression. Bounded by `NlExpr.max_depth`;
    /// exposed because it is the number in the error you get when a `+`
    /// chain runs away.
    #[getter]
    fn depth(&self) -> u32 {
        self.depth
    }

    /// The deepest expression this class will build. See the class
    /// docstring for why deep trees are refused rather than allowed to
    /// crash the interpreter.
    #[classattr]
    fn max_depth() -> u32 {
        MAX_DEPTH
    }

    /// Opt out of NumPy's ufunc protocol.
    ///
    /// Without this, `np.array([1.0, 2.0]) * x[0]` silently produces a
    /// `dtype=object` ndarray of `NlExpr` — NumPy broadcasts elementwise
    /// and calls `float.__mul__` per cell — while the forward form
    /// `x[0] * np.array(...)` correctly raises. Setting it to `None`
    /// makes the reflected form raise too, so the two directions agree
    /// and a vectorized expression has to be spelled explicitly
    /// (`NlExpr.sum(c * x[i] for ...)`).
    #[classattr]
    #[allow(non_snake_case)]
    fn __array_ufunc__() -> Option<()> {
        None
    }

    fn __abs__(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Abs, self.operand())
    }

    fn sqrt(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Sqrt, self.operand())
    }

    fn exp(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Exp, self.operand())
    }

    /// Natural logarithm.
    fn log(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Log, self.operand())
    }

    fn log10(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Log10, self.operand())
    }

    fn sin(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Sin, self.operand())
    }

    fn cos(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Cos, self.operand())
    }

    fn tan(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Tan, self.operand())
    }

    fn asin(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Asin, self.operand())
    }

    fn acos(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Acos, self.operand())
    }

    fn atan(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Atan, self.operand())
    }

    fn sinh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Sinh, self.operand())
    }

    fn cosh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Cosh, self.operand())
    }

    fn tanh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Tanh, self.operand())
    }

    fn asinh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Asinh, self.operand())
    }

    fn acosh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Acosh, self.operand())
    }

    fn atanh(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Atanh, self.operand())
    }

    /// Gauss error function. Reachable only from here — AMPL has no `erf`
    /// opcode, so no `.nl` round trip can carry it (issue #469).
    fn erf(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::Erf, self.operand())
    }

    /// Value of this expression alone at `x`, through the same AD tape a
    /// model uses. `x` must be long enough to cover every variable index
    /// the expression references.
    fn eval(&self, x: Vec<Number>) -> PyResult<Number> {
        let inner = &self.inner;
        // The tape build recurses once per level, so it runs under the
        // depth guard; only plain data crosses back (pounce#472).
        let value: Result<Number, String> = on_deep_stack(self.depth, move || {
            let tape = checked_tape(inner, x.len(), "eval")?;
            Ok(tape.eval(&x))
        });
        value.map_err(PyValueError::new_err)
    }

    /// Gradient of this expression alone at `x`, length `len(x)`.
    fn gradient<'py>(
        &self,
        py: Python<'py>,
        x: Vec<Number>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let inner = &self.inner;
        let grad: Result<Vec<Number>, String> = on_deep_stack(self.depth, move || {
            let tape = checked_tape(inner, x.len(), "gradient")?;
            let mut grad = vec![0.0; x.len()];
            tape.gradient_seed(&x, 1.0, &mut grad);
            Ok(grad)
        });
        Ok(grad.map_err(PyValueError::new_err)?.into_pyarray_bound(py))
    }

    /// Sorted variable indices this expression references.
    fn variables(&self) -> Vec<usize> {
        let inner = &self.inner;
        on_deep_stack(self.depth, move || Tape::build(inner).variables())
    }

    fn __repr__(&self) -> String {
        let mut count = 0usize;
        const CAP: usize = 64;
        node_count_capped(&self.inner, CAP, &mut count);
        if count > CAP {
            format!("NlExpr(<{CAP}+ nodes>)")
        } else {
            format!("NlExpr({})", render_expression(&self.inner, &[]))
        }
    }
}

impl PyNlExpr {}

/// Rebuild `root` with a `Cse` node left only where the same
/// subexpression really is reached more than once.
///
/// Operators reference their operands rather than copying them, so a
/// freshly built expression is a chain of `Cse` nodes even when nothing is
/// shared. The tape does not care — it walks through a `Cse` and memoizes
/// the body — but the problem assembler does: `split_top_sums` treats a
/// `Cse` as an opaque leaf, so an un-inlined `a + b + c` would tape as one
/// summand instead of three, and the whole model would share one tape and
/// one color set in `eval_h` instead of a small one per term. Inlining the
/// references that are not really shared gives `from_expressions` exactly
/// the tree it would have gotten from a copy-on-every-operator build, at
/// one pass over the DAG instead of one per operator.
///
/// Genuine sharing (`t` used twice) is *kept* as a `Cse`, which is better
/// than the copy it replaces: the body is taped once and its adjoint sums
/// the contributions from every reference — the same value and the same
/// derivatives, off a smaller tape.
fn materialize(root: &Expr) -> Expr {
    let mut refs: HashMap<*const Expr, usize> = HashMap::new();
    count_refs(root, &mut refs);
    let mut shared: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    rebuild(root, &refs, &mut shared)
}

/// Count how many references reach each `Cse` body. Descends into a body
/// only the first time it is seen, so a DAG costs one visit per distinct
/// node rather than one per path to it.
fn count_refs(e: &Expr, refs: &mut HashMap<*const Expr, usize>) {
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            count_refs(a, refs);
            count_refs(b, refs);
        }
        Expr::Unary(_, a) | Expr::Not(a) => count_refs(a, refs),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                count_refs(a, refs);
            }
        }
        Expr::Cond { cond, then_, else_ } => {
            count_refs(cond, refs);
            count_refs(then_, refs);
            count_refs(else_, refs);
        }
        Expr::Funcall { args, .. } => {
            for a in args {
                if let FuncallArg::Real(inner) = a {
                    count_refs(inner, refs);
                }
            }
        }
        Expr::Cse(body) => {
            let seen = refs.entry(Arc::as_ptr(body)).or_insert(0);
            *seen += 1;
            if *seen == 1 {
                count_refs(body, refs);
            }
        }
    }
}

fn rebuild(
    e: &Expr,
    refs: &HashMap<*const Expr, usize>,
    shared: &mut HashMap<*const Expr, Arc<Expr>>,
) -> Expr {
    match e {
        Expr::Const(_) | Expr::Var(_) => e.clone(),
        Expr::Binary(op, a, b) => Expr::Binary(
            *op,
            Box::new(rebuild(a, refs, shared)),
            Box::new(rebuild(b, refs, shared)),
        ),
        Expr::Unary(op, a) => Expr::Unary(*op, Box::new(rebuild(a, refs, shared))),
        Expr::Compare(op, a, b) => Expr::Compare(
            *op,
            Box::new(rebuild(a, refs, shared)),
            Box::new(rebuild(b, refs, shared)),
        ),
        Expr::And(a, b) => Expr::And(
            Box::new(rebuild(a, refs, shared)),
            Box::new(rebuild(b, refs, shared)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(rebuild(a, refs, shared)),
            Box::new(rebuild(b, refs, shared)),
        ),
        Expr::Not(a) => Expr::Not(Box::new(rebuild(a, refs, shared))),
        Expr::Sum(args) => Expr::Sum(args.iter().map(|a| rebuild(a, refs, shared)).collect()),
        Expr::MinList(args) => {
            Expr::MinList(args.iter().map(|a| rebuild(a, refs, shared)).collect())
        }
        Expr::MaxList(args) => {
            Expr::MaxList(args.iter().map(|a| rebuild(a, refs, shared)).collect())
        }
        Expr::Cond { cond, then_, else_ } => Expr::Cond {
            cond: Box::new(rebuild(cond, refs, shared)),
            then_: Box::new(rebuild(then_, refs, shared)),
            else_: Box::new(rebuild(else_, refs, shared)),
        },
        Expr::Funcall { id, args } => Expr::Funcall {
            id: *id,
            args: args
                .iter()
                .map(|a| match a {
                    FuncallArg::Real(inner) => FuncallArg::Real(rebuild(inner, refs, shared)),
                    FuncallArg::Str(s) => FuncallArg::Str(s.clone()),
                })
                .collect(),
        },
        Expr::Cse(body) => {
            let key = Arc::as_ptr(body);
            if refs.get(&key).copied().unwrap_or(0) < 2 {
                // One reference: this is an operand hand-off, not sharing.
                return rebuild(body, refs, shared);
            }
            if let Some(done) = shared.get(&key) {
                return Expr::Cse(Arc::clone(done));
            }
            let built = Arc::new(rebuild(body, refs, shared));
            shared.insert(key, Arc::clone(&built));
            Expr::Cse(built)
        }
    }
}

/// Tape for `inner`, rejecting a variable index `x` cannot supply.
/// Without the check the tape's forward sweep would index `x` out of
/// bounds — a panic across the pyo3 boundary rather than a catchable
/// Python error.
///
/// A free function returning `Result<_, String>` rather than a method
/// returning `PyResult`: it is called from inside [`on_deep_stack`], and
/// only `Send` plain data may cross back out of that worker.
fn checked_tape(inner: &Expr, x_len: usize, what: &str) -> Result<Tape, String> {
    let tape = Tape::build(inner);
    match tape.variables().iter().max() {
        Some(&max) if max >= x_len => Err(format!(
            "{what}: expression references variable {max} but x has length {x_len}"
        )),
        _ => Ok(tape),
    }
}

/// Decode an optional float vector, filling `default` when it is `None`.
fn opt_vec(
    v: Option<&Bound<'_, PyAny>>,
    len: usize,
    default: Number,
    what: &str,
) -> PyResult<Vec<Number>> {
    match v {
        None => Ok(vec![default; len]),
        Some(b) => {
            let mut out = Vec::with_capacity(len);
            for item in b.iter()? {
                out.push(item?.extract::<Number>()?);
            }
            if out.len() != len {
                return Err(PyValueError::new_err(format!(
                    "build_nl_problem: {what} has length {}, expected {len}",
                    out.len()
                )));
            }
            Ok(out)
        }
    }
}

/// Build an evaluable [`crate::PyNlProblem`] from expressions, with no `.nl`
/// file involved (issue #469).
///
/// The returned object is the same `NlProblem` class `read_nl` hands back
/// and supports the same surface — `objective`, `gradient`, `constraints`,
/// `jacobian` / `jacobian_structure`, `hessian` / `hessian_structure`,
/// `hessian_vector_product`, and `variant`.
///
/// Bound vectors default to unbounded (`±1e19`, the `.nl` sentinel) and
/// `x0` to zeros. `var_names` / `con_names` are optional; when given they
/// must match `n` / `len(constraints)`.
#[pyfunction]
#[pyo3(signature = (
    n,
    objective,
    constraints=None,
    x_l=None,
    x_u=None,
    x0=None,
    g_l=None,
    g_u=None,
    minimize=true,
    obj_constant=0.0,
    var_names=None,
    con_names=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn build_nl_problem(
    n: usize,
    objective: &Bound<'_, PyAny>,
    constraints: Option<&Bound<'_, PyAny>>,
    x_l: Option<&Bound<'_, PyAny>>,
    x_u: Option<&Bound<'_, PyAny>>,
    x0: Option<&Bound<'_, PyAny>>,
    g_l: Option<&Bound<'_, PyAny>>,
    g_u: Option<&Bound<'_, PyAny>>,
    minimize: bool,
    obj_constant: Number,
    var_names: Option<Vec<String>>,
    con_names: Option<Vec<String>>,
) -> PyResult<PyNlProblem> {
    let objective = coerce(objective, "build_nl_problem: objective")?;
    let (constraints, con_depth) = match constraints {
        None => (Vec::new(), 0),
        Some(c) => coerce_all(c, "build_nl_problem: constraints")?,
    };
    let m = constraints.len();
    // Assembly walks every expression recursively — materialization, the
    // variable-index validation, the top-level sum split, and one tape
    // build per summand — so it runs under the depth guard the query
    // methods use (pounce#472).
    let depth = objective.depth.max(con_depth);
    let objective = objective.expr;

    let parts = NlProblemParts {
        minimize,
        objective,
        obj_constant,
        constraints,
        x_l: opt_vec(x_l, n, -INF, "x_l")?,
        x_u: opt_vec(x_u, n, INF, "x_u")?,
        x0: opt_vec(x0, n, 0.0, "x0")?,
        g_l: opt_vec(g_l, m, -INF, "g_l")?,
        g_u: opt_vec(g_u, m, INF, "g_u")?,
        var_names: var_names.unwrap_or_default(),
        con_names: con_names.unwrap_or_default(),
    };

    let tnlp = on_deep_stack(depth, move || {
        // Each row is materialized on its own, so an expression used in
        // two of them is a plain tree in both rather than a `Cse` shared
        // across them — a row whose *root* were a `Cse` would tape as a
        // single summand.
        let mut parts = parts;
        parts.objective = materialize(&parts.objective);
        for c in &mut parts.constraints {
            *c = materialize(c);
        }
        let prob = NlProblem::from_expressions(parts)?;
        NlTnlp::try_new(prob)
    })
    .map_err(|e| PyValueError::new_err(format!("build_nl_problem: {e}")))?;
    // The problem keeps the expression trees it taped, so it inherits
    // their depth for its own copies and teardown.
    PyNlProblem::from_tnlp(tnlp, "build_nl_problem", depth)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the sharing/materialization pair (issue #472). They
    //! drive the real constructors — `binary`, `unary`, `operand` — which
    //! never touch the interpreter on the success path, so no Python is
    //! needed to build an expression exactly as a modeler would.

    use super::*;

    fn num(v: Number) -> Operand {
        Operand {
            expr: Expr::Const(v),
            depth: 1,
        }
    }

    fn node(inner: Expr, child_depth: u32) -> PyNlExpr {
        PyNlExpr::nested(inner, child_depth).unwrap()
    }

    /// `x0 + 1 + 1 + ...`, the accumulation loop, `adds` operators long.
    fn chain(adds: usize) -> PyNlExpr {
        let mut e = PyNlExpr::leaf(Expr::Var(0));
        for _ in 0..adds {
            e = binary(BinOp::Add, e.operand(), num(1.0)).unwrap();
        }
        e
    }

    /// The same expression a copy-on-every-operator build would produce.
    fn plain_chain(adds: usize) -> Expr {
        let mut e = Expr::Var(0);
        for _ in 0..adds {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(Expr::Const(1.0)));
        }
        e
    }

    fn count_cse(e: &Expr) -> usize {
        let mut refs = HashMap::new();
        count_refs(e, &mut refs);
        refs.values().sum()
    }

    #[test]
    fn building_a_chain_shares_rather_than_copies() {
        // One reference per operator, less the first, whose operand is the
        // bare `Var` leaf: the payload is never copied.
        assert_eq!(count_cse(&chain(4).inner), 3);
    }

    #[test]
    fn materialize_inlines_operand_references() {
        // Those references are hand-offs, not sharing, so the assembled
        // model gets the plain left-leaning chain — which is what lets
        // `split_top_sums` see the `+`s and give each term its own tape.
        let flat = materialize(&chain(4).inner);
        assert_eq!(count_cse(&flat), 0);
        assert_eq!(format!("{flat:?}"), format!("{:?}", plain_chain(4)));
    }

    #[test]
    fn a_leaf_operand_is_never_wrapped() {
        let x = PyNlExpr::leaf(Expr::Var(0));
        let y = PyNlExpr::leaf(Expr::Var(1));
        let e = binary(BinOp::Mul, x.operand(), y.operand()).unwrap();
        assert_eq!(count_cse(&e.inner), 0);
        assert!(matches!(&*e.inner, Expr::Binary(BinOp::Mul, _, _)));
    }

    #[test]
    fn materialize_keeps_a_twice_used_subexpression_shared() {
        let t = unary(UnaryOp::Sin, PyNlExpr::leaf(Expr::Var(0)).operand()).unwrap();
        let e = binary(BinOp::Add, t.operand(), t.operand()).unwrap();
        match materialize(&e.inner) {
            Expr::Binary(BinOp::Add, a, b) => match (*a, *b) {
                // One body, referenced twice — not two copies of it.
                (Expr::Cse(x), Expr::Cse(y)) => {
                    assert!(Arc::ptr_eq(&x, &y));
                    assert!(matches!(&*x, Expr::Unary(UnaryOp::Sin, _)));
                }
                other => panic!("expected two references to one body, got {other:?}"),
            },
            other => panic!("expected an Add, got {other:?}"),
        }
    }

    #[test]
    fn sharing_survives_the_n_ary_and_control_flow_nodes() {
        // `count_refs` / `rebuild` have an arm per variant; this walks the
        // ones the binary tests do not reach.
        let t = unary(UnaryOp::Exp, PyNlExpr::leaf(Expr::Var(0)).operand()).unwrap();
        let cond = node(
            Expr::Cond {
                cond: Box::new(t.operand().expr),
                then_: Box::new(t.operand().expr),
                else_: Box::new(num(0.0).expr),
            },
            t.depth,
        );
        let min = node(
            Expr::MinList(vec![t.operand().expr, cond.operand().expr]),
            cond.depth,
        );
        let e = node(
            Expr::Sum(vec![min.operand().expr, t.operand().expr]),
            min.depth,
        );

        // `t` is reached four times, `min` and `cond` once each.
        let flat = materialize(&e.inner);
        assert_eq!(count_cse(&flat), 4, "{flat:?}");
        let mut bodies = Vec::new();
        collect_cse_ptrs(&flat, &mut bodies);
        bodies.sort_unstable();
        bodies.dedup();
        assert_eq!(bodies.len(), 1, "one shared body, four references");
    }

    fn collect_cse_ptrs(e: &Expr, out: &mut Vec<*const Expr>) {
        match e {
            Expr::Cse(body) => {
                out.push(Arc::as_ptr(body));
                collect_cse_ptrs(body, out);
            }
            Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
                collect_cse_ptrs(a, out);
                collect_cse_ptrs(b, out);
            }
            Expr::Unary(_, a) | Expr::Not(a) => collect_cse_ptrs(a, out),
            Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
                for a in args {
                    collect_cse_ptrs(a, out);
                }
            }
            Expr::Cond { cond, then_, else_ } => {
                collect_cse_ptrs(cond, out);
                collect_cse_ptrs(then_, out);
                collect_cse_ptrs(else_, out);
            }
            Expr::Const(_) | Expr::Var(_) | Expr::Funcall { .. } => {}
        }
    }

    #[test]
    fn a_shared_body_is_taped_and_differentiated_like_the_copy_it_replaces() {
        // sin(x0) used twice, against the same thing written out twice.
        let t = unary(UnaryOp::Sin, PyNlExpr::leaf(Expr::Var(0)).operand()).unwrap();
        let shared = materialize(&binary(BinOp::Mul, t.operand(), t.operand()).unwrap().inner);
        let copied = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)))),
            Box::new(Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)))),
        );
        let x = [0.7];
        let (ts, tc) = (Tape::build(&shared), Tape::build(&copied));
        assert!((ts.eval(&x) - tc.eval(&x)).abs() < 1e-15);
        let (mut gs, mut gc) = (vec![0.0], vec![0.0]);
        ts.gradient_seed(&x, 1.0, &mut gs);
        tc.gradient_seed(&x, 1.0, &mut gc);
        assert!((gs[0] - gc[0]).abs() < 1e-15, "{gs:?} vs {gc:?}");
        // The point of keeping it shared: one `Sin` on the tape, not two.
        assert!(
            ts.ops.len() < tc.ops.len(),
            "{} vs {}",
            ts.ops.len(),
            tc.ops.len()
        );
    }

    #[test]
    fn the_depth_limit_still_bites() {
        let mut e = PyNlExpr::leaf(Expr::Var(0));
        for _ in 0..MAX_DEPTH - 1 {
            e = binary(BinOp::Add, e.operand(), num(1.0)).unwrap();
        }
        assert_eq!(e.depth, MAX_DEPTH);
        assert!(binary(BinOp::Add, e.operand(), num(1.0)).is_err());
    }
}
