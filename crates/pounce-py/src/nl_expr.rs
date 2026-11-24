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
use pyo3::types::{PyBool, PyString};

use std::collections::HashMap;
use std::sync::Arc;

use pounce_common::types::Number;
use pounce_nl::nl_reader::{
    BinOp, CmpOp, Expr, FuncallArg, NlBody, NlProblem, NlProblemParts, NlTnlp, UnaryOp,
    render_expression,
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
/// * every such walk runs on a worker thread with a [`WORKER_STACKS`]-sized (1 GiB, falling back to 256 / 64 MiB)
///   stack (see [`on_deep_stack`]), so what is survivable stops depending
///   on the calling thread — 8 MB on a macOS/Linux main thread, 1 MB on
///   Windows, less on a `threading.Thread`;
/// * this limit then keeps the depth well inside what that stack holds.
///
/// 10 000 is the arithmetic: the deepest frames measured are ~300 bytes in
/// a release build and up to ~26 KB in an unoptimised build, so the 1 GiB
/// reservation covers ~3 million levels released and ~40 000 unoptimised — a 3x margin at this limit even
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
#[cfg(not(target_os = "emscripten"))]
const WORKER_STACKS: [usize; 3] = [1 << 30, 256 << 20, 64 << 20];

/// Run `f` on a worker thread with a stack sized for recursion over a
/// deep expression, whatever the caller's stack is.
///
/// For work whose depth is not known ahead of time — parsing a `.nl` file
/// is exactly that, since the depth is what the file says — rather than
/// [`on_deep_stack`], which skips the worker for shallow input.
///
/// Pyodide cannot start a thread, so the emscripten build runs `f` on the
/// calling thread instead, with the `.nl` reader told how much of that
/// thread's stack is free. The reader then refuses a file nested deeper than
/// that stack holds, as it does on a worker whose reservation came up short.
pub(crate) fn on_worker_stack<T, F>(f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    #[cfg(target_os = "emscripten")]
    {
        on_calling_stack(f)
    }
    #[cfg(not(target_os = "emscripten"))]
    {
        on_spawned_stack(f)
    }
}

/// The emscripten build's stand-in for the worker thread: run `f` here.
/// `.nl` parsing gets a stack budget of what is free now, less a margin
/// for the frames between this call and the reader's recursion.
#[cfg(target_os = "emscripten")]
fn on_calling_stack<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    /// Restores the reader's default budget however `f` returns, panics
    /// included, so a later parse on this thread does not inherit it.
    struct ResetBudget;
    impl Drop for ResetBudget {
        fn drop(&mut self) {
            pounce_nl::nl_reader::set_parse_stack_budget(None);
        }
    }
    let free = emscripten_stack_free();
    pounce_nl::nl_reader::set_parse_stack_budget(Some(free.saturating_sub(CALL_MARGIN)));
    let _reset = ResetBudget;
    f()
}

/// Stack left free on the calling thread, from emscripten's own record of
/// the stack bounds. Pyodide gives its one thread 5 MiB.
#[cfg(target_os = "emscripten")]
fn emscripten_stack_free() -> usize {
    unsafe extern "C" {
        fn emscripten_stack_get_free() -> usize;
    }
    // SAFETY: a pure read of the current stack pointer and the stack's
    // limit, which emscripten's runtime always provides.
    unsafe { emscripten_stack_get_free() }
}

/// Stack kept back from a budget on the calling thread for the frames
/// between the budget's caller and the recursive walk.
#[cfg(target_os = "emscripten")]
const CALL_MARGIN: usize = 256 << 10;

/// Stack one level of a recursive walk is assumed to need on the calling
/// thread, the reader's own worst case for an optimised build.
#[cfg(target_os = "emscripten")]
const LEVEL_BYTES: usize = 4 << 10;

#[cfg(not(target_os = "emscripten"))]
fn on_spawned_stack<T, F>(f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    // Reserve the largest stack the OS grants (1 GiB, then 256 MiB, then
    // 64 MiB), and tell the `.nl` reader what it got so its depth guard fires
    // before the stack does at any build profile (gh#986). `f` runs exactly
    // once, on whichever spawn succeeds.
    // The body sits in a slot outside the scope: a failed `spawn_scoped`
    // drops its closure unrun, leaving the body here for the next attempt.
    let slot = std::sync::Mutex::new(Some(f));
    std::thread::scope(|scope| {
        for stack in WORKER_STACKS {
            let slot_ref = &slot;
            let spawned =
                std::thread::Builder::new()
                    .stack_size(stack)
                    .spawn_scoped(scope, move || {
                        pounce_nl::nl_reader::set_parse_stack_budget(Some(stack));
                        let taken = slot_ref.lock().unwrap_or_else(|e| e.into_inner()).take();
                        let Some(job) = taken else {
                            panic!("pounce: worker body already consumed")
                        };
                        job()
                    });
            if let Ok(handle) = spawned {
                return match handle.join() {
                    Ok(v) => v,
                    // Carry a panic across the join so pyo3 still turns it
                    // into a Python exception rather than losing it here.
                    Err(panic) => std::panic::resume_unwind(panic),
                };
            }
        }
        // Nothing safe is left to do: running the walk here is the
        // segfault this function exists to prevent. Panic instead, so
        // pyo3 raises it as a Python exception.
        panic!("pounce: cannot spawn the expression worker thread (no stack could be reserved)")
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
    // With no worker to move to, a walk deeper than the free stack holds
    // would overflow it and kill the interpreter, so it is refused here
    // and pyo3 raises the panic as a Python exception.
    #[cfg(target_os = "emscripten")]
    if (depth as usize).saturating_mul(LEVEL_BYTES)
        > emscripten_stack_free().saturating_sub(CALL_MARGIN)
    {
        panic!(
            "pounce: this expression is nested {depth} levels deep, more than the \
             browser's {} KiB of free stack can walk",
            emscripten_stack_free() >> 10
        );
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

    /// `centropy(a, b) = a·ln(a/b)`, with the `a = 0` limit `0`.
    ///
    /// A fused op, not sugar for `a * log(a / b)`: the composed form's second
    /// derivatives route through `ln''(a) = -1/a²` and `a/b²`, both of which
    /// leave `f64` range long before `a·ln(a/b)` and its own derivatives do.
    /// Prefer this over hand-expanding it in a model.
    #[staticmethod]
    fn centropy(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::CEntropy,
            coerce(a, "centropy: a")?,
            coerce(b, "centropy: b")?,
        )
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

    /// `x·ln(x)`, with the `x = 0` limit `0`. GAMS `entropy` is `-xlogx`.
    ///
    /// Fused rather than written as `x * log(x)` because the composed second
    /// derivative goes through `ln''(x) = -1/x²`, which overflows below
    /// `x ≈ 1e-154` even though the true `(x ln x)'' = 1/x` is representable
    /// all the way down to `1e-308`.
    fn xlogx(&self) -> PyResult<PyNlExpr> {
        unary(UnaryOp::XLogX, self.operand())
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
///
/// `build_nl_problem` materializes the whole model at once
/// ([`materialize_model`]); this one-row form is the per-row tree that
/// build is held to, so since gh#1235 only the tests call it.
#[cfg(test)]
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

/// The rows of a model as `build_nl_problem` stores them: one compact copy
/// of the caller's `NlExpr` DAG, with every node copied at most once
/// (gh#1235).
///
/// Every row (the objective is one more) is copied as [`materialize`] would
/// inline it on its own, except in three ways, none of which changes what a
/// row evaluates:
///
/// * **A subexpression reached from more than one row is one shared body**,
///   not a copy per row (gh#1232). Reference counts are taken over the
///   summand leaves of the whole model, so a body two leaves reach is one
///   `Arc` and everything below it that only it reaches is inlined into it.
/// * **A summand leaf several summands reach is copied once**, and each of
///   them holds it behind a `Cse` of its own. Copying it per reference made
///   a rate shared by several rows cost one copy of its top node per row.
///   That `Cse` is a hand-off, not sharing: the problem is marked
///   [`NlProblem::frontend_dag`], so `NlTnlp` looks through it when it
///   splits the row, and the leaf it tapes is the plain node behind it, as
///   the per-reference copy was. That matters to the hybrid tape, which
///   promotes a `Cse` that several summands reach into its shared prelude:
///   promoting the leaf would make its Hessian sum adjoints in another order
///   and move it in the last bits.
/// * **A row the quadratic recognizer admits keeps its own per-row tree**,
///   since the recognizer refuses some `Cse` placements a plain tree passes
///   (a `Cse` on the sum spine of a factored form, or inside the scaling
///   around a square). It is asked only of a row that could pass:
///   [`polynomial_ops_only`].
///
/// The rest is inlined exactly as [`materialize`] would, because what a row
/// evaluates depends on where a `Cse` sits in three places. The summand split
/// walks the row's sum spine, which is therefore rebuilt per row without
/// hand-offs ([`spine`]). `Tape::build` emits a `Cse` body once, as the ops
/// an inlined copy would get, so every flat tape is op for op what it was.
/// And the quadratic recognizer, covered above.
///
/// What this does **not** do is keep the caller's own nodes. An `NlExpr`
/// node holds each operand as a boxed `Cse`, which costs about three times a
/// tree node. A problem holding the DAG would pin all of it after the caller
/// let go: on rows that share nothing that measured 45% more live heap than
/// this copy once the caller dropped its expressions. Here the problem holds
/// only its copy, so a caller that frees its expressions frees them.
///
/// Values, derivatives and both sparsity patterns are bit-identical to the
/// copy-per-row build this replaced, on every evaluator, the hybrid ones
/// included (`stored_rows_evaluate_exactly_as_the_copied_rows_did`).
fn materialize_model(roots: &[&Expr]) -> Vec<Expr> {
    // Pass 1: which rows keep their per-row tree; reference counts over the
    // summand leaves of every other row; how often each leaf is reached.
    let mut global: HashMap<*const Expr, usize> = HashMap::new();
    let mut leaf_refs: HashMap<*const Expr, usize> = HashMap::new();
    let mut keep_tree: Vec<bool> = Vec::with_capacity(roots.len());
    for root in roots {
        let mut row = HashMap::new();
        count_refs(root, &mut row);
        let keep = polynomial_ops_only(root) && quadratic_row(root, &row);
        if !keep {
            spine(root, &row, &mut |leaf: &Expr, _| {
                count_refs(leaf, &mut global);
                *leaf_refs.entry(leaf as *const Expr).or_insert(0) += 1;
                Expr::Const(0.0)
            });
        }
        keep_tree.push(keep);
    }
    // Pass 2: copy each row's spine, each body shared across the model once,
    // and each leaf reached more than once once.
    let mut shared: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    let mut leaves: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    roots
        .iter()
        .zip(keep_tree)
        .map(|(root, keep)| {
            let mut row = HashMap::new();
            count_refs(root, &mut row);
            if keep {
                let mut private = HashMap::new();
                return rebuild(root, &row, &mut private);
            }
            spine(root, &row, &mut |leaf: &Expr, reached: &Expr| {
                let key = leaf as *const Expr;
                // Only a leaf the spine reached through a hand-off can be
                // wrapped in one: `frontend_dag` looks through a `Cse` only,
                // and a leaf the row references twice is a `Cse` already.
                let handoff = matches!(reached, Expr::Cse(_)) && !matches!(leaf, Expr::Cse(_));
                if !handoff || leaf_refs.get(&key).copied().unwrap_or(0) < 2 {
                    return rebuild(leaf, &global, &mut shared);
                }
                if let Some(done) = leaves.get(&key) {
                    return Expr::Cse(Arc::clone(done));
                }
                let copy = Arc::new(rebuild(leaf, &global, &mut shared));
                leaves.insert(key, Arc::clone(&copy));
                Expr::Cse(copy)
            })
        })
        .collect()
}

/// Whether `e` reaches only the operations a quadratic form can be read out
/// of: constants, variables, sums, differences, negation, products,
/// quotients and powers. Both forms the recognizer admits are built from
/// these alone — the expanded form's monomials and the factored form's
/// spine, constant weights and squared affine bases — so a row reaching any
/// other operation (`exp`, `sin`, a comparison, …) is admitted by neither,
/// and [`materialize_model`] need not build its tree to ask. Each `Cse` body
/// is visited once.
fn polynomial_ops_only(e: &Expr) -> bool {
    let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
    let mut work: Vec<&Expr> = vec![e];
    while let Some(e) = work.pop() {
        match e {
            Expr::Const(_) | Expr::Var(_) => {}
            Expr::Cse(body) => {
                if seen.insert(Arc::as_ptr(body)) {
                    work.push(body);
                }
            }
            Expr::Sum(args) => work.extend(args.iter()),
            Expr::Unary(UnaryOp::Neg, a) => work.push(a),
            Expr::Binary(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow, a, b) => {
                work.push(a);
                work.push(b);
            }
            _ => return false,
        }
    }
    true
}

/// The rows as `build_nl_problem` stored them before gh#1235, kept as the
/// reference [`materialize_model`] is tested against. Its original doc:
///
/// [`materialize`] for a whole model at once: every row (the objective is
/// one more) inlined exactly as [`materialize`] would inline it on its own,
/// except that a subexpression reached from more than one row stays **one**
/// shared `Cse` instead of becoming a private copy in each (gh#1232).
///
/// Materializing row by row made a body referenced once per row look
/// unshared in every row, so the problem held one expanded copy per row: a
/// model that is 50 shared rate expressions referenced by 20 000 rows built
/// as large as one with 20 000 private copies. Here the sharing is counted
/// across the model, so memory follows the `NlExpr` DAG.
///
/// What a row *evaluates* is unchanged, bit for bit, because three things
/// downstream are sensitive to where a `Cse` sits and each is held to the
/// per-row tree:
///
/// * **The summand split.** `split_top_sums` treats a `Cse` as an opaque
///   leaf, and the split decides how a row's terms are added up. So the
///   region that split walks through — sums, differences, negations, and
///   scaling or division by a constant — is inlined per row exactly as
///   [`materialize`] would, see [`spine`]. Sharing starts below it, at the
///   summand leaves.
/// * **The flat tape of each summand.** `Tape::build` emits a `Cse` body
///   once, as the same ops an inlined copy would get. A body referenced once
///   in a row is therefore the same tape shared or not, and one referenced
///   twice in a row was already a `Cse` in the per-row tree — so every flat
///   tape comes out op for op what it was.
/// * **The quadratic recognizer**, which refuses some `Cse` placements an
///   inlined tree passes (a `Cse` on the sum spine of a factored form, or
///   inside the scaling around a square). A row the recognizer admits as
///   per-row tree keeps that tree verbatim ([`quadratic_row`]); every other
///   row may share, since a `Cse` only ever narrows what it admits.
///
/// What does change is that a body shared across summands can now reach the
/// constraint block's shared-CSE prelude, which evaluates it once per sweep
/// instead of once per summand. Values and Jacobians come out of it as they
/// did; its Hessian sums the λ-weighted contributions before propagating
/// them through the body, so `eval_h` can differ in the last bits.
///
/// **Sharing is kept for every model** (gh#1235). It used to be kept only
/// where an estimate said the prelude would *replace* the flat tapes,
/// because below the prelude's op-ratio gates `NlTnlp` built it *next to*
/// them, and its per-summand bookkeeping outweighed what sharing saved: a
/// model whose shared bodies are small — discopt hands over one `NlExpr`
/// per distinct node, so `exp(p_j)` is one object in thousands of rows —
/// came out 19% larger than unshared (gh#1232 review). `NlTnlp` now keeps
/// the prelude only when a derivative evaluator reads it, so below the gates
/// sharing costs nothing and still shrinks the stored rows: gh#1235's
/// kinetics rows build at 5.2 KiB a row shared against 6.1 unshared — and
/// 11.4 when the estimate declined to share and an unused prelude was built
/// anyway. Between the gates the prelude is kept for `eval_h`, next to the
/// flat tapes, as the measured Hessian gate says it should be.
#[cfg(test)]
fn materialize_model_copied(roots: &[&Expr]) -> Vec<Expr> {
    // Pass 1: which rows keep their per-row tree, and — over the summand
    // leaves of every other row — how often each `Cse` body is referenced
    // model-wide.
    let mut global: HashMap<*const Expr, usize> = HashMap::new();
    let mut keep_tree: Vec<bool> = Vec::with_capacity(roots.len());
    for root in roots {
        let mut row = HashMap::new();
        count_refs(root, &mut row);
        let keep = quadratic_row(root, &row);
        if !keep {
            spine(root, &row, &mut |leaf: &Expr, _| {
                count_refs(leaf, &mut global);
                Expr::Const(0.0)
            });
        }
        keep_tree.push(keep);
    }
    // Pass 2: rebuild the spines; below them, reference the caller's own
    // nodes (gh#1235). A summand leaf reached through a `Cse` hand-off is
    // copied one node deep — `clone` copies the node and bumps the `Arc` of
    // each operand — so everything below it is the frontend's own DAG rather
    // than a private copy. The tape builders treat a `Cse` as transparent,
    // so each summand tapes op for op as the deep copy did.
    //
    // The leaf node itself must stay a fresh node, not the caller's `Cse`:
    // a leaf several rows reach (every row's `exp(…)` in gh#1232's model)
    // would otherwise become a body the hybrid tape promotes into its
    // prelude, which the deep copy never was, and the hybrid Hessian, which
    // sums adjoints per promoted body, would move in the last bits. Below a
    // node unique to its summand, the first body a hybrid tape can promote
    // on any path is one referenced from two places, which is exactly where
    // the deep copy kept a `Cse`. A leaf that arrived bare is rebuilt as
    // before, one `Arc` per body shared across the model.
    let mut shared: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    roots
        .iter()
        .zip(keep_tree)
        .map(|(root, keep)| {
            let mut row = HashMap::new();
            count_refs(root, &mut row);
            if keep {
                let mut private = HashMap::new();
                rebuild(root, &row, &mut private)
            } else {
                spine(root, &row, &mut |leaf: &Expr, orig: &Expr| match orig {
                    Expr::Cse(_) => leaf.clone(),
                    _ => rebuild(leaf, &global, &mut shared),
                })
            }
        })
        .collect()
}

/// Whether the per-row tree of `root` is one the quadratic recognizer
/// admits (`push_body_form`'s two tests), in which case it is kept as is so
/// that the recognizer sees exactly what it saw before gh#1232. Built and
/// dropped here, one row at a time.
fn quadratic_row(root: &Expr, row: &HashMap<*const Expr, usize>) -> bool {
    let mut private = HashMap::new();
    let body = NlBody::Tree(rebuild(root, row, &mut private));
    body.admitted_quad_form().is_some() || body.admitted_factored_form().is_some()
}

/// Look through every `Cse` [`materialize`] would inline in this row (one
/// referenced fewer than twice in it), to the node that stands in its place
/// in the per-row tree.
fn see_through<'a>(mut e: &'a Expr, row: &HashMap<*const Expr, usize>) -> &'a Expr {
    while let Expr::Cse(body) = e {
        if row.get(&Arc::as_ptr(body)).copied().unwrap_or(0) >= 2 {
            break;
        }
        e = body;
    }
    e
}

/// Rebuild the part of a row that `split_top_sums` descends through, as
/// [`materialize`] would inline it, handing each summand leaf to `leaf` —
/// as the node that stands in its place in the per-row tree, and as the
/// expression that reached it, which is that node behind the `Cse`
/// hand-offs [`see_through`] looked past.
///
/// Mirrors `split_top_sums`'s `go` node for node — `Sum`, `+`, `-`, unary
/// `-`, a product with a constant on either side, a quotient by a nonzero
/// constant — on the tree with this row's single-reference `Cse`s seen
/// through, so the leaves it splits into, and the constant factors it
/// folds into them, are the ones the per-row tree gave. Anything else is a
/// leaf.
fn spine(
    e: &Expr,
    row: &HashMap<*const Expr, usize>,
    leaf: &mut dyn FnMut(&Expr, &Expr) -> Expr,
) -> Expr {
    let orig = e;
    let e = see_through(e, row);
    let konst = |x: &Expr| match see_through(x, row) {
        Expr::Const(c) => Some(*c),
        _ => None,
    };
    match e {
        Expr::Sum(args) => Expr::Sum(args.iter().map(|a| spine(a, row, leaf)).collect()),
        Expr::Binary(op @ (BinOp::Add | BinOp::Sub), a, b) => {
            let a = spine(a, row, leaf);
            let b = spine(b, row, leaf);
            Expr::Binary(*op, Box::new(a), Box::new(b))
        }
        Expr::Unary(UnaryOp::Neg, a) => Expr::Unary(UnaryOp::Neg, Box::new(spine(a, row, leaf))),
        Expr::Binary(BinOp::Mul, a, b) => match (konst(a), konst(b)) {
            (Some(c), _) => Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(c)),
                Box::new(spine(b, row, leaf)),
            ),
            (None, Some(c)) => Expr::Binary(
                BinOp::Mul,
                Box::new(spine(a, row, leaf)),
                Box::new(Expr::Const(c)),
            ),
            (None, None) => leaf(e, orig),
        },
        Expr::Binary(BinOp::Div, a, b) => match konst(b) {
            Some(c) if c != 0.0 => Expr::Binary(
                BinOp::Div,
                Box::new(spine(a, row, leaf)),
                Box::new(Expr::Const(c)),
            ),
            _ => leaf(e, orig),
        },
        _ => leaf(e, orig),
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

/// The `con_linear=` argument of [`build_nl_problem`]: each row's linear part
/// as a CSR triple `(indptr, indices, data)` over the `m` rows.
///
/// Validated here, while it is still a diagnosable user error, for the same
/// reasons `from_expressions` validates the trees: an index at or beyond `n`
/// is an out-of-bounds read in `eval_g` / `eval_jac_g`, and a non-finite
/// coefficient would be read as a number by every evaluator (gh#847 refuses
/// the same in a `.nl` file's bounds). A repeated column within a row is
/// allowed -- every consumer of `con_linear` accumulates, as it does for a
/// `.nl` file's `J` segment.
fn csr_linear(
    v: Option<&Bound<'_, PyAny>>,
    m: usize,
    n: usize,
) -> PyResult<Option<Vec<Vec<(usize, Number)>>>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let err = |msg: String| PyValueError::new_err(format!("build_nl_problem: con_linear {msg}"));
    let not_triple = || {
        PyTypeError::new_err(
            "build_nl_problem: con_linear must be a CSR triple (indptr, indices, data)",
        )
    };
    let (py_indptr, py_indices, py_data): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>) =
        v.extract().map_err(|_| not_triple())?;
    // A boolean extracts as an integer, so a mask passed where indices belong
    // would silently become columns 0 and 1. Refuse it, for a NumPy bool
    // array (dtype kind 'b') and for a sequence holding Python bools.
    for (what, obj) in [("indptr", &py_indptr), ("indices", &py_indices)] {
        let bool_dtype = obj
            .getattr("dtype")
            .and_then(|d| d.getattr("kind"))
            .and_then(|k| k.extract::<String>())
            .is_ok_and(|k| k == "b");
        let bool_item = !bool_dtype
            && obj.downcast::<pyo3::types::PySequence>().is_ok()
            && obj
                .iter()?
                .any(|x| x.is_ok_and(|x| x.is_instance_of::<PyBool>()));
        if bool_dtype || bool_item {
            return Err(PyTypeError::new_err(format!(
                "build_nl_problem: con_linear {what} must be integers, got booleans"
            )));
        }
    }
    let indptr: Vec<i64> = py_indptr.extract().map_err(|_| not_triple())?;
    let indices: Vec<i64> = py_indices.extract().map_err(|_| not_triple())?;
    let data: Vec<Number> = py_data.extract().map_err(|_| not_triple())?;
    if indptr.len() != m + 1 {
        return Err(err(format!(
            "indptr has length {}, expected len(constraints) + 1 = {}",
            indptr.len(),
            m + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(err(format!(
            "indices and data have lengths {} and {}",
            indices.len(),
            data.len()
        )));
    }
    // Monotonicity first, over the whole of `indptr`: checked row by row
    // inside the loop below, a later decrease let an earlier row run past
    // the end of `indices` before it was seen (`[0, 5, 2]` with two indices
    // read `indices[2]`), a panic rather than this error.
    if let Some(i) = indptr.windows(2).position(|w| w[1] < w[0]) {
        return Err(err(format!("indptr decreases at row {i}")));
    }
    if indptr[0] != 0 || indptr[m] != indices.len() as i64 {
        return Err(err(format!(
            "indptr must run from 0 to len(indices) = {}, got {}..{}",
            indices.len(),
            indptr[0],
            indptr[m]
        )));
    }
    let mut rows = Vec::with_capacity(m);
    for i in 0..m {
        // In range by the two checks above: 0 = indptr[0] <= a <= b <= indptr[m].
        let (a, b) = (indptr[i], indptr[i + 1]);
        let mut row = Vec::with_capacity((b - a) as usize);
        for k in a as usize..b as usize {
            let (j, c) = (indices[k], data[k]);
            if j < 0 || j as usize >= n {
                return Err(err(format!("row {i} references variable {j} but n = {n}")));
            }
            if !c.is_finite() {
                return Err(err(format!("row {i} has a non-finite coefficient {c}")));
            }
            row.push((j as usize, c));
        }
        rows.push(row);
    }
    Ok(Some(rows))
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
///
/// `con_linear`, optional, is each row's *linear part* as a CSR triple
/// `(indptr, indices, data)`: row `i` is `constraints[i](x) + sum(data[k] *
/// x[indices[k]] for k in indptr[i]:indptr[i+1])`. It is the `.nl` `J`
/// segment by another name, and gets what a parsed model's linear part gets:
/// no expression tree, no AD tape, a constant Jacobian entry. Pass `0.0` as
/// the expression of a row that is linear outright. A frontend that already
/// knows which terms are linear saves the tree, the tape and the `NlExpr`
/// nodes it would otherwise build for them (discopt#2293 measured a
/// three-term linear row at ~590 B in the problem plus ~490 B of transient
/// `NlExpr` DAG this way, against ~250 B with the terms passed here).
///
/// `keep_expressions=False` frees the expression trees once they are taped
/// (gh#1261), for a frontend that only evaluates. Values, derivatives,
/// sparsity and every solver query are unchanged — what the evaluator still
/// asks of a row is cached when its tree goes — but nothing can read a tree
/// back afterwards: a rebuild from the problem is refused. On discopt#2293's
/// nonlinear rows the trees were the largest single item, 544–700 B a row in
/// 14 small allocations.
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
    con_linear=None,
    keep_expressions=true,
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
    con_linear: Option<&Bound<'_, PyAny>>,
    keep_expressions: bool,
) -> PyResult<PyNlProblem> {
    let objective = coerce(objective, "build_nl_problem: objective")?;
    let (constraints, con_depth) = match constraints {
        None => (Vec::new(), 0),
        Some(c) => coerce_all(c, "build_nl_problem: constraints")?,
    };
    let m = constraints.len();
    let con_linear = csr_linear(con_linear, m, n)?;
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
        // All rows at once, so a subexpression several rows reference is
        // one shared body rather than a copy per row (gh#1232); each row's
        // own top-level sum structure is still inlined, so it splits into
        // the summands it always did. See `materialize_model`.
        let mut parts = parts;
        let mut built = {
            let roots: Vec<&Expr> = std::iter::once(&parts.objective)
                .chain(parts.constraints.iter())
                .collect();
            materialize_model(&roots)
        }
        .into_iter();
        parts.objective = built.next().expect("one materialized root per input root");
        parts.constraints = built.collect();
        let mut prob = NlProblem::from_expressions(parts)?;
        // The rows are the caller's DAG, hand-offs and all.
        prob.frontend_dag = true;
        // Set after `from_expressions`, which builds every row's linear part
        // empty: every evaluator, the Jacobian structure and presolve already
        // read `con_linear`, because a parsed `.nl` model fills it.
        if let Some(lin) = con_linear {
            prob.con_linear = lin;
        }
        let mut tnlp = NlTnlp::try_new(prob)?;
        if !keep_expressions {
            // Dropped here, on the deep stack, like the build's own walks.
            drop(tnlp.drop_expression_trees());
        }
        Ok::<_, String>(tnlp)
    })
    .map_err(|e| PyValueError::new_err(format!("build_nl_problem: {e}")))?;
    // A problem that keeps the expression trees it taped inherits their
    // depth for its own copies and teardown. One that dropped them holds at
    // most a trivially-zero `Const` per body, depth 1.
    let depth = if keep_expressions { depth } else { 1 };
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

    // ---- gh#1232: sharing across rows ----

    fn v(i: usize) -> PyNlExpr {
        PyNlExpr::leaf(Expr::Var(i))
    }

    /// A rate expression of the kind a kinetics model reuses in every
    /// species balance, deep enough that sharing it across rows pays:
    /// `e ← exp(e · x_j) · 0.5`, twelve times over, ending in `· 0.5`.
    fn rate() -> PyNlExpr {
        let mut e = v(0);
        for j in 0..12 {
            let p = binary(BinOp::Mul, e.operand(), v(j % 2).operand()).unwrap();
            let x = unary(UnaryOp::Exp, p.operand()).unwrap();
            e = binary(BinOp::Mul, x.operand(), num(0.5)).unwrap();
        }
        e
    }

    /// Rows as `build_nl_problem` receives them: each one an operand, i.e.
    /// a `Cse` reference to the row's payload.
    fn roots(rows: &[PyNlExpr]) -> Vec<Expr> {
        rows.iter().map(|r| r.operand().expr).collect()
    }

    fn bodies(e: &Expr) -> std::collections::HashSet<*const Expr> {
        let mut refs = HashMap::new();
        count_refs(e, &mut refs);
        refs.into_keys().collect()
    }

    /// `n`-variable model of `rows`, unbounded, with the rows marked as a
    /// frontend DAG or not.
    fn assemble(n: usize, rows: Vec<Expr>, frontend_dag: bool) -> NlTnlp {
        let m = rows.len();
        let mut prob = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: rows,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![0.3; n],
            g_l: vec![-1e19; m],
            g_u: vec![1e19; m],
            var_names: vec![],
            con_names: vec![],
        })
        .unwrap();
        prob.frontend_dag = frontend_dag;
        NlTnlp::try_new(prob).unwrap()
    }

    /// Every evaluator of `a` against `b`, bit for bit: both sparsity
    /// patterns, and `g`, the Jacobian, the Lagrangian Hessian and a
    /// Hessian-vector product at three points with nonuniform multipliers.
    /// Also that they agree on keeping a shared-CSE tape, since the hybrid
    /// paths compute the same numbers by a different route.
    fn assert_evaluate_alike(a: &mut NlTnlp, b: &mut NlTnlp, what: &str) {
        use pounce_nlp::tnlp::{SparsityRequest, TNLP};
        let (ia, ib) = (a.get_nlp_info().unwrap(), b.get_nlp_info().unwrap());
        assert_eq!(
            (ia.n, ia.m, ia.nnz_jac_g, ia.nnz_h_lag),
            (ib.n, ib.m, ib.nnz_jac_g, ib.nnz_h_lag),
            "{what}: sizes"
        );
        assert_eq!(
            a.has_shared_cse_tape(),
            b.has_shared_cse_tape(),
            "{what}: hybrid kept"
        );
        let (n, m) = (ia.n as usize, ia.m as usize);
        let (nj, nh) = (ia.nnz_jac_g as usize, ia.nnz_h_lag as usize);
        let structure = |t: &mut NlTnlp| {
            let (mut jr, mut jc) = (vec![0; nj], vec![0; nj]);
            let (mut hr, mut hc) = (vec![0; nh], vec![0; nh]);
            t.eval_jac_g(
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut jr,
                    jcol: &mut jc,
                },
            );
            t.eval_h(
                None,
                true,
                1.0,
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut hr,
                    jcol: &mut hc,
                },
            );
            (jr, jc, hr, hc)
        };
        assert_eq!(structure(a), structure(b), "{what}: sparsity");
        let bits = |v: &[Number]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (k, shift) in [0.0, 0.37, -0.81].into_iter().enumerate() {
            let x: Vec<Number> = (0..n)
                .map(|j| 0.2 + shift + 0.6 * ((j * 7919 + k * 104729) % 1000) as Number / 1000.0)
                .collect();
            let lam: Vec<Number> = (0..m).map(|i| 0.5 - (i % 7) as Number * 0.13).collect();
            let v: Vec<Number> = (0..n).map(|j| 1.0 - 0.01 * (j % 13) as Number).collect();
            let run = |t: &mut NlTnlp| {
                let mut g = vec![0.0; m];
                let mut jv = vec![0.0; nj];
                let mut hv = vec![0.0; nh];
                let mut hvp = vec![0.0; n];
                assert!(t.eval_g(&x, true, &mut g));
                assert!(t.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jv }));
                assert!(t.eval_h(
                    Some(&x),
                    true,
                    0.7,
                    Some(&lam),
                    true,
                    SparsityRequest::Values { values: &mut hv }
                ));
                t.hessian_vector_products(&x, &v, 1, 0.7, Some(&lam), &mut hvp)
                    .unwrap();
                (bits(&g), bits(&jv), bits(&hv), bits(&hvp))
            };
            let (ra, rb) = (run(a), run(b));
            assert_eq!(ra.0, rb.0, "{what}: eval_g at point {k}");
            assert_eq!(ra.1, rb.1, "{what}: eval_jac_g at point {k}");
            assert_eq!(ra.2, rb.2, "{what}: eval_h at point {k}");
            assert_eq!(ra.3, rb.3, "{what}: hessian_vector_products at point {k}");
        }
    }

    /// gh#1235's kinetics rows, small: `y_s - x_s - h·(r_a - r_b + 2·r_c)`
    /// with each rate `exp(p)·y·y - exp(p')·y` shared by several rows, so the
    /// spine sees through hand-offs at every level and splits each rate into
    /// two summands, scaled by `-h`, `h` and `-2h`.
    fn kinetics_rows(steps: usize) -> (usize, Vec<PyNlExpr>) {
        let s_n = 6;
        let nx = (steps + 1) * s_n;
        let n = nx + 4;
        let ep: Vec<PyNlExpr> = (0..4)
            .map(|j| unary(UnaryOp::Exp, v(nx + j).operand()).unwrap())
            .collect();
        let mut rows = Vec::new();
        for k in 0..steps {
            let y = |s: usize| v((k + 1) * s_n + s);
            let rates: Vec<PyNlExpr> = (0..3)
                .map(|j| {
                    let t = binary(BinOp::Mul, ep[j % 4].operand(), y(j % s_n).operand()).unwrap();
                    let t = binary(BinOp::Mul, t.operand(), y((j + 1) % s_n).operand()).unwrap();
                    let u = binary(
                        BinOp::Mul,
                        ep[(j + 1) % 4].operand(),
                        y((j + 2) % s_n).operand(),
                    )
                    .unwrap();
                    binary(BinOp::Sub, t.operand(), u.operand()).unwrap()
                })
                .collect();
            for s in 0..s_n {
                let d = binary(
                    BinOp::Sub,
                    rates[s % 3].operand(),
                    rates[(s + 1) % 3].operand(),
                )
                .unwrap();
                let c = binary(BinOp::Mul, num(2.0), rates[(s + 2) % 3].operand()).unwrap();
                let rhs = binary(BinOp::Add, d.operand(), c.operand()).unwrap();
                let hr = binary(BinOp::Mul, num(0.01), rhs.operand()).unwrap();
                let l = binary(BinOp::Sub, y(s).operand(), v(k * s_n + s).operand()).unwrap();
                rows.push(binary(BinOp::Sub, l.operand(), hr.operand()).unwrap());
            }
        }
        (n, rows)
    }

    /// gh#1232's shape, small: deep bodies each shared by many rows, whose
    /// summand leaf — `exp(…)` behind the `· 0.5` the split distributes — is
    /// itself one node every row of a body reaches. That clears both hybrid
    /// gates, and it is the case where storing the leaf as the caller's `Cse`
    /// rather than as a node of its own once moved the hybrid Hessian.
    fn deep_shared_rows() -> (usize, Vec<PyNlExpr>) {
        let n = 30;
        let pool: Vec<PyNlExpr> = (0..3)
            .map(|i| {
                let mut e = v(i);
                for j in 0..14 {
                    let p = binary(BinOp::Mul, e.operand(), v((i + j) % 10).operand()).unwrap();
                    let x = unary(UnaryOp::Exp, p.operand()).unwrap();
                    e = binary(BinOp::Mul, x.operand(), num(0.5)).unwrap();
                }
                e
            })
            .collect();
        let rows = (0..60)
            .map(|i| binary(BinOp::Add, pool[i % 3].operand(), v(10 + i % 20).operand()).unwrap())
            .collect();
        (n, rows)
    }

    /// Spine shapes the split has to see through exactly as the inlined tree
    /// shows them: scaling and division by a constant, subtraction and
    /// negation, an n-ary sum, a quotient by zero (no scaling), a constant
    /// reached through a hand-off, and a `Cse` the row references twice,
    /// which stays a leaf of its own.
    fn spine_shape_rows() -> (usize, Vec<PyNlExpr>) {
        let k = rate();
        let twice = binary(BinOp::Mul, k.operand(), v(3).operand()).unwrap();
        let three = binary(BinOp::Add, num(1.0), num(2.0)).unwrap();
        let rows = vec![
            {
                let d = binary(BinOp::Sub, k.operand(), v(2).operand()).unwrap();
                let m = binary(BinOp::Mul, num(2.0), d.operand()).unwrap();
                binary(BinOp::Div, m.operand(), num(3.0)).unwrap()
            },
            {
                let n_ = unary(UnaryOp::Neg, k.operand()).unwrap();
                binary(BinOp::Add, n_.operand(), twice.operand()).unwrap()
            },
            binary(BinOp::Div, k.operand(), num(0.0)).unwrap(),
            PyNlExpr::nested(
                Expr::Sum(vec![k.operand().expr, twice.operand().expr, Expr::Var(4)]),
                1,
            )
            .unwrap(),
            binary(
                BinOp::Mul,
                three.operand(),
                binary(BinOp::Sub, v(1).operand(), v(4).operand())
                    .unwrap()
                    .operand(),
            )
            .unwrap(),
            binary(BinOp::Add, k.operand(), k.operand()).unwrap(),
        ];
        (5, rows)
    }

    /// The point of gh#1235's last step: `build_nl_problem` stores one
    /// compact copy of the caller's DAG rather than a copy per row, and
    /// every evaluator gives exactly what the copied rows gave — against the
    /// copy-based build this replaced (cross-row sharing and all), and
    /// against each row inlined on its own. The models reach each branch that matters: rows
    /// that split through hand-offs at every level and keep no hybrid tape
    /// (kinetics, spine shapes), small bodies shared by thousands of rows
    /// (`exp(p_j)`), and a model above both hybrid gates whose summand leaf
    /// many rows share (deep).
    #[test]
    fn stored_rows_evaluate_exactly_as_the_copied_rows_did() {
        let small = {
            let rows_n = 200;
            let p: Vec<PyNlExpr> = (0..10)
                .map(|j| unary(UnaryOp::Exp, v(rows_n + j).operand()).unwrap())
                .collect();
            let rows: Vec<PyNlExpr> = (0..rows_n)
                .map(|i| {
                    let (a, b, c) = (v(i), v((i + 1) % rows_n), v((i + 2) % rows_n));
                    let t1 = binary(BinOp::Mul, p[i % 10].operand(), a.operand()).unwrap();
                    let t1 = binary(BinOp::Mul, t1.operand(), b.operand()).unwrap();
                    let t2 = binary(BinOp::Mul, p[(i + 1) % 10].operand(), c.operand()).unwrap();
                    let d = binary(BinOp::Sub, t1.operand(), t2.operand()).unwrap();
                    let s = binary(BinOp::Mul, num(0.01), d.operand()).unwrap();
                    let l = binary(BinOp::Sub, a.operand(), b.operand()).unwrap();
                    binary(BinOp::Sub, l.operand(), s.operand()).unwrap()
                })
                .collect();
            (rows_n + 10, rows)
        };
        let models = [
            ("kinetics", kinetics_rows(12)),
            ("deep shared", deep_shared_rows()),
            ("spine shapes", spine_shape_rows()),
            ("small shared bodies", small),
        ];
        for (what, (n, rows)) in models {
            let roots = roots(&rows);
            let refs: Vec<&Expr> = roots.iter().collect();
            let mut stored = assemble(n, materialize_model(&refs), true);
            let mut copied = assemble(n, materialize_model_copied(&refs), false);
            assert_evaluate_alike(&mut stored, &mut copied, &format!("{what} vs copied"));
            let mut per_row = assemble(n, roots.iter().map(materialize).collect(), false);
            if !stored.has_shared_cse_tape() {
                // Without a hybrid tape there is no cross-row sharing to
                // differ in, so each row inlined on its own must agree too.
                assert_evaluate_alike(&mut stored, &mut per_row, &format!("{what} vs per row"));
            }
        }
        // The branch the deep model exists for.
        let (n, rows) = deep_shared_rows();
        let roots = roots(&rows);
        let refs: Vec<&Expr> = roots.iter().collect();
        assert!(assemble(n, materialize_model(&refs), true).has_shared_cse_tape());
    }

    /// `build_nl_problem`'s assembly of `objective` and `rows`, through
    /// `materialize_model` and with the rows marked a frontend DAG.
    fn assemble_model(n: usize, objective: &PyNlExpr, rows: &[PyNlExpr]) -> NlTnlp {
        let obj = objective.operand().expr;
        let roots = roots(rows);
        let refs: Vec<&Expr> = std::iter::once(&obj).chain(roots.iter()).collect();
        let mut built = materialize_model(&refs).into_iter();
        let objective = built.next().unwrap();
        let constraints: Vec<Expr> = built.collect();
        let m = constraints.len();
        let mut prob = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective,
            obj_constant: 0.0,
            constraints,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![0.3; n],
            g_l: vec![-1e19; m],
            g_u: vec![1e19; m],
            var_names: vec![],
            con_names: vec![],
        })
        .unwrap();
        prob.frontend_dag = true;
        NlTnlp::try_new(prob).unwrap()
    }

    /// One row per answer a [`NlBody::Taped`] caches, each on the branch
    /// that is not the common one: an affine tree (`Some(true)`, with a
    /// quadratic read-out), a bilinear one (`Some(false)`), a transcendental
    /// one (`None`, no read-out), a row whose quadratic fold lost a term
    /// (`quad_terms_dropped`), and a trivially-zero row, which keeps its
    /// tree. The objective is nonlinear in a variable no row is.
    fn taped_branch_rows() -> (usize, PyNlExpr, Vec<PyNlExpr>) {
        let affine = {
            let a = binary(BinOp::Mul, num(2.0), v(0).operand()).unwrap();
            binary(BinOp::Add, a.operand(), v(1).operand()).unwrap()
        };
        let bilinear = binary(BinOp::Mul, v(0).operand(), v(1).operand()).unwrap();
        let transcendental = {
            let e = unary(UnaryOp::Exp, v(2).operand()).unwrap();
            binary(BinOp::Mul, e.operand(), v(1).operand()).unwrap()
        };
        let lost = {
            // 2⁵³·x₃² + x₃² − 2⁵³·x₃²: `fl(2⁵³ + 1)` absorbs the middle term.
            let big = 9_007_199_254_740_992.0;
            let sq = |c: Number| {
                let s = binary(BinOp::Mul, v(3).operand(), v(3).operand()).unwrap();
                binary(BinOp::Mul, num(c), s.operand()).unwrap()
            };
            let a = binary(BinOp::Add, sq(big).operand(), sq(1.0).operand()).unwrap();
            binary(BinOp::Sub, a.operand(), sq(big).operand()).unwrap()
        };
        let zero = PyNlExpr::leaf(Expr::Const(0.0));
        let objective = {
            let s = unary(UnaryOp::Sin, v(4).operand()).unwrap();
            binary(BinOp::Add, s.operand(), bilinear.operand()).unwrap()
        };
        (
            5,
            objective,
            vec![affine, bilinear, transcendental, lost, zero],
        )
    }

    /// Everything a solve asks of the problem's bodies rather than of its
    /// tapes, as one comparable value.
    fn body_queries(t: &mut NlTnlp) -> String {
        use pounce_nlp::tnlp::{Linearity, TNLP};
        let info = t.get_nlp_info().unwrap();
        let (n, m) = (info.n as usize, info.m as usize);
        let mut cl = vec![Linearity::Linear; m];
        let mut vl = vec![Linearity::Linear; n];
        let mut ol = vec![Linearity::Linear; n];
        assert!(t.get_constraints_linearity(&mut cl));
        assert!(t.get_variables_linearity(&mut vl));
        assert!(t.get_objective_variables_linearity(&mut ol));
        let k = t.get_number_of_nonlinear_variables() as usize;
        let mut list = vec![0; k];
        assert!(t.get_list_of_nonlinear_variables(&mut list));
        let proofs = t.derivative_proofs();
        let prob = t.problem();
        format!(
            "{cl:?}\n{vl:?}\n{ol:?}\n{list:?}\n{proofs:?}\n{:?}\n{:?}\n{:?}",
            pounce_nl::nl_reader::constraint_jacobian_sparsity(prob),
            pounce_nl::nl_scaling::quad_row_coefs(prob),
            pounce_nl::nl_scaling::curvature_scaling(prob),
        )
    }

    /// `keep_expressions=False` (gh#1261): freeing the trees changes no
    /// value, no derivative, no structure and no answer the solver asks of
    /// a body, and leaves no way to read a tree back or re-tape.
    ///
    /// The models reach every branch of the cache: the kinetics, deep (above
    /// both hybrid gates) and spine-shape rows the stored-row test uses, the
    /// one-row-per-branch model above, and an all-quadratic model, the only
    /// shape on which `curvature_scaling` answers at all.
    #[test]
    fn dropping_the_trees_changes_no_answer() {
        let (n, objective, rows) = taped_branch_rows();
        let quadratic = {
            let sq = |i: usize| binary(BinOp::Mul, v(i).operand(), v(i).operand()).unwrap();
            let obj = binary(BinOp::Add, sq(0).operand(), sq(1).operand()).unwrap();
            let row = binary(BinOp::Mul, v(0).operand(), v(1).operand()).unwrap();
            (2, obj, vec![row, sq(1)])
        };
        let zero = PyNlExpr::leaf(Expr::Const(0.0));
        let models = [
            ("kinetics", kinetics_rows(4)),
            ("deep", deep_shared_rows()),
            ("spine", spine_shape_rows()),
        ]
        .into_iter()
        .map(|(w, (n, rows))| (w, n, zero.clone(), rows))
        .chain([
            ("branches", n, objective, rows),
            ("quadratic", quadratic.0, quadratic.1, quadratic.2),
        ]);
        let mut seen = (false, false, false, false, false);
        for (what, n, objective, rows) in models {
            let mut kept = assemble_model(n, &objective, &rows);
            let mut dropped = assemble_model(n, &objective, &rows);
            let trees = dropped.drop_expression_trees();
            assert!(!trees.is_empty(), "{what}: some tree was dropped");
            drop(trees);
            assert!(dropped.expression_trees_dropped(), "{what}");
            assert!(!kept.expression_trees_dropped(), "{what}");

            for b in std::iter::once(&dropped.problem().obj_nonlinear)
                .chain(dropped.problem().con_nonlinear.iter())
            {
                match b {
                    NlBody::Taped(t) => {
                        seen.0 |= t.affine == Some(true);
                        seen.1 |= t.affine == Some(false);
                        seen.2 |= t.affine.is_none();
                        seen.3 |= t.quad_terms_dropped;
                        seen.4 |= t.quad_form.is_some();
                    }
                    NlBody::Tree(e) => assert!(
                        matches!(e, Expr::Const(c) if *c == 0.0),
                        "{what}: only a trivially-zero tree is kept, got {e:?}"
                    ),
                    NlBody::Quad(_) => unreachable!("no parsed bodies here"),
                }
            }

            assert_evaluate_alike(&mut dropped, &mut kept, &format!("{what}: dropped vs kept"));
            {
                use pounce_nlp::tnlp::TNLP;
                let bits = |v: &[Number]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let x: Vec<Number> = (0..n).map(|j| 0.4 + 0.07 * j as Number).collect();
                let f = |t: &mut NlTnlp| {
                    let mut g = vec![0.0; n];
                    assert!(t.eval_grad_f(&x, true, &mut g));
                    (t.eval_f(&x, true).unwrap().to_bits(), bits(&g))
                };
                assert_eq!(f(&mut dropped), f(&mut kept), "{what}: objective");
            }
            assert_eq!(
                body_queries(&mut dropped),
                body_queries(&mut kept),
                "{what}: body queries"
            );

            // Loud, not silent: a re-tape is refused, and a copy (the batch
            // path's `clone_tnlp`, `variant`) still evaluates.
            let err = NlTnlp::try_new(dropped.problem().clone()).err().unwrap();
            assert!(err.contains("keep_expressions=False"), "{what}: {err}");
            let mut copy = dropped.variant(&Default::default()).unwrap();
            assert_evaluate_alike(&mut copy, &mut kept, &format!("{what}: variant"));
        }
        assert_eq!(
            seen,
            (true, true, true, true, true),
            "the fixtures reach every cached branch: (affine, curved, unknown, \
             lost a term, has a read-out)"
        );
    }

    /// The panic a consumer of a dropped tree gets says why.
    #[test]
    #[should_panic(expected = "keep_expressions=False")]
    fn reading_a_dropped_tree_panics_by_name() {
        let (n, objective, rows) = taped_branch_rows();
        let mut t = assemble_model(n, &objective, &rows);
        drop(t.drop_expression_trees());
        let _ = t.problem().con_expr(2);
    }

    /// Every `Arc` a stored row reaches, by address.
    fn arcs(e: &Expr, out: &mut std::collections::HashSet<*const Expr>) {
        let mut refs = HashMap::new();
        count_refs(e, &mut refs);
        out.extend(refs.into_keys());
    }

    /// The problem keeps a copy, not the caller's nodes: once the caller
    /// drops its expressions they are freed, rather than pinned by the
    /// problem in their bulkier `NlExpr` form (gh#1235).
    #[test]
    fn the_stored_rows_hold_none_of_the_callers_nodes() {
        for (_, rows) in [kinetics_rows(3), deep_shared_rows(), spine_shape_rows()] {
            let roots = roots(&rows);
            let refs: Vec<&Expr> = roots.iter().collect();
            let mut theirs = std::collections::HashSet::new();
            for r in &roots {
                arcs(r, &mut theirs);
            }
            let mut ours = std::collections::HashSet::new();
            for b in materialize_model(&refs) {
                arcs(&b, &mut ours);
            }
            assert!(!theirs.is_empty());
            assert!(
                ours.is_disjoint(&theirs),
                "a stored row pins a caller's node"
            );
        }
    }

    /// A summand leaf several summands reach — each rate's `exp(p)·y·y` in
    /// the kinetics rows — is copied once and shared, not copied per
    /// reference. And a body several rows reach is one body (gh#1232).
    #[test]
    fn a_leaf_several_summands_reach_is_copied_once() {
        let (_, rows) = kinetics_rows(2);
        let roots = roots(&rows);
        let refs: Vec<&Expr> = roots.iter().collect();
        let built = materialize_model(&refs);
        // Each row's summand leaves, as `NlTnlp` would split them.
        let mut leaf_arcs: HashMap<*const Expr, usize> = HashMap::new();
        for b in &built {
            let mut row = HashMap::new();
            count_refs(b, &mut row);
            spine(b, &row, &mut |_, reached: &Expr| {
                if let Expr::Cse(a) = reached {
                    *leaf_arcs.entry(Arc::as_ptr(a)).or_insert(0) += 1;
                }
                Expr::Const(0.0)
            });
        }
        // 2 steps x 3 rates x 2 products, each reached from several rows.
        assert_eq!(leaf_arcs.len(), 12, "one copy per distinct leaf");
        assert!(leaf_arcs.values().all(|&k| k >= 2));
    }

    /// The same rows split as their inlined trees would: the summand tapes
    /// of a frontend-DAG model are op for op those of each row inlined on its
    /// own, read through `Tape::build` of the summands the split hands out.
    /// Checked on the whole-row tape too, which no split is involved in.
    #[test]
    fn a_stored_row_tapes_as_its_inlined_tree() {
        for (_, rows) in [kinetics_rows(2), spine_shape_rows()] {
            for r in roots(&rows) {
                assert_eq!(
                    format!("{:?}", Tape::build(&r).ops),
                    format!("{:?}", Tape::build(&materialize(&r)).ops),
                    "whole-row tape of {}",
                    render_expression(&r, &[])
                );
            }
        }
    }

    /// A row the quadratic recognizer admits from its per-row tree keeps that
    /// tree verbatim. The recognizer refuses some `Cse` placements the plain
    /// tree passes — here `peel_square`, which reads the weight of `w·r²`
    /// through products and negations but not through a `Cse` — so sharing
    /// a coefficient `w = -(2)` that every row uses would silently move each
    /// row off the constant-structure fast path and back onto the AD tape.
    #[test]
    fn a_row_the_quadratic_recognizer_admits_keeps_its_private_tree() {
        // r = x0 - x1 and w = -(2), both shared by every quadratic row:
        // w r^2 + x_{2+i}, a least-squares shape. Next to them, rows sharing
        // a deep rate expression, which do share it.
        let r = binary(BinOp::Sub, v(0).operand(), v(1).operand()).unwrap();
        let sq = binary(BinOp::Pow, r.operand(), num(2.0)).unwrap();
        let w = unary(UnaryOp::Neg, num(2.0)).unwrap();
        let k = rate();
        let mut rows = Vec::new();
        for i in 0..40 {
            let s3 = binary(BinOp::Mul, w.operand(), sq.operand()).unwrap();
            rows.push(binary(BinOp::Add, s3.operand(), v(2 + i).operand()).unwrap());
            rows.push(binary(BinOp::Add, k.operand(), v(2 + i).operand()).unwrap());
        }
        let roots = roots(&rows);
        let refs: Vec<&Expr> = roots.iter().collect();
        let built = materialize_model(&refs);
        for (i, b) in built.iter().enumerate() {
            let body = NlBody::Tree(b.clone());
            let admitted =
                body.admitted_quad_form().is_some() || body.admitted_factored_form().is_some();
            if i % 2 == 0 {
                assert!(admitted, "row {i}: the quadratic row lost its fast path");
                assert_eq!(count_cse(b), 0, "row {i}: a kept tree has no Cse");
            } else {
                assert!(!admitted);
            }
        }
        // The rate rows do share the rate among themselves.
        assert!(!bodies(&built[1]).is_empty());
        assert_eq!(bodies(&built[1]), bodies(&built[3]));
    }
}
