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
//! * Optional initial primal (`x`) segment. Initial dual (`d`) is
//!   read and discarded.
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

use crate::nl_tape::Tape;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, MetaData, NlpInfo, Solution,
    SparsityRequest, StartingPoint, IDX_NAMES, TNLP,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::rc::Rc;

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
    /// one `Rc`, so the parsed problem is a DAG. Walking through `Cse`
    /// is mathematically equivalent to inlining the body at each
    /// occurrence (every reference is an independent occurrence in the
    /// chain rule), so eval/grad/collect_vars just recurse into the
    /// inner `Expr`.
    Cse(Rc<Expr>),
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
}

/// Parsed `.nl` problem in the form needed by `NlTnlp`.
#[derive(Debug, Clone)]
pub struct NlProblem {
    pub n: usize,
    pub m: usize,
    pub num_obj: usize,
    pub minimize: bool,
    pub obj_nonlinear: Expr,
    pub obj_linear: Vec<(usize, Number)>,
    pub obj_constant: Number,
    /// Per-constraint nonlinear part (length m).
    pub con_nonlinear: Vec<Expr>,
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
    let mut prob = parse_nl_text(&txt)?;
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

/// Parse `.nl` text content. Public so tests can use string literals.
pub fn parse_nl_text(txt: &str) -> Result<NlProblem, String> {
    let mut p = Parser::new(txt);
    p.parse_header()?;
    let n = p.n;
    let m = p.m;
    let num_obj = p.num_obj;

    let mut con_nonlinear: Vec<Expr> = (0..m).map(|_| Expr::Const(0.0)).collect();
    let mut obj_nonlinear = Expr::Const(0.0);
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
                let idx = parse_segment_index(&_hdr, 'C')?;
                if idx >= m {
                    return Err(format!("C{idx} out of range; m={m}"));
                }
                con_nonlinear[idx] = p.parse_expr()?;
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
                    obj_nonlinear = p.parse_expr()?;
                } else {
                    // Extra objectives are read but ignored.
                    let _ = p.parse_expr()?;
                }
            }
            'r' => {
                p.eat_segment_header()?;
                for i in 0..m {
                    let line = p.next_data_line()?;
                    let (lo, hi) = parse_bound_line(&line)?;
                    g_l[i] = lo;
                    g_u[i] = hi;
                }
            }
            'b' => {
                p.eat_segment_header()?;
                for i in 0..n {
                    let line = p.next_data_line()?;
                    let (lo, hi) = parse_bound_line(&line)?;
                    x_l[i] = lo;
                    x_u[i] = hi;
                }
            }
            'k' => {
                // Column counts in the Jacobian; we don't need them
                // for evaluation since J segments give explicit lists.
                p.eat_segment_header()?;
                let count = if n == 0 { 0 } else { n - 1 };
                for _ in 0..count {
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
                    let (var, coef) = parse_var_coef(&line)?;
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
                    let (var, coef) = parse_var_coef(&line)?;
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
                    let (idx, val) = parse_var_coef(&line)?;
                    if idx < n {
                        x0[idx] = val;
                    }
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
                    let (idx, val) = parse_var_coef(&line)?;
                    if idx < m {
                        lambda0[idx] = val;
                    }
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
        imported_funcs,
        // `.nl` text carries no names; `read_nl_file` fills these from the
        // sibling `.col`/`.row` files when present.
        var_names: Vec::new(),
        con_names: Vec::new(),
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

fn parse_bound_line(line: &str) -> Result<(Number, Number), String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Err("empty bound line".into());
    }
    let kind: i32 = parts[0].parse().map_err(|e| format!("bound kind: {e}"))?;
    let lo;
    let hi;
    match kind {
        0 => {
            // 0  lo  hi
            if parts.len() < 3 {
                return Err(format!("bound kind 0 needs 2 values: '{line}'"));
            }
            lo = parts[1].parse().map_err(|e| format!("lo: {e}"))?;
            hi = parts[2].parse().map_err(|e| format!("hi: {e}"))?;
        }
        1 => {
            // 1  hi
            if parts.len() < 2 {
                return Err(format!("bound kind 1 needs 1 value: '{line}'"));
            }
            lo = -1e19;
            hi = parts[1].parse().map_err(|e| format!("hi: {e}"))?;
        }
        2 => {
            // 2  lo
            if parts.len() < 2 {
                return Err(format!("bound kind 2 needs 1 value: '{line}'"));
            }
            lo = parts[1].parse().map_err(|e| format!("lo: {e}"))?;
            hi = 1e19;
        }
        3 => {
            // 3  (free)
            lo = -1e19;
            hi = 1e19;
        }
        4 => {
            // 4  eq
            if parts.len() < 2 {
                return Err(format!("bound kind 4 needs 1 value: '{line}'"));
            }
            let v: Number = parts[1].parse().map_err(|e| format!("eq: {e}"))?;
            lo = v;
            hi = v;
        }
        5 => return Err("complementarity (kind 5) bounds are not supported".into()),
        other => return Err(format!("unknown bound kind {other}")),
    }
    Ok((lo, hi))
}

fn parse_var_coef(line: &str) -> Result<(usize, Number), String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(format!("malformed var/coef line: '{line}'"));
    }
    let v: usize = parts[0].parse().map_err(|e| format!("var idx: {e}"))?;
    let c: Number = parts[1].parse().map_err(|e| format!("coef: {e}"))?;
    Ok((v, c))
}

struct Parser<'a> {
    lines: Vec<&'a str>,
    pos: usize,
    n: usize,
    m: usize,
    num_obj: usize,
    /// Number of AMPL imported (external) functions declared in the header.
    n_funcs: usize,
    /// Common subexpressions (`V` segments). Index in this vec is the
    /// CSE-local index, i.e. the global `.nl` index minus `n`.
    cses: Vec<Rc<Expr>>,
}

impl<'a> Parser<'a> {
    fn new(txt: &'a str) -> Self {
        let lines: Vec<&str> = txt.lines().collect();
        Self {
            lines,
            pos: 0,
            n: 0,
            m: 0,
            num_obj: 0,
            n_funcs: 0,
            cses: Vec::new(),
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

    fn next_data_line(&mut self) -> Result<String, String> {
        let raw = self
            .next_line()
            .ok_or_else(|| "unexpected end of file in data line".to_string())?;
        Ok(strip_comment(raw).trim().to_string())
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

        // Header line 2: n_vars n_cons n_objs ranges eqns
        let l2 = self.next_data_line()?;
        let nums: Vec<&str> = l2.split_whitespace().collect();
        if nums.len() < 3 {
            return Err(format!("malformed line 2: '{l2}'"));
        }
        self.n = nums[0].parse().map_err(|e| format!("n: {e}"))?;
        self.m = nums[1].parse().map_err(|e| format!("m: {e}"))?;
        self.num_obj = nums[2].parse().map_err(|e| format!("num_obj: {e}"))?;

        // Lines 3..5 are metadata we skip.
        for _ in 0..3 {
            self.next_data_line()?;
        }
        // Line 5 (0-indexed from `g`-header): `nwv nfunc arith flags`
        let l5 = self.next_data_line()?;
        let nums5: Vec<&str> = l5.split_whitespace().collect();
        self.n_funcs = nums5.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        // Lines 6..10 are metadata we don't need — skip 4 more lines.
        for _ in 0..4 {
            self.next_data_line()?;
        }
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
    fn eat_segment_header(&mut self) -> Result<(String, String), String> {
        let raw = self
            .next_line()
            .ok_or_else(|| "expected segment header".to_string())?;
        let (hdr, comment) = split_comment(raw);
        Ok((hdr.trim().to_string(), comment.trim().to_string()))
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let raw = self
            .next_line()
            .ok_or_else(|| "expected expression token".to_string())?;
        let tok = strip_comment(raw).trim().to_string();
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
                Ok(Expr::Const(v))
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
        let tok = strip_comment(raw).trim().to_string();
        let first = tok.chars().next().ok_or("empty funcall arg token")?;
        if first == 'h' {
            // `h<len>:<chars>` — strip the `<len>:` prefix.
            let rest = &tok[1..];
            let s = match rest.find(':') {
                Some(i) => rest[i + 1..].to_string(),
                None => String::new(),
            };
            Ok(FuncallArg::Str(s))
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
            let (var, coef) = parse_var_coef(&line)?;
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
        self.cses.push(Rc::new(combined));
        Ok(())
    }
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
            if truth {
                1.0
            } else {
                0.0
            }
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
pub fn collect_vars(e: &Expr, out: &mut BTreeSet<usize>) {
    match e {
        Expr::Const(_) => {}
        Expr::Var(i) => {
            out.insert(*i);
        }
        Expr::Binary(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Unary(_, a) => collect_vars(a, out),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                collect_vars(a, out);
            }
        }
        // Collect from every child, including the condition: even
        // though the comparison/branch-test contributes no derivative,
        // the variables it reads are genuinely "used" by the problem,
        // and being conservative here only ever adds structural zeros
        // to the Jacobian/Hessian (never drops a real nonzero).
        Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Not(a) => collect_vars(a, out),
        Expr::Cond { cond, then_, else_ } => {
            collect_vars(cond, out);
            collect_vars(then_, out);
            collect_vars(else_, out);
        }
        Expr::Cse(body) => collect_vars(body, out),
        Expr::Funcall { args, .. } => {
            for a in args {
                if let FuncallArg::Real(e) = a {
                    collect_vars(e, out);
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

#[derive(Debug)]
pub struct NlTnlp {
    prob: NlProblem,
    /// Per-summand objective tapes (one `Tape` per top-level
    /// summand after `split_top_sums`).
    obj_tapes: Vec<Tape>,
    /// Per-constraint, per-summand tapes. Length `m`; row `i` holds
    /// one `Tape` per summand of constraint `i`.
    con_tapes: Vec<Vec<Tape>>,
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
    final_x: Option<Vec<Number>>,
    final_obj: Number,
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
    let body = render_body(&prob.con_linear[k], &prob.con_nonlinear[k], prob);
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
        collect_vars(&prob.con_nonlinear[k], &mut support);
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
fn greedy_hessian_coloring(n: usize, lower_pairs: &[(usize, usize)]) -> (Vec<u32>, usize) {
    if n == 0 {
        return (Vec::new(), 0);
    }

    // For each variable k, list of rows in which column k has a
    // nonzero in the FULL (symmetric) Hessian. Built from lower
    // pairs: (i, j) with i >= j contributes row i to column j and
    // row j to column i (when i != j); diagonals contribute once.
    let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut row_cols: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(i, j) in lower_pairs {
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
        // Variable `j` has no Hessian entries → skip (no color).
        if col_rows[j].is_empty() {
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

    (var_color, n_colors as usize)
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
        // Resolve any AMPL imported (external) functions. Walk every
        // nonlinear expression to collect the funcall ids actually
        // referenced; load the libraries named in $AMPLFUNC and bind
        // each id to its (library, registered-name) pair so the tape
        // builder can emit live `TapeOp::Funcall` ops.
        let mut referenced: BTreeSet<usize> = BTreeSet::new();
        super::nl_external::collect_funcall_ids(&prob.obj_nonlinear, &mut referenced);
        for c in &prob.con_nonlinear {
            super::nl_external::collect_funcall_ids(c, &mut referenced);
        }
        let resolver = if referenced.is_empty() {
            super::nl_external::ExternalResolver::default()
        } else {
            super::nl_external::ExternalResolver::build_for_problem(
                &prob.imported_funcs,
                &referenced,
            )?
        };

        // Flatten objective and each constraint into independent
        // summands. Each summand becomes its own `Tape` (CSE bodies
        // are deduplicated within a tape via Rc identity in
        // `Tape::build`; bodies shared across summands are
        // duplicated, which we accept as a simplicity tradeoff).
        let obj_summands = split_top_sums(&prob.obj_nonlinear);
        let obj_tapes: Vec<Tape> = obj_summands
            .iter()
            .map(|e| Tape::build_with_externals(e, &resolver))
            .collect();

        let mut con_tapes: Vec<Vec<Tape>> = Vec::with_capacity(prob.m);
        for k in 0..prob.m {
            let summands = split_top_sums(&prob.con_nonlinear[k]);
            con_tapes.push(
                summands
                    .iter()
                    .map(|e| Tape::build_with_externals(e, &resolver))
                    .collect(),
            );
        }

        // Hessian-of-Lagrangian sparsity: union of each tape's own
        // structural Hessian sparsity.
        let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
        for t in &obj_tapes {
            pairs.extend(t.hessian_sparsity());
        }
        for row in &con_tapes {
            for t in row {
                pairs.extend(t.hessian_sparsity());
            }
        }
        let mut h_irow = Vec::with_capacity(pairs.len());
        let mut h_jcol = Vec::with_capacity(pairs.len());
        let mut hess_map = HashMap::with_capacity(pairs.len());
        for (k, (hi, lo)) in pairs.iter().enumerate() {
            h_irow.push(*hi as i32);
            h_jcol.push(*lo as i32);
            hess_map.insert((*hi, *lo), k);
        }

        // Hessian column coloring. The chromatic number of the
        // column-intersection graph bounds how many directional
        // Hessian-vector products we need per `eval_h` call —
        // typically O(stencil) for PDE-mesh problems.
        let lower_pairs: Vec<(usize, usize)> = pairs.iter().copied().collect();
        let (var_color, n_colors) = greedy_hessian_coloring(prob.n, &lower_pairs);

        // Per-color seed vectors (dense for O(1) Var lookup in
        // `Tape::hessian_directional`).
        let mut seeds: Vec<Vec<f64>> = vec![vec![0.0; prob.n]; n_colors];
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
        let mut decoding: Vec<Vec<ColorWrite>> = vec![Vec::new(); n_colors];
        for (&(i, j), &idx) in hess_map.iter() {
            let c = var_color[j];
            debug_assert!(
                c != u32::MAX,
                "column {j} has Hessian pair {idx} but no color"
            );
            decoding[c as usize].push(ColorWrite {
                row: i as u32,
                hess_idx: idx as u32,
            });
        }

        // Per-tape distinct color set: for each tape, the colors
        // its variables fall into. `eval_h` loops over only these
        // (tape, color) pairs instead of n_tapes × n_colors.
        let tape_colors = |t: &Tape| -> Vec<u32> {
            let mut s: BTreeSet<u32> = BTreeSet::new();
            for v in t.variables() {
                let c = var_color[v];
                if c != u32::MAX {
                    s.insert(c);
                }
            }
            s.into_iter().collect()
        };
        let obj_tape_colors: Vec<Vec<u32>> = obj_tapes.iter().map(tape_colors).collect();
        let con_tape_colors: Vec<Vec<Vec<u32>>> = con_tapes
            .iter()
            .map(|row| row.iter().map(tape_colors).collect())
            .collect();

        // Per-row Jacobian sparsity = union of tape vars plus
        // linear-segment vars.
        let mut jac_cols: Vec<Vec<usize>> = Vec::with_capacity(prob.m);
        let mut jac_nnz = 0;
        for i in 0..prob.m {
            let mut set: BTreeSet<usize> = BTreeSet::new();
            for t in &con_tapes[i] {
                for v in t.variables() {
                    set.insert(v);
                }
            }
            for (v, _) in &prob.con_linear[i] {
                set.insert(*v);
            }
            let cols: Vec<usize> = set.into_iter().collect();
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
        }

        let compressed: Vec<Vec<f64>> = vec![vec![0.0; prob.n]; n_colors];

        Ok(Self {
            prob,
            obj_tapes,
            con_tapes,
            h_irow,
            h_jcol,
            jac_cols,
            jac_nnz,
            seeds,
            decoding,
            obj_tape_colors,
            con_tape_colors,
            final_x: None,
            final_obj: 0.0,
            scratch_row_grad: Vec::new(),
            vals_scratch: vec![0.0; max_tape_n],
            dot_scratch: vec![0.0; max_tape_n],
            adj_scratch: vec![0.0; max_tape_n],
            adj_dot_scratch: vec![0.0; max_tape_n],
            compressed,
        })
    }

    pub fn final_x(&self) -> Option<&[Number]> {
        self.final_x.as_deref()
    }

    pub fn final_obj(&self) -> Number {
        self.final_obj
    }
}

impl pounce_nlp::expression_provider::ExpressionProvider for NlTnlp {
    /// Per-`.nl`-row constraint expression tape, with the linear
    /// part folded in. Returns `None` for constraints that contribute
    /// neither a nonlinear expression nor any linear coefficients
    /// (so FBBT skips them — there's nothing to tighten).
    fn constraint_expression(&self, i: usize) -> Option<pounce_nlp::FbbtTape> {
        let nonlinear = self.prob.con_nonlinear.get(i)?;
        let linear = self
            .prob
            .con_linear
            .get(i)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        crate::nl_fbbt_translate::translate_constraint(nonlinear, linear)
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
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut nl: Number = 0.0;
        for t in &self.obj_tapes {
            nl += t.eval(x);
        }
        let lin: Number = self.prob.obj_linear.iter().map(|(i, c)| c * x[*i]).sum();
        let v = self.prob.obj_constant + nl + lin;
        let signed = if self.prob.minimize { v } else { -v };
        Some(signed)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad.fill(0.0);
        for t in &self.obj_tapes {
            t.gradient_seed(x, 1.0, grad);
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
        for i in 0..self.prob.m {
            let mut nl: Number = 0.0;
            for t in &self.con_tapes[i] {
                nl += t.eval(x);
            }
            let lin: Number = self.prob.con_linear[i].iter().map(|(j, c)| c * x[*j]).sum();
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
                let xs = x.unwrap_or(&self.prob.x0);
                if self.scratch_row_grad.len() < n {
                    self.scratch_row_grad.resize(n, 0.0);
                }
                let mut k = 0;
                for i in 0..self.prob.m {
                    for &j in &self.jac_cols[i] {
                        self.scratch_row_grad[j] = 0.0;
                    }
                    for t in &self.con_tapes[i] {
                        t.gradient_seed(xs, 1.0, &mut self.scratch_row_grad);
                    }
                    for &(v, c) in &self.prob.con_linear[i] {
                        self.scratch_row_grad[v] += c;
                    }
                    for &j in &self.jac_cols[i] {
                        values[k] = self.scratch_row_grad[j];
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

                if let Some(lam) = lambda {
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
        // identity zero left over from initial allocation (post-parse
        // identity for "no `C<idx>` segment touched this row").
        for (i, t) in types.iter_mut().enumerate() {
            *t = match &self.prob.con_nonlinear[i] {
                Expr::Const(c) if *c == 0.0 => Linearity::Linear,
                _ => Linearity::NonLinear,
            };
        }
        true
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
        let f = eval_expr(&p.obj_nonlinear, &[0.0, 0.0]);
        assert!((f - 5.0).abs() < 1e-12);
        // f(1,2) = 0
        let f = eval_expr(&p.obj_nonlinear, &[1.0, 2.0]);
        assert!(f.abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_analytic() {
        let p = parse_nl_text(SIMPLE).expect("parse");
        let x = [0.5, 1.0];
        let mut g = [0.0_f64; 2];
        grad_expr(&p.obj_nonlinear, &x, 1.0, &mut g);
        // d/dx0 = 2*(x0-1) = -1.0
        // d/dx1 = 2*(x1-2) = -2.0
        assert!((g[0] - (-1.0)).abs() < 1e-12);
        assert!((g[1] - (-2.0)).abs() < 1e-12);
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
        let f = eval_expr(&p.obj_nonlinear, &[1.0, 2.0]);
        assert!((f - 12.0).abs() < 1e-12, "got {f}");
        // d/dx0 = 2*(x0+x1) + 1 = 7 at (1,2). Same for x1.
        let mut g = [0.0_f64; 2];
        grad_expr(&p.obj_nonlinear, &[1.0, 2.0], 1.0, &mut g);
        assert!((g[0] - 7.0).abs() < 1e-12, "g[0]={}", g[0]);
        assert!((g[1] - 7.0).abs() < 1e-12, "g[1]={}", g[1]);
        // collect_vars reaches into the CSE body and finds {0, 1}.
        let mut vs = BTreeSet::new();
        collect_vars(&p.obj_nonlinear, &mut vs);
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
2 0
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
        prob.con_nonlinear = vec![Expr::Const(0.0), Expr::Const(0.0)];
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
            Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(2))),
            Expr::Const(0.0),
        ];
        prob.g_l = vec![0.0, 0.0];
        prob.g_u = vec![0.0, 0.0];

        let (irow, jcol) = constraint_jacobian_sparsity(&prob);
        // Sorted, deduped per row: row 0 → cols 0,1,2; row 1 → col 2.
        assert_eq!(irow, vec![0, 0, 0, 1]);
        assert_eq!(jcol, vec![0, 1, 2, 2]);
    }
}
