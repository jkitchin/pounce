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
//!   `o44` (exp), `o15` (abs), `o41` (sin), `o46` (cos), plus
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
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
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
pub fn read_nl_file(path: &Path) -> Result<NlProblem, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
    parse_nl_text(&txt)
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
            'F' => return Err("F (imported function) segments are not supported".into()),
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
    let nentries: usize = parts[1]
        .parse()
        .map_err(|e| format!("S nentries: {e}"))?;
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
        let first = trimmed
            .chars()
            .next()
            .ok_or("empty header line")?;
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

        // Lines 3..10 are metadata we don't need — skip 8 more lines.
        for _ in 0..8 {
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
        let first = tok
            .chars()
            .next()
            .ok_or("empty expression token")?;
        match first {
            'n' => {
                let v: Number = tok[1..].trim().parse().map_err(|e| format!("n value: {e}"))?;
                Ok(Expr::Const(v))
            }
            'v' => {
                let i: usize = tok[1..].trim().parse().map_err(|e| format!("v index: {e}"))?;
                Ok(self.var_or_cse(i)?)
            }
            'o' => {
                let code: i32 = tok[1..].trim().parse().map_err(|e| format!("opcode: {e}"))?;
                self.parse_opcode(code)
            }
            'f' | 't' | 'u' => {
                Err(format!("unsupported expression token '{tok}'"))
            }
            other => Err(format!("unexpected expression token start '{other}': '{tok}'")),
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
            other => Err(format!("unsupported opcode o{other}")),
        }
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
            }
        }
        Expr::Sum(args) => args.iter().map(|a| eval_expr(a, x)).sum(),
        Expr::Cse(body) => eval_expr(body, x),
    }
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
            };
            grad_expr(a, x, seed * d, grad);
        }
        Expr::Sum(args) => {
            for arg in args {
                grad_expr(arg, x, seed, grad);
            }
        }
        Expr::Cse(body) => grad_expr(body, x, seed, grad),
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
        Expr::Sum(args) => {
            for a in args {
                collect_vars(a, out);
            }
        }
        Expr::Cse(body) => collect_vars(body, out),
    }
}

// --------------------------------------------------------------------
// TNLP wrapper — backed by `Tape` reverse-mode AD for value, gradient,
// Jacobian, and Hessian. Built once at construction; every solve-time
// callback is a tape sweep, no expression-tree recursion.
// --------------------------------------------------------------------

#[derive(Debug)]
pub struct NlTnlp {
    prob: NlProblem,
    /// One tape per top-level summand of the objective's nonlinear
    /// part. Variadic `o54` sums (and nested `o0` adds at the top of
    /// the tree) are split into independent terms so that the
    /// forward-over-reverse Hessian walks each small subgraph
    /// separately instead of the full tape once per variable —
    /// turning a separable obj's O(n²) Hessian cost into O(n).
    obj_tapes: Vec<Tape>,
    /// One tape per top-level summand of each constraint's nonlinear
    /// part (length m). Same separable-Sum split as `obj_tapes`.
    con_tapes: Vec<Vec<Tape>>,
    /// Lower-triangle Hessian sparsity (row >= col), one entry per
    /// structurally nonzero second derivative in the Lagrangian. The
    /// `(row, col) -> values index` map lets each tape's
    /// `hessian_accumulate` scatter into the right slot.
    h_irow: Vec<i32>,
    h_jcol: Vec<i32>,
    hess_map: HashMap<(usize, usize), usize>,
    /// Per-row sorted variable indices for the constraint Jacobian
    /// (union of nonlinear-tape vars and linear-segment vars).
    jac_cols: Vec<Vec<usize>>,
    jac_nnz: usize,
    final_x: Option<Vec<Number>>,
    final_obj: Number,
}

/// Recursively flatten top-level Sum and binary-Add nodes into a list
/// of independent summands. Non-Sum/Add expressions are returned as a
/// single-element vector. This lets `NlTnlp` build one small tape per
/// term so the per-variable Hessian sweep only walks the term that
/// actually depends on that variable.
fn split_top_sums(expr: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    fn go(e: &Expr, out: &mut Vec<Expr>) {
        match e {
            Expr::Sum(terms) => {
                for t in terms {
                    go(t, out);
                }
            }
            Expr::Binary(BinOp::Add, l, r) => {
                go(l, out);
                go(r, out);
            }
            _ => out.push(e.clone()),
        }
    }
    go(expr, &mut out);
    if out.is_empty() {
        out.push(Expr::Const(0.0));
    }
    out
}

impl NlTnlp {
    pub fn new(prob: NlProblem) -> Self {
        // Build tapes up front. The objective and each constraint's
        // nonlinear part are split at top-level `Sum`/`Add` so each
        // summand gets its own small tape — see `split_top_sums`.
        let obj_tapes: Vec<Tape> = split_top_sums(&prob.obj_nonlinear)
            .iter()
            .map(Tape::build)
            .collect();
        let con_tapes: Vec<Vec<Tape>> = (0..prob.m)
            .map(|k| {
                split_top_sums(&prob.con_nonlinear[k])
                    .iter()
                    .map(Tape::build)
                    .collect()
            })
            .collect();

        // Structural Hessian sparsity: union of per-tape sparsities.
        // Linear segments have zero 2nd derivative so they contribute
        // nothing here.
        let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
        for t in &obj_tapes {
            for p in t.hessian_sparsity() {
                pairs.insert(p);
            }
        }
        for ts in &con_tapes {
            for t in ts {
                for p in t.hessian_sparsity() {
                    pairs.insert(p);
                }
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

        // Per-row Jacobian sparsity = union over all summand tapes of
        // their variables, plus the linear-segment vars for that row.
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

        Self {
            prob,
            obj_tapes,
            con_tapes,
            h_irow,
            h_jcol,
            hess_map,
            jac_cols,
            jac_nnz,
            final_x: None,
            final_obj: 0.0,
        }
    }

    pub fn final_x(&self) -> Option<&[Number]> {
        self.final_x.as_deref()
    }

    pub fn final_obj(&self) -> Number {
        self.final_obj
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
        let nl: Number = self.obj_tapes.iter().map(|t| t.eval(x)).sum();
        let lin: Number = self
            .prob
            .obj_linear
            .iter()
            .map(|(i, c)| c * x[*i])
            .sum();
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
            let nl: Number = self.con_tapes[i].iter().map(|t| t.eval(x)).sum();
            let lin: Number = self.prob.con_linear[i]
                .iter()
                .map(|(j, c)| c * x[*j])
                .sum();
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
                let mut row_grad = vec![0.0; n];
                let mut k = 0;
                for i in 0..self.prob.m {
                    // gradient_seed writes only to positions that appear
                    // in the constraint's tape (its Var nodes); linear
                    // contributions touch only `con_linear[i]`. Both
                    // sets are subsets of `jac_cols[i]`, so clearing
                    // just those entries is sufficient and avoids an
                    // O(n) fill per row.
                    for &j in &self.jac_cols[i] {
                        row_grad[j] = 0.0;
                    }
                    for t in &self.con_tapes[i] {
                        t.gradient_seed(xs, 1.0, &mut row_grad);
                    }
                    for &(v, c) in &self.prob.con_linear[i] {
                        row_grad[v] += c;
                    }
                    for &j in &self.jac_cols[i] {
                        values[k] = row_grad[j];
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
                for t in &self.obj_tapes {
                    t.hessian_accumulate(x, obj_seed, &self.hess_map, values);
                }

                if let Some(lam) = lambda {
                    for k in 0..self.prob.m {
                        for t in &self.con_tapes[k] {
                            t.hessian_accumulate(x, lam[k], &self.hess_map, values);
                        }
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
        let r = p.suffixes.var_real.get("sens_state_value_1").expect("var_real");
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
}
