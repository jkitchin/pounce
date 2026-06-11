//! Reader for the **Conic Benchmark Format** (CBF / `.cbf`), the format the
//! CBLIB conic benchmark library (<https://cblib.zib.de>) ships its instances
//! in, plus a mapping to a pounce conic program.
//!
//! # Format (the subset CBLIB's exponential-cone GPs use)
//!
//! A CBF file is a sequence of keyword blocks, blank-line separated, with `#`
//! comments. The blocks this reader understands:
//!
//! - `VER` — format version (read and ignored).
//! - `OBJSENSE` — `MIN` or `MAX`.
//! - `POWCONES` — power-cone parameter table: each entry's weight vector
//!   `(α₀, α₁)` gives the exponent `α = α₀/(α₀+α₁)`, referenced as `@k:POW`.
//! - `VAR n k` — `n` scalar variables partitioned into `k` cones, one cone
//!   per following line as `CONE dim` (`F`/`L+`/`L-`/`L=`/`EXP`/`Q`/`@k:POW`).
//! - `CON m k` — `m` scalar constraint rows `Ax + b`, each lying in one of `k`
//!   cones (same syntax). `L=` ⇒ `Ax+b = 0`, `L-` ⇒ `≤ 0`, `L+` ⇒ `≥ 0`.
//! - `OBJACOORD` / `OBJBCOORD` — sparse objective `c` and constant `c₀`.
//! - `ACOORD` / `BCOORD` — sparse `A` (`row col val`) and `b` (`row val`).
//! - `PSDCON` + `HCOORD` / `DCOORD` — affine PSD constraints
//!   `D_c + Σ_k x_k H_{c,k} ⪰ 0`, mapped to a `Psd` cone on the slack.
//!
//! The problem is `min/max cᵀx + c₀  s.t.  x ∈ K_var,  Ax + b ∈ K_con`,
//! plus any affine PSD constraints.
//!
//! # Exponential-cone convention
//!
//! CBF's primal exponential cone is `{(u₀,u₁,u₂) : u₀ ≥ u₁·exp(u₂/u₁), u₁>0}`
//! (the **first** coordinate is the bound), whereas pounce's is
//! `{(x,y,z) : z ≥ y·exp(x/y), y>0}` (the **third** is the bound). The triple
//! therefore **reverses**: pounce `(x,y,z) = (u₂, u₁, u₀)`. See
//! `dev-notes/hsde.md` (the CBLIB benchmark-tier plan).

use pounce_convex::{ConeSpec, QpProblem, Triplet};
use std::fmt;

/// A parsed CBF cone declaration: a kind and the number of scalar rows it
/// spans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConeDecl {
    pub kind: ConeKind,
    pub dim: usize,
    /// The power-cone exponent `α ∈ (0, 1)` for [`ConeKind::Pow`]; `None`
    /// for every other kind.
    pub alpha: Option<f64>,
}

/// The CBF cone kinds this reader supports (`F`/`L=`/`L+`/`L-`/`EXP`/`Q`,
/// plus the 3-D power cone `@k:POW` resolved against `POWCONES`). Unsupported
/// kinds (PSD `DCOORD`, the rotated SOC `QR`, dual power cones) are rejected
/// at parse time with a clear error rather than silently mis-handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConeKind {
    /// `F` — free (ℝ): no constraint.
    Free,
    /// `L=` — the zero cone: the rows are equalities.
    Zero,
    /// `L+` — nonnegative orthant.
    Nonneg,
    /// `L-` — nonpositive orthant.
    Nonpos,
    /// `EXP` — the 3-D exponential cone (CBF order; reversed for pounce).
    Exp,
    /// `Q` — the second-order cone.
    SecondOrder,
    /// `@k:POW` — the 3-D power cone, with the exponent `α` resolved from the
    /// referenced `POWCONES` parameter set (stored on [`ConeDecl::alpha`]).
    Pow,
}

impl ConeKind {
    /// Parse a plain (non-parametric) cone token. Parametric cones
    /// (`@k:POW`) are handled by [`parse_cone_token`].
    fn parse(tok: &str) -> Option<ConeKind> {
        Some(match tok {
            "F" => ConeKind::Free,
            "L=" => ConeKind::Zero,
            "L+" => ConeKind::Nonneg,
            "L-" => ConeKind::Nonpos,
            "EXP" => ConeKind::Exp,
            "Q" => ConeKind::SecondOrder,
            _ => return None,
        })
    }
}

/// A parsed CBF instance: the objective, the variable / constraint cone
/// partitions, and the sparse `A`/`b` (and objective `c`/`c₀`).
#[derive(Debug, Clone)]
pub struct CbfModel {
    /// `true` for `OBJSENSE MIN`, `false` for `MAX`.
    pub minimize: bool,
    pub num_var: usize,
    pub var_cones: Vec<ConeDecl>,
    pub num_con: usize,
    pub con_cones: Vec<ConeDecl>,
    /// Objective linear term `c`, dense (length `num_var`).
    pub c: Vec<f64>,
    /// Objective constant `c₀`.
    pub c0: f64,
    /// Constraint matrix `A` as `(row, col, val)` triplets.
    pub a: Vec<(usize, usize, f64)>,
    /// Constraint constant `b`, dense (length `num_con`).
    pub b: Vec<f64>,
    /// Matrix sizes of the affine PSD constraints (`PSDCON`): constraint `c`
    /// asserts `D_c + Σ_k x_k H_{c,k} ⪰ 0` over a `psdcon_dims[c]`-square
    /// matrix.
    pub psdcon_dims: Vec<usize>,
    /// `HCOORD` entries `(con, var, i, j, val)`: `H_{con,var}[i][j] = val`
    /// (lower triangle, `i ≥ j`) — the coefficient of scalar variable `var`
    /// on entry `(i,j)` of PSD constraint `con`.
    pub hcoord: Vec<(usize, usize, usize, usize, f64)>,
    /// `DCOORD` entries `(con, i, j, val)`: `D_con[i][j] = val` (lower
    /// triangle) — the constant term of PSD constraint `con`.
    pub dcoord: Vec<(usize, usize, usize, f64)>,
}

/// A CBF instance mapped to a pounce conic program
/// `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ⪯_K h` (here `P = 0`). The `cones`
/// partition the rows of `G` in order; `obj_constant` (`c₀`, sign-adjusted)
/// is added to `solution.obj` to recover the CBF objective value.
#[derive(Debug, Clone)]
pub struct ConicProgram {
    pub prob: QpProblem,
    pub cones: Vec<ConeSpec>,
    pub obj_constant: f64,
}

impl ConicProgram {
    /// Recover the CBF objective value from a pounce solution objective
    /// `½xᵀPx + cᵀx`. For a `MAX` instance the linear term was negated when
    /// building, so the value is `−pounce_obj + c₀`.
    pub fn cbf_objective(&self, pounce_obj: f64, minimize: bool) -> f64 {
        if minimize {
            pounce_obj + self.obj_constant
        } else {
            -pounce_obj + self.obj_constant
        }
    }
}

/// A CBF parse / mapping failure, with enough context to locate the problem.
#[derive(Debug, Clone, PartialEq)]
pub enum CbfError {
    /// A required section or token was missing / malformed.
    Malformed(String),
    /// A cone kind appeared that this reader does not yet support.
    UnsupportedCone(String),
    /// An exponential cone was declared with a dimension other than 3.
    BadExpDim(usize),
}

impl fmt::Display for CbfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CbfError::Malformed(s) => write!(f, "malformed CBF: {s}"),
            CbfError::UnsupportedCone(s) => write!(f, "unsupported CBF cone '{s}'"),
            CbfError::BadExpDim(d) => write!(f, "EXP cone must have dim 3, got {d}"),
        }
    }
}

impl std::error::Error for CbfError {}

/// A cursor over the meaningful (non-blank, non-comment) lines of a CBF file.
struct Lines<'a> {
    rows: Vec<&'a str>,
    pos: usize,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        let rows = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        Lines { rows, pos: 0 }
    }

    fn next(&mut self) -> Option<&'a str> {
        let row = self.rows.get(self.pos).copied();
        if row.is_some() {
            self.pos += 1;
        }
        row
    }

    fn require(&mut self, what: &str) -> Result<&'a str, CbfError> {
        self.next()
            .ok_or_else(|| CbfError::Malformed(format!("expected {what}, got end of file")))
    }
}

fn parse_usize(tok: &str, what: &str) -> Result<usize, CbfError> {
    tok.parse()
        .map_err(|_| CbfError::Malformed(format!("expected integer for {what}, got '{tok}'")))
}

fn parse_f64(tok: &str, what: &str) -> Result<f64, CbfError> {
    tok.parse()
        .map_err(|_| CbfError::Malformed(format!("expected number for {what}, got '{tok}'")))
}

/// Resolve a cone token to its `(kind, alpha)`. Plain tokens (`F`, `EXP`,
/// …) go through [`ConeKind::parse`]; a parametric `@k:POW` token looks up
/// power-cone parameter set `k` in `pow_params` and resolves the exponent
/// `α = α₀ / (α₀ + α₁)` for the 3-D power cone (parameter vector `(α₀, α₁)`).
fn parse_cone_token(
    tok: &str,
    pow_params: &[Vec<f64>],
) -> Result<(ConeKind, Option<f64>), CbfError> {
    if let Some(rest) = tok.strip_prefix('@') {
        // `@k:KIND` — a reference into a parameter table (only POW today).
        let (idx, kind) = rest
            .split_once(':')
            .ok_or_else(|| CbfError::Malformed(format!("bad parametric cone '{tok}'")))?;
        if kind != "POW" {
            return Err(CbfError::UnsupportedCone(format!("@{idx}:{kind}")));
        }
        let k = parse_usize(idx, "POW reference index")?;
        let params = pow_params
            .get(k)
            .ok_or_else(|| CbfError::Malformed(format!("POW references @{k}, not declared")))?;
        if params.len() != 2 {
            return Err(CbfError::UnsupportedCone(format!(
                "POW with {} parameters (only the 3-D power cone, 2 parameters, is supported)",
                params.len()
            )));
        }
        let alpha = params[0] / (params[0] + params[1]);
        Ok((ConeKind::Pow, Some(alpha)))
    } else {
        let kind =
            ConeKind::parse(tok).ok_or_else(|| CbfError::UnsupportedCone(tok.to_string()))?;
        Ok((kind, None))
    }
}

/// Read a `VAR`/`CON`-style cone partition: a header `total k`, then `k`
/// lines of `CONE dim`. Returns `(total, cones)` and validates the dims sum.
fn parse_cone_block(
    lines: &mut Lines,
    what: &str,
    pow_params: &[Vec<f64>],
) -> Result<(usize, Vec<ConeDecl>), CbfError> {
    let header = lines.require(what)?;
    let mut it = header.split_whitespace();
    let total = parse_usize(it.next().unwrap_or(""), &format!("{what} total"))?;
    let k = parse_usize(it.next().unwrap_or(""), &format!("{what} cone count"))?;
    let mut cones = Vec::with_capacity(k);
    let mut sum = 0;
    for _ in 0..k {
        let line = lines.require(&format!("{what} cone"))?;
        let mut t = line.split_whitespace();
        let tok = t.next().unwrap_or("");
        let (kind, alpha) = parse_cone_token(tok, pow_params)?;
        let dim = parse_usize(t.next().unwrap_or(""), &format!("{what} cone dim"))?;
        if kind == ConeKind::Exp && dim != 3 {
            return Err(CbfError::BadExpDim(dim));
        }
        if kind == ConeKind::Pow && dim != 3 {
            return Err(CbfError::Malformed(format!(
                "{what}: only the 3-D power cone is supported, got POW dim {dim}"
            )));
        }
        sum += dim;
        cones.push(ConeDecl { kind, dim, alpha });
    }
    if sum != total {
        return Err(CbfError::Malformed(format!(
            "{what} cone dims sum to {sum}, header says {total}"
        )));
    }
    Ok((total, cones))
}

/// Parse a CBF instance from its text. Errors on malformed input or a cone
/// kind outside the supported subset.
pub fn parse(text: &str) -> Result<CbfModel, CbfError> {
    let mut lines = Lines::new(text);

    let mut minimize = true;
    let mut num_var = 0usize;
    let mut var_cones = Vec::new();
    let mut num_con = 0usize;
    let mut con_cones = Vec::new();
    let mut c = Vec::new();
    let mut c0 = 0.0;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut pow_params: Vec<Vec<f64>> = Vec::new();
    let mut psdcon_dims: Vec<usize> = Vec::new();
    let mut hcoord: Vec<(usize, usize, usize, usize, f64)> = Vec::new();
    let mut dcoord: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut seen_var = false;

    while let Some(kw) = lines.next() {
        match kw {
            "VER" => {
                lines.require("VER value")?;
            }
            // Power-cone parameter table: `n total`, then for each of the `n`
            // cones a length followed by that many α weights. Must precede the
            // `VAR`/`CON` that reference it via `@k:POW`.
            "POWCONES" => {
                let header = lines.require("POWCONES header")?;
                let mut it = header.split_whitespace();
                let ncones = parse_usize(it.next().unwrap_or(""), "POWCONES count")?;
                let _total = parse_usize(it.next().unwrap_or(""), "POWCONES total")?;
                for _ in 0..ncones {
                    let len = parse_usize(lines.require("POWCONES cone length")?, "POWCONES len")?;
                    let mut params = Vec::with_capacity(len);
                    for _ in 0..len {
                        params.push(parse_f64(
                            lines.require("POWCONES alpha")?,
                            "POWCONES alpha",
                        )?);
                    }
                    pow_params.push(params);
                }
            }
            // Affine PSD constraints: header `count`, then one matrix size
            // per constraint. The constraint `c` is `D_c + Σ_k x_k H_{c,k} ⪰ 0`.
            "PSDCON" => {
                let count = parse_usize(lines.require("PSDCON count")?, "PSDCON count")?;
                for _ in 0..count {
                    psdcon_dims.push(parse_usize(lines.require("PSDCON dim")?, "PSDCON dim")?);
                }
            }
            // Variable coefficient matrices of the PSD constraints.
            "HCOORD" => {
                let nnz = parse_usize(lines.require("HCOORD nnz")?, "HCOORD nnz")?;
                for _ in 0..nnz {
                    let line = lines.require("HCOORD entry")?;
                    let mut t = line.split_whitespace();
                    let con = parse_usize(t.next().unwrap_or(""), "HCOORD con")?;
                    let var = parse_usize(t.next().unwrap_or(""), "HCOORD var")?;
                    let i = parse_usize(t.next().unwrap_or(""), "HCOORD i")?;
                    let j = parse_usize(t.next().unwrap_or(""), "HCOORD j")?;
                    let val = parse_f64(t.next().unwrap_or(""), "HCOORD val")?;
                    hcoord.push((con, var, i, j, val));
                }
            }
            // Constant matrices of the PSD constraints.
            "DCOORD" => {
                let nnz = parse_usize(lines.require("DCOORD nnz")?, "DCOORD nnz")?;
                for _ in 0..nnz {
                    let line = lines.require("DCOORD entry")?;
                    let mut t = line.split_whitespace();
                    let con = parse_usize(t.next().unwrap_or(""), "DCOORD con")?;
                    let i = parse_usize(t.next().unwrap_or(""), "DCOORD i")?;
                    let j = parse_usize(t.next().unwrap_or(""), "DCOORD j")?;
                    let val = parse_f64(t.next().unwrap_or(""), "DCOORD val")?;
                    dcoord.push((con, i, j, val));
                }
            }
            "OBJSENSE" => {
                let s = lines.require("OBJSENSE value")?;
                minimize = match s {
                    "MIN" => true,
                    "MAX" => false,
                    other => {
                        return Err(CbfError::Malformed(format!("bad OBJSENSE '{other}'")));
                    }
                };
            }
            "VAR" => {
                let (n, cones) = parse_cone_block(&mut lines, "VAR", &pow_params)?;
                num_var = n;
                var_cones = cones;
                c = vec![0.0; n];
                seen_var = true;
            }
            "CON" => {
                let (m, cones) = parse_cone_block(&mut lines, "CON", &pow_params)?;
                num_con = m;
                con_cones = cones;
                b = vec![0.0; m];
            }
            "OBJACOORD" => {
                if !seen_var {
                    return Err(CbfError::Malformed("OBJACOORD before VAR".into()));
                }
                let nnz = parse_usize(lines.require("OBJACOORD nnz")?, "OBJACOORD nnz")?;
                for _ in 0..nnz {
                    let line = lines.require("OBJACOORD entry")?;
                    let mut t = line.split_whitespace();
                    let col = parse_usize(t.next().unwrap_or(""), "OBJACOORD col")?;
                    let val = parse_f64(t.next().unwrap_or(""), "OBJACOORD val")?;
                    if col >= num_var {
                        return Err(CbfError::Malformed(format!("OBJACOORD col {col} ≥ n")));
                    }
                    c[col] += val;
                }
            }
            "OBJBCOORD" => {
                c0 = parse_f64(lines.require("OBJBCOORD value")?, "OBJBCOORD")?;
            }
            "ACOORD" => {
                let nnz = parse_usize(lines.require("ACOORD nnz")?, "ACOORD nnz")?;
                a.reserve(nnz);
                for _ in 0..nnz {
                    let line = lines.require("ACOORD entry")?;
                    let mut t = line.split_whitespace();
                    let row = parse_usize(t.next().unwrap_or(""), "ACOORD row")?;
                    let col = parse_usize(t.next().unwrap_or(""), "ACOORD col")?;
                    let val = parse_f64(t.next().unwrap_or(""), "ACOORD val")?;
                    a.push((row, col, val));
                }
            }
            "BCOORD" => {
                if b.is_empty() && num_con > 0 {
                    b = vec![0.0; num_con];
                }
                let nnz = parse_usize(lines.require("BCOORD nnz")?, "BCOORD nnz")?;
                for _ in 0..nnz {
                    let line = lines.require("BCOORD entry")?;
                    let mut t = line.split_whitespace();
                    let row = parse_usize(t.next().unwrap_or(""), "BCOORD row")?;
                    let val = parse_f64(t.next().unwrap_or(""), "BCOORD val")?;
                    if row >= num_con {
                        return Err(CbfError::Malformed(format!("BCOORD row {row} ≥ m")));
                    }
                    b[row] += val;
                }
            }
            // Integrality markers: solve the continuous relaxation, so the
            // index list is read and discarded.
            "INT" => {
                let nnz = parse_usize(lines.require("INT count")?, "INT count")?;
                for _ in 0..nnz {
                    lines.require("INT entry")?;
                }
            }
            other => {
                return Err(CbfError::UnsupportedCone(format!("section '{other}'")));
            }
        }
    }

    if !seen_var {
        return Err(CbfError::Malformed("no VAR section".into()));
    }

    Ok(CbfModel {
        minimize,
        num_var,
        var_cones,
        num_con,
        con_cones,
        c,
        c0,
        a,
        b,
        psdcon_dims,
        hcoord,
        dcoord,
    })
}

impl CbfModel {
    /// Row-major dense access to `A` is avoided; instead group `A` by row so
    /// constraint-cone rows can pull their own coefficients.
    fn rows_of_a(&self) -> Vec<Vec<(usize, f64)>> {
        let mut rows = vec![Vec::new(); self.num_con];
        for &(r, col, val) in &self.a {
            rows[r].push((col, val));
        }
        rows
    }

    /// Map this instance to a pounce conic program. Variable cones become
    /// slack blocks `s = −Gx ∈ K` (a `G = −I` selection, `h = 0`);
    /// constraint cones use `s = h − Gx = Ax + b ∈ K`. `L=` rows become
    /// equalities `Ax = −b`. Exponential triples are reversed, and power
    /// triples rotated, into pounce cone order (see the per-arm comments).
    pub fn to_conic(&self) -> Result<ConicProgram, CbfError> {
        let n = self.num_var;
        let a_rows = self.rows_of_a();

        let mut g: Vec<Triplet> = Vec::new();
        let mut h: Vec<f64> = Vec::new();
        let mut cones: Vec<ConeSpec> = Vec::new();
        let mut a_eq: Vec<Triplet> = Vec::new();
        let mut b_eq: Vec<f64> = Vec::new();

        // Push one cone row whose slack must equal the affine form `(coeffs,
        // constant)`: `s = h − Gx = Σ coeffs·x + constant` ⇒ `G = −coeffs`,
        // `h = constant`.
        let push_row =
            |g: &mut Vec<Triplet>, h: &mut Vec<f64>, coeffs: &[(usize, f64)], constant: f64| {
                let r = h.len();
                for &(col, val) in coeffs {
                    g.push(Triplet::new(r, col, -val));
                }
                h.push(constant);
            };

        // --- Variable cones: the affine form is the variable itself. ---
        let mut v = 0usize; // running scalar-variable index
        for cone in &self.var_cones {
            match cone.kind {
                ConeKind::Free => {}
                ConeKind::Nonneg => {
                    for j in 0..cone.dim {
                        push_row(&mut g, &mut h, &[(v + j, 1.0)], 0.0);
                    }
                    cones.push(ConeSpec::Nonneg(cone.dim));
                }
                ConeKind::Nonpos => {
                    // x ≤ 0 ⇒ slack −x ≥ 0.
                    for j in 0..cone.dim {
                        push_row(&mut g, &mut h, &[(v + j, -1.0)], 0.0);
                    }
                    cones.push(ConeSpec::Nonneg(cone.dim));
                }
                ConeKind::SecondOrder => {
                    for j in 0..cone.dim {
                        push_row(&mut g, &mut h, &[(v + j, 1.0)], 0.0);
                    }
                    cones.push(ConeSpec::SecondOrder(cone.dim));
                }
                ConeKind::Exp => {
                    // Reverse to pounce order (x,y,z) = (u₂,u₁,u₀).
                    for j in (0..3).rev() {
                        push_row(&mut g, &mut h, &[(v + j, 1.0)], 0.0);
                    }
                    cones.push(ConeSpec::Exponential);
                }
                ConeKind::Pow => {
                    // CBF power cone (x₀,x₁,x₂): x₀^β₀·x₁^β₁ ≥ |x₂|. pounce
                    // K_α = {|x| ≤ y^α z^{1−α}} ⇒ (x,y,z) = (x₂, x₀, x₁) with
                    // α = β₀. Emit slack rows in that pounce order.
                    let alpha = cone.alpha.ok_or_else(|| {
                        CbfError::Malformed("POW cone missing its exponent".into())
                    })?;
                    push_row(&mut g, &mut h, &[(v + 2, 1.0)], 0.0); // x ← x₂
                    push_row(&mut g, &mut h, &[(v, 1.0)], 0.0); // y ← x₀
                    push_row(&mut g, &mut h, &[(v + 1, 1.0)], 0.0); // z ← x₁
                    cones.push(ConeSpec::Power(alpha));
                }
                ConeKind::Zero => {
                    // x = 0 — an equality on the variable.
                    for j in 0..cone.dim {
                        a_eq.push(Triplet::new(b_eq.len(), v + j, 1.0));
                        b_eq.push(0.0);
                    }
                }
            }
            v += cone.dim;
        }

        // --- Constraint cones: the affine form is row `r` of `Ax + b`. ---
        let mut r = 0usize; // running constraint-row index
        for cone in &self.con_cones {
            match cone.kind {
                ConeKind::Zero => {
                    // Ax + b = 0 ⇒ Ax = −b.
                    for i in 0..cone.dim {
                        let row = r + i;
                        for &(col, val) in &a_rows[row] {
                            a_eq.push(Triplet::new(b_eq.len(), col, val));
                        }
                        b_eq.push(-self.b[row]);
                    }
                }
                ConeKind::Nonneg => {
                    // Ax + b ≥ 0 ⇒ slack = Ax + b ≥ 0.
                    for i in 0..cone.dim {
                        let row = r + i;
                        push_row(&mut g, &mut h, &a_rows[row], self.b[row]);
                    }
                    cones.push(ConeSpec::Nonneg(cone.dim));
                }
                ConeKind::Nonpos => {
                    // Ax + b ≤ 0 ⇒ slack = −(Ax + b) ≥ 0.
                    for i in 0..cone.dim {
                        let row = r + i;
                        let neg: Vec<(usize, f64)> =
                            a_rows[row].iter().map(|&(c, v)| (c, -v)).collect();
                        push_row(&mut g, &mut h, &neg, -self.b[row]);
                    }
                    cones.push(ConeSpec::Nonneg(cone.dim));
                }
                ConeKind::SecondOrder => {
                    for i in 0..cone.dim {
                        let row = r + i;
                        push_row(&mut g, &mut h, &a_rows[row], self.b[row]);
                    }
                    cones.push(ConeSpec::SecondOrder(cone.dim));
                }
                ConeKind::Exp => {
                    // Slack must be ((Ax+b)₂, (Ax+b)₁, (Ax+b)₀) — reversed.
                    for i in (0..3).rev() {
                        let row = r + i;
                        push_row(&mut g, &mut h, &a_rows[row], self.b[row]);
                    }
                    cones.push(ConeSpec::Exponential);
                }
                ConeKind::Pow => {
                    // pounce (x,y,z) = ((Ax+b)₂, (Ax+b)₀, (Ax+b)₁), α = β₀.
                    let alpha = cone.alpha.ok_or_else(|| {
                        CbfError::Malformed("POW cone missing its exponent".into())
                    })?;
                    for &i in &[2usize, 0, 1] {
                        let row = r + i;
                        push_row(&mut g, &mut h, &a_rows[row], self.b[row]);
                    }
                    cones.push(ConeSpec::Power(alpha));
                }
                ConeKind::Free => {} // a free constraint row imposes nothing
            }
            r += cone.dim;
        }

        // --- Affine PSD constraints (PSDCON): D_c + Σ_k x_k H_{c,k} ⪰ 0. ---
        // The slack svec entry (i,j) is `D[i][j] + Σ_k x_k H_k[i][j]`, scaled
        // by √2 off the diagonal so smat(s) reconstructs the matrix. Appended
        // after the VAR/CON cone rows as Psd blocks.
        if !self.psdcon_dims.is_empty() {
            use std::collections::HashMap;
            let r2 = std::f64::consts::SQRT_2;
            let mut h_by: HashMap<(usize, usize, usize), Vec<(usize, f64)>> = HashMap::new();
            for &(con, var, i, j, val) in &self.hcoord {
                h_by.entry((con, i, j)).or_default().push((var, val));
            }
            let mut d_by: HashMap<(usize, usize, usize), f64> = HashMap::new();
            for &(con, i, j, val) in &self.dcoord {
                *d_by.entry((con, i, j)).or_insert(0.0) += val;
            }
            for (con, &dim) in self.psdcon_dims.iter().enumerate() {
                // svec order: column by column, lower triangle (j ≤ i).
                for j in 0..dim {
                    for i in j..dim {
                        let scale = if i == j { 1.0 } else { r2 };
                        let constant = scale * d_by.get(&(con, i, j)).copied().unwrap_or(0.0);
                        let coeffs: Vec<(usize, f64)> = h_by
                            .get(&(con, i, j))
                            .map(|v| v.iter().map(|&(var, val)| (var, scale * val)).collect())
                            .unwrap_or_default();
                        push_row(&mut g, &mut h, &coeffs, constant);
                    }
                }
                cones.push(ConeSpec::Psd(dim));
            }
        }

        // Objective: minimize cᵀx (negate for MAX), constant carried out.
        let c: Vec<f64> = if self.minimize {
            self.c.clone()
        } else {
            self.c.iter().map(|v| -v).collect()
        };

        let prob = QpProblem {
            n,
            p_lower: Vec::new(),
            c,
            a: a_eq,
            b: b_eq,
            g,
            h,
            lb: Vec::new(),
            ub: Vec::new(),
        };
        Ok(ConicProgram {
            prob,
            cones,
            obj_constant: self.c0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_GP: &str = "\
VER
2

OBJSENSE
MIN

VAR
4 2
F 1
EXP 3

CON
1 1
L= 1

OBJACOORD
1
0 1.0

ACOORD
2
0 1 1.0
0 3 -1.0

BCOORD
1
0 -2.0
";

    #[test]
    fn parses_sections() {
        let m = parse(TINY_GP).unwrap();
        assert!(m.minimize);
        assert_eq!(m.num_var, 4);
        assert_eq!(m.var_cones.len(), 2);
        assert_eq!(m.var_cones[0].kind, ConeKind::Free);
        assert_eq!(m.var_cones[1].kind, ConeKind::Exp);
        assert_eq!(m.num_con, 1);
        assert_eq!(m.con_cones[0].kind, ConeKind::Zero);
        assert_eq!(m.c, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.b, vec![-2.0]);
        assert_eq!(m.a.len(), 2);
    }

    #[test]
    fn rejects_bad_exp_dim() {
        let bad = TINY_GP.replace("EXP 3", "EXP 2");
        assert!(matches!(parse(&bad), Err(CbfError::BadExpDim(2))));
    }

    #[test]
    fn rejects_unsupported_cone() {
        let bad = TINY_GP.replace("EXP 3", "POW 3");
        assert!(matches!(parse(&bad), Err(CbfError::UnsupportedCone(_))));
    }

    #[test]
    fn cone_dim_sum_is_checked() {
        let bad = TINY_GP.replace("4 2", "5 2");
        assert!(matches!(parse(&bad), Err(CbfError::Malformed(_))));
    }

    #[test]
    fn to_conic_builds_exp_and_equality() {
        let m = parse(TINY_GP).unwrap();
        let cp = m.to_conic().unwrap();
        // One exp cone over vars {1,2,3}; the L= row is an equality.
        assert_eq!(cp.cones, vec![ConeSpec::Exponential]);
        assert_eq!(cp.prob.m_eq(), 1); // the L= constraint
        assert_eq!(cp.prob.m_ineq(), 3); // the exp cone's 3 rows
        assert_eq!(cp.obj_constant, 0.0);
        // The exp rows reverse CBF (vars 1,2,3) to pounce order (3,2,1):
        // G row 0 selects var 3, row 1 var 2, row 2 var 1 (each with −1·−? ).
        // push_row uses G = −coeffs with coeff +1 ⇒ G entry −1.
        let row0: Vec<_> = cp.prob.g.iter().filter(|t| t.row == 0).collect();
        assert_eq!(row0.len(), 1);
        assert_eq!(row0[0].col, 3);
    }

    const TINY_POW: &str = "\
VER
2

OBJSENSE
MAX

POWCONES
1 2
2
3.0
1.0

VAR
3 1
@0:POW 3

CON
0 0

OBJACOORD
1
2 1.0
";

    #[test]
    fn parses_powcones_and_resolves_alpha() {
        let m = parse(TINY_POW).unwrap();
        assert_eq!(m.var_cones.len(), 1);
        assert_eq!(m.var_cones[0].kind, ConeKind::Pow);
        // α = α₀/(α₀+α₁) = 3/(3+1) = 0.75.
        let a = m.var_cones[0].alpha.unwrap();
        assert!((a - 0.75).abs() < 1e-12, "alpha {a}");
    }

    #[test]
    fn to_conic_builds_power_cone_with_permutation() {
        let m = parse(TINY_POW).unwrap();
        let cp = m.to_conic().unwrap();
        assert_eq!(cp.cones, vec![ConeSpec::Power(0.75)]);
        assert_eq!(cp.prob.m_ineq(), 3); // the power cone's 3 rows
                                         // pounce (x,y,z) = (CBF x₂, x₀, x₁): row 0 selects var 2.
        let row0: Vec<_> = cp.prob.g.iter().filter(|t| t.row == 0).collect();
        assert_eq!(row0[0].col, 2);
        let row1: Vec<_> = cp.prob.g.iter().filter(|t| t.row == 1).collect();
        assert_eq!(row1[0].col, 0);
        let row2: Vec<_> = cp.prob.g.iter().filter(|t| t.row == 2).collect();
        assert_eq!(row2[0].col, 1);
    }

    #[test]
    fn pow_reference_to_undeclared_set_errors() {
        let bad = TINY_POW.replace("@0:POW", "@5:POW");
        assert!(matches!(parse(&bad), Err(CbfError::Malformed(_))));
    }

    const TINY_SDP: &str = "\
VER
2

OBJSENSE
MAX

VAR
1 1
F 1

PSDCON
1
2

OBJACOORD
1
0 1.0

HCOORD
2
0 0 0 0 -1.0
0 0 1 1 -1.0

DCOORD
2
0 0 0 2.0
0 1 1 5.0
";

    #[test]
    fn parses_psdcon_hcoord_dcoord() {
        let m = parse(TINY_SDP).unwrap();
        assert_eq!(m.psdcon_dims, vec![2]);
        assert_eq!(m.hcoord.len(), 2);
        assert_eq!(m.dcoord.len(), 2);
    }

    #[test]
    fn to_conic_builds_psd_constraint() {
        let m = parse(TINY_SDP).unwrap();
        let cp = m.to_conic().unwrap();
        // One affine PSD constraint of size 2 → a Psd(2) cone over 3 rows.
        assert_eq!(cp.cones, vec![ConeSpec::Psd(2)]);
        assert_eq!(cp.prob.m_ineq(), 3);
        // s = svec(M − λI) = [2 − λ, 0, 5 − λ]: h = [2, 0, 5] and the diagonal
        // svec rows (0 and 2) carry +λ from G (push_row negates H = −1).
        assert_eq!(cp.prob.h, vec![2.0, 0.0, 5.0]);
        let row0: Vec<_> = cp.prob.g.iter().filter(|t| t.row == 0).collect();
        assert_eq!(row0.len(), 1);
        assert!((row0[0].val - 1.0).abs() < 1e-12); // −H = −(−1) = +1
    }
}
