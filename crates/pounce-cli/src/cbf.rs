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
//! - `VAR n k` — `n` scalar variables partitioned into `k` cones, one cone
//!   per following line as `CONE dim` (`F`/`L+`/`L-`/`L=`/`EXP`/`Q`/`QR`).
//! - `CON m k` — `m` scalar constraint rows `Ax + b`, each lying in one of `k`
//!   cones (same syntax). `L=` ⇒ `Ax+b = 0`, `L-` ⇒ `≤ 0`, `L+` ⇒ `≥ 0`.
//! - `OBJACOORD` / `OBJBCOORD` — sparse objective `c` and constant `c₀`.
//! - `ACOORD` / `BCOORD` — sparse `A` (`row col val`) and `b` (`row val`).
//!
//! The problem is `min/max cᵀx + c₀  s.t.  x ∈ K_var,  Ax + b ∈ K_con`.
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
}

/// The CBF cone kinds this reader supports. Unsupported kinds (PSD `DCOORD`,
/// power cones needing a `POWCONES` parameter table) are rejected at parse
/// time with a clear error rather than silently mis-handled.
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
}

impl ConeKind {
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

/// Read a `VAR`/`CON`-style cone partition: a header `total k`, then `k`
/// lines of `CONE dim`. Returns `(total, cones)` and validates the dims sum.
fn parse_cone_block(lines: &mut Lines, what: &str) -> Result<(usize, Vec<ConeDecl>), CbfError> {
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
        let kind =
            ConeKind::parse(tok).ok_or_else(|| CbfError::UnsupportedCone(tok.to_string()))?;
        let dim = parse_usize(t.next().unwrap_or(""), &format!("{what} cone dim"))?;
        if kind == ConeKind::Exp && dim != 3 {
            return Err(CbfError::BadExpDim(dim));
        }
        sum += dim;
        cones.push(ConeDecl { kind, dim });
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
    let mut seen_var = false;

    while let Some(kw) = lines.next() {
        match kw {
            "VER" => {
                lines.require("VER value")?;
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
                let (n, cones) = parse_cone_block(&mut lines, "VAR")?;
                num_var = n;
                var_cones = cones;
                c = vec![0.0; n];
                seen_var = true;
            }
            "CON" => {
                let (m, cones) = parse_cone_block(&mut lines, "CON")?;
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
    /// equalities `Ax = −b`. Exponential triples are reversed to pounce order.
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
                ConeKind::Free => {} // a free constraint row imposes nothing
            }
            r += cone.dim;
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
}
