//! Maros-Mészáros QPS (QP-extended MPS) reader — pure Rust, no
//! external dependencies. Loads the standard subset that the
//! Maros-Mészáros test set (Maros & Mészáros 1999, *Optim. Methods
//! Softw.* **11/12**) uses:
//!
//! * `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`,
//!   `QUADOBJ` / `QSECTION`, `ENDATA` section headers.
//! * Free-format token parsing (whitespace-separated). Comment
//!   lines start with `*`.
//! * Row senses: `N` (objective), `L` (≤), `G` (≥), `E` (=).
//! * Bound types: `LO`, `UP`, `FX`, `FR`, `MI`, `PL`. (BV, LI, UI
//!   are MILP markers and are rejected as unsupported.)
//! * Quadratic-objective entries treated as upper-triangular per
//!   the QPS convention; the half-factor `½ xᵀ H x` is implicit
//!   in the QP form, so values are stored as-is.
//!
//! RANGES semantics (per MPS reference):
//!   L row, range r:  bl = rhs − |r|, bu = rhs
//!   G row, range r:  bl = rhs,       bu = rhs + |r|
//!   E row, r > 0:    bl = rhs,       bu = rhs + r
//!   E row, r < 0:    bl = rhs + r,   bu = rhs
//!   E row, r = 0:    bl = bu = rhs   (no-op)
//!
//! Not yet supported (rejected with a clear error):
//! * `OBJSENSE` (assumes minimization).
//! * MIP markers / `INTORG` / `INTEND` / binary variables.
//!
//! These cover the basic QPS profile; loading a real MM instance
//! that uses `RANGES` will require extending the parser. The
//! design-note §8.1 lists Maros-Mészáros as the QP correctness
//! benchmark for Phase 5a; this module is the on-ramp.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use std::collections::HashMap;

/// In-memory representation of a parsed QPS file. Holds owned
/// sparse-triplet data ready to be wrapped into a
/// [`crate::QpProblem`] via [`QpsModel::to_problem_data`].
#[derive(Debug, Clone)]
pub struct QpsModel {
    pub name: String,
    pub n: usize,
    pub m: usize,
    pub var_names: Vec<String>,
    pub row_names: Vec<String>,
    pub g: Vec<f64>,
    pub obj_constant: f64,

    /// Hessian triplet (1-based, lower triangle).
    pub h_irow: Vec<i32>,
    pub h_jcol: Vec<i32>,
    pub h_val: Vec<f64>,

    /// Constraint-Jacobian triplet (1-based).
    pub a_irow: Vec<i32>,
    pub a_jcol: Vec<i32>,
    pub a_val: Vec<f64>,

    pub bl: Vec<f64>,
    pub bu: Vec<f64>,
    pub xl: Vec<f64>,
    pub xu: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Quadobj,
    Endata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSense {
    N,
    L,
    G,
    E,
}

pub fn parse_qps(text: &str) -> Result<QpsModel, String> {
    let mut name = String::new();
    let mut section = Section::None;
    // Whether the active quadratic section uses the *full-matrix*
    // convention (`QMATRIX`, both triangles listed) versus the
    // *single-triangle* convention (`QUADOBJ` / `QSECTION`). For a
    // full-matrix section every off-diagonal H_ij is given twice —
    // once as (i,j) and once as the mirror (j,i). After lower-triangle
    // normalization both collapse onto the same triplet and the
    // evaluator sums them, doubling the off-diagonal. We drop the
    // strict-upper mirror of a full-matrix section so each off-diagonal
    // survives exactly once.
    let mut quad_is_full = false;

    // Row metadata.
    let mut obj_row: Option<String> = None;
    let mut row_sense: HashMap<String, RowSense> = HashMap::new();
    let mut row_names: Vec<String> = Vec::new();
    let mut row_idx: HashMap<String, usize> = HashMap::new();

    // Column metadata + linear coefficients.
    let mut var_names: Vec<String> = Vec::new();
    let mut var_idx: HashMap<String, usize> = HashMap::new();
    let mut g_entries: HashMap<usize, f64> = HashMap::new();
    let mut a_entries: Vec<(usize, usize, f64)> = Vec::new(); // (row, col, val)

    // Right-hand side, ranges, and bounds (variable bounds).
    let mut rhs: HashMap<String, f64> = HashMap::new();
    let mut ranges_map: HashMap<String, f64> = HashMap::new();
    let mut obj_constant: f64 = 0.0;
    let mut bnd_lo: HashMap<usize, f64> = HashMap::new();
    let mut bnd_up: HashMap<usize, f64> = HashMap::new();

    // Quadratic objective entries (upper triangle as in QPS).
    let mut h_entries: Vec<(usize, usize, f64)> = Vec::new();

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        // A line starting at column 0 is a section header (and is
        // not indented). MPS allows the leading column to be a
        // single-letter row-sense indicator, but those lines start
        // with whitespace.
        if !line.starts_with(char::is_whitespace) {
            let mut tokens = line.split_whitespace();
            match tokens.next() {
                Some("NAME") => {
                    name = tokens.next().unwrap_or("").to_string();
                    continue;
                }
                Some("ROWS") => section = Section::Rows,
                Some("COLUMNS") => section = Section::Columns,
                Some("RHS") => section = Section::Rhs,
                Some("RANGES") => section = Section::Ranges,
                Some("BOUNDS") => section = Section::Bounds,
                Some("QUADOBJ") | Some("QSECTION") => {
                    section = Section::Quadobj;
                    quad_is_full = false;
                }
                Some("QMATRIX") => {
                    section = Section::Quadobj;
                    quad_is_full = true;
                }
                Some("ENDATA") => {
                    let _ = Section::Endata;
                    break;
                }
                Some("OBJSENSE") => {
                    return Err(format!(
                        "line {}: OBJSENSE not yet supported (parser assumes MIN)",
                        line_no + 1
                    ));
                }
                Some(other) => {
                    return Err(format!(
                        "line {}: unknown section header `{other}`",
                        line_no + 1
                    ));
                }
                None => continue,
            }
            continue;
        }

        // Indented line — section content.
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match section {
            Section::Rows => {
                if tokens.len() < 2 {
                    return Err(format!("line {}: ROWS entry needs sense+name", line_no + 1));
                }
                let sense = match tokens[0] {
                    "N" => RowSense::N,
                    "L" => RowSense::L,
                    "G" => RowSense::G,
                    "E" => RowSense::E,
                    other => {
                        return Err(format!(
                            "line {}: row sense `{other}` not recognized (need N/L/G/E)",
                            line_no + 1
                        ));
                    }
                };
                let name = tokens[1].to_string();
                if sense == RowSense::N {
                    if obj_row.is_none() {
                        obj_row = Some(name.clone());
                    }
                    row_sense.insert(name, sense);
                } else {
                    if row_idx.contains_key(&name) {
                        return Err(format!("line {}: duplicate row `{name}`", line_no + 1));
                    }
                    row_idx.insert(name.clone(), row_names.len());
                    row_names.push(name.clone());
                    row_sense.insert(name, sense);
                }
            }
            Section::Columns => {
                if tokens.contains(&"'MARKER'") {
                    return Err(format!(
                        "line {}: MIP markers not supported in pounce-qp",
                        line_no + 1
                    ));
                }
                if tokens.len() < 3 {
                    return Err(format!(
                        "line {}: COLUMNS entry needs col+row+value",
                        line_no + 1
                    ));
                }
                let col_name = tokens[0].to_string();
                let col = *var_idx.entry(col_name.clone()).or_insert_with(|| {
                    var_names.push(col_name.clone());
                    var_names.len() - 1
                });

                let mut i = 1;
                while i + 1 < tokens.len() {
                    let row_name = tokens[i];
                    let val: f64 = tokens[i + 1].parse().map_err(|e| {
                        format!("line {}: bad value `{}`: {e}", line_no + 1, tokens[i + 1])
                    })?;
                    if Some(row_name) == obj_row.as_deref() {
                        let prev = g_entries.entry(col).or_insert(0.0);
                        *prev += val;
                    } else if let Some(&row) = row_idx.get(row_name) {
                        a_entries.push((row, col, val));
                    } else {
                        return Err(format!(
                            "line {}: row `{row_name}` not declared",
                            line_no + 1
                        ));
                    }
                    i += 2;
                }
            }
            Section::Rhs => {
                if tokens.len() < 3 {
                    return Err(format!(
                        "line {}: RHS entry needs name+row+value",
                        line_no + 1
                    ));
                }
                // tokens[0] is the RHS-set name (often "RHS" or
                // "B"); ignored.
                let mut i = 1;
                while i + 1 < tokens.len() {
                    let row_name = tokens[i];
                    let val: f64 = tokens[i + 1].parse().map_err(|e| {
                        format!(
                            "line {}: bad RHS value `{}`: {e}",
                            line_no + 1,
                            tokens[i + 1]
                        )
                    })?;
                    if Some(row_name) == obj_row.as_deref() {
                        // RHS on the N row contributes a constant
                        // offset in the objective (with the
                        // standard MPS convention `objective +
                        // const = rhs` ⇒ `const = -rhs`).
                        obj_constant -= val;
                    } else {
                        rhs.insert(row_name.to_string(), val);
                    }
                    i += 2;
                }
            }
            Section::Ranges => {
                if tokens.len() < 3 {
                    return Err(format!(
                        "line {}: RANGES entry needs name+row+value",
                        line_no + 1
                    ));
                }
                // tokens[0] is the ranges-set name (often "RNG");
                // ignored.
                let mut i = 1;
                while i + 1 < tokens.len() {
                    let row_name = tokens[i];
                    let val: f64 = tokens[i + 1].parse().map_err(|e| {
                        format!(
                            "line {}: bad RANGES value `{}`: {e}",
                            line_no + 1,
                            tokens[i + 1]
                        )
                    })?;
                    if !row_idx.contains_key(row_name) {
                        return Err(format!(
                            "line {}: RANGES row `{row_name}` not declared",
                            line_no + 1
                        ));
                    }
                    ranges_map.insert(row_name.to_string(), val);
                    i += 2;
                }
            }
            Section::Bounds => {
                if tokens.len() < 3 {
                    return Err(format!(
                        "line {}: BOUNDS entry needs type+set+col[+value]",
                        line_no + 1
                    ));
                }
                let btype = tokens[0];
                // tokens[1] is the bound-set name; ignored.
                let col_name = tokens[2];
                let col = *var_idx.get(col_name).ok_or_else(|| {
                    format!(
                        "line {}: BOUNDS column `{col_name}` not in COLUMNS",
                        line_no + 1
                    )
                })?;
                let parse_val = |t: &str| -> Result<f64, String> {
                    t.parse::<f64>()
                        .map_err(|e| format!("line {}: bad BOUNDS value `{t}`: {e}", line_no + 1))
                };
                match btype {
                    "LO" => {
                        if tokens.len() < 4 {
                            return Err(format!("line {}: LO needs value", line_no + 1));
                        }
                        bnd_lo.insert(col, parse_val(tokens[3])?);
                    }
                    "UP" => {
                        if tokens.len() < 4 {
                            return Err(format!("line {}: UP needs value", line_no + 1));
                        }
                        bnd_up.insert(col, parse_val(tokens[3])?);
                    }
                    "FX" => {
                        if tokens.len() < 4 {
                            return Err(format!("line {}: FX needs value", line_no + 1));
                        }
                        let v = parse_val(tokens[3])?;
                        bnd_lo.insert(col, v);
                        bnd_up.insert(col, v);
                    }
                    "FR" => {
                        bnd_lo.insert(col, NLP_LOWER_BOUND_INF);
                        bnd_up.insert(col, NLP_UPPER_BOUND_INF);
                    }
                    "MI" => {
                        bnd_lo.insert(col, NLP_LOWER_BOUND_INF);
                    }
                    "PL" => {
                        bnd_up.insert(col, NLP_UPPER_BOUND_INF);
                    }
                    "BV" | "LI" | "UI" => {
                        return Err(format!(
                            "line {}: MILP bound type `{btype}` not supported in pounce-qp",
                            line_no + 1
                        ));
                    }
                    other => {
                        return Err(format!(
                            "line {}: bound type `{other}` not recognized",
                            line_no + 1
                        ));
                    }
                }
            }
            Section::Quadobj => {
                if tokens.len() < 3 {
                    return Err(format!(
                        "line {}: QUADOBJ entry needs col_i+col_j+value",
                        line_no + 1
                    ));
                }
                let i_col = *var_idx.get(tokens[0]).ok_or_else(|| {
                    format!("line {}: QUADOBJ row `{}` unknown", line_no + 1, tokens[0])
                })?;
                let j_col = *var_idx.get(tokens[1]).ok_or_else(|| {
                    format!("line {}: QUADOBJ col `{}` unknown", line_no + 1, tokens[1])
                })?;
                let val: f64 = tokens[2]
                    .parse()
                    .map_err(|e| format!("line {}: bad QUADOBJ value: {e}", line_no + 1))?;
                // In a full-matrix (`QMATRIX`) section the mirror entry
                // (j,i) carries the same value, so keep only the lower
                // triangle and the diagonal; otherwise the off-diagonal
                // is double-counted after normalization below. A
                // single-triangle section lists each pair once, so keep
                // everything.
                if quad_is_full && i_col < j_col {
                    continue;
                }
                h_entries.push((i_col, j_col, val));
            }
            Section::None | Section::Endata => continue,
        }
    }

    // Build dense vectors per the column ordering.
    let n = var_names.len();
    let m = row_names.len();
    let mut g = vec![0.0; n];
    for (col, &v) in &g_entries {
        g[*col] = v;
    }

    // MPS default variable bounds are [0, +∞). Apply that, then
    // override with any BOUNDS entries.
    let mut xl = vec![0.0; n];
    let mut xu = vec![NLP_UPPER_BOUND_INF; n];
    for (&col, &v) in &bnd_lo {
        xl[col] = v;
    }
    for (&col, &v) in &bnd_up {
        xu[col] = v;
    }

    // Constraint bounds derived from row sense + RHS + RANGES.
    // RANGES (per MPS spec):
    //   L row, range r: bl = rhs − |r|, bu = rhs
    //   G row, range r: bl = rhs,       bu = rhs + |r|
    //   E row, r > 0:   bl = rhs,       bu = rhs + r
    //   E row, r < 0:   bl = rhs + r,   bu = rhs
    //   E row, r = 0:   unchanged
    let mut bl = vec![NLP_LOWER_BOUND_INF; m];
    let mut bu = vec![NLP_UPPER_BOUND_INF; m];
    for (row_name, &i) in &row_idx {
        let r = rhs.get(row_name).copied().unwrap_or(0.0);
        match row_sense[row_name] {
            RowSense::L => {
                bu[i] = r;
                if let Some(&rng) = ranges_map.get(row_name) {
                    bl[i] = r - rng.abs();
                }
            }
            RowSense::G => {
                bl[i] = r;
                if let Some(&rng) = ranges_map.get(row_name) {
                    bu[i] = r + rng.abs();
                }
            }
            RowSense::E => {
                bl[i] = r;
                bu[i] = r;
                if let Some(&rng) = ranges_map.get(row_name) {
                    if rng > 0.0 {
                        bu[i] = r + rng;
                    } else if rng < 0.0 {
                        bl[i] = r + rng;
                    }
                }
            }
            RowSense::N => {}
        }
    }

    // Hessian triplets — normalize to lower triangle and convert
    // to 1-based.
    let mut h_irow = Vec::with_capacity(h_entries.len());
    let mut h_jcol = Vec::with_capacity(h_entries.len());
    let mut h_val = Vec::with_capacity(h_entries.len());
    for (i, j, v) in h_entries {
        let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
        h_irow.push((hi + 1) as i32);
        h_jcol.push((lo + 1) as i32);
        h_val.push(v);
    }

    // A triplets — convert to 1-based.
    let mut a_irow = Vec::with_capacity(a_entries.len());
    let mut a_jcol = Vec::with_capacity(a_entries.len());
    let mut a_val = Vec::with_capacity(a_entries.len());
    for (row, col, v) in a_entries {
        a_irow.push((row + 1) as i32);
        a_jcol.push((col + 1) as i32);
        a_val.push(v);
    }

    Ok(QpsModel {
        name,
        n,
        m,
        var_names,
        row_names,
        g,
        obj_constant,
        h_irow,
        h_jcol,
        h_val,
        a_irow,
        a_jcol,
        a_val,
        bl,
        bu,
        xl,
        xu,
    })
}
