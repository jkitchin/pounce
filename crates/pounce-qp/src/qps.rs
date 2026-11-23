//! MPS / QPS (QP-extended MPS) reader — pure Rust, no external
//! dependencies. Reads LPs and convex QPs in the form the Maros-Mészáros
//! QP set (Maros & Mészáros 1999, *Optim. Methods Softw.* **11/12**), the
//! NETLIB LP set and the Kennington LP set are distributed in:
//!
//! ```text
//! min/max  ½ xᵀHx + gᵀx + c   s.t.  bl ≤ Ax ≤ bu,  xl ≤ x ≤ xu
//! ```
//!
//! # Sections
//!
//! `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `QUADOBJ` /
//! `QSECTION` / `QMATRIX`, `OBJSENSE`, `ENDATA`. Lines starting with `*`
//! are comments.
//!
//! * **Rows:** `N` (objective — the first `N` row; any later `N` row is a
//!   free row with no bounds, and its entries are dropped), `L` (≤), `G`
//!   (≥), `E` (=).
//! * **RHS on the objective row** is the negative of the objective
//!   constant: `c = −rhs`, the MPS / CPLEX / HiGHS convention.
//! * **RANGES** (per the MPS reference):
//!   - `L` row, range r: `bl = rhs − |r|`, `bu = rhs`
//!   - `G` row, range r: `bl = rhs`, `bu = rhs + |r|`
//!   - `E` row, r > 0: `bl = rhs`, `bu = rhs + r`
//!   - `E` row, r < 0: `bl = rhs + r`, `bu = rhs`
//!   - `E` row, r = 0: `bl = bu = rhs`
//! * **Bounds:** `LO`, `UP`, `FX`, `FR`, `MI`, `PL`. The default box is
//!   `[0, +∞)`. A negative `UP` on a column with no explicit lower bound
//!   also sets the lower bound to −∞ (the CPLEX / Gurobi rule; without it
//!   the box would be the empty `[0, u]`), and adds a warning. HiGHS 1.16
//!   keeps the empty box instead.
//! * **Quadratic objective:** `QUADOBJ` / `QSECTION` list one triangle of
//!   the symmetric `H`; `QMATRIX` lists both, and the strict-upper mirror
//!   is dropped so each off-diagonal is kept once. Either way the stored
//!   triplets are normalized to the **lower** triangle, and the objective
//!   is `½ xᵀHx` (the ½ is implicit, so values are stored as written).
//! * **`OBJSENSE`** `MIN` / `MAX` (or `MINIMIZE` / `MAXIMIZE`), on the
//!   header line or on the indented line after it; recorded in
//!   [`QpsModel::minimize`]. The data is stored as written, so a consumer
//!   of a maximize model negates the objective itself.
//!
//! # Free and fixed format
//!
//! The file is first read as **free** MPS (whitespace-separated fields).
//! The set name in `RHS`, `RANGES` and `BOUNDS` is optional there, as it
//! is in fixed format when its columns are blank: an `RHS` / `RANGES` line
//! with an even number of fields has none, and so does a `BOUNDS` line with
//! three fields for a type that takes a value (`LO`, `UP`, `FX`) or two for
//! one that does not (`FR`, `MI`, `PL`).
//!
//! If the free reading fails, the file is read again as **fixed** MPS,
//! where each field sits in its own columns (2–3, 5–12, 15–22, 25–36,
//! 40–47, 50–61) and a name may contain spaces (NETLIB's FORPLAN has rows
//! named `DEDO3 1R`). Every section checks its field count, so a fixed
//! file with spaced names cannot be read as free MPS without an error; and
//! the fixed reading rejects text between the field columns, so a free
//! file is never misread as fixed.
//!
//! # Duplicates
//!
//! Handled as HiGHS does, so a model read here matches one read through
//! HiGHS (as `benchmarks/*/mps_to_nl.py` do):
//!
//! * a repeated `COLUMNS` coefficient (same column and row, objective row
//!   included), `RHS` value or `RANGES` value: the **first** is kept, the
//!   rest ignored;
//! * a repeated `QUADOBJ` entry (same unordered pair): **summed**;
//! * `BOUNDS` are statements applied in order, so a later bound on the
//!   same column replaces an earlier one, as in every MPS reader.
//!
//! Each ignored or summed duplicate is counted in
//! [`QpsModel::warnings`] rather than passing silently.
//!
//! # Rejected
//!
//! Integer data — `MARKER` / `INTORG` / `INTEND` lines and the `BV`,
//! `LI`, `UI` bound types — with a clear error: this is a continuous
//! reader.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use std::collections::{HashMap, HashSet};

/// In-memory representation of a parsed QPS file, as owned sparse
/// triplets.
#[derive(Debug, Clone)]
pub struct QpsModel {
    pub name: String,
    /// `false` when an `OBJSENSE` section asked for `MAX`. The objective
    /// data (`g`, `obj_constant`, the Hessian) is the file's, unnegated:
    /// the model is `max ½xᵀHx + gᵀx + c` in that case.
    pub minimize: bool,
    pub n: usize,
    pub m: usize,
    pub var_names: Vec<String>,
    pub row_names: Vec<String>,
    pub g: Vec<f64>,
    pub obj_constant: f64,

    /// Hessian triplet (1-based, lower triangle). May hold the same
    /// position more than once; the entries are summed.
    pub h_irow: Vec<i32>,
    pub h_jcol: Vec<i32>,
    pub h_val: Vec<f64>,

    /// Constraint-Jacobian triplet (1-based), one entry per position.
    pub a_irow: Vec<i32>,
    pub a_jcol: Vec<i32>,
    pub a_val: Vec<f64>,

    pub bl: Vec<f64>,
    pub bu: Vec<f64>,
    pub xl: Vec<f64>,
    pub xu: Vec<f64>,

    /// Things the reader resolved by a convention rather than by what the
    /// file says outright: ignored or summed duplicates, a negative `UP`
    /// that freed a lower bound, and that the file was read as fixed
    /// format. Empty for a file that needed none. Callers decide whether
    /// to show them.
    pub warnings: Vec<String>,
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
    Objsense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSense {
    N,
    L,
    G,
    E,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Free,
    Fixed,
}

/// Counts duplicates of one kind and remembers the first, for one
/// warning line per kind instead of one per entry.
#[derive(Default)]
struct DupLog {
    count: usize,
    first: Option<String>,
}

impl DupLog {
    fn note(&mut self, what: impl FnOnce() -> String) {
        self.count += 1;
        if self.first.is_none() {
            self.first = Some(what());
        }
    }

    fn report(&self, warnings: &mut Vec<String>, kind: &str, action: &str) {
        if let Some(first) = &self.first {
            warnings.push(format!(
                "{} duplicate {kind} {action} (first: {first})",
                self.count
            ));
        }
    }
}

/// Parse an MPS / QPS file: free format first, fixed format if that
/// fails. See the module docs for the conventions applied.
pub fn parse_qps(text: &str) -> Result<QpsModel, String> {
    match parse_with(text, Format::Free) {
        Ok(m) => Ok(m),
        Err(free_err) => match parse_with(text, Format::Fixed) {
            Ok(mut m) => {
                m.warnings.insert(
                    0,
                    format!("read as fixed-format MPS (as free format: {free_err})"),
                );
                Ok(m)
            }
            Err(fixed_err) => Err(format!("{free_err} (as fixed-format MPS: {fixed_err})")),
        },
    }
}

/// Bound types that take a value. `FR`, `MI`, `PL` do not.
fn bound_takes_value(t: &str) -> bool {
    matches!(t, "LO" | "UP" | "FX" | "LI" | "UI" | "SC")
}

/// Split a free-format data line into fields, making an omitted set name
/// in `RHS` / `RANGES` / `BOUNDS` explicit as `""` so every section sees
/// the same layout whichever format the file is in.
fn free_fields(line: &str, section: Section) -> Vec<String> {
    let mut f: Vec<String> = line.split_whitespace().map(str::to_string).collect();
    match section {
        Section::Rhs | Section::Ranges if f.len() % 2 == 0 => f.insert(0, String::new()),
        Section::Bounds if !f.is_empty() => {
            let no_set = if bound_takes_value(&f[0]) {
                f.len() == 3
            } else {
                f.len() == 2
            };
            if no_set {
                f.insert(1, String::new());
            }
        }
        _ => {}
    }
    f
}

/// Split a fixed-format data line into its fields (1-based columns 2–3,
/// 5–12, 15–22, 25–36, 40–47, 50–61), trimmed. Text in the columns
/// between fields is an error: it means the line is not fixed format.
fn fixed_fields(line: &str, section: Section) -> Result<Vec<String>, String> {
    const FIELDS: [(usize, usize); 6] = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)];
    const GAPS: [(usize, usize); 5] = [(3, 4), (12, 14), (22, 24), (36, 39), (47, 49)];
    if !line.is_ascii() {
        return Err("non-ASCII text in a fixed-format line".to_string());
    }
    let col = |a: usize, b: usize| -> &str {
        line.get(a.min(line.len())..b.min(line.len())).unwrap_or("")
    };
    for (a, b) in GAPS {
        if !col(a, b).trim().is_empty() {
            return Err(format!(
                "text in columns {}-{}, between fixed-format fields",
                a + 1,
                b
            ));
        }
    }
    if line.len() > 61 && !line[61..].trim().is_empty() {
        return Err("text past column 61".to_string());
    }
    let f: Vec<String> = FIELDS
        .iter()
        .map(|&(a, b)| col(a, b).trim().to_string())
        .collect();
    let mut out: Vec<String> = match section {
        Section::Rows => vec![f[0].clone(), f[1].clone()],
        Section::Bounds => vec![f[0].clone(), f[1].clone(), f[2].clone(), f[3].clone()],
        Section::Objsense => vec![line.trim().to_string()],
        // COLUMNS, RHS, RANGES, QUADOBJ: field 1 is unused.
        _ => {
            if !f[0].is_empty() {
                return Err(format!("unexpected `{}` in columns 2-3", f[0]));
            }
            f[1..].to_vec()
        }
    };
    // Drop trailing empty fields, but keep a blank set name in place.
    let keep = match section {
        Section::Rhs | Section::Ranges => 1,
        Section::Bounds => 2,
        _ => 0,
    };
    while out.len() > keep && out.last().is_some_and(String::is_empty) {
        out.pop();
    }
    Ok(out)
}

fn parse_with(text: &str, format: Format) -> Result<QpsModel, String> {
    let mut name = String::new();
    let mut section = Section::None;
    let mut minimize = true;
    let parse_sense = |tok: &str, line_no: usize| -> Result<bool, String> {
        match tok.to_ascii_uppercase().as_str() {
            "MIN" | "MINIMIZE" => Ok(true),
            "MAX" | "MAXIMIZE" => Ok(false),
            other => Err(format!(
                "line {}: OBJSENSE `{other}` not recognized (need MIN or MAX)",
                line_no + 1
            )),
        }
    };
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
    let mut a_seen: HashSet<(usize, usize)> = HashSet::new();

    // Right-hand side, ranges, and bounds (variable bounds).
    let mut rhs: HashMap<String, f64> = HashMap::new();
    let mut ranges_map: HashMap<String, f64> = HashMap::new();
    let mut obj_rhs: Option<f64> = None;
    let mut bnd_lo: HashMap<usize, f64> = HashMap::new();
    let mut bnd_up: HashMap<usize, f64> = HashMap::new();
    // Columns whose upper bound was last set by `UP` (as opposed to `FX`,
    // `MI`, `PL`, `FR`), for the negative-`UP` rule.
    let mut up_by_up: HashSet<usize> = HashSet::new();

    // Quadratic objective entries, as written.
    let mut h_entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut h_seen: HashSet<(usize, usize)> = HashSet::new();

    let mut dup_coef = DupLog::default();
    let mut dup_rhs = DupLog::default();
    let mut dup_range = DupLog::default();
    let mut dup_quad = DupLog::default();
    let mut free_row_entries = 0usize;

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
                Some("ENDATA") => break,
                Some("OBJSENSE") => {
                    // Free MPS puts the sense on the header line; fixed
                    // MPS on the indented line that follows.
                    if let Some(tok) = tokens.next() {
                        minimize = parse_sense(tok, line_no)?;
                        section = Section::None;
                    } else {
                        section = Section::Objsense;
                    }
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
        let fields: Vec<String> = match format {
            Format::Free => free_fields(line, section),
            Format::Fixed => {
                fixed_fields(line, section).map_err(|e| format!("line {}: {e}", line_no + 1))?
            }
        };
        let tokens: Vec<&str> = fields.iter().map(String::as_str).collect();
        if tokens.iter().all(|t| t.is_empty()) {
            continue;
        }
        let count_err = |what: &str| -> String {
            format!(
                "line {}: {what}, got {} field(s): {:?}",
                line_no + 1,
                tokens.len(),
                tokens
            )
        };
        match section {
            Section::Rows => {
                if tokens.len() != 2 {
                    return Err(count_err("ROWS entry needs sense+name"));
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
                if row_sense.contains_key(&name) {
                    return Err(format!("line {}: duplicate row `{name}`", line_no + 1));
                }
                if sense == RowSense::N {
                    if obj_row.is_none() {
                        obj_row = Some(name.clone());
                    }
                } else {
                    row_idx.insert(name.clone(), row_names.len());
                    row_names.push(name.clone());
                }
                row_sense.insert(name, sense);
            }
            Section::Columns => {
                if tokens.contains(&"'MARKER'") {
                    return Err(format!(
                        "line {}: MIP markers not supported in pounce-qp",
                        line_no + 1
                    ));
                }
                if tokens.len() != 3 && tokens.len() != 5 {
                    return Err(count_err(
                        "COLUMNS entry needs col + one or two (row, value) pairs",
                    ));
                }
                let col_name = tokens[0].to_string();
                let col = *var_idx.entry(col_name.clone()).or_insert_with(|| {
                    var_names.push(col_name.clone());
                    var_names.len() - 1
                });

                for pair in tokens[1..].chunks(2) {
                    let row_name = pair[0];
                    let val: f64 = pair[1].parse().map_err(|e| {
                        format!("line {}: bad value `{}`: {e}", line_no + 1, pair[1])
                    })?;
                    if Some(row_name) == obj_row.as_deref() {
                        if g_entries.contains_key(&col) {
                            dup_coef
                                .note(|| format!("column `{col_name}` in objective `{row_name}`"));
                        } else {
                            g_entries.insert(col, val);
                        }
                    } else if let Some(&row) = row_idx.get(row_name) {
                        if a_seen.insert((row, col)) {
                            a_entries.push((row, col, val));
                        } else {
                            dup_coef.note(|| format!("column `{col_name}` in row `{row_name}`"));
                        }
                    } else if row_sense.get(row_name) == Some(&RowSense::N) {
                        // A second `N` row: a free row with no bounds,
                        // which constrains nothing.
                        free_row_entries += 1;
                    } else {
                        return Err(format!(
                            "line {}: row `{row_name}` not declared",
                            line_no + 1
                        ));
                    }
                }
            }
            Section::Rhs => {
                if tokens.len() != 3 && tokens.len() != 5 {
                    return Err(count_err(
                        "RHS entry needs [set] + one or two (row, value) pairs",
                    ));
                }
                // tokens[0] is the RHS-set name (often "RHS" or "B", or
                // blank); ignored.
                for pair in tokens[1..].chunks(2) {
                    let row_name = pair[0];
                    let val: f64 = pair[1].parse().map_err(|e| {
                        format!("line {}: bad RHS value `{}`: {e}", line_no + 1, pair[1])
                    })?;
                    if Some(row_name) == obj_row.as_deref() {
                        // RHS on the N row is the negative of a constant
                        // in the objective (`objective + const = rhs` ⇒
                        // `const = -rhs`).
                        if obj_rhs.is_some() {
                            dup_rhs.note(|| format!("objective row `{row_name}`"));
                        } else {
                            obj_rhs = Some(val);
                        }
                    } else if rhs.contains_key(row_name) {
                        dup_rhs.note(|| format!("row `{row_name}`"));
                    } else {
                        rhs.insert(row_name.to_string(), val);
                    }
                }
            }
            Section::Ranges => {
                if tokens.len() != 3 && tokens.len() != 5 {
                    return Err(count_err(
                        "RANGES entry needs [set] + one or two (row, value) pairs",
                    ));
                }
                // tokens[0] is the ranges-set name (often "RNG", or
                // blank); ignored.
                for pair in tokens[1..].chunks(2) {
                    let row_name = pair[0];
                    let val: f64 = pair[1].parse().map_err(|e| {
                        format!("line {}: bad RANGES value `{}`: {e}", line_no + 1, pair[1])
                    })?;
                    if !row_idx.contains_key(row_name) {
                        return Err(format!(
                            "line {}: RANGES row `{row_name}` not declared",
                            line_no + 1
                        ));
                    }
                    if ranges_map.contains_key(row_name) {
                        dup_range.note(|| format!("row `{row_name}`"));
                    } else {
                        ranges_map.insert(row_name.to_string(), val);
                    }
                }
            }
            Section::Bounds => {
                let btype = tokens[0];
                let needs_value = bound_takes_value(btype);
                if tokens.len() != 4 && !(tokens.len() == 3 && !needs_value) {
                    return Err(count_err(
                        "BOUNDS entry needs type + [set] + column (+ value)",
                    ));
                }
                // tokens[1] is the bound-set name (or blank); ignored.
                let col_name = tokens[2];
                let col = *var_idx.get(col_name).ok_or_else(|| {
                    format!(
                        "line {}: BOUNDS column `{col_name}` not in COLUMNS",
                        line_no + 1
                    )
                })?;
                let value = || -> Result<f64, String> {
                    let t = tokens[3];
                    t.parse::<f64>()
                        .map_err(|e| format!("line {}: bad BOUNDS value `{t}`: {e}", line_no + 1))
                };
                match btype {
                    "LO" => {
                        bnd_lo.insert(col, value()?);
                    }
                    "UP" => {
                        bnd_up.insert(col, value()?);
                        up_by_up.insert(col);
                    }
                    "FX" => {
                        let v = value()?;
                        bnd_lo.insert(col, v);
                        bnd_up.insert(col, v);
                        up_by_up.remove(&col);
                    }
                    "FR" => {
                        bnd_lo.insert(col, NLP_LOWER_BOUND_INF);
                        bnd_up.insert(col, NLP_UPPER_BOUND_INF);
                        up_by_up.remove(&col);
                    }
                    "MI" => {
                        bnd_lo.insert(col, NLP_LOWER_BOUND_INF);
                    }
                    "PL" => {
                        bnd_up.insert(col, NLP_UPPER_BOUND_INF);
                        up_by_up.remove(&col);
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
                if tokens.len() != 3 {
                    return Err(count_err("QUADOBJ entry needs col_i + col_j + value"));
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
                let key = (i_col.max(j_col), i_col.min(j_col));
                if !h_seen.insert(key) {
                    dup_quad.note(|| format!("({}, {})", tokens[0], tokens[1]));
                }
                h_entries.push((i_col, j_col, val));
            }
            Section::Objsense => {
                if tokens.len() != 1 {
                    return Err(count_err("OBJSENSE entry needs MIN or MAX"));
                }
                minimize = parse_sense(tokens[0], line_no)?;
            }
            Section::None => {
                return Err(format!(
                    "line {}: data line before any section header",
                    line_no + 1
                ));
            }
        }
    }

    let mut warnings: Vec<String> = Vec::new();
    dup_coef.report(
        &mut warnings,
        "COLUMNS coefficient(s)",
        "ignored, first kept",
    );
    dup_rhs.report(&mut warnings, "RHS value(s)", "ignored, first kept");
    dup_range.report(&mut warnings, "RANGES value(s)", "ignored, first kept");
    dup_quad.report(&mut warnings, "QUADOBJ entr(ies)", "summed");
    if free_row_entries > 0 {
        warnings.push(format!(
            "{free_row_entries} COLUMNS entr(ies) on free rows (N rows after the first) dropped"
        ));
    }

    // Build dense vectors per the column ordering.
    let n = var_names.len();
    let m = row_names.len();
    let mut g = vec![0.0; n];
    for (col, &v) in &g_entries {
        g[*col] = v;
    }
    let obj_constant = obj_rhs.map_or(0.0, |r| 0.0 - r);

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
    // Negative `UP` with no lower bound given: the lower bound becomes
    // −∞ rather than leaving the empty box [0, u].
    let mut freed: Vec<usize> = up_by_up
        .iter()
        .copied()
        .filter(|c| xu[*c] < 0.0 && !bnd_lo.contains_key(c))
        .collect();
    freed.sort_unstable();
    for &c in &freed {
        xl[c] = NLP_LOWER_BOUND_INF;
    }
    if let Some(&c) = freed.first() {
        warnings.push(format!(
            "{} column(s) with a negative UP bound and no lower bound given a lower \
             bound of -inf (first: `{}`)",
            freed.len(),
            var_names[c]
        ));
    }

    // Constraint bounds derived from row sense + RHS + RANGES.
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
        minimize,
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
        warnings,
    })
}
