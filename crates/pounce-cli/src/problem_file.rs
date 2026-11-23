//! Problem-file loading for the CLI: AMPL `.nl`, plus QPS / MPS (gh#1251).
//!
//! Every downstream stage of the CLI — classification, the convex LP/QP
//! dispatch, the NLP arm, `.sol` writing, `verify`, `check-x0`, `diagnose` —
//! consumes an [`NlProblem`]. A QPS / MPS file is therefore not given a
//! pipeline of its own: [`qps_to_nl_problem`] assembles the same in-memory
//! `NlProblem` a modelling frontend would ([`NlProblem::from_expressions`]),
//! with the linear objective and rows in the linear slots and `½ xᵀHx` as the
//! objective's nonlinear body. No `.nl` file is written or re-parsed.
//!
//! The format is chosen by file extension, case-insensitively: `.qps` and
//! `.mps` — or `.qps.gz` / `.mps.gz`, decompressed on the fly — read through
//! [`pounce_qp::qps::parse_qps`]; any other `.gz` is refused with a message
//! saying what is read compressed; anything else (including an extensionless
//! AMPL stub) goes to [`nl_reader::read_nl_file`] as before.
//!
//! The QPS reader's [`QpsModel::warnings`] — duplicates it ignored or summed,
//! a negative `UP` that freed a lower bound, a fallback to fixed format — are
//! printed to stderr, one line each, so a convention applied to the file is
//! never silent.

use crate::nl_reader::{self, BinOp, Expr, NlProblem, NlProblemParts};
use pounce_qp::qps::{QpsModel, parse_qps};
use std::collections::BTreeMap;
use std::path::Path;

fn is_qps_ext(e: &str) -> bool {
    e.eq_ignore_ascii_case("qps") || e.eq_ignore_ascii_case("mps")
}

/// Whether `path` names a gzip-compressed file (`.gz`, any case).
fn is_gz_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
}

/// Whether `path` names a QPS / MPS file: `.qps` / `.mps`, or the same
/// followed by `.gz` (by extension, case-insensitive).
pub fn is_qps_path(path: &Path) -> bool {
    let inner = if is_gz_path(path) {
        path.file_stem().map(Path::new)
    } else {
        Some(path)
    };
    inner
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .is_some_and(is_qps_ext)
}

/// Read a problem file, dispatching on its extension: `.qps` / `.mps`
/// (optionally `.gz`) through the QPS reader, any other `.gz` refused,
/// everything else through the `.nl` reader.
pub fn read_problem_file(path: &Path) -> Result<NlProblem, String> {
    if is_qps_path(path) {
        read_qps_file(path)
    } else if is_gz_path(path) {
        Err(format!(
            "{} is gzip-compressed; only .mps.gz and .qps.gz are read compressed. \
             Decompress it first (gunzip).",
            path.display()
        ))
    } else {
        nl_reader::read_nl_file(path)
    }
}

/// Read a QPS / MPS file (decompressing a `.gz`), print the reader's
/// warnings to stderr, and convert it with [`qps_to_nl_problem`].
pub fn read_qps_file(path: &Path) -> Result<NlProblem, String> {
    let model = parse_qps_file(path)?;
    for w in &model.warnings {
        eprintln!("pounce: warning: {}: {w}", path.display());
    }
    qps_to_nl_problem(&model)
}

/// Read and parse a QPS / MPS file, decompressing a `.gz`. Bytes that are
/// not UTF-8 (old files carry Latin-1 in comments and names) are replaced
/// rather than refused; the fixed-format reader rejects them where they
/// would matter.
pub fn parse_qps_file(path: &Path) -> Result<QpsModel, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let bytes = if is_gz_path(path) {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| format!("could not decompress {}: {e}", path.display()))?;
        out
    } else {
        bytes
    };
    parse_qps(&String::from_utf8_lossy(&bytes))
}

/// Assemble the [`NlProblem`] for a parsed QPS model:
///
/// ```text
/// min/max  ½ xᵀHx + gᵀx + c   s.t.  bl ≤ Ax ≤ bu,  xl ≤ x ≤ xu
/// ```
///
/// `H` arrives as a lower triangle (1-based triplets, duplicates summed),
/// so a diagonal entry contributes `½·h·xᵢ²` and an off-diagonal one
/// `h·xᵢxⱼ`. A model with no quadratic entries gets a zero nonlinear body
/// and classifies as an LP.
pub fn qps_to_nl_problem(model: &QpsModel) -> Result<NlProblem, String> {
    let n = model.n;
    let m = model.m;

    // Objective quadratic, as an explicit sum of monomials. Duplicate
    // triplets are folded first so the tree carries one term per pair.
    let mut h: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for ((&r, &c), &v) in model.h_irow.iter().zip(&model.h_jcol).zip(&model.h_val) {
        let (r, c) = (r as usize - 1, c as usize - 1);
        *h.entry((r.max(c), r.min(c))).or_insert(0.0) += v;
    }
    let terms: Vec<Expr> = h
        .into_iter()
        .filter(|&(_, v)| v != 0.0)
        .map(|((i, j), v)| {
            let coef = if i == j { 0.5 * v } else { v };
            Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(coef)),
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Var(i)),
                    Box::new(Expr::Var(j)),
                )),
            )
        })
        .collect();
    let objective = match terms.len() {
        0 => Expr::Const(0.0),
        1 => terms.into_iter().next().unwrap_or(Expr::Const(0.0)),
        _ => Expr::Sum(terms),
    };

    let mut prob = NlProblem::from_expressions(NlProblemParts {
        minimize: model.minimize,
        objective,
        obj_constant: model.obj_constant,
        constraints: vec![Expr::Const(0.0); m],
        x_l: model.xl.clone(),
        x_u: model.xu.clone(),
        x0: vec![0.0; n],
        g_l: model.bl.clone(),
        g_u: model.bu.clone(),
        var_names: model.var_names.clone(),
        con_names: model.row_names.clone(),
    })
    .map_err(|e| format!("QPS model `{}`: {e}", model.name))?;

    prob.obj_linear = model
        .g
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != 0.0)
        .map(|(j, &v)| (j, v))
        .collect();

    // Linear rows, duplicates within a row summed, ascending by column.
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); m];
    for ((&r, &c), &v) in model.a_irow.iter().zip(&model.a_jcol).zip(&model.a_val) {
        *rows[r as usize - 1].entry(c as usize - 1).or_insert(0.0) += v;
    }
    prob.con_linear = rows
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();

    Ok(prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "\
NAME          TINY
ROWS
 N  COST
 L  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
RHS
    RHS       C1         1.0
BOUNDS
 UP BND       X1         1.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X1         0.5
    X2        X2         1.0
ENDATA
";

    #[test]
    fn extension_dispatch_is_case_insensitive() {
        assert!(is_qps_path(Path::new("a/b.qps")));
        assert!(is_qps_path(Path::new("B.QPS")));
        assert!(is_qps_path(Path::new("c.mps")));
        assert!(is_qps_path(Path::new("c.MPS")));
        assert!(!is_qps_path(Path::new("c.nl")));
        assert!(!is_qps_path(Path::new("stub")));
        assert!(is_qps_path(Path::new("d/afiro.mps.gz")));
        assert!(is_qps_path(Path::new("QAFIRO.QPS.GZ")));
        assert!(!is_qps_path(Path::new("model.nl.gz")));
        assert!(!is_qps_path(Path::new("x.gz")));
    }

    #[test]
    fn gz_input_reads_like_the_plain_file() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("pounce_pf_gz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("tiny.qps");
        std::fs::write(&plain, TINY).unwrap();
        let gz = dir.join("tiny.QPS.gz");
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(TINY.as_bytes()).unwrap();
        std::fs::write(&gz, enc.finish().unwrap()).unwrap();
        let a = read_problem_file(&plain).unwrap();
        let b = read_problem_file(&gz).unwrap();
        assert_eq!(a.obj_linear, b.obj_linear);
        assert_eq!(a.con_linear, b.con_linear);
        assert_eq!((a.x_l, a.x_u), (b.x_l, b.x_u));

        // Any other `.gz` is refused with a message that says why, not
        // handed to the `.nl` reader.
        let nlgz = dir.join("model.nl.gz");
        std::fs::write(&nlgz, b"not read").unwrap();
        let err = read_problem_file(&nlgz).unwrap_err();
        assert!(err.contains("Decompress it first"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn objective_matches_half_xhx_plus_gx() {
        let model = parse_qps(TINY).unwrap();
        let prob = qps_to_nl_problem(&model).unwrap();
        assert_eq!((prob.n, prob.m), (2, 1));
        assert_eq!(prob.obj_linear, vec![(0, -1.0), (1, -2.0)]);
        assert_eq!(prob.con_linear, vec![vec![(0, 1.0), (1, 1.0)]]);
        assert_eq!(prob.g_u, vec![1.0]);
        assert_eq!(prob.var_names, vec!["X1", "X2"]);
        // ½(x1² + x2²) + 0.5·x1·x2 at (2, 3): 0.5·(4 + 9) + 3 = 9.5
        let x = [2.0, 3.0];
        let q = nl_reader::eval_expr(&prob.obj_expr(), &x);
        assert!((q - 9.5).abs() < 1e-12, "{q}");
    }

    #[test]
    fn lp_has_zero_nonlinear_objective() {
        let lp = TINY.split("QUADOBJ").next().unwrap().to_string() + "ENDATA\n";
        let prob = qps_to_nl_problem(&parse_qps(&lp).unwrap()).unwrap();
        assert!(prob.obj_nonlinear.is_trivially_zero());
        assert!(prob.con_nonlinear.iter().all(|b| b.is_trivially_zero()));
    }

    #[test]
    fn objsense_max_sets_maximize() {
        let max = TINY.replacen("ROWS", "OBJSENSE\n    MAX\nROWS", 1);
        assert!(!parse_qps(&max).unwrap().minimize);
        let max_free = TINY.replacen("ROWS", "OBJSENSE MAXIMIZE\nROWS", 1);
        let prob = qps_to_nl_problem(&parse_qps(&max_free).unwrap()).unwrap();
        assert!(!prob.minimize);
    }
}
