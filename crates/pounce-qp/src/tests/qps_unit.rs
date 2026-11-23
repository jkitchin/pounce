//! QPS reader tests — parse → build QpProblem → solve →
//! verify the recovered solution matches the closed-form answer.
//! The on-ramp for §8.1 Maros-Mészáros benchmarking.

use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::qps::parse_qps;
use crate::solver::{ParametricActiveSetSolver, QpSolver};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()))
}

#[test]
fn parse_qps_recovers_basic_metadata_for_tiny_qp() {
    // A 2-variable, 1-constraint convex QP in QPS form:
    //   min ½(x₁² + x₂²) − x₁ − 2x₂   s.t.  x₁ + x₂ ≤ 1,
    //                                       0 ≤ x_i ≤ 1
    let text = "\
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
 LO BND       X1         0.0
 UP BND       X1         1.0
 LO BND       X2         0.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";

    let model = parse_qps(text).unwrap();
    assert_eq!(model.name, "TINY");
    assert_eq!(model.n, 2);
    assert_eq!(model.m, 1);
    assert_eq!(model.var_names, vec!["X1", "X2"]);
    assert_eq!(model.row_names, vec!["C1"]);
    assert_eq!(model.g, vec![-1.0, -2.0]);
    assert_eq!(model.bl, vec![NLP_LOWER_BOUND_INF]);
    assert_eq!(model.bu, vec![1.0]);
    assert_eq!(model.xl, vec![0.0, 0.0]);
    assert_eq!(model.xu, vec![1.0, 1.0]);
    assert_eq!(model.h_irow.len(), 2);
    assert_eq!(model.a_irow.len(), 2);
}

#[test]
fn parse_qps_round_trip_solves_to_expected_optimum() {
    // Same TINY problem as above.
    //
    // Closed form:
    //   Unconstrained min: x = (1, 2). Violates x ≤ 1 and the
    //   sum constraint. Bounds clamp x₂ = 1; x₁ = 1 clamps to xu.
    //   Sum: x₁ + x₂ = 2 > 1. So the sum constraint binds.
    //   Reduce to x₁ + x₂ = 1 with 0 ≤ x_i ≤ 1. Substitute
    //   x₂ = 1 − x₁, minimize over x₁ ∈ [0, 1]:
    //     f(x₁) = ½(x₁² + (1 − x₁)²) − x₁ − 2(1 − x₁)
    //           = x₁² − x₁ + ½ + x₁ − 2 = x₁² − 1.5
    //   Minimum at x₁ = 0, x₂ = 1. f = -1.5.
    let text = "\
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
 LO BND       X1         0.0
 UP BND       X1         1.0
 LO BND       X2         0.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X2         1.0
ENDATA
";

    let model = parse_qps(text).unwrap();

    // Wrap the parsed model in pounce-linalg sparse types so the
    // existing solver can consume it. SymTMatrixSpace and
    // GenTMatrixSpace own their indices via `Rc`; the values
    // arrays are set in place after construction.
    let h_space = SymTMatrixSpace::new(model.n as i32, model.h_irow.clone(), model.h_jcol.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&model.h_val);

    let a_space = GenTMatrixSpace::new(
        model.m as i32,
        model.n as i32,
        model.a_irow.clone(),
        model.a_jcol.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&model.a_val);

    let qp = QpProblem {
        n: model.n,
        m: model.m,
        h: &h,
        g: &model.g,
        a: &a,
        bl: &model.bl,
        bu: &model.bu,
        xl: &model.xl,
        xu: &model.xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver = new_solver();
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, crate::QpStatus::Optimal);

    // The optimum (0, 1) lies at a corner: x₁ = xl_1 = 0,
    // x₂ = xu_2 = 1, and the general constraint binds.
    assert!((sol.x[0] - 0.0).abs() < 1e-6, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", sol.x[1]);
    assert!(
        (sol.obj - (-1.5)).abs() < 1e-6,
        "obj = {} but expected -1.5",
        sol.obj
    );
}

#[test]
fn parse_qps_ranges_l_row_two_sided_lower_from_abs_range() {
    // L row: bu = rhs, bl = rhs − |range|.
    //   rhs = 5, range = 3 ⇒ bl = 2, bu = 5.
    //   rhs = 5, range = -3 ⇒ bl = 2, bu = 5 (|range|).
    let text = "\
NAME          T
ROWS
 N  C
 L  R1
COLUMNS
    X1        C          1.0   R1         1.0
RHS
    RHS       R1         5.0
RANGES
    RNG       R1         3.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!((m.bl[0] - 2.0).abs() < 1e-12, "bl = {}", m.bl[0]);
    assert!((m.bu[0] - 5.0).abs() < 1e-12, "bu = {}", m.bu[0]);

    let text2 = text.replace("3.0", "-3.0");
    let m2 = parse_qps(&text2).unwrap();
    assert!((m2.bl[0] - 2.0).abs() < 1e-12);
    assert!((m2.bu[0] - 5.0).abs() < 1e-12);
}

#[test]
fn parse_qps_ranges_g_row_two_sided_upper_from_abs_range() {
    // G row: bl = rhs, bu = rhs + |range|.
    let text = "\
NAME          T
ROWS
 N  C
 G  R1
COLUMNS
    X1        C          1.0   R1         1.0
RHS
    RHS       R1         5.0
RANGES
    RNG       R1         3.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!((m.bl[0] - 5.0).abs() < 1e-12);
    assert!((m.bu[0] - 8.0).abs() < 1e-12);
}

#[test]
fn parse_qps_ranges_e_row_positive_range_extends_upper() {
    // E row, range > 0: bl = rhs, bu = rhs + range.
    let text = "\
NAME          T
ROWS
 N  C
 E  R1
COLUMNS
    X1        C          1.0   R1         1.0
RHS
    RHS       R1         5.0
RANGES
    RNG       R1         2.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!((m.bl[0] - 5.0).abs() < 1e-12);
    assert!((m.bu[0] - 7.0).abs() < 1e-12);
}

#[test]
fn parse_qps_ranges_e_row_negative_range_extends_lower() {
    // E row, range < 0: bl = rhs + range, bu = rhs.
    let text = "\
NAME          T
ROWS
 N  C
 E  R1
COLUMNS
    X1        C          1.0   R1         1.0
RHS
    RHS       R1         5.0
RANGES
    RNG       R1        -2.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!((m.bl[0] - 3.0).abs() < 1e-12);
    assert!((m.bu[0] - 5.0).abs() < 1e-12);
}

#[test]
fn parse_qps_rejects_mip_markers() {
    let text = "\
NAME          T
ROWS
 N  C
COLUMNS
    MARKER1   'MARKER'                   'INTORG'
    X1        C          1.0
    MARKER2   'MARKER'                   'INTEND'
ENDATA
";
    let err = parse_qps(text).unwrap_err();
    assert!(
        err.contains("MIP") || err.contains("MARKER"),
        "expected MIP/MARKER rejection but got: {err}"
    );
}

/// Sum the parsed Hessian triplets that land on the lower-triangle
/// position (`irow`, `jcol`) (both 1-based). The evaluator sums all
/// triplets, so this is the *effective* H entry the solver sees.
fn h_at(model: &crate::qps::QpsModel, irow: i32, jcol: i32) -> f64 {
    model
        .h_irow
        .iter()
        .zip(&model.h_jcol)
        .zip(&model.h_val)
        .filter(|&((&r, &c), _)| r == irow && c == jcol)
        .map(|(_, &v)| v)
        .sum()
}

#[test]
fn parse_qps_qmatrix_full_matrix_does_not_double_off_diagonals() {
    // QMATRIX uses the *full-matrix* convention: both (i,j) and the
    // mirror (j,i) are listed. For H = [[2, 1], [1, 2]] the file
    // carries X1·X2 = 1 and X2·X1 = 1. After lower-triangle
    // normalization both land on (2,1); the evaluator sums all
    // triplets, so a naive parser reports H_21 = 2 — double the true
    // value. The diagonal is listed once and must be unaffected.
    let text = "\
NAME          QFULL
ROWS
 N  COST
COLUMNS
    X1        COST       0.0
    X2        COST       0.0
QMATRIX
    X1        X1         2.0
    X1        X2         1.0
    X2        X1         1.0
    X2        X2         2.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!(
        (h_at(&m, 2, 1) - 1.0).abs() < 1e-12,
        "off-diagonal H_21 = {} but expected 1.0 (doubled?)",
        h_at(&m, 2, 1)
    );
    assert!(
        (h_at(&m, 1, 1) - 2.0).abs() < 1e-12,
        "H_11 = {}",
        h_at(&m, 1, 1)
    );
    assert!(
        (h_at(&m, 2, 2) - 2.0).abs() < 1e-12,
        "H_22 = {}",
        h_at(&m, 2, 2)
    );
}

#[test]
fn parse_qps_quadobj_single_triangle_keeps_off_diagonal() {
    // QUADOBJ uses the *single-triangle* convention: each off-diagonal
    // pair is listed exactly once. The same H = [[2, 1], [1, 2]] is
    // expressed with a single X1·X2 = 1 entry, and must round-trip to
    // H_21 = 1 (this path was already correct — guards against the
    // QMATRIX fix regressing the triangle convention).
    let text = "\
NAME          QTRI
ROWS
 N  COST
COLUMNS
    X1        COST       0.0
    X2        COST       0.0
QUADOBJ
    X1        X1         2.0
    X1        X2         1.0
    X2        X2         2.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert!(
        (h_at(&m, 2, 1) - 1.0).abs() < 1e-12,
        "off-diagonal H_21 = {}",
        h_at(&m, 2, 1)
    );
    assert!((h_at(&m, 1, 1) - 2.0).abs() < 1e-12);
    assert!((h_at(&m, 2, 2) - 2.0).abs() < 1e-12);
}

#[test]
fn parse_qps_rejects_binary_variable_bound() {
    let text = "\
NAME          T
ROWS
 N  C
COLUMNS
    X1        C          1.0
BOUNDS
 BV BND       X1
ENDATA
";
    let err = parse_qps(text).unwrap_err();
    assert!(err.contains("BV"), "expected BV rejection but got: {err}");
}

// ---------------------------------------------------------------------------
// gh#1256: optional set names, fixed-format fallback, duplicates, negative UP.
// ---------------------------------------------------------------------------

/// Rows `C1`..`C3` with x1 in each, so RHS / RANGES land somewhere visible.
const THREE_ROWS: &str = "\
NAME          T
ROWS
 N  COST
 L  C1
 L  C2
 E  C3
COLUMNS
    X1        COST       1.0    C1         1.0
    X1        C2         1.0    C3         1.0
    X2        COST       1.0    C1         1.0
";

fn three_rows(tail: &str) -> crate::qps::QpsModel {
    parse_qps(&format!("{THREE_ROWS}{tail}ENDATA\n")).unwrap()
}

#[test]
fn rhs_without_set_name_one_and_two_pairs() {
    // Two fields (row, value) and four (two pairs): no set name. The
    // four-field form is NETLIB BLEND's, which the old reader took as
    // set `C1` + pair (`5.0`, `C2`) and dropped as an unknown row,
    // silently zeroing every RHS.
    let m = three_rows("RHS\n    C1  5.0  C2  6.0\n    C3  7.0\n");
    assert_eq!(m.bu, vec![5.0, 6.0, 7.0]);
    assert!(m.warnings.is_empty(), "{:?}", m.warnings);
}

#[test]
fn rhs_with_set_name_equal_to_a_row_name() {
    // Maros-Meszaros DPKLO1 names its rows 1, 2, ... and its RHS set `1`.
    // Three fields are set + pair, whatever the set is called.
    let text = "\
NAME          D
ROWS
 N  OBJ
 E  1
 E  2
COLUMNS
    X         1          1.0    2          1.0
RHS
    1         2          4.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert_eq!(m.bl, vec![0.0, 4.0]);
}

#[test]
fn ranges_without_set_name() {
    let m = three_rows("RHS\n    RHS  C1  5.0\nRANGES\n    C1  2.0\n");
    assert_eq!((m.bl[0], m.bu[0]), (3.0, 5.0));
}

#[test]
fn bounds_without_set_name_valued_and_valueless() {
    // `LO col value` (3 fields) and `FR col` (2 fields): no set name. With
    // a set name the same bounds take 4 and 3 fields.
    let m = three_rows("BOUNDS\n LO X1 -10.\n FR X2\n");
    assert_eq!((m.xl[0], m.xu[0]), (-10.0, NLP_UPPER_BOUND_INF));
    assert_eq!(
        (m.xl[1], m.xu[1]),
        (NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF)
    );
    let m = three_rows("BOUNDS\n LO BND X1 -10.\n FR BND X2\n");
    assert_eq!(m.xl[0], -10.0);
    assert_eq!(m.xl[1], NLP_LOWER_BOUND_INF);
}

/// One fixed-format data line: fields at 1-based columns 2-3, 5-12, 15-22,
/// 25-36, 40-47, 50-61, exactly where the format puts them.
fn fx(fields: [&str; 6]) -> String {
    let starts = [1usize, 4, 14, 24, 39, 49];
    let mut line = vec![b' '; 61];
    for (f, &at) in fields.iter().zip(&starts) {
        line[at..at + f.len()].copy_from_slice(f.as_bytes());
    }
    String::from_utf8(line).unwrap().trim_end().to_string() + "\n"
}

#[test]
fn fixed_format_blank_set_name_columns() {
    // EXDATA's shape: the set-name columns (5-12) left blank. Free
    // splitting sees `LO X1 -10.` and `C1 5.0`, the no-set-name forms.
    let tail = format!(
        "RHS\n{}BOUNDS\n{}",
        fx(["", "", "C1", "5.0", "", ""]),
        fx(["LO", "", "X1", "-10.", "", ""])
    );
    let m = three_rows(&tail);
    assert_eq!(m.bu[0], 5.0);
    assert_eq!(m.xl[0], -10.0);
}

#[test]
fn fixed_format_names_with_spaces_fall_back_and_warn() {
    // NETLIB FORPLAN's shape: names containing spaces, only readable by
    // column position. Free format fails on the field count, the fixed
    // reading succeeds, and the fallback is reported.
    let text = [
        "NAME          SPACED\nROWS\n".to_string(),
        fx(["N", "COST", "", "", "", ""]),
        fx(["E", "DEDO3 1R", "", "", "", ""]),
        fx(["E", "DEDO3 2R", "", "", "", ""]),
        "COLUMNS\n".to_string(),
        fx(["", "X 1", "COST", "1.", "DEDO3 1R", "1."]),
        fx(["", "X 1", "DEDO3 2R", "2.", "", ""]),
        "RHS\n".to_string(),
        fx(["", "RHS 1", "DEDO3 1R", "4.", "DEDO3 2R", "6."]),
        "RANGES\n".to_string(),
        fx(["", "RNG 1", "DEDO3 1R", "1.", "", ""]),
        "BOUNDS\n".to_string(),
        fx(["UP", "BND 1", "X 1", "10.", "", ""]),
        "ENDATA\n".to_string(),
    ]
    .concat();
    let m = parse_qps(&text).unwrap();
    assert_eq!(m.row_names, vec!["DEDO3 1R", "DEDO3 2R"]);
    assert_eq!(m.var_names, vec!["X 1"]);
    assert_eq!(m.g, vec![1.0]);
    assert_eq!(m.a_val, vec![1.0, 2.0]);
    // E rows; a positive range on an E row gives [rhs, rhs + r].
    assert_eq!(
        (m.bl.clone(), m.bu.clone()),
        (vec![4.0, 6.0], vec![5.0, 6.0])
    );
    assert_eq!(m.xu, vec![10.0]);
    assert!(
        m.warnings[0].starts_with("read as fixed-format MPS"),
        "{:?}",
        m.warnings
    );
}

#[test]
fn free_format_file_is_not_misread_as_fixed() {
    // A genuinely broken free file (three pairs on one RHS line) reports
    // both readings' errors; the fixed reading refuses text between its
    // field columns rather than producing a model.
    let err = parse_qps(&format!(
        "{THREE_ROWS}RHS\n    RHS  C1  5.0  C2  6.0  C3  7.0\nENDATA\n"
    ))
    .unwrap_err();
    assert!(err.contains("RHS entry needs"), "{err}");
    assert!(err.contains("as fixed-format MPS"), "{err}");
}

#[test]
fn duplicates_first_kept_for_linear_data_summed_for_quadobj() {
    // Matches HiGHS: COLUMNS / RHS / RANGES keep the first entry, QUADOBJ
    // entries for the same unordered pair are summed. Each is warned.
    let text = "\
NAME          DUP
ROWS
 N  OBJ
 L  C1
COLUMNS
    X  OBJ  1.0  C1  1.0
    X  OBJ  9.0  C1  2.0
    Y  OBJ  1.0  C1  1.0
RHS
    RHS  C1  4.0  C1  6.0
    RHS  OBJ  1.0  OBJ  2.0
RANGES
    RNG  C1  1.0  C1  3.0
QUADOBJ
    X  X  2.0
    X  X  5.0
    Y  X  1.0
    X  Y  1.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert_eq!(m.g, vec![1.0, 1.0]);
    assert_eq!(m.a_val, vec![1.0, 1.0]);
    assert_eq!((m.bl[0], m.bu[0]), (3.0, 4.0));
    assert_eq!(m.obj_constant, -1.0);
    assert_eq!(h_at(&m, 1, 1), 7.0);
    assert_eq!(h_at(&m, 2, 1), 2.0);
    let w = m.warnings.join("\n");
    assert!(w.contains("2 duplicate COLUMNS"), "{w}");
    assert!(w.contains("2 duplicate RHS"), "{w}");
    assert!(w.contains("1 duplicate RANGES"), "{w}");
    assert!(w.contains("2 duplicate QUADOBJ"), "{w}");
}

#[test]
fn negative_up_without_lower_frees_the_lower_bound() {
    // CPLEX / Gurobi rule: a negative UP with no lower bound given means
    // lower = -inf, not the empty box [0, u]. An explicit LO, or FX, keeps
    // what the file says.
    let m = three_rows("BOUNDS\n UP BND X1 -5.0\n");
    assert_eq!((m.xl[0], m.xu[0]), (NLP_LOWER_BOUND_INF, -5.0));
    assert!(
        m.warnings.iter().any(|w| w.contains("negative UP")),
        "{:?}",
        m.warnings
    );

    let m = three_rows("BOUNDS\n LO BND X1 -8.0\n UP BND X1 -5.0\n");
    assert_eq!((m.xl[0], m.xu[0]), (-8.0, -5.0));
    let m = three_rows("BOUNDS\n FX BND X1 -5.0\n");
    assert_eq!((m.xl[0], m.xu[0]), (-5.0, -5.0));
    assert!(m.warnings.is_empty(), "{:?}", m.warnings);
    // Non-negative UP leaves the default lower bound alone.
    let m = three_rows("BOUNDS\n UP BND X1 3.0\n");
    assert_eq!((m.xl[0], m.xu[0]), (0.0, 3.0));
}

#[test]
fn entries_on_a_second_n_row_are_dropped_with_a_warning() {
    let text = "\
NAME          FREE
ROWS
 N  OBJ
 N  SPARE
 L  C1
COLUMNS
    X  OBJ  1.0  SPARE  3.0
    X  C1  1.0
ENDATA
";
    let m = parse_qps(text).unwrap();
    assert_eq!((m.m, m.a_val.clone()), (1, vec![1.0]));
    assert!(
        m.warnings.iter().any(|w| w.contains("free rows")),
        "{:?}",
        m.warnings
    );
}

#[test]
fn crlf_line_endings_parse() {
    let m = parse_qps(&format!("{THREE_ROWS}RHS\n    C1  5.0\nENDATA\n").replace('\n', "\r\n"))
        .unwrap();
    assert_eq!(m.bu[0], 5.0);
}
