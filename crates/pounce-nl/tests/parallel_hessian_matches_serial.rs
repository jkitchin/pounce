//! The parallel flat-Hessian walk must reproduce the serial one **exactly**.
//!
//! `eval_h`'s flat path walks tapes and, inside each, the colors it touches,
//! so several rows accumulate into one `compressed[c]`. That is unshareable
//! across threads, so the parallel path walks the same work *color-major*
//! instead, one rayon task per color, each owning its accumulator.
//!
//! The reordering is the whole risk. Two things make it safe, and this file
//! pins both:
//!
//! 1. Within a color, the (row, tape) pairs stay in ascending order — the
//!    order the serial walk contributes them in — so the sums are identical
//!    term by term. Floating-point addition is not associative, so "identical
//!    term by term" is the only claim worth asserting: the test compares
//!    **bit patterns**, not a tolerance. A tolerance here would pass just as
//!    happily on a genuinely reordered sum, which is the defect it exists to
//!    catch.
//! 2. The objective's tapes are accumulated before the constraint block and
//!    stay serial, so a color's order is still "objective, then rows
//!    ascending".
//!
//! Mutation table (each run against this file):
//!
//! | Break | Result |
//! |---|---|
//! | Visit a color's pairs in reverse order (`.rev()`) | **run**: 265 of 6,003 entries differ, first by 4.5e-16 relative |
//!
//! That mutation is also what proves this fixture *reaches* the parallel walk
//! rather than comparing the serial path against itself — the failure mode the
//! file would otherwise have no defence against. It is worth noting what the
//! difference it produces looks like: 4.5e-16 relative, which every tolerance
//! a reviewer would reach for accepts. The bitwise comparison is not
//! fastidiousness; it is the only assertion that fails here.
//!
//! Note that dropping the `lam[k] == 0` skip is *not* a detectable mutation:
//! a zero multiplier contributes `0 · H`, so the skip is an optimization and
//! not semantics. It is listed here so the next reader does not mistake its
//! absence from the table for an untested branch.
//!
//! The model is built from expressions rather than a `.nl` fixture so the test
//! carries no data file, and is sized past `HESS_PAR_MIN_PAIRS` so the
//! parallel path is actually taken — `a_model_below_the_threshold_is_serial`
//! guards the other side of that gate, since a test whose model never crosses
//! it would compare the serial path against itself and pass forever.

use pounce_nl::nl_reader::{BinOp, Expr, NlProblem, NlProblemParts, NlTnlp, UnaryOp};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

/// `rows` nonlinear rows over `rows + 2` variables, each row coupling three
/// neighbours so the Hessian needs several colors:
///
/// ```text
/// g_i(x) = sin(x_i · x_{i+1}) + x_{i+1}^2 · x_{i+2}
/// ```
///
/// Overlapping supports are the point: a coloring with one color per row would
/// make the walk trivially parallel and prove nothing.
fn model(rows: usize) -> NlProblem {
    let n = rows + 2;
    let v = |i: usize| Expr::Var(i);
    let mul = |a: Expr, b: Expr| Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::Binary(BinOp::Add, Box::new(a), Box::new(b));

    let constraints: Vec<Expr> = (0..rows)
        .map(|i| {
            let sin = Expr::Unary(UnaryOp::Sin, Box::new(mul(v(i), v(i + 1))));
            add(sin, mul(mul(v(i + 1), v(i + 1)), v(i + 2)))
        })
        .collect();
    let objective = Expr::Sum((0..n).map(|i| mul(v(i), v(i))).collect::<Vec<_>>());

    NlProblem::from_expressions(NlProblemParts {
        minimize: true,
        objective,
        obj_constant: 0.0,
        constraints,
        x_l: vec![-5.0; n],
        x_u: vec![5.0; n],
        x0: vec![0.5; n],
        g_l: vec![-1.0; rows],
        g_u: vec![1.0; rows],
        var_names: Vec::new(),
        con_names: Vec::new(),
    })
    .expect("model builds")
}

/// The Lagrangian Hessian at a deliberately varied point. A constant `x` or a
/// constant multiplier vector hides an index error by making the wrong entry
/// equal to the right one.
fn hessian(rows: usize, parallel: bool) -> Vec<f64> {
    // SAFETY: single-threaded test body, set before any evaluator is built.
    unsafe {
        if parallel {
            std::env::remove_var("POUNCE_NL_PARALLEL_EVAL");
        } else {
            std::env::set_var("POUNCE_NL_PARALLEL_EVAL", "0");
        }
    }
    let mut t = NlTnlp::new(model(rows));
    let info = t.get_nlp_info().expect("info");
    let (n, m, nnzh) = (info.n as usize, info.m as usize, info.nnz_h_lag as usize);
    let x: Vec<f64> = (0..n)
        .map(|i| 0.3 + 0.7 * ((i % 13) as f64) / 13.0)
        .collect();
    // Some multipliers are zero on purpose: the walks skip those rows, and a
    // skip applied in one walk but not the other shows up here.
    let lam: Vec<f64> = (0..m)
        .map(|i| {
            if i % 7 == 0 {
                0.0
            } else {
                -1.0 + (i % 5) as f64 * 0.4
            }
        })
        .collect();
    let mut values = vec![0.0; nnzh];
    assert!(t.eval_h(
        Some(&x),
        true,
        1.3,
        Some(&lam),
        true,
        SparsityRequest::Values {
            values: &mut values
        },
    ));
    values
}

/// 3,000 rows is comfortably past `HESS_PAR_MIN_PAIRS` (4,096 pairs at ~3
/// colors per tape), so the parallel walk is the one under test.
const ROWS: usize = 3_000;

#[test]
fn parallel_matches_serial_bit_for_bit() {
    let serial = hessian(ROWS, false);
    let parallel = hessian(ROWS, true);

    assert_eq!(serial.len(), parallel.len());
    let nonzero = serial.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > 1_000,
        "the comparison must have something to compare: {nonzero} nonzero entries"
    );

    let differing: Vec<usize> = (0..serial.len())
        .filter(|&i| serial[i].to_bits() != parallel[i].to_bits())
        .collect();
    assert!(
        differing.is_empty(),
        "{} of {} entries differ; first at {}: serial {:.17e} vs parallel {:.17e}",
        differing.len(),
        serial.len(),
        differing[0],
        serial[differing[0]],
        parallel[differing[0]],
    );
}

/// The other side of the gate. A small model must stay on the serial walk —
/// below the threshold the color-major inversion costs more than it saves,
/// because it forwards each tape once per color it touches instead of once.
///
/// Asserted through the answer rather than through a private flag: a tiny
/// model evaluated with the switch on and off must agree bit-for-bit for the
/// trivial reason that both took the same path. It is a weak assertion on its
/// own, which is why it exists next to the strong one above rather than
/// instead of it — together they say the gate changes speed, never values.
#[test]
fn a_model_below_the_threshold_is_serial() {
    let with = hessian(8, true);
    let without = hessian(8, false);
    assert!(with.iter().any(|v| *v != 0.0), "degenerate model");
    assert!(
        with.iter()
            .zip(&without)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a model below the parallel threshold must evaluate identically either way"
    );
}
