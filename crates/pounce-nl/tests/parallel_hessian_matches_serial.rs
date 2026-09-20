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
use std::sync::{Mutex, OnceLock};

/// Serializes "set the switch, build, evaluate".
///
/// The switch is `POUNCE_NL_PARALLEL_EVAL`, which is **process-wide**, and
/// libtest runs these tests as concurrent threads of one process. A test that
/// loses the race does not fail — it passes *vacuously*, because both arms of
/// its comparison end up on the same path and agree for the wrong reason.
///
/// Measured, with the reverse-order mutation applied to the parallel Hessian
/// walk and the file run 20 times: **with this lock the mutation is caught
/// 20/20; without it, 16/20**. So four runs in twenty proved nothing, silently.
/// That is the failure this file exists to prevent, occurring inside the file
/// itself, and it is why the guard is held across the evaluation rather than
/// just across the `set_var`.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

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
    // Held until this function returns: see `env_lock`.
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the lock above makes this the only thread touching the variable.
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

/// The Jacobian's parallel walk splits **rows**, not colors, and does not
/// reorder anything: each row zeroes only its own columns of the gradient
/// scratch and writes a contiguous run of `values`. So the bar is higher than
/// for the Hessian — not "the same sum in the same order" but "the same
/// arithmetic entirely" — and the bitwise comparison is correspondingly
/// stricter than it looks.
///
/// What it really guards is the bookkeeping: each group gets its own slice of
/// `values`, sized from the row offsets, and writes into it from index zero.
/// Getting that wrong shifts a row's gradient onto its neighbour's nonzeros —
/// plausible-looking numbers in the wrong places, which no residual check
/// downstream would flag.
///
/// Mutation table:
///
/// | Break | Result |
/// |---|---|
/// | Reset the write cursor per row instead of per group | **run**: all 9,000 entries differ, from the first |
#[test]
fn parallel_jacobian_matches_serial_bit_for_bit() {
    let serial = jacobian(ROWS, false);
    let parallel = jacobian(ROWS, true);

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

/// The constraint Jacobian at the same varied point as [`hessian`].
fn jacobian(rows: usize, parallel: bool) -> Vec<f64> {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the lock above makes this the only thread touching the variable.
    unsafe {
        std::env::set_var("POUNCE_NL_PARALLEL_EVAL", if parallel { "1" } else { "0" });
    }
    let mut t = NlTnlp::new(model(rows));
    let info = t.get_nlp_info().expect("info");
    let (n, nnz) = (info.n as usize, info.nnz_jac_g as usize);
    let x: Vec<f64> = (0..n)
        .map(|i| 0.3 + 0.7 * ((i % 13) as f64) / 13.0)
        .collect();
    let mut values = vec![0.0; nnz];
    assert!(t.eval_jac_g(
        Some(&x),
        true,
        SparsityRequest::Values {
            values: &mut values
        },
    ));
    values
}

/// Constraint values, the third walk and the simplest: row `i` reads only `x`
/// and writes only `g[i]`, so the parallel version is the serial loop with a
/// per-worker forward arena. Nothing is reordered or duplicated.
///
/// It earns a test anyway, for the reason the other two do: "obviously
/// parallel" is what every incorrect parallelization was called first. The
/// specific thing it pins is that the per-worker arena really is per worker —
/// sharing one `vals` buffer across rows is the mistake this shape invites,
/// and it corrupts values non-deterministically rather than failing.
///
/// Mutation table:
///
/// | Break | Result |
/// |---|---|
/// | Write `g[i + 1]` instead of `g[i]` (an off-by-one in the row index) | **run**: 2,999 of 3,000 entries differ |
#[test]
fn parallel_eval_g_matches_serial_bit_for_bit() {
    let serial = constraint_values(ROWS, false);
    let parallel = constraint_values(ROWS, true);

    assert_eq!(serial.len(), parallel.len());
    let nonzero = serial.iter().filter(|v| **v != 0.0).count();
    assert!(nonzero > 1_000, "{nonzero} nonzero entries");
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

/// `g(x)` at the same varied point as the other two walks.
fn constraint_values(rows: usize, parallel: bool) -> Vec<f64> {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the lock above makes this the only thread touching the variable.
    unsafe {
        std::env::set_var("POUNCE_NL_PARALLEL_EVAL", if parallel { "1" } else { "0" });
    }
    let mut t = NlTnlp::new(model(rows));
    let info = t.get_nlp_info().expect("info");
    let (n, m) = (info.n as usize, info.m as usize);
    let x: Vec<f64> = (0..n)
        .map(|i| 0.3 + 0.7 * ((i % 13) as f64) / 13.0)
        .collect();
    let mut g = vec![0.0; m];
    assert!(t.eval_g(&x, true, &mut g));
    g
}
