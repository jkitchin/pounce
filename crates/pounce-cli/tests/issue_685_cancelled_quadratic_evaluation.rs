//! A degree-2 row whose quadratic coefficients cancel must not be
//! **evaluated** from those coefficients (gh #685, part 1).
//!
//! This is the storage-side half of gh #683, and it is a separate defect
//! with a separate gate. `provably_affine` answers a question about
//! *degree*; `admitted_quad_form` answers a question about *evaluation* —
//! it hands Q4's constant-structure evaluator a matrix and a vector to use
//! **in place of** the row's tape. Its gate, `is_expanded_quadratic`, is a
//! test on the *shape* the coefficients were derived from: a flat sum of
//! monomials passes it. It says nothing about whether the derivation kept
//! them.
//!
//! So `2⁵³·x₀² + x₀² − 2⁵³·x₀²` passes the shape gate, folds to an empty
//! term map, and the row is evaluated as identically `0` where its own tape
//! gives `16` at `x₀ = 3`. Nothing here is opt-in: the second row of each
//! model is a `sin`, the model classifies `NLP`, and the substitution
//! happens on the default route with no option set. gh #685 reports the
//! end of it — an objective of `-1.0e6` against a true optimum a couple of
//! units away, reported `Optimal`.
//!
//! The fix keys off [`Quad2::lost_terms`] and not off the map being
//! empty, and the difference is the whole of `partial_cancellation` below:
//! `2⁵³·x₀² + x₀² − 2⁵³·x₀² + x₁²` keeps `x₁²`, so the map is *not* empty
//! and `provably_affine` says `Some(false)` quite correctly — while the
//! read-out is still short an entire `x₀²`. An emptiness test passes that
//! model and stays wrong.
//!
//! Part 2 of the issue — the classifier calling such a model an LP — was
//! fixed after this one, in `issue_685_cancelled_quadratic_routing`. That
//! fix moved the gate onto `analyze_quadratic_full` itself, so the models
//! here now also refuse to hand a dropped form to the classifier; the
//! `admitted_quad_form` gate this file asserts is the separate one that
//! governs *evaluation*, and it stays where it is.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::nl_reader::{NlProblem, NlTnlp, parse_nl_text_with_quadratic};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

// ---------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------

/// Two variables, two rows, a linear objective `x₀ + x₁`. Row 0 is
/// `body ≤ 5` — an inequality, so it lands in `jac_d` — and row 1 is
/// `sin(x₁) = 0`, which keeps the classifier on the general NLP route.
///
/// Deliberately the same shape as `issue_683_cancelled_quadratic_degree`'s
/// model, so a difference between the two files is a difference in what is
/// being asserted rather than in what is being solved.
fn model(body: &str) -> String {
    format!(
        "g3 0 1 0\n\
         2 2 1 0 1\n\
         2 0\n\
         0 0\n\
         2 1 1\n\
         0 0 0 1\n\
         0 0 0 0 0\n\
         1 2\n\
         0 0\n\
         0 0 0 0 0\n\
         C0\n{body}\
         C1\n\
         o41\n\
         v1\n\
         O0 0\n\
         n0\n\
         x2\n\
         0 1.0\n\
         1 1.0\n\
         r\n\
         1 5.0\n\
         4 0.0\n\
         b\n\
         3\n\
         3\n\
         k1\n\
         0\n\
         J1 1\n\
         1 0\n\
         G0 2\n\
         0 1.0\n\
         1 1.0\n"
    )
}

/// `2⁵³·x₀² + x₀² − 2⁵³·x₀²`, folded front to back: `2⁵³`, then `2⁵³ + 1`
/// rounds back to `2⁵³`, then `2⁵³ − 2⁵³` is exactly `0`. The body is
/// `x₀²`; the stored form is nothing at all.
///
/// The middle term is spelled `x₀^2` where the outer two are `x₀·x₀` so
/// that the tape does not hash-cons all three onto one node — with one node
/// the tape's own derivative cancels too and there is nothing to compare
/// against. Same construction, and same reason, as gh #683's reproducer.
fn cancelling_body() -> String {
    let big = (1u64 << 53) as f64;
    format!(
        "o54\n3\n\
         o2\nn{big:.1}\no2\nv0\nv0\n\
         o5\nv0\nn2\n\
         o2\nn{neg:.1}\no2\nv0\nv0\n",
        neg = -big,
        big = big,
    )
}

/// `2⁵³·x₀² + x₀² − 2⁵³·x₀² + x₁²` — the same cancellation with a live term
/// beside it. The map keeps `x₁²`, so it is not empty, and the degree
/// answer is a correct `Some(false)`; the read-out is nonetheless missing
/// `x₀²` entirely. This is the model that separates a `lost_terms` gate
/// from an emptiness gate.
fn partially_cancelling_body() -> String {
    let big = (1u64 << 53) as f64;
    format!(
        "o54\n4\n\
         o2\nn{big:.1}\no2\nv0\nv0\n\
         o5\nv0\nn2\n\
         o2\nn{neg:.1}\no2\nv0\nv0\n\
         o5\nv1\nn2\n",
        neg = -big,
        big = big,
    )
}

/// `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)`: one monomial, degree 2, whose coefficient
/// `10⁻⁴⁰⁰` is not representable and flushes to zero on the multiply.
fn underflowing_body() -> &'static str {
    "o2\no2\nn1e-200\nv0\no2\nn1e-200\nv0\n"
}

/// An honest degree-2 body, to check the gate did not simply close.
fn ordinary_body() -> &'static str {
    "o54\n2\no2\nv0\nv0\no5\nv1\nn2\n"
}

/// `x₀·x₀ − x₀^2 + x₁²`: a term that cancels **exactly**, beside a live one.
/// `fl(1) + fl(−1)` is `0` with nothing rounded away, so the read-out is the
/// whole body and the row keeps its matrix evaluation (gh #687). The two
/// spellings of `x₀²` keep the tape from hash-consing the pair.
fn exactly_cancelling_body() -> &'static str {
    "o54\n3\no2\nv0\nv0\no16\no5\nv0\nn2\no5\nv1\nn2\n"
}

/// `(2²⁷x₀)² + (x₀ + x₁)² − (x₀·2²⁷)²` — the same defect reached through
/// gh #673's factored door, and the witness gh #711's review built.
///
/// Three honest squares. There is no monomial spine here, so
/// `is_expanded_quadratic` has no opinion and the shape gate in
/// `admitted_factored_form` lets it through; the loss happens later, when
/// `Σ 2wₖbₖbₖᵀ` accumulates `(0, 0)` as `2·2⁵⁴ + 2 − 2·2⁵⁴`. The `+ 2` is
/// half an ulp at `2⁵⁵` and ties back to even, so the entry reaches exactly
/// `0.0` and is not stored — while the tape declares it.
///
/// The two spellings of the same square (`2²⁷·x₀` and `x₀·2²⁷`) keep the
/// tape from hash-consing the pair into a `Cse`, which would refuse the
/// body for an unrelated reason and prove nothing.
fn factored_cancelling_body() -> String {
    let big = (1u64 << 27) as f64;
    format!(
        "o54\n3\n\
         o5\no2\nn{big:.1}\nv0\nn2\n\
         o5\no0\nv0\nv1\nn2\n\
         o16\no5\no2\nv0\nn{big:.1}\nn2\n"
    )
}

/// The off-diagonal form of the same thing, and the one that matters more:
/// `(10⁹x₀ + 10⁹x₁)² + (x₀ + x₁)² + (10⁹x₀ − 10⁹x₁)²`.
///
/// Every weight is positive — no difference of squares, nothing anyone
/// would call contrived — and `H(0, 1)` still folds to exactly zero and
/// vanishes. The diagonal mechanism needs a negative weight to cancel;
/// off the diagonal the `−10⁹x₁` inside a *squared* term supplies the sign,
/// so an ordinary badly scaled least-squares model reaches it.
fn factored_offdiagonal_body() -> &'static str {
    "o54\n3\n\
     o5\no0\no2\nn1e9\nv0\no2\nn1e9\nv1\nn2\n\
     o5\no0\nv0\nv1\nn2\n\
     o5\no1\no2\nn1e9\nv0\no2\nn1e9\nv1\nn2\n"
}

/// Parse with parse-time quadratic recognition on and off — the two arms of
/// `admitted_quad_form`, which reaches its form from a `Quad` body one way
/// and re-derives it from a `Tree` the other.
fn both_paths(txt: &str) -> [(&'static str, NlProblem); 2] {
    [
        (
            "recognizing",
            parse_nl_text_with_quadratic(txt, true).expect("parse (recognizing)"),
        ),
        (
            "trees",
            parse_nl_text_with_quadratic(txt, false).expect("parse (trees)"),
        ),
    ]
}

// ---------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------

/// The regression at the level the defect lives: a row that lost a term to
/// floating point is not admitted for evaluation, on either parse path.
///
/// Before the fix every one of these returned `Some(form)`, and the form was
/// short the cancelled coefficient.
#[test]
fn a_row_that_dropped_a_term_is_not_admitted_for_evaluation() {
    for (what, txt) in [
        ("cancellation", model(&cancelling_body())),
        ("partial cancellation", model(&partially_cancelling_body())),
        ("underflow", model(underflowing_body())),
    ] {
        for (path, prob) in both_paths(&txt) {
            assert!(
                prob.con_nonlinear[0].admitted_quad_form().is_none(),
                "{what} ({path}): a row with a dropped term was admitted for \
                 constant-structure evaluation",
            );
        }
    }
}

/// The same gate, against the second door gh #673 opened.
///
/// `admitted_factored_form` serves the bodies `admitted_quad_form` refuses
/// for their *shape*, and these are refused for their *loss* — but they are
/// also, read structurally, sums of squares: `2⁵³·x₀·x₀ + x₀² − 2⁵³·x₀·x₀`
/// is three of them. Without the explicit `is_expanded_quadratic` test in
/// that accessor they walk straight back onto the fast path this file
/// exists to keep them off.
///
/// It is worth knowing what goes wrong when they do, because it is not that
/// the fast path gets *worse*. Measured, with the gate removed: this row's
/// tape answers `16.0` at `x₀ = 3` and the factored arm answers `9.0` —
/// and `9.0` is the right value of `x₀²`, which the compensated outer sum
/// (gh #702) recovers where the tape's naive fold cannot.
/// `the_cli_answer_does_not_depend_on_the_dropped_form` below then moves
/// from `−1.812` to `−2.236`, which is `−√5`: the true optimum of the model
/// these bytes describe.
///
/// That is still the defect. As this file's own doc comment puts it, the
/// tape is the reference because it is what the row means *to this solver*,
/// not because it is exact — and two routes over the same bytes answering
/// `9` and `16` is exactly the divergence gh #685 is about, whichever of
/// them is closer to the algebra. The Hessian **pattern** parts company
/// too: `Σ 2wₖbₖbₖᵀ` folds `(0, 0)` to exactly `0.0` (`2⁵⁴ + 2` ties back
/// to `2⁵⁴`) and a zero entry is not stored, while the tape declares one —
/// `nnz_h` 2 → 1.
#[test]
fn a_row_that_dropped_a_term_is_not_admitted_as_a_factored_form_either() {
    for (what, txt) in [
        ("cancellation", model(&cancelling_body())),
        ("partial cancellation", model(&partially_cancelling_body())),
        ("underflow", model(underflowing_body())),
    ] {
        for (path, prob) in both_paths(&txt) {
            assert!(
                prob.con_nonlinear[0].admitted_factored_form().is_none(),
                "{what} ({path}): a row with a dropped term came back in \
                 through the factored read-out",
            );
        }
    }
}

/// The partial-cancellation model is the one an emptiness test would let
/// through, and this is why: its term map is not empty and its *degree*
/// answer is correct. Only the read-out is wrong.
#[test]
fn partial_cancellation_is_not_caught_by_looking_for_an_empty_form() {
    let txt = model(&partially_cancelling_body());
    for (path, prob) in both_paths(&txt) {
        assert_eq!(
            prob.con_nonlinear[0].provably_affine(),
            Some(false),
            "{path}: the surviving x₁² makes this row provably degree 2 — if \
             this ever becomes None the model no longer separates the two gates",
        );
        assert!(
            prob.con_nonlinear[0].admitted_quad_form().is_none(),
            "{path}: a non-empty but incomplete form was admitted",
        );
    }
}

/// The gate is on the *loss*, not on the drop (gh #687): a row whose term
/// cancelled exactly is admitted, because its read-out is the whole body.
/// On the pre-#687 code this row was refused along with the absorbing one,
/// and the row went back to the tape for nothing.
#[test]
fn an_exactly_cancelled_row_is_still_admitted_for_evaluation() {
    for (path, prob) in both_paths(&model(exactly_cancelling_body())) {
        let form = prob.con_nonlinear[0]
            .admitted_quad_form()
            .unwrap_or_else(|| panic!("{path}: an exactly cancelling row lost its fast path"));
        let (h, _lin, _c) = form;
        // The cancelled `x₀²` is absent because it is absent, and `x₁²` is
        // there because it is there.
        assert_eq!(h.get(&(0, 0)), None, "{path}");
        assert_eq!(h.get(&(1, 1)), Some(&2.0), "{path}");
    }

    // And the fast path it keeps agrees with the tape it replaces.
    let x = [3.0, 1.0];
    let g = |quad: bool| -> f64 {
        let prob =
            parse_nl_text_with_quadratic(&model(exactly_cancelling_body()), true).expect("parse");
        let mut t = NlTnlp::try_new_with_quadratic(prob, quad).expect("build TNLP");
        let m = t.get_nlp_info().expect("nlp info").m as usize;
        let mut out = vec![0.0; m];
        assert!(t.eval_g(&x, true, &mut out), "eval_g failed");
        out[0]
    };
    assert_eq!(g(true), g(false), "the admitted form is not the row");
}

/// The gate is on the drop, not on the shape: an ordinary `x₀² + x₁²` is
/// still admitted, so Q4 keeps the reach it was written for.
#[test]
fn an_ordinary_quadratic_row_is_still_admitted() {
    for (path, prob) in both_paths(&model(ordinary_body())) {
        assert!(
            prob.con_nonlinear[0].admitted_quad_form().is_some(),
            "{path}: an honest degree-2 row lost its constant-structure \
             evaluation",
        );
    }
}

// ---------------------------------------------------------------------
// What the gate is protecting
// ---------------------------------------------------------------------

/// The claim `admitted_quad_form` makes is that the form evaluates the row,
/// so the row's own tape is what checks it: `g₀` from the default route and
/// `g₀` from `POUNCE_DBG_NO_QUAD`'s route have to agree.
///
/// Before the fix the default route answered `0` for the cancelling model
/// where the tape answered `16`, and `5` for the partially cancelling one
/// where the tape answered `14` (at `x₀ = 3`, `x₁ = √5`... here `x = (3, 1)`,
/// so `9` against `10`).
#[test]
fn the_fast_path_and_the_tape_agree_on_the_row() {
    for (what, txt) in [
        ("cancellation", model(&cancelling_body())),
        ("partial cancellation", model(&partially_cancelling_body())),
        ("ordinary", model(ordinary_body()).to_string()),
    ] {
        let x = [3.0, 1.0];
        let g = |quad: bool| -> f64 {
            let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");
            let mut t = NlTnlp::try_new_with_quadratic(prob, quad).expect("build TNLP");
            let m = t.get_nlp_info().expect("nlp info").m as usize;
            let mut out = vec![0.0; m];
            assert!(t.eval_g(&x, true, &mut out), "{what}: eval_g failed");
            out[0]
        };
        let (fast, tape) = (g(true), g(false));
        assert_eq!(
            fast, tape,
            "{what}: the constant-structure evaluator disagrees with the tape \
             on row 0 at x = {x:?}",
        );
    }
}

/// And the same for the row's Jacobian, which is the quantity Q6 then goes
/// on to freeze: a form missing a term is missing that term's derivative
/// too, which is the shape that turns into a wrong `Optimal`.
#[test]
fn the_fast_path_and_the_tape_agree_on_the_row_jacobian() {
    for (what, txt) in [
        ("cancellation", model(&cancelling_body())),
        ("partial cancellation", model(&partially_cancelling_body())),
    ] {
        let x = [3.0, 1.0];
        let row0 = |quad: bool| -> Vec<f64> {
            let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");
            let mut t = NlTnlp::try_new_with_quadratic(prob, quad).expect("build TNLP");
            let nnz = t.get_nlp_info().expect("nlp info").nnz_jac_g as usize;
            let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
            assert!(t.eval_jac_g(
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut irow,
                    jcol: &mut jcol,
                },
            ));
            let mut vals = vec![0.0; nnz];
            assert!(t.eval_jac_g(
                Some(&x),
                true,
                SparsityRequest::Values { values: &mut vals },
            ));
            let mut out = vec![0.0; 2];
            for k in 0..nnz {
                if irow[k] == 0 {
                    out[jcol[k] as usize] += vals[k];
                }
            }
            out
        };
        assert_eq!(
            row0(true),
            row0(false),
            "{what}: the constant-structure evaluator disagrees with the tape \
             on ∂g₀/∂x at x = {x:?}",
        );
    }
}

// ---------------------------------------------------------------------
// The reported symptom
// ---------------------------------------------------------------------

/// The reported reproduction, rebuilt: `min x₀` subject to `body ≤ 5`, a
/// nonlinear second row to keep the classifier on the general NLP route,
/// and a floor of `-10⁶` under `x₀`.
///
/// The true model is `min x₀ s.t. x₀² + x₁² ≤ 5`, whose optimum is
/// `x₀ = −√5 ≈ −2.236` at `x₁ = 0`. Read row 0 without its `x₀²` and `x₀`
/// is unconstrained: the solve walks to its bound and reports `−10⁶`,
/// `Optimal`, with no warning of any kind.
///
/// Row 1 is `sin(x₁) ≤ 2`, which every point satisfies. It is there to be
/// nonlinear — an all-quadratic model classifies QP and leaves the route
/// this test is about — and slack so that it contributes nothing else.
fn reported_model(body: &str) -> String {
    format!(
        "g3 0 1 0\n\
         2 2 1 0 0\n\
         2 0\n\
         0 0\n\
         2 1 1\n\
         0 0 0 1\n\
         0 0 0 0 0\n\
         3 1\n\
         0 0\n\
         0 0 0 0 0\n\
         C0\n{body}\
         C1\n\
         o41\n\
         v1\n\
         O0 0\n\
         n0\n\
         x2\n\
         0 1.0\n\
         1 1.0\n\
         r\n\
         1 5.0\n\
         1 2.0\n\
         b\n\
         2 -1000000.0\n\
         3\n\
         k1\n\
         2\n\
         J0 2\n\
         0 0\n\
         1 0\n\
         J1 1\n\
         1 0\n\
         G0 1\n\
         0 1.0\n"
    )
}

/// End to end, through the CLI, on the default route: the answer the solver
/// reports must not depend on whether Q4's constant-structure evaluator
/// stood in for row 0.
///
/// `POUNCE_DBG_NO_QUAD=1` is the branch's own oracle — the same bytes down
/// the AD tape, which is where these rows were evaluated before Q4 — so the
/// two runs are an A/B on exactly the substitution this gate governs. After
/// the fix they agree because the row is no longer substituted; before it,
/// the fast path evaluated row 0 as identically zero, the `≤ 5` stopped
/// constraining anything, and the solve walked `x₀` to its `-10⁶` floor and
/// called it `Optimal`.
///
/// Note what is *not* claimed here. A body whose coefficients cancel in the
/// recognizer's fp arithmetic also cancels in the tape's, so neither route
/// computes the mathematical `x₀² + x₁²`: at `x₀ = 3` the tape gives `16`
/// where the algebra gives `9`. The tape is the reference because it is what
/// the row means *to this solver*, not because it is exact. What gh #685
/// reports is the six-order-of-magnitude gap between the two, and that is
/// what this asserts away.
#[test]
fn the_cli_answer_does_not_depend_on_the_dropped_form() {
    let dir = std::env::temp_dir().join("pounce_issue_685");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");

    for (what, body) in [
        ("cancellation", cancelling_body()),
        ("partial cancellation", partially_cancelling_body()),
    ] {
        let path = dir.join(format!("{}.nl", what.replace(' ', "_")));
        std::fs::write(&path, reported_model(&body)).expect("write model");

        // A cancelling row is genuinely ill-conditioned on either route, so
        // cap the iterations rather than wait out a thrash: the divergence
        // this is watching for shows up in the first few.
        let run = |no_quad: bool| -> f64 {
            let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
            cmd.arg(&path).arg("max_iter=200");
            if no_quad {
                cmd.env("POUNCE_DBG_NO_QUAD", "1");
            }
            let out = cmd.output().expect("run pounce");
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            objective(&text)
                .unwrap_or_else(|| panic!("{what} (no_quad={no_quad}): no objective in:\n{text}"))
        };
        let (fast, tape) = (run(false), run(true));
        assert_eq!(
            fast, tape,
            "{what}: the default route reports {fast} where the row's own \
             tape reports {tape} — Q4 substituted a form that had lost a term",
        );
    }
}

/// The unscaled objective from the end-of-run summary block.
fn objective(text: &str) -> Option<f64> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))?;
    line.split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .next_back()
}

/// gh #711 review, the merge-blocking finding: a **factored** body that
/// loses a Hessian entry to floating point must not be evaluated from its
/// structure either.
///
/// The gate `admitted_factored_form` carries is `is_expanded_quadratic`,
/// which answers about the *shape* the coefficients came from. It catches a
/// lossy flat sum of monomials. It has nothing to say about three honest
/// squares whose `Σ 2wₖbₖbₖᵀ` cancels — and before the fix, both bodies
/// below went straight onto the fast path and disagreed with their own
/// tapes:
///
/// | | fast path | tape |
/// |---|---|---|
/// | `eval_g[0]` at `x = (3, 1)` | `16.0` | `0.0` |
/// | `nnz_h` pattern | `[(1,0), (1,1)]` | `[(0,0), (1,0), (1,1)]` |
///
/// The fix refuses in `push_factored_form`, where the accumulation is, so
/// the body keeps its tape. This asserts the property that matters rather
/// than the mechanism: whatever the routing, the two routes agree.
#[test]
fn a_factored_row_that_dropped_a_hessian_entry_keeps_its_tape() {
    for (what, txt) in [
        ("diagonal", model(&factored_cancelling_body())),
        ("off-diagonal", model(factored_offdiagonal_body())),
    ] {
        let x = [3.0, 1.0];
        let mut answers = Vec::new();
        for quad in [true, false] {
            let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");
            let mut t = NlTnlp::try_new_with_quadratic(prob, quad).expect("tnlp");

            let mut g = [0.0; 2];
            assert!(t.eval_g(&x, true, &mut g), "{what}: eval_g failed");

            let nnz = t.get_nlp_info().expect("info").nnz_h_lag as usize;
            let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
            assert!(
                t.eval_h(
                    None,
                    true,
                    1.0,
                    None,
                    true,
                    SparsityRequest::Structure {
                        irow: &mut irow,
                        jcol: &mut jcol,
                    },
                ),
                "{what}: eval_h structure failed",
            );
            let pattern: Vec<(i32, i32)> = irow.iter().copied().zip(jcol.iter().copied()).collect();
            answers.push((g[0], pattern));
        }

        let (fast, tape) = (&answers[0], &answers[1]);
        assert_eq!(
            fast.0, tape.0,
            "{what}: the fast path and the tape disagree on the row's value",
        );
        assert_eq!(
            fast.1, tape.1,
            "{what}: the fast path and the tape disagree on the Hessian pattern",
        );
    }
}

/// The refusal above is about *loss*, not about factored bodies with a
/// cancelling Hessian in general. gh #687's rule holds on this arm too: a
/// term that cancels **exactly** did not go missing, and refusing it would
/// give up the fast path for arithmetic that never rounded.
///
/// `(x₀ + x₁)² − (x₀ − x₁)²` has `H(0, 0) = 2 − 2` and `H(1, 1) = 2 − 2`,
/// both exactly zero by an add that rounded nothing, and its true Hessian
/// really is `[[0, 4], [4, 0]]`. It stays on the fast path, and the two
/// routes agree — which is the same assertion as the test above, made
/// against the opposite expectation.
#[test]
fn an_exactly_cancelling_factored_row_is_still_admitted() {
    // (x0 + x1)^2 - (x0 - x1)^2
    let body = "o54\n2\n\
                o5\no0\nv0\nv1\nn2\n\
                o16\no5\no1\nv0\nv1\nn2\n";
    let txt = model(body);

    let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");
    assert!(
        prob.con_nonlinear[0].admitted_factored_form().is_some(),
        "an exactly cancelling factored row should still be recognized",
    );

    let x = [3.0, 1.0];
    let mut values = Vec::new();
    for quad in [true, false] {
        let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");
        let mut t = NlTnlp::try_new_with_quadratic(prob, quad).expect("tnlp");
        let mut g = [0.0; 2];
        assert!(t.eval_g(&x, true, &mut g));
        values.push(g[0]);
    }
    // 4·x₀·x₁ = 12.
    assert_eq!(values[0], 12.0);
    assert_eq!(values[0], values[1]);
}
