//! The Q4 constant-structure evaluator against the AD tape it replaces
//! (gh #588).
//!
//! Q4 stops building a tape for every objective and row the recognizer
//! proves is degree ≤ 2, and evaluates `g`, `∇g` and `∇²L` from the constant
//! matrix instead. If those two ways of computing a derivative ever disagree,
//! POUNCE hands the algorithm a wrong Hessian and converges somewhere else —
//! or fails to — on a model no fixture solves. The fixture sweep cannot see
//! that: it only reaches rows a *solve* depends on, and it reports one number
//! per model rather than one per matrix entry.
//!
//! So this is the differential check, modelled on Q3's
//! `quad_recognizer_differential.rs`: every `.nl` file in the repository is
//! loaded **twice** — once with the fast path, once with
//! `NlTnlp::try_new_with_quadratic(prob, false)`, which is what
//! `POUNCE_DBG_NO_QUAD=1` selects — and the two are compared entry by entry
//! at several points, on:
//!
//! * `eval_g` — the line search's inner loop;
//! * `eval_jac_g` — compared as a `(row, col) -> value` map, because the two
//!   paths need not agree on the *pattern*: a coefficient that cancels to
//!   exactly zero is dropped by the recognizer and kept by the tape's
//!   structural sparsity;
//! * `eval_h` — same, over the assembled lower triangle.
//!
//! ### Which comparisons are bitwise
//!
//! `∇²L` is: the tape's entry for `0.5·((c·xᵢ)·xⱼ)` is the product of
//! constants `0.5·c` and the decode adds `w·(0.5·c)`, while the scatter
//! computes `w·q_val` with the same `q_val`. Nothing about that is an
//! *approximation* of the other. **Measured, that is right for the shape the
//! note reasoned about and wrong in general**: over this corpus 27 174 of
//! 27 176 assembled Hessian entries are bit-identical, and the two that are
//! not (`eigena2.nl`, entry (8, 8)) differ by exactly **one ulp**.
//!
//! The mechanism is worth stating because it decides where else it can
//! happen: **an entry written by both paths at once**. `eigena2` has 55
//! quadratic rows and a non-quadratic objective, so `∇²L[8, 8]` takes a
//! constant contribution from a row and an `x`-dependent one from the
//! objective's tape. On the tape path both land in the same compressed
//! column pass and reach `values` as one add; on the fast path the row's
//! share is scattered first and the objective's decode adds on top. Same
//! terms, different association, one ulp — and only where a model mixes the
//! two, which is why the entry moves with `x` even though the row's Hessian
//! does not. A model whose objective and rows are all recognized (the whole
//! `qcqp` family) has no entry with a foot in both camps.
//!
//! The disagreement is therefore bounded, not asserted away: the worst ulp
//! distance over the corpus is pinned at `MAX_HESS_ULPS`. Q1's 2-ulp line is
//! why that is a pin and not a tolerance — a one-ulp coefficient difference
//! moved a fixture from 17 to 12 conic iterations, and only a differential
//! check saw it.
//!
//! `f` and `∇f` are compared too, and that is not decoration: on a model
//! whose *rows* are not quadratic the objective is the only thing the phase
//! touches, and `infeasible_equalities.nl` (cubic rows, quadratic objective)
//! is exactly such a model. Adding those two comparisons immediately found
//! the phase's one real accuracy defect — expanding `(x − 500000)²` cancels
//! five digits where the tape squares a small residual — which is why the
//! fast path is now gated on `is_expanded_quadratic` and takes only forms
//! the writer had already expanded.
//!
//! `g` and `∇g` are **not** bitwise and the design note says so in advance:
//! the tape sums one summand at a time in file order while the matvec sums a
//! merged row, so the association differs. They are held to a tight relative
//! tolerance, and the *observed* worst deviation over the corpus is pinned as
//! a number so that a future regression has to move it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pounce_cli::nl_quadratic::recognize_expr;
use pounce_cli::nl_reader::{
    BinOp, Expr, NlProblem, NlProblemParts, NlTnlp, UnaryOp, read_nl_file,
};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

/// Relative tolerance for the two summation orders in `eval_g` / `eval_jac_g`.
///
/// Not a bound anyone derived — a ceiling well above what the corpus
/// actually produces (see `WORST_OBSERVED_REL`, asserted separately), set
/// loose enough that reassociating a long sum cannot trip it and tight
/// enough that a wrong coefficient cannot hide under it.
const REL_TOL: f64 = 1e-12;

/// The largest relative deviation any fixture actually produces, over every
/// `g` and `∇g` entry at every probe point. Pinned so that "within
/// tolerance" stays a measurement rather than a hope.
const WORST_OBSERVED_REL: f64 = 1e-14;

/// The worst Hessian disagreement the corpus produces, in representable
/// doubles. See the module docs: the design note forecast bit-identity here
/// and the corpus refutes it in general while confirming it on the shape the
/// note reasoned about.
const MAX_HESS_ULPS: u64 = 1;

// ---------------------------------------------------------------------
// Probing one model both ways
// ---------------------------------------------------------------------

/// A deterministic xorshift, so a failure is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // [-2, 2), which keeps `x` away from the all-equal points where a
        // sign error in a cross term cancels itself.
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 * 4.0 - 2.0
    }
}

/// The probe points: the model's own starting point, then pseudo-random
/// perturbations of it. `x0` alone is not enough — a quadratic row at a
/// point where half the variables are zero hides half its coefficients.
fn probe_points(prob: &NlProblem, k: usize) -> Vec<Vec<f64>> {
    let mut out = vec![prob.x0.clone()];
    let mut rng = Rng(0x5eed_1234_9e37_79b9);
    for _ in 0..k {
        out.push(prob.x0.iter().map(|&v| v + rng.next_f64()).collect());
    }
    out
}

/// Sparse triplets as a map, so two paths with different *patterns* can
/// still be compared entry by entry.
fn as_map(irow: &[i32], jcol: &[i32], values: &[f64]) -> BTreeMap<(i32, i32), f64> {
    let mut m = BTreeMap::new();
    for k in 0..values.len() {
        m.insert((irow[k], jcol[k]), values[k]);
    }
    m
}

/// Worst deviation between two values, relative to the reference — but
/// never to a scale smaller than `floor`.
///
/// `floor = 0.0` is the pure relative measure the corpus is held to (with
/// an absolute fallback when the reference is zero). The synthetic battery
/// passes `floor = 1.0` instead, because its rows are `O(1)` by
/// construction and a cancelling sum of `O(1)` terms lands on a residue of
/// `1e-17` on one path and exactly `0.0` on the other: that is a relative
/// deviation of 1 and an absolute deviation of nothing, and pinning it as
/// the former would only be pinning the reassociation the module docs open
/// by admitting to.
///
/// A probe point is random, so it can land outside a model's domain — a
/// `sqrt` of something negative, a `log` of zero. Both paths then produce
/// `NaN` and that is agreement, not a difference; **one** of them producing
/// `NaN` is the loudest possible disagreement and comes back infinite.
fn rel_dev(a: f64, b: f64, floor: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if a == b {
        return 0.0;
    }
    if a == 0.0 && floor == 0.0 {
        return b.abs();
    }
    ((a - b) / a.abs().max(floor)).abs()
}

/// Bit equality, with the same `NaN` convention as [`rel_dev`]: two `NaN`s
/// agree even if their payloads differ, because neither path promises a
/// particular quiet-`NaN` bit pattern.
fn bit_equal(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    a.to_bits() == b.to_bits()
}

/// Everything one probe of one model reports.
#[derive(Default)]
struct Report {
    /// Models where the fast path was actually taken.
    models_with_quadratic: usize,
    /// Constraint rows (over all models) routed through a form.
    quadratic_rows: usize,
    /// Worst relative deviation seen on `g` or `∇g`.
    worst_rel: f64,
    /// Hessian entries compared, and how many were not bit-identical.
    hess_entries: usize,
    hess_bit_diffs: usize,
    worst_hess_ulps: u64,
    worst_hess_where: String,
    /// `eval_f` and `eval_grad_f` values compared, and how many were not
    /// bit-identical.
    obj_entries: usize,
    obj_bit_diffs: usize,
}

/// Distance in representable doubles between two finite values of the same
/// sign. Used only to *bound* a disagreement, so mixed signs and non-finite
/// values come back saturated rather than being reasoned about.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

fn compare_model(path: &Path, rep: &mut Report) {
    let Ok(prob) = read_nl_file(path) else { return };
    compare_problem(&path.display().to_string(), prob, 0.0, rep);
}

/// The comparison itself, over a problem from wherever — a `.nl` file or the
/// synthetic battery below.
fn compare_problem(name: &str, prob: NlProblem, floor: f64, rep: &mut Report) {
    // Recognition is what decides whether this model exercises anything; a
    // model with no quadratic part builds two identical TNLPs and the
    // comparison is vacuous but free.
    let n = prob.n;
    let m = prob.m;
    if n == 0 {
        return;
    }
    let points = probe_points(&prob, 3);

    let Ok(mut fast) = NlTnlp::try_new_with_quadratic(prob.clone(), true) else {
        return;
    };
    let Ok(mut slow) = NlTnlp::try_new_with_quadratic(prob.clone(), false) else {
        return;
    };

    let quad_rows = (0..m).filter(|&i| fast.quadratic_row(i)).count();
    let quad_obj = fast.quadratic_objective();
    if quad_rows == 0 && !quad_obj {
        return;
    }
    rep.models_with_quadratic += 1;
    rep.quadratic_rows += quad_rows;

    // The Jacobian and Hessian patterns are asked for once each; they do not
    // depend on `x`.
    let (fast_jac, slow_jac) = (
        structure(&mut fast, Kind::Jac),
        structure(&mut slow, Kind::Jac),
    );
    let (fast_h, slow_h) = (
        structure(&mut fast, Kind::Hess),
        structure(&mut slow, Kind::Hess),
    );

    for (p, x) in points.iter().enumerate() {
        // --- eval_f / eval_grad_f ---
        // The objective is on the fast path independently of the rows, and
        // is the only thing that moves on a model whose *rows* are not
        // quadratic — which is how `infeasible_equalities` (cubic rows,
        // quadratic objective) turned up in the fixture sweep.
        let (ff, fs) = (
            fast.eval_f(x, true).expect("eval_f (fast)"),
            slow.eval_f(x, true).expect("eval_f (tape)"),
        );
        let d = rel_dev(fs, ff, floor);
        rep.worst_rel = rep.worst_rel.max(d);
        assert!(
            d <= REL_TOL,
            "{name}: probe {p}: eval_f disagrees: tape {fs:?} vs quad {ff:?} (rel {d:.3e})"
        );
        rep.obj_entries += 1;
        if !bit_equal(ff, fs) {
            rep.obj_bit_diffs += 1;
        }

        let (mut gradf, mut grads) = (vec![0.0; n], vec![0.0; n]);
        assert!(
            fast.eval_grad_f(x, true, &mut gradf),
            "{name}: eval_grad_f (fast)"
        );
        assert!(
            slow.eval_grad_f(x, true, &mut grads),
            "{name}: eval_grad_f (tape)"
        );
        for j in 0..n {
            let d = rel_dev(grads[j], gradf[j], floor);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: grad_f[{j}] disagrees: tape {:?} vs quad {:?} (rel {d:.3e})",
                grads[j],
                gradf[j]
            );
            rep.obj_entries += 1;
            if !bit_equal(gradf[j], grads[j]) {
                rep.obj_bit_diffs += 1;
            }
        }

        // --- eval_g ---
        let (mut gf, mut gs) = (vec![0.0; m], vec![0.0; m]);
        assert!(fast.eval_g(x, true, &mut gf), "{name}: eval_g (fast)");
        assert!(slow.eval_g(x, true, &mut gs), "{name}: eval_g (tape)");
        for i in 0..m {
            let d = rel_dev(gs[i], gf[i], floor);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: eval_g row {i} disagrees: tape {:?} vs quad {:?} (rel {d:.3e})",
                gs[i],
                gf[i]
            );
        }

        // --- eval_jac_g ---
        let mut vf = vec![0.0; fast_jac.0.len()];
        let mut vs = vec![0.0; slow_jac.0.len()];
        assert!(
            fast.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut vf }),
            "{name}: eval_jac_g (fast)"
        );
        assert!(
            slow.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut vs }),
            "{name}: eval_jac_g (tape)"
        );
        let jf = as_map(&fast_jac.0, &fast_jac.1, &vf);
        let js = as_map(&slow_jac.0, &slow_jac.1, &vs);
        for key in jf.keys().chain(js.keys()) {
            let (a, b) = (
                js.get(key).copied().unwrap_or(0.0),
                jf.get(key).copied().unwrap_or(0.0),
            );
            let d = rel_dev(a, b, floor);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: jac {key:?} disagrees: tape {a:?} vs quad {b:?} (rel {d:.3e})"
            );
        }

        // --- eval_h ---
        // Multipliers that are neither all-ones nor all-equal: a sign or an
        // index error in the λ-weighting survives both of those.
        let obj_factor = 0.75;
        let lambda: Vec<f64> = (0..m).map(|i| 1.0 + (i % 7) as f64 * 0.5).collect();
        let mut hf = vec![0.0; fast_h.0.len()];
        let mut hs = vec![0.0; slow_h.0.len()];
        assert!(
            fast.eval_h(
                Some(x),
                true,
                obj_factor,
                Some(&lambda),
                true,
                SparsityRequest::Values { values: &mut hf }
            ),
            "{name}: eval_h (fast)"
        );
        assert!(
            slow.eval_h(
                Some(x),
                true,
                obj_factor,
                Some(&lambda),
                true,
                SparsityRequest::Values { values: &mut hs }
            ),
            "{name}: eval_h (tape)"
        );
        let mf = as_map(&fast_h.0, &fast_h.1, &hf);
        let ms = as_map(&slow_h.0, &slow_h.1, &hs);
        for key in mf.keys().chain(ms.keys()) {
            let (a, b) = (
                ms.get(key).copied().unwrap_or(0.0),
                mf.get(key).copied().unwrap_or(0.0),
            );
            rep.hess_entries += 1;
            if !bit_equal(a, b) {
                rep.hess_bit_diffs += 1;
                let u = ulp_distance(a, b);
                if u > rep.worst_hess_ulps {
                    rep.worst_hess_ulps = u;
                    rep.worst_hess_where = format!("{name}: probe {p}: hessian {key:?}");
                }
            }
        }
    }
}

enum Kind {
    Jac,
    Hess,
}

fn structure(t: &mut NlTnlp, kind: Kind) -> (Vec<i32>, Vec<i32>) {
    let info = t.get_nlp_info().expect("nlp info");
    let nnz = match kind {
        Kind::Jac => info.nnz_jac_g,
        Kind::Hess => info.nnz_h_lag,
    } as usize;
    let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
    let req = SparsityRequest::Structure {
        irow: &mut irow,
        jcol: &mut jcol,
    };
    let ok = match kind {
        Kind::Jac => t.eval_jac_g(None, true, req),
        Kind::Hess => t.eval_h(None, true, 1.0, None, true, req),
    };
    assert!(ok, "structure request declined");
    (irow, jcol)
}

// ---------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------

fn all_fixtures() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "nl") {
                out.push(p);
            }
        }
    }
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut out = Vec::new();
    walk(&base.join("fixtures"), &mut out);
    walk(&base.join("fixtures_issue_49"), &mut out);
    out.sort();
    out
}

#[test]
fn every_quadratic_fixture_evaluates_the_same_both_ways() {
    let fixtures = all_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected the fixture corpus, found {} files",
        fixtures.len()
    );
    let mut rep = Report::default();
    for f in &fixtures {
        compare_model(f, &mut rep);
    }

    // A floor, not a target: if a refactor ever leaves this walking two
    // models it should fail rather than pass vacuously.
    assert!(
        rep.models_with_quadratic >= 20,
        "the corpus should exercise the fast path on many models, got {}",
        rep.models_with_quadratic
    );
    assert!(
        rep.hess_entries >= 1_000,
        "too few Hessian entries compared: {}",
        rep.hess_entries
    );
    eprintln!(
        "[quad differential] {} models, {} quadratic rows, {} hessian entries \
         ({} not bit-identical, worst {} ulp at {}), worst g/jac rel deviation {:.3e}",
        rep.models_with_quadratic,
        rep.quadratic_rows,
        rep.hess_entries,
        rep.hess_bit_diffs,
        rep.worst_hess_ulps,
        rep.worst_hess_where,
        rep.worst_rel
    );
    eprintln!(
        "[quad differential] objective: {} values compared, {} not bit-identical",
        rep.obj_entries, rep.obj_bit_diffs
    );
    assert!(
        rep.worst_hess_ulps <= MAX_HESS_ULPS,
        "hessian disagreement grew past what the corpus produced: {} ulp at {}",
        rep.worst_hess_ulps,
        rep.worst_hess_where
    );
    assert!(
        rep.hess_bit_diffs * 1000 <= rep.hess_entries,
        "too many Hessian entries stopped being bit-identical: {} of {}",
        rep.hess_bit_diffs,
        rep.hess_entries
    );
    assert!(
        rep.worst_rel <= WORST_OBSERVED_REL,
        "g/jac deviation grew past what the corpus produced: {:.3e} > {WORST_OBSERVED_REL:.0e}",
        rep.worst_rel
    );
}

/// A model with nothing quadratic in it must be untouched — same structures,
/// same values, bit for bit, on both constructions. This is what makes the
/// phase's blast radius statable: it is exactly the recognized set.
#[test]
fn a_model_with_no_quadratic_part_is_byte_identical_on_both_paths() {
    let mut checked = 0usize;
    for f in all_fixtures() {
        let Ok(prob) = read_nl_file(&f) else { continue };
        if prob.n == 0 {
            continue;
        }
        let Ok(mut fast) = NlTnlp::try_new_with_quadratic(prob.clone(), true) else {
            continue;
        };
        if fast.quadratic_objective() || (0..prob.m).any(|i| fast.quadratic_row(i)) {
            continue;
        }
        let Ok(mut slow) = NlTnlp::try_new_with_quadratic(prob.clone(), false) else {
            continue;
        };
        let (a, b) = (
            structure(&mut fast, Kind::Hess),
            structure(&mut slow, Kind::Hess),
        );
        assert_eq!(a, b, "{}: Hessian pattern moved", f.display());
        let x = prob.x0.clone();
        let (mut gf, mut gs) = (vec![0.0; prob.m], vec![0.0; prob.m]);
        assert!(fast.eval_g(&x, true, &mut gf));
        assert!(slow.eval_g(&x, true, &mut gs));
        for i in 0..prob.m {
            assert!(
                bit_equal(gf[i], gs[i]),
                "{}: row {i} moved on a model with nothing quadratic",
                f.display()
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 5,
        "expected some non-quadratic models, got {checked}"
    );
}

// ---------------------------------------------------------------------
// A synthetic battery
// ---------------------------------------------------------------------
//
// The corpus is two tests wide and made of models AMPL and Pyomo wrote, so
// it exercises the *shapes those writers emit* — which is the hole gh #683
// sat in: the recognizer and the evaluator agreed on every fixture while
// disagreeing on a shape no fixture contains. Q3's
// `quad_recognizer_differential` closes the same hole for the recognizer
// with a 4 000-expression battery, but it compares two implementations of
// the same floating-point arithmetic, so it cannot see a form on which the
// coefficients themselves are the wrong answer. This one compares the fast
// path against the **tape** — a genuinely different computation of the same
// derivative — over expressions the writers do not produce.

/// Expressions are built over this many variables. Small, so that a
/// four-term body has a real chance of putting two terms on one monomial
/// and exercising a merge.
const BATTERY_VARS: usize = 4;

/// The worst relative deviation the battery actually produces, pinned the
/// same way [`WORST_OBSERVED_REL`] pins the corpus. It is looser than the
/// corpus number because a random sum of monomials is not as well
/// conditioned as a model somebody meant.
const BATTERY_WORST_REL: f64 = 1e-14;

/// One monomial: a coefficient, or a coefficient times one or two
/// variables, spelled every way `is_monomial` admits (`Mul`, `Pow`, `Div`,
/// `Neg`, and a `Cse` wrapper, which is what a `V`-segment reference
/// becomes).
///
/// Coefficients stay within a few orders of magnitude on purpose. The
/// ill-scaled case — where the sum of a row's coefficients is catastrophic
/// and the two paths part company by more than any tolerance — is a real
/// defect on the *storage* side (see the `lost_terms` skip below), not
/// something for this battery to rediscover once per seed.
fn battery_monomial(rng: &mut Rng2) -> Expr {
    fn c(rng: &mut Rng2) -> Expr {
        const COEFS: [f64; 8] = [1.0, -1.0, 2.0, -3.0, 0.5, -0.25, 4.0, 0.125];
        Expr::Const(COEFS[rng_below(rng, 8) as usize])
    }
    fn v(rng: &mut Rng2) -> Expr {
        Expr::Var(rng_below(rng, BATTERY_VARS as u64) as usize)
    }
    let coef = c(rng);
    let body = match rng_below(rng, 6) {
        // `c`
        0 => return coef,
        // `c · xᵢ`
        1 => Expr::Binary(BinOp::Mul, Box::new(coef), Box::new(v(rng))),
        // `c · xᵢ · xⱼ`
        2 => Expr::Binary(
            BinOp::Mul,
            Box::new(coef),
            Box::new(Expr::Binary(BinOp::Mul, Box::new(v(rng)), Box::new(v(rng)))),
        ),
        // `xᵢ²`, the `Pow` spelling — a *different tape node* from `xᵢ·xᵢ`,
        // which is what keeps a repeated monomial from being hash-consed
        // into one node with one adjoint.
        3 => Expr::Binary(BinOp::Pow, Box::new(v(rng)), Box::new(Expr::Const(2.0))),
        // `xᵢ / c`
        4 => Expr::Binary(BinOp::Div, Box::new(v(rng)), Box::new(c(rng))),
        // `−(c · xᵢ · xⱼ)`
        _ => Expr::Unary(
            UnaryOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(coef),
                Box::new(Expr::Binary(BinOp::Mul, Box::new(v(rng)), Box::new(v(rng)))),
            )),
        ),
    };
    if rng_below(rng, 4) == 0 {
        Expr::Cse(std::sync::Arc::new(body))
    } else {
        body
    }
}

/// `w·(bᵀx + d)²` — a **squared affine form**, the leaf gh #673 taught the
/// fast path to keep factored rather than expand.
///
/// This battery predates that arm and emitted only flat sums of monomials,
/// so nothing random ever reached the factored read-out: all of its
/// coverage was the repo's 24 fixtures, which are real-writer models with
/// unit weights and well-scaled residuals. That is the same hole gh #683
/// sat in, and gh #711's review found a witness straight through it.
///
/// The base is `battery_affine`, so the forms inside a square are as varied
/// as the flat sums are: `Add`/`Sub`/`Neg` spines, constants on either
/// side, repeated variables.
fn battery_square(rng: &mut Rng2) -> Expr {
    const WEIGHTS: [f64; 6] = [1.0, -1.0, 2.0, -0.5, 3.0, 0.25];
    let terms = 1 + rng_below(rng, 3) as usize;
    let base = battery_affine(rng, terms);
    let square = Expr::Binary(BinOp::Pow, Box::new(base), Box::new(Expr::Const(2.0)));
    match rng_below(rng, 3) {
        // A bare `(…)²`, weight 1.
        0 => square,
        // `w · (…)²` — the weight on the left, where `peel_square` meets it
        // in a writer's output.
        1 => Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(WEIGHTS[rng_below(rng, 6) as usize])),
            Box::new(square),
        ),
        // `(…)² · w`, the other association, which folds in reverse order.
        _ => Expr::Binary(
            BinOp::Mul,
            Box::new(square),
            Box::new(Expr::Const(WEIGHTS[rng_below(rng, 6) as usize])),
        ),
    }
}

/// A degree-≤1 body, for use as the base of a square. Leaves that cannot
/// raise the degree only — a squared base of degree 2 would make the body
/// quartic and neither read-out would take it.
fn battery_affine(rng: &mut Rng2, terms: usize) -> Expr {
    fn leaf(rng: &mut Rng2) -> Expr {
        const COEFS: [f64; 6] = [1.0, -1.0, 2.0, -3.0, 0.5, 4.0];
        let c = Expr::Const(COEFS[rng_below(rng, 6) as usize]);
        let v = Expr::Var(rng_below(rng, BATTERY_VARS as u64) as usize);
        match rng_below(rng, 3) {
            0 => c,
            1 => v,
            _ => Expr::Binary(BinOp::Mul, Box::new(c), Box::new(v)),
        }
    }
    let mut acc = leaf(rng);
    for _ in 1..terms {
        acc = match rng_below(rng, 3) {
            0 => Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf(rng))),
            1 => Expr::Binary(BinOp::Sub, Box::new(acc), Box::new(leaf(rng))),
            _ => Expr::Unary(
                UnaryOp::Neg,
                Box::new(Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf(rng)))),
            ),
        };
    }
    acc
}

/// A sum whose leaves are monomials, squared affine forms, or both.
///
/// A body of pure monomials reaches the **expanded** read-out; one with a
/// square in it reaches the **factored** one (gh #673); one that mixes a
/// square with a cross term reaches neither and keeps its tape. All three
/// are the point — the assertion is that whatever route a body takes, it
/// agrees with its own tape.
///
/// The spine is randomized over `Sum`, `Add`, `Sub` and `Neg`, because both
/// gates admit all four and a battery that only ever emitted one would test
/// the gate rather than the arithmetic.
fn battery_body(rng: &mut Rng2, terms: usize) -> Expr {
    // One body in four may contain squares. Kept a minority so the expanded
    // arm — still the common case in real models — does not lose coverage.
    let squares = rng_below(rng, 4) == 0;
    fn leaf(rng: &mut Rng2, squares: bool) -> Expr {
        if squares && rng_below(rng, 2) == 0 {
            battery_square(rng)
        } else {
            battery_monomial(rng)
        }
    }
    let mut acc = leaf(rng, squares);
    let mut left = terms.saturating_sub(1);
    while left > 0 {
        acc = match rng_below(rng, 4) {
            0 => Expr::Sum(vec![acc, leaf(rng, squares), leaf(rng, squares)]),
            1 => Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf(rng, squares))),
            2 => Expr::Binary(BinOp::Sub, Box::new(acc), Box::new(leaf(rng, squares))),
            _ => Expr::Unary(
                UnaryOp::Neg,
                Box::new(Expr::Binary(
                    BinOp::Add,
                    Box::new(acc),
                    Box::new(leaf(rng, squares)),
                )),
            ),
        };
        left = left.saturating_sub(if matches!(acc, Expr::Sum(_)) { 2 } else { 1 });
    }
    acc
}

/// A deterministic xorshift for the battery. Separate from [`Rng`] because
/// that one yields probe coordinates and this one yields tree shapes; a
/// shared stream would couple the two.
struct Rng2(u64);

fn rng_below(rng: &mut Rng2, n: u64) -> u64 {
    rng.0 ^= rng.0 << 13;
    rng.0 ^= rng.0 >> 7;
    rng.0 ^= rng.0 << 17;
    rng.0 % n
}

/// The fast path and the tape, over expressions no `.nl` writer emitted.
///
/// A body whose recognized form has
/// [`lost_terms`](pounce_cli::nl_quadratic::Quad2::lost_terms) set is
/// **skipped**, and the skip is the honest part of this test. Such a body's
/// coefficients were rounded and then cancelled, or underflowed, on the way
/// into the form, so the fast path is evaluating a different function from
/// the tape — by design, and wrongly: `2⁵³·x² + x² − 2⁵³·x²` reads out as
/// the zero form while its own tape gives 16 at `x = 3`. That is the
/// storage-side half of gh #683, filed as gh #685; a battery that did not
/// skip it would be reporting that defect once per seed instead of guarding
/// this one.
///
/// The skip used to fire on any drop, which took **204 of 1 500** seeds out
/// of the comparison — most of them for cancellations that lost nothing
/// (`c·xᵢxⱼ − c·xᵢxⱼ` is a shape this battery emits readily, and every
/// coefficient it draws from is a power of two or a small integer). Since
/// gh #687 it fires on the *loss*, 7 seeds remain skipped, and the 197
/// problems that came back are compared like any other — at the same
/// tolerance, which they meet.
#[test]
fn a_synthetic_battery_evaluates_the_same_both_ways() {
    let mut rep = Report::default();
    let mut skipped = 0usize;
    let mut seeds_used = 0usize;

    let mut factored_bodies = 0usize;
    for seed in 1..=1_500u64 {
        let mut rng = Rng2(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let terms = 1 + rng_below(&mut rng, 5) as usize;
        let rows: Vec<Expr> = (0..3).map(|_| battery_body(&mut rng, terms)).collect();
        let objective = battery_body(&mut rng, terms);

        // The skip. Asked of the same recognizer the evaluator asks, so it
        // cannot drift out of step with what the fast path admits.
        let lost = |e: &Expr| recognize_expr(e).is_some_and(|q| q.lost_terms());
        if lost(&objective) || rows.iter().any(lost) {
            skipped += 1;
            continue;
        }

        let n = BATTERY_VARS;
        let m = rows.len();
        let prob = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective,
            obj_constant: 0.0,
            constraints: rows,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            // Away from the origin and from each other, so a probe grid
            // built around it separates terms a symmetric point would hide.
            x0: (0..n).map(|i| 0.7 + i as f64 * 0.3).collect(),
            g_l: vec![-1e19; m],
            g_u: vec![1.0; m],
            var_names: Vec::new(),
            con_names: Vec::new(),
        })
        .expect("assemble battery problem");
        // Coverage, asserted below rather than assumed: how many bodies
        // actually reached the factored read-out. A generator change that
        // stopped emitting squares would otherwise leave gh #673's arm
        // silently uncovered again, which is the state gh #711's review
        // found it in.
        for b in std::iter::once(&prob.obj_nonlinear).chain(prob.con_nonlinear.iter()) {
            if b.admitted_quad_form().is_none() && b.admitted_factored_form().is_some() {
                factored_bodies += 1;
            }
        }

        seeds_used += 1;
        compare_problem(&format!("battery seed {seed}"), prob, 1.0, &mut rep);
    }

    eprintln!(
        "[quad differential] battery: {seeds_used} problems built ({skipped} skipped for \
         lost terms), {} reached the fast path, {} quadratic rows, {} hessian entries \
         ({} not bit-identical, worst {} ulp at {}), worst g/jac rel deviation {:.3e}",
        rep.models_with_quadratic,
        rep.quadratic_rows,
        rep.hess_entries,
        rep.hess_bit_diffs,
        rep.worst_hess_ulps,
        rep.worst_hess_where,
        rep.worst_rel,
    );

    // gh #711: the battery must actually reach gh #673's arm. Before the
    // squared-affine leaf was added this counted **zero** — every factored
    // body the suite ever saw came from the 24 repo fixtures, which are
    // real-writer models with unit weights and well-scaled residuals. The
    // bound is loose on purpose: it pins that the arm is exercised, not how
    // often.
    eprintln!("[quad differential] battery: {factored_bodies} bodies took the factored arm");
    assert!(
        factored_bodies >= 100,
        "the battery stopped covering the factored read-out: only {factored_bodies} bodies \
         reached it (gh #673, gh #711)",
    );

    // Reach floors: a generator that stopped producing recognizable bodies
    // would otherwise pass this test by comparing nothing.
    assert!(
        rep.models_with_quadratic >= 500,
        "the battery stopped reaching the fast path: {} of {seeds_used}",
        rep.models_with_quadratic
    );
    assert!(
        rep.hess_entries >= 1_000,
        "too few Hessian entries compared: {}",
        rep.hess_entries
    );
    assert!(
        rep.worst_hess_ulps <= MAX_HESS_ULPS,
        "hessian disagreement grew past what the battery produced: {} ulp at {}",
        rep.worst_hess_ulps,
        rep.worst_hess_where
    );
    assert!(
        rep.worst_rel <= BATTERY_WORST_REL,
        "g/jac deviation grew past what the battery produced: {:.3e} > {BATTERY_WORST_REL:.0e}",
        rep.worst_rel
    );
}
