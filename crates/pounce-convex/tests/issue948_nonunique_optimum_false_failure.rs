//! gh#948 — a non-unique optimum was reported as `numerical_failure`, with
//! all-zero multipliers, on a QP whose primal point was optimal to machine
//! precision.
//!
//! ## The chain, because the reported symptom is three layers from the cause
//!
//! `model_step_cap` computes the model's own minimizer along the search
//! direction, `α* = −slope/curv`. Below the curvature floor the model is at
//! best linear along `p`, and the cap was `+∞` whenever `slope < 0.0` — a bare
//! sign test on an accumulated inner product `pᵀ(Hx + g)`.
//!
//! On a direction along which the model is **flat**, the exact slope is zero,
//! so what that sum returns is round-off of order `eps·‖p‖·‖r‖` whose sign is
//! a coin flip. Half of those read as descent and returned `+∞`. The
//! active-set loop's recession-ray branch takes an infinite cap as proof of
//! unboundedness *without* consulting `ray_is_unbounded_descent` — whose own
//! descent clause is scale-relative and would have thrown it out — so the
//! engine returned `Unbounded`. Every `Unbounded` return fills its multipliers
//! with zeros by construction. `pounce_convex`'s driver then did exactly the
//! right thing with a claim it could not certify: it declined to propagate the
//! unboundedness, fell back on the returned point, and re-derived the KKT
//! error — which, at `z = 0`, is `‖c‖`. Hence `numerical_failure`, CLI exit 1,
//! and `Overall NLP error` equal to `‖c‖`.
//!
//! So the status and the zero duals in the report are both *consequences*. The
//! defect is one bare sign test.
//!
//! ## Why a flat direction is the whole population
//!
//! A flat direction is what a **non-unique optimum** is: an LP whose cost is
//! parallel to a constraint normal has an optimal face rather than a vertex,
//! and a QP with rank-deficient `P` is flat along `null(P)`.
//!
//! The issue's own statistics are the signature of a round-off sign test, and
//! are why the fix is relative rather than a wider tolerance: **integer data,
//! where the tie is exact and the round-off identically zero, never failed**;
//! float data carrying ulp noise around the same tie failed about half the
//! time; and a `1e-8` perturbation that makes the optimum unique took 51 of
//! 100 failures to 0 of 100.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_active_set,
    solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(p: &QpProblem) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set(
        p,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
}

fn ipm(p: &QpProblem) -> QpSolution {
    solve_qp_ipm(p, &QpOptions::default(), backend)
}

fn dense(rows: &[&[f64]]) -> Vec<Triplet> {
    let mut t = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            if v != 0.0 {
                t.push(Triplet::new(i, j, v));
            }
        }
    }
    t
}

fn lower(rows: &[&[f64]]) -> Vec<Triplet> {
    let mut t = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            if j <= i && v != 0.0 {
                t.push(Triplet::new(i, j, v));
            }
        }
    }
    t
}

/// `‖c + Gᵀz + Aᵀy‖∞` — stationarity for a `P = 0` problem, which is the
/// residual the zero multipliers violated. Reported in the issue as
/// "at `z = 0`, stationarity `c + G'z = c ≠ 0`".
fn stationarity_lp(prob: &QpProblem, sol: &QpSolution) -> f64 {
    let mut r = prob.c.clone();
    for t in &prob.g {
        r[t.col] += t.val * sol.z[t.row];
    }
    for t in &prob.a {
        r[t.col] += t.val * sol.y[t.row];
    }
    r.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// The LP from the issue: `c` is, to rounding, `-1.0837 · G[2]`, so the
/// optimal set is an edge of row 2 rather than a vertex.
///
/// Oracle: `scipy.optimize.linprog` (HiGHS) reports `-2.6237240275913822`, and
/// the POUNCE IPM agrees to 10 digits with `z = [~0, ~0, 1.0837]`.
#[test]
fn the_lp_with_an_edge_optimum_is_solved_not_failed() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.4286197740541566, -0.41791584441169793],
        a: vec![],
        b: vec![],
        g: dense(&[
            &[1.8267565599574231, -3.0783319101980338],
            &[0.9580639753088469, 0.06963722766094482],
            &[1.3182500241810684, 0.385629249998389],
        ]),
        h: vec![4.072533868670402, 1.927008361538642, 2.421025052034367],
        lb: vec![],
        ub: vec![],
    };
    let sol = active_set(&prob);

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "gh#948: a non-unique optimum was reported as {:?}; obj={:?} z={:?}",
        sol.status,
        sol.obj,
        sol.z
    );
    // HiGHS's objective, to the last digit it published.
    assert!(
        (sol.obj - -2.6237240275913822).abs() < 1e-9,
        "objective {:?} against the HiGHS oracle -2.6237240275913822",
        sol.obj
    );
    // The multiplier is the substantive half of the report: `z = 0` is not a
    // rounding slip, it is a vector that violates stationarity outright.
    assert!(
        stationarity_lp(&prob, &sol) < 1e-8,
        "multipliers must satisfy stationarity; residual {:e} at z={:?}",
        stationarity_lp(&prob, &sol),
        sol.z
    );
    assert!(
        (sol.z[2] - 1.0837244434548188).abs() < 1e-6,
        "the binding row's multiplier should match the IPM's 1.0837; got {:?}",
        sol.z
    );
    // The POINT is deliberately not pinned. The optimal set is an edge, so
    // the two engines legitimately return different points on it — the IPM's
    // is `[1.8148, 0.0744]` and the active-set engine's is `[1.6918, 0.4949]`.
    // Asserting a point here would be asserting which of two correct answers
    // an engine happens to reach.
}

/// The QP from the issue: rank-1 PSD `P` and one equality row, built from a
/// chosen `(x*, y*)` via `c = −(P x* + Aᵀ y*)`, so the optimum is flat along
/// `null(P)`.
///
/// The equality multiplier is what makes this case unambiguous: a `y` of zero
/// on an equality row cannot be a rounding artefact of a nearly-inactive
/// constraint, because an equality row is always active.
#[test]
fn the_rank_deficient_qp_is_solved_not_failed() {
    let prob = QpProblem {
        n: 3,
        p_lower: lower(&[
            &[4.165350860035665, -5.215905630264575, 0.8533059309117347],
            &[-5.215905630264575, 6.531423752282118, -1.0685206022098608],
            &[0.8533059309117347, -1.0685206022098608, 0.17480664563342632],
        ]),
        c: vec![9.829203142125198, -8.441553002215453, 2.343515884887553],
        a: dense(&[&[
            -0.5677696061279298,
            -0.45264929211044586,
            -0.2155971630897659,
        ]]),
        b: vec![1.438408240200998],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = active_set(&prob);

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "gh#948: reported as {:?}; obj={:?} y={:?}",
        sol.status,
        sol.obj,
        sol.y
    );
    assert!(
        (sol.obj - -12.352227511919441).abs() < 1e-6,
        "objective {:?} against the closed-form oracle -12.352227511919441",
        sol.obj
    );
    assert!(
        (sol.y[0] - 3.3229995166325903).abs() < 1e-6,
        "the equality multiplier should match the IPM's 3.323, not 0; got {:?}",
        sol.y
    );
}

/// A deterministic LCG, so the population below is reproducible without
/// taking a dependency. Numerically it only has to produce *float* data:
/// the defect needs ulp-level noise around an exact tie, which any
/// non-integer generator supplies and integer data does not.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 53 bits to [0, 1), then to [-1, 1).
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn next_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + ((self.next_f64().abs() * (hi - lo + 1) as f64) as usize).min(hi - lo)
    }
}

/// The population, not the two instances: 100 LPs with a non-unique optimum
/// by construction, none of which may fail.
///
/// Construction, following the issue: draw `G` and a strictly positive `h` (so
/// `x = 0` is feasible), pick a row `k`, and set `c = −μ·G[k]` for `μ > 0`.
/// Then `min cᵀx = −μ · max G[k]·x`, which row `k` itself caps at `h[k]` — so
/// every instance is feasible and bounded, and its optimal set is the face
/// `G[k]·x = h[k]`, a face and not a point for `n ≥ 2`. That face is exactly
/// the flat direction the bare sign test misread.
///
/// The IPM is the oracle, as it is in the issue.
///
/// **Measured on THIS population: 9 of 100 fail before the fix, 0 after.**
/// The issue reports 51 of 100, but that is its own KKT-by-construction
/// family, which is more adversarial than the single-row tie built here; its
/// looser "mixed random convex QPs" figure is about 5%, which this brackets.
/// The number to keep is the one this file can reproduce, so the assertion
/// message quotes 9 rather than borrowing 51.
#[test]
fn a_population_of_non_unique_optima_never_reports_a_false_failure() {
    let mut rng = Lcg(0x9481_9481_9481_9481);
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for case in 0..100 {
        let n = rng.next_usize(2, 6);
        let m = rng.next_usize(2, 6);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for _ in 0..m {
            rows.push((0..n).map(|_| rng.next_f64()).collect());
        }
        // Strictly positive, so the origin is interior and the LP is feasible.
        let h: Vec<f64> = (0..m).map(|_| 1.0 + rng.next_f64().abs()).collect();
        let k = rng.next_usize(0, m - 1);
        let mu = 0.5 + rng.next_f64().abs();
        let c: Vec<f64> = rows[k].iter().map(|v| -mu * v).collect();

        let g: Vec<Triplet> = rows
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                r.iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0.0)
                    .map(move |(j, &v)| Triplet::new(i, j, v))
            })
            .collect();

        let prob = QpProblem {
            n,
            p_lower: vec![],
            c,
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![],
            ub: vec![],
        };

        let reference = ipm(&prob);
        // Only instances the oracle itself certifies are evidence about the
        // active-set arm; an instance the IPM cannot solve says nothing here.
        if reference.status != QpStatus::Optimal {
            continue;
        }
        checked += 1;

        let sol = active_set(&prob);
        if sol.status != QpStatus::Optimal {
            failures.push(format!(
                "case {case} (n={n}, m={m}, tie on row {k}): {:?}, obj={:?} \
                 against the IPM's {:?}, stationarity {:e}",
                sol.status,
                sol.obj,
                reference.obj,
                stationarity_lp(&prob, &sol)
            ));
            continue;
        }
        let scale = reference.obj.abs().max(1.0);
        if (sol.obj - reference.obj).abs() > 1e-6 * scale {
            failures.push(format!(
                "case {case}: objective {:?} against the IPM's {:?}",
                sol.obj, reference.obj
            ));
        }
    }

    assert!(
        checked >= 80,
        "the construction should produce solvable instances; only {checked} \
         of 100 were certified by the oracle, so this test is not measuring \
         what it claims"
    );
    assert!(
        failures.is_empty(),
        "{} of {checked} non-unique optima failed. This population puts 9 \
         failures on the bare-sign-test build and 0 on the fixed one, so a \
         non-empty list here is the gh#948 defect and not a tolerance \
         quibble:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
