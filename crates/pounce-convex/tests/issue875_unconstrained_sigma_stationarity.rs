//! Regression for gh #875: on an **unconstrained** ill-conditioned QP the
//! cost-normalized (`σ`) path returned a materially wrong `x` under `Optimal`,
//! because the componentwise guard gh #846 installed had no rows to run over.
//!
//! This is pre-existing — present at `v0.10.0` — and adjacent to gh #846 but
//! not caused by `b545172b` and not closed by it, which is why it has its own
//! file rather than a case in `issue846_sigma_flat_direction.rs`.
//!
//! # The gap
//!
//! `sigma_complementarity_is_genuine` loops over `prob.m_ineq()` and over the
//! finite entries of `lb`/`ub`. With **neither** it reaches the end and
//! returns `true` having executed no test at all, so
//! `normalized_optimum_is_genuine` fell back to the aggregate
//! `unscaled_relative_kkt <= cut` arm alone — the exact aggregate the
//! componentwise guard exists to stop trusting. The fix is the sibling arm
//! `sigma_stationarity_is_genuine`, which asks the same question of the
//! stationarity rows; those exist whatever the constraint count is.
//!
//! # The instance
//!
//! Two variables and nothing else: `min ½(x₀−3)² + ½·10¹²(x₁−½)²`, so
//! `x* = (3, ½)` **by identity**, with no solver in the loop and no oracle.
//! The `σ` path returned `x₀ = 0.027` in one iteration — wrong by `2.97` on a
//! coordinate whose optimum is `3` — and printed `Dual infeasibility
//! 5.50e+01` on the line *above* `EXIT: Optimal Solution Found.`, so the
//! report contradicted itself. `qp_hsde=no` and Ipopt both return `(3, ½)`.
//!
//! # The aggregate is blind twice over
//!
//! At that returned point row 1 carries `|r₁| = 55.0` against its own scale
//! `5e11` — relative `1.1e-10`, genuinely converged — while row 0 carries
//! `|r₀| = 2.97` against its own scale `3`, a relative `0.99` and completely
//! unconverged. The reported `‖r‖∞` is `55.0`, i.e. row **1**'s, and over
//! `gscale = 5e11` it reads `1.1e-10` and sails through. The row that is
//! wrong is invisible in the aggregate both because its residual is not the
//! largest and because its scale is not the largest.
//!
//! # What this file is NOT evidence about
//!
//! **Coupled (non-separable) instances.** The componentwise denominator is
//! the largest term that built the row, which is a *directional* scale, not a
//! reduced one — the same distinction CLAUDE.md draws for the sensitivity
//! classifier, one crate over. On a diagonal `P` row `i`'s residual is
//! `eᵢ(xᵢ − tᵢ)` against a scale of order `eᵢ|tᵢ|`, so the ratio *is* the
//! relative error in `xᵢ` and the test is exact. Rotate the same spectrum and
//! the stiff mode appears in every row, every denominator collapses back to
//! the aggregate, and the arm rejects nothing. Measured over a 72-instance
//! unconstrained census (`cond` `1e2 ‥ 1e12` × magnitude `1e-3 ‥ 1e3` × `n`
//! ∈ {2, 5} × rotated or not), this fix takes claimed-optimal-but-wrong from
//! **32/72 to 17/72**, and the split is total: all 15 repairs are separable
//! instances, and every rotated instance is bit-identical to the baseline
//! (same `x`, same iteration count, same reported dual infeasibility).
//!
//! The coupled half was a **larger and different** defect, not a weaker form of
//! this one, and it is gh #880 — fixed separately, in
//! `tests/issue880_coupled_sigma_forward_error.rs`, which carries the full
//! census table. What closed it is not a better ratio but a different *kind* of
//! quantity: `sigma_forward_error_is_small` measures the affine-scaling Newton
//! step `‖Δ‖∞` — a norm of a vector, so basis-free, where any per-row ratio
//! (this file's included) is not. Over the same 72-instance census the two
//! fixes together take claimed-optimal-but-wrong from **32/72 → 17/72 → 9/72**
//! (the first arrow is against the census's original `x* = t` reference; the
//! second is against the exact rational one gh #882 replaced it with, under
//! which this file's own result still reads 17/72). The remaining 9 are all at
//! `cond ≥ 1e10` and all real — but eight of them are smaller than `ε·cond`,
//! the arithmetic floor of the `f64` estimator that would have to reject them,
//! so what is left is out of reach of a double-precision guard rather than
//! out of reach of a tighter threshold.
//!
//! So a green run of *this* file still does not cover the coupled arm, for the
//! same reason it never did — every fixture below is separable, and the
//! componentwise rule it pins is exactly the one that goes blind under a
//! rotation. Read the two files as covering the two halves, and add a coupled
//! case to #880's file rather than here.
//!
//! **The fixture corpus.** `scripts/sweep-fixtures.sh` moves **zero** of 180
//! fixture-legs across this change, which per CLAUDE.md is the expected
//! result and not evidence: exactly one fixture in the corpus reaches the `σ`
//! path at all. The corpus is a no-collateral-damage check here and nothing
//! more.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min Σ ½ k·eᵢ(xᵢ − tᵢ)²` with **no constraints and no bounds**, written as
/// `½xᵀPx + cᵀx` with `P = diag(k·e)` and `cᵢ = −k·eᵢtᵢ`. Diagonal, so the
/// exact minimiser is `t` itself — by identity, independent of `k`, and with
/// no oracle in the loop.
fn unconstrained_qp(e: &[f64], t: &[f64], k: f64) -> (QpProblem, Vec<f64>) {
    (
        QpProblem {
            n: e.len(),
            p_lower: e
                .iter()
                .enumerate()
                .map(|(i, &v)| Triplet::new(i, i, k * v))
                .collect(),
            c: e.iter().zip(t).map(|(&v, &ti)| -k * v * ti).collect(),
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; e.len()],
            ub: vec![f64::INFINITY; e.len()],
        },
        t.to_vec(),
    )
}

fn rel_x_err(x: &[f64], exact: &[f64]) -> f64 {
    let scale = exact.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

/// The reported instance, verbatim: `min ½(x₀−3)² + ½·10¹²(x₁−½)²`.
const EIG: [f64; 2] = [1.0, 1e12];
const TGT: [f64; 2] = [3.0, 0.5];

/// The headline. The issue reports `x₀ = 0.027039` against a true `3.0` —
/// `|x − x*|∞ = 2.973` — reported `Optimal` in one iteration.
#[test]
fn the_reported_unconstrained_qp_is_solved_at_the_default_tolerance() {
    let (prob, exact) = unconstrained_qp(&EIG, &TGT, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal, "the instance is trivial");
    let err = rel_x_err(&sol.x, &exact);
    assert!(
        err < 1e-6,
        "x = {:?} off the closed form {exact:?} by {err:.3e} relative \
         (the issue reports x₀ = 0.027039 against 3.0)",
        sol.x
    );
}

/// The status line and the metric printed beside it have to agree. The issue's
/// sharpest observation is that they did not: `Dual infeasibility 5.4976e+01`
/// sat one line above `EXIT: Optimal Solution Found.`, and the JSON carried
/// `final_kkt_error = 54.97` next to `status = 'SolveSucceeded'`. A report
/// that contradicts itself is the cheapest possible place to catch this, so
/// it is asserted directly rather than only through `x`.
///
/// **The aggregate `‖r‖∞` cannot be asserted tightly here, and that is the
/// whole point of the issue.** On this instance it has a hard floor of
/// `‖P‖∞·ε·|x₁| ≈ 1.1e-4`: `x₁ = ½` is representable, `10¹²·½` is not
/// perturbable by less than an ulp, so no correct solve can drive row 1 below
/// that. The defect produced `55`, five orders above the floor, so the loose
/// bar below still separates it — but the sharp assertion is the
/// **componentwise** one after it, which is exactly the measure the fix
/// installed. `P` is diagonal, so both are arithmetic.
#[test]
fn optimal_is_not_reported_beside_a_large_dual_infeasibility() {
    let (prob, _) = unconstrained_qp(&EIG, &TGT, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let r = sol.kkt_residuals(&prob);
    assert!(
        r.dual_infeasibility < 1e-2,
        "reported Optimal with dual infeasibility {:.4e} (the issue reports \
         5.4976e+01 printed directly above `EXIT: Optimal Solution Found.`; \
         the representation floor on this instance is 1.1e-4)",
        r.dual_infeasibility
    );
    // Componentwise, against the largest term that built each row. At the
    // reported failure this reads 0.99 for row 0 and 1.1e-10 for row 1: the
    // guilty row is the one whose residual is *not* the largest.
    for i in 0..EIG.len() {
        let (px, pt) = (EIG[i] * sol.x[i], EIG[i] * TGT[i]);
        let (r_i, d_i) = (px - pt, px.abs().max(pt.abs()));
        assert!(
            r_i.abs() <= 1e-6 * d_i,
            "row {i}: |r| = {:.4e} against its own scale {d_i:.4e}, ratio \
             {:.3e} (the issue reports 0.99 for row 0 while the aggregate \
             read 1.1e-10)",
            r_i.abs(),
            r_i.abs() / d_i
        );
    }
}

/// Tightening `tol` was not a workaround: the issue reports `1e-4` through
/// `1e-10` all returning `0.027039`, and `1e-12` taking 42 iterations to
/// return `2.809257` — still wrong, and wrong in a different direction. A
/// spot check at one tolerance is worthless on this path, because
/// `hsde_cost_scale` reads `tol` and so tightening it *pulls* models in.
#[test]
fn every_tolerance_is_correct_not_just_the_ends() {
    for tol in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
        let (prob, exact) = unconstrained_qp(&EIG, &TGT, 1.0);
        let opts = QpOptions {
            tol,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "tol {tol:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        assert!(
            err < 1e-6,
            "tol {tol:.0e}: x off by {err:.3e} relative (the issue reports \
             0.027039 from 1e-4 through 1e-10 and 2.809257 at 1e-12)"
        );
    }
}

/// The issue's condition-number table: the error is a clean function of `cond`
/// — `3e-4` at `1e6`, `3e-2` at `1e8`, `1.5` at `1e10`, `3.0` at `1e12`, where
/// `3.0` means `x₀ ≈ 0` against a true `3`, i.e. 100% wrong — while the
/// **relative objective** error stays at `1e-11 ‥ 1e-16` throughout. That last
/// number is the point: an objective-parity check rates every row of this
/// table solved, which is the blind spot CLAUDE.md names.
#[test]
fn the_condition_number_sweep_is_correct_in_x_not_just_in_the_objective() {
    for cond in [1e4, 1e6, 1e8, 1e10, 1e12] {
        let (prob, exact) = unconstrained_qp(&[1.0, cond], &TGT, 1.0);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        assert!(
            err < 1e-6,
            "cond {cond:.0e}: x off by {err:.3e} relative (the issue reports \
             3e-4 / 3e-2 / 1.5 / 3.0 at 1e6 / 1e8 / 1e10 / 1e12)"
        );
    }
}

/// Multiplying the objective by `k > 0` leaves the argmin unchanged **by
/// identity**, and the issue reports the error is independent of `k` — so this
/// is a contradiction with no oracle at all. The sweep spans `σ` off (small
/// `k`: the gate is `max(‖P‖∞, ‖c‖∞)·ε > tol`, which on this instance turns on
/// at `k ≈ 4.5e-5`) through `σ` on by ten decades, so the "below the gate" and
/// "above it" branches are both asserted.
///
/// The bar is `k`-dependent **because the stopping test is absolute and the
/// flat curvature is `k`**. Row 0's residual is `k(x₀ − 3)`, so a solve that
/// stops at `‖r‖∞ ≤ tol` pins `x₀` no better than `tol/k` — `1e-2` at
/// `k = 1e-6`. Asserting a flat `1e-6` there would be asserting something no
/// correct solve can deliver, so the bar is `max(3e-6, 10·tol/k)`. It still
/// separates the defect by orders at every `k`, because the defect did not
/// scale with `k` at all: it plateaued at `|x₀ − 3| ≈ 2.97` throughout, which
/// is the issue's "independent of coefficient magnitude" claim.
#[test]
fn an_inert_objective_rescaling_stays_inert() {
    let tol = QpOptions::default().tol;
    for k in [1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8] {
        let (prob, exact) = unconstrained_qp(&EIG, &TGT, k);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "k = {k:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        let bar = 3e-6_f64.max(10.0 * tol / k);
        assert!(
            err < bar,
            "k = {k:.0e}: x off by {err:.3e} relative, bar {bar:.3e} \
             (the defect plateaus at 9.9e-1 regardless of k)"
        );
    }
}

/// The **other branch** of the new rule, per CLAUDE.md's rule that a fixture
/// which always takes one branch says nothing about the other.
///
/// Everything above reaches `sigma_stationarity_is_genuine`'s *reject* branch.
/// A guard that rejected unconditionally would pass every one of them and be a
/// pure cost: each reject buys one un-normalized re-solve. So this asserts the
/// *accept* branch still accepts — a well-conditioned unconstrained QP is
/// solved to the same accuracy and in no more iterations than it was before
/// the arm existed. The baseline numbers are the same binary with the conjunct
/// removed, measured at the fix commit.
#[test]
fn a_well_conditioned_unconstrained_qp_is_not_newly_rejected() {
    for cond in [1.0, 1e1, 1e2, 1e3] {
        let (prob, exact) = unconstrained_qp(&[1.0, cond], &TGT, 1.0);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        assert!(rel_x_err(&sol.x, &exact) < 1e-8, "cond {cond:.0e}");
        assert!(
            sol.iters <= 20,
            "cond {cond:.0e} took {} iterations; a rejected certificate costs \
             an un-normalized re-solve, and a guard that rejects a \
             well-conditioned model pays that on every solve",
            sol.iters
        );
    }
}

/// The arm must not fire where there is nothing wrong to find: an
/// unconstrained QP that is ill-conditioned but whose `σ` answer is already
/// right must keep it. `P = I` scaled up is perfectly conditioned but large
/// enough to engage `σ` (`max(‖P‖∞, ‖c‖∞)·ε > tol`), which is the one shape
/// where the gate is on and the answer was never in doubt.
#[test]
fn sigma_is_engaged_but_a_conditioned_answer_survives_it() {
    let (prob, exact) = unconstrained_qp(&[1e9, 1e9, 1e9], &[3.0, 0.5, -2.0], 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    // `1e-8`, not machine precision: `tol` is `1e-8` and `P = 1e9·I`, so the
    // stopping test licenses `|Δx| ≈ tol/1e9`; the measured error is `4.3e-10`.
    assert!(
        rel_x_err(&sol.x, &exact) < 1e-8,
        "x = {:?} against {exact:?}",
        sol.x
    );
}
