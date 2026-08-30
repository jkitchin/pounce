//! Regression for gh #846: the cost-normalized (`σ`) convex path returned a
//! materially wrong `x` under `Optimal` at the **default** tolerance, because
//! every test standing behind that verdict was an **aggregate**.
//!
//! # The instance
//!
//! A 6-variable **diagonal** box QP, `min Σ ½eᵢ(xᵢ − tᵢ)²` on `[-1, 1]` with
//! `e = [1e3, 3.98e4, 1.59e6, 6.31e7, 2.51e9, 1e11]` (condition number `1e8`).
//! It is separable, so `x* = clamp(t, -1, 1)` **with no solver in the loop** —
//! the oracle in this file is arithmetic, not another solver. The issue
//! additionally reports Ipopt 3.14.19, cvxpy/Clarabel at `1e-14`, cvxpy/SCS,
//! POUNCE's own `solver_selection=nlp` and its own `qp_hsde=no` all agreeing
//! with that closed form.
//!
//! At the default `tol = 1e-8` the path returned `x₀ = 0.831` against a true
//! `1.0`, and at `1e-6` it returned `0.039` — 96% wrong on a unit box — both
//! reporting `EXIT: Optimal Solution Found.`
//!
//! # Why every guard was blind
//!
//! The flat direction carries `1e3` in a spectrum reaching `1e11`, so:
//!
//! - **the objective cannot see it.** `-1.17834000816e10` against
//!   `-1.17834002580e10`: a relative objective error of `1.5e-8` for a **17%
//!   error in x₀**. An objective-parity check rates this problem solved, which
//!   is exactly the substitute CLAUDE.md records `4c02817d` making for the
//!   fixture sweep on this arm.
//! - **the aggregate KKT ratios cannot see it either.** Stationarity was
//!   normed by `gscale = ‖Px‖∞ ∨ ‖c‖∞ = 4.0e10` — a number belonging to the
//!   *stiffest* coordinate, used as the denominator for every other one — and
//!   complementarity by the same. `‖r‖∞ = 5.68` over `4.0e10` reads `2.7e-9`,
//!   comfortably inside the `100·tol = 1e-6` cut, so
//!   `normalized_optimum_is_genuine` accepted.
//!
//! The relative arm is now asked **one row at a time**, against the largest
//! term that built that row. The binding measure is complementarity: at the
//! returned point `x₀` sits `2.9e-3` off its bound carrying a multiplier of
//! `6.3e3` that insists it is on it, so componentwise the two factors read
//! `2.9e-3` and `1.0` against a `1e-6` cut, where the aggregate read
//! `4.5e-10`. One number is wrong by seven orders about the same point, and
//! the difference is entirely which denominator the row is held to.
//!
//! A companion arm over the **stationarity** rows was written and then
//! removed: on this family it rejects nothing complementarity does not, and
//! the same spectrum unconstrained, in a wide box, or under an equality row
//! comes back exact to `3e-16`. The failure needs an *active bound*, because
//! the slack it spends is bought by the embedding's objective-relative gap
//! test and a gap is spent on the bound multipliers.
//!
//! # `σ` is the amplifier, not the origin
//!
//! Rejecting the normalized certificate only moved the answer from `1.6e-1`
//! wrong to `2.9e-3` wrong, because the fallback is the *same embedding*
//! un-normalized, and the embedding's own stopping test normalizes the duality
//! gap by the objective's magnitude (`gap / (1 + |obj|)`). At `|obj| = 1.18e10`
//! that licenses an absolute gap of `tol·|obj| = 118`, which on curvature `1e3`
//! is `√(2·118/1e3) ≈ 0.49` in `x`. Measured across `mag = 1e7 ‥ 1e14` the
//! embedding degrades from `1e9` up while the direct driver behind Ruiz is
//! accurate at every magnitude, so a `σ` answer that fails the test twice now
//! falls through to that driver — and its answer is taken only if it passes
//! the same test.
//!
//! # Tolerance non-monotonicity
//!
//! `hsde_cost_scale` **reads `tol`**, so tightening `tol` *pulls* models into
//! the `σ` path. The reported table is `1e-4` correct → `1e-6` 96% wrong →
//! `1e-8` 17% wrong → `1e-14` correct, `Optimal` throughout. A spot check at
//! one tolerance is therefore worthless here, and `every_tolerance_is_correct`
//! is the test.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min Σ ½ k·eᵢ(xᵢ − tᵢ)²` on `[-1, 1]`, as `½xᵀPx + cᵀx` with
/// `P = diag(k·e)` and `cᵢ = −k·eᵢtᵢ`. Diagonal and separable, so the exact
/// minimiser is `clamp(t, −1, 1)` and does not depend on `k` at all.
fn box_qp(e: &[f64], t: &[f64], k: f64) -> (QpProblem, Vec<f64>) {
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
            lb: vec![-1.0; e.len()],
            ub: vec![1.0; e.len()],
        },
        t.iter().map(|v| v.clamp(-1.0, 1.0)).collect(),
    )
}

fn abs_x_err(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// The reported instance, verbatim.
const EIG: [f64; 6] = [1e3, 3.98e4, 1.59e6, 6.31e7, 2.51e9, 1e11];
const TGT: [f64; 6] = [1.5, -1.3, 0.3, -0.7, 2.0, -0.4];

/// The tolerance the issue reports as the headline failure. Two coordinates
/// are strictly inside the box and four are clamped to it, so the fixture
/// exercises both the active and the inactive complementarity branch.
#[test]
fn the_reported_box_qp_is_solved_at_the_default_tolerance() {
    let (prob, exact) = box_qp(&EIG, &TGT, 1.0);
    let opts = QpOptions::default();
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "the instance is trivial");
    let err = abs_x_err(&sol.x, &exact);
    assert!(
        err < 1e-6,
        "x = {:?} off the closed form {exact:?} by {err:.3e} \
         (the issue reports x₀ = 0.831 against 1.0, err 1.69e-1)",
        sol.x
    );
}

/// The whole reported tolerance table. `1e-4` and `1e-14` were already correct
/// and are here so the fix cannot be credited for a regime it does not touch,
/// nor regress it; `1e-6` and `1e-8` are the failures.
#[test]
fn every_tolerance_is_correct_not_just_the_ends() {
    for tol in [1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-14] {
        let (prob, exact) = box_qp(&EIG, &TGT, 1.0);
        let opts = QpOptions {
            tol,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "tol {tol:.0e}");
        let err = abs_x_err(&sol.x, &exact);
        assert!(
            err < 1e-6,
            "tol {tol:.0e}: x off by {err:.3e} (the issue reports 9.6e-1 at \
             1e-6 and 1.7e-1 at 1e-8)"
        );
    }
}

/// Multiplying the objective by `k > 0` leaves the argmin unchanged **by
/// identity** — it is the justification `solve_qp_core`'s own comment cites
/// for `σ` — so this is a contradiction with no oracle needed at all. The
/// issue reports one decade of this inert rescaling costing eight orders of
/// accuracy (`6.2e-8` at `k = 1e-1`, `8.0e-10` at `k = 1e0`, `8.0e-2` at
/// `k = 1e1`) and then *plateauing* at `8.2e-2 ‥ 1.85e-1` rather than
/// recovering, `Optimal` on every row.
///
/// The sweep spans `σ` off (`k ≤ 1e-4`, where `mag·ε ≤ tol` and the wrapper is
/// a no-op) through `σ` on by ten decades, so the "unchanged below the gate"
/// and "repaired above it" branches are both asserted here.
#[test]
fn an_inert_objective_rescaling_stays_inert() {
    for k in [
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14,
    ] {
        let (prob, exact) = box_qp(&EIG, &TGT, k);
        let opts = QpOptions::default();
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "k = {k:.0e}");
        let err = abs_x_err(&sol.x, &exact);
        assert!(
            err < 1e-6,
            "k = {k:.0e}: an argmin-invariant rescaling moved x by {err:.3e}"
        );
    }
}

/// The population, not the instance. The issue reports **84 of 157** members of
/// a generated family returning a wrong `x` under `Optimal` at the default
/// tolerance, worst `‖x − x*‖∞ = 5.72e-01` on a `[-1, 1]` box — so a single
/// fixture passing says very little.
///
/// The family below is deterministic (a fixed LCG, no `rand` dependency) and
/// sweeps the three axes the defect lives on: the dimension, the spread of the
/// spectrum (which decides how flat the flattest direction is relative to the
/// stiffest), and the top magnitude (which decides whether `σ` engages and how
/// hard). Targets straddle the box so each instance carries both clamped and
/// interior coordinates.
///
/// The spread stops at `1e8`, the reported instance's own condition number,
/// and that is a limit on the *fixture*, not on the fix. Past about `1e10` the
/// flattest curvature is small enough that `tol = 1e-8` does not pin `x` to
/// `1e-4` by any route: at `span = 11, top = 1e6` the direct driver — this
/// file's independent oracle — is itself `5.8e-5` off the closed form. A
/// fixture there would be measuring double precision, not this guard.
#[test]
fn the_generated_family_returns_the_closed_form_everywhere() {
    let mut seed = 0x846u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let mut worst = 0.0_f64;
    let mut worst_case = String::new();
    let mut checked = 0usize;
    for n in [2usize, 4, 6, 9] {
        for span in [3.0, 5.0, 7.0, 8.0] {
            for top in [1e6, 1e9, 1e11, 1e13] {
                let e: Vec<f64> = (0..n)
                    .map(|i| top * 10f64.powf(-span * (n - 1 - i) as f64 / (n - 1) as f64))
                    .collect();
                // Straddle the box: roughly half the targets outside it.
                let t: Vec<f64> = (0..n).map(|_| 4.0 * next() - 2.0).collect();
                let (prob, exact) = box_qp(&e, &t, 1.0);
                let opts = QpOptions::default();
                let sol = solve_qp_ipm(&prob, &opts, backend);
                assert_eq!(
                    sol.status,
                    QpStatus::Optimal,
                    "n={n} span={span} top={top:.0e}"
                );
                // The reference is the closed form; the *budget* is what an
                // independent driver reaches on the same data, floored at
                // `1e-5`. Neither number is a hedge: the worst member of this
                // family comes back at `9.9e-7`, so the floor carries a factor
                // of ten, and comparing against `qp_hsde=no` — the issue's own
                // oracle #6 — is what keeps the assertion honest where the
                // data itself is the limit rather than this guard.
                //
                // The floor used to be `1e-4`, for a residual that is now
                // fixed: `n=9, span=6, top=1e13` came back at `1.13e-5`
                // against the direct driver's `1.1e-16`, and the gap was not
                // the `σ` chain but gh #414's `verify_or_repair_optimum` one
                // layer above, which took the first *genuine* candidate rather
                // than the best one. It now ranks them by absolute KKT error,
                // and that member reaches `1.1e-16` too.
                let direct = solve_qp_ipm(
                    &prob,
                    &QpOptions {
                        use_hsde: false,
                        ..opts
                    },
                    backend,
                );
                let budget = (10.0 * abs_x_err(&direct.x, &exact)).max(1e-5);
                let err = abs_x_err(&sol.x, &exact) / budget;
                checked += 1;
                if err > worst {
                    worst = err;
                    worst_case = format!(
                        "n={n} span={span} top={top:.0e} t={t:?} \
                         (abs {:.3e}, direct {:.3e})",
                        abs_x_err(&sol.x, &exact),
                        abs_x_err(&direct.x, &exact)
                    );
                }
            }
        }
    }
    assert_eq!(checked, 64, "the family should not have shrunk");
    assert!(
        worst <= 1.0,
        "worst member is {worst:.2}x its budget ({worst_case}); the issue \
         reports 84/157 wrong, worst 5.72e-1"
    );
}

/// The `σ`-**accepted** branch, which every assertion above reaches only
/// through a *reject*. Per CLAUDE.md a fixture that always takes one branch is
/// no evidence about the other, and "always reject" would pass every test in
/// this file while turning `σ` into dead weight and paying three solves for
/// every large-coefficient QP.
///
/// A *well*-conditioned huge-coefficient QP is the accepting case: there is no
/// flat direction for an aggregate scale to hide, so the normalized solve is
/// genuinely accurate and the first gate lets it through. Asserted on the
/// iteration count, which is what a needless second and third solve would move.
#[test]
fn a_well_conditioned_huge_coefficient_qp_is_still_accepted_first_time() {
    let e = [1e11, 1.3e11, 0.7e11, 1.1e11];
    let t = [1.5, -1.3, 0.3, -0.7];
    let (prob, exact) = box_qp(&e, &t, 1.0);
    let opts = QpOptions::default();
    // Premise: `σ` really does engage here (`mag·ε > tol`).
    let mag = prob
        .p_lower
        .iter()
        .map(|t| t.val.abs())
        .chain(prob.c.iter().map(|v| v.abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        mag * f64::EPSILON > opts.tol,
        "premise: this instance must reach the σ path (mag = {mag:.3e})"
    );
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(abs_x_err(&sol.x, &exact) < 1e-6);
    let one_solve = solve_qp_ipm(
        &prob,
        &QpOptions {
            use_hsde: false,
            ..opts
        },
        backend,
    )
    .iters;
    assert!(
        sol.iters < 3 * one_solve,
        "a genuine normalized optimum must be accepted without paying the \
         two fallback solves: {} iterations against a single direct solve's {}",
        sol.iters,
        one_solve
    );
}

/// Why an objective-parity corpus is blind here, stated as a number rather
/// than as a warning. This is the assertion that would have to be *deleted*
/// for "check the objectives agree" to be a defensible substitute for checking
/// `x` on this arm.
#[test]
fn the_objective_cannot_see_the_error_that_x_can() {
    let (prob, exact) = box_qp(&EIG, &TGT, 1.0);
    // The objective at the true minimiser, and at a point wrong by the amount
    // the issue reports in the flat coordinate.
    let obj = |x: &[f64]| {
        let mut v = 0.0;
        for i in 0..prob.n {
            v += 0.5 * EIG[i] * x[i] * x[i] + prob.c[i] * x[i];
        }
        v
    };
    let mut wrong = exact.clone();
    wrong[0] = 0.831182;
    let rel = ((obj(&wrong) - obj(&exact)) / obj(&exact)).abs();
    assert!(
        rel < 1e-7,
        "premise: the objective must be nearly blind to this error, got {rel:.3e}"
    );
    let xerr = abs_x_err(&wrong, &exact);
    assert!(
        xerr > 1e-1,
        "premise: while x is wrong by a sixth of the box, got {xerr:.3e}"
    );
}

/// The `σ` chain's last-resort fallback must **not** fire on a cone-carrying
/// problem, and this is the branch every other test in this file misses.
///
/// `direct_driver_fallback` enters at `solve_qp_ipm_core`, which is an
/// **orthant-only** door: Ruiz is a row scaling and that function's own
/// comment records that SOC/exp/power problems reach the solver through
/// `solve_socp_ipm` instead. Handing it a QCQP silently reads `h − Gx ∈ K` as
/// `h − Gx ≥ 0` row by row, so what comes back is the correct answer to a
/// *different, looser* problem — and it is a claimed `Optimal`, so the chain
/// would happily prefer it.
///
/// This is not hypothetical. The first draft of the gh #846 fix had no cone
/// gate, and `scripts/sweep-fixtures.sh` moved exactly one fixture on both
/// legs: `qcqp_columns_illcond`, `-364.2102538 → -210.5328764` at
/// `SolveSucceeded`, where `solver_selection=nlp` and `qp_hsde=no` — two
/// independent routes — both put the optimum at `-364.2102`. Every orthant
/// fixture in this file stayed green throughout, because none of them takes
/// this branch. With the gate the sweep diff is empty across all 158
/// fixture-legs.
///
/// # The oracle
///
/// `min ½s‖x‖² − s·gᵀx` over `‖x‖ ≤ 1` has the closed form `x* = g/‖g‖`
/// whenever `‖g‖ ≥ 1`, at objective `s(½ − ‖g‖)` — arithmetic, no solver.
/// With `g = (1.5, 0.4)` and `s = 1e11` that is `x* = (0.96625, 0.25767)` and
/// `-1.05524e11`. Dropping the cone to its rows instead gives the
/// *unconstrained* `x = g` at `-1.205e11`, five percent lower and outside the
/// ball, so the two readings are plainly distinguishable — which is what makes
/// this a test rather than a formality.
#[test]
fn a_cone_carrying_problem_never_reaches_the_orthant_fallback() {
    use pounce_convex::{ConeSpec, solve_socp_ipm};

    let s = 1e11;
    let g = [1.5, 0.4];
    // `[t; x₀; x₁] ∈ SOC(3)` with `t = x₂`, plus `t ≤ 1`.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, s), Triplet::new(1, 1, s)],
        c: vec![-s * g[0], -s * g[1], 0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 2, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(2, 1, -1.0),
            Triplet::new(3, 2, 1.0),
        ],
        h: vec![0.0, 0.0, 0.0, 1.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(3), ConeSpec::Nonneg(1)];
    let opts = QpOptions::default();

    // Premise: this instance really does reach the `σ` path.
    let mag = prob
        .p_lower
        .iter()
        .map(|t| t.val.abs())
        .chain(prob.c.iter().map(|v| v.abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        mag * f64::EPSILON > opts.tol,
        "premise: σ must engage here (mag = {mag:.3e})"
    );

    let sol = solve_socp_ipm(&prob, &cones, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let gnorm = g[0].hypot(g[1]);
    let exact = [g[0] / gnorm, g[1] / gnorm];
    let norm = sol.x[0].hypot(sol.x[1]);
    // Asserted on the cone itself and not only on the objective, because
    // "respects the cone" is the property at stake and an orthant reading
    // leaves the ball rather than merely scoring differently.
    assert!(
        norm <= 1.0 + 1e-6,
        "the returned point left the second-order cone: ‖(x₀,x₁)‖ = {norm:.6}          (an orthant reading lands at ‖g‖ = {gnorm:.6})"
    );
    let err = abs_x_err(&sol.x[..2], &exact);
    assert!(
        err < 1e-6,
        "x = {:?} off the closed form {exact:?} by {err:.3e}",
        &sol.x[..2]
    );
    let want = s * (0.5 - gnorm);
    assert!(
        ((sol.obj - want) / want).abs() < 1e-7,
        "objective {:.10e} against the closed form {want:.10e} \
         (the orthant misreading gives {:.10e})",
        sol.obj,
        -0.5 * s * (g[0] * g[0] + g[1] * g[1])
    );
}

/// gh #414's `verify_or_repair_optimum` must take the **best** genuine
/// candidate, not the first one.
///
/// That guard sits one layer above the `σ` chain: when a claimed optimum fails
/// `optimum_is_genuine`, it re-solves equilibrated and returns that retry if
/// the retry is genuine. On this instance the retry *is* genuine and is still
/// twelve times worse in `x` than the point it replaces, while a third
/// candidate — the same equilibrated solve on the **direct** driver — is
/// better than both on every measure at once:
///
/// ```text
///   candidate                    kkt_error   equil. rel   genuine   |x-x*|inf
///   the point handed in           1.30e2      4.77e-3       no       9.5e-7
///   equilibrated HSDE retry       1.26e2      1.96e-10      yes      1.1e-5
///   equilibrated DIRECT driver    1.22e-4     7.58e-24      yes      1.1e-16
/// ```
///
/// Six orders of absolute KKT, fourteen of equilibrated relative, eleven of
/// `x`, and it was never asked for. The choice is now made on absolute
/// `kkt_error` in the caller's own coordinates — a ranking, not another
/// threshold — over the genuine candidates only, so it cannot promote a point
/// gh #414 exists to reject.
#[test]
fn the_repair_takes_the_best_genuine_candidate_not_the_first() {
    // The family member that exposed it, verbatim.
    let n = 9usize;
    let (span, top) = (6.0, 1e13);
    let e: Vec<f64> = (0..n)
        .map(|i| top * 10f64.powf(-span * (n - 1 - i) as f64 / (n - 1) as f64))
        .collect();
    let t = [
        -1.8190576992928982,
        1.2812658958137035,
        1.9588236287236214,
        1.8292641211301088,
        0.004131857305765152,
        1.0691035818308592,
        -1.0484796334058046,
        -0.3830815851688385,
        -0.24765135161578655,
    ];
    let prob = QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, e[i])).collect(),
        c: (0..n).map(|i| -e[i] * t[i]).collect(),
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0; n],
        ub: vec![1.0; n],
    };
    let exact: Vec<f64> = t.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let opts = QpOptions::default();

    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let err = abs_x_err(&sol.x, &exact);
    assert!(
        err < 1e-9,
        "the repair should have reached the direct driver's answer (1.1e-16), \
         got {err:.3e} — 1.1e-5 is the equilibrated HSDE retry, which is the \
         first genuine candidate rather than the best"
    );

    // And it really is the direct driver's answer, not a coincidence: the same
    // solve with the embedding off must agree.
    let direct = solve_qp_ipm(
        &prob,
        &QpOptions {
            use_hsde: false,
            ..opts
        },
        backend,
    );
    assert_eq!(direct.status, QpStatus::Optimal);
    assert!(
        abs_x_err(&sol.x, &direct.x) < 1e-12,
        "the repaired answer should coincide with the direct driver's"
    );
}
