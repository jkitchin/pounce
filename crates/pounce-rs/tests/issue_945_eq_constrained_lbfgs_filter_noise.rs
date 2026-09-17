//! gh#945 — the limited-memory arm on an **equality-constrained**,
//! ill-conditioned convex QP.
//!
//! Reported against `pounce.minimize(f, x0, jac=g, constraints=[...])`,
//! which selects `hessian_approximation=limited-memory` (no Hessian
//! supplied) — the mode the Python frontend and the CasADi plugin pick on
//! their own. On a 10-variable separable quadratic with one linear equality
//! and no bounds at `cond(H) = 3.2e3`, the solve exited
//! `Error_In_Step_Computation` at iteration 131 with `x` wrong by 3.6e-5
//! relative, while POUNCE's own exact-Hessian arm reaches the same model's
//! KKT point in **one** iteration and its convex QP engine solves it to
//! 7.4e-13. The iteration count and the status were identical at `max_iter`
//! 200, 1000 and 5000 — a breakdown, not an exhausted budget.
//!
//! **What was actually wrong.** Not the model, not the quasi-Newton update,
//! and not the linear algebra. Over all 576 augmented-system solves on the
//! failing trajectory the Sherman-Morrison-Woodbury path's
//! iterative-refinement residual ratio never exceeds **4.1e-11** — inside
//! its own `residual_ratio_max` of 1e-10 every time — so each step is the
//! solution of the system it was handed. What failed is the **filter**, in
//! `line_search::filter::Filter::dominated_by_any`. This model's constraint
//! is linear, so `theta` is satisfied to round-off from the first feasible
//! iterate on; every filter entry then records a `theta` of one or two
//! `ulp`, and upstream's bare `theta > e.theta` ranks later trials on
//! nothing but which way the last constraint sum rounded. Measured at
//! iteration 94, trial 3, `alpha = 0.125`: the trial decreases `phi` by
//! 5.1e-9 and passes the Armijo test on it, and the filter rejects it on
//! `theta = 5.551115123125783e-16` against an entry's
//! `1.110211922394910e-16` — a gap of 4.4e-16 between two roundings of the
//! same identically-zero quantity. (The `phi` arm of that same comparison is
//! doing real work: the trial is 2.0e-8 above the entry. Only the `theta`
//! arm is noise, and only the `theta` arm gets the fix.) The α-loop
//! backtracks to `alpha_min`, hands a point that is feasible to **0.0** to
//! the restoration phase — which has no violation to minimize — and the
//! solve reports a failure from the optimum.
//!
//! **The fix, and why it is shaped the way it is.** The last sentence above
//! is the part that generalizes: the hand-off the driver is about to make
//! cannot work. So before making it,
//! `BacktrackingLineSearch::run_filter_line_search` runs the α-loop **once
//! more**, with the filter's `theta` axis measured against `theta`'s own
//! evaluation noise, and only when the iterate it failed at is feasible to
//! that noise. If the retry finds nothing either, the hand-off proceeds
//! byte-for-byte as before — which is why the whole thing can only be
//! reached on a trajectory that was otherwise about to enter restoration.
//! It is on by default; `filter_theta_roundoff_retry=no` restores the plain
//! hand-off, and the tests below use that rather than a second build.
//!
//! The narrowness is not caution, it is the measurement. Applying the floor
//! to *every* filter decision fixes this model too and costs MacMPEC's
//! `qpec_small` its answer at exactly the floor this model needs, with no
//! constant separating them. The retry gate separates them because it asks
//! about the **iterate** rather than about a pair of entries:
//! `qpec_small`'s line search fails 2.6× above its own noise floor, so
//! restoration there has real violation to work on and the gate declines.
//! That is pinned in `pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`,
//! `the_gh945_retry_gate_is_what_this_fixture_relies_on`, which carries the
//! measured table. Where an MPCC *does* reach the retry the effect runs the
//! other way, which is the second half of the same point: the CLI reproducer
//! `mpcc_qpec_small_biactive` converges on its own at `Optimal`/100 with a
//! KKT error of `5.65e-10` (dual infeasibility `6.26e-10`) and stops needing
//! gh#884's promotion, which used
//! to buy it `9.96e-8`
//! (`pounce-cli/tests/issue884_promotion_gate_reads_the_answer.rs`,
//! `the_reproducer_no_longer_needs_the_retry`).
//!
//! **What this file is not evidence about.** It pins the reported model,
//! the shape it instances, and the option that moves it. It does not pin the
//! fixture corpus — that is `scripts/sweep-fixtures.sh`, whose ledger is in
//! CHANGELOG.md (exact leg byte-identical, one moved `lbfgs` leg, no status
//! changes) — and it says nothing about the exact-Hessian arm, which
//! `pounce-rs`'s builder cannot drive at all.

use pounce_rs::builder::{Nlp, Problem};

/// `min ½ Σ dᵢ(xᵢ − tᵢ)²  s.t.  Σx = s`, with `d = 10^linspace(0, lg, n)`
/// so `cond(H) = 10^lg`. The KKT system of an equality-constrained QP is
/// linear, so the solution is closed form:
///
/// ```text
///   λ  = (s − Σtᵢ) / Σ(1/dᵢ)
///   x*ᵢ = tᵢ + λ/dᵢ
/// ```
struct EqualityConstrainedQp {
    d: Vec<f64>,
    t: Vec<f64>,
    s: f64,
}

impl EqualityConstrainedQp {
    fn new(n: usize, cond: f64) -> Self {
        let lg = cond.log10();
        Self {
            d: (0..n)
                .map(|i| 10f64.powf(lg * i as f64 / (n - 1) as f64))
                .collect(),
            t: vec![1.0; n],
            s: 0.0,
        }
    }

    fn solution(&self) -> Vec<f64> {
        let lam =
            (self.s - self.t.iter().sum::<f64>()) / self.d.iter().map(|&di| 1.0 / di).sum::<f64>();
        self.t
            .iter()
            .zip(&self.d)
            .map(|(&ti, &di)| ti + lam / di)
            .collect()
    }

    /// The issue's `relKKT`: an independent dual-residual check computed
    /// **outside** the solver. Stationarity here is `d⊙(x − t) − λ·1 = 0`,
    /// so `‖g − mean(g)‖∞ / max(1, ‖g‖∞)` is the relative KKT error at the
    /// returned point, with `λ` eliminated rather than read back from
    /// POUNCE. A solver that reported success at a non-solution could not
    /// hide from this.
    fn rel_kkt(&self, x: &[f64]) -> f64 {
        let g: Vec<f64> = x
            .iter()
            .zip(&self.t)
            .zip(&self.d)
            .map(|((&xi, &ti), &di)| di * (xi - ti))
            .collect();
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        let num = g.iter().map(|&gi| (gi - mean).abs()).fold(0.0, f64::max);
        let den = g.iter().map(|&gi| gi.abs()).fold(1.0, f64::max);
        num / den
    }
}

impl Problem for EqualityConstrainedQp {
    fn objective(&self, x: &[f64]) -> f64 {
        0.5 * x
            .iter()
            .zip(&self.t)
            .zip(&self.d)
            .map(|((&xi, &ti), &di)| di * (xi - ti) * (xi - ti))
            .sum::<f64>()
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
        for (i, gi) in grad.iter_mut().enumerate() {
            *gi = self.d[i] * (x[i] - self.t[i]);
        }
        true
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) {
        out[0] = x.iter().sum();
    }
    fn jacobian(&self, _x: &[f64], jac: &mut [f64]) -> bool {
        jac.fill(1.0);
        true
    }
}

#[derive(Debug)]
struct Outcome {
    success: bool,
    iters: i32,
    rel_kkt: f64,
}

fn solve(n: usize, cond: f64, max_iter: i32) -> Outcome {
    solve_with(n, cond, max_iter, None)
}

/// `retry = None` is the shipped default; `Some("no")` is the pre-fix
/// behaviour, reached through the registered option rather than a separate
/// build.
fn solve_with(n: usize, cond: f64, max_iter: i32, retry: Option<&str>) -> Outcome {
    let p = EqualityConstrainedQp::new(n, cond);
    let mut nlp = Nlp::new(EqualityConstrainedQp::new(n, cond))
        .x0(&vec![0.0; n])
        .constraint_bounds(&[p.s], &[p.s])
        .option_int("max_iter", max_iter)
        .option_int("print_level", 0)
        .option_str("hessian_approximation", "limited-memory");
    if let Some(r) = retry {
        nlp = nlp.option_str("filter_theta_roundoff_retry", r);
    }
    let sol = nlp.solve();
    Outcome {
        success: sol.success,
        iters: sol.stats.iteration_count,
        rel_kkt: if sol.x.len() == n {
            p.rel_kkt(&sol.x)
        } else {
            f64::INFINITY
        },
    }
}

/// The reported case, verbatim: `n = 10`, `d = logspace(0, 3.5, 10)`,
/// `t = 1`, `Σx = 0`, `x0 = 0`, default tolerance, no Hessian.
///
/// Before the fix this returned `Error_In_Step_Computation` at iteration
/// 131 with `relKKT = 3.8e-3` — five orders above the default `tol = 1e-8`.
/// The `relKKT` bound here is the assertion that matters: it is computed
/// from the closed form outside POUNCE, so it cannot be satisfied by a
/// status the solver merely reports.
#[test]
fn issue_945_the_reported_model_solves() {
    let o = solve(10, 3.1622776601683795e3, 1000);
    assert!(
        o.success,
        "gh#945's model did not converge ({} iterations, relKKT {:.3e})",
        o.iters, o.rel_kkt
    );
    assert!(
        o.rel_kkt < 1e-6,
        "converged to a non-solution: relKKT {:.3e} after {} iterations \
         (the closed-form optimum reads 2.3e-14)",
        o.rel_kkt,
        o.iters
    );
}

/// The base solve stands on its own — no second-opinion rung is opened.
///
/// This pins the mechanism rather than the verdict. gh#945's exit was not a
/// single bad step: the α-loop ran out of steps it was willing to take,
/// handed a point feasible to **0.0** to the restoration phase — which has
/// no constraint violation to minimize and can only report
/// `Restoration_Failed` — and that failing verdict is what opened the
/// second-opinion ladder. On the reported model the baseline reaches
/// `RestorationFailed` at 156 iterations with a rung attached; after the
/// fix the first solve converges and nothing is spent on recovery.
///
/// Asserting the *absence of a rung* is what makes this a mechanism test:
/// a future change that got this model to converge only by rescuing it
/// would pass every other test in this file and fail this one. (The
/// restoration counters on `stats` cannot say it — they describe whichever
/// solve produced the returned `Solution`, so a rung-rescued answer reports
/// the rung's zero, not the base solve's excursion.)
#[test]
fn issue_945_needs_no_second_opinion_rung() {
    let n = 10;
    let cond = 3.1622776601683795e3;
    let p = EqualityConstrainedQp::new(n, cond);
    let sol = Nlp::new(EqualityConstrainedQp::new(n, cond))
        .x0(&vec![0.0; n])
        .constraint_bounds(&[p.s], &[p.s])
        .option_int("max_iter", 1000)
        .option_int("print_level", 0)
        .option_str("hessian_approximation", "limited-memory")
        .solve();
    assert!(
        sol.second_opinion.is_none(),
        "the base solve failed and was rescued by a second-opinion rung \
         ({:?} after {} iterations); gh#945 is the claim that the base \
         solve should not fail here at all",
        sol.status,
        sol.stats.iteration_count
    );
    assert!(sol.success, "{:?}", sol.status);
}

/// **The retry is what moves it** — the same model, the same build, the
/// option off.
///
/// Every other test here asserts a verdict, and a verdict can be reached by
/// any number of routes. This one pins the *cause*: with
/// `filter_theta_roundoff_retry=no` the reported model reproduces gh#945
/// exactly — a breakdown well short of `max_iter`, from a point that is not
/// a solution — and with the shipped default it converges. Anything that
/// fixed this model by some other mechanism would leave this test red, and
/// anything that quietly stopped running the retry would leave it green
/// while the rest of the file went red, which is the pairing that makes the
/// two together evidence.
#[test]
fn issue_945_the_retry_is_the_mechanism() {
    let n = 10;
    let cond = 3.1622776601683795e3;
    let off = solve_with(n, cond, 1000, Some("no"));
    assert!(
        !off.success,
        "with the retry off the reported model converged in {} iterations; \
         gh#945 is the report that it does not",
        off.iters
    );
    assert!(
        off.iters < 900,
        "with the retry off the model ran to {} of 1000 iterations — that \
         is a budget, and gh#945 reported a breakdown (identical exits at \
         max_iter 200, 1000 and 5000)",
        off.iters
    );
    // Two orders under the measured 1.0e-5 and two over the default
    // `tol = 1e-8`: the claim is "not a solution", and it should not go red
    // because something unrelated improved the retry-off path a little.
    assert!(
        off.rel_kkt > 1e-7,
        "with the retry off the returned point reads relKKT {:.3e} — it is \
         supposed to be a non-solution (measured 1.0e-5 here, against the \
         3.8e-3 gh#945 reports through the Python frontend, which does not \
         open this crate's second-opinion ladder)",
        off.rel_kkt
    );

    let on = solve(n, cond, 1000);
    assert!(on.success && on.rel_kkt < 1e-6, "{on:?}");
}

/// **It is fewer breakdowns, not a different tolerance** — the prevalence
/// claim, sampled.
///
/// gh#945 reports the failure at 9-17 of 40 instances at `cond = 1e3` and
/// 35-39 of 40 at `cond = 1e5`, so a fix for one instance is not a fix.
/// Measured over 480 fresh instances of the reported shape
/// (`n ∈ {8, 12, 20}` × `cond ∈ {1e3, 3.2e3, 1e4, 1e5}`, 40 each, random
/// `t` and `s`, counting any status but `Solve_Succeeded`), the option off
/// against the shipped default:
///
/// | `max_iter` | off | on |
/// |---|---|---|
/// | 1000  | 133 bad — 123 breakdowns, 10 out of budget | 94 bad — 49 breakdowns, 45 out of budget |
/// | 10000 | 122 bad — **every one a breakdown** | 44 bad — 43 breakdowns |
///
/// Ten times the budget buys the old code eleven solves and the new one
/// fifty. What is left at `cond = 1e5` is a six-pair L-BFGS window that
/// cannot span a five-decade spectrum — gh#818's documented limit, with
/// `limited_memory_max_history` as its remedy — and not this defect.
///
/// The 480-instance run takes minutes, so what is pinned here is one cell
/// of it, at the size the assertion can afford: the count must go **down**,
/// strictly, on a sample where it was measured to.
#[test]
fn issue_945_the_failure_rate_falls_across_the_family() {
    // LCG, so the instances are the same on every platform and every run.
    let mut state: u64 = 12345;
    let mut unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let instances: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..24)
        .map(|k| {
            let n = [8usize, 12, 20][k % 3];
            let cond = [1e3f64, 3.1622776601683795e3, 1e4][(k / 3) % 3];
            let lg = cond.log10();
            let d = (0..n)
                .map(|i| 10f64.powf(lg * i as f64 / (n - 1) as f64))
                .collect();
            let t = (0..n).map(|_| 2.0 * unit() - 1.0).collect();
            (d, t, 2.0 * unit() - 1.0)
        })
        .collect();

    let count_bad = |retry: &str| {
        instances
            .iter()
            .filter(|(d, t, s)| {
                let n = d.len();
                !Nlp::new(EqualityConstrainedQp {
                    d: d.clone(),
                    t: t.clone(),
                    s: *s,
                })
                .x0(&vec![0.0; n])
                .constraint_bounds(&[*s], &[*s])
                .option_int("max_iter", 2000)
                .option_int("print_level", 0)
                .option_str("hessian_approximation", "limited-memory")
                .option_str("filter_theta_roundoff_retry", retry)
                .solve()
                .success
            })
            .count()
    };

    let off = count_bad("no");
    let on = count_bad("yes");
    assert!(
        off > 0,
        "the premise of this test is that this sample reaches the defect; \
         none of the {} instances failed with the retry off",
        instances.len()
    );
    assert!(
        on < off,
        "the retry did not reduce the failure count on the sampled family: \
         {off} failed with it off, {on} with it on"
    );
}

/// Conditioning degrades the *cost*, not the *verdict*. gh#945's second
/// table is a status column that turns red at `cond = 3.2e3` and stays red,
/// with `relKKT` climbing to 1.28 at `cond = 1e6` — a returned point that
/// is not a solution in any useful sense. Across the band the issue calls
/// "unremarkable for a scaled engineering model" the answer must now be
/// right at every rung.
#[test]
fn issue_945_the_conditioning_ladder_stays_solved() {
    for cond in [1e1f64, 1e2, 1e3, 3.1622776601683795e3, 1e4] {
        let o = solve(8, cond, 10_000);
        assert!(
            o.success,
            "cond = {cond:.1e} did not converge ({} iterations, relKKT {:.3e})",
            o.iters, o.rel_kkt
        );
        assert!(
            o.rel_kkt < 1e-6,
            "cond = {cond:.1e} converged to a non-solution: relKKT {:.3e}",
            o.rel_kkt
        );
    }
}

/// The issue's first oracle, checked directly: `x*ᵢ = tᵢ + λ/dᵢ` in closed
/// form. `relKKT` above is a *dual* residual, so on its own it cannot rule
/// out a primal point that is wrong in a direction the stationarity test is
/// blind to; this is the primal half, and the two together are the issue's
/// own standard of proof. Before the fix the returned `x` was wrong by
/// 3.6e-5 relative on the reported model.
///
/// The issue's other three oracles are not reachable from this crate and
/// are not restated here: `pounce-rs`'s builder takes no analytic Hessian
/// (`builder.rs`'s `eval_h` returns `false` and the facade pins
/// `hessian_approximation=limited-memory`), so the exact arm cannot be
/// driven through it, and the convex QP engine is behind the `convex`
/// feature. The exact arm's evidence for this change is the sweep's exact
/// leg, which is byte-identical across all 97 fixtures — it never enters
/// the round-off band the fix is about.
#[test]
fn issue_945_the_primal_point_matches_the_closed_form() {
    let n = 10;
    let cond = 3.1622776601683795e3;
    let p = EqualityConstrainedQp::new(n, cond);
    let x_star = p.solution();
    let sol = Nlp::new(EqualityConstrainedQp::new(n, cond))
        .x0(&vec![0.0; n])
        .constraint_bounds(&[p.s], &[p.s])
        .option_int("max_iter", 1000)
        .option_int("print_level", 0)
        .option_str("hessian_approximation", "limited-memory")
        .solve();
    assert!(sol.success, "the reported model did not converge");
    let rel = (0..n)
        .map(|i| (sol.x[i] - x_star[i]).abs() / x_star[i].abs().max(1.0))
        .fold(0.0f64, f64::max);
    assert!(
        rel < 1e-7,
        "missed the closed-form optimum by {rel:.3e} relative; gh#945 \
         reported 3.6e-5 at the Error_In_Step_Computation exit"
    );
}
