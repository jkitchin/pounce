//! gh#884 — the biactive complementarity pair, end to end through the CLI.
//!
//! `crates/pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`
//! owns the invariants and the branch coverage, against hand-built TNLPs.
//! This file owns the **number the issue reported**. It is the only place
//! `7.8965510781517834e+04` appears against a file a user could hand the
//! binary, and the only place the two columns that made it invisible are
//! read off the same line of output a user would read.
//!
//! **The model.** MacMPEC `qpec_small` under `ncp_eq`/`prod_eq` lowering,
//! started at the origin. Its solution is `(1, 1, 0)`, and there the
//! complementarity pair is **biactive**: `G₂ = y₂` and
//! `H₂ = x + 2y₂ − 1` are *both* zero. The lowered product row `G·H = 0`
//! has gradient `H∇G + G∇H`, so at that point its gradient vanishes
//! identically. The row is still satisfied, the primal is still exact —
//! but the multiplier that would certify it is **arbitrary rather than
//! nonexistent**, and the barrier drives it to infinity.
//!
//! **Why nothing caught it.** The convergence gate reads an aggregate
//! normalised by `s_d`, and `s_d` grows with the mean multiplier
//! magnitude. So the runaway divides itself out. Measured here, on one
//! line of the summary block:
//!
//! ```text
//!                                    (scaled)                 (unscaled)
//! Overall NLP error.......:   8.2335532426389998e-11    7.8965510781517834e+04
//! ```
//!
//! Fifteen orders apart, and the gate reads the left one. That is the
//! whole defect: a `Solved_To_Acceptable_Level` on a point whose
//! Lagrangian gradient is 7.9e+04 in the model's own units.
//!
//! **What changed under gh#945, and what it means for this file.** That
//! `Solved_To_Acceptable_Level` is not where the trouble starts. It is where
//! the solve settles *after* the filter line search gives up at an iterate
//! feasible to round-off and hands it to a restoration phase with no
//! violation to minimize. gh#945's retry finds the step instead, so at the
//! shipped default this model never reaches the state above: the base attempt
//! converges by itself, `Optimal`, KKT error `5.65e-10` and unscaled dual
//! infeasibility `6.26e-10` against the
//! `7.9e+04` this file is named for, at `[0.99998, 0.99999, 8.70e-6]` — two
//! orders better than the promoted answer, without the `perturb_always_cd`
//! re-solve. [`the_reproducer_no_longer_stalls_at_the_default`] pins that.
//!
//! Which leaves gh#884's own machinery without an input, so the tests whose
//! subject *is* that machinery — the detector, the kill switch, the dominance
//! gate — run under [`REPRO_PRE945`], which holds the base trajectory fixed
//! with `filter_theta_roundoff_retry=no`. Read that constant's doc before
//! deciding it is a workaround: the state is still reachable on other models,
//! the detector still fires here at the default, and the alternative is a
//! family of tests that pass by never reaching their own subject. The numbers
//! quoted throughout this comment are that trajectory's, not the default's.
//!
//! **The primal was never wrong.** With the retry off the point is
//! `[1.0000000000000, 1.0000000000000, 1.5e-14]` — right to 14 digits.
//! The retry's promoted answer is *further* out, `3.7e-06` in `y₂`, and
//! that is the trade the fix makes on purpose: a slightly looser primal
//! that comes with a multiplier a reader can check, in exchange for nine
//! orders of unscaled dual residual. A test that graded these two answers
//! on the primal alone would prefer the broken one.
//!
//! **Why the options.** `bound_relax_factor=0` is what makes the
//! reproducer bite: at the default relaxation the bound sits far enough
//! off the kink that the pair never goes biactive to working precision.
//! `mu_strategy_fallback=no` holds off the μ-strategy flip, which
//! independently rescues this model — see
//! `the_default_configuration_never_reaches_the_retry`, which is the
//! evidence that the new retry is the *outer* wrapper and costs nothing
//! where an existing one already wins.
//!
//! | test | what it owns | which of gh#884's criteria |
//! |---|---|---|
//! | `the_reproducer_converges_with_a_multiplier_a_reader_can_check` | the fix, on the file the issue is about | 1 |
//! | `the_kill_switch_shows_what_the_scaled_aggregate_was_hiding` | the defect, and the `s_d` normalisation that hid it | 1 (as an implication), 2 |
//! | `a_diverging_iterate_is_not_the_signature` | gh#274's `-exp(x)` case, measured both ways | 4 |
//! | `the_default_configuration_never_reaches_the_retry` | wrapper ordering, and that it is free | — |
//! | `a_generic_exhaustion_exit_does_not_buy_a_retry` | the status scope, on `deb7` — the fixture that measured it | 5 (the one line the sweep moved) |
//! | `a_run_that_recovers_from_the_signature_does_not_buy_a_retry` | the dominance gate does not cost the reproducer its retry; the negative is a unit test in `pounce-algorithm`, because `deb7` is not portable (gh#887) | — |
//!
//! Criterion 3 (`ralph1` must still fail) and the detector's branch
//! coverage are the algorithm-level file's; criterion 5 is
//! `scripts/sweep-fixtures.sh`, recorded in
//! `dev-notes/mpcc-biactive-dual-divergence.md`.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// Fixture stem under `tests/fixtures` — i.e. inside the sweep corpus,
/// which had no MPCC lowering in it before this issue.
const FIXTURE: &str = "mpcc_qpec_small_biactive";

/// The solution of `qpec_small`: `x = 1`, `y₁ = 1`, `y₂ = 0`, `f* = 0`.
const OPTIMUM_X: [f64; 3] = [1.0, 1.0, 0.0];

/// The options that reproduce the issue. See the module comment.
const REPRO: &[&str] = &["bound_relax_factor=0", "mu_strategy_fallback=no"];

/// [`REPRO`] plus `filter_theta_roundoff_retry=no` — the base trajectory
/// gh#884's machinery was built against, held fixed (gh#945).
///
/// The stall this file is about does not start in the convergence gate. It
/// starts one layer down, with the filter line search giving up at an iterate
/// feasible to round-off and handing it to a restoration phase that has no
/// violation to minimize; `Solved_To_Acceptable_Level` at an unscaled dual of
/// `7.90e4` is what the solve settles for afterwards. gh#945's retry finds the
/// step instead, so **at the shipped default this model no longer reaches that
/// state at all**: the base attempt converges by itself at `Optimal`/100,
/// KKT error `5.65e-10` (unscaled dual infeasibility `6.26e-10`),
/// `x = [0.99998, 0.99999, 8.70e-6]`. That is pinned
/// by [`the_reproducer_no_longer_stalls_at_the_default`].
///
/// Which leaves gh#884's own machinery — the detector, the kill switch, the
/// dominance gate — with no input to be tested on unless the input is held
/// fixed. That is what this constant is for, and it is not preserving a bug:
/// the state it reproduces is one a different model can still reach (the gate
/// for that is `runaway_is_the_whole_residual` in `pounce-algorithm`), and the
/// detector still fires at the default here, so nothing in this family has
/// lost its subject.
const REPRO_PRE945: &[&str] = &[
    "bound_relax_factor=0",
    "mu_strategy_fallback=no",
    "filter_theta_roundoff_retry=no",
];

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// gh#274's reproducer, `min -exp(x) s.t. x >= 0`. Lives in the same
/// corpus directory; see `a_diverging_iterate_is_not_the_signature`.
const UNBOUNDED: &str = "unbounded_exp";

fn fixture(stem: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{stem}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh884_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve the fixture with `opts` appended verbatim; return the report and
/// stdout. Both are needed: the residual table is printed but not
/// serialised (`StatisticsInfo` carries only the scaled column), and the
/// two flags are serialised but not printed as such.
fn solve(opts: &[&str]) -> (SolveReport, String) {
    solve_stem(FIXTURE, opts)
}

fn solve_stem(stem: &str, opts: &[&str]) -> (SolveReport, String) {
    let tag = format!("{stem}_{}", opts.join("_")).replace(['=', '.', '-'], "_");
    let json = tmp_path(&tag, "json");
    // Explicit, so a solved fixture does not drop a `.sol` beside the `.nl`.
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(stem))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {stem} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    (
        serde_json::from_str(&text).expect("parse SolveReport JSON"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// The `(scaled, unscaled)` pair off the **last** summary row whose label
/// starts with `label`.
///
/// Last, not first: a run that retries prints one summary block per
/// attempt, and the block that describes the answer being reported is the
/// final one. Reading the first would grade the base attempt's residuals
/// against the promoted attempt's status.
fn residual(stdout: &str, label: &str) -> (f64, f64) {
    let line = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with(label))
        .next_back()
        .unwrap_or_else(|| panic!("no `{label}` row in:\n{stdout}"));
    let cols: Vec<&str> = line.split_whitespace().collect();
    let parse = |s: &str| {
        s.parse::<f64>()
            .unwrap_or_else(|e| panic!("`{label}` column {s:?} is not a number ({e})"))
    };
    let n = cols.len();
    (parse(cols[n - 2]), parse(cols[n - 1]))
}

/// Every `Number of Iterations....:` the run printed, in order.
///
/// One per summary block, so one per *attempt*: on a run that retries,
/// `[base, retry]`, and on one that also climbs the second-opinion ladder,
/// one such pair per rung. This is the only place an attempt's iteration
/// count survives — `statistics.iteration_count` is overwritten by each
/// attempt rather than accumulated, so the JSON report carries the last
/// attempt's count and no record of the others.
fn iteration_counts(stdout: &str) -> Vec<i64> {
    stdout
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("Number of Iterations"))
        .filter_map(|rest| rest.rsplit(':').next())
        .map(|n| {
            n.trim()
                .parse::<i64>()
                .unwrap_or_else(|e| panic!("iteration count {n:?} is not an integer ({e})"))
        })
        .collect()
}

fn max_x_error(r: &SolveReport) -> f64 {
    assert_eq!(
        r.solution.x.len(),
        OPTIMUM_X.len(),
        "{FIXTURE}: expected a 3-variable model, got {}",
        r.solution.x.len()
    );
    r.solution
        .x
        .iter()
        .zip(OPTIMUM_X)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// **Criterion 1 of the issue**, which is not "must converge": the solver
/// must not *report success* at an unscaled dual residual of 7.9e+04.
///
/// It converges here, which is better than the criterion asked for, so
/// this test asserts the stronger thing — but read the assertion order.
/// The one that matters is the residual: a future change that reaches
/// `Solve_Succeeded` by a different route and still carries a runaway
/// multiplier fails on the residual line, not on the status line.
#[test]
fn the_reproducer_converges_with_a_multiplier_a_reader_can_check() {
    let (r, stdout) = solve(REPRO_PRE945);
    let (_, unscaled_du) = residual(&stdout, "Dual infeasibility");
    assert!(
        unscaled_du <= 1e-6,
        "{FIXTURE}: unscaled dual infeasibility {unscaled_du:.6e} — gh#884 is \
         about *this* number (it read 7.8965510781517834e+04), not about the \
         status; stdout=\n{stdout}"
    );
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE}: stdout=\n{stdout}"
    );
    assert!(
        r.statistics.dual_divergence_retry_promoted,
        "{FIXTURE}: the answer must be the retry's — if it converged without \
         one, the base path changed and the numbers in this file's module \
         comment are stale; stdout=\n{stdout}"
    );
    // 3.7e-06 measured, in `y₂`. Deliberately not tighter: the promoted
    // answer is *further* from `(1,1,0)` than the broken one, and pinning
    // it hard would make an honest trajectory change look like a
    // regression. See the module comment.
    let err = max_x_error(&r);
    assert!(
        err <= 1e-4,
        "{FIXTURE}: {err:.3e} from (1, 1, 0) at {:?}; stdout=\n{stdout}",
        r.solution.x
    );
}

/// **At the shipped default the reproducer does not stall** (gh#945), and
/// that is why every test in this family that needs the stall now names
/// [`REPRO_PRE945`].
///
/// The `Solved_To_Acceptable_Level` at an unscaled dual of `7.90e4` is what
/// the solve settles for after the filter line search gives up at an iterate
/// feasible to round-off and hands it to a restoration phase with nothing to
/// minimize. gh#945's retry finds the step, so the base attempt converges by
/// itself: `Optimal`, KKT error `5.65e-10` and unscaled dual infeasibility
/// `6.26e-10` against the
/// reported `7.8965510781517834e+04`, at `[0.99998, 0.99999, 8.70e-6]` — two
/// orders *better* in the residual than the promoted answer it replaces
/// (`9.96e-8`), and reached without the `perturb_always_cd` re-solve.
///
/// One attempt, and gh#884's criterion 1 read off that attempt: this is the
/// assertion that would catch a future change reaching `Solve_Succeeded` here
/// while carrying a runaway multiplier again.
#[test]
fn the_reproducer_no_longer_stalls_at_the_default() {
    let (r, stdout) = solve(REPRO);
    let counts = iteration_counts(&stdout);
    assert_eq!(
        counts.len(),
        1,
        "{FIXTURE}: expected a single attempt at the default, saw {counts:?}; \
         stdout=\n{stdout}"
    );
    assert!(
        r.statistics.dual_divergence_signature,
        "{FIXTURE}: the detector must still fire at the default — the \
         signature is a property of the iterate, and `REPRO_PRE945`'s whole \
         rationale is that nothing in this family has lost its subject. If \
         this goes red, that rationale is wrong and gh#884 needs a second, \
         non-synthetic reproducer rather than a pinned option; \
         stdout=\n{stdout}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "{FIXTURE}: nothing should be promoted when the base attempt \
         converges; stdout=\n{stdout}"
    );
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE}: stdout=\n{stdout}"
    );
    let (_, unscaled_du) = residual(&stdout, "Dual infeasibility");
    assert!(
        unscaled_du <= 1e-6,
        "{FIXTURE}: unscaled dual infeasibility {unscaled_du:.6e} — gh#884 is \
         about *this* number (it read 7.8965510781517834e+04), and a base \
         attempt that converges is held to it exactly as a promoted one is; \
         stdout=\n{stdout}"
    );
    let err = max_x_error(&r);
    assert!(
        err <= 1e-4,
        "{FIXTURE}: {err:.3e} from (1, 1, 0) at {:?}; stdout=\n{stdout}",
        r.solution.x
    );
}

/// The defect itself, reachable through the kill switch — and the two
/// columns that hid it.
///
/// This is the one test here that describes rather than constrains, and
/// it is written to stay honest about that. It does **not** assert that
/// the base path still fails; it asserts the *implication* the issue is
/// about: whatever verdict the base path reaches, it must not be a
/// success paired with a runaway multiplier. Measured today the base path
/// exits `Solved_To_Acceptable_Level` at 7.8965510781517834e+04 unscaled
/// against 8.2335532426389998e-11 scaled, and the second assertion pins
/// that fifteen-order gap, because the gap is the mechanism: `s_d` grows
/// with the mean multiplier magnitude, so the runaway divides itself out
/// of the aggregate the gate reads.
///
/// If a future change makes the base path converge honestly here, this
/// test goes red on the `s_d` assertion. That is the right place to learn
/// it: retire the assertion deliberately, and check the detector is still
/// reached by something before deleting anything else in this family.
#[test]
fn the_kill_switch_shows_what_the_scaled_aggregate_was_hiding() {
    let opts: Vec<&str> = REPRO_PRE945
        .iter()
        .copied()
        .chain(["dual_divergence_retry=no"])
        .collect();
    let (r, stdout) = solve(&opts);
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "{FIXTURE}: `dual_divergence_retry=no` must mean no retry; \
         stdout=\n{stdout}"
    );
    assert!(
        r.statistics.dual_divergence_signature,
        "{FIXTURE}: the *detector* is not the kill switch — turning the retry \
         off must still record what was seen, or the report loses the only \
         field that distinguishes a runaway multiplier from an unsettled \
         iterate; stdout=\n{stdout}"
    );
    let (scaled_du, unscaled_du) = residual(&stdout, "Dual infeasibility");
    let (scaled_err, unscaled_err) = residual(&stdout, "Overall NLP error");
    // gh#884's criterion 1, as an implication rather than a pin.
    if matches!(r.solution.status, ApplicationReturnStatus::SolveSucceeded) {
        assert!(
            unscaled_du <= 1e-6,
            "{FIXTURE}: `Solve_Succeeded` at unscaled dual infeasibility \
             {unscaled_du:.6e} is gh#884 verbatim; stdout=\n{stdout}"
        );
    }
    // The mechanism: the aggregate the gate reads is clean while the
    // residual in the model's own units is not.
    assert!(
        unscaled_err >= 1e2 && scaled_err <= 1e-8,
        "{FIXTURE}: the `s_d` normalisation no longer hides the runaway \
         (overall NLP error {scaled_err:.6e} scaled, {unscaled_err:.6e} \
         unscaled; dual infeasibility {scaled_du:.6e} / {unscaled_du:.6e}). \
         Read this test's doc comment before changing the bound; \
         stdout=\n{stdout}"
    );
}

/// The new retry is the **outermost** wrapper, and that is why it is
/// nearly free.
///
/// Its "base solve" is `dispatch_standard_solve`, which already contains
/// the μ-strategy fallback. `retry_worthy` excludes `Solve_Succeeded`, so
/// wherever the μ flip promotes, the dual-divergence retry never runs. At
/// default options this fixture is exactly that case: the detector fires
/// mid-solve — the flag is `true` — and no second solve is spent.
///
/// The flag being `true` on a clean answer is also why the console line
/// is suppressed on `Solve_Succeeded`: passing through a biactive runaway
/// and recovering is routine on an MPCC lowering, and a warning printed
/// over a correct answer is noise.
#[test]
fn the_default_configuration_never_reaches_the_retry() {
    let (r, stdout) = solve(&[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE}: stdout=\n{stdout}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "{FIXTURE}: an existing wrapper already wins here — a promotion means \
         the ordering changed and every model this fixture stands for now \
         pays for a second solve; stdout=\n{stdout}"
    );
    assert!(
        !stdout.contains("Biactive dual divergence"),
        "{FIXTURE}: the summary must stay silent on a clean answer; \
         stdout=\n{stdout}"
    );
    let err = max_x_error(&r);
    assert!(
        err <= 1e-4,
        "{FIXTURE}: {err:.3e} from (1, 1, 0) at {:?}; stdout=\n{stdout}",
        r.solution.x
    );
}

/// **Criterion 4 of the issue**: evidence the fix does not relabel a
/// `-exp(x)`-shaped case.
///
/// `unbounded_exp.nl` is gh#274's reproducer — `min -exp(x) s.t. x >= 0`,
/// unbounded below — and it is the closest thing in the corpus to a
/// false positive for the *detector*. It satisfies **two of the three
/// conjuncts outright**: the constraint row stays satisfied while the
/// iterates run off, so `inf_pr` is at zero and the unscaled dual
/// infeasibility is enormous (`8.7e+20` at the exit, `8.8e+47` at the
/// iterate gh#274 was written about).
///
/// The **step** conjunct is what holds the detector off, and that is
/// gh#884's discriminator stated as a measurement rather than an
/// intention: this iterate is not settled, it is running away. Both
/// halves are asserted, because "the detector does not fire" is only
/// evidence if the reason is the one claimed —
///
/// * at the default `1e-5` the signature is never set; and
/// * at `dual_divergence_retry_step_tol=1e30`, the step conjunct
///   disabled and nothing else changed, it *is* set. So nothing else
///   was quietly doing the work.
///
/// The forced leg then shows the **second, independent** barrier. Even
/// with the detector fully deceived, no retry runs at all: this model
/// exits `Error_In_Step_Computation`, a generic exhaustion status, and
/// the retry is scoped to the two verdicts the vanishing-gradient row
/// produces directly (`Solved_To_Acceptable_Level`,
/// `Restoration_Failed`). Re-add a generic status to that set and this
/// test goes red — which is the point, because doing so is what costs
/// `deb7` 3000 iterations (see the test below).
#[test]
fn a_diverging_iterate_is_not_the_signature() {
    let (base, base_out) = solve_stem(UNBOUNDED, &[]);
    let (forced, forced_out) = solve_stem(UNBOUNDED, &["dual_divergence_retry_step_tol=1e30"]);

    // The fixture is only evidence while it still runs off to -inf.
    assert!(
        base.solution.objective < -1e10,
        "{UNBOUNDED}: objective {} — this model is supposed to diverge, and \
         every assertion below is about the iterate that does; \
         stdout=\n{base_out}",
        base.solution.objective
    );
    assert!(
        !base.statistics.dual_divergence_signature,
        "{UNBOUNDED}: gh#274's diverging iterate must not read as a settled \
         one; stdout=\n{base_out}"
    );
    assert!(
        forced.statistics.dual_divergence_signature,
        "{UNBOUNDED}: with the step conjunct disabled the other two must \
         hold — if they do not, the leg above is passing for a reason this \
         test does not describe; stdout=\n{forced_out}"
    );
    assert!(
        !forced_out.contains("dual-divergence retry:"),
        "{UNBOUNDED}: no retry may run on a generic exhaustion status, even \
         with the detector deceived; stdout=\n{forced_out}"
    );
    assert!(
        !forced.statistics.dual_divergence_retry_promoted,
        "{UNBOUNDED}: stdout=\n{forced_out}"
    );
    assert_eq!(
        (base.solution.status, base.solution.objective),
        (forced.solution.status, forced.solution.objective),
        "{UNBOUNDED}: a deceived detector must cost nothing at all; \
         stdout=\n{forced_out}"
    );
    assert_ne!(
        base.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "gh#274: an unbounded model must not be reported solved; \
         stdout=\n{base_out}"
    );
}

/// `deb7` under L-BFGS: the fixture that scoped the retry.
///
/// This is the corpus's **second true positive for the detector**, and
/// it is a true positive by a wide margin — measured at iteration 346, a
/// scale-relative step of `6.5e-6`, `inf_pr` of `3.0e-12`, and an
/// unscaled `inf_du` of `9.2e+05`, which is an *order above* the gh#884
/// reproducer's `7.9e+04` — so no dual floor excludes it, and the step
/// conjunct separates the two only by fitting the default onto this one
/// fixture and spending the margin that holds `ralph1` out. The point of
/// writing that down is that nobody should try: the detector is right
/// here, and the *remedy* is what does not apply.
///
/// An earlier draft of this fix retried on `Error_In_Step_Computation`
/// too. Across this fixture that cost 715 -> 3000 iterations — 4x the
/// trajectory — ending `Maximum_Iterations_Exceeded` at an unscaled KKT
/// error of `6.7e+01` against the base attempt's `9.9e+01`, with the
/// shipped verdict and objective unchanged. It was the only line the
/// fixture sweep moved, on either leg, and it is the whole reason
/// `retry_worthy` names two statuses instead of four.
///
/// So this test is not about `deb7`. It is about the scope, pinned on
/// the model that measured it.
#[test]
fn a_generic_exhaustion_exit_does_not_buy_a_retry() {
    let (r, stdout) = solve_stem(
        "deb7",
        &["hessian_approximation=limited-memory", "max_iter=3000"],
    );
    assert!(
        r.statistics.dual_divergence_signature,
        "deb7/lbfgs: the detector is supposed to fire here — if it no longer \
         does, this test has stopped measuring the scope and the comment \
         above is stale; stdout=\n{stdout}"
    );
    assert!(
        !stdout.contains("dual-divergence retry:"),
        "deb7/lbfgs: a retry here costs 4x the trajectory and buys nothing; \
         stdout=\n{stdout}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "deb7/lbfgs: stdout=\n{stdout}"
    );
    // The trajectory itself, which is what the sweep would have caught.
    // Generous: the claim is "did not multiply", not "715".
    assert!(
        r.statistics.iteration_count < 1500,
        "deb7/lbfgs: {} iterations — it takes 715 without a retry and 3000 \
         with one; stdout=\n{stdout}",
        r.statistics.iteration_count
    );
}

/// gh#887 — "one extra solve" is a cost claim, and this is the half of
/// it a CLI test can actually hold.
///
/// The detector is a statement about an **iterate**. Nothing in it says
/// the solve *ends* at that iterate, so a run can pass through a settled
/// point with a diverged multiplier, work its way back down, and report
/// something ordinary — and then there is nothing left for
/// `perturb_always_cd` to repair. The gate for that is
/// `runaway_is_the_whole_residual` in `pounce-algorithm`, and the numbers
/// behind it are pinned there, as unit tests, on purpose.
///
/// **Why not here.** `deb7` on the L-BFGS leg under the gh#818 rung is
/// the fixture that measured gh#887 (6.08 s → 25.17 s), and it is not a
/// portable witness for the rule. The same invocation reaches a
/// materially different answer on the two platforms CI runs:
///
/// | | objective | unscaled dual | viol | compl | ratio |
/// |---|---|---|---|---|---|
/// | macOS | `99.677` | `9.90e1` | `8.0e-13` | `4.65e0` | `4.7e-2` |
/// | Linux | `99.651` | `5.5743e3` | `5.6e-14` | `2.08e-5` | `3.7e-9` |
///
/// On Linux that answer genuinely *is* gh#884's shape — scaled overall
/// error `5.28e-1` against unscaled `5.57e3`, the `s_d` normalisation
/// hiding a runaway exactly as it did on `qpec_small` — so the retry
/// there is the designed cost, not the waste gh#887 filed. An assertion
/// that `deb7` declines is false on Linux at *any* threshold. It cost a
/// red CI to learn that, and the honest reading was that the fixture was
/// wrong, not that the constant needed loosening.
///
/// So what this test holds is the part that does not depend on any
/// fixture's trajectory: the reproducer, whose answer is gh#884's shape
/// on every platform, still buys its retry and still promotes. The
/// *negative* — an answer that is merely unconverged buys nothing — is
/// `an_unconverged_point_does_not_open_the_retry`, which pins the macOS
/// row above directly.
#[test]
fn the_gate_that_reads_the_answer_does_not_cost_the_reproducer_its_retry() {
    let (r, stdout) = solve(REPRO_PRE945);
    let counts = iteration_counts(&stdout);
    assert_eq!(
        counts.len(),
        2,
        "{FIXTURE}: expected a base attempt and one retry, saw {counts:?}; \
         stdout=\n{stdout}"
    );
    assert!(
        r.statistics.dual_divergence_retry_promoted,
        "{FIXTURE}: the dominance gate must not cost the reproducer its \
         promotion — its answer is gh#884's shape by twelve orders \
         (viol 1.1e-16, compl 1.1e-9, against an unscaled dual of 7.9e4); \
         stdout=\n{stdout}"
    );
}
