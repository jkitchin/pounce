//! gh#884 follow-up — the promotion gate must rank the two attempts as
//! **answers**, not only as certificates.
//!
//! `crates/pounce-cli/tests/issue_884_biactive_dual_divergence.rs` owns the
//! number gh#884 reported and the fact that the retry works. This file owns
//! the other half: what the retry is allowed to hand back.
//!
//! **What was wrong.** The gate promoted on `Solve_Succeeded` + unscaled
//! KKT error and constraint violation within `acceptable_tol` + a strictly
//! better unscaled KKT error than the base attempt. Every one of those is a
//! statement about the *certificate*. The argument that this was enough
//! reads, in `run_with_dual_divergence_retry`:
//!
//! > that deferral exists because the μ flip returns a different *local
//! > solution* … This retry cannot: conjunct 4 requires the promoted answer
//! > to satisfy the KKT conditions in the model's own units.
//!
//! Any *other* KKT point satisfies the KKT conditions in the model's own
//! units too, so the inference does not hold. Measured on 400 random QPECs
//! under the `prod_eq` lowering at `bound_relax_factor=0
//! mu_strategy_fallback=no tol=1e-8`: **68 promotions, 42 of which moved the
//! objective materially** — i.e. returned a different local solution — and
//! **three of which returned a strictly worse feasible point**, worst case
//! `-13.0057 → -1.2072`, both independently `pounce verify`-feasible.
//!
//! **Two fixtures, because the rule branches**, and a green test on the
//! branch a fixture does not take is worth nothing (CLAUDE.md):
//!
//! | fixture | base f | retry f | which rule refuses it |
//! |---|---|---|---|
//! | `mpcc_scholtes4_biactive` | `+1.8176e-09` | `-6.6088e-05` | 2 — an improvement bought with primal slack |
//! | `mpcc_worse_local_solution` | `-1.3006e+01` | `-1.2072e+00` | 1 — a strictly worse feasible point |
//!
//! `scholtes4` is the more serious of the two and is why this is a
//! *correctness* fix rather than a quality one: its `f*` is **exactly 0**,
//! for the MPCC and for the smooth lowering alike (`x₁x₂ = 0` forces one of
//! the pair to zero, hence `x₃ ≤ 0`, hence `f = x₁ − x₃ ≥ 0`), so
//! `-6.6088e-05` is a value no feasible point attains. It was returned
//! under `EXIT: Optimal Solution Found.`
//!
//! That is the failure gh#884's own safety argument names and claims to
//! exclude — `perturb_always_cd=yes` on `ralph1` reaching `f = -2.71e-5`
//! below `f* = 0` — and `the_detector_must_not_fire_on_ralph1` is
//! mutation-checked against it. But the barrier was fitted to `ralph1`'s
//! scale-relative step (`7.2e-3`, against `qpec_small`'s `4.3e-8`), and
//! `scholtes4` is the same failure class arriving on the other side of it:
//! MPCC-LICQ fails, no S-stationary point exists, and its step settles.
//! A threshold validated against one member of the class it has to exclude
//! is not evidence about the class.
//!
//! **Provenance.** `mpcc_scholtes4_biactive` is `benchmarks/mpcc/cases.py`'s
//! `scholtes4` — `min x₁+x₂−x₃ s.t. −4x₁+x₃ ≤ 0, −4x₂+x₃ ≤ 0,
//! 0 ≤ x₁ ⟂ x₂ ≥ 0` — under the `prod_eq` lowering, started at the origin;
//! its optimum is derived above and in that file. `mpcc_worse_local_solution`
//! is seed 116 of the random QPEC family in
//! `dev-notes/mpcc-biactive-dual-divergence.md`; no global optimum is
//! claimed for it and none is needed — the assertion is only that the base
//! attempt's feasible point is better than the retry's, which `pounce
//! verify` establishes on both points independently.
//!
//! ## Mutation table
//!
//! | change | what goes red |
//! |---|---|
//! | drop conjunct 6 (rule 1) | `a_worse_feasible_local_solution_is_not_promoted` |
//! | drop conjunct 7 (rule 2) | `scholtes4_is_not_promoted_below_its_known_optimum` |
//! | drop both | both, and `a_declined_retry_reports_one_answer` |
//! | drop the `answer_restored_from_floor` swap in `main.rs` | `a_declined_retry_reports_one_answer` |
//! | let the gh#486 correction run on the swapped capture | `a_declined_retry_reports_one_answer_under_variable_scaling` only — the unscaled leg stays green, which is the point |
//! | tighten the objective tolerance below `5.8e-11` | `the_reproducer_still_promotes` |

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// The options that put the retry in play. `bound_relax_factor=0` is the
/// one that matters — at the default relaxation the pair never goes
/// biactive to working precision — and it is an ordinary documented
/// option, set by anyone who wants the model they declared.
const REPRO: &[&str] = &[
    "bound_relax_factor=0",
    "mu_strategy_fallback=no",
    "tol=1e-8",
];

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

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
        "pounce_gh884gate_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

fn solve(stem: &str, opts: &[&str]) -> (SolveReport, String) {
    let tag = format!("{stem}_{}", opts.join("_")).replace(['=', '.', '-'], "_");
    let json = tmp_path(&tag, "json");
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
    let report: SolveReport = serde_json::from_str(&text).expect("parse report");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    (report, stdout)
}

/// `scholtes4`'s `f*` is exactly 0, so a *negative* objective is a value
/// no feasible point of the model attains.
#[test]
fn scholtes4_is_not_promoted_below_its_known_optimum() {
    let (r, out) = solve("mpcc_scholtes4_biactive", REPRO);

    assert!(
        r.statistics.dual_divergence_signature,
        "the detector is supposed to fire here — if it stopped, this test \
         is no longer about the promotion gate. stdout:\n{out}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "the retry was promoted at f = {:.6e}, which is below this model's \
         exactly-known f* = 0. stdout:\n{out}",
        r.solution.objective,
    );
    // The answer that ships is the base attempt's, at f* to nine digits.
    assert!(
        r.solution.objective >= -1e-8,
        "reported objective {:.6e} is below f* = 0",
        r.solution.objective,
    );
    assert!(
        r.solution.objective < 1e-6,
        "reported objective {:.6e} is not f* = 0 either",
        r.solution.objective,
    );
    // And it says which gate refused it, in the words that distinguish
    // this decline from an ordinary one.
    assert!(
        out.contains("declined on the ANSWER, not the certificate"),
        "no explanation of the decline on stdout:\n{out}"
    );
}

/// Rule 1, on a model where the objective moved the *other* way — the
/// branch `scholtes4` does not take.
#[test]
fn a_worse_feasible_local_solution_is_not_promoted() {
    let (r, out) = solve("mpcc_worse_local_solution", REPRO);

    assert!(
        r.statistics.dual_divergence_signature,
        "the detector is supposed to fire here. stdout:\n{out}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "the retry was promoted at f = {:.6e}, giving up the base \
         attempt's feasible f = -1.3005680756e1. stdout:\n{out}",
        r.solution.objective,
    );
    assert!(
        r.solution.objective < -1.3e1,
        "the base attempt's answer was not the one reported: f = {:.10e}",
        r.solution.objective,
    );
}

/// The other side of the gate: over-tightening it costs gh#884 its fix.
///
/// `qpec_small`'s promotion moves the objective *worse* by `5.8e-11` —
/// deliberately, to buy nine orders of unscaled dual residual — so a rule
/// that refused any worsening at all would refuse the reproducer.
///
/// **`filter_theta_roundoff_retry=no` is what keeps this reachable, and it
/// is pinned rather than incidental (gh#945).** The trajectory the gate was
/// designed against is the one where this model's base solve stalls at
/// `Solved_To_Acceptable_Level`/41 with an unscaled KKT error of `7.90e4`,
/// which is what makes the retry worth promoting. gh#945 removed the
/// line-search failure that stall comes from, so at the shipped default the
/// base solve reaches the optimum on its own and there is no promotion left
/// to guard — see [`the_reproducer_no_longer_needs_the_retry`], which pins
/// that. Turning the option off here is not preserving a bug; it is holding
/// the gate's *input* fixed so the gate itself stays under test.
#[test]
fn the_reproducer_still_promotes() {
    let mut opts = REPRO.to_vec();
    opts.push("filter_theta_roundoff_retry=no");
    let (r, out) = solve("mpcc_qpec_small_biactive", &opts);
    assert!(
        r.statistics.dual_divergence_retry_promoted,
        "the reproducer lost its fix. stdout:\n{out}"
    );
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
}

/// …and at the shipped default it does not need the retry at all (gh#945).
///
/// The stall that gh#884's detector fires on — `Solved_To_Acceptable_Level`
/// at 41 iterations with an unscaled KKT error of `7.90e4` — begins with the
/// filter line search giving up at an iterate feasible to round-off and
/// handing it to a restoration phase with nothing to minimize. With the
/// gh#945 retry the α-loop finds the step instead, and the *base* attempt
/// converges: `Optimal Solution Found` at 100 iterations,
/// `f = 4.5454149802e-10` at `x = [0.99998, 0.99999, 8.70e-6]`, KKT error
/// `5.65e-10` (unscaled dual infeasibility `6.26e-10`) and constraint
/// violation exactly `0`. That is two orders
/// *better* than the promoted retry it replaces (`9.96e-8`), reached without
/// the `perturb_always_cd` re-solve gh#884 documents as lying on `ralph1`.
///
/// The detector still fires — the signature is a property of the iterate,
/// not of the fix — so what this pins is the gate declining a retry there is
/// nothing to gain from, and the base answer shipping.
#[test]
fn the_reproducer_no_longer_needs_the_retry() {
    let (r, out) = solve("mpcc_qpec_small_biactive", REPRO);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "stdout:\n{out}"
    );
    assert!(
        r.statistics.dual_divergence_signature,
        "the detector must still fire at the default — the signature is a \
         property of the iterate, and it is what says this family has not \
         lost its subject to `filter_theta_roundoff_retry=no`. stdout:\n{out}"
    );
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "the base attempt is supposed to stand on its own here. stdout:\n{out}"
    );
    assert!(
        (0.0..1e-8).contains(&r.solution.objective),
        "reported objective {:.6e} is not f* = 0",
        r.solution.objective,
    );
    for (i, want) in [1.0, 1.0, 0.0].iter().enumerate() {
        assert!(
            (r.solution.x[i] - want).abs() <= 1e-4,
            "x[{i}] = {:.6e} is not within 1e-4 of the optimum {want}",
            r.solution.x[i],
        );
    }
}

/// One run, one answer.
///
/// `set_on_converged` fires once per *attempt*, and the three-sink floor
/// does not reach it — so before this was fixed a declined retry shipped
/// the **discarded** attempt's `x` in `solution.x` and the `.sol` while
/// `status`, `objective` and every statistic beside them had been floored
/// back. Measured: the `.sol` held `f = -6.3274` and the JSON report next
/// to it said `-6.1768`.
///
/// Checked on `scholtes4` because its objective is `x₁ + x₂ − x₃`, so the
/// invariant needs no evaluator — the point and the number it is supposed
/// to have produced are both in the report.
#[test]
fn a_declined_retry_reports_one_answer() {
    let (r, out) = solve("mpcc_scholtes4_biactive", REPRO);
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "this test is about the declined path. stdout:\n{out}"
    );
    assert_eq!(r.solution.x.len(), 3, "unexpected model shape");
    let f_at_x = r.solution.x[0] + r.solution.x[1] - r.solution.x[2];
    assert!(
        (f_at_x - r.solution.objective).abs() <= 1e-9,
        "the reported point and the reported objective are from different \
         attempts: f(solution.x) = {f_at_x:.10e} against solution.objective \
         = {:.10e}. stdout:\n{out}",
        r.solution.objective,
    );
}

/// The same invariant **under a change of variables** — which is the leg
/// that would have caught R1, and the reason
/// `a_declined_retry_reports_one_answer` alone was not enough.
///
/// `CountingTnlp` sits *inside* the gh#486 scaling wrapper and *outside*
/// the presolve one, so its `finalize_solution` payload has already had
/// `x /= d`, `z *= d` applied and has *not* been lifted out of the reduced
/// presolve space. The CLI's floor swap therefore has to skip the gh#486
/// correction and keep the presolve lift. Getting that backwards squares
/// the factor: measured on this fixture under
/// `nlp_scaling_method=curvature-based` (`d = [2, 2, 0.5007]`), the
/// declined retry reported `x = [7.28e-14, 7.28e-14, -3.63e-09]` against
/// the correct `[1.46e-13, 1.46e-13, -1.82e-09]`, with the objective
/// beside it unchanged and right.
///
/// `curvature-based` rather than `user-scaling` deliberately: it derives
/// its own factors, so the leg needs no suffix in the fixture and cannot
/// silently become `d = 1` if one is dropped. The assertion is against the
/// `dual_divergence_retry=no` run rather than a literal, so it stays true
/// if the model's solution moves.
#[test]
fn a_declined_retry_reports_one_answer_under_variable_scaling() {
    const SCALED: &[&str] = &[
        "nlp_scaling_method=curvature-based",
        "bound_relax_factor=0",
        "tol=1e-8",
    ];
    let (r, out) = solve("mpcc_scholtes4_biactive", SCALED);
    assert!(
        !r.statistics.dual_divergence_retry_promoted,
        "this leg is about the declined path. stdout:\n{out}"
    );
    assert!(
        r.statistics.dual_divergence_signature,
        "the detector must fire here or the leg tests nothing. stdout:\n{out}"
    );

    // 1. The reported point and the reported objective agree.
    let f_at_x = r.solution.x[0] + r.solution.x[1] - r.solution.x[2];
    assert!(
        (f_at_x - r.solution.objective).abs() <= 1e-9,
        "f(solution.x) = {f_at_x:.10e} against solution.objective = {:.10e} \
         — the scaling correction landed a different number of times on the \
         point than on the objective. stdout:\n{out}",
        r.solution.objective,
    );

    // 2. ...and they are the answer the run without the retry gives, which
    //    is what pins the *number of times* the correction was applied
    //    rather than merely its self-consistency.
    let mut off = SCALED.to_vec();
    off.push("dual_divergence_retry=no");
    let (base, _) = solve("mpcc_scholtes4_biactive", &off);
    assert_eq!(
        r.solution.x.len(),
        base.solution.x.len(),
        "shape moved between the two runs"
    );
    for (i, (a, b)) in r.solution.x.iter().zip(&base.solution.x).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "x[{i}] = {a:.10e} with the retry against {b:.10e} without it — \
             the declined answer is not the base attempt's point"
        );
    }
}

/// None of the above is reachable at defaults, which is what keeps the
/// cost of the whole mechanism where gh#884 said it was.
#[test]
fn the_default_configuration_reaches_none_of_this() {
    for stem in ["mpcc_scholtes4_biactive", "mpcc_worse_local_solution"] {
        let (r, out) = solve(stem, &[]);
        assert!(
            !r.statistics.dual_divergence_signature,
            "{stem} reaches the detector at default options. stdout:\n{out}"
        );
        assert_eq!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{stem} does not solve cleanly at defaults. stdout:\n{out}"
        );
    }
}
