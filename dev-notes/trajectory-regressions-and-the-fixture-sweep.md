# Trajectory regressions: how #544 shipped in 0.10.0 and came back as #592

A post-mortem, written after #592 was fixed. The point is not that a bug
shipped — that happens — but that this one was **measured, recorded, and
reclassified as an accepted cost** three days before the release, by a chain of
individually defensible decisions. The chain is worth knowing because every
link in it will look reasonable again next time.

The operational conclusion is one line, and it is in `CLAUDE.md`: a change that
reroutes *which* correction the solver reaches for is a trajectory change, and
trajectory changes need `scripts/sweep-fixtures.sh` before merge.

## What happened

| when | what |
|---|---|
| Aug 8 | `a0362be4` — #540's fix. Adds `feral_inertia_pivot_floor`, default `1e-12`. Reroutes a mismatching inertia count from the `δ_x` ladder to `δ_c`. |
| Aug 8 | Same change takes `pooling_rt2stp` from 206 to 812 iterations. Known at the time. |
| Aug 8 | `3eea43bb` — #547. `pooling_rt2stp` has been timing out in CI. Diagnosed correctly as a wall-clock cap sized like a perf budget; caps replaced with one 300 s hang guard. |
| Aug 11 | `v0.10.0` tagged. |
| ~Aug 14 | #592 filed: `Solve_Succeeded` at a point POUNCE immediately improves on restart. Same root cause. |

## The four decisions

**1. An explicit argument for skipping the corpus.** From `a0362be4`'s own
commit message:

> "The new feral_inertia_pivot_floor (default 1e-12) is consulted only once the
> count has already mismatched — on a factorization the caller was going to
> reject either way — so it can never turn a usable factor into a failure; **it
> only changes which perturbation is reached for first. That is what makes the
> default safe to change without the benchmark corpus.**"

The premise is true and the conclusion does not follow. The safety argument
reasons about *factorization outcomes*: no good factor becomes a failure.
The risk lives in *step sequence*, and the same sentence names it — "changes
which perturbation is reached for first." Correct reasoning, wrong axis, used
to waive the one check that covers the other axis.

This is the transferable lesson. "It cannot produce a wrong answer" is not the
relevant safety property for a step-computation change, because a trajectory
regression produces the *right* answer, slowly — or a differently-wrong
tolerance-legal answer, which is what #592 was.

**2. A dimension-dependent quantity pinned to a scalar.** The 0.10.0 release
notes describe the default as "`1e-12`, the middle of the `n·eps` range over
which an equilibrated pivot loses its sign." A quantity acknowledged in the
same clause to depend on `n` was fixed by taking the middle of its range.
`1e-12` is `n·eps` at `n ≈ 4500`; the KKT systems in #592 are order 165–311,
where `n·eps` is 3.7e-14 … 6.9e-14 — two decades away. Nothing tied the
constant to the system being factored. It now does
(`pounce_feral::inertia_trust_floor`).

Worth stating plainly: the defect was legible in the release notes at release
time. Writing the rationale down was not enough; nothing re-read it against
the models it would act on.

**3. A measured regression became an accepted cost with no tracking.**
`pooling_rt2stp` 206 → 812 was known. #547's commit calls it "a recorded cost
of that fix — and the cap was not revisited then." A 4× iteration regression on
a corpus model is a defect report about the fix that caused it. Recorded as a
cost, with no issue and no owner, it is indistinguishable from noise by the
next reader.

**4. The last signal was retired for good reasons.** #547 is a *good* commit.
Its reasoning is right: nothing in that file asserts a time, no assertion
weakens as the cap grows, `max_wall_time` is wall clock under `cargo test`
contention, and the timeout had been misattributed to `dual_diverging_streak`.
Raising 10 s → 300 s was correct. But the timeout was, accidentally, the only
instrument in the suite pointed at trajectory length — and fixing the
instrument's calibration removed the reading without anyone deciding to.

## Why nothing else caught it

- **No corpus sweep runs anywhere.** CI is: Test, pounce-rs facade, hsl
  compile-check, WASM, two Python lanes. The 57 fixtures are already in-tree at
  `crates/pounce-cli/tests/fixtures`; nothing ran all of them.
- **The assertions are status + objective, not trajectory.** 28 test files
  check `iteration_count`, but each pins the single model it was written for.
  `pooling_rt2stp` still reached the right objective with the right status —
  four times slower. Invisible to every assertion in the suite.
- **n = 2 confirmation.** The only models exercised were `eigena2` and
  `eigenb2` — the two the fix was developed against. Both improved. A fix
  validated only on the population it was tuned on has not been validated.

## The structural problem, which is still open

#540 and #592 are the same defect class: a wrong *first guess* between `δ_c`
(regularise the constraint block) and `δ_x` (regularise the Hessian). Before
#540 the guess was wrong for degenerate-Jacobian models; #540 moved it and made
it wrong for full-rank-Jacobian models it had no visibility into. It traded one
population for another while holding only the first in view.

The #592 fix does **not** close this class. The walk-back is a bounded recovery
from a wrong guess (three rungs, then withdraw), not a discriminator. The
discriminator needs `min_pivot_index` from feral — which block owns the
smallest pivot — and feral is an external crates.io dependency, so that is
follow-up work, recorded in `issue-592-restart-non-idempotence.md`.

**Anything that touches this guess should be assumed to trade populations
until a sweep shows otherwise.**

## What to do instead

`scripts/sweep-fixtures.sh` — 57 fixtures already in the repo, status +
objective + **iteration count**, sorted and diffable, about two minutes:

    git stash && cargo build --release && cp target/release/pounce /tmp/p-base
    git stash pop && cargo build --release
    scripts/sweep-fixtures.sh /tmp/p-base           /tmp/base.txt
    scripts/sweep-fixtures.sh target/release/pounce /tmp/new.txt
    diff /tmp/base.txt /tmp/new.txt

An empty diff is the expected result for a change not meant to move the corpus.
Every line that moves should be explainable **before** merge. On #592 the diff
was two lines (`eigena2` 27 → 26, `eigenb2` 68 → 67, both improvements, no
status changes), and that is what made it safe to land a default change. Run
against #544 it would have printed `pooling_rt2stp 206 → 812` immediately.

Two things it does not do, deliberately: it is not wired into CI (57 solves is
too slow for every PR, and it needs a *baseline* binary to be meaningful), and
it does not assert anything. It is an instrument, not a gate. The judgment
about which moved lines are acceptable stays with the person merging — the
failure here was never a missing assertion, it was a missing reading.

## Round two: the leg the sweep did not run (#677)

The instrument above has a blind spot, and it cost a second silent defect.

Until #677 the sweep ran each fixture **once**, on the default exact-Hessian
path. Nothing in the corpus ever set `hessian_approximation=limited-memory`,
and no benchmark did either. So `limited_memory_initialization` — registered
with Ipopt's `scalar1` default and never read, pinning every limited-memory
solve to `scalar2` — was invisible to the one instrument built to catch
exactly this. The option had been inert since the option port.

What makes it worth recording is how *thoroughly* invisible it was:

- The **unit tests passed, correctly.** `initial_hessian_scalar` has a test
  per formula, and each one computes its formula right. They assert what σ is
  *once you have chosen a rule*; nothing asserted which rule got chosen. A
  formula test cannot see a selection bug.
- The **option registry was right.** `upstream_options.rs` declared
  `scalar1`, matching Ipopt. The registry and the behaviour disagreed, and
  nothing compares them.
- `unimplemented_options.rs` did not list it, so setting the option was a
  silent no-op — no effect, no warning, and no way for a user to tell.
- The **corpus never ran the path**, so the trajectory guard was silent too.

Read together: four independent layers of coverage, and the defect sat in the
seam between all of them. It surfaced only when an outside user compared
POUNCE against Ipopt on a 59,939-variable collocation model and posted both
iteration logs — the two agreed to the last digit at iteration 1 and separated
at iteration 2, which is where the first curvature pair lands and σ stops
being the hard-coded `1.0` of the empty-history branch.

The fix to the instrument is the cheap half: `sweep-fixtures.sh` now runs
**two legs**, `exact` and `lbfgs`, prefixing each line with the leg name. Same
corpus, twice, one diffable file. That is not exotic coverage — both the
Python frontend and the CasADi plugin select `limited-memory` on their own
whenever no exact Lagrangian Hessian is available, so the L-BFGS leg is what
an embedder gets *by default*, without any user typing the option. The corpus
was not sweeping a rare opt-in; it was not sweeping the common embedded path.

The generalisable lesson is narrower than "add more tests". It is: **a
registered option whose value is never read is invisible to every layer of
testing that exists here**, because each layer checks behaviour *given* a
configuration and none checks that configuration reaches the algorithm. The
structural guard is to fail the build for any registered option with no read
site, which turns this class from a user-reported trajectory mystery into a
compile error. #551 is the standing list of the rest of them; the audit for
#677 found 26 core options still in that state.
