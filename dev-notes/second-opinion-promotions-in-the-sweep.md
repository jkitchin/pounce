# A promoted second opinion used to read as a speed-up (gh#850)

`scripts/sweep-fixtures.sh` could not see a second-opinion ladder promotion.
When the base solve fails and a rung recovers it, the JSON report's `status`
and `statistics.iteration_count` both become the **promoted rung's**, and
nothing else said the base solver had failed — so a fixture that *lost* its
baseline solve and is now only rescued by a retry read in the diff as a large
improvement.

This is the same shape of invisibility the engine column was added to close
(gh#760). CLAUDE.md already states the rule for engines: "Status, objective and
iteration count can all be unchanged while a model silently changes arms … so a
routing regression used to leave no trace in the diff." A ladder promotion is
that shape.

It is worse than a gap in the evidence. `scripts/sweep-fixtures.sh` is the
repo's primary trajectory guard and CLAUDE.md makes it the *required* evidence
for a trajectory change, so a guard that converts a lost solve into a recorded
win produces positive evidence for the wrong conclusion.

## What was fixed

- `SecondOpinionOutcome` now carries `base_status`, `base_iteration_count` and
  `rung_iteration_counts`, so the base solve's verdict and cost survive a
  promotion instead of being overwritten by the rung's.
- The JSON report gained a `second_opinion` block (additive; absent entirely
  when the verdict opened no ladder, so its *presence* is itself the signal).
- `scripts/sweep-fixtures.sh` gained a `2nd=` column, built from that block:
  `-`, `kept(n),tot=N`, or `<rung>@<base status>/<base iters>,tot=N`.

Pinned by `crates/pounce-cli/tests/issue850_second_opinion_is_recorded.rs`.

## What the new column immediately revealed — and who owns it

Two fixtures in the corpus are solved **only** by a ladder rung. Measured at
the commit that added the column (`infeasibility_perturbed_start_retry=no`
turns the rung off):

| fixture | defaults | rung off |
|---|---|---|
| `square_flowsheet_resto` | `SolveSucceeded`, 54 iters | **`RestorationFailed`, 131 iters** |
| `degenerate_start_hs008` | `SolveSucceeded`, 5 iters | **`InfeasibleProblemDetected`, 7 iters** |

`square_flowsheet_resto` is the one gh#850 reports, and it is a **regression**,
not merely a fixture that has always needed the rung:

| | status | iters | final constr viol |
|---|---|---|---|
| `v0.10.0`, defaults | `SolveSucceeded` | 116 | 4.2e-10 |
| HEAD, defaults | `SolveSucceeded` | 54 | 3.9e-10 |
| HEAD, rung off | `RestorationFailed` | 131 | 6.7e-4 |

`v0.10.0` does not have `infeasibility_perturbed_start_retry` at all — it
rejects the option with `OPTION_INVALID` — so that 116 is the *base solver*
converging, and HEAD's base solver no longer does. The rung that saves it was
added in the same release window. gh#850 bisects the loss to `2c4f25f1`
("perf(feral): wire increase_quality, and turn the backend refinement off for
the IPM (gh#698 obs 5)").

## The regression itself, and what it turned out to be

Made visible first, then fixed. The visible form is what the sweep now prints
on that line — `start_point_perturbation=1e-2@Restoration_Failed/131,tot=185`
— and it is what let the second, worse instance be found at all.

**The lbfgs leg was worse than the reported exact one and nothing was rescuing
it.** With the column in place, `lbfgs square_flowsheet_resto` reads
`MaximumIterationsExceeded`, `it=3000`, `2nd=-`: it ran to the cap, and no
ladder opened because that verdict does not trigger one. The exact leg at
least came back `SolveSucceeded`.

**Which half of `2c4f25f1` did it.** That commit does two things — wires
`increase_quality`, and turns the backend refinement off — and only the first
is responsible. Measured, one binary, the rung switchable:

| `increase_quality` | `feral_refine` | exact leg |
|---|---|---|
| on (0.11 default) | off (default) | `RestorationFailed`, 131 |
| on | **on** | `RestorationFailed`, 131 |
| **off** | off | **`Optimal`, 99** |
| off | on | `Optimal`, 99 |

Refinement makes no difference to this model in either direction, so the
68.9 s → 18.8 s win that `2c4f25f1` bought — which lives entirely in
`refine = false` — is untouched by the fix.

**Why the rung costs a solve.** Ipopt calls `IncreaseQuality` when
`PdFullSpaceSolver`'s refinement stalls, and MA57 answers by raising `pivtol`
toward `pivtolmax`: strictly more conservative each time, so keeping it raised
for the rest of the solve can only make the factorization safer. FERAL's ladder
changes *which pivots are taken*, which is a lateral move in trajectory terms,
and it persists the same way — across every later factorization, including a
restoration sub-solve's. On this fixture it fires exactly twice: once in the
main solve at iteration 25, and once inside restoration at `76r`.

**There is no firing-count policy that separates the cases.** `deb7` and
`square_flowsheet_resto` each fire the rung exactly twice on their exact legs;
one gains 16% of its iterations and the other loses its verdict. So the fix is
a default, not a cap: `feral_increase_quality` is off, and `=yes` restores the
0.11 behaviour for a problem that needs it.

**The trade, from the sweep — 18 fixture-legs move.** Within the CLI corpus the
rung costs a *verdict* twice and buys *iterations* five times:

| | with the rung | without |
|---|---|---|
| `exact square_flowsheet_resto` | `RestorationFailed`/131, rescued at 185 total | **`Optimal`/99, no ladder** |
| `lbfgs square_flowsheet_resto` | `MaximumIterationsExceeded`/3000 | **`Optimal`/178** |
| `exact deb7` | 147 | 171 |
| `exact pooling_rt2stp` | 109 | 128 |
| `lbfgs eigena2` | 186 | 202 (`ErrorInStepComputation` either way) |
| `lbfgs pooling_rt2stp` | 295 | **273** |
| three ladder `tot=` counts on infeasible fixtures | | +3 ‥ +24 |

On that evidence alone the rung looks like a bad trade, and the first draft of
this work flipped the default off. **The workspace suite refuted it**, which is
worth recording because the fixture corpus could not:
`pounce-rs/tests/watchdog_trial_is_not_a_divergence_verdict.rs`'s 12-variable
model ends `SolvedToAcceptableLevel` at `obj = 3.7e-6` **with** the rung and at
`obj = 3.42` against `f* = 0` **without** it. That is a wrong-ish answer under a
success-shaped status — worse than an honest cap failure — and the 158-leg sweep
is blind to it because the model is not a CLI fixture. The same lesson as the
corpus notes in CLAUDE.md, one layer out.

**And nothing separates the two sides.** Measured with a process-global firing
cap, the rung fires exactly twice on `square_flowsheet_resto` — once in the main
solve at iteration 25 and once inside restoration at `76r` — and allowing *only
the first* still loses the leg, so declining it for the restoration sub-solve
would not help. Nor does a count: `deb7` and `square_flowsheet_resto` each fire
it exactly twice on their exact legs, one gaining 16% of its iterations and the
other losing its verdict.

**So the default stands and the rung gets a lever, not a flip.**
`feral_increase_quality=no` recovers both legs of `square_flowsheet_resto`
cleanly (99 and 178 iterations), and is the documented recovery for a model this
rung costs. Pinned by
`crates/pounce-cli/tests/issue850_increase_quality_regression.rs`, which asserts
the trade in both directions.

**What a real fix needs.** A *revertible* escalation — one that does not govern
every later factorization, including a restoration sub-solve's. FERAL's
`quality_level` cannot express that today: it only ratchets up, with no reset.

That is filed upstream as **jkitchin/feral#192** (a `reset_quality()`, or a
scoped escalate-for-one-factor form), and tracked here as **gh#857**, which
also carries the two things still owed on this side: re-running `2c4f25f1`'s
`laptime` 126k benchmark, and retiring `feral_increase_quality` if the reset
makes it moot.

**One thing checked and worth knowing:** gh#590's badly-scaled LP grid
(`issue_590_primal_noise_floor_component`, data scale `1e10` and `1e11`, six
seeds), which `2c4f25f1` cites as needing the escalation once refinement came
off, passes with the rung off — so that particular justification no longer
binds. The perf claim that commit measured lives in `feral_refine`, which none
of this touches.

The cost is understated on the same lines, and the `tot=` field is what says
so. `square_flowsheet_resto` really costs `131 + 54 = 185`, 3.4× its reported
`it=54`; `degenerate_start_hs008` costs 30 against a reported 5; and among the
fixtures where the ladder runs and promotes *nothing*,
`issue_508_infeasible_gap_1em4` costs 982 against a reported 441. Fifteen
fixture-legs carry a `2nd=` entry, and every one of them was previously
reporting a fraction of its true cost.

## Note for the next sweep baseline

Adding the column moves **every** line in the sweep output, so a diff taken
across this commit is not comparable field-by-field with an older baseline.
Re-baseline against a binary built at or after it.
