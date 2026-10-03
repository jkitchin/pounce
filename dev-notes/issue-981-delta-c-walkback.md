# gh#981 finding 1 — the `δ_c` walk-back on a rank-deficient Jacobian (open)

**Status: measured, not fixed.** Both candidate fixes below were built, swept
and run against the full `pounce-cli` suite, and both cost other models. The
walk-back is left exactly as gh#592 shipped it (`perturb_delta_c_max_rungs`
default `3`). This note is the starting point for a fix that needs a
discriminator nobody has yet.

## The report

A CSTR (A → B → C, Arrhenius rates) with one equality row written twice —
`balA_again` is exactly `2 · balA` — has a rank-deficient Jacobian at every
point, so `δ_c` is the right perturbation. The gh#592 walk-back withdraws it
anyway. The issue, driven through discopt, measured δ_w climbing to 6.99e19
and a restoration call, against 12 iterations at a peak δ_w of 2.13e-3 with
`perturb_delta_c_max_rungs=0`. Reproduced through the CLI on
`crates/pounce-cli/tests/fixtures/issue981_cstr_dup_row.nl` (Pyomo
transcription, `issue981.py`), at `672320d`:

| run | iterations | restoration | factorizations | peak δ_w |
|---|---|---|---|---|
| default (rungs = 3) | 16 | 0 | 53 | 1.09 |
| `perturb_delta_c_max_rungs=0` | 12 | 0 | 41 | 2.13e-3 |
| default, `mu_strategy=adaptive` | 31 | 2 calls | 94 | 1e10 |
| rungs = 0, `mu_strategy=adaptive` | 10 | 0 | 38 | 1 |

The adaptive rungs-0 row is identical to the CSTR *without* the duplicated
row.

## Why the walk-back fires here

The gh#592 discriminator is "how many δ_x rungs are climbed while δ_c is
up": when δ_c is the right medicine it is *immediately* right. That holds on
eigena2/eigenb2, where the Hessian block is fine. Here δ_c fixes the
constraint block, the nonconvex Hessian block still needs δ_x, the ladder
climbs three rungs under δ_c, and δ_c is withdrawn. Every factorization after
that reports `Singular` — no δ_x makes `[W + δ_x I, Jᵀ; J, 0]` nonsingular
when `J` has a dependent row. Traced with temporary instrumentation:

```
withdraw at δ_x = 2.7e-4      (three rungs under δ_c, all WrongInertia)
Singular  at δ_x = 3.3e-5     (restarted ladder, δ_c off)
Singular  at δ_x = 2.7e-4     (same δ_x the δ_c-on factor was nonsingular at)
...
```

## Candidate A — reinstate δ_c on a refuting `Singular`

A `Singular` from a δ_c-free factorization at or above the rung where δ_c was
withdrawn looks like a controlled comparison: at that δ_x the factor *with*
δ_c was nonsingular, so the singularity should be in the constraint block.
Reinstate δ_c once per aug-system in that case. (Reinstating on *any*
post-withdrawal `Singular`, from the low rungs of the restarted ladder too,
was tried first: `pooling_rt2stp`'s median over a 9-draw round-off `mu_init`
screen went 127 → 245.)

* Fixes the CSTR: 12 iterations, peak δ_w 2.13e-3; adaptive 10 / none / 1.
* `pooling_rt2stp` exact: 17-draw median 127 → 244, four draws above 900
  iterations against one, and the default-start draw lands on the other
  local optimum (−4391.83), which breaks two `issue_592_delta_c_walkback.rs`
  pins.
* `mpcc_qpec_small_biactive` under `bound_relax_factor=0
  mu_strategy_fallback=no` (gh#884's reproducer): `Solve_Succeeded` with
  unscaled dual infeasibility 2.2e-4 instead of 6.3e-10 — the gh#884 failure
  class.

The handler's own state does not separate the CSTR from these: `jac_degenerate`
reads `Degenerate` at every reinstatement on both the CSTR and
`pooling_rt2stp`.

## Candidate B — default `perturb_delta_c_max_rungs = 0`

Swept against `672320d` with only that option changed, both legs, 200
fixture-legs:

| fixture-leg | walk-back on (3) | walk-back off (0) |
|---|---|---|
| exact `pooling_rt2stp` | 162 it, q=4 | 116 it, q=0 (same optimum) |
| lbfgs `pooling_rt2stp` | `ErrorInStepComputation`, 716 it | `SolveSucceeded`, 146 it |
| exact `mu_fallback_point_floor` | 31 it | 46 it (same status, same objective) |
| exact `unbounded_exp` | `ErrorInStepComputation`, 23 it | same status, 32 it |
| exact `infeasible_square_scaled_1em4` | 2nd-opinion total 78 | 74 |
| lbfgs `issue_508_infeasible_gap_1em2` | 2nd-opinion total 284 | 289 |

On that sweep alone it looks like a clear win (the pooling L-BFGS fix holds on
9/9 round-off draws). **The sweep runs default options only, and the
`pounce-cli` suite found three costs it cannot see:**

* `issue_884_biactive_dual_divergence::the_reproducer_no_longer_stalls_at_the_default`
  — the same 2.2e-4 `Solve_Succeeded` as candidate A, under
  `bound_relax_factor=0 mu_strategy_fallback=no`.
* `issue884_promotion_gate_reads_the_answer::a_worse_feasible_local_solution_is_not_promoted`
  — `mpcc_worse_local_solution` under `bound_relax_factor=0
  mu_strategy_fallback=no tol=1e-8` lands on the worse local solution
  (−1.2072 instead of −13.0057) in the base attempt itself.
* `issue_616_ls_init_downgrades::a_declined_step_is_not_the_same_as_never_asking`
  — `pooling_rt2stp`'s two initialization routes become identical (116 vs
  116). With the walk-back off the run is no longer chaotic enough for the
  linear-solver carry-over that test describes to show.

It would also reopen gh#592's over-damped exit on that reporter's model
(not vendorable).

So the walk-back is load-bearing on two MPCC models under
`bound_relax_factor=0`, and counter-productive on the CSTR and on
`pooling_rt2stp`. Both MPCC cases and the CSTR show the same handler-level
signature — δ_c up, three `WrongInertia` rungs, `Singular` without it — so a
fix needs information the perturbation handler does not have today. The
gh#592 note already names the candidate: *which block owns the smallest
pivot* (`min_pivot_index` from feral). An exact structural check for
duplicate rows would cover the reported model but not near-dependence.

## Lessons

* A trajectory change judged by `scripts/sweep-fixtures.sh` alone is judged
  at default options. The MPCC costs above appear only under
  `bound_relax_factor=0`, which the suite exercises and the sweep does not —
  run the suite before believing an empty-or-better sweep.
* `pooling_rt2stp` is chaotic under the walk-back: over the round-off screen
  it scatters between 81 and 3000 iterations and across two local optima.
  Single-draw comparisons on it mean little.
