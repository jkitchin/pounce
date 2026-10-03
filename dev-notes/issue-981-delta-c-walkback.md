# gh#981 finding 1 — the `δ_c` walk-back on a rank-deficient Jacobian

**Status: fixed by the certified-singular reinstatement — see "The fix" at
the end.** The first two candidates below were built, swept and run against
the full `pounce-cli` suite, and both cost other models; they are kept
because the measurements that disqualified them are what the fix is
measured against. The discriminator nobody had is the linear solver's own
verdict on *why* it reported `Singular`.

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

## The fix — a `Singular` the solver can vouch for

`pounce-feral` reports `ESymSolverStatus::Singular` for three different
reasons, and the handler could not tell them apart:

1. the factorization found a **zero pivot** (`inertia.zero > 0`, or feral's
   own `FactorStatus::Singular`) — a rank deficiency it measured;
2. gh#540's **untrusted inertia**: the count disagrees with the expected one
   and the smallest pivot is under the `n·ε` trust floor — evidence about the
   measurement, not about the Jacobian;
3. the absolute `feral_singular_pivot_floor` (`1e-20`).

Traced with a per-factorization pivot dump (`inertia`, smallest pivot and
its original index, the `Singular` reason) at every withdrawal event:

| model (options) | how `δ_c` came up | first `δ_c`-free factor | zero-pivot `Singular`s in the run |
|---|---|---|---|
| `issue981_cstr_dup_row` | seeded, `jac_degenerate` | `Singular`, **zero pivot**, `inertia=(3,3,1)`, pivot in the constraint block (index 6 of `x:0-3, c:4-6`) — at every rung | 25 |
| `mpcc_qpec_small_biactive` (`bound_relax_factor=0 mu_strategy_fallback=no`) | untrusted-inertia `Singular`, `neg=7 exp=6` | untrusted-inertia `Singular`, `min_piv=5.5e-17`, `neg=7 exp=6` — at every rung up to `δ_x=2e19` | **0** |
| `mpcc_worse_local_solution` (same + `tol=1e-8`) | untrusted-inertia `Singular` (iter 105); the "`δ_x` exhausted, `δ_c==0`" recovery (iters 0, 113) | `WrongInertia`, `neg=7..8 exp=6` | **0** |
| `pooling_rt2stp` exact | seeded, `jac_degenerate` (8 withdrawals) | `WrongInertia` or untrusted; a zero pivot appears at one rung of two long climbs (`δ_x=2.3e3`, `5.8e3`) and not at its neighbours | 8 |
| `pooling_rt2stp` L-BFGS, iter 56 | untrusted-inertia `Singular` | **zero pivot**, at every rung from `3e-5` to `70` | 1988 |

Both MPCC models' `Singular`s are the gh#540 kind, every one; the CSTR's are
zero pivots, every one. The MPCC runs are also a different physics: the
extra negative eigenvalue persists to `δ_x = 3.6e17` because the runaway
multiplier (gh#884) makes `W` indefinite at that scale — `δ_x` is the only
medicine, and the walk-back is right to withdraw `δ_c`.

**Rule.** `SparseSymLinearSolverInterface::singularity_certified()` (new,
defaulted `false`, implemented by `pounce-feral` as "reason 1") is carried
through `SymLinearSolver` / `AugSystemSolver` to `PDFullSpaceSolver`, which
passes it to `PdPerturbationHandler::perturb_for_singular_with(mu,
certified, ..)`. After a gh#592 withdrawal, a certified `Singular` from the
**first** `δ_c`-free factorization refutes it: the factor at the withdrawal
rung *with* `δ_c` was nonsingular (`WrongInertia`), only `δ_c` changed, so
the zero pivot is in the constraint block and no `δ_x` removes it. `δ_c`
goes back at `δ_cd(μ)`, the ladder resumes from the withdrawal rung, and
`delta_c_reinstated` latches so neither the walk-back nor the reinstatement
fires again in that aug-system. Anything else — an uncertified `Singular`,
a `WrongInertia`, or a zero pivot that only shows up later in the climb —
leaves the withdrawal standing (`withdrawal_untested` closes after one
report). The "first factor only" clause is what the pooling exact leg
demanded: honouring the later zero pivots too (the first prototype) moved
the default draw onto the `−4391.83` optimum and failed two
`issue_592_delta_c_walkback.rs` pins; with it the exact leg is unchanged.
Reasons 2 and 3 are deliberately *not* certified; reason 3 has not been
measured as a discriminator.

**Measured (prototype, branch head as baseline).**

| run | before | after |
|---|---|---|
| CSTR default | 16 it, 53 fact., peak δ_w 1.09 | 12 it, 42 fact., peak δ_w 2.13e-3 (rungs=0: 12 / 41 / 2.13e-3) |
| CSTR `mu_strategy=adaptive` | 31 it, 2 restoration calls, δ_w 1e10 | 10 it, none, peak 1 |
| `mpcc_qpec_small_biactive` (gh#884 options) | `Solve_Succeeded`, unscaled dual 6.26e-10, 100 it | identical (0 reinstatements) |
| `mpcc_worse_local_solution` (gh#884 options) | −13.0057, 17 it | identical (0 reinstatements) |
| `pooling_rt2stp` exact, 17-draw round-off screen | median 127, min 81, max 3000; 10/17 at −3273.95 | median 126, min 81, max 1542; 10/17 at −3273.95; 11 draws bit-identical |
| `pooling_rt2stp` L-BFGS, same screen | 17/17 `ErrorInStepComputation`, 716 it | 17/17 `SolveSucceeded`, 146 it |

Fixture sweep against the branch head, both legs: three lines move —
`exact issue981_cstr_dup_row` 16 → 12; `lbfgs pooling_rt2stp` as above;
`exact infeasible_square_scaled_1em4` second-opinion total 78 → 74 (the
third rung reinstates at iter 3 on a zero pivot in its constraint block,
`inertia=(2,1,1)`, and accepts one rung later; status and objective
unchanged). The two MPCC fixtures do not move on either leg.
