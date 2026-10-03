# gh#981 finding 1 — the `δ_c` walk-back on a rank-deficient Jacobian, and its default

## The report

A CSTR (A → B → C, Arrhenius rates) with one equality row written twice —
`balA_again` is exactly `2 · balA` — has a rank-deficient Jacobian at every
point, so `δ_c` is the right perturbation. With the defaults the gh#592
walk-back withdrew it anyway. The issue, driven through discopt, measured
δ_w climbing to 6.99e19 and a restoration call, against 12 iterations at a
peak δ_w of 2.13e-3 with `perturb_delta_c_max_rungs=0`. Reproduced through
the CLI on `crates/pounce-cli/tests/fixtures/issue981_cstr_dup_row.nl`
(Pyomo transcription, `issue981.py`), at `672320d`:

| run | iterations | restoration | factorizations | peak δ_w |
|---|---|---|---|---|
| default (rungs = 3) | 16 | 0 | 53 | 1.09 |
| `perturb_delta_c_max_rungs=0` | 12 | 0 | 41 | 2.13e-3 |
| default, `mu_strategy=adaptive` | 31 | 2 calls | 94 | 1e10 |
| rungs = 0, `mu_strategy=adaptive` | 10 | 0 | 38 | 1 |

The adaptive rung-0 row is identical to the CSTR *without* the duplicated
row (10 iterations, peak δ_w 1).

## Why the walk-back fires here

The gh#592 discriminator is "how many δ_x rungs are climbed while δ_c is
up": when δ_c is the right medicine it is *immediately* right. That holds on
eigena2/eigenb2, where the Hessian block is fine. On this model it is not:
δ_c fixes the constraint block, the nonconvex Hessian block still needs δ_x,
and the ladder climbs three rungs under δ_c — so δ_c is withdrawn. Every
factorization after that reports `Singular`, because no δ_x makes
`[W + δ_x I, Jᵀ; J, 0]` nonsingular when `J` has a dependent row. Traced
(temporary instrumentation, not committed):

```
withdraw at δ_x = 2.7e-4      (three rungs under δ_c, all WrongInertia)
Singular  at δ_x = 3.3e-5     (restarted ladder, δ_c off)
Singular  at δ_x = 2.7e-4     (same δ_x the δ_c-on factor was nonsingular at)
...
```

## The reinstatement (kept, for the opt-in walk-back)

A `Singular` from a δ_c-free factorization at or above the rung where δ_c was
withdrawn is a controlled comparison: at that δ_x the factor *with* δ_c was
nonsingular (`WrongInertia`), so the singularity is in the constraint block.
`perturb_for_singular` now undoes the withdrawal in that case — δ_c back at
its old value, the ladder climbing on — once per aug-system, so withdraw and
reinstate cannot cycle. A `Singular` from the low rungs of the restarted
ladder does *not* reinstate: that is the gh#540 unmeasurable-inertia evidence
the walk-back exists to discount.

The first version reinstated on *any* post-withdrawal `Singular`. Over a
round-off `mu_init` screen (`0.1·(1 ± k·1e-12)`, 9 draws) that moved
`pooling_rt2stp`'s median from 127 to 245 iterations — it undid the walk-back
exactly where it was meant to pay — so the rung condition was added.

With the condition it fixes the CSTR (12 iterations, peak δ_w 2.13e-3, 43
factorizations; adaptive 10 / none / 1). But on the corpus it still cost
`pooling_rt2stp` on the exact leg: 17-draw median 127 → 244, four draws above
900 iterations against one. The handler's own state does not separate the two
populations — `jac_degenerate` reads `Degenerate` on both the CSTR and
`pooling_rt2stp` at every reinstatement — so no tighter rule was found.

## Why the default became 0

Swept against `672320d` with nothing changed but `perturb_delta_c_max_rungs=0`
(both legs, 200 fixture-legs):

| fixture-leg | walk-back on (3) | walk-back off (0) |
|---|---|---|
| exact `pooling_rt2stp` | 162 it, q=4 | **116 it, q=0** (same optimum) |
| lbfgs `pooling_rt2stp` | `ErrorInStepComputation`, 716 it | **`SolveSucceeded`, 146 it** |
| exact `mu_fallback_point_floor` | 31 it | 46 it (same status, same objective) |
| exact `unbounded_exp` | `ErrorInStepComputation`, 23 it | same status, 32 it |
| exact `infeasible_square_scaled_1em4` | 2nd-opinion total 78 | 74 |
| lbfgs `issue_508_infeasible_gap_1em2` | 2nd-opinion total 284 | 289 |

Everything else is byte-identical. Under the round-off screen the rung-0
`pooling_rt2stp` run is 116 iterations at −3273.95 on six of nine draws,
where the walk-back run scatters between 81 and 3000 iterations and across
two local optima (−3273.95 and −4391.83). The L-BFGS fix is 9/9 draws.
`mu_fallback_point_floor`'s cost is robust (9/9 at 46) and is a trajectory
difference from iteration 0: with δ_c kept the first step is accepted at
δ_w = 1e4, without it at 1e6, and the two runs reach the same
`Solved_To_Acceptable_Level` point. `unbounded_exp` is an unbounded model that
fails either way.

Together with the gh#693 record in
`crates/pounce-cli/tests/issue_592_delta_c_walkback.rs` — no measured problem
robustly helped by the walk-back, `steenbrd`/`steenbrf`/`steenbrg` robustly
hurt — there is no vendored model on which the walk-back pays. The one it was
written for is gh#592's reporter's LyoPRONTO model, which is GPL-3.0 and
cannot be vendored. **Turning the default off reopens gh#592's over-damped
exit on that model** (measured there in gh#592: 19 iterations, 31810.840,
against 27 and 31785.744 with the walk-back). That cost was accepted by the
maintainer on gh#981 in exchange for the rows above; `perturb_delta_c_max_rungs=3`
restores the walk-back, now with the reinstatement.

## Sweep of the whole gh#981 change

Against `672320d`, both legs, 192 fixture-legs (the three gh#981 fixtures
included): the rows in the table above, unchanged, plus
`issue981_cstr_dup_row` 16 -> 12 iterations, plus an objective-column move on
infeasibility verdicts that is finding 4 (the returned point is now the
restoration iterate): `issue981_wachter_biegler` −1.0885 (exact) and
−1.0714 (L-BFGS) -> −1.0, and last-digit moves on `issue981_water_main`
and `issue_372_infeasible_bounds`. No status and no other iteration count
moved.

## Coverage

* `crates/pounce-common/src/pd_perturbation.rs` — state-machine unit tests:
  off by default; withdrawal after three rungs when opted in; a low-rung
  `Singular` does not reinstate; a `Singular` at the withdrawal rung does,
  at the old δ_c; a reinstated δ_c is not withdrawn again.
* `crates/pounce-cli/tests/issue_981_textbook_findings.rs` — the CSTR at the
  default, at rungs 0, and at rungs 3 (the reinstatement end to end), plus
  the adaptive-μ case.
* `crates/pounce-cli/tests/issue_592_delta_c_walkback.rs` — the default
  equals rungs 0 on `pooling_rt2stp`; the opt-in walk-back still reaches a
  certificate.
