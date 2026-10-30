# Active-Set-Aware Advanced-Step NMPC for a CSTR

Notebook
[`36_active_set_parametric_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/36_active_set_parametric_sensitivity.ipynb)
corrects a held NLP solution for a measured-state change without solving
the NLP again. The correction is taken from the KKT factorization the
solve left behind. The notebook then closes the loop: a 100-interval
collocation controller drives an independently integrated plant, with
guarded fallbacks to a fresh solve.

The reusable closed-loop driver is `pounce.examples.asnmpc_cstr`.
`pyomo-pounce/tests/test_asnmpc_example.py` is its regression test. That
test needs `pyomo-cvp`, which is not a dependency of `pounce-solver`, and
it skips when `pyomo-cvp` is absent. The pyomo-pounce CI job installs
`pyomo-cvp==0.7.2`, so the test does run there. A local run without it
skips silently.

The numbers on this page come from the notebook's saved output, which is
stamped with POUNCE 0.10.0 at source commit `9e11922a`, model revision
`hicks-ray-collocation-asnmpc-v1`, solver `tol=1e-8`, on Linux/WSL2 with
Python 3.13.5. Its wall times belong to that machine.

This is the NMPC case of the applications epic
[gh#780](https://github.com/jkitchin/pounce/issues/780) (child issue
[gh#774](https://github.com/jkitchin/pounce/issues/774), implemented in
[#778](https://github.com/jkitchin/pounce/pull/778)).

## The cumbersome baseline

Advanced-step NMPC (Zavala and Biegler, 2009) moves the expensive solve
off the critical path. The controller solves the next horizon in the
background from the state it *predicts*. When the measurement arrives, it
corrects that solution with one back-solve against the held
factorization. The correction is sIPOPT's parametric step (Pirnay,
López-Negrete and Biegler, 2012).

The plain step is linear in the measurement change and treats every bound
as fixed. That is exactly wrong when a control sits on a limit and the
measurement should move it off. A conventional implementation therefore
either re-solves in full whenever the active set might change, or bolts
active-set logic onto the step by hand. The comparator in this case is the
**full NMPC re-solve after every measurement**, together with the
uncorrected stale prediction as a failure mode.

## The model

This is the Hicks–Ray (1971) CSTR in dimensionless concentration `zc` and
temperature `zt`. It has a coolant valve `v1 ∈ [1/6, 1]` and a
residence-time valve `v2 ∈ [0.025, 1]`, both piecewise constant over 100
one-minute intervals. The model is collocated with 3 Radau points per
element:

```text
dzc/dt = (1 - zc) / (40 v2) - 300 zc exp(-5/zt)
dzt/dt = (0.395 - zt) / (40 v2) + 300 zc exp(-5/zt)
         - 1.95e-4 * 600 v1 (zt - 0.38)
```

The objective is a quadratic regulation cost to the steady state
`(zc, zt) = (0.6416, 0.5387)` with a 1000× terminal weight. The initial
state enters through two declared sensitivity parameters, `zc0` and
`zt0`. From the upset `(0.5216, 0.4887)` the optimum cuts cooling to the
valve minimum and holds it there for the first 13 intervals. That active
bound is what the correction has to handle.

## Open loop: what each correction mode does at an active bound

Each experiment holds one solve and corrects it for a measured state, then
compares the correction against a warm re-solve at that state. The error
reported below is the largest relative error over all four trajectories,
floored at 0.01.

- **`mode="linear"`.** With `clamp=False`, measuring 25% *farther* from
  the steady state (case A) drives `v1` below its minimum on 8 intervals,
  by up to 0.35. `clamp=True` pulls those values back, but every other
  value was computed as though the valve had crossed. The clamp hides the
  error in the valve plot and moves it into the states (error 1.30).
  `sens_solution_report` names the first crossing, `v1[13]` at 4.3% of
  the step, without fixing it.
- **`mode="fix_relax"`.** This is sIPOPT's fix-relax on the same factor.
  It pins the crossing values and re-solves the step with the pin, so the
  rest of the trajectory moves consistently. Error 0.29.
- **`mode="path"`.** This applies the change in pieces and records every
  bound event. Over the full change to the steady state, `v1` leaves its
  minimum interval by interval from minute 12 back to minute 0, at
  fractions 0.013, 0.039, …, 0.819. The first fraction is how much
  measurement discrepancy leaves the held active set intact. No other mode
  produces this record. (`sens_active_set_changes` returns it directly.)
- **Budgets.** `predictor_iter` caps the passes or applied events.
  Halfway to the steady state the error falls 0.763 → 0.417 as the cap
  rises from 1 to 16. `path` costs 16–41 ms and `fix_relax` 21–72 ms,
  against 0.31 s for the re-solve.
- **Corrector.** `corrector_iter` adds Newton back-solves against the same
  factor. It converges while the predictor hands over a few crossings (a
  2% change goes from 2.4e-3 to 1e-6) and gains little past that. It aims
  at the barrier solution at the final `mu`, not at a re-solve.
- **`degeneracy`.** A solve held exactly at the first breakpoint has
  `v1[12]` ambiguous: it sits on the bound with a multiplier of order the
  slack. There `sens_jacobian` warns that its value is one-sided.
  `degeneracy="directional"` (the default) decides the bound for the
  perturbation's own direction from the directional-derivative QP, and
  all three modes reach error 0.0018. Under `"one_sided"`, `linear` and
  `path` stay at 0.0077.

## Closed loop

`run_campaigns` runs 30 samples in each of three seeded campaigns. All six
policies receive identical measurements:

- `full_resolve`
- `no_correction`
- `clamped_linear`
- `fix_relax`
- `path`
- `guarded_path`, which is `path` behind a full-point guard and a fresh-solve fallback.

The plant is integrated separately with RK4, with disturbances and model
mismatch the controller does not see. The controller temperature limit is
0.55. The deadline is an illustrative 100 ms.

| campaign | policy | IAE | max T violation | online median (ms) | fallbacks |
| --- | --- | ---: | ---: | ---: | ---: |
| nominal | full_resolve | 1.5567 | 0 | 258.9 | — |
| nominal | path, fix_relax, clamped_linear, guarded_path | 1.5567 | 0 | 18–72 | 0 |
| constraint_switching | full_resolve | 1.4091 | 0 | 258.9 | — |
| constraint_switching | no_correction | 10.621 | **0.541** | 0 | — |
| constraint_switching | path, fix_relax, guarded_path | 1.4091 | 0 | 69–80 | 0 |
| constraint_switching | clamped_linear | 1.4131 | 0 | 20.0 | 0 |
| stress_model_mismatch | full_resolve | 2.3331 | 0 | 264.9 | — |
| stress_model_mismatch | guarded_path | 2.3331 | 0 | 75.8 | 1 of 30 |
| stress_model_mismatch | path, fix_relax | 2.3796 | 0 | 77–88 | 0 |
| stress_model_mismatch | clamped_linear | 2.3707 | 0 | 34.5 | 0 |

`full_resolve` misses the 100 ms deadline on all 30 samples in every
campaign. The corrected policies keep their roughly 260 ms solve in the
background.

- **Nominal.** Every corrected policy reproduces the full solve, with an
  IAE difference of at most 7.9e-7.
- **Constraint switching.** The stale prediction violates the temperature
  limit, while `path` and `fix_relax` track the full solve. `path` records
  the one event that matters, `v1[1]` leaving its minimum at fraction
  0.556 of sample 11.
- **Stress.** The initial concentration bias is 4.5× the configured trust
  scale, so it deliberately forces an out-of-validity update. `path`
  records 11 events on sample 0 and applies them. `guarded_path` rejects
  that sample (reasons: measurement displacement, corrector feasibility,
  corrector progress, stationarity) and falls back to a fresh solve, which
  is why it alone matches `full_resolve`. The fallback is
  `guarded_path`'s one deadline miss. `path` and `fix_relax` also miss
  the deadline once each in this campaign.

The guard accepts a corrected point only when all of the following hold:

- the scaled measurement displacement is inside the trust region;
- full-point feasibility, stationarity and complementarity are each below
  their limits;
- the corrector made progress;
- the path budget was not exhausted;
- the predicted temperature is safe;
- no manipulated variable's activity is ambiguous.

A rejection re-solves and resets both the warm start and the held factor.

## Verification

The notebook's last code cell asserts:

1. the campaign × policy matrix is 3 × 6, with 30 samples each;
2. the nominal corrected-versus-full IAE difference is below 1e-6;
3. the stale prediction violates the limit in the switching campaign,
   while no corrected policy violates it in any campaign;
4. the guarded fallback rates are 0 / 0 / 0.033;
5. the event ledger holds 24 active-set transitions;
6. at a kink, the one-sided and directional controls differ (by 2.3e-3).
   The guard rejects both probes there rather than silently choosing one
   derivative.

## Reproduce it

This case needs the optional Pyomo stack:

```sh
pip install pyomo-pounce pyomo-cvp==0.7.2
python -m pytest -q pyomo-pounce/tests/test_asnmpc_example.py
jupyter nbconvert --to notebook --execute --inplace \
  python/notebooks/36_active_set_parametric_sensitivity.ipynb
```

## Limitations

- This is a deterministic research case, not a production controller or a
  safety certificate. The RK4 plant is independent of the collocation
  controller but is still a model.
- The 100 ms deadline and every wall time belong to the recorded machine.
  The guard thresholds are illustrative.
- The online timing includes the applied estimate and a separate
  diagnostic replay, because the public API computes their
  predictor/corrector work independently. It excludes the third replay
  that builds the event ledger.
- At large measurement changes every first-order prediction misses parts
  of the trajectory badly. At the full change to the steady state,
  `fix_relax` and `path` still have error 0.63, against 0.83 for `linear`.
  The active-set modes repair the active set, not the curvature.
- A real deployment still needs state estimation, robust constraint
  handling, interlocks, deadline-aware scheduling, and validation on
  target hardware.

## References

- J. G. Hicks and W. H. Ray, *Can. J. Chem. Eng.* 49 (1971) 522–528,
  [doi:10.1002/cjce.5450490416](https://doi.org/10.1002/cjce.5450490416).
- V. M. Zavala and L. T. Biegler, *Automatica* 45 (2009) 86–93,
  [doi:10.1016/j.automatica.2008.06.011](https://doi.org/10.1016/j.automatica.2008.06.011).
- H. Pirnay, R. López-Negrete and L. T. Biegler, *Math. Prog. Comp.* 4
  (2012) 307–331,
  [doi:10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).

## See also

- [Sensitivity Analysis](sensitivity.md): the held-factor update and the
  active-set modes.
- [Pyomo](pyomo.md): `sens_solution`, `sens_solution_report` and
  `sens_active_set_changes`.
- [Path Following & Inverse Mapping](path-following.md): the path-record
  machinery behind bound crossings.
