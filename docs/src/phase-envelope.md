# Differentiable Phase Envelopes and Inverse Mixture Design

Notebook
[`34_phase_envelope_peng_robinson.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/34_phase_envelope_peng_robinson.ipynb)
traces the full phase envelope of a Peng–Robinson mixture with one
continuation run. It then refines the cricondenbar and cricondentherm as
augmented fold equations, differentiates both with respect to composition
and a binary interaction parameter, and designs a composition that hits a
target maxcondentherm. Every result is checked outside the solve that
produced it.

The reusable implementation is `pounce.examples.phase_envelope`, and
`python/tests/test_phase_envelope_example.py` is its regression test. The
numbers on this page come from the notebook's saved output. That output is
stamped with POUNCE 0.10.0 at source commit `fefa6ec3`, JAX 0.9.2 on CPU.
Re-execute the notebook to restamp it.

This is the phase-envelope case of the applications epic
[gh#780](https://github.com/jkitchin/pounce/issues/780) (child issue
[gh#773](https://github.com/jkitchin/pounce/issues/773), implemented in
[#777](https://github.com/jkitchin/pounce/pull/777)).

## The cumbersome baseline

A phase envelope is conventionally built by marching flash or saturation
calculations in pressure or temperature, then restarting by hand at each
turning point. Its design derivatives come from finite-difference
retracing: perturb the composition, trace the whole curve again, and
difference the extrema. Both steps fail in a characteristic way, and the
notebook keeps both failures in:

| method | outcome on the 4-component gas | points |
| --- | --- | ---: |
| pressure ladder, `jp.solve` in a loop | fails loudly at 98 bar; no solution exists past the cricondenbar | 36 |
| `PathFollower.follow` in `ln P` | reports `status="ok"` on the **trivial branch** `K_i = 1` | 9 |
| `PathFollower.trace_arclength` | traces the whole envelope through both folds and the critical point | 323 |

The second row is the warning the notebook is built around. `K_i = 1`
makes the two phases identical, which satisfies the equilibrium equations
exactly at every `(T, P)`. The `follow` run converges onto that branch to
a residual of `1e-15`, and its KKT monitor cannot object because nothing
is wrong with the point. A residual says nothing about whether the answer
is physical. `max|ln K|` has to be watched explicitly away from the
critical neighbourhood.

## Equations

The mixture is 80/10/6/4 methane/ethane/propane/n-butane with all
`k_ij = 0` (`NATURAL_GAS`). The incipient-phase condition of Michelsen
(1980), at fixed vapour fraction `beta`, is

```text
x_i = z_i / (1 - beta + beta K_i),   y_i = K_i x_i
g_i = ln K_i + ln phi_i^V(y, T, P) - ln phi_i^L(x, T, P) = 0,   i = 1..N
g_{N+1} = sum_i (y_i - x_i) = 0
```

The unknowns are `X = [ln K_1 .. ln K_N, ln T]`, with `theta = ln P` as the
continuation parameter, which makes the system square, `N + 1` by `N + 1`.
Swapping the roles, carrying `ln P` as a state and continuing in
`ln T`, gives an independent parameterization of the same curve.

POUNCE has no `root()` entry point and needs none. As in
[Boundary Value Problems](bvp.md), the square system is posed as the
feasibility NLP `min 0` s.t. `F(X) = 0`. For a nonsingular Jacobian the
interior-point step reduces to the Newton step.

The Peng–Robinson cubic is solved in closed (trigonometric Cardano) form
so that the residual stays one differentiable expression. Under
`jax.jacobian` both root branches are evaluated, and a `NaN` in the dead
branch would poison the live gradient, so each inactive branch is clamped
before its `sqrt`/`arccos`. The vapour takes the largest real root and the
liquid the smallest. That is a local root-selection convention, **not** a
tangent-plane-distance stability test.

The EOS is checked before anything is built on it:

- `Z -> 1` as `P -> 0`;
- gradients are finite in both the one-root and three-root regions;
- pure n-butane vapour pressure is 1.034 / 2.573 / 9.469 bar at
  273.15 / 300 / 350 K, against literature values of about
  1.03 / 2.58 / 9.42.

## What the POUNCE workflow does

1. **Trace.** `PathFollower(jp).trace_arclength` starts from a Wilson
   K-value anchor at 1 bar. It walks the dew branch, passes the
   cricondenbar (97.59 bar) and the critical point (246.35 K, 92.16 bar,
   where `max|ln K|` reaches 0.004), and comes out on the **bubble**
   branch without being told to. Methane's `K` inverts from 1.244 to 0.906
   across the critical point. The `ln T` run detects the cricondentherm
   (283.91 K). Each run's turning point agrees with the plain maximum of
   the other run to 0.018 K and 0.002 bar.
2. **Read the slope from the factor.** `sensitivity_at` returns
   `d ln T / d ln P` along the boundary from the KKT factor. It agrees with
   a finite difference along the path to 1% relative and grows 122× toward
   the cricondenbar, where it diverges. The number the continuation needs
   and the number that blows up are the same number. This is why
   parameter continuation cannot pass the fold.
3. **Refine the folds.** `PathTrace.turning_points` only brackets a fold
   to the trace spacing. `refine_fold` solves the augmented simple-fold
   system `[F, F_X v, v^T v - 1] = 0` with POUNCE, seeded from that
   bracket. Its residual is `max|G| ~ 1e-14`.
4. **Differentiate the extrema.** The same implicit solve returns
   `d(extremum)/dq` for the `N - 1` log-ratio composition coordinates
   `eta_i = ln(z_i / z_N)` and one `k_ij`.
5. **Design.** `design_composition_for_extremum` solves for the smallest
   log-ratio change that moves the maxcondentherm to a target. Its design
   variables are unconstrained log ratios, so every iterate stays inside
   the simplex.

## Validation independent of the solve

The four-component gas is a worked example, not a reference. The
independent benchmark is Deiters and Bell's (2019) Peng–Robinson
methane/propane liquid isopleth at `x_methane = 0.8`, `k_12 = 0.042823`
(`DEITERS_BELL_METHANE_PROPANE`):

| check | result |
| --- | --- |
| published maxcondentherm 282.53 K | POUNCE refined fold 282.5253 K (Δ 0.0047 K) |
| trace step size `ds = 0.10` vs `0.08`, after refinement | all four fold coordinates agree to ≤ 8.4e-12 |
| implicit fold derivatives vs **four full independent retraces** (central differences in `eta` and `k_12`) | max relative error 1.07e-8 |
| inverse design to 280 K, then a fresh Wilson anchor, full retrace and independent fold refinement | 280.000000000 K; independent error 3.0e-11 K |
| physical guards at every refined fold (root admissibility, distance from the trivial branch) | pass; branch distance ≥ 0.77 |
| retrograde condensation: isotherms between `T_c` and the cricondentherm | 3 of 3 cross the dew branch twice |

The derivative check deliberately pays for full retraces rather than
differentiating a second expression at the same fold. The design is
accepted only after a retrace that starts from scratch, so what is checked
is the designed mixture itself, not the optimizer's own constraint
residual. The designed composition is `(0.81107, 0.18893)`, from
`(0.8, 0.2)`.

## Reproduce it

From the repository root, with the Python development environment
installed:

```sh
PYTHONPATH=python python -m pytest -q python/tests/test_phase_envelope_example.py
jupyter nbconvert --to notebook --execute --inplace \
  python/notebooks/34_phase_envelope_peng_robinson.ipynb
```

The notebook's last code cell asserts each row of the validation table
against a threshold rather than the printed value. Examples are the
published error `< 0.02 K`, the retrace relative error `< 1e-5` and the
inverse-design error `< 1e-6 K`. A re-execution that drifts past a
threshold therefore fails instead of silently restamping.

## Limitations

- This is an inspectable application example, not a production property
  package. Classical one-fluid Peng–Robinson with these mixing rules may
  be inadequate for strongly associating or polar mixtures.
- The largest/smallest-`Z`, residual, normalization and trivial-branch
  guards are local checks. They do **not** replace a tangent-plane-distance
  phase-stability calculation.
- `trace_arclength` requires a scalar parameter and equality constraints.
  The augmented solve assumes a simple fold, and a higher-order
  singularity needs a different formulation. General branch switching is
  out of scope.
- Both traces end with `status="corrector_failed"`. This happens at the
  cold end of the bubble branch, where the heavy components' `K` runs past
  a few hundred and the corrector cannot take a full `ds` step; the
  envelope is complete well before that point. A terminal rejection is
  coverage evidence, not proof that every branch was found.
- The inverse problem is nonconvex. The fresh retrace verifies the target
  and the physical branch. It does not certify that the composition change
  is globally minimal.

## References

- D.-Y. Peng and D. B. Robinson, *Ind. Eng. Chem. Fundam.* 15 (1976)
  59–64, [doi:10.1021/i160057a011](https://doi.org/10.1021/i160057a011).
- M. L. Michelsen, *Fluid Phase Equilibria* 4 (1980) 1–10,
  [doi:10.1016/0378-3812(80)80001-X](https://doi.org/10.1016/0378-3812(80)80001-X).
- U. K. Deiters and I. H. Bell, *AIChE Journal* 65 (2019) e16730,
  [doi:10.1002/aic.16730](https://doi.org/10.1002/aic.16730).

## See also

- [Path Following & Inverse Mapping](path-following.md): `follow`,
  `trace_arclength`, and fold refinement.
- [Sensitivity Analysis](sensitivity.md): where `dx*/dtheta` comes from.
- [Phase-Changing Flash as an MPCC](flash-mpcc.md): the same
  Peng–Robinson constants, solved across a phase change at fixed
  temperature.
