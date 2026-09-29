# Phase-Changing Flash as a Complementarity Problem

Notebook
[`38_phase_changing_flash_mpcc.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/38_phase_changing_flash_mpcc.ipynb)
solves a single vapour–liquid equilibrium stage across a temperature path
that crosses single-liquid, two-phase and single-vapour — with the number
of variables and equations fixed at every point, and the phase logic
carried by two complementarity conditions.

The reusable implementation is `pounce.examples.flash_mpcc`. The evidence
harness that runs the full route matrix, both traversal directions and the
machine-readable result contract is
[`benchmarks/flash/`](https://github.com/jkitchin/pounce/tree/main/benchmarks/flash);
`python/tests/test_flash_mpcc.py` is the fast regression.

This is Gate 1 of [gh#776](https://github.com/jkitchin/pounce/issues/776).
Gate 0 ([gh#794](https://github.com/jkitchin/pounce/issues/794)) established
which POUNCE route is supported for small MPCCs on a corpus that was
deliberately not a process model; this is the first model with physics in
it, and it stops at one flash.

## Scope

| | |
| --- | --- |
| mixture | ethane / n-butane, equimolar feed |
| thermodynamics | Peng–Robinson, classical one-fluid mixing, `k_ij = 0` |
| constants | the ethane and n-butane rows of `phase_envelope.NATURAL_GAS` |
| condition | 10 bar, 34 temperatures from 230 to 360 K |
| unknowns | vapour fraction and both phases' mole numbers — 5 variables, 8 rows |
| switch points | bubble 268.8896 K, dew 323.3849 K, both interior to the path |

It is **not** a property package, a tray model, or evidence about anything
with more than one equilibrium stage.

## What the formulation is, and is not, for

The notebook measures the honest comparison before making any claim:
ordinary successive substitution solves every point on this path correctly
in 7–10 sweeps, against roughly a hundred interior-point iterations for the
complementarity form. **As a way of computing one flash, the MPCC loses.**

The case for it is composability. Successive substitution is a procedure
with an inner loop and a regime branch, its vapour fraction comes out of a
Rachford–Rice solve that *clips* to `[0, 1]` (so its derivative with respect
to an upstream variable is identically zero in the single-phase regions —
wrong, and silently so), and it carries no phase-state certificate. None of
that can be embedded in a simultaneous transcription and handed to one NLP
solver, which is what a tray, a column, or a dynamic startup model needs.

The complementarity form gives up speed to buy a fixed-dimension algebraic
system with exact derivatives. The notebook's job is to establish that it
gets the right answers on a problem where the answer is independently known,
before it is used where it is not.

## The complementarity pairs

The pairing is **phase amount against stability slack** — not `L ⟂ V`, which
would say the two phases cannot coexist:

```text
z_i = (1 - beta) x_i + beta y_i                          balance
ln x_i + ln phi_i^L(x/Sx) = ln y_i + ln phi_i^V(y/Sy)    isofugacity

0 <= beta      _|_  1 - Sy >= 0                          pair V
0 <= 1 - beta  _|_  1 - Sx >= 0                          pair L
```

At `beta = 0` the balance forces `x = z`, and the isofugacity rows collapse
to Michelsen's tangent-plane stationarity for the trial vapour; `Sy <= 1` is
then exactly `TPD >= 0`. The complementarity is not an encoding of the
stability test — it *is* the stability test. At the bubble point `beta = 0`
and `Sy = 1` hold together, so pair V is biactive; at the dew point pair L
is. The two regime switches are exactly the two biactive points.

## Verification

Every answer is checked against an independent calculation — Michelsen
stability with a multistart per phase label, then Rachford–Rice with a
Newton polish — which shares only the Peng–Robinson fugacity primitive with
the model. Across all 34 temperatures and all four traversal legs (up and
down, cold and warm starts) the regime label always agrees, the vapour
fraction and both phase sums agree to about 1e-11, source complementarity
products land at 1e-12, and no leg is path-dependent.

That check is not ceremony. The first implementation of the model
normalised inside the logarithm as well as inside `phi`, which adds
`ln(Sy/Sx)` to every isofugacity row. The term **vanishes identically in the
two-phase region**, so the broken model solved, converged, and agreed with
the reference at every two-phase temperature — and was wrong only in the
single-phase regimes, the ones the fixture exists to reach. The notebook
reproduces that defect side by side with the correct row.

### Cross-repository result contract

The POUNCE-only harness continues to emit `pounce-flash-results/1`. The reduced
DiscOpt comparison tracked by
[DiscOpt #1526](https://github.com/jkitchin/discopt/issues/1526) emits
`pounce-flash-results/2`, whose schema ships in the Python package as
`pounce.examples/flash_results_v2.schema.json`. The version-2 comparison keeps
source and lowered residuals separate, preserves every residual's definition, and
distinguishes methods that were not run, failed, returned a local point, or returned a
global certificate. The DiscOpt point records the selected liquid/vapour compressibility
roots and their one-root/three-root branches; its source block requires separate EOS,
root-selection, and nontrivial-stationary-point residuals, so an artifact cannot validate
while omitting the algebraic logic unique to the global encoding. A local record is
structurally forbidden from carrying a certified gap or bound.
A completed comparison requires at least one GDP, SOS1, and Scholtes record,
plus DiscOpt provenance with `present: true` and a commit of at least seven
characters. Exact one-per-cell coverage of the declared temperature-by-method
matrix is a producer invariant (Draft 7 cannot express that cross-array join);
the DiscOpt artifact builder checks it before validation.

When DiscOpt is unavailable, version 2 can instead report
`comparison.state: "not_run"`, empty records, and a reason. Its DiscOpt stamp
uses `present: false` and `comparison_run: false`; the commit may be omitted or
null. An installed DiscOpt environment may also report `not_run`, retaining its
known provenance.

## Two findings

**The supported route's second half needs a fallback on a square flash.**
Gate 0's `scholtes_then_ncp` finishes with one exact-product (`prod_eq`)
solve. Here that solve is refused with `Not_Enough_Degrees_Of_Freedom`
after zero iterations at every temperature: a square flash has no objective
and no slack, so making both product rows equalities gives six equality rows
against five variables, and POUNCE refuses that before iteration zero,
mirroring upstream Ipopt. Gate 0's corpus had an objective and never met
this; every equilibrium-stage model meets it by construction.

The gate is correct and is not worked around — the fix is in the route.
`Route.finish_fallback` substitutes the direct `G*H <= 0` lowering where
the equality form cannot run; the two have identical feasible sets (with
`G, H >= 0` the inequality is active only at `G*H = 0`) but an inequality
does not count against the gate. It is a **fallback and not a
replacement**, and that is measured: run unconditionally over Gate 0's
corpus the inequality finish ties on cells solved and cells at the known
optimum, beats the equality finish on complementarity by 27×, and returns
three cells *below* the MPCC optimum where the equality finish returns
none — the one axis Gate 0 chose that finish for. Re-measured with the
conditional in place, Gate 0's corpus is bit-identical, so the fallback
buys Gate 1 its finishing solve at no cost to Gate 0.

**Reverse-mode autodiff loses the Hessian at the bubble point.** Under
`jax.jit`, `jax.hessian` — which is `jacfwd(jacrev(·))` — is wrong by O(20)
at exactly the two path points straddling the bubble temperature, where the
cubic approaches a double root and Cardano's trigonometric branch runs
`arccos` into its endpoint singularity. Forward-over-forward is exact there
and costs the same, so `pounce.examples.flash_mpcc` uses it. The Jacobian is
unaffected in every mode, which is what makes this dangerous rather than
merely wrong: gradients, KKT residuals and converged answers were all
correct while the Hessian was not.

## Limitations

- One equilibrium stage, two components, one operating parameter. Nothing
  here bounds a tray, a column, or a dynamic transcription.
- The reduced DiscOpt GDP/SOS1/Scholtes comparison is tracked by
  [DiscOpt #1526](https://github.com/jkitchin/discopt/issues/1526). Version 1 records
  that no comparison was run; version 2 is the shared artifact once that work runs.
- The path stays far from the mixture critical point. Ethane is above its
  own `Tc` over the top third of it, which is ordinary and is not a
  supercritical *mixture* state.
- POUNCE is a local solver. The reference calculation's stability test is a
  multistart, a stronger claim than the MPCC's single stationary point —
  which is why agreement is evidence and a disagreement would be a finding
  about the MPCC.
