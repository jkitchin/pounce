# Equivalent-Circuit AC-OPF Suite (`grid_ecf` / `grid_polar`)

A **controlled formulation comparison** on AC optimal power flow. The same
pglib-opf cases are emitted twice, from one parser, with one objective, one
thermal-limit convention, and one starting point — and only the formulation
differs:

* **`grid_ecf`** — the equivalent-circuit formulation (ECF) of Jereminov,
  Pandey & Pileggi, *Equivalent Circuit Formulation for Solving AC Optimal
  Power Flow* (CMU, 2018). Rectangular voltages, generators as a
  negative-conductance / susceptance **GB macro-model**, `|V|²` carried as
  the state variable `dsq`. **No trigonometry**: every row is bilinear or
  quadratic.
* **`grid_polar`** — the textbook polar power-balance model, as the control.

Both are written by `ecf_nl_export.py`. It is the only tracked source here;
the downloaded `.m` cases and the generated `.nl` are regenerated locally,
like every other suite.

## Why this is a separate suite from `benchmarks/grid/`

`benchmarks/grid/` is also polar AC-OPF on the same cases, so it looks like
the control is already there. It is not usable as one. Its `.nl` were written
by Egret, whose encoding carries per-branch flow variables — 191 variables for
`case14_ieee` against 38 (polar) and 62 (ECF) here. A formulation difference measured against it
would be confounded by an encoding difference, which is exactly the error this
suite exists to avoid. `grid_polar` is the matched baseline; `grid` stays as
the historical polar reference and is not touched.

## What the comparison is for

The paper's claim is that ECF is far more robust to the starting point,
because the polar model's trigonometry is what lets an interior-point method
wander into a region it reports as infeasible. The paper couples that claim to
a bespoke solver (ESCAPE: SPICE-style per-variable limiting, diode-form
complementarity, Tx-stepping homotopy) and argues a general-purpose
primal-dual interior-point method is the wrong tool — but formulation and
solver are separable, and the paper never runs the separation.

This suite runs it. Both `.nl` sets go through the same shared NL driver as
every other suite, so the only variable is the model.

## Correctness check

Under the default `--thermal apparent`, **both** forms reproduce the published
pglib-opf optimal objectives. That is the check that the parser, the
transformer pi-model, the shunts and the angle limits are all right, and it is
checkable against a reference outside this repo:

| case | published | polar | ECF |
|---|---|---|---|
| case3_lmbd | 5812.6430 | 5812.642959 | 5812.642962 |
| case5_pjm | 17551.8908 | 17551.890839 | 17551.890840 |
| case14_ieee | 2178.0804 | 2178.080411 | 2178.080413 |
| case30_ieee | 8208.5154 | 8208.515428 | 8208.515436 |
| case118_ieee | 97213.6070 | 97213.606943 | 97213.606986 |

A run whose objectives drift from these has a modelling bug, not a solver
finding.

## Two conventions worth understanding before reading results

**Thermal limits.** The paper replaces the apparent-power limit
`Pf² + Qf² ≤ Smax²` with a current limit `|I|² ≤ (Smax/Vnom)²` (its eq. 19),
arguing a conductor's rating is physically a current rating. That is a
*different feasible set*, so objectives may legitimately differ wherever a
limit binds. `--thermal` applies the choice to **both** forms so the
comparison stays controlled either way. The default is `apparent` because it
is the one that reproduces the published numbers above; `--thermal current`
gives the paper's model literally.

**Angle-difference limits.** pglib carries `angmin`/`angmax` on every branch
(typically ±30°) and they bind on several cases. The paper does not mention
them. Dropping them would make ECF a *relaxation* of the polar model and every
objective comparison meaningless, so they are included in both — in polar as
`amin ≤ θf − θt ≤ amax`, in ECF in the standard bilinear rectangular form on
`V_f · conj(V_t)`. `--no-angle-limits` drops them from both.

The two `.nl` sets are **not interchangeable across conventions**, and a `.nl`
records neither. Each output directory therefore gets a `manifest.json` naming
the form, the thermal convention, the angle-limit setting and the per-case
sizes; a results file without it cannot be interpreted.

## The one asymmetry, stated

Under `--thermal current`, ECF carries `isq` as a state variable with a box,
defined by an equality — that is the paper's (31) and the form its `dsq`
treatment implies. The polar model keeps the thermal inequality inline, which
is standard practice there. So under that convention ECF's variable count is
higher by two per rated branch beyond the structural difference
(`case14_ieee`: 102 against 38, where the default `apparent` gives 62 against
38). Under the default, both forms keep the thermal limit inline. This is
deliberate — each formulation is emitted in its own idiom — but it means
variable counts are not a clean like-for-like measure. Iteration counts and
status are.

## What the suite has measured so far

**Flat start, `--thermal apparent`, all 20 cases.** Both forms solve
everything, objectives agreeing to <=3e-8. ECF costs 10-30% more iterations on
most cases but is *faster in wall clock* on the large ones, because polynomial
Jacobians and Hessians are cheaper to evaluate than trigonometric ones:

| case | polar it / s | ECF it / s |
|---|---|---|
| case1354_pegase | 41 / 1.30 | 47 / 1.03 |
| case1888_rte | 131 / 5.43 | 471 / 18.11 |
| case4661_sdet | 76 / 51.58 | 76 / 22.68 |
| case10000_goc | 132 / 34.33 | 95 / 19.54 |

`case1888_rte` is the one clear ECF loss, and it is the same case that is the
iteration outlier in `benchmarks/grid/` (778 iterations there).

**Random start, `--angle-spread 0.5`, 17 cases x 10 seeds x 2 forms.**
Convergence is 170/170 in *both* forms -- no robustness difference at this
perturbation size. What does move is the iteration count: ECF needs about half
as many, geometric mean **1.91x** over the 17 cases, and polar degrades far
faster off the flat start (case240_pserc 65 -> 352 iterations, against ECF's
84 -> 136).

So the flat-start ordering **inverts** under perturbation. Both facts are
about the same thing: a flat start already sits near the polar solution
manifold, where the trigonometry costs nothing.

**Random start, `--angle-spread 3.14159` (the full angle box), 17 cases x 5
seeds x 2 forms.** This is the perturbation size at which the formulations
separate, and they separate hard:

| | converged | at the reference optimum |
|---|---|---|
| polar | **8/85 (9%)** | 8/85 |
| ECF | **81/85 (95%)** | 77/85 (91%) |

Polar solves only the three smallest cases at all, and from `case30_ieee`
upward it is **0/5 on every single case**. Two failure modes, and they differ
by size: `Infeasible_Problem_Detected` -- a *false* infeasibility verdict on a
demonstrably feasible model -- through `case793_goc`, then
`Maximum_CpuTime_Exceeded` at 1354 and 2000 buses.

Two things to read carefully before quoting these numbers:

* **A timeout is a statement about the cap, not about convergence.** The
  sweep used `max_cpu_time=60`, generous against flat-start times of under
  2.5s for these cases, but polar's ten timeouts at 1354/2000 buses might
  converge given longer. They are not evidence of non-convergence, only of
  cost.
* **"Converged" and "at the reference optimum" are different columns on
  purpose.** AC-OPF is nonconvex with genuinely multiple local solutions, so a
  wild start reaching a *different* local optimum is correct behaviour, not a
  failure. ECF does this on `case300_ieee` (5/5 converged, 3 at the reference)
  and `case179_goc` (4/5 converged, 2 at the reference). Collapsing the two
  columns would understate ECF by four runs.

`case3_lmbd` is the one case where polar wins (3/5 against 2/5) -- the same
3-bus case that is the sole polar win in the iteration comparison above. At
n=5 that difference is noise; what it is not is a counterexample to the trend,
because the trend is monotone in size from `case5_pjm` on.

**The mild sweep does not show any of this**, which is the methodological
point worth keeping: at `--angle-spread 0.5` both forms are 170/170 and the
only visible difference is iteration count. A robustness claim about these
formulations is a claim about a *perturbation size*, and reporting one without
the other is how a null result gets mistaken for equivalence.

## How to run

```bash
# regenerate both .nl sets (downloads pglib .m on first use)
python3 benchmarks/grid_ecf/ecf_nl_export.py --form ecf
python3 benchmarks/grid_ecf/ecf_nl_export.py --form polar

# solve
make -C benchmarks grid-ecf-run      # -> benchmarks/grid_ecf/pounce.json
make -C benchmarks grid-polar-run    # -> benchmarks/grid_polar/pounce.json

# one case directly
pounce "$POUNCE_BENCH_DATA/grid_ecf/nl/pglib_case118_ieee.nl" print_level=5
```

Other flags: `--thermal current`, `--no-angle-limits`, `--no-fetch`,
`--list`, and a positional case list to restrict the set.

### Random starts

`--start random --seed N --angle-spread R` replaces the flat start with a
random *physical* operating point (per-bus voltage magnitude and angle,
per-generator real and reactive output, each uniform in its box; angles within
`±R` radians). The draw is seeded per case from `N`, so the same case and seed
give both forms the same physical start. A random-start set **requires**
`--out-dir`, so it can never overwrite the flat-start suite; its
`manifest.json` records the seed and spread.

One cell of the sweeps above is one generator invocation per form and seed:

```bash
for seed in 0 1 2 3 4; do
  for form in ecf polar; do
    python3 benchmarks/grid_ecf/ecf_nl_export.py --form $form \
      --start random --seed $seed --angle-spread 3.14159 \
      --out-dir "$POUNCE_BENCH_DATA/grid_ecf_rand/$form/s$seed"
  done
done
# then solve each directory, e.g. with max_cpu_time=60 as the sweep used
```

The driver that ran the sweeps reported above is **not** tracked, and it did
not record which 17 of the 20 default cases it used or which seeds; treat those
tables as measured but not yet reproducible from this repo alone. The
generator side is fully determined by `--seed` and `--angle-spread`.

## Contents

- `ecf_nl_export.py` — the generator (tracked)
- `data/` — downloaded pglib `.m` cases (untracked)
- `pounce.json` — latest per-problem results (untracked)
- `<bench root>/grid_ecf/nl/`, `<bench root>/grid_polar/nl/` — the `.nl`
  and their `manifest.json` (untracked)
