# `kkt_scaling` — where does superlinear solve time come from?

Phase 0b of the structured-KKT decomposition work. Given a family of instances
that differ in one size parameter — the number of blocks, e.g. horizon hours of
a multi-period model — `sweep.py` solves each under several fill-reducing
orderings and fits an exponent in that parameter to every quantity that could
carry superlinear growth.

| Metric | Growth here means | Can a block-structured KKT solve fix it? |
|---|---|---|
| `factor_nnz`, `factor_flops` | **fill**: the ordering cuts the block graph in the wrong place | yes |
| `delayed_per_fac` (with `fac_per_iter` outgrowing `factor_flops`) | **pivoting**: delayed and 2×2 pivots carry work across block boundaries | yes |
| `iterations` | the IPM needs more iterations as the model grows | no |
| `other_per_iter` | non-factorization work per iteration grows superlinearly | no — a defect outside the linear solve |
| `wall` | the total the other rows decompose | — |

Factorization metrics include the restoration phase's factorizations
(`linear_solver.restoration` in the solve report), which on a cold-started
multi-period model can be a large share of the total.

## Usage

```bash
# Solve every instance under each ordering; one report JSON per solve plus
# a runs.csv row. Instances are named ..._H<size>.nl (or _T / _K).
python benchmarks/kkt_scaling/sweep.py run \
    --bin target/release/pounce --out results/ \
    --ordering auto metis auto_race --timeout 3600 \
    path/to/family_H006.nl path/to/family_H012.nl ...

# Fit exponents (log-log slopes) per metric and ordering.
python benchmarks/kkt_scaling/sweep.py fit results/            # in the size parameter
python benchmarks/kkt_scaling/sweep.py fit results/ --by n     # in the variable count
```

Each solve runs with `max_wall_time` just under `--timeout`, so a long solve
still writes its report. Time-limited solves count toward the per-iteration and
per-factorization fits but not toward `iterations` or `wall`.

## Instances

The instance families are not tracked; like every suite here, inputs are
regenerated locally. The first family is a GasLib-40 transient model (public
[GasLib](https://gaslib.zib.de/) topology, CC-BY 3.0): hourly finite-volume
time stepping over a horizon of `H` hours, initial state fixed to the steady
state, a daily sinusoidal demand, and **no** periodic terminal constraint — a
cyclic constraint ties the last time point to the first and turns the temporal
chain into a ring, which is a different block graph.

Findings are recorded in `dev-notes/kkt-scaling-phase0b.md`.
