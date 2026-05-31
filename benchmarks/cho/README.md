# CHO Parameter Estimation Suite

Parameter estimation for a CHO (Chinese Hamster Ovary) cell kinetic model,
originally formulated as a `parmest` example in Pyomo. The objective is
the sum of squared residuals between measured and predicted species
concentrations; the constraints are the differential mass balances for the
kinetic ODEs after discretisation.

The single-problem NLP is produced by exporting the Pyomo model to AMPL
`.nl` format via `parmest_nl_export.py`, then solved through POUNCE's
AMPL NL parser. The problem exercises the NL-front end as well as a
moderately sized dense NLP with bound constraints and nonlinear equalities.

Neither POUNCE nor Ipopt currently converges to a clean optimum on this
problem (both hit numerical difficulties on the kinetic equations near
steady state), so it is a useful stress test for restoration and fallback
logic.

## Contents

- `parmest_nl_export.py` — Pyomo script that builds the CHO model and
  writes an `.nl` file to `nl_export_results/`
- `nl_export_results/cho_parmest.nl` — exported AMPL NL problem
- `cho_results.json` — latest benchmark results (when available)

## Prerequisites

1. Python environment with `pyomo` and the `parmest` tools installed:
   `pip install pyomo`
2. An NLP solver that Pyomo can see; any AMPL-compatible one works since we
   only need the `.nl` export, not a solve, e.g.:
   `conda install -c conda-forge ipopt`
3. Regenerate the `.nl` file once after cloning:
   `cd benchmarks/cho && python parmest_nl_export.py`

## How to run

From the repo root:

```bash
make cho-run
```

Or solve the exported model directly:

```bash
pounce benchmarks/cho/nl_export_results/cho_parmest.nl print_level=5
```

The `.nl` files are produced by `parmest_nl_export.py`; rerun it if the
parameter-estimation model changes.

## Output

- `cho_results.json` — per-solver results (written by the example when it
  completes)
- `nl_export_results/cho_parmest.sol` — AMPL solution file from the last
  solve attempt
- `cho_stderr.txt` — solver chatter (gitignored)

The CHO suite is loaded opportunistically by `benchmark_report.py` when a
`cho_results.json` exists. Since neither solver currently produces a clean
Optimal on this problem, the CHO numbers should be treated as a stress
test rather than as part of the headline performance comparison.
