# Quick Start

This page assumes POUNCE is built and on your `PATH`
(see [Installation](installation.md)).

## Solve an AMPL `.nl` file

```sh
pounce problem.nl
```

This solves the problem and writes a sibling `problem.sol` next to the
input, following the AMPL solver convention. The console output
mirrors upstream `ipopt`'s banner, per-iteration table, and final
summary.

Append `KEY=VALUE` pairs to override options — the syntax and
semantics match the upstream Ipopt CLI:

```sh
pounce problem.nl print_level=8 max_iter=500 tol=1e-10
```

See [Solver Options](options.md) for details.

## Solve a QPS / MPS file

LPs and QPs in (free- or fixed-column) MPS form, including the QPS
extension for a quadratic objective (`QUADOBJ` / `QSECTION` / `QMATRIX`),
are read directly — no conversion to `.nl`:

```sh
pounce model.mps
pounce QSCFXM1.QPS
```

The extension (`.mps` or `.qps`, any case, optionally followed by `.gz`)
selects the reader. The model is classified and routed exactly like a `.nl`
input, so an LP or convex QP reaches the convex interior-point engine, and a
sibling `model.sol` is written. `OBJSENSE MAX` is honored; integer markers
and `BV`/`LI`/`UI` bounds are rejected, since POUNCE is a continuous solver.

The file is read as free MPS first, and by fixed column positions if that
fails, so names containing spaces work. Set names in `RHS`, `RANGES` and
`BOUNDS` are optional. Where the format leaves a choice, the reader follows
HiGHS — a repeated coefficient, `RHS` or `RANGES` value keeps the first,
repeated `QUADOBJ` entries are summed — and a negative `UP` bound on a
column with no lower bound makes that lower bound −∞, as CPLEX and Gurobi
do. Each such choice is printed as a warning on stderr.

## Try a built-in problem

POUNCE ships several self-contained test problems that exercise the
full pipeline without parsing a `.nl` file (run `pounce --list-problems`
for the full set):

```sh
pounce --list-problems
pounce --problem rosenbrock
pounce --problem quadratic
```

## From Python

```python
import numpy as np
from pounce import minimize

res = minimize(lambda x: ((x - 1) ** 2).sum(), x0=np.zeros(3))
print(res.fun, res.x)
```

See the [Python API](python.md) chapter for the full
cyipopt-compatible interface.

## From Pyomo

```python
import pyomo_pounce  # registers 'pounce'
from pyomo.environ import SolverFactory

SolverFactory('pounce').solve(model)
```

See the [Pyomo](pyomo.md) chapter for details.

## Full help

```sh
pounce --help
```
