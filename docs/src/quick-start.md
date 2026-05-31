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
