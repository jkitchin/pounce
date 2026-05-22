# pyomo-pounce

Pyomo solver plugin for [POUNCE](https://github.com/jkitchin/pounce), a
pure-Rust interior-point NLP solver (a Rust port of IPOPT).

POUNCE speaks the AMPL NL/SOL protocol, so Pyomo drives it through the
AMPL Solver Library interface — exactly how Pyomo integrates with IPOPT.

## Installation

```bash
pip install pyomo-pounce
```

This registers the solver plugin. The `pounce` binary must be available
either bundled in the wheel or on your `PATH`.

## Usage

```python
import pyomo_pounce  # registers the solver
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(initialize=0.5)
model.obj = Objective(expr=(model.x - 2)**2)

solver = SolverFactory('pounce')
result = solver.solve(model, tee=True)
print(f"x* = {value(model.x)}")  # 2.0
```

## Solver Options

Pass options the same way as IPOPT:

```python
solver = SolverFactory('pounce')
solver.options['max_iter'] = 1000
solver.options['tol'] = 1e-10
solver.options['print_level'] = 5
```

Options are forwarded to POUNCE's `OptionsList` (ipopt.opt-compatible
keys).

## Building from Source

If a pre-built wheel is not available for your platform, build the
`pounce` binary from the [pounce](https://github.com/jkitchin/pounce)
repository and put it on your `PATH`:

```bash
cargo build --release --bin pounce      # in the pounce repo
export PATH="$PWD/target/release:$PATH"
pip install pyomo-pounce
```

The plugin finds `pounce` via `shutil.which`, so any `pounce` on `PATH`
works.

## License

EPL-2.0, same as POUNCE.
