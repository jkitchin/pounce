# pyomo-pounce

Pyomo solver plugin for [POUNCE](https://github.com/jkitchin/pounce), a
pure-Rust interior-point NLP solver (a Rust port of IPOPT).

POUNCE speaks the AMPL NL/SOL protocol, so Pyomo drives it through the
AMPL Solver Library interface — exactly how Pyomo integrates with IPOPT.

## Installation

```bash
pip install pyomo-pounce
```

That single command pulls in the `pounce-solver` dependency, which
ships a per-platform wheel bundling the `pounce` executable. After
install, `pounce` is on your `PATH` and Pyomo finds it automatically.

## Usage

```python
import pyomo_pounce  # registers the solver — REQUIRED before SolverFactory('pounce')
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(initialize=0.5)
model.obj = Objective(expr=(model.x - 2)**2)

solver = SolverFactory('pounce')
result = solver.solve(model, tee=True)
print(f"x* = {value(model.x)}")  # 2.0
```

> **`import pyomo_pounce` is required.** Without it, `SolverFactory('pounce')`
> raises a clear `UnknownSolver` / "plugin not registered" error — it does not
> silently run some other `pounce`. With it imported, the plugin runs the
> `pounce` binary **bundled in the `pounce-solver` wheel**, independent of
> `PATH`. Only a source/dev install without that wheel falls back to a `pounce`
> on `PATH`, in which case the plugin warns.
>
> To see exactly which binary will run (and whether a stale or unrelated
> `pounce` earlier on `PATH` would shadow it), call:
>
> ```python
> import pyomo_pounce
> pyomo_pounce.check_binary()
> ```
>
> The check compares the git **commit** embedded in `pounce --about`, not the
> version string — two builds can share the same `X.Y.Z` while differing in
> behavior (as a binary from before/after a fix does).

## Pyomo's modern (`pyomo.contrib.solver`) interface

The same solver is registered against Pyomo's newer solver interface —
the one `ipopt_v2` uses — carrying every extra on this page:

```python
import pyomo_pounce
from pyomo.contrib.solver.common.factory import SolverFactory as SolverFactoryV2

solver = SolverFactoryV2('pounce')          # returns a v2 Results object
results = solver.solve(model)
print(results.solution_status, results.incumbent_objective)
```

`SolverFactory('pounce_v2')` gives the same engine behind the legacy API,
mirroring Pyomo's own `ipopt` / `ipopt_v2` split. `SolverFactory('pounce')`
is unchanged and remains fully supported.

Both routes return the same numbers — a test solves one model through
each and compares primals, objective, duals and reduced costs. They
differ in API (v2 returns `Results` and hands the solution back through a
loader, so `load_solutions=False` gives you values without touching the
model; options are `solver_options={...}` rather than `options={...}`)
and in per-solve overhead outside the solve, which on IDAES-shaped
collocation models is roughly 0.25 s/solve lower on v2. See
[the Pyomo docs page](https://jkitchin.github.io/pounce/pyomo.html) for
the measurements.

Pointing `ipopt_v2` at the `pounce` binary by hand also works, but
silently drops all of the above — the integer-variable guard, the
`scaling_factor` handling, the sensitivity path and the bundled-binary
resolution. Prefer one of the two registrations above.

> **Requirements for the v2 route** — `pip install pyomo-pounce[pyomo-v2]`
> asks for both:
>
> - **Pyomo ≥ 6.10.1**, which is where the `SolutionLoader` / `get_vars` API
>   this builds on landed. (`pyomo.contrib.solver.common` exists from 6.9.2,
>   but 6.9.2–6.10.0 ship the older `SolutionLoaderBase` / `get_primals`.)
> - **pounce-solver > 0.9.0**. The v2 route reads the `.sol` through Pyomo's
>   `asl_sol_reader`, which is strict where the legacy reader is lenient, so it
>   needs the per-model `Options` echo added after 0.9.0.
>
> Neither applies to `SolverFactory('pounce')`. On an older Pyomo,
> `import pyomo_pounce` still works and the legacy plugin behaves exactly as
> before; `pyomo_pounce.HAVE_V2_INTERFACE` reports whether the v2 names are
> available.

## Complementarity (`pyomo.mpec`)

`Complementarity` components are solved directly by
`SolverFactory('pounce')`, one-sided and two-sided ("mixed") pairs alike:

```python
from pyomo.mpec import Complementarity, complements
m.c1 = Complementarity(expr=complements(m.y >= 0, m.x - m.y >= 0))
m.c2 = Complementarity(expr=complements(pyo.inequality(0, m.x, 1), m.x - m.y))
```

Pyomo's `mpec.nl` writer drops a mixed pair (or writes it as an equation),
and merges a one-sided pair's bound into a different declared bound on the
same variable; either way the solve used to return `optimal` at a point that
violates the complementarity. `pyomo_pounce.mpec` writes those pairs itself
and restores the model afterwards, and the plugin counts the complementarity
rows in the written `.nl`: if a pair is still missing it raises
`ComplementarityLoweringError` rather than solve. Use the legacy
`SolverFactory('pounce')` for these models; the v2 interface's writer raises
on an active `Complementarity`.

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

## User scaling

The standard `scaling_factor` Suffix works as it does with IPOPT:

```python
model.scaling_factor = Suffix(direction=Suffix.EXPORT)
model.scaling_factor[model.obj] = 1e-3
model.scaling_factor[model.mass_balance] = 1e2

solver.solve(model, options={'nlp_scaling_method': 'user-scaling'})
```

Both halves are needed — the Suffix alone is inert (it also drives
Pyomo's own `core.scale_model`), and the option alone has nothing to
apply, which pyomo-pounce warns about. Untagged components are
unscaled; inactive constraints and fixed variables are skipped.

POUNCE models objective and constraint scaling only, so a
`scaling_factor` on a `Var` **raises** rather than being silently
dropped ([#483](https://github.com/jkitchin/pounce/issues/483)) —
rescale those variables in the model instead.

## Local development / unsupported platforms

If `pounce-solver` does not ship a wheel for your platform, the pip
install fails on the dependency. Two workarounds:

1. **Build POUNCE from source and put it on `PATH`** — the plugin
   resolves `pounce` via `shutil.which`, so any binary on `PATH`
   works:

   ```bash
   # in the pounce repo
   cargo build --release --bin pounce
   export PATH="$PWD/target/release:$PATH"
   pip install --no-deps pyomo-pounce pyomo
   ```

2. **Install `pounce-solver` from source** via maturin:

   ```bash
   make -C pounce dev     # maturin develop --release + the bundled CLI
   pip install pyomo-pounce
   ```

   `make dev` rather than a bare `maturin develop --release`: maturin builds
   the extension module and not the CLI, and this plugin shells out to the
   CLI. Without it the plugin falls through to the checkout's
   `target/release/pounce` (warning that it did) or, failing that, to
   whatever `pounce` is on `PATH` — see gh #816.

## Running the tests locally

Which binary the tests exercise depends on whether a *bundled* one is
present: the plugin prefers `pounce/bin/pounce` inside the installed
`pounce-solver` package, then the surrounding checkout's own
`target/release/pounce` or `target/debug/pounce`, and reaches `PATH` only
when there is neither (gh #816). Setup 1 above bundles nothing, so a plain
source checkout takes one of the fallback rungs while CI takes the bundled
one. Tests that describe the bundled arrangement then skip, and the suite you
run locally is not the suite CI runs.

To match CI, stage the freshly built CLI into the package *before* building
the wheel — this is what `.github/workflows/ci.yml` does:

```bash
cargo build --release --bin pounce
mkdir -p python/pounce/bin
cp target/release/pounce python/pounce/bin/pounce   # the step that bundles it
(cd python && maturin build --release --out dist)
pip install python/dist/*.whl                       # pounce module + bundled CLI
pip install --no-deps -e pyomo-pounce
pytest pyomo-pounce/tests -q
```

`networkx` and `scipy` are needed for `test_block_init.py` and
`test_repair.py`; without them those two files skip.

If a test involving duals, multipliers, or bound reduced costs fails, check
the binary before the code:

```python
import pyomo_pounce; pyomo_pounce.check_binary()
```

It reports which executable will actually run, its build *commit*, and
whether anything on `PATH` shadows it. Two builds can share a version
string while differing in commit, so a stale binary reports a plausible
`X.Y.Z` while returning pre-fix results — that is the failure mode gh #315
added `check_binary()` for, and the one gh #366 turned out to be about.

## License

EPL-2.0, same as POUNCE.
