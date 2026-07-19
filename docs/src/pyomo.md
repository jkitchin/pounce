# Pyomo

Because POUNCE speaks the AMPL NL/SOL protocol, it drops into
[Pyomo](https://www.pyomo.org/) through the AMPL Solver Library
interface — exactly how Pyomo drives Ipopt.

The [`pyomo-pounce`](https://github.com/jkitchin/pounce/tree/main/pyomo-pounce)
package registers `pounce` as a Pyomo `SolverFactory` solver:

```python
import pyomo_pounce  # registers 'pounce'
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory

model = ConcreteModel()
model.x = Var(bounds=(-10, 10))
model.obj = Objective(expr=(model.x - 3) ** 2)

solver = SolverFactory('pounce')
solver.solve(model)
```

Options pass through the usual Pyomo mechanism:

```python
solver.solve(model, options={'tol': 1e-10, 'max_iter': 500})
```

Under the hood, Pyomo writes the model to an AMPL `.nl` file, invokes
`pounce problem.nl -AMPL`, and reads the result back from the `.sol`
file. See [Running Solves](cli.md) for the `-AMPL` solver mode.

## Preflight and initialization

A `Var` whose `.value` was never set is written as **0** into the
`.nl` file, so an uninitialized model actually starts at the origin
(see [Initialization and Warm Starts](initialization.md)). The package
ships a preflight check plus an initialization pipeline for exactly
this:

```python
import pyomo_pounce

report = pyomo_pounce.preflight(model)   # what will POUNCE see at x0?
print(report)                            # unset vars, bound/constraint
if report.fatal:                         # violations, NaN/inf evaluations
    ...

# fill -> repair -> block-solve, with the decisions held constant:
rep = pyomo_pounce.initialize(model, decisions=[m.feed, m.reflux])
if not rep.block.square:
    print(rep)          # names of what you forgot to specify
```

`preflight` evaluates every active constraint and the objective at the
current values with unset values treated as 0 (exactly what the NL
writer sends), restores the model untouched, and reports what
iteration 0 will see; `report.fatal` means the solve would abort with
`Invalid_Number_Detected`.

`initialize` follows the workflow you would run by hand on, say, a
distillation column: set the decisions (feed, reflux, boilup), solve
for a physical profile with them held constant, then let the optimizer
move them. Its three stages are also available individually:

```python
pyomo_pounce.initialize_missing_values(model)   # bounds-aware fill
                                                # (midpoint / one unit
                                                # inside / zero)

pyomo_pounce.project_to_feasible(model)         # min-norm repair: move the
                                                # current point onto the
                                                # model's own constraints
                                                # (one POUNCE solve)

rep = pyomo_pounce.block_initialize(            # solve the equality
    model, decisions=[m.feed, m.reflux])        # system's square blocks
                                                # in calculation order
```

`initialize_missing_values` fills each variable independently, so the
fill can be internally inconsistent (mole fractions that do not sum to
one); `project_to_feasible` repairs that by minimizing
`sum((v - v0)**2)` subject to the model's active constraints and
bounds — the full nonlinear projection, solved with POUNCE, with the
original objective restored afterwards.

`block_initialize` is IDAES-flavored initialization without
hand-written routines. `decisions=` holds the listed variables at
their current values for the solve and releases them afterwards (each
must have a value). The active equality constraints are decomposed
(Dulmage-Mendelsohn, via `pyomo.contrib.incidence_analysis`); the
square part is solved block by block in topological order by Pyomo's
`solve_strongly_connected_components` (1x1 blocks by Newton, larger
blocks by POUNCE), filling `Var.value` along the way. When the system
is **not** square, `report.square` is False and the offending
variables and constraints are reported **by name** —
`underconstrained_variables` is the list of things you forgot to
specify or flag as decisions, `overconstrained_constraints` the
redundant or conflicting specifications. Permanently-known inputs can
simply be `fix()`ed instead of listed as decisions.

The analysis half is also available on its own:

```python
rep = pyomo_pounce.block_analyze(               # the DM partition only:
    model, decisions=[m.feed, m.reflux])        # nothing seeded or solved
rep.underconstrained_variables                  # VarData objects, uncapped
rep.n_extra_degrees_of_freedom                  # how many specs are missing
rep.variable_blocks                             # the calculation order
```

`block_analyze` runs the same decision handling and the same
Dulmage-Mendelsohn decomposition, but touches nothing: no values are
read or written (so, unlike `block_initialize`, the decisions do not
need values), and no solve happens. Where the initialization reports
cap their name lists for display, `block_analyze` returns the **full**
partition as the component objects themselves: the underconstrained
and overconstrained subsystems, the square part, and its
block-triangular calculation order. Use it to diagnose a large model's
specification, or as the structural front end for tooling that decides
*what* to specify before calling `initialize` /
`block_initialize` to do the work.
