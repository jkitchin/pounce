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

### Which `pounce` binary runs

`import pyomo_pounce` is **required** before `SolverFactory('pounce')`.
Without it Pyomo does not know the solver and raises a clear
`UnknownSolver` / "plugin not registered" error — it does not silently run
some other `pounce`. With it imported, the plugin runs the binary **bundled
in the `pounce-solver` wheel**, independent of `PATH`; only a source/dev
install lacking that wheel falls back to a `pounce` on `PATH` (and the plugin
warns when it does).

Because two builds can report the same version string (`X.Y.Z`) while
behaving differently — a binary from before and after a fix does — a stale
`pounce` on `PATH` is otherwise hard to notice. To see exactly which
executable will run, its build (the git commit from `pounce --about`), and
whether a different `pounce` earlier on `PATH` would shadow it:

```python
import pyomo_pounce
pyomo_pounce.check_binary()   # prints a report; returns a dict
```

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

Both stages guarantee that a failed solve leaves variable values
exactly as they were: a diverged projection restores the
pre-projection point, and a failed block solve restores that block's
seeds and stops, so initialization can never make your starting point
worse than it found it.

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

## Repairing a bad specification

Some specifications are structurally wrong, not just badly started. On
a distillation column at steady state, holding **all** the flow
controls leaves the drum levels undetermined while the holdup balances
become redundant — square by count, singular in structure, and no
starting point fixes that. `block_repair_plan` plans a valid
specification instead of failing on the broken one:

```python
plan = pyomo_pounce.block_repair_plan(
    model,
    decision_candidates=[m.LT, m.VB, m.D, m.B])  # what you would like held
plan.decisions   # candidates a square system can hold
plan.pruned      # candidates the equalities claim: solved for instead
plan.pinned      # what nothing determines: hold at values you choose
```

The candidates are pruned to the subset a valid specification can
hold: matching prefers plain variables over candidates, which provably
minimizes the number pruned, and among candidates **earlier-listed
ones are preferentially kept**, so the listing order acts as an
implicit priority when a pruning tie could go either way. The pins
need **no user input**: a
variable is pinned when every one of its edges is provably unusable —
the key case being an equation `0 == f/g`, which cannot determine a
variable appearing only in the denominator `g`, since its sensitivity
there vanishes at every solution. That is exactly the shape
substituting `d/dt = 0` into a dynamic balance produces, which is how
loose integrators (drum levels with no weir feedback) hide in
steady-state models. Like `block_analyze` it is a plan, not an action:
nothing is fixed, read, or written, and no values are needed.
`loose_variables` (undetermined, not repairable) and
`redundant_constraints` (satisfiable by no specification) are genuine
model defects.

`initialize` and `block_initialize` run the same check on their
`decisions` **automatically** (`repair="auto"`, the default). A square
specification is used exactly as given, the shipped behavior. A broken
one is repaired: the decisions become the candidate pool, conflicting
ones are pruned (they need no values), pins are seeded bounds-aware
and never at zero when valueless (a pin lives in denominators, so zero
is the one forbidden seed), and `report.repair` records the plan (None
when nothing was needed). Pass `repair="off"` for the strict path:
decisions held exactly as given, and a non-square specification is
reported (`report.square`, the name lists) instead of repaired. The repair is call-scoped exactly like the
decisions themselves (fixed flags restored, values only), so it never
changes your model's own specification. To *apply* a plan to a model
you intend to solve — a square simulation, say — fix `plan.decisions`
and `plan.pinned` and leave `plan.pruned` free; which variables to fix
is a modeling decision, so the plan leaves it to you.
