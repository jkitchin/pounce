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

### Which *interface* runs — and why it matters for timing

Pyomo has more than one way to drive an NL/SOL solver, and they are
genuinely different code paths, not aliases. All of these reach POUNCE
(verified against Pyomo 6.10.1):

| call | works | carries `pyomo-pounce`'s extras |
|---|---|---|
| `SolverFactory('pounce')` | yes | **yes** |
| `contrib.solver` `SolverFactory('pounce')` | yes | **yes** |
| `SolverFactory('pounce_v2')` | yes | **yes** |
| `SolverFactory('ipopt_v2', executable=<pounce>)` | yes | no |
| `SolverFactory('ipopt', executable=<pounce>)` | yes | no |
| `SolverFactory('asl', executable=<pounce>, solver='pounce')` | yes | no |
| `SolverFactory('appsi_ipopt', …)` | no — takes no `executable` | — |

The first three are `pyomo-pounce`'s own registrations and the supported
routes; they are the only ones that bring the rest of this page with
them: the `scaling_factor` suffix handling, the sensitivity path, the
preflight/repair helpers, the guard against handing a model with live
integer variables to a continuous solver, and the bundled-binary
resolution above. The generic routes run the same solver and return the
same answer, but silently do without all of that.

`ipopt` and `asl` are Pyomo's *legacy* solver interface; `ipopt_v2` is
the newer `pyomo.contrib.solver` one. Driving POUNCE through `ipopt_v2`
needs a build carrying the two ASL-compatibility fixes noted in the
CHANGELOG under "Pyomo's modern solver interface could not drive POUNCE
at all" — before them it failed on every model, because Pyomo v2 passes
options as `key="value"` in a single `argv` entry (quotes and all, since
no shell is involved) and because POUNCE's `.sol` wrote an `Options`
count of `0`, which the v2 `.sol` reader rejects.

### Choosing between the legacy and v2 interfaces

Both of `pyomo-pounce`'s interfaces carry the same extras and return the
same numbers — a test in `pyomo-pounce/tests/test_v2.py` solves one model
through both and compares primals, objective, duals and reduced costs, so
this is checked on every CI run rather than asserted here.

```python
import pyomo_pounce
from pyomo.environ import SolverFactory
from pyomo.contrib.solver.common.factory import SolverFactory as SolverFactoryV2

solver = SolverFactory('pounce')       # legacy interface
solver = SolverFactoryV2('pounce')     # v2 interface (a Results object)
solver = SolverFactory('pounce_v2')    # v2 engine, legacy-style API
```

The v2 route needs **Pyomo ≥ 6.10.1** (where the `SolutionLoader` /
`get_vars` API it builds on landed — `pyomo.contrib.solver.common`
exists from 6.9.2, but 6.9.2–6.10.0 ship the older
`SolutionLoaderBase` / `get_primals`) and **pounce-solver > 0.9.0**
(Pyomo's `asl_sol_reader` is strict where the legacy reader is lenient
and needs the per-model `.sol` `Options` echo added after 0.9.0).
`pip install pyomo-pounce[pyomo-v2]` asks for both. Neither applies to
`SolverFactory('pounce')`: on an older Pyomo the legacy plugin works
exactly as before and `pyomo_pounce.HAVE_V2_INTERFACE` reports `False`.

They differ in API and in per-solve overhead. The v2 interface returns a
`Results` object and hands the solution back through a solution loader
(so `load_solutions=False` gives you the values without touching the
model); the legacy one returns a `SolverResults` and loads into the model
as a side effect. Options are `solver_options={...}` on v2 against
`options={...}` on the legacy route.

**The v2 route can be materially faster outside the solve**, and how much
depends on the model's shape. Same POUNCE binary, wall clock around
`solve()` minus POUNCE's own reported time:

| model | legacy remainder | v2 remainder |
|---|---|---|
| plain `pyomo.dae` four-tank collocation, n = 3,010 | 0.109 s | 0.104 s |
| drto/IDAES `quad_tank` N=100, n = 2,910 | 0.553 s | 0.301 s |
| drto/IDAES `cart_pole` N=100, n = 2,810 | 0.566 s | 0.295 s |

On the plain model the two are indistinguishable; on the IDAES-shaped
ones the legacy interface adds roughly 0.25 s per solve (~1.8×). If your
models are of that kind and you are solving many of them, the v2 route is
worth taking. (Figures from the [#552](https://github.com/jkitchin/pounce/issues/552)
measurements; the first row was measured on Linux, the other two on
Windows, so read down the columns rather than across the rows.)

**If you are benchmarking, put both solvers on the same interface.**
`solver.solve(model)` is not only the solve: it is Pyomo writing the
`.nl`, launching the process, reading the `.sol` back and loading it into
the model. Timing around that call and subtracting the solver's own
reported time leaves a remainder that is mostly *Pyomo's* work, and — as
the table above shows — it is not the same work on every interface. On
the 3,010-variable collocation model that remainder breaks down as
~0.082 s Pyomo writing the `.nl`, ~0.020 s process spawn plus POUNCE's
own `.nl` read and setup, and ~0.008 s Pyomo reading the `.sol` and
loading it. So comparing `SolverFactory('pounce')` against
`SolverFactory('ipopt_v2', …)` compares two Pyomo interfaces as much as
two solvers, and attributing the remainder to either solver's file
handling will mislead you. Use the same interface on both sides, or
compare the solvers' own reported times.

### How an accepted solve is reported

POUNCE reports the AMPL solve codes IPOPT's own driver reports, so a model
that swaps `ipopt` for `pounce` gets the same `SolverResults` shape. In
particular a solve that stops at the
[acceptable level](options.md#solved_to_acceptable_level-and-acceptable_progress_kappa)
— the strict tolerances were missed, `acceptable_tol` was met — is an
*accepted* solve on every route:

| interface | reported as |
|---|---|
| legacy `SolverFactory('pounce')` | `status=ok`, `termination_condition=optimal` |
| v2 | `TerminationCondition.convergenceCriteriaSatisfied`, `SolutionStatus.optimal` |
| declared-parameter (in-process) | `status=ok`, `termination_condition=optimal` |

Which convergence you got is in the solver message
(`POUNCE X.Y.Z: SolvedToAcceptableLevel` against
`POUNCE X.Y.Z: SolveSucceeded`) and, on the legacy `.sol` route, in
`results.solver.id` — the AMPL code, `1` against `0`.

Up to and including 0.10.0 the acceptable-level solve was written as AMPL
code `100`, which put it in the "solved, with a warning" band
([#591](https://github.com/jkitchin/pounce/issues/591)). The legacy route
then reported `status=warning` and Pyomo logged a load warning that IPOPT
does not; the v2 route, whose reader maps that band to
`TerminationCondition.error`, went further and raised
`NoOptimalSolutionError` under the default
`raise_exception_on_nonoptimal_result=True`. If your code special-cased
POUNCE for either, it no longer needs to.

## Complementarity constraints (`pyomo.mpec`)

`Complementarity` components need no transformation: `SolverFactory("pounce")`
writes them as complementarity rows, and POUNCE picks the formulation and
certifies the answer.

```python
from pyomo.mpec import Complementarity, complements
m.c = Complementarity(expr=complements(m.y >= 0, m.x + 2 * m.y - 1 >= 0))
pyo.SolverFactory("pounce").solve(m, tee=True)  # tee shows the certificate line
```

Each pair must be one-sided. See
[Complementarity Constraints](complementarity.md) for the formulations, the
stationarity certificate and `complementarity_strategy`.

## User scaling with the `scaling_factor` Suffix

A badly conditioned model converges poorly, and often you know its
natural units better than the solver can infer from gradients at `x0`.
The standard Pyomo channel for saying so is the `scaling_factor`
Suffix, read exactly as Ipopt reads it:

```python
model.scaling_factor = Suffix(direction=Suffix.EXPORT)
model.scaling_factor[model.obj] = 1e-3           # objective in MW, not W
model.scaling_factor[model.mass_balance] = 1e2   # one constraint
model.scaling_factor[model.energy_balance] = 1e2 # or a whole container

solver.solve(model, options={'nlp_scaling_method': 'user-scaling'})
```

Both halves are required: without `nlp_scaling_method=user-scaling` the
Suffix is inert (a `scaling_factor` Suffix also drives Pyomo's own
`core.scale_model` transformation, which never involves the solver), and
without the Suffix the option has nothing to apply — pyomo-pounce warns
in that case rather than leaving you to wonder.

Rules, matching AMPL/Ipopt:

* Only an **export-enabled** Suffix counts (`Suffix.EXPORT` or
  `Suffix.IMPORT_EXPORT`).
* Components you do not list are unscaled, as are components listed with
  a factor of `0`.
* An entry on a container applies to every member.
* Entries on **inactive** constraints/objectives and on **fixed**
  variables are skipped — none is a row or column of the problem the
  solver is handed.
* Scaling changes conditioning, never the answer: solutions, duals, and
  everything the [sensitivity](sensitivity.md) accessors report come back
  in your model's units.

**Variables can be scaled**, and a factor on a `Var` is applied as a
change of variables inside the solver: the algorithm works in the
scaled coordinates and the solution, the duals, and the bound
multipliers come back in your model's own units. No clone of the model
is made and no `propagate_solution` step is needed, which is what
distinguishes this from Pyomo's `core.scale_model` transformation.
Factors must be positive and finite. A negative factor would reverse a
variable's direction and swap its bounds, so it raises rather than
being applied.

This works on both solve paths: the ordinary ASL/subprocess solve and
the in-process path taken when the model carries [sensitivity
declarations](sensitivity.md) — including the accessors themselves.
`sens_covariance()`, `sens_information()`, `sens_jacobian()`,
`sens_solution()` and `sens_solution_report()` read the
solver's KKT factorization directly rather than through the scaling
layer, so they carry the factors through their own natural-units
translation and answer in your model's units on a variable-scaled solve
([issue #486](https://github.com/jkitchin/pounce/issues/486)).

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
`sum(w**2 * (v - v0)**2)` subject to the model's active constraints
and bounds — the full nonlinear projection, solved with POUNCE, with
the original objective restored afterwards. The weights `w` and the
row scaling that goes with them are described under [Scaling the
projection](#scaling-the-projection) below.

Both stages guarantee that a failed solve leaves variable values
exactly as they were: a diverged projection restores the
pre-projection point, and a failed block solve restores that block's
seeds, so initialization can never make your starting point worse than
it found it. A failed block does **not** stop the traversal — see
[When a block fails](#when-a-block-fails).

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

## Tuning initialization: `InitOptions`

Every stage of the pipeline takes its solver options and its policy
from one object:

```python
from pyomo_pounce import InitOptions

rep = pyomo_pounce.initialize(
    model,
    decisions=[m.feed, m.reflux],
    options=InitOptions(
        solver_options={"tol": 1e-10},  # reaches EVERY solve below
        scaling="auto",                 # projection merit + row scaling
        cond_tol=1e-8,                  # what counts as a weak block
        fallback="regularized",         # where a weak block is routed
        on_block_failure="skip-dependents",
    ),
)
```

`solver_options` reaches the projection, every block subsystem solve,
and any fallback solve. A bare dict still works and still means solver
options — `options={"tol": 1e-8}` is unchanged — and is *never*
reinterpreted as policy, because POUNCE has solver options whose names
collide with these fields.

### Scaling the projection

The projection merit is `sum(w**2 * (v - v0)**2)`. With the default
`scaling="auto"`:

* **Variables** are weighted by their own magnitude (`w = 1/|v0|`), so
  the merit measures *relative* movement. An unweighted merit measures
  absolute distance, which across mixed units is not a distance anyone
  wants: with a pressure at `1e6` beside a mole fraction at `1e-4`, it
  dumps the whole repair on the mole fraction. An anchor at (or very
  near) zero has no relative scale, so it keeps `w = 1` and stays free
  to move, and the spread between the smallest and largest weight is
  capped.
* **Rows** are normalised two-sided (`1/||grad c||_inf`) through the
  model's own `scaling_factor` Suffix. The solver's default
  gradient-based rule only ever scales a row *down*, so a row in units
  of `1e-6` keeps its magnitude and an absolute convergence test
  enforces it far more loosely than a row in units of `1e6`.

Entries the model already carries in its `scaling_factor` Suffix
**win** over the automatic ones, and the Suffix is restored exactly
afterwards (a model that declared none does not acquire one). See
[User scaling with the `scaling_factor` Suffix](#user-scaling-with-the-scaling_factor-suffix)
for the Suffix itself. `scaling="user"` uses only your entries;
`scaling="none"` restores the old unweighted merit.

Rescaling a row by a constant does not change the feasible set, and it
does not move where the projection lands — to within the solver's own
convergence tolerance, which is what sets the floor (measured: `1.2e-9`
at the default tolerance, `2.2e-16` at `tol=1e-12`).

### When a block fails

`block_initialize` walks the equality blocks in calculation order, and
that order is a DAG: `block_analyze(model).block_dependencies[k]` lists
the blocks block `k` consumes values from. When a block fails, only its
**descendants** are skipped; branches that do not depend on it are
still initialized:

```python
rep = pyomo_pounce.block_initialize(model)
rep.initialized_blocks   # solved as square systems, values kept
rep.fallback_blocks      # weak blocks routed to a fallback
rep.failed_blocks        # solve failed; seed values restored
rep.skipped_blocks       # descendants of a failure (with the reason)
```

Each entry is a `BlockOutcome` carrying the block's index, size,
leading constraint name, status, `rcond`, and `depends_on`. Pass
`on_block_failure="stop"` to go back to abandoning the traversal at the
first failure.

### Weak blocks: structurally square is not numerically solvable

Structural matching proves a block is solvable *in principle*. Whether
its Jacobian has usable rank at the current point is a separate,
numerical question, and a square-but-near-singular block handed to a
Newton step lands wherever the near-null direction takes it, silently.

Each block is therefore rank-checked before it is solved. The check is
run on the **scaled** Jacobian — rows divided by their own gradient
norm, columns multiplied by their variable's magnitude — so it answers
a numerical question and not a units one; a mixed-units block is not
weak. A block whose reciprocal condition number falls below `cond_tol`
is recorded in `rep.diagnostics` and routed to `fallback`:

* `"regularized"` (default) minimises the block's scaled squared
  residuals plus `regularization` times the scaled squared step from
  the seed, which selects the minimum-norm solution instead of an
  arbitrary point on the near-null direction. The residual it leaves is
  the ridge bias, so it falls linearly with `regularization` — which is
  why the default is small (`1e-8`).
* `"coupled"` merges the weak block with the blocks that depend
  directly on it and regularizes the union, for a deficiency that only
  downstream rows resolve. Its precision on the near-null direction is
  set by `regularization` against the solver tolerance rather than by
  the ridge bias, so a *larger* `regularization` (around `1e-6`) is the
  better choice there.
* `"off"` reports the diagnostic and solves the block squarely anyway.

`conditioning="off"` skips the check entirely.


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
