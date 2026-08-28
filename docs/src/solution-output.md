# Solution Output

## The `.sol` file

Following the AMPL solver convention, solving a positional `.nl` file
writes a sibling `<stub>.sol` next to it — `pounce problem.nl`
produces `problem.sol`. The file carries the primal `x` and dual
`lambda` blocks plus an `objno` line with the AMPL `solve_result_num`,
so AMPL (or any `.sol` reader) can pull the solution back:

```sh
pounce problem.nl                       # writes problem.sol
pounce problem.nl --sol-output out.sol  # write to an explicit path
pounce problem.nl --no-sol              # skip the .sol write
```

A `.sol` is written even when the solve fails, so the
`solve_result_num` is always recoverable. Built-in problems
(`--problem …`) have no `.nl` stub, so they only produce a `.sol`
when `--sol-output` is given explicitly.

## Reading `solve_result_num`

The `objno` line carries an AMPL `solve_result_num` (Gay 2005, *Hooking Your
Solver to AMPL* §5). Consumers key on the **band**, not the exact number:

| Band | Meaning |
|---|---|
| `0`–`99` | solved |
| `100`–`199` | solved, with a warning |
| `200`–`299` | infeasible |
| `300`–`399` | unbounded |
| `400`–`499` | limit reached (iterations, time) |
| `500`–`599` | failure |

Pyomo maps each band to a `TerminationCondition`, so anything in `200`–`299`
arrives as `TerminationCondition.infeasible`.

### Solved: strict, acceptable, and square

POUNCE writes the same codes IPOPT's AMPL driver writes, so a model can be
moved between the two solvers without a reader change:

| Code | Verdict | What it means |
|---|---|---|
| `0` | `Solve_Succeeded` | The convergence criteria (`tol` and friends) were met. |
| `1` | `Solved_To_Acceptable_Level` | The [acceptable-level fallback](options.md#solved_to_acceptable_level-and-acceptable_progress_kappa): the strict tolerances were not met, but `acceptable_tol` was, for `acceptable_iter` consecutive iterations. |
| `2` | `Feasible_Point_Found` | A **square** problem — as many equality rows as variables — recovered to feasibility. |

All three are accepted solves and all three sit in the `0`–`99` band. That matters
beyond tidiness: Pyomo's legacy `.sol` reader loads the `0`–`99` band as
`SolverStatus.ok` and the `100`–`199` band as `SolverStatus.warning`, with
`TerminationCondition.optimal` either way. POUNCE reported acceptable-level
solves as `100` up to and including 0.10.0, so Pyomo logged

```text
WARNING: Loading a SolverResults object with a warning status into model...
    - termination condition: optimal
    - message from solver: POUNCE 0.10.0: SolvedToAcceptableLevel
```

on a solve IPOPT loads clean
([#591](https://github.com/jkitchin/pounce/issues/591)). The distinction
between strict and acceptable convergence is still there — in the code itself
(`1`, not `0`), in the `.sol` message line, and in the JSON report's `status`
field — it simply no longer reads as a warning.

`Feasible_Point_Found` was `100` up to and including 0.10.0 for the same
reason and was fixed the same way
([#815](https://github.com/jkitchin/pounce/issues/815)). It is worth being
precise about why a feasible point counts as *solved* here, because in general
it would not: POUNCE emits this status only when the problem is square, which
is the condition IPOPT uses for its own `2`. On a square problem there is
nothing to optimise — the objective is constant over a feasible set the
equalities have already pinned — so a point that satisfies the constraints is
the solution, and there is no further criterion it could be said to have
missed. For a non-square problem the status is never produced.

The stakes were higher than a logged warning. Pyomo's newer `.sol` reader
(`pyomo.contrib.solver`) maps the `100`–`199` band to
`TerminationCondition.error`, not to `optimal`-with-a-warning, so on that route
a square flowsheet that POUNCE had solved to a constraint violation of
2.2e-06 was delivered to the caller as a solver error.

### Infeasible: proved vs. local

Within the infeasible band POUNCE distinguishes *how* it knows:

| Code | Verdict | What it means |
|---|---|---|
| `200` | `InfeasibleProblemDetected` | The solver converged to a point of **local** infeasibility — a stationary point of the constraint violation with the violation bounded away from zero. |
| `201` | `... (detected by presolve: …)` | Presolve's bound propagation / interval arithmetic found the feasible region empty before any iteration. |

The difference is real, not cosmetic. `201` is a *structural* detection made on
the model's bounds before iterating, not a certified proof — it is subject to
the same floating-point limits as any interval computation, and is withheld
whenever the violation is smaller than the feasibility tolerance. `200` is
different in kind — on a nonconvex problem a positive local minimum of the
violation does **not** rule out a feasible point elsewhere, which is why the
console message says "Problem may be infeasible."

Because `200` is an inference rather than a proof, it is withdrawn when POUNCE
holds a point that contradicts it. Before any numerical path reports `200`, the
model's own starting point is evaluated against every constraint; if it
satisfies them all, the feasible set is demonstrably non-empty and the verdict
becomes `Error_In_Step_Computation` (`500`) — an honest "the solve broke down"
rather than a wrong answer. This can only ever *withdraw* a verdict: a model
with no feasible point cannot produce such a point, so a correct `200` is
unaffected. Supplying a feasible starting point is therefore worth doing on a
model you believe is feasible but POUNCE reports otherwise.

When the region is found empty the solve is skipped entirely and the message
names how it was found, so the claim is checkable:

```text
POUNCE 0.9.0: InfeasibleProblemDetected (detected by presolve: bound propagation)
objno 0 201
```

`201` requires [presolve](auxiliary-presolve.md) to be enabled (`presolve=yes`);
it is off by default. A presolve-derived infeasibility is only reported when the
contradiction holds on the *original* box — one produced by presolve's own
auxiliary elimination is re-checked after rollback and never certified.

### One more route to `200`: over-determined systems

An **over-determined** model — more equality rows than free variables, such as
`x == 0.2` with `x == 0.8` — cannot be solved at all: it fails a structural gate
before the first iteration. That used to be reported as
`Not_Enough_Degrees_Of_Freedom` (`504`, the failure band), which says "cannot
attempt this" for a model whose answer is already decided.

POUNCE now checks such a model for a bound-propagation contradiction on that
failure path and reports `200` when it finds one. This does not need
`presolve=yes` — nothing is transformed and no solve runs through the check — so
it is the one way to reach the infeasible band with the default options and no
iterations. A *consistent* over-determined system is unaffected and still
reports `504`.

Because the solve provably cannot run here, this route measures constraint
residuals against each row's declared magnitude rather than an absolute
tolerance, so the verdict does not change when every row is multiplied by a
constant. Elsewhere — wherever a solve *can* run — an infeasibility smaller than
the feasibility tolerance is still withheld, as described above.

## Choosing an output format

| You want… | Use |
|---|---|
| AMPL / Pyomo to read the result back | the `.sol` file (default) |
| A structured, schema-versioned report for tooling | `--json-output` (see [JSON Solve Report](json-output.md)) |
| Just the console summary | `--no-sol` |

The `.sol` and JSON outputs are not exclusive — you can request both
in the same run.
