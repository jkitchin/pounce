# Initialization and Warm Starts

POUNCE is a local NLP solver: every solve starts from a point, and that
point often decides whether the solve takes 15 iterations or 150, or
whether it converges at all. This page collects the initialization
story in one place: where the starting point comes from on each
frontend, what the solver does with it (the part that surprises
people), how to warm-start each algorithm path, and how to diagnose a
bad start. The per-algorithm details live in their own pages; this is
the map.

## Where the starting point comes from

| Frontend | Primal starting point |
|---|---|
| Python `Problem.solve(x0=...)` | the `x0` argument |
| Python `minimize(fun, x0, ...)` | the `x0` argument |
| CLI / AMPL | the `.nl` file's initial-guess segment; zeros for variables without one |
| Pyomo | each `Var`'s `.value`, serialized into the `.nl` by Pyomo's writer |
| GAMS | variable levels (`x.L`) via GMO |
| Rust | `Nlp::new(problem).x0(&[...])`, or `TNLP::get_starting_point` |

Two silent-zero traps hide in that table:

* **Pyomo:** a `Var` whose `.value` was never set is written as `0`
  in the `.nl` file. A model initialized "nowhere" is actually
  initialized at the origin, which for many process models is outside
  every variable's meaningful range (and a domain error for `log`,
  `/`, and friends).
* **GAMS:** levels default to `0` unless assigned. Set `x.L` before
  the `solve` statement.

Dual estimates can be seeded too: `Problem.solve` accepts
`lagrange=`, `zl=`, `zu=` keyword arguments, and the `.nl` format
carries constraint-dual guesses when the modeling layer writes them.
Dual seeds are ignored unless you opt into a warm start (below). The
scipy-style `minimize` facade does not expose dual seeding; use
`pounce.Problem` directly when you need it.

## What the solver does with your point (cold start)

The default interior-point path ports Ipopt's iterate initializer
(`crates/pounce-algorithm/src/init/default.rs`). The sequence:

1. **The primal point is pushed into the interior of the bounds.**
   Per component, with bounds `lo <= x <= hi`:
   `p_l = min(bound_push * max(|lo|, 1), bound_frac * (hi - lo))`,
   likewise `p_u`, and `x` is clamped into `[lo + p_l, hi - p_u]`.
   One-sided bounds use the `bound_push` term alone; free variables
   are untouched. With the defaults (`bound_push = bound_frac =
   1e-2`), a variable sitting exactly on its lower bound `1.0` starts
   at `1.01` instead. Your point is honored *approximately*, and the
   deliberately-at-a-bound part of it is not honored at all. This is
   the single most common reason a "perfect" starting point does not
   behave like one.
2. **Slacks** are set to `s = d(x)` and pushed into the slack bounds
   the same way.
3. **Duals** get fixed defaults: constraint multipliers `y = 0` (or a
   least-square estimate, see `bound_mult_init_method` below) and
   bound multipliers `z = v = bound_mult_init_val = 1.0`.
4. **The barrier parameter** starts at `mu_init = 0.1` (monotone
   `mu_strategy`, the default) regardless of how good your point is.

The knobs, all Ipopt-compatible:

| Option | Default | Meaning |
|---|---|---|
| `bound_push` | `1e-2` | Absolute push off each bound (relative to `max(|bound|, 1)`). |
| `bound_frac` | `1e-2` | Cap on the push as a fraction of the bound interval. |
| `slack_bound_push` / `slack_bound_frac` | `1e-2` | Same, for inequality slacks. |
| `bound_mult_init_val` | `1.0` | Initial bound-multiplier value. |
| `bound_mult_init_method` | `constant` | `constant` / `mu-based` / `least-square`. |
| `constr_mult_init_max` | `1e3` | Cap on the least-square constraint-multiplier estimate; `0` keeps `y = 0`. |
| `least_square_init_primal` | `no` | Replace the starting `x` with the min-norm solution of the linearized constraints before the interior push. |
| `mu_init` | `0.1` | Initial barrier parameter (monotone strategy). |
| `start_with_resto` | `no` | Jump straight into feasibility restoration at iteration 1 (aborts if the start is already feasible). |

An *infeasible* starting point is fine: the IPM does not require
feasibility, and `least_square_init_primal=yes` can cheaply reduce
iteration-0 infeasibility on mostly-linear models (the
`mehrotra_algorithm` LP/QP cascade turns it on for you, along with
more aggressive `bound_push` / `bound_frac` / `bound_mult_init_val`).
A point where a function *fails to evaluate* is not fine; see
[Diagnosing a bad start](#diagnosing-a-bad-start).

## Warm-starting the interior-point path

From Python, the packaged form is one object:

```python
x, info = prob.solve(x0=x0)                  # cold solve
ws = pounce.WarmStart.from_info(x, info)     # captures x, duals, mu
x2, info2 = prob.solve(warm_start=ws)        # warm re-solve
ws.save("state.npz")                         # reuse across processes
```

`warm_start=` is accepted by `Problem.solve` and `pounce.minimize`,
seeds the primal and dual iterates, applies the enabling options
below, and forwards the SQP working set when the state was captured
from that path. The rest of this section is what it does under the
hood (and the only route from the CLI or an options file).

Passing a previous solution as `x0` is **not** a warm start by
itself. The IPM warm start is a package of three things, and skipping
any one of them silently degrades to (roughly) a cold solve:

1. **Opt in and seed the duals.** Set `warm_start_init_point=yes` and
   pass the previous multipliers.
2. **Lower `mu_init`.** The default `0.1` makes the solver walk the
   barrier schedule down from scratch even when started at the
   optimum. Seed it near the converged complementarity (e.g. `1e-7`
   after a `tol=1e-8` solve).
3. **Tighten the warm-start pushes.** The warm initializer applies
   its own interior clamp with `warm_start_bound_push` / `_frac`
   (default `1e-3`), which shoves an at-the-bound solution back off
   its bounds. Tighten them to keep the point.

```python
x, info = make_problem().solve(x0=x0_cold)      # cold solve

warm = make_problem()
warm.add_option("warm_start_init_point", "yes")
warm.add_option("mu_init", 1e-7)
for k in ("warm_start_bound_push", "warm_start_bound_frac",
          "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
          "warm_start_mult_bound_push"):
    warm.add_option(k, 1e-9)

x2, info2 = warm.solve(
    x0=x,
    lagrange=np.asarray(info["mult_g"]),
    zl=np.asarray(info["mult_x_L"]),
    zu=np.asarray(info["mult_x_U"]),
)
```

On HS071 this takes the re-solve from 11 iterations to 5, while
`warm_start_init_point=yes` alone saves nothing; the full runnable
comparison is `python/examples/hs071_warm_start.py`. On the CLI the
same options apply as `KEY=VALUE` pairs, with dual seeds coming from
the `.nl` file's dual segment when present.

| Option | Default | Meaning |
|---|---|---|
| `warm_start_init_point` | `no` | Master switch: honor supplied primal *and* dual seeds. |
| `warm_start_bound_push` / `warm_start_bound_frac` | `1e-3` | Interior clamp used instead of `bound_push` / `bound_frac`. |
| `warm_start_slack_bound_push` / `warm_start_slack_bound_frac` | `1e-3` | Same, for slacks. |
| `warm_start_mult_bound_push` | `1e-3` | Floor on seeded bound multipliers (a carried-in `z = 0` must not start on the barrier's boundary). |
| `warm_start_mult_init_max` | `1e6` | Cap on seeded equality multipliers. |

Even a well-executed IPM warm start has a structural limit: the
barrier pushes iterates off the bounds, so the active-set information
in a converged solution cannot be fully exploited. When you are
solving a *sequence* of related NLPs (MPC steps, branch-and-bound
nodes, homotopy paths), that limit is the reason the active-set SQP
path exists.

## Warm-starting the active-set SQP path

With `algorithm=active-set-sqp`, the warm-start payload is different:
alongside the primal/dual seeds it carries the **working set** (which
bounds and constraints are active), and an unchanged working set means
the next solve converges in a handful of QP iterations.

```python
prob.add_option("algorithm", "active-set-sqp")

ws = None
for k in range(horizon_steps):
    x, info = prob.solve(x0=x_prev, working_set=ws)
    ws = info["working_set"]
    x_prev = x
```

The two paths' warm-start inputs are deliberately path-local: the
IPM-side options above (`warm_start_init_point`, `mu_init`,
`bound_push`, ...) are silently ignored on the SQP path, and
`working_set=` is ignored on the IPM path. Details, the
`classify_working_set` helper for reconstructing a working set from
multipliers, and the GAMS `sqp_state_file` / marginal-based routes are
in [Active-Set SQP & Warm Starts](active-set-sqp.md). Note the GAMS
warm-start features currently live in the native C link only, not the
pip link (see [GAMS](gams.md)).

## Sequences of solves: batch chaining and sessions

For MPC chains, parametric sweeps, and B&B node relaxations from
Python, `solve_nlp_batch` packages the whole IPM warm-start recipe
for you:

```python
results = pounce.solve_nlp_batch(batch_t)                   # cold
results = pounce.solve_nlp_batch(batch_t1, warms=results)   # warm
```

Each instance is seeded with the previous primal and duals, the
converged `mu` is threaded into `mu_init`, and
`warm_start_init_point=yes` is forced; see
[Batched NLP solving](python.md#batched-nlp-solving-solve_nlp_batch).
For post-solve sensitivity queries against the converged KKT factor
(a different kind of reuse, no re-solve at all), see
[Sessions](sessions.md). JAX users get warm-start hand-off along a
parameter trajectory via `JaxProblem`; see
[the Python guide](python.md).

## Diagnosing a bad start

The first stop is the preflight check, which evaluates the model once
at its starting point (no solve) and reports everything this page has
warned about: NaN/inf evaluations, bound violations, how far the
interior clamp will move the point, initial constraint violation, and
derivative scale spread.

```sh
pounce check-x0 model.nl              # text report; --json for tools
pounce check-x0 model.nl --x0-file candidate.txt
```

```python
report = pounce.preflight(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
print(report)          # report.fatal, report.warnings, report.to_dict()
```

Exit code 0 means the model evaluates cleanly at x0 (warnings allowed);
21 means a solve from this point would abort. The other diagnostics:

* **`Invalid_Number_Detected`** means an evaluator returned NaN/inf,
  and the very first evaluation at the starting point is the usual
  culprit (`log(0)` or a division at an all-zeros default start).
  The interior clamp only repairs bound violations; it cannot fix
  domain errors on free variables. Move the start into the domain,
  or add bounds that keep the clamp inside it.
* **`derivative_test=first-order`** runs the derivative checker at
  the starting point; wrong derivatives look exactly like a bad
  start (immediate restoration, tiny steps).
* **The [interactive debugger](debugger.md)** (`--debug`) breaks at
  iteration 0, so you can inspect the initial objective, `inf_pr`,
  and `inf_du` before a single step is taken, and `resolve` from an
  edited iterate.
* **Presolve** (`presolve=yes`) reports structural trouble that no
  starting point can fix, like rank-deficient equality blocks
  (LICQ check), and its bound tightening shrinks the box the
  interior clamp places you in. See
  [Troubleshooting Recipes](troubleshooting.md) and [FBBT](fbbt.md).
* **`pounce-studio analyze-nl`** gives a structural pre-flight of a
  model file without solving.

## No good starting point at all?

Three composable primitives cover the "generate or repair a point"
workflows from Python:

```python
# N diverse starts (the sampler behind find_minima): sobol / uniform /
# jitter / bounds midpoint. Feed them to solve_nlp_batch or race them.
starts = pounce.generate_starts(16, bounds=bounds, seed=0)

# Min-norm repair of a candidate onto the linearized constraints +
# bounds (the standalone form of least_square_init_primal).
x0 = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)

# Cheap tournament: a few iterations from each start, ranked; continue
# the winner at full effort with a WarmStart.
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10)[0]
res = pounce.minimize(fun, best.x,
                      warm_start=pounce.WarmStart.from_info(best.x, best.info))
```

When the model has many local minima and you want *all* of them (or a
managed search rather than a tournament), the
[global search drivers](find-minima.md) (`multistart`, `mlsl`,
`deflation`, `flooding`, `tunneling`, `basinhopping`) manage
populations of starting points and warm-start bookkeeping for you,
from Python (`pounce.find_minima`) or the CLI (`--minima`).
