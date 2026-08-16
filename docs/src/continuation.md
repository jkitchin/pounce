# Continuation over a parametric NLP sequence

When you solve the *same* NLP many times with a moving parameter — an MPC
controller stepping its horizon, a flowsheet swept over an operating
variable, an uncertainty map traced through a design space — the
orchestration is always the same: carry the previous iterate forward,
transfer it onto the new problem, seed the solve, notice when the active
set moves, and count what it all cost.

`pounce.Continuation` is that orchestration, for the generic `Problem`
API. `pyomo_pounce.continuation` is the same thing over a Pyomo model.

This page also reports what the capability is *worth*, measured, because
the answer is not the one the literature on active-set warm starting
would lead you to expect. Read [the measurement](#what-it-is-worth)
before you reach for the tangent predictor.

## The short version

```python
import numpy as np
import pounce

def update(theta):
    """Install theta and hand back the Problem. This is the only
    frontend-specific part."""
    rhs = np.array([0.0, 0.0, theta[0], theta[1]])
    p = pounce.Problem(n=5, m=4, problem_obj=model,
                       lb=lb, ub=ub, cl=rhs, cu=rhs)
    p.add_option("print_level", 0)
    return p

driver = pounce.Continuation(update, pins=[2, 3])
trace = driver.run([np.array([5.0 - 0.5 * t, 1.0 + 0.2 * t])
                    for t in np.linspace(0, 1, 8)], x0=x0)

print(trace.report())
```

```text
continuation: ok
  points            8
  corrections       8
  predictor accepts 0
  step rejections   0
  active-set events 1
  worst predictor residual 4.309e-02
  solver iterations 28
  total evaluations 232
  solve time        6.1 ms
```

## The two modes, and which one you want

`run(thetas)` traces a **prescribed** sequence: you have a list of
parameter values and you want each one solved. Every point is corrected,
because every point is an answer you asked for.

`follow(theta_of_s, s_span)` traces a **path**: the intermediate points
are a means, not an end. Here the driver picks its own step size, and a
predicted point whose KKT residual is under `monitor_tol` is accepted
**with no solve at all**. That is where continuation pays.

The distinction matters more than it looks, because it decides whether
the predictor has anything to do. See below.

## Pieces

### The parameter must enter through pin rows for a tangent predictor

`pins=` are the 0-based indices of the equality rows `g_i(x) = θ_i`
(so `cl[i] == cu[i] == θ_i`) — the sIPOPT convention that
[`Problem.solve_with_sens`](sensitivity.md) and
`Solver.parametric_step` already take. Given them, the predictor is a
back-solve against the previous solve's held KKT factor: no extra
factorization, no extra function evaluation.

Omit `pins` and the driver falls back to a **zero-order warm transfer** —
the previous iterate carried over unchanged. That is a supported mode,
not a degraded one; `Continuation.has_tangent` tells you which you got.

### Transfer maps, for when the problem changes shape

A horizon shift or a remesh changes *which* variables exist. Supply
`transfer=`, a mapper with the same protocol as `WarmStart.transfer`:

```python
def shift(ctx):
    x = ctx.source.x.reshape(N + 1, -1)
    return {"x": np.vstack([x[1:], x[-1]]).ravel()}

driver = pounce.Continuation(update, pins=[0, 1], transfer=shift)
```

The mapper returns a dict of replacement arrays — any of `x`,
`lagrange`, `zl`, `zu`, `working_set`, `mu` — and anything it leaves out
is carried over unchanged. Every array is length-checked against the
target problem before the solve, so an off-by-one fails at the mapper
rather than three function evaluations into the solve.

### The monitor is what lets `follow` skip a solve

`follow` cannot accept a predicted point without a way to score it, so
supply `monitor=`. `pounce.kkt_residual_monitor(problem_obj, bounds)`
builds one from the same cyipopt-shaped callbacks the `Problem` already
has:

```python
monitor = pounce.kkt_residual_monitor(model, bounds_at)
driver = pounce.Continuation(update, pins=[2, 3], monitor=monitor,
                             bounds=bounds_at, monitor_tol=3e-2)
```

Without a monitor `follow` corrects every step — the safe degradation,
not an error.

### What the trace reports

`ContinuationTrace` carries every counter, per step and aggregated:
predictor residual, corrections, step rejections, active-set events,
solver iterations, and total callback evaluations (pass `counter=` an
object with `reset_counts()` / `counts()` to fill the last one).

## What it is worth

> **Summary: on an interior-point method the tangent predictor does not
> make a warm-started solve meaningfully cheaper.** Measured over
> pounce's own warm-start corpus it is within ±3% of a plain
> previous-solution warm start. Continuation pays here by *skipping
> solves* in `follow`, not by accelerating them in `run`.

### The four-way benchmark

`benchmarks/warmstart` carries the four arms as `cold-ipm`, `warm-ipm`,
`pred-ipm` (tangent primal seed) and `predcorr-ipm` (tangent primal *and*
dual seed, re-anchored on an active-set event):

```
python -m warmstart.run --families mpc_horizon_40 \
    --arms cold-ipm,warm-ipm,pred-ipm,predcorr-ipm
```

Summed over the `mpc_horizon_{10,20,40,80}` families at all three step
scales — 12 cells, 240 solves per arm, warm-start-eligible steps only:

| arm | iterations | vs. `warm-ipm` |
|---|---|---|
| `cold-ipm` | 2360 | +115% |
| `warm-ipm` | 1097 | — |
| `pred-ipm` | 1125 | **+2.6%** |
| `predcorr-ipm` | 1083 | **−1.3%** |

Warm starting is worth 2.15×. The tangent predictor on top of it is
noise, and at the largest step scale `pred-ipm` is reliably *worse*
(+18% at horizon 80) because it extrapolates across critical-region
boundaries the corrector then has to undo.

### Why — the one-iteration floor

The mechanism is visible per step. In the continuation regime (the
suite's `tiny` scale), a warm-started IPM solve converges in **one
iteration**:

```
warm  k= 2 iters=  1     warm  k=11 iters=  1
warm  k= 3 iters=  1     warm  k=12 iters=  1
warm  k= 4 iters=  1     ...
```

There is no headroom. A better seed cannot take a solve below one
iteration, so the predictor has nothing to remove. This is the
interior-point analogue of the well-known result that IPMs warm-start
weakly: the barrier restart, not the seed's distance from the solution,
is what the iterations are spent on.

At larger steps there is headroom, but there the active set moves and
the first-order predictor is extrapolating through a kink — the error is
`O(Δθ²)` at best and unbounded across a region boundary.

### Where continuation *does* pay: skipping the solve

`follow` can accept a point outright. On the upstream sIPOPT
`ParametricTNLP` fixture, tracing the same path at different
`monitor_tol`:

| `monitor_tol` | points | solves | accepts | iterations | endpoint error |
|---|---|---|---|---|---|
| `1e-6` | 8 | 8 | 0 | 28 | 2.8e-15 |
| `1e-2` | 8 | 7 | 1 | 26 | 2.8e-15 |
| `3e-2` | 8 | 4 | 4 | 18 | 6.2e-05 |
| `1e-1` | 8 | 2 | 6 | 12 | 8.3e-04 |
| `1e+0` | 8 | 1 | 7 | 9  | 6.8e-03 |

Half the solves for 6e-5 of endpoint error is a real trade, and it is the
one continuation is for. But notice what the currency is: **accuracy, not
free speed**. If you need every point at solver tolerance, use `run` and
expect warm-start performance.

### Which to use

* Every point must be solved to tolerance → `run(...)`, with or without
  `pins`. The predictor is not the reason to use this; the orchestration
  and the counters are.
* Intermediate points are a means → `follow(...)` with a `monitor` and a
  `monitor_tol` you have chosen deliberately, knowing it sets the
  accuracy of the accepted points.
* Your problem is an active-set SQP (`algorithm=active-set-sqp`) → the
  working-set warm start is the mechanism that pays there; see
  [active-set SQP warm starts](active-set-sqp-warm-start.md).

## Relationship to the differentiable frontend

`pounce.jax.PathFollower` (see [path following](path-following.md)) is the
same algorithm over a `JaxProblem`, and predates this. The two share their
step-size policy — `pounce.StepController` — so "how far to step next" has
one implementation. They differ only in how the predictor and the monitor
are obtained: the AD frontend gets them from `jax.grad` / `jax.jacobian`
and `jvp_from_state`, this one from the problem's own callbacks and
`Solver.parametric_step`.

`PathFollower` additionally offers pseudo-arclength continuation past
folds (`trace_arclength`). That is **not** exposed here — see below.

## Not implemented

* **Pseudo-arclength continuation** past turning points. `PathFollower`'s
  version is written directly against `jax.jacobian` of the stationarity
  system and an SVD of it; the generic frontend has no equivalent of that
  dense `(d, d+1)` Jacobian, and manufacturing one from `Solver.kkt_solve`
  is a separate piece of work rather than an adaptation. pounce#608 marks
  it optional; use `pounce.jax.PathFollower.trace_arclength` if you need
  it.
* **CLI and GAMS adapters.** pounce#608 sequences these after the
  `Problem` and Pyomo ones. Both need a path manifest / repeated-solve
  protocol that does not exist yet, and neither can hold a KKT factor
  across process boundaries, so both would run the zero-order fallback
  only — which is what `--warm-start` already does.
