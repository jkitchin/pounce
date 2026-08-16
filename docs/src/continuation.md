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
scales — 12 cells, 240 solves per arm, warm-start-eligible steps only.
Measured on **`cfc1121`**, at both settings of `warm_start_recentering`
(pounce#606), because the warm baseline is what every margin here is
quoted against and a claim about the predictor has to name the baseline
it was taken against:

| arm | iterations (`recentering=residual`) | vs. `warm-ipm` | iterations (`recentering=none`) | vs. `warm-ipm` |
|---|---|---|---|---|
| `cold-ipm` | 2492 | +102.8% | 2492 | +102.8% |
| `warm-ipm` | 1229 | — | 1229 | — |
| `pred-ipm` | 1258 | **+2.4%** | 1257 | **+2.3%** |
| `predcorr-ipm` | 1242 | **+1.1%** | 1215 | **−1.1%** |

Warm starting is worth 2.03×. The tangent predictor on top of it is
noise — within ±3% either way — and at the largest step scale `pred-ipm`
is reliably *worse* (+17.5% at horizon 80) because it extrapolates
across critical-region boundaries the corrector then has to undo.

**`warm_start_recentering` does not move this verdict, and it is worth
saying why.** The `cold-ipm`, `warm-ipm` and `pred-ipm` columns are
*bit-identical* between the two settings: over 240 steps each, zero
steps change iteration count. Only `predcorr-ipm` moves, on 14 of 240
steps. That is the mechanism, not a coincidence — recentering measures
the supplied iterate and adapts to how far off-centre it is, and
`predcorr-ipm` is the only arm that supplies a *perturbed* multiplier
seed (the previous solution stepped along the tangent). The other arms
hand over either a cold point or an exactly-converged one, and on these
single-block, zero-inequality-row models there is nothing for the
measurement to change.

The consequence for reading the table: the `−1.1%` at `recentering=none`
is not the predictor being better there. It is the same predictor with
its off-centre dual seed left alone, landing better on 14 steps and
worse on others, against an *identical* 1229-iteration baseline. Since
the baseline does not move between settings, the failure mode of "the
predictor looks better because the warm baseline was raised under it"
does not arise here — it could not, because nothing raised it.

The baseline *did* move against the previous measurement on `70bf53de`
(warm 1097 → 1229, cold 2360 → 2492). Because `recentering=none` is by
definition pre-pounce#606 behaviour and reproduces 1229 exactly,
pounce#606/#620 contributes **none** of that move; it comes from the
other merges in between (pounce#605/#619, #602/#614, #607/#623).

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

Both frontends now offer pseudo-arclength continuation past folds;
`Continuation.trace_arclength` is described above.

## Subdividing a prescribed path

`run` traces the points you asked for. When a corrector rejects between
two of them, it does not have to give up or record the runaway iterate
as an answer: it halves the gap and re-predicts from the last point
known good, and repeats. Inserted points carry `prescribed=False` and
are counted by `trace.n_inserted`; every prescribed point still appears,
in order.

```python
trace = drv.run(thetas, subdivide=True, max_subdivisions=10)
print(trace.n_inserted, trace.n_rejections)
```

This is on by default and is a **no-op on a healthy path** — the step
count and the per-step iteration counts are unchanged when nothing goes
wrong. `subdivide=False` restores the one-solve-per-point behaviour
exactly.

Monitor-driven subdivision is opt-in and separate, via `subdivide_tol`:
with a `monitor` supplied, a predicted point whose KKT residual exceeds
it is not even attempted, and the gap is halved first. It is a *separate*
knob from `monitor_tol` on purpose — `monitor_tol` is `follow`'s
accept-without-solving threshold, typically `1e-6`, which as a
subdivision trigger would subdivide on essentially every step.

`max_subdivisions` caps halvings, not inserted points; once the budget
is spent the driver attempts the prescribed point directly rather than
abandoning it.

## Past a fold: `trace_arclength`

`run` and `follow` both march in `θ`, so both stop dead at a turning
point — past a fold there is no solution at the next `θ` for the
corrector to find. `trace_arclength` parametrises the solution curve of

```
R(x, λ, θ) = [ ∇f(x) + A(x)ᵀλ ; g(x) − c(θ) ] = 0
```

by its own arclength instead, so `θ` is free to stop and reverse.

```python
trace = drv.trace_arclength(x0, theta0, callbacks=obj,
                            ds=0.25, n_steps=40, direction=-1.0)
```

**Why not the held factor.** `Solver.parametric_step` back-solves
against the factor of a *converged* solve. A fold has none: `∂x*/∂θ` is
singular there — that is what "fold" means — and past it there is no
solution at that `θ` for any factor to belong to. So the tangent is
taken instead as the null vector of the `(d, d+1)` augmented matrix
`[∂R/∂z | ∂R/∂θ]`, obtained by **bordering** it with the previous
tangent and solving

```
[ ∂R/∂z   ∂R/∂θ ] [ t ]   [ 0 ]
[     t_prevᵀ   ]       = [ 1 ]
```

which is nonsingular *at* a simple fold — that is the point of the
pseudo-arclength formulation — and needs no SVD, unlike `PathFollower`'s
dense route. `R` and its Jacobian are assembled sparsely from the
problem's own cyipopt-shaped callbacks; the Hessian of the Lagrangian is
exactly `∂/∂x` of the stationarity block, so nothing is approximated and
no third derivative appears.

**The cost.** One sparse LU per Newton iteration, where parameter
continuation gets a back-solve against a factor the solver already
built. That is the honest price of going round the corner at all: there
is no factor to reuse there. On the test fixture the whole 40-point
traverse takes single-digit milliseconds; on a large model the LU is the
dominant cost and this mode is correspondingly more expensive per point
than `run`.

**Measured.** On `min x₀³/3 + x₁²/2  s.t.  x₀ + x₁ = θ`, whose solution
curve `x₀ + x₀² = θ` folds at `x₀ = −1/2, θ = −1/4`:

| | result |
|---|---|
| `run`, marching θ from 2.0 to −0.4 | fails at θ = −0.4 with `Diverging_Iterates`, \|x₀\| > 10¹⁰ |
| `run` with `subdivide=True` | walks in to θ = −0.250, x₀ = −0.49995 — locating the fold to 1.2×10⁻³ — then reports `subdivision_exhausted` |
| `trace_arclength`, same start | turns at θ ≈ −0.25 and continues onto the branch with x₀ < −1/2, every point a root of `R` to < 10⁻⁷ |

**Scope (v1)** — deliberately `PathFollower`'s: scalar `θ`, equality /
unconstrained families, fixed active set along the traced branch.
Two-sided inequality rows are rejected with an error rather than
mis-traced; a branch that runs into a variable bound ends the trace with
`status="bound_active"` rather than reporting a wrong curve. Bifurcation
and branch switching are out of scope.

The fixture is built so LICQ holds at the fold and `λ` stays finite
there. A fold resting on a vanishing constraint gradient sends `λ` to
infinity, and no arclength scheme in `(x, λ, θ)` passes that — such a
fixture would flatter the method rather than test it.

## The CLI: a path manifest

`pounce-continue` traces a whole parametric path from one command:

```
pounce-continue path.json --out trace.json
pounce-continue path.json --cold          # the baseline it is measured against
```

The manifest names the models the modeling system already emitted, one
per parameter value:

```json
{
  "version": 1,
  "points": [
    {"model": "mpc_000.nl", "theta": [1.5, 0.0]},
    {"model": "mpc_001.nl", "theta": [1.45, 0.03]}
  ],
  "options": {"tol": "1e-8"},
  "warm": true
}
```

**There is no tangent predictor here, and there cannot be.** The
predictor is a back-solve against the KKT factor the previous solve left
in memory, and that factor does not survive `exec`. A CLI path is a
sequence of separate processes, so the transfer is zero-order. The trace
says so in its `predictor` field rather than leaving it to be inferred.

What *does* cross the boundary is more than the primal point. An AMPL
`.nl` file carries an initial primal point (the `x` segment) *and*
initial duals (the `d` segment), and pounce's reader honours both, so
the driver folds the previous point's answer into the next model and
turns on `warm_start_init_point`. The bound multipliers and the barrier
parameter have nowhere to go in the `.nl` format, and that is the gap
against the in-process driver.

**Measured**, on a 20-point van der Pol NMPC path (horizon 40, n = 122):

| | iterations | evaluations | wall |
|---|---|---|---|
| repeated cold `pounce model.nl` | 226 | 1230 | 233 ms |
| `pounce-continue` (warm transfer) | 193 | 1166 | 221 ms |
| | **−14.6%** | **−5.2%** | **−5%** |

Iteration counts are exactly reproducible run to run; the wall-clock
figure is the mean of three and the spread overlaps. The gap between
−14.6% of iterations and −5% of wall clock is process startup, `.nl`
parse and presolve, which at these sizes dominate the solve — repeating
at horizon 300 (n = 902) moves the iteration saving not at all
(225 → 193) and the wall time not at all. **If you are paying per
process, that overhead is what you are paying, and the warm transfer
does not address it.** The in-process driver is the answer there.

One thing worth knowing: a linear-quadratic MPC is a convex QP, and the
CLI routes convex QPs to `pounce-convex`, which never reaches the NLP
warm-start path at all. The measurement above uses van der Pol dynamics
for that reason; a linear-quadratic path shows exactly 0% because the
warm-start options are not consulted.

## GAMS

`pounce.gams.continuation.trace` drives a GAMS path through the same
driver. The pip link builds an ordinary `pounce.Problem` from a GMO
view, so when the points are driven from one Python process the whole
driver applies — **including** the tangent predictor, unlike the CLI
path:

```python
from pounce.gams import continuation as gams_cont
trace = gams_cont.trace(view_of_theta, thetas, pins=[0, 1],
                        options={"tol": 1e-8})
```

Driving it the other way — `option nlp = pounce;` inside a GAMS loop —
is one link invocation per solve, GAMS owns the process, and the same
process-boundary limit as the CLI applies.

**The native C link's state file does not help here**, and it is worth
being precise about why. `gams/gams_pounce.c`'s `sqp_state_file` holds
the *discrete working set only*: one byte of `bound_status` per variable
and one of `cons_status` per constraint, behind a magic string and a
checksum over `(n, m, bounds)`. No primal point, no multipliers, no
barrier parameter — it feeds `IpoptSetWarmStartWorkingSet` on the
active-set SQP path. For interior-point continuation that is strictly
less than the pip link already holds in memory. And the checksum is
taken over the bounds, so any change of problem *shape* — a horizon
shift, a remesh — invalidates it, which is precisely the case the
transfer map exists to serve.

## Not implemented

* **Bifurcation detection and branch switching.** `trace_arclength`
  follows the branch it starts on. At a bifurcation (as opposed to a
  simple fold) the bordered system is singular and the trace stops
  rather than picking a branch.
* **Folds with a moving active set.** The arclength residual `R` treats
  every general row as an active equality, so a branch whose active set
  changes as it turns is out of scope, and two-sided inequality rows are
  rejected rather than mis-traced.
* **Vector-parameter arclength.** "Past the fold" is not defined for a
  solution manifold of dimension > 1; reparametrise onto a scalar path
  and trace that.
