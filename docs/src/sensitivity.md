# Sensitivity Analysis

POUNCE includes a parametric sensitivity capability compatible with
upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler
2012, DOI
[10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
It computes the first-order change in the optimal primal solution with
respect to a problem parameter, reusing the KKT factorization from the
converged solve. Four entry points cover the common workflows.

## AMPL CLI

The main `pounce` driver auto-detects the sIPOPT suffixes
(`sens_state_1`, `sens_state_value_1`, `sens_init_constr`) in an input
`.nl`, runs a post-optimal sensitivity step after the solve, and
writes the perturbed primal back as a `sens_sol_state_1` suffix — no
separate binary or flag needed:

```sh
pounce problem.nl                   # writes problem.sol
pounce problem.nl out.sol --json-output result.json --json-detail full
```

`pounce_sens` is retained as a thin backward-compatibility alias:
`pounce_sens in.nl out.sol` is identical to `pounce in.nl out.sol`, so
existing AMPL / solver scripts keep working unchanged.

Related flags:

- `--sens-boundcheck` / `--sens-bound-eps EPS` — hold the perturbed
  primal `x* + Δx` at the declared bounds by pinning each crossing
  coordinate there and re-solving, so the others move with it.
- `--compute-red-hessian` / `--rh-eigendecomp` — compute the reduced
  Hessian (and its eigendecomposition) over the variables tagged by
  the `red_hessian` integer var-suffix.

### The sIPOPT option names

The same requests can be made by upstream sIPOPT's own option names —
on the command line as `key=value`, or in an `ipopt.opt` — so a script
written for sIPOPT keeps working. They are read by the `pounce` driver
and by the `pounce-sensitivity` builder alike:

| option | effect |
|---|---|
| `run_sens=no` | solve, but skip the sensitivity step the `.nl`'s suffixes ask for |
| `compute_red_hessian=yes` | as `--compute-red-hessian` |
| `rh_eigendecomp=yes` | as `--rh-eigendecomp` (implies the reduced Hessian) |
| `sens_boundcheck=yes` | as `--sens-boundcheck` |
| `sens_bound_eps=EPS` | the margin that refinement measures a bound crossing against (default `1e-3`); it does not enable the refinement on its own |
| `sens_max_pdpert=P` | refuse to report sensitivity outputs when the converged KKT factor carries an inertia-correction perturbation above `P` |

Two of these deliberately differ from upstream's registered default,
because honouring that default would change results for anyone who
never set the option. `run_sens` is registered `no` upstream, but
pounce runs the step whenever the input declares the suffixes, and only
an explicit `run_sens=no` turns it off. `sens_max_pdpert` is registered
`1e-3`, but pounce applies **no** cap unless you set one — an unset
`sens_max_pdpert` reports the step however hard the factor was
regularized, as it always has. Check
[`SensResult::kkt_perturbations`] (Python: `info["kkt_perturbations"]`)
if you want to see the perturbation without capping on it.

`n_sens_steps` is the one sIPOPT key pounce does not honour: only the
single `sens_state_1` perturbation tier is implemented, so any value
other than the default `1` is refused with an explanation rather than
silently rounded down (gh#677).

[`SensResult::kkt_perturbations`]: https://docs.rs/pounce-sensitivity/latest/pounce_sensitivity/struct.SensResult.html

## Rust library

Reach the sensitivity path through the `pounce-rs` facade, with the
`sensitivity` feature on:

```toml
[dependencies]
pounce-rs = { version = "0.9", features = ["sensitivity"] }
```

`SensSolve` is a builder that wraps the `on_converged` callback
plumbing into a single call:

```rust
use pounce_rs::sensitivity::SensSolve;

let result = SensSolve::new(vec![2, 3])
    .with_deltas(vec![0.05, 0.0])
    .with_reduced_hessian()
    .run(&mut app, tnlp);
// result.dx, result.reduced_hessian, result.status
```

`with_reduced_hessian_eigen()` adds the eigendecomposition, and
`with_boundcheck(eps)` enables the bound refinement described under
[Bending the estimate around a bound](#bending-the-estimate-around-a-bound-modefix_relax).

### Eigenvector sign convention

Every eigendecomposition POUNCE hands back — the reduced Hessian's
here and through the CLI and Python wrappers, the QP one from
`QpSensitivity.reduced_hessian`, and `covariance().eigen()` /
`information().eigen()` in `pyomo-pounce` — returns **sign-pinned**
eigenvectors: the largest-magnitude component of each column is
positive, ties broken by the earliest row. `v` and `-v` are equally
valid eigenvectors, so without a convention the direction you read
back depends on the arithmetic that produced it and is not
reproducible across builds or machines.

The sign is all that is pinned. A repeated eigenvalue leaves the basis
*within* its eigenspace arbitrary — any rotation of those columns
diagonalizes equally well — so read a degenerate block as a subspace,
not column by column.

## Python

`solve_with_sens` exposes the same capability from the
cyipopt-compatible Python wrapper:

```python
# pin_constraint_indices is required; pass deltas=..., compute_reduced_hessian=True,
# or both. Returns (x, info) — sensitivity outputs live in the info dict.
x, info = prob.solve_with_sens(x0, pin_constraint_indices=[2, 3],
                               deltas=[0.05, 0.0], sens_boundcheck=True)
# info["dx"], info["reduced_hessian"], info["reduced_hessian_eigenvalues"], ...
```

`compute_reduced_hessian=True` returns the reduced Hessian in
`info["reduced_hessian"]`; `rh_eigendecomp=True` adds its
eigendecomposition; `sens_bound_eps=…` tunes the bound refinement. See
[`python/notebooks/04_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/04_sensitivity.ipynb)
for a walkthrough.

## Pyomo

`pyomo_pounce` wraps the same machinery in a declare-then-query
interface: flag the parameters that matter while building the model
(no perturbed values required), solve normally, then ask for
derivatives. Parameters are declared with `declare_sens_param`
(mutable `Param` or fixed `Var`, scalar or indexed); when declarations
are present, `SolverFactory("pounce").solve(m)` runs in-process and
keeps the converged KKT factorization, so every query afterwards is a
single backsolve.

```python
import pyomo.environ as pyo
import pyomo_pounce
from pyomo_pounce import declare_sens_param, gradient, estimate

m.p = pyo.Param(initialize=2.0, mutable=True)
declare_sens_param(m.p)                 # a flag, not a perturbation

pyo.SolverFactory("pounce").solve(m)    # ordinary solve

gradient(m.x, wrt=m.p)                  # dx*/dp (float)
gradient(m.con, wrt=m.p)                # d(multiplier of con)/dp
G = gradient(m.z, wrt=m.r)              # containers -> Gradient object
G[m.z[1], m.r[2]]; G.to_dataframe()     # element access / full Jacobian
estimate(m, [(m.p, 2.5)])               # first-order solution estimate at
                                        # new values, clamped to bounds
```

`gradient` returns exact first-order derivatives (unit-perturbation
backsolves, no finite differencing); `estimate` combines the stored
derivative columns for arbitrary perturbed values after the fact. Its
perturbation is measured from the solve point, not the Param's current
value, so writing a measurement into the Param before asking (the
receding-horizon pattern) does not change the answer. It also
warns when the linear step leaves the variable bounds, and
`mode="fix_relax"` pins those variables and re-solves instead, covered
in [Bending the estimate around a bound](#bending-the-estimate-around-a-bound-modefix_relax)
below. `mode="path"` applies the change a little at a time and records
where the active set changes along it, covered in
[Applying the change a little at a time](#applying-the-change-a-little-at-a-time-modepath)
below. There is one exception to the warning, a bound written on a declared Param, covered in
[Declared Params in variable bounds](#declared-params-in-variable-bounds)
below. `estimate_report()` measures the same step and reports where the
active set changes along it, covered in
[What the step did about the bounds](#what-the-step-did-about-the-bounds-estimate_report). Multiplier sensitivities are available for equality constraints.
Models without declarations solve through the ordinary AMPL/CLI path,
unchanged. See
[`python/notebooks/25_pyomo_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/25_pyomo_sensitivity.ipynb)
for a worked optimal-control example (initial conditions as
parameters; the first-move gradient IS the NMPC feedback gain).

### Bending the estimate around a bound: `mode="fix_relax"`

`estimate()` takes the linear step, and where that step leaves a
variable's bound it clamps the value and warns. Clamping is all the
linear step can do, and it costs more than the one variable: every
other variable keeps the value the step gave it, computed on the
assumption that the clamped one was free to move where the step said.
The result satisfies the bounds and no longer satisfies the
constraints.

`mode="fix_relax"` repairs the active set the step implies instead,
which is upstream sIPOPT's strategy of that name and both of its cases.
A variable the step carries past a bound is pinned there, activating it.
A bound multiplier the step drives negative is set to zero, deactivating
that bound so the variable can move. Each adds a row to the held
factorization and re-solves, so the other variables move with it:

```python
estimate(m, [(m.setpoint, 3.0)])                      # clamps
estimate(m, [(m.setpoint, 3.0)], mode="fix_relax")    # pins and re-solves
```

Both halves matter and they fail differently. On a model where
`y = 2x + 1` and `x` hits its lower bound, the linear step returns
`y = -5`, which does not satisfy the constraint at all, while pinning
`x` returns `y = 1`, matching a full re-solve. On a model whose bound
wants to release, the linear step is stuck at `x = 0` where the answer
is `x = 1.667`, because the step preserves complementarity and nothing
but the release lets the variable off its bound.

Both modes also carry a correction for the barrier. The step is taken
against a factorization held at the solve's final `mu`, so on its own it
estimates where the BARRIER problem's solution moves rather than the
original problem's, and the two differ by `O(mu)`. That is invisible at
a converged tolerance and is not at a loose one: against sIPOPT the
uncorrected step differs by 9e-6 at `tol = 1e-3` and by 2e-9 at
`tol = 1e-8`. There is no option for it, since there is no reason to
want the barrier problem's answer.

A pass takes every crossing it can see, pins them together and
re-solves, which is upstream's own loop: one violation list per pass,
and `while (bounds_violated)` as the termination condition. Each pass
rebuilds the Schur complement over the pins so far, so a pass carrying
`k` of them costs one dense `k × k` solve and `k + 1` back-solves. A
pin never rebuilds the factorization, which is what keeps it cheaper
than re-solving.

`bound_eps` sets how far outside a variable bound a step has to end to
count as having left it, and so decides what a pass pins, what
`estimate()` clamps, and what `crossed` reports. It is absolute, as the
refinement's own test is. Unset, it is how far outside the solve itself
was willing to settle, so nothing moves for a caller who does not set
it. A constraint row keeps its own floor, and a bound is released when
the step drives its multiplier negative past the solve's own margin,
whatever `bound_eps` is. `mode="path"` reads no such margin, and
passing it under `linear` or `path` warns.

`max_pdpert` refuses rather than answering when the converged factor
carries an inertia correction above the value given, since every
sensitivity output inverts that factor and a perturbed one answers for
a nearby problem. `estimate_report().perturbations` reports the same
numbers for a caller who would rather read them.

`predictor_iter` caps the passes, and is a safety limit rather than a budget.
It was a budget while a pass took only the worst crossing, which needed
as many passes as there were crossings — and on a model with more
crossings than passes the limit, not the violations, decided where the
loop stopped. On the CSTR of notebook 36 that put the pin count at
exactly the budget for every budget tried, and at 100 pins (half that
problem's degrees of freedom) the refined step came back 8.6 times
worse than the unrefined one (gh#732). `estimate()`'s warning now names
which stopping condition was reached rather than inferring it from the
pin count.

A **release** does re-factor, once per released set. It has to. An
active bound contributes `sigma = z / s` to the KKT's `x` diagonal, and
the tighter the solve the larger that term, so the released system's
information is destroyed in the converged factor to about `eps · sigma`.
Computing a release from the held factorization therefore gets *worse*
the better the solve converged — at `tol = 1e-10` the released answer
was off by 2e-4 while at a looser `1e-6` it was off by 7e-9. Dropping
the bound's `sigma` and re-factoring removes the dependence entirely.
One factorization still sits an order of magnitude under the twenty to
a hundred a re-solve runs, and a step that releases nothing pays
nothing. This is the one place the loop departs from upstream, which
puts the multiplier's row in the same violation list as the primal
crossings and takes a Schur row over it: that is the computation the
`eps · sigma` cancellation above is measuring. The pins survive a
release either way — their right-hand sides are re-measured against the
re-solved base rather than the pin set being cleared.

Releasing is the half that has to be careful about how much it does at
once, and the asymmetry is worth stating. A pin ADDS a condition, so
asking for too many shows up honestly as an augmented system that
cannot be solved. A release REMOVES one: each bound taken out of the
active set is stiffness that is no longer holding its variable there.
Take too many at once and variables that were sitting on their bounds
are carried off them, with no degrees of freedom left to pin them back
— and no failed solve anywhere to say so. So a release batch is kept
only when the step it produces is no further outside the bounds than
the step in hand; otherwise the most negative multiplier goes alone and
the next pass re-measures the rest under it.

Three things stop it short of holding every bound. The pass limit,
which a caller can raise. The problem's degrees of freedom, which no
limit helps: pinning uses one degree of freedom each, and past that no
step holds every bound at once, so the pin is refused rather than
returned from a singular system. And the refinement ending further
outside the bounds than the step it started from, which returns the
unrefined step instead — repairing an active set has to beat not
repairing it. In each case `estimate()` warns, names the variables
still outside, and says which of the three it was. `clamp` then decides
what happens to them, exactly as under `linear`.

A pass is also refused when its correction is out of scale with the
step it corrects, not only when a pinned row misses its target.
Checking the pinned rows alone is what let gh#732's hundred pins each
land within `1e-3` of where they were asked to go while the step as a
whole came back unusable: hitting the pinned coordinates says nothing
about what the correction did to the other thirteen hundred.

What counts as outside a bound is not a tolerance you pass. It comes
from the solve, which was willing to leave a converged point
`bound_relax_factor` outside its bound, so anything within that is on
the bound rather than past it.

This is what `sens_boundcheck` turns on for the CLI and the Rust API,
and it mirrors upstream sIPOPT's option of that name.

### Applying the change a little at a time: `mode="path"`

`mode="fix_relax"` decides every active-set change from full steps
taken at the base point. `mode="path"` follows the solution along the
perturbation instead: it takes the largest fraction of the change the
current active set allows, applies the one change that happens there,
and continues under the updated set. The prediction is piecewise
linear in the parameter. For a QP that is the exact solution path,
since a QP's solution is piecewise affine in the parameter. For an NLP
the one error left is the linearization at the base point, because
nothing is re-linearized between breakpoints.

Three kinds of breakpoint end a segment. A free variable reaches a
bound and is held there. A bound active at the base has its multiplier
fall to zero and the variable leaves it. A bound the path itself
started holding stops binding under a later direction and the variable
leaves it again. That last kind is what no decision at the base point
can represent: a variable can arrive at a bound partway through the
change and depart before the end.

`active_set_changes()` returns that record, which is the part no other
mode produces. It takes the same perturbation argument `estimate()`
takes:

```python
from pyomo_pounce import active_set_changes, estimate

estimate(m, [(m.setpoint, 3.0)], mode="path")
for c in active_set_changes(m, [(m.setpoint, 3.0)]):
    print(c.fraction, c.var.name, c.bound, c.action)
```

Each entry holds the fraction of the perturbation at which the change
happens, the variable, which bound (`"lower"` or `"upper"`), and
whether the variable `"reaches"` it or `"leaves"` it. The first
entry's fraction is how much of the perturbation the held solve's
active set survives unchanged.

Where the two modes settle the same active set they give the same
prediction. Where the changes are spread out along the perturbation
they differ: on the notebook's CSTR at a change large enough to
release thirteen bounds, `fix_relax` decides all thirteen at once
from base-point multipliers and its prediction lands below even
`mode="linear"` (worst relative miss 0.950 against 0.833), while
`mode="path"` applies each release at the fraction the record names
and stays the most accurate of the three (0.626). At changes this
large every first-order prediction degrades: the CSTR trajectories
read high near the start of the horizon in every mode, which is the
base-point linearization and not something more segments repair.

`predictor_iter` is the same knob it is under `fix_relax`: it caps the
active-set changes applied, and past the cap the rest of the
perturbation is taken in one step under the active set reached, with
the warning naming the cap. On the cost side a reach adds a Schur row
without re-factoring, each release re-factors once, and the wall time
grows about linearly with the changes applied, well under a re-solve.

See
[`python/notebooks/36_active_set_parametric_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/36_active_set_parametric_sensitivity.ipynb)
for the worked CSTR example behind those numbers, including `predictor_iter`
sweeps of both modes against re-solve wall time.

### A held solve at a kink: `degeneracy`

A solve can converge with a bound weakly active, the variable on the
bound with a multiplier of the same order as the slack, both of order
the square root of the barrier parameter. The solution as a function
of the parameter has a kink there, with a different one-sided
derivative on each side, and no single linear step is right for both.
The activity classifier reports such a bound as ambiguous, which is
its honest answer.

The factorization carries every bound as `sigma = z / s` on the
variable's diagonal. At a strongly active bound that is around 1e8
and the variable cannot move, at an inactive one around 1e-8 and the
bound imposes nothing, and at a kink it is of order one, so the bound
is only partly enforced, which is wrong for both sides. The thresholds that
decide activity elsewhere have no answer at a kink, since the two
quantities they compare are the same size.

`degeneracy` on `estimate()`, `estimate_report()`, and
`active_set_changes()` selects what happens then:

```python
estimate(m, [(m.p, 2.5)], degeneracy="directional")   # the default
estimate(m, [(m.p, 2.5)], degeneracy="one_sided")     # the thresholds' answer
```

`"directional"` decides each weakly active bound for the
perturbation's own direction by the directional-derivative QP (the
sIPOPT paper's eq. 14). The weakly active rows are released, removing
the order-one `sigma`, in one factorization that serves the whole
decision, and the direction of the released system is computed. Rows
it moves into violation are the ones the direction engages, and their
pin forces solve a small quadratic program, one variable per engaged
row with a nonnegativity bound, whose optimality conditions are
eq. 14's complementarity: each engaged bound either holds with a
nonnegative force or releases and moves feasibly. The active-set QP
engine solves it, the decided direction is checked against every weak
row, and the engaged set grows until no new row violates.

A row engages only when its movement exceeds the square root of the
barrier parameter relative to the direction's norm. A weak bound's
slack and multiplier carry an uncertainty equal to their own size,
so a movement below that band cannot be resolved against the bound,
and deciding it exactly would assert precision the solve does not
contain.

All three modes consume the decision: `linear` takes the QP direction
itself, `fix_relax` takes it as the predictor its refinement iterates
from, and `path` starts with the held rows pinned and the left rows
in its base-activity table, so a bound that is genuinely active for
the first stretch of the perturbation, which happens when the held
solve sits inside the ambiguous band rather than exactly at the kink,
releases at the fraction where its multiplier reaches zero rather
than at the start. The record then carries that departure at its
measured fraction.

`"one_sided"` takes the single-sided value the thresholds produce,
bit-identical to the behavior without the argument. On the CSTR held
at the record's first breakpoint, a 2% step toward the steady state
puts the thresholds on the wrong side: `linear` and `path` miss by
0.0077 with an empty record where `directional` puts all three modes
at 0.0018, and `fix_relax` reaches 0.0018 either way because its own
release test happens to read the right sign there, a favorable read
that `directional` replaces with a guarantee.

The cost is gated by the condition, and budgeted by
`degeneracy_iter` (default 16): the released solve, one further
back-solve per engaged row, and one more to recover the direction all
count against it, so the decision costs a handful of back-solves at a
kink that engages a handful of bounds. A decision whose engaged set
grows pays that recovering solve once per pass, so a set reached in
two passes costs one more than the same set reached in one.

A direction that engages more rows than the budget covers falls back to
the one-sided step with a warning. Only a budget of zero fails before
any work: which rows engage is not known until the released solve has
run, so a budget too small to finish still pays that one factorization
before reporting the shortfall. The warning names the engaged count and
the number to raise `degeneracy_iter` to, which is the retry price and
is a floor, since a later pass can engage more rows. It is always
strictly above what the failed call spent, so each retry buys progress.
`predictor_iter` keeps its meaning as the mode's own work and plays no
part in the decision. Detection also returns
nothing on a solve with relaxed bounds, where the classifier cannot
read the slacks.

`gradient()` cannot take a side, since it is asked for a derivative
without a direction, so at a degenerate base point it warns, names
the variables and bounds, and returns the one-sided value. The
direction-aware answer is `estimate()`'s.

### Refining the step: `corrector_iter`

Every mode returns a step, and that step leaves a residual in the
barrier KKT system at the perturbed parameter values. Newton iterations
against the held factorization drive that residual down, one back-solve
each and no factorization. `corrector_iter` is how many to run, on
`estimate()` and `estimate_report()`, and it stops early when an
iteration fails to improve the residual, so it is a budget rather than
a count. It defaults to zero.

The correction aims at the barrier solution at the `mu` the solve
finished on, not at a re-solve, so the accuracy it can reach is bounded
by that offset. It does not converge to the exact answer and does not
claim to.

What lets it work past a bound crossing is that the predictor already
decided which bounds moved, and the corrector applies that decision
once before iterating. A bound the step takes off its minimum comes out
of the operator, its multiplier held at zero and its complementarity
row gone. A bound the step brings onto its minimum has its diagonal
raised to the stiffness the barrier assigns there. Every other row
keeps the base point's term. Both directions are the same change to one
diagonal, so a single factorization serves the whole correction.

That decision is where the modes start from different places.
`fix_relax` and `path` compute an active set and hand it over.
`mode="linear"` holds the active set fixed as it builds the step, so
all the correction has to work with is whatever the clamp left sitting
on a bound. On the CSTR at a quarter of the change to its steady state
that is one bound against the seven the other two pass over, which is
why the linear estimate stays furthest from a re-solve. Below the first
crossing all three are the same step.

How far the correction reaches is set by how many crossings the
predictor hands over rather than by the size of the perturbation
directly. On the CSTR the notebook uses, whose first crossing is at
1.3% of the change to its steady state, `fix_relax` with eight
iterations takes the largest relative error from 2.4e-3 to 1e-6 at a 2%
change, from 1.4e-2 to 2e-6 at 5%, and from 6.0e-2 to 6.0e-5 at 10%,
which is four crossings. At seven crossings the same call improves the
estimate by about 5% and stops.

The reason it stops is the multipliers rather than the operator. They
arrive extrapolated over the whole perturbation, nothing sets them at
handoff, and once the perturbation is large that is the dominant error.
Fitting them to minimize the stationarity residual at the predictor's
variables does not help: it absorbs the error into the multipliers and
removes the signal the iterations need, which is why the algorithm uses
that estimate only to initialize multipliers before its first
iteration.

So a budget past the crossing count the correction carries buys little,
and at large perturbations it can return an estimate no better than the
step it was handed. `estimate()` warns when a correction ends without
at least halving the residual, so an uncorrected step is never passed
off as a corrected one, and `estimate_report(corrector_iter=...)`
carries the iterations spent, the residual before and after, and that
residual split into stationarity, feasibility and complementarity. The
three carry different units and different consequences: a correction
can leave the model's equations nearly satisfied and the multipliers
complementary while the Lagrangian's gradient is far from zero, and
only the first two say whether the values can be acted on.

### What the step did about the bounds: `estimate_report()`

The clamp warning names the variables it clamped and stops there.
`estimate_report()` takes the same perturbation argument `estimate()`
takes and measures the same step, so a caller can see how far along the
perturbation the active set changes:

```python
from pyomo_pounce import estimate_report

r = estimate_report(m, [(m.setpoint, 3.0)])
r.alpha          # fraction of the perturbation that fits before a
                 # bound is reached; inf when none lies in the way
r.first          # which variable or constraint is reached there
r.crossed        # {var data: distance past its bound} for the full
r.crossed_rows   # step, and the same for inequality constraints
r.violation      # constraint violation at the predicted point
r.activity       # per coordinate: inactive / weakly_active /
r.row_activity   # strongly_active / ambiguous / unidentified / ...
```

`refine_stop` says why the `"fix_relax"` refinement stopped, one of
`"settled"`, `"iteration_limit"`, `"degrees_of_freedom"` or
`"worse_than_plain"`, and is None under the other two modes. A pass
pins every crossing it sees, so the pin count says nothing about which
limit was reached and this is the only thing that does.

`mode` and `predictor_iter` select which step is measured and match
`estimate()`'s arguments of the same names. `violation` and `corrector`
are properties of the step, so a `fix_relax` estimate needs
`estimate_report(mode="fix_relax")` to be described by its own numbers
rather than the linear step's.

Under `"fix_relax"` and `"path"` the step stops at the bound, so
`alpha` is 1.0 and `crossed` is empty for every model. What those two
did about the bounds is what `"linear"` reports at the same
perturbation. `activity`, `row_activity` and `mu` come from the
converged base point and do not depend on the mode.

`alpha` comes from a ratio test along the step. Coordinates already on
a bound take no part in it, on that side: the gap left at an active
bound is the slack the barrier leaves rather than room to move, so
scoring it divides two small quantities and would become the minimum on
any model carrying an active bound. Which coordinates those are comes
from the same classifier [Activity classification](#activity-classification)
describes, and for the ones it declines to rule on, from the size of
the gap measured against `sqrt(mu)`.

Three further fields say what separates this prediction from the exact
value at the perturbed active set, which is what a caller needs when
the estimate and a re-solve disagree: `mu`, the barrier parameter the
factorization sits at; `perturbations`, the factor's inertia
corrections, non-zero when it was regularized; and `bounds_relaxed`,
true when the solve ran with a non-zero `bound_relax_factor`. That last
case is reported rather than raised on, and it empties the two
classification maps, because relaxed bounds shift the slacks the
classifier reads.

`violation` is the primal half of the residual. The dual half needs the
multipliers at the perturbed point and belongs to a corrector step,
which holds them.

### Declared Params in variable bounds

A limit is often most naturally written as a bound rather than a
constraint:

```python
m.u_max = pyo.Param(initialize=1.0, mutable=True)
declare_sens_param(m.u_max)
m.u = pyo.Var(m.t, bounds=(0, m.u_max))   # the cap, as a bound
```

`pyomo.contrib.sensitivity_toolbox`, which supplies the expression
surgery underneath, substitutes declared Params in *constraint*
expressions only. A Param left in a bound is written to the `.nl` file
as a constant at its pre-perturbation value, so the bound never moves
and `gradient(m.u[t], wrt=m.u_max)` reads exactly `0.0` — a wrong
answer that is indistinguishable from a legitimate insensitivity.

POUNCE rewrites such a bound as a constraint over the substituted
variable before the solve, so both spellings of the same limit give the
same derivative. Expression bounds work too, e.g.
`bounds=(0, 2 * m.p + 1)`. Two kinds of variable are deliberately left
alone: **fixed** Vars, whose bounds the solver never enforces, and Vars
on **deactivated** Blocks.

This is a deliberate divergence from `sensitivity_calculation`, which
still reports zero for the same model. Four things follow from it:

- **The bound is dropped on the clone that is solved.** `m.x.ub` reads
  `None` there and the NL row carries the reader's no-bound sentinel
  `1e19` — finite, so an `isinf()` test will not catch it. The model you
  wrote is never modified.
- **`estimate()` does not clamp against a rewritten bound**, and raises
  no clamp warning for it. That is correct rather than an oversight: the
  bound now moves with the perturbation, so the linear step already
  respects it to first order.
- **`covariance()`'s bound-active projection still fires.** The value
  the bound held at the solve point is recorded and read back for the
  activity test, so a `declare_fitted` variable capped by a declared
  Param is still projected and still warns.
- **It costs a row.** A simple bound is handled directly in the barrier;
  a general inequality costs a slack and a Jacobian row. A model with
  many Param-dependent bounds trades roughly one row per bound. Only
  models that write a bound in terms of a declared Param pay this.

### A Param pinned to exactly a bound

A related case, and one that used to be silent. A declared Param can pin
a variable through an ordinary equality:

```python
m.zc0 = pyo.Param(initialize=1.0, mutable=True)
declare_sens_param(m.zc0)
m.zc = pyo.Var(m.t, bounds=(0, 1))
m.zc_init = pyo.Constraint(expr=m.zc[0] == m.zc0)
```

`d zc[0]/d zc0` is `1` by construction: the equality is linear and says
so. When `zc0` sits strictly inside `zc`'s box that is what comes back.
When it sits *on* a bound — `zc0 = 1.0` here, or `0.0` — the variable is
held by the bound and the equality at once, the force that holds it has
no unique split between them, and the solve lands with a bound
multiplier far larger than the geometry needs over a slack near
roundoff. The barrier diagonal `Σ = z/s` is the product of both, and it
can reach `1e27` against Jacobian entries of `1`.

At that point the constraint rows through the variable stop being
representable against its own diagonal, and before
[#737](https://github.com/jkitchin/pounce/issues/737) the whole
derivative column read `0.00000` — `estimate()` returned the baseline
value, and nothing warned. `Σ` is now capped at the stiffness those rows
can still be seen against, so the equality is enforced again and the
column reads what the model says. The cap is a ceiling and not a
release: a bound that genuinely holds a variable still holds it, to
within roundoff of the variable's own scale, and a bound-pinned variable
that appears in no constraint row is not capped at all — there is no row
for its diagonal to swamp, and there the stiffness is exactly what
[Crossover and the barrier
diagonal](crossover.md#what-it-does-to-a-downstream-sensitivity-result)
wants every digit of.

The ceiling holds on every diagonal the sensitivity system builds, the
one [`corrector_iter`](#refining-the-step-corrector_iter) assembles when
a step brings a variable onto a bound included — a bound the corrector
newly pins arrives as `mu / s²` off the step's own endpoint, which is
the same quantity by another name.

Nothing about the solve changes; this is the sensitivity system only.

### Solver options and warm starts

Solver options reach the in-process path the same two ways they reach an
ordinary solve: factory-level (`SolverFactory("pounce", options={...})`
or `solver.options[...]`) and per-call (`solve(m, options={...})`), with
the per-call mapping winning on conflict. Everything the CLI accepts
works here: tolerances, `max_iter`, scaling, warm-start knobs.

With `warm_start_init_point=yes` (Python `True` works too) among the
options, the initial multipliers come from the model's suffixes, the
same ones the ASL path uses: `dual` for equality multipliers,
`ipopt_zL_in` / `ipopt_zU_in` for bound multipliers, matched by
component name (a constraint rewritten by the declared-parameter
surgery is reached through its internal alias). Sign conventions are
handled: `dual` holds the AMPL marginal and `ipopt_zU_in` Ipopt's
negative-at-upper value, and both are translated to the solver's
internal conventions on the way in.

One deliberate improvement over the ASL path: entries you do not
supply take the solver's own default initialization rather than zero.
Through a dense ASL array an absent entry is indistinguishable from a
zero multiplier, and a zero bound multiplier on an active bound is a
contradictory KKT certificate the solver must first recover from. A
suffix knows which entries exist, so an explicit zero is honored
(then floored at `warm_start_mult_bound_push`, exactly as a
round-tripped inactive multiplier is) and absence means "initialize as
you normally would": the solver's own `bound_mult_init_val` for bound
multipliers, and for equality duals the warm path's 0, which is not
the cold path's least-squares estimate. Seed everything from a prior
solve and the two paths behave identically; seed partially and the
in-process path degrades gracefully.

### Watching the solve (`tee=True`)

`SolverFactory("pounce").solve(m, tee=True)` streams the solver's full
Ipopt-style log — banner, problem statistics, iteration table, and
end-of-run summary — live to standard output, including inside a Jupyter
notebook cell. The log is emitted by the engine itself (the same blocks the
`pounce` CLI prints), so the in-process path just tails it: a long solve
shows its iteration table as it runs rather than as one block at the end.
Without `tee=True` the solve is silent, matching the Pyomo convention.

## Parameter covariance and identifiability

For a parameter-estimation model whose objective is a **plain sum of
squared residuals**, the factorization from ONE ordinary solve yields
the asymptotic covariance of the fitted parameters. Declare the
fitted variables (they stay free) and the residual container while
building the model, solve, and ask:

```python
from pyomo_pounce import covariance, declare_fitted, declare_residual

m.A = pyo.Var(); m.k = pyo.Var()        # the fitted parameters, free
declare_fitted(m.A, m.k)

m.r = pyo.Var(m.I)                      # residuals, one per data point
m.res = pyo.Constraint(m.I, rule=...)   # r[i] == y[i] - model(A, k, t[i])
declare_residual(m.r)

m.obj = pyo.Objective(expr=sum(m.r[i]**2 for i in m.I))
pyo.SolverFactory("pounce").solve(m)    # one solve

cov = covariance(m)                     # no further information needed

cov[m.A, m.k]               # covariance entry (either order)
cov.std_err[m.k]            # standard error of one parameter
cov.correlation[m.A, m.k]   # correlation matrix entry
cov.matrix                  # dense numpy array, ordered like cov.params
w, V = cov.eigen()          # eigendecomposition, for identifiability
```

The recipe: the parameter block of the inverse KKT matrix, one
backsolve per parameter against the held factor, equals the inverse
reduced Hessian of the eliminated problem, and for a sum-of-squares
objective `cov = 2 sigma^2 (K^-1)_pp`. The factor 2 belongs to the
unscaled sum of squares (a Gaussian negative log-likelihood objective,
`SSR / (2 sigma^2)`, would drop it). The scaling is pinned by test
against the analytical linear-regression covariance
`sigma^2 inv(X^T X)` (`pyomo-pounce/tests/test_covariance.py`).

The noise variance comes from, in order of precedence: `sigma_sq=`
(known measurement variance); the declared residuals (estimated as
`SSR / (n - n_params)`, with both numbers derived from the container);
or the `n_data=` fallback for models without explicit residuals, whose
SSR is the objective value *at the solve* — like `estimate()`'s
baseline, writing into the model afterwards (a measurement, a warm
start for the next horizon) does not move the answer. The
solve warns if the declared residuals do not reproduce the objective
value (weights or regularization terms would silently corrupt the
estimate).

**Groups.** `declare_residual(m.r_conc, group="conc")` partitions
residuals into noise groups by arbitrary user strings: containers
sharing a group (or all ungrouped containers) pool into one estimated
variance; distinct groups get their own (`cov.sigma_sq` becomes a
dict), and the covariance switches to the heteroscedastic sandwich
form, whose per-group pieces come from the same backsolves. When
groups genuinely differ, weighting the objective itself (dividing each
group's residuals by its sigma) is the statistically efficient fix;
the sandwich is the truthful report on the unweighted fit.

`cov.eigen()` returns ascending eigenvalues and matching eigenvectors.
An eigenvalue much larger than the rest flags a poorly identified
problem: its eigenvector is the parameter combination the data cannot
pin down, and the corresponding `cov.correlation` entries approach
+/-1. The returned signs follow the project-wide
[eigenvector sign convention](#eigenvector-sign-convention) —
**the largest-magnitude component of each eigenvector is positive**,
ties broken by the earliest position in `cov.params` — so the
direction reproduces across machines instead of coming back as
whatever LAPACK's build chose. `information().eigen()` is the same.
`covariance` warns when the held factor carries
inertia-correction perturbations (typically an exactly unidentifiable
parameterization) and when the covariance diagonal comes out negative
(not a least-squares minimum).

Bound and constraint activity is classified from the solve's own
barrier geometry, not a slack threshold. A STRONGLY ACTIVE bound pins
its parameter: zero variance, correlations 0, conditional on the
bound, warned. A WEAKLY ACTIVE bound (slack and multiplier vanish
together) is KEPT at its full finite variance, corrected for the
barrier weight the held factor carries; AMBIGUOUS (loosely converged)
and UNIDENTIFIED (curvature below the model's own noise scale) stay
in the free block, each with a warning. A strongly active inequality
CONSTRAINT over the fitted parameters pins a combination rather than
a coordinate: the matrix is projected on the constraint's null space,
going singular by one per binding row, and the warning names the
constraint, the pinned combination, and its conditional information.
The same limit written as a bound or as a row returns the same matrix.
A binding row that reaches the fitted parameters through free
eliminated variables cannot be represented by a restricted normal and
is kept unprojected with an explicit warning.

To classify honestly, the declaration-triggered solve sets
`bound_relax_factor = 0` (slacks must measure distance to your own
bounds). This applies to every solve routed through the sensitivity
session, not only ones that end in `covariance()`. If you need the
relaxation, pass `bound_relax_factor` explicitly in `options=`: your
value wins, and `covariance()` then refuses with a clear error rather
than classifying against shifted slacks.

The AMBIGUOUS class is the one this machinery cannot argue away: the
interior iterate simply does not carry enough information to decide
whether the constraint binds. [Crossover](crossover.md) (`crossover=yes`)
attacks that directly — it pivots to the active-set path after
convergence and returns a point at which a linearly independent set of
constraints holds with *equality*, collapsing the ambiguity into a
STRONGLY or WEAKLY ACTIVE verdict. It is a different remedy to the same
problem the `bound_relax_factor = 0` rule above addresses, and the two
compose — genuinely independently, since
[#654](https://github.com/jkitchin/pounce/issues/654); see [Crossover and
the barrier
diagonal](crossover.md#what-it-does-to-a-downstream-sensitivity-result)
for the measurement. A crossed-over point sits *on* the declared bounds,
i.e. `bound_relax_factor` inside the box the barrier measured against, so
its `Σ = z/s` used to read `z/δ` and hold the bound more loosely than an
interior iterate would have — degrading, rather than improving, every
quantity read off the held factor unless the relaxation was also switched
off. `Σ` is now re-measured against the declared bounds whenever
crossover is accepted, so the two options no longer have to be set
together.

`classify_activity()` still requires `bound_relax_factor = 0`, for the
separate reason above: the central-path checks it makes read the
barrier's own slacks, which the relaxation shifts. A solve routed through
the declaration-triggered path already sets it; a `Solver` or `SensSolve`
session you configure yourself does not, unless you ask.

**Relation to `pounce.curve_fit`.** This uses the same
scale-and-invert-the-reduced-Hessian recipe as
[`pounce.curve_fit`](curve-fitting.md) — both read a reduced-Hessian
block from the held KKT factor and scale it by `2 sigma^2` with
`sigma^2 = SSR / (n - p)` — but with one substantive difference for
**nonlinear** models: `curve_fit` factors the **Gauss-Newton** Hessian
(`pcov = 2 sigma^2 (J^T J)^-1`, the expected-information / scipy /
`pycse.nlinfit` convention, always positive semidefinite), while
`covariance()` here feeds the **exact Lagrangian Hessian** through the
`.nl` bridge, so it reports the **observed-information** covariance —
the full reduced Hessian including the residual-curvature term that
Gauss-Newton drops. The two are identical for linear models and in the
small-residual / large-`n` limit, and differ by `O(residual x model
curvature)` otherwise (a few percent on a strongly-curved fit). Neither
is uniquely "correct": Gauss-Newton is the conventional, robust default
(it cannot produce a negative variance); observed information is the
honest local curvature of the objective you actually solved (Efron &
Hinkley 1978) but can go indefinite — which is what the negative-variance
warning above is telling you. `covariance()` offers both: the default
`hessian="lagrangian"` inverts the exact reduced Hessian of the
Lagrangian, and `covariance(m, hessian="gauss-newton")` rebuilds the
expected-information form from the residual Jacobian, recovered from
the same backsolves at no extra solve (declared residuals required).
Reach for it when the numbers must match scipy/`nls`, when
`covariance()` warns about a negative diagonal, or when the covariance
must stay positive semidefinite by construction, e.g. feeding an
arrival-cost update in moving horizon estimation.
The other difference is the input surface.
`curve_fit(f, xdata, ydata, ...)` is the batteries-included fitter for a
callable model `f(x, *params)` and data arrays: it chooses a starting
point, offers robust losses, per-point `sigma` weights, confidence
intervals, prediction bands, `dpopt/ddata`, and out-of-core streaming,
and it *projects* the covariance onto the active-constraint nullspace
when a parameter sits on a bound. `covariance()` is the post-solve
primitive for a model you have **already written in Pyomo** — residuals
as constraints, arbitrary surrounding structure — where you want the
covariance of the fit as posed without re-expressing it as
`f(x, *params)`. Use `curve_fit` when the fit is naturally a
model-plus-data call; use `covariance()` to interrogate an existing
Pyomo estimation model. Both project a bound-active fitted parameter
onto the active-constraint nullspace: `covariance()` reports the
covariance conditional on the active bound (zero variance in the
pinned direction, computed by inverting the free block of the
information matrix) and still warns, since boundary asymptotics are
nonstandard. Only variable bounds on the fitted parameters themselves
are detected; a parameter held at the same value by an active
*constraint row* is treated as free
([#362](https://github.com/jkitchin/pounce/issues/362)). A bound
rewritten into a constraint by the rule in
[Declared Params in variable bounds](#declared-params-in-variable-bounds)
is the one exception: the value it held at the solve point is recorded,
so it is still detected and still projected.

**Relation to `pyomo.contrib.parmest`.** parmest is an estimation
workflow: multi-experiment data management, bootstrap resampling, and
likelihood-ratio confidence regions, at the price of restructuring the
problem into its experiment framework, with covariance computed by
finite differences or an ipopt re-solve. `covariance()` is a
post-solve primitive: the model as written, one declaration per
component, the asymptotic covariance and identifiability diagnostics
from the factorization the solve already produced. Use parmest for
multi-experiment campaigns and non-asymptotic intervals; use this to
interrogate the fit you already have.

See
[`python/notebooks/26_parameter_covariance.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/26_parameter_covariance.ipynb)
for a worked example with a Monte Carlo validated confidence ellipse
and an identifiability diagnosis.

## Activity classification

Which bounds and constraint rows are actually *holding* the solution
is a question the converged iterate answers only ambiguously: at a
weakly active bound the slack and its multiplier are both `O(√μ)`, so
no fixed threshold on either one alone separates "just touching" from
"not binding". `Solver.classify_activity()` keys on the ratio of
barrier curvature to the model's own curvature instead, which is
`O(μ)`, `O(1)` and `O(1/μ)` in the three regimes:

```python
solver = pounce.Solver(problem)   # problem.add_option("bound_relax_factor", 0.0)
x, info = solver.solve(x0=x0)

rep = solver.classify_activity()
rep["var_status"]        # ["inactive", "unbounded", "fixed", "strongly_active"]
rep["row_status"]        # ["equality", "strongly_active"]
rep["var_ratio"]         # the ratio behind each call (NaN where nothing was classified)
rep["mu"]                # the barrier parameter the calls were made at
```

Statuses are `inactive`, `weakly_active`, `strongly_active`,
`ambiguous` (the ratio fell in a gap where this `μ` cannot decide —
re-solve tighter), and `unidentified` (the curvature is below noise
scale, so the question does not arise). `unbounded`, `fixed` and
`equality` mark entries with no barrier geometry to classify.

Both arrays are indexed in **user space**: `var_*` follows your `n`
variables and `row_*` your `m` constraints, in your order. A variable
that `fixed_variable_treatment = make_parameter` removed from the
solve (`lb == ub`) reports `fixed` at its own index rather than
shifting everything after it.

Two per-entry flags report on the assumptions rather than the
geometry: `off_central_path` (`s·z` differs from `μ` by more than 10×
on some side) and `contaminated` (classified inactive yet carrying
barrier curvature well above the `O(μ)` an inactive bound should have
— typically a bound that sits close enough to the optimum to bend it).

Inequality rows classify through the same rule, via the curvature
along the constraint normal. That is the point of classifying rows at
all: move a bound off a variable and onto a row and the activity
disappears from the bound-multiplier view entirely, while any
covariance or identifiability heuristic keyed on `z` alone silently
stops seeing it
([#362](https://github.com/jkitchin/pounce/issues/362)).

The call requires the solve to have run with `bound_relax_factor=0`
(the Ipopt default is `1e-8`) and raises `ValueError` otherwise:
relaxed bounds shift the very slacks the classifier reads. The guard
tests the value that solve ran under, so setting the option after the
fact does not change the answer — set it on the `Problem` and solve
again.

## The information matrix

`information(model)` is the un-inverted sibling of `covariance()`: the
reduced Hessian over the declared fitted block, from the same single
solve, in natural units with no `sigma^2` anywhere. For a homoscedastic
Lagrangian fit, `covariance()` equals `2*sigma^2*inv(information())` on
the free block. `hessian=` selects the observed (`"lagrangian"`,
default) or expected (`"gauss-newton"`) form exactly as in
`covariance()`.

The Lagrangian form is built by tangent recovery against the held
factorization rather than by inverting the covariance back: the
K-inverse columns' x-blocks are `T*M`, so `T = Zx*inv(M)` exactly and
`R = T'HT` with the exact Lagrangian Hessian. The barrier weight
cancels multiplicatively, so equality and variable-bound activity
carries machine precision at any barrier parameter, including on
pinned parameters where a subtract-the-barrier route loses
`log10(Sigma/q)` digits. A binding inequality row is the one
exception: it couples through its slack barrier and leaves ~1e-6
relative residue at practical barrier parameters.

Membership and warnings follow `covariance()`. One disposition is
opposite by design: a strongly active (pinned) parameter's entry is
`S`, the reduction onto the pinned set, NOT a zero row — zero
information is the opposite of what a pinned parameter carries —
conditional on the rest of the pinned set, with zero cross blocks to
the free parameters. Binding constraint rows project the free block on
both sides (the pseudo-inverse of the projected covariance). An
indefinite Lagrangian block is returned as computed with a warning
naming Gauss-Newton as the PSD alternative: refusing would withhold
the finding that the point is not a minimum or the model is
over-parameterized. `eigen()` reads identifiability directly: a
near-zero eigenvalue is a direction the data does not inform; its
eigenvector's sign follows the project-wide
[convention](#eigenvector-sign-convention).

## Choosing the block: wrt=

Both accessors take `wrt=` to reduce onto any block of the solve's
variables off the held factor, post-solve; the declared fitted block is
the default, so omitting it is exactly the prior behavior. Accepted
forms: a Var (scalar or indexed, every member), an indexed slice
(`m.x[2, :]`), a `(Var, iterable)` pair, data objects, or a list mixing
these.

```python
cov = covariance(m)                      # the fitted block, as before
cov_a = covariance(m, wrt=[m.a])         # one parameter's marginal
band = covariance(m, wrt=m.r)            # a predicted trajectory
info_a = information(m, wrt=[m.a])
```

Each call re-reduces onto its own argument, so one solve serves as many
blocks as are asked about, and each block gets its MARGINAL: everything
outside it is profiled out, not held fixed. Sigma estimation always
divides by the fit's own degrees of freedom (a property of the solve,
not of the question being asked), so a sub-block's numbers agree
exactly with the corresponding entries of the default answer.

A rank-deficient block, one with more coordinates than the fit has
degrees of freedom or with linearly dependent coordinates (a
duplicated design point), is the trajectory-band case: `covariance()`
returns its (rank-deficient) marginal, `2 sigma^2 M`, the confidence
band on the fitted trajectory (add the observation noise for a
prediction band), with the membership handling bypassed, and
`information()` raises an error pointing to `covariance()`, since such
a block carries no information matrix. For `information()`,
a block that parameterizes the constraint manifold (size equal to the
degrees of freedom) gets the exact tangent construction; a sub-block of
the fitted set gets its marginal as a Schur complement of the exact
tangent R over the fitted block (never inverting a covariance, so a
pinned member costs no digits); other blocks reduce off the held factor
with the item-1 corrections, which is benign for free coordinates.

One exception is returned rather than hidden: a strongly active
variable OUTSIDE the block is not deleted from the factor, so the
block's numbers are the values conditional on that bound, not the
marginal over it. The result carries the list as `.conditioned_on`
(empty when there is none); inside-block activity is membership, not
conditioning, and is handled as before. The list is decided by the
same classification the block members get, applied per candidate as a
singleton block, so it is scale-invariant; only near-bound variables
pay the extra backsolve.

## Keeping and releasing the factor: retain_kkt(), release_kkt()

The solve factors the KKT matrix to solve the NLP; the only question is
whether that factor is kept for post-solve queries. Any declaration
keeps it. `retain_kkt(model)` keeps it with no declaration at all,
which is what `wrt=` queries with nothing declared need: the MHE case,
where the arrival state and the parameters are each queried by `wrt=`
and neither is THE fitted set. It defaults off, so a solve with no
sensitivity pays nothing.

```python
retain_kkt(m)
SolverFactory("pounce").solve(m)
arrival = covariance(m, sigma_sq=s2, wrt=m.x[:, t0])
params = information(m, wrt=[m.k1, m.k2])
release_kkt(m)          # done asking: give the memory back now
```

| setup | factor kept | `covariance(model)` | `covariance(model, wrt=T)` |
|---|---|---|---|
| nothing | no | error | error |
| `declare_fitted(S)` | yes | over S | over T |
| `retain_kkt()` only | yes | error, no default | over T |
| `retain_kkt()` + `declare_fitted(S)` | yes | over S | over T |

The retention policy in one place: the factor is kept if anything is
declared or `retain_kkt()` was called, and a `Covariance` or
`Information` result whose lazy `conditioned_on` has not been read
keeps the session alive through its pending computation until first
access. `release_kkt(model)` is the exit: it drops the model's hold
on the factor immediately, freeing the memory, while declarations and
the retain flag still apply to the next solve. Release drops the
model's hold, not a result's: a `Covariance` or `Information` with a
pending `conditioned_on`, and a `Gradient` (which reads the factor on
every lookup), each hold their own reference, so they keep working
across the release and keep the factor in memory until they are
discarded. Noise is a separate question: `retain_kkt()` keeps the
factor, not a noise model, and with nothing declared fitted the
degrees of freedom for a noise ESTIMATE are unknown, so
`covariance()` under retain-only needs `sigma_sq=`; the estimation
routes (declared residuals, `n_data=`) raise an error saying so.

Like any declaration, `retain_kkt()` routes the solve through the
in-process sensitivity path, whose `solve()` surface is not
keyword-identical to the ordinary subprocess path (for example,
`load_solutions=False` is not honored there). Adding it to an
existing script changes how the solve runs, not just what is kept.

## Units and NLP scaling

All sensitivity outputs are in **natural (unscaled) units**. The IPM
holds its converged KKT factor in an internally scaled space whenever
NLP scaling is active (the default `nlp_scaling_method =
"gradient-based"` fires when an objective gradient or constraint row
exceeds `nlp_scaling_max_gradient = 100` at the starting point);
pounce undoes that scaling in every held-factor back-solve, so `dx`,
`kkt_solve`, and the reduced Hessian are independent of how the
problem was scaled internally
([#128](https://github.com/jkitchin/pounce/issues/128)).

That covers **user scaling** too, on all three of its axes. A
per-variable `scaling_factor` is applied as a change of variables
`x̃ = d ⊙ x` below the algorithm, so the held factor is the scaled
problem's; the factors are carried into the same translation, and
every accessor answers in your units
([#486](https://github.com/jkitchin/pounce/issues/486)). The factors a
solve ran under are readable back from `Solver.nlp_scaling["x_scaling"]`
(Python) / `Solver::variable_scaling` (Rust) — diagnostic rather than a
correction to apply, since the outputs already carry it.

`classify_activity()` is scale-invariant for the same reason, and
mostly by construction rather than by undoing anything: its ratios are
formed so that rescaling a constraint row or the objective leaves them
fixed. Writing a constraint as `1000·x ≥ 0` instead of `x ≥ 0` does
not move a status, and neither does the solver's own per-row
`d_scale`. A change of variables is the one case the ratios do not
absorb on their own — the identification floor is a single number
shared across entries, so a *non-uniform* `d` would move entries
across it — and there the factors are divided out of the geometry
before anything is classified, which keeps a status from depending on
the conditioning you asked for. The values the report exports follow
the natural-units contract like everything else: `var_sigma` and
`row_sigma` are the barrier diagonals in the model's own units,
`row_normal(j)` is the constraint gradient with the solver's per-row
scale divided out, and `hessian_vec(v)` is the exact Lagrangian
Hessian times a user-space vector with the objective scale divided
out; classification happens on the scaled quantities internally, the
report never shows them.

**Variable indices are user-space, factor rows are not.** Everything
the sensitivity API reports or accepts — the `.col` file's order, the
activity report's `var_*` arrays, `row_normal(j)`'s entries — indexes
the variables you wrote. The converged factor does not: a variable
whose bounds are equal is removed from the solve
(`fixed_variable_treatment = make_parameter`, the default), so its
column is absent and every later variable sits one row earlier. The
two orders coincide exactly when the model has no fixed variable,
which makes the difference easy to miss. Translate with
`Solver.primal_rows(indices)` — `None` marks a removed variable —
before indexing a `kkt_solve` or `parametric_step_full` result, just
as `multiplier_rows` has always been required for the `y_c` block.

In particular, for a parameter-estimation NLP with the parameters
pinned by equality constraints, `-inv(info["reduced_hessian"])` is
directly the parameter covariance — no per-problem scale factor, no
need to set `nlp_scaling_method = "none"`. (Sign convention: over pin
*constraint* rows, `B K⁻¹ Bᵀ` equals the multiplier sensitivity
`∂λ/∂p = −∂²f*/∂p²`, hence the minus in the covariance recipe.)

For callers that calibrated against the pre-#128 behavior, the
solver-space value and the factors that relate the two are exposed:

- Python: `info["reduced_hessian_scaled"]`,
  `info["obj_scaling_factor"]`, `info["pin_g_scaling"]`;
  `Solver.reduced_hessian(pins, scaled=True)`,
  `Solver.kkt_solve(rhs, scaled=True)`, and the `Solver.nlp_scaling`
  dict (`{"obj": df, "c_scale": …, "d_scale": …, "x_scaling": …}`).
- Rust: `SensResult::{reduced_hessian_scaled, obj_scaling_factor,
  pin_g_scaling}`, `Solver::{compute_reduced_hessian_scaled,
  kkt_solve_scaled, nlp_scaling, pin_g_scaling}`, and
  `PdSensBacksolver::solve_scaled_space`.

The relation is `H_scaled[i,j] = df / (dc_i·dc_j) · H[i,j]`, where
`df` is the objective scaling factor and `dc_i` the pin rows'
constraint scaling factors.

One caveat: the IPM's inertia-correction perturbations (`δ_x`, `δ_s`,
`δ_c`, `δ_d`) are added to the factor in *scaled* space, so on a
problem whose final factorization needed regularization (e.g.
linearly dependent pin rows) the unscaling maps a slightly different
perturbed system per scaling method. The perturbations are reported —
`info["kkt_perturbations"]` / `Solver.kkt_perturbations` (Python),
`SensResult::kkt_perturbations` / `Solver::kkt_perturbations` (Rust)
— so a covariance workflow can assert they are all zero before
trusting `-inv(reduced_hessian)`; on well-posed estimation problems
the final factor is unregularized and the invariance is exact.

## Verification

All three entry points are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to within roughly 6e-9 per component.

The bound refinement is verified on that same example, which crosses a
bound under upstream's own perturbation: against a full re-solve the
refinement lands within 6e-9 where clamping the crossing coordinate is
off by 0.12. It is also checked on a model with three degrees of
freedom, where three coordinates cross at once and all three pins hold,
and for the refusal when the pins would exceed the degrees of freedom.

Both halves of fix-relax and the barrier correction are also checked
against sIPOPT 3.14.19 itself, driven through
`pyomo.contrib.sensitivity_toolbox`, on cases built to separate them:

| what it exercises | pounce vs sIPOPT |
|---|---|
| pinning a variable the step carries past a bound | 2e-8 |
| releasing a bound the step drives the multiplier off | 1e-6 |
| the barrier correction, at `tol = 1e-3` | 2.4e-7 |
| the barrier correction, at `tol = 1e-8` | 4e-10 |

Each case is one the other two do not reach. The release case returns
`x = 0` without it where the answer is `1.667`, since the linear step
preserves complementarity and holds the variable on its bound. The
barrier case differs by 9e-6 without the correction at `tol = 1e-3`,
and by 2e-9 at `tol = 1e-8`, which is why it is only visible where the
solve leaves `mu` loose.

## Beyond one perturbation

Everything above answers "how does `x*` move for *this* \\(\Delta\theta\\)"
— a first-order step off one converged factor. Repeat it and you are
tracing a path, at which point the questions become where the linear
prediction stops being good enough, when the active set changes under
you, and what to do where \\(\partial x^*/\partial\theta\\) goes singular.

The Python frontend answers those with `PathFollower`, which turns the
same held factor into a predictor–corrector continuation loop (and a
pseudo-arclength mode that traces through folds), plus `inverse_map_rhs`
for running the map backwards as an ODE. See
[Path Following & Inverse Mapping](path-following.md).
