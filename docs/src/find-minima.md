# Finding Multiple Minima

`pounce.minimize` finds a single local minimum from a starting point.
`pounce.find_minima` is its global-search companion: it drives the same
local solver in a loop to discover **many distinct minima**, or the global
one among them.

```python
import pounce

result = pounce.find_minima(
    fun, x0,
    method="deflation",     # see the method families below
    jac=jac, hess=hess,     # same as minimize; analytic derivatives recommended
    bounds=bounds,
    n_minima=6,             # target number of distinct minima
    max_solves=None,        # budget; default 4 * n_minima
    patience=8,             # give up after this many solves with nothing new
    dedup=1e-3,             # minima closer than this are "the same"
    seed=0,
)

result.minima   # list of minima, sorted by objective (lowest first)
result.values   # their objective values
result.x        # the best (lowest) minimum
result.status   # "target_reached" | "converged" | "budget_exhausted"
result.n_solves # solver calls used
result.trace    # per-solve diagnostics
```

Every method reuses `minimize`, so **bounds and constraints carry through
unchanged**, and the acceptance test is shared: each candidate is polished
on the clean objective, checked against the bounds, and — when a Hessian is
supplied — certified as a true minimum (positive-semidefinite Hessian, so
saddles and maxima are rejected) before being de-duplicated and recorded.

> **Logging note.** The unconstrained `minimize` facade emits a harmless
> `jacobian(): AttributeError` line per solve. Because `find_minima` issues
> many solves, set `RUST_LOG=off` (or `error`) to silence it.

The six methods fall into three families by *how* they escape a minimum
they have already found.

## Repulsion — transform the problem and re-solve

These modify the problem so the solver can no longer settle where it just
did, then re-solve. They share the lineage of the **filled-function**
method and **metadynamics**: make the found minimum unattractive.

### `flooding`

Add a repulsive **Gaussian bump** to the objective at each found minimum
`x*_k`:

> `F(x) = f(x) + Σ_k A_k · exp(−‖x − x*_k‖² / 2σ_k²)`

The bump does not move the stationary point (a Gaussian is flat on top); it
*flips its curvature*. The minimum turns into a saddle once the bump is
taller than the basin's curvature — precisely when
`A/σ² > λ_min(∇²f(x*))` — and the solver rolls off it into a new
basin. The bump is smooth with an analytic gradient and Hessian, so the
flooded problem is as solvable as the original.

* **Knobs** (`strategy_kw`): `sigma` (≈ spacing between minima), `amplitude`
  (taller than a basin is deep).
* **Best for** broad enumeration of all minima of a smooth objective.
* **References.** Ge, R. "A filled function method for finding a global
  minimizer of a function of several variables." *Mathematical Programming*
  **46**, 191–204 (1990).
  [doi:10.1007/BF01585737](https://doi.org/10.1007/BF01585737). Laio, A. &
  Parrinello, M. "Escaping free-energy minima." *PNAS* **99**(20),
  12562–12566 (2002).
  [doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399).
  Grubmüller, H. "Predicting slow structural transitions in macromolecular
  systems: Conformational flooding." *Phys. Rev. E* **52**(3), 2893–2906
  (1995). [doi:10.1103/PhysRevE.52.2893](https://doi.org/10.1103/PhysRevE.52.2893).
  Adaptive bump heights: Barducci, A., Bussi, G. & Parrinello, M.
  "Well-tempered metadynamics." *Phys. Rev. Lett.* **100**, 020603 (2008).
  [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).

### `deflation`

Instead of a finite local bump, add a **singular pole penalty**:

> `F(x) = f(x) + Σ_k η / (‖x − x*_k‖² + s)^(p/2)`

Each found minimum becomes infinitely costly. The pole reaches further than
a Gaussian (it decays as `1/r^p` rather than vanishing exponentially), so
it can clear a basin a narrow Gaussian would miss. This is the additive,
minimization-friendly realization of the **deflation** idea, whose original
form multiplies the residual of a nonlinear system by a deflation operator
to exclude known roots for a Newton iteration.

* **Knobs**: `eta` (penalty strength), `power` `p`, `soft` `s` (softening
  that keeps the pole finite for the solver).
* **Best for** enumeration on problems where the longer-reach repulsion
  helps; the most Newton/IPM-native of the repulsion methods.
* **References.** Brown, K.M. & Gearhart, W.B. "Deflation techniques for the
  calculation of further solutions of a nonlinear system." *Numerische
  Mathematik* **16**, 334–342 (1971).
  [doi:10.1007/BF02165004](https://doi.org/10.1007/BF02165004). Farrell,
  P.E., Birkisson, Á. & Funke, S.W. "Deflation techniques for finding
  distinct solutions of nonlinear partial differential equations." *SIAM J.
  Sci. Comput.* **37**(4), A2026–A2045 (2015).
  [doi:10.1137/140984798](https://doi.org/10.1137/140984798).

### `tunneling`

Rather than climb out of a basin, tunneling crosses **sideways at constant
height** to a point past the barrier, then descends. Between local solves it
seeks a point at the height of the most-recently found minimum while being
repelled from all known minima, and then re-minimizes there. The result is a
*monotonically non-increasing* sequence of minima.

* **Knobs**: `eta`, `power`, `soft` (the repelling poles).
* **Best for** finding the **global** minimum and a descending trail to it,
  not exhaustive enumeration.
* **Reference.** Levy, A.V. & Montalvo, A. "The tunneling algorithm for the
  global minimization of functions." *SIAM J. Sci. Stat. Comput.* **6**(1),
  15–29 (1985). [doi:10.1137/0906002](https://doi.org/10.1137/0906002).

A worked example of all three is in
[`python/notebooks/15_find_minima_repulsion.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/15_find_minima_repulsion.ipynb).

## Restart — choose the next start cleverly

These leave the objective untouched and only change *where* each local solve
begins.

### `multistart`

Random (or Sobol low-discrepancy) sampling of the bounds box, one local
solve per start. Simple, a strong baseline, and embarrassingly parallel.

* **Knobs**: `sobol` (low-discrepancy sampling, on by default),
  `restart_jitter` (used when no bounds box is given).
* **Best for** a robust default, especially when local solves are cheap and
  can be parallelized.

### `mlsl`

**Multi-Level Single Linkage** grows a pool of sample points and starts a
local solve from a sample only when (a) no *better* sample lies within a
shrinking "reduced distance," and (b) it is not near an already-found
minimum. The effect is that each basin is descended approximately **once**,
instead of many times as plain multistart re-discovers knowns.

* **Knobs**: `samples_per_round`, `gamma` (reduced-distance scale).
* **Best for** expensive local solves on funneling landscapes, where
  avoiding redundant descents matters.
* **Reference.** Rinnooy Kan, A.H.G. & Timmer, G.T. "Stochastic global
  optimization methods part II: Multi level methods." *Mathematical
  Programming* **39**, 57–78 (1987).
  [doi:10.1007/BF02592071](https://doi.org/10.1007/BF02592071).

See [`python/notebooks/16_find_minima_restart.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/16_find_minima_restart.ipynb),
which shows multistart spending ~15 solves (9 redundant) to find all six
camel minima where MLSL needs ~6 (0 redundant).

## Hopping — a Markov chain over minima

### `basinhopping`

From the current minimum, apply a random perturbation, locally minimize to a
neighboring minimum, and **accept or reject** by a Metropolis rule on the
objective. The chain is biased downhill, so it reliably reaches the global
minimum while collecting the distinct minima it visits.

* **Knobs**: `step` (perturbation size), `temperature` (acceptance).
* **Best for** the global minimum on rugged, high-dimensional landscapes —
  the workhorse of cluster and protein optimization.
* **References.** Li, Z. & Scheraga, H.A. "Monte Carlo-minimization approach
  to the multiple-minima problem in protein folding." *PNAS* **84**(19),
  6611–6615 (1987).
  [doi:10.1073/pnas.84.19.6611](https://doi.org/10.1073/pnas.84.19.6611).
  Wales, D.J. & Doye, J.P.K. "Global optimization by basin-hopping…" *J.
  Phys. Chem. A* **101**(28), 5111–5116 (1997).
  [doi:10.1021/jp970984n](https://doi.org/10.1021/jp970984n). Cousin with
  history feedback: Goedecker, S. "Minima hopping…" *J. Chem. Phys.*
  **120**(21), 9911–9917 (2004).
  [doi:10.1063/1.1724816](https://doi.org/10.1063/1.1724816).

See [`python/notebooks/17_find_minima_hopping.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/17_find_minima_hopping.ipynb).

## Beyond minima: saddle points and critical points

The same ideas extend to *every* stationary point of `f` — saddles
(transition states) and maxima included. A critical point has `∇f(x) = 0`;
its **Morse index** (the number of negative Hessian eigenvalues) classifies
it: `0` = minimum, `1` = transition state, …, `n` = maximum. Two entry
points are provided.

### `find_critical_points` — enumerate and classify

Stationary points are the roots of `∇f(x) = 0`, which are exactly the minima
of the gradient-norm merit `½‖∇f(x)‖²` (zero there). So `find_critical_points`
runs `find_minima` on that merit — using any enumeration `method`
(`"deflation"`, `"multistart"`, …) — then keeps the points where `‖∇f‖` is
truly zero and labels each by its Morse index. This treats pounce as a
root-finder and reuses the whole `find_minima` machine.

```python
r = pounce.find_critical_points(
    fun, x0, grad=grad, hess=hess, bounds=bounds,
    method="deflation", n_points=12, dedup=1e-2,
)
r.minima      # index 0
r.saddles     # 0 < index < n  (transition states)
r.maxima      # index n
for p in r.points:
    print(p.kind, p.x, p.f, p.index)
```

### `find_saddles` — eigenvector following

A saddle is a *minimum* in most directions and a *maximum* along a few. By
walking uphill along the `index` softest Hessian eigenvectors and Newton-
downhill in the rest, eigenvector following lands directly on an
index-`index` saddle; multistart enumerates several.

```python
s = pounce.find_saddles(fun, x0, grad=grad, hess=hess, bounds=bounds,
                        index=1, n_saddles=4)
```

Together with the minima, the index-1 saddles between them form the
transition-state network / disconnectivity graph — flooding fills the
basins, and the saddles are the barriers crossed between filled basins.

* **References.** Cerjan, C.J. & Miller, W.H. "On finding transition
  states." *J. Chem. Phys.* **75**, 2800 (1981). Henkelman, G. & Jónsson, H.
  "A dimer method for finding saddle points…" *J. Chem. Phys.* **111**,
  7010 (1999). [doi:10.1063/1.480097](https://doi.org/10.1063/1.480097).
  Henkelman, G., Uberuaga, B.P. & Jónsson, H. "A climbing image nudged
  elastic band method…" *J. Chem. Phys.* **113**, 9901 (2000).
  [doi:10.1063/1.1329672](https://doi.org/10.1063/1.1329672). E, W. & Zhou,
  X. "The gentlest ascent dynamics." *Nonlinearity* **24**, 1831 (2011).
  [doi:10.1088/0951-7715/24/6/008](https://doi.org/10.1088/0951-7715/24/6/008).

Runnable demos: a landscape with 4 minima, 4 saddles, and 1 maximum in
[`python/examples/critical_points.py`](https://github.com/jkitchin/pounce/blob/main/python/examples/critical_points.py),
and a **molecular reaction barrier** on the Müller-Brown potential —
locating the stable states and the transition states between them, then
reading off barrier heights and the minimum-energy path — in
[`python/examples/reaction_barrier.py`](https://github.com/jkitchin/pounce/blob/main/python/examples/reaction_barrier.py).

## Termination

The search stops on whichever fires first, reported in `result.status`:

| condition | meaning | `status` |
|---|---|---|
| `n_minima` distinct minima found | got what you asked for | `target_reached` |
| `patience` solves in a row find nothing new | landscape appears exhausted | `converged` |
| `max_solves` reached | spent the budget | `budget_exhausted` |

`patience` is what makes the "fewer minima exist than requested" case
efficient: ask for 6, find 2, try a few more times, and stop with
`converged` rather than burning the whole budget. `find_minima` always
returns however many minima it actually found — falling short of `n_minima`
is not an error.

A solve is many function evaluations; `max_solves` counts *solver calls*. A
true per-evaluation ceiling belongs inside each solve via
`options={"max_iter": ...}`.

## Choosing a method

See [Choosing a Multiple-Minima Method](find-minima-choosing.md), including
how the families behave as the dimension grows.

## Scope

`find_minima` covers methods that drive pounce's local solver as their inner
loop. Rigorous deterministic global optimization (branch-and-bound, DIRECT),
population/stochastic globals (differential evolution, CMA-ES — already in
SciPy), and homotopy continuation (all stationary points of *polynomial*
systems) are different machinery and out of scope.
