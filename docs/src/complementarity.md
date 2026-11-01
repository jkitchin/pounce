# Complementarity Constraints (MPCC)

A complementarity constraint says that of two nonnegative quantities, at
least one is zero:

```text
0 <= G(x)  ⟂  H(x) >= 0      i.e.   G >= 0,  H >= 0,  G·H = 0
```

Contact (a gap or a force), phase existence (a phase amount or a
fugacity slack), switching logic and bilevel problems are all written this
way. A model with such constraints is a *mathematical program with
complementarity constraints* (MPCC), and it violates the constraint
qualifications an interior-point method relies on at every feasible point.
Written by hand as `G·H = 0`, it often stalls; written as a relaxation it
needs a schedule; written as a penalty it needs a weight. Which of those
works is model-dependent.

POUNCE takes the pairing **declared** instead, and picks the formulation
itself. Nothing needs to be set.

## Declaring pairs

**AMPL** — write `complements`. The `.nl` file carries each pair as a
complementarity row, which the CLI reads:

```ampl
var x >= 0, <= 2;  var y1 >= 0;  var y2 >= 0;
minimize f: (x-1)^2 + (y1-1)^2 + y2^2;
s.t. c1: 0 <= y1 complements 2*y1 - 1 - x >= 0;
s.t. c2: 0 <= y2 complements x + 2*y2 - 1 >= 0;
```

```bash
pounce qpec_small.nl
```

**Pyomo** — use `pyomo.mpec`. `SolverFactory("pounce")` writes the
`Complementarity` components as complementarity rows; no transformation is
needed:

```python
from pyomo.mpec import Complementarity, complements
m.c1 = Complementarity(expr=complements(m.y1 >= 0, 2 * m.y1 - 1 - m.x >= 0))
pyo.SolverFactory("pounce").solve(m)
```

**Python API** — pass `complementarity=[(row, var), ...]` to
`pounce.Problem`. Pair `(i, j)` pairs constraint row `g_i` with variable
`x_j`, each measured from its one finite bound:

```text
0 <= x_j - lb_j   ⟂   g_i(x) - cl_i >= 0
```

or, for an upper bound, `ub_j - x_j` (respectively `cu_i - g_i`). Three
shapes are lowered:

| shape | variable `x_j` | row `g_i` | meaning |
|---|---|---|---|
| one-sided | one finite bound | one finite side | `0 <= x_j - lb_j ⟂ g_i - cl_i >= 0`, either orientation |
| one-sided, plus a plain bound | both bounds finite | one finite side | the row's side names the variable bound in the pair (`g >= cl` with `lb`, `g <= cu` with `ub`); the other bound is an ordinary limit |
| mixed | both bounds finite | free | `lb_j <= x_j <= ub_j ⟂ g_i`: `g_i >= 0` at `lb_j`, `g_i <= 0` at `ub_j`, `g_i = 0` between |

A mixed pair is lowered as two product rows, `(x_j - lb_j)·g_i <= 0` and
`(ub_j - x_j)·g_i >= 0`, with `g_i` free in sign; in a `.nl` file it is a
complementarity row with flag 3 (AMPL `l <= x <= u complements g`). A box
on the row side is not lowered; put it on a variable instead (`g(x) = y`
as a row, `l <= y <= u ⟂ ...`).

```python
prob = pounce.Problem(n=3, m=2, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu,
                      complementarity=[(0, 1), (1, 2)])
x, info = prob.solve(x0)
info["complementarity"]["certificate"]["class"]   # 'S'
```

`pounce.read_nl(path).complementarity` lists the pairs of a `.nl` file as
`(row, var, flags)`.

## What POUNCE does with them

The model is solved as an NLP (never routed to a convex engine — with the
pairing it is not convex), through a short ladder of formulations, the
first one that *answers* winning:

1. **Continuation** — Scholtes relaxation `G·H <= τ`, `τ` from 1 down to
   1e-8, each solve warm-started from the last, then an exact finish
   (`G·H = 0`, or `G·H <= 0` when the equality form leaves too few degrees
   of freedom).
2. **Exact** — two solves from your start. A relaxed path: `G·H <= 1e-6`,
   one warm step to `1e-8`, then the same exact finish as the continuation.
   And the exact product solved cold, `G·H = 0` (with the same fallback).
3. **Penalty** — the complementarity products moved into the objective with
   an exact ℓ₁ weight; run only if neither of the first two answers.

The first two both run, and the answer with the lower objective wins. On a
nonconvex model each reaches local minima the other avoids: on the 60
feasible cells of the MPCC benchmark corpus (`benchmarks/mpcc/declared.py`)
the exact product alone reaches the optimum on 48 and the continuation alone
on 60, while on the CALIPSO bunny-hop the continuation alone settles on a
trajectory 2.4× worse on 3 of 5 starts that the exact product avoids.
Running both reaches the best known answer on every one of them.

The exact stage runs two solves because they, too, find different minima.
The cold exact product has no interior to work in (each product row is
parallel to its pair's active bound), so on larger models it stalls, but it
takes the path least like the continuation's and is the only one that
reaches MacMPEC `water-net`'s better branch (927 against 953). The relaxed
path converges where the cold solve stalls (on the 28 MacMPEC models where
the exact stage costs or matters most it takes about 1,900 iterations
against 25,700) and reaches branches the cold solve misses: `dempe` (28.25)
and `ex9.2.5` (5), both below their published values, and `pack-comp2p-8`.
Running both costs about a fifth more time on MacMPEC and CALIPSO and loses
no answer on either: MacMPEC goes from 137 at the best known value, 10 better
and 4 worse to 136, 12 and 3, and one CALIPSO bunny-hop start improves.

Once an answer is in hand, the exact stage is a second opinion and runs on
a budget: three times the iterations spent so far, at least 200 (never more
than your `max_iter`). The relaxed path is budgeted once the continuation
has answered with a tight S point, the cold solve once the continuation or
the relaxed path has. With no answer in hand (the continuation failed, or
found only a W point or one inside the tolerance band) the stage runs in
full. When nothing answers, the verdict reported is the cold solve's own
whenever it returned a point; the relaxed path's (its finish, or its
relaxation reported infeasible) only when the cold solve was refused. An
infeasibility verdict is local to where the solver ended, so it does not
override a cold solve that merely ran out of iterations. Every exact-product answer that
changed a MacMPEC outcome fitted inside that budget (`water-net`, 470
iterations against the continuation's 189), as do the bunny-hop starts
above. The continuation's exact finish is budgeted too, at 500 iterations or
what the schedule spent if more. A finish that has not converged by then
leaves the schedule's own answer, whose products are within `τ = 1e-8`. On
MacMPEC the budgets cut the iterations spent by 30% (112k to 78k) with no
outcome worse, and one better (`pack-rig1p-32` reaches its best known value).
The price is that a point the finish would have tightened to `1e-10` after
thousands of iterations is now reported at the schedule's `1e-8`: on one
CALIPSO ball-in-cup start the answer moved from violation 8e-11 to 1e-8
(objective 2.4e-6 lower, the slack τ allows), still certified S, in half
the time.

The remaining cost of running both is a choice you can undo when you know
your model. Two
local solves from different starts raise the odds of the better local
minimum; they guarantee nothing, and on a model where one formulation is
known to work the other is overhead. `complementarity_strategy` picks:

| value | runs |
| --- | --- |
| `both` (default) | the continuation and the exact product; the better S answer wins |
| `exact` | the exact product only |
| `continuation` | the continuation (and its exact finish) only |

```bash
pounce model.nl complementarity_strategy=exact
```

The penalty still runs under every setting if nothing answers, and the
verdict reported then is still an exact-product solve's — with
`continuation`, the continuation's own exact finish — never a relaxed
solve's success.

A stage *answers* only if it converged, every product is within
`constr_viol_tol`, and the point is certified strongly stationary (S, see
below). A solve that exits "acceptable" at a point that is not stationary
does not end the ladder, and neither does a weakly stationary (W) point:
it is kept, and returned only if no later stage finds an S point. If no
stage answers, the point reported is, in order: the first W point; else the
first converged point with every product in tolerance, reported as
`Solved_To_Acceptable_Level` since it is not certified stationary; else the exact
product's own verdict (for an infeasible model,
`Infeasible_Problem_Detected`) — never a relaxed solve's success. The
console line says which. Your options apply to every stage and are restored after
the solve. Unless you set them yourself, declared pairs also run with
`bound_relax_factor = 0` and `honor_original_bounds = yes`: a relaxed bound
moves the point a complementarity is measured from.

## What you get back

The CLI prints one summary line on stderr, e.g. for `qpec_small`:

```text
pounce: complementarity: answered by the Scholtes continuation formulation (11 lowered solves, 171 iterations in all, violation = 0.0e0); 2 pairs, 1 biactive; stationarity: S (strongly stationary).
```

`violation` is the largest complementarity violation at the reported point:
`|G·H|` for a one-sided pair, and the positive part of each product row for
a mixed pair, whose products are legitimately nonzero at a bound. The same
number is `max_abs_product` in Python and in the JSON report.

From Python, `info["complementarity"]` holds `accepted_by` (the stage that
answered with an S point, or `None` if none did), `fallback` (the stage
whose W point is reported when none answered, else `None`), `n_solves`,
`total_iterations`, `max_abs_product`, the per-solve records, and
`certificate`:

| `class` | meaning |
| --- | --- |
| `S` | strongly stationary: a multiplier exists with the right sign on every biactive pair. A local minimum candidate in the usual sense. |
| `W` | weakly stationary only: converged and feasible, but every multiplier has the wrong sign on some biactive pair. There may be a descent direction that changes which side of a pair is zero. ralph1's origin is the textbook case. |
| `none` | the stationarity system is not satisfied at all, or the point violates a row or bound by more than `constr_viol_tol` (`feasible` is then false and `primal_violation` says by how much). |
| `not_searched` | not determined: the multipliers are not unique (MPCC-LICQ fails) and the set is too large to search — more than 4·10⁶ entries in the stationarity system, or more than 600 sign-constrained columns. Accepted like S. |

along with `n_pairs`, `n_biactive`, `n_neither`, whether the multipliers
are unique, the stationarity residual against its tolerance, `primal_violation`
and `feasible`, and `tight`.

The residual is judged against `max(1e-6, tol)` plus a `sqrt(tol)` band, both
times the dual scale: an interior-point answer near a bound reproduces `∇f`
only to about `sqrt(tol)`, so a tighter test rejects optimal points. An `S` point
whose sign-free residual is inside the band but not the `max(1e-6, tol)` floor
alone is not `tight`: the driver keeps it, runs the remaining formulations, and
reports it only if none of them produces a tight answer.

### B-stationarity: when the certificate is not S

S is the strongest first-order statement, but a point can fail it and still
be a local minimum: ralph1's origin is W and is the minimizer. The question
that settles it is **B-stationarity** — no direction that keeps the pairs
complementary descends, along any branch of the biactive pairs (which side of
each stays zero). S implies B, so the driver checks B only when the reported
point is feasible and not certified S:

- **The check** solves one LP per branch: `min ∇f·d` over the linearized
  active constraints, with each pair's held side kept at zero and the released
  side non-negative, in the box `-1 <= d <= 1`. The point is B-stationary
  exactly when every LP's optimum is 0. Up to 10 biactive pairs all `2^b`
  branches are solved (no MILP: the LPs go to POUNCE's convex LP solver).
  Above that, one heuristic branch, picked from the relaxed LP's direction,
  is solved; it can find a descent but never certify B.
- **The escape.** When a branch descends, the driver re-solves that branch as
  an ordinary NLP — each pair's held side pinned, warm-started at the point —
  and keeps the result if it is feasible and strictly lower, then checks
  again (at most 3 times). The console says what happened:

  ```text
  pounce: complementarity: B-stationarity (1 biactive pair, 2 branch LPs): B-stationary -- no branch descends.
  ```

From Python, `info["complementarity"]["b_stationarity"]` holds `verdict`
(`B`, `not_B`, `not_checked`), `reason` (for `not_checked`), `n_biactive`,
`branches_solved` and `enumerated`; it is `None` when the check did not run
(an S answer, which is B, or an infeasible verdict). `escapes` counts branch
NLPs that moved the point.

On MacMPEC the check confirms the W fallbacks `ralph1`, `scholtes4` and `qpec2`
(10 biactive pairs, 1024 branch LPs, about a second) as B-stationary. On
`siouxfls` it finds a descending branch at the continuation's point
(minimized objective −207.36); the branch NLP reaches −208.258, the published
best known, where the check then reads B.

The check is on the **linearized** constraints. Where constraint
qualifications fail it can read B at a point the multiplier certificate calls
`none` — the escaped `siouxfls` point, with 1748 pairs, is one — and each is
right about its own question: no linearized branch descends, and no
multipliers reproduce `∇f`. The status then stays
`Solved_To_Acceptable_Level`.

### The exit status and the certificate answer different questions

The exit status — `info["status"]` / `info["status_msg"]`, the CLI's
`EXIT:` line, the `.sol` file, and so Pyomo's termination condition — is the
convergence level of the solve that produced the reported point. The
certificate is the verdict on the complementarity model at that point. They
can disagree, and when they do, each is right about its own question:

| status | certificate | what it means |
| --- | --- | --- |
| `Solve_Succeeded` | `S` | converged to `tol`, strongly stationary |
| `Solved_To_Acceptable_Level` | `S` | strongly stationary, but the solve that found it met only the acceptable tolerances. Common at an optimum with a biactive pair, and on badly scaled models (look for `large_dual_scale` in the log). An answer, at reduced accuracy. |
| `Solved_To_Acceptable_Level` | `none` | feasible but **not** stationary: no formulation found a stationary point. Not an answer; the status is lowered to say so whatever the solve reported. |
| any converged status | `W` | weakly stationary only, reported as the fallback (`accepted_by` is `None`, `fallback` names the stage); `b_stationarity` says whether it is B-stationary |
| anything else | — | the exact product's verdict, e.g. `Infeasible_Problem_Detected` |

Pyomo reads `Solved_To_Acceptable_Level` as `optimal`, as it does for any
model, so the certificate is the only place the `none` and `W` cases show.
Read it from `info["complementarity"]`, the console line (`tee=True`), or
the `complementarity` block of the JSON solve report (`--json-output`, or
`solve(report_path=...)` from Python).

## Limits

- A pair whose **row** is boxed, or whose variable and row both have no
  finite bound, is refused with the reason, never solved as if the pairing
  were absent. The three shapes in the table under "Declaring pairs" are
  lowered. Pyomo's `complements(m.sL >= 0, m.L >= 0)` with `m.L` declared
  `bounds=(0, 1)` writes the pair on `L`'s lower bound (flag 1) and keeps
  `L <= 1` a plain bound; an AMPL `0 <= L <= 1 complements sL` writes the
  mixed pair (flag 3).
- Complementarity between **cones** (friction cones, `β ∘ η = 0`) is not
  supported; lower such a model to scalar pairs.
- Presolve is declined for a model with pairs (it renumbers what they
  index), and the CLI says so. `--minima`, the sensitivity and
  reduced-Hessian requests, the interactive debugger, `pounce verify`,
  `check-x0`, the WASM build, `pounce.Solver`, `find_minima`,
  `solve_nlp_batch` and `solve_with_sens` refuse pairs rather than solve
  the model without them.
- The certificate is sparse above 2.5·10⁵ entries in its stationarity
  system, so a model with unique multipliers is certified at any size (a
  3 000-pair chain, `n = 9 000`, in 2.3 s for the whole solve). Where the
  multipliers are a set, the search over it is dense, and past 4·10⁶
  entries it is not attempted: the point is reported `not_searched`, and a
  stall there goes undetected.

## Worked example

Notebook
[`47_declared_complementarity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/47_declared_complementarity.ipynb)
designs a string resting on a floor (contact force ⟂ gap): the hand-written
product reports `Solve_Succeeded` at the same wrong design for every target,
the declared pairs meet every target with an `S` certificate, and it measures
`complementarity_strategy`, a `W` point, and the scaling trap: on badly
scaled rows a wrong point certifies `S`, because every tolerance is relative
to the model's own scale.

## Evidence

`benchmarks/calipso/` ports the trajectory-optimization problems of the
CALIPSO paper (bunny hop, drifting car, ball-in-cup, rocket, quadruped) and
runs every formulation route by hand next to Ipopt and CALIPSO; it is where
the "which formulation works is model-dependent" claim above was measured.
The design record is `dev-notes/research/calipso-contact-plan.md`.
