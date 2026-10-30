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

or, for an upper bound, `ub_j - x_j` (respectively `cu_i - g_i`). So a
pair is a variable with exactly one finite bound and a row with exactly one
finite side:

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
2. **Exact** — `G·H = 0` per pair from your start, with the same fallback.
3. **Penalty** — the complementarity products moved into the objective with
   an exact ℓ₁ weight; run only if neither of the first two answers.

The first two both run, and the answer with the lower objective wins. On a
nonconvex model each reaches local minima the other avoids: on the 60
feasible cells of the MPCC benchmark corpus (`benchmarks/mpcc/declared.py`)
the exact product alone reaches the optimum on 48 and the continuation alone
on 60, while on the CALIPSO bunny-hop the continuation alone settles on a
trajectory 2.4× worse on 3 of 5 starts that the exact product avoids.
Running both reaches the best known answer on every one of them, at roughly
the iterations of both.

A stage *answers* only if it converged, every product is within
`constr_viol_tol`, and the point is certified strongly stationary (S, see
below). A solve that exits "acceptable" at a point that is not stationary
does not end the ladder, and neither does a weakly stationary (W) point:
it is kept, and returned only if no later stage finds an S point. If no
stage answers, the exact product's own verdict is reported (for an
infeasible model, `Infeasible_Problem_Detected`) — never a relaxed solve's
success. Your options apply to every stage and are restored after
the solve. Unless you set them yourself, declared pairs also run with
`bound_relax_factor = 0` and `honor_original_bounds = yes`: a relaxed bound
moves the point a complementarity is measured from.

## What you get back

The CLI prints one summary line on stderr, e.g. for `qpec_small`:

```text
pounce: complementarity: answered by the Scholtes continuation formulation (11 lowered solves, 171 iterations in all, max |G·H| = 0.0e0); 2 pairs, 1 biactive; stationarity: S (strongly stationary).
```

From Python, `info["complementarity"]` holds `accepted_by` (the stage, or
`None` if no stage answered), `n_solves`, `total_iterations`,
`max_abs_product`, the per-solve records, and `certificate`:

| `class` | meaning |
| --- | --- |
| `S` | strongly stationary: a multiplier exists with the right sign on every biactive pair. A local minimum candidate in the usual sense. |
| `W` | weakly stationary only: converged and feasible, but every multiplier has the wrong sign on some biactive pair. There may be a descent direction that changes which side of a pair is zero. ralph1's origin is the textbook case. |
| `none` | the stationarity system is not satisfied at all. |
| `not_searched` | too many biactive pairs to search the multiplier set. |

along with `n_pairs`, `n_biactive`, `n_neither`, whether the multipliers
are unique, and the stationarity residual against its tolerance.

## Limits

- Pairs must be **one-sided**: a variable or row bounded on both sides (an
  AMPL "mixed" complementarity) is refused with the reason, never solved as
  if the pairing were absent.
- Complementarity between **cones** (friction cones, `β ∘ η = 0`) is not
  supported; lower such a model to scalar pairs.
- Presolve is declined for a model with pairs (it renumbers what they
  index), and the CLI says so. `--minima`, the interactive debugger,
  `solve_nlp_batch` and `solve_with_sens` refuse pairs.
- The JSON solve report does not carry the certificate yet; read the
  console line or the Python `info`.

## Evidence

`benchmarks/calipso/` ports the trajectory-optimization problems of the
CALIPSO paper (bunny hop, drifting car, ball-in-cup, rocket, quadruped) and
runs every formulation route by hand next to Ipopt and CALIPSO; it is where
the "which formulation works is model-dependent" claim above was measured.
The design record is `dev-notes/research/calipso-contact-plan.md`.
