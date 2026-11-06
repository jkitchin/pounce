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

A mixed pair is `complements(pyo.inequality(lo, r, hi), f)`, in either
order. Pyomo's `mpec.nl` writer loses it (and merges a one-sided pair's
bound into a different declared bound on the same variable), so
pyomo-pounce writes those pairs itself as flag-3 (or flag-1) rows on an
auxiliary variable, and refuses the solve if the written file still lacks a
pair — see [Pyomo](pyomo.md#complementarity-constraints-pyomompec).

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
   1e-8, then an exact finish (`G·H = 0`, or `G·H <= 0` when the equality
   form leaves too few degrees of freedom). By default `τ = 1` is solved as
   an ordinary relaxation and the rest of the way is one solve in which `τ`
   falls with the barrier parameter (below); the schedule of one
   warm-started solve per `τ` is the fallback, and
   `complementarity_continuation=schedule` selects it outright.
2. **Exact** — two solves from your start. A relaxed path: `G·H <= 1e-6`,
   one warm step to `1e-8`, then the same exact finish as the continuation.
   And, until the continuation and that path both hold an exact answer,
   the exact product solved cold, `G·H = 0` (with the same fallback).
3. **Penalty** — the complementarity products moved into the objective with
   an exact ℓ₁ weight; run only if neither of the first two answers.

The first two both run, and the answer with the lower objective wins. On a
nonconvex model each reaches local minima the other avoids: on the 60
feasible cells of the MPCC benchmark corpus (`benchmarks/mpcc/declared.py`)
the exact product alone reaches the optimum on 48 and the continuation alone
on 60, while on the CALIPSO bunny-hop the continuation alone settles on a
trajectory 2.4× worse on 3 of 5 starts that the exact product avoids.
Running both reaches the best known answer on every one of them.

The exact stage can run two solves because they, too, find different minima.
The cold exact product has no interior to work in (each product row is
parallel to its pair's active bound), so on larger models it stalls, but it
takes the path least like the continuation's. It runs only until the
continuation and the relaxed path both hold an exact answer (below). The relaxed
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

Three more rules keep the exact-product solves from costing more than they
return; together they cut MacMPEC's time by 31% (3,663 s to 2,528 s) and its
iterations by a third, with no answer changed:

- **A finish never replaces a better relaxed point.** A finish solves the
  degenerate exact product and can wander: `pack-comp1p-32`'s continuation
  reaches 0.65304 at `τ = 1e-8`, and its finish used to crawl to 0.783 and
  replace it. A finish is kept unless its objective is worse than the relaxed
  point's by more than the relaxation's own slack can explain, which at a
  biactive corner is about `sqrt(τ)`, not `τ` (`scholtes4`'s `τ = 1e-8` point
  sits at `-2e-4` against the exact `0`), so the allowance is
  `max(1e-4, 10·sqrt(product))·max(1, |f|)`.
- **The cold solve runs only until two exact answers are in hand.** Once the
  continuation has a certified answer and the exact stage's relaxed path has
  answered too, the cold `G·H = 0` solve is skipped if the two agree (to
  `1e-4` relative: one point reached twice), or if both came from a
  converged exact-product finish (two branches, of which the lower is kept).
  It still runs when either answer is only a relaxed point. A continuation
  whose finish fails keeps its `τ = 1e-8` point, and that is where the cold
  solve has been measured to win: CALIPSO bunny-hop seed 1, 462.605 against
  463.195. The console says which case applied.

  Skipping it otherwise changed no outcome on MacMPEC (178 models, every
  status and objective identical to `1e-9`), `mpcc_nl`, CALIPSO, the OPF,
  capacity and transmission suites, or `relu_nn`. It cut MacMPEC's solve
  time from 1,595 s to 1,149 s and `relu_nn`'s from 178 s to 88 s. On
  `relu_nn` the relaxed path produced all 12 of the exact stage's
  improvements. `water-net`, which the cold solve once lifted to 927, reads
  953 with or without it.
- **The continuation tightens `τ` with the barrier parameter.** A schedule
  of one solve per `τ` pays at each step for starting outside the tighter
  relaxation: on the `relu_nn` suite the `τ = 0.1` and `0.01` steps alone
  are 60% of the continuation's iterations, because each starts from a point
  whose products sit at the previous `τ`. The coupled continuation keeps
  `G·H <= τ` with `τ = max(1e-8, μ)` inside one solve, cut at most tenfold
  per barrier update, so the relaxation and the barrier fall together (the
  ratio CCOpt's relaxation uses). `τ` is carried in the row, `G·H + (τ₀ − τ)
  <= τ₀`, so no bound moves and every slack stays interior; the solver
  re-evaluates the point and resets its filter when `τ` moves.

  Three things keep it safe, each measured:
  - **`τ = 1` stays an ordinary solve.** Solving the smoothest relaxation
    fully is what reaches the better basin. Coupled from the start, MacMPEC
    `bilevel1` settles on 5 where the schedule finds 0.
  - **Its finish has three starts.** The exact product is degenerate at the
    coupled endpoint, and which start converges from it differs by model:
    the warm point at `μ = 1e-6`, the warm point as it ended, the point with
    fresh duals, 100 iterations each.
  - **Its answer passes the schedule's own test or is dropped.** Converged,
    every product in tolerance, a tight S certificate; otherwise the schedule
    resumes at `τ = 0.1` from the `τ = 1` solve. The coupled solve runs on a
    budget (five times the `τ = 1` solve's iterations, at least 200), so one
    that crawls hands over early.

  Against the schedule, on the same build:

  | suite | answers | cost |
  |---|---|---|
  | MacMPEC (178) | 158 at the best known value, 7 better, 5 worse, 0 unsolved, against 157, 7, 5, 1 | solve time 1,149 s → 1,000 s |
  | `relu_nn` (42) | same 32 answers; 4 objectives lower, 1 higher | iterations −25%, wall 88 s → 68 s |
  | OPF volt-var/volt-watt (37) | unchanged | iterations −27% |
  | capacity equilibria (36) | same 32; 4 near-ties land in the other well each way | iterations −27% |
  | OPF transmission (59) | unchanged | wall 240 s → 270 s (+12%): the coupled solve reaches its budget and hands over |
  | CALIPSO declared (12) | one start better (ball-in-cup), one worse (bunny-hop, 0.13%) | iterations −9% |

- **A failed relaxation step is retried warm.** It restarts from the previous
  step's point and barrier parameter, with twice that step's iterations,
  rather than from `mu_init`'s 0.1. Skipping straight to the finish was tried
  and dropped: on the CALIPSO rocket the later steps reach a better branch.
- **A warm step that crawls is cut short** (the schedule). Each warm-started step may take
  10× the previous step's iterations, at least 500 (never more than your
  `max_iter`). Tightening `τ` leaves the previous point slightly infeasible,
  and at a small barrier parameter the iterate can creep along its bounds:
  on an IEEE 300-bus volt-var case one step took all 3000 iterations after
  steps of 25–41. Cut at 500, it is re-solved cold by `mu_strategy_fallback`
  and the run takes a third of the time, to the same answer.

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

`complementarity_continuation` picks how the continuation tightens `τ`:
`coupled` (the default, above) or `schedule` (one warm-started solve per
`τ`). On the OPF transmission suite `schedule` is 12% faster to the same
answers.

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
`bound_relax_factor = 0` and `honor_original_bounds = yes` (a relaxed bound
moves the point a complementarity is measured from) and
`perturb_slack_max_rows = 0`: that option's row budget targets models with
thousands of inactive inequality rows, and a lowered model crosses it through
its own near-active product rows. Measured on MacMPEC, turning the budget off
for these solves moves one model back to its best known value and changes no
other outcome, at under 1% in total time.

### Wide rows are lifted

A pair's product row `G·H(x)` repeats the whole row `g(x)` in the
Jacobian, and its second derivative couples the pair's variable to every
column of that row. For a row with a few entries that costs nothing. For a
wide one — a ReLU unit `y ⟂ y - (W h + b)`, whose row holds a full row of
the weight matrix — every product row becomes a dense block of the KKT
factorization, and the factorization becomes the solve.

So by default a one-sided pair whose row has 8 or more Jacobian entries is
*lifted*: its row `g(x) >= c` becomes the equality `g(x) - w = c` with a
new variable `w >= 0`, and the product row is `G·w`, with two Jacobian
entries and one Hessian entry. The two models have the same solutions, and
the KKT system the same dimension (`w` takes the place of the row's slack).
Everything you get back is in your model's terms: `x` has your `n` entries,
and a lifted row's multiplier is the one its inequality would carry.
`complementarity_lift` sets it:

| value | lifts |
| --- | --- |
| `auto` (default) | a one-sided pair whose row has at least 8 Jacobian entries |
| `yes` | every one-sided pair |
| `no` | none: every product row is `G·H(x)` |

A mixed pair (the row free) is never lifted, since its row has no sign to
give `w`. The report's `complementarity.lifted` counts the lifted pairs.

On a 64-128-128-128-10 ReLU network (384 pairs, rows of 65 and 129
entries), lifting cuts the floating-point work of each factorization from
98 to 41 Mflop. Time per iteration falls from about 115 ms to 21–30 ms, and
the default solve falls from 89 s to 24 s. `complementarity_strategy=
continuation` falls from 26 s to 1.6 s. Both reach a lower objective than
unlifted (−7.0786 against −7.0137).

The ℓ1 penalty stage (the last resort when no formulation answers) always
solves the model unlifted. The penalty makes every equality row elastic,
and a lifted row `g(x) - w = c` is an equality. Lifted, the penalty would
relax the paired row's own feasibility, which it keeps hard when unlifted.

A Pyomo model is never lifted. Pyomo's `mpec.nl` transformation already
writes each pair's expression as an equality on a new variable and
declares the pair on a one-entry row. So `auto` lifts nothing on the OPF,
capacity and `relu_nn` suites, which are built through Pyomo, and leaves
them as they were. Lifting applies to models that pose the paired
expression directly: `Problem(complementarity=...)` from Python, and AMPL
`complements`.

On MacMPEC, `auto` lifts pairs on 30 of 178 models. Run one at a time,
those 30 take 16% fewer iterations and 19% less time in total. The other
148 are unchanged. The effect is not uniform: qpec-200-4 takes 2.8 times
as long through a longer exact stage, and four models change local
optimum, two onto the best-known value and two off it. No fixture or
`benchmarks/mpcc/declared.py` paired row is wide enough for `auto`.
Lifting every pair (`yes`) is not uniformly better. It cuts the iterations
on `declared.py` by 30% to the same answers, and on CALIPSO's declared
models by 14%, but CALIPSO's rocket ends 0.14% worse after 2.6 times the
iterations. That is why `yes` is not the default.

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
`total_iterations`, `max_abs_product`, `natural_residual` (how
complementary the reported point is; see below), `polish`, the per-solve
records, and `certificate`:

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

### Complementary to `tol`, not to `sqrt(tol)`

Every formulation stops at a pair product near the last relaxation, `τ =
1e-8`. Where a pair is strictly complementary that is harmless: the other side
is `O(1)`, so the smaller one is about `τ`. Where a pair is **nearly biactive**
at the answer, both sides sit near `sqrt(τ) = 1e-4`. On a hinge
`r − s = V − knot` with slope 7.5 behind it, that leaves the controlled
quantity 6e-4 off its curve, at an objective that reads better than the true
optimum by what the slack bought (MPCCModels.jl's OPF volt-var cases, 13 of
37).

So before the B-stationarity check, the driver measures the reported point's
largest `min(|G_k|, |H_k|)`. It acts only if that exceeds `10·tol`. A strictly
complementary pair ends near `τ = tol`, and a re-solve there would buy
nothing. When it acts, it re-solves once on the branch that holds each pair's
smaller side at zero. The re-solve is warm-started from the whole point,
except that the pairs' own multipliers are zeroed; they are inactive on the
branch. The re-solved point replaces the answer when four conditions hold:

- it converged;
- it tightened the residual to `max(tol, 1e-3 × before)`. The second term
  is there because a held row is pinned only to its constraint tolerance;
- the objective did not rise by more than removing the residual can cost:
  `100 × residual × max(1, |f|)`. Over 54 measured re-solves the largest rise
  was 15.7 of those units, on a curve of slope 16. A *fall* is a lower,
  complementary point, and is accepted whatever its size;
- it certifies as the stages require: S or not searched for an answer, W for
  the weak fallback. An S answer is demoted to W only when the B-stationarity
  check at the new point finds no descending branch. The stages report W only
  when no S exists, and here the S was S only because a nearly biactive pair
  had counted as one-sided.

Otherwise the answer is kept as it was, and the console says which happened
and why:

```text
pounce: complementarity: the answer's pairs were complementary only to 7.8e-5 (min(|G|,|H|)); re-solved on its branch with each pair's smaller side held at zero: 6.0e-14, minimized objective -3.954779e-1 -> -3.954731e-1.
pounce: complementarity: the answer's pairs are complementary only to 7.8e-5 (min(|G|,|H|)), and the branch re-solve did not tighten it (the re-solve returned Infeasible_Problem_Detected); keeping it.
```

The re-solve appears in `solves` as a `branch polish` stage, distinct from a
B-escape's `branch NLP`. A kept answer is no worse than before, but its pairs
are complementary only to the residual printed.

The report says the same without the console. `natural_residual` is the
reported point's largest `min(|G_k|, |H_k|)`, on every run. `polish`,
present only when the re-solve ran, holds `taken`, `residual_before`,
`residual_after` and the `reason` it was not taken. Read
`natural_residual` alongside `certificate.class`: the kept answer above
certifies S with its pairs complementary only to 7.8e-5. `pounce-studio
diagnose` warns about it (`mpcc_loose_pairs`) when `natural_residual`
exceeds 1e-6, or `polish_threshold` (the `10·tol` above which the driver
re-solves) when that is larger: under a looser `tol`, a residual below the
threshold was left alone on purpose.

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
NLPs that moved the point. The JSON solve report (`--json-output`, or
`report_path=` from Python) carries both under `complementarity`, with the
same keys; there an unrun check is omitted rather than `null`
([schema](schema/solve-report-v1.md)).

On MacMPEC the check confirms the W fallbacks `ralph1`, `scholtes4` and `qpec2`
(10 biactive pairs, 1024 branch LPs, about a second) as B-stationary. On
`siouxfls` (1748 pairs, too many to certify, so `not_searched`), the
continuation's answer is first re-solved onto its branch (residual 1.7e-7 →
2.4e-11), and the check then reads B at −208.258, the published best known.

The check is on the **linearized** constraints. Where constraint
qualifications fail it can read B at a point the multiplier certificate calls
`none`, and each is right about its own question: no linearized branch
descends, and no multipliers reproduce `∇f`. The status then stays
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

Notebook
[`48_relu_networks_as_complementarity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/48_relu_networks_as_complementarity.ipynb)
optimizes over a trained ReLU network. Each hidden unit `y = max(0, z)` is
the pair `0 <= y ⟂ y - z >= 0`, so the network is linear rows plus one pair
per unit. From 40 starts:

- **Declared pairs** reach the MILP-certified global minimum 18 times, and
  every answer is a local minimum of the network.
- **Products written by hand** converge just as often, but never reach the
  global minimum, and 23 of their answers sit on a kink the network still
  descends from.

It also times the global alternative: the big-M MILP's time to a proof grows
from 0.1 s to beyond 120 s between 16 and 96 hidden units, while one MPCC
solve stays near a second.

## Evidence

`benchmarks/calipso/` ports the trajectory-optimization problems of the
CALIPSO paper (bunny hop, drifting car, ball-in-cup, rocket, quadruped) and
runs every formulation route by hand next to Ipopt and CALIPSO; it is where
the "which formulation works is model-dependent" claim above was measured.
The design record is `dev-notes/research/calipso-contact-plan.md`.
