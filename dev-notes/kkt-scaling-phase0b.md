# KKT scaling, Phase 0b: where GasLib-40's superlinear solve time comes from

Structured-KKT decomposition, Phase 0b: fit an exponent in the number of time
blocks to every quantity that could carry superlinear solve time, and place the
growth in one of four rows — fill, pivoting, iterations, other per-iteration
work. Only fill and pivoting are things a block-structured KKT solve can change.

Harness: `benchmarks/kkt_scaling/sweep.py` (the report fields it reads were
added in the same branch: factorization seconds, work proxy, delayed / 2×2 /
tiny pivots, largest front, and restoration factorizations reported apart).

## First family: horizon extension

GasLib-40 transient (public GasLib topology): hourly backward-Euler finite
volumes, initial state fixed to the steady state, daily sinusoidal demand,
**no periodic terminal constraint** (that would close the chain into a ring),
cold start. Horizon `H` = 6, 12, 24, 48, 96, 192 hours → 7 to 193 time points,
5 788 to 162 772 free variables (≈ 844 per hour, exactly linear). pounce at
`cecadc3d`, release build, default options except `feral_ordering`. All 18
solves converged (`Solve_Succeeded`).

## Result

Exponent in `H` (log-log least squares over all six sizes):

| metric | auto | metis | auto_race |
|---|---|---|---|
| iterations | 0.05 | 0.05 | 0.09 |
| nnz(L) | 1.19 | 1.13 | 1.14 |
| work proxy per factorization | **1.71** | **1.54** | **1.54** |
| delayed columns per factorization | 0.86 | 0.82 | 0.88 |
| factorization s / iteration | 1.23 | 1.10 | 1.07 |
| other s / iteration | 1.11 | 1.13 | 1.10 |
| wall | **1.24** | **1.16** | **1.17** |

| H | n | wall (auto) | iterations | largest front auto / metis | factorization share |
|---|---|---|---|---|---|
| 6 | 5 788 | 1.1 s | 152 | 95 / 112 | 64% |
| 24 | 20 980 | 6.5 s | 177 | 256 / 320 | 55% |
| 96 | 81 748 | 34.2 s | 180 | 698 / 683 | 68% |
| 192 | 162 772 | 67.8 s | 189 | 1 002 / 773 | 72% |

## Reading

1. **This family does not reproduce quadratic scaling.** Wall time grows as
   `H^1.16–1.24` over a 28× range of problem size. Whatever produced the
   observed near-`N²` behaviour is not present in this configuration.
2. **Iterations are flat** (exponent ≤ 0.09, 152–219). The iteration row is
   empty here.
3. **Pivoting is not the source.** Delayed columns per factorization grow
   slightly *less* than linearly, i.e. the delayed work per unit of problem is
   constant or falling.
4. **Fill is the one superlinear signal, and it is moderate.** The work proxy
   per factorization grows as `H^1.5–1.7` and the largest front grows with the
   horizon under every ordering (metis 112 → 773). For a chain of fixed-size
   time blocks, a nested dissection that cuts *between time points* has a top
   separator of one interface, independent of `H`; a front that keeps growing
   means the partitioner is cutting elsewhere. That is the case Phase 2
   (structure-derived block-level ordering) exists to test. It is bounded,
   though: factorization time per iteration grows only as `H^1.1–1.2`, so a
   perfect temporal ordering would buy roughly `32^0.2 ≈ 2×` at the largest
   size, not an exponent change from 2 to 1.
5. **`metis` is no better than the default here** in wall time, despite lower
   fill at 96–192 h: it takes more iterations at 12 h and 96 h. Ordering
   changes the trajectory (it is also a pivoting decision), so fill alone is
   not the wall-time predictor.
6. **Non-factorization work is ~30% of the time and grows ~linearly** (1.1),
   so it is not a hidden quadratic either.

## Second family: the observed `n^2.3`, reproduced and attributed

GasLib-40-T transient optimal control (Radau collocation, 3 points per element,
`n_seg = 10` volumes per pipe, 775 states per time point, six compressor-ratio
controls, 36 bar pressure floor, simulated warm start): a **fixed 24 h
horizon, time step refined** from 2 h to 15 min — 12, 24, 48, 96 elements,
27 978 to 223 782 variables. This is the family pounce#947 reported. Solved
through the JAX front end at `cecadc3d`; every solve converged to the
documented objectives (160.7544, 160.5897, 160.5534, 160.5360 MWh).

Exponent in `n`:

| metric | auto | metis |
|---|---|---|
| iterations | 0.13 | −0.09 |
| nnz(L) | 1.50 | 1.34 |
| work proxy per factorization | **2.41** | **1.96** |
| delayed columns per factorization | 1.15 | 1.22 |
| factorization s / iteration | **2.41** | **1.69** |
| other s / iteration | 1.14 | 1.10 |
| wall | **2.35** | **1.47** |

| n | auto: wall, iters, factorization share, largest front | metis: wall, iters, share, front |
|---|---|---|
| 27 978 | 22 s, 68, 65%, 937 | 16 s, 63, 64%, 1 034 |
| 55 950 | 144 s, 83, 87%, 1 809 | 100 s, 94, 80%, 1 932 |
| 111 894 | 495 s, 58, 93%, 3 535 | 142 s, 41, 89%, 2 644 |
| 223 782 | **3 408 s**, 103, **96%**, 4 856 | **428 s**, 68, 85%, 3 414 |

**The quadratic is the fill row.** Iterations are flat, non-factorization work
is linear, and delayed pivoting grows only slightly faster than the problem;
factorization time per iteration tracks the work proxy exactly under `auto`
(2.41 and 2.41), and the largest front grows almost linearly with `n`. At 224k
variables factorization is 96% of the solve.

**`metis` removes most of it**: 8× faster at 224k, wall exponent 2.35 → 1.47,
and over the last doubling its factorization time per iteration grows as only
`n^0.9` (fill ≈ linear, 49M → 102M). For a two-dimensional mesh, nested
dissection's asymptotic cost is `N^1.5` flops; `metis` is at or below that at
the top of this range. The pounce#947 comparison point: the same 224k problem
took 4 357 s at the default ordering before feral 0.18; now the default `auto`,
which never selects nested dissection, takes 3 408 s and `metis` takes 428 s.

**`auto_race` finds it without being told.** Racing AMD against METIS on fill,
it chose the `metis` factorization at every size (identical `nnz(L)` and
fronts) and matched its time to within 1%: 16.5, 101.6, 145.4, 429.2 s, wall
exponent 1.46. The race's extra symbolic pass is noise at this scale.

## Prior measurement: time-axis decomposition was tried and lost

The study behind pounce#947 (the second family above) also measured the
decomposition this plan proposes. That was before feral 0.18, but the fix
(feral#203) changed only FERAL's own nested-dissection separators; a
caller-supplied ordering and the Schur path's interface do not go through it.

It lost:

- four hand-built orderings that eliminate along the time axis, supplied
  through `Problem.set_ordering`, were **exactly linear in the horizon and
  20–40× slower** than the generic ordering;
- a one-level Schur partition of the horizon (`set_kkt_schur_block`, a
  793-index interface, 0.18% of the KKT dimension) was **about 3× slower per
  iteration**.

The recorded reason is structural: each time point carries 775 states whose
spatial coupling has only 1 977 nonzeros (average degree under 3), so cutting
between time points turns a sparse interior into a dense front. In the plan's
terms, the linking set between consecutive blocks is the whole state slice —
the same size as the block — so the premise `m ≪ n` does not hold. The model is
a two-dimensional space-by-time mesh, sparse in both directions, and a
two-dimensional nested dissection (what `metis` computes) is the right
elimination; a one-dimensional temporal one is not.

## A structure-derived space × time ordering, measured

The earlier study's structure was the wrong one (time only). The obvious
structure-derived alternative uses the network as well: every state belongs to
a network cell (a node pressure, a pipe-interior pressure, a segment flow, the
energy integral) and every compressor control is its own cell; the cell graph
is split by recursive vertex separators (on a pipe network these are cut
pipes), time is split at element boundaries, and a two-dimensional nested
dissection eliminates across whichever cut is cheaper at each level. Every KKT
unknown is placed from the declared layout plus the Jacobian. Built outside the
repository (it imports the model code) and supplied through `set_ordering`.

Three pieces of structural knowledge turned out to be needed, each found by a
measurement that failed without it:

- **spatial cells** — the network cuts were exact: zero KKT edges cross a
  spatial separator;
- **time spans** — a compressor-control knot's piecewise-linear support covers
  several elements, so it belongs in any time separator it straddles; placed at
  a single point, one time cut leaked 2 282 edges;
- **primal-dual pairs** — each collocation equation is pivoted with its own
  state; separating them at time cuts produced 36 506 delayed pivots per
  factorization and a solve that hit 500 iterations; keeping the dual with its
  state brought delays to 31 per factorization (`metis`: 146–285).

End to end, same build, same run (28k: 33 s / 72 iterations against 16 s / 63):

| n | structure-derived: wall, iterations, factorization s / iteration | `metis` |
|---|---|---|
| 55 950 | 95 s, 78, 1.04 | 102 s, 94, 0.87 |
| 111 894 | 209 s, 75, 2.49 | 145 s, 41, 3.14 |
| 223 782 | 654 s, 96, 6.16 | 424 s, 68, 5.28 |

A single-factorization probe had shown this ordering 1.4× faster than `metis`
at 56k and 112k; over a full solve the per-iteration cost is within
0.8–1.2× of `metis`, and the iteration count is higher at the larger sizes (an
ordering is also a pivoting decision, so it moves the trajectory). **This
structure-derived ordering does not beat the generic one on gas.** It is one
design — leaf ordering, separator choice, and which dimension to cut are all
open — so it does not show that no structure can; it shows that the network ×
time structure, applied this way, buys nothing METIS does not already find.

## A side finding: 2×2 pairing, not ordering, dominates the factor cost

Parked here because it is a factorization question, not a decomposition one,
but it bounds what any ordering — generic or structure-derived — can buy on
this model. `feral_ordering_preprocess` (added for this) shows that `metis`'s
automatic preprocessing is `ldlt_compress`: MC64 matching pairs each state
with its collocation equation and the pairs are ordered as super-variables.
At 112k variables that pairing costs 7× in flops (6.5e10 against 8.8e9 per
factorization, 2.9 s against 0.42 s). Dropping it is not usable as is —
~100k delayed columns per factorization, tiny pivots, quality escalations, and
at 56k 190 iterations and 234 s against 94 and 100 s.

A value-based *selective* pairing — keep a matched pair only when neither
member passes the Bunch-Kaufman one-pivot test at the first factorization —
was built and measured and does not separate the two: at 56k, keeping 61% of
the pairs halves the delays and saves 18% of the flops; keeping 3% saves 70% of
the flops and keeps nearly all the delays. Which pivots go weak is decided by
Schur updates during elimination and by how the iterate evolves, not by the
original diagonal, so a static test on the first matrix cannot find them.
Structure does not identify them either: every collocation state carries a
Hessian diagonal. The literature on this (Duff-Pralet matching compression;
Schenk-Wächter-Hagemann on IPOPT KKT matchings; Hogg-Scott compressed threshold
pivoting; quasidefinite regularization; static pivoting with refinement) is a
separate line of work from decomposition by structure and is left for later.

The consequence for decomposition: **a declared block structure must keep each
primal-dual pair inside one block.** The first structure-derived ordering
split collocation equations from their states at time cuts and delayed 36 506
pivots per factorization; keeping the pairs together brought that to 31.

## Phase 0c: the arrowhead — corrective N-1 AC-SCOPF

`benchmarks/kkt_scaling/gen_scopf.py`: a base-case polar AC-OPF plus K
contingency copies of a pglib network, each with one branch out, coupled only
through the base-case dispatch of the non-reference generators (the shared
block); each contingency carries its own voltages, reactive power, reference
generator and a corrective re-dispatch within ±20% of capacity. (Preventive —
one dispatch for every contingency — is jointly infeasible on case118_ieee from
K = 64 although every single contingency is feasible; Ipopt agrees. Outages are
screened individually and cached.) Base cases reproduce pglib's published
optima (97 214 and 1 258 844). Two families, all 45 solves converged:

- `case118_ieee`, K = 1…128: 686 to 44 247 variables, 53 shared, 343 per block;
- `case1354_pegase`, K = 1…64: 6 454 to 209 755 variables, 259 shared, ~3 200
  per block.

Exponent in `n` (auto / metis / auto_race within ±0.1 of each other):

| metric | case118 | case1354 |
|---|---|---|
| iterations | 0.23 | 0.10 |
| nnz(L) | 0.99 | 1.00–1.04 |
| work proxy per factorization | 0.96 | 1.36–1.49 |
| factorization s / iteration | 0.95 | 1.10–1.15 |
| other s / iteration | 1.01 | 1.04–1.08 |
| wall | 1.22 | 1.17–1.19 |

**The generic orderings already find the arrowhead.** Fill is exactly linear in
the block count and the largest front is bounded by the block (case118: 38–44
rows at every K from 8 up; case1354: it grows toward the block size and
plateaus, 1 014 at K = 32 and 1 025 at K = 64): they eliminate each block and
then the shared dispatch, which is the block elimination a structured solver
would perform. The three orderings are within about 10% of each other. There is
no fill for declared structure to remove on one core.

**Factorization is not most of the time.** It is 25–40% of the solve; at 210k
variables (77 s) the breakdown is function evaluation 37.5 s (48%, of which the
Lagrangian Hessian 27.8 s), factorization 27.1 s (35%), back-solve 8.2 s (11%).
Both large terms are block-separable: every contingency's residuals, Jacobian
and Hessian, and every block's factorization, are independent given the shared
dispatch. **On the arrowhead the value of declared structure is parallelism —
across block evaluations as much as block factorizations — and memory, not a
better ordering.** Not measured here: how much of that FERAL's tree
parallelism already delivers inside the factorization, and whether evaluation
through the AMPL `.nl` interface can be run per block at all.

(Aside, checked and **not** a defect: `FireIntermediateCallback` takes 6.6 s of
that 77 s solve. It fires at the top of each iteration, right after the iterate
moves, so it is the first caller to ask for the objective and the primal/dual
infeasibility at the new point and is billed for evaluating them; the
convergence check later reuses the cached values. The counters confirm no
duplicated work: 60 objective / constraint / Jacobian / Hessian evaluations for
59 iterations.)

### What parallelism is actually available

**FERAL's tree parallelism delivers ~1.2× here, and the factorization is
overhead-bound, not flop-bound.** On the 210k-variable instance (`case1354`,
K = 64), factorization time against rayon threads: 25.3 s serial, 22.6 s at 2,
**21.2 s at 4**, 24.9 s at 8, 26.5 s at 14 — and back-solves get monotonically
*slower* (5.6 s → 7.8 s). Lowering feral's parallel-dispatch flop gate changes
nothing (21.3 s at 4 threads either way), so the gate is not the limiter. The
reason is granularity: the factorization does 1.22e9 flops spread over 196 561
supernodes — about 6 k flops each, ~50 MFlop/s — so per-supernode overhead
dominates and rayon's per-task cost cannot be amortised. Block-granular
parallelism (65 coarse independent tasks, one per contingency) is exactly the
shape this workload lacks, which is the structural opening.

**Evaluation is block-separable and vectorises ~20× per block.** The
contingency blocks are the same network with one branch out, so a frontend that
knows this can evaluate them together. Measured in JAX on `case1354` (one block:
2 969 variables, 6 690 rows; Jacobian 29 colors, Hessian 22 colors): a
directional derivative over **all 64 blocks at once costs 0.99 ms, 0.044× the
one-at-a-time cost**, and a Lagrangian-Hessian-vector product 1.0 ms (0.03×).
At 29 / 22 colors that is ≈ 29 ms for the whole Jacobian and ≈ 22 ms for the
whole Hessian, against the `.nl` path's measured 107 ms and 463 ms per
evaluation — roughly 4× and 20×. Part of that gap is JAX/XLA against the ASL
evaluator rather than structure; the structural part is that identical blocks
share one sparsity pattern and one coloring, so they can be evaluated in a
single vectorised pass. Evaluation is 48% of this solve.

## Phase 5b prototype: block-parallel factorization, measured

`benchmarks/kkt_scaling/blockfac` (+ `export_kkt_blocks.py`) factors a real
dumped KKT both ways and checks the inertias agree. Input: the 210k-variable
SCOPF (`case1354_pegase`, K = 64) dumped at iteration 10 — n = 883 551, 4.09 M
lower-triangle nonzeros. The blocks are recovered **from the matrix alone**: the
shared columns have degree > 100 where everything else is under 72, and
removing them leaves exactly 65 components of ~13 600, matching the model.

Steady state (pattern cached, as in an IPM), 14 threads:

| path | time |
|---|---|
| monolithic refactorization, best ordering (`amf`/`auto`, feral's own parallelism) | **0.34 s** |
| block refactorization alone, parallel | 0.035 s |
| block factor + Schur, feral's native `factorize_multifrontal_with_schur` | **0.047 s** + 0.002 s border → **7.1×** |
| block factor + Schur, dense multi-RHS solves | 0.40 s → 0.7× (*slower*) |
| **back-solve**, monolithic | 0.049 s |
| **back-solve**, block-parallel (two block solves + the border system) | **0.015 s → 3.3×** |
| **whole KKT solve** (factor + border + back-solve) | 0.389 s → **0.064 s = 6.1×** |

`inertia(K) = Σ_k inertia(A_kk) + inertia(S)` holds exactly — (458 643, 424 908,
0) both ways — so the block path reproduces the pivot information the IPM's
inertia correction runs on. The block solve is also at least as accurate:
relative residual 1.53e-14 against the monolithic solve's 1.94e-14 and the
2.05e-14 of the solution pounce itself produced for this system (both dumped
with the matrix).

Three things this pins down:

1. **Block granularity is what feral's elimination tree cannot reach.** The
   same factorization is 10× faster when it is 65 coarse tasks (0.035 s against
   0.34 s), while feral's own tree parallelism on the whole matrix tops out at
   1.2×. The work was always embarrassingly parallel; the tasks were too small.
2. **How the Schur complement is formed decides whether any of that survives.**
   Forming it with dense multi-RHS solves costs 0.37 s and gives the whole gain
   back. It is an implementation artefact: each border column has ~2 nonzeros
   in a block (mean 1.98, max 2), yet a dense solve moves a 13 589 × 259 buffer
   per block. feral's native path forms the same complement inside the
   factorization and costs 0.012 s on top of the block factors.
3. **The monolithic baseline must be the best ordering.** `metis` needs 2.6–3.3 s
   on this matrix against `amf`'s 0.34 s, and an unfair baseline turns 7× into
   a reported 10–200×.

Scope: one matrix, one iterate, one machine, 14 threads. The linear algebra is
~46% of this solve (factorization 35%, back-solve 11%), so 6.1× on it is ~1.8×
end to end before any evaluation work. The prototype factors each block twice
(once through `Solver` for the back-solve, once through the native Schur call);
a real implementation keeps the factors the native call already returns, so the
factor figure is if anything pessimistic. What it establishes is that the arrowhead's block
parallelism is real and reachable with feral as it ships.

## Phase 5b, implemented: the block path inside pounce

`pounce_feral::FeralBlockSolver` + `kkt::BlockAugSystemSolver`, engaged either
by a caller-supplied partition (`IpoptApplication::set_kkt_block_structure`,
`Problem.set_kkt_block_structure` in Python) or by `kkt_block_detect=yes`,
which looks for the structure in the assembled KKT. End to end on the
`case118_ieee` arrowhead family, same binary, `.nl` through the CLI:

| K | blocks / border | factorization std → block | wall | iterations | objective |
|---|---|---|---|---|---|
| 32 | 33 / 18 | 1.077 → 0.271 s (**4.0×**) | 2.79 → 2.07 s | 43 = 43 | identical |
| 64 | 65 / 18 | 2.290 → 0.592 s (**3.9×**) | 6.01 → 4.50 s | 48 = 48 | identical |
| 128 | 129 / 18 | 7.110 → 1.953 s (**3.6×**) | 20.05 → 15.85 s | 81 = 81 | identical |

Identical iteration counts and objectives are the correctness evidence: the
block path reproduces the monolithic *trajectory*, not merely a comparable
answer. The end-to-end factor of ~1.3× is exactly what the attribution above
predicts — factorization is ~35% of these solves, and 3.9× on that share is
1.35×. The rest is evaluation (Phase 5a), which pounce does not own.

Three things the implementation had to get right, each found by a measurement:

1. **Unscale the Schur block.** feral factors `D·A·D`, so the complement comes
   back scaled. Inertia is a congruence invariant, so every inertia assertion
   passes either way and only a solve notices.
2. **Surface `Singular`, do not fall back on it.** Unlike the Schur arm — where
   a rank-deficient *eliminated* block is outside δ_c's reach — every dual row
   here lives inside a block, so `perturb_for_singular` fixes it on the next
   factor. A KKT whose structurally empty constraint rows carry only δ_c reads
   as singular on the monolithic path too (2 240 such rows on a 64-contingency
   SCOPF), and that path reports rather than gives up. Falling back instead
   cost the whole feature on the first end-to-end run.
3. **Fold tiny components into a block, never into the border.** Those 2 240
   empty rows are isolated, so each is its own "block"; putting them in the
   border would grow it from 18 to thousands and make its dense complement cost
   more than the factorization it replaces.

**Detection's limits, measured.** It ranks columns by degree and peels: it finds
the structure when the shared columns stand out (`case118_ieee`: degree 130
against a median of 6) and **fails to on `case1354_pegase`**, where the shared
generator columns have degree ~50, no more than many ordinary columns — there it
reports 581 blocks with a 7 301-column border, which the border-size guard
rejects, and the solve proceeds monolithically. That is the argument for
*declared* structure: the model knows which columns are shared; a degree
heuristic only sometimes does. The Phase 1 mapper (NLP-space labels → KKT
indices) is what turns that into the usable path, and it is not built yet.

## Verdict for the structured-KKT plan

Across both families the attribution is the same: **the only superlinear row is
fill.** Iterations are flat, non-factorization work is linear, and delayed
pivoting grows at most slightly faster than the problem.

On the family that showed the quadratic, that fill is removed by the elimination
the model's geometry calls for — a *two-dimensional* nested dissection of the
space-by-time mesh, which `metis` computes and `auto_race` selects — and not by
the one-dimensional temporal decomposition the plan proposes, which was
measured 20–40× worse as an ordering and 3× worse per iteration as a Schur
split, because the linking set between time points is the whole 775-state
slice. The plan's premise that linking variables are few relative to block size
does not hold for this model.

So, against the Phase 0 gate:

1. **Gas does not justify a block-structured KKT solver** (Phases 3–4) as its
   motivating case. The decomposable row is present, but the decomposition that
   works is already available, and near its asymptotic cost at the top of the
   measured range. A structure-derived network × time ordering, built with
   the spatial cells, time spans and primal-dual pairs it turned out to need,
   matched `metis` at 56k and lost by 44–54% at 112k–224k end to end.
2. **The actionable finding is routing.** The default `auto` never selects
   nested dissection and costs 8× at 224k variables; `auto_race` gets it right
   on this family. Making that choice automatic for space-by-time models —
   through FERAL's routing, or through declared structure that says "this is a
   mesh" — is a default-path change and needs the fixture sweep on both legs
   plus `benchmarks/qp`, where `metis` is known to lose.
3. **The arrowhead's block parallelism is real: 7.1× on the factorization,
   prototyped and inertia-checked** (Phase 5b above), on top of a
   block-separable evaluation that is the larger half of the solve. This is
   the capability's case, and it is now measured rather than assumed.
4. **The arrowhead's ordering is already right (Phase 0c).** On N-1 SCOPF up to 210k variables
   the generic orderings reach linear fill with bounded fronts, and the
   remaining cost is split between block-separable function evaluation (~half)
   and block-separable factorization (~a third). Declared structure's case
   there is running the blocks in parallel (plan Phase 5), not reordering them.

Not covered here: periodic (ring) terminal constraints, sizes above 224k in the
refinement family, and the 72 h / 20 min 503k case, which the earlier study
solved in 1 h 30 min with `metis`.
