# GLOBALLib — proven-optimum global benchmark (`pounce-global`)

An **external, `.nl`-driven** benchmark for the spatial branch-and-bound global
solver, complementing the self-contained synthetic suite in
[`../global/`](../global/README.md). Where that suite hand-builds classic
functions in Rust, this one drives real AMPL `.nl` files through the same CLI a
user hits — `pounce <model>.nl solver_selection=global` — and checks the
**certified** objective against a *proven* global optimum.

## What it is

- **Problems:** the [GLOBALLib][globallib] collection (Floudas/GAMS nonconvex
  NLP & QP test set, 2–9 variables, finite box bounds — the natural shape for
  spatial B&B), as redistributed in AMPL `.mod` form by
  [`ampl/global-optimization`][ampl-go].
- **Subset:** only the models that have a **proven** global optimum
  (`=opt=`) in MINLPLib's [`minlplib.solu`][solu] — so every check is against
  ground truth, not a best-known heuristic value. That is **104** models
  (1 GLOBALLib model, `nemhaus`, has no AMPL `.mod` and is excluded).
- **Ground truth:** [`optima.txt`](optima.txt) — one `<stem> <objective>` per
  line, copied verbatim from the `=opt=` entries of `minlplib.solu`.

## How the `.nl` are produced (reproducible)

The `.nl` files live in the bench-data tree (Dropbox), next to every other
supplied tier (`lp/nl`, `qp/nl`, `vanderbei/nl`, …), at
`$POUNCE_BENCH_DATA/globallib/nl/`. They are *generated*, not committed:

```sh
# clones ampl/global-optimization, runs AMPL `write` on each proven-optimum
# model, drops <stem>.nl into the bench-data globallib/nl dir
benchmarks/globallib/translate.sh        # needs `ampl` on PATH
# or via the suite Makefile:
make -C benchmarks globallib-translate
```

`.mod → .nl` is done by AMPL itself (`model x.mod; option auxfiles rc; write gx;`),
the same translation the `mittelmann` tier uses.

## Running

```sh
make -C benchmarks globallib-run                    # 30s/problem cap (default)
make -C benchmarks globallib-run GLOBALLIB_TIMEOUT=120
# or directly:
python3 benchmarks/globallib/run_globallib.py --timeout 30 --out report.json
python3 benchmarks/globallib/run_globallib.py ex2_1_1 ex8_1_1   # a few by name
python3 benchmarks/globallib/run_globallib.py --max-vars 4      # small only
```

The harness runs each model, parses the solver's certificate line
(`obj=… gap=… nodes=…`), and classifies the run:

| verdict | meaning |
|---|---|
| **OK** | `Global optimum found` **and** certified obj matches the known optimum (abs-tol `1e-6` **or** rel-tol `1e-4`) |
| **WRONG** | solver certified optimality at a value that disagrees with the proven optimum — a **correctness bug** (none observed) |
| **TIMEOUT** | hit the per-problem wall-clock cap before closing the gap |
| **other** | node-limit / infeasible / crash |

The OK check is **combined absolute+relative** (`--atol 1e-6`, `--tol 1e-4`): a
proven optimum of *exactly* 0 (common here — `ex14_1_*`, `ex9_2_3`) makes a pure
*relative* metric explode for a certified `~1e-7` that is in fact correct to
~1e-6 absolute. Accepting on either bound stops those near-zero optima from being
mis-flagged as `WRONG`.

The distinction that matters: a `WRONG` would mean the solver *claimed* a
certified global optimum that is provably not one — the only true failure. A
`TIMEOUT`/node-limit means "didn't finish in budget," a performance limit, not
a soundness bug. The global solver has no node/time CLI flag yet, so the budget
is enforced by an external process timeout.

## Notes on coverage

- The global CLI path **caps unbounded variables to ±1e6** and warns; GLOBALLib
  models are bounded, so this rarely triggers here.
- Expect honest performance limits at this stage: **concave** quadratics
  (e.g. `ex2_1_*`, negative-definite Hessian) and **high-degree univariate
  polynomials** (e.g. `ex4_1_2`, degree 16) are the hardest cases for the
  McCormick/αBB relaxations and tend to time out — exactly the regime a
  benchmark should expose. The headline correctness claim is that **no run
  certifies a wrong optimum**.

## Results

<!-- RESULTS: regenerate with `make -C benchmarks globallib-rerun`; see pounce.json -->
Latest run — Apple M-series, `--release`, **30 s/problem** cap, abs-tol `1e-6` /
rel-tol `1e-4`, 104 proven-optimum models:

| outcome | count | meaning |
|---|--:|---|
| **certified correct global optimum** | **59** | matched the known optimum |
| **wrong certified value** | **0** | no soundness failure of this kind |
| **false "infeasible"** | **0** | no feasible problem certified infeasible |
| timed out (30 s) | 45 | performance limit, not a correctness failure |

**Headline (good):** every run that returned a *value* certified the correct
optimum — **0 wrong objectives, 0 false-infeasible**. The remaining 45 are pure
performance timeouts, not soundness failures.

### Fix: the `.nl` infinity-sentinel false-infeasible (4 problems recovered)

An earlier run flagged **4** problems (`dispatch`, `ex2_1_10`, `ex3_1_1`,
`ex7_2_1`) as certified *infeasible* despite each having a proven finite optimum.
Root cause: AMPL writes a *missing* constraint bound as the sentinel `±1e19`
(not an IEEE infinity), and the global CLI was passing that sentinel straight
through as a **finite** bound. `pounce-global` treats a finite bound as an
*active* side, so a genuinely one-sided constraint (`g ≤ ub`) became spuriously
two-sided (`1e19 ≥ g ≤ ub`); at GLOBALLib scale the bilinear relaxation terms
(~1e7) against a 1e19 wall make the relaxed region read as empty. Fix:
`nl_constraint_bound()` in `crates/pounce-cli/src/dispatch.rs` maps `±1e19 → ±∞`
before the constraints reach the relaxation (unit-tested). All four now certify
their proven optima (`ex3_1_1 → 7049.249`, `dispatch → 3155.288`,
`ex2_1_10 → 49318.018`, `ex7_2_1 → 1227.226`) when given enough budget. Within
the 30 s screen three of them now close (`dispatch` 1.9 s, `ex2_1_10` 11.0 s,
`ex7_2_1` 13.5 s); only `ex3_1_1` still exceeds it (closes in ~113 s) and shows
as TIMEOUT above, with the certified value correct once it finishes.

### Fixed: the `chance` false-infeasible (near-singular envelope tangent)

`chance` (proven optimum `29.894`, solved by both the NLP filter-IPM and BARON)
used to be certified *infeasible* at the root node. The first hypothesis was an
FBBT reverse-propagation bug, but instrumenting the run cleared FBBT: at the root
box it correctly tightens to `[0,1]⁴` and never prunes a box containing the
optimum. The real fault was one level down, in the **relaxation LP**.

The `√(Σ aᵢxᵢ²)` constraint relaxes through the `sqrt` envelope, whose concave
**over**-cuts are tangent lines `df = 0.5/√t`. At the singular endpoint `t = 0`
that slope is ≈`5e149` — a *valid* but astronomical cut. Feeding it into the
relaxation LP's constraint matrix wrecks the conditioning, and the HSDE conic IPM
responds by emitting a spurious Farkas infeasibility certificate (the tell: it
reported `obj ≈ 29.636`, right next to the true optimum, before declaring the LP
empty). So a perfectly feasible relaxation read as infeasible and the root node
was pruned.

Fix (in `crates/pounce-global/src/relax.rs`): a `cut_is_finite` guard with
`MAX_CUT_MAGNITUDE = 1e8` drops any cut whose slope or intercept exceeds that
bound, in both `emit_univariate` and `sandwich_cuts`. Dropping a cut only
*loosens* the relaxation, so it is always sound — spatial branching re-tightens
the bound on later, better-conditioned subboxes. `chance` now certifies
`29.894378` in **3 nodes / 0.11 s**, and the better conditioning also flipped
`ex14_1_2` from TIMEOUT to OK. Regression-tested end-to-end
(`chance_constraint_is_not_falsely_infeasible`, `drops_astronomical_sqrt_tangent`
in `relax.rs`); the full GLOBALLib sweep shows zero OK→worse regressions.

### Fixed: ill-conditioned relaxation LPs discarded their bound (+11 net)

A cluster of division+log models — the **Wilson VLE consistency** set
`ex14_2_*` — timed out despite the relaxation *reaching* the correct objective
(`~1e-8`, the proven optimum `0`) at the root. The cause was one level down again,
in the conic IPM that solves the relaxation. These relaxations are severely
ill-scaled: the McCormick **division** columns `w = a/c` with a denominator box
bottoming out near `0` carry bounds up to `~1.2e6`, and the `ln` envelope tangents
at `x ≈ 1e-6` have slope `1/x ≈ 1e6`, so the inequality matrix spans
`|G| ∈ [1.8e-7, 1e6]` (condition number `~1e12`). On such data the HSDE driver's
embedded KKT factorization breaks down and returns `NumericalFailure` — and
`process_node` then has no choice but to fall back to the *inherited* parent
bound (`-∞` at the root). With no finite lower bound the node can never be pruned,
so the search runs to the wall-clock cap even though it sat on the optimum the
whole time.

The HSDE driver deliberately skips Ruiz pre-scaling (it conditions itself through
per-cone NT scaling, like Clarabel/ECOS, and Ruiz composes badly with presolve on
the well-scaled NETLIB LPs). The fix keeps that happy path intact and adds a
**fallback**: when an HSDE solve returns `NumericalFailure` *and* equilibration is
enabled (the default), `solve_qp_ipm` retries the solve **once** with Ruiz
equilibration and accepts the result if it converges
(`crates/pounce-convex/src/ipm.rs`). This is sound by construction — the retry
only runs after the un-equilibrated solve has already failed, so there is no
well-conditioned case left to regress; equilibration either recovers a usable
bound or fails the same way and the original result stands.

Net effect on the 30 s screen: **48 → 59 OK** (`+11`). Twelve models flipped
TIMEOUT→OK — the eight solvable `ex14_2_*` (each now **1–3 nodes**, e.g.
`ex14_2_1` in 0.29 s), plus `ex14_1_5`, `ex2_1_7`, `ex5_4_2`, and `ex7_3_2`. One
model, `ex9_2_6`, crossed the screen the other way (OK→TIMEOUT) — *not* a
correctness change: it still certifies its proven optimum `-1.0` (gap 0), but the
recovered bound reorders the best-first frontier and it now closes in ~41 s
instead of under 30 s (79 → 209 nodes — the familiar "a different valid bound
grows a different tree" anomaly of spatial B&B). `ex14_2_4` is the one `ex14_2_*`
that still times out: its equilibrated retry also fails to certify, a harder
conditioning case left for future work. Zero models certified a wrong value.

### Timing context vs BARON (true global solver peer)

To put pounce's solve times in context we cross-check against **BARON**, the
canonical spatial-branch-and-bound global solver, via AMPL's bundled build.
That build is **demo-limited** (≤10 variables / constraints for nonlinear
models), so it covers a 33-problem subset — but on that subset it is the gold
standard, and **every BARON optimum matches the proven value** (independent
confirmation of our ground truth). The BARON sweep is committed as
[`baron_sweep.tsv`](baron_sweep.tsv); reproduce the table with
`python3 compare_baron.py` (defaults to the committed `pounce.json` +
`baron_sweep.tsv`):

| | BARON (demo) | pounce-global |
|---|---|---|
| certify proven optimum (33-subset) | 33/33 | 27/33 within 30 s |
| median wall | **0.061 s** | 0.434 s |
| max wall | 1.91 s | 21.06 s |

So where both close the gap they agree to ~7 digits; pounce is currently **~1–2
orders of magnitude slower** and times out on the harder ~1/5 of the subset.
BARON is a mature commercial solver — the gap is expected and the useful read is
the *shape*: pounce is competitive on the small/well-conditioned cases and loses
ground exactly where its relaxations are loosest.

**Performance (the 45 timeouts):** the global solver has no node/time CLI flag,
so 30 s is a deliberately tight screen. The dominant slow cases are concave
quadratics (`ex2_1_*`, negative-definite Hessian → loose secant relaxation) and
high-degree polynomials (`ex4_1_2`, degree 16). A longer cap recovers more (e.g.
`ex3_1_1` closes in ~113 s), but the McCormick/αBB relaxation blow-up on these
shapes is the real lever, not wall clock. Re-run with `GLOBALLIB_TIMEOUT=120` to
measure the budget sensitivity.

[globallib]: https://www.minlplib.org/
[ampl-go]: https://github.com/ampl/global-optimization
[solu]: https://www.minlplib.org/minlplib.solu
