VERDICT: improved

# gh #611 — warm-start benchmark coverage

Work branch `claude/pounce-issue-611-loop`, head **`fe6c658`**, based on
`origin/main` at **`eeca47f922372834a7e8d925ce7fe7b6160c8c7b`**. Every
number below was taken on that SHA with a `maturin develop --release`
build, on a 4-core Linux 6.18.5 container. `origin/main` was re-fetched
at the end and had not moved, so the merge-up was a clean no-op and the
expected `CHANGELOG.md` conflict never arose.

**No Rust changed.** `git diff origin/main -- crates/ Cargo.toml Cargo.lock`
is empty, and `cargo fmt --all` produced no diff. This is benchmark and
documentation work only — so the solver behaviour measured here *is*
`main@eeca47f9`.

> **On this document.** Two sessions worked this branch concurrently and
> each produced a `LOOP_REPORT.md`. This is the union of both, resolved
> in `claude/loop-report-611`. The collision was worth more than it
> cost: the two ran the full sweep **independently, in separate
> containers**, and agree *exactly* — arm for arm, across all 15 arms
> and 1335 steps — on every noise-free column (iterations, evaluations,
> speedups, `bad`, `better`, max-KKT). The only divergence is wall time
> (`cold-ipm` 11.86 s against 12.31 s; `cold-sqp` 208.21 against
> 233.36), which is the 10–30% between-run drift the suite already
> documents and exactly why it tells the reader to read ratios rather
> than milliseconds. Wall times quoted below are the committed run's.

---

## 1. The point of the exercise

The suite this issue asks to extend was already substantial — 17
families, 10 arms, a real measurement protocol. What it could not do
was produce a result *against* warm starting, because every family in
it had been chosen by earlier warm-start work. That is the failure this
repo has already paid for once, and it is the risk the whole deliverable
carries.

So the centre of the work is the falsification arm, and the headline
results are negative. They are in §3 before anything favourable.

---

## 2. What was built

Eleven non-merge commits plus two merges, all referencing gh #611.
`git diff --stat origin/main...HEAD`: 22 files, +7619 / −38 — of which
**+3156 / −38 across 20 files is hand-written**; the remaining +4463 in
2 files is the generated composite report and its JSON.

```
fe6c658  Merge: take the pushed composite; two independent runs agree
699ccbe  warmstart: regenerate the composite on the fixed resistive_network
51a6881  warmstart: the generated composite report
b6408aa  Merge branch 'claude/pounce-issue-611-loop'
232fa79  warmstart: the generated composite report and its raw JSON
ec861f6  warmstart: the documented -hom reproduction crashed the report
73eec88  warmstart: resistive_network was infeasible at V >= 120
961487f  warmstart: warn when a narrowed --arms drops the correctness reference
ad07ac2  warmstart: composite report, profiles, and docs wired up
6fa8c2c  warmstart: changed-structure arms — horizon shift and mesh prolongation
f516f45  warmstart: Ipopt through cyipopt, and one KKT residual for both
db6c257  warmstart: the five initialization arms gh #611 was missing
c048d35  warmstart: families outside the ones warm-start work chose
```

The two merges and the duplicated composite commits are the two
concurrent sessions converging; `fe6c658` resolves the composite to a
single committed version. **No force-push was used at any point.**

| area | file | what |
|---|---|---|
| families | `families/global_nonconvex.py` | `rastrigin_drift`, `rastrigin_scatter` — the falsification arm |
| | `families/pde.py` | `elliptic_control_{40,80,160,600}` — 1-D Poisson control, tridiagonal, conditioning ~ h⁻² |
| | `families/network.py` | `resistive_network_{60,120,800}` — graph-incidence sparsity, quartic loss (not a QP) |
| | `families/scaling.py` | `badly_scaled_qp` — 10⁸ Hessian conditioning, 10³ row scaling |
| arms | `adapters/base.py`, `adapters/pounce_adapter.py` | `warm-ipm-primal`, `warm-ipm-norecenter`, `cold-ipm-lsq`, `race-fixed`, `race-halving` |
| metric | `spec.py` | `StepResult.init_time` — initialization overhead, separate from `solve_time` |
| external | `adapters/ipopt_adapter.py` | Ipopt via cyipopt, on the *same* callback object |
| fairness | `kkt.py` | one KKT residual (dual / primal / complementarity) computed by the harness for both solvers |
| structure | `transfer.py` | horizon shift (`WarmStart.reindex`), mesh prolongation (`WarmStart.transfer`) |
| report | `composite.py` | composite report + performance/data profiles, every table generated from raw JSON |

Families 17 → 27. Arms 10 → 15. All 27 families pass
`python -m warmstart.selftest` (central-difference gradient / Jacobian /
Hessian checks, plus QP-form re-derivation for the four new families
claiming `quadratic`).

### Exact commands for everything below

```
cd benchmarks
python -m warmstart.run       --scales all --arms all --tier default --out warmstart/results.json
python -m warmstart.run       --solver ipopt --scales all --arms all --tier default \
                              --out warmstart/results-ipopt.json
python -m warmstart.transfer  --experiment all --horizons 10,20,40 --meshes 20,40,80 \
                              --out warmstart/transfer.json
python -m warmstart.composite --results warmstart/results.json \
    --ipopt warmstart/results-ipopt.json --transfer warmstart/transfer.json \
    -o ../dev-notes/warm-start-611-composite.md \
    --json-out ../dev-notes/warm-start-611-composite.json
```

or `make -C benchmarks warmstart-all`. 66 family × scale cells, 1335
steps per full arm.

---

## 3. The negative results

### 3.1 Warm starting is faster and wrong on unrelated instances

`rastrigin_scatter` is the issue's "unrelated global/nonconvex cases
where continuation should not be expected to help": θ is redrawn from a
fixed seed each step, so consecutive instances share nothing but a
shape. Bad steps out of 20, per scale:

| arm | tiny | small | large | iterations (t/s/l) |
|---|--:|--:|--:|---|
| `cold-ipm` | 0 | **0** | **0** | 116 / 231 / 237 |
| `warm-ipm` | 0 | **17** | **9** | 86 / 199 / 206 |
| `warm-ipm-primal` | 0 | 17 | 9 | 96 / 208 / 213 |
| `warm-ipm-norecenter` | 0 | 17 | 9 | 86 / 199 / 206 |
| `cold-ipm-lsq` | 0 | 0 | 0 | 116 / 231 / 237 |
| `race-fixed` | 0 | 5 | 5 | 100 / 100 / 104 |
| `race-halving` | 8 | 11 | 10 | 100 / 105 / 121 |
| `cold-sqp` | 0 | 20 | 20 | 60 / 6230 / 9025 |
| `warm-sqp` | 10 | 20 | 20 | 86 / 7048 / 9017 |

**The warm arm is faster and wrong.** 199 iterations against cold's
231, and 17 of 20 steps land on a worse optimum. Nothing fails — every
one of those steps returns `Solve_Succeeded` with a small KKT residual.
It is only visible because the harness scores against a reference
objective rather than a status code.

**Ipopt reproduces it exactly.** Same family, same scales, an
independent solver sharing only the Python callbacks:

| arm | solver | tiny | small | large |
|---|---|--:|--:|--:|
| `warm-ipm` | pounce | 0 | **17** | **9** |
| `warm-ipm` | ipopt | 0 | **17** | **9** |

Identical counts. This is a property of warm starting on this workload,
not of pounce's implementation.

The scale axis behaves as designed and one part of it is worth noting:
`small` (17 bad) is **worse** than `large` (9 bad). At `large` the draws
are so scattered that the seed is effectively random and globalization
often recovers; at `small` the seed is close enough to look plausible
and lands in the adjacent wrong basin. Non-monotone, and the reason the
suite sweeps three step sizes rather than picking one.

`rastrigin_drift` sweeps the same landscape smoothly. Even at `tiny`
— per-coordinate step 0.03 against a basin spacing of 1 — `warm-ipm`
loses 2 of 20 steps where `cold-ipm` loses none.

### 3.2 Start racing costs enormously and is wrong more often

Over the whole sweep (1335 steps per arm):

| arm | iterations | evaluations | solve s | **init s** | bad |
|---|--:|--:|--:|--:|--:|
| `cold-ipm` | 15640 | 91 779 | 12.3 | 0.0 | 0 |
| `warm-ipm` | 5825 | 40 090 | 6.2 | 0.0 | 32 |
| `race-fixed` | 12513 | **630 232** | 10.7 | **1762.0** | 35 |
| `race-halving` | 13436 | **525 176** | 12.8 | **689.6** | 75 |

`race-fixed` spends **1762 seconds choosing a start for 10.7 seconds of
solving** — 165× — and burns 6.9× the evaluations of a plain cold solve.
On the evaluation performance profile it scores **0.00 at every τ**,
including τ = 4: it is never within 4× of the best arm on any instance.
`race-halving` is half the cost and twice as wrong (75 bad against 35),
which matches the concern already documented on
`race_starts(policy="halving")`.

Both raced with **the family's own cold start as candidate 0**. That was
deliberate: without it a loss could be blamed on never having sampled
the point `cold-ipm` uses. With it, the ranking rule discarded a winning
start it was already holding.

**Caveat that limits how far this generalizes.** On the Rastrigin
families the cold start is the origin and θ is centred there, so
`cold-ipm` begins inside the correct basin by construction — an
unusually good cold start. The defensible claim is the narrow one: at
`_RACE_STARTS = 8` / `_RACE_ITERS = 10`, a 10-iteration ranking over a
10-dimensional multi-basin landscape is noise with respect to the
eventual optimum and pays ~7× the evaluations for it. Not "racing is
useless".

### 3.3 The horizon shift does not pay

Iterations over steps 1+:

| family | cold | carry (unshifted) | shift by one stage |
|---|--:|--:|--:|
| `mpc_horizon_10` | 94 | **23** | 87 |
| `mpc_horizon_20` | 103 | **48** | 92 |
| `mpc_horizon_40` | 125 | **59** | 114 |
| `nmpc_vanderpol` (genuinely closed-loop) | 136 | **71** | 73 |

The suite's own notes said carrying the previous solution unshifted
"understates what an MPC implementation would do". Measured, it does
not. On the closed-loop family the shift is a wash (73 vs 71); on
`mpc_horizon_*` it is much worse, because that family walks its initial
state around a *circle* — consecutive problems are a rotation of one
another and a time shift is not the operation relating them. The
unseeded final stage costs more than the shift saves.

Both families were run precisely so the result would be attributable.
Running only the closed-loop one would have hidden it; running only the
rotating one would have blamed the shift for the family's shape.

### 3.4 A prediction of mine that the data refuted

For mesh prolongation I derived the mesh-dependent multiplier scaling
(PDE rows ~ h³, pin rows and bound multipliers ~ h) and predicted the
unscaled control would backfire. Iterations on the fine mesh:

| arm | 20→41 | 40→81 | 80→161 |
|---|--:|--:|--:|
| `cold` | 14 | 15 | 16 |
| `prolong-primal` | 13 | 10 | 12 |
| `prolong-dual` (scaled) | **6** | **4** | **3** |
| `prolong-dual-raw` (unscaled) | 7 | 5 | **3** |

Unscaled dual prolongation still cuts 16 → 3, and at the finest mesh it
*ties* the scaled version. Carrying the duals is worth far more than
scaling them correctly is. The `-raw` arm stays in the suite as the
control that says so.

### 3.5 The first full sweep found one of my own families broken

`resistive_network_120` returned `Infeasible_Problem_Detected` on **all
60 of its steps** while `resistive_network_60` was clean. I had
validated at V = 60 and generalized. Three defects, each found only by
measuring:

1. The demand was a sinusoid over the *whole ring*, so induced flow grew
   with the ring (≈ `D_AMP·V/2π`: 3.8 at V = 60, 7.6 at V = 120, against
   capacity 1.2). V = 60 passed only because the chords shortened paths
   enough. Fixed with a fixed spatial *period*, so transport is local.
2. A pure sinusoid loads every cell identically, so capacity admits all
   or none — the instance jumped from "nothing active" at `f_max = 0.70`
   to "infeasible everywhere" at 0.68. Fixed with a coprime envelope.
3. Ring-only nodes had degree 2, putting the feasibility floor at
   `max|d|/2`, which sits *above* the uncapped p99 flow. No capacity was
   both feasible and congesting. Fixed by giving every node a chord
   (minimum degree 4, floor 0.125). E goes 1.5V → 2V.

`_F_MAX = 0.40` is now read off the measured distribution (p99 = 0.204,
max = 0.781, per-step max swinging 0.206 → 0.781), verified feasible on
every step of every scale at V = 60, 120, 800.

**I also retracted a claim I had written into that family's docstring.**
It said the active bounds "move around the graph as θ rotates". They
barely do: nothing is active at `tiny` or `small`, and at `large` the
count runs 0–4 (V = 60, 120) and 0–2 (V = 800). The docstring now says
that, explains why the feasible window is narrow, and tells readers to
use the family for its sparsity pattern, non-constant Hessian and size —
not as an active-set probe.

**The entire sweep was re-run on the fixed family.** Every number in this
report is from that second run.

The stale rows were **not confined to their own family**, which is worth
recording: an infeasible instance is a failed step for every arm that
meets it, so those 60 per-arm failures were sitting in the sweep-wide
`bad` column and made four arms look broken when they were clean.

| arm | bad, pre-fix sweep | bad, after |
|---|--:|--:|
| `cold-ipm` | 60 | **0** |
| `cold-ipm-lsq` | 60 | **0** |
| `pred-ipm` | 60 | **0** |
| `predcorr-ipm` | 60 | **0** |
| `warm-ipm` | 92 | 32 |
| `cold-sqp` | 246 | 186 |
| `warm-sqp` | 267 | 207 |

The headline speedup moved the *unfavourable* way — `warm-ipm` 3.23× →
**3.05×** — because the removed steps were ones cold could not solve
either. Publishing the pre-fix sweep would have overstated the speedup
*and* libelled four arms as failing.

### 3.6 The reproduction command this branch documents had never worked

`dev-notes/warm-start-benchmark.md` tells the reader to reproduce the
`elliptic_control_*` finding (item 9 in §7) with

```
cd benchmarks && python -m warmstart.run --families elliptic_control_40 \
    --scales small --arms cold-sqp,cold-sqp-hom
```

Running it as documented raises `KeyError: 'warm-sqp'` — **after** every
solve has already run, at the point of writing results out.
`warmstart/report.py` selects its homotopy section on `cold-sqp-hom`
being present, then reads all four of `cold-sqp`, `cold-sqp-hom`,
`warm-sqp` and `warm-sqp-hom` unconditionally, so any narrowed `--arms`
that drops the warm twins dies there.

`report.py` is `main`'s file, untouched by the rest of this branch, so
the bug is pre-existing — but this branch is what documents a command
that triggers it. Fixed in `ec861f6`: the warm pair is looked up
defensively and renders as `—` when absent, and the cold pair (what the
section is actually about) is what gates it. Re-rendering the full
66-cell sweep from its own `results.json` gives a **byte-identical**
report across the change, so nothing that already worked moved. The
documented command now exits 0, and the narrowed-`--arms` correctness
warning added by `961487f` fires as designed.

This also means the fix is *not* in the pre-`fe6c658` history that the
concurrent session's report describes.

---

## 4. The positive results

Warm starting works on the workload it was built for, and two solvers
agree about it.

| arm | solver | iterations | evaluations | solve s | bad |
|---|---|--:|--:|--:|--:|
| `cold-ipm` | pounce | 15640 | 91 779 | 12.3 | 0 |
| `cold-ipm` | ipopt | 15739 | 88 411 | 10.2 | 0 |
| `warm-ipm` | pounce | **5825** | 40 090 | 6.2 | 32 |
| `warm-ipm` | ipopt | **5579** | 35 963 | 4.5 | 35 |
| `warm-ipm-primal` | pounce | 9056 | 67 238 | 8.5 | 30 |
| `warm-ipm-primal` | ipopt | 11918 | 67 404 | 8.1 | 38 |

The two cold columns agree to 0.6%, which is the strongest available
evidence that the harness is driving both solvers equivalently — they
share only the Python callbacks.

**Carrying the duals is most of the benefit.** Seeded with the previous
primal point alone, 9056 iterations; with the complete
primal-dual-barrier state, 5825. Ipopt splits the same way and wider
(11918 vs 5579). That is the issue's separation of arm 2 from arm 3,
measured on two solvers.

**Recentering is worth ~4% overall, and it is concentrated.** 5825
(`warm-ipm`) against 6054 (`warm-ipm-norecenter`). Per family it is not
spread at all: `hanging_chain` splits 116/139, while `simplex_proj` and
`mpc_horizon_10` show no difference whatever. Verified the control works
by running the pair under both global settings — the control arm tracks
global `none` and ignores global `residual`.

**The predictor arms are the best warm arms measured.** `pred-ipm` 2204
iterations over its 540 applicable steps with **zero** bad steps, at
0.5 s total initialization overhead.

**`cold-ipm-lsq` is a wash.** 15195 iterations against `cold-ipm`'s
15640 (−2.8%) but 94 354 evaluations against 91 779 (+2.8%). The sparse
safeguarded normal step neither helps nor hurts materially on this
workload.

Performance profile by iterations (fraction of instances within τ of the
best arm; τ=1 is "how often it was best"):

| arm | τ=1 | τ=1.5 | τ=2 | τ=4 | τ=∞ |
|---|--:|--:|--:|--:|--:|
| `cold-ipm` | 0.02 | 0.19 | 0.28 | 0.66 | 1.00 |
| `warm-ipm` | **0.79** | 0.92 | 0.96 | 0.97 | 1.00 |
| `warm-ipm-primal` | 0.17 | 0.50 | 0.78 | 0.91 | 1.00 |
| `warm-ipm-norecenter` | 0.70 | 0.89 | 0.94 | 0.97 | 1.00 |
| `cold-ipm-lsq` | 0.03 | 0.21 | 0.37 | 0.66 | 1.00 |
| `race-fixed` | 0.18 | 0.33 | 0.48 | 0.75 | 1.00 |
| `race-halving` | 0.13 | 0.29 | 0.43 | 0.67 | 1.00 |

---

## 5. External solvers — what actually happened

The instructions predicted Ipopt would not be installable here. **It
was**, on the fourth attempt. The failures are recorded because the
working path is not obvious.

| # | command | result |
|---|---|---|
| 1 | `pip install cyipopt` | fail — `OSError: pkg-config was not able to find any of the requested packages ['ipopt']` |
| 2 | `pip install --only-binary=:all: cyipopt` | fail — `Could not find a version that satisfies the requirement cyipopt (from versions: none)`. No wheel exists. |
| 3 | `apt-get install -y coinor-libipopt-dev` (no prior `update`) | fail — `Failed to fetch .../openssh-client_9.6p1-3ubuntu13.15_amd64.deb 404 Not Found`; one stale index entry aborted the transaction |
| 4 | `apt-get update && apt-get install -y coinor-libipopt-dev pkg-config` | ok — `pkg-config --modversion ipopt` → 3.11.9 |
| 5 | `pip install --no-build-isolation cyipopt` | fail — `/usr/bin/ld: cannot find -llapack` / `-lblas` |
| 6 | `apt-get install -y liblapack-dev libblas-dev`, repeat 5 | ok — cyipopt 1.7.0 |

Working recipe, now in the suite README and the `warmstart-ipopt`
Makefile target:

```
apt-get install -y coinor-libipopt-dev liblapack-dev libblas-dev
pip install cython && pip install --no-build-isolation cyipopt
```

`--solver ipopt` without cyipopt exits printing exactly those lines.

Arms Ipopt does not offer are skipped with a recorded reason each, not
dropped: no active-set SQP path (4 arms), no matrix-form QP entry point
(2), sIPOPT not built (2), pounce-only options (2), racing is a pounce
API (2).

**Second external solver: not wired.** `nlopt 2.11.0` and `casadi 3.7.2`
both installed cleanly here and both were verified solving a Rosenbrock
instance. This is a "chose not to wire" result, not a "could not
install" one — see §7.2.

---

## 6. Acceptance criteria — per-criterion verdict

| # | criterion | verdict |
|---|---|---|
| 1 | Reproducible scripts and machine-readable raw results | **met, with a caveat.** Every run is a documented command with a `make` target. Caveat: per-run `results.json` inside a suite directory is gitignored by repo convention (`.gitignore:44`, and the comment above `!/benchmarks/warmstart/*.py` says these outputs "are regenerated and ignored like every other suite's"). Raw files are therefore *regenerable*, not committed; the committed machine-readable artifact is `dev-notes/warm-start-611-composite.json`, following the precedent of `BENCHMARK_REPORT.json` and `qp_three_way.json`. |
| 2 | Equal stopping criteria and documented solver-specific settings | **met for what is equal; the inequalities are documented, not removed.** Same `tol`, `constr_viol_tol`, `max_iter`; correctness judged by `kkt.py` from the returned point for both solvers rather than from either status line. Not equal, and stated in the report's settings table: linear solver (MUMPS vs pounce's own sparse LDLᵀ), Ipopt 3.11.9's age, `warm_start_*_push` set to 1e-9 (the 1e-2 default discards most of a warm start, which would have made Ipopt's warm start look useless as a settings artifact), and `mu` read from the last intermediate callback rather than a true final value. |
| 3 | Separate cold-start, repeated-solve, and changed-structure claims | **met.** Three harnesses, three report sections. Cold arms — including both racing arms — never receive the previous solution; `run.py` is repeated-solve; `transfer.py` is changed-structure. |
| 4 | Composite report wired into the existing benchmark documentation | **met.** `dev-notes/warm-start-611-composite.md` is generated by `composite.py` and linked from `dev-notes/warm-start-benchmark.md`, whose "Not done yet" list is rewritten to match reality. It stays out of `BENCHMARK_REPORT.md`, which is per-problem cold-solve rows and still has no shape for a per-sequence result. |
| 5 | Any SOTA claim scoped to the tested workload and supported by data | **met by making no such claim.** No file added or changed here contains one, and §3 is placed before §4 on purpose. |

### Problem families requested

| requested | status |
|---|---|
| MPC with horizon shifts and active-set changes | already present, plus `transfer.py`'s shift arms |
| PDE/DAE discretizations and mesh refinement | **PDE yes** (`elliptic_control_*` over a mesh sweep, plus prolongation). **DAE no** |
| large sparse process/energy models | `resistive_network_800` (n = 1600), `elliptic_control_600` (n = 1202) |
| rank-deficient or poorly scaled constraints | rank-deficient already present (3 families); `badly_scaled_qp` adds scaling |
| unrelated global/nonconvex where continuation should not help | `rastrigin_scatter`, `rastrigin_drift` |

### Metrics requested

| requested | status |
|---|---|
| time to first acceptable point | `solve_time`, gated on the harness's own convergence check |
| function / Jacobian / Hessian evaluations | `n_obj`/`n_grad`/`n_cons`/`n_jac`/`n_hess`, counted by the harness so they mean the same thing across solvers |
| final KKT residual | `kkt.py`, one definition for both solvers |
| robustness rate | `bad` / `failed` columns |
| initialization overhead | `init_time`, new |
| corrector iterations | **partial** — for `predcorr-ipm` the corrector count *is* `iters`; no separate predictor/corrector split is recorded |
| memory peak | **not measured** — see §7.1 |
| performance / data profiles | both, over iterations, evaluations and wall time |

---

## 7. What I did NOT do, and what is unmet

1. **Memory peak is not measured.** The issue asks for it. A
   `tracemalloc` figure would report the Python harness's allocations,
   not the Rust solver's, and RSS is too coarse to separate arms at
   these sizes. Dropped deliberately; not attempted.

2. **No second external solver adapter.** nlopt 2.11.0 and casadi 3.7.2
   install and run here — a choice, not a blocker. nlopt has no dual
   warm-start API so it could only populate the "previous primal only"
   arm; casadi bundles its own Ipopt, so an adapter through it would
   measure the same solver through a second binding. The seam is
   `KNOWN` in `adapters/__init__.py`.

3. **No DAE or collocation family.** `elliptic_control_*` is an elliptic
   PDE. Index-1 DAEs behave differently under a warm start because the
   algebraic variables have no dynamics carrying them.

4. **No HSL.** The Ipopt arm is MUMPS-backed; `ref/Ipopt/install-ma57`
   does not exist in this container. Wall-time numbers against it are
   against MUMPS-backed Ipopt 3.11.9, and are **not** comparable to the
   MA57 reference the rest of `benchmarks/` uses.

5. **Predictor/corrector iterations are not split.** The issue lists
   "corrector iterations" as its own metric; what is recorded is total
   iterations after the predictor step.

6. **The `large` tier was not swept.** Every number here is the
   `default` tier. `elliptic_control_600`, `resistive_network_800` and
   `mpc_horizon_{200,400,800}` are defined, self-tested, verified
   feasible and runnable with `--tier large`, but no large-tier sweep
   was run. This is the largest single gap: the sparsity and
   conditioning claims for the new families are demonstrated at
   n ≤ 322, not at n ≈ 1200–1600.

7. **The racing arms use one field size and one budget**
   (`_RACE_STARTS = 8`, `_RACE_ITERS = 10`). Sensitivity to those two
   constants is not measured; §3.2's numbers hold at that setting only.

8. **`rastrigin_*` cold starts are unusually well placed** (origin, with
   θ centred there), which strengthens every "cold beats warm" row on
   those families. Stated at §3.1/§3.2 rather than buried.

9. **A solver finding is recorded, not fixed and not filed.**
   `elliptic_control_*` does not solve under the conventional
   phase-1/phase-2 inner QP: `cold-sqp` returns
   `Maximum_Iterations_Exceeded` with **zero** completed outer
   iterations (`n_qp_solves = 1`, `n_qp_ws_changes = 88`), and
   `sqp_max_iter=5000` changes nothing because the exhausted budget is
   the inner QP's. `cold-sqp-hom` solves the identical problem in one
   outer iteration to KKT 3.1e-13.

   ```
   cd benchmarks && python -m warmstart.run --families elliptic_control_40 \
       --scales small --arms cold-sqp,cold-sqp-hom
   ```

   Independently re-verified in the second session on the regenerated
   sweep, including the `sqp_max_iter` control run directly: 500 → 5000
   changes **nothing** — same `Maximum_Iterations_Exceeded`, same
   `iters = 0`, same `n_qp_solves = 1`, same `n_qp_ws_changes = 88`.

   **It scales with the mesh, and the homotopy stops rescuing it.** The
   claim above holds at meshes 40 and 80; at 160 both arms fail:

   | family @ small | `cold-sqp` | `cold-sqp-hom` |
   |---|---|---|
   | `elliptic_control_40` | 10 of 20 steps exceed max-iter | **all 20 solve**, 1 iter, KKT 3.1e-13 |
   | `elliptic_control_80` | 18 of 20 exceed | **all 20 solve**, 1 iter, KKT 1.7e-12 |
   | `elliptic_control_160` | **20 of 20** exceed | **20 of 20 exceed** |

   So `dev-notes/warm-start-benchmark.md`'s "the homotopy variant solves
   the identical problem" is true of the problem its reproduction
   command names, and stops being true one mesh refinement further on.
   Not chased further.

   This is `main`'s behaviour on a new problem class — no Rust changed
   on this branch. Recorded in `dev-notes/warm-start-benchmark.md`, not
   filed, per the standing instruction against follow-up issues.

10. **Two in-flight PRs will move arms measured here.** #638
    (seed-rejection guards in `init/warm_start.rs`) moves `warm-ipm` and
    `warm-ipm-norecenter`; #639 (model-probe facet, `transfer()`
    evaluation timing) moves the `transfer.py` arms. Neither is merged.
    Re-measuring is `make -C benchmarks warmstart-all`; every table in
    the composite report regenerates from the raw JSON, so nothing needs
    hand-editing.

11. **The SQP arms carry high `bad` counts on the new nonconvex
    families** (e.g. `cold-sqp` 186 bad over the sweep) and I did not
    investigate them. They may be the same inner-QP issue as item 9, or
    genuine multi-basin behaviour, or a third thing. Unexamined.

---

## 8. Verification run in this session

Combined across both sessions, on the merged tree:

| check | result |
|---|---|
| `python -m warmstart.selftest` | 27 families, **all pass** (finite-difference gradient / Jacobian / Hessian, plus QP-form re-derivation) |
| `python -m pytest tests -q` (full `python/` suite) | **1082 passed, 23 skipped, 0 failed** (479 s) |
| `pytest tests/test_warm_start*.py tests/test_starts*.py tests/test_sqp_warm_start.py` | **103 passed** |
| `bash scripts/check-docs-consistency.sh` | **OK** — "46 pages, all reachable from SUMMARY.md, no dead links" |
| `bash scripts/check-release-consistency.sh` | **OK** — versions agree at 0.10.0 across `Cargo.toml` / `python` / `pyomo-pounce`; publish list 20 crates, topological; docker ARG matches |
| `cargo fmt --all` | exit 0, no diff (`git status --porcelain` empty afterwards) |
| `git diff origin/main -- crates/ Cargo.toml Cargo.lock` | **empty** |
| full 4-stage pipeline | all 66 cells completed, exit 0 at every stage, twice |
| reference arm (`cold-ipm`) failures across the final sweep | **0** |

The compiled extension was **rebuilt** in release (`cargo build -p
pounce-cli --release` + `maturin develop --release`) rather than
bypassing the staleness guard.

`cargo clippy` was **not** run: no Rust source changed on this branch,
and a full workspace clippy on 4 cores costs more than it could find.

**One note on the full pytest line, because its first run was red.** In
a bare environment,
`test_continuation.py::test_step_controller_is_what_the_jax_follower_uses`
*fails* rather than skipping: it guards with
`pytest.importorskip("pounce.jax._path")`, but `pounce/jax/__init__.py`
re-raises the missing-JAX `ModuleNotFoundError` as a chained "useful
error", which `importorskip` does not treat as a skip. Installing JAX
turns it green (32/32 in that file). This branch touches **no** file
under `python/` — `git diff --stat origin/main...HEAD -- python/` is
empty — so it is an environment artefact, not something this branch
caused. But a guard that fails instead of skipping when its optional
dependency is absent is arguably a real if minor defect in `main`. Not
filed, per the standing instruction; noted here for the parent.

The `origin/main` SHA was re-fetched before the final push and had not
moved from `eeca47f9`, so the merge-up was a no-op. The branch *did*
need two merges — but of the two concurrent sessions' own work, not of
`main`. Neither was force-pushed.
