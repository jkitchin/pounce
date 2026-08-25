# `robot_a` / `robot_b` / `robot_c`: the stall is the filter's `theta` ceiling

Follow-on to `robot-abc-per-iteration-cost.md`, for [pounce#476](https://github.com/jkitchin/pounce/issues/476).
That note closed with the mechanism "characterized, not yet diagnosed" and named
the next step: run the `POUNCE_DBG_LS` trace out past 200 iterations. This is
that trace and what it found.

**Result: all three instances converge to verified optima in 109–262 iterations
under a supported option, `theta_max_fact`, with the filter, the Armijo test and
restoration all still armed.** The earlier note's "no safe knob reproduces it"
table never tried this one.

Everything below was measured on a **4-core x86 Linux container**, POUNCE 0.9.0
at `270a050`, on the AMPL-generated `.nl` files (not the `gen_robot_nl.py`
reconstruction):

```
robot_a.nl  sha256:213d351bb9e2f65f083788bab88524bc30d2b0f42044e0a5868c66c288f50b0c
robot_b.nl  sha256:25b9ce5bf15f5f033344a6322515777da447e195899a8229fd1aea81955ee0f3
robot_c.nl  sha256:63918983b430613fbdf1057ebcb8c528aefa0f343a69ea772c851f20708cff72
```

Sizes match the published `n = 1001`, `m = 52013`, `nzJ = 196781`. **Wall-clock
numbers here are not comparable to the Mac mini reference** — this box is
roughly 4x slower per iteration and runs FERAL effectively serial. Ratios within
one table are meaningful; absolute seconds are not.

## 1. The wall

`line_search/filter_acceptor.rs:277`:

```rust
let theta_max = self.theta_max
    .unwrap_or_else(|| self.theta_max_fact * theta.max(1.0));   // theta_max_fact = 1e4
```

Any trial point with `theta_trial > theta_max` is rejected outright
(Wächter-Biegler Eqn. 21). Three facts turn that into a wall on this model:

- **The starting point is exactly feasible.** `pounce check-x0 robot_a.nl`:
  *"rows violated: 0, max violation: 0.000e0"*. So `max(1, theta_0)` collapses
  to `1` and the ceiling becomes the bare constant `1e4`, carrying no
  information about the problem.
- **`theta` is a 1-norm over rows**, not a max-norm — `ipopt_cq.rs:1677` is
  `c.asum() + dms.asum()`. With `m = 52013` and no equalities, `theta` is
  `Σ|d_i − s_i|` over 52013 terms, so it grows with `m` while the ceiling does
  not. The observed `theta = 9.997e3` is a mean per-row residual of **0.19**.
- **On this model `theta` is not measuring infeasibility at all.** Every row is
  a one-sided inequality, and `d − s > 0` *satisfies* the row however large the
  gap grows (see the comment at `ipopt_cq.rs:903`). The `inf_pr` column reads
  `0.00e+00` throughout — the original constraints are met — while the filter
  accumulates 52013 slack-lag terms and refuses steps on the total.

The trace shows the result. `POUNCE_DBG_LS=1 RUST_LOG=pounce::linesearch=debug`:

```
alpha=1.221e-4  theta=9.997e3  theta_max=1.000e4  phi=-5.500e3  n_steps=13
```

`theta` at 99.97 % of the ceiling, `alpha` cut to 1.2e-4 after 13 backtracks,
and `lg(mu)` pinned at −1.0. The line search never *fails* (it always finds
something that fits under the ceiling), so restoration never engages and the
solve grinds until `max_iter`.

## 2. The ceiling is what binds, not the iteration budget

Re-running the converged solve with the trace on:

| | |
| --- | --- |
| peak `theta` on the converged run | **9.371e7** |
| default ceiling | 1.0e4 |
| trial evaluations above the default ceiling | **149 of 251** |

The route to the optimum passes through `theta ≈ 9.4e7`, four orders of
magnitude above the default ceiling, and more than half the line search lives
up there. Under the default that route is unreachable **at any iteration
budget** — which retires "3000 iterations just wasn't enough" as an explanation.

## 3. Raising it converges all three

`feral_ordering=kahip` throughout (see §5); the ordering does not affect the
outcome, only the clock.

| `theta_max_fact` | `robot_a` |
| --- | --- |
| **1e4 (default)** | no convergence in 3000 iterations; `lg(mu)` pinned at −1.0 |
| 1e6 | `lg(mu)` reaches −5.7; still unconverged at 1000 |
| 1e7 | **Optimal**, 202 iterations |
| 1e8 | **Optimal**, 128 iterations, 52 s |

At `1e8` the barrier parameter walks the full ladder —
`−1.0 → −1.7 → −3.8 → −5.7 → −8.6 → −9.0` — which is what an IPM is supposed to
do and what neither solver managed under the default.

| instance | `theta_max_fact=1e8` | objective | IPOPT's 3000-iteration grind |
| --- | --- | --- | --- |
| `robot_a` | Optimal, 128 iters, 52 s | 1.0431952 | 8.173 (not converged) |
| `robot_b` | Optimal, 262 iters, 105 s | 2.3330990 | 15.485 (not converged) |
| `robot_c` | Optimal, 109 iters, 44 s | 1.4059756 | 29.040 (not converged) |

`pounce verify` on `robot_a`, reading the `.sol` independently of the solve:

```
max constraint violation: 0.000e0     max bound violation: 0.000e0
KKT stationarity residual: 2.582e-11  complementarity: 1.030e-9
VERDICT: VERIFIED
```

`robot_a`'s objective matches what `accept_every_trial_step=yes` reaches to 10
digits (1.0431952061), but this run keeps every safeguard. That also re-reads
the earlier note's conclusion: `accept_every_trial_step` worked because it
bypasses `theta_max` along with everything else, so the 80x was located
correctly and attributed to the wrong component.

## 4. Corrections to `robot-abc-per-iteration-cost.md`

- **§2's baseline is the wrong IPOPT.** It measures against `ipopt` + MUMPS at
  0.138 s/iter. The repo's own reference for this exact instance is MA57
  (`benchmarks/mittelmann/ipopt_ma57.json`: 3000 iterations in 170.1 s =
  **0.057 s/iter**, same Mac mini, single-threaded), and `benchmarks/README.md`
  makes MA57 the comparison convention. Against MA57 the per-iteration handicap
  is **3.9x** (5.0x for the 0.9.0 build Mittelmann ran), not 1.6x — which
  *brackets* the published 3.11x rather than contradicting it. Nothing about the
  published number is anomalous.
- **§5's "ahead of IPOPT for the first time"** (0.12 vs 0.138 s/iter after
  condensing) is ahead of *MUMPS*. Against MA57's 0.057 it is still ~2.3x
  behind, so condensing should not be scoped as "beat IPOPT on m ≫ n".
- **§4b's "none of the safe knobs reproduce it"** is false: `theta_max_fact`
  does. Every knob in that table tunes `mu` or the corrector; none touches the
  ceiling.
- **§4b's falsified `theta ≈ 0` hypothesis** — the replacement reading is
  `theta ≈ theta_max`. The note's own trace lines (`theta=9.820150e3`,
  `9.935307e3`, `9.716279e3`) are all 97–99 % of a 1e4 ceiling that was never
  printed alongside them.
- **§6's `inf_pr` "loose end" is not cosmetic.** It is the same fact as §1's
  third bullet — the internal and original measures diverge because slack lag is
  not infeasibility — and it is what disguised the mechanism.

## 5. FERAL ordering: 18–22 % on this shape, and `auto` picks the loser

Independent of the stall, and worth having regardless. `robot_a`, 50 iterations,
median of 3 runs:

| `feral_ordering` | total | vs default | factor | backsolve | nnz(L) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `auto` (default) | 46.0 s | — | 24.5 | 9.5 | 565 848 |
| `amd` | 46.3 s | +0.5 % | 24.8 | 9.4 | 565 848 |
| `amf` | 48.0 s | +4.2 % | 25.5 | 10.2 | 565 848 |
| **`metis`** | **37.9 s** | **−17.6 %** | 18.3 | 7.6 | 576 277 |
| **`kahip`** | **35.8 s** | **−22.3 %** | 17.0 | 7.0 | 576 384 |
| `scotch` | 54.7 s | +18.8 % | 23.8 | 18.9 | 967 579 |
| `auto_race` | 53.8 s | +16.8 % | 29.9 | 9.5 | 565 848 |

Transfers: `robot_b` 43.4 → 35.3 s (metis), `robot_c` 28.8 → 24.5 s (kahip).

- `auto`, `amd`, `amf` and `auto_race` all produce a **bit-identical iteration
  table** — four settings, one ordering. feral 0.14's dispatcher
  (`symbolic/mod.rs:84`) routes `n > 100_000 && nnz/n < 5` to AMD, and this KKT
  (105 027, nnz/n ≈ 3.4) lands in that catch. feral's own doc says that catch is
  calibrated "outside known IPM workloads".
- **Less fill is the wrong objective here.** METIS/KaHIP carry ~1.9 % *more*
  nonzeros in L and still factor 25–31 % faster — the squarer-front effect
  feral's option doc predicts for "banded / nearly-1D structure", which this
  model is (every Jacobian row touches 4 consecutive variables; `W` has
  half-bandwidth 3). Mechanism plausible but not measured — no flop counts.
- **`auto_race` therefore selects the loser**: it keeps the smallest
  `factor_nnz`, which is AMD's, and pays ~3.5 s of extra symbolic to get there.
  Racing on estimated factor flops, or on one timed numeric factorization, would
  pick differently.

Symbolic cost is one-time (`n_pattern_changes = 1` over the run): AMD 0.56 s,
METIS 1.19 s, KaHIP 1.87 s, `auto_race` 4.06 s.

Item 3 of §7 below asked for this to be measured "before `feral_ordering` gets
recommended anywhere user-facing". It was recommended first; the suite-wide
measurement landed as [gh#768](https://github.com/jkitchin/pounce/issues/768)
and agrees with this table — `auto_race` loses on 27 of 42 Mittelmann
instances, median 0.82×. See `feral-ordering-auto-race-mittelmann.md`.

Forcing FERAL's parallel dispatch (`POUNCE_FERAL_MIN_PAR_FLOPS=0`) buys 11 % for
16 % more CPU on 4 cores — threading is not hiding anything here.

## 6. Separate bug: the end-of-run "Constraint violation" line

On the **converged, independently verified** `robot_a` run:

| source | constraint violation |
| --- | --- |
| end-of-run summary | `3.109e-5` unscaled / `2.826e-10` scaled |
| `pounce verify` on the `.sol` | `0.000e0` |

Same point, and the verifier is the one to trust. The same gap shows up
mid-solve: at iteration 200 of a default run the `inf_pr` column reads
`0.00e+00` while the summary reports `5.304e6`. Whatever the summary line is
computing, the doc comment on `curr_unscaled_nlp_constraint_violation_max`
(`ipopt_cq.rs:894`) says it should be the original-NLP max-norm violation, and
it is not matching it. Worth its own issue; it is what made #476's §6 look
cosmetic.

## 7. What to check locally

Nothing here changes solver behaviour — this note is a record. Each item below
is a measurement this container could not make, ordered by how much it would
change the conclusion.

**1. Does IPOPT converge under the same option?** The highest-value check, ~30
seconds. If it does, this stops being a POUNCE note and becomes an upstream
Ipopt default problem — `theta_max_fact = 1e4` is Ipopt's default too, and it is
costing Mittelmann's IPOPT column these same three instances.

```
ipopt robot_a -AMPL theta_max_fact=1e8
```

*Confirms* if it converges in roughly 130 iterations at objective ≈ 1.043.
*Refutes* if it still hits 3000 — which would mean POUNCE and Ipopt diverge
somewhere the digit-for-digit comparison in the earlier note did not reach.

**2. Is the residual per-iteration gap FERAL or the algorithm?** Requires
`--features ma57` and the `libcoinhsl` already at `ref/Ipopt/install-ma57`;
unavailable in this container.

```
cargo build --release --bin pounce --features ma57
pounce robot_a.nl linear_solver=ma57 max_iter=200 print_timing_statistics=yes
```

Landing near IPOPT+MA57's 0.057 s/iter would make `robot_a` a linear-solver
story and retire the KKT-condensing project in §5 of the earlier note. Staying
high points at the assembly/refinement path rather than the factorization
kernel — which is where the earlier note's FERAL sampling already pointed
("refinement outweighs the factorization").

**3. Re-time the ordering sweep on the Mac mini.** The §5 table is from a 4-core
x86 box with FERAL effectively serial. Squarer fronts usually pay *more* on
wider vector units, so the win may be larger there — but it needs measuring
before `feral_ordering` gets recommended anywhere user-facing.

```
for o in auto amd amf metis scotch kahip auto_race; do
  pounce robot_a.nl max_iter=50 feral_ordering=$o print_timing_statistics=yes --no-sol
done
```

**4. Blast radius of a raised `theta_max`, before any default moves.** This is
the gate on everything in §1–§3. `theta_max` is a real global-convergence
safeguard; raising it admits iterates far from feasibility, and the
Wächter-Biegler argument assumes a bounded `theta` region. The measurement is
the full corpus both ways — status and objective per problem, not just the
geomean — with attention to problems that currently *rely* on the ceiling to
reject a bad path.

**5. Is the trigger really large `m` with a feasible start?** The mechanism in
§1 predicts that `theta_0 = 0` plus large `m` is what breaks it. Cheap to test
across the suite: record `pounce check-x0`'s initial violation and `m` for every
instance, and check whether the stalling ones cluster where `theta_0 = 0` and
`m` is large. `robot_1600` (same family, `m = 9601`, converges in 35 iterations)
is the control that fits. A coarser discretization of `robot_a` converging under
the *default* would be the clean confirmation — it needs `robot_a.mod`, which
plato.asu.edu would not serve to this container.

**6. Reproduce the reporting bug in §6** and decide whether it is the summary or
the column that is wrong. `pounce verify` disagreeing with our own end-of-run
line on a converged solve is the sharp case.

**7. Confirm file identity** before comparing any number here: the sha256s are
at the top. `gen_robot_nl.py`'s reconstruction is close but not byte-identical
to the AMPL output (it emits 12003 defined variables against AMPL's 11999), so
trajectories will differ.
