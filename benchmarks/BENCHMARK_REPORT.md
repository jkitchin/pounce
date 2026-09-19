# POUNCE Benchmark Report

Generated: 2026-09-19 01:19:54

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.12.0 (bench/v0.12.0-sweep @ 7db5738a) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.6.0 arm64 |

POUNCE results were produced this run by `make -C benchmarks
<suite>-run` (pounce only). The Ipopt column is a saved reference
(`make -C benchmarks ipopt-reference`), rerun only when explicitly
regenerated — generated 2026-06-11 21:49:49 EDT on Johns-Mac-mini.local (Darwin 25.5.0 arm64), git 659d98a, timelimit 300s. Ipopt solve *times* are
from that reference machine and only comparable to POUNCE when this
report is generated on the same host.

The GAMS solver-link path is exercised separately as a liveness
smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.

> **Threading & timing.** These POUNCE runs carry no per-suite thread stamp — they predate it, so the
> settings they ran under were not recorded.
> At report time they are not pinned (`OMP_NUM_THREADS` unset, `OPENBLAS_NUM_THREADS` unset, `VECLIB_MAXIMUM_THREADS` unset, `RAYON_NUM_THREADS` unset), which says nothing about the runs themselves.
> Treat POUNCE-vs-Ipopt time comparisons below as unverified on this axis.

> **Time limits.** The saved Ipopt reference ran at `max_cpu_time` = 300s unless overridden below. The POUNCE arm carries no time-limit flag — it is wrapped in `timeout $BENCH_TIMELIMIT` and a kill is recorded as `Maximum_CpuTime_Exceeded` — but the limit in force is stamped per suite into `<suite>/pounce.env.json` and is read from there below.
> Override — **mittelmann**: Ipopt reference at 1800s (regenerated 2026-08-07, threads pinned to 1 (OMP/OPENBLAS/VECLIB/RAYON)).
> Reason given: the 300s max_cpu_time in the base stamp was reached at ~90s wall on unpinned multithreaded BLAS, leaving 6 instances truncated; see dev-notes/research/mittelmann-post-546-sweep.md
> POUNCE ran this suite at 300s (recorded in `pounce.env.json`) against the reference's 1800s, so the two columns are **not** held to the same clock here.
> Decided by that gap: **WM_CFy** — POUNCE cut off at 300s (102 iters), Ipopt Optimal at 1147s (556 iters), i.e. past POUNCE's cutoff. It is counted here as an Ipopt-only solve, on a limit POUNCE was never given. Measured out of band on 2026-08-11 at the reference's own 1800s limit (same host, threads pinned, binary from ce41b5bc): POUNCE returns `Optimal Solution Found` in 673s / 239 iterations, vs Ipopt's 1147s / 556. Objectives differ by 0.136% (two local optima; Ipopt's is the better point). Deliberately not merged into the results, which record the sweep as configured — see dev-notes/research/wm-cfy-timelimit.md.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1284/1326** (96.8%) | **1241/1326** (93.6%) |
| Acceptable (informational, *not* counted as solved) | 3 | 24 |
| Solved exclusively (strict Optimal) | 48 | 5 |
| Both Optimal | 1236 | |
| Matching objectives (< 0.01%) | 1178/1236 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Performance Profiles

[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) performance profiles pooled over every suite with an Ipopt reference. ρ_s(τ) is the fraction of problems a solver solves within a factor τ of the fastest solver on each problem: the **height at τ=1** is how often it was the quickest, and the **right-hand plateau** is its overall robustness (fraction solved at all). A problem counts as solved only at strict/acceptable success; failures and timeouts are charged infinite cost. Regenerate or slice these with `python3 scripts/perf_profile.py <suite…> [--metric iters] [--mode data]`.

![**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).](figures/profile_performance_time.png)

**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).
  
_1311 problems; solvers: pounce, ipopt._

![**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.](figures/profile_performance_iters.png)

**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.
  
_1311 problems; solvers: pounce, ipopt._

![**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.](figures/profile_data_time.png)

**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.
  
_1345 problems; solvers: pounce, ipopt._

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 697 (95.1%) | 683 (93.2%) | 18 | 4 | 679 | 657/679 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 1 (100.0%) | 1 (100.0%) | 0 | 0 | 1 | 1/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 5 | 5 (100.0%) | 5 (100.0%) | 0 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 46 (97.9%) | 41 (87.2%) | 6 | 1 | 40 | 39/40 |
| QP | 138 | 138 (100.0%) | 133 (96.4%) | 5 | 0 | 133 | 125/133 |
| LP | 371 | 369 (99.5%) | 352 (94.9%) | 17 | 0 | 352 | 327/352 |
| LPopt | 4 | 2 (50.0%) | 0 (0.0%) | 2 | 0 | 0 | 0/1 |

## Problems without an Ipopt reference — 19 problems

These problems were added to a suite after its saved ipopt-ma57 reference was taken. They are **excluded from every POUNCE-vs-Ipopt count in this report** (the executive summary, the per-suite table, the profiles, wins and regressions) rather than scored as Ipopt failures. Refresh the suite's reference with `make -C benchmarks ipopt-ref-<suite>` to bring them into the comparison.

| Problem | Suite | n | m | POUNCE status | POUNCE objective |
|---------|-------|---|---|---------------|------------------|
| pglib_case10000_goc | Grid | 125029 | 173770 | Optimal | 1.354031e+06 |
| pglib_case118_ieee | Grid | 1742 | 2379 | Optimal | 9.721361e+04 |
| pglib_case1354_pegase | Grid | 17675 | 25120 | Optimal | 1.258844e+06 |
| pglib_case14_ieee | Grid | 191 | 262 | Optimal | 2.178080e+03 |
| pglib_case162_ieee_dtc | Grid | 2485 | 3598 | Optimal | 1.080756e+05 |
| pglib_case179_goc | Grid | 2312 | 3307 | Optimal | 7.542664e+05 |
| pglib_case1888_rte | Grid | 23291 | 32836 | Optimal | 1.402531e+06 |
| pglib_case2000_goc | Grid | 29425 | 43482 | Optimal | 9.734325e+05 |
| pglib_case240_pserc | Grid | 3841 | 5348 | Optimal | 3.329670e+06 |
| pglib_case300_ieee | Grid | 3908 | 5415 | Optimal | 5.652200e+05 |
| pglib_case30_ieee | Grid | 388 | 541 | Optimal | 8.208515e+03 |
| pglib_case4661_sdet | Grid | 56671 | 79212 | Optimal | 2.251344e+06 |
| pglib_case500_goc | Grid | 6703 | 9274 | Optimal | 4.549460e+05 |
| pglib_case57_ieee | Grid | 738 | 1045 | Optimal | 3.758934e+04 |
| pglib_case588_sdet | Grid | 6728 | 9283 | Optimal | 3.131398e+05 |
| pglib_case73_ieee_rts | Grid | 1220 | 1503 | Optimal | 1.897641e+05 |
| pglib_case793_goc | Grid | 8936 | 12395 | Optimal | 2.601978e+05 |
| pglib_case89_pegase | Grid | 1748 | 2565 | Optimal | 1.072857e+05 |
| laptime | LargeScale | 58014 | 62014 | Optimal | 6.529464e+01 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 662 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 8 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 662 / 684 expected-solvable (96.8%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (22):**

> britgas coshfun cresc100 cresc50 dallasl dallasm deconvb discs eigenc2 flosp2hh grouping himmelbj nonmsqrt orthrds2 palmer5e polak3 sineali ssebnln steenbrc steenbrd steenbrf steenbrg

**Objective disagreements vs. cute_table reference (20)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.963305e+03 | 2.499976e+01 | 7.8e+01 |
| liswet8 | 7.144874e+02 | 2.499977e+01 | 2.8e+01 |
| liswet7 | 4.987922e+02 | 2.499979e+01 | 1.9e+01 |
| eigenbco | 4.732885e-19 | 9.000000e+00 | 1.0e+00 |
| liswet10 | 4.948391e+01 | 2.499967e+01 | 9.8e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| liswet1 | 3.612062e+01 | 2.500304e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| liswet12 | -3.314381e+03 | -5.026353e+03 | 3.4e-01 |
| cresc132 | 8.577054e-01 | 6.848460e-01 | 1.7e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| steenbre | 2.851495e+04 | 2.745916e+04 | 3.8e-02 |
| haldmads | 3.303041e-02 | 1.223712e-04 | 3.3e-02 |
| errinros | 4.040449e+01 | 3.990415e+01 | 1.3e-02 |
| trainh | 1.231200e+01 | 1.236996e+01 | 4.7e-03 |

## Vanderbei Suite — Performance

On 679 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 29.2ms | 44.3ms |
| Total time | 271.93s | 234.62s |
| Mean iterations | 47.0 | 46.9 |
| Median iterations | 15 | 16 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.3x
- POUNCE faster: 522/679 (77%)
- POUNCE 10x+ faster: 1/679
- Ipopt faster: 157/679

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 26.3ms | 37.6ms |
| Total time | 353.7ms | 503.3ms |
| Mean iterations | 12.3 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.4x
- **Median speedup**: 1.4x
- POUNCE faster: 13/13 (100%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 0/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 29.3ms | 41.9ms |
| Total time | 118.7ms | 157.2ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.3x
- **Median speedup**: 1.4x
- POUNCE faster: 4/4 (100%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 0/4

## CHO Suite — Performance

On 1 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 7.08s | 1.76s |
| Total time | 7.08s | 1.76s |
| Mean iterations | 20.0 | 33.0 |
| Median iterations | 20 | 33 |

- **Geometric mean speedup**: 0.2x
- **Median speedup**: 0.2x
- POUNCE faster: 0/1 (0%)
- POUNCE 10x+ faster: 0/1
- Ipopt faster: 1/1

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 99.0ms | 122.5ms |
| Total time | 644.2ms | 696.0ms |
| Mean iterations | 198.2 | 205.2 |
| Median iterations | 183 | 209 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 0.9x
- POUNCE faster: 2/6 (33%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 4/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 77.0ms | 113.3ms |
| Total time | 261.9ms | 374.0ms |
| Mean iterations | 39.7 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 1.5x
- **Median speedup**: 1.5x
- POUNCE faster: 3/3 (100%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 0/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 2.39s | 573.2ms |
| Total time | 10.72s | 9.43s |
| Mean iterations | 309.6 | 305.6 |
| Median iterations | 5 | 2 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.6x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 40 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 5.88s | 6.88s |
| Total time | 829.28s | 1488.30s |
| Mean iterations | 153.6 | 110.7 |
| Median iterations | 55 | 55 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 0.8x
- POUNCE faster: 14/40 (35%)
- POUNCE 10x+ faster: 4/40
- Ipopt faster: 26/40

## QP Suite — Performance

On 133 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 78.6ms | 92.9ms |
| Total time | 85.81s | 172.97s |
| Mean iterations | 18.6 | 75.6 |
| Median iterations | 18 | 24 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.3x
- POUNCE faster: 91/133 (68%)
- POUNCE 10x+ faster: 3/133
- Ipopt faster: 42/133

## LP Suite — Performance

On 352 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 139.7ms | 157.8ms |
| Total time | 150.75s | 422.20s |
| Mean iterations | 25.2 | 107.3 |
| Median iterations | 23 | 56 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.1x
- POUNCE faster: 198/352 (56%)
- POUNCE 10x+ faster: 10/352
- Ipopt faster: 154/352

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 3 | 6 |
| Diverging_Iterates | 3 | 0 |
| Error_In_Step_Computation | 4 | 0 |
| Infeasible_Problem_Detected | 3 | 4 |
| Invalid_Number_Detected | 1 | 3 |
| Maximum_CpuTime_Exceeded | 2 | 8 |
| Maximum_Iterations_Exceeded | 12 | 16 |
| Not_Enough_Degrees_Of_Freedom | 2 | 0 |
| Restoration_Failed | 2 | 3 |
| Search_Direction_Becomes_Too_Small | 2 | 1 |
| Solver_Error | 2 | 2 |
| Unknown_Error | 0 | 7 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 1 | 2 |
| Maximum_Iterations_Exceeded | 0 | 3 |
| Solver_Error | 0 | 1 |

### QP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 0 | 4 |
| Maximum_CpuTime_Exceeded | 0 | 1 |

### LP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 0 | 14 |
| Infeasible_Problem_Detected | 2 | 1 |
| Maximum_CpuTime_Exceeded | 0 | 1 |
| Maximum_Iterations_Exceeded | 0 | 1 |
| Restoration_Failed | 0 | 1 |
| Unknown_Error | 0 | 1 |

### LPopt Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 2 | 4 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| WM_CFy | Mittelmann | 8709 | 12850 | Maximum_CpuTime_Exceeded | 1.221824e+00 |
| discs | Vanderbei | 36 | 69 | Infeasible_Problem_Detected | 1.444952e+01 |
| orthrds2 | Vanderbei | 203 | 100 | Acceptable | 1.544297e+03 |
| ssebnln | Vanderbei | 194 | 96 | Error_In_Step_Computation | 1.617060e+07 |
| steenbrd | Vanderbei | 468 | 108 | Error_In_Step_Computation | 9.030082e+03 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 48 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BOYD1 | QP | 93261 | 18 | Acceptable | -6.173522e+07 |
| BOYD2 | QP | 93263 | 186531 | Maximum_CpuTime_Exceeded | 2.125677e+01 |
| QPILOTNO | QP | 2172 | 975 | Acceptable | 4.728587e+06 |
| QRECIPE | QP | 180 | 91 | Acceptable | -2.666160e+02 |
| QSCORPIO | QP | 358 | 388 | Acceptable | 1.880510e+03 |
| aa4 | LP | 7195 | 426 | Acceptable | 2.587761e+04 |
| air05 | LP | 7195 | 426 | Acceptable | 2.587761e+04 |
| bore3d | LP | 315 | 233 | Acceptable | 1.373080e+03 |
| brainpc0 | Vanderbei | 6905 | 6900 | Restoration_Failed | 1.499639e-03 |
| brainpc1 | Vanderbei | 6905 | 6900 | Restoration_Failed | 4.382655e-04 |
| brainpc2 | Vanderbei | 13805 | 13800 | Maximum_CpuTime_Exceeded | 4.406987e-04 |
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.750404e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.927066e-04 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| co5 | LP | 7993 | 5715 | Acceptable | 7.144696e+05 |
| complex | LP | 1408 | 1023 | Acceptable | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| cq5 | LP | 7530 | 5025 | Acceptable | 4.001338e+05 |
| cresc132 | Vanderbei | 6 | 2654 | Infeasible_Problem_Detected | 8.577054e-01 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 |
| cvxqp3 | Vanderbei | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.157111e+08 |
| dallass | Vanderbei | 46 | 31 | Invalid_Number_Detected | -3.239323e+04 |
| de063155 | LP | 1488 | 852 | Restoration_Failed | 9.883094e+09 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| finnis | LP | 614 | 497 | Acceptable | 1.727911e+05 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.000000e+01 |
| greenbea | LP | 5405 | 2389 | Maximum_Iterations_Exceeded | -7.248917e+07 |
| greenbeb | LP | 5405 | 2389 | Acceptable | -4.302260e+06 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741512e-01 |
| maros | LP | 1443 | 845 | Acceptable | -5.806374e+04 |
| nql180 | Mittelmann | 129601 | 130080 | Solver_Error | -9.277211e-01 |
| palmer7e | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 1.015390e+01 |
| pilot.ja | LP | 1988 | 940 | Acceptable | -6.113136e+03 |
| pilotnov | LP | 2172 | 975 | Acceptable | -4.497276e+03 |
| polak6 | Vanderbei | 5 | 4 | Unknown_Error | -4.400000e+01 |
| qap15 | LPopt | 22275 | 6330 | Maximum_CpuTime_Exceeded | 1.040994e+03 |
| qcqp1500-1c | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 3.882979e+06 |
| qcqp1500-1nc | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 4.778480e+06 |
| recipe | LP | 180 | 91 | Acceptable | -2.666160e+02 |
| robot_a | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 1.043195e+00 |
| robot_b | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 2.333099e+00 |
| robot_c | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 1.405976e+00 |
| scfxm1-2r-27 | LP | 6189 | 4088 | Acceptable | 2.886965e+03 |
| scorpion | LP | 358 | 388 | Acceptable | 1.878125e+03 |
| scrs8-2r-256 | LP | 9765 | 7196 | Maximum_CpuTime_Exceeded | 1.144161e+03 |
| steenbre | Vanderbei | 540 | 126 | Acceptable | 2.851495e+04 |
| supportcase10 | LPopt | 14630 | 165684 | Maximum_CpuTime_Exceeded | 3.383924e+00 |

## Acceptable (not Optimal) — 3 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| orthrds2 | Vanderbei | 203 | 100 | Optimal | 1.461041e+03 | 1.544297e+03 |
| steenbrc | Vanderbei | 540 | 126 | Unknown_Error | 4.364283e+04 | 2.597624e+04 |
| steenbrf | Vanderbei | 468 | 108 | Acceptable | 7.205566e+02 | 1.321652e+03 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 0 | 300.15s |
| irish-electricity | 61,728 | 104,259 | Maximum_CpuTime_Exceeded | N/A | 779 | 300.05s |
| qap15 | 22,275 | 6,330 | Optimal | 1.0410e+03 | 26 | 36.46s |
| supportcase10 | 14,630 | 165,684 | Optimal | 3.3839e+00 | 32 | 119.16s |

POUNCE: **2/4 Optimal** in 755.83s total

## Dedicated Convex Solver vs. General NLP (head-to-head)

The same LP / convex-QP `.nl` problems solved twice by the **same**
pounce binary: once routed to the dedicated convex interior-point
solver (`pounce-convex`, via `solver_selection=lp-ipm` / `qp-ipm`) and
once through the general NLP filter-IPM (`solver_selection=nlp`). This
quantifies the speedup the dedicated solver buys on its home turf. It
is a pounce-vs-pounce comparison and is independent of the Ipopt
reference used by the suites above.

> **The Ipopt-MA57 reference is a status and timing reference, not an
> objective oracle.** It carries Ipopt's `bound_relax_factor` widening,
> which changes the answer on constraint-degenerate models by `delta`
> times the bound's multiplier. On the `LISWET*` / `YAO` families POUNCE
> disagrees with it by tens of percent and POUNCE is correct: scored on
> the published Maros-Meszaros optima (DOC 97/6) the convex arm is
> 138/138 and Ipopt-MA57 misses 8. Score an objective disagreement
> against DOC 97/6 (`compare_qp_four_way.py`) before calling it a
> regression -- gh #744 was filed by not doing that.

### LP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 368/371 (99.2%) | 363/371 (97.8%) |
| Solved exclusively | 6 | 1 |
| Both Optimal | 362 | |
| Matching objectives (< 0.01%) | 336/362 | |

On 362 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 140.7ms | 189.2ms |
| Total time | 155.71s | 648.15s |
| Mean iterations | 25.2 | 117.4 |
| Median iterations | 23 | 57 |

- **Geometric-mean speedup (pounce-convex over pounce-nlp)**: 1.4x
- **Median speedup**: 1.1x
- pounce-convex faster: 249/362 (69%)
- pounce-convex 10x+ faster: 11/362
- pounce-nlp faster: 113/362

### QP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 137/138 (99.3%) | 138/138 (100.0%) |
| Solved exclusively | 0 | 1 |
| Both Optimal | 137 | |
| Matching objectives (< 0.01%) | 129/137 | |

On 137 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 78.1ms | 91.9ms |
| Total time | 331.36s | 499.08s |
| Mean iterations | 19.3 | 82.8 |
| Median iterations | 18 | 25 |

- **Geometric-mean speedup (pounce-convex over pounce-nlp)**: 1.1x
- **Median speedup**: 1.0x
- pounce-convex faster: 61/137 (45%)
- pounce-convex 10x+ faster: 4/137
- pounce-nlp faster: 76/137

### Mittelmann — exact vs L-BFGS

| Metric | exact Hessian | L-BFGS |
|--------|---------------|------------|
| Optimal | 46/47 (97.9%) | 32/47 (68.1%) |
| Solved exclusively | 14 | 0 |
| Both Optimal | 32 | |
| Matching objectives (< 0.01%) | 29/32 | |

On 32 problems solved by both arms:

| Metric | exact Hessian | L-BFGS |
|--------|---------------|------------|
| Median time | 7.91s | 9.37s |
| Total time | 833.26s | 970.13s |
| Mean iterations | 76.5 | 229.7 |
| Median iterations | 48 | 70 |

- **Geometric-mean speedup (exact Hessian over L-BFGS)**: 1.5x
- **Median speedup**: 1.8x
- exact Hessian faster: 22/32 (69%)
- exact Hessian 10x+ faster: 0/32
- L-BFGS faster: 10/32

---
*Generated by benchmark_report.py*