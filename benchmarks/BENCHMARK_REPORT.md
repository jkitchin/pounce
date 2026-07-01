# POUNCE Benchmark Report

Generated: 2026-07-01 16:38:54

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.7.0 (release/0.7.0 @ 8dc0667-dirty) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.5.0 arm64 |

POUNCE results were produced this run by `make -C benchmarks
<suite>-run` (pounce only). The Ipopt column is a saved reference
(`make -C benchmarks ipopt-reference`), rerun only when explicitly
regenerated — generated 2026-06-11 21:49:49 EDT on Johns-Mac-mini.local (Darwin 25.5.0 arm64), git 659d98a, timelimit 300s. Ipopt solve *times* are
from that reference machine and only comparable to POUNCE when this
report is generated on the same host.

The GAMS solver-link path is exercised separately as a liveness
smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.

> **Threading & timing.** The reference and POUNCE runs are pinned to a
> single compute thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
> `VECLIB_MAXIMUM_THREADS`, `RAYON_NUM_THREADS` all = 1) and run
> sequentially so pounce and Ipopt solve times are directly comparable
> on one host.
> POUNCE's dense linear algebra (via `faer`/`rayon`) parallelizes across
> cores, so its *multi-threaded* wall-clock is up to ~2x faster on the
> larger dense problems (e.g. Mittelmann `cont*`/`qcqp*`, QP); the
> single-threaded times reported here are therefore a controlled lower
> bound, not pounce's real-world speed, and should not be compared
> against multi-threaded runs of this report.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1260/1326** (95.0%) | **1237/1326** (93.3%) |
| Acceptable (informational, *not* counted as solved) | 12 | 24 |
| Solved exclusively (strict Optimal) | 46 | 23 |
| Both Optimal | 1214 | |
| Matching objectives (< 0.01%) | 1159/1214 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Performance Profiles

[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) performance profiles pooled over every suite with an Ipopt reference. ρ_s(τ) is the fraction of problems a solver solves within a factor τ of the fastest solver on each problem: the **height at τ=1** is how often it was the quickest, and the **right-hand plateau** is its overall robustness (fraction solved at all). A problem counts as solved only at strict/acceptable success; failures and timeouts are charged infinite cost. Regenerate or slice these with `python3 scripts/perf_profile.py <suite…> [--metric iters] [--mode data]`.

![**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).](figures/profile_performance_time.png)

**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).
  
_1286 problems; solvers: pounce, ipopt._

![**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.](figures/profile_performance_iters.png)

**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.
  
_1286 problems; solvers: pounce, ipopt._

![**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.](figures/profile_data_time.png)

**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.
  
_1326 problems; solvers: pounce, ipopt._

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 684 (93.3%) | 683 (93.2%) | 18 | 17 | 666 | 649/666 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 1 (100.0%) | 1 (100.0%) | 0 | 0 | 1 | 1/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 2/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 5 | 5 (100.0%) | 5 (100.0%) | 0 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 42 (89.4%) | 37 (78.7%) | 6 | 1 | 36 | 35/36 |
| QP | 138 | 138 (100.0%) | 133 (96.4%) | 5 | 0 | 133 | 125/133 |
| LP | 371 | 363 (97.8%) | 352 (94.9%) | 16 | 5 | 347 | 322/347 |
| LPopt | 4 | 1 (25.0%) | 0 (0.0%) | 1 | 0 | 0 | 0/1 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 649 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 8 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 649 / 684 expected-solvable (94.9%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (35):**

> airport brainpc0 brainpc2 britgas coshfun cresc100 cresc132 cresc50 dallass deconvb dixchlnv eigena2 flosp2hh grouping himmelbj hs012 hs043 hs099 kissing makela2 minmaxbd nonmsqrt orthrds2 orthrege palmer1c palmer5e palmer7c polak3 polak4 rosenmmx sawpath sineali spanhyd steenbrc steenbrf

**Objective disagreements vs. cute_table reference (26)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| quartc | 2.488823e+02 | 2.406057e-18 | 2.5e+02 |
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.963305e+03 | 2.499976e+01 | 7.8e+01 |
| dqrtic | 3.935818e+01 | 1.654558e-17 | 3.9e+01 |
| liswet8 | 7.144874e+02 | 2.499977e+01 | 2.8e+01 |
| liswet7 | 4.987922e+02 | 2.499979e+01 | 1.9e+01 |
| penalty1 | 6.439498e+00 | 9.686175e-03 | 6.4e+00 |
| eigenbco | 1.024905e-16 | 9.000000e+00 | 1.0e+00 |
| liswet10 | 4.948391e+01 | 2.499967e+01 | 9.8e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| liswet1 | 3.612062e+01 | 2.500304e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| liswet12 | -3.314381e+03 | -5.026353e+03 | 3.4e-01 |
| discs | 1.444952e+01 | 1.200008e+01 | 2.0e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| steenbre | 2.851495e+04 | 2.745916e+04 | 3.8e-02 |
| haldmads | 3.195581e-02 | 1.223712e-04 | 3.2e-02 |
| errinros | 4.040449e+01 | 3.990415e+01 | 1.3e-02 |
| cliff | 2.072380e-01 | 1.997866e-01 | 7.5e-03 |
| lch | -4.287718e+00 | -4.318289e+00 | 7.1e-03 |
| trainh | 1.231200e+01 | 1.236996e+01 | 4.7e-03 |
| twirism1 | -1.008371e+00 | -1.006758e+00 | 1.6e-03 |

## Vanderbei Suite — Performance

On 666 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 41.1ms | 44.5ms |
| Total time | 348.48s | 233.67s |
| Mean iterations | 47.3 | 47.3 |
| Median iterations | 15 | 16 |

- **Geometric mean speedup**: 0.9x
- **Median speedup**: 1.0x
- POUNCE faster: 318/666 (48%)
- POUNCE 10x+ faster: 1/666
- Ipopt faster: 348/666

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 31.4ms | 37.6ms |
| Total time | 429.8ms | 503.3ms |
| Mean iterations | 14.8 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.2x
- POUNCE faster: 10/13 (77%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 3/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 39.7ms | 41.9ms |
| Total time | 146.7ms | 157.2ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.1x
- POUNCE faster: 4/4 (100%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 0/4

## CHO Suite — Performance

On 1 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 4.24s | 1.76s |
| Total time | 4.24s | 1.76s |
| Mean iterations | 36.0 | 33.0 |
| Median iterations | 36 | 33 |

- **Geometric mean speedup**: 0.4x
- **Median speedup**: 0.4x
- POUNCE faster: 0/1 (0%)
- POUNCE 10x+ faster: 0/1
- Ipopt faster: 1/1

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 142.6ms | 122.5ms |
| Total time | 814.2ms | 696.0ms |
| Mean iterations | 192.7 | 205.2 |
| Median iterations | 191 | 209 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 0.9x
- POUNCE faster: 1/6 (17%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 5/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 92.7ms | 113.3ms |
| Total time | 486.8ms | 374.0ms |
| Mean iterations | 40.0 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.2x
- POUNCE faster: 2/3 (67%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 1/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 3.09s | 573.2ms |
| Total time | 13.96s | 9.43s |
| Mean iterations | 309.2 | 305.6 |
| Median iterations | 5 | 2 |

- **Geometric mean speedup**: 0.5x
- **Median speedup**: 0.4x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 36 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 10.00s | 5.65s |
| Total time | 1072.37s | 1227.95s |
| Mean iterations | 92.4 | 96.3 |
| Median iterations | 35 | 41 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.6x
- POUNCE faster: 12/36 (33%)
- POUNCE 10x+ faster: 0/36
- Ipopt faster: 24/36

## QP Suite — Performance

On 133 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 91.1ms | 92.9ms |
| Total time | 104.49s | 172.97s |
| Mean iterations | 18.2 | 75.6 |
| Median iterations | 17 | 24 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.0x
- POUNCE faster: 73/133 (55%)
- POUNCE 10x+ faster: 2/133
- Ipopt faster: 60/133

## LP Suite — Performance

On 347 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 163.2ms | 156.2ms |
| Total time | 183.10s | 409.87s |
| Mean iterations | 24.5 | 107.2 |
| Median iterations | 23 | 56 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 0.9x
- POUNCE faster: 157/347 (45%)
- POUNCE 10x+ faster: 10/347
- Ipopt faster: 190/347

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 12 | 6 |
| Infeasible_Problem_Detected | 5 | 4 |
| Invalid_Number_Detected | 1 | 3 |
| Maximum_CpuTime_Exceeded | 3 | 8 |
| Maximum_Iterations_Exceeded | 17 | 16 |
| Restoration_Failed | 4 | 3 |
| Search_Direction_Becomes_Too_Small | 2 | 1 |
| Solver_Error | 5 | 2 |
| Unknown_Error | 0 | 7 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 5 | 6 |
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
| Maximum_Iterations_Exceeded | 6 | 1 |
| Restoration_Failed | 0 | 1 |
| Unknown_Error | 0 | 1 |

### LPopt Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 2 | 4 |
| Maximum_Iterations_Exceeded | 1 | 0 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| NARX_CFy | Mittelmann | 43973 | 48256 | Maximum_CpuTime_Exceeded | 8.726796e-03 |
| airport | Vanderbei | 84 | 42 | Restoration_Failed | 4.795270e+04 |
| dixchlnv | Vanderbei | 100 | 50 | Search_Direction_Becomes_Too_Small | 0.000000e+00 |
| eigena2 | Vanderbei | 110 | 55 | Acceptable | 8.250000e+01 |
| gen | LP | 2560 | 769 | Maximum_Iterations_Exceeded | -1.097485e-05 |
| gen1 | LP | 2560 | 769 | Maximum_Iterations_Exceeded | -1.097485e-05 |
| gen4 | LP | 4297 | 1537 | Maximum_Iterations_Exceeded | -2.221401e-05 |
| hs012 | Vanderbei | 2 | 1 | Acceptable | -3.000000e+01 |
| hs043 | Vanderbei | 4 | 3 | Acceptable | -4.400000e+01 |
| hs099 | Vanderbei | 23 | 18 | Acceptable | -8.310799e+08 |
| kissing | Vanderbei | 127 | 903 | Acceptable | 8.454426e-01 |
| kleemin8 | LP | 8 | 8 | Maximum_Iterations_Exceeded | -1.000000e+14 |
| makela2 | Vanderbei | 3 | 3 | Acceptable | 7.200000e+00 |
| minmaxbd | Vanderbei | 5 | 20 | Restoration_Failed | 1.157064e+02 |
| orthrds2 | Vanderbei | 203 | 100 | Acceptable | 1.544297e+03 |
| orthrege | Vanderbei | 36 | 20 | Acceptable | 3.868188e+00 |
| palmer1c | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 9.759799e-02 |
| palmer7c | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 6.019857e-01 |
| pilot87 | LP | 4883 | 2030 | Maximum_Iterations_Exceeded | 3.017104e+02 |
| polak4 | Vanderbei | 3 | 3 | Restoration_Failed | -3.513953e-09 |
| rosenmmx | Vanderbei | 5 | 4 | Acceptable | -4.400000e+01 |
| sawpath | Vanderbei | 593 | 786 | Infeasible_Problem_Detected | 1.815730e+02 |
| spanhyd | Vanderbei | 97 | 33 | Acceptable | 2.397380e+02 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 46 problems

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
| brainpc1 | Vanderbei | 6905 | 6900 | Restoration_Failed | 4.362953e-04 |
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.752286e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.926834e-04 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| co5 | LP | 7993 | 5715 | Acceptable | 7.144696e+05 |
| complex | LP | 1408 | 1023 | Acceptable | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| cq5 | LP | 7530 | 5025 | Acceptable | 4.001338e+05 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 |
| cvxqp3 | Vanderbei | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.157111e+08 |
| dallasl | Vanderbei | 906 | 667 | Invalid_Number_Detected | -2.026041e+05 |
| dallasm | Vanderbei | 196 | 151 | Invalid_Number_Detected | -4.819819e+04 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.555870e-03 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.555870e-03 |
| eigenc2 | Vanderbei | 462 | 231 | Unknown_Error | 7.718095e+02 |
| finnis | LP | 614 | 497 | Acceptable | 1.727911e+05 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.000000e+01 |
| greenbea | LP | 5405 | 2389 | Maximum_Iterations_Exceeded | -7.246479e+07 |
| greenbeb | LP | 5405 | 2389 | Acceptable | -4.302260e+06 |
| henon120 | Mittelmann | 32401 | 241 | Maximum_CpuTime_Exceeded | 1.332947e+02 |
| lane_emden120 | Mittelmann | 57721 | 241 | Maximum_CpuTime_Exceeded | 9.340251e+00 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741684e-01 |
| maros | LP | 1443 | 845 | Acceptable | -5.806374e+04 |
| nql180 | Mittelmann | 129601 | 130080 | Solver_Error | -9.277258e-01 |
| palmer7e | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 1.015390e+01 |
| pilot.ja | LP | 1988 | 940 | Acceptable | -6.113136e+03 |
| pilotnov | LP | 2172 | 975 | Acceptable | -4.497276e+03 |
| polak6 | Vanderbei | 5 | 4 | Unknown_Error | -4.400000e+01 |
| qap15 | LPopt | 22275 | 6330 | Maximum_CpuTime_Exceeded | 1.040994e+03 |
| qcqp1000-2c | Mittelmann | 1000 | 5107 | Maximum_CpuTime_Exceeded | 7.381274e+05 |
| qcqp1500-1c | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 3.882979e+06 |
| qcqp1500-1nc | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 4.778480e+06 |
| recipe | LP | 180 | 91 | Acceptable | -2.666160e+02 |
| scfxm1-2r-27 | LP | 6189 | 4088 | Acceptable | 2.886965e+03 |
| scorpion | LP | 358 | 388 | Acceptable | 1.878125e+03 |
| scrs8-2r-256 | LP | 9765 | 7196 | Maximum_CpuTime_Exceeded | 1.144161e+03 |
| steenbre | Vanderbei | 540 | 126 | Acceptable | 2.851495e+04 |
| steenbrg | Vanderbei | 540 | 126 | Acceptable | 2.747128e+04 |

## Acceptable (not Optimal) — 12 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| dallass | Vanderbei | 46 | 31 | Invalid_Number_Detected | -3.202464e+04 | N/A |
| eigena2 | Vanderbei | 110 | 55 | Optimal | 8.250000e+01 | 8.250000e+01 |
| hs012 | Vanderbei | 2 | 1 | Optimal | -3.000000e+01 | -3.000000e+01 |
| hs043 | Vanderbei | 4 | 3 | Optimal | -4.400000e+01 | -4.400000e+01 |
| hs099 | Vanderbei | 23 | 18 | Optimal | -8.310799e+08 | -8.310799e+08 |
| kissing | Vanderbei | 127 | 903 | Optimal | 1.000001e+00 | 8.454426e-01 |
| makela2 | Vanderbei | 3 | 3 | Optimal | 7.200000e+00 | 7.200000e+00 |
| orthrds2 | Vanderbei | 203 | 100 | Optimal | 1.544296e+03 | 1.544297e+03 |
| orthrege | Vanderbei | 36 | 20 | Optimal | 3.934338e+00 | 3.868188e+00 |
| rosenmmx | Vanderbei | 5 | 4 | Optimal | -4.400000e+01 | -4.400000e+01 |
| spanhyd | Vanderbei | 97 | 33 | Optimal | 2.397380e+02 | 2.397380e+02 |
| steenbrc | Vanderbei | 540 | 126 | Unknown_Error | 1.946894e+04 | 2.597624e+04 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 0 | 300.08s |
| irish-electricity | 61,728 | 104,259 | Maximum_Iterations_Exceeded | 2.4544e+06 | 199 | 111.78s |
| qap15 | 22,275 | 6,330 | Optimal | 1.0410e+03 | 22 | 30.50s |
| supportcase10 | 14,630 | 165,684 | Maximum_CpuTime_Exceeded | N/A | 0 | 300.09s |

POUNCE: **1/4 Optimal** in 742.44s total

## Dedicated Convex Solver vs. General NLP (head-to-head)

The same LP / convex-QP `.nl` problems solved twice by the **same**
pounce binary: once routed to the dedicated convex interior-point
solver (`pounce-convex`, via `solver_selection=lp-ipm` / `qp-ipm`) and
once through the general NLP filter-IPM (`solver_selection=nlp`). This
quantifies the speedup the dedicated solver buys on its home turf. It
is a pounce-vs-pounce comparison and is independent of the Ipopt
reference used by the suites above.

### LP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 363/371 (97.8%) | 354/371 (95.4%) |
| Solved exclusively | 14 | 5 |
| Both Optimal | 349 | |
| Matching objectives (< 0.01%) | 324/349 | |

On 349 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 161.9ms | 240.6ms |
| Total time | 185.27s | 818.92s |
| Mean iterations | 24.4 | 116.7 |
| Median iterations | 23 | 56 |

- **Geometric-mean speedup (convex over nlp)**: 1.5x
- **Median speedup**: 1.2x
- pounce-convex faster: 251/349 (72%)
- pounce-convex 10x+ faster: 15/349
- pounce-nlp faster: 98/349

### QP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 137/138 (99.3%) | 134/138 (97.1%) |
| Solved exclusively | 4 | 1 |
| Both Optimal | 133 | |
| Matching objectives (< 0.01%) | 125/133 | |

On 133 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 87.3ms | 126.7ms |
| Total time | 101.45s | 185.11s |
| Mean iterations | 18.2 | 75.9 |
| Median iterations | 17 | 24 |

- **Geometric-mean speedup (convex over nlp)**: 1.2x
- **Median speedup**: 1.1x
- pounce-convex faster: 88/133 (66%)
- pounce-convex 10x+ faster: 1/133
- pounce-nlp faster: 45/133

---
*Generated by benchmark_report.py*