# CALIPSO contact-implicit problems (`benchmarks/calipso/`)

Two problems from Howell, Tracy, Le Cleac'h & Manchester, *CALIPSO: A
Differentiable Solver for Trajectory Optimization with Conic and
Complementarity Constraints* ([arXiv 2205.09255](https://arxiv.org/abs/2205.09255)),
ported from the authors' Julia sources
([thowell/CALIPSO.jl](https://github.com/thowell/CALIPSO.jl) @ `0e3bdcf`,
2022-12-14) to run through POUNCE's NLP arm. The question they answer is
narrow: **can POUNCE solve, today, the problems the paper says Ipopt
cannot?**

| problem | source | n | rows | hard part |
|---|---|---:|---:|---|
| bunny-hop | `examples/contact_implicit/bunnyhop.jl`, `models/mountain_bike.jl` | 110 | 94 | 24 NCP pairs `eta ⟂ d(q)` (wheel and rider contact) |
| drifting car | `examples/contact_implicit/drifting.jl`, `models/cybertruck.jl` | 286 | 345 | 28 **second-order-cone** complementarity pairs (Coulomb friction, max dissipation) |

## Running it

Needs the built extension (`make python-ext`) and `jax[cpu]` (derivatives
are JAX; Jacobian and Hessian sparsity are detected structurally).

```sh
cd benchmarks/calipso
python -I -c "import sys; sys.path.insert(0, '.'); sys.argv=['x', 'bunnyhop', 'scholtes_then_ncp', '0']; import run_one; run_one.main()"
PYTHON=python bash run_all.sh      # 100 cells, ~40 min on 4 cores
python summarize.py               # tables below
```

## What is being compared

The complementarity block is lowered the way `benchmarks/mpcc/` lowers it
(`mpcc_nlp.py`), with the same base options (`tol=1e-8`,
`bound_relax_factor=0`, `honor_original_bounds=yes`; `max_iter` raised to
3000) and the same 1 → 1e-8 Scholtes schedule with full warm starts:

| route | lowering |
|---|---|
| `defaults_eq` | `comp = 0`, **POUNCE defaults** (`bound_relax_factor=1e-8`): the closest thing here to handing the model to a stock solver |
| `direct` | `comp <= 0` |
| `ncp_eq` | `comp = 0` |
| `ncp_eq_l1` | `comp = 0` + `l1_exact_penalty_barrier=yes` |
| `jordan_eq` | drifting only: the paper's three Jordan-product rows `beta ∘ eta = 0` |
| `scholtes_then_ncp` | Gate 0's supported route: continuation, then one `comp = 0` solve |
| `scholtes_then_ineq` | continuation, then one `comp <= 0` solve |

An NLP solver has no cones, so the drifting car's cone membership is
lowered two ways: `drifting_norm` uses `t - ||x|| >= 0`, which is what the
authors' own Ipopt script (`examples/experimental/cyberdrift_ipopt.jl`)
does; `drifting_squared` uses `t^2 - ||x||^2 >= 0, t >= 0`, which is smooth
and nonconvex. For two vectors in the self-dual SOC, `beta·eta = 0` is
equivalent to the full Jordan product, so the scalar is the
complementarity function every route except `jordan_eq` lowers.

"Violation" is source-level: the max of equality residual, the Jordan
product (not the scalar the NLP saw), cone, obstacle and bound violation.
A cell counts as physical at violation `<= 1e-4`.

## Results (POUNCE `22d0e23`, 5 seeds per cell)

The paper reports, for the same models, CALIPSO / Ipopt:

- bunny-hop: obj **462.61** / 1503.42, violation 1.8e-6 / 9.0e-8, 101 / 1409 iterations
- drifting: obj **0.24** / 8.66, violation 1.4e-5 / 1.0e-1, 189 / 194 iterations

### bunny-hop

| route | converged | viol <= 1e-4 | obj (median, best) | max viol (median) | iters (median) |
|---|---:|---:|---|---|---:|
| `defaults_eq` | 5 | 5 | 463.2, 463.2 | 1.5e-13 | 41 |
| `direct` | 5 | 5 | 462.6, 462.6 | 8.5e-14 | 103 |
| `ncp_eq` | 5 | 5 | 463.2, 463.2 | 1.0e-13 | 70 |
| `ncp_eq_l1` | 4 | 4 | 13670, 13670 | 4.4e-11 | 550 |
| `scholtes_then_ncp` | 5 | 5 | 463.2, 462.6 | 1.5e-13 | 212 |
| `scholtes_then_ineq` | 4 | 5 | 463.2, 462.6 | 3.5e-13 | 318 |

Every route but the ℓ₁ one reaches CALIPSO's objective (462.6) or a
neighbouring local solution (463.2) at violation ~1e-13, in a few seconds.
The ℓ₁ route converges to a feasible point 30× worse on all five seeds.

### drifting car, smooth (`squared`) cones

| route | converged | viol <= 1e-4 | obj (median, best) | max viol (median) | iters (median) |
|---|---:|---:|---|---|---:|
| `defaults_eq` | 5 | **0** | – | 3.5e-4 | 872 |
| `direct` | 2 | 3 | 0.219, 0.215 | 4.1e-5 | 2669 |
| `ncp_eq` | 0 | 3 | 0.452, 0.206 | 1.6e-5 | 3000 |
| `ncp_eq_l1` | 5 | 5 | 0.2027, 0.2026 | 2.2e-5 | 560 |
| `jordan_eq` | 1 | 5 | 1.305, 0.215 | 3.4e-9 | 3000 |
| `scholtes_then_ncp` | 5 | 5 | 0.2026, 0.2026 | 7.2e-6 | 1015 |
| `scholtes_then_ineq` | 5 | 5 | 0.2027, 0.2026 | 4.1e-6 | 693 |

### drifting car, `norm` cones (the authors' Ipopt transcription)

| route | converged | viol <= 1e-4 | obj (median, best) | max viol (median) | iters (median) |
|---|---:|---:|---|---|---:|
| `defaults_eq` | 0 | 1 | 1.06 | 3.0e-4 | 54 |
| `direct` | 0 | 2 | 0.94, 0.87 | 5.0e-4 | 3000 |
| `ncp_eq` | 0 | 3 | 1.05, 1.04 | 6.4e-5 | 1513 |
| `ncp_eq_l1` | 0 | 0 | – | 2.0e-1 | 3000 |
| `jordan_eq` | 0 | 5 | 26.4, 0.58 | 4.1e-8 | 3000 |
| `scholtes_then_ncp` | 0 | 0 | – | 2.3 | 6 |
| `scholtes_then_ineq` | 0 | 0 | – | 2.3 | 6 |

## What the numbers say

1. **Both problems are solvable with POUNCE today, through a route the user
   has to assemble.** `scholtes_then_ncp`, the route `benchmarks/mpcc/`
   chose on 11 toy cases, is the only one that is 5/5 physical on both
   problems. It lands on CALIPSO's bunny-hop objective and on 0.2026 for the
   drift, below CALIPSO's reported 0.24, at violation ~1e-5. It is not a
   solver option: it lives in this driver and in `benchmarks/mpcc/routes.py`.
   It is also slow in iterations: 1015 for the drift against CALIPSO's 189.
2. **No single-solve route works on both.** `ncp_eq_l1` is the fastest
   route on the drift (5/5, 560 iterations) and the worst on bunny-hop (a
   30×-worse local solution on every seed). `ncp_eq` is the reverse.
3. **The cone lowering decides the drift, not the complementarity route.**
   With `t - ||x|| >= 0`, no route produces a converged, physical answer:
   the norm has no derivative at the cone vertex, a sticking contact sits
   exactly there, and the continuation routes hit `Invalid_Number_Detected`
   (NaN) at iteration 6 of their first stage. That reproduces the paper's
   claim about Ipopt on this transcription. The squared lowering is smooth
   and is what makes the drift solvable.
4. **`Solve_Succeeded` is not "physical" on the vertex.** Under POUNCE
   defaults the squared drift reports `Solve_Succeeded` on all five seeds
   at violation 3.5e-4. `bound_relax_factor=1e-8` relaxes
   `t^2 - ||x||^2 >= 0` by ~1e-8, and at a sticking contact (`t → 0`) that
   becomes `||x|| <= 1e-4`: the square root of the relaxation. The Jordan
   residual inherits the same square root from the scalar `beta·eta`. Every
   route that pins `bound_relax_factor=0` stays below 1e-5 on the cone.
5. **Four `scholtes_then_ineq` solves were certified through the
   scale-relative dual floor** (`dual_scale` 1.5e16, `dual_inf` 2.3e4): its
   `G·H <= 0` finish at a biactive pair has no multiplier, the known case in
   `docs/src/options.md`, "A settled point with a runaway multiplier". No
   `scholtes_then_ncp` result depends on it.

## What this is not evidence about

- **POUNCE versus Ipopt.** Ipopt was not run on this port (no `cyipopt` wheel
  installs here). POUNCE's defaults solve this bunny-hop in 41 iterations
  where the paper's Ipopt took 1409 and finished at a 3× worse objective, but
  the port differs from whatever the paper handed Ipopt: the contact
  nonnegativity is variable bounds here, the Julia RNG initialization cannot
  be reproduced, and no bunny-hop Ipopt script exists in CALIPSO.jl. Running
  Ipopt on these exact callbacks is the attribution test.
- **CALIPSO's own numbers.** CALIPSO was not run. Its drift objective (0.24)
  came at `tol=1e-3`; whether 0.2026 is the same local solution polished
  tighter, or a different one, is unmeasured.
- **Iteration parity.** The continuation's iteration count is a sum over
  ten solves, not comparable one-for-one with a single CALIPSO solve.
- **The other three paper problems** (ball-in-cup, quadruped gait,
  entry-vehicle). Not ported.
