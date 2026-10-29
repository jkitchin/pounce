# CALIPSO corpus (`benchmarks/calipso/`)

The problems from Howell, Tracy, Le Cleac'h & Manchester, *CALIPSO: A
Differentiable Solver for Trajectory Optimization with Conic and
Complementarity Constraints* ([arXiv 2205.09255](https://arxiv.org/abs/2205.09255)),
ported from the authors' Julia sources
([thowell/CALIPSO.jl](https://github.com/thowell/CALIPSO.jl) @ `0e3bdcf`,
2022-12-14) and run through three solvers: POUNCE's NLP arm, stock Ipopt,
and CALIPSO.jl itself. It is workstream A of
[`dev-notes/research/calipso-contact-plan.md`](../../dev-notes/research/calipso-contact-plan.md):
the measurement every later claim in that plan is checked against.

| model | source (`examples/`) | n | structure |
|---|---|---:|---|
| `bunnyhop` | `contact_implicit/bunnyhop.jl` | 110 | 24 NCP pairs (wheel / rider contact) |
| `drifting_squared`, `drifting_norm` | `contact_implicit/drifting.jl` | 286 | 28 second-order-cone (SOC) complementarity pairs (Coulomb friction); cones lowered as `t² − ‖x‖² ≥ 0` or `t − ‖x‖ ≥ 0` |
| `ballincup` | `contact_implicit/ball_in_cup.jl` | 248 | 20 NCP pairs (string tension ⟂ slack) |
| `quadruped` | `contact_implicit/quadruped_gait.jl` | 2302 | NCP contact pairs + 2-D SOC friction pairs, 31 steps |
| `rocket` | `state_triggered/rocket_landing.jl` | 856 | 98 NCP pairs (state-triggered keep-out zones) |
| `complementarity` | `nonconvex/complementarity.jl` | 8 | 3 NCP pairs; known optimum 17 |
| `maratos` | `nonconvex/maratos.jl` | 2 | plain NLP (Maratos effect) |
| `wachter` | `nonconvex/wachter.jl` | 3 | plain NLP (Wächter–Biegler example) |

## Running it

```sh
cd benchmarks/calipso
PYTHON=python IPOPT_PYTHON=/path/to/cyipopt/python \
JULIA=julia JULIA_PROJECT=/path/to/env CALIPSO_JL=/path/to/CALIPSO.jl \
bash run_corpus.sh
```

Each baseline is skipped when its interpreter is unset; `MODELS`, `JOBS` and
`TIMEOUT` restrict and pace the run (see the script's header). POUNCE needs
the built extension (`make python-ext`) and `jax[cpu]`; Ipopt runs through
`cyipopt` (here: conda-forge, Ipopt 3.14.20 + MUMPS); CALIPSO.jl needs
Julia 1.8 and an environment with CALIPSO, RoboDojo 0.1.6 and JSON
(CALIPSO pins Symbolics 0.1.x). One cell:
`python -I -c "import sys; sys.path.insert(0, '.'); sys.argv=['x', 'rocket', 'scholtes_then_ncp', '0']; import run_one; run_one.main()"`.
`python stationarity.py` is the classifier's solver-free selftest.

| file | role |
|---|---|
| `models.py` | registry: model, initial point, source example, CALIPSO-answer mapping; job list |
| `bunnyhop.py`, `drifting.py`, `ballincup.py`, `quadruped.py`, `rocket.py`, `nonconvex.py` | the ports (each docstring records every transcription choice) |
| `mpcc_nlp.py` | lowering of the complementarity block, the routes, the POUNCE and Ipopt backends; JAX derivatives compressed by column coloring |
| `stationarity.py` | MPCC stationarity (S / M / C / W for NCP pairs, KKT for plain NLPs) |
| `calipso_baseline.jl` | runs a CALIPSO.jl example with the authors' own setup code |
| `evaluate_calipso.py` | scores CALIPSO's answers with the port |
| `summarize.py` | the tables below |

## What is compared

**Routes** (`mpcc_nlp.py`), with base options `tol=1e-8`,
`bound_relax_factor=0`, `honor_original_bounds=yes`, `max_iter=3000`, and
the `benchmarks/mpcc/` Scholtes schedule (τ = 1 → 1e-8, full warm starts):

| route | lowering of the complementarity block |
|---|---|
| `defaults_eq` | `comp = 0` with **solver defaults** (`bound_relax_factor=1e-8`) — the model handed over as-is |
| `direct` | `comp ≤ 0` |
| `ncp_eq` | `comp = 0` |
| `ncp_eq_l1` | `comp = 0` + `l1_exact_penalty_barrier=yes` (POUNCE only) |
| `jordan_eq` | SOC models: the paper's full Jordan-product rows `β ∘ η = 0` |
| `scholtes_then_ncp` | continuation, then one `comp = 0` solve |
| `scholtes_then_ineq` | continuation, then one `comp ≤ 0` solve |

**CALIPSO.jl** runs each example's source verbatim up to its `solve!`
(the authors' setup, initial guess and options, default tolerance 1e-4),
times a second solve after a warm-up, and writes its solution.

**Physical** means source-level violation ≤ 1e-4: equality residual, the
complementarity products (full Jordan product for SOC pairs), cones,
inequalities, bounds. **Certified** means physical *and* S-stationary (NCP
pairs) or KKT (plain NLPs), per `stationarity.py`, judged at the tolerance
the solver was asked for (1e-8; 1e-4 for CALIPSO). SOC complementarity is
`n/a`: the NCP stationarity classes do not apply.

### The ports are checked against CALIPSO's own answers

Each CALIPSO solution, mapped into the port and evaluated by the port, has
the violation CALIPSO itself reports — so the transcriptions are the same
problems:

| model | CALIPSO's equality residual | port's violation at CALIPSO's answer |
|---|---:|---:|
| drifting | 1.4e-5 | 1.0–1.6e-5 |
| ball-in-cup | 5.2e-5 | 5.2e-5 |
| quadruped | 5.4e-5 | 5.4e-5 |
| rocket | — | 7.3e-9 |
| complementarity / maratos / wachter | — | at the documented solutions |

The quadruped port was also checked piece by piece against RoboDojo
(dynamics, contact, objective and kinematics functions agree to ≤ 1.1e-13).
`bunnyhop` has no CALIPSO run here: its whole-problem symbolic code
generation exceeds this container's 15 GB (killed at 14 GB, alone).

## Results (POUNCE on this branch, FERAL; Ipopt 3.14.20 + MUMPS; CALIPSO.jl @ `0e3bdcf`)

5 seeds per cell; 2 for the quadruped (up to ~20 min per solve); 1 for
models whose start has no random part (rocket and the three nonconvex
examples). CALIPSO ran 5 seeds where it could (1 for the quadruped).

### Best route per solver

The route with the most physical runs, ties broken by objective:

| model | CALIPSO | POUNCE (best route) | Ipopt (best route) |
|---|---|---|---|
| bunny-hop | paper: 462.61, 101 it | `direct`: 5/5, **462.6**, 113 it | `direct`: 5/5, 462.6, 85 it |
| drifting (squared cones) | 5/5, 0.2405, 130 it | `scholtes_then_ineq`: 5/5, **0.2026**, 1759 it | `scholtes_then_ncp`: 5/5, **0.2026**, 436 it |
| drifting (`t − ‖x‖` cones) | 5/5, **0.2405**, 130 it | `direct`: 3/5, 1.008, 3000 it (unconverged) | `direct`: 4/5, 1.537, 2176 it |
| ball-in-cup | 5/5, **12.31**, **56 it** | `ncp_eq_l1`: 5/5, 12.38, 172 it | `scholtes_then_ncp`: 5/5, 12.38, 110 it |
| quadruped | 1/1, **0.2615**, **168 it** | `ncp_eq_l1`: 2/2, 0.288, 1718 it | `ncp_eq`: 1/2, 0.283, 1066 it |
| rocket | 5/5, **1529**, **74 it** | `scholtes_then_ncp`: 1/1, 1532, 278 it | `scholtes_then_ineq`: 1/1, 1532, 233 it |
| complementarity | 17, 12 it | 17 (most routes) | 17 (most routes) |
| maratos | −1, 6 it | −1, 6 it | −1, 6 it |
| wachter | **1, 15 it** | **fails** (local infeasibility) | **fails** (local infeasibility) |

### Every route

Cells: physical/seeds · certified/seeds · median objective (physical runs)
· median iterations. Regenerate with `python summarize.py results_*.jsonl`.

**bunny-hop** (paper: CALIPSO 462.61 / 1.8e-6 / 101 it; Ipopt 1503.42 / 9.0e-8 / 1409 it)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 5/5 · 4/5 · 463.2 · 41 | 5/5 · 4/5 · 463.2 · 39 |
| `direct` | 5/5 · 4/5 · 462.6 · 113 | 5/5 · 3/5 · 462.6 · 85 |
| `ncp_eq` | 5/5 · 4/5 · 463.2 · 70 | 5/5 · 5/5 · 463.2 · 66 |
| `ncp_eq_l1` | 5/5 · 5/5 · 13670 · 569 | n/a |
| `scholtes_then_ncp` | 5/5 · 5/5 · 463.2 · 225 | 5/5 · 5/5 · 1165 · 209 |
| `scholtes_then_ineq` | 5/5 · 5/5 · 463.2 · 257 | 5/5 · 5/5 · 1165 · 237 |

**drifting, squared cones** (CALIPSO here: 5/5 · n/a · 0.2405 · 130)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 0/5 · n/a · – · 1030 | 1/5 · n/a · 0.357 · 1157 |
| `direct` | 3/5 · n/a · 1.683 · 3000 | 5/5 · n/a · 0.299 · 1823 |
| `ncp_eq` | 3/5 · n/a · 0.208 · 2110 | 5/5 · n/a · 0.301 · 1053 |
| `ncp_eq_l1` | 5/5 · n/a · 0.2027 · 616 | n/a |
| `jordan_eq` | 5/5 · n/a · 0.388 · 3000 | 5/5 · n/a · 0.211 · 1479 |
| `scholtes_then_ncp` | 4/5 · n/a · 0.2034 · 1955 | 5/5 · n/a · 0.2026 · 436 |
| `scholtes_then_ineq` | 5/5 · n/a · 0.2026 · 1759 | 2/5 · n/a · 0.204 · 848 |

**drifting, `t − ‖x‖` cones** (CALIPSO here: 5/5 · n/a · 0.2405 · 130)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 0/5 · n/a · – · 39 | 3/5 · n/a · 1.038 · 62 |
| `direct` | 3/5 · n/a · 1.008 · 3000 | 4/5 · n/a · 1.537 · 2176 |
| `ncp_eq` | 3/5 · n/a · 1.046 · 3000 | 1/5 · n/a · 1.079 · 3000 |
| `ncp_eq_l1` | 0/5 · n/a · – · 3000 | n/a |
| `jordan_eq` | 3/5 · n/a · 23.46 · 3000 | 3/5 · n/a · 29.36 · 3000 |
| `scholtes_then_ncp` | 0/5 · n/a · – · 6 | 0/5 · n/a · – · 7 |
| `scholtes_then_ineq` | 0/5 · n/a · – · 6 | 0/5 · n/a · – · 7 |

**ball-in-cup** (paper: CALIPSO 11.96 / 131 it, Ipopt 68.18; CALIPSO here: 5/5 · 0/5 · 12.31 · 56)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 5/5 · 1/5 · 13.77 · 214 | 5/5 · 0/5 · 13.36 · 379 |
| `direct` | 5/5 · 3/5 · 12.38 · 506 | 5/5 · 2/5 · 12.38 · 154 |
| `ncp_eq` | 5/5 · 2/5 · 12.62 · 446 | 5/5 · 0/5 · 16.45 · 126 |
| `ncp_eq_l1` | 5/5 · 5/5 · 12.38 · 172 | n/a |
| `scholtes_then_ncp` | 5/5 · 4/5 · 12.38 · 158 | 5/5 · 5/5 · 12.38 · 110 |
| `scholtes_then_ineq` | 5/5 · 5/5 · 12.38 · 203 | 5/5 · 5/5 · 12.38 · 120 |

**quadruped** (paper: CALIPSO 574.84, see below; CALIPSO here: 1/1 · n/a · 0.2615 · 168)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 0/2 · n/a · – · 3000 | 1/2 · n/a · 0.332 · 3000 |
| `direct` | 0/2 · n/a · – · 2367 | 0/2 · n/a · – · 3000 |
| `ncp_eq` | 0/2 · n/a · – · 3000 | 1/2 · n/a · 0.283 · 1066 |
| `ncp_eq_l1` | 2/2 · n/a · 0.288 · 1718 | n/a |
| `jordan_eq` | 2/2 · n/a · 0.343 · 1760 | 1/2 · n/a · 0.503 · 858 |
| `scholtes_then_ncp` | 0/2 · n/a · – · 3000 | 0/2 · n/a · – · 4105 |
| `scholtes_then_ineq` | 0/2 · n/a · – · 3000 | 0/2 · n/a · – · 4034 |

**rocket** (CALIPSO here: 5/5 · 0/5 · 1529 · 74)

| route | POUNCE | Ipopt |
|---|---|---|
| `defaults_eq` | 0/1 · 0/1 · – · 103 | 0/1 · 0/1 · – · 146 |
| `direct` | 1/1 · 0/1 · 1543 · 104 | 1/1 · 0/1 · 1543 · 71 |
| `ncp_eq` | 1/1 · 0/1 · 1594 · 209 | timeout (900 s) |
| `ncp_eq_l1` | 1/1 · 0/1 · 1618 · 450 | n/a |
| `scholtes_then_ncp` | 1/1 · 1/1 · 1532 · 278 | 1/1 · 1/1 · 1532 · 184 |
| `scholtes_then_ineq` | 1/1 · 0/1 · 1532 · 244 | 1/1 · 1/1 · 1532 · 233 |

**complementarity / maratos / wachter**: POUNCE and Ipopt agree on every
route to the iteration on most of them. `complementarity` returns 17
(S-stationary optimum) on `direct`, `ncp_eq_l1` and both continuations, 25
(a genuine competing S-stationary point) on `ncp_eq`, and 85 (M- but not
S-stationary) under defaults; CALIPSO returns 17. `maratos`: all solve in
6 iterations. `wachter`: CALIPSO solves; POUNCE and Ipopt both stop at
`[-1, 0, 0]` with `Infeasible_Problem_Detected`, as do POUNCE's
`l1_exact_penalty_barrier` and `l1_fallback_on_restoration_failure`.

## What the numbers say

1. **The paper's headline does not survive a careful transcription.**
   Stock Ipopt solves bunny-hop (462.6) and, through the right route, the
   drifting car (0.2026). What fails is the formulation, not the solver.
2. **The route decides the outcome, on every MPCC model, for POUNCE and
   Ipopt alike** — and no route wins everywhere. `ncp_eq_l1` is POUNCE's
   only route that solves the quadruped on both seeds and is 5/5 on the
   drift, but finds a 30× worse bunny-hop; Ipopt's continuations find 1165
   on bunny-hop where POUNCE's find 463. This is the case for workstream B
   (declared complementarity, automatic route choice, a certificate).
3. **The cone lowering decides the drift.** With `t − ‖x‖ ≥ 0` no route
   converges to a physical answer on either solver; CALIPSO's native cones
   solve it every time. With the smooth lowering both solvers find 0.2026,
   below CALIPSO's 0.24. This is the case for workstream C.
4. **Where both solve, CALIPSO needs far fewer iterations and POUNCE/Ipopt
   find equal or better objectives.** Rocket 74 vs 233–278, ball-in-cup 56
   vs 110–172, quadruped 168 vs 1066–1718; drift 0.2026 vs 0.24, rocket
   1532 vs 1529, ball-in-cup 12.38 vs 12.31. Where the stationarity check
   applies, POUNCE's best routes certify S-stationary; CALIPSO's answers,
   judged at its own 1e-4 tolerance, do not (rocket: M-stationary;
   ball-in-cup: no class established — not yet investigated, and possibly
   an artefact of a relaxed pair at that tolerance).
5. **`Solve_Succeeded` under defaults is not physical on two models, for
   both solvers.** Of the reported successes that violate the physics, all
   but one are `defaults_eq` runs on the drift and the rocket:
   `bound_relax_factor=1e-8` loosens a cone or complementarity row, and at a
   cone tip or corner that becomes a ~1e-4 violation. (The exception is an
   Ipopt quadruped continuation that ended `Solved_To_Acceptable_Level` at
   6.7e-4.) Safe defaults on declared structure belong in workstream B.
6. **Some results sit on a knife edge.** Moving POUNCE's derivatives from
   dense to colored evaluation changed them only in the last bit (Hessians
   agree to 2.2e-16), yet the drift moved: `scholtes_then_ncp` 5/5 at 1015
   iterations → 4/5 at 1955, `direct` median 0.219 → 1.683. Single-route
   rankings on this model are partly noise; a fallback ladder is robust to
   that, a fixed route is not.
7. **`wachter` is the one failure no formulation fixes** — and the reason an
   augmented-Lagrangian fallback rung is a candidate, pending a larger
   degenerate corpus.

## The two follow-up experiments

### 1. An augmented-Lagrangian fallback rung (`al_fallback.py`, `exp_al_corpus.py`, `exp_al_control.py`)

`wachter` is the one failure no formulation fixes, and CALIPSO solves it.
To see whether that generalizes, a prototype AL rung was built *around*
POUNCE — equalities in a textbook augmented Lagrangian (Nocedal & Wright
alg. 17.4, nothing tuned), inequalities and bounds left to POUNCE's IPM,
then one warm POUNCE polish so the verdict is POUNCE's own — and tried on
every failure of every NLP fixture in `crates/pounce-cli/tests/fixtures`
(63 models; own start plus 10 starts perturbed by the variables' own
scale; 686 runs). A run fails if POUNCE's status is not success /
acceptable / feasible-point-found or the source violation exceeds 1e-4
(POUNCE's own `constr_viol_tol`); a rescue must meet the same bar, so the
rung cannot "rescue" an infeasible model — a rescue *proves* feasibility.

110 runs fail. 88 are on models no method solves from any start (six
fixtures infeasible by design, plus `circle_parabola_t1p5` and
`issue981_water_main`); the rung rescues none of them, as it must not. 22
are on known-feasible models:

| model | failures | AL rung | best existing option | any existing option |
|---|---:|---:|---:|---:|
| `issue981_wachter_biegler` | 11 (every start) | **11** | 0 | 0 |
| `circle_parabola_t0p5` | 2 | **2** | 1 | 1 |
| `discs` | 2 | 2 | 2 | 2 |
| `deb7` | 2 | **2** | 1 | 1 |
| `scale_audit_retry_meyer3` | 5 | 0 | 1 (`mu_strategy=adaptive`) | 1 |
| **total** | **22** | **17** | 4 (`l1_exact_penalty_barrier`) | 5 |

Existing options tried as the control: `l1_exact_penalty_barrier`,
`l1_fallback_on_restoration_failure`, `mu_strategy=adaptive`. The rung's
rescues reach the own-start optimum in all but one case (`discs` start 5:
21.38 against 12.00). It is expensive as prototyped: ~50 inner iterations
on `wachter` but 7–12k on `discs` and `deb7`, against a few hundred for a
plain solve.

**Reading:** a real effect, larger than every existing option combined
(17 vs 5), but 11 of the 17 are one model; on the rest it is 6 vs 5, with
two rescues nothing else gets. Medium evidence; the decisive test is a
larger trap corpus (CUTEst / MacMPEC from random starts), not available in
this container.

### 2. Derivatives at a contact switch (`exp_contact_sens.py`, `exp_contact_sens.jl`)

One implicit contact step of a falling particle,
`z⁺ = argmin ½(z − ẑ)² s.t. z ≥ 0`, `ẑ = z₀ + h·v − h²g`, so
`z⁺ = max(0, ẑ)` and the exact one-sided derivatives in the initial
velocity `v` are known: `h` in flight, 0 in contact, and at the switch `h`
for an increase and 0 for a decrease. (Trajectory *optimization* with
complementarity generically has a solution that **jumps** at a contact
switch — the control drops when the floor takes the load — and at a jump
no derivative is right; a contact *step* is the well-posed case, and the
one a differentiable simulator or CALIPSO's policy tuning differentiates
through.) Two transcriptions: the contact as the **bound** `z ≥ 0`, and as
explicit **complementarity** `z − ẑ − γ = 0, z·γ = 0` (CALIPSO's way).
Base points at the switch and ±1e-6, ±1e-3 away; steps ±1e-2, ±1e-3.

| method | bound form | complementarity form |
|---|---|---|
| POUNCE `sens_solution`, `degeneracy="directional"` (default) | **correct on both sides**, at the switch and off it (error = the IPM base offset only) | averaged on the increasing side (≈ h/2) |
| POUNCE `mode="fix_relax"` | exact to 1e-13 for increases | averaged |
| POUNCE `sens_jacobian` (plain) | averaged, h/2 — and warns that the point is degenerate | averaged |
| CALIPSO.jl (default tol 1e-4) | `dz/dv = 0.050` everywhere within ±1e-3 of the switch | same |
| CALIPSO.jl (tol 1e-8) | 0.0500 at the switch; 0.065 / 0.035 at ±1e-3 (exact 0.1 / 0) | 0.0500 at the switch; 0.091 / 0.009 at ±1e-3 |

**Reading:** POUNCE can already do what CALIPSO cannot — return the
correct one-sided derivative through a contact switch — but only when the
contact is a variable bound; its degeneracy machinery inspects bounds, not
complementarity rows, so written the usual way it averages like CALIPSO.
CALIPSO's implicit-function derivative is averaged across a band around
the switch (~√μ wide) at any tolerance. Connecting declared
complementarity pairs to the existing kink machinery is a concrete,
demonstrated win for workstream E.

## What this is not evidence about

- **The paper's quadruped numbers.** CALIPSO.jl's last commit (`0e3bdcf`,
  "quadruped mpc") changed `quadruped_gait.jl`; it now solves to 0.2615,
  not the paper's 574.84. The example also puts its friction cones on
  `u[9:24]`, which leaves η₃, η₄ in no cone and gives the normal forces a
  spurious cone — apparently an index offset of 4. The port reproduces the
  source as written; `quadruped.make_model("intended")` is the corrected
  model and is not run here.
- **Bunny-hop under CALIPSO** (out of memory here; the paper's number is
  shown).
- **MA57 or wall time.** Ipopt ran with MUMPS (the paper's linear solver);
  an SPRAL leg was cut off at 46 of 90 cells and is not reported. Wall
  times are not compared: CALIPSO.jl is Julia research code, and the
  POUNCE / Ipopt runs pay Python + JAX callbacks.
- **Iteration parity for the continuations**: their counts sum ten solves.
- **Scale.** The largest model is 2302 variables; the plan's claims about
  benchmark scale need `benchmarks/qp` and friends, not this corpus.
