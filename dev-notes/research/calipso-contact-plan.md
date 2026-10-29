# Research note — contact-implicit problems: complementarity and cones (CALIPSO)

**Status: research note / plan, not started.** Written after porting two
problems from the CALIPSO paper and running them through POUNCE. It
proposes what to build, what each piece buys, and how we would know it
worked. Work happens on the `feature/contact-mpcc-cones` branch until we
decide it is done. No issues are filed yet; the tracker holds planned
work only, and this is not planned yet.

## 0. Where this comes from

Howell, Tracy, Le Cleac'h & Manchester, *CALIPSO: A Differentiable Solver
for Trajectory Optimization with Conic and Complementarity Constraints*
([arXiv 2205.09255](https://arxiv.org/abs/2205.09255)). CALIPSO is a
primal-dual augmented-Lagrangian (equalities) + interior-point (cones)
solver with native second-order cones, a symmetric KKT reduction,
Ipopt-style inertia correction and filter line search, and
implicit-function-theorem derivatives with a least-squares fallback when
the KKT matrix is singular. The paper reports that Ipopt fails on its
contact-implicit and state-triggered problems.

`benchmarks/calipso/` ports two of them (bunny-hop: 110 variables, 24 NCP
pairs; drifting car: 286 variables, 28 second-order-cone complementarity
pairs) from CALIPSO.jl @ `0e3bdcf` and runs them through the
`benchmarks/mpcc/` route set. Results and caveats are in
[`benchmarks/calipso/README.md`](../../benchmarks/calipso/README.md).

### What the port measured (POUNCE `22d0e23`, FERAL, 5 seeds)

- **Bunny-hop needs nothing new.** Every single-solve route reaches
  CALIPSO's objective (462.6) or a neighbouring local solution (463.2) at
  violation ~1e-13.
- **The paper's bunny-hop Ipopt failure did not reproduce on the port.**
  Stock Ipopt 3.14.20 + MUMPS (conda-forge) on the same callbacks, seed 0:
  44 iterations with defaults (POUNCE: 41), 65 with `G·H = 0` (POUNCE:
  65), at the same objectives. The 1503 / 1409-iteration result in the
  paper comes from their formulation, not from the solver. The full Ipopt
  matrix (MUMPS, SPRAL) was still running when this note was written;
  MA57 needs a licensed HSL build (`IPOPT_HSLLIB`).
- **`scholtes_then_ncp` is the only route physical on 5/5 starts for both
  problems.** It lives in benchmark code, not the solver, and costs 1015
  iterations on the drifting car against CALIPSO's 189.
- **No single-solve route works on both.** `ncp_eq_l1` is 5/5 and fastest
  on the drift, and 30× worse in objective on every bunny-hop start;
  `ncp_eq` is the reverse.
- **Cones are the real gap.** Written `t − ‖x‖ ≥ 0`, no route converges to
  a physical answer: a gripping wheel sits at the cone tip, where the norm
  has no derivative (continuations hit NaN at iteration 6). The smooth
  `t² − ‖x‖² ≥ 0` works, but only with `bound_relax_factor=0`. With
  defaults it reports `Solve_Succeeded` at violation 3.5e-4, because a
  1e-8 relaxation becomes ~1e-4 at the tip (its square root).

### What POUNCE already has, so we do not build it

Filter line search, inertia correction with primal and dual
regularization, iterative refinement, IFT sensitivities (sIPOPT-style and
the JAX/Torch layers), the ℓ₁ exact-penalty wrapper, the MPCC stationarity
classifier (`benchmarks/mpcc/stationarity.py`), SOC cone algebra with
Nesterov–Todd scaling (`crates/pounce-convex/src/cones/soc.rs`), and an
active-set SQP arm with `pounce-qp` subproblems.

CALIPSO's augmented-Lagrangian treatment of equalities is **not** worth
porting as the main algorithm. Its stated purpose is a full-rank KKT
without LICQ, which dual regularization already provides, and the closest
POUNCE relative (the ℓ₁ wrapper) gave mixed results on the port. **It may
be worth having as a fallback rung**, though, for a reason the paper does
not state: on `wachter` (CALIPSO.jl `nonconvex/wachter.jl`) CALIPSO
reaches the documented solution in 15 iterations, while POUNCE, stock
Ipopt, `l1_exact_penalty_barrier=yes` and
`l1_fallback_on_restoration_failure=yes` all stop at `[-1, 0, 0]` with
`Infeasible_Problem_Detected`. That point is a genuine local minimizer of
the constraint violation, so restoration and the ℓ₁ penalty are behaving
as designed; the quadratic augmented-Lagrangian path with multiplier
updates simply does not lead into it. One tiny problem is not a case for
building anything, but it is the only kind of failure in this corpus that
no formulation choice fixes.

## 1. Rules that apply throughout

- Any change that reroutes or rescales the NLP solver's steps (B2, C) runs
  `scripts/sweep-fixtures.sh`, both legs, before merge, and every moved
  line is explained (CLAUDE.md, "Trajectory changes need the fixture
  sweep").
- Any change to the sensitivity layer (E) runs the invariance legs and the
  re-solve oracle, with one fixture per branch the rule can take
  (CLAUDE.md, "Sensitivity changes need the invariance legs").
- Thresholds below are set before measuring. If one is missed, the result
  is recorded as missed, not re-scoped.

### Design principle: it just works by default

The target user should not need to know what a Scholtes relaxation, a
Jordan product or `bound_relax_factor` is. An option is acceptable as an
override or escape hatch, but **the default must get these problems
right with no options set**. Most of what failed in the port — for the
paper's Ipopt runs and for POUNCE alike — was a formulation choice the
user had to make (`t − ‖x‖` vs `t² − ‖x‖²`, `G·H = 0` vs `≤ 0`,
`bound_relax_factor` left at its default), not a missing algorithm. That
implies four things, which cut across B, C and E:

1. **Declare structure, never reformulate it.** Users say "these are
   complementary" or "this lies in a cone" (Python API, Pyomo `mpec`,
   AMPL `complements`); the solver chooses the lowering.
2. **Detect the common forms when nothing is declared**, the way convex
   quadratic constraints are already detected and routed to the conic
   solver: a product of two nonnegative quantities held at `= 0` or
   `≤ 0` is a complementarity pair; `‖x‖ ≤ t` or its square is a
   second-order cone.
3. **A fallback ladder with a certificate, not one fixed route.** No
   single route won on the port: ℓ₁ was best on the drifting car and worst
   on bunny-hop, `ncp_eq` the reverse, and only the Scholtes continuation
   was 5/5 on both, at ~4× the iterations. So: try the fast route, classify
   the answer (`benchmarks/calipso/stationarity.py`), fall back to the
   robust route only on failure, and report the class. This is the
   pattern POUNCE already uses in `mu_strategy_fallback` and the
   `square_system` ladder. Never report a bare success on a point that is
   not physical (the squared-cone drift under defaults reports
   `Solve_Succeeded` at 3.5e-4).
4. **Safe defaults for declared structure**: no bound relaxation on cone
   and complementarity rows, which is what turns a 1e-8 relaxation into a
   1e-4 violation at a cone tip.

**Acceptance for "just works"**: with no options set, a physical answer
certified S-stationary (or KKT) on every corpus problem where *any* of
POUNCE, Ipopt or CALIPSO finds one; zero wrong certificates; the fixture
sweep clean on everything POUNCE already solves. It cannot mean "finds the
best local solution": `complementarity` has S-stationary points at 17 and
25, and both are correct answers for a local method.

## 2. Workstreams

### A. Measurement foundation (first; small)

Without a wider corpus, every later claim rests on two models.

- Port the rest of the paper: ball-in-cup, quadruped gait, rocket landing
  (`state_triggered/rocket_landing.jl`), and the auto-tuning examples
  (cart-pole, acrobot); plus CALIPSO.jl's `nonconvex/complementarity.jl`,
  `maratos.jl`, `wachter.jl`.
- Run CALIPSO.jl itself (install Julia), so "better than CALIPSO" is
  measured, not read from a 2022 table.
- Classify every MPCC solution with `benchmarks/mpcc/stationarity.py`:
  "solved" means S-stationary, not "residuals small" (on `ctrap` a
  C-stationary point has every residual at zero and is not a local
  minimum).
- Fixed metrics on every row: physical-solve rate over 5 seeds, objective
  vs best known, source-level violation, stationarity class, iterations,
  wall time, and warm re-solve time for MPC use.
- Baselines: POUNCE today (best route), Ipopt with MUMPS / SPRAL / MA57,
  CALIPSO.jl.

**Done when** the corpus, the three baselines, and the stationarity column
exist and run from one command.

### B. First-class complementarity in the NLP solver

1. **Declare pairs** `G ⟂ H` in the Python API, Pyomo `mpec`, and `.nl`
   (AMPL's format has native complementarity, which is how MacMPEC is
   written; check whether `pounce-nl` reads it).
2. **One solve instead of ten.** Tie the relaxation to the barrier,
   `G·H ≤ δ·μ` (Raghunathan–Biegler, IPOPT-C), so it tightens as μ → 0.
3. **Certify the answer.** After converging, read the active side of each
   pair, solve the tightened NLP once (LICQ holds there), and report the
   stationarity class in the JSON. CALIPSO reports residuals only.

**Measured by:**
- Gate 0 corpus: 60/60 solved, 0 below `f*` (no regression).
- CALIPSO port: 5/5 physical on both problems; drifting-car iterations
  1015 → **≤ 300**.
- Share of solutions certified S-stationary across corpus A.

### C. Native second-order cones in the NLP solver (the real gap)

- `h(x) ∈ K`, K a product of second-order cones, via a cone slack with
  barrier `−log(t² − ‖x‖²)`; reuse `soc.rs` for the cone arithmetic,
  scaling and step-to-boundary rule. Then extend B to cone pairs
  `β ∘ η = 0`.
- **Buys:** friction- and thrust-cone problems without user
  reformulation; no "succeeded but 3.5e-4 off" at a gripping contact; a
  new problem class (nonconvex dynamics with cone constraints, e.g.
  powered descent).

**Measured by:**
- Drifting car with native cones: **5/5 physical**, cone-tip violation
  **≤ 1e-8 under default options**.
- Rocket landing with thrust cones.
- **Free correctness check:** CBLIB convex SOCPs through the NLP solver vs
  `pounce-convex` (global optima): objectives match to 1e-6 on **≥ 95%**.

### D. Sequential conic programming (research-grade; after C)

- Let the active-set SQP arm solve a cone-constrained QP with
  `pounce-convex` at each step instead of a QP with `pounce-qp`: the
  GuSTO/SCvx idea from the paper's related work, with exact second
  derivatives.
- **Buys:** SQP warm starts for MPC re-solves; convex subproblems solved
  to a certified optimum; a second route to compare against C.

**Measured by:** warm re-solve time over an MPC rollout (drifting car,
rocket) against C and CALIPSO; iterations to a given accuracy.

### E. Correct derivatives through contact (likely the most novel)

- At a contact switch the system CALIPSO differentiates is singular, and
  its least-squares fallback returns an averaged gradient.
  `crates/pounce-sensitivity/tests/sens_resolve_oracle.rs` documents the
  same effect at a kink (the plain step is the two-sided average, off by
  `A/2·δ`), and POUNCE's directional / fix-relax modes return the
  one-sided derivative.
- Route the sensitivity layer through B's declared pairs, so contact
  switches are handled the way kinks already are.

**Measured by:**
1. Derivative error vs finite-difference re-solves on both sides of a
   contact switch, POUNCE vs a CALIPSO-style least-squares emulation
   (extends the re-solve oracle). Run this first: it decides whether E is
   worth implementing.
2. Reproduce the paper's Table 3 (cart-pole, acrobot auto-tuning).
3. Contact MPC tuning (hopper): tuning-loop convergence with correct vs
   averaged gradients.

### F. Speed from trajectory structure (low priority)

At these sizes (≤ 300 variables) wall time is dominated by Python
callbacks, and CALIPSO does not exploit trajectory structure either.
Measure time per iteration for horizon T = 15 → 500 before doing
anything.

## 3. Order

| | Size | Unlocks | Key number |
|---|---|---|---|
| A | small | trustworthy claims | full paper corpus, 3 baselines |
| B1 + B3 | medium | usable MPCC API, certified answers | % certified S-stationary |
| E (measure only) | small | decides whether E is worth building | derivative error at a switch |
| B2 | medium | speed | drift iterations 1015 → ≤ 300 |
| C | large | new problem class | norm-form drift 0/5 → 5/5; CBLIB parity |
| E (implement) | medium | correct bilevel gradients | tuning convergence with contact |
| D | large / research | MPC warm starts | warm re-solve time |

## 4. Open questions

- ~~Does Ipopt's continuation also solve the drifting car?~~ **Yes**:
  Ipopt 3.14.20 + MUMPS, `scholtes_then_ncp`, 5/5 physical at 0.2041 in
  729 iterations (POUNCE: 0.2026, 1015), and Ipopt's plain `ncp_eq` is 5/5
  where POUNCE's is 3/5. So B is a usability feature, not a capability
  Ipopt lacks.
- ~~Is 0.2026 the same drifting-car solution as CALIPSO's 0.24?~~ **No**:
  CALIPSO.jl run here returns 0.239–0.253 across five seeds (violation
  ~1.4e-5, matching its own residual), so POUNCE and Ipopt find genuinely
  lower-cost trajectories.
- Does `pounce-nl` already parse AMPL complementarity constraints?
- The paper's quadruped numbers (objective 574.84) are not reproducible
  from CALIPSO.jl at `0e3bdcf`: that commit changed `quadruped_gait.jl`,
  and it now solves to 0.26 in 168 iterations. The example also puts its
  friction cones on `u[9:24]`, which leaves η₃, η₄ in no cone and gives
  the normal forces a spurious cone — an index offset of 4, apparently.
  `benchmarks/calipso/quadruped.py` reproduces the source as written and
  keeps the corrected model as `cones="intended"`.
