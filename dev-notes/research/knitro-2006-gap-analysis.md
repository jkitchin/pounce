# Research note — KNITRO (Byrd/Nocedal/Waltz 2006) vs pounce: gap analysis

**Status: analysis. No code proposed here.** Source: Byrd, Nocedal &
Waltz, "KNITRO: An Integrated Package for Nonlinear Optimization",
in *Large-Scale Nonlinear Optimization*, Springer 2006 — describing
**KNITRO 5.0**. That vintage matters: this is the paper's KNITRO, not
today's (no MINLP, no multistart, no parallelism, no complementarity
constraints in scope). Everything below is measured against the 2006
feature set only.

The paper's three algorithms:

| KNITRO name | What it is |
|---|---|
| INTERIOR/DIRECT | line-search primal-dual IPM, direct factorization, trust-region step as safeguard |
| INTERIOR/CG | trust-region IPM, Byrd–Omojokun normal + tangential steps, projected CG with Steihaug truncation |
| ACTIVE | SLQP — LP phase (simplex) picks the working set, EQP phase (projected CG) computes the step |

plus **crossover** between the interior and active-set paths, and a
shared **projected preconditioned CG** module.

pounce ≈ INTERIOR/DIRECT's line-search half (via the Ipopt lineage),
with a filter instead of a merit function, and a separate active-set
SQP that is *not* SLQP.

## 1. Already at parity — do not re-plan these

| Paper §  | Feature | pounce |
|---|---|---|
| §3 | monotone (Fiacco–McCormick) barrier update | `mu/monotone.rs` |
| §3 | LOQO barrier rule | `mu/oracle/loqo.rs` |
| §3 | Mehrotra probing rule | `mu/oracle/probing.rs` |
| §3 | quality-function rule | `mu/oracle/quality_function.rs` |
| §3 | safeguarded corrector steps | `corrector_type` / `skip_corr_*` — runs |
| §6 | BFGS / SR1 / L-BFGS, Powell damping | `hess/lim_mem_quasi_newton.rs` |
| §6 | least-squares initial multipliers | `eq_mult/least_square.rs`, `bound_mult_init_method` |
| §6 | infeasibility detection | shipped (C2); plus refutation, which KNITRO has no analogue for |
| §6 | LP / QP special-casing | `docs/src/lp-qp-routing.md` — and pounce **auto-detects**, where KNITRO 5.0 requires the user to declare it |

`future-work-roadmap.md` §3.2 lists C1 (active-set path for warm
starts) as a Tier-3 gap. **That entry is stale** — the active-set SQP
shipped (`docs/src/active-set-sqp.md`, CHANGELOG "Phase 5b + 5c +
5d"). It matters here because it changes the cost of item 2.1 below
from "build a second solver" to "build a bridge".

## 2. Already designed in-repo, not implemented

INTERIOR/CG is the paper's largest single block (§3.2, §5) and is
already covered by two design notes:

- `composite-step-byrd-omojokun.md` — the normal/tangential step
  decomposition and the trust-region spine pounce does not have.
- `interior-cg-matrix-free.md` — the Krylov solve on top of it, and
  the projected-CG/Steihaug module the paper's §5 describes.

Nothing in the paper changes those plans. Two details in it are worth
folding into the notes when they get implemented:

- The **preconditioner/projection operator** (5.18) is reused for
  three jobs — projection inside CG, the normal step, and the
  Lagrange multipliers — so the second and third cost one back-solve
  against a factorization already formed. KNITRO 5.0 ships `D = I`.
- Scaling the slacks (3.13) before forming the projection is what
  keeps `s → 0` from ill-conditioning the CG iteration.

## 3. Genuinely absent, and worth having

Ranked by value ÷ effort.

### 3.1 Crossover: interior-point → active-set (§7)

**What it is.** After the IPM converges to `E_tol`, estimate the
active set by a tolerance test on primal-dual feasibility and
complementarity, take one EQP step (4.6) plus a line search on the
model (4.5), and stop if that already satisfies the tolerances. Only
if it does not, start the full active-set algorithm from
`(x_k, s_k, y_k, z_k)` with an LP trust region sized (7.22) to exclude
every inactive constraint, and `ν₀` a little above the largest
|multiplier| at the interior solution. Where strict complementarity
holds, the tolerance test is right and crossover costs one iteration
and zero LPs.

**Why pounce should care.** Three reasons, none hypothetical:

1. pounce already wrote the argument for it, for LPs.
   `pounce-convex/src/crossover.rs`'s header describes exactly the
   IPM failure mode at a degenerate solution — `α → 1e-4`, `μ`
   freezes, the residual plateaus above tolerance, right objective,
   no certificate. That failure is not LP-specific; a degenerate NLP
   does the same thing, and there the NLP path has no crossover.
2. `pounce-sensitivity` (the sIPOPT port) and any reduced-Hessian
   consumer are downstream of *which constraints are active*. An
   interior solve gives that set only approximately.
3. The active-set SQP's warm start currently comes from a previous
   **SQP** solve (`docs/src/active-set-sqp.md` §warm start). Crossover
   would let an IPM solve seed a working set, which is the natural
   entry point for MPC/homotopy sequences whose first solve is cold.

**Cost.** Both endpoints exist. Unlike KNITRO's, pounce's active-set
path is SQP rather than SLQP, so the "EQP step from a tolerance-based
active-set estimate" (crossover step 3) has to be expressed against
`pounce-qp`'s working-set interface rather than an EQP phase — the
LP-crossover bridge in `pounce-convex` is the precedent for that
translation. Opt-in and post-convergence, so it moves no default
trajectory.

### 3.2 Feasible mode (§6)

**What it is.** An option under which every iterate satisfies
`c_I(x) ≥ 0`. Requires a strictly feasible start w.r.t. the
inequalities; equalities are unaffected. The line-search adaptation is
three lines of algebra: after `x⁺ = x + d_x`, *reset* the slacks
`s⁺ = max(s + d_s, c_I(x⁺))` — really `s_i⁺ = c_i(x⁺)` where the step
would have left the region — then test the merit function. Trial
points outside the region give `+∞`, so they are rejected without a
special case, and the `-μ Σ log s_i` term also rejects points that
merely come too close to the boundary.

**Why pounce should care.** pounce's user base is process/chemistry
models where `f` or `c` is *undefined* outside the constraints —
`log`, `sqrt`, division. `docs/src/initialization.md` already names
this as the top starting-point failure ("0 is a domain error for
`log`, `/`, and friends"). Today the only answers pounce has are
bound pushing (bounds only, not general inequalities) and asking the
user to reformulate. Nothing in the repo tracks this.

**Cost.** Small and self-contained on the line-search path: a slack
reset plus a guard, both inside the existing accept/reject loop, all
behind an option. The caveat is honest advertising — it constrains
inequalities only, and needs a feasible start, so a model whose
difficulty is an *equality* domain error is not served.

### 3.3 Affine-scaling initial point (§6, Gertz–Nocedal–Sartenaer)

**What it is.** At the user's `x₀`, compute the affine-scaling step by
setting `μ = 0` in the primal-dual system (3.4). Then reset the slacks
and inequality multipliers componentwise to the magnitudes of that
step, leave `x` and `y` alone, and set `μ₁ = s₁ᵀz₁ / m`. One further
trick: evaluate the **first** Hessian of the Lagrangian at `z₀`, not
`z₁` — a large `z_i` against an indefinite `∇²c_i` manufactures
indefiniteness that has nothing to do with the problem.

**Why pounce should care.** pounce ports Ipopt's initializer:
`bound_push`/`bound_frac` clamping, then `y = 0` or least-squares,
`z = v = 1.0`. `docs/src/initialization.md` is candid that this is the
most common reason a good starting point does not behave like one —
but that page is about the *primal* clamp; the fixed `z = 1.0` is the
half nobody instruments. KNITRO's strategy respects the user's `x₀`
by construction (it never moves `x`) and fixes only the duals and
`μ`, which is where the arbitrary values are.

**Cost.** One extra RHS against a factorization the first iteration
already forms. But it *is* a trajectory change on every solve, so per
`CLAUDE.md` it needs `scripts/sweep-fixtures.sh` against a baseline
before merge — that, not the code, is the work. Ship it behind an
option first (KNITRO 5.0 itself made it INTERIOR/DIRECT-only).

### 3.4 Gradient-only Hessian information (§6)

KNITRO takes Hessian-vector products three ways: user-supplied,
finite differences of gradients of the Lagrangian (KNITRO drives it,
user supplies gradients only), or quasi-Newton. pounce's `TNLP` has
`eval_h` and nothing else; `gradient_approximation` and
`jacobian_approximation` are registered-but-unimplemented
(`unimplemented_options.rs:116`), so the only no-Hessian answer is
L-BFGS.

This is Level B of `interior-cg-matrix-free.md` §3.1 and depends on
the same evaluation-layer addition (`.nl` reader + CUTEst FFI
exposing `Hv`). Worth noting that the FD-of-gradients variant needs
*no* new user API — only gradients — and so could land on the
existing direct path, independent of Interior/CG.

### 3.5 SLQP as an alternative active-set path (§4)

KNITRO's ACTIVE decouples what pounce's SQP fuses: an LP (4.3),
solved by simplex, identifies the working set; an EQP (4.6) over that
working set, solved by projected CG, computes the step; the total step
is a Cauchy step plus a segment toward the EQP point. The stated
motivation is scaling — general QP subproblems cap problem size, and
second derivatives are awkward to put in an SQP.

pounce has both ingredients (`pounce-convex/src/simplex.rs`,
`pounce-qp`). But pounce's SQP exists for *warm-started sequences*,
where it measures well (0.02–0.17× cold, `warm-start-benchmark.md`),
and that is a different objective than SLQP's. File as "understood,
not needed", unless cold large-scale active-set solves become a
target.

Two pieces of §4 are separately reusable regardless:

- The **per-iteration penalty update** (Algorithm Penalty Update):
  choose `ν` every iteration by re-solving the LP with `ν` increased
  in steps of 10 until the linearized violation drops enough, rather
  than holding `ν` fixed and reacting to poor feasibility progress.
  The paper reports this pays for the extra LPs, which warm-start
  cheaply. `pounce-l1penalty` is fixed-`ρ`-at-construction with
  dynamic `ρ` deferred to Phase 3 of pounce#10 — this is a concrete
  recipe for that phase.
- **Penalty-based infeasibility detection**: `ν → ∞` is the natural
  signal, which complements the IPM-side test pounce already ships.

### 3.6 Automatic reduction to special problem classes (§6)

KNITRO reduces automatically: unconstrained → Newton-CG with trust
regions; **square nonlinear systems → Levenberg–Marquardt** (compute
the normal step only), or line-search Newton on `‖residual‖₂` with LM
fallback when the Jacobian is singular.

pounce auto-routes LP/QP (ahead of KNITRO 5.0 there) but does nothing
for square systems — and pounce *ships frontends that generate them*:
`docs/src/ode.md`, `dae.md`, `bvp.md`. Today those go through the
general IPM with a null objective. Most of the LM path is downstream
of the composite-step note (the normal step *is* the LM step), so the
sequencing is: composite-step first, this nearly free afterwards.

### 3.7 Merit function as a filter alternative (§3.3)

KNITRO uses `φ_ν = f - μ Σ log s_i + ν‖(c_E, c_I - s)‖` with `ν`
re-chosen every iteration so the model decrease is `≥ ρ ν ×` the
linearized-constraint decrease (3.17), and — for the line-search
variant — a `σ` switch (3.18) that drops the quadratic term when the
model is nonconvex, guaranteeing a descent direction. The paper
argues (against Gurwitz/Overton and citing Wächter–Biegler's own
Table 2) that this is as tolerant as a filter.

pounce is filter-only on the IPM path; the penalty-line-search knobs
inherited from Ipopt are in the unimplemented table. Lowest priority
of anything here: it is a pure trajectory change with no failure mode
it uniquely fixes, and per `CLAUDE.md` that is the most expensive kind
of change to justify.

## 4. Where pounce is ahead of the 2006 paper

Not the question asked, but it bounds the comparison: conic/SDP/SOS
global optimization, presolve (FBBT + bound tightening), post-optimal
sensitivity, automatic LP/QP routing, infeasibility *refutation*, and
the WASM/browser and Pyomo/GAMS surfaces have no counterpart in
KNITRO 5.0.

## 5. Suggested order

1. **Crossover** (§3.1) — both endpoints exist; opt-in; unblocks
   IPM→SQP warm starts and firms up the active set that sensitivity
   already depends on.
2. **Feasible mode** (§3.2) — smallest change here, aimed straight at
   pounce's documented top starting-point failure.
3. **Affine-scaling initial point** (§3.3) — cheap code, but budget
   the fixture sweep, not the patch.
4. **FD-of-gradients Hessian-vector products** (§3.4) — the half of
   Interior/CG Level B that needs no new user API.
5. Everything else follows the composite-step / Interior/CG notes.

Items 1, 2 and 4 are additive and opt-in: they do not move the default
trajectory, so they do not need a baseline sweep. Item 3 does. That
distinction is the real effort ranking.
