# pounce — vision / positioning

> Draft for discussion. The goal is a statement that says *where pounce sits*
> in the optimization-software landscape and *why it is different*, not a
> feature list. Three candidate framings below, then the supporting pillars
> and the one-liners they roll up into.

---

## The one-sentence version (lead candidate)

**pounce is one pip-installable optimization stack — LP through MINLP — built
to live inside modern ML and agent pipelines: differentiable where you need a
solver in the loop, constraint-guaranteeing where you need the answer to be
*feasible*, and legible to the LLMs and agents that increasingly drive the
modeling.**

---

## Why now — the gap pounce fills

The optimization-software world is split into camps that don't talk to each
other:

- **Classical solvers** (Ipopt, the commercial MI(N)LP engines) are fast and
  trustworthy but live behind C/Fortran ABIs and file formats. They were built
  before autodiff frameworks and before LLMs, and they treat the solve as a
  black box you call once and read the log of.
- **Differentiable-optimization layers** (cvxpylayers, theseus, the
  implicit-diff toolkits) plug a solver into JAX/PyTorch, but each covers a
  narrow problem class (usually convex QP/cone programs), ships its own
  numerics, and stops at the boundary of what its backend can express.
- **Modeling layers** (Pyomo, JuMP, CVXPY) are great for humans authoring
  models, but the solver underneath is still an opaque dependency you install
  separately and debug by hand.

pounce's bet is that these stop being separate concerns. One numerical
backbone should:

1. **span the whole ladder** — LP, QP, SOCP, SDP, exp/power cones, general
   NLP, and certified-global nonconvex — so a project never hits a wall where
   the problem class outgrew the tool;
2. **be differentiable as a first-class mode** — the solver is a layer you can
   put *inside* a learned model and backprop through, not just a thing you call
   at the end;
3. **guarantee feasibility** — a differentiable layer whose forward pass is a
   real interior-point solve returns a point that *satisfies the constraints*,
   which a learned approximator can't promise;
4. **be legible to agents** — the same diagnostics a human reads are exposed
   over MCP, so an LLM can author, run, and *debug* a model end to end.

---

## The four pillars (what makes the claim true today)

### 1. One roof: LP → MINLP, `pip install`, pure Rust
- `pip install pounce-solver` gets the whole family: the Ipopt-faithful NLP
  core, the convex/conic IPM (`pounce-convex`), SOS/Lasserre global, and the
  spatial branch-and-bound global solver (`pounce-global`).
- Pure Rust by default — no Fortran, no HSL, no system BLAS. One wheel, every
  platform, reproducible. This is the thing that makes "one roof" not just a
  slogan: there is genuinely one numerical backbone, not a meta-package
  shelling out to six binaries.
- `auto` routing classifies a problem and sends it to the right solver, so the
  "ladder" is invisible until you need to reason about it.
- The discrete top of the ladder is [discopt](https://github.com/jkitchin/discopt):
  a MINLP modeling language + spatial branch-and-bound that uses pounce as its
  primary NLP backend. Co-designed rather than plugged in — warm state, dual
  bounds, infeasibility certificates, the shared AD/problem IR, and the debug
  surface flow through the B&B tree instead of being rebuilt per node — so
  pounce+discopt behave like *one MINLP engine*, not a B&B loop dispatching to a
  generic solver. See `dev-notes/discopt-pounce-integration.md`.

### 2. Differentiable optimization that guarantees constraints
- `pounce.jax`: `from_jax` builds a solver problem straight from traced
  `f(x)`, `g(x)`; `solve` is wrapped in `jax.custom_vjp` so `jax.grad` flows
  through a solve via the implicit-function theorem on the KKT system.
- `QpLayer` / `solve_qp` / `solve_socp`: differentiable conic layers whose
  forward pass is a *real* IPM solve — the returned point is feasible by
  construction, not a learned projection that's "close." This is the headline
  for ML: a constraint layer you can trust.
- Built for the loop, not the one-shot: warm starts, factor reuse across a
  path (`PathFollower`), batched/parallel solves, sparse colored AD so the
  derivative cost scales with structure, not dimension.
- **Framework-agnostic by construction.** The differentiable layer is *one Rust
  IPM with a KKT-based implicit backward* — the autodiff framework is just a
  frontend over it. JAX is the first; PyTorch is a thin adapter, not a rewrite
  (the solver core and the implicit-function-theorem math don't change — only
  the array namespace and the `custom_vjp`↔`autograd.Function` wrapper do). This
  is the "one roof" thesis extended from problem classes to autodiff
  frameworks, and it's where cvxpylayers/theseus-style projects ship *separate*
  per-framework numerics while pounce ships one backbone under both. Tracked in
  [#109](https://github.com/jkitchin/pounce/issues/109).

### 3. Native to ML pipelines (JAX today, PyTorch next)
- x64-correct, JIT-compatible, vmap-aware. The integration is designed around
  how JAX actually composes (custom batching rule rather than lifting an impure
  callback), not bolted on.
- A PyTorch frontend ([#109](https://github.com/jkitchin/pounce/issues/109))
  mirrors the same surface — and is *smaller* to build, because PyTorch's eager
  mode drops the `pure_callback`/shape-declaration machinery JAX's traced model
  forces. Reaching the PyTorch-first half of the ML/research audience is mostly
  binding work, not new numerics.
- The target user is someone building a model where *part* of the forward pass
  is "solve this optimization exactly" — inverse problems, control/MPC layers,
  structured prediction, physics- or constraint-informed learning.

### 4. Legible to agents and LLMs (the differentiator)
- An interactive solver **debugger**: break into a live solve, inspect the
  iterate (primals, duals, KKT residuals, μ, inertia), sweep/multistart/replay.
- The same diagnostics exposed over **MCP** (`pounce-studio`), so an LLM agent
  can analyze a model, run it, read the convergence trace, and explain *why* it
  stalled — closing the loop from "agent writes a model" to "agent debugs the
  solve." Few, if any, classical solvers were designed to be driven this way.
- Signed solve receipts (`pounce verify`) — verifiable provenance for an
  answer, which matters when an agent (not a human) is the one trusting it.

---

## For teaching & research (the legibility pillar, pointed at people)

The same introspection that makes pounce legible to agents makes it a teaching
and research instrument no other solver can match. Plot it on two axes —
**introspectable internals** × **LLM-grounded explanation** — and the quadrant
pounce occupies is empty: classical solvers (Ipopt, SNOPT) print a log wall with
no live debugger and no LLM; commercial engines (Gurobi, BARON) are black boxes
by design; modeling layers (CVXPY, Pyomo) leave the solver opaque; and toy
teaching solvers aren't faithful to a production algorithm, so nothing transfers.
pounce is a **faithful production algorithm** (the Ipopt port — skills transfer)
that is **fully introspectable** and **explained by an LLM grounded in the real
trace and the literature**.

- **Education** — a glass-box IPM students watch *run* (μ, inertia, filter,
  restoration), a TA-over-MCP that reads *their* trace and explains the stall in
  algorithm terms with a citation, a zero-setup classroom (`pip install`, pure
  Rust, no licenses), and assignments graded on the *process* (the signed,
  reproducible solve report), not just the final number.
- **Research** — the iteration trace as a reproducible dataset, a hackable
  faithful baseline to perturb (swap a barrier rule and A/B it in one readable
  Rust codebase), an LLM that drives the MCP surface to *run and write up*
  experiments, and one diagnostic lens across NLP / conic / global / MINLP.

This is publishable in its own right — an LLM-drivable interactive debugger for
interior-point methods as a pedagogical and research instrument. Full treatment
in `dev-notes/education-research.md`.

---

## Taglines to choose from

- *"From LP to MINLP, in your ML pipeline and your agent's hands."*
- *"The solver that's differentiable, feasible, and legible — under one pip
  install."*
- *"One numerical backbone for the whole optimization ladder — built for the
  era of differentiable programs and AI agents."*
- *"Optimization that ML can backprop through, agents can drive, and you can
  trust to be feasible."*

---

## What we are *not* claiming (keep it honest)

- Not (yet) competing on raw speed with mature commercial MI(N)LP engines.
- "MINLP under one roof" is the *trajectory*: NLP + convex/conic + certified
  global B&B are here; the integer side is the spatial-B&B path maturing toward
  general MINLP. State it as direction, not a finished checkbox, until the
  mixed-integer story is fully wired.
- Differentiable-everything is real for the convex/QP/NLP layers; be precise
  about which classes have the `custom_vjp` path today.
- "Any autodiff frontend" is **JAX today, PyTorch tracked** ([#109](https://github.com/jkitchin/pounce/issues/109)),
  not both-shipping. The architectural claim (one framework-agnostic core) is
  true now; the PyTorch *binding* is roadmap. Don't imply a shipped PyTorch
  package until the adapter lands.
