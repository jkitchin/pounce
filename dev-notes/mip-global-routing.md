# Mixed-integer & global optimization — design note

**Status: design only.** No code changes yet. This note captures the
architecture for adding mixed-integer (MILP / MIQP / convex-MINLP) and
deterministic global (nonconvex NLP / MINLP) optimization on top of the
LP/QP routing already designed in
[`lp-qp-routing.md`](./lp-qp-routing.md). It assumes that note's LP/QP
plan has landed: a `pounce-convex` IPM family, the `pounce-qp`
active-set solver, and the `solver_selection` dispatch seam.

Three scope decisions drive everything below. The first two depart from
the LP/QP note's stated boundaries; the third is the *thesis* — the
reason to build any of this.

1. **Pure Rust, no wrapping.** The LP/QP note's escape hatch — "wrap
   HiGHS / SCIP / OSQP / BARON behind the dispatch layer" — is *off the
   table*. Every algorithm here is in-house. This re-promotes simplex
   from "out of scope, use HiGHS" to a *required dependency* (see
   "Simplex is back" below). The payoff is **distribution**: no Fortran,
   no HSL, no system BLAS, no C++ solver to build means a single
   `pip install` of a `manylinux` / macOS / Windows wheel — it works in
   Colab, locked-down CI, and air-gapped clusters out of the box. That
   is a direct adoption edge over the differentiable-convex incumbents
   (cvxpylayers needs the SCS / diffcp C stack), and it matters most for
   exactly the Python/ML audience this note targets.
2. **Nonconvex global optimization is in scope.** The LP/QP note
   explicitly excludes it ("Use BARON / Gurobi-nonconvex",
   lp-qp-routing.md "Nonconvex QP / global optimization"). This note
   reverses that: deterministic global optimization via spatial
   branch-and-bound becomes a first-class POUNCE algorithm family.
3. **Differentiability, JAX, and Python are the point — not a
   follow-on.** The goal is *not* to beat Gurobi/BARON on the Mittelmann
   sheets. It is **vertical integration**: a pure-Rust, JAX-native,
   `vmap`-batched *differentiable* solver stack spanning LP → QP → NLP →
   MIP → global, so that any of these solves can sit inside an ML model
   and pass gradients. This is the thing that does not exist anywhere —
   cvxpylayers / diffcp is convex-only; there is no differentiable MINLP
   or global-optimization layer, commercial or open. It is a **hard
   requirement on every solver**, not a fast-follow (see
   "Differentiability and the JAX/Python surface" below).

## Positioning

Four claims, in priority order. Each later one supports the one above it.

1. **Differentiability is the primary differentiator.** The reason to
   reach for POUNCE rather than Gurobi / BARON / HiGHS is that the solve
   is a differentiable, batched JAX primitive — a layer you can train
   through. No other solver across this problem class (MIP, MINLP,
   global) offers this.
2. **Competitive performance is the credible reason to *trust* it.**
   Differentiability is worthless if the forward solve is a toy. The bar
   is "good enough that a practitioner would pick it on the merits even
   without the gradient" — credible on MIPLIB / MINLPLib, not
   embarrassing next to HiGHS / SCIP. Performance is *table stakes that
   make the differentiator believable*, not the headline.
3. **The architecture gives a long runway for performance.** The
   trait seams (NodeSolver, Brancher, Relaxation, DiffHandoff) decouple
   *algorithm quality* from the *differentiable surface*. Better cuts,
   branching, pricing, relaxations, and parallel B&B all land behind
   stable interfaces without touching the JAX-facing API. POUNCE does
   not need to be fastest on day one; it needs a design where every year
   of work closes the gap and *none of it breaks the gradient*. The
   performance ceiling is high and approached incrementally.
4. **Pure Rust makes distribution frictionless.** `pip install`, no
   system dependencies, every platform — the lowest-friction path into a
   Python/ML workflow, and a standing advantage over C/C++/Fortran-backed
   competitors. (Decision 1.)

The synthesis: *lead with the capability nobody else has, back it with
performance good enough to trust, on an architecture that keeps
improving without API churn, delivered as a one-line install.*


## Relationship to discopt — division of labor, not duplication

[`discopt`](https://github.com/jkitchin/discopt) (same author, same
EPL-2.0 license) is a hybrid MINLP solver that **already implements much
of this thesis**: a Python-modeled, JAX-differentiated, Rust-backed
spatial branch-and-bound for MINLP/global, with a McCormick envelope
library (21 functions incl. NN activations), piecewise-McCormick, αBB,
AMP (adaptive multivariate partitioning), FBBT/OBBT presolve, and KKT
implicit-differentiation sensitivity. It is **prior art that validates
the design** — and it is what POUNCE composes *with*, not against. This
note's `pounce-mip` / `pounce-relax` are therefore **not a from-scratch
reimplementation**; the plan is reframed as the Rust/performance layer
under discopt's modeling/orchestration layer.

The dividing principle is **where the work is best done**:

| Concern | Home | Why |
|---|---|---|
| Modeling, problem setup, user intent | **discopt (Python)** | Users model in Python; the algebraic API, GDP, NN embedding, ONNX import, DAE collocation live here |
| **Modeling-language presolve** — reformulation that needs symbolic structure / user intent | **discopt (Python)** | Big-M / disjunction handling, NN-embedding reductions, expression-level simplification: best done where the model semantics are still visible, before lowering to numerics |
| B&B orchestration (tree, branching, fathoming) | **discopt today**, optionally lifted to Rust later | Already working; the hot loop is the *node solve*, not the tree bookkeeping |
| **Performance-heavy numeric presolve** — FBBT sweeps, OBBT LPs, activity/bound reductions, coefficient strengthening | **pounce (Rust)** | Thousands of repetitions per solve; this is the `pounce-presolve` + `pounce-convex` presolve merged in PR #70 |
| Node relaxation solves (LP/QP/NLP/conic) | **pounce (Rust)** | The wall-clock-dominant inner loop; pure-Rust, warm-started, differentiable |
| Single-problem NLP backend | **pounce (Rust)** | Directly replaces discopt's `ripopt` — see below |

### Two presolve layers, by design

This is the heart of why the two projects fit. Presolve is not one thing
done in one place — it is **two layers** that belong in two languages:

1. **Symbolic / modeling-language presolve (discopt, Python).** Done
   once, on the rich model representation, while user intent and
   structure (disjunctions, indicator constraints, NN blocks, problem
   templates) are still legible. Lowering loses this information, so it
   *must* happen upstream. discopt is the right home.
2. **Numeric presolve (pounce, Rust).** Done on the lowered numeric
   problem and repeated at every B&B node — FBBT propagation, OBBT,
   activity-bound reductions, coefficient strengthening. Hot-loop,
   allocation-sensitive, pure-Rust. This is the "highest-ROI performance
   lever" section below, and it is what POUNCE owns.

The handoff is one-directional and clean: discopt does symbolic
reductions, lowers to a numeric problem (`.nl` or a direct Rust
problem), and POUNCE does the numeric presolve + solves from there. No
reduction is implemented twice.

### ripopt → pounce

discopt's `ripopt` is "a custom Rust IPM via PyO3, distinct from Ipopt"
— its fastest single-problem NLP backend. POUNCE *is* a mature pure-Rust
Ipopt port (filter line search, restoration, `feral` linsol,
`pounce-sensitivity`, a `custom_vjp` JAX layer, and the convex-conic
family landing now). The natural consolidation: **POUNCE replaces
ripopt** as discopt's single-problem NLP/LP/QP/conic backend. discopt
gains a more complete, better-tested solver and the conic family for
free; POUNCE gains a real downstream consumer that drives its
requirements. This is the first concrete integration step and removes
the most obvious duplication (two Rust IPMs).

### What this note becomes

Read `pounce-mip` / `pounce-relax` throughout this note as **the
pure-Rust performance layer that discopt orchestrates**, not a competing
MINLP product. The differentiable JAX surface (Seam 5, the
"Differentiability" section) is the shared contract: discopt already
differentiates via KKT implicit differentiation, and POUNCE's
`custom_vjp` layer is the same mechanism — so the gradient flows through
the language boundary, not just within one side of it.

## The central reframe

Mixed-integer programming is a **branch-and-bound shell**, not a new
numerical kernel. Every node of a B&B tree solves a continuous
relaxation POUNCE already has (or will have once LP/QP lands). The new
work is combinatorial orchestration — a search tree, branching,
incumbent / gap bookkeeping, cuts — not a new factorization.

Deterministic global optimization is the *same B&B shell* extended in
two ways: it branches on **continuous** variables (spatial
branch-and-bound), and its lower bound comes from a **machine-constructed
convex relaxation** of the nonconvex problem rather than from the
problem itself. Integer B&B and spatial B&B are therefore **one engine**,
differing only in (a) what they branch on and (b) where the node lower
bound comes from. Convex MINLP-global runs both branching modes at once
— the BARON / Couenne architecture.

## Local and global: one solver, tiered guarantees

POUNCE is a **mixed local/global solver** — not two separate products.
The user chooses where to sit on a guarantee-vs-cost spectrum via
`solver_selection`, and the tiers share machinery rather than duplicating
it:

| Tier | What it does | Guarantee | Cost |
|---|---|---|---|
| `local` (default for NLP) | one IPM-NLP solve (`pounce-algorithm`) | local optimum, no certificate | one solve |
| `multistart` | several local solves from sampled starts; keep best | better global odds, still no certificate | k solves |
| `global` (opt-in) | spatial B&B over machine-built convex relaxations | numerical global optimum + gap certificate | a tree of solves |

Three things make this *one* solver, not a bundle:

1. **Global's upper bound *is* the local solver.** The `IncumbentSearch`
   seam inside spatial B&B (Phase 4) gets its feasible points from
   multi-start `pounce-algorithm` NLP solves. "Local solve" is therefore
   a *component* of the global solver, not a throwaway path — building
   the local tier well builds half the global tier. Likewise `multistart`
   is just the upper-bounding loop run standalone, without the
   lower-bounding tree.
2. **The default never surprises.** A nonconvex problem with no explicit
   `global` request gets the fast local solve it gets today — no silent
   100× slowdown. Global is always a deliberate opt-in (the
   convexity-undecidability dodge in "Limitations").
3. **Differentiability is identical across tiers.** Seam 5 differentiates
   through the *returned* solution's KKT — it does not know or care
   whether that point came from a single local solve or the winning leaf
   of a global tree. Same `DiffHandoff`, same `custom_vjp`, same `vmap`.
   So `pj.solve` (local), `pj.solve_global`, and `pj.solve_mip` are the
   same backward pass over different forward passes.

Per problem class this is *already* mixed: convex-MI is "global over the
integers (exact B&B), exact on the continuous relaxations"; nonconvex
MINLP is "global over integers, **and** local-or-global on the continuous
part per the chosen tier"; a pure nonconvex NLP is local-or-global by
choice. The integer dimension is always handled by exact B&B; the
*continuous* nonconvexity is the axis the local/global tier selects.

## The two consistency wins

The LP/QP note's decision principle puts an algorithm family *in-house*
when it scores strongly on two consistency wins (lp-qp-routing.md
"Decision principle"). The conic-IPM family is anchored on the **sparse
symmetric augmented system** (feral). This note adds a second, equally
clean anchor for the global-optimization family: the **factorable
expression DAG and interval arithmetic** that FBBT already consumes.

Deterministic global optimization in the McCormick "factorable
programming" tradition walks the *same* expression tree FBBT walks
(`crates/pounce-cli/src/nl_reader.rs` `Expr`, a DAG with `Cse`
sharing), but instead of propagating intervals it builds convex/concave
relaxations term by term. The assets already in the repo:

| Global-opt component | Reuses today |
|---|---|
| Factorable decomposition (split exprs into unary/binary ops + aux vars) | `crates/pounce-presolve/src/auxiliary.rs` reformulation pipeline; the `Expr` DAG with `Cse` sharing |
| Interval bounds on every DAG node | `crates/pounce-presolve/src/fbbt/interval.rs` + `forward.rs` / `reverse.rs` — shipped |
| FBBT as cheap node bound-tightening | shipped, issue #62 (`docs/src/fbbt.md`) |
| Convex (LP/NLP) relaxation → valid **lower** bound | `pounce-convex` (IPM-LP/QP) + the new simplex |
| Local solve → **upper** bound / incumbent | `pounce-algorithm` IPM-NLP, today |
| OBBT (min/max each var over the relaxation) | the relaxation solver, looped |

So `pounce-relax` is "a new consumer of the expression DAG, the way
feral is a consumer of the augmented system." That is the architectural
argument that makes nonconvex global belong in-house and pure-Rust
rather than as a wrapped bolt-on — the same *shape* of justification the
conic family received, anchored on the symbolic graph instead of the
linear algebra.

## Decisions

1. **One unified branch-and-bound engine (`pounce-mip`).** Integer and
   spatial branching share the tree, node queue, incumbent, and gap
   machinery. Branching mode and lower-bound source are pluggable.
   Resists building two near-duplicate B&B drivers.
2. **Simplex is deferred to the global-optimization work, and its LU
   lives in feral — not a separate crate.** Pure-Rust means no HiGHS to
   wrap for warm-started node LPs, so a dual simplex *is* eventually
   needed — but **only where it pays off: warm-started LP-relaxation
   nodes inside spatial B&B (global)**. The convex-MI deliverable
   (Phases 0–2) does *not* need it: those nodes solve on the existing
   symmetric path (IPM `pounce-convex`, active-set `pounce-qp`). This
   matches the repo history — a simplex (`Harris two-pass + EXPAND`) was
   built and then deliberately *stripped* from the merged LP/QP path
   (PR #70), i.e. parked until global needs it. Two consequences:
   - **No `pounce-lu` / `pounce-simplex` crates.** The robust sparse LU
     (Bartels-Golub / Forrest-Tomlin updates) is landing **inside
     feral**, alongside its symmetric LDLᵀ — one sparse-factorization
     backend serving both the symmetric (IPM/QP) and unsymmetric
     (simplex) systems behind `pounce-linsol`. The simplex driver itself
     becomes a module in `pounce-convex`, not a standalone crate.
   - This **reconciles the old "Simplex is back" reversal with decision
     8** ("not chasing MILP"): you do not build the hardest numerical
     component on the convex-MI critical path. It arrives with global.
3. **The relaxation engine is its own crate (`pounce-relax`).** It is
   symbolic, not linear-algebra — factorable decomposition, McCormick
   envelopes, αBB, OBBT. It is the research-grade analog of
   `pounce-convex` and the heart of the global capability.
4. **MIQP is the front door, not MILP.** `pounce-qp`'s active-set
   solver warm-starts natively node-to-node (qpOASES-style homotopy —
   lp-qp-routing.md "Active-set vs IPM-QP"), and there is no dominant
   open-source convex-MIQP solver to clear. MILP-as-such is the
   fiercely-contested class (HiGHS / Gurobi / COPT) and depends on the
   new simplex landing first.
5. **Convex MI is a hard prerequisite for nonconvex global.** Spatial
   B&B's node lower-bounding solves convex relaxations that are
   themselves LP / convex-NLP — i.e., the convex-MI machinery. You
   cannot build the global layer without first running a B&B tree over
   convex relaxations. The phasing below is therefore forced, not a
   soft preference.
6. **Differentiability is a hard requirement — definition of done, not
   a fast-follow.** A solver does not "land" until its `jax.custom_vjp`
   backward pass and its `vmap`/batched wrapper exist and pass tests.
   This is deliberate: differentiable-optimization projects fail when
   the backward pass is retrofitted onto a forward solver that wasn't
   designed to expose `(x*, λ*, multipliers, active set)`. Designing it
   in from the first commit of each phase keeps the residual hand-off
   (Seam 5 below) a first-class output of every solve. The cost is real
   — each phase carries its JAX work — and it is paid on purpose.
7. **LP/polyhedral relaxations for global.** Node lower bounds come from
   *linearized* McCormick/αBB envelopes solved as LPs on the dual
   simplex, not from convex-NLP relaxations. Three reasons, in priority
   order: (a) **one** warm-startable engine serves MILP nodes, MIQP
   relaxation LPs, and global nodes; (b) LP relaxations are themselves
   differentiable through the *same* KKT machinery, so relaxation-gap
   gradients come for free if ever wanted; (c) it matches Couenne, the
   EPL-2.0 reference. Cost: `pounce-relax` must emit linear cuts of the
   envelopes and re-linearize as boxes tighten. The hybrid (convex-NLP
   root bound, LP nodes below) is a Phase-6+ refinement, not the
   starting architecture.
8. **MILP is "degenerate MIQP," not a HiGHS challenger.** Pure-MILP
   works correctly via the zero-Hessian path; cut/heuristic effort
   concentrates where POUNCE differentiates (MIQP / MINLP / global /
   *differentiability*). The simplex still gets exercised hard — the
   global LP-relaxation path (decision 7) drives it — so it becomes good
   out of necessity, not out of chasing MIPLIB. Bounds are **numerical**
   (BARON/Couenne-style floating point); rigorous interval-validated
   bounds are explicitly *not* a goal — ML wants gradients, not
   certified optima.

## Architecture

### Crate layout

```
crates/
  pounce-relax/    # NEW, heavy: factorable decomposition, McCormick
                   #   envelopes, αBB for general C² terms, auxiliary-
                   #   variable reformulation. Builds a convex LP/NLP
                   #   relaxation from the Expr DAG. Reuses fbbt/interval.rs.
                   #   Symbolic analog of pounce-convex.
  pounce-mip/      # NEW: unified branch-and-bound engine — tree, node
                   #   management, incumbent, gap. Branching: integer
                   #   (pseudocost / reliability) AND spatial (continuous,
                   #   on the var driving the relaxation gap). Consumes
                   #   pounce-relax for bounds and pounce-{convex,
                   #   algorithm,qp} for node solves.
  ── existing / in flight ──
  pounce-convex/   # IPM-LP/QP + the conic family (merged, PR #70). Later
                   #   gains a dual-simplex MODULE (not a crate) for the
                   #   global LP-relaxation nodes — gated on the global work.
  pounce-qp/       # active-set QP — the MIQP node engine
  pounce-algorithm/# IPM-NLP — upper-bound / incumbent local solves
  pounce-presolve/ # bound-tightening, FBBT, BTF, DM, components, auxiliary
  feral/           # sparse symmetric LDLᵀ today; sparse LU (Bartels-Golub
                   #   / Forrest-Tomlin updates) landing now → the one
                   #   backend behind pounce-linsol for BOTH the symmetric
                   #   (IPM/QP) and unsymmetric (simplex) node systems.
```

There is no `pounce-lu` and no `pounce-simplex` crate: the robust sparse
LU lives in **feral** next to its LDLᵀ, and the simplex driver is a
later **module in `pounce-convex`**. Both are gated on the
global-optimization work (Phases 3–5), not the convex-MI deliverable.

Note `pounce-mip` is **not** a `pounce-linsol` consumer the way the
solver crates are — it consumes *solvers*, which in turn consume
`pounce-linsol`. It is a thinner, combinatorial crate, with
`pounce-relax` carrying the genuinely new numerical/symbolic weight.

### Dependency graph and phase gating

Arrows are "depends on / calls into". The two anchors at the bottom are
the existing consistency wins; everything new is a consumer of one or
both. `[Pn]` tags the phase that introduces each new crate.

```text
                         pounce-mip  [P1]
            ┌──────────────┬───────────────┬──────────────┐
            ▼              ▼               ▼              ▼
      pounce-relax       (simplex      pounce-qp      pounce-convex
         [P3]         module of         (existing,    (merged, PR #70)
            │         pounce-convex,    MIQP node)         │
            │         global-gated)         │              │
            ▼              └───────┬────────┴──────────────┘
       ┌─────────────────┐        ▼
       │ Expr DAG +       │   ┌──────────────────────────────────┐
       │ interval arith   │   │ feral behind pounce-linsol:        │
       │ (fbbt/, auxiliary)│  │  symmetric LDLᵀ (IPM/QP) +          │
       │  ── anchor #1 ──  │  │  sparse LU (simplex, landing now)   │
       └─────────────────┘   │  ── anchor #2 ──                    │
                             └──────────────────────────────────┘
   (also: pounce-algorithm IPM-NLP — incumbent upper bounds, anchor #2)
```

Note anchor #2 now hosts *both* factorization kinds in one backend: the
symmetric LDLᵀ that IPM/QP nodes use, and the sparse LU the simplex needs
— so the simplex, when it arrives with the global work, adds no new
linear-algebra crate, just a new factorization mode in feral.

Phase gating (what must exist before what):

```text
P0 plumbing ─▶ P1 B&B shell + MIQP ─▶ P2 cuts + MIP presolve
                                            │
                        convex-MI deliverable ◀──┘   (IPM/active-set
                                            │          node solves only —
                                            │          NO simplex, NO LU)
                                            ▼
                       P3 pounce-relax (McCormick/αBB) ──┐
                                            │            │
              feral sparse LU + simplex ────┤  (both     │
              module in pounce-convex       │   land here, with global)
                                            ▼            │
                                   P4 spatial B&B ◀───────┘
                                            │
                                            ▼
                                   P5 MINLP-global
```

Two crossbars now matter:
1. **Global gates on convex-MI:** P3–P5 cannot start until the convex-MI
   substrate (P1–P2) can run a B&B tree over convex relaxations.
2. **Simplex/LU gates on global, not convex-MI:** the dual simplex and
   feral's sparse LU arrive *with* the global LP-relaxation nodes — the
   convex-MI deliverable never touches them. This is the deferral
   recorded in decision 2, matching the PR-#70 strip of the built simplex.

### Dispatch and classification

Extends the `solver_selection` / `ProblemClass` seam from LP/QP
Phase 1. `ProblemClass` gains integer and nonconvexity flags; the
classifier reads integer-variable counts from the `.nl` header (the
fields `nl_reader::parse_header` currently skips — see
lp-qp-routing.md "NL-header inspection") and detects nonconvexity from
the Hessian-pattern / AST walk already used to split convex vs
nonconvex QP.

```rust
enum ProblemClass { Lp, ConvexQp, NonconvexQp, Nlp,
                    Milp, Miqp, ConvexMinlp, GlobalNlp, GlobalMinlp }
```

`solver_selection` gains `mip-bb` (integer B&B over convex relaxations)
and `global` (spatial B&B). `auto` routes integer problems to the
B&B engine and, when a nonconvex structural problem is detected and the
user has opted into global mode, to the spatial path. Default behavior
for a nonconvex NLP **without** opt-in stays exactly as today (a local
IPM-NLP solve) — global is never silently substituted, because it can
be orders of magnitude slower.

### What modeling languages see

- **AMPL / Pyomo (`.nl`):** integer variables already round-trip through
  the NL format. The CLI auto-detects integrality and routes to the B&B
  engine. Global mode is opt-in via
  `solver.options['solver_selection'] = 'global'`.
- **Python API (`pounce-py`):** add `pounce.solve_mip(...)` and
  `pounce.solve_global(...)` alongside the existing entry points.
- **C ABI:** follows the same `TNLP`-bridge pattern as the LP/QP entry
  points.

## The relaxation engine (`pounce-relax`)

The research-grade core. Standard factorable-programming stack:

1. **Factorable decomposition.** Walk the `Expr` DAG, introducing an
   auxiliary variable per intermediate operation so every constraint
   becomes a set of simple unary/binary defining constraints
   (`w = u·v`, `w = exp(u)`, …). Reuses `auxiliary.rs` reformulation
   machinery and the `Cse` sharing already in the DAG.
2. **McCormick envelopes.** Convex/concave under/over-estimators for
   bilinear products, univariate convex/concave terms, fractional and
   power terms. Tight on each variable box; tightened as boxes shrink.
3. **αBB.** For general twice-differentiable nonconvex terms, a convex
   underestimator via a quadratic perturbation using interval bounds on
   the Hessian eigenvalues (reuses the Hessian-pattern computation in
   `pounce-nlp`; the Adjiman–Floudas construction).
4. **Relaxation assembly.** The estimators plus the linear/aux defining
   constraints form a convex LP (or convex NLP) whose optimum is a valid
   **lower bound** on the nonconvex problem over the current box.
5. **OBBT.** Optimization-based bound tightening: minimize / maximize
   each variable subject to the relaxation to shrink the box. Expensive
   (two LPs per variable); applied selectively. FBBT is the cheap pass
   that runs every node.

Upper bounds come from multi-start local NLP solves
(`pounce-algorithm`); the global gap is `upper − lower`, and spatial
branching subdivides the box on the variable contributing most to the
relaxation gap until the gap closes to tolerance.

### Conic vs. factorable relaxations — and why the branch makes this free

The factorable-McCormick stack above is the *general* lower-bounding
route (it handles `exp`, `log`, trig, arbitrary C² terms). But for the
**quadratic** structured classes — nonconvex QP, QCQP, and the quadratic
parts of MINLP — a second, often *tighter* lower bound is available:
**conic (SDP/SOCP) relaxation**. And PR #70 already merged exactly the
cone machinery this needs (SOCP with NT scaling, the HSDE driver,
composite cones), so it costs little to add as an alternative relaxation
backend.

The contrast, for the global lower bound on a quadratic term:

| | Factorable / McCormick-LP | Conic (SDP / SOCP) |
|---|---|---|
| Generality | any C² expression | quadratic (QP/QCQP) only |
| Bound tightness on quadratics | looser (separable envelopes) | tighter (Shor SDP dominates RLT — Anstreicher 2009) |
| Node solver | the dual simplex (decision 7) | `pounce-convex` cone IPM (merged, PR #70) |
| Warm-start in B&B | excellent (simplex basis) | weaker (IPM reseed) |
| Differentiable | yes, via simplex KKT | yes — the branch already ships a `cone-aware OptNet` layer |
| Cost to add | the planned `pounce-relax` | mostly *reuse* of the landing cone solver |

So they are **complementary, not competing**, and the `Relaxation` seam
(Seam 3) is exactly where the choice plugs in — a `FactorableRelaxation`
(LP, general) and a `ConicRelaxation` (SDP/SOCP, quadratic-only)
implement the same trait, and dispatch picks per problem structure or
runs both and takes the tighter bound. The Shor/RLT and
completely-positive reformulations (Burer 2009 lineage) are the standard
constructions; SOCP relaxation is the cheaper, weaker cousin that
warm-starts better inside the tree.

**Phasing implication.** Conic relaxations are a *Phase 4+ enhancement*,
not a prerequisite — `pounce-relax`'s factorable-LP path (decision 7) is
the baseline that makes the global solver work at all. The conic
backend is the natural "tighten the bound on quadratic structure"
follow-on, and it is unusually cheap here precisely because the cone IPM
and its differentiable layer are landing for the convex-conic family
anyway. Crucially, it does **not** reverse decision 7: LP relaxations
stay the default node engine (warm-start throughput); conic is the
opt-in tighter bound for hard quadratic instances where node count, not
per-node cost, dominates.

## Presolve: the highest-ROI performance lever

Presolve is the single largest practical speedup in modern MIP
(Achterberg & Wunderling 2013 attribute order-of-magnitude factors to
it). For the positioning where *competitive performance is the
table-stakes that make the differentiator trustworthy* (Positioning #2),
presolve is where credibility is bought most cheaply — so it gets a
section, not a sub-bullet.

### Build on what already exists — two presolve systems, not zero

Far more presolve already ships or is in flight than a from-scratch plan
would assume. There are **two** systems, and MIP builds on both:

- **`pounce-presolve` (NLP / structural):** bound tightening, FBBT
  (issue #62), redundant-row removal, BTF, Dulmage–Mendelsohn, connected
  components, auxiliary-equality preprocessing (issue #53), incidence /
  matching.
- **Convex presolve in `pounce-convex` (PaPILO-style, merged in PR
  #70):** LP/QP activity-bound
  reductions, free-column singleton substitution, forcing constraints,
  parallel-row detection, dominated columns, dual bound tightening, all
  iterated to a fixpoint — with **transaction-stack postsolve** and
  `qp_presolve` / `presolve_conic` toggles. Pure-Rust, rayon-parallel,
  PaPILO-informed.

That same branch is executing far more than LP/QP — IPM-LP/QP (Mehrotra),
an HSDE driver, full SOCP (NT scaling, `solve_socp`), exponential-cone
groundwork, **and differentiable convex/cone layers** (`cone-aware
OptNet`, JAX QP matrix gradients P/G/A). MIP/global inherits all of it.

### Two distinct presolve phases in B&B

- **Root presolve (once, aggressive).** Run both pipelines on the
  original problem before the tree, adding MIP-specific reductions on top:
  integer-aware bound tightening (round fractional bounds on integers),
  coefficient strengthening, **probing** (fix a binary, propagate —
  Savelsbergh 1994), clique / implication detection, GCD reductions on
  all-integer rows, and (later) symmetry detection. Cost is amortized
  over the whole tree.
- **Node presolve (cheap, repeated thousands of times).** Propagate the
  branching-tightened bounds: FBBT every node (shipped), light probing,
  *selective* OBBT (Gleixner et al. 2017). The engine loop already
  exposes the hook (`fbbt_and_obbt(&mut node, &relax)`).

### Presolve under the differentiability requirement — already half-solved

Presolve transforms the problem, so the backward pass must map gradients
back through every reduction to the *original* variables. The
**transaction-stack postsolve merged in PR #70 is exactly this
mechanism for the solution**; MIP extends it from solution-mapping to
gradient-mapping, and the merged differentiable cone layer
(`cone-aware OptNet`) already demonstrates differentiating through a
presolved-then-solved convex problem. Reductions that fix or substitute
a variable contribute a known/zero gradient (pinned, like integers and
active bounds in Seam 5). Per decision 6, a reduction does not land until
its gradient pass-through is defined — but postsolve means the hard part
is mostly built.

## Interfaces and crate skeletons

The whole design rests on four trait seams. They are sketched here so
the phasing below builds against fixed interfaces, and so the
"one engine, pluggable parts" claim is concrete rather than aspirational.
Names are illustrative, not final.

### Seam 1 — `NodeSolver`: warm-startable relaxation solve

The single integration point with the continuous solvers. Thin adapters
implement it over `pounce-qp` (active-set), `pounce-convex` (IPM, and
later its dual-simplex module), and `pounce-algorithm` (NLP). The
associated `WarmState` is what threads node-to-node — a simplex basis,
an active set, or a (symbolic) factor. The engine is generic over
`S: NodeSolver`, so dispatch monomorphizes one B&B per solver arm; no
`dyn` and no object-safety constraint on the warm state.

```rust
pub trait NodeSolver {
    /// Carried parent→child: basis (simplex), active set (pounce-qp),
    /// or symbolic factor (IPM). `Clone` so siblings can both inherit.
    type WarmState: Clone;

    /// Solve the relaxation at this node (branching-narrowed bounds plus
    /// any cuts in scope), optionally warm-started from the parent.
    fn solve_node(
        &mut self,
        node: &NodeData<'_>,
        parent: Option<&Self::WarmState>,
    ) -> NodeResult<Self::WarmState>;
}

pub struct NodeData<'a> {
    pub var_lb: &'a [f64],   // branching narrows these
    pub var_ub: &'a [f64],
    pub cuts: &'a [CutRef],  // global cut-pool handles in scope (Phase 2)
    pub depth: u32,
}

pub enum NodeStatus { Optimal, Infeasible, Unbounded, IterLimit }

pub struct NodeResult<W> {
    pub status: NodeStatus,
    pub objective: f64,      // the node LOWER bound (relaxation optimum)
    pub x: Vec<f64>,         // relaxation solution (in relaxed-var space)
    pub warm: Option<W>,
}
```

### Seam 2 — `Brancher`: integer *and* spatial branching, one type

The unification claim made concrete: `BranchDecision` carries both
modes, so the same engine drives MILP, spatial global, and MINLP-global
by swapping the brancher (or letting one brancher emit both kinds).

```rust
pub trait Brancher {
    /// `None` ⇒ node solution already feasible for the ORIGINAL problem
    /// (integer-feasible and, for global, within relaxation tolerance):
    /// a leaf / candidate incumbent.
    fn select(&mut self, node: &NodeData<'_>, x: &[f64], info: &ProblemInfo)
        -> Option<BranchDecision>;

    /// Feedback after children are bounded — pseudocost / reliability
    /// statistics (Achterberg–Koch–Martin). No-op for most-fractional.
    fn observe(&mut self, decision: &BranchDecision, child_bounds: [f64; 2]);
}

pub enum BranchDecision {
    /// x_j ≤ ⌊v⌋  |  x_j ≥ ⌈v⌉
    Integer { var: usize, value: f64 },
    /// box split at `point` on a continuous nonconvex variable
    Spatial { var: usize, point: f64 },
}
```

### Seam 3 — `Relaxation`: the `pounce-relax` boundary

What converts a (possibly nonconvex) problem into something a
`NodeSolver` can bound. For convex MI it is a near-identity wrapper that
just drops integrality; for global it is the factorable McCormick/αBB
construction over the `Expr` DAG. Same engine consumes both.

```rust
pub trait Relaxation {
    /// Re-derive envelope coefficients for a node's narrowed box
    /// (McCormick/αBB depend on the bounds). Cheap; called per node.
    fn tighten(&mut self, var_lb: &[f64], var_ub: &[f64]);

    /// The current convex relaxation, as a problem the node solver reads.
    /// LP for polyhedral relaxations; convex TNLP otherwise.
    fn as_problem(&self) -> Rc<RefCell<dyn TNLP>>;

    /// Drop the auxiliary variables introduced by factorable
    /// decomposition, mapping a relaxation point to original-var space.
    fn project(&self, x_relaxed: &[f64]) -> Vec<f64>;
}

/// Convex-MI path: no relaxation construction, integrality handled by
/// the brancher. `tighten` only updates bounds; `project` is identity.
pub struct PassthroughRelaxation { /* wraps the original TNLP */ }

/// Global path (Phase 3): factorable decomposition + envelopes.
pub struct FactorableRelaxation { /* Expr DAG, aux vars, McCormick/αBB */ }
```

### Seam 4 — `IncumbentSearch`: upper bounds / primal heuristics

Decouples "find a feasible point for the original problem" from the
bounding loop. MILP uses rounding / diving / RINS; global uses
multi-start local NLP via `pounce-algorithm`.

```rust
pub trait IncumbentSearch {
    fn improve(&mut self, node: &NodeData<'_>, relaxed_x: &[f64])
        -> Option<Incumbent>;
}
```

### Seam 5 — `DiffHandoff`: the differentiable backward pass

The seam that makes differentiability a definition-of-done (decision 6)
rather than a retrofit. When B&B finishes, the winning leaf is a
*continuous* problem — the original with integers fixed to the optimal
assignment (and, for global, with the box pinned to the winning basin).
Its KKT data is exactly what the existing `pounce.jax.solve` backward
consumes (`python/pounce/jax/_diff.py:128`): `(x*, λ*, mult_x_L,
mult_x_U)` plus the active set. **Integer variables differentiate like
active bounds** — `dx/dp = 0` on their columns, via the same
identity-augment trick the active-set handling already uses (the
pounce#73 fix). So the `SolveReport` from B&B must surface this hand-off
as a first-class output, and `pounce-sensitivity` consumes it unchanged.

```rust
/// Everything the implicit-function-theorem backward pass needs, emitted
/// by every solve (continuous or B&B-leaf). Fed to pounce-sensitivity
/// and, above it, the jax.custom_vjp residual.
pub struct DiffHandoff {
    pub x: Vec<f64>,              // primal solution
    pub lambda: Vec<f64>,        // constraint multipliers
    pub mult_x_lower: Vec<f64>,  // bound multipliers → active-set mask
    pub mult_x_upper: Vec<f64>,
    /// Variables pinned in the backward (dx/dp = 0): active bounds, AND
    /// the integer variables fixed at the optimal assignment.
    pub pinned: BitVec,
    /// Converged KKT factor of the fixed-integer / final continuous
    /// problem, reused across back-solves (pounce_linsol::Factorization).
    pub factor: Option<Factorization>,
}
```

The gradient this yields is **correct conditional on the discrete
solution being locally stable** — the optimal integer assignment and
active set do not change under the infinitesimal parameter perturbation.
That is the right and useful object for decision-focused learning when
the assignment is stable (most of training). It carries *no* signal
*through* the combinatorial switch itself; for learning that must flow
gradient through the discrete decision, a smoothing layer
(perturbed-optimizer / blackbox-combinatorial — see references) wraps
the solver on the JAX side as an alternative `custom_vjp` rule. **The
Rust core is unchanged either way**; smoothing is a Python/JAX concern.

### The engine loop

`pounce-mip` owns the tree, queue, incumbent, and gap; everything
problem-specific is behind the four seams above.

```rust
pub fn branch_and_bound<S, B, R, H>(
    mut solver: S, mut brancher: B, mut relax: R, mut heur: H,
    info: ProblemInfo, opts: &OptionsList,
) -> SolveReport
where S: NodeSolver, B: Brancher, R: Relaxation, H: IncumbentSearch
{
    // best-bound queue; incumbent = best original-feasible point so far
    while let Some(node) = queue.pop_best_bound() {
        if node.lower >= incumbent.value - gap_tol { continue; }  // prune

        relax.tighten(node.var_lb, node.var_ub);   // Seam 3: per-box envelopes
        fbbt_and_obbt(&mut node, &relax);          // reuse pounce-presolve
        let r = solver.solve_node(&node.data,      // Seam 1: warm-started solve
                                  node.parent_warm.as_ref());

        match r.status {
            NodeStatus::Infeasible | NodeStatus::Unbounded => continue,
            _ => {}
        }
        if r.objective >= incumbent.value - gap_tol { continue; }  // bound prune

        if let Some(inc) = heur.improve(&node.data, &r.x) {        // Seam 4
            incumbent.update(inc);
        }
        match brancher.select(&node.data, &r.x, &info) {          // Seam 2
            None => incumbent.update_from_leaf(&relax.project(&r.x), r.objective),
            Some(decision) => {
                let (lo, hi) = split(&node, &decision);   // integer or spatial
                lo.parent_warm = r.warm.clone();          // inherit warm state
                hi.parent_warm = r.warm;
                brancher.observe(&decision, [lo.lower, hi.lower]);
                queue.push(lo); queue.push(hi);
            }
        }
    }
    report_from(incumbent, global_lower_bound, gap)
}
```

The warm-state inheritance on the two `parent_warm` lines is the single
most performance-critical detail: it is why the MIQP-over-`pounce-qp`
front door (native active-set homotopy) is the cheapest first target,
and why IPM node solves need the symbolic-factor-reuse seam
(lp-qp-routing.md "Session-style factorization reuse") before they are
competitive node engines.

### Crate skeletons

```text
crates/pounce-mip/
  Cargo.toml            # deps: pounce-common, pounce-nlp, pounce-presolve,
                        #       pounce-convex, pounce-qp, pounce-algorithm,
                        #       pounce-relax
  src/
    lib.rs              # branch_and_bound<S,B,R,H> + the four seam traits
    tree.rs            # Node, best-bound / depth-first queue, gap accounting
    node.rs            # NodeData, NodeResult, NodeStatus
    branch/
      mod.rs            # Brancher trait, BranchDecision
      most_fractional.rs# Phase 1
      pseudocost.rs     # Phase 2 (reliability — Achterberg et al. 2005)
      spatial.rs        # Phase 4 (gap-driven continuous split)
    incumbent.rs        # IncumbentSearch trait + rounding/diving (Phase 2)
    cuts/               # Phase 2: pool + Gomory/MIR/cover separators
      mod.rs  gomory.rs  mir.rs  cover.rs
    adapters/           # NodeSolver impls (thin wrappers)
      active_set.rs  ipm.rs  nlp.rs   # simplex.rs added in Phase 4

crates/pounce-relax/    # the heavy symbolic crate (Phase 3)
  Cargo.toml            # deps: pounce-common, pounce-nlp, pounce-presolve (fbbt)
  src/
    lib.rs              # Relaxation trait, PassthroughRelaxation
    factorable.rs       # Expr-DAG → aux-var defining constraints
    mccormick.rs        # bilinear / univariate envelopes
    alpha_bb.rs         # αBB underestimator (interval Hessian eigenvalues)
    obbt.rs             # optimization-based bound tightening loop
    project.rs          # aux-var → original-var projection

# Phase 4 (with the global LP-relaxation nodes):
#   feral             — sparse LU (Bartels-Golub / Forrest-Tomlin) beside LDLᵀ
#   pounce-convex     — gains a dual-simplex module (bound-flipping ratio test)
# No pounce-lu / pounce-simplex crates.
```

## Differentiability and the JAX/Python surface

This is the thesis (scope decision 3), so it gets first-class treatment
rather than a closing paragraph. The headline is that **the hard
machinery already exists** — the MIP/global differentiability story is
an *integration* of shipped infrastructure, not new differentiation
theory.

### What already ships

POUNCE already differentiates continuous solves end-to-end through JAX:

- `pounce.jax.solve` / `solve_with_warm` — `jax.custom_vjp` wrappers
  whose backward is KKT implicit-function-theorem differentiation with
  active-set handling (`python/pounce/jax/_diff.py:128`): active bounds
  → `dx/dp = 0`; active constraint rows form the KKT block; inactive
  rows drop out (the pounce#73 slack-inequality fix).
- `vmap_solve` / `vmap_solve_parallel` / `batched_solve` — batched and
  GIL-released parallel solves with the per-element KKT backward
  `vmap`-ed (pounce#74).
- `JaxProblem` — build-once / solve-many handle with factor reuse across
  the backward (pounce#75–#77).
- `pounce-sensitivity` — the Rust-side converged-KKT-factor reuse the
  backward stands on (`pounce_linsol::Factorization`).

### The forward/backward split

| Pass | Does | Differentiable? |
|---|---|---|
| **Forward** | B&B / spatial search — finds the optimal integer assignment and winning basin | No (combinatorial / global) — but it runs entirely in Rust |
| **Backward** | implicit-diff through the *winning leaf*: original problem with integers fixed + box pinned | **Yes** — reuses `_diff.py` `bwd` unchanged, integers pinned like active bounds (Seam 5) |

So the new Python surface mirrors the existing one exactly:

```python
import pounce.jax as pj

x_star = pj.solve_mip(params, x0)          # custom_vjp: fwd = B&B,
                                           #   bwd = fixed-integer KKT
x_star = pj.solve_global(params, x0)       # same, winning local basin
grads  = jax.grad(loss)(params)            # flows through the solve
batch  = pj.vmap_solve_mip(param_batch)    # decision-focused learning
                                           #   over a whole dataset
```

`vmap_solve_mip` is the ML payoff: a *batch* of mixed-integer / global
programs, each differentiated w.r.t. the (neural-net-produced) parameters
that defined it, solved and back-propagated in parallel. That is
end-to-end **decision-focused learning** (predict-then-optimize) for a
problem class — MINLP / global — that no existing differentiable-layer
library reaches.

### Exact vs. smoothed gradients (the honest boundary)

The implicit-diff gradient is **exact, conditional on the discrete
solution being locally stable** (optimal assignment + active set
invariant under the infinitesimal perturbation). The argmin of a MIP is
piecewise-constant in its parameters: the true gradient is zero almost
everywhere and undefined at the switching surfaces. Two regimes:

- **Assignment stable (most of training):** the conditional gradient is
  the right object; it is what OptNet-style layers and differentiable
  MPC with a fixed active set already use. Cheap, exact, reuses
  everything.
- **Learning *through* the discrete switch:** wrap the solver in a
  smoothing rule on the JAX side — perturbed optimizers (Berthet et al.
  2020), blackbox-combinatorial interpolation (Vlastelica et al. 2020),
  or an SPO+ surrogate loss (Elmachtoub & Grigas 2022). These are
  alternative `custom_vjp` rules in `_diff.py`; **the Rust core does not
  change.**

Designing Seam 5 in from each solver's first commit (decision 6) is what
keeps both regimes available without a forward-solver rewrite.

## Implementation phasing

The ordering is forced: convex MI (Phases 0–2) is the substrate the
global layer (Phases 3–5) stands on. Each phase is independently
shippable. **Simplex and feral's sparse LU are not a convex-MI phase —
they land with the global LP-relaxation nodes (Phase 4), per decision
2.**

**Definition of done (every phase that ships a solver).** Per decision
6, a phase is not complete until: (i) the forward solver passes its
numerical tests; (ii) it emits the Seam-5 `DiffHandoff`; (iii) a
`pounce.jax.solve_*` `custom_vjp` wraps it and passes a finite-
difference gradient check; and (iv) a `vmap_*` batched form exists. The
JAX work is *part of* each phase, not a trailing phase — that is the
whole point of making differentiability first-class.

**Phase 0 — Integer plumbing (mostly already done).** The
`ProblemClass` classifier, NL-header parsing, and `solver_selection`
routing merged in PR #70. Phase 0 is therefore *additive*: extend the
existing classifier with integer and nonconvexity flags, parse the
integer-var counts from the `.nl` header, carry `is_integer: BitVec` on
the problem, and run **root presolve** (the full `pounce-presolve`
pipeline plus the LP reductions merged in PR #70). Dispatch errors
cleanly on integrality / nonconvexity it cannot yet handle. *No new
algorithm, no JAX surface yet.*

**Phase 1 — B&B shell + first convex MI + `solve_mip` VJP.** The unified
tree with integer branching only, driven by **MIQP over `pounce-qp`**
first (native warm-starts, weakest competition). Most-fractional
branching, depth-first, incumbent + gap. **DoD:** the leaf emits
`DiffHandoff`; `pj.solve_mip` differentiates with integers pinned;
`vmap_solve_mip` batches. This is the first end-to-end differentiable
mixed-integer solve and the minimum that justifies `pounce-mip`.

**Phase 2 — Cuts + MIP presolve.** Gomory / MIR / cover cuts; the
MIP-specific **root presolve** reductions (probing, coefficient
strengthening, clique/implication, GCD on integer rows) and cheap
**node presolve** (FBBT every node, selective OBBT) — all extending
`pounce-presolve` on top of the LP reductions merged in PR #70, not a
new crate. Pseudocost / reliability branching. This is where the
"table-stakes performance" (Positioning #2) is actually earned. Cuts and
reductions must preserve the leaf's active-set hand-off so the backward
stays valid. Mostly combinatorial engineering. **This completes the
convex-MI deliverable — still no simplex, no LU.**

**Phase 3 — `pounce-relax`.** Factorable decomposition, McCormick
envelopes (linearized to LP cuts per decision 7), αBB, OBBT. The
multi-quarter research lift; the heart of the global capability.
Validates by reproducing known global optima on small MINLPLib
instances. **DoD:** the relaxation is differentiable — relaxation-gap
gradients available if wanted.

**Phase 4 — Spatial B&B + simplex/LU + `solve_global` VJP.** Continuous
branching on the relaxation-gap variable; relaxation lower bounds from
Phase 3; multi-start local NLP upper bounds; convergence by global-gap
closure. **This is where the warm-started LP-relaxation nodes finally
need the dual simplex** — so feral's sparse LU (Bartels-Golub /
Forrest-Tomlin updates, landing now) and the simplex driver (a module in
`pounce-convex`) come online here, *not* earlier. **DoD:**
`pj.solve_global` differentiates through the winning local basin (box
pinned); `vmap_solve_global` batches. A genuine *differentiable*
deterministic global solver.

**Phase 5 — MINLP-global unified.** Both branching modes active at once
— the BARON / Couenne capability, end to end, pure Rust, and
differentiable / batched throughout.

**Phase 6 (optional, JAX-side) — smoothed/through-the-switch gradients.**
Perturbed-optimizer and blackbox-combinatorial `custom_vjp` rules in
`_diff.py` for learning that must flow gradient *through* the discrete
decision. No Rust changes; gated on an ML workload that needs it.

### Cost summary (rough, single engineer)

Each solver phase's effort below *includes* its JAX backward + `vmap`
wrapper (decision 6) — roughly a +15–25% tax over a forward-only solver,
already folded into the ranges.

Simplex + feral sparse LU is *not* a convex-MI line item — its effort
folds into Phase 4 (spatial B&B), where the warm-started LP-relaxation
nodes first need it. feral's LU is landing independently of this plan, so
the Phase-4 range below assumes the LU backend already exists and counts
only the simplex driver + integration.

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Integer plumbing | 2–4 weeks | 1 month |
| 1 — B&B shell + MIQP + VJP | 3–5 months | 4–6 months |
| 2 — Cuts + MIP presolve | 3–6 months | 7–12 months |
| 3 — Relaxation engine (`pounce-relax`) | 6–12 months | 13–24 months |
| 4 — Spatial B&B + simplex driver + VJP | 6–10 months | 19–34 months |
| 5 — MINLP-global | 3–6 months | 22–40 months |
| 6 — Smoothed gradients (opt) | 1–2 months | gated on demand |

Phases 0–2 are the differentiable convex-MI deliverable — incremental,
shippable on top of LP/QP, and **free of the hardest numerics** (no
simplex, no LU). Phases 3–5 are a flagship, multi-year, BARON-class
effort whose differentiability has no existing analog.

**Effort caveat given discopt.** These estimates assume a from-scratch
build. Because discopt already implements the spatial-B&B, McCormick/αBB,
and AMP machinery (see "Relationship to discopt"), the realistic path is
much shorter: the Rust phases narrow to *porting/optimizing* the
hot-loop pieces under discopt's orchestration and replacing `ripopt`,
rather than reinventing the algorithms. Treat the table as the ceiling
for an independent pure-Rust product; the compose-with-discopt path is
the floor.

## Limitations to design around

1. **Global optimization requires the symbolic expression graph.** Only
   `.nl`-loaded problems expose a structural representation today;
   Python (`PyTnlp`), C-callback, and Rust-closure problems "silently
   opt out" (`docs/src/fbbt.md`). Black-box NLPs *cannot* be globally
   solved with this machinery — that would need interval / Lipschitz
   black-box methods, a separate family. So global optimization is
   gated on the `ExpressionProvider`, and broadening it (a structural
   Python modeling surface) is a prerequisite for global to be useful
   beyond the AMPL path. This may pull modeling-layer work forward.
2. **Default behavior never changes.** A nonconvex NLP without explicit
   `solver_selection=global` opt-in still gets a local IPM-NLP solve, as
   today. Global is opt-in because it can be orders of magnitude slower
   and users rarely want a certificate by default.

## Reference implementations (study, do not wrap)

Pure Rust means reimplementing, not linking. The blueprints:

- **Couenne** — COIN-OR, **EPL-2.0 (same license as POUNCE / Ipopt)**.
  Its architecture (expression library → spatial B&B → LP relaxation +
  Cgl cuts → Clp / Cbc) is precisely this design. The natural reference.
- **SCIP** — now Apache-2.0; the strongest open MINLP / constraint-integer
  framework.
- **BARON** (Tawarmalani–Sahinidis) — the commercial bar for nonconvex
  global; the factorable-relaxation + range-reduction literature.
- **αBB** (Adjiman–Floudas) — the general-C² convex-underestimator
  construction for Phase 3.

## Benchmark suites

To validate against once each layer lands (listed in adoption order):

### MILP / MIQP (convex MI)

- **MIPLIB 2017** — the de-facto MILP standard; Mittelmann publishes
  head-to-head runs. The credible bar for an MILP / MIQP solver.
- **MIPLIB LP relaxations** — already noted in lp-qp-routing.md as an LP
  secondary set; the root relaxations double as B&B-node sanity checks.

### Convex MINLP

- **MINLPLib** — the standard MINLP library; tag the convex subset for
  Phases 1/2/3 validation.
- **CMU-IBM / Bonmin convex-MINLP test set** — classic convex MINLP
  instances (process synthesis, layout).

### Nonconvex / global

- **MINLPLib nonconvex subset** — the global-optimization bar; report
  against BARON / Couenne / SCIP global runs.
- **GLOBALLib / Lib2** — historical global-optimization instances.
- **Mittelmann global / MINLP benchmarks** — curated head-to-head runs.

### What "competitive" means

Competitiveness here is a *constraint, not the objective* (scope
decision 3). For convex MI, HiGHS / SCIP are the open bar; Gurobi / COPT
lead by a wide margin and POUNCE deliberately does not chase them. For
nonconvex global, BARON / commercial solvers lead. The bar POUNCE
actually sets for itself is **"correct and credible on MINLPLib"** — and
then the differentiator no benchmark sheet measures: the *same* solves
are JAX-differentiable and `vmap`-batched. The competitive claim is not
"fastest MINLP solver" but "the only **differentiable** one."

## What does not change

- `TNLP` stays algorithm-agnostic and object-safe; the B&B engine
  drives node solves through it, exactly as the LP/QP solvers do.
- The `.sol` writer is problem-type-agnostic; no change.
- The filter line search stays the default NLP globalization; the local
  IPM-NLP path is unchanged and remains the default for nonconvex NLP
  absent explicit global opt-in.
- `pyomo-pounce` gets MIP / global routing transparently via CLI
  dispatch; integer variables already round-trip through the NL format.
- The `jax.custom_vjp` pattern in `python/pounce/jax/_diff.py` is
  *extended, not redesigned* — `solve_mip` / `solve_global` reuse the
  existing KKT-implicit-diff backward and the `vmap`/batching machinery
  (pounce#73–#77) wholesale. The Rust-side `pounce-sensitivity`
  factor-reuse seam is unchanged; integers are just one more class of
  pinned variable in the active-set mask.

## Literature references

Organized to mirror the sections above. These are the algorithms to
reimplement (pure Rust — study, don't link) and the sources that pin
down the design choices. Canonical / foundational entries are marked ★.

### Branch-and-bound and MILP foundations

- ★ Land, A.H. & Doig, A.G. (1960). "An automatic method of solving
  discrete programming problems." *Econometrica* 28(3):497–520. The
  original branch-and-bound.
- ★ Dakin, R.J. (1965). "A tree-search algorithm for mixed integer
  programming problems." *The Computer Journal* 8(3):250–255. Branching
  on fractional variables — the scheme `pounce-mip` Phase 1 implements.
- Nemhauser, G.L. & Wolsey, L.A. (1988). *Integer and Combinatorial
  Optimization.* Wiley. The standard reference.
- Conforti, M., Cornuéjols, G. & Zambelli, G. (2014). *Integer
  Programming.* Springer GTM 271. Modern textbook.
- Achterberg, T. & Wunderling, R. (2013). "Mixed integer programming:
  analyzing 12 years of progress." In *Facets of Combinatorial
  Optimization*, Springer, 449–481. What actually moved the needle in
  practice — the prioritization input for Phase 3.
- Lodi, A. (2010). "Mixed integer programming computation." In *50 Years
  of Integer Programming*, Springer, 619–645.

### Cutting planes (Phase 3)

- ★ Gomory, R.E. (1958). "Outline of an algorithm for integer solutions
  to linear programs." *Bulletin of the AMS* 64(5):275–278. Gomory cuts.
- Marchand, H. & Wolsey, L.A. (2001). "Aggregation and mixed integer
  rounding to solve MIPs." *Operations Research* 49(3):363–371. MIR cuts.
- Balas, E., Ceria, S. & Cornuéjols, G. (1993). "A lift-and-project
  cutting plane algorithm for mixed 0–1 programs." *Mathematical
  Programming* 58:295–324.
- Cornuéjols, G. (2008). "Valid inequalities for mixed integer linear
  programs." *Mathematical Programming* 112:3–44. Survey.

### Branching and node selection (Phases 1, 3)

- Bénichou, M. et al. (1971). "Experiments in mixed-integer linear
  programming." *Mathematical Programming* 1:76–94. Pseudocost branching.
- ★ Achterberg, T., Koch, T. & Martin, A. (2005). "Branching rules
  revisited." *Operations Research Letters* 33(1):42–54. Reliability
  branching — the Phase 2 default. doi:10.1016/j.orl.2004.04.002
- Linderoth, J.T. & Savelsbergh, M.W.P. (1999). "A computational study
  of search strategies for mixed integer programming." *INFORMS Journal
  on Computing* 11(2):173–187. Node-selection strategies.
- Achterberg, T. (2009). "SCIP: solving constraint integer programs."
  *Mathematical Programming Computation* 1(1):1–41. The
  constraint-integer-programming framework this design echoes.

### Simplex and basis factorization (Phase 4 — feral sparse LU + `pounce-convex` simplex module)

- ★ Dantzig, G.B. (1963). *Linear Programming and Extensions.*
  Princeton University Press.
- ★ Bartels, R.H. & Golub, G.H. (1969). "The simplex method of linear
  programming using LU decomposition." *Communications of the ACM*
  12(5):266–268. The feral sparse-LU update scheme (landing in feral
  beside its LDLᵀ, gated on the global work).
- ★ Forrest, J.J.H. & Tomlin, J.A. (1972). "Updated triangular factors
  of the basis to maintain sparsity in the product form simplex method."
  *Mathematical Programming* 2:263–278. The sparsity-preserving
  alternative update.
- Goldfarb, D. & Reid, J.K. (1977). "A practicable steepest-edge simplex
  algorithm." *Mathematical Programming* 12:361–371. Steepest-edge
  pricing.
- Maros, I. (2003). *Computational Techniques of the Simplex Method.*
  Kluwer. The implementation reference for a from-scratch simplex.
- Koberstein, A. (2008). "Progress in the dual simplex algorithm for
  solving large scale LP problems: techniques for a fast and stable
  implementation." *Computational Optimization and Applications*
  41(2):185–204. Dual simplex + bound-flipping ratio test.
- Huangfu, Q. & Hall, J.A.J. (2018). "Parallelizing the dual revised
  simplex method." *Mathematical Programming Computation* 10(1):119–142.
  The HiGHS dual simplex — the open-source bar for the node LP.
  doi:10.1007/s12532-017-0130-5

### Presolve and bound tightening (reuses `pounce-presolve`)

- Brearley, A.L., Mitra, G. & Williams, H.P. (1975). "Analysis of
  mathematical programming problems prior to applying the simplex
  algorithm." *Mathematical Programming* 8:54–83. Classic LP presolve.
- Savelsbergh, M.W.P. (1994). "Preprocessing and probing techniques for
  mixed integer programming problems." *ORSA Journal on Computing*
  6(4):445–454. Probing and coefficient tightening for Phase 3.
- ★ Belotti, P., Cafieri, S., Lee, J. & Liberti, L. (2010). "Feasibility-
  based bound tightening via fixed points." In *Combinatorial
  Optimization and Applications*, LNCS 6508, 65–76. The FBBT this repo
  already ships (issue #62, `docs/src/fbbt.md`).
- Gleixner, A.M., Berthold, T., Müller, B. & Weltge, S. (2017). "Three
  enhancements for optimization-based bound tightening." *Journal of
  Global Optimization* 67(4):731–757. OBBT made affordable — the Phase 3
  OBBT loop. doi:10.1007/s10898-016-0450-4

### Convex MINLP (Phases 1, 3 upper layer)

- ★ Duran, M.A. & Grossmann, I.E. (1986). "An outer-approximation
  algorithm for a class of mixed-integer nonlinear programs."
  *Mathematical Programming* 36:307–339. Outer approximation.
- Geoffrion, A.M. (1972). "Generalized Benders decomposition." *Journal
  of Optimization Theory and Applications* 10:237–260.
- Fletcher, R. & Leyffer, S. (1994). "Solving mixed integer nonlinear
  programs by outer approximation." *Mathematical Programming*
  66:327–349.
- ★ Quesada, I. & Grossmann, I.E. (1992). "An LP/NLP based branch and
  bound algorithm for convex MINLP problems." *Computers & Chemical
  Engineering* 16(10–11):937–947. The LP/NLP-BB single-tree scheme.
- Bonami, P. et al. (2008). "An algorithmic framework for convex mixed
  integer nonlinear programs." *Discrete Optimization* 5(2):186–204.
  Bonmin.
- Kronqvist, J., Bernal, D.E., Lundell, A. & Grossmann, I.E. (2019). "A
  review and comparison of solvers for convex MINLP." *Optimization and
  Engineering* 20:397–455. The current survey of the convex-MINLP
  landscape; use it to pick Phase-1/3 defaults.

### Deterministic global optimization — factorable relaxations (Phases 3–5)

- ★ McCormick, G.P. (1976). "Computability of global solutions to
  factorable nonconvex programs: Part I — Convex underestimating
  problems." *Mathematical Programming* 10:147–175. The foundation of
  the `pounce-relax` envelopes.
- Falk, J.E. & Soland, R.M. (1969). "An algorithm for separable
  nonconvex programming problems." *Management Science* 15(9):550–569.
  Early spatial branch-and-bound.
- ★ Smith, E.M.B. & Pantelides, C.C. (1999). "A symbolic reformulation/
  spatial branch-and-bound algorithm for the global optimisation of
  nonconvex MINLPs." *Computers & Chemical Engineering* 23(4–5):457–478.
  The auxiliary-variable factorable reformulation `pounce-relax` Phase 3
  implements.
- ★ Adjiman, C.S., Dallwig, S., Floudas, C.A. & Neumaier, A. (1998). "A
  global optimization method, αBB, for general twice-differentiable
  constrained NLPs — I. Theoretical advances." *Computers & Chemical
  Engineering* 22(9):1137–1158. The αBB underestimator.
- Androulakis, I.P., Maranas, C.D. & Floudas, C.A. (1995). "αBB: A global
  optimization method for general constrained nonconvex problems."
  *Journal of Global Optimization* 7:337–363.
- Ryoo, H.S. & Sahinidis, N.V. (1996). "A branch-and-reduce approach to
  global optimization." *Journal of Global Optimization* 8:107–138.
  Range reduction / OBBT inside the spatial tree.
- ★ Tawarmalani, M. & Sahinidis, N.V. (2002). *Convexification and Global
  Optimization in Continuous and Mixed-Integer Nonlinear Programming.*
  Kluwer. The BARON monograph.
- Tawarmalani, M. & Sahinidis, N.V. (2005). "A polyhedral branch-and-cut
  approach to global optimization." *Mathematical Programming*
  103:225–249. The polyhedral-relaxation BARON design.
- ★ Belotti, P., Lee, J., Liberti, L., Margot, F. & Wächter, A. (2009).
  "Branching and bounds tightening techniques for non-convex MINLP."
  *Optimization Methods and Software* 24(4–5):597–634. The Couenne paper
  — closest published analog of this whole note, and EPL-2.0 lineage.
- Misener, R. & Floudas, C.A. (2014). "ANTIGONE: Algorithms for
  coNTinuous / Integer Global Optimization of Nonlinear Equations."
  *Journal of Global Optimization* 59:503–526.

### Conic relaxations for quadratic structure (`ConicRelaxation`, Phase 4+)

The tighter-bound alternative for QP / QCQP, cheap because the LP/QP
branch lands the cone IPM and its differentiable layer anyway.

- ★ Shor, N.Z. (1987). "Quadratic optimization problems." *Soviet
  Journal of Computer and Systems Sciences* 25:1–11. The SDP relaxation
  of nonconvex QP.
- ★ Anstreicher, K.M. (2009). "Semidefinite programming versus the
  reformulation-linearization technique for nonconvex quadratically
  constrained quadratic programming." *Journal of Global Optimization*
  43:471–484. Shows the SDP relaxation dominates RLT — the reason to add
  a conic bound. doi:10.1007/s10898-008-9372-0
- Burer, S. (2009). "On the copositive representation of binary and
  continuous nonconvex quadratic programs." *Mathematical Programming*
  120:479–495. The completely-positive reformulation lineage.
- Chen, J. & Burer, S. (2012). "Globally solving nonconvex quadratic
  programming problems via completely positive programming."
  *Mathematical Programming Computation* 4:33–52. CP-based global QP.
- Kim, S. & Kojima, M. (2003). "Exact solutions of some nonconvex
  quadratic optimization problems via SDP and SOCP relaxations."
  *Computational Optimization and Applications* 26:143–154. The cheaper
  SOCP relaxation that warm-starts better in B&B.

### McCormick relaxation theory (for a correct, convergent `pounce-relax`)

- Mitsos, A., Chachuat, B. & Barton, P.I. (2009). "McCormick-based
  relaxations of algorithms." *SIAM Journal on Optimization*
  20(2):573–601. Generalized/propagated McCormick — the form a DAG
  implementation needs. doi:10.1137/080717341
- Scott, J.K., Stuber, M.D. & Barton, P.I. (2011). "Generalized
  McCormick relaxations." *Journal of Global Optimization* 51:569–606.
- Bompadre, A. & Mitsos, A. (2012). "Convergence rate of McCormick
  relaxations." *Journal of Global Optimization* 52:1–28. Why
  branch-and-bound on these envelopes converges — and how fast.

### Interval arithmetic (reuses `pounce-presolve/src/fbbt/interval.rs`)

- ★ Moore, R.E. (1966). *Interval Analysis.* Prentice-Hall.
- Neumaier, A. (2004). "Complete search in continuous global optimization
  and constraint satisfaction." *Acta Numerica* 13:271–369. Survey
  connecting interval methods, constraint propagation, and global B&B.

### Differentiable optimization layers & decision-focused learning (the thesis)

The literature behind scope decision 3 — putting a solve inside an ML
model. The convex-layer entries are what POUNCE's existing
`pounce.jax.solve` already implements via KKT implicit diff; the
combinatorial entries are the Phase-7 smoothing escape hatches.

- ★ Amos, B. & Kolter, J.Z. (2017). "OptNet: differentiable optimization
  as a layer in neural networks." *ICML 2017.* The QP-as-a-layer
  founding paper; KKT implicit differentiation — the exact mechanism
  `_diff.py` generalizes.
- ★ Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S. & Kolter,
  J.Z. (2019). "Differentiable convex optimization layers." *NeurIPS
  2019.* cvxpylayers / diffcp — **convex-only**, the boundary this note's
  MIP/global layer crosses.
- Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-P. & Bach,
  F. (2020). "Learning with differentiable perturbed optimizers."
  *NeurIPS 2020.* Perturbed-optimizer smoothing for through-the-switch
  gradients (Phase 6).
- Vlastelica, M., Paulus, A., Musil, V., Martius, G. & Rolínek, M.
  (2020). "Differentiation of blackbox combinatorial solvers." *ICLR
  2020.* arXiv:1912.02175. Informative gradients through a blackbox
  combinatorial solver via loss interpolation (Phase 6).
- Paulus, A., Rolínek, M., Musil, V., Amos, B. & Martius, G. (2021).
  "CombOptNet: fit the right NP-hard problem by learning integer
  programming constraints." *ICML 2021.*
- ★ Elmachtoub, A.N. & Grigas, P. (2022). "Smart 'Predict, then
  Optimize'." *Management Science* 68(1):9–26. The SPO+ surrogate loss
  for predict-then-optimize. doi:10.1287/mnsc.2020.3922
- Ferber, A., Wilder, B., Dilkina, B. & Tambe, M. (2020). "MIPaaL: mixed
  integer program as a layer." *AAAI 2020*, 34(02):1504–1511. Closest
  prior art to differentiable MIP — but MILP-only and built on a
  commercial solver, not a pure-Rust differentiable stack.
- Wilder, B., Dilkina, B. & Tambe, M. (2019). "Melding the data-decisions
  pipeline: decision-focused learning for combinatorial optimization."
  *AAAI 2019.* The decision-focused-learning framing that motivates the
  `vmap_solve_mip` batched payoff.

### Prior art in this lineage (the sibling project)

- ★ **discopt** — <https://github.com/jkitchin/discopt>. Same author,
  EPL-2.0. A hybrid Python/JAX/Rust MINLP solver: spatial B&B, McCormick
  (21 functions incl. NN activations), piecewise-McCormick, αBB, AMP
  (adaptive multivariate partitioning), FBBT/OBBT, and KKT
  implicit-differentiation sensitivity. The working implementation this
  note's Rust layer composes under (see "Relationship to discopt"); also
  the reference for AMP and the NN-activation envelope library, neither
  of which the classic global-optimization references cover.
