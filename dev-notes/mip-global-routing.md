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
2. **Simplex is back, as a dependency.** Pure-Rust means no HiGHS to
   wrap for warm-started node LPs. B&B throughput lives or dies on
   warm-starting node relaxations, and IPM-LP warm-starts poorly (must
   reseed the barrier — lp-qp-routing.md "Active-set vs IPM-QP"). A
   pure-Rust **dual simplex with LU-with-updates** becomes the node
   engine for MILP and for the LP relaxations inside spatial B&B. This
   reverses the LP/QP note's "Simplex (LP) — was Phase 4, removed"
   decision; it was removed *because* HiGHS was the fallback, and that
   fallback is now disallowed.
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
  pounce-lu/       # NEW: sparse LU with Bartels-Golub / Forrest-Tomlin
                   #   updates — the simplex factorization.
  pounce-simplex/  # NEW (or a module in pounce-convex): dual simplex with
                   #   bound-flipping ratio test, warm-startable node LP.
  pounce-mip/      # NEW: unified branch-and-bound engine — tree, node
                   #   management, incumbent, gap. Branching: integer
                   #   (pseudocost / reliability) AND spatial (continuous,
                   #   on the var driving the relaxation gap). Consumes
                   #   pounce-relax for bounds and pounce-{simplex,convex,
                   #   algorithm,qp} for node solves.
  ── existing ──
  pounce-convex/   # IPM-LP/QP (from the LP/QP plan)
  pounce-qp/       # active-set QP — the MIQP node engine
  pounce-algorithm/# IPM-NLP — upper-bound / incumbent local solves
  pounce-presolve/ # bound-tightening, FBBT, BTF, DM, components, auxiliary
```

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
            ┌──────────────┬───────┴───────┬──────────────┐
            ▼              ▼               ▼              ▼
      pounce-relax   pounce-simplex   pounce-qp      pounce-convex
         [P4]         [P2]            (existing,      (LP/QP plan)
            │              │           MIQP node)         │
            │              ▼               │              │
            │         pounce-lu [P2]       │              │
            │                              │              │
            ▼                              ▼              ▼
       ┌─────────────────┐          ┌──────────────────────────┐
       │ Expr DAG +       │          │ sparse symmetric          │
       │ interval arith   │          │ augmented system          │
       │ (fbbt/, auxiliary)│         │ (pounce-linsol + feral)   │
       │  ── anchor #1 ──  │          │  ── anchor #2 ──          │
       └─────────────────┘          └──────────────────────────┘
   (also: pounce-algorithm IPM-NLP — incumbent upper bounds, anchor #2)
```

Phase gating (what must exist before what):

```text
P0 plumbing ─▶ P1 B&B shell + MIQP ─▶ P2 simplex/LU ─▶ P3 cuts+presolve
                                                              │
                          convex-MI deliverable ◀────────────┘
                                                              │
                                   P4 pounce-relax ◀──────────┘ (needs a
                                          │                      working node
                                          ▼                      solver + B&B)
                                   P5 spatial B&B
                                          │
                                          ▼
                                   P6 MINLP-global
```

The crossbar is the whole point: **P4–P6 (global) cannot start until the
convex-MI substrate (P1–P3) can run a B&B tree over convex relaxations.**
`pounce-relax` only produces those relaxations; something has to bound
and branch on them.

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

## Interfaces and crate skeletons

The whole design rests on four trait seams. They are sketched here so
the phasing below builds against fixed interfaces, and so the
"one engine, pluggable parts" claim is concrete rather than aspirational.
Names are illustrative, not final.

### Seam 1 — `NodeSolver`: warm-startable relaxation solve

The single integration point with the continuous solvers. Thin adapters
implement it over `pounce-simplex` (dual simplex), `pounce-qp`
(active-set), `pounce-convex` (IPM), and `pounce-algorithm` (NLP). The
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
    pub cuts: &'a [CutRef],  // global cut-pool handles in scope (Phase 3)
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

/// Global path (Phase 4): factorable decomposition + envelopes.
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
                        #       pounce-relax, pounce-simplex
  src/
    lib.rs              # branch_and_bound<S,B,R,H> + the four seam traits
    tree.rs            # Node, best-bound / depth-first queue, gap accounting
    node.rs            # NodeData, NodeResult, NodeStatus
    branch/
      mod.rs            # Brancher trait, BranchDecision
      most_fractional.rs# Phase 1
      pseudocost.rs     # Phase 3 (reliability — Achterberg et al. 2005)
      spatial.rs        # Phase 5 (gap-driven continuous split)
    incumbent.rs        # IncumbentSearch trait + rounding/diving (Phase 3)
    cuts/               # Phase 3: pool + Gomory/MIR/cover separators
      mod.rs  gomory.rs  mir.rs  cover.rs
    adapters/           # NodeSolver impls (thin wrappers)
      simplex.rs  active_set.rs  ipm.rs  nlp.rs

crates/pounce-relax/    # the heavy symbolic crate (Phase 4)
  Cargo.toml            # deps: pounce-common, pounce-nlp, pounce-presolve (fbbt)
  src/
    lib.rs              # Relaxation trait, PassthroughRelaxation
    factorable.rs       # Expr-DAG → aux-var defining constraints
    mccormick.rs        # bilinear / univariate envelopes
    alpha_bb.rs         # αBB underestimator (interval Hessian eigenvalues)
    obbt.rs             # optimization-based bound tightening loop
    project.rs          # aux-var → original-var projection

crates/pounce-lu/       # Phase 2: sparse LU + Bartels-Golub/Forrest-Tomlin
crates/pounce-simplex/  # Phase 2: dual simplex, bound-flipping ratio test
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

The ordering is forced: convex MI (Phases 0–3) is the substrate the
global layer (Phases 4–6) stands on. Each phase is independently
shippable.

**Definition of done (every phase that ships a solver).** Per decision
6, a phase is not complete until: (i) the forward solver passes its
numerical tests; (ii) it emits the Seam-5 `DiffHandoff`; (iii) a
`pounce.jax.solve_*` `custom_vjp` wraps it and passes a finite-
difference gradient check; and (iv) a `vmap_*` batched form exists. The
JAX work is *part of* each phase, not a trailing phase — that is the
whole point of making differentiability first-class.

**Phase 0 — Integer plumbing.** Parse integer-var counts from the `.nl`
header; extend `ProblemClass`; carry `is_integer: BitVec` on the
problem. Dispatch errors cleanly on integrality / nonconvexity it
cannot yet handle. Mirrors LP/QP Phase 1. *No new algorithm, no JAX
surface yet.*

**Phase 1 — B&B shell + first convex MI + `solve_mip` VJP.** The unified
tree with integer branching only, driven by **MIQP over `pounce-qp`**
first (native warm-starts, weakest competition). Most-fractional
branching, depth-first, incumbent + gap. **DoD:** the leaf emits
`DiffHandoff`; `pj.solve_mip` differentiates with integers pinned;
`vmap_solve_mip` batches. This is the first end-to-end differentiable
mixed-integer solve and the minimum that justifies `pounce-mip`.

**Phase 2 — Pure-Rust simplex (`pounce-lu` + `pounce-simplex`).** Dual
simplex with bound-flipping ratio test and LU-with-updates. Native
node-to-node warm-starts. Unlocks MILP and the LP relaxations the
spatial path needs. **DoD:** the LP solve emits `DiffHandoff` (LP KKT is
trivially differentiable), so `solve_mip` over LP relaxations
differentiates too.

**Phase 3 — Cuts + node presolve.** Gomory / MIR / cover cuts; probing;
coefficient tightening reusing `pounce-presolve`. Pseudocost /
reliability branching. Closes the gap to real solvers for convex MI.
Cuts must preserve the leaf's active-set hand-off so the backward stays
valid. Mostly combinatorial engineering.

**Phase 4 — `pounce-relax`.** Factorable decomposition, McCormick
envelopes (linearized to LP cuts per decision 7), αBB, OBBT. The
multi-quarter research lift; the heart of the global capability.
Validates by reproducing known global optima on small MINLPLib
instances. **DoD:** the relaxation LP is differentiable (free, via the
simplex hand-off) — relaxation-gap gradients available if wanted.

**Phase 5 — Spatial B&B + `solve_global` VJP.** Continuous branching on
the relaxation-gap variable; relaxation lower bounds from Phase 4;
multi-start local NLP upper bounds; convergence by global-gap closure.
**DoD:** `pj.solve_global` differentiates through the winning local
basin (box pinned); `vmap_solve_global` batches. A genuine
*differentiable* deterministic global solver.

**Phase 6 — MINLP-global unified.** Both branching modes active at once
— the BARON / Couenne capability, end to end, pure Rust, and
differentiable / batched throughout.

**Phase 7 (optional, JAX-side) — smoothed/through-the-switch gradients.**
Perturbed-optimizer and blackbox-combinatorial `custom_vjp` rules in
`_diff.py` for learning that must flow gradient *through* the discrete
decision. No Rust changes; gated on an ML workload that needs it.

### Cost summary (rough, single engineer)

Each solver phase's effort below *includes* its JAX backward + `vmap`
wrapper (decision 6) — roughly a +15–25% tax over a forward-only solver,
already folded into the ranges.

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Integer plumbing | 2–4 weeks | 1 month |
| 1 — B&B shell + MIQP + VJP | 3–5 months | 4–6 months |
| 2 — Simplex + LU (+ VJP) | 5–9 months | 9–15 months |
| 3 — Cuts + node presolve | 3–6 months | 12–21 months |
| 4 — Relaxation engine | 6–12 months | 18–33 months |
| 5 — Spatial B&B + VJP | 5–9 months | 23–42 months |
| 6 — MINLP-global | 3–6 months | 26–48 months |
| 7 — Smoothed gradients (opt) | 1–2 months | gated on demand |

Phases 0–3 are the differentiable convex-MI deliverable — incremental
and shippable on top of LP/QP. Phases 4–6 are a flagship, multi-year,
BARON-class effort whose differentiability has no existing analog.

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
  construction for Phase 4.

## Benchmark suites

To validate against once each layer lands (listed in adoption order):

### MILP / MIQP (convex MI)

- **MIPLIB 2017** — the de-facto MILP standard; Mittelmann publishes
  head-to-head runs. The credible bar for an MILP / MIQP solver.
- **MIPLIB LP relaxations** — already noted in lp-qp-routing.md as an LP
  secondary set; the root relaxations double as B&B-node sanity checks.

### Convex MINLP

- **MINLPLib** — the standard MINLP library; tag the convex subset for
  Phases 1/3/4 validation.
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
  branching — the Phase 3 default. doi:10.1016/j.orl.2004.04.002
- Linderoth, J.T. & Savelsbergh, M.W.P. (1999). "A computational study
  of search strategies for mixed integer programming." *INFORMS Journal
  on Computing* 11(2):173–187. Node-selection strategies.
- Achterberg, T. (2009). "SCIP: solving constraint integer programs."
  *Mathematical Programming Computation* 1(1):1–41. The
  constraint-integer-programming framework this design echoes.

### Simplex and basis factorization (Phase 2 — `pounce-lu`, `pounce-simplex`)

- ★ Dantzig, G.B. (1963). *Linear Programming and Extensions.*
  Princeton University Press.
- ★ Bartels, R.H. & Golub, G.H. (1969). "The simplex method of linear
  programming using LU decomposition." *Communications of the ACM*
  12(5):266–268. The `pounce-lu` update scheme (already named in
  `pounce-simplex` crate comments).
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
  Global Optimization* 67(4):731–757. OBBT made affordable — the Phase 4
  OBBT loop. doi:10.1007/s10898-016-0450-4

### Convex MINLP (Phases 1, 4 upper layer)

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

### Deterministic global optimization — factorable relaxations (Phases 4–6)

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
  The auxiliary-variable factorable reformulation `pounce-relax` Phase 4
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
  gradients (Phase 7).
- Vlastelica, M., Paulus, A., Musil, V., Martius, G. & Rolínek, M.
  (2020). "Differentiation of blackbox combinatorial solvers." *ICLR
  2020.* arXiv:1912.02175. Informative gradients through a blackbox
  combinatorial solver via loss interpolation (Phase 7).
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
