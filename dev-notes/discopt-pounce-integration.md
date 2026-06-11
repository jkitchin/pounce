# pounce ⟷ discopt — the value of a deep, co-designed integration

> Discussion note. [discopt](https://github.com/jkitchin/discopt) is a MINLP
> modeling language + spatial branch-and-bound (B&B) orchestrator. It already
> lists POUNCE as one of three NLP backends (alongside a pure-JAX IPM and
> cyipopt). This note is about what going **beyond a generic solver-plugin
> interface** to a deep, co-designed integration unlocks — and why it changes
> what the combined system *is*.

## The core insight

Spatial B&B calls the NLP solver **thousands of times** over a tree in which
each child node differs from its parent by **one changed bound**. A generic
plugin treats every node as a cold, independent solve across a serialization
boundary (`.nl` file / fresh process state). Almost all the leverage of a deep
integration comes from refusing that — letting warm state, certificates,
relaxations, the AD graph, and diagnostics *flow through the tree* instead of
being rebuilt at every node.

A generic plugin makes discopt a **dispatcher** that hands problems to whichever
solver. Deep co-design makes pounce+discopt **one solver that happens to have a
modeling front-end and a B&B loop wrapped around the same numerical state.**
That is the difference between "a fast NLP solver under a B&B loop" and "an
MINLP engine."

## Value map

### B1 — Warm-starting across the tree (the biggest single win)
- Child = parent + one tightened bound → warm-start primal **and** dual **and**
  the barrier μ. pounce already has this primitive: `solve_with_warm` with dual
  + μ threading (pounce#86). A generic plugin discards it at every node.
- KKT sparsity is identical across the whole tree → symbolic-factorize once,
  numeric-refactor per node. `pounce-feral` could expose a "same pattern, new
  values" fast path.
- `pounce-qp`'s parametric active-set corrector is literally a "solve a small
  perturbation of the last problem" engine — exactly the node→node step.

### B2 — Bounds & certificates flowing both ways
- **Early-fathom from dual bounds:** B&B needs a *valid lower bound*, not a fully
  converged node. Expose pounce's mid-solve dual bound so discopt fathoms without
  solving to optimality.
- **Infeasibility certificates → instant prune:** pounce-convex emits
  Farkas/infeasibility certificates; a certified-infeasible relaxation prunes the
  subtree *with proof*, not a tolerance.
- **Sensitivity → branching:** `pounce-sensitivity` (sIPOPT) gives ∂x*/∂(bound) —
  exactly the signal for strong-branching pseudo-costs, free from a solve already
  done.

### B3 — One relaxation / convexification engine (kill the duplication)
- Both sides do McCormick + bound-tightening today: `pounce-global` (McCormick +
  OBBT/FBBT), `pounce-presolve` (FBBT + auxiliary elimination), and discopt
  (McCormick + AMP adaptive partitioning). Co-design → **one** relaxation library
  and cone catalog used by both the node relaxation and the tree, not two
  parallel implementations that can silently disagree.
- discopt's AMP and pounce-global's spatial B&B are the same algorithm class.
  Co-design decides who owns the tree *once*.

### B4 — One problem IR, no `.nl` round-trip
- The modeling language compiles **once** to a structure pounce consumes natively:
  sparsity pattern, colored-AD coloring, Hessian-of-Lagrangian structure,
  variable/constraint partition. Both sides already use JAX AD — the traced graph
  is a *shared asset*, not a bridge to be serialized per node.

### B5 — Differentiable MINLP (moonshot differentiator)
- pounce.jax already makes the *NLP* differentiable. With the integer/branching
  decisions fixed at the solution, discopt could expose ∂(MINLP solution)/∂(params)
  → a **differentiable mixed-integer layer** you backprop through. Almost nobody
  ships this. Ties directly into vision.md pillar 2.

### B6 — Tree-level diagnostics & agent-drivability
- pounce has an interactive debugger + MCP surface (`pounce-studio`). Lift it from
  per-solve to **per-tree**: which node stalled, which relaxation was loosest,
  where the gap stopped closing, why a subtree won't prune. An LLM agent driving
  an MINLP debug session is something no classical MINLP stack (BARON, Couenne,
  SCIP) was built for.

### B7 — Distribution, trust, certification
- `pip install`, pure-Rust core, **no GAMS/BARON/commercial license** underneath —
  discopt ships pounce embedded, reproducible.
- Extend signed solve receipts (`pounce verify`) to the **whole MINLP proof**: a
  verifiable certificate of the global optimality gap with node-level bounds.
  "Certified global, and here's the signed proof," end to end.

## Priorities (impact × effort)

```
HIGH IMPACT / LOWER EFFORT  (do first — proof points)
  ✓ B1  warm-start primal+dual+μ across nodes   (primitive exists: solve_with_warm, pounce#86)
  ✓ B2  dual-bound early fathom + certificate pruning
  ✓ B4  shared in-memory JAX problem IR (no .nl round-trip per node)

HIGH IMPACT / HIGHER EFFORT  (strategic bets)
  ★ B3  single relaxation/bound-tightening engine shared by both
  ★ B6  tree-level debugger + MCP (the agent differentiator)
  ★ B7  certified-gap signed receipts for the full MINLP

MOONSHOT
  ◇ B5  differentiable MINLP layer

SUPPORTING
  ○ B1  KKT symbolic-factorize-once / numeric-refactor fast path
  ○ B2  sensitivity-driven branching pseudo-costs
```

## The co-design API surface

What the interface must expose that a generic plugin *cannot*:

- **warm-state in/out** — primal, duals (`mult_g`, `mult_x_L/U`), and μ, threaded
  node→node (already prototyped in `solve_with_warm`).
- **valid-bound-without-full-convergence** — a dual lower bound mid-solve for
  early fathoming.
- **certificate out** — infeasibility/Farkas certificate for proof-based pruning.
- **shared sparsity / IR handle** — hand pounce the in-memory traced problem, not
  a serialized file.
- **sensitivity out** — ∂x*/∂(bound) for branching heuristics.
- **per-node diagnostic stream** — feed the studio/MCP tree-level debugger.

## Next steps

- Prototype B1 end-to-end: discopt threads pounce's `solve_with_warm` warm-state
  down the tree; measure node-solve speedup vs. cold `.nl` dispatch.
- Open tracking issues (pounce and/or discopt) for B1 / B2 / B4 — the lower-effort,
  high-impact trio — mirroring pounce#109.
- Decide tree ownership (B3): does the spatial B&B live in `pounce-global`,
  `discopt-core`, or a shared crate? This is the load-bearing architectural call.
