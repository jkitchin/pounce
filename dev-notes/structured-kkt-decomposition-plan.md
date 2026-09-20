# Structured KKT Decomposition for `discopt` + POUNCE

> Working plan, kept on the `feat/feral-factor-stats` branch alongside the work
> it describes. Phase status is in §57; the Phase 0b measurements are in
> `dev-notes/kkt-scaling-phase0b.md`.

## Purpose

This document proposes a general mechanism for exploiting block structure in large sparse nonlinear programs modeled in `discopt` and solved with POUNCE.

## Why decomposition, in general

Structure-exploiting interior-point methods are established practice. What `discopt` + POUNCE lack is a path that carries a model's block structure into the KKT solve. Decomposition buys things that no fill-reducing ordering of a single monolithic KKT system can provide:

- **Parallel and distributed factorization.** Independent blocks factor concurrently, and a scenario set too large for one node's memory can be spread across nodes.
- **Per-block reuse.** Identical blocks, partial refactorization when one block changes, rolling horizons, parameter sweeps.
- **Per-block solver choice.** Dense or sparse, a different backend, or different regularization for each block.
- **Robustness in pivoting.** Pivoting is kept inside a block, with inertia recovered by adding the block inertias, rather than pivots delayed across the whole matrix.
- **Declared rather than guessed structure.** Semantic structure from the model survives transformations and does not depend on a graph partitioner finding it again (§41).
- **A better scaling exponent**, *when* the generic solve scales superlinearly because it misses the structure (§38).

The last item is what motivated this document, but it is only one of six, and the capability should not be judged on it alone (§58).

## Prior art to position against, not reinvent

- **Plasmo.jl (OptiGraph):** a graph of model blocks with linking constraints. The closest analogue to the block graph in §11.
- **Parapint (Pyomo):** an interior-point method with structured linear algebra behind a solver-neutral interface. The closest analogue to §8 and §45.
- **PIPS-NLP, Schur-complement IPOPT, OOPS:** the arrowhead (scenario) Schur solve, inertia by adding block inertias, and regularized blocks to keep pivoting local (§22, §24).
- **HPIPM / acados:** the chain (optimal-control) case. There the block-tridiagonal solve of §18 is a Riccati recursion.

Phase 0 (§57) should record how each of these handles restoration, inertia correction and quasi-Newton Hessians. Those are exactly where POUNCE's current structured hooks stop.

## The first instance: multi-horizon gas planning

The first application is a large multi-horizon gas-network planning NLP in which:

- each horizon contains a large local nonlinear network model,
- neighboring horizons are coupled through a relatively small number of temporal state variables,
- some design or investment variables may be shared globally across all horizons,
- POUNCE can already solve very large sparse NLPs, but observed runtime scaling was roughly quadratic in total problem size (pounce#947),
- IPOPT appears to show similar scaling behavior.

**Phase 0b result (2026-09-19, `dev-notes/kkt-scaling-phase0b.md`).** The quadratic comes from refining the time step at a fixed horizon (GasLib-40-T optimal control, 28k–224k variables), not from extending the horizon, which scales as about \(H^{1.2}\). It is the **fill** row: under the default ordering wall time grows as \(n^{2.35}\) with factorization 96% of the solve at 224k, and iterations are flat. `metis` takes it to \(n^{1.47}\) and 8× faster at 224k; `auto_race` finds the same factorization without being told; the default `auto` never selects nested dissection. And the **one-dimensional temporal chain is the wrong block structure** for this model: an earlier study measured time-axis orderings 20–40× worse and a Schur split of the horizon 3× worse per iteration, because the link between consecutive time points is the whole 775-state slice, not a small separator.

So gas does not motivate the *chain* strategy. It may still motivate a different declared structure (§38.3): the model is a space-by-time mesh, and a block structure taken from the **network** (subnetworks separated by compressor stations or cut pipes) crossed with time blocks is untested. Other problem classes — scenario and contingency (arrowhead) models above all — carry the general case on their own merits.

Gas remains a good first instance for the infrastructure, because every piece it needs (metadata, mapping, validation, diagnostics) is general. From Phase 1 the benchmark set carries an **arrowhead** problem as well (§57), the topology where decomposition's advantages are best established.

The central idea is:

> Do not treat a structured 500,000-variable NLP as one arbitrary 500,000-variable sparse KKT system if the model is naturally composed of many weakly coupled blocks.

Instead:

1. preserve or detect the model-level decomposition in `discopt`,
2. pass a solver-neutral block structure to POUNCE,
3. let POUNCE construct a decomposition-aware KKT solve,
4. eliminate large local blocks,
5. solve a much smaller structured coupling system,
6. back-substitute to recover the full Newton/IPM direction.

This should remain an **exact Newton/KKT solve** within each POUNCE iteration. It is not necessarily an iterative outer decomposition method such as ADMM, Benders, or Lagrangian relaxation.

## Where the work happens

This plan executes in the **POUNCE repository**. Everything on the POUNCE side is built and tested there without depending on `discopt`: `Problem.set_block_structure` takes NLP indices (§12), and POUNCE's own benchmark generators supply the structure. Work needed on the `discopt` side is listed in §28 as **draft enhancement issues**. Each one is filed in `discopt` only when it is needed and approved, not in advance.

## Open decisions (resolve before Phase 0 closes)

- **D1. Arrowhead benchmark.** Proposed: N-1 security-constrained AC-OPF generated by extending `benchmarks/grid_ecf/ecf_nl_export.py` (base case plus \(k\) contingency copies sharing dispatch; \(k\) is the block-count knob). *TBD.*
- **D2. Source of the gas instances.** *Resolved:* generated locally from the existing GasLib-40 models (horizon family exported as `.nl`; the optimal-control family driven through the JAX front end). No `discopt` issue needed.
- **D3. Gate thresholds** (§58). Proposed defaults: an exponent drop of at least 0.3 in \(T\), or at least 2× at the largest instance, or at least 30% less peak memory; parallel efficiency above 50% at 8 threads. *TBD.*
- **D4. Entry-point names** (§12). Proposed: `Problem.set_block_structure(...)` in POUNCE; `solve(structured_kkt=...)` in `discopt`. *TBD.*

---

# 1. High-Level Architecture

The responsibilities should be divided cleanly.

```text
application / domain package
        |
        | declares physical model and known structure
        v
     discopt
        |
        | blocks, coupling variables, stages, graph
        v
solver-neutral decomposition description
        |
        v
      POUNCE
        |
        | chooses structured KKT strategy
        | maps model entities to KKT indices
        | coordinates elimination / Schur solve
        v
 pounce-linsol / FERAL
        |
        | sparse factorization primitives
        | block solves
        | triangular solves
        | inertia / regularization support
        v
 Newton/IPM direction
```

The key design principle is:

- `discopt` knows the **semantics and model structure**.
- POUNCE knows the **optimization algorithm and KKT structure**.
- FERAL / `pounce-linsol` know the **linear algebra**.

The application should not manually implement Schur complements, KKT partitioning, or block back-substitution.

---

# 2. Motivating Example: Multi-Horizon Gas Network

Suppose there are \(T\) planning horizons.

Each horizon has a large local variable vector

\[
x_t
\]

containing quantities such as:

- pressures,
- flows,
- compressor operating variables,
- nodal injections,
- local slack variables,
- local operating decisions.

Adjacent horizons are coupled by a smaller set of temporal variables

\[
z_t,
\]

for example:

- linepack,
- storage inventory,
- compressor state,
- ramping variables,
- state-of-charge-like quantities,
- maintenance state,
- carry-over capacity or operating state.

There may also be global planning variables

\[
y,
\]

such as:

- pipe-build decisions,
- compressor investments,
- capacity expansions,
- global design parameters.

A schematic model is

```text
           global variables y
            /   |   |   \
           /    |   |    \
       x1 ---- x2 ---- x3 ---- ... ---- xT
        \ z1 /  \ z2 / \ z3 /
```

The nonlinear equations for one horizon might be

\[
F_t(x_t,z_t,z_{t+1},y)=0,
\]

with bounds and inequalities

\[
G_t(x_t,z_t,z_{t+1},y)\le 0.
\]

The important structural property is that most entries of \(x_t\) do not directly interact with distant horizons.

---

# 3. What the Current Generic Sparse Solve Does

A conventional sparse NLP solver linearizes the full NLP and constructs a KKT system.

Conceptually:

\[
K \Delta w = r.
\]

Even if \(K\) is sparse, it is treated as one large sparse indefinite matrix.

A generic sparse factorizer can exploit:

- sparsity,
- ordering,
- symbolic factorization,
- fill reduction,
- pivoting,
- repeated sparsity patterns.

But it generally does not know:

> these 10,000 variables are horizon 17, these 10,000 are horizon 18, and only 80 variables connect them.

That information is stronger than ordinary sparsity.

The proposed feature preserves this higher-level structure and uses it to determine the elimination strategy.

---

# 4. Exact Structured KKT Elimination

This is not necessarily an approximate or iterative decomposition.

Consider a KKT system partitioned into local variables \(x\) and linking variables \(z\):

\[
\begin{bmatrix}
A & B^T \\
B & D
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta z
\end{bmatrix}
=
\begin{bmatrix}
r_x \\
r_z
\end{bmatrix}.
\]

If \(A\) is the collection of large local blocks, solve

\[
\Delta x = A^{-1}(r_x-B^T\Delta z).
\]

Substitution gives the reduced system

\[
S\Delta z =
r_z - BA^{-1}r_x,
\]

where

\[
S = D - BA^{-1}B^T.
\]

Then:

1. factor the local block(s),
2. assemble or apply the Schur complement,
3. solve for the coupling step \(\Delta z\),
4. back-substitute for \(\Delta x\).

This gives the same Newton direction as solving the full KKT system, subject to equivalent numerical pivoting and regularization choices.

---

# 5. Multi-Block Structure

For many horizons, the KKT system can often be rearranged into a form like

\[
\begin{bmatrix}
K_1 & C_1^T & 0 & 0 & \cdots \\
C_1 & K_2 & C_2^T & 0 & \cdots \\
0 & C_2 & K_3 & C_3^T & \cdots \\
0 & 0 & C_3 & K_4 & \cdots \\
\vdots & & & & \ddots
\end{bmatrix}.
\]

This is block-banded or block-tridiagonal when temporal coupling is local.

If global variables are also present, the structure becomes approximately

```text
             global
          / / / | \ \ \
        B1 B2 B3 B4 ... BT
         |  |  |  |
        local temporal links
```

This can be viewed as a block graph rather than a flat matrix.

The solver should exploit the graph explicitly.

---

# 6. Why This Could Change Scaling

Assume:

- \(T\) horizons,
- \(n\) local KKT unknowns per horizon,
- \(m\) linking unknowns per horizon,
- \(m \ll n\).

A monolithic solve sees approximately

\[
N \sim Tn.
\]

If empirical runtime behaves like

\[
T_{\mathrm{solve}}(N)\propto N^2,
\]

then the monolithic cost behaves roughly like

\[
O(T^2n^2).
\]

With block elimination, the expensive local work becomes approximately

\[
T \cdot O(n^2),
\]

plus a reduced coupling solve.

If the reduced coupling system is itself banded or block-tridiagonal, its cost can be far below a dense solve.

The idealized gain is therefore potentially very large.

**Caveat:** the \(N^2\) above is an observation, not a property of sparse direct factorization. On a chain with small separators, an elimination that follows the structure costs \(O(T)\) whether a block solver performs it or a multifrontal factorization given the right ordering does. So the gain is available only where the observed \(N^2\) lives in fill or pivoting, not in iteration count or work outside the factorization. §38.2 sets out how to tell which.

The practical questions are stated in §38: the general one per topology, and for gas, where its \(T^2\) comes from.

---

# 7. Relationship to Existing `discopt.decomposition`

`discopt.decomposition` should remain the natural home for model-level structural information.

The existing decomposition machinery is conceptually aimed at optimization-level methods such as:

- Benders decomposition,
- generalized Benders,
- Lagrangian relaxation,
- decomposition advisors,
- block and coupling detection.

The proposed work reuses the same structural description for a different purpose:

> accelerating a single Newton/IPM KKT solve inside POUNCE.

This is an important distinction.

## Optimization-level decomposition

```text
solve subproblem A
solve subproblem B
coordinate master problem
repeat
```

Examples:

- Benders,
- Lagrangian relaxation,
- ADMM-like methods.

## Linear-algebra-level decomposition

```text
one POUNCE iteration
    |
    v
one linearized KKT system
    |
    v
factor local blocks
    |
    v
solve exact reduced coupling system
    |
    v
back-substitute
    |
    v
one full Newton direction
```

The second approach should be transparent to the nonlinear optimization algorithm.

---

# 8. Proposed `discopt` Additions

*Correction 2026-09-20: this section, and §45's structure object, were written
without reading `discopt.decomposition`, and they propose something that is
largely already there. `DecompositionStructure` carries `block_of_var`,
`complicating_vars` and a `block_of_constraint` that is **already `-1` for
coupling rows** — the convention POUNCE settled on independently — resolved
from `set_block` / `mark_coupling` / `first_stage` annotations or auto-detected
(bridge constraints + connected components). The advisor registers
`MethodKind.SCHUR` "for later phases", its design doc's taxonomy table already
names the structural signature ("KKT sparsity graph | a border (bordered
block-diagonal)"), and `ROADMAP.md` lists Schur among the remaining methods.
So the discopt-side ask is not a structure export; it is the **consumer** for a
slot already reserved, plus the index-space mapping and its stability through
discopt's transformations. jkitchin/discopt#1370 was revised to say that.*

*The distinction worth keeping straight, because it is why this is not
redundant with what `decomposition/` already does: Benders / GBD / Lagrangian
decompose the **algorithm** — a master and subproblem loop whose soundness is
conditional (GBD is exact only on convex recourse, Lagrangian yields a bound),
which is why that module has a `Soundness` gatekeeper at all. The block-KKT
path decomposes the **linear algebra inside one solve**: same IPM, same
iterates, exact on any nonconvex NLP because it is a permutation plus block
elimination of a symmetric indefinite system. Its evidence is identical
iteration counts and objectives, not a convexity classification. They compose —
a block-structured Benders subproblem can be solved this way.*

The first goal is to expose a solver-neutral structural description.

## 8.1 Core object

A possible structure:

```python
@dataclass
class NLPBlock:
    name: str
    variables: Sequence[Variable]
    constraints: Sequence[Constraint]


@dataclass
class NLPLink:
    blocks: tuple[str, ...]
    variables: Sequence[Variable] = ()
    constraints: Sequence[Constraint] = ()


@dataclass
class NLPDecomposition:
    blocks: list[NLPBlock]
    links: list[NLPLink]
    global_variables: Sequence[Variable]
    global_constraints: Sequence[Constraint]
    graph: BlockGraph
```

The names are illustrative.

The key requirement is that the representation be:

- solver-neutral,
- independent of POUNCE internals,
- stable under model transformations,
- able to map back to original model entities.

---

# 9. Explicit Model Annotation

For known temporal problems, explicit structure is preferable to rediscovery.

Possible API:

```python
with model.block("horizon_0"):
    build_network(model, data[0])

with model.block("horizon_1"):
    build_network(model, data[1])

model.link(
    "horizon_0",
    "horizon_1",
    variables=[linepack[0], storage[0]]
)

model.shared_variables(investment_variables)
```

Alternative syntax:

```python
model.set_block(vars=horizon_vars[t], cons=horizon_cons[t], block=t)

model.mark_coupling(
    vars=link_vars[t],
    blocks=(t, t + 1),
)

model.mark_global(investment_vars)
```

The exact API should fit existing `discopt.decomposition` conventions rather than inventing duplicate abstractions.

---

# 10. Automatic Structure Detection

Explicit annotations should be preferred when available, but automatic analysis is valuable.

Construct a bipartite incidence graph:

```text
variables <----> constraints
```

with an edge whenever

\[
\frac{\partial c_i}{\partial x_j}\neq 0.
\]

Useful structural analysis includes:

- connected components,
- articulation variables,
- separator detection,
- block-angular structure,
- chain structure,
- tree structure,
- weakly coupled components,
- global high-degree variables.

The decomposition advisor could produce something like:

```python
structure = analyze_decomposition(model)

print(structure.kind)
# "chain"

print(structure.num_blocks)
# 48

print(structure.max_separator_size)
# 76
```

The key output is not merely a recommendation such as “use Benders.” It should include a concrete block graph suitable for solver use.

---

# 11. Proposed Solver-Neutral Block Graph

The most general representation is a graph:

\[
G_B=(V_B,E_B),
\]

where:

- each node is a local NLP block,
- each edge represents coupling,
- global variables can be represented separately or as a root/global block.

Examples:

## Independent scenarios with shared controls

```text
      B1
       \
B2 ---- G ---- B3
       /
      B4
```

This suggests an arrowhead Schur complement.

## Temporal horizon model

```text
B1 -- B2 -- B3 -- B4 -- B5
```

This suggests block-tridiagonal elimination.

## Tree structured model

```text
        B1
       /  \
     B2    B3
    / \     \
   B4 B5     B6
```

This suggests recursive tree elimination.

## Generic model

```text
dense irregular graph
```

Fallback to ordinary sparse LDLᵀ.

---

# 12. Proposed POUNCE Additions

POUNCE consumes solver-neutral structure through one entry point on `Problem`, mirroring `set_ordering` and `set_kkt_schur_block`:

```python
prob.set_block_structure(structure)   # NLP indices; see §45
prob.clear_block_structure()
```

The structure uses **NLP variable and constraint indices**, never KKT positions; POUNCE owns the mapping to KKT indices (§45). This is the only interface POUNCE needs, and it is what POUNCE's own tests and benchmark generators call, so the POUNCE side never waits on `discopt`.

On the `discopt` side, the user-facing form is expected to be:

```python
result = model.solve(solver="pounce", structured_kkt=True)
```

which builds the structure and calls `set_block_structure`. That is `discopt` work (draft issue I2, §28).

---

# 13. POUNCE Internal Responsibilities

POUNCE should:

1. receive model structure,
2. map model variables and constraints to current NLP indices,
3. map NLP indices to KKT indices,
4. determine the block topology,
5. decide whether structured elimination is worthwhile,
6. select a strategy,
7. coordinate factorization and solves,
8. preserve inertia and regularization logic,
9. fall back safely to the generic KKT solver when necessary.

POUNCE should not depend on domain-specific concepts such as “gas horizon” or “grid contingency.”

It should only understand generic structures:

- blocks,
- separators,
- global variables,
- block graph.

---

# 14. Strategy Selection

A simple first version could classify structure as:

```text
no useful structure
    -> generic sparse LDLᵀ

single small separator
    -> existing Schur complement path

star / arrowhead
    -> independent block elimination + Schur complement

chain
    -> structure-derived ordering (Phase 2);
       block-tridiagonal elimination only if Phase 4 is justified

tree
    -> recursive tree elimination

generic sparse block graph
    -> graph partitioning / nested decomposition / fallback
```

The first implementation does not need all of these. Choosing among them automatically is §47, and comes after Phase 4.

---

# 15. Existing POUNCE Hooks to Exploit First

Before adding new FERAL algorithms, use existing POUNCE mechanisms as a prototype.

Two useful capabilities already discussed are:

```python
prob.set_ordering(...)
```

and

```python
prob.set_kkt_schur_block(...)
```

These suggest a staged implementation.

## Stage A: ordering only

Use `discopt` structure to generate a KKT permutation. Choose it by topology:

- **Arrowhead (scenarios plus globals):** all block-local unknowns first, block by block; globals last.
- **Chain (horizons):** block-level **nested dissection**. The middle separator is eliminated last, then each half is handled recursively, giving independent subtrees per block and small separator fronts.

Do **not** use "all locals first, all links last" on a chain. It puts every linking unknown (as many as \(T \cdot m\)) into one root front, a dense matrix that reproduces the §48 dense trap through the ordering.

Stage A is not a detour into faster ordering. It is the decomposition expressed as an elimination order, and it needs the same metadata and index mapper as a structured solver does. On a chain it directly tests the fill row of §38.2.

Benchmark whether better ordering reduces:

- fill,
- factorization time,
- peak memory,
- total runtime.

This requires minimal new solver code.

## Stage B: current Schur-block mechanism

Map the KKT indices of the **global** variables (and the duals of constraints that touch only globals) into:

```python
prob.set_kkt_schur_block(global_indices)
```

Benchmark against the best generic ordering. Do not use this path for temporal links; see the limitation below and Phase 3.

Limitation:

If the current Schur implementation forms a dense reduced matrix, it will stop being attractive when the linking system grows.

For example:

- 50 linking variables × 100 horizons = 5,000 reduced unknowns.

A dense \(5000\times5000\) factorization is undesirable when the actual temporal coupling graph is banded.

---

# 16. Block-Graph Solver (Phase 4, if justified)

If Phases 2–3 leave a measured gap on some topology, the target is a block-graph solver, with one reduced solve per topology (dense or parallel Schur for arrowhead, block-tridiagonal for chain):

```text
large local blocks
      |
      v
independent / partially independent factorizations
      |
      v
sparse reduced coupling system
      |
      v
structured solve
      |
      v
parallel back-substitution
```

The critical difference from a single dense Schur complement is:

> preserve sparsity in the reduced block graph.

For a temporal problem, the reduced system should remain block-tridiagonal or narrowly banded. §17–18 describe that case; they apply only if Phase 4 is taken for the chain.

---

# 17. Local Block Elimination

Suppose each horizon produces a local linear system

\[
K_t \Delta x_t
+
E_t \Delta z_t
+
F_t \Delta z_{t+1}
=
r_t.
\]

After factoring \(K_t\),

\[
\Delta x_t
=
K_t^{-1}
\left(
r_t
-
E_t\Delta z_t
-
F_t\Delta z_{t+1}
\right).
\]

Substitution contributes only to coupling blocks involving \(z_t\) and \(z_{t+1}\).

Therefore the reduced matrix remains local in time.

This is the structure that should be preserved.

---

# 18. Block-Tridiagonal Reduced System

The reduced coupling system may have the form

\[
\begin{bmatrix}
S_1 & C_1^T & 0 & \cdots \\
C_1 & S_2 & C_2^T & \cdots \\
0 & C_2 & S_3 & \cdots \\
\vdots & & & \ddots
\end{bmatrix}
\Delta z
=
\hat r.
\]

This can be solved using:

- block LDLᵀ,
- block Thomas-like elimination,
- recursive elimination,
- sparse direct factorization with guaranteed block ordering.

If separator blocks are small, this should be far cheaper than factoring the full KKT system.

---

# 19. Relationship to Multiple Shooting / Optimal Control

The structure is closely related to methods used in:

- multiple shooting,
- optimal control,
- MPC,
- dynamic optimization.

The lesson from those fields is that temporal coupling should not be flattened into a generic sparse problem *unless* the generic solve recovers the same elimination. A structure-derived ordering (§15, Stage A) may do exactly that, and Phase 2 measures whether it does. What an explicit chain solve adds beyond the right ordering is pivoting confined to blocks (§24) and Riccati-style reuse.

The gas-network model is particularly attractive because:

- local networks may be large,
- temporal state dimension may be comparatively small,
- the horizon graph is naturally ordered,
- the same decomposition repeats over many IPM iterations.

---

# 20. Repeated Symbolic Structure

POUNCE should exploit the fact that the structural pattern is usually constant across IPM iterations.

The ideal sequence is:

```text
once:
    determine block graph
    determine local KKT sparsity
    determine reduced sparsity
    compute orderings
    allocate workspaces

every POUNCE iteration:
    update numeric KKT entries
    refactor local blocks
    assemble reduced numeric matrix
    factor reduced system
    back-substitute
```

Symbolic analysis should not be repeated unnecessarily.

---

# 21. Parallelism

Many local factorizations may be independent.

For scenario or contingency problems:

```text
factor K1 ─┐
factor K2  ├── parallel
factor K3  │
factor K4 ─┘
```

For a temporal chain, blocks are coupled through separators, but local block factorization may still be parallelizable before reduced-system assembly.

Parallelism could occur at several levels:

- across blocks,
- within FERAL sparse factorization,
- across triangular solves,
- potentially on GPU in future work.

FERAL already parallelizes factorization over the elimination tree (the `feral_min_par_flops` gate). With a nested-dissection ordering, independent subtrees, and so independent blocks, already factor concurrently. Block-level parallelism (Phase 5) must therefore be measured against FERAL's own threading, not against a single thread.

The first implementation should focus on structural gains before adding complex concurrency.

---

# 22. Inertia and Indefinite KKT Systems

A major issue is that POUNCE depends on properties of the KKT factorization beyond simply solving \(Ax=b\).

The structured solver must preserve or reconstruct:

- inertia counts,
- regularization behavior,
- pivot stability,
- failure detection,
- iterative refinement,
- residual checks.

For symmetric block elimination, inertia obeys additive relationships under suitable factorizations.

For example, for

\[
K=
\begin{bmatrix}
A & B \\
B^T & D
\end{bmatrix},
\]

when the elimination is valid,

\[
\operatorname{inertia}(K)
=
\operatorname{inertia}(A)
+
\operatorname{inertia}(D-B^TA^{-1}B).
\]

This is potentially very useful:

- local block inertia can be accumulated,
- reduced-system inertia can be added,
- POUNCE can retain its usual inertia-based regularization logic.

This should be designed explicitly rather than added later.

---

# 23. Numerical Stability

Potential issues include:

- nearly singular local KKT blocks,
- unstable separator choice,
- local pivots requiring cross-block pivoting,
- poor conditioning in the Schur complement,
- amplification from explicitly forming \(A^{-1}B\),
- dense fill in the reduced system,
- inconsistent regularization between blocks.

Important principle:

> never explicitly form an inverse.

Compute quantities through solves:

\[
A^{-1}B
\]

means solve

\[
AX=B.
\]

Likewise, Schur contributions should be built through stable factor/solve operations.

---

# 24. Pivoting Complications

A generic indefinite LDLᵀ solver may pivot across the proposed block boundary.

A strict block-elimination strategy restricts that freedom.

Therefore there is a tradeoff between:

- preserving decomposition,
- allowing unrestricted numerical pivoting.

Possible approaches:

1. regularize local KKT blocks sufficiently to make block-local factorization stable,
2. use static or bounded pivoting within each block,
3. detect unstable blocks and fall back to generic factorization,
4. merge adjacent blocks when necessary,
5. permit adaptive decomposition.

This is likely one of the most important implementation details.

---

# 25. Adaptive Block Merging

A useful robustness mechanism is:

```text
try block-local factorization
        |
        +-- stable -> continue
        |
        +-- unstable -> merge with neighbor / separator
```

For a temporal chain:

```text
B1 -- B2 -- B3 -- B4
```

could become:

```text
B1 -- [B2+B3] -- B4
```

if one local block is numerically problematic.

This avoids forcing decomposition when the numerical structure does not support it.

---

# 26. Proposed POUNCE API

A minimal high-level interface:

```python
result = model.solve(
    solver="pounce",
    structured_kkt=True,
)
```

Optional explicit control:

```python
result = model.solve(
    solver="pounce",
    structured_kkt={
        "strategy": "auto",
        "fallback": True,
        "parallel": True,
    },
)
```

Advanced use:

```python
structure = discopt.decomposition.analyze(model)

result = pounce.solve(
    model,
    block_structure=structure,
    kkt_strategy="block",
)
```

Strategy names do not depend on topology: `generic`, `ordering`, `schur_globals`, `block`. The topology is reported, not encoded in the name.

Debugging:

```python
result = pounce.solve(
    model,
    block_structure=structure,
    kkt_strategy="generic",
)
```

This makes A/B testing straightforward.

---

# 27. Diagnostics

The solver should expose decomposition diagnostics.

Example:

```text
Structured KKT analysis
-----------------------
blocks:                   48
topology:                 chain
local KKT size:
    min:                  8,921
    median:               9,114
    max:                  9,307

separator size:
    min:                     42
    median:                  47
    max:                     53

global variables:            38

estimated full KKT nnz:   12.8 M
estimated reduced nnz:     0.19 M

strategy:                 ordering
fallback enabled:             yes
path actually taken:      ordering (no fallback)
```

Runtime statistics should include:

```text
symbolic analysis
local numeric factorization
Schur assembly
reduced factorization
backsolve
iterative refinement
total KKT solve
total POUNCE iteration
```

These measurements are critical for determining whether the architecture actually improves scaling.

---

# 28. Work Split and Draft `discopt` Issues

POUNCE-side work is §57. `discopt`-side work is listed here as **drafts**. Each is filed in `discopt` only when the phase that needs it is reached, and only after approval. None is needed to start.

*Status 2026-09-20 (later the same day): **implemented and merged on the
discopt side** — `python/discopt/block_structure.py`, a `block_structure=`
passthrough in `solvers/nlp_pounce.py`, compressed derivatives over declared
blocks for Part B, and `python/tests/test_1370_block_structure.py`. Verified
end to end here against this branch's POUNCE: a 24-block arrowhead declared in
discopt reaches `block-parallel KKT path engaged blocks=24 border=1` and the
partition is reported back through `NLPResult.linear_solver`. That run also
found the crossover this branch now guards (`kkt_block_min_size`) — see
`dev-notes/kkt-scaling-phase0b.md`. Two of the numbers in the issue were
corrected by their measurements: evaluation is 20-24% of the solve on the tape
path rather than 48% on the `.nl` path, so Part B is worth ~1.45x after Part A
rather than being "the larger half".*

*Status 2026-09-20: filed, as **one** issue — jkitchin/discopt#1370, "Export
model block structure to pounce, and evaluate identical blocks in one pass".
I1 and I5 are its two halves and I2 is folded in as the passthrough, because
I5 needs I1's membership and I1 alone buys only the factorization share, which
is about a third of these solves. I3 is dropped: the gas question was answered
by measurement and does not need a block-structured KKT. I4 stays unfiled —
POUNCE has a detector, and the declaration is what covers the cases it misses.
Two things the sketch in §45 asked for were dropped from the ask as well:
`coupling_groups` / `graph_edges`, because the implemented solver is arrowhead
and labels alone suffice, and §46's `structure_validation` modes, because
POUNCE validates and falls back on its own.*

| Draft | Title | Needed by | Summary |
|---|---|---|---|
| **I1** | Export solver-neutral block structure with NLP indices | Phase 1, end-to-end test | Block membership for variables **and** constraints, links, globals, block graph, in the NLP index space POUNCE sees (§8, §45). Stable identity through `discopt`'s transformations (§41). Validation of declared structure against incidence. |
| **I2** | `solve(structured_kkt=...)` passthrough to `Problem.set_block_structure` | Phase 2 on models built in `discopt` | Builds I1's structure and hands it to POUNCE. Depends on I1. |
| **I3** | Multi-horizon gas model emitted at \(T = 1 \dots 64\) | Phase 0b, *only if* D2 says the gas instances come from `discopt` | `.nl` instances per horizon count, same network and data. |
| **I5** | Block-parallel / vectorised evaluation of identical blocks | Phase 5a — **the largest measured win** | Evaluate every block in one pass, sharing the one sparsity pattern and coloring they have in common: measured 0.044× the one-at-a-time cost per directional derivative on 64 SCOPF blocks, i.e. ≈ 29 ms (Jacobian) and ≈ 22 ms (Hessian) against the `.nl` path's 107 ms and 463 ms. Needs I1's block membership. POUNCE needs no change. |
| **I4** | Automatic structure detection yielding a block graph | After Phase 4 | §10. Not on the critical path. |

---

# 29. (Superseded)

Merged into §57, Phase 2. Note: benchmark against the **best** generic ordering (§38.1), not the default.

---

# 30. (Superseded)

Merged into §57, Phase 3. The existing Schur path is used for **global variables only**. Its sweep is capped at a size whose two dense \(n_f \times n_s\) buffers fit in memory (about 40 GB at \(n_f = 5 \times 10^5\), \(n_s = 5000\)), unless streaming formation of \(S\) is implemented first.

---

# 31. (Superseded)

Merged into §57, Phase 4. The first block-solver topology is whichever one Phases 2–3 leave a measured gap on, not the chain by default.

---

# 32. Proposed FERAL / `pounce-linsol` Additions

**Phase 0a needs FERAL-side visibility.** `FactorStats` exposes `nnz_l`, `fill_ratio`, `n_tiny`, and `ordering_info` (`n_supernodes`, `max_front_rows`), and the symbolic analysis carries `factor_flops`. There is **no aggregate delayed-pivot count** in the public API (it exists per front only). Either add one to FERAL, or use `n_tiny` plus factor time against `nnz_l` as the proxy for the pivoting row of §38.2.

Phases 1–3 need no other FERAL changes. For Phase 4, useful primitives may include:

```rust
factor(A)
solve(factor, rhs)
solve_multiple(factor, B)
inertia(factor)
refine(factor, A, rhs, x)
```

Potential structured abstractions:

```rust
BlockFactor
BlockSchurSystem
ChainFactorization
```

However, keep model-level topology out of FERAL if possible.

POUNCE should coordinate the block algorithm.

FERAL should provide efficient numerical kernels.

---

# 33. Acceptance Test: Exactness

The structured linear solve should reproduce the generic KKT solution.

For a test system:

\[
Kx=b
\]

compare:

```text
x_generic
x_structured
```

Require:

\[
\frac{\|x_s-x_g\|}{1+\|x_g\|}
< \epsilon
\]

and independently:

\[
\frac{\|Kx_s-b\|}{1+\|b\|}
< \epsilon.
\]

Also compare inertia:

```text
generic inertia == structured inertia
```

where applicable.

---

# 34. Acceptance Test: POUNCE Equivalence

Solve the same NLP using:

```python
solve(..., structured_kkt=False)
```

and

```python
solve(..., structured_kkt=True)
```

Compare:

- objective,
- primal feasibility,
- dual feasibility,
- complementarity,
- final variables,
- multiplier values,
- termination status.

Assert status, objective and KKT residuals within tolerance. **Report** the change in iteration count as a measurement; do not require it to match. Elimination order changes pivoting and inertia-based regularization, and so changes the trajectory (§58).

---

# 35. Synthetic Benchmark Family

Before relying only on the gas model, create controlled synthetic problems.

Two families, one per topology:

```text
chain:  T blocks, n local variables/block, m linking variables/interface
star:   k blocks, n local variables/block, g global variables
```

Sweep:

```text
T = 2, 4, 8, 16, 32, 64, 128
n = 500, 1000, 2500, 5000, 10000
m = 5, 10, 25, 50, 100, 250
```

Generate sparse local NLPs with fixed local degree. Sweep one axis at a time around a base point before attempting the full grid.

This makes it possible to distinguish:

- dependence on number of blocks,
- local block size,
- separator size,
- total NLP size.

---

# 36. Critical Scaling Plots

Measure:

\[
T_{\text{KKT}}
\]

versus total variables \(N\).

Compare:

```text
best generic ordering (auto / metis / auto_race)
structure-derived ordering
Schur on globals
block solver (if Phase 4)
IPOPT
```

Fit empirical scaling:

\[
T = cN^\alpha.
\]

The key question is whether structured elimination reduces the exponent or mainly reduces the constant.

Also plot:

```text
factor nnz vs N
peak memory vs N
numeric factorization time vs N
separator solve time vs T
iteration count vs N
```

---

# 37. Gas-Network Benchmark

Use the real multi-horizon gas planning problem as the principal application benchmark.

Recommended sequence:

```text
1 horizon
2 horizons
4 horizons
8 horizons
16 horizons
32 horizons
64 horizons
full production model
```

For every case, record:

- primal variables,
- equality constraints,
- inequality constraints,
- KKT dimension,
- KKT nnz,
- local block dimension,
- separator dimension,
- fill,
- IPM iterations,
- factorization time,
- total solve time,
- peak memory,

plus the Phase 0 columns (§57): `factor_flops`, delayed pivots or their proxy, and time per iteration outside the factorization. This directly locates the observed near-\(N^2\) behavior in §38.2.

---

# 38. Hypothesis to Test

## 38.1 General hypothesis

> For NLPs made of weakly coupled blocks, carrying declared block structure into the KKT solve improves at least one of wall time, memory, parallel speedup, or scaling exponent, compared with the generic sparse solve under its best ordering. The improvement differs by topology and is measurable per topology.

"Best ordering" means the best of `auto`, `metis` and `auto_race`, not the default. Otherwise a gain that is really an ordering gain gets credited to decomposition.

## 38.2 The gas instance: attribute the exponent first

For a chain of blocks with small separators, sparse direct factorization is not intrinsically quadratic. An exact block elimination is itself an elimination order, and its cost is linear in the number of horizons \(T\). So an observed \(T^2\) under `metis` lives somewhere specific, and whether decomposition fixes it depends on where:

| Where the \(T^2\) lives | Diagnostic (fit an exponent in \(T\)) | Fixed by block decomposition? |
|---|---|---|
| **Fill:** the partitioner cuts in the wrong place (through the network in space, or a front as wide as the whole horizon set built from globals or hidden couplings) | factor nnz and flops | **Yes.** Declared structure forces the temporal cut. |
| **Pivoting:** delayed and 2×2 pivots on the indefinite KKT carry work across block boundaries | delayed-pivot count; factor time against factor nnz | **Yes.** Pivoting confined to blocks (§24) is the payoff here. |
| **Iterations:** IPM iteration count grows with \(T\) | iterations | **No.** No KKT solve changes this. Look instead at initialization, e.g. from single-horizon solves. |
| **Other per-iteration work** grows superlinearly | per-iteration time outside the factorization | **No.** This is a code defect outside the linear solve. |

The gas hypothesis is then stated per row:

> Under `metis`, the superlinear growth in factorization cost with \(T\) is in the fill row, the pivoting row, or both. A block-level nested-dissection ordering built from declared horizon structure (§15, Stage A) brings fill to linear in \(T\). If factor time is still superlinear after that, pivoting confined to blocks (Phase 4) brings it to linear.

Each clause is falsifiable, and the Phase 0 measurements decide which clause the rest of the gas work tests.

## 38.3 The gas instance after Phase 0b: which block structure?

Phase 0b placed the gas growth in the fill row and showed the temporal chain is the wrong structure (§ "The first instance"). The structure worth testing is the model's own geometry:

- **spatial blocks**: subnetworks of the pipeline graph, with compressor stations or a few cut pipes as separators, each carrying its full time trajectory;
- **space-time tiles**: a network partition crossed with coarse time blocks, eliminated by a *structure-derived* two-dimensional nested dissection.

The test is Phase 2's (a structure-derived ordering through the existing external-ordering hook) against `metis` / `auto_race`, not the chain Schur solver. The bar is high: `metis` is already near the two-dimensional asymptote at the top of the measured range, so a win shows up as a constant, and must clear D3's thresholds.

---

# 39. Expected Failure Modes

The idea may fail to help when:

- separator dimension is large relative to local block size,
- coupling is dense across horizons,
- global variables connect nearly everything,
- local blocks are numerically singular,
- pivoting destroys the desired structure,
- Schur complements become dense,
- local factorization dominates anyway,
- POUNCE iterations increase because of altered regularization,
- current sparse ordering already discovers almost the same elimination structure.

These cases should be measured rather than assumed away.

---

# 40. (Merged into §38.1)

The control experiment, generic ordering against structure-derived ordering against explicit elimination, is now the "best ordering" baseline of §38.1 and the arms of §36.

---

# 41. Why Model Provenance Matters

Automatic sparse graph analysis sees only:

```text
matrix entry (i,j) is nonzero
```

`discopt` can know:

```text
constraint 37 belongs to horizon 19
variable 88 is linepack connecting horizons 19 and 20
variable 3 is a global investment decision
```

That semantic provenance can be much more reliable than reverse-engineering structure from the final matrix.

Therefore decomposition metadata should survive transformations from:

```text
domain model
    -> discopt expressions
    -> NLP representation
    -> POUNCE
    -> KKT indexing
```

---

# 42. Generality Beyond Gas Networks

The same mechanism applies to many problem families.

## Grid security-constrained OPF

```text
shared dispatch
   |
   +-- contingency 1
   +-- contingency 2
   +-- contingency 3
```

## Multi-scenario stochastic optimization

```text
first-stage variables
   |
   +-- scenario 1
   +-- scenario 2
   +-- scenario 3
```

## Batch optimization

```text
shared design
   |
   +-- batch 1
   +-- batch 2
```

## Dynamic process optimization

```text
time 1 -- time 2 -- time 3 -- time 4
```

## Spatial domain decomposition

```text
region 1 -- interface -- region 2
```

This is why the capability belongs in generic `discopt` + POUNCE infrastructure.

---

# 43. Relationship to `discopt-grid`

A future `discopt-grid` package would benefit directly.

For security-constrained OPF:

- each contingency is a block,
- dispatch or corrective controls may be shared,
- each grid block is large and sparse,
- block solves can potentially run in parallel.

For multi-period OPF:

- each time point is a block,
- storage state and ramping variables create temporal links,
- the reduced system is naturally banded.

Thus `discopt-grid` should consume the generic structure rather than implement its own decomposition engine.

---

# 44. Suggested Module Boundaries

POUNCE is Rust with PyO3 bindings. Proposed placement:

```text
pounce/
    crates/pounce-algorithm/src/kkt/
        block_structure.rs       # NLP-index structure, validation (§46)
        kkt_index_map.rs         # NLP -> KKT mapping (§45, Phase 1)
        structured_ordering.rs   # block-level ND / locals-then-globals (Phase 2)
        schur_aug_system_solver.rs   # existing; globals (Phase 3)
        block_aug_system_solver.rs   # Phase 4, if justified
    crates/pounce-py/src/problem.rs  # set_block_structure (§12)
    benchmarks/                      # gas sweep, N-1 SCOPF generator, synthetics

discopt/                             # via approved issues only (§28)
    decomposition/ ...
```

Exact names should follow existing project conventions.

---

# 45. Solver Structure Object

A useful boundary object might look like:

```python
@dataclass
class SolverBlockStructure:
    # NLP indices throughout
    block_of_variable: list[int]      # -1 = global / separator
    block_of_constraint: list[int]    # -1 = linking / global
    coupling_groups: list[CouplingGroup]
    graph_edges: list[tuple[int, int]]
```

Constraint membership is required, not optional: the mapper places each constraint's dual on one side of the partition. A dual whose row touches only separator or global variables must go with them, or the eliminated block is singular.

Important:

The object passed from `discopt` to POUNCE should use stable model/NLP identifiers, not KKT positions.

POUNCE should own the mapping to KKT indices because KKT layout is solver-specific.

---

# 46. Structure Validation

Before structured solving, POUNCE should validate that the supplied structure matches the actual derivatives.

For example:

1. compute Jacobian **and** Lagrangian-Hessian sparsity (a cross-block objective term couples blocks without appearing in the Jacobian),
2. examine all nonzeros,
3. verify that each coupling is allowed by the declared block graph,
4. detect undeclared cross-block terms.

If a hidden cross-block derivative exists:

```text
declared:
B1 -- B2 -- B3

actual:
B1 ------ B3
```

POUNCE should:

- update the graph automatically, or
- reject the structure in strict mode.

Possible modes:

```python
structure_validation="strict"
structure_validation="repair"
structure_validation="off"
```

---

# 47. Automatic Strategy Heuristics

POUNCE could estimate whether decomposition is worthwhile.

Example metrics:

```text
local block sizes
separator size
separator / total size
block graph degree
estimated reduced nnz
estimated dense Schur memory
estimated fill
```

Heuristic:

```text
if separator is tiny:
    dense Schur

elif graph is a chain and separators are modest:
    structure-derived ordering (block solver if Phase 4 justified it)

elif graph is a star:
    arrowhead Schur

else:
    generic sparse LDLᵀ
```

Initially, keep heuristics conservative. Automatic selection comes after Phase 4. Once it routes the default path it is a trajectory change and needs the fixture sweep on both legs (§58).

---

# 48. Dense Schur Crossover

The existing Schur path may be ideal when the separator dimension \(n_s\) is small.

Dense storage:

\[
O(n_s^2)
\]

Dense factorization:

\[
O(n_s^3).
\]

Therefore a key benchmark is identifying the crossover where dense Schur loses.

This depends on:

- hardware,
- BLAS,
- local block cost,
- generic sparse fill,
- separator density.

Do not hard-code a threshold without empirical data.

---

# 49. Sparse Schur Assembly

For temporal problems, avoid globally dense assembly.

Each horizon should contribute only to nearby separator blocks.

Conceptually:

```python
for block in blocks:
    contribution = block.compute_schur_contribution()

    reduced.add(
        rows=block.separator_rows,
        cols=block.separator_cols,
        values=contribution,
    )
```

Because each block touches only one or two separators, the reduced matrix remains sparse.

---

# 50. Matrix-Free Schur Option

A later alternative is to avoid explicitly forming the Schur matrix.

Define the action:

\[
v \mapsto Sv
=
Dv
-
BA^{-1}B^Tv.
\]

This permits Krylov methods on the reduced system.

Potentially useful when:

- separators become large,
- local solves are cheap,
- good preconditioners are available.

This should not be the first implementation because it introduces iterative-solver complexity.

---

# 51. Relation to Iterative Solves

The structured decomposition itself does **not** imply an iterative decomposition.

There are three nested levels to keep distinct:

## Nonlinear/IPM iteration

POUNCE updates the nonlinear solution.

## Structured direct KKT solve

Local factorization + reduced solve + back-substitution.

## Optional iterative linear solve

A future matrix-free Schur implementation might use Krylov methods.

The first structured implementation should ideally remain direct.

---

# 52. Warm Starts and Repeated Solves

This architecture could also help when repeatedly solving closely related planning problems.

Reusable information includes:

- block graph,
- symbolic factorization patterns,
- orderings,
- memory allocations,
- reduced graph sparsity,
- possibly numerical factors for unchanged blocks.

Potential use cases:

- rolling horizons,
- parameter sweeps,
- scenario updates,
- repeated planning solves,
- continuation.

---

# 53. Potential Further Optimization: Partial Refactorization

If only one horizon changes between solves, a structured representation might eventually permit:

```text
reuse factors for B1
reuse factors for B2
refactor B3
reuse factors for B4
...
update reduced system
```

This is much harder with a monolithic KKT factorization.

This is a later optimization, not part of the initial implementation.

---

# 54. Debugging Mode

Add a mode that assembles both representations:

```python
debug_structured_kkt=True
```

For small test cases:

1. assemble full KKT,
2. solve generically,
3. solve structurally,
4. compare steps,
5. compare residuals,
6. compare inertia.

This will be invaluable during development.

---

# 55. Unit Tests

Recommended unit-test categories:

## Structure detection

- independent blocks,
- chain,
- star,
- tree,
- global variable,
- hidden cross-block constraint.

## Mapping

- model variable -> NLP index,
- NLP index -> KKT index,
- eliminated/fixed variables,
- reordered constraints.

## Linear algebra

- two-block Schur,
- three-block chain,
- block-tridiagonal solve,
- indefinite blocks,
- singular block handling,
- inertia combination.

## Solver integration

- identical Newton step at the same iterate (linear-algebra test),
- same status and objective within tolerance,
- fallback on failure.

---

# 56. Performance Tests

Performance tests should be separate from correctness tests.

Track regressions in:

```text
factor nnz
peak allocation
symbolic time
numeric factorization time
reduced solve time
overall solve time
```

Set broad performance thresholds to avoid flaky CI.

Run larger scaling studies offline or in benchmark CI.

---

# 57. Suggested Development Order

Two benchmark problems run through every phase from Phase 1 on:

- **Chain:** the multi-horizon gas-network model (the first instance; source per D2).
- **Arrowhead:** N-1 security-constrained OPF (per D1). It keeps the block-graph interface from ending up shaped only for chains, and it is the case the existing Schur path suits.

All phases are POUNCE work unless marked **[discopt: In]**, which means a draft issue from §28 that is filed only when that phase is reached and approved.

## Phase 0 — inspect, instrument, measure, attribute

*Status 2026-09-19: Phase 0 is complete, and it reorders what follows —
parallelism (Phase 5) ahead of the block solver (Phases 3–4), which nothing
measured so far justifies. See "After Phase 0" in §58 and
`dev-notes/kkt-scaling-phase0b.md`.*

**0a done** (branch `feat/feral-factor-stats`: factorization time, work proxy, delayed / 2×2 / tiny pivots, largest front, ordering used, Schur breakdown, restoration reported apart; two summary-reporting defects fixed). **0b done** (`benchmarks/kkt_scaling/sweep.py`): fill row; chain structure refuted for gas, and a network × time structure-derived ordering measured and beaten by `metis`. **0c done** (`benchmarks/kkt_scaling/gen_scopf.py`): the arrowhead's ordering is already right, and the remaining cost is block-separable evaluation (48%) and factorization (35%). **0d** (routing) open. All numbers: `dev-notes/kkt-scaling-phase0b.md`.

**0a. Instrumentation (one PR, POUNCE).** Surface per-factorization FERAL stats in `info` and the solve report: `nnz_l`, `fill_ratio`, `factor_flops`, `max_front_rows`, `n_tiny`, `ordering_info.used`, and a delayed-pivot count if FERAL can provide one (§32). Make sure factorization time per iteration is available (`timing_statistics="yes"` today). Without this PR, §38.2 cannot be measured.

**0b. Gas horizon sweep (no solver changes).** Run the gas model at \(T = 1, 2, 4, \dots, 64\) under `auto`, `metis` and `auto_race`. **[discopt: I3]** only if D2 says the instances come from `discopt`.

**0d. Ordering routing (from 0b).** The default `auto` never selects nested dissection and costs 8× at 224k on the space-by-time family; `auto_race` gets it right there. Make the right ordering reachable without the user knowing to ask. A default-path change: needs the fixture sweep on both legs and `benchmarks/qp`, where `metis` loses.

**0c. Arrowhead generator (POUNCE `benchmarks/`).** *Done:*
`benchmarks/kkt_scaling/gen_scopf.py` — corrective N-1 AC-SCOPF, `case118_ieee`
to K = 128 and `case1354_pegase` to K = 64 (210k variables). The synthetic
chain and star families (§35) are still open.

Also inspect and document:

- current `discopt.decomposition` representations,
- POUNCE's KKT construction and its extension point (`AugSystemSolver`, which `SchurAugSystemSolver` already wraps),
- the existing hooks and their coverage gaps: `set_ordering` and `set_kkt_schur_block` take full-KKT indices (x, s, y_c, y_d); restoration ignores both; the Schur path is exact-Hessian and FERAL only, and L-BFGS bypasses it; fallback is silent and permanent,
- how the prior art (Purpose section) handles restoration, inertia correction and quasi-Newton Hessians.

Measure in 0b:

- IPM iterations,
- factor nnz and flops,
- delayed and 2×2 pivot counts,
- factorization time per iteration,
- per-iteration time outside the factorization.

Fit an exponent in \(T\) for each, and place the gas \(T^2\) in the rows of §38.2.

**Exit criteria:** every §38.2 row has a fitted exponent for each of the three orderings, and D1–D4 are resolved.

## Phase 1 — block structure API and index mapper (general)

```text
declared structure (NLP indices)      <- benchmark generators now; discopt via I1 later
      ->
Problem.set_block_structure (§12)
      ->
KKT index map
```

- **`Problem.set_block_structure`** taking the §45 object in NLP indices, fed in tests by the Phase 0c generators.
- **Mapper in POUNCE** from model entities to KKT indices. It must handle slacks, variables removed by `fixed_variable_treatment=make_parameter`, and the placement of linking-constraint duals. A dual whose row touches only separator variables must go to the separator side, or the eliminated block is singular.
- **Validation (§46)** against Jacobian **and** Lagrangian-Hessian sparsity. A cross-horizon objective term couples blocks without appearing in the Jacobian.
- **Diagnostics (§27)** reporting which path actually ran, including any fallback.

No new numerical algorithm. Every item is independent of topology.

**[discopt: I1]** when the first end-to-end test on a model built in `discopt` is wanted. POUNCE's Phase 1 is complete without it.

## Phase 2 — structure-derived ordering

Build the permutation from the declared structure inside POUNCE, and apply it through the existing external-ordering path (§15, Stage A): block-level nested dissection on the chain, locals-then-globals on the arrowhead. Benchmark against the best generic ordering (§36).

**[discopt: I2]** once gas is built directly from `discopt` rather than from emitted `.nl`.

## Phase 3 — existing Schur path, arrowhead only (**not justified by Phase 0; do not start without a new reason**)

Use `set_kkt_schur_block` for the **global** variables of the arrowhead problem and of the gas model (\(n_s\) in the tens). Do not use it for the temporal links: it factors the eliminated block as one matrix and materializes two dense \(n_f \times n_s\) buffers, about 40 GB at \(n_f = 5 \times 10^5\), \(n_s = 5000\). Either cap the §30 sweep at a size that fits in memory, or implement column-streamed formation of \(S\) first.

## Phase 4 — block solver on the `AugSystemSolver` extension point (**not justified by Phase 0**; reachable only as 5b's serial fallback or if a new topology shows fill the ordering misses)

Generalize `SchurAugSystemSolver` from a two-block partition to a block graph:

- per-block factorization with pivoting kept inside the block,
- inertia by adding the block inertias,
- a reduced solve for the topology: dense or parallel Schur for arrowhead, block-tridiagonal (Riccati-like) for chain,
- fallback to merging blocks (§25).

Start with the topology whose Phase 2–3 results left a measured gap that ordering could not close.

## Phase 5 — parallel blocks (**the phase Phase 0 promoted**)

Phase 0c measured where the time goes on a 210k-variable arrowhead: evaluation 48% (Lagrangian Hessian 36%), factorization 35%, back-solve 11% — and both large terms are block-separable, while the ordering has nothing left to give. So this phase, not a new KKT solver, carries the general capability.

**5a. Block-parallel evaluation (frontend; the larger half).** Identical blocks share one sparsity pattern and one coloring, so a frontend that knows the structure evaluates them in one vectorised pass. Measured in JAX on `case1354_pegase`: a directional derivative over all 64 blocks costs 0.044× the one-at-a-time cost, giving ≈ 29 ms (Jacobian, 29 colors) and ≈ 22 ms (Hessian, 22 colors) against the `.nl` path's 107 ms and 463 ms. This is `discopt` work (draft issue I5, §28); POUNCE's part is only to accept the derivatives.

**5b. Block-parallel factorization (POUNCE / FERAL).** FERAL's tree parallelism reaches 1.2× on this workload and degrades past four threads, because the factorization is per-supernode-overhead bound: 1.22e9 flops over 196 561 supernodes. Declared blocks give 65 coarse independent tasks instead. The prototype question is whether factoring each block on its own thread — with a Schur complement on the ~259 shared columns — recovers the parallelism the elimination tree cannot.

Both are constants, not exponents. Measure against FERAL's own threading, never against a single thread.

## Phase 6 — coverage

Extend structure into the restoration KKT system and into quasi-Newton Hessians: a Hessian approximation partitioned by block (`partitioned_quasi_newton.rs` exists) instead of bypassing the structured path. Until this lands, iterations in restoration and every L-BFGS solve get no benefit, and diagnostics must say so.

## Phase 7 — trees and arbitrary block graphs

Only if justified by real applications.

---

# 58. Go / No-Go Gates

Gates are judged **per topology**, against the **best generic ordering**, and on the full set of benefits in the Purpose section, not only the scaling exponent. A no-go on one topology does not stop the capability; it stops that strategy for that topology.

"Material" and "substantial" below mean the D3 thresholds. Proposed: an exponent drop of at least 0.3 in the block count, or at least 2× at the largest instance, or at least 30% less peak memory; for parallelism, efficiency above 50% at 8 threads, measured against FERAL's own threading.

## After Phase 0 — **complete; it reorders the plan**

The original test (which row of §38.2 carries the growth) resolved to **fill**, on both topologies, and then the follow-through changed the plan:

- **Gas.** The chain structure is refuted (20–40× worse as an ordering). A network × time structure-derived ordering, built with the spatial cells, time spans and primal-dual pairs it needs, matched `metis` at 56k and lost 44–54% at 112k–224k end to end. The available win is *routing* (0d): `auto` never selects nested dissection and costs 8× at 224k.
- **Arrowhead.** Generic orderings already find it: fill exactly linear in the block count, largest front bounded by the block, the three orderings within ~10%. Factorization is 25–40% of the solve.
- **Therefore:** on one core there is no fill left for declared structure to remove on either topology. What remains is **parallelism and memory** — Phase 5, promoted — with evaluation the larger half. Phases 3–4 are **not justified** by anything measured; do not start them without a topology that shows fill the ordering misses.
- Phase 2 keeps one open question per topology: for gas, whether a better network-derived design exists (this one is not it); for the arrowhead, nothing — the ordering is already right.

## After Phase 2

Per topology, proceed if the structure-derived ordering materially improves at least one of fill, memory, factorization time or its exponent. On the chain, this tests the fill row directly.

## After Phase 3

Proceed with the Schur path for globals on the arrowhead if it beats the generic solve at realistic global counts. Record the \(n_s\) at which dense Schur stops winning (§48).

## After Phase 4

Per topology, proceed to broader generalization if the block solver shows a substantial improvement in wall time, memory or exponent that Phase 2 did not already deliver. On the chain, this is where the pivoting row is tested.

## After Phase 5

Proceed if parallel speedup across blocks is material on the arrowhead benchmark at a realistic block count, **measured against FERAL's own threading** (1.2× at 4 threads on the 210k case, degrading past that) and against the `.nl` evaluator for 5a. This is now the gate that decides whether the capability ships at all: if block parallelism does not beat those baselines, declared structure has no measured win left on either topology.

## Standing requirements at every gate

- **§33 exactness** for the linear solve: matching step and matching inertia.
- **NLP equivalence (§34):** assert status, objective and KKT residuals. Report the change in iteration count as a measurement, not a pass/fail condition. Elimination order changes pivoting and inertia-based regularization, and so changes the trajectory.
- **Fixture sweep:** any change that routes the default path (e.g. `structured_kkt="auto"`) is a trajectory change and needs `scripts/sweep-fixtures.sh` on both legs (exact and L-BFGS) before merge.

---

# 59. Key Benchmark Question

The most important experiment is:

> For a fixed local horizon size, how does solve time grow as the number of horizons increases?

Compare empirical scaling:

\[
T_{\mathrm{generic}}(H)
\]

and

\[
T_{\mathrm{structured}}(H).
\]

If generic behaves roughly like

\[
H^2
\]

and structured approaches something closer to

\[
H
\]

or

\[
H^{1+\epsilon},
\]

that would be a major result.

Even a substantial reduction in constant factor or memory could be valuable.

---

# 60. Longer-Term Research Possibilities

If the architecture works, several interesting directions become possible.

## Automatic decomposition discovery

Use graph partitioning to identify solver-useful separators automatically.

## Adaptive decomposition

Merge or split blocks based on numerical conditioning.

## Hybrid direct/iterative Schur

Direct local factorizations + Krylov reduced solve.

## GPU block solves

Independent repeated blocks may be suitable for batched GPU linear algebra.

## Distributed solves

Large scenario sets could potentially distribute local blocks across nodes.

## Learned decomposition policy

Use structural features to predict which KKT strategy will be fastest.

These should remain secondary to a robust exact chain implementation.

---

# 61. Main Risks

The major technical risks are:

1. **Pivoting across block boundaries**
2. **Loss of robust inertia information**
3. **Dense reduced systems**
4. **Unexpected hidden coupling**
5. **Separator size too large**
6. **Generic sparse ordering already near-optimal**
7. **Implementation complexity outweighs practical gain**

These risks are precisely why the staged benchmarking strategy is important.

---

# 62. Minimal First Prototype

The smallest useful prototype is probably:

```text
declared block membership (NLP indices)
  from the benchmark generator (discopt via I1 later)
        |
        v
POUNCE
  set_block_structure -> KKT permutation
        |
        v
existing generic FERAL factorization
```

No new factorizer.

No new Schur algorithm.

No parallelism.

Benchmark that first.

Then:

```text
same block metadata
        |
        v
existing POUNCE Schur-block path   (global variables / arrowhead only)
```

Only after those results are known, and only for a topology where they leave a measured gap (§58), should a block solver be implemented.

---

# 63. Example Desired User Experience

The final goal should look simple:

```python
from discopt import Model
from discopt.decomposition import horizon

m = Model()

for t in range(T):
    with horizon(m, t):
        build_gas_network(m, data[t])

link_temporal_states(m, linepack, storage)
mark_shared(m, investments)

result = m.solve(
    solver="pounce",
    structured_kkt="auto",
)
```

Output:

```text
POUNCE structured KKT
---------------------
detected topology: chain
blocks: 72
median local KKT size: 8,430
median separator size: 46
global variables: 31

selected strategy: ordering   (path taken: ordering, no fallback)

KKT solve:
  local factorization:   1.84 s
  reduced assembly:      0.07 s
  reduced factorization: 0.03 s
  backsolve:             0.21 s
  total:                 2.15 s
```

The user should not need to understand the linear algebra to benefit from it.

---

# 63.5 Phase 5b: `BlockAugSystemSolver` — **implemented**

*Status 2026-09-19: built and measured through the CLI on the arrowhead family
— `pounce_feral::FeralBlockSolver`, `kkt::BlockAugSystemSolver`,
`IpoptApplication::set_kkt_block_structure`, `Problem.set_kkt_block_structure`,
and the `kkt_block_detect` option. On `case118_ieee`: factorization 3.6–4.0×
faster at K = 32…128, identical iteration counts and objectives, ~1.3× end to
end (factorization is ~35% of these solves). Detection is a degree heuristic
and fails where the shared columns do not stand out (measured on
`case1354_pegase`); the Phase 1 mapper in §63.6, which turns a model's own
block labels into KKT indices, is what covers that case and is also built.*

Concrete because the prototype measured it (`benchmarks/kkt_scaling/blockfac`,
`dev-notes/kkt-scaling-phase0b.md`): on a 210k-variable N-1 SCOPF, factor +
border + back-solve is **6.1× faster** than the best monolithic path, with
inertia identical and residual no worse. This is what shipping that looks like.

## Where it plugs in

A third arm beside `StdAugSystemSolver` and `SchurAugSystemSolver`, built by
`AlgorithmBuilder::build_with_backend` when a block structure is installed
(§12's `Problem::set_block_structure`), wrapping `StdAugSystemSolver` for
assembly and fallback exactly as the Schur arm does. It owns:

* `blocks: Vec<Vec<Index>>` and `border: Vec<Index>`, in KKT space, from the
  Phase 1 mapper;
* one `feral::SymbolicFactorization` per block, built once per pattern with
  `symbolic_factorize_with_schur(block ∪ border, schur_indices = border)`;
* per-iteration: `factorize_multifrontal_with_schur` per block (rayon), which
  returns that block's factors **and** its dense Schur contribution without
  forming any `n_block × n_border` buffer — the step that decides whether the
  gain survives (0.012 s against 0.37 s for dense multi-RHS solves);
* the summed border complement `S = A_bb − Σ_k A_bk A_kk⁻¹ A_kb`, factored on
  its own.

## Contract with the IPM

* **Inertia** by Haynsworth: `inertia(K) = Σ_k inertia(A_kk) + inertia(S)`,
  verified exact in the prototype. This is what `perturb_for_wrong_inertia`
  consumes, so the existing inertia-correction ladder works unchanged.
* **Regularization.** `δ_w` / `δ_c` are diagonal, so a retry adds them to the
  block diagonals and the border and refactors: the same cost again, no new
  symbolic analysis.
* **Back-solve** by block elimination: `y_k = A_kk⁻¹ b_k`, `S Δ_s = b_s − Σ_k
  A_sk y_k`, `x_k = A_kk⁻¹ (b_k − A_ks Δ_s)` — two block solves and one border
  solve, measured 3.3× the monolithic back-solve and at least as accurate.
* **Fallback**, first-class as in `SchurAugSystemSolver`: a malformed
  partition, a block-to-block coupling the structure did not declare, a
  singular block, or any backend error routes the rest of the solve through
  `inner`. `linear_solver.blocks` present in the report is the signal it
  actually ran, mirroring `linear_solver.schur`.
* **Validation** before the first factor: every off-diagonal KKT entry must be
  block-local or block-to-border. This is the structural check §46 describes;
  it is cheap (one pass over the assembled triplet) and it is what makes a
  wrong declaration fall back instead of silently costing fill.

## Parallelism policy

Blocks are the coarse tasks feral's elimination tree cannot see (measured: 1.2×
from tree parallelism against 10× from block tasks). So: rayon across blocks,
feral's own parallelism **off** inside each block, and the whole thing measured
against feral's threading rather than against one thread. Thread count follows
pounce's existing policy; a serial fallback (one block at a time) must stay
correct because it is also the debugging path.

## Three things the implementation had to get right

Each was found by a measurement, not by inspection:

* **Unscale the Schur block.** feral factors `D·A·D`, so the complement comes
  back scaled by the border's own factors. Inertia is a congruence invariant,
  so every inertia assertion passes either way — only a solve notices.
* **Surface `Singular` rather than falling back on it.** Unlike the Schur arm,
  every dual row here lives inside a block, so `perturb_for_singular`'s δ_c
  reaches it; a KKT with structurally empty constraint rows reads as singular
  on the monolithic path too.
* **Fold tiny components into a block, never into the border.** Isolated rows
  (empty constraints keeping only δ_c) each form their own component; putting
  them in the border makes its dense complement cost more than the
  factorization it replaces.

## What is not designed yet

* **Restoration** builds a different KKT (extra `p`/`n` columns) and has no
  structure. It falls back until Phase 6.
* **L-BFGS** routes through the low-rank wrapper, which owns the (2,2) block;
  the block arm is exact-Hessian only, like the Schur arm.
* **Nested blocks** (a block graph deeper than one border) — out of scope; this
  is the arrowhead, and the chain is refuted for gas (§38.3).

## Acceptance

§33 exactness against `StdAugSystemSolver` on the same iterate (step and
inertia), §34's NLP equivalence with iteration count *reported* rather than
required, the fallback exercised by a deliberately wrong structure, and a
scaling run on the Phase 0c arrowhead family. Opt-in — it engages only when a
structure is installed — so no default path moves and the fixture sweep is not
required, though the arrowhead family should be swept before and after.

---

# 63.6 Phase 1: the block-structure mapper — **implemented**

*Status 2026-09-19: `map_block_structure_to_kkt` +
`IpoptApplication::set_block_structure`, `Problem.set_block_structure` in
Python, `block_structure_file` for the `.nl` path. Measured on
`case1354_pegase`, the family detection could not read: 17/33/65 blocks over a
259-column border at K = 16/32/64, factorization 4.3× / 4.8× / 3.8× faster,
identical iteration counts and objectives. The section below is the design it
was built to; what it did not anticipate is that a fixed variable moves the
border (53 declared shared columns become a border of 18 on `case118_ieee`) and
that restoration's factorizations were 92% of the remaining factor time on the
infeasible K = 64 run. Restoration is covered too as of the commit after:
`AugRestoSystemSolver` reduces onto the original 4-block system, so the same
labels describe it, and the only thing missing was a way to hand them to a
builder minted before the mapping existed (`kkt_blocks_shared`,
`kkt_block_restoration`). That took the K = 64 run's restoration factor time
from 35.0 s to 11.4 s and its wall from 113 s to 83 s. Numbers in
`dev-notes/kkt-scaling-phase0b.md`.*

The block solver works; what it lacked was a usable way to be *told* the
structure. Today it takes KKT-space labels (`x | slack | eq-dual | ineq-dual`,
in the solver's internal order), which a modeller cannot be expected to
produce, or it detects them, which works only when the shared columns stand out
by degree.

The mapper is the missing piece, and it is small:

* **Input** (§45): a block id per *model* variable and per *model* constraint,
  `< 0` for shared — exactly what an indexed model already knows (`pg[g]` is
  shared; everything in contingency `k` is block `k`).
* **Mapping**: `x` takes the variable's label; a slack and its inequality dual
  take the constraint's; an equality dual likewise. The pitfall is presolve: a
  dropped row or fixed column shifts every later index, so the mapper must work
  from the *post-presolve* model the KKT is assembled from, and refuse (fall
  back) on a length mismatch rather than mislabel.
* **Surfaces**: `Problem.set_block_structure(var_blocks, con_blocks)` in Python,
  and for the `.nl` path either a labels file or — better — the block index
  already carried by the `.row`/`.col` symbolic names, which is how
  `gen_scopf.py`'s instances identify their contingencies.

With it, `case1354_pegase` (where detection fails) becomes the second
end-to-end measurement, and Phase 5a's evaluation work has the same structure
object to key on. Both surfaces shipped, plus `block_structure_file` so the
`.nl` path has one without a modelling layer; the `.row`/`.col` symbolic-name
route was not built, because a labels file is the same information and does not
tie pounce to one generator's naming convention.

**Restoration is done** — it needed no new mapping at all, only the cell that
gets the labels to a builder minted before the mapping existed. What is left on
this line is Phase 5a's block-parallel evaluation, the larger half of every one
of these solves, which pounce does not own (discopt draft I5, unfiled).

# 64. Bottom Line

The proposed feature is not primarily a new decomposition algorithm at the optimization level.

It is a mechanism for carrying structural knowledge from `discopt` into POUNCE so that POUNCE can solve its KKT systems in a way that respects the model's block structure.

The clean division is:

```text
discopt
    identifies and preserves structure

POUNCE
    chooses and coordinates structured KKT elimination

FERAL / pounce-linsol
    performs robust sparse factorization and solves
```

The general target is a block-graph solver on POUNCE's `AugSystemSolver` extension point, fed by solver-neutral structure from `discopt`, and judged per topology on the full set of benefits in the Purpose section.

For the multi-horizon gas-network problem, its first instance, the target is:

> exact elimination of horizon-local KKT variables that leaves a block-banded reduced system in the temporal linking variables, and that brings factorization cost to linear in the number of horizons.

The immediate development sequence should be:

1. instrument FERAL stats, then attribute the gas \(T^2\) to a row of §38.2 (Phase 0),
2. build `Problem.set_block_structure`, the index mapper, validation and diagnostics in POUNCE (general), with the `discopt` export (I1) filed when an end-to-end test needs it — **done** (§63.6),
3. benchmark structure-derived ordering, block-level nested dissection on the chain, on the gas model **and** an arrowhead problem,
4. benchmark the existing Schur path on global variables only,
5. implement the block solver for whichever topology the measurements leave a gap on,
6. extend structure into restoration and quasi-Newton Hessians.

If successful, the capability should generalize naturally to:

- multi-period process optimization,
- gas-network planning,
- security-constrained OPF,
- multi-period OPF,
- stochastic programming,
- scenario optimization,
- dynamic optimization,
- batch and spatially decomposed NLPs.

That would make structured large-scale NLP solving a genuine architectural capability of the `discopt` + POUNCE ecosystem rather than a one-off optimization for a single application.
