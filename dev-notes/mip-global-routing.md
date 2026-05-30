# Mixed-integer & global optimization — design note

**Status: design only.** No code changes yet. This note captures the
architecture for adding mixed-integer (MILP / MIQP / convex-MINLP) and
deterministic global (nonconvex NLP / MINLP) optimization on top of the
LP/QP routing already designed in
[`lp-qp-routing.md`](./lp-qp-routing.md). It assumes that note's LP/QP
plan has landed: a `pounce-convex` IPM family, the `pounce-qp`
active-set solver, and the `solver_selection` dispatch seam.

Two scope decisions drive everything below and depart from the LP/QP
note's stated boundaries:

1. **Pure Rust, no wrapping.** The LP/QP note's escape hatch — "wrap
   HiGHS / SCIP / OSQP / BARON behind the dispatch layer" — is *off the
   table*. Every algorithm here is in-house. This re-promotes simplex
   from "out of scope, use HiGHS" to a *required dependency* (see
   "Simplex is back" below).
2. **Nonconvex global optimization is in scope.** The LP/QP note
   explicitly excludes it ("Use BARON / Gurobi-nonconvex",
   lp-qp-routing.md "Nonconvex QP / global optimization"). This note
   reverses that: deterministic global optimization via spatial
   branch-and-bound becomes a first-class POUNCE algorithm family.

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

## Implementation phasing

The ordering is forced: convex MI (Phases 0–3) is the substrate the
global layer (Phases 4–6) stands on. Each phase is independently
shippable.

**Phase 0 — Integer plumbing.** Parse integer-var counts from the `.nl`
header; extend `ProblemClass`; carry `is_integer: BitVec` on the
problem. Dispatch errors cleanly on integrality / nonconvexity it
cannot yet handle. Mirrors LP/QP Phase 1. *No new algorithm.*

**Phase 1 — B&B shell + first convex MI.** The unified tree with
integer branching only. Drive it with **MIQP over `pounce-qp`** first
(native warm-starts, weakest open-source competition). Most-fractional
branching, depth-first, incumbent + gap. Minimum that justifies
`pounce-mip`.

**Phase 2 — Pure-Rust simplex (`pounce-lu` + `pounce-simplex`).** Dual
simplex with bound-flipping ratio test and LU-with-updates. Native
node-to-node warm-starts — the reason simplex re-enters scope. Unlocks
MILP and fast LP relaxations for the later spatial path.

**Phase 3 — Cuts + node presolve.** Gomory / MIR / cover cuts; probing;
coefficient tightening reusing `pounce-presolve`. Pseudocost /
reliability branching. Closes the gap to real solvers for convex MI.
Mostly combinatorial engineering, independent of the linear algebra.

**Phase 4 — `pounce-relax`.** Factorable decomposition, McCormick
envelopes, αBB, OBBT. The multi-quarter research lift; the heart of the
global capability. Validates by reproducing known global optima on
small MINLPLib instances.

**Phase 5 — Spatial branch-and-bound.** Continuous branching on the
relaxation-gap variable; relaxation lower bounds from Phase 4;
multi-start local NLP upper bounds; convergence by global-gap closure.
This is a genuine deterministic global solver.

**Phase 6 — MINLP-global unified.** Both branching modes active at once
— the BARON / Couenne capability, end to end, pure Rust.

### Cost summary (rough, single engineer)

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Integer plumbing | 2–4 weeks | 1 month |
| 1 — B&B shell + MIQP | 2–4 months | 3–5 months |
| 2 — Simplex + LU | 4–8 months | 7–13 months |
| 3 — Cuts + node presolve | 3–6 months | 10–19 months |
| 4 — Relaxation engine | 6–12 months | 16–31 months |
| 5 — Spatial B&B | 4–8 months | 20–39 months |
| 6 — MINLP-global | 3–6 months | 23–45 months |

Phases 0–3 are the convex-MI deliverable — incremental and shippable on
top of LP/QP. Phases 4–6 are a flagship, multi-year, BARON-class effort.

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

For convex MI, HiGHS / SCIP are the open bar to clear; Gurobi / COPT
lead by a wide margin. For nonconvex global, BARON and the commercial
solvers lead, and **a pure-Rust deterministic global optimizer
essentially does not exist today** — so the bar is "correct and
credible on MINLPLib," with competitiveness against Couenne / SCIP as
the realistic medium-term target.

## What does not change

- `TNLP` stays algorithm-agnostic and object-safe; the B&B engine
  drives node solves through it, exactly as the LP/QP solvers do.
- The `.sol` writer is problem-type-agnostic; no change.
- The filter line search stays the default NLP globalization; the local
  IPM-NLP path is unchanged and remains the default for nonconvex NLP
  absent explicit global opt-in.
- `pyomo-pounce` gets MIP / global routing transparently via CLI
  dispatch; integer variables already round-trip through the NL format.

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
