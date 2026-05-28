# LP/QP solver routing — design note

**Status: design only.** No code changes yet. This note captures the
architecture for adding specialized LP and convex-QP solvers alongside
the existing IPM-NLP pipeline, so the work can resume cleanly when LP/QP
development starts.

POUNCE today routes every problem — linear, convex quadratic, or general
nonlinear — through the same Wächter-Biegler filter-IPM. This is
correct (LP ⊂ convex QP ⊂ NLP) but leaves performance on the table:

- IPM-QP with Mehrotra predictor-corrector closes 30–50% of iteration
  count vs IPM-NLP on convex QPs.
- Simplex / active-set LP solvers beat IPM-LP on small LPs and
  warm-started sequences (MPC, branch-and-bound subproblems).

## Decisions

1. **Single `pounce` binary with `--solver` flag.** Default behavior:
   auto-detect from the `.nl` header + a linearity walk. Explicit
   override via `pounce --solver=lp foo.nl` or `solver_selection=lp`
   in `ipopt.opt`. Mirrors Gurobi/CPLEX UX; preserves a single Pyomo
   `SolverFactory('pounce')` entry.
2. **One `pounce-convex` crate** for the IPM-based convex algorithms
   (IPM-LP, IPM-QP, and the conic extensions). Resists workspace sprawl;
   related algorithms share warm-start logic, presolve adapters, and the
   predictor-corrector machinery.
3. **Active-set QP stays in its own `pounce-qp` crate.** A sparse
   Schur-complement parametric active-set QP solver (qpOASES lineage —
   Kirches 2011; Janka/Kirches/Sager/Schlöder 2016) is already in
   flight on the `claude/active-set-sqp-warm-start-BnjLA` branch
   (`crates/pounce-qp/`, ~59 commits across Phases 5a–d). It is
   complementary to the IPM-QP proposed here, not duplicative — the two
   algorithms have different sweet spots (see "Active-set vs IPM-QP"
   below) — so it keeps its own crate and ships as a separate dispatch
   target.

## Architecture

### Routing layer

New module `crates/pounce-cli/src/dispatch.rs` sits between problem
loading (`nl_reader::read_nl_file` at
`crates/pounce-cli/src/nl_reader.rs:570`) and the call to
`app.optimize_tnlp()` (currently `crates/pounce-cli/src/main.rs:412`).

It does three things:

1. **Classifies the problem.** Extends `nl_reader::parse_header` to
   capture `n_nl_cons`, `n_nl_objs`, and the `n_nl_vars_*` triplet
   currently skipped at `nl_reader.rs:591`. Walks the parsed `Expr`
   AST (`nl_reader.rs:45-65`) to confirm linearity and detect
   quadratic objectives and constraints. Produces:
   ```rust
   enum ProblemClass { Lp, ConvexQp, ConvexQcqp, NonconvexQp, Nlp }
   ```
2. **Resolves the solver choice** by combining `ProblemClass` with the
   `solver_selection` option:
   - `auto` (default): most specialized solver matching the class
   - `nlp`: always IPM-NLP (current behavior)
   - `lp-ipm`, `qp-ipm`, `qp-active-set`: force; error if the problem
     doesn't fit (e.g., `qp-ipm` on a problem with a non-quadratic
     objective).
3. **Dispatches.** Each solver implements (or is wrapped behind) the
   existing `TNLP` trait (`crates/pounce-nlp/src/tnlp.rs:157`); the
   trait is already algorithm-agnostic and object-safe, so dispatch is
   a `match` over the resolved choice that calls a thin per-solver
   entry point in either `pounce-convex` or `pounce-qp`.

### Crate layout

```
crates/
  pounce-algorithm/    # existing — IPM-NLP, unchanged
  pounce-convex/       # NEW — IPM-LP, IPM-QP, conic (SOCP/exp/pow/SDP)
  pounce-qp/           # existing (on active-set-sqp-warm-start branch)
                       #   — sparse Schur-complement parametric active-set QP
  pounce-nlp/          # existing — TNLP trait, unchanged
  pounce-linsol/       # existing — sparse LDLᵀ contract, unchanged
  pounce-feral/        # existing — pure-Rust LDLᵀ backend, unchanged
  pounce-hsl/          # existing — MA57 backend, unchanged
  pounce-presolve/     # existing — extended with LP-specific reductions
```

`pounce-convex` exposes per-algorithm entry points for the IPM family:
```rust
pub fn solve_lp_ipm(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
pub fn solve_qp_ipm(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
// SOCP / exp / pow / SDP reuse solve_qp_ipm's cone-generic scaffolding
// (see src/cones/), selected by the cone types present — not a new fn.
```

`pounce-qp` already exposes its own active-set entry point; dispatch
calls into it for `qp-active-set`:
```rust
// in pounce-qp (existing on the branch)
pub fn solve_qp_active_set(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
```

All IPM solvers reuse `pounce-linsol` for the augmented-system
factorization (`SparseSymLinearSolverInterface` — same trait feral and
MA57 implement today). Mehrotra predictor-corrector and Gondzio
higher-order correctors live inside `pounce-convex` because the same
iteration scaffolding serves both IPM-LP and IPM-QP (and the conic
extensions). `pounce-qp` keeps its own Schur-complement KKT
machinery — different from the IPM augmented system — so it does not
share the IPM scaffolding.

Unlike the NLP path, the convex entry points exploit the constant-matrix
structure: for an LP/QP the Hessian `P` and constraint matrix `A` (and
`c`, `b`) do *not* depend on `x`, so they are extracted **once** at
setup via a single `eval_h` / `eval_jac_g` call and cached for the rest
of the solve. The `TNLP` contract is built for nonlinear problems and
suggests per-iteration re-evaluation; the convex solver must *not* be a
thin per-iteration `TNLP` driver like the NLP path, or it forfeits the
specialization that justifies it (and the Phase 2 "specialized path
wins" benchmark claim).

### Active-set vs IPM-QP: why both

| Property                        | IPM-QP (`pounce-convex`)        | Active-set (`pounce-qp`)              |
|---------------------------------|----------------------------------|---------------------------------------|
| Iteration cost                  | one big sparse LDLᵀ per step    | one Schur update per active-set step  |
| Iteration count                 | ~10–30 (predictor-corrector)    | grows with active-set churn           |
| Cold start (large/dense QP)     | strong                          | weaker                                |
| Warm start (parametric, MPC,    | weak (must reseed barrier)      | excellent (homotopy through QP        |
| B&B subproblems)                |                                 | sequences, qpOASES-style)             |
| Returns exact active set        | only after crossover            | yes, natively                         |
| Best for                        | one-shot convex QPs, LPs        | QP sequences, SQP inner solver,       |
|                                 |                                 | MPC, MIP node QPs                     |

Dispatch picks between them via `solver_selection`. Under `auto`,
convex LP/QP always goes to IPM-LP/IPM-QP — **the active-set path is
opt-in**, never auto-selected from the NL path. The reason: an `.nl`
file describes a single instance, and neither the format nor
`solver_selection` carries a "this is one of a parametric sequence,
warm-start it" signal for the classifier to act on. So `pounce-qp` is
reached only (a) explicitly via `solver_selection = qp-active-set`, or
(b) programmatically via the Python/C warm-start API, where the caller
holds state across solves and *is* the warm-start signal. A future
extension could let a caller mark a problem as warm-startable through a
`solver.options` hint, at which point `auto` could route it to
`pounce-qp`; until that hint exists, auto-routing to active-set is not
possible and is not claimed.

### What modeling languages see

- **AMPL / Pyomo (via NL files):** No change to the user-facing solver
  name. `solve with pounce;` or `SolverFactory('pounce')` continues to
  work for any problem. The CLI auto-detects and routes; users who
  want to force a solver pass options through Pyomo's
  `solver.options['solver_selection'] = 'lp'`. Everything still flows
  through NL files for the AMPL path.
- **Python API (`pounce-py`):** Add `pounce.solve_lp(problem, ...)`,
  `pounce.solve_qp(problem, ...)` alongside the existing
  `pounce.solve(problem, ...)` (NLP). Programmatic users know the
  problem type already — explicit entry points are more ergonomic
  than auto-detection. The existing `Problem.solve()` keeps NLP
  semantics for backward compatibility.
- **C ABI (`pounce-cinterface`):** Add `IpoptSolveLp()`,
  `IpoptSolveQp()` alongside `IpoptSolve()`. Same callback-driven
  `TNLP` bridge.

### NL-header inspection

The NL format header (Gay 2005 §3) lines currently skipped at
`crates/pounce-cli/src/nl_reader.rs:591` contain exactly the fields
needed:

- Line 2: `n_vars n_cons n_objs ranges eqns` (already parsed)
- Line 4: `n_nl_cons n_nl_objs` — count of constraints/objectives with
  a *nonlinear part*. Zero means purely linear; see the LP/QP caveat
  below.
- Line 5: `n_nl_net n_lin_net` — network structure (future routing
  target)
- Line 6: `n_nl_vars_in_both n_nl_vars_in_cons n_nl_vars_in_obj`

The NL format has no dedicated quadratic section: each row's linear
part lives in the `G`/`J` (gradient/Jacobian) coefficient segments,
while *any* higher-order term — including the quadratic terms of a QP —
is written into the nonlinear expression tree (`O`/`C` segments) as
`Mul`/`Pow` nodes. Consequently a QP objective registers as nonlinear,
so the header alone does **not** distinguish LP from QP:

- `n_nl_cons == 0` and `n_nl_objs == 0` → class is **LP** (all
  structure is in the linear `G`/`J` segments; no AST walk needed).
- Otherwise walk the nonlinear AST of every row (objective *and*
  constraints) that carries a nonlinear part. If any nonlinear term is
  not a degree-2 polynomial (transcendental, higher-degree `Pow`, etc.)
  → **NLP**. If all nonlinear terms are degree-2 polynomials, extract
  the Hessians and split on convexity (PSD test via numerical
  factorization / attempted Cholesky — *not* the Hessian *pattern* from
  `pounce-nlp`):
  - quadratic objective, **linear** constraints, objective Hessian PSD
    → **ConvexQp** (→ IPM-QP);
  - quadratic objective and/or **quadratic** constraints, all convex
    (objective Hessian PSD and each ≤-inequality's constraint Hessian
    PSD) → **ConvexQcqp** (→ SOCP / conic solver, Phase 4+). A convex
    QCQP is SOCP-representable via the epigraph / rotated-second-order-
    cone reformulation, so it routes to the same conic IPM as native
    SOCP rather than to the dense NLP path;
  - any indefinite Hessian (objective or a constraint) → **NonconvexQp**
    (falls through to NLP-IPM for a local min).
- **Conservative fallback (correctness guard).** Whenever the walk
  cannot *prove* the stronger class — parse failure, an inconclusive /
  near-singular PSD test, or a quadratic constraint whose sense is
  incompatible with its curvature — fall back to the more general class,
  ultimately **NLP**. Misclassifying an indefinite or non-quadratic
  problem *into* a convex solver would return a spurious KKT point as if
  globally optimal; falling back to NLP is always sound. The PSD test
  therefore uses a tolerance, and "inconclusive within tolerance" routes
  to NLP, never to the convex path.
- Until Phase 4 (SOCP) lands, **ConvexQcqp** falls through to NLP-IPM;
  the distinct class is the dispatch seam the conic solver later
  intercepts (same pattern as `NonconvexQp`).

This mirrors how QP-capable AMPL solvers detect QPs (ASL's `nqpcheck`
walks the nonlinear tree to recover `Q`); the header is a fast reject
for the LP case only.

### Option plumbing

Single new option on `OptionsList`:

- Key: `solver_selection`
- Values: `auto` (default), `nlp`, `lp-ipm`, `qp-ipm`, `qp-active-set`
- Validation: `auto` always works; explicit values error if the
  loaded problem doesn't match the class (with a message naming the
  detected class).
- Routing: `lp-ipm` / `qp-ipm` resolve into `pounce-convex` entry
  points; `qp-active-set` resolves into the existing `pounce-qp` crate.

Follows the precedent of `linear_solver`, which selects `Ma57`/`Feral`
via the `LinearBackendFactory` at
`crates/pounce-algorithm/src/alg_builder.rs:45-57`.

### What does not change

- `TNLP` trait stays exactly as it is — algorithm-agnostic,
  object-safe (`crates/pounce-nlp/src/tnlp.rs:157-249`).
- `.sol` writer (`crates/pounce-cli/src/nl_writer.rs`) is already
  problem-type-agnostic; takes `(x, lambda, status)`. No change.
- `pounce-restoration`, `pounce-l1penalty`, `pounce-sensitivity` stay
  coupled to IPM-NLP only — the convex solvers don't use them (no
  filter restoration, no penalty reformulation; sensitivity stays
  NLP-coupled for now, though it's the natural seam for differentiable
  convex layers later).
- A barrier parameter μ is *not* optional, though: every IPM has one.
  The convex IPM supplies its own **Mehrotra adaptive σ·μ centering**
  (in `pounce-convex`, Phase 3), which is distinct from the NLP
  `mu_strategy` (Monotone / Adaptive) in `pounce-mu`. Open question for
  Phase 2/3: reuse `pounce-mu`'s strategy abstraction if it fits, or
  keep the convex μ logic local to `pounce-convex`. Either way it is a
  required component, not a skipped one.
- `pyomo-pounce` doesn't change at all; users get LP/QP routing
  transparently via the CLI dispatch.

### Presolve integration

Presolve is a 2–10× factor on the Mittelmann/Maros-Mészáros sets, so
*wall-clock* competitiveness with HiGHS/Clarabel depends on it — Phase 3
delivers an *algorithmically* competitive iteration (low iteration
counts), and Phase 3.5 (presolve) is what turns that into competitive
end-to-end wall-clock. Presolve is *not* optional for that bar, even
though it is not blocking for *correctness*. Two parts: the integration
seam (favorable, mostly inherited) and the reduction work (largely
net-new for LP/QP).

**Integration seam — inherited for free.** `pounce-presolve` is already
a *composable TNLP wrapper* (TNLP-in → reduced-TNLP-out, with a
postsolve path that reinstates dropped rows and forwards multipliers;
see `crates/pounce-presolve/src/lib.rs` Phases 0–5). Because the convex
solvers also consume `TNLP`, `pounce-convex` sits *behind*
`PresolveTnlp` exactly as the IPM does today — no new plumbing. This is
the part that is genuinely "not blocking."

**IPM-aware reduction policy — the seam differs from a simplex
presolve.** Gondzio (1997) shows an IPM cares about Cholesky/LDLᵀ
*fill-in*, not a basis: reductions that help simplex (aggressive
variable substitution) can *hurt* an IPM by densifying the factor.
Since `pounce-convex` factors through `pounce-linsol` LDLᵀ, substitution
must be gated on fill growth (Mészáros & Suhl 2003 bound model-size
increase before each elimination). This is a *policy*, not just a
reduction set.

**Reduction catalog to implement.** Grounded in the literature review
(citations below):

- *Core LP reductions (Andersen & Andersen 1995):* empty / singleton /
  forcing / dominated rows; singleton / duplicate columns; bound
  tightening. Most already exist in `pounce-presolve` for the NLP path
  and carry over.
- *Modern strengthening (Achterberg et al. 2020):* coefficient
  strengthening, dual reductions, parallel/dominated row–column
  detection. The modern bar; add incrementally.
- *QP/Hessian-consistent reductions (Gould & Toint 2004) — net-new:*
  variable substitution and duplicate-column detection must account for
  the Hessian `Q` (elimination fills `Q` with cross-terms), and the
  **postsolve must recover the dual consistently with the quadratic
  term**. The existing NLP-shaped presolve has no notion of a `P`
  block, so this is the genuinely new work for the convex-QP path.

**Postsolve / restoration stack — the missing architectural piece.**
Every reduction must carry its undo and recover *primal and dual* for
the original problem (Andersen & Andersen 1995; PaPILO's
transaction/reduction-stack design). The current crate does this for
its NLP reductions; LP/QP variable substitution and bound shifts need
their own dual-recovery transforms.

**Equilibration front-end.** Ruiz (2001) row–column norm balancing
(optionally + Pock–Chambolle), as used by OSQP/Clarabel, conditions the
KKT system before the IPM solve. Adjacent to presolve proper; bundle it
with the dispatch into `pounce-convex`.

**Build in pure Rust; learn from PaPILO, don't wrap it.** POUNCE's
default build is pure Rust by design (no Fortran/C/C++, no system BLAS —
see README and `docs/src/introduction.md`), so wrapping PaPILO
(header-only C++) is out: it would break the pure-Rust guarantee that
`pounce-feral` exists to uphold. PaPILO (Gleixner, Gottwald & Hoen
2023; INFORMS JOC; arXiv:2206.10709) is still the best *reference
architecture* — its **transaction-based reduction stack** (each
reduction is a transaction with an undo, conflict-checked so reductions
can be applied in parallel) is exactly the postsolve design
`pounce-presolve` needs, and it is Apache-2.0 so studying the source is
unencumbered. The plan is therefore to extend `pounce-presolve`
in-house, porting PaPILO's *ideas* (transaction model, the LP/QP
reduction set) rather than its code. Parallelism uses **rayon** (the
idiomatic Rust data-parallel crate; not yet a workspace dependency) for
the same recursive/data-parallel routines PaPILO parallelizes with
Intel TBB — probing, dominated-column detection, constraint
sparsification — keeping the transaction model as the conflict-avoidance
mechanism.

**Key references**

- E. D. Andersen & K. D. Andersen, *Presolving in linear programming*,
  Math. Prog. 71:221–245 (1995). — reduction catalog + restoration.
- J. Gondzio, *Presolve analysis of linear programs prior to applying
  an interior point method*, INFORMS JOC 9(1):73–91 (1997); Addendum
  13(2):169 (2001). — IPM-specific (fill-in) presolve.
- C. Mészáros & U. Suhl, *Advanced preprocessing techniques for linear
  and quadratic programming*, OR Spectrum 25:575–595 (2003). —
  fill-/row-growth control during elimination.
- N. Gould & P. Toint, *Preprocessing for quadratic programming*,
  Math. Prog. Ser. B 100:95–132 (2004). — QP/Hessian-aware reductions
  and dual recovery.
- T. Achterberg, R. Bixby, Z. Gu, E. Rothberg & D. Weninger, *Presolve
  Reductions in Mixed Integer Programming*, INFORMS JOC 32(2):473–506
  (2020). — modern taxonomy (Gurobi).
- A. Gleixner, L. Gottwald & A. Hoen, *PaPILO: A Parallel Presolving
  Library for Integer and Linear Optimization with Multiprecision
  Support*, INFORMS JOC (2023); arXiv:2206.10709. — Apache-2.0
  reference implementation (LP/MIP/QP).
- D. Ruiz, *A scaling algorithm to equilibrate both rows and columns
  norms in matrices*, RAL-TR-2001-034 (2001). — equilibration.

## Implementation phasing

Each phase is independently shippable. The headline shift from the
original plan is that `pounce-convex` is *the* in-house home for the
entire IPM/conic family — LP, QP, SOCP, SDP, exponential cone, power
cone — built incrementally on a single Mehrotra + HSDE scaffolding
sharing `pounce-linsol`. Active-set QP stays in `pounce-qp` on its own
track. Other algorithm families (ADMM, AL+semismooth Newton,
banded/Riccati IPM, simplex) are explicitly *out of scope* — see the
"Out of scope and why" section below.

**Phase 1 — Dispatch scaffolding.** Header parsing, classifier,
`solver_selection` option, dispatcher that supports only `auto` and
`nlp` (auto → nlp for now). Ship to verify no regression. *No new
algorithm.*

**Phase 2 — IPM-QP in `pounce-convex` (+ Ruiz equilibration).** Bare
IPM-QP (no Mehrotra yet); route LP and QP problems to it under `auto`.
**Build the iteration over the `Cone` abstraction (`src/cones/`) from
the start, with only `nonneg` implemented** — this is what makes
Phases 4–6 cone *extensions* rather than a rewrite; a QP-specific solve
retrofitted for cones later would make the Phase 4 "cheap incremental
win" claim false. Bring in **Ruiz equilibration** here — it is a
conditioning prerequisite for the IPM KKT solve, effectively part of the
solver rather than deferrable presolve (see "Presolve integration").
Compare
iteration counts and wall-clock against the existing IPM-NLP path on the
`quadratic`, `bounded-quadratic`, `eq-quadratic` builtins. This is the
minimum that justifies the `pounce-convex` crate.

**Phase 3 — Mehrotra predictor-corrector + HSDE.** Add the
predictor-corrector iteration and homogeneous self-dual embedding for
infeasibility detection and a self-starting iterate. Note this must be
the **quadratic-objective HSDE variant** (as in Clarabel; Goulart &
Chen) that carries the `P` term inside the embedding — *not* the
textbook LP/conic HSDE, which assumes a linear objective. Handling the
quadratic objective in the embedding is the QP-specific part and does
not transfer from LP-HSDE verbatim. Should reduce iteration counts
~30-50% on convex QPs. Validate on Mittelmann LP
subset and Maros-Mészáros QP set. After this phase `pounce-convex` is
algorithmically competitive with Clarabel and HiGHS for the LP/QP
problem class. This is *algorithmic* competitiveness (iteration count
and convergence); *wall-clock* competitiveness on the full benchmark
sets additionally needs presolve (Phase 3.5).

**Phase 3.5 — Presolve (reduction catalog + postsolve stack).** Now
that the iteration is algorithmically competitive, presolve is the
multiplier that closes the benchmark gap to HiGHS/Clarabel (a 2–10×
factor on the standard sets). Land the LP/QP reduction catalog, the
IPM-aware reduction policy, and the pure-Rust transaction-based
postsolve stack (PaPILO ideas, rayon for parallelism — not a wrap), per
the "Presolve integration" section. Sequenced *after* Phase 3 on
purpose: debugging the postsolve dual-recovery against a solver you
already trust avoids chasing two unknowns at once. Benchmark-driven —
add the reductions that actually move the Mittelmann / Maros-Mészáros
numbers. Equilibration (Phase 2) is the prerequisite already in place;
this phase adds the size-reducing transformations on top.

**Phase 4 — SOCP via second-order cone.** Add the second-order cone as
a constraint type. Nesterov-Todd scaling on the SOC block; rotated-SOC
as a derived form. Validate on Mittelmann SOCP set. This is a cheap
incremental win once Mehrotra is in place — the symmetric-cone IPM
machinery extends from LP/QP unchanged.

**Phase 5 — Exponential and power cones (non-symmetric).** Add the
three-dimensional exponential cone, three-dimensional power cone, and
generalized power cone via the dual-scaling Mehrotra implementation
from Chen & Goulart 2023 (arXiv:2305.12275). The augmented system stays
sparse quasi-definite, so `pounce-linsol` (feral/MA57) covers it
unchanged. This unlocks geometric programming, logistic regression at
high accuracy, entropy optimization, KL divergence, and $p$-norm
constraints. Validate on the GP / entropy / KL benchmark sets used by
Clarabel and MOSEK.

**Phase 6 — SDP via PSD cone + chordal decomposition (optional).** Add
the PSD cone with triangular storage, the Nesterov-Todd scaling for
PSD (the cone is symmetric), and the clique-graph merging strategy of
Garstka, Cannon & Goulart 2020 (arXiv:1911.05615) as a presolve step.
This is the biggest lift in the conic family — the chordal
decomposition is more engineering than the IPM iteration itself —
and is gated on user demand. Skip until SDP is justified by a real
application; for one-off SDP needs in the meantime, wrap MOSEK or
Clarabel as a backend behind the dispatch layer.

The active-set QP track in `crates/pounce-qp/` (currently 59 commits
on `claude/active-set-sqp-warm-start-BnjLA`) runs *in parallel* and is
not phase-ordered against `pounce-convex`. It targets a different sweet
spot — parametric warm-start (MPC, SQP inner solver, B&B node QPs) —
and ships when its own phases 5a–d are complete.

### Cost summary (rough, single engineer)

| Phase | Effort | Cumulative |
|------|--------|-----------|
| 1 — Dispatch | 2–4 weeks | 1 month |
| 2 — Bare IPM-QP (+ equilibration) | 3–6 months | 4–7 months |
| 3 — Mehrotra + HSDE | 2–3 months | 6–10 months |
| 3.5 — Presolve | 2–4 months | 8–14 months |
| 4 — SOCP | 1–2 months | 9–16 months |
| 5 — Exp/power cones | 2–4 months | 11–20 months |
| 6 — SDP + chordal | 6+ months | 17+ months (optional) |

Phases 1–3 are the minimum to justify the dispatch architecture and
deliver a *correct* LP/QP solver; Phase 3.5 (presolve) is what makes it
*benchmark-competitive* with HiGHS/Clarabel — required for that bar,
though not for correctness. Phases 4–5 are the natural extension that
closes most of the convex-conic-IPM gap to Clarabel. Phase 6 is gated
on demand.

## Out of scope and why

The "QP solver families" analysis (see
[`/Users/jkitchin/projects/pounce/.crucible/wiki/concepts/qp-solver-families.org`](../.crucible/wiki/concepts/qp-solver-families.org))
identifies five production-algorithm families for QP and several more
for the conic generalizations. POUNCE deliberately *does not* plan to
build the following families in-house:

### ADMM / operator-splitting (OSQP-class)

The first-order operator-splitting family (OSQP, SCS, ProxQP-ADMM) is
the right answer for embedded MPC at moderate accuracy (1e-3 to 1e-5)
and for very large structured QPs. The algorithm is well-documented and
implementable, but the "leverage feral + consistency with NLP-IPM"
argument is shallow here: ADMM factors a *different* matrix (typically
$P + \sigma I + A^\top R A$) and uses it factor-once / solve-many
without inertia checks. Feral can host the factor but the algorithm
scaffolding (step-size adaptation, restarts, scaling, polishing) has no
overlap with POUNCE's filter-IPM core.

*Escape hatch:* wrap OSQP as a dispatch backend (`solver_selection =
qp-osqp`) if a user needs it. OSQP is MIT-licensed and mature.

### Augmented Lagrangian + semismooth Newton (QPALM-class)

The robust-on-degeneracy family (QPALM, ProxQP-AL). Newer, less
standardized, and overlaps significantly with what filter-IPM
regularization already does in `pounce-algorithm`. The use case
(ill-conditioned QPs with active-set degeneracy) is niche enough that
POUNCE's IPM-QP path is likely good enough; the marginal value of a
separate QPALM-class solver is low.

*Escape hatch:* none planned. If the use case becomes a priority,
revisit; otherwise skip.

### Banded / Riccati IPM for MPC (HPIPM-class)

The MPC-specialist family (HPIPM, FORCES Pro, acados-internal). Uses
the IPM scaffolding but with banded LDLᵀ via Riccati recursion instead
of generic sparse factor. The block-tridiagonal MPC structure gives an
order-of-magnitude factor speedup that generic sparse linalg cannot
match.

The "feral consistency" argument cuts the *wrong* way here: feral is a
general sparse symmetric-indefinite backend, and the right linsol for
banded MPC is a different code path (a banded LDLᵀ implementation) that
shares the `SparseSymLinearSolverInterface` trait but not feral's
factorization. Building it would be a separate `pounce-mpc` crate with
a banded `pounce-linsol` backend; only worth doing if optimal control
is a deliberate POUNCE target. Currently it is not.

*Escape hatch:* wrap HPIPM if MPC users emerge; or use the active-set
`pounce-qp` warm-start path, which is the alternative MPC-friendly
solver and is in flight.

### Simplex (LP)

Was Phase 4 in the original plan; removed. Simplex is the right answer
for small LPs and warm-started LP sequences (B&B node relaxations,
sensitivity analysis on degenerate LPs). It needs LU-with-updates,
which is a substantial engineering effort separate from the
LDLᵀ-based IPM/conic scaffolding.

*Escape hatch:* IPM-LP from Phases 2/3 plus presolve (Phase 3.5) covers
the medium-to-large LP case and benchmarks competitively with HiGHS-IPM
on the Mittelmann sets. For small LPs and warm-start LP sequences, defer
simplex until a specific application forces it; alternative is to wrap
HiGHS as a backend.

### Nonconvex QP / global optimization

Inherently combinatorial (spatial branch-and-bound + convex
relaxation). Out of scope *for now* — neither the NLP-IPM nor the
convex-IPM finds global optima today, and the B&B shell is substantial
new engineering. But it is deliberately left *reachable*: the
lower-bounding subproblem at each B&B node is itself a convex
relaxation (Shor/SDP, RLT/LP, or convex-QP), which is precisely the
conic family this note already plans to build. So the per-node solver
is free; only the B&B shell is new.

Architectural choices that keep global QP in scope for later, without
redesign:

1. **`NonconvexQp` stays a first-class `ProblemClass`**, never folded
   into `Nlp`. It falls through to NLP-IPM (local min) today, but the
   distinct class is the dispatch seam a future `qp-global` target
   intercepts.
2. **Reserve option space** — a future `solver_selection = qp-global`
   value, or (cleaner) an orthogonal `require_global` flag, so the
   dispatch `match` grows by one arm rather than being reworked.
3. **Branching-rule-agnostic B&B shell.** The future `pounce-mip` B&B
   shell (see "Mixed-integer" in the outlook) should parameterize the
   branching rule and relaxation builder so that *spatial* branching
   (continuous vars, for global QP) and *integer* branching (MIP) share
   one tree / incumbent / pruning / node-queue core.
4. **Preserve the classifier's Hessian factorization.** The PSD test in
   the classifier already computes the eigenstructure of `P`; a global
   solver reuses it for the DC split (`P = P⁺ − P⁻`) and relaxation
   construction. Expose it rather than recomputing.
5. **Factor-reuse / warm-start across nodes** (outlook items 1–2) is
   what makes any B&B tractable — the same argument as MIP.

*Escape hatch (until then):* use BARON / Gurobi-nonconvex for problems
with indefinite Hessians where local minima are insufficient.

### Decision principle

The criterion that puts a family *in-house* in `pounce-convex` versus
*out-of-scope* is the strength of two consistency wins:

1. **Sparse symmetric-indefinite augmented system.** If the per-iteration
   linear system has the same shape as POUNCE's existing NLP-IPM, feral
   is reused as-is and the regularization / inertia machinery in
   `pounce-linsol` is shared. The conic-IPM family (LP/QP/SOCP/SDP/exp/pow)
   passes this test cleanly; ADMM, AL, and banded-IPM do not.
2. **`TNLP` + `Solver` + `OptionsList` integration.** Algorithms that
   accept a `TNLP` problem and slot into the existing dispatch /
   session / Python / C scaffolding get unified ergonomics for free.
   IPM-class algorithms inherit this naturally; first-order and
   operator-splitting methods need the same scaffolding but with
   different convergence-criterion plumbing.

When both wins are strong (conic IPM family), in-house is right. When
both are weak (ADMM, AL), wrap or defer. When only one is strong
(banded IPM), it's a judgment call gated on application demand.

## Files to modify or add

### Modify
- `crates/pounce-cli/src/nl_reader.rs:570-594` — extend `parse_header`
  to capture the additional header fields
- `crates/pounce-cli/src/main.rs:~179, ~412` — call into the new
  dispatcher between problem loading and `optimize_tnlp`
- `crates/pounce-algorithm/src/options.rs` (or equivalent) — register
  `solver_selection`
- `Cargo.toml` (workspace) — add `pounce-convex` as a member
- `crates/pounce-presolve/` — LP/QP reductions, IPM-aware reduction
  policy, and a pure-Rust transaction-based postsolve stack (PaPILO
  ideas, rayon for parallelism — not a wrap); see the "Presolve
  integration" section for the scoped catalog and references. Not
  blocking for correctness, but required for the Phase 3 benchmark bar.

### Add
- `crates/pounce-cli/src/dispatch.rs` — `classify_problem(&NlProblem)
  -> ProblemClass` plus the `match`-based router
- `crates/pounce-convex/` — new crate scaffolded with `solve_lp_ipm`
  and `solve_qp_ipm` entry points; `src/ipm.rs` (the shared Mehrotra +
  HSDE scaffolding) plus `src/cones/` (per-cone barrier, gradient,
  Hessian, scaling-update — one module per cone: `nonneg.rs`, `soc.rs`,
  `psd.rs`, `exp.rs`, `pow.rs`, `gpow.rs`). The first implementation
  target is `cones/nonneg.rs` (covers LP) plus the IPM scaffolding; QP
  comes for free via the explicit $P$ block in the augmented system.
  Subsequent cones land incrementally per the phasing above.
- (no new crate for active-set QP — `crates/pounce-qp/` already exists
  on the `claude/active-set-sqp-warm-start-BnjLA` branch and is the
  dispatch target for `qp-active-set`)
- (no new crate for ADMM / AL / banded-IPM / simplex — see "Out of
  scope and why" above)

## Verification

Phase 1 (routing scaffolding, no behavior change):

- `cargo test -p pounce-cli` covers new dispatcher with unit tests on
  `classify_problem`: feed it parsed `NlProblem` structs for known
  LP / convex QP / convex QCQP / nonconvex QP / NLP cases (builtins +
  Mittelmann fixtures already on disk), plus boundary cases that must
  fall back to NLP (inconclusive PSD test, parse failure), and assert
  the right `ProblemClass`.
- `make benchmark-mittelmann` produces identical results to current
  behavior — `auto` routes everything to NLP-IPM until `pounce-convex`
  lands.
- Integration test: `pounce --solver=lp builtin:rosenbrock` should
  error with "problem class NLP does not match forced solver LP".

Phase 2 (LP/QP actually dispatched):

- Comparison harness: run each Mittelmann LP through both
  `--solver=nlp` and `--solver=lp-ipm`, assert objective values match
  to 1e-6, log iteration counts and wall-clock to confirm the
  specialized path wins.
- `studio/mcp` MCP tools can render `compare_runs` between the two
  paths for any individual benchmark — `compare_runs` was built for
  exactly this kind of side-by-side analysis.

Phase 3 (Mehrotra + HSDE):

- Iteration-count regression: assert the predictor-corrector cuts
  iterations vs the bare Phase-2 IPM on the same Mittelmann LP /
  Maros-Mészáros QP instances — the ~30–50% claim is a checked
  regression, not an aspiration.
- Infeasibility / unboundedness: feed known-infeasible and
  known-unbounded LP/QP fixtures and assert HSDE reports the correct
  status instead of stalling or hitting the iteration cap.

Phase 3.5 (presolve) — the highest correctness risk is postsolve dual
recovery, so it gets the most coverage:

- Round-trip primal *and* dual: for each Mittelmann / Maros-Mészáros
  instance, solve with presolve on and off and assert the recovered
  `x` *and* the duals (`λ`, bound multipliers) match to 1e-6 after
  postsolve. Primal-only matching hides the most common postsolve bug.
- Per-reduction unit tests: each reduction (singleton / doubleton /
  forcing / dominated row; singleton / duplicate column; bound
  tightening) gets a fixture where postsolve must reconstruct the
  eliminated primal *and* dual entries exactly.
- Detection: presolve-only infeasibility / unboundedness fixtures
  (e.g. contradictory singleton bounds) assert the correct status
  without invoking the IPM at all.
- QP-specific: a fixture where a variable substitution fills the
  Hessian, asserting `P` is transformed consistently and the dual is
  recovered with the quadratic term (the net-new Gould–Toint path).

Phases 4–6 (conic):

- Objective-value cross-check against Clarabel / MOSEK on the matching
  cone benchmark set (SOCP / GP-entropy / SDP) to 1e-6.
- Regression guard: adding a cone must not change LP/QP results — re-run
  the Phase-2/3 suite and assert stable iteration counts on the pure
  LP/QP instances. Convex-QCQP fixtures route to the SOCP path and are
  cross-checked against the NLP-IPM local solution (same optimum, since
  the QCQP is convex).

Python / C APIs:

- `pyomo-pounce` smoke test in CI passes unchanged (proves no
  regression for the modeling-language user).
- New Python-side test in `python/tests/` that constructs a known LP
  and calls `pounce.solve_lp(...)`, asserting it succeeds and that
  `--solver=nlp` would also succeed on the same input.

## Benchmark suites

Standard external test sets to validate against once specialized solvers
land. Listed roughly in the order POUNCE should adopt them.

### LP

- **Mittelmann LP benchmark** — the de-facto modern standard. Curated by
  Hans Mittelmann (ASU); mix of Netlib, Mészáros, Kennington, and large
  industrial LPs up to millions of variables. He publishes regular
  head-to-head runs of Gurobi / CPLEX / COPT / HiGHS. Subset already on
  disk under `benchmarks/`; full set is the credible bar for an LP
  solver paper. <http://plato.asu.edu/bench.html>
- **Netlib LP** — the historical standard (~100 problems, mostly small
  by modern standards). Useful as a smoke test; mostly absorbed into
  Mittelmann.
- **MIPLIB LP relaxations** — root-node relaxations of MIPLIB instances.
  Harder than pure Netlib; common secondary report.

### Convex QP

- **Maros-Mészáros QP test set** — the standard convex QP benchmark
  (~138 problems, tiny to ~100k vars). Every QP solver paper reports on
  this. The credible bar for IPM-QP validation.
- **CUTEst QP subset** — problems with `objtype='Q'` and only linear
  constraints. Already accessible via `benchmarks/cutest/`, so it costs
  nothing to add.
- **Mittelmann convex-QP benchmark** — smaller curated set on the same
  Mittelmann site; head-to-head Gurobi / CPLEX / MOSEK / OSQP.
- **OSQP benchmark set** — Stellato et al.'s ~120 problems (control,
  portfolio, lasso, SVM, Huber). Most useful for the ADMM / first-order
  comparison and the active-set vs IPM-QP split, since the control
  subset favors warm-startable solvers.

### NLP (already in scope today)

- **CUTEst** — the standard NLP benchmark suite (~1500 problems).
  Already wired in `benchmarks/cutest/`.
- **Hock-Schittkowski (HS)** — ~120 small classical NLPs. POUNCE has
  HS071 as an integration test
  (`crates/pounce-algorithm/tests/optimize_hs71.rs`).

### Domain-specific (consider after the core sets)

- **MPC** — no canonical suite, but `acados` / `HPIPM` publish their
  own MPC-shaped benchmarks (varying horizon length, state/input
  dimension). Relevant once `pounce-qp` warm-start is exercised
  end-to-end.
- **Portfolio QP** — typically constructed from real market data;
  not standardized.
- **PDE-constrained / large-scale** — POPS and the Biegler /
  Heinkenschloss optimal-control benchmarks; relevant for the NLP
  path, not for LP/QP routing.

### What "competitive" means in 2025

Reading Mittelmann's site sets expectations:

- **LP**: Gurobi / COPT lead by ~2–3× over HiGHS. **HiGHS is the
  open-source bar to clear.**
- **Convex QP**: MOSEK and Gurobi lead. OSQP is competitive on its
  sweet spot (medium-accuracy, structured problems) — that is the
  realistic target for IPM-QP + the existing active-set `pounce-qp`.

## Outlook: other solver classes that would reuse `pounce-linsol`

`pounce-linsol`'s contract (sparse symmetric indefinite LDLᵀ with
inertia) is the right primitive for a whole family of solvers beyond
LP/QP/NLP. Anything that reduces to a saddle-point or KKT system is a
natural fit. This section is forward-looking — none of these are
planned yet — but it shapes how the LP/QP work should leave the
workspace.

### Conic / barrier-based (closest cousins to IPM-LP/QP/NLP)

- **SOCP** (second-order cone programming). Same augmented-system
  structure as IPM-QP, plus cone barriers on the diagonal. Used for
  robust optimization, portfolio with VaR/CVaR, antenna design,
  anything with `‖Ax+b‖ ≤ cᵀx+d`. Slots into a `pounce-convex`
  extension with no new linsol dependency. Comparable: Mosek / ECOS /
  Clarabel.
- **SDP** (semidefinite programming). Mixed — the KKT Schur complement
  is often *dense*, so SDPA / SDPT3 / Mosek use dense Cholesky for the
  bottleneck. Feral helps for chordal / structured-sparse SDPs (e.g.
  polynomial optimization after chordal decomposition), not general
  dense SDPs.
- **Exponential / power cones.** Same IPM scaffolding as SOCP with
  different barriers. Entropy-regularized OT, geometric programming,
  constrained logistic regression. Clarabel and Mosek 10 support
  these.
- **Homogeneous self-dual embedding / symmetric-cone IPM.** The
  modern conic-IPM formulation (Clarabel, ECOS, SCS). Augmented system
  is symmetric quasi-definite — feral's natural sweet spot.

### Complementarity / variational

- **MCP / NCP** (mixed / nonlinear complementarity). PATH is canonical;
  Newton step on a sparse symmetric system. Game-theory equilibria,
  traffic assignment, electricity market clearing,
  general-equilibrium economics.
- **MPCC / MPEC**. Reformulate complementarity via smooth penalties
  (Scholtes, NCP-function) and solve as NLP — POUNCE already handles
  the smoothed form. A dedicated `pounce-mpcc` crate on top of the
  same linsol is plausible.
- **Bilevel optimization** reduces to MPCC via KKT replacement of the
  inner problem.

### Stochastic / decomposition

- **Stochastic programming with recourse.** Block-angular LPs/QPs (one
  block per scenario + linking constraints). Benders / L-shaped /
  progressive hedging — each subproblem is a sparse symmetric solve,
  and the master is too. Feral handles each block; scenario-parallel
  structure is an architectural layer above.
- **Multi-period / banded KKT.** HPIPM exploits banded structure with
  Riccati recursion because general sparse factor is overkill; the
  general fallback still uses feral-class linsol.

### Differential algebraic / time-stepping

- **Implicit ODE/DAE integrators** (BDF, Radau IIA, IRK). Each step is
  a Newton solve on a Jacobian that, for many physical systems, has
  symmetric saddle-point structure (constrained mechanical systems,
  index-1 chemical-engineering DAEs). Sundials / IDA, Assimulo.
  POUNCE doesn't do simulation, but a DAE integrator sharing the
  linsol backbone is plausible.
- **Trajectory optimization / direct transcription.** Collocation
  produces large sparse symmetric KKT systems — already an NLP for
  POUNCE, but specialized CasADi-style transcription solvers would
  benefit from sharing the linsol.

### Linear algebra / eigenproblems

- **Sparse symmetric eigensolvers via shift-invert.** ARPACK / SLEPc
  shift-invert needs `(A - σI)⁻¹v` repeatedly — exactly the
  factor-once / solve-many pattern feral already supports internally.
- **Sparse linear least squares via augmented system.** Min `‖Ax-b‖²`
  with constraints reformulates as a symmetric indefinite saddle-point
  system; often beats normal equations on conditioning.

### PDE-constrained optimization

- **All-at-once / full-space methods.** Optimize-then-discretize PDE
  problems give huge sparse KKT systems with saddle-point structure.
  POUNCE could plausibly serve as the inner solver for a PDE-opt
  framework; preconditioning becomes the dominant concern at scale.

### Mixed-integer (transitive)

- **MIP / MINLP.** B&B itself isn't a linsol consumer, but every node
  solves an LP or NLP relaxation. A future `pounce-mip` would be a
  B&B shell over `pounce-convex` / `pounce-algorithm`, sharing feral
  through them. Warm-starting across nodes (the factor-reuse
  capability discussed in the batched-solving notes) is what makes
  MIP competitive.

### Architectural implication

Feral is correctly factored as a generic capability. The trait
(`SparseSymLinearSolverInterface`) makes no assumption about *what kind
of optimizer* is calling it — it's the right abstraction layer. The
plausible long-run growth path:

```
crates/
  pounce-linsol/    # contract
  pounce-feral/     # pure-Rust backend
  pounce-hsl/       # MA57 backend
  ┌─ consumers ─────────────────────────────────────┐
  pounce-algorithm/ # IPM-NLP (today)
  pounce-convex/    # IPM-LP/QP + conic (planned)
  pounce-qp/        # active-set QP (in flight)
  pounce-socp/      # SOCP / conic IPM (future)
  pounce-mcp/       # complementarity (future)
  pounce-mip/       # B&B over the above (future)
  └────────────────────────────────────────────────┘
```

Two capabilities that would benefit *every* future consumer if landed
in `pounce-linsol` / `pounce-feral` once:

1. **Factor-once / solve-many as a public API.** ✅ **Landed** —
   `pounce_linsol::Factorization` (`crates/pounce-linsol/src/factorization.rs`)
   exposes the previously-internal `multi_solve(new_matrix: bool, …)`
   semantics as a value handle: `Factorization::new` / `.solve` /
   `.refactor`. Any future LP/QP consumer can hold a factor across
   back-solves without touching the IPM.
2. **Session-style factorization reuse across top-level `solve()`
   calls.** Partially landed: `pounce_sensitivity::Solver`
   (`crates/pounce-sensitivity/src/solver.rs`) keeps the converged KKT
   factor alive across sensitivity / parametric-step / reduced-Hessian
   / raw KKT-back-solve calls, with Python (`pounce.Solver`) and C
   (`IpoptSolver`) surfaces also shipped. Symbolic-factor reuse across
   IPM-level `resolve()` (warm-start MPC / B&B) is the remaining piece
   and is tracked separately; the value-typed `Solver` API is the
   intended seam for plumbing it in.

These two unlock MPC, sensitivity, parametric/warm-start MIP,
differentiable layers, and shift-invert eigensolves all at once — they
are not specific to LP/QP. That's the real argument for investing in
them before the second-solver consumer ships. See
[`docs/src/sessions.md`](../docs/src/sessions.md) for the user-facing
walkthrough.
