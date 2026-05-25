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
   (IPM-LP, IPM-QP, and a future simplex). Resists workspace sprawl;
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
   quadratic objectives. Produces:
   ```rust
   enum ProblemClass { Lp, ConvexQp, NonconvexQp, Nlp }
   ```
2. **Resolves the solver choice** by combining `ProblemClass` with the
   `solver_selection` option:
   - `auto` (default): most specialized solver matching the class
   - `nlp`: always IPM-NLP (current behavior)
   - `lp-ipm`, `lp-simplex`, `qp-ipm`, `qp-active-set`: force; error
     if the problem doesn't fit (e.g., `simplex` on a problem with a
     quadratic objective).
3. **Dispatches.** Each solver implements (or is wrapped behind) the
   existing `TNLP` trait (`crates/pounce-nlp/src/tnlp.rs:157`); the
   trait is already algorithm-agnostic and object-safe, so dispatch is
   a `match` over the resolved choice that calls a thin per-solver
   entry point in either `pounce-convex` or `pounce-qp`.

### Crate layout

```
crates/
  pounce-algorithm/    # existing — IPM-NLP, unchanged
  pounce-convex/       # NEW — IPM-LP, IPM-QP, simplex
  pounce-qp/           # existing (on active-set-sqp-warm-start branch)
                       #   — sparse Schur-complement parametric active-set QP
  pounce-nlp/          # existing — TNLP trait, unchanged
  pounce-linsol/       # existing — sparse LDLᵀ contract, unchanged
  pounce-feral/        # existing — pure-Rust LDLᵀ backend, unchanged
  pounce-hsl/          # existing — MA57 backend, unchanged
  pounce-presolve/     # existing — extended with LP-specific reductions
```

`pounce-convex` exposes per-algorithm entry points for the IPM family
and (eventually) simplex:
```rust
pub fn solve_lp_ipm(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
pub fn solve_qp_ipm(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
pub fn solve_simplex(tnlp: Rc<RefCell<dyn TNLP>>, opts: &OptionsList) -> Status;
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
iteration scaffolding serves both IPM-LP and IPM-QP. Simplex grows its
own LU-with-updates module (eventually a separate `pounce-lu` crate
when justified). `pounce-qp` keeps its own Schur-complement KKT
machinery — different from the IPM augmented system — so it does not
share the IPM scaffolding.

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

Dispatch picks between them via `solver_selection`; `auto` defaults to
IPM-QP for one-shot convex QPs and routes parametric / warm-startable
calls (when that signal is exposed by the caller) to `pounce-qp`.

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
- Line 4: `n_nl_cons n_nl_objs` — if both zero, problem is at-most
  quadratic (could be LP or QP; need AST walk to decide)
- Line 5: `n_nl_net n_lin_net` — network structure (future routing
  target)
- Line 6: `n_nl_vars_in_both n_nl_vars_in_cons n_nl_vars_in_obj`

If `n_nl_cons == 0` and `n_nl_objs == 0` → class is LP or QP.
If furthermore the objective AST contains only linear terms → LP.
If the objective AST has degree-2 `Mul` or `Pow` nodes only → QP
(check positive-semidefiniteness for convex/nonconvex split via the
Hessian-pattern computation already in `pounce-nlp`).

### Option plumbing

Single new option on `OptionsList`:

- Key: `solver_selection`
- Values: `auto` (default), `nlp`, `lp-ipm`, `lp-simplex`, `qp-ipm`,
  `qp-active-set`
- Validation: `auto` always works; explicit values error if the
  loaded problem doesn't match the class (with a message naming the
  detected class).
- Routing: `lp-ipm` / `qp-ipm` / `lp-simplex` resolve into
  `pounce-convex` entry points; `qp-active-set` resolves into the
  existing `pounce-qp` crate.

Follows the precedent of `linear_solver`, which selects `Ma57`/`Feral`
via the `LinearBackendFactory` at
`crates/pounce-algorithm/src/alg_builder.rs:45-57`.

### What does not change

- `TNLP` trait stays exactly as it is — algorithm-agnostic,
  object-safe (`crates/pounce-nlp/src/tnlp.rs:157-249`).
- `.sol` writer (`crates/pounce-cli/src/nl_writer.rs`) is already
  problem-type-agnostic; takes `(x, lambda, status)`. No change.
- `pounce-restoration`, `pounce-l1penalty`, `pounce-sensitivity`,
  `pounce-mu` stay coupled to IPM-NLP only — convex solvers don't
  need most of them.
- `pyomo-pounce` doesn't change at all; users get LP/QP routing
  transparently via the CLI dispatch.

## Implementation phasing

Each phase is independently shippable.

1. **Dispatch plumbing without new algorithms.** Add header parsing,
   classifier, `solver_selection` option, dispatcher that only
   supports `auto` and `nlp` (auto → nlp for now). Ship to verify no
   regression.
2. **IPM-QP in `pounce-convex`.** Bare IPM-QP (no Mehrotra yet); route
   LP and QP problems to it under `auto`. Compare iteration counts and
   wall-clock against the existing IPM-NLP path on the `quadratic`,
   `bounded-quadratic`, `eq-quadratic` builtins.
3. **Mehrotra predictor-corrector** in `pounce-convex`. Should reduce
   iteration counts ~30-50% on convex QPs. Validate on Mittelmann LP
   subset.
4. **Simplex.** Separate effort, separate dev-note. Adds `pounce-lu`
   or equivalent dependency.

Phases 1 and 2 are the minimum that justifies the dispatch
architecture. Phases 3-4 are independent improvements once the
scaffolding is in place.

## Files to modify or add

### Modify
- `crates/pounce-cli/src/nl_reader.rs:570-594` — extend `parse_header`
  to capture the additional header fields
- `crates/pounce-cli/src/main.rs:~179, ~412` — call into the new
  dispatcher between problem loading and `optimize_tnlp`
- `crates/pounce-algorithm/src/options.rs` (or equivalent) — register
  `solver_selection`
- `Cargo.toml` (workspace) — add `pounce-convex` as a member
- `crates/pounce-presolve/` — LP-specific reductions over time
  (singleton rows/cols, dual-bound tightening); not blocking

### Add
- `crates/pounce-cli/src/dispatch.rs` — `classify_problem(&NlProblem)
  -> ProblemClass` plus the `match`-based router
- `crates/pounce-convex/` — new crate scaffolded with `solve_lp_ipm`,
  `solve_qp_ipm`, `solve_simplex` entry points; `src/ipm_qp.rs`
  (covers LP via H=0) is the first implementation target
- (no new crate for active-set QP — `crates/pounce-qp/` already exists
  on the `claude/active-set-sqp-warm-start-BnjLA` branch and is the
  dispatch target for `qp-active-set`)

## Verification

Phase 1 (routing scaffolding, no behavior change):

- `cargo test -p pounce-cli` covers new dispatcher with unit tests on
  `classify_problem`: feed it parsed `NlProblem` structs for known
  LP / convex QP / nonconvex QP / NLP cases (builtins + Mittelmann
  fixtures already on disk) and assert the right `ProblemClass`.
- `make bench-mittelmann` produces identical results to current
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
  pounce-convex/    # IPM-LP/QP, simplex (planned)
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
