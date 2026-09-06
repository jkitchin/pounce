# Rust API

POUNCE is written in Rust, so the Rust API is the solver itself rather than a
binding over it. Everything is reached through one crate — **`pounce-rs`**, a
facade that re-exports a curated public surface so your code does not depend
on POUNCE's internal crate layout.

```sh
cargo add pounce-rs
```

```toml
[dependencies]
pounce-rs = "0.9"
```

The default build is the NLP path. The convex/conic, active-set QP, and
sensitivity solvers are behind [feature flags](#feature-flags).

> **Why one crate.** The solver is split across ~21 workspace crates
> (`pounce-nlp`, `pounce-algorithm`, `pounce-convex`, …) whose boundaries move
> as the code evolves. Depending on them directly couples you to that layout.
> `pounce-rs` is the stability boundary; everything below is internal.

## Two APIs

### The builder — for the common case

Implement [`Problem`] — only `objective` is required — then configure and
solve. Anything you leave out is supplied: missing gradients and Jacobians are
approximated by finite differences, and the Hessian defaults to a
limited-memory (L-BFGS) approximation.

```rust
use pounce_rs::prelude::*;

// min (x0−1)² + (x1−2)²  s.t.  x0 + x1 == 3,  0 ≤ x ≤ 5
struct P;
impl Problem for P {
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }
    fn n_constraints(&self) -> usize { 1 }
    fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
}

let sol = Nlp::new(P)
    .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
    .constraint_bounds(&[3.0], &[3.0])   // equality: lower == upper
    .x0(&[0.0, 0.0])
    .option_num("tol", 1e-10)
    .solve();

assert!(sol.success);
assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
```

`n` is inferred from `var_bounds` or `x0` (they must agree). Options use the
same names as the CLI and upstream Ipopt — `option_num`, `option_int`,
`option_str`; see [Solver Options](options.md).

Option names, value types, ranges, and choices are validated against the
option registry, and a rejected option is never applied silently — a
misspelled name would otherwise leave the default in effect while the solve
looked like it had honoured the request. `solve` panics on a rejected option;
`try_solve` returns the same conditions as `Err(NlpError)` instead:

```rust
use pounce_rs::builder::{Nlp, NlpError};

let err = Nlp::new(P).x0(&[0.0, 0.0])
    .option_str("mu_stratgey", "adaptive")   // typo
    .try_solve();
assert!(matches!(err, Err(NlpError::InvalidOption { .. })));
```

`try_solve` reports setup failures only. A solve that runs and does not
converge is `Ok` with `success == false` and the reason in `status`.

The returned `Solution` carries `success` / `status`, `x`, `objective`,
`multipliers`, the constraint values `g`, the bound multipliers `z_l` / `z_u`,
and `stats` (wall time, iteration count, evaluation counts, final
infeasibilities). With `presolve=yes` and `presolve_fbbt=yes`, implement
`Problem::constraint_expression` with `FbbtTape` values to receive
`Solution::fbbt_report`. The vector fields are filled by `finalize_solution`,
so they stay **empty** if a solve aborts before finalizing — check `success`
before indexing.

Setting `presolve_fbbt=yes` without `presolve=yes` is a no-op.

Each tape must exactly restate the corresponding value from `constraints()`.
`try_solve` checks the starting point and box midpoint, but that sampling is not
a proof — and the comparison at those two points allows a relative mismatch of
~1.5e-8. An undetected mismatch can cut off the optimum without a diagnostic.
Every slot must additionally influence the tape's root value (gh #877).
Generate both representations from one source when possible.

To supply exact derivatives, implement `gradient` and `jacobian` and return
`true`; returning `false` (the default) selects finite differences for that
callback.

### `TNLP` — for full control

For an exact Hessian, custom Jacobian/Hessian *sparsity*, or NLP scaling,
implement the [`TNLP`] trait directly and drive it with `IpoptApplication`.
This is the same trait the CLI and the C ABI sit on.

```rust
use pounce_rs::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

let mut app = IpoptApplication::new();
app.initialize()?;
let prob = Rc::new(RefCell::new(MyTnlp::default()));
let status = app.optimize_tnlp(Rc::clone(&prob) as Rc<RefCell<dyn TNLP>>);
assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
```

You provide `get_nlp_info` (sizes and nonzero counts), `get_bounds_info`,
`get_starting_point`, the evaluators (`eval_f`, `eval_grad_f`, `eval_g`,
`eval_jac_g`, `eval_h`), and `finalize_solution` to receive the answer.
`eval_jac_g` and `eval_h` are called in two modes — `SparsityRequest::Structure`
for the pattern, then `SparsityRequest::Values` — so the pattern is declared
once and reused across iterations.

The [crate documentation on docs.rs](https://docs.rs/pounce-rs) has a complete
HS071 walkthrough.

## Iteration capture and logging

Opt into the per-iteration trajectory with `.capture_iterations()` on the
builder; the records land in `sol.stats.iterations`. Outside the builder,
`with_iter_capture` wraps any closure and returns the records alongside its
result:

```rust
use pounce_rs::prelude::*;

let (sol, iters) = with_iter_capture(|| {
    Nlp::new(P)
        .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
        .constraint_bounds(&[3.0], &[3.0])
        .solve()
});
assert!(sol.success && !iters.is_empty());
```

On the `IpoptApplication` path, install `collector_scope()` for the duration of
the solve and read the history back from `statistics()`. `init_subscriber()`
turns on console logging without your crate taking a `tracing` dependency.

## Feature flags

Everything beyond the NLP path is off by default and lands in its own module.
The two QP families both name their types `QpProblem` / `QpSolution` /
`QpStatus`, so they cannot share one flat namespace.

| feature | module | covers |
|---|---|---|
| `convex` | `pounce_rs::convex` | LP, convex QP, SOCP / exponential / power / PSD cones, SOS; batched and warm-started solves; symbolic-factorization reuse; QP sensitivity |
| `qp` | `pounce_rs::qp`, `pounce_rs::sqp` | sparse **parametric active-set** QP, and the SQP working-set warm-start contract |
| `sensitivity` | `pounce_rs::sensitivity` | sIPOPT-style `∂x*/∂p` predictors, parametric warm starts, reduced Hessian |
| `full` | — | all three |

```toml
[dependencies]
pounce-rs = { version = "0.9", features = ["convex", "sensitivity"] }
```

Enabling a feature widens what the crate **exports**, not what it builds: the
default NLP path already compiles `pounce-qp`, `pounce-linsol`, and
`pounce-feral` transitively, so `qp` costs nothing at build time and only
`convex` and `sensitivity` add crates.

`convex` and `qp` also enable **`pounce_rs::linsol`**, which supplies the
sparse symmetric factorization those solvers take as an argument —
`backend()` for the default parallel FERAL factor, `serial_backend()` for the
inner-serial one used under an outer-parallel batch.

`pounce_rs::presolve` and `pounce_rs::restoration` are modules too, but they
are **not** feature-gated — the default NLP path already compiles both crates.
See [Presolve and restoration](#presolve-and-restoration).

### Convex: LP, QP, and conic

```rust
use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_rs::linsol::backend;

// min ‖x‖² − 0.5·x0 − 1.5·x1  s.t.  x0 + x1 == 1,  0 ≤ x ≤ 5
let prob = QpProblem {
    n: 2,
    p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
    c: vec![-0.5, -1.5],
    a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
    b: vec![1.0],
    g: vec![],
    h: vec![],
    lb: vec![0.0, 0.0],
    ub: vec![5.0, 5.0],
};

let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
assert_eq!(sol.status, QpStatus::Optimal);
```

Both convex engines accept a solve-wide wall-clock budget through
`QpOptions::time_limit: Option<std::time::Duration>`. `None` (the default)
preserves the uncapped behavior. A limit creates one monotonic deadline for
the entire top-level solve: equilibration/HSDE retries, active-set homotopy and
phase-1, feasibility probes, seeded retries, and LP crossover do not receive a
fresh budget. Batch entries are separate top-level solves and each receives
the full duration. Expiration returns `QpStatus::TimeLimit` with the latest
finite iterate; a linear-system factorization already in flight is not
interruptible, so the solve may overshoot by one factorization.

These are public API additions. Downstream exhaustive `QpOptions { ... }`
literals must initialize `time_limit` (or use `..QpOptions::default()`), and
exhaustive matches on `QpStatus` must handle `TimeLimit`.

`P` is the **lower triangle** of the Hessian in triplet form; an empty `P` is
an LP. Cone blocks beyond the nonnegative orthant are declared with `ConeSpec`
and solved by `solve_socp_ipm`. For many instances, `solve_qp_batch_parallel`
runs one per rayon worker, and `QpFactorization` reuses the AMD ordering and
symbolic analysis across instances that share a sparsity pattern. See
[Convex Solver](convex-solver.md).

A *family* of convex QPs — one parameter moving, the structure fixed — is what
`ActiveSetSession` is for. It is a persistent handle over the **active-set**
driver (`solver_selection=qp-active-set`'s engine) that owns the convex →
`pounce-qp` translation and the presolve/postsolve wrapper, keeps the previous
solve, and traces a parametric homotopy to the next problem instead of solving
it cold. Reuse is a cost claim only: a warm answer passes through the same
verification a cold one does, and anything that does not stand up falls back to
the full cold driver.

```rust
use pounce_rs::convex::{ActiveSetSession, QpProblem, QpStatus, Reuse, Triplet};
use pounce_rs::linsol::backend;

let qp = |t: f64| QpProblem {
    n: 2,
    p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
    c: vec![-2.0 * t, -2.0 * t],
    a: vec![],
    b: vec![],
    g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
    h: vec![1.0],
    lb: vec![0.0, 0.0],
    ub: vec![5.0, 5.0],
};

let mut session = ActiveSetSession::new(backend);
for t in [0.2, 0.3, 0.4, 0.9] {
    let sol = session.solve(&qp(t));
    assert_eq!(sol.status, QpStatus::Optimal);
}
assert_eq!(session.last_reuse(), Reuse::Homotopy);
```

`last_reuse` names the route the **engine** took, which is not the same as
whether reuse was attempted. `solve_parametric` declines the homotopy when the
Hessian changes or a row's equality/fixed status changes, and answers from the
previous working set instead — still warm, but not the traced path
(`Reuse::WorkingSet`); if the previous solve is not a usable base it solves
cold internally (`Reuse::EngineCold`). `Reuse::is_warm()` is the coarse
question, and `stats()` breaks the counts out the same way
(`homotopy_accepted`, `working_set_accepted`, `engine_cold_accepted`, plus
`warm_accepted()`). A family whose reuse count is high but whose
`homotopy_accepted` is zero is not being traced — usually because something in
`P` moves between members.

`solve_cold` forces a cold solve and `reset` drops the reuse state.
`with_presolve(false)` turns the reduction off when the reported iterate has to
be in the coordinates of the problem exactly as posed. `cargo run -p
pounce-convex --example active_set_session` is the measurement: on an 8-step
path at `n = 40`, ~104-111 ms of wall clock cold against ~19-24 ms through a
session, with all 7 reusing steps tracing the path.

If you are driving the engine yourself rather than through a session — a
frontend doing something the session does not cover — the recipe is four steps,
and the first and last are the ones that are easy to skip and wrong to:

1. `screen_variable_box`. `BoxScreen::Empty` is a certified `PrimalInfeasible`
   with no solve behind it; `Snapped` hands back a repaired copy to solve
   instead. Skip it and a reversed box reaches the engine as an
   `InvertedBounds` error, while a *present* `+∞` lower bound is dropped as if
   absent and the solve returns `Optimal` at a point that violates it.
2. `ActiveSetQp::from_convex` on whatever step 1 handed back. For an
   indefinite Hessian, add `.with_hessian_inertia(HessianInertia::Indefinite)`.
3. `engine_options` for the settings this path was measured under, then solve
   `ActiveSetQp::problem` with `pounce_qp`. It takes the *same* inertia value
   as step 2 — one of the settings turns on it — so pass whatever you passed
   there, and `HessianInertia::Psd` for a convex QP.
4. `back_translate_verified`. It applies the dual sign transform, recomputes
   the objective in convex coordinates and re-derives the verdict — the
   engine's `Optimal` is a claim, not a verdict (see `verify_status`).
   `back_translate` and `verify_status` are exported separately for callers
   that need to do something between them.

What the recipe does *not* reproduce is the cold driver's retry ladder (Ruiz
equilibration, the simplex-seeded retry, the objective-free feasibility probe).
Those are `solve_qp_active_set`'s and `ActiveSetSession`'s job; if you want
them, call one of those instead.

For a **nonconvex** QP with the whole ladder, that call is
`solve_qp_active_set_inertia(prob, opts, engine, HessianInertia::Indefinite,
backend)` — `solve_qp_active_set` is the same driver under a standing PSD
claim. The convex IPM has no such entry point: without a PSD Hessian its
optimality test accepts a saddle point and reports it as `Optimal`. What comes
back for an indefinite `P` is a *local* solution, and the constraints must be
linear — the curvature this engine controls is the objective's.
`ActiveSetSession` stays convex-only: its homotopy is a predictor built for
that case, so a nonconvex sequence goes through the free function one solve at
a time.

### Active-set QP and SQP warm starts

`pounce_rs::qp` is the parametric active-set engine — a different solver
family from the convex IPM, for sequences of nearby QPs, and it accepts an
indefinite Hessian. `pounce_rs::sqp` is the NLP-level counterpart: carrying a
working set from one SQP solve into the next. See
[Active-Set SQP & Warm Starts](active-set-sqp.md).

### Sensitivity

```rust
use pounce_rs::prelude::*;
use pounce_rs::sensitivity::SensSolve;

let result = SensSolve::new(vec![2, 3])      // pinned constraint rows
    .with_deltas(vec![-0.5, 0.0])            // Δp
    .with_reduced_hessian()
    .run(&mut app, tnlp);

let dx = result.dx.expect("populated when with_deltas was set");
```

A sensitivity-stage failure is reported through `result.error`, **not**
`result.status` — the underlying solve can converge while the post-solve step
fails. See [Sensitivity Analysis](sensitivity.md) and
[Sessions](sessions.md).

## Presolve and restoration

Two modules that are **not** behind a feature — the NLP path already compiles
both crates — and that most callers never need, because the ordinary paths run
them already. Reach for these when you want the *reports*, or when you drive
`IpoptApplication` directly.

### `pounce_rs::presolve`

`optimize_tnlp` applies presolve itself when `presolve=yes`, so setting the
option is the whole of the common case. This module is for reading what
preprocessing found, which means holding the concrete wrapper — the accessors
hang off `PresolveTnlp`, not off `dyn TNLP`:

```rust
use pounce_rs::presolve::{LicqVerdict, PresolveOptions, PresolveTnlp, wrap_with_presolve};

let wrapped = wrap_with_presolve(inner, PresolveOptions::defaults())?;

let mut app = IpoptApplication::new();
app.initialize()?;
app.set_presolve_already_applied(true);   // or the problem is presolved twice
let status = app.optimize_tnlp(wrapped);
```

| accessor | returns |
|---|---|
| `licq_verdict()` | `Option<&LicqVerdict>` — structural rank of the equality rows |
| `tighten_report()` | `TightenReport` — how many bounds Phase 1 moved |
| `cached_bounds()` | `Option<&CachedBounds>` — the box after tightening |
| `auxiliary_diagnostics()` | `AuxiliaryPreprocessingDiagnostics` — Phase 0 blocks, timings, rejections |
| `fbbt_report()` | `Option<FbbtReport>` — nonlinear bound propagation |
| `certified_infeasible()` | `Option<InfeasibilityProof>` — infeasibility proved before iteration 1 |

Every one of those types is exported here too, so a report can be bound,
matched and stored rather than only `{:?}`-printed.

Two things to get right:

- **`wrap_with_presolve`, not `PresolveTnlp::new`.** The linear-equality
  elimination (`presolve_linear_eq_reduction`) is a *separate* wrapper stacked
  outside `PresolveTnlp`, so `PresolveTnlp::new` on its own reads the option
  and removes no column. `wrap_with_presolve` stacks both;
  `wrap_from_options` does the same from an `OptionsList`, which is what
  `app.options()` hands you.
- **`set_presolve_already_applied(true)` after wrapping by hand**, since
  `optimize_tnlp` would otherwise wrap again.

### `pounce_rs::restoration`

`run_second_opinion_ladder` re-solves a failing verdict along up to four
deliberately different trajectories — different scaling, a different barrier
strategy, a displaced start — and promotes one only if it converges. A
converged solve pays nothing: the ladder reads the status and returns.

The builder runs it already, and reports what it did on
`Solution::second_opinion`; so do the CLI and the Python and C frontends. The
`IpoptApplication` path does not, so call it yourself there:

```rust
use pounce_rs::restoration::run_second_opinion_ladder;

let status = app.optimize_tnlp(Rc::clone(&tnlp));
let outcome = run_second_opinion_ladder(
    &mut app, tnlp, status, app.statistics(), &mut |_line| {},
);
```

`outcome.statistics` is the **shipped** solve's alone, so the true cost of a
promotion is `outcome.total_iteration_count()`, and `outcome.base_status` is
the only remaining trace that the base solver did not converge. Each rung
writes solver options and the driver restores them, so an application that
solves twice does not inherit a rung's settings.

Not for multi-start drivers: a failed start is routine there, and four extra
solves per failure buy nothing.

## Escape hatch

Each feature module also re-exports the crate behind it — `pounce_rs::convex`
re-exports `pounce_convex`, `pounce_rs::presolve` re-exports
`pounce_presolve`, and so on — so anything outside the curated surface stays
reachable without adding a dependency. Reaching for it is a signal the facade
is missing something; those are worth
[filing](https://github.com/jkitchin/pounce/issues).

## See also

- [docs.rs/pounce-rs](https://docs.rs/pounce-rs) — the full API reference
- [Choosing a Solver](choosing-a-solver.md) — which solver fits which problem
- [Solver Options](options.md) — the option names shared by every frontend
- [Python API](python.md) — the same solvers from Python
