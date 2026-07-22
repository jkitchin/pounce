# feral factorization interrupt hook — spec for the #254 residual

`max_wall_time` / `max_cpu_time` are enforced by a shared
`pounce_common::timing::Deadline` that the outer IPM loop, the line search,
the restoration inner IPM, and the KKT solver all consult. The chain of
issues that built this out:

- **#242** — honor the budget at all; check it in the between-iteration
  convergence check. Overshoot bounded to *one outer iteration*.
- **#245** — check it *between* the major KKT factorizations inside
  `PdFullSpaceSolver` (the inertia-correction / iterative-refinement sweep).
  Overshoot bounded to *one factorization*.
- **#246** — check it inside the initialization / restoration entry so a
  bad-start grind returns promptly.
- **#254** (this note) — poll it *inside* a single factorization so one
  factorization on a large NLP cannot itself exceed the whole budget.

#245's own scope note is explicit that it stops at the factorization
boundary:

> It does **not** implement cancellation *inside* a single
> factorization/back-solve — if one individual factorization of a huge
> system runs longer than the budget, that one factorization still
> completes. Aborting mid-factorization would require cancellation hooks in
> the FERAL/MUMPS backend and is out of scope here.

#254 is exactly that residual: on `emfl050_5_5` (5675 vars, 5625 cons) a
`max_wall_time=5` solve returned `TIME_LIMIT` after **48.8 s** — the
between-op check passed while still under budget, one factorization then ran
~44 s uninterrupted, and only the *next* check tripped. feral's own issue #8
records the mechanism: a delayed-pivot cascade turns "an otherwise
sub-second problem" into an 87 s factorization.

## What pounce did (the proactive guard) and why it is only half

`PdFullSpaceSolver` now measures each factorization's wall/CPU cost and, via
`predict_factor_overshoot` / `factor_overshoot_predicted`, refuses to *start*
a new factorization once a factorization has been observed to cost a large
fraction of the budget and the remaining budget can no longer cover one of
that size. This bounds the overshoot **proactively** — the doomed final
factorization never begins — for the "large factor, several-factor budget"
regime (e.g. discopt's ~10 s per-node budgets over multi-second
factorizations), where #245 would previously run one more factorization to
completion before the reactive check caught it.

It cannot close the case #254 actually reports: a **single** factorization
already larger than the entire budget — the first one, before any estimate
exists. There is nothing to predict from, and the factorization cannot be
stopped once it is running. Closing that needs cooperative cancellation
*inside* feral's numeric factorization.

## Why it can't be done in the pounce repo today

The default backend is `feral` (`crates/pounce-feral`), an external crate
pinned at **0.14.0** (the latest published version). Its numeric entry point
is

```rust
impl feral::Solver {
    pub fn factor(&mut self, matrix: &CscMatrix, check_inertia: Option<Inertia>) -> FactorStatus;
}
```

Neither `factor` nor `NumericParams` exposes any interrupt / deadline /
progress-callback / cancel-flag surface (verified against the 0.14.0 source:
`src/numeric/solver.rs`, `src/numeric/factorize.rs`). feral *does* have a
`DelayBudgetExceeded` path, but that is a symbolic-time **column-count**
budget on the delayed-pivot cascade, not a wall-clock deadline, and it is
already armed by default. So a wall-clock abort inside the supernodal driver
is not reachable from pounce without a feral change.

A thread-based watchdog was considered and rejected: the backend
(`Box<dyn SparseSymLinearSolverInterface>`) holds non-`Send`, stateful
symbolic-cache buffers that are reused across `multi_solve` calls, and
abandoning a factorization mid-flight would either leave those buffers being
mutated by an orphaned thread (unsound) or force a fresh `feral::Solver` per
factorization (destroying the symbolic-cache reuse the IPM depends on).

## Proposed feral-side hook

Add a cooperative cancellation flag that the supernodal driver polls at a
coarse-but-frequent granularity (per supernode, or every N eliminated
columns — often enough to bound overshoot to well under a second, rare
enough to stay off the hot inner kernels). Poll a shared `AtomicBool` rather
than a clock so feral stays clock-agnostic and pounce owns the
wall/CPU-vs-budget policy.

Sketch (feral):

```rust
// feral::numeric — new optional field on NumericParams (default None).
pub struct NumericParams {
    // ...
    /// When `Some`, the numeric driver checks `flag.load(Relaxed)` at each
    /// supernode boundary and returns `FactorStatus::Interrupted` (a new
    /// variant) if it is set. `None` (default) keeps the uninterruptible
    /// path, zero overhead.
    pub interrupt: Option<Arc<AtomicBool>>,
}
```

or, keeping it off `Solver`:

```rust
impl feral::Solver {
    /// Arm cooperative cancellation for subsequent `factor` calls.
    pub fn with_interrupt(self, flag: Arc<AtomicBool>) -> Self;
}
// and a new terminal status:
pub enum FactorStatus { /* ... */ Interrupted }
```

Contract on `Interrupted`: the `Solver`'s factors are left invalid (same as
any failed factor); the caller must not `solve` against them; a subsequent
`factor` re-runs cleanly. No partial results are promised.

## Wiring on the pounce side once feral ships it

1. `pounce_linsol::SparseSymLinearSolverInterface` grows an optional
   `set_interrupt(&mut self, Option<Arc<AtomicBool>>)` (default no-op), and
   `ESymSolverStatus` grows an `Interrupted` variant that
   `TSymLinearSolver::multi_solve` forwards.
2. `pounce_feral::FeralSolverInterface` arms `feral::Solver` with the flag
   and maps `FactorStatus::Interrupted -> ESymSolverStatus::Interrupted`.
3. `PdFullSpaceSolver` owns an `Arc<AtomicBool>`, hands it to the backend,
   and — because the factorization now runs on the caller thread and cannot
   be signalled from it — either (a) spawns a lightweight watchdog thread
   that sets the flag when `Deadline::exceeded()` first trips, or (b) relies
   on feral polling `Deadline` directly if the hook is expressed as a
   deadline rather than a flag. `ESymSolverStatus::Interrupted` from
   `aug_solver.solve` then returns `false` from `solve_once`, routing through
   the existing post-KKT `deadline_status()` check exactly like the
   between-op abort does today — no new termination path.

With that in place a `max_wall_time=5` solve returns in ~5 s plus one
supernode's worth of slack even when a single factorization would otherwise
run for 44 s, and the proactive guard added for #254 becomes a cheap
early-out on top of a hard mid-factorization bound.
