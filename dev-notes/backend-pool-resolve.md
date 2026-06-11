# BackendPool + Solver::resolve() — design note

**Status: design only, not implemented.** This note captures the
remaining piece of Phase 3b from the `factor-reuse-session-api` plan
so the work can resume cleanly with the unknowns called out up front.

## What's shipped vs deferred

Already shipped on `claude/factor-reuse-session-api`:

* `pounce_linsol::Factorization` — public factor-once / solve-many.
* `pounce_sensitivity::Solver` — hold the converged KKT factor; issue
  parametric_step / reduced_hessian / kkt_solve against it.
* Python `pounce.Solver`, C `IpoptSolver` — same surface in Python / C.
* Examples + measurement: `examples/sensitivity_session.rs`,
  `examples/parametric_mpc.rs`, `examples/factor_reuse_bench.rs`
  (measured ~10× speedup on the parametric-step half of an MPC loop).

Deferred: **IPM-level warm-start across `solve()` calls.** A
`Solver::resolve()` that re-runs the IPM on a perturbed problem while
reusing the symbolic factor + AMD ordering from the previous solve.
This is the MPC / B&B path; the sensitivity path doesn't need it.

## Why it's not a one-line plumbing change

The plan summary made this sound simple:

> `crates/pounce-algorithm/src/alg_builder.rs:332-340` — accept a
> `BackendPool` instead of a `LinearBackendFactory` (or wrap the
> factory). The pool returns either a fresh backend (first solve) or
> the existing one with `reset_for_new_solve()` called on it.

Three real problems hide behind that line.

### 1. Backend ownership is single-owner today

The factory hands out `Box<dyn SparseSymLinearSolverInterface>`. The
box is moved into `TSymLinearSolver::new`
(`crates/pounce-linsol/src/sparse_sym_linear_solver.rs`), which is
moved into `StdAugSystemSolver`, then into `PdFullSpaceSolver`, then
into `PdSearchDirCalc`, then into `AlgorithmBundle.search_dir`, then
into the algorithm proper. The whole chain is dropped when the algorithm
goes out of scope.

To reuse the backend across two top-level `solve()` calls, one of:

* **Drop-and-recover.** Add a way to extract the backend back out of
  `PdSearchDirCalc` / `PdFullSpaceSolver` / `StdAugSystemSolver` /
  `TSymLinearSolver` at the end of a solve, hand it back to the pool,
  pull it out again on the next call. Adds a `take_backend()` method
  on each of those layers. Mechanical but touches a lot of files.
* **Shared ownership.** Change the chain to hold
  `Rc<RefCell<Box<dyn SparseSymLinearSolverInterface>>>` so the pool
  keeps a clone alongside the algorithm. Trait methods are `&mut self`,
  so every callsite has to do `RefCell::borrow_mut`. Less file churn
  but more contention surface and harder to reason about.

Both are real refactors. Neither is hard; both need to be done
carefully so existing tests stay green.

### 2. `reset_for_new_solve()` semantics

A second solve against the same backend needs to clear per-solve scratch
without losing the symbolic factor:

* **feral** (`crates/pounce-feral/src/lib.rs`) caches its symbolic
  factor via a sparsity-pattern fingerprint. If we feed it the same
  `(ia, ja)` and call `factor()` again it already reuses the cached
  symbolic — no `reset` needed. Confirmed.
* **MA57** (`crates/pounce-hsl/src/ma57.rs`) caches the symbolic
  factor in `keep`. The `keep` workspace must be preserved; the
  numeric workspaces (`fact`, `ifact`) can be reused. Adding a
  `reset_for_new_solve()` no-op that zeros only the numeric scratch
  is straightforward.

The trait extension is small: an `fn reset_for_new_solve(&mut self) {}`
default-implemented method. Both backends override or accept the default.

### 3. What's allowed to change between resolves?

This is the **real** unknown. Upstream Ipopt's re-solve path has
non-trivial behavior here. Decisions we'd need to make:

| Change between resolves | Plausible behavior                                              |
|---|---|
| `x0` (starting point)   | Re-read from TNLP. Cheap.                                        |
| Variable bounds `x_l`/`x_u`     | Re-read. May invalidate fixed-variable detection — re-run that pass. |
| Constraint bounds `g_l`/`g_u`   | Re-read. Same as above; equality detection depends on `g_l == g_u`. |
| Objective / constraint values   | Always re-evaluated; nothing to do.                              |
| Jacobian / Hessian sparsity     | **Disallowed.** Would invalidate the symbolic factor.            |
| Scaling                         | `nlp_scaling_method` is per-application. If reapplied it has to be re-run. Easier to lock to first-solve scaling. |
| Options (`tol`, `max_iter`, …)  | Allow most; `linear_solver` change disallowed (defeats the point). |
| Restoration state               | Always re-initialized; restoration shouldn't survive across resolves. |

The right answer is probably "allow x0 / bounds / non-structural
options; document everything else as undefined behavior", but that's
worth pinning down with a concrete consumer (MPC controller? warm-start
B&B?) before committing to the API.

## Suggested implementation order

If picked up:

1. **Add `reset_for_new_solve()` to the trait** (default no-op). Wire
   the MA57 override. Confirm feral fingerprint reuse on its own
   already gives symbolic-factor reuse. **Tests:** repeat factor on a
   fixed pattern with new values; assert symbolic AMD ordering doesn't
   change between calls (instrument or count work in a stub backend).
2. **Pick an ownership model** (drop-and-recover vs Rc-shared) and
   refactor the linsol chain. This is mostly mechanical but is the
   work item where a regression could land. Ship as its own PR.
3. **Implement `BackendPool`** as a `LinearBackendFactory` closure
   that captures a slot. First call constructs; subsequent calls
   recover the cached backend.
4. **Implement `Solver::resolve()`.** Read `x0` / bounds / options
   freshly from the TNLP, reuse the cached backend via the pool,
   re-run `optimize_tnlp`. Stops short of true iterate warm-start
   (carrying `x*` and multipliers forward as a starting point); that
   can come second.
5. **True iterate warm-start.** Optional follow-up — would save more
   than just the symbolic factor.

## What we'd want to measure

* **`bench_resolve_vs_cold`**: 100 perturbed solves of HS071 via
  `solve + 99 resolves` vs `100 fresh solves`. Plan target: ≥2×
  speedup. Realistic: depends on (IPM iterations × per-iteration KKT
  cost) vs (symbolic factor + ordering) ratio for the problem size.
* **Symbolic-factor reuse fraction**: instrument the backends to
  count symbolic vs numeric work and confirm symbolic is zero on
  every `resolve()` after the first.

## Why this isn't shipped today

Two reasons, in order of importance:

1. **No concrete consumer in tree.** The MPC / B&B / warm-start
   workloads are real but none are wired up in the POUNCE codebase
   yet, so we'd be building the API without a forcing function for
   the semantics in §3.
2. **The ownership refactor (§1) is genuinely invasive** — it's the
   kind of change that should ride in on its own with isolated tests,
   not bundled with new functionality.

The shipped `Solver` value type is the seam this work plugs into. The
public API doesn't need to grow to accommodate `resolve()`; the
internals do.
