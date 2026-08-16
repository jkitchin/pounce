VERDICT: improved

Fixes #605

## Problem

`pounce.project_to_feasible` materialized dense projection matrices, and both
it and the solver's `least_square_init_primal` accepted the linearized
least-squares point without checking whether it improved the *true* nonlinear
constraints. Two independent defects with one root cause.

**1. The projection allocates `O(n²)` on `O(n)` data.** It built `P = eye(n)`
and a dense `m × n` Jacobian regardless of sparsity. On a chain model with 2
nonzeros per row, peak allocation quadruples on every doubling of `n` — the
signature of a quadratic allocation on linear data. The library's own
`PounceSparsityWarning` fires on the way through.

| `n` (chain, 2 nnz/row) | before | after | ratio |
|---|---|---|---|
| 300 | 2.97 MB | 0.41 MB | 7.2× |
| 1000 | 32.32 MB | 1.52 MB | 21× |
| 3000 | 225.66 MB | 4.65 MB | 49× |
| 6000 | 901.31 MB | 9.37 MB | 96× |

Before: ×4.0 per doubling. After: ×2.0 per doubling.

**2. The linearized step can make the true violation far worse.** On
`x₀² + x₁² = 1` started at `(0.05, 0.05)`, the Jacobian is `(0.1, 0.1)`: the
min-norm linearized correction is ~7 units long and lands at `(5.025, 5.025)`.

| | violation at `x₀` | violation at returned point | |
|---|---|---|---|
| Python, before | 0.995 | **49.50** | 49.7× **worse** |
| Python, after | 0.995 | 0.000976 | 1019× better |
| Rust init, before | 0.995 | **48.50** | 48.7× **worse** |
| Rust init, after | 0.995 | 0.1139 | 8.7× better |

Merit is `‖max(cl − g, g − cu, 0)‖₂` (Python) and
`curr_unscaled_nlp_constraint_violation_max()` (Rust) — the quantity the CLI
already reports as the model's constraint violation.

**3. An inconsistent linearization is all-or-nothing.** Two conflicting rows
(`x₀+x₁ = 1` and `x₀+x₁ = 3`) made the projection QP infeasible, and the helper
raised `RuntimeError`, returning nothing usable. Violation 3.162 → 3.162.

## Cause

A linearized solution is a local model step, not automatically a better NLP
starting point. Where `‖J‖` is small relative to the residual, the linear model
demands a correction long enough to leave the region where it is valid; the
true violation at the far end is unrelated to the residual the model predicted
it would close. Neither code path had a trust region, a fraction-to-boundary
control, or an actual-versus-predicted reduction test, so there was nothing to
detect that.

The dense allocation is separate and simpler: `solve_qp` accepts scipy-sparse
inputs, and the projection just never built them.

## Fix

**Python** — each outer iteration solves the sparse elastic QP

```
min_{d,p,q}  σ/2‖Dd‖² + ½‖Wp‖² + ½‖Wq‖² + ρ1ᵀ(p+q)
s.t.         cl − p ≤ g(x) + Jd ≤ cu + q
             max(lb − x + margin, −Δ/D) ≤ d ≤ min(ub − x − margin, Δ/D)
             p, q ≥ 0
```

`P` is diagonal, `J` stays scipy-sparse. The trust region is the ∞-norm box,
not `‖Dd‖₂ ≤ Δ` as the issue wrote it: the box keeps the subproblem a QP rather
than a QCQP and composes directly with the bound box, at the cost of a slightly
larger trust region in the corners. `P` is PSD by construction, so `solve_qp`'s
dense `O(k³)` PSD check is disabled — with the elastic columns widening the
variable block, that check had become the largest single allocation left (it is
what made `n=300` cost 6.7 MB before it was turned off).

`σ` defaults to **1**, not the issue's small regularizer. At `σ = 1e-8` the
step-norm term is negligible against `ρ = 1e3` and the QP returns an arbitrary
feasible point: the existing
`test_project_to_feasible_equality_and_inequality` caught this, returning
`(0.155, 0.845)` where min-norm is `(0.5, 0.5)`. At `σ = 1` the `d` term is
exactly the `½‖x − x0‖²` of a min-norm projection, and with a consistent
linearization the elastics price themselves out.

**Rust** — the direction is computed once, then walked back: `α = 1, ½, ¼, …`,
accepting the first with `θ(α) ≤ (1 − ηα)θ₀`. Since the linear model predicts
`θ → 0` at `α = 1`, predicted reduction is `αθ₀` and this is exactly "actual
reduction ≥ η × predicted". Each candidate is pushed into the bound interior
*before* being measured, so the accepted merit is the merit of the point the
algorithm really starts from and interiority holds by construction. If nothing
qualifies, the user's point is kept.

Each trial costs one constraint evaluation and no Jacobian or KKT solve.

**Resolved — `least_square_init_primal` is now a registered option, and this
branch is rebased onto that.** The original plan rejected registering it here
because PR #613 was already doing exactly that and duplicating it would have
conflicted. #613 has since merged (it is in this branch's merge base), so the
option is settable from every frontend and this branch simply uses it. The
tests now set `least_square_init_primal=yes` directly rather than reaching the
path through `mehrotra_algorithm=yes`, whose cascade also rewrites
`bound_push`, `bound_frac` and `bound_mult_init_val` — three confounds in a
test about one code path. The report is the same either way on that model (its
`x` bounds are far enough from every point the safeguard visits that neither
push moves it); the point is that the test no longer depends on that.

**Corrected:** `LeastSquareInitReport`'s doc comment claimed the report is
"printed at `print_level >= 5`". Nothing prints it — there is no such call
site. It is reachable only through
`IpoptApplication::least_square_init_report()`, and the comment now says so.

**Deliberate:** the full-length trial reuses `x_ls` itself rather than
`x0 + 1.0·(x_ls − x0)`. Those differ in the last bit, which was enough to move
two fixtures by an iteration each; reusing `x_ls` keeps an accepted step
bit-identical so the only trajectory change is on models where the step is
actually rejected.

## Merging #613

#613 registered the cold-start initialization options and refused two
unimplemented option *values*. It touched `init/default.rs`, `application.rs`,
`CHANGELOG.md` and `docs/src/initialization.md` — most of this branch's
footprint — so the branch was merged against it and re-measured rather than
re-reported.

`CHANGELOG.md` was the only textual conflict: both changes prepended to
`[Unreleased]`. Both entries are kept.

The code files auto-merged, and the hunks compose. In `init/default.rs` the two
changes are adjacent but disjoint: this branch rewrote step 1.5 (the
`least_square_init_primal` block), #613 removed the `bound_mult_init_method`
fallback in step 3 and the `cap_constraint_multipliers` helper it was the only
caller of. That removal did not land inside rewritten code, and nothing
references the deleted helper. #613's new `mu-based` refusal sits *after* the
safeguard rather than inside it, so a caller who builds the initializer
directly with both `least_square_init_primal=yes` and
`bound_mult_init_method=mu-based` pays for the safeguard's constraint
evaluations before being refused; the refusal a real caller sees is raised at
the application layer before any work, so this is only reachable from the
library backstop. In `application.rs` the two changes do not touch the same
regions.

`least_square_init_max_trials` and `least_square_init_accept_ratio` are struct
fields, never read from the options registry, so they do not trip #613's new
"read implies registered" invariant test. `docs/src/initialization.md` now says
they are not settable, which #613 made worth stating: every other knob on that
page now is.

**One thing did not compose, and it was not in a merged hunk.** #613 added
`option_behavior::least_square_init_primal_replaces_the_starting_point`, which
asserts that `least_square_init_primal=yes` moves the iterate to the stub
solver's `x_ls = [1, 3]`. Its stub NLP has `c(x) ≡ 1.0` — a *constant*
violation, which no step can reduce — so under this branch the safeguard
correctly declines the step and the starting point stays at the user's pushed
`x0`. The test encoded the pre-#605 contract: take the step unconditionally.
The stub's row is now `c(x) = x0 + x1 − 4`, which `x_ls = [1, 3]` satisfies
exactly. That keeps #613's assertion and its intent — the option is wired and
visibly moves the iterate, not a silent no-op — and makes the stub more honest
than it was, since `x_ls` is supposed to be the point that solves the
linearized constraints and now is one. All seven `option_behavior` tests pass.

## Blast radius

Baseline: **`70bf53ded3d893cfa2da6ead5195fda5ac096f68`** (`main` with #613),
release binary built from a `git worktree` at that commit, same machine, both
sides swept with `scripts/sweep-fixtures.sh`.

**Default options: bit-identical across all 57 fixtures.**
`least_square_init_primal` is `no` by default, so nothing moves unless it is
turned on. Since #613 there are two ways to turn it on — setting it directly,
or `mehrotra_algorithm=yes`, whose cascade turns it on — and both are swept
below.

### `mehrotra_algorithm=yes`: 12 lines move

| | baseline | after |
|---|---|---|
| fixtures solved | 27 | 27 (same set) |
| objectives on those 27 | — | all bit-identical |
| total iterations across those 27 | 292 | 292 |

**The set of moving fixtures is exactly the twelve measured against the old
`5d8ad36` baseline, and each moves the same way.** #613 was itself bit-identical
across the corpus, and re-measuring against it reproduces the earlier result
line for line. No fixture changed between success and failure in either
direction.

Every initializer decision below was re-measured against `70bf53d` from a
temporary probe compiled into the initializer, not carried over from the
earlier report:

| fixture | movement | initializer decision |
|---|---|---|
| `eigena2` | Succeeded, 21→20 it, same obj | full step rejected, accepted α=½ (θ 1.0→0.25) |
| `eigenb2` | Succeeded, 20→21 it, same obj | full step rejected, accepted α=½ (θ 1.0→0.25) |
| `boxed_qp_fixed_var` | RestorationFailed both | θ₀=0, step declined ("x0 already feasible") |
| `unbounded_cubic` | RestorationFailed both | θ₀=0, step declined |
| `unbounded_exp` | ErrorInStepComputation → InvalidNumberDetected | θ₀=0, step declined |
| `pooling_rt2stp` | RestorationFailed both | all 4 trials rejected, declined (θ 31.4 unchanged) |
| `deb7` | InfeasibleProblemDetected both | all 4 trials rejected, declined (θ 287.5 unchanged) |
| `hs71_obj1e8` | RestorationFailed → InfeasibleProblemDetected | all 4 trials rejected, declined (θ 1.76 unchanged) |
| `issue_372_infeasible_bounds` | RestorationFailed both | all 4 trials rejected, declined (θ 0.22 unchanged) |
| `cresc4` | InfeasibleProblemDetected → RestorationFailed | accepted α=¼ (θ 14014.45→13869.92) |
| `user_scaling_suffix` | RestorationFailed → InfeasibleProblemDetected | accepted α=⅛ (θ 0.5→0.4375) |
| `user_scaling_var_suffix` | RestorationFailed → InfeasibleProblemDetected | accepted α=⅛ (θ 0.5→0.4375) |

`eigena2`/`eigenb2` are the only two that solve; they move by one iteration in
opposite directions and cancel. The other ten all fail under Mehrotra on both
sides — Mehrotra disables globalization outright
(`adaptive_mu_globalization=never-monotone-mode`), and all ten solve or fail
identically under default options. Only *how* they fail changes.

`airport`, the LP-shaped case the code comments call this step critical for,
still accepts the **full-length step at α=1 with zero rejections**
(θ 103.6 → 25.96) and does not move: 9 iterations, `RestorationFailed`,
identical objective on both sides.

### `least_square_init_primal=yes`: 14 lines move — new, and not all good

This route did not exist when the original sweep was run: before #613 the
option was not settable, so `mehrotra_algorithm=yes` was the only way in. It is
now the cleaner way to reach the changed path, since it changes one thing
instead of four, and it shows a materially different picture that the Mehrotra
sweep hides.

| | baseline | after |
|---|---|---|
| `SolveSucceeded` | 46 | **44** |
| solved-or-acceptable | 46 | 46 (same set) |
| total iterations across those 46 | 2067 | **1687** (−18%) |
| fixtures using fewer iterations | — | 10 |
| fixtures using more iterations | — | 2 |

Ten fixtures get to the same answer in fewer iterations, several by a lot:
`deb7` 479→202, `pooling_rt2stp` 134→81, `hs71_obj1e8` 19→11, `eigena2` 78→65,
`linear_eq_aggregation` and `..._row_constant` 8→6, `user_scaling_suffix` and
`..._var_suffix` 11→8, `boxed_qp_fixed_var` 9→6, `csfi2` 53→35.

**Three regressions, stated plainly:**

- **`csfi2` and `eigenb2` come back `SolvedToAcceptableLevel` where the
  baseline returned `SolveSucceeded`.** Both still return an objective
  (`csfi2` bit-identical at 55.0176045, `eigenb2` 1.6 → 1.599999991), but they
  no longer meet the tight tolerance. This is a real quality downgrade on two
  of 57 models. Verified deterministic across repeated runs.
- **`pooling_rt2stp` lands on a worse local optimum**, −4391.83 → −3273.95.
  `deb7` moves the other way on the same mechanism, 249.75 → 97.56 (better).
  Both are nonconvex; a different starting point is entitled to a different
  local minimum, and the safeguard changes the starting point by design. It is
  still a worse answer on that model.
- `unbounded_cubic` takes 290 iterations to reach `DivergingIterates` where the
  baseline took 91. It diverges either way.

None of this is reachable without explicitly setting the option, and the
default-options sweep is bit-identical. But "the safeguard never makes things
worse" is true only of the *starting point's violation*, which is what it
measures and guarantees. It is not true of the trajectory that follows, and on
this route two fixtures pay for it.

**Behaviour changes, all deliberate and all in the CHANGELOG:**

- `project_to_feasible` no longer raises `RuntimeError` on an inconsistent
  linearization — the issue asks for elastic handling instead of an
  all-or-nothing solve. `test_project_to_feasible_inconsistent_raises` was
  rewritten to pin the new contract.
- It re-linearizes up to `max_iter` (default 3) times rather than once, so it
  spends more constraint evaluations and returns a smaller residual.
  `max_iter=1` restores the old budget.
- The Rust initializer leaves an already-feasible start alone rather than
  replacing it with the min-norm point. No step can improve a violation of
  zero, but it does change what three already-failing Mehrotra fixtures do.

**Not measured:** `nuffield2_trap`, named in the code comments as the model this
step exists for, is not in the CLI fixture corpus, so the sweep says nothing
about it.

## Feasibility gained per evaluation

The benchmark the issue asks for, both methods in one process on one machine.
The baseline is given repeated calls, since its own docstring recommends them.

| model | evals before → after | gain/eval before → after | evals to θ<1e-6 |
|---|---|---|---|
| chain n=300 | 10 → 9 | 0.676 → 0.751 | 9 → 8 |
| chain n=1000 | 10 → 9 | 1.236 → 1.373 | 9 → 8 |
| chain n=3000 | 10 → 9 | 2.141 → 2.379 | 9 → 8 |
| circle | 19 → 10 | 0.052 → 0.099 | never / never |
| rank-deficient | 4 → **5** | 0.354 → **0.283** | 3 → **4** |
| inconsistent rows | 3 (raised) → 9 | 0 → 0.194 | never / never |

Better on four of six. **Rank-deficient is a real regression: one extra
constraint evaluation**, the price of verifying a step that was going to be
accepted anyway. On a model where the linearization is already good, the
safeguard is pure overhead. That is the cost of the guarantee and it is bounded
at one evaluation per accepted step. Inconsistent rows go from "raises, returns
nothing" to 9 evaluations for a real improvement — more work, but the baseline
number is not a saving.

## Tests

`crates/pounce-algorithm/tests/issue_605_safeguarded_ls_init.rs` (2 tests) and
6 new tests in `python/tests/test_starts.py`. Plus the stub-row fix to #613's
`option_behavior` module described under **Merging #613** above.

**Verified to fail on the parent commit, for the right reason:**

Python — the three new assertions were run against the verbatim pre-#605
implementation with the new-API kwargs stripped, so each failure is behavioural
rather than a `TypeError`:

```
FAIL poor_linearization_never_worsens_violation
     AssertionError: projection made the violation worse: 0.995 -> 49.501249991749454
FAIL inconsistent_rows_degrade_via_elastics
     RuntimeError: project_to_feasible: projection QP ended with status
     'primal_infeasible' — the linearized constraints may be inconsistent
FAIL large_sparse_model_allocates_no_dense_blocks
     AssertionError: peak allocation 225.6 MB suggests a dense block
```

Rust — the parent exposes no report accessor, so the assertion was expressed
against a probe compiled into the parent's initializer in the baseline
worktree:

```
GH605-PARENT-PROBE theta0=0.995000 thetaN=48.501200
```

**Not pinned:** the end-to-end solve on the poor-linearization model. It hits
its iteration cap on both sides — a nonlinear equality from that start is not
something this configuration solves regardless of where it begins. The tests
assert on the initializer's own diagnostics instead, which is the unit of
behaviour that actually changed.

---

- [x] Tests fail on the parent commit for the stated reason
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms, alongside #613's
- [x] Book page under `docs/src/` updated (`initialization.md`; existing page,
      already in `SUMMARY.md`)
- [x] `cargo fmt --all -- --check` clean; `cargo clippy --workspace --exclude
      pounce-hsl --all-targets -- -D clippy::correctness -D clippy::suspicious`
      (CI's gate) clean
- [x] `cargo test --workspace --exclude pounce-hsl --no-fail-fast` — 2928
      passed / 0 failed / 6 ignored across 225 test binaries; Python 779 passed
      / 38 skipped; `check-release-consistency.sh` and
      `check-docs-consistency.sh` both OK
- [x] Every claim in this body is re-checked against the merged diff and
      re-measured against `70bf53d`

**One pre-existing flaky test.** `optimize_hs71::hs071_max_cpu_time_terminates`
sets `max_cpu_time = 1e-12` and expects `MaximumCpuTimeExceeded`; when the
solve beats the timer's granularity it returns `SolveSucceeded` instead. It
fails intermittently on the *unmodified* `70bf53d` baseline at about the same
rate as on this branch (2 of 6 runs there, 1 of 5 here, same binary, no code
change between runs). It touches nothing this branch changes and is not a
regression from it.
