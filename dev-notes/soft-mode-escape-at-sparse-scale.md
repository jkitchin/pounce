# Soft-mode escape moves: does the primitive scale? (gh#936)

gh#936 proposes drawing `--minima` escape moves along the soft modes of the
reduced Hessian instead of isotropically in raw `x`-space, and scopes its own
blocker first:

> **The dense eigendecomposition does not scale, and that is the whole
> problem.** […] Getting `k` soft modes at scale needs an iterative extraction
> — Lanczos against a reduced-Hessian matvec, or a randomized range finder —
> and `grep` finds no Lanczos/Arnoldi/randomized eigensolver anywhere in
> `crates/pounce-linalg/src/`. **This is the bulk of the work, not the hop
> logic.** Scope it first; if the matvec is not cheaply available the whole
> idea may not pay for itself.

This note answers that scoping question and nothing else. The experiment is
`crates/pounce-sensitivity/examples/soft_mode_scaling.rs`:

```
cargo run --release -p pounce-sensitivity --example soft_mode_scaling
```

**Verdict: the matvec is cheaply available, and the extraction is `O(1)`
back-solves in `n`.** The soft subspace at 100 000 variables costs **16
back-solves**, about **11% of one NLP solve** on the fixture measured. Two
facts the issue did not have are what change the answer, and both cut in the
idea's favour.

## 1. `B K⁻¹ Bᵀ` is the *inverse* reduced Hessian, so soft modes are the easy end

The issue's step 1 reads:

> `compute_reduced_hessian_eigen` returns eigenvalues in **ascending** order
> plus a column-major, sign-pinned eigenvector matrix. The soft modes are the
> leading columns, already in the order we want.

That is backwards, and `reduced_hessian.rs`'s own unit test is the proof.
`compute_reduced_hessian` computes `H_R = B K⁻¹ Bᵀ` — a submatrix of the KKT
**inverse**. For the test's `K = tridiag(-1,2,-1)` on rows `{0,2}` it returns
`[[3/4,1/4],[1/4,3/4]]`, whose inverse `[[3/2,-1/2],[-1/2,3/2]]` is exactly
the Schur complement of `K` onto that block — i.e. the true reduced Hessian.
So the returned matrix is the reduced Hessian's inverse, its eigenvalues are
reciprocals, and **the leading (smallest) columns of the ascending
decomposition are the stiffest modes, not the softest.**

Anyone implementing gh#936 off the issue text as written would have hopped
along the stiff directions — the exact failure the issue is about, with the
sign of the error hidden behind a plausible-looking spectrum. The
`1/sqrt(lambda)` weighting in step 2 inverts with it.

The consequence for scale is the good one. The operator we can *apply* is the
inverse reduced Hessian, so the soft modes are its **dominant** eigenpairs —
the end of the spectrum Lanczos reaches first, in a handful of iterations,
rather than the end that needs shift-invert and a second factorization.

## 2. The matvec is one back-solve against a factor the IPM already holds

`H_R v` is: scatter `v` into the `x` block, `kkt_solve`, gather the `x` block.
`PdSensBacksolver` retains the converged factor (it exists precisely so the
session API can make repeated `parametric_step` / `kkt_solve` calls after the
IPM returns), so this is a triangular back-solve, not a refactorization.
`kkt_solve_many` batches it. `pounce-feral/examples/shift_invert.rs` is the
same factor-once/solve-many pattern already written down in this repo.

The issue is right that no iterative eigensolver exists in the codebase; that
part of the estimate stands. It is wrong that the matvec is the risk.

## What was measured

Fixture: the 1-D chain `f = ½κ Σ(x_{i+1}-x_i)² + ½ε Σ x_i² − bᵀx`, Hessian
`H = κ·tridiag(-1,2,-1) + εI`, with closed-form eigenpairs

```
λ_k = 4κ sin²(kπ / 2(n+1)) + ε        v_k(i) = sin((i+1)kπ / (n+1))
```

chosen so that **ground truth is analytic at every `n`**. A study that could
only validate below `PSD_MAX_N = 256` would be measuring the regime where the
dense path already works — the `CLAUDE.md` branch rule, and the whole premise
of gh#936 is that the interesting regime is the one no fixture reaches. Its
soft modes are the chain's long-wavelength modes, which is also the cleanest
instance of the collective-variable analogy the issue borrows from MD.

### Matvec count is flat in `n`; it tracks the soft-end gap instead

Gap-matched (`ε = 1e-12`, so the chain's own `k²` structure sets the gaps),
`k = 8` softest modes, budget `m = 40`:

| `n` | cond(H) | soft-end rel. gap | worst-case subspace capture | matvecs |
|---|---|---|---|---|
| 1 000 | 4.1e5 | 3.00 | 1.000000 | 40 |
| 10 000 | 4.1e7 | 3.00 | 1.000000 | 40 |
| 50 000 | 1.0e9 | 3.00 | 1.000000 | 40 |
| 100 000 | 4.0e9 | 3.00 | 1.000000 | 40 |

Eigenvalues agree with the analytic ones to `9e-8` relative or better and
every `|cos∠|` is `1.000000`, at `n = 100 000` and condition number `4e9`.

The negative control is in the same program. Holding `ε = 1e-6` fixed while
`n` grows lets the `ε` floor swallow the `k²` gaps — at `n = 100 000` the
soft-end relative gap collapses from `3.0` to `3.0e-3` and the eight softest
eigenvalues sit within 0.1% of each other. Capture falls to **0.32** at the
same budget. So the precondition is a **soft-end relative gap**, not a
dimension; `n` is not the variable. This matters for the idea because the
sloppy-model cascade the issue cites (Gutenkunst) is *geometric*, which is
precisely the well-separated case.

### The budget is ~2k matvecs, not the 40 assumed

At `n = 100 000`, gap-matched, sweeping the Lanczos budget:

| `m` (= matvecs) | worst-case subspace capture |
|---|---|
| 10 | 0.464 |
| 12 | 0.592 |
| 14 | 0.987 |
| **16** | **0.999988** |
| 20 | 1.000000 |

So `k = 8` soft modes cost **16 back-solves** — about `2k`, independent of
`n`, against the `n` back-solves plus `O(n³)` Jacobi of the dense path.

### Cost against the thing it is meant to save

At `n = 100 000` (distinct right-hand sides on both paths — a repeated
identical RHS hits the solver's last-solve fast path and overstates batching
by ~3x):

| | per solve | 40 solves vs one NLP solve |
|---|---|---|
| one NLP solve | 3.11 s | — |
| `kkt_solve`, one at a time | 48.0 ms | 62% |
| `kkt_solve_many`, batched | 21.9 ms | 28% |

At the measured budget of 16, batched, that is **~11% of one NLP solve**. One
wasted isotropic hop costs 100% of one. **Break-even is a duplicate-rate
reduction of roughly 1 in 9** — which is the number gh#936's proposed
experiment should be scored against.

Full reorthogonalization is this program's own `O(m²n)` bookkeeping and is
minor (0.29 s of 2.22 s at `n = 100 000`, `m = 40`); at a budget of 16 it is
smaller still.

### Active bounds are suppressed automatically

gh#936's third design note:

> Bound and constraint activity interacts with this: a soft mode pointing into
> an active bound is not usable. Either project the mode into the free
> subspace or drop it.

No work is needed. An active bound puts `Σ = z/s` (large) on that `x`
diagonal, so `(K⁻¹)_xx` is *small* there and the dominant end avoids it
unaided. Measured on the chain at `n = 10 000` with 200 actively-bounded
variables (a live 200-row `z_u` block in the factor): each soft mode carries
between `5e-31` and `1e-27` of its squared norm on the bounded coordinates,
against the `0.0200` a direction with no preference would carry — 25 orders
of magnitude of suppression, for free.

This is the same mechanism `reduced_activity` relies on, and it is why the
`x`-block operator is the right one to iterate on rather than an explicitly
projected free-variable operator.

## What this does *not* establish

Per the branch rule, stated explicitly so a green result here is not read as
evidence about something it never touched:

* **Efficacy is untested.** This measures whether soft modes can be *obtained*
  at scale. Whether *hopping along them* finds more distinct minima per NLP
  solve — gh#936's own metric, and the stated go/no-go — is not measured here
  and remains the decision point. The issue is right that it needs a
  purpose-built multimodal benchmark at scale, and right that the existing
  corpus would return a meaningless null.
* **One KKT shape.** The fixture is a convex QP with a tridiagonal Hessian:
  the cheapest back-solve and the cheapest NLP solve there is. Both sides of
  the cost ratio move on a real multimodal NLP — the solve gets much more
  expensive (many iterations, each with a factorization) while the back-solve
  stays one triangular solve, so the ratio should improve; but a denser or
  less structured factor makes each back-solve dearer. Unmeasured.
* **Equality/inequality constraints are absent** (`m = 0` except for bounds in
  the third case). The `y_c` / `y_d` / `s` blocks are never exercised, and the
  `x`-block-of-`K⁻¹` argument, while structurally unchanged, is not measured
  with them live.
* **No Lanczos exists in the library.** The one in the example is scoping
  scaffolding — no reorthogonalization strategy beyond the brute-force one, no
  restart, no convergence test. Shipping this needs a real implementation; the
  issue's estimate of that work is not disputed.

## Recommendation

The Lanczos work is worth starting, and it is smaller than gh#936 budgeted for
— the dominant end plus an existing back-solve is the easy case, not the hard
one. Two corrections to the issue before anyone implements it: the soft modes
are the **trailing** columns of the current ascending decomposition, not the
leading ones, and design note 3's free-subspace projection is unnecessary.

But the order in gh#936 still stands, for the reason gh#936 gives. The
extraction being cheap does not make the hop effective, and the efficacy
experiment is the one that can kill the idea. Build the multimodal benchmark
and score duplicate rate against the ~1-in-9 break-even above **before**
writing a production Lanczos.
