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

**Verdict: the matvec is cheaply available, the extraction is `O(1)`
back-solves in `n`, the iterative eigensolver the issue budgets for is not
needed at all, and on a purpose-built multimodal benchmark the hop pays ~40x
at `n = 10 000` — but only when barriers lie along soft directions, and ~0.04x
when they do not.** The soft subspace at 100 000 variables costs **16
back-solves**, about **11% of one NLP solve** on the fixture measured — and a
randomized range finder plus a `k×k` dense Rayleigh-Ritz matches the Lanczos
path at the same cost with none of the machinery (see *Do we need the
eigensolver at all?* below). Two facts the issue did not have are what change
the answer, and both cut in the idea's favour.

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

* **Efficacy is measured, but on synthetic fixtures only.** See the efficacy
  section below: the method pays ~40x when barriers lie along soft modes and
  ~0.04x when they do not. Both branches are purpose-built synthetic
  landscapes with the answer designed in; neither is evidence about which
  regime a real `--minima` workload occupies, which is now the open question.
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
* **The extraction code in the example is scoping scaffolding.** Brute-force
  reorthogonalization, no restart, no convergence test, fixed budgets. The
  section below argues the shippable version is a randomized range finder
  rather than any of this, but that version is not written either.

## Efficacy: it works, and only when the premise holds

Everything above is about *extraction*. The `soft_mode_scaling.rs` fixture is a
convex quadratic with exactly one minimum, so it cannot say whether hopping
along these directions escapes anything. `minima_hop_efficacy.rs` is the other
half, scored on gh#936's own metric — distinct minima per NLP solve, both
strategies hopping with the same step norm from the same incumbent under the
same seed stream, so only the *direction* differs.

Two fixtures had to be discarded first, and both failures are informative.

* **A Frenkel-Kontorova chain** (every coordinate in its own periodic well) is
  degenerate for this metric: exponentially many minima mean *every* hop lands
  somewhere new. Duplicate rate `0.0%` for both strategies at `n` = 200 /
  2 000 / 20 000, ratio exactly `1.00x`. The metric saturates and
  discriminates nothing — gh#936's own trap one level up, a *metric* uniform
  in the dimension the change acts on.
* **Corrugating a collective coordinate** failed for a deeper reason: the
  corrugation that creates a well also supplies the curvature at the bottom of
  it, so the barriered direction is *stiff* and the soft modes are the
  uncorrugated ones that lead nowhere. Soft-mode hops moved the collective
  coordinates by exactly `0.0000` at every `n`.

The working fixture is `½Σc_i x_i² + ½ρΣ(x_{j+1}−x_j)² + AΣ_{i∈C}(1−cos(2πx_i/P))`
with `c_i` small on `M` designated coordinates and large elsewhere, `ρ` a weak
tridiagonal coupling, and **`C` a switch**: corrugate the soft coordinates
(gh#936's premise holds) or an equal number of stiff ones (it fails).
Everything else is held fixed, so the switch isolates the premise. Per the
branch rule, one branch alone would have been worthless.

**Branch 1 — barriers along soft modes.** The advantage is real and it *grows
with dimension*, which is the specific claim gh#936 makes and the one no
existing fixture could test:

| `n` | isotropic dup rate | soft-mode dup rate | distinct/solve ratio |
|---|---|---|---|
| 16 | 12.2% | 4.9% | 1.08x |
| 100 | 63.4% | 7.3% | 2.53x |
| 1 000 | 97.6% | 0.0% | **41.0x** |
| 10 000 | 97.6% | 2.4% | **40.0x** |

At `n ≥ 1000` isotropic hopping finds *nothing* — 1 minimum in 41 solves,
0.00 well crossings per hop — while soft-mode hopping crosses ~2.9 wells per
hop and finds a new minimum almost every time. This is "plausible at 813 and
implausible at 93 263" as a measurement. Against the ~1-in-9 break-even
computed above, 40x is not marginal.

**Branch 2 — barriers along stiff modes.** The same machinery is *worse than
doing nothing*:

| `n` | isotropic distinct/solve | soft-mode distinct/solve | ratio |
|---|---|---|---|
| 16 | 0.561 | 0.024 | **0.04x** |
| 100 | 0.171 | 0.024 | **0.14x** |
| 1 000 | 0.024 | 0.024 | 1.00x |
| 10 000 | 0.024 | 0.024 | 1.00x |

Where isotropic still works, soft-mode hopping finds exactly one minimum and
never crosses a well — it spends every solve travelling along directions with
nothing at the end of them. A 25x *regression* at `n = 16`.

(Branch 2 needed its amplitude sized separately: at branch 1's `A = 0.02` a
stiff coordinate's quadratic swamps the corrugation and there is only one
minimum, which the first run reported as `1` for both strategies — not "the
premise fails" but "there is nothing to find". `wells_exist` asserts the
fixture is non-vacuous. That a barrier on a stiff direction must be
proportionally higher to exist at all is itself structural.)

So the feature is not a general improvement to be switched on. It is a bet
that the problem's barriers lie along its soft directions, paying ~40x when
right and ~0.04x when wrong, and **nothing in the solver can tell which
regime a user's model is in.** That makes fallback and opt-in load-bearing
rather than defensive, and it makes "which regime do real `--minima`
workloads live in?" the question that decides whether this ships — not any
remaining question about cost.

## Do we need the eigensolver at all? (mostly no)

The measurements above extract *eigenpairs*. An escape move does not need
them — it needs a random direction that is long for its objective rise. So
section F of the example scores candidate move constructions on two axes at
once, on the same fixture at `n = 100 000`:

* **reach** — `‖y‖ / sqrt(½ yᵀHy)`, distance travelled per unit objective
  rise. This is gh#936's "travel far for little change in the objective",
  made into a number, and it is invariant to `‖y‖` so constructions compare
  without tuning a step size for each. The ceiling is travel along the
  softest mode alone.
* **diversity** — mean `|cos∠|` between independent moves. This axis is the
  one gh#936's own metric cares about: moves that all point the same way
  re-enter the same basin, which is a duplicate and a wasted NLP solve.

The two trade off against each other, and no single number decides it: the
ceiling is reached only by always moving along `v₁`, which has diversity 1.0
— every hop identical.

| construction | back-solves | reach, well-separated | reach, degenerate | diversity |
|---|---|---|---|---|
| isotropic `N(0,I)` | 0 | 0.0% | 0.1% | 0.002 |
| one back-solve `A g` | 1 | **69.3%** | **69.6%** | 0.73 / 0.15 |
| `A^{1/2} g`, Krylov, `m = 160` | 160 | 0.3% | 1.7% | 0.41 / 0.10 |
| range finder, `k = 8`, 2 passes | 16 | 18.3% | 87.1% | 0.29 |
| range finder + `8×8` Ritz, `1/sqrt(λ)` | 24 | **45.5%** | **87.8%** | 0.54 / 0.31 |
| gh#936 eigen: 8 modes, `1/sqrt(λ)` | 20 | 42.8% | 54.9% | 0.58 / 0.32 |

Three results, in order of how much they change the plan.

**A randomized range finder replaces the eigensolver.** `Q = orth(AᵠΩ)` for a
Gaussian `Ω` spans the soft subspace in `q·k` back-solves that **batch into a
single `kkt_solve_many` call**, with no restart, no locking, no convergence
test — and no dependence on the gaps *between* the soft modes, only between
the block and the rest. Adding a `k×k` Rayleigh-Ritz to recover the
`1/sqrt(λ)` weighting matches or beats the full Lanczos path on both axes
(45.5% vs 42.8% well-separated, 87.8% vs 54.9% degenerate) for `24`
back-solves against `20`. The `k×k` symmetric eigendecomposition is
`pounce_linalg::symmetric_eigen` on an 8×8 matrix — arithmetic, not an
eigensolver.

This is the alternative gh#936 lists in passing ("or a randomized range
finder") and then drops in favour of Lanczos. It is the better half of that
sentence, and it means **the item the issue calls "the bulk of the work, not
the hop logic" does not need to be built.**

**The matrix-function shortcut does not work — measured, and it was my own
hypothesis.** `A^{1/2}g` with `g ~ N(0,I)` is a draw with covariance `H_R⁻¹`,
which is *exactly* gh#936's stated `1/sqrt(λ)` weighting obtained without
naming a mode, and it is gap-independent. It should have been the clean
answer. It is not: approximating `sqrt` by a Krylov polynomial across a
spectrum spanning nine orders converges far too slowly. The self-check
`yᵀHy / gᵀg`, which is exactly `1` for an accurate square root, reads
**113.7** at `m = 20`, `27.5` at `m = 40`, `6.7` at `m = 80` and still
**1.87** at `m = 160` — eight times the range finder's cost for a worse
answer. Recorded because it is the obvious thing to try second.

**One back-solve is the reach champion and the diversity disaster.** `A g` is
a single power-iteration step, so it wins reach outright (69% of ceiling for
**one** back-solve) by collapsing onto `v₁` — diversity 0.73 where the soft
end is well separated. Since duplicate rate is the metric, that is the wrong
trade. It is, however, an almost-free *fallback*, and strictly better than
isotropic on both axes in the degenerate case (69.6% reach at diversity 0.15).

The knob across the whole table is how hard to concentrate the draw toward the
softest mode: more concentration buys reach and costs diversity. Where to sit
on that frontier is a question only the efficacy experiment can answer, which
is the next section's point.

## Recommendation

**Do not build an iterative eigensolver for this.** A randomized range finder
plus a `k×k` dense Rayleigh-Ritz gets the soft subspace and the `1/sqrt(λ)`
weighting in ~`2k` back-solves, batched into one or two `kkt_solve_many`
calls, and matches or beats the Lanczos path on both reach and diversity.
That removes what gh#936 scoped as the bulk of the work.

Three corrections to the issue before anyone implements it:

1. The soft modes are the **trailing** columns of the current ascending
   decomposition, not the leading ones — `B K⁻¹ Bᵀ` is the reduced Hessian's
   inverse.
2. Design note 3's free-subspace projection is unnecessary; `Σ = z/s`
   suppresses active bounds by ~25 orders unaided.
3. The Lanczos/randomized-eigensolver work item can be struck.

The go/no-go has moved. Extraction is cheap on every axis and efficacy is no
longer hypothetical: ~40x when gh#936's premise holds, ~0.04x when it does
not, with the crossover governed by whether barriers lie along soft
directions. What is *not* established is which regime real `--minima`
workloads occupy — and since the solver cannot detect it, that decides both
whether this ships and whether it can ever be a default.

So: keep it opt-in with an isotropic fallback, and spend the next effort on
real multimodal models rather than more synthetic ones or any production
extraction code.
