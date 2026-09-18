# RSLAB as a POUNCE KKT backend — assessment

Measured on `crates/pounce-rslab` against RSLAB
`milanofthe/rslab@8268b2e3` (0.36.0), FERAL 0.17.0, POUNCE at
`ba95130`. Every number below is reproducible from the crate:

```sh
cargo test    -p pounce-rslab                                  # the contracts
cargo run -p pounce-rslab --release --example rslab_kkt_replay # real KKT systems
cargo run -p pounce-rslab --release --example rslab_nlp_solve  # whole NLPs
cargo run -p pounce-rslab --release --example rslab_bench       # phase timings
cargo run -p pounce-rslab --release --example rslab_scale_smoke # end-to-end, 3 scales
cargo run -p pounce-rslab --release --example kkt_zero_census   # the explicit-zero census
```

MA57 was not measured: `libcoinhsl` is not on the link path here. The harness
carries the arm (`--features ma57`) and records it as unavailable rather than
silently dropping it from the comparison.

---

## Summary

**RSLAB in its exact mode cannot factor a POUNCE KKT.** Over 90 KKT systems
captured from six real models it completed 13; FERAL completed 90. The cause is
structural and RSLAB documents it: Bunch-Kaufman pivoting is restricted to each
front's fully-summed block and there is no delayed pivoting, so a saddle-point
row factors only when its 2×2 partner happens to land in the same front.

**With static pivoting it can**, and the factor it then produces is a
preconditioner: pivots are lifted, the reported inertia is that of `A + E`, and
the residual is 10³–10⁹ times FERAL's on the same system.

**And on one model that is better than FERAL.** On `eigena2` — the model pounce
gh#540 was filed against — RSLAB under static pivoting converges
(`SolveSucceeded`, 21 iterations, `constr_viol` 3.1e-13) where FERAL enters
restoration and fails (17 iterations, `constr_viol` 1.1e-9). Deterministic over
three runs, and attributable to the factorization rather than to the adapter's
inertia gate: the two arms that differ only in that gate both converge.

That last result is probably **not a reason to adopt RSLAB**, but the obvious
explanation for it is wrong. RSLAB is a fork of FERAL (its `NOTICE` says so) and
was running lift-to-floor static pivoting, which FERAL has and POUNCE could not
reach — so the natural prediction was that exposing
`Solver::with_static_pivot_threshold` would reproduce it. It was exposed, swept
over three decades, and **reproduces none of it**. The `eigena2` result stands
unexplained. What the same sweep did find is that `feral_static_pivoting`, a
knob POUNCE already had, is the only setting that converges `lp_degen2`. Both
in §4, "The control, and where the `eigena2` win actually comes from".

**And it is 3.8–6.3× slower end to end** on a banded KKT at three scales
(1.5k / 15k / 150k rows), with identical iteration counts on every arm so the
ratio is comparable. 92% of that time is RSLAB's own numeric factorization and
~2% is the adapter, so it is not an integration artefact.

The verdict is therefore not "RSLAB is worse". It is that RSLAB is missing one
specific thing POUNCE needs, that the thing is nameable and fixable, and that
what is on the other side of it is interesting enough to be worth the fix.

---

## 1. The POUNCE linear-solver contract

### The trait

`pounce_linsol::SparseSymLinearSolverInterface` (a port of Ipopt's
`IpSparseSymLinearSolverInterface.hpp`) is what a backend implements.
`pounce-feral` and `pounce-hsl` implement it; so does `pounce-rslab`. The
lifecycle:

```text
matrix_format()                        -> the layout the caller must marshal into
initialize_structure(dim, nnz, ia, ja) -> pattern, once
values_array_mut()                     -> the caller fills this slice
multi_solve(new_matrix = true,  ...)   -> factor, then back-substitute
multi_solve(new_matrix = false, ...)   -> back-substitute only
number_of_neg_evals()                  -> the inertia POUNCE steers on
```

There is no factor-only entry point. Every `multi_solve` also back-substitutes;
`pounce_linsol::Factorization::do_factor` issues a throwaway zero RHS to get a
factorization out of it, and accepts the cost of one triangular solve.

`EMatrixFormat` offers five layouts. Both FERAL and RSLAB take
`TripletFormat`: **COO over the lower triangle, 1-based indices**, duplicates
permitted and summed. That is also the format `Factorization::new` documents as
"the universal denominator the trait expects".

### What the caller expects beyond `solve(Ax = b)`

| Expectation | Where it lives | RSLAB |
| --- | --- | --- |
| symbolic reuse across refactorizations | implicit — the pattern is fixed by `initialize_structure` and only values change | `analyze_with` once, `factor_numeric` many |
| numeric refactorization, pattern unchanged | `Factorization::refactor` | same |
| multiple RHS | `multi_solve(nrhs, …)`, column-major | row-major; the adapter transposes |
| matrix scaling | `TSymScalingMethod` one layer up, plus whatever the backend does internally | one-pass inf-norm inside `LdltSolver`; the adapter reproduces it |
| permutation | `factor_pattern()`, diagnostic only (`--dump kkt:*+L`) | available but not surfaced; the adapter returns `None` |
| factorization statistics | `LinearSolverSummary` | partially — see §3 |
| bit-identical multi-RHS | `multi_solve_matches_single_solve(nrhs)`, default `false` | left at `false` |
| a quality ladder | `increase_quality()` | none; the adapter reports `false` |
| degeneracy detection | `provides_degeneracy_detection` / `determine_dependent_rows` | not implemented |

### How POUNCE consumes inertia

This is the part a backend most easily gets subtly wrong, so it is worth being
exact.

POUNCE reads **one number**: `number_of_neg_evals()`, the count of *strictly
negative* pivots — MA57's `INFO(24)` convention. Not a triple. The positive and
zero counts never reach the algorithm.

Rank deficiency is signalled **separately**, by returning `Singular` rather than
by folding zeros into the negative count. That distinction is load-bearing:
`Singular` routes `PdPerturbationHandler` to `perturb_for_singular`, which
bumps `δ_c` on the constraint block; `WrongInertia` routes it to
`perturb_for_wrong_inertia`, which bumps `δ_x` by ×8 per retry. Folding zero
into the negative count — the SSIDS bookkeeping convention — is correct
spectral accounting and breaks Ipopt's singularity branch on LP-shaped KKTs
whose (3,3) block is structurally zero (pounce gh#52, feral gh#54).

There is a **reliability test** on top of that, and preserving it was the main
constraint on the adapter's design. `pounce_feral::FeralSolverInterface::factor`
reports `Singular` instead of `WrongInertia` when the count mismatches *and* the
smallest accepted pivot magnitude sits below
`pounce_feral::inertia_trust_floor(configured, dim)`, which defaults to `n·ε` —
the backward-error bound on the smallest pivot of an equilibrated matrix of
order `n`. Below it, a pivot's sign is a rounding artefact and the count built
from those signs is noise (pounce gh#540: on `eigena2` the same factorization
returned 64 / 58 / 62 negatives for the same expected 55, and LAPACK's exact
count agreed with none of them). `1e-12`, the fixed floor this replaced, is more
than an order of magnitude too generous at the dimensions an IPM actually
factors (gh#592).

There is also a second, independent floor: `singular_pivot_floor`, the MA57
`CNTL(2)` analogue, which reports `Singular` on any completed factorization
whose smallest pivot falls below an absolute threshold. It is off by default. A
`min/max` *ratio* test would be wrong here — an interior-point KKT is designed
to become ill-conditioned as `μ → 0`, so the ratio collapses on healthy
full-rank systems near the solution.

Both floors need one quantity: **the smallest accepted pivot magnitude, with
2×2 blocks counted by their smaller eigenvalue.** FERAL exposes it as
`Solver::min_pivot_magnitude()`, in the scaled space of the matrix it actually
factored. RSLAB does not expose it anywhere, which is the whole reason the
adapter is built the way it is.

---

## 2. RSLAB's LDLᵀ pivoting and inertia

### The kernel

`src/dense/ldlt_generic.rs` is the unblocked Bunch-Kaufman scheme of LAPACK's
`?sytf2`, generic over the scalar field, with the classical threshold
`α = (1 + √17)/8 ≈ 0.6404`. `src/numeric/multifrontal_ldlt.rs` applies it
front-by-front, in two drivers: `FactorMethod::LeftLooking` (the shipped
default) and `FactorMethod::Multifrontal`. `D` comes back as `d_diag`,
`d_subdiag` and a `two_by_two: Vec<bool>` marking each 2×2 block's first
column — a cleaner representation than FERAL's `d_subdiag[k] != 0.0`
convention.

### Pivoting scope, and the defect

RSLAB's own module documentation, `multifrontal_ldlt.rs` lines 16–25:

> Pivoting is restricted to the **fully-summed block** of each front … There is
> no delayed pivoting: a fully-summed block that is singular in exact mode
> surfaces as `RslabError::NumericallyRankDeficient`, and the static-pivot mode
> lifts the pivot to the floor instead.

FERAL delays such a pivot to the parent front, SSIDS/MUMPS-style — `n_delayed`,
`PivotStepResult::Delayed`, `DelayBudgetExceeded`. RSLAB has the error variant
but no delay machinery on the symmetric path.

This is not a corner case for POUNCE. A KKT's (2,2) block is structurally zero,
so every constraint row needs either a 2×2 pivot or an off-diagonal partner, and
RSLAB gets one only when the amalgamation happens to place both in the same
front. `crates/pounce-rslab/tests/cross_solver.rs::a_sparse_jacobian_saddle_point_reproduces_the_failure`
is the minimal form: a grid saddle point `[[W + Σ, Jᵀ], [J, 0]]` factors fine
with a dense-row `J` at every size up to `n = 160 400`, and fails at every size
with one Jacobian entry per row — the arrow shape a slack or simple-bound row
gives a KKT. FERAL factors both.

### Inertia

`rslab::Inertia { positive, negative, zero }` — exactly POUNCE's shape,
accumulated per front and summed over the assembly tree (`multifrontal_ldlt.rs`
lines 3255–3311 for the left-looking emit, 1026–1116 for the multifrontal
front). The two paths use the same rule.

**1×1 pivots** are classified by the sign of `d.real()`, with **exact
comparison to zero**: `d > 0 → positive`, `d < 0 → negative`, else `zero`. No
tolerance. FERAL uses `|d| <= zero_tol` with `zero_tol = f64::EPSILON` by
default.

**2×2 blocks are classified correctly** — this was worth checking, and the
answer is yes. The rule is `sign(det)` and `sign(trace)` over the *whole* block,
not the signs of `a` and `c`:

```rust
let det_r = (d[pp] * d[pp + 1] - d_subdiag[pp] * d_subdiag[pp]).real();
let tr_r  = (d[pp] + d[pp + 1]).real();
if det_r < 0.0 { ipos += 1; ineg += 1; }             // eigenvalues straddle zero
else if det_r > 0.0 { if tr_r >= 0.0 { ipos += 2 } else { ineg += 2 } }
else { izero += 1; if tr_r >= 0.0 { ipos += 1 } else { ineg += 1 } }
```

`[[0, 1], [1, 0]]` therefore reads `(1, 1, 0)`, which
`tests/inertia.rs::antidiagonal_2x2_block_inertia` pins against a real
factorization, checking `two_by_two_pivots == 1` so the test cannot pass by
taking the 1×1 branch.

**The determinant is evaluated naively.** `a*c - b*b` loses every significant
digit — and can round to exactly `0.0` or flip sign — when the two products are
close, which is the shape of a borderline KKT pivot. FERAL evaluates the same
determinant with Kahan's fused difference-of-products (`det_sym2x2`, relative
error `≤ 2u` for any inputs, so the sign is exact unless the block is genuinely
singular). `pounce_rslab::inertia::det_sym2x2` reproduces the fused form and
`fused_determinant_keeps_a_sign_the_naive_one_loses` exhibits an `O(1)` block —
`a = 1 + 2⁻⁵²`, `b = 1`, `c = 1 - 2⁻⁵³` — where the naive expression evaluates
to exactly `0.0` and reports a zero eigenvalue for a positive definite block.
Not observed to fire on a real KKT here; it is a latent difference, and the
adapter cross-checks for it on every factorization.

### Near-zero pivots and perturbation

RSLAB has three arms, and their names do not mean what a POUNCE reader expects:

| `ZeroPivotAction` | 1×1 | 2×2 |
| --- | --- | --- |
| `Fail` (default) | `d == 0.0` exactly → `NumericallyRankDeficient` | `\|det\| <= 1e-14·scale²` → same |
| `PerturbToEps { abs_floor }` | `\|d\| < floor` → `sign(d)·floor` | `\|det\| < max(floor², 1e-14·scale²)` → add `lift` to **both** diagonals, then to `det` itself if still short |
| `ForceAccept` | **also perturbs**, at `max(‖A‖_max, 1)·ε` | same |

`scale` is the largest block-entry magnitude, so the 2×2 test is
scale-invariant; the 1×1 test is not (it is exact zero, or an absolute floor).

`ForceAccept` is the trap. FERAL's arm of that name accepts the tiny pivot at
face value and books it in the `zero` bucket, preserving the inertia's meaning;
RSLAB's lifts it. `tests/inertia.rs::a_perturbing_policy_reports_the_perturbed_inertia_and_flags_it`
measures the consequence: `diag(3, -2, 0)` reports `(2, 1, 0)`, not `(1, 1, 1)`.
The zero eigenvalue comes back positive.

Worse for the inertia contract, the 2×2 lift adds to **both** diagonals and, as
a last resort, to the determinant. Both operations move eigenvalues across zero,
so a perturbed factorization's inertia is that of `A + E` and need not equal
`A`'s at all. FERAL's `n_tiny` is explicitly documented as *not* affecting
inertia because its 1×1 perturbation preserves the sign; RSLAB's carries no such
guarantee. The adapter therefore treats any `n_perturbed > 0` as disqualifying.

`n_perturbed` is reported (`LdltNumeric::n_perturbed`,
`NumericReport::perturbed`), so "how many, and did it happen" is answerable.
"By how much" is not: the floor is an input, and the realised perturbation per
pivot is not recorded.

---

## 3. Compatibility assessment

Eight mismatches. Six are mechanical and live in the adapter; two are not.

| # | Mismatch | Resolution |
| --- | --- | --- |
| 1 | **No delayed pivoting** | *Not resolvable in an adapter.* Static pivoting is the only workaround, and it changes the factor from a direct solve to a preconditioner. |
| 2 | **No `min_pivot_magnitude`** | `inertia::scan_d` recovers `(min\|λ\|, max\|λ\|)` from `D` in one O(n) pass, 2×2 blocks by the cancellation-free `\|det\|/\|λ\|_max`. Forced the adapter onto RSLAB's low-level entry points. |
| 3 | **Multi-RHS layout** | RSLAB's `solve_ldlt_many` reads and writes **row-major** `n × nrhs`; POUNCE packs columns. The adapter transposes, fused with the equilibration. Silent for every `nrhs > 1` — the transposed answer has the right shape and no dimension error fires. Caught by `packed_multi_rhs_matches_one_at_a_time`, not by inspection. |
| 4 | **Equilibration is private** | `LdltSolver` equilibrates and the low-level path does not; `equilibrate_with` is `pub(crate)`. The adapter reproduces the one-pass inf-norm step (`scaling.rs`, nine lines), pinned end-to-end against `LdltSolver::factor` by `the_adapters_equilibration_matches_rslabs`. |
| 5 | **`ForceAccept` is not FERAL's** | The adapter does not re-export `ZeroPivotAction`. `PivotPolicy` is adapter-owned, and its static-pivot arm computes the floor per factorization from the equilibrated matrix, which is RSLAB's own recommended recipe and not something a caller can supply at construction time. |
| 6 | **`nnz(L)` means something different** | RSLAB drops numerically-zero entries when it materializes `L`; FERAL keeps the structural slot. Reporting RSLAB's stored-nonzero count gave 330 against FERAL's 5721 on `airport`'s first system, a 17× "win" that is pure accounting — and one that evaporates by the last factorization of the same solve, pattern unchanged (§4). The adapter reports the structural size and exposes the exact-zero count separately. |
| 7 | **No quality ladder** | `α` is a compile-time constant with no `SolverSettings` route to it. `increase_quality()` returns `false` — "already at maximum", which terminates the caller's retry loop rather than spinning. |
| 8 | **Naive 2×2 determinant** | *Not resolvable in an adapter*, since RSLAB computes the inertia internally. The adapter recounts from the same `D` with the fused determinant and marks a disagreement unreliable — it does not pick a winner. |

Things that match cleanly and needed no work: the triplet→CSC conversion
(same 0-based lower-triangle CSC, same duplicate-summing, same sorted rows as
FERAL), analyse-once/factor-many, the `Inertia` triple's shape, and
`NumericallyRankDeficient → Singular`.

**Diagnostics coverage** against the list POUNCE's `LinearSolverSummary` carries
and the prompt asked for:

| | FERAL | RSLAB (via the adapter) |
| --- | --- | --- |
| inertia | yes | yes |
| 1×1 / 2×2 pivot counts | 2×2 not exposed through `pounce-feral` | yes (`two_by_two`) |
| perturbed pivots | `n_tiny` | yes (`n_perturbed`) |
| min / max \|pivot\| | yes | computed by the adapter |
| factor nnz, fill ratio | yes | yes (structural) |
| symbolic / numeric / solve time | not split | yes (`PhaseTimings`) |
| residual or condition estimate | neither | neither |

No report schema changed. `LinearSolverSummary` is populated exactly as
`pounce-feral` populates it, and the same `linear_solve` tracing-span fields are
recorded, so an existing subscriber reads an RSLAB solve without modification.

---

## 4. Results

### Real KKT systems — `rslab_kkt_replay`

Captured from six models through
`IpoptApplication::set_linear_backend_factory`, so each carries the
negative-eigenvalue count the IPM expected at that iteration. A dense
cyclic-Jacobi eigensolve supplies the outside number, because two backends
agreeing tells you they agree, not that either is right.

| model | systems | FERAL | RSLAB exact | RSLAB static-pivot | sp inertia vs FERAL |
| --- | --- | --- | --- | --- | --- |
| eigena2 | 9 | 9 | **0** | 9 | 7 agree / 2 |
| eigenb2 | 7 | 7 | **0** | 7 | 5 / 2 |
| deb7 | 30 | 30 | **0** | 30 | 0 / 30 |
| convex_qp_qscfxm1 | 5 | 5 | **0** | 5 | 4 / 1 |
| airport | 4 | 4 | 4 | 4 | 4 / 0 |
| lp_degen2 | 35 | 35 | 9 | 35 | 6 / 29 |
| **total** | **90** | **90** | **13** | **90** | **26 / 64** |

On `eigena2`'s first system the oracle reports inertia `(110, 55, 0)` with
`min|λ| = 1.0` and `κ = 2.56` — full rank, about as well conditioned as a KKT
gets. FERAL solves it to `residual_ratio 7.1e-17`. RSLAB returns
`NumericallyRankDeficient` for every ordering (`Auto` / `Amd` / `MetisND`), both
kernels, every panel width, every `nemin`, and through its own `LdltSolver`
handle.

On the systems **both** factored:

| | airport (4 paired) | lp_degen2 (9 paired) |
| --- | --- | --- |
| inertia agree | 4 / 4 | 6 / 9 |
| smaller `residual_ratio` | FERAL 4 / 4 | FERAL 9 / 9 |
| worst RSLAB/FERAL residual ratio | 1.0e3 | 5.4e9 |
| structural nnz(L) | 23 241 vs 13 140 | 422 005 vs 354 582 |
| factor / refactor / solve ms | 6.5/2.4/0.46 vs 6.1/2.4/0.06 | 119/30/2.8 vs 69/36/1.4 |

**Scaling-neutral control.** FERAL's default equilibration is
`Auto` (MC64 matching or Knight-Ruiz) and RSLAB exposes neither, so a
default-vs-default residual gap is a gap between *solvers*, not kernels.
`run_scaling_neutral_pair` turns equilibration off on both sides: 13/90 becomes
14/90, and FERAL still has the smaller residual on 13 of 14 paired systems. The
gap is not the scaling.

### Whole NLPs — `rslab_nlp_solve`

Each backend drives the interior-point loop, so a differing inertia produces a
differing trajectory — the thing a replay cannot show.

| model | FERAL | RSLAB exact | RSLAB static-pivot |
| --- | --- | --- | --- |
| airport | Succeeded, 15 it | Succeeded, 15 it | Succeeded, 15 it |
| wyndor_min | Succeeded, 8 it | Succeeded, 8 it | Succeeded, 8 it |
| eigena2 | **RestorationFailed**, 17 it | ErrorInStepComputation, 1 it | **Succeeded, 21 it** |
| eigenb2 | Succeeded, 21 it | ErrorInStepComputation, 0 it | Succeeded, 21 it |
| deb7 | RestorationFailed, 54 it | ErrorInStepComputation, 0 it | ErrorInStepComputation, 1 it |
| convex_qp_qscfxm1 | RestorationFailed, 23 it | RestorationFailed, 23 it | RestorationFailed, 23 it |
| lp_degen2 | RestorationFailed, 165 it | RestorationFailed, 190 it | RestorationFailed, 138 it |

On `airport` and `wyndor_min` all configurations agree to the last printed
digit — RSLAB drives the IPM down an identical trajectory.

The `eigena2` row is the interesting one and it is not a fluke: three runs,
identical. The `rslab-sp+t` arm (identical except that a perturbed
factorization's inertia is acted on rather than routed to `Singular`) also
converges in 21 iterations, so the win is the **factorization**, not the
adapter's reliability gate. That gate is not free either — on `deb7` it turns 51
iterations into a failure at iteration 1 — which is recorded rather than tuned
away.

### Phase timings — `rslab_bench`

Milliseconds, best of 5. `material` is the adapter's own overhead (materialising
`L` in CSC because RSLAB's supernodal `SolvePlan` is `pub(crate)`).

| case | backend | convert | symbolic | numeric | material | first | refactor | solve |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n = 160 400 | FERAL | – | – | – | – | 1038 | 488 | 61.5 |
| n = 160 400 | RSLAB | 15.7 | 318 | 505 | 79.6 | 923 | 543 | 51.2 |
| airport KKT | FERAL | – | – | – | – | 1.81 | 0.61 | 0.090 |
| airport KKT | RSLAB | 0.08 | 0.60 | 0.80 | 0.06 | 1.57 | 0.35 | 0.004 |

Symbolic reuse works: 318 ms on the first factor at `n = 160 400`, **0** on
every refactorization. The cached triplet→CSC slot scatter takes conversion from
20.2 ms to 1.9 ms at 640k nonzeros — an 11× saving, and the same device
`pounce-feral` uses (gh#562).

At scale the two are at rough parity, with the split going the wrong way for an
IPM: RSLAB is 11% faster on the first factorization and **11% slower on the
refactorization**, which is the one paid every iteration.

### Aside: the explicit zeros, and why they are not waste — `kkt_zero_census`

An earlier draft of this note said "POUNCE's KKT carries 88% explicit zeros".
That was measured on `airport`'s **first** factorization and generalised to the
solve, which it does not survive:

| model | slots | factzns | zero @ it0 | zero @ last | zero in *every* factzn |
| --- | --- | --- | --- | --- | --- |
| airport | 2100 | 17 | 88.2% | **2.0%** | 2.0% |
| eigena2 | 1825 | 43 | 88.5% | **9.0%** | 0.0% |
| deb7 | 8063 | 150 | 38.3% | 11.1% | 11.1% |
| convex_qp_qscfxm1 | 4395 | 25 | 24.2% | 7.5% | 7.5% |
| lp_degen2 | 5402 | 251 | 8.2% | **0.0%** | 0.0% |
| wyndor_min | 15 | 10 | 20.0% | 20.0% | 20.0% |

Iteration 0 cannot be representative. The pattern is fixed once by
`initialize_structure` and reused for every later factorization, so it is the
union over the whole solve; at the starting point much of the barrier diagonal
and the Hessian contribution has no value yet, but the slot still has to exist
because the same entry is nonzero five iterations later. **The zeros are the
price of the fixed symbolic pattern** — the thing that makes refactorization
cheap, and the thing §4's timings show working (symbolic 318 ms → 0 ms).

The only column that measures waste is the last one: slots zero in *every*
factorization. That is 0–11%, and exactly zero on two of six models. There is
no POUNCE-side inefficiency here to go and fix.

What it does settle is the accounting question in §3 row 6. RSLAB's
stored-nonzero figure for the same unchanged pattern moves from 88% to 2% of
the slots across one `airport` solve. A factor-size number that moves with the
values is not a fill metric, so the harness reports the structural size — which
is also what FERAL reports and what memory costs.

### Does keeping the zeros cost memory? — factor size at scale

FERAL keeps a structurally-present entry of `L` whether or not it evaluates to
zero; RSLAB drops the numerically-zero ones when it materializes `L`. The
natural worry is that FERAL's choice inflates the factor — in the limit, that a
large KKT ends up stored densely. Measured on the LQ model at three scales, it
does not:

| KKT dim | nnz(A) | nnz(L) | fill | stored L | dense lower triangle | L as % of dense |
| --- | --- | --- | --- | --- | --- | --- |
| 1 502 | 3 003 | 10 495 | 3.49× | ≥0.1 MiB | 9 MiB | 1.39% |
| 15 002 | 30 003 | 104 989 | 3.50× | ≥1.2 MiB | 859 MiB | 0.140% |
| 150 002 | 300 003 | 1 049 989 | 3.50× | ≥12 MiB | **83.8 GiB** | **0.014%** |

Three things to read off it.

**The fill ratio is flat at 3.50× across a 100× range in `n`.** The factor grows
*linearly* with the problem, not quadratically — which is the entire purpose of
the fill-reducing ordering the symbolic phase computes. The percentage-of-dense
column falls by 100× precisely because the numerator is linear and the
denominator is quadratic. (3.50× is this model's number, not a universal one:
the real fixtures run 2.8× on `airport` and 7.8–9.4× on `lp_degen2`. The
constancy across scale is the transferable part, the value is not.)

**FERAL and RSLAB store the same factor.** 1 049 989 against 1 049 988 — one
entry apart in a million, at every scale. The keep-versus-drop difference is not
a memory difference here at all. It showed up on `airport` only because that
measurement was taken at iteration 0, where 88% of `A`'s *values* were zero; on
a matrix whose values are all nonzero the two counts coincide exactly. That is
also a consistency check on the §3 row-6 fix: reporting RSLAB's structural size
makes the two arms agree where they should.

**Keeping them is required by the contract anyway.** The pattern is fixed once
by `initialize_structure` and never revisited, so a slot that is zero at this
iteration and nonzero at the next cannot be dropped without redoing the symbolic
analysis — the one cost the whole design exists to avoid paying per iteration.

Peak RSS for the entire three-scale run, all arms, is **396 MiB**. The stored
factor is 12 MiB of that, so the working set is dominated by the frontal panels,
the IPM's own vectors and the model — not by `L`, and nowhere near the 83.8 GiB
a dense factor would need. That working set is the number this note does *not*
measure separately, and RSLAB's a-priori `MemoryEstimate` (§5 F) is the tool for
it.

### End-to-end at three scales — `rslab_scale_smoke`

Whole POUNCE solves of one LQ optimal-control model at three sizes, only the
backend swapped. All three scales reach **identical status, iteration count and
objective on every arm**, so the wall-clock ratio is a like-for-like comparison
and not a trajectory artefact.

| KKT dim | FERAL | RSLAB exact | RSLAB static-pivot |
| --- | --- | --- | --- |
| 1 502 | 0.017 s | 0.076 s (4.4×) | 0.075 s (4.3×) |
| 15 002 | 0.156 s | 0.974 s (6.3×) | 0.731 s (4.7×) |
| 150 002 | 1.950 s | 9.908 s (5.1×) | 7.457 s (3.8×) |

**FERAL is 3.8–6.3× faster end to end, and the gap is not the adapter.** The
example accumulates `PhaseTimings` across the whole solve precisely so that
claim can be checked rather than assumed. At 150 002 rows:

* `factor_numeric` alone is **9106 ms of RSLAB's 9908 ms — 92% of the solve**;
* everything the adapter owns (conversion 53 ms, equilibration 96 ms,
  `materialize` 51 ms) is **200 ms, about 2%**;
* the CSC reference back-solve that replaces RSLAB's supernodal sweep is 57 ms,
  against FERAL's 1950 ms for the entire solve.

Removing every adapter inefficiency would take 9.9 s to roughly 9.7 s.

Two things this does **not** say. It does not contradict the raw-kernel
benchmark above, where the two were at rough parity at `n = 160 400` — that
matrix is a dense-row grid KKT and this one is banded, which is where FERAL's
ordering and kernel do best. One shape is not a verdict on the other, and
`benchmarks/qp` or Maros-Meszaros would be the corpus to settle it. And
`rslab-sp` being faster than `rslab` here is not a speed-up: static pivoting
lifts small pivots and does less work, and it happens to land on the same
answer on this model.

This model is also banded and benign enough that RSLAB's **exact** mode
survives it, where it fails on 77 of 90 real KKT systems. That is the corpus
trap in miniature, and it is why the real-KKT capture was necessary.

---

## 5. Answers

### A. Can RSLAB serve as a correct POUNCE KKT backend?

**Not in its exact mode — 13 of 90 real KKT systems.** Under static pivoting it
completes every one, and then it is a preconditioner rather than a direct
solver: the inertia is that of `A + E`, and the residual is 10³–10⁹ times
FERAL's. POUNCE's `PdFullSpaceSolver` does run iterative refinement, so that is
survivable, and on 2 of 7 whole NLPs it produced an outcome at least as good as
FERAL's. It is not a correct direct backend today.

### B. Does its inertia definition match what POUNCE requires?

**Yes, for the definition; with two caveats for the arithmetic.**
`rslab::Inertia` is POUNCE's `(positive, negative, zero)`, the negative count is
the strict one POUNCE steers on, and 2×2 blocks are classified by the block's
determinant and trace rather than by its diagonal — verified against
`[[0, 1], [1, 0]]` on a real factorization.

The caveats: the zero bucket uses an exact comparison where FERAL uses
`|d| ≤ ε`, so a pivot of `1e-300` reads positive; and the 2×2 determinant is
evaluated naively, so cancellation can flip a sign. Neither was observed to fire
on a real KKT here, and both are cross-checked on every factorization.

The third caveat is bigger and is not about arithmetic: under static pivoting —
the only mode that works — the inertia describes `A + E`, and RSLAB's 2×2 lift
moves eigenvalues across zero. The count is then not a measurement of anything
POUNCE asked about.

### C. Can POUNCE detect when RSLAB's reported inertia is numerically unreliable?

**Yes, and the detection is stronger than FERAL's.** `InertiaInfo::reliable`
carries three disqualifiers:

1. `perturbed_pivots > 0` — the factor is of `A + E`;
2. `min |λ(D)| < inertia_trust_floor(configured, n)` — the pounce gh#540
   condition, using POUNCE's own floor via POUNCE's own function, not an
   invented RSLAB-specific criterion;
3. RSLAB's count disagrees with a cancellation-free recount of the same `D`.

FERAL's rule is (2) alone. (1) exists because RSLAB's perturbation is not
sign-preserving; (3) because its determinant is not cancellation-free. Both are
O(n) on data already in hand.

The quantity (2) needs is not in RSLAB's API at all; `inertia::scan_d` recovers
it from `D`, and for a 2×2 block it is the smaller eigenvalue magnitude —
`|det| / |λ|_max` with the fused determinant, never a diagonal entry.

### D. Does RSLAB handle difficult POUNCE KKT systems more robustly than FERAL?

**No overall, yes in one specific and interesting place.**

Overall: FERAL 90/90 against RSLAB's 13/90 on captured systems, and 3 of 7
whole NLPs where RSLAB's exact mode fails within one iteration.

The exception is the case the question was probably aimed at. On `eigena2` —
the small-pivot model gh#540 was filed against — RSLAB under static pivoting
converges to `constr_viol` 3.1e-13 where FERAL enters restoration and stops at
1.1e-9. Static pivoting is doing exactly what an IPM wants on a near-singular
KKT: it completes rather than reporting a rank deficiency that sends the outer
loop up a perturbation ladder. That is worth pursuing independently of RSLAB.

On speed, separately: FERAL is 3.8–6.3× faster end to end on a banded KKT at
three scales, with 92% of RSLAB's time in its own numeric factorization and
~2% in the adapter — so that gap is RSLAB's, not the integration's.

#### The control, and where the `eigena2` win actually comes from

RSLAB's defining structural difference from FERAL is the absence of delayed
pivoting. FERAL can be told to drop it too — POUNCE already exposes that as
`feral_static_pivoting` / `POUNCE_FERAL_STATIC_PIVOTING` — so the comparison
has an obvious control that the first pass of this note was missing. Run it
(the `feral-sp` arm of `rslab_nlp_solve`) and the answer is **negative**:

| model | feral | feral-sp | rslab-sp |
| --- | --- | --- | --- |
| eigena2 | RestorationFailed, 17 it | RestorationFailed, 17 it | **Succeeded, 21 it** |
| eigenb2 | Succeeded, 21 it | Succeeded, 21 it | Succeeded, 21 it |
| deb7 | RestorationFailed, 54 it | RestorationFailed, 35 it | failed at 1 it |

On `eigena2`, `feral-sp` is identical to plain `feral` to every printed digit.
Turning off delayed pivoting is **not** what gets RSLAB the win.

What does is the *other* half of the configuration, and the two are easy to
conflate. POUNCE's `feral_static_pivoting` bool maps to
`ZeroPivotAction::ForceAccept`, which in FERAL accepts the tiny pivot **at face
value**, zeroes the `L` column and books it as a zero. RSLAB's static-pivot mode
**lifts** the pivot to an absolute floor and keeps the `L` column live. Those
are different numerics, and §2 already recorded that `ForceAccept` does not mean
the same thing in the two libraries — this is that difference showing up as a
converged solve.

**FERAL has the lift-to-floor mode.** `NumericParams::static_pivot_threshold`,
reachable as `feral::Solver::with_static_pivot_threshold(t)`, enforces an
absolute floor of `t · ‖D·A·D‖∞` on the scaled matrix — the same MA57 recipe,
computed against the same equilibrated matrix, as
[`crate::PivotPolicy::StaticPivotRelative`] in this adapter. `factorize.rs`
notes it "applies regardless of `on_zero_pivot`", so it is independent of the
ForceAccept/Fail trichotomy.

**POUNCE does not wire it.** `pounce_feral::FeralConfig` carries the
`static_pivoting` bool and not the threshold, and `static_pivot_threshold`
appears nowhere in `crates/`. So the configuration that produced the only
result in this note where RSLAB beats FERAL is one that FERAL supports and
POUNCE cannot currently ask for.

#### The prediction, and its falsification

An earlier revision of this note predicted the fix: add
`static_pivot_threshold: Option<f64>` to `FeralConfig` and `eigena2` would
converge, making the win FERAL's and RSLAB merely the instrument that found it.

**That has now been run, and the prediction is wrong.** The field is wired
(`pounce_feral::FeralConfig::static_pivot_threshold`, default `None`) and swept
over the band FERAL's C ABI documents as useful:

| model | feral | spt 1e-12 | spt 1e-10 | spt 1e-8 | feral-sp | rslab-sp |
| --- | --- | --- | --- | --- | --- | --- |
| eigena2 | RestFail 17 | RestFail 17 | RestFail 16 | RestFail 15 | RestFail 17 | **OK 21** |
| eigenb2 | OK 21 | OK 67 | OK 43 | OK 46 | OK 21 | OK 21 |
| deb7 | RestFail 54 | RestFail 54 | RestFail 54 | RestFail 54 | RestFail 35 | fail 1 |
| lp_degen2 | RestFail 165 | RestFail 151 | RestFail 164 | RestFail 97 | **OK 209** | RestFail 138 |
| airport, qscfxm1, wyndor | unchanged across every arm | | | | | |

The floor rescues **nothing**, `eigena2` included, and on `eigenb2` it costs
2–3× the iterations. So RSLAB's `eigena2` win is *not* explained by
lift-to-floor static pivoting, and remains unexplained. The knob is kept because
the capability gap was real — FERAL had a mode POUNCE could not reach — not
because a measurement asked for it, and its doc comment says so.

**The sweep did turn up a result, on a different row.** `feral_static_pivoting`
— the boolean POUNCE *already* exposed, which drops delayed pivoting and
force-accepts — is the only one of eight arms that converges `lp_degen2`: 209
iterations at `constr_viol` 1.98e-10 and `dual_inf` 7.2e-14, against restoration
failure for the default, for all three floors, and for all three RSLAB arms.
Deterministic over three runs. That has nothing to do with RSLAB; it surfaced
because the control arm was in the harness at all.

Two cautions before anyone turns it on by default. It is a **trajectory change**
in the `CLAUDE.md` sense, so it needs the fixture sweep first. And it is not
free elsewhere — on `deb7` it takes 54 iterations to 35 and still fails, and the
earlier RSLAB comparison found it makes no difference on `eigena2` or `eigenb2`.
A per-model win is not a default.

### E. Are its 2×2 Bunch-Kaufman pivots beneficial on those cases?

**Where the factorization completes, yes, and measurably.** On `airport` RSLAB
uses 2×2 blocks, agrees with FERAL about the inertia on 4/4 systems, and
produces a factor 1.8× smaller with a back-solve 5–20× faster (0.004 ms against
0.090 ms). On `lp_degen2`'s paired systems it uses 135 2×2 blocks per
factorization, 1.2× less fill, 1.7× faster factorization.

The benefit does not survive the accuracy comparison: FERAL has the smaller
residual on every paired system, by up to 5.4e9. And the pivots do not rescue
the cases that matter — the models where RSLAB fails, fail because a 2×2
partner was **unavailable within the front**, which is the absence of delayed
pivoting rather than a shortage of 2×2 blocks.

### F. Is there anything in RSLAB that should be ported back into FERAL?

**RSLAB is a fork of FERAL** (see its `NOTICE`), which reframes the question:
most of what is good in it came from FERAL. Two things did not, and both are
now filed:

1. **An a-priori memory and work estimate** (`MemoryEstimate`, `estimate_memory`)
   — computed from the symbolic analysis, so it answers "what will this
   factorization cost" *before* paying for it. `factor_flops` is what POUNCE's
   `predict_factor_overshoot` needs: that guard currently estimates from the
   worst factorization observed so far, i.e. it learns the cost by overrunning
   once. `panel_live_peak_bytes` / `transient_peak_bytes` is the number §4's
   memory table could not report — the frontal working set between the 12 MiB
   stored factor and the 396 MiB process peak. Filed as
   [feral#204](https://github.com/jkitchin/feral/issues/204).
2. **The `Decisions` report** — specifically `ordering_requested` vs
   `ordering_used`. FERAL routes adaptively (`OrderingMethod::Auto`,
   `with_ordering_escalation`) and `FactorStats` names the *scaling* choice via
   `ScalingInfo` but not the ordering one, so a routing change leaves no trace
   in any reported number. That is the same defect POUNCE fixed one layer up by
   adding the engine column to the fixture sweep. Filed as
   [feral#205](https://github.com/jkitchin/feral/issues/205).

**Two candidates were checked and dropped**, which is worth recording because an
earlier draft of this note asserted the first of them:

* *Phase-resolved diagnostics.* This note previously called them "the clearest
  borrow". **That was wrong** — FERAL has `Solver::with_profiling(true)` →
  `profile_report()` / `symbolic_profile_report()`, and its `ProfileReport`
  (per-supernode buckets, prologue sub-phase breakdown, validation warnings) is
  richer than RSLAB's stage report. The real gap was only ever the *a-priori*
  half, which is item 1. What `pounce-feral` reconstructs from outside
  (`PhaseTimings` here) it reconstructs because POUNCE never wires the profiling
  through, not because FERAL lacks it.
* *The explicit `two_by_two: Vec<bool>`.* FERAL infers block starts from
  `d_subdiag[k] != 0.0` at each consumer site. That convention is sound — a 2×2
  block's off-diagonal is nonzero by construction — and FERAL already exposes
  the derived quantities an external caller would otherwise walk `D` for
  (`inertia()`, `min_pivot_magnitude()`), so nobody outside has to reimplement
  the walk. The adapter here only had to because *RSLAB* lacks those accessors.
  Not filed.

The traffic in the other direction is heavier, and is the actionable half of
this note. RSLAB should take from FERAL: **delayed pivoting**, without which it
cannot factor a saddle point reliably; **`det_sym2x2`**, the fused
difference-of-products; **a tolerance on the 1×1 zero test**; and a
**sign-preserving 2×2 perturbation**, so `n_perturbed > 0` does not void the
inertia. The first is a prerequisite for RSLAB being a KKT solver at all.

### G. Is the adapter clean enough for a future GPU/Metal implementation underneath it?

**Yes for the trait surface, with one honest qualification.**

`RslabSolverInterface` touches RSLAB through eight entry points —
`CscMatrix::from_triplets`, `analyze_with`, `factor_numeric`,
`LdltNumeric::{d_diag, d_subdiag, two_by_two, n_perturbed, inertia,
into_factors}`, `solve_ldlt_many` — and everything else it does (index shift,
slot-map scatter, equilibration, RHS transpose, inertia scan, status mapping) is
backend-agnostic arithmetic over plain slices. A different implementation behind
`analyze_with` / `factor_numeric` / `solve_ldlt_many` needs no change above it,
and POUNCE's optimizer is untouched either way — the whole assessment ran
without a line changing in `pounce-algorithm`.

The qualification is that two of those eight exist only because the ergonomic
API was insufficient. `into_factors` materialises `L` in CSC on every
factorization (79 ms of 923 ms at `n = 160 400`, and it would be pure waste
against a device-resident factor), and the adapter reads `D` field-by-field
because no accessor reports a pivot magnitude. A GPU backend would want to keep
its factor on the device and answer `min |λ(D)|`, `n_perturbed` and the inertia
itself as *queries*. That is one accessor on the factor handle, not a redesign —
but it should be added before a device implementation, not after.

---

## 6. What was deliberately not done

* **RSLAB is not wired into `pounce-algorithm`**, and cannot be. It has no
  crates.io release under its own name — the registry `rslab` is an unrelated
  slab allocator (`os-module/rslab` 0.2.1) — so it is a git dependency, and
  cargo refuses to publish a crate carrying one even behind a disabled feature.
  An optional `pounce-rslab` dependency would break the crates.io release of
  `pounce-algorithm` and everything above it, which
  `scripts/check-release-consistency.sh` guards. Hence `publish = false` and no
  edge into the publishable graph. Nothing was lost: the backend is a
  `Box<dyn SparseSymLinearSolverInterface>` and
  `IpoptApplication::set_linear_backend_factory` is public API, which is how the
  whole-NLP solves ran.
* **No trajectory sweep.** `scripts/sweep-fixtures.sh` was not run, because
  nothing on POUNCE's own path changed — `pounce-rslab` is not a default-member
  and no crate in the default build depends on it. If RSLAB is ever wired in,
  the sweep is mandatory: a backend change is a trajectory change by definition.
* **`pounce_common::timing::time_linear_system` was not used** for the phase
  split. It aggregates into the convex-path timing scope, and these numbers are
  a standalone benchmark's, not a solve report's.
* **No GPU/Metal work, no modification to RSLAB, no changes to FERAL.**
