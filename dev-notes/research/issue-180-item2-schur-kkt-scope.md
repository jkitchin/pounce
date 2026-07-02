# Issue #180 item 2 — block-triangular / Schur KKT path: scope

> Scoping note for the third and largest part of pounce#180. Items 1 (external
> ordering passthrough) and 3 (per-solve timing) have shipped on
> `claude/github-issue-180-4uijx2`. This note scopes item 2 — adopting the
> Schur-complement / block-triangular KKT solve of Parker, Garcia & Bent
> (arXiv:2602.17968), driven by a caller-supplied block partition from a
> reduced-space / variable-aggregation presolve (Naik, Biegler, Bent & Parker,
> arXiv:2502.13869).

## TL;DR / recommendation

- **FERAL 0.13 already ships the hard linear algebra.** The "F3" feature series
  gives us `symbolic_factorize_with_schur`, `factorize_multifrontal_with_schur`
  → `(SparseFactors, Inertia(A_FF), SchurBlock)`, a dense `SchurBlock`
  factor/solve, and `compute_schur_aware_perm`. So item 2 is mostly
  **orchestration + plumbing in pounce** wrapping an existing primitive — much
  like item 1 — **not** a from-scratch dense/sparse Schur build.
- **It is not pure plumbing, though.** Three real risks gate a production path:
  (a) inertia must be recovered a priori via Sylvester's law (correctness core);
  (b) one **open FERAL dependency** — the full-system block backsolve is not
  demonstrated/exported; (c) a current FERAL **scope limit** (single-supernode
  Schur tail, F3.2b) that not all aggregation problems will satisfy.
- **Recommended path:** a ½–1 day **spike** to resolve the backsolve dependency,
  then a ~2–3 week 3-phase build, contingent on the spike. Do the spike first.

## The method (and how it maps onto FERAL)

The paper: isolate a **nonsingular block-triangular submatrix** of the KKT,
Schur-complement it out, factor **only the diagonal blocks**, block-backsolve,
and recover the full-system inertia **a priori via Sylvester's law of inertia**
(no factorization of the Schur complement's spectrum needed to know it). The
aggregation presolve hands pounce the block structure (the "LM" nonsingular
submatrix).

Mapping to FERAL's `factorize_multifrontal_with_schur(matrix, symbolic, params)`
which returns `(SparseFactors, Inertia, SchurBlock)` and forms
`S = A_SS − A_SF · A_FF⁻¹ · A_FS`:

- **A_FF (eliminated block)** = the large block-triangular nonsingular submatrix
  — cheap to factor multifrontally.
- **`schur_indices` (the Schur tail)** = the small coupling complement left over.
- FERAL returns the **dense** `S`; pounce factors `S` densely (Bunch-Kaufman via
  `SchurBlock::solve_with`, which yields *its* inertia) and adds the two inertias.

The paper's reported win (up to 15× over MA57/MA86 on NN-surrogate-constrained
KKTs) comes from factoring only the diagonal blocks of A_FF and keeping `S`
small; it holds when the Schur block ≪ the eliminated block.

## What already exists (so we don't rebuild it)

**pounce (verified):**
- The `AugSystemSolver` trait (`crates/pounce-algorithm/src/kkt/aug_system_solver.rs:73`)
  is the clean insertion point; `StdAugSystemSolver`
  (`std_aug_system_solver.rs`) assembles the 4×4 KKT into one lower-triangular
  1-based triplet, `dim = n_x + n_s + n_c + n_d`.
- The inertia-correction loop lives in
  `pd_full_space_solver.rs:866–1085` (`num_neg_evals = rhs.y_c.dim() +
  rhs.y_d.dim()`, i.e. `m`), driving `PdPerturbationHandler`
  (`perturbation_handler.rs`) — bumps δ_x/δ_s on `WrongInertia`, δ_c/δ_d on
  `Singular`, refactor-until-success (no fixed cap; bails to restoration when
  δ_x > 1e20).
- FERAL wrapper `FeralSolverInterface` (`crates/pounce-feral/src/lib.rs`) already
  reads inertia via `solver.inertia()` / `num_negative_eigenvalues()`, splits
  `zero > 0 → Singular` vs `neg ≠ expected → WrongInertia` (gh#52 / feral#54).

**FERAL 0.13 (verified in the registry source):**
- `symbolic_factorize_with_schur(matrix, snode_params, schur_indices)`
  (`symbolic/mod.rs:1308`) → `SymbolicFactorization { is_schur_tail: Some(n_schur), … }`.
  Forces `preprocess = None`, `amalgamation = Adjacency`; pins the Schur set to
  perm tail `[n − n_schur, n)`.
- `factorize_multifrontal_with_schur(...)` (`numeric/factorize.rs:1707`) →
  `(SparseFactors, Inertia, SchurBlock)`. **Inertia is of A_FF only** (docs
  `factorize.rs:761–763`).
- `SchurBlock { dim, data }` (`factorize.rs:764`) with `get`, `symv`, `solve` /
  `solve_with` (dense Bunch-Kaufman LDLᵀ — gives S's inertia).
- `compute_schur_aware_perm` (`ordering/schur.rs:46`); `Solver::with_sqd_mode`
  (symmetric-quasi-definite fast path) as a possible companion lever.
- **Neither pounce nor any caller wires this yet** (tree grep: no
  `factorize_multifrontal_with_schur` / `SchurBlock` references in pounce).

**pounce's own Schur code is unrelated** (do not reuse for the NLP inner solve,
but `pounce-sensitivity/src/schur_driver.rs` is a useful reference for block
backsolve mechanics): `pounce-qp/src/schur.rs` (active-set QP SMW updates),
`pounce-sensitivity` (post-optimal sIPOPT), `pounce-convex/src/hsde.rs`
(cone-solver HSDE). None is a condensed/Schur reduction of the NLP IPM KKT.

## Design — where it slots in

1. **`SchurAugSystemSolver`** (new, `crates/pounce-algorithm/src/kkt/`) — a second
   `AugSystemSolver` impl behind the existing trait. Assembles the *same* KKT
   triplet as `Std` (reuse that code), then routes factor+solve through a new
   pounce-feral Schur path. Selected only when a KKT block partition is
   installed; otherwise the solver stack is unchanged.
2. **pounce-feral Schur backend** (new path on/next to `FeralSolverInterface`) —
   drives `symbolic_factorize_with_schur` + `factorize_multifrontal_with_schur`,
   caches `(SparseFactors, SchurBlock)`, and implements factor / backsolve /
   inertia against the same `SparseSymLinearSolverInterface` contract.
3. **Application / Problem API** — `set_kkt_schur_block(indices)` on
   `IpoptApplication` and `Problem`, threaded exactly like item 1's
   `external_ordering` side-channel. `indices` are **KKT-space** (the augmented
   system's `0..dim`), matching the issue's "hook to supply the (rows, cols) of
   the reducible block."

## Inertia via Sylvester's law (the correctness core)

- Required: `(n_pos, n_neg) = (n_x + n_s, n_c + n_d)`.
- Sylvester: `inertia(M) = inertia(A_FF) + inertia(S)`. FERAL gives
  `inertia(A_FF)`; pounce factors `S` (dense BK) for `inertia(S)`; sum, compare
  the negative count to `num_neg_evals`, and return `WrongInertia` / `Singular`
  **exactly as `Std` does** so the existing perturbation loop is untouched.
- **Refactor cadence:** diagonal perturbations δ change *values* not *pattern*, so
  the symbolic Schur factor is computed **once** and numeric-refactored per
  inertia-correction retry (matches FERAL's same-pattern cache); the dense `S`
  is re-formed + re-factored each retry.
- **Subtle risks to verify:**
  - Sylvester requires A_FF factored **without pivots crossing the F/S
    boundary.** FERAL's per-front `NPIV ≤ NASS − NVSCHUR` stopping rule keeps
    delayed pivots inside F — good, but must be confirmed under μ→0.
  - A near-singular A_FF pivot (μ→0) hitting FERAL static pivoting would corrupt
    the inertia read; pounce must route A_FF zero/tiny pivots to `Singular` the
    same way the current `zero > 0` path does. Confirm FERAL surfaces A_FF
    zero-pivots on the Schur path.

## Open dependency — the full-system block backsolve (biggest risk)

FERAL returns the **partial** `SparseFactors` (A_FF eliminated) + dense `S`, but
does **not** export a combined "solve the full permuted system" helper, and its
own F3.4 test only solves the `S` subsystem (`factorize.rs:4767–4780`). The full
solve of `M [x_F; x_S] = [b_F; b_S]` is:

1. `y = A_FF⁻¹ b_F`
2. `b_S' = b_S − A_SF y`
3. `x_S = S⁻¹ b_S'`  (dense, `SchurBlock::solve`)
4. `x_F = A_FF⁻¹ (b_F − A_FS x_S)`

Steps 2 & 4 apply the coupling `A_SF` / `A_FS` — pounce **holds the KKT triplet**,
so it can apply those directly. The unknown is steps 1 & 4: **does
`solve_sparse(partial_factors, v)` compute `A_FF⁻¹ v`?** Unverified. Outcomes:

- **(1) It does** → pounce assembles the backsolve from `solve_sparse` +
  its own coupling apply + `SchurBlock::solve`. Medium effort, no FERAL change.
- **(2) It doesn't** → companion FERAL request ("F3.5": a partial `A_FF⁻¹` solve
  or a one-shot `solve_with_schur(factors, schur_block, rhs)`), then a
  crates.io round-trip — same pattern as item 1.
- **(3) Ask the FERAL author to add the one-shot solve regardless** (cleanest for
  reuse across refinement).

**Phase 0 spike resolves (1) vs (2) definitively before committing to the build.**

## FERAL scope limit to design around (F3.2b)

The Schur tail must be a **single root supernode ending at column `n−1`**;
multi-supernode Schur tails → `InvalidInput` (deferred to FERAL F3.3).
Forest-structured Schur sets rejected; `schur_indices.len() == n` → `InvalidInput`.
Implications:

- The coupling block must be small & connected enough to land in one supernode
  (true for "Schur ≪ eliminated" aggregation, but not guaranteed).
- pounce **must fall back to `StdAugSystemSolver`** when FERAL returns
  `InvalidInput`, rather than failing the solve — the block structure is a
  *hint*, not a contract. This graceful fallback is a first-class requirement.
- Measure in Phase 3 whether real aggregation problems exceed the single-supernode
  limit; if so, a FERAL F3.3 follow-up is needed.

## Interactions that must keep working

- **Iterative refinement** (`PdFullSpaceSolver`): support `resolve` (back-solve
  only) by reusing cached A_FF + S factors; `multi_solve` ideally too. Feasible.
- **Scaling:** FERAL's Schur path forces `preprocess = None`; MC64 symmetric
  scaling permutes/reweights and may fight the fixed Schur tail. Start with
  `Identity` / symmetric scaling on the Schur path, revisit.
- **Debugger `kkt_triplets` / `l_factor`:** the factor representation changes;
  `l_factor` may return `None` initially.
- **Timing/diagnostics:** bump the same `linear_system_factorization` /
  `linear_system_back_solve` counters (item 3) so the perf study is measurable.
- Defer `try_resolve_many_flat` (sensitivity backward fast path).

## Validation

- **Acceptance gate:** identical solutions to `StdAugSystemSolver` (within tol)
  per-solve, and inertia parity at *every* IPM iteration — item 2 changes only
  factorization structure, solutions are identical.
- **Corpus:** NN-surrogate-constrained / aggregation problems (the paper's
  Table-6 set) + the standard HS/CUTEst regression subset.
- **Perf:** factor+solve wall-time vs `Std`, attributed via the item-3 timing
  breakdown; look for the paper's up-to-15× on the surrogate KKTs.

## Effort estimate & phasing

- **Phase 0 — Spike (½–1 day): DONE (green).** See § Phase-0 above.
- **Phase 1 — pounce-feral Schur backend: DONE.** `crates/pounce-feral/src/schur.rs`
  — `FeralSchurSolver` (init/values/factor/backsolve), self-formed-`S` design,
  Sylvester inertia, per-block iterative refinement (`cfg.refine`), graceful
  `FatalError` on a malformed partition. Extracted `configure_solver` so both the
  monolithic and Schur backends configure feral identically. 7 unit tests vs a
  `FeralSolverInterface` oracle: SPD + indefinite `A_FF`, scattered (non-tail)
  Schur sets, multi-RHS, refactor-same-pattern, `WrongInertia` flagging,
  malformed-partition rejection — all machine-precision solution + exact inertia.
- **Phase 2 — `SchurAugSystemSolver` + app/Problem API (3–5 days):** trait impl,
  block-partition hook, `InvalidInput`→`Std` fallback, wire through
  `PdFullSpaceSolver`, inertia-loop integration.
- **Phase 3 — validation + perf (2–4 days):** corpus correctness/inertia parity,
  perf study, docs.
- **Total ≈ 2–3 weeks** *if* Phase 0 hits outcome (1). Outcome (2) adds a FERAL
  request + release round-trip (as with item 1). A FERAL F3.3 (multi-supernode
  Schur tail) follow-up may be needed for larger aggregation problems.

## Phase-0 spike results (RAN — green)

Reproducer: `issue-180-item2-schur-spike.rs` (standalone `feral = "0.13.0"`
crate). It builds KKT-shaped symmetric-indefinite matrices with an explicit
F/S partition, runs the **self-formed Schur design** (factor `A_FF` standalone
via `feral::Solver`, form `S = A_SS − A_FSᵀ A_FF⁻¹ A_FS` with `n_schur`
backsolves, factor `S` densely, block-backsolve), and checks against a
monolithic-factorization oracle. Sweeps `n` in both dimensions.

**Q1 — correctness: PASS.** `max|x_schur − x_oracle|` is `4e-16 … 2e-15`
(machine precision) across every case, including `n = 256 008`.

**Q2 — inertia via Sylvester: PASS.** `inertia(A_FF) + inertia(S)` equals the
oracle's full-system inertia in every case — including an **indefinite**
eliminated block where both blocks carry negatives (`(25,29,0)`,
`(1000,1016,0)`), not just the SPD-`A_FF` case. Haynsworth/Sylvester additivity
holds exactly, which is the correctness core of the whole method.

**Q3 — the dense trap: located precisely, and it is confined to `n_schur`.**
- **Sweep A** (grow the sparse block `n_F` from 1k→256k, hold `n_S = 8`):
  `nnz(L_AFF)` is exactly linear in `n_F` (≈9.5·`n_F`, tridiagonal → banded, no
  fill blow-up); factor / form-S / backsolve times all scale ~linearly; the
  dense-`S` cost is flat. **No n² anywhere in the sparse dimension.**
- **Sweep B** (hold `n_F = 8000`, grow `n_S` from 2→1024): `nnz(L_AFF)` flat;
  dense `S` storage `= n_S²`; the transient `W = A_FF⁻¹A_FS` buffer `= n_F·n_S`
  (64 MB at `n_S=1024`); form-`S` superlinear; factor-`S` ~`n_S³` (0.055→0.262 s
  as `n_S` 512→1024). This is the trap — **entirely a function of `n_schur`.**
- **Crossover vs monolithic factor:** Schur is ~3.3× *faster* than the oracle
  full factor at `n_S = 32` (`n_F=8000`) and ~15× *slower* at `n_S = 1024`. The
  method wins iff `n_schur ≪ n_F`.

**Open FERAL dependency — RESOLVED (unblocked).** The self-formed design uses
only `feral::Solver::factor` / `solve` (proven, stable) and needs **no**
un-exported partial-backsolve, so **item 2 is not blocked at the FERAL layer and
needs no companion FERAL request** for a baseline. As an independent
cross-check, feral's native `factorize_multifrontal_with_schur` produced a dense
`S` that **matched the self-formed `S` to < 1e-9 in every case** — and it did
**not** hit the F3.2b single-supernode limit even at `n_S = 1024` (feral's
HALO-SCHUR amalgamation merges the tail). The native path remains attractive as
a *memory* optimization (it forms `S` without materializing the `O(n_F·n_S)`
`W` buffer) but would reintroduce the partial-backsolve question; **use the
self-formed design as the baseline, revisit native as an optimization.**

**Design guardrails this surfaces for Phase 1/2:**
1. **Gate on `n_schur ≪ n_F`.** Refuse (fall back to `Std`) or warn when the
   supplied Schur block is not small — otherwise the O(n_S³) factor + O(n_F·n_S)
   memory make it lose to the monolithic solve. This threshold is the single
   most important knob.
2. **Stream the `S` formation** column-by-column (solve one `A_FS` column,
   accumulate into `S`, discard) to keep transient memory at O(n_F) instead of
   O(n_F·n_S).
3. Per-IPM-iteration overhead adds `O(n_S · nnz(L_FF)) + O(n_S³)`; fine for
   small `n_S`, and the item-3 timing breakdown already lets us measure it.

**Effort revision:** Phase 0 (this spike) done. The baseline design carries **no
FERAL round-trip** (the previously-largest risk), so the estimate tightens to the
~2–3 week Phases 1–3 with the biggest unknown removed. Sylvester inertia — the
remaining correctness risk — is empirically confirmed (incl. indefinite `A_FF`).

## Recommendation

**Green light.** Item 2 is **much cheaper and lower-risk than it first looked**:
FERAL already ships the Schur primitive, the block backsolve + Sylvester inertia
are proven correct to machine precision (incl. an indefinite eliminated block),
and the design needs **no companion FERAL change** — the self-formed-`S` path
uses only stable exported APIs. The one real hazard, the dense `n_schur²/n_schur³`
cost, is fully understood and confined to the (intended-small) Schur block; guard
it with an `n_schur ≪ n_F` gate + `Std` fallback. Proceed to Phase 1 (the
pounce-feral Schur backend) when ready.
