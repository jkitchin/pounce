# Verification of the code-review fix batch — 2026-06-10

Read-only verification of the 114 fix commits in `dbe2d6a..bc6e856` against the
findings in `code-review-2026-06.md`. Every fix was re-reviewed by a dedicated
verifier: the diff was read, the original failure scenario re-derived against
the new code, the new code scrutinized for introduced bugs, and test coverage
checked. All crate test suites pass at `bc6e856` (verified locally per crate;
PR CI is green: Rust tests, Python jax/torch, both wheel smokes). Where the
verdict depended on external facts, they were checked directly (upstream Ipopt
3.14 sources for L2/L7; the CLI binary was run to probe H3; the publishability
checker was re-run against a simulated git-pin to prove H14).

## Overall result

| Verdict | Count |
|---|---|
| CORRECT / CORRECT-WITH-NITS | 60 findings |
| Dispositions reviewed: justified | M1 (deferred), L2 (not-a-bug), L25 (fixed-by-H4), M10/M24/L26/L46 (documented deferrals) |
| **INCOMPLETE** | **H1, H3, H11 (dormant), M9, L10, L56** |
| **Disposition wrong — reopen** | **L7** |
| INCORRECT | none |
| New bugs introduced | 1 (H1's unboundedness heuristic — see N1) |

Both Critical fixes (C1, C2) are **sound**. C1's mask realignment was re-derived
including the rollback path; C2's hard gate was checked against all four
failure sub-modes (a)–(d), with the raw-COO sparsity scan closing the
probe-zero hole. The highest-regression-risk changes all hold up under deep
review: M23's closed-form `W ⊗ₛ W` algebra was independently re-derived
(including the √2 svec factors) and is pinned entry-for-entry against the old
construction; M16/M17/M18's hot-path caching preserves numerics bit-for-bit by
construction; M35's GIL release reuses the proven SendGuard pattern with
callbacks re-acquiring via `with_gil`; M30's covariance projection math is
exact and basis-invariant; H14's feral pin is a genuine registry dependency
(Cargo.toml + lockfile agree, compiles clean, the new guard demonstrably
catches the original failure and is wired into CI and the publish script).

---

## Follow-ups required (ordered by priority)

### F1. H3 INCOMPLETE — duals still off by `obj_scale_factor` (High)

`pack_lambda_for_user` applies `c_scale`/`d_scale` but not the
`obj_scale_factor` division; the in-tree dead function
`OrigIpoptNlp::finalize_solution_lambda`
(`crates/pounce-nlp/src/orig_ipopt_nlp.rs:1152-1163`, "Mirror of upstream
`IpOrigIpoptNLP::FinalizeSolution`", zero callers) documents the correct
convention. **Reproduced by running the binary**: a fixture with objective
gradient 6000 (> `nlp_scaling_max_gradient` 100 ⇒ obj_scale = 1/60) reports
`lambda = [0.0333, 99.967]` by default vs the true `[2.0, 5998.0]` with
`nlp_scaling_method=none`. The permutation half of H3 is correctly fixed and
test-pinned (`json_report.rs::lambda_is_in_original_g_order_...`), but the
regression fixture's gradient (60) is too small to trigger scaling, so the
remaining bug is invisible to the test. Fix: divide by `obj_scale_factor` in
the hook (`pounce-cli/src/main.rs:643`) and `finalize_via_orig_nlp`
(`application.rs:2189`) — or call the existing dead API — plus a fixture whose
initial gradient exceeds 100. Note the same convention question is in play in
main's #130 (sensitivity unscaled units); align at merge time.

### F2. H1 INCOMPLETE + new false positive (Medium-High)

The fix captures the inertia shift δ only at the `solve_equality_only` site
(`crates/pounce-qp/src/solver.rs:563-611`); the six other
`factorize_with_inertia_control` call sites (incl. line 731, the **general
active-set inner loop**, and 1689, the Schur path) still discard δ — so an
unbounded QP with any inequality row still returns a δ-dependent `Optimal`.
The commit acknowledges deferring this; the finding remains live on the most
common QP path. **N1 (new, Low-Medium):** the new discriminator
`δ·‖x‖∞ > 1e-3·max(‖g‖∞, 1)` false-positives on bounded singular QPs:
`min −x₁ + ½·1e-6·x₁²` with a flat second variable (H = diag(1e-6, 0)) gets
δ=1e-8, x₁ ≈ 1e6, and is reported `Unbounded` with obj = −∞ though the true
minimum is finite. Any bounded singular one-shot QP with solution magnitude
≳ 1e5·max(‖g‖∞,1) trips it.

### F3. H11 fix is dormant on every production path (High residual)

The mechanism (union `NonLinear`-tagged variables into `obj_support`) is
correct, but protection engages only when the inner TNLP implements
`get_variables_linearity` — and only the default stub exists
(`crates/pounce-nlp/src/tnlp.rs:223`); neither the .nl reader nor the Python
bridge supplies tags, and `have_var_linearity=false` falls back to the
single-probe gradient. The exact reviewed failure (`f=(x−x₀)²` warm-started at
`x₀`, `presolve_auxiliary=yes`, Safe policy) still silently changes the
optimum. Either implement the trait method in `NlTnlp` (the tape knows its
nonlinear variables) or make the untagged case conservative.

### F4. L7 disposition is wrong — reopen (Medium)

Commit 5ed70aa declares the watchdog FTB-cap reuse "NOT A BUG / exact match to
upstream", but upstream `IpBacktrackingLineSearch.cpp` (stable/3.14, verified
against source) **recomputes** `alpha_primal_max` from the reverted snapshot
direction at the top of the post-`StopWatchDog` retry (lines ~696/709/718);
pounce's `handle_watchdog_failure` (`backtracking.rs:733-740`) reuses
`alpha_init` — the failed direction's cap at the pre-revert iterate. The
divergence the review described is real (the review's pointer to the stored
field was imprecise — upstream recomputes rather than reusing — but the
substance stands), and the new comment at `backtracking.rs:716-733` bakes the
misreading in. Faithful fix: recompute
`aff_step_alpha_primal_max(snap_delta, tau)` at the reverted iterate.

### F5. L56 INCOMPLETE — session API still unguarded (Medium)

`ffi_guard` wraps `IpoptSolve`/`IpoptSolveWarmStart` (and `ipsolve_` via
delegation), but `IpoptSolverSolve` (`crates/pounce-cinterface/src/solver.rs:128`)
drives the identical solver surface with no `catch_unwind` anywhere in
solver.rs — an internal panic there still aborts the embedding process,
contradicting the commit's "only entry points that drive the solver" claim.
`IpoptSolverKktSolve`/`ParametricStep`/`ReducedHessian` and
`IpoptWriteSolveReport` are also unwrapped. Also undocumented: after a caught
panic, the handle's `last_solve` holds the previous solve's stats, so
`GetIpoptIterCount` etc. silently report stale data alongside `Internal_Error`.

### F6. H12 residual — no Phase-0 rollback on FBBT infeasibility (Medium)

The fix correctly masks dropped rows and restores the pre-FBBT box on an
infeasibility witness, but unlike Phase 1 (#53: filter **+ rollback**), an
FBBT witness with a non-empty reduction stack does not roll back Phase 0. If
an aux clamp made a kept *nonlinear* row infeasible, the IPM now cleanly
certifies infeasibility of a problem presolve itself broke — a wrong
"infeasible" verdict on a feasible original (better than the pre-fix corrupted
bounds, but still wrong).

### F7. L10 INCOMPLETE — MA57 growth paths unguarded (Low)

`ma57_symbolic_sizes`/`ma57_scaled_size` correctly guard the initial sizing,
but the `info[0] = -3/-4` retry paths `grow_fact`/`grow_ifact`
(`crates/pounce-hsl/src/ma57.rs:384-388, 419-421`) still compute
`(info[16] as Number * pre_alloc).ceil() as Index` unguarded — float→int
saturation plus `.max(self.lfact + 1)` can overflow i32 and produce the absurd
allocation the finding asked to convert into a clean `FatalError`.

### F8. M9 residue — zero-fill survives in pounce-sensitivity (Low)

The restoration half is complete, but the commit's "scope correction" rests on
a grep that missed the `match`-form of the pattern: `dense_to_vec`'s
`None => vec![0.0; v.dim()]` arms survive at
`crates/pounce-sensitivity/src/solver.rs:412` and `convenience.rs:442`
(populating `SensResult.x` / KKT residual extraction). L16 even edited the
`Some` arm of these exact functions and left the `None` arm in place.

---

## Merge status warning

The branch does **not** merge cleanly with the updated `main` (which gained
#129 batched-NLP and #130 sensitivity-unscaled-units after the review
baseline). Conflicts in 5 files: `crates/pounce-sensitivity/src/convenience.rs`,
`crates/pounce-sensitivity/src/solver.rs`, `crates/pounce-py/src/problem.rs`,
`python/pounce/_curve_fit.py`, `python/tests/test_problem.py`. The
sensitivity overlap is semantic, not just textual — #130 changes
scaling/units in the same functions the H2/M6 fixes rewrite, and F1's
`obj_scale_factor` question is the same convention #130 addresses. Green CI on
this branch was measured against the old base; re-run after rebase/merge.

---

## New issues introduced by the fix batch

- **N1 (Low-Medium):** H1's unboundedness heuristic false-positives on bounded
  singular one-shot QPs (see F2).
- No other fix was found to introduce a bug. (CI + per-crate suites green;
  the riskiest rewrites — M23 algebra, M16-M18 hot-path caches, M35 GIL,
  L41 sparse SOC blocks — were each verified in depth.)

## Nits worth a small cleanup pass (no behavior bugs)

1. **`_minimize.py` / `_curve_fit.py` drift (recurring theme):** curve_fit's
   KKT-fallback (added by H15) lacks L50's status-5 gate (latent — curve_fit
   has no intermediate callback today); `_curve_fit.py:669, 1199` still probe
   constraints at the origin instead of `p0` (L47 fixed only `_minimize.py`'s
   cited site).
2. **L48:** `disp` sits in `_CONVEX_HONORED` but the convex routes only read
   `tol`/`max_iter`, so `disp=True` is still silently dropped there.
3. **M31:** the jax/torch PSD guard is not skippable (unconditional `eigvalsh`
   per forward for n ≤ 1500) and cannot be forced above 1500; a `check_psd`
   kwarg on the layers would close both.
4. **M34:** the probe cache stores user return objects by reference; a
   buffer-reusing `jac`/`hess` callable can poison entries (degrades toward
   the safe NLP route; `np.array(copy=True)` on store would close it).
5. **H14 residual assumption:** registry feral 0.10.0 is the release
   checkpoint *before* the old git rev's MC64/scaling perf commits; compile
   compatibility verified, behavioral equivalence asserted only by the commit
   message. Direction of drift is conservative, but released binaries may
   iterate slightly differently than what was benchmarked.
6. **M15:** `options_from_env` silently drops tokens without `=` (AMPL's
   `keyword value` spelling); dotted stubs write `my.sol` instead of AMPL's
   `my.model.sol`; `pounce_options` is honored even without `-AMPL`.
7. **L43:** the finding's `auxiliary.rs` assignment sites survive
   (`auxiliary.rs:683, :804`) — verified benign (the summing residual check
   conservatively rejects), but the inconsistency remains.
8. **M1 annotation:** the stale doc-comment at `opt_error.rs:212` ("each
   **unscaled** component…") contradicts the code the annotation documents.
9. **M21:** if flat truncation holds but atom extraction returns empty,
   `is_exact` is kept with zero validated atoms; `num_minimizers` can then
   disagree with the empty list.
10. **H8:** exp/power certificate validators are strict-interior tests —
    boundary certificates are conservatively rejected (false negatives only).
11. **M36:** still no automated writer↔reader parity check between
    `pounce-solve-report` and the studio-core mirror; a fifth variant would
    re-introduce the bug class.
12. **L5:** the new `getrusage`-based test can flake under concurrent
    CPU-burning sibling tests (acknowledged in the commit).
13. **C1/C2 test gaps:** C1 lacks an end-to-end test combining a real Phase-0
    drop with Phase-2 removal; C2 sub-modes (a)–(c) are covered by
    construction, only (d) by a dedicated test. C2 sibling note: probe-zero
    entries are still dropped from `InequalityIncidence` (`incidence.rs:231`)
    — coupling miss yields suboptimality/false infeasibility (never silent
    violation); H11's linearity tags would close it too.

## Dispositions verified

- **M1 (scaled convergence gates → deferred):** justified; every factual claim
  in the annotation checks out, the two-PR plan is concrete.
- **L2 (tiny-step dual test → not a bug):** confirmed against upstream
  `IpBacktrackingLineSearch.cpp` — upstream's test is absolute; the original
  review's premise was wrong.
- **L25 (fixed by H4):** verified accurate — all three `.sol`-write sites
  log-and-continue; no exit-2 write-failure return remains.
- **M10/M24/L26/L46 (documented deferrals):** accurate docs, characterization
  tests anchor future fixes; L26's scaled/unscaled column duplication is
  correctly attributed to `SolveStatistics` carrying only scaled residuals.
- **L7 (not a bug):** **wrong** — see F4.
