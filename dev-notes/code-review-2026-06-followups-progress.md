# Code-review 2026-06 — follow-ups (F1–F8) progress

Tracks the follow-ups raised in `code-review-2026-06-verification.md` (the
re-verification of the L1–L56 fix batch). Each entry: verification by running
code, a fail-first test where constructible, the fix, and the result.

| ID | Title | Sev | Status |
|----|-------|-----|--------|
| F1 | H3 duals off by `obj_scale_factor` | High | ✅ fixed |
| F2 | H1 inertia-shift δ + unbounded-QP false positive | Med-High | ✅ fixed |
| F3 | H11 fix dormant (no `get_variables_linearity`) | High | ✅ fixed |
| F4 | L7 watchdog `alpha_primal_max` — reopen | Medium | ✅ fixed |
| F5 | L56 incomplete — session FFI unguarded | Medium | ✅ fixed |
| F6 | H12 no Phase-0 rollback on FBBT infeasibility | Medium | ✅ fixed |
| F7 | L10 MA57 grow paths unguarded | Low | ✅ fixed |
| F8 | M9 zero-fill in pounce-sensitivity `dense_to_vec` | Low | ✅ fixed |

---

## F1 detail — duals reported scaled by `obj_scale_factor` (High)

**Finding.** When gradient-based objective scaling triggers
(`‖∇f(x0)‖∞ > nlp_scaling_max_gradient`, default 100), the solution duals
(`lambda`, and also the bound duals `z_l`/`z_u`) were reported scaled by
`obj_scale_factor` instead of in the user's unscaled-Lagrangian convention
`∇f + λ·∇g + z = 0`.

**Root cause.** `OrigIpoptNlp` carries two parallel dual-lifting families:
- `pack_lambda_for_user` / `pack_z_l_for_user` / `pack_z_u_for_user` — apply
  `c_scale`/`d_scale` but **not** the `1/obj_scale_factor` division. These feed
  the *scaled* `eval_h` and are correct there.
- `finalize_solution_lambda` / `finalize_solution_z_l` / `finalize_solution_z_u`
  — apply scale **and** divide by `obj_scale_factor`. These are the correct
  *final-solution* convention (mirror of upstream
  `IpOrigIpoptNLP::FinalizeSolution`). They existed but had **zero callers**
  (dead code).

The solution hooks called the `pack_*` family:
- `crates/pounce-cli/src/main.rs:643` (`on_converged` → JSON / `.sol`),
- `crates/pounce-algorithm/src/application.rs:2215` (`finalize_via_orig_nlp`,
  the `finalize_solution` TNLP callback used by the Python bindings),
- `application.rs:2371` (`finalize_via_sqp`, the SQP analog),

so every dual came back scaled whenever scaling kicked in. Only `lambda` was
flagged in the review, but `z_l`/`z_u` shared the identical root cause via
`pack_z_*_for_user`.

**Verification by running code.** New fixture `dual_scaled.nl` =
`dual_order.nl` with the y-target moved 30 → 3000:
`min (x-3)^2 + (y-3000)^2  s.t.  x ≤ 2 (active),  y == 1`. At x0=(0,0)
`‖∇f‖∞ = 2·3000 = 6000`, so `obj_scale_factor = 100/6000 = 1/60`. Running the
release binary:

```
# before fix, default scaling:        lambda = [0.0333, 99.97]   (scaled, WRONG)
# nlp_scaling_method=none:            lambda = [2.0,   5998.0]   (true)
# after fix, default scaling:         lambda = [2.0,   5998.0]   (true, FIXED)
```

The pre-fix `[0.0333, 99.97]` is exactly `[2, 5998]/60` — confirming the missing
`obj_scale_factor` division. Regression: `dual_order.nl` (obj_scale=1) is
unchanged at `[2.0, 58.0]`.

**Fix.** Switch all six call sites in `finalize_via_orig_nlp` /
`finalize_via_sqp` (and the lambda site in `main.rs`) from the `pack_*` family
to the `finalize_solution_*` family. This fixes `lambda` (F1's explicit ask)
**and** the latent `z_l`/`z_u` bug consistently, and retires the dead code the
verifier flagged. The `pack_*` methods stay (still used by the `eval_h` path).

**Test.** `crates/pounce-cli/tests/json_report.rs::lambda_is_unscaled_by_obj_scale_factor_under_gradient_scaling`
— solves `dual_scaled.nl` with default scaling ON and asserts the unscaled
`lambda = [≈2, ≈5998]` (decisive guard `|lambda[1]| > 1000`; pre-fix it was
≈99.97). Fail-first demonstrated directly on the pre-fix binary (`[0.033, 99.97]`
fails both `>1000` and `≈5998`).

**Result.** `pounce-cli` full suite green (incl. 7 json_report tests);
`pounce-algorithm` 258 lib tests, `pounce-nlp` 39 lib tests green; fmt + clippy
(correctness/suspicious gate) clean.

---

## F2 detail — inertia-shift δ and unbounded-QP detection (Med-High)

The verifier raised two coupled defects around the QP inertia-control shift δ
in `crates/pounce-qp/src/solver.rs`:

- **N1 false positive** — the equality-only fast path (`solve_equality_only`)
  used a *magnitude heuristic* (`δ·‖x‖∞ > 1e-3·‖g‖∞`) to declare unboundedness.
  It could not distinguish a large-but-finite minimizer in a **curved**
  direction from a genuine blow-up along a **flat** descent ray, so a bounded
  singular QP like `H = diag(1e-6, 0)`, `g = (-1, 0)` (true minimizer
  `x₁ = 1e6`, obj `-5e5`) was wrongly reported `Unbounded`.
- **F2(a) δ discarded** — the general active-set loop (`solve_general`) and the
  opt-in Schur loop (`solve_general_schur`) threw away the δ returned by
  `factorize_with_inertia_control`. An unbounded QP carrying a general
  inequality (which routes off the equality-only path) just took unbounded full
  steps until `MaxIter` — never certifying the recession ray.

**Root cause.** Both defects are the same missing piece: a `δ > 0` solve is
consistent with *both* a bounded QP (the regularizer picks the min-norm point
along a flat, gradient-free direction) and an unbounded one, so the shift alone
proves nothing. Magnitude of `‖x‖` is not a discriminator either — a shallow but
genuinely curved bowl has a large finite minimizer.

**Fix — certified recession ray.** A QP `min ½xᵀHx + gᵀx s.t. Ax = b` is
unbounded below iff there is a direction `d` with `Hd = 0` (zero curvature; for
PSD H ⟺ `dᵀHd = 0`), `Ad = 0` (feasible), and `gᵀd < 0` (descent). New shared
helper `ray_is_unbounded_descent(h, g, dir)` checks the two intrinsic clauses
— **zero curvature** `dᵀHd ≈ 0` relative to `‖H‖`, and **strict descent**
`gᵀd < 0` — leaving feasibility of the ray to the caller:

- `solve_equality_only`: the saddle solve maintains `Ax = b`, so the candidate
  ray `d = x/‖x‖` has `Ad = b/‖x‖ → 0`; verified by an inline `‖Ad‖∞` guard,
  then `δ > 0 && feasible && ray_is_unbounded_descent` ⇒ `Unbounded`.
- `solve_general`: the ratio test having an **empty candidate list** certifies
  `+p` is feasible for every step length (and `p` lies in the active null
  space); gated additionally on `δ > 0` (captured from the factorize call).
- `solve_general_schur`: same empty-candidate certificate, but δ is hidden
  inside `SchurState`, so it relies on the curvature clause alone to reject
  curved (PD-reduced-Hessian) steps — `ray_is_unbounded_descent` returns
  `false` on any `dᵀHd ≉ 0`, so the unconditional check is safe.

Why the curvature clause is robust: as the inertia shift `δ → 0` the flat
descent component of `-g` is amplified by `1/δ`, so `d = x/‖x‖` converges to the
recession ray and `dᵀHd/‖H‖` shrinks like `O((δ/‖H‖)²)`; a curved minimizer
keeps `dᵀHd ≈ ‖H‖` (ratio `O(1)`). The `1e-3·‖H‖` threshold sits in that gap,
and ambiguous near-floor-curvature cases fall on the conservative side
(reported bounded, never falsely unbounded).

**Verification by running code (fail-first).** Three new analytical tests in
`crates/pounce-qp/src/tests/analytical.rs`:

```
n1_bounded_singular_qp_is_not_falsely_unbounded   pre-fix: Unbounded x=[990099,0]  → Optimal (FIXED)
n1_partial_curvature_descent_ray_is_unbounded     pre/post: Unbounded              (guard)
f2_general_active_set_detects_unbounded_ray        pre-fix: MaxIter x=[0,2.0e10]   → Unbounded (FIXED)
                                                   (+ same problem on the Schur path)
```

The N1 case (`Unbounded` pre-fix) and the active-set case (`MaxIter` pre-fix,
`x₂` ran to `2.01e10`) both fail before the fix and pass after.

**Result.** `pounce-qp` full suite green — 83 lib tests + integration
(`mm_published_optima` real Maros-Meszaros QPs confirm bounded problems are not
falsely flagged). fmt + clippy (correctness/suspicious gate) clean.

---

## F3 detail — H11 presolve safeguard was dormant (High)

**Finding.** The L-batch fix H11 added a guard in the Phase-0 presolve
auxiliary-elimination pass (`pounce-presolve/src/auxiliary.rs:270-281`): it
unions every variable upstream tagged `NonLinear` into the objective support,
so a variable that is nonlinear in the objective but merely *zero-gradient at
the single probe point* (the canonical case `f = (x − x0)²` warm-started at
`x0`, where `∇f(x0) = 0`) is not mis-classified objective-free and eliminated.
The guard reads the tags from `TNLP::get_variables_linearity`. But the only
implementation of that trait method was the **default stub**
(`pounce-nlp/src/tnlp.rs:223`, `-> false`, slice untouched), so on every
production path `have_var_linearity` was `false`, `probe.var_linearity` was
`None`, and the H11 union never ran. The safeguard was dead code: its unit
tests (`auxiliary.rs:1002-1065`) pass *tags by hand* and so never exercised the
real (untagged) production path.

**Root cause.** `NlTnlp` (the `.nl`-backed TNLP, the production entry point for
every AMPL/`pounce`-CLI solve) implemented `get_constraints_linearity` but not
`get_variables_linearity`, falling through to the `false` default. The tape
already knows exactly which variables are nonlinear, so the information was
available — just never surfaced.

**Fix.** Implement `get_variables_linearity` on `NlTnlp`
(`crates/pounce-nl/src/nl_reader.rs`, beside `get_constraints_linearity`) with
the upstream **global** semantics: a variable is `NonLinear` iff it appears in
the nonlinear part of the objective or of any constraint, else `Linear`. The
parsed `.nl` already separates each row into a linear coefficient list and a
nonlinear `Expr`, so the nonlinear set is exactly the structural union of the
existing `collect_vars` walk over `obj_nonlinear` and every `con_nonlinear`
row. This honors the documented contract and engages H11 on every solve. (The
alternative the verifier floated — making the *untagged* case conservative in
the presolve — was rejected: it would broadly regress legitimate
auxiliary-elimination for every TNLP that genuinely returns no tags.)

**Verification by running code (fail-first).** New unit test
`crates/pounce-nl/src/nl_reader.rs::variables_linearity_tags_obj_nonlinear_vs_linear_vars`
builds `min (x0 − 1)² + 3·x1` (x0 nonlinear in the objective, x1 only in the
linear part) and asserts `get_variables_linearity` returns `true` with
`[NonLinear, Linear]`. Demonstrated fail-first by temporarily reverting the
body to the stub (`-> false`, slice untouched): the test panics
`get_variables_linearity must report it filled the slice`. Post-fix it passes.

**Result.** `pounce-nl` full suite green (89 lib tests, incl. the new one);
`pounce-presolve` 226 lib tests green (the H11 path now active on the real
no-hand-tags route, no regression). fmt + clippy (correctness/suspicious gate)
clean.

---

## F4 detail — watchdog StopWatchDog retry reused the failed direction's FTB cap (Medium)

**Finding.** The filter line-search watchdog snapshots an iterate + search
direction; after `watchdog_trial_iter_max` (default 3) failed outer iterations
it reverts ("StopWatchDog") to that snapshot and re-runs the alpha-loop on the
saved `delta` with `skip_first = true`. Upstream
`IpBacktrackingLineSearch::FindAcceptableTrialPoint` (stable/3.14) recomputes
`alpha_primal_max` / `alpha_dual_max` from `actual_delta_` — which
`StopWatchDog` has reverted to the snapshot — so the entire body re-runs on the
recovered direction, fraction-to-the-boundary (FTB) caps included. pounce's
`handle_watchdog_failure` (`backtracking.rs`) instead **reused the
`alpha_init` / `alpha_dual` it had been handed** — the caps computed for the
*pre-revert* iterate and the *now-abandoned* failed direction — and fed them to
`run_alpha_loop` over `snap_delta`.

**Root cause.** The retry's first trial length is `cap × alpha_red_factor`. With
a stale cap the first trial is mis-sized for `snap_delta` at the reverted
iterate: if the failed cap is looser than the snapshot's true FTB limit, the
trial overshoots the boundary (negative slack / bound multiplier → non-finite
barrier objective, wasted backtracking trials); if tighter, it needlessly
shortens an otherwise-feasible step. Either way the recovered search starts from
the wrong place.

**Fix.** In the StopWatchDog branch, after `set_curr(snap)` (so the CQ now
reflects the snapshot), recompute both caps from `snap_delta` at the reverted
iterate and stop reusing the handed-in values:

```rust
let tau = data.borrow().curr_tau;
let (alpha_primal_retry, alpha_dual_retry) = {
    let cq_ref = cq.borrow();
    (1.0_f64.min(cq_ref.aff_step_alpha_primal_max(&snap_delta, tau)),
     1.0_f64.min(cq_ref.aff_step_alpha_dual_max(&snap_delta, tau)))
};
```

clamped to the full step `1.0` (the default `alpha_max`, matching the main
path's `alpha_init.min(alpha_primal_max)` at `ipopt_alg.rs:1045`;
`BacktrackingLineSearch` carries no `alpha_max` field). The now-unused
`alpha_init` parameter was removed from `handle_watchdog_failure`'s signature
and its call site in `run_filter_line_search`. Both the primal **and** dual cap
are recomputed: the verification doc named only `alpha_primal_max`, but the dual
cap reuse is the identical staleness bug — applying the failed direction's dual
cap to `snap_delta`'s bound-multiplier components at the reverted iterate can
violate FTB on the `z`'s just as readily.

**Verification by running code (fail-first).** New focused unit test
`crates/pounce-algorithm/src/line_search/backtracking.rs::stop_watchdog_retry_recomputes_ftb_cap_from_snapshot_direction`.
The watchdog/backtracking path had no existing unit harness, so the test builds
one: an `F4MockNlp` (n=1, `x[0] ≥ 0`, `f = x[0]²`, no constraints) and a
`RecordingAcceptor` that records the *first* trial alpha offered and always
accepts. It arms the watchdog (snapshot `x = 2` ⇒ slack 2, `z_L = 0.5`,
`τ = 1`; `snap_delta` `Δx = -4` ⇒ true FTB cap `τ·s/|Δx| = 1·2/4 = 0.5`) with
`watchdog_trial_iter = watchdog_trial_iter_max`, then calls
`handle_watchdog_failure`. It asserts `Outcome::Accepted` and that the recorded
first retry alpha is `0.25` (= recomputed cap `0.5` × `alpha_red_factor` `0.5`).
Fail-first was demonstrated by temporarily hard-coding the call to pass a
*failed-direction* cap of `0.6` in place of `alpha_primal_retry`: the recorded
first alpha became `0.3` (≠ asserted `0.25`) — the test fails on the pre-fix
behavior and passes after. Numbers were chosen so both the pre-fix (`0.3`,
`x = 0.8`) and post-fix (`0.25`) first trials are *feasible*, avoiding the trap
where an infeasible pre-fix trial backtracks and coincidentally lands on `0.25`.

**Result.** `pounce-algorithm` 258 lib tests + the new one green (one unrelated
pre-existing flake, `iter_dump::tests::header_writes_magic_and_version`, races on
a shared `ENV_DUMP_PATH` env var under parallel execution and passes in
isolation — not touched by this change). fmt + clippy (correctness/suspicious
gate) clean.

---

## F5 detail — session-style C ABI was unguarded against panics (Medium)

**Finding.** The L56 fix wrapped `IpoptSolve` / `IpoptSolveWarmStart` (the
classic one-shot C entry points) in `ffi_guard` — a `catch_unwind` that
converts a pounce-internal panic into an `Internal_Error` / `FALSE` return
instead of letting it unwind across the `extern "C"` boundary (UB, in practice
a process abort that takes the embedding application down). But the **session
API** in `crates/pounce-cinterface/src/solver.rs` — `IpoptSolverSolve` (which
drives the identical solver surface) plus `IpoptSolverKktSolve`,
`IpoptSolverKktSolveScaled`, `IpoptSolverParametricStep`,
`IpoptSolverReducedHessian` — and the report writer `IpoptWriteSolveReport`
(`lib.rs`) had **no `catch_unwind` anywhere**, contradicting the L56 commit's
"only entry points that drive the solver" claim. An internal panic in any of
them still aborted the embedding process.

A second, related defect the verifier flagged: after a (now-)caught panic in
`IpoptSolverSolve`, the handle's `last_solve` would still hold the *previous*
solve's stats and `session` its *previous* converged factor, so the post-solve
accessors and a later `IpoptSolverKktSolve` would silently report / back-solve
against stale data alongside the `Internal_Error` return.

**Fix.**
1. **Guards.** Promoted `ffi_guard` to `pub(crate)` and wrapped every session
   entry point and `IpoptWriteSolveReport` in it (fallback: `Internal_Error`
   for the solve, `FALSE` for the Bool-returning ops). `kkt_solve_impl` is
   wrapped once, covering both the scaled and natural-units KKT entry points.
2. **State hygiene.** `IpoptSolverSolve` now invalidates `session` **and**
   `problem.last_solve` to `None` *up front*, before the guarded solve runs.
   Those fields are only repopulated at the *end* of a completed solve, so if
   the guarded body bails early or a caught panic returns `Internal_Error`,
   the failure-consistent state is "no data" (no held factor, no stats) rather
   than a stale factor / stale stats. The same up-front clear was applied to
   the classic `IpoptSolve` in `lib.rs`, which shares the `last_solve` accessor
   surface (`GetIpoptIterCount`, `IpoptWriteSolveReport`, …) and had the
   identical latent staleness.

A panic inside a user-supplied `extern "C"` callback still aborts at *that*
callback's own ABI boundary, before unwinding can reach `ffi_guard` — that is
the caller's responsibility, exactly as in upstream Ipopt's C/C++ original.
`ffi_guard` guards panics originating in *pounce's own* Rust code (solver core,
callback bridge, numerical kernels, the report serializer).

**Verification by running code (fail-first).** Two new tests assert the
state-hygiene half of the fix (the directly observable behavior):
- `solver.rs::stale_session_state_cleared_when_resolve_bails` (session arm) —
  reads the private `session` / `problem.last_solve` fields directly.
- `lib.rs::stale_stats_cleared_when_resolve_bails` (classic arm) — observes
  via the public `GetIpoptIterCount` accessor.

A caught panic cannot be injected deterministically through the public C ABI
(a panic in a user `extern "C"` callback aborts at its own boundary before
reaching `ffi_guard`; the panic-catch mechanism itself is already covered by
`lib.rs::ffi_guard_converts_panic_to_fallback`). Each test therefore drives the
**equivalent control-flow shape**: after a successful solve it corrupts a cached
dimension (`n`/`m` → −1) so the next solve returns `InvalidProblemDefinition`
from *inside the guarded body* without reaching the trailing
`session = Some(..)` / `last_solve = Some(..)` writes — exactly where a caught
panic also bails. Post-fix the handle reports "no data" (`session` None,
`last_solve` None, `GetIpoptIterCount == 0`, `GetKktDim == -1`); pre-fix the
previous solve's factor and stats survived. Fail-first demonstrated by
neutralizing the two up-front clears: both tests then `FAILED` (the stale
`session`/`last_solve` persisted); restoring the clears makes them pass.

**Result.** `pounce-cinterface` full lib suite green — 49 tests (47 prior + 2
new). fmt + clippy (correctness/suspicious gate) clean.

## F6 detail — FBBT infeasibility did not roll back a Phase-0 aux clamp (Medium)

**Finding.** Presolve runs Phase 0 (auxiliary-equality elimination — clamps
variables, drops rows, pushes a `ReductionFrame` onto `reduction_stack`) before
Phase 1b (FBBT — interval propagation over the kept nonlinear constraint DAGs,
`crates/pounce-presolve/src/lib.rs`). Phase 1 already guards against a bad aux
clamp: `if tighten_report.infeasible && !reduction_stack.is_empty()` it **rolls
back Phase 0** and re-runs tightening on the un-clamped box (#53). The FBBT
infeasibility branch (lib.rs ~704) had **no symmetric guard** — on a witness it
restored *only* the pre-FBBT box (`fbbt_x_l_pre`/`fbbt_x_u_pre`), which is the
*aux-clamped* box. So if an aux clamp made a kept **nonlinear** row infeasible,
the clamp stayed in force and the IPM was handed a reduced problem that is
infeasible *because presolve broke it* — the solver then cleanly certifies a
"wrong infeasible" verdict on a problem whose original may be feasible. Presolve
has no channel to certify infeasibility itself, so it must never bake in a
reduction whose infeasibility it cannot attribute to the original.

**Fix.** Mirror the Phase-1 aux rollback in the FBBT branch. When
`report.infeasibility_witness.is_some() && !reduction_stack.is_empty()`:
1. Restore the inner box (`inner_x_l`/`inner_x_u`), re-key every dropped row
   (`row_kept_inner` all `true`), clear `reduction_stack`, and rebuild the full
   `linear_rows` set so Phase 2's redundancy mask stays aligned (C1).
2. Re-run Phase 1 bound tightening on the full (un-filtered) linear rows, with
   the M25 guard (a genuine Phase-1 infeasibility on the un-clamped box restores
   the inner box rather than handing the IPM crossed bounds).
3. Re-run FBBT on the un-clamped, all-kept box. Only an infeasibility that
   **survives** there is genuine — it then falls through to the existing
   "discard FBBT's undefined bounds, proceed on the pre-FBBT box, let the IPM
   certify" handling.

The empty-`reduction_stack` case (no aux active) keeps the original behavior
(restore the pre-FBBT box) — a witness there cannot be a Phase-0 artifact.
`let report` became `let mut report` so the re-run can overwrite it (and the
surfaced `fbbt_report()` reflects the authoritative un-clamped re-run).

**Verification by running code (fail-first).** New test
`lib.rs::fbbt_infeasibility_with_aux_clamp_rolls_back_phase0`. A 3-var/3-row
TNLP: rows 0/1 are a square linear-equality block (`x0+x1=3`, `x0-x1=1` →
clamps `x0=2`, `x1=1`); row 2 is tagged `NonLinear` (FBBT-handled, never a
Phase-1 linear row) and reads `x0+x2=20` with `x2 ∈ [0,1]`, supplied as the
FBBT tape `Add(Var(0), Var(2))` by a paired `ExpressionProvider`. Over the
aux-clamped box `x0 ∈ [2,2]` FBBT sees `x0+x2 ∈ [2,3]`, disjoint from `20`, and
witnesses infeasibility on row 2 — an infeasibility that exists *only* because
the clamp pinned `x0`. The test asserts the scenario fired
(`auxiliary_diagnostics().vars_eliminated == 2`), then that the rollback
restored the dropped rows (`info.m == 3`), that the re-run FBBT found no witness
(`fbbt_report().infeasibility_witness.is_none()`), and that `x0` is no longer
pinned to 2 (FBBT re-tightens it to `[19,20]` on the free box; bounds stay
valid). Fail-first demonstrated by neutralizing the new guard
(`&& !reduction_stack.is_empty()` → `&& false`): the test then `FAILED` with
`got m=1` (Phase 0's dropped rows survived); restoring the guard makes it pass.

**Result.** `pounce-presolve` full lib suite green — 227 tests (226 prior + 1
new). fmt + clippy (correctness/suspicious gate) clean.

## F7 detail — MA57 `info[0] = -3/-4` grow paths were unguarded against i32 overflow (Low)

**Finding.** The L10 fix guarded the *symbolic* workspace sizing
(`ma57_symbolic_sizes`, `ma57_scaled_size`) against i32 overflow, but the
numerical-factor grow loop's two retry paths in
`crates/pounce-hsl/src/ma57.rs` were missed. `grow_fact` (info[0] = −3) and
`grow_ifact` (info[0] = −4) each computed the new workspace size as

```rust
let suggested = (self.info[16] as Number * self.options.pre_alloc).ceil() as Index;
let new_lfact = suggested.max(self.info[16]).max(self.lfact + 1);
let mut newfac: Vec<Number> = vec![0.0; new_lfact as usize];
```

Two unguarded hazards: (1) the `… as Index` float→int cast *saturates* to
`i32::MAX` on a large suggestion (no longer wraps, but an i32::MAX-element
allocation is absurd), and (2) the strictly-growing `self.lfact + 1` bump is a
plain i32 add that **overflows** — when `lfact == i32::MAX` it wraps to
`i32::MIN`, and `new_lfact as usize` then becomes a colossal value
(`-2147483648 as usize` ≈ 9.2e18), so `vec![0.0; new_lfact as usize]` aborts on
a ~74 EB allocation instead of failing cleanly.

**Fix.** Added `ma57_grown_size(base, scale, current)`, a sibling to the
existing `ma57_scaled_size`: it routes the scale through `ma57_scaled_size`
(reusing the saturation/floor guard) and bumps via `current.checked_add(1)`,
returning `None` if either step leaves the 32-bit index range. Both grow paths
now return `Result<(), ESymSolverStatus>` — `Err(FatalError)` when
`ma57_grown_size` is `None` — and the `info[0]` match arms (`-3 =>`/`-4 =>`)
propagate that as a clean `return status`, matching how the symbolic phase
already maps an out-of-range suggestion to `FatalError`.

**Verification by running code.** New unit test
`ma57_grown_size_guards_overflow_and_grows_strictly` (mirrors the existing L10
`ma57_scaled_size_guards_*` test): normal grow `max(ceil(1000·1.05), 2000+1) =
2001`; scaled-base wins over the bump; `scale < 1` never shrinks below the
MA57-suggested base; scaled-size overflow → `None`; and `current == i32::MAX` →
`None` (the `+1`-bump overflow guard).

`pounce-hsl`'s test *binary* cannot link in this environment — the crate has
`links = "coinhsl"` and `COINHSL_DIR` is unset here, so MA57's Fortran symbols
are absent at link time (the pre-existing L10 tests share this constraint and
run in CI where CoinHSL is installed). Verification was therefore done two ways:
`cargo check --tests -p pounce-hsl` confirms the fix and the new test compile
(the `Result` signatures, the match arms, the helper); and the pure helper logic
(`ma57_scaled_size` + `ma57_grown_size`, no FFI) was extracted to a standalone
program and run — all five assertions pass, and it prints the pre-fix wrap
(`current + 1 = i32::MIN = -2147483648`) that the guard now rejects. fmt +
clippy (correctness/suspicious gate) clean.

## F8 detail — `dense_to_vec` silently zero-filled a `CompoundVector` iterate (Low)

**Finding.** The M9 review flagged two copies of a private `dense_to_vec`
helper — `crates/pounce-sensitivity/src/solver.rs` and `convenience.rs` — that
flatten the converged primal iterate `x` into a `Vec<Number>` (populating
`ConvergedState.x` / `SensResult.x` and the KKT-residual extraction). The L16
follow-up had edited only the `Some(DenseVector)` arm (swapping `values()` for
`expanded_values()` to dodge a homogeneous-vector `debug_assert`), but left the
fallback as

```rust
None => vec![0.0; v.dim() as usize],
```

The iterate's primal block is not always a `DenseVector`: a partitioned problem
hands back a `CompoundVector` (the other concrete `pounce_linalg::Vector`
impl). For such an iterate the `downcast_ref::<DenseVector>()` fails and the
function **silently fabricated an all-zero vector of the right length**,
poisoning `SensResult.x` and any KKT residual computed from it with zeros — no
panic, no error, just wrong numbers.

**Fix.** Both `dense_to_vec` copies now also match the `CompoundVector` case
and flatten its components in order — recursively, so a nested compound also
works:

```rust
if let Some(c) = any.downcast_ref::<pounce_linalg::compound_vector::CompoundVector>() {
    let mut out = Vec::with_capacity(v.dim() as usize);
    for i in 0..c.n_comps() {
        out.extend(dense_to_vec(c.comp(i)));
    }
    return out;
}
```

The `Vector` trait has no generic element accessor, so a genuinely-unknown
concrete impl still falls back to zeros — but now behind a
`debug_assert!(false, …)`, so a newly-added `Vector` type is caught by the test
suite in debug builds rather than silently emitting zeros in release.

**Verification by running code.** New unit tests in `solver.rs`:
`dense_to_vec_flattens_compound_vector_components` builds a two-block
`CompoundVector` (`[1.5, -2.0]` ‖ `[7.0, 0.25, -9.0]`) and asserts
`dense_to_vec` returns the concatenated real values (and explicitly *not*
`vec![0.0; 5]`); `dense_to_vec_handles_plain_dense_vector` covers the existing
`DenseVector` path. Fail-first was demonstrated by neutralizing the new
`CompoundVector` arm (`if false /* FAILFIRST */`): the compound test then falls
through to the fallback and fails — the `debug_assert!(false, "… returning
zeros (dim 5)")` fires in the debug test build (in release it would have
returned the zero vector the fix exists to prevent) — and passes again once the
arm is restored. Full `pounce-sensitivity` lib suite (51 tests) green; fmt +
clippy (correctness/suspicious gate) clean.

> **Merge note.** This finding's files (`solver.rs` / `convenience.rs`) overlap
> the area touched by the updated `main` (#129/#130). The fix is applied here
> against the branch as-is; reconcile with main at merge time.
