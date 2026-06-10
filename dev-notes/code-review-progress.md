# Code-review remediation progress (dev-notes/code-review-2026-06.md)

Worked one issue per `/loop` iteration: verify by running code → write a
regression test that fails pre-fix and passes post-fix → fix → `cargo test`.

## Status

| ID | Title (short) | Status | Notes |
|----|---------------|--------|-------|
| C1 | presolve: Phase-2 redundancy mask misaligned after Phase-0 row drop | **FIXED** | `apply_redundant_verdicts` helper guards on `row_kept_inner`; rollback path rebuilds `linear_rows`. Test `c1_redundancy_mask_realigned_after_phase0_drop`. |
| C2 | presolve: Phase-0 block elimination assumes non-block columns are constants (4 sub-cases) | **FIXED** | Conservative soundness gate rejects any block whose rows reference a free non-block column; `x_running` clamped to fixed value for trivially-fixed vars. Test `c2_gate_rejects_block_with_probe_hidden_free_dependency`. |
| H1 | qp: inertia-shift regularization silently discarded — unbounded QPs reported `Optimal` with δ-dependent garbage | **FIXED** (`solve_equality_only` path) | Re-verify unshifted stationarity `δ·‖x‖∞` after a shifted one-shot solve; report `Unbounded` when it exceeds `1e-3·‖g‖∞` (gradient scale, not `opt_tol`). Test `h1_zero_hessian_linear_objective_is_unbounded`; repointed `inertia_control_shift_succeeds_on_psd_singular_hessian` to a bounded singular case. |
| H2 | sensitivity: pin-row mapping omits `full_g_to_c_block` — silently wrong sensitivities with inequality constraints | **FIXED** | Translate user full-g pin indices through the c/d split before indexing `y_c`; reject pinned inequalities. Fixed `Solver::parametric_step`, `Solver::compute_reduced_hessian`, and the `convenience` (`SensSolve`) path; added `PdSensBacksolver::full_g_to_c_block` accessor. Tests in `cd_split_pin_mapping.rs`. |
| H3 | cli: `.sol`/JSON constraint duals written in internal c/d-split order, unscaled | **FIXED** | `on_converged` hook now reassembles `lambda` via `pack_lambda_for_user` (inverts the c/d split via `c_map`/`d_map` AND unwinds `c_scale`/`d_scale`) instead of concatenating raw `y_c`+`y_d`; manual concatenation kept only as a fallback for non-`OrigIpoptNlp`. Test `lambda_is_in_original_g_order_not_cd_split_order` in `json_report.rs`. |
| H4 | cli: convex LP/QP/SOCP dispatch ignores the `-AMPL` exit-code contract | **FIXED** | Threaded `args.ampl` into `run_convex_qp`/`run_convex_socp`; new `convex_exit_code(ok, ampl)` returns 0 for any non-fatal outcome under `-AMPL` (mirrors NLP path), 1 otherwise. Also dropped the `.sol`-write-failure `exit 2` (log-and-continue like the NLP path). Test `ampl_mode_honors_exit_code_contract_on_infeasible_convex_qp`. |
| H5 | nl: external-function errors detected on the wrong channel — failed evals silently return garbage | **FIXED** | `ExternalLibrary::eval` now decodes both `funcadd` error channels via `decode_external_errmsg`: the **reassigned** `al->Errmsg` pointer (conforming path) and the caller buffer. Previously only `errmsg_buf[0]` was checked, so a library doing `al->Errmsg = "...";` was invisible and the IPM consumed NaN f/∇f/∇²f. Tests `reassigned_errmsg_pointer_is_detected_end_to_end` + `decode_external_errmsg_buffer_and_none_channels`. |
| H6 | qp: `select_blocker` EXPAND branch can panic (`best.expect`) on valid near-degenerate input | **FIXED** | The Harris two-pass admitted nothing in Pass 2 when every candidate's τ-relaxed ratio `r + τ/\|a·p\|` exceeded the artificial `α_min_relaxed = 1.0` init cap by more than `tol` (reachable when `\|a·p\| ≈ feas_tol` inflates `τ/\|a·p\|`). `best` stayed `None` → `expect` panicked. Now falls back to the strict minimum-ratio blocker (always exists since `α_min < 1.0`) and steps exactly `α_min`. Tests `expand_tau_inflation_falls_back_to_strict_min_no_panic` + 2 more in `solver::select_blocker_tests`. |
| H7 | convex: dual-infeasibility certificate validates recession `Gd` componentwise — false `DualInfeasible` on SOC/PSD | **FIXED** | `detect_infeasibility_with` gained a `primal_recession_ok` closure: the dual-inf branch now checks `−Gd ∈ K` (orthant ⇒ componentwise `Gd ≤ 0`; SOC/PSD ⇒ `cone.in_dual_cone(−Gd)`, valid since the composite cone is self-dual) instead of `gd_max ≤ tol`. A direction with `Gd ≤ 0` but `−Gd ∉ K` (e.g. `−Gd=(0.1,0.5) ∉ SOC`) no longer yields a bogus unboundedness proof. Tests `soc_recession_not_in_cone_is_not_dual_infeasible` + 2 in `ipm::detect_infeasibility_tests`. |
| H8 | convex: non-symmetric HSDE driver validates Farkas/recession certs with the orthant test — wrong in both directions for exp/power | **FIXED** | `hsde_nonsym.rs:840` now calls `detect_infeasibility_nscone` (new helper) instead of the componentwise `detect_infeasibility`. Added `NsCone::in_dual_cone`/`in_primal_cone` (per-block dispatch; exp/power use their `BarrierCone` tests). The dual exp cone requires `u < 0`, so componentwise `z ≥ 0` both **rejected** genuine exp Farkas certs (→ `IterationLimit`) and **accepted** all-nonnegative `z ∉ K_exp*` (false `PrimalInfeasible`); both fixed. `detect_infeasibility_with` made `pub(crate)`; the plain componentwise `detect_infeasibility` is now test/docs-only. Tests `exp_farkas_certificate_rejected_componentwise_accepted_cone_aware`, `nonneg_z_not_in_dual_exp_cone_is_false_positive_componentwise`, `nscone_exp_membership_disagrees_with_componentwise`. |
| H9 | convex: `presolve_conic` protects only `SecondOrder` rows — unsound reductions / wrong `Infeasible` for PSD/exp/power rows | **FIXED** | Two layers fixed. (1) `presolve_conic` now protects **every** non-`Nonneg` cone block (`!matches!(spec, ConeSpec::Nonneg(_))`), not just `SecondOrder`. (2) The deeper bug: `build_rows` independently collapsed empty rows — a post-substitution empty cone row with `h<0` returned `Err`→`Infeasible`, and a feasible empty cone row (`h≥0`) was silently dropped, desyncing `reduced_cones`. `build_rows` now takes a `protected` mask and keeps coupled cone rows verbatim (the `0·x ≤ h` slack `s=h` is legal — e.g. `(−1,1,5) ∈ K_exp`); `pivot_divisor` guards empty rows. Tests `exp_cone_empty_row_negative_h_is_not_infeasible`, `exp_cone_activity_redundant_row_not_dropped` in `tests/presolve_conic.rs`. |
| H10 | presolve: postsolve does not zero `z_l`/`z_u` at aux-fixed variables — reported duals violate stationarity | **FIXED** | `finalize_solution` (`lib.rs:1049`) forwarded `sol.z_l`/`sol.z_u` verbatim, but `recover_dropped_multipliers` folds the entire fixed-var stationarity residual into the recovered λ assuming `z_l = z_u = 0` there — double-counting against the IPM's large clamp multipliers. Now copies `z_l`/`z_u` into mutable buffers and zeros each `frame.fixed_vars` entry immediately after that frame's λ is recovered (only on `Ok` recovery; a failed recovery leaves λ=0 so the clamp multiplier is still legitimate). Test `phase0_finalize_zeroes_bound_multipliers_at_fixed_vars` (recording mock inner). |
| H11 | presolve: objective coupling classified from the gradient at a single probe point — a nonlinear objective variable reading zero gradient at the probe is mis-classified `PureEquality` and wrongly eliminated | **FIXED** | `run_auxiliary_phase0` built `obj_support` solely from `objective_gradient_support(grad_f)` — one sample. A variable whose objective gradient happens to vanish at the probe (classic `f=(x−x₀)²` started at `x₀`) reads as objective-free, so its square block is classed `PureEquality` and eliminated even under `Safe`. `PresolveTnlp` now fetches `get_variables_linearity` (`lib.rs:354`) and passes it via a new `Phase0Probe::var_linearity` field; `run_auxiliary_phase0` (`auxiliary.rs:221`) unions every `NonLinear`-tagged variable into `obj_support`, so nonlinear vars are always treated objective-coupled. When the TNLP declines (default), `var_linearity=None` → falls back to the probe gradient (no behavior change; no production TNLP implements the hook). Test `phase0_nonlinear_var_with_zero_probe_grad_blocks_elimination_under_safe`. |

## C1 detail

- **Bug**: `redundant_mask` from `find_redundant_rows` is aligned to the
  *kept* linear rows (`linear_rows`, filtered by `row_kept_inner`), but the
  mapping loop advanced the mask iterator on *every* `Some(linear_row)`,
  including ones Phase 0 already dropped. Every kept linear row after a
  Phase-0-dropped linear row received its predecessor's verdict → a binding
  constraint silently dropped, reinstated at postsolve with λ=0 (wrong answer).
- **Fix**: extracted `apply_redundant_verdicts()` which advances the mask only
  on rows that are both `Some` *and* still `row_kept_inner[i]`. Also made
  `linear_rows` mutable and rebuilt it to the full set inside the Phase-0
  rollback path (lines ~556-583), so the mask stays aligned with the restored
  all-kept mask there too.
- **Test**: `c1_redundancy_mask_realigned_after_phase0_drop` builds a
  3-linear-row map with inner row 0 dropped by Phase 0 and a mask flagging the
  2nd *kept* row; asserts the fixed helper drops inner row 2 (correct) while the
  inlined old loop drops inner row 1 (the documented bug). Deterministic, no FFI.
- **Verified**: `cargo test -p pounce-presolve` → 202 unit + 1 e2e + 9 doc, all pass.

## C2 detail

- **Bug**: Phase-0 block elimination drops a block's rows from the IPM problem
  but folds any *non-block* column into the RHS at a fixed value
  (`solve_linear_block` auxiliary.rs:551), and the residual check evaluates at
  that same point — so it can never catch a non-block column the IPM is still
  free to move. Four ways a free non-block column slips in: (a) a rejected
  earlier block leaves its columns free; (b) DM can leave a Square row adjacent
  to an Over column; (c) trivially-fixed vars are folded at probe value, not
  their fixed value; (d) a nonlinear row's derivative that is zero *at the
  probe* is dropped from incidence, hiding a real dependency. All four yield a
  feasible-looking presolve and a final solution silently violating the dropped
  equality. Opt-in (`presolve_auxiliary`), so not catastrophic today.
- **Fix**: conservative soundness gate (auxiliary.rs, before block solve) —
  scan each block row's **raw Jacobian sparsity** (not incidence, which drops
  probe-zero entries — covers (d)); if any non-block column is neither
  trivially fixed (`x_l==x_u`) nor pinned by an earlier accepted block
  (`fixed_mask`, updated on accept — covers (a)/(b)), reject the block as
  `NonBlockColumnFree`. Separately, clamp `x_running` to the fixed value for
  trivially-fixed vars at init — covers (c).
- **Test**: `c2_gate_rejects_block_with_probe_hidden_free_dependency` builds the
  (d) case (`x0 + x1^2 = 5`, ∂/∂x1=0 at probe x1=0) so incidence forms a clean
  square block {row0,x0} while x1 is a hidden free dependency; asserts the gate
  rejects (`blocks_eliminated==0`, reason `NonBlockColumnFree`).
- **Verified the bug by running code**: with the gate stubbed to `if false`,
  the same test eliminates the block (`blocks_eliminated: 1`) — the silent
  wrong elimination reproduced; restored gate → rejected. Full suite green
  (203 unit + 1 e2e + 9 doc); `pounce-cli`/`pounce-algorithm` build clean with
  the new enum variant.

## H1 detail

- **Bug**: `factorize_with_inertia_control` (solver.rs:104) returns the final
  diagonal shift δ it had to add to factor the KKT, but callers dropped it and
  declared stationarity from the *shifted* system `H+δI`. For `min gᵀx, H=0`
  (or any QP unbounded along a flat/negative-curvature direction) the shift
  regularizes the singular KKT and returns `x = -g/δ` — a δ-dependent garbage
  point — reported as `Optimal`. `QpStatus::Unbounded` was declared in
  `error.rs` but never constructed: unbounded detection did not exist.
- **Fix** (scoped to the one-shot `solve_equality_only` path, solver.rs:586):
  capture δ; the true unshifted primal stationarity residual is exactly `-δx`,
  so after a shifted solve (`δ > 0`) re-verify `δ·‖x‖∞`. A *bounded* singular
  QP regularizes to a min-norm point (residual ≈ `δ_initial·O(1)`, Tikhonov
  noise); an *unbounded* one blows `x` up like `‖g_null‖/δ` (residual ≈
  `‖g_null‖ = O(‖g‖)`) — an ~8-order gap. Threshold is `1e-3·max(‖g‖∞, 1)`
  (gradient scale), **not** `opt_tol`: `opt_tol`=1e-9 < `inertia_shift_initial`
  =1e-8, so comparing to `opt_tol` would false-positive every bounded shifted
  solve. On trip, return `QpStatus::Unbounded` with `obj = -∞`.
- **Scope note**: the other six shift call sites (238/384/441/682/943/1569)
  share the root cause but are iterative paths where a *transient* shift on one
  inner iteration is normal and must not abort — re-verifying there needs the
  shift to persist to convergence, so those are deliberately left for a
  follow-up. H1's concrete reproducer (`min gᵀx, H=0`) routes through
  `solve_equality_only`, which is fixed.
- **Test**: `h1_zero_hessian_linear_objective_is_unbounded` (`H=0`, `g=(1,-2)`,
  no constraints, infinite bounds) asserts `status == Unbounded`. Also
  repointed the pre-existing `inertia_control_shift_succeeds_on_psd_singular_hessian`
  from `g=(-1,-2)` (which is *genuinely unbounded* and was wrongly asserting
  `Optimal` — it encoded the bug) to `g=(0,-2)` (bounded singular: g has no
  component along the null direction), which still exercises the shift
  mechanism and correctly stays `Optimal`.
- **Verified the bug by running code**: neutralizing the new guard
  (`if false && delta > 0.0`) makes `h1_…` report `Optimal` with
  `x = [-1e8, 2e8]` (the δ-dependent clamp point) — the bug reproduced;
  restored → `Unbounded`. Full `pounce-qp` suite green (71 unit + tests).

## H2 detail

- **Bug**: the pin-constraint → KKT-row mapping computed the flat row of a
  pinned equality as `n_x + n_s + user_g_index`, but the `y_c` multiplier
  block holds **equality rows only**. With any inequality preceding the pinned
  equality in `g(x)`, the inequality lands in the `d` block and shifts every
  later equality's `y_c` position down — so the raw user index selects the
  wrong constraint's row (or a `y_d`/slack row) and `parametric_step` /
  `compute_reduced_hessian` return plausible-but-wrong numbers with no error.
  Three sites: `Solver::parametric_step` (solver.rs:316), `Solver::compute_reduced_hessian`
  (solver.rs:357), and the `convenience`/`SensSolve` closure (convenience.rs:285).
  The CLI driver (`pounce-cli/src/sens.rs`) already did it right via
  `full_g_to_c_block` — duplicated logic that had diverged. Existing tests
  passed only because every fixture was equality-only (identity c-map).
- **Fix**: route all three sites through the c/d-split map. Added
  `PdSensBacksolver::full_g_to_c_block` (delegates to the held NLP) and a
  `pin_rows_for` helper in solver.rs; convenience.rs translates inline against
  its `nlp` handle. A pinned inequality (no `y_c` row) is now rejected with an
  error instead of silently pinning a `d`/slack row.
- **Test** (`tests/cd_split_pin_mapping.rs`): a fixture with one inactive
  leading inequality then three equalities (`min x0²` s.t. `x0+x1+x2≤1000`,
  `x0=x1+x2`, `x1=p1`, `x2=p2`). Pinning the x1-fixing equality must move x1
  and x0 but not x2 (`dx=[Δ,Δ,0]`); the pre-fix bug pins the x2-fixing
  equality instead. Plus two inequality-rejection tests (parametric_step and
  reduced_hessian).
- **Verified the bug by running code**: pre-fix, the new test reported
  `dx=[0.1, 0, …]` (x1 unmoved — wrong row pinned) and pinning the inequality
  returned `Ok([0.1, 0, 0])` silently; post-fix → `dx=[0.1,0.1,0]` and the
  inequality is rejected. Full `pounce-sensitivity` suite green (43 + 6 + 3 + …
  across test bins); `pounce-cli` builds clean.

## H3 detail

- **Bug**: the `on_converged` hook (`pounce-cli/src/main.rs:602-624`) built the
  captured `lambda` as the raw internal multipliers — all `y_c` (equalities)
  expanded, then all `y_d` (inequalities) expanded. But `OrigIpoptNlp` splits
  the user's `g(x)` into c (equality) and d (inequality) blocks *interleaved by
  original `.nl` g-index* (`c_map`/`d_map`), and the canonical
  `pack_lambda_for_user` both inverts that permutation **and** unwinds the
  `c_scale`/`d_scale` scaling. The hook did neither, so on any `.nl` with
  interleaved eq/ineq rows the JSON/`.sol` dual block was permuted (AMPL/Pyomo
  read it positionally → each constraint gets the wrong dual), and off by scale
  factors whenever default `gradient-based` scaling fires. The correct backfill
  at main.rs:934-938 only ran when the nominal capture was empty (active-set
  route), so the NLP path always took the buggy branch.
- **Fix**: reassemble via `nlp.borrow().pack_lambda_for_user(&*curr.y_c,
  &*curr.y_d)`; keep the raw `y_c`-then-`y_d` concatenation only as a fallback
  for a non-`OrigIpoptNlp` whose trait default returns an empty vector.
- **Test** (`json_report.rs::lambda_is_in_original_g_order_not_cd_split_order`):
  pyomo-generated `dual_order.nl` interleaves `g0: x ≤ 2` (active inequality,
  dual ≈ 2) then `g1: y == 1` (equality, dual ≈ 58). Correct g-order is
  `lambda = [≈2, ≈58]`; the pre-fix concatenation gives `[≈58, ≈2]`. Magnitudes
  an order apart so the swap is unambiguous regardless of sign convention. Runs
  the binary with `solver_selection=nlp` to force the general filter-IPM path.
- **Verified the bug by running code**: pre-fix binary emitted
  `lambda = [58.0, 2.0]` on `dual_order.nl`; post-fix → `[2.0, 58.0]`. Forcing
  the fallback branch (`if true || lambda.is_empty()`) reproduced the failure in
  the test harness (`lambda[0] = 58 expected ≈2`); restored → green. Full
  `pounce-cli` suite green (154 unit + all integration bins).

## H4 detail

- **Bug**: `run_convex_qp` (`pounce-cli/src/main.rs`) and `run_convex_socp`
  never received `args.ampl` and ended with `if ok { SUCCESS } else { from(1) }`
  — exit 1 on every non-fatal *unsuccessful* outcome (infeasible / unbounded /
  iteration limit). But these paths handle every default-routed (`auto`)
  LP / convex-QP / QCQP `.nl`, and the AMPL solver protocol conveys termination
  through the `.sol`'s `solve_result_num`: a non-zero process exit makes Pyomo /
  the ASL interface raise `ApplicationError` and never read the `.sol`. The NLP
  path already documents and implements this (`_ if args.ampl => SUCCESS`,
  main.rs:1116). So `pounce model.nl -AMPL` on an infeasible LP broke the Pyomo
  integration. Secondary inconsistency: a failed `.sol` write exited 2 on the
  convex paths but only logged-and-continued on the NLP path.
- **Fix**: thread `args.ampl` into both functions; extract
  `convex_exit_code(ok, ampl) -> ExitCode` returning `SUCCESS` when `ok || ampl`
  (mirrors the NLP contract) and `1` otherwise. Dropped the two
  `.sol`-write-failure `return ExitCode::from(2)` early-returns in favor of
  log-and-continue, matching the NLP path so the exit code uniformly follows the
  solve outcome.
- **Test** (`qp_dispatch_end_to_end.rs::ampl_mode_honors_exit_code_contract_on_infeasible_convex_qp`):
  runs the infeasible-QP fixture both ways — `-AMPL --sol-output` must exit 0
  with the verdict (`solve_result_num` 200) written to the `.sol`; plain
  `--no-sol` must still exit non-zero. The existing
  `infeasible_qp_reports_infeasible` (non-AMPL, exit non-zero) is unchanged.
- **Verified the bug by running code**: pre-fix binary exited 1 on
  `infeasible_qp.nl -AMPL` (with the `.sol` written); post-fix → exit 0, and
  non-AMPL stays exit 1 / feasible `-AMPL` exits 0. Neutralizing the `|| ampl`
  guard reproduced the test failure (`right: Some(0)`); restored → green. Full
  `pounce-cli` suite green (154 unit + integration; qp_dispatch 16 tests).

## H5 detail

- **Bug**: the AMPL `funcadd` ABI lets an external library report an error two
  ways. The conforming path is to **reassign** `arglist.errmsg` to the
  library's own string (`al->Errmsg = "T out of range";`); the alternative is
  to write into a caller-provided buffer. `ExternalLibrary::eval`
  (`pounce-nl/src/nl_external.rs`) pre-pointed `al.errmsg` at a zeroed 1024-byte
  buffer and only checked `errmsg_buf[0] != 0` afterward. A library that
  reassigns the pointer (the standard behavior — e.g. IDAES Helmholtz on
  out-of-domain `(h,p)`) leaves the buffer untouched, so the error was
  invisible: `eval` returned `Ok` with the function's NaN/garbage value. This
  defeated the NaN-poisoning design in `nl_tape.rs::ext_eval_or_nan` (written so
  the line search backs off on out-of-domain evals); the IPM silently consumed
  wrong f/∇f/∇²f.
- **Fix**: remember the buffer's address, and after the call decode via a new
  `decode_external_errmsg(errmsg_field, orig_buf_ptr, buf_first)`: if the field
  no longer equals our buffer (and is non-null) the library reassigned it →
  read from the new pointer; otherwise fall back to the buffer when its first
  byte is non-zero; else no error.
- **Test**: `reassigned_errmsg_pointer_is_detected_end_to_end` builds the real
  `Arglist` and invokes a conforming Rust `extern "C"` `rfunc` that reassigns
  `al->Errmsg` to a static string and returns NaN — exercising the real
  function-pointer call and the real post-call decode. It asserts the caller
  buffer stays zeroed (so the pre-fix `errmsg_buf[0]` check saw nothing) and
  that the fixed decode surfaces `"T out of range"`.
  `decode_external_errmsg_buffer_and_none_channels` covers the buffer-write
  channel and the no-error / explicit-NULL cases.
- **Verified the bug by running code**: the end-to-end test demonstrates
  channel 1 in-process — after a reassigning call `errmsg_buf[0] == 0`, proving
  the old check was blind to it, while `decode_external_errmsg` returns
  `Some("T out of range")`. Full `pounce-nl` suite green (75 + …); no external
  dylib required (the IDAES-dependent tests still skip when absent).

## H6 detail

- **Bug**: `select_blocker`'s `AntiCyclingChoice::Expand` arm
  (`pounce-qp/src/solver.rs`) runs the GMSW EXPAND Harris two-pass. Pass 1
  initializes `alpha_min_relaxed = 1.0` (a *cap*, not `+∞`) and records
  `min(1.0, minᵢ r_relaxedᵢ)` where `r_relaxed = r + τ/|a·p|`. Pass 2 admits
  candidates with `r_relaxed ≤ alpha_min_relaxed + tol`, then
  `best.expect("non-empty candidates above")` reads the winner. When *every*
  candidate's `r_relaxed > 1.0` the recorded minimum is the artificial `1.0`
  cap that **no real candidate attains**, so Pass 2's admission test
  (`r_relaxed > 1.0 + tol`) rejects all of them → `best = None` → panic.
- **Reachable on valid input**: a candidate with a true blocking ratio `r < 1`
  (so the `alpha_min ≥ 1.0` early-return at the top is *not* taken) but a tiny
  `|a·p| ≈ feas_tol` has `τ/|a·p|` blow `r_relaxed` far above `1`. If all
  candidates are near-degenerate like this, the panic fires. The review doc
  itself notes "Narrow but reachable on near-degenerate data" — confirmed
  **not** a false positive (an earlier note claimed otherwise; that was wrong).
- **Fix**: replace the `best.expect(...)` with a `match`; in the `None` arm,
  fall back to the strict minimum-ratio blocker — scan `candidates` for the
  first with `r ≤ alpha_min` (guaranteed to exist, since `alpha_min < 1.0` past
  the early-return) and step exactly `alpha_min`. This never freezes (α > 0),
  never panics, and never oversteps the first blocking constraint (it does
  **not** floor at the bogus `alpha_min_relaxed = 1.0`, which would jump past
  the blocker).
- **Test**: `solver::select_blocker_tests` (a `#[cfg(test)] mod` *inside*
  `solver` so it can reach the private `select_blocker`/`BlockerTarget`).
  `expand_tau_inflation_falls_back_to_strict_min_no_panic` passes a single
  `(Bound(0,AtLower), r=0.5, |a·p|=1e-9)` with `τ=1e-3` → pre-fix panics at the
  `expect` (verified by reverting the fix: *"panicked at solver.rs:1518:
  non-empty candidates above"*), post-fix returns `(0.5, Some(Bound(0,…)))`.
  Two companions: `expand_fallback_selects_strict_minimum_among_inflated`
  (picks the min-ratio one among several inflated) and
  `expand_normal_case_admits_in_pass_two` (healthy `|a·p|` ⇒ ordinary Pass-2
  admission, no fallback).
- **Verified by running code**: full `pounce-qp` suite green (74 lib + 1 + 5
  integration); the targeted test fails (panics) when the fix is reverted and
  passes with it in place.

## H7 detail

- **Bug**: `detect_infeasibility_with` (`pounce-convex/src/ipm.rs`) validates the
  dual-infeasibility / unboundedness certificate's recession direction `d` with
  `Pd≈0, Ad≈0, cᵀd<0` and `Gd ≤ 0` **componentwise** (`gd_max ≤ ctol·‖x‖∞`).
  For a cone inequality `Gx ⪯_K h`, the correct recession condition is
  `−Gd ∈ K`, which is *stronger* than componentwise for any non-orthant cone.
  The cone-aware entry point `detect_infeasibility_cone` (reached from the
  direct driver `ipm.rs:1397` and the symmetric HSDE driver `hsde.rs:235`) only
  fixed the *primal* (Farkas) certificate's `z ∈ K*` test; the dual branch
  still used the componentwise check. So a direction with `−Gd = (0.1, 0.5)`
  (componentwise OK, but `0.1 < ‖0.5‖` ⇒ **not** in the SOC) was accepted as a
  genuine unboundedness ray, violating the function's documented "a false
  positive is impossible" contract.
- **Fix**: thread a second closure `primal_recession_ok(gd, tol)` through
  `detect_infeasibility_with` (mirroring the existing `dual_cone_ok`). The
  orthant default keeps componentwise (`(Gd)ᵢ ≤ tol`); the cone-aware path
  tests `−Gd ∈ K` via `cone.in_dual_cone(−Gd, tol)` — valid because every cone
  reaching `CompositeCone` is symmetric/self-dual (orthant/SOC/PSD; exp/power
  route to `hsde_nonsym`, which is the separate H8 issue). Updated the
  certificate doc comment from `Gd ≤ 0` to `−Gd ∈ K`.
- **Test**: `ipm::detect_infeasibility_tests` (calls the `pub(crate)` detectors
  directly). `soc_recession_not_in_cone_is_not_dual_infeasible` builds
  `G=[[−0.1],[−0.5]]`, `d=(1)` so `Gd=(−0.1,−0.5)` (componentwise ≤0) but
  `−Gd=(0.1,0.5) ∉ SOC`: asserts the componentwise `detect_infeasibility`
  (wrongly) returns `DualInfeasible` while the fixed `detect_infeasibility_cone`
  returns `None`. Companions `soc_genuine_recession_still_dual_infeasible`
  (`−Gd=(1,0) ∈ SOC` ⇒ still `DualInfeasible`, no false negative) and
  `orthant_unbounded_lp_detected_both_paths` (orthant parity).
- **Verified by running code**: reverting just the cone-aware recession closure
  to componentwise makes `detect_infeasibility_cone` return
  `Some(DualInfeasible)` and the test fails (`left: Some(DualInfeasible), right:
  None`); with the fix it returns `None`. Full `pounce-convex` suite green (100
  lib + integration).
- **Note**: H8 (`hsde_nonsym.rs:840` using the componentwise default for
  exp/power Farkas multipliers) is the *primal*-certificate analogue in the
  non-symmetric driver and is tracked separately.

## H8 detail

- **Bug**: the non-symmetric HSDE driver (`hsde_nonsym.rs:840`, exp/power
  blocks, also carries SOC) called the orthant componentwise
  `detect_infeasibility` to validate its as-`τ→0` infeasibility certificate.
  The dual exponential cone is `K_exp* = cl{ (u,v,w) : −u·e^{v/u} ≤ e·w, u<0 }`
  (`exp.rs:110`) — it **requires `u < 0`**. The componentwise `z ≥ 0` test is
  therefore wrong in *both* directions: it (a) **rejects** every genuine exp
  Farkas multiplier (which has `u<0`), so a primal-infeasible exp problem
  silently degrades to `IterationLimit`; and (b) **accepts** an all-nonnegative
  `z ∉ K_exp*`, emitting a false `PrimalInfeasible`. The recession branch had
  the analogous `Gd ≤ 0` flaw (H7's defect, here on a non-self-dual cone).
- **Fix**: added `NsCone::in_dual_cone` / `in_primal_cone` (per-block dispatch:
  orthant componentwise, SOC self-dual via `SecondOrderCone::in_dual_cone`,
  exp/power via their `BarrierCone` primal/dual tests). Made
  `detect_infeasibility_with` `pub(crate)` and added a `detect_infeasibility_nscone`
  helper that routes the Farkas test through `cone.in_dual_cone(z)` and the
  recession test through `−Gd ∈ K` via `cone.in_primal_cone(−Gd)` (the
  non-symmetric cone is **not** self-dual, so primal ≠ dual here — unlike H7).
  Line 840 now calls it. The plain componentwise `detect_infeasibility` has no
  production caller anymore (both drivers are cone-aware); kept `#[allow(dead_code)]`
  as the documented orthant baseline + test contrast oracle.
- **Test** (`hsde_nonsym::tests`, contrast componentwise vs cone-aware):
  `exp_farkas_certificate_rejected_componentwise_accepted_cone_aware` — a real
  exp Farkas cert `z = interior_reference` (`u<0`, `∈ K_exp*`) with `G=0`,
  `h=(1,0,0)` so `hᵀz=z₀<0`: componentwise `detect_infeasibility` returns
  `None` (misses it), cone-aware returns `PrimalInfeasible`.
  `nonneg_z_not_in_dual_exp_cone_is_false_positive_componentwise` — `z=(1,1,1)`
  (`u>0 ∉ K_exp*`) with `h=(−1,0,0)`: componentwise FALSE-positives
  `PrimalInfeasible`, cone-aware returns `None`.
  `nscone_exp_membership_disagrees_with_componentwise` — unit-checks the new
  `NsCone` membership against the exp cone's `u<0` requirement.
- **Verified by running code**: both contrast tests show the old componentwise
  path (the literal pre-fix line-840 call) returning the wrong status while the
  new cone-aware path returns the correct one. Full `pounce-convex` suite green
  (103 lib + integration); no warnings.

## H9 detail

- **Bug**: `presolve_conic` (`presolve.rs:388`) built its `soc_row` protection
  mask only for `ConeSpec::SecondOrder` blocks. Exp/power/PSD cone rows were
  therefore treated as plain orthant `≤` rows by the reduction catalog, which
  is unsound: a non-orthant cone row is coupled to its block, its `h<0` is
  legal (e.g. `(−1,1,5) ∈ K_exp` since `1·e^{−1}≈0.37 ≤ 5`), and dropping any
  one row of a fixed-layout block (3-row exp/power, `svec` PSD) corrupts the
  layout AND desyncs `reduced_cones`, which assumes non-orthant blocks keep
  full dimension.
- **Two layers**:
  1. `presolve_conic` now marks **every** non-`Nonneg` block:
     `if !matches!(spec, ConeSpec::Nonneg(_))` (was `matches!(.., SecondOrder)`).
     Variable renamed `soc_row` → `protected_row`. This guards the in-pass
     reductions (`is_soc_row` at the empty-row, activity-drop, forcing, and
     bound-tightening sites) for all cone rows.
  2. The masking at step 1 alone was **insufficient** — the post-substitution
     row builder `build_rows` collapsed empty rows independently of the mask:
     an empty cone row with `h<0` returned `Err(())` → `Infeasible`
     (`presolve.rs:1205`), and a feasible empty cone row (`h≥0`) was silently
     `continue`-dropped (desyncing `reduced_cones`). `build_rows` now takes a
     `protected: &[bool]` mask (the ineq call passes `soc_row`, the eq call
     `&[]`) and pushes protected empty rows verbatim — the `0·x ≤ h` row is the
     cone slack `s = h`, not an orthant feasibility check. `pivot_divisor`
     guards `coeffs.first()` so an empty protected row can't panic the
     parallel-row normalization (it's excluded from dedup grouping anyway).
- **Tests** (`tests/presolve_conic.rs`):
  `exp_cone_empty_row_negative_h_is_not_infeasible` — `n=1`, empty `G`,
  `h=(−1,1,5)`, `cones=[Exponential]`: pre-fix returned bogus `Infeasible`;
  post-fix `Reduced` with all 3 rows kept and `reduced_cones==[Exponential]`.
  `exp_cone_activity_redundant_row_not_dropped` — row 0 `−x0 ≤ 10` with
  `x0∈[0,1]` (max-activity `0 ≤ 10`, the orthant rule would drop it): pre-fix
  dropped rows to leave 1; post-fix keeps all 3.
- **Verified by running code**: both tests FAILED pre-fix exactly as predicted
  (test1 panicked on the bogus `Infeasible`; test2 `left:1 right:3`) and PASS
  post-fix. The step-1 mask fix alone left both still failing (`left:1`), which
  is what surfaced the deeper `build_rows` layer. Full `pounce-convex` suite
  green (103 lib + integration); `cargo fmt --check` clean.

## H10 detail

- **Bug**: `PresolveTnlp::finalize_solution` (`lib.rs:1049`) constructed the
  inner `Solution` with `z_l: sol.z_l, z_u: sol.z_u` forwarded **unchanged**.
  Phase 0 fixes block variables by clamping `x_l = x_u = v`, so the IPM emits
  large bound multipliers at those variables. The dropped-row recovery
  `recover_dropped_multipliers` (`reduction_frame.rs:205`) solves
  `∇f − Jᵀλ = 0` at the fixed vars under the documented assumption
  `z_l = z_u = 0` there — so the recovered λ already accounts for the full
  residual. Forwarding the clamp multipliers too double-counts the
  contribution, and the reported KKT point violates the stationarity
  `∇f − Jᵀλ − z_l + z_u = 0`.
- **Fix** (`lib.rs`): copy `sol.z_l`/`sol.z_u` into mutable `z_l_full`/
  `z_u_full`; in the per-frame recovery loop, on a **successful** (`Ok`) λ
  recovery, zero `z_l_full[i] = z_u_full[i] = 0` for every `i` in
  `frame.fixed_vars` (length-guarded). Forward the buffers to the inner
  `finalize_solution`. Zeroing only on `Ok` is deliberate: a failed recovery
  leaves the dropped rows' λ at 0, so the IPM's clamp multiplier is still the
  legitimate carrier of that variable's stationarity and must survive.
- **Test** (`lib.rs` test module): `RecordingTwoVar` — same model as
  `TwoVarSquareEq` (`x+y=3, x−y=1` → fixes `(2,1)`, both rows dropped, frame
  `fixed_vars={0,1}`) but records the `z_l`/`z_u` its `finalize_solution`
  receives. `phase0_finalize_zeroes_bound_multipliers_at_fixed_vars` drives a
  reduced `Solution` with clamp multipliers `z_l=[7,0]`, `z_u=[0,3]` and
  asserts the inner sees `[0,0]`/`[0,0]`.
- **Verified by running code**: pre-fix FAILED (`left:[7.0,0.0]`, the
  multipliers forwarded verbatim); post-fix PASSES. `m_inner = info_inner.m`
  is the **full** row count, so the recovery+zeroing block runs even though
  the reduced problem has 0 rows. Full `pounce-presolve` suite green (204 lib
  + integration); `cargo fmt --check` clean.

## H11 detail

- **Bug**: `run_auxiliary_phase0` (`auxiliary.rs`) derived the objective
  support that drives coupling classification from a single gradient sample:
  `obj_support = objective_gradient_support(probe.grad_f, 1e-12)`. The probe
  `grad_f` is `∇f` evaluated at **one** point (`x_l`/probe). For a variable
  that appears nonlinearly in the objective, a zero entry there does NOT prove
  the variable is objective-free — the canonical `f = (x − x₀)²` evaluated at
  the stationary point `x₀` has `∂f/∂x = 0`. `classify_block`
  (`coupling.rs`) then sees the block as touching no objective variable,
  returns `PureEquality`, and `run_auxiliary_phase0` eliminates it even under
  the `Safe` policy — silently changing the objective (the eliminated var is
  pinned to its equality-implied value, dropping the `(x−x₀)²` curvature).
- **Fix**: surface per-variable linearity from the inner TNLP and treat every
  `NonLinear` variable as objective-coupled regardless of the probe gradient.
  - `PresolveTnlp::run_phase0` (`lib.rs:354`) calls
    `get_variables_linearity(&mut var_linearity)` (default-`NonLinear` buffer)
    and records whether the TNLP supplied tags (`have_var_linearity`).
  - New field `Phase0Probe::var_linearity: Option<&[Linearity]>`
    (`auxiliary.rs:64`); set to `Some(&var_linearity)` only when
    `have_var_linearity` (`lib.rs:484`), else `None`.
  - `run_auxiliary_phase0` (`auxiliary.rs:221`) unions every `NonLinear`
    variable into `obj_support` after the gradient-support seed. `None`
    (TNLP declined) falls back to the probe gradient alone — the prior
    behavior.
- **Soundness**: a `Linear` variable with zero probe gradient is genuinely
  objective-free (linear ⇒ constant gradient ⇒ zero everywhere) — safe to
  eliminate. A `NonLinear` variable is the only ambiguous case, and it is now
  always protected. The default `get_variables_linearity` returns `false`
  (no tags), and no production TNLP overrides it, so the path is dormant —
  zero regression risk on real solves; it engages only when a caller opts in.
- **Test** (`auxiliary.rs` test module):
  `phase0_nonlinear_var_with_zero_probe_grad_blocks_elimination_under_safe`
  builds a 2×2 linear equality block (`x+y=3, x−y=1`) with `grad_f=[0,0]`
  (probe reads no objective coupling) and `var_lin=[NonLinear, Linear]`. A
  control probe with `var_linearity: None` eliminates 1 block (gradient-only
  classification → `PureEquality`); the tagged probe
  (`Phase0Probe { var_linearity: Some(&var_lin), ..base }`) eliminates **0**,
  produces no frame, and reports `class_counts.objective_coupled == 1`.
- **Verified by running code**: pre-fix (augmentation temporarily disabled)
  FAILED (`left:1 right:0` — the nonlinear-tagged block was still
  eliminated); post-fix PASSES. Full `pounce-presolve` suite green (205 lib +
  integration + doctests); `cargo fmt --check` clean; no build warnings.
