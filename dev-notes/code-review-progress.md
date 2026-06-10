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
| H12 | presolve: FBBT lacks both the Phase-0 row mask and any infeasibility handling | **FIXED** | Two layers. (1) **Row mask**: `run_fbbt` (`fbbt/orchestrator.rs`) gained a `row_kept: Option<&[bool]>` param; the call site (`lib.rs`) passes `Some(&row_kept_inner)`, so propagation skips any row Phase 0 dropped — over the aux-clamped box a dropped row could fabricate a spurious infeasibility (the #53 hazard Phase 1 already filters). (2) **Infeasibility handling**: `fbbt_report.infeasibility_witness` was never inspected, so FBBT's "undefined and must not be trusted" partially-tightened bounds reached the IPM. The call site now snapshots `x_l`/`x_u` before FBBT and, on a witness, restores them (mirrors the Phase 1 rollback — presolve has no channel to certify infeasibility, so the IPM runs on the pre-FBBT box and certifies it). Tests `dropped_row_is_skipped_and_does_not_flag_infeasible` (orchestrator) + `fbbt_infeasibility_discards_corrupted_bounds` (lib integration). |
| H13 | cinterface: `IpoptSolverSolve` silently discards all user options after the first solve | **FIXED** | The session solve does `mem::replace(&mut info.problem.app, IpoptApplication::new())` to move the configured app into the `RustSolver`, leaving a **blank** app behind that nothing restored (the doc's claimed `app_template` field never existed — grep-confirmed). The second `IpoptSolverSolve` on a handle then read default options — linear solver, tolerances, scaling all lost — and the `feral_config_from_options` snapshot read the blanked app too. Fix: clone the `OptionsList` (it derives `Clone`) before the `mem::replace` and write it back into the fresh blank app via `options_mut()`, so options survive every solve. Stale doc comment on `IpoptSolverInfo::problem` corrected. Test `options_survive_repeated_session_solves` (`solver.rs`): sets `max_iter=7`, creates the session, solves twice, asserts the option persists after each. |
| H14 | release: crates.io automation guaranteed to fail mid-batch (irreversible partial publish), invisible to the consistency guard | **FIXED (guard + pre-flight; root pin out of scope)** | Verified by running `cargo publish -p pounce-feral --dry-run`: hard-fails with "all dependencies must have a version requirement specified … dependency `feral` does not specify a version". The root `feral` dep (`Cargo.toml:89`) is a versionless git pin (`req:"*"`, source `git+…`); it is crate #4 of 19 in publish order, so a `vX.Y.Z` tag uploads 3 crates then hard-fails — an irreversible partial release. The root pin cannot be lifted here (feral must first cut a crates.io release carrying the pinned commits — `feral` is on crates.io only at 0.10.0, which lacks them). Two-layer fix: (1) new `scripts/check_dep_publishability.py` flags any normal/build dep of a publishable crate that is git-sourced or wildcard/versionless; wired as check #4 in `check-release-consistency.sh` (the per-PR/pre-tag guard) so the blocker is no longer invisible. (2) `publish-crates.sh` pre-flight runs the same scan and **aborts before uploading crate 1**, converting the irreversible mid-batch failure into a safe no-op. Tests: `scripts/tests/test_check_dep_publishability.py` (7 synthetic-fixture cases, tree-state-independent). |
| H15 | python: `curve_fit` reports `success=False` for `Solved_To_Acceptable_Level` | **FIXED** | `_solve_fit` (`_curve_fit.py:712`, shared by `curve_fit`, `curve_fit_streaming`, `curve_fit_minima`) gated `success` on `int(info["status"]) == 0`, so an acceptable-level stall (status 1) was reported failed despite a fully populated `popt`/`pcov` — and it lacked the `final_kkt_error` fallback `minimize` already had (gh #119/#123). **Verified by running code**: built the native ext (`maturin develop`) and ran an exp-decay FD fit at `tol=1e-12` → `status=1`, `success=False`, valid `popt≈[2.5,1.31,0.505]`. Fix reuses `_minimize._NLP_SUCCESS_STATUS` (`{0,1}`) plus the finite-KKT-≤-`acceptable_tol` second gate. Post-fix the same fit reports `success=True`. Tests `test_curve_fit_acceptable_level_reports_success` (e2e, asserts status 1 → success) + `test_curve_fit_success_mapping_matches_nlp_minimize`; pre-fix the e2e FAILS (`assert False is True`), post-fix PASSES. Full `test_curve_fit.py` (42) + `test_minima.py`/`test_minimize.py` (30) green. |
| M1 | algorithm: convergence gates use internally *scaled* residuals where upstream uses unscaled | **VERIFIED — DEFERRED** (cross-crate scaling-unwind + core convergence-criteria change; unsafe to ship in an autonomous edit) | **Mechanism confirmed by code inspection**: `check_convergence_with_state` / `current_is_acceptable_with_state` (`conv_check/opt_error.rs:215-222, 301-307`) gate `dual_inf_tol`/`constr_viol_tol`/`compl_inf_tol`/`acceptable_*` on the **scaled** CQ accessors `curr_dual_infeasibility_max` / `curr_primal_infeasibility_max` / `curr_complementarity_max` / `curr_f`; `ipopt_cq.rs` exposes **no** unscaled component accessor (only `unscaled_curr_f`), and `nlp_scaling_method` defaults to **gradient-based** (`upstream_options.rs:361`), so scaling is on by default. Direction (`orig_ipopt_nlp.rs:897-916`): `c_scaled = c_scale·c_orig` with `c_scale ≤ 1`, so the user-space violation = `c_scaled/c_scale ≥ c_scaled` can exceed `constr_viol_tol` by `1/c_scale` while pounce declares `Success` — the reported harm. **Why deferred, not fixed here**: (a) a correct unscaled constraint-violation accessor needs `c_scale`/`d_scale`, which are private to `OrigIpoptNlp` — exposing them means new `IpoptNlp` trait methods on every implementor; (b) unscaled dual-inf and complementarity need the scaling-object unwind pounce explicitly defers (`orig_ipopt_nlp.rs:52-54`) and, because x-scaling is identity but obj-scaling `df` is not, are **not** simple divisions (`∇ₓL_scaled = df·∇f + Jᵀλ` vs unscaled `∇f + Jᵀλ`), so a careless port silently corrupts termination; (c) this is core convergence criteria (high blast radius) deserving reference-validated review. See `## M1 detail` for the scoped two-PR plan and the tests it needs. No code changed. |
| M2 | algorithm: `accept_trial_point` silently nulls `curr` when no trial is staged | **FIXED** | **Mechanism confirmed by code inspection**: `accept_trial_point` (`ipopt_data.rs:203-205`) did `self.curr = self.trial.take()` unconditionally; `ipopt_alg.rs:1121` calls it every iteration. In the documented bookkeeping-only `iterate()` path (no NLP + no `search_dir`, module docs `ipopt_alg.rs:17-22`), step 5 (`ipopt_alg.rs:724-727`) is skipped, so `delta` stays `None`, `have_delta == false` (`ipopt_alg.rs:994`), and no trial is staged — yet accept still ran, nulling `curr`. The next iteration's `IpoptCq::curr_iv` (`ipopt_cq.rs:107-112`) then hits `unreachable!("curr iterate not set")`. **Fix**: guard the promotion — `if let Some(trial) = self.trial.take() { self.curr = Some(trial); }`, preserving `curr` when nothing is staged (normal path unchanged: trial is always `Some` after a line search, so it still promotes and clears `trial`). **Test** (`ipopt_data.rs` tests): `accept_trial_point_preserves_curr_when_no_trial_staged` sets `curr`, leaves `trial` unset, asserts `curr.is_some()` after accept. Pre-fix FAILS (`curr` nulled); post-fix PASSES alongside the existing `accept_trial_point_promotes_trial_to_curr`. Full `pounce-algorithm` suite green (323 passed, 0 failed). |
| M3 | algorithm: `LeastSquareMults` lacks the δ_c/δ_d inertia workaround its sibling has | **FIXED** (trigger not synthetically reproducible — see note) | **Mechanism confirmed by code inspection**: `calculate_y_eq` (`eq_mult/least_square.rs:106-119`) solved the W=0 augmented system with `delta_c = delta_d = 0.0`, while the dual initializer (`init/default.rs:154-194`) solves the *identical* W=0 / structurally-zero (3,3)/(4,4)-block system but perturbs `delta_c = delta_d = 1e-8` specifically because pounce-feral's LDLᵀ mis-reports the inertia of that block (counted 0 negative eigenvalues on `nuffield2_trap` where the true count is `n_c+n_d`, raising `WrongInertia`). With `check_neg = aug_solver.provides_inertia()` (feral → true) and `num_eq = n_c+n_d` passed to `solve` (`least_square.rs:133-135`), the LS solve can spuriously fail; the caller then **silently leaves `y_c=y_d=0`** (`init/default.rs:388-390`) — the iter-0 `inf_du` blow-up this step exists to prevent. "Duplicate logic that diverged." **Fix**: mirror the sibling's `1e-8` perturbation (`least_square.rs:115,118`), with a cross-reference comment to keep the two in sync. **Verification**: the fail-first trigger is feral's *data-dependent* inertia mis-report on a CUTEst matrix (`nuffield2_trap`) **not in the repo**; the aug-solver unit harness uses `DenseMock` (an exact LU oracle) which cannot reproduce it, so a synthetic fail-first test is not constructible — the *sibling* fix itself shipped on the same basis (no synthetic fail-first test, integration-validated). Regression-safety is verified by running: `constr_mult_init_max` defaults to `1e3 > 0`, so every constrained solve traverses `calculate_y_eq`; the constrained-problem integration tests (`optimize_hs71`, `optimize_hs14`, `hock_schittkowski_subset`) and the full `pounce-algorithm` suite stay green (323 passed, 0 failed), confirming the `1e-8` perturbation is numerically inert (the constraint Jacobian dominates). See `## M3 detail`. |
| M4 | linalg: `symmetric_eigen` reports `true` on non-convergence | **FIXED** | **Confirmed by code inspection**: the doc (`eigen.rs:32-35`) promises `false` when the Jacobi sweeps run out, but the cyclic-Jacobi loop only `break`s on early convergence; after `max_sweeps` (50) it fell through to `return true` unconditionally (old `eigen.rs:153`). Callers branch on the verdict (`pounce-convex/src/cones/psd.rs:108,145,163,231`, `sos.rs:615,672,717`), so a stalled matrix would feed unconverged eigenpairs into PSD projections / SOS decompositions instead of the error path. **Fix**: track a `converged` flag (set on the early-`break`), recompute the off-diagonal mass once after the loop (to credit convergence achieved on the final sweep, whose state the top-of-loop check never sees), and `return converged`. Eigenpair extraction stays unconditional so callers still get best-effort values. To make the otherwise-unreachable `false` path testable, the body moved to a private `symmetric_eigen_impl(.., max_sweeps)`; the public `symmetric_eigen` delegates with `50` (signature/callers unchanged). **Tests** (`eigen.rs`): `eigen_reports_false_when_sweeps_exhausted` — a coupled 4×4 with `max_sweeps=1` must return `false` (pre-fix FAILS, returning `true`); `eigen_reports_true_when_converged` — same matrix at `max_sweeps=50` returns `true`, and an already-diagonal matrix converges even at `max_sweeps=1`. Pre-fix the first test FAILS; post-fix all 8 `eigen` tests pass, and `pounce-linalg` + `pounce-convex` (the consumers) stay green (328 passed, 0 failed). See `## M4 detail`. |
| M5 | QP: warm start can return `Optimal` at an infeasible point; unmarked equality rows never enforced | **FIXED** | **Mechanism confirmed by code inspection + reproduced by a failing test**: `ParametricActiveSetSolver::solve_general` (`crates/pounce-qp/src/solver.rs`) trusts the caller's warm-start `(x, working)` and steps with a zero-RHS active-set system (`rhs[n..] = 0`, lines 729-732), so the residuals of caller-marked-active rows are frozen and never re-audited; the `Optimal` return (lines 827-841) had **no** feasibility check, contradicting `QpStatus::Optimal`'s own contract ("KKT residual **and feasibility** within tolerance", `error.rs:8-9`). Separately, an equality row (`bl==bu`) the caller left `Inactive` is skipped by the ratio test (`if qp.bl[i]==qp.bu[i] { continue; }`, lines 883-884) and can **never** enter the working set, so it is never enforced. Net effect: a warm start at an infeasible point converges to a KKT-stationary point of the wrong working set and is returned as a silent `Optimal` (the doc claimed it "may diverge or hit max_iter" — the real failure is worse). **Fix**: add a post-solve feasibility audit in the public `solve` (the one entry point for both `solve_general` and `solve_general_schur`): a free fn `point_is_feasible` checks every general row **including equalities** and every variable bound against `feas_tol`; when a result claims `Optimal` but fails the audit, recover through `solve_elastic` — the exact recovery the cold path already uses when `cold_general_initial` returns an infeasible point. **Recursion-safe by construction**: `solve_elastic` recurses through `solve_general` *directly* (not the public `solve`), seeding a slack-feasible augmented problem, so the recovery is never re-audited and cannot loop. Feasible warm/cold results pass the audit untouched (happy path unchanged). The audit is the "`OptimalityCheck` audit pass" the doc comment (lines 668-671) explicitly deferred. **Test** (`tests/analytical.rs`): `m5_warm_start_inactive_equality_is_not_a_false_optimal` — `min ½‖x‖² s.t. x₁+x₂=2`, warm-started at `(0,0)` with the equality row `Inactive`; pre-fix returns `Optimal` at `(0,0)` (residual 2.0 — **FAILS** the feasibility assertion), post-fix recovers to the true optimum `(1,1)` reported `Optimal`. Full `pounce-qp` suite green (75 + 6 integration) and the `pounce-algorithm` QP consumer green (245 + SQP integration, 0 failed). See `## M5 detail`. |
| M6 | sensitivity: `SensSolve` swallows sensitivity-stage failures | **FIXED** | **Mechanism confirmed by code inspection + reproduced by a failing test**: the `on_converged` callback in `SensSolve::run` (`crates/pounce-sensitivity/src/convenience.rs`) writes a diagnostic into `CallbackOut.error` on *every* sensitivity-stage failure (no current iterate, inequality/invalid pin, `PdSensBacksolver::new` / `IndexSchurData::from_parts` error, `parametric_step` / `compute_reduced_hessian[_eigen]` returning false) and bails. But `CallbackOut.error` carried `#[allow(dead_code)]` and was **never copied into `SensResult`** (the result builder at the old lines 382-396 read every other `out.*` field but not `error`). Because the *underlying solve* still converged, `status` is `SolveSucceeded` and the requested `dx`/`reduced_hessian` are simply `None` — **indistinguishable from "sensitivity not requested."** A failed `parametric_step` therefore looked like success with no step computed. **Fix**: add a public `error: Option<String>` field to `SensResult` (documented as the sole signal separating a sensitivity failure from a not-requested computation), copy `out.error.clone()` into it in the builder, and drop the `#[allow(dead_code)]`. Updated the two unit-test `SensResult` literals in `diff_handoff.rs` (`error: None`). Also surfaced it end-to-end: the Python `info` dict now carries `info["sens_error"]` (`pounce-py/src/problem.rs`), since the Python binding is the primary user-facing consumer and previously had no way to see the failure either. **Test** (`tests/convenience_api.rs`): `sens_solve_surfaces_sensitivity_stage_failure` — solves the known-good `ParametricTNLP` (converges) but pins an out-of-range index, so the callback hits the "not in the equality c-block" branch and writes `error`. Post-fix asserts `status == SolveSucceeded`, `error.is_some()`, `dx.is_none()`; a paired happy-path solve asserts `error.is_none()` + `dx.is_some()`. **Pre-fix the assertion FAILS** ("failure must be surfaced … not swallowed; dx = None, status = SolveSucceeded") — verified by temporarily forcing `error: None` in the builder. Full `pounce-sensitivity` suite green (64 across 7 binaries, 0 failed); `pounce-py` builds clean. See `## M6 detail`. |
| M7 | QP: QPS parser doubles Hessian off-diagonals for `QMATRIX` files | **FIXED** | **Mechanism confirmed by code inspection + reproduced by a failing test**: `parse_qps` (`crates/pounce-qp/src/qps.rs`) mapped all three quadratic-section headers to the same state — `Some("QUADOBJ") \| Some("QSECTION") \| Some("QMATRIX") => section = Section::Quadobj` (old line 132). But the conventions differ: `QUADOBJ`/`QSECTION` list each off-diagonal pair **once** (single triangle), whereas `QMATRIX` lists the **full** matrix — both `(i,j)` and the mirror `(j,i)`. The content parser pushed every raw `(i_col, j_col, val)` triplet to `h_entries`; the lower-triangle normalization (`let (lo, hi) = if i>=j {(j,i)} else {(i,j)}`) then collapses both QMATRIX mirror entries onto the **same** lower triplet, and the evaluator sums all triplets → every off-diagonal is **doubled** (diagonal `i==j` is listed once, so unaffected). A QMATRIX file thus solves a different objective (`½xᵀHx` with off-diagonals 2×) and returns a wrong optimum. **Fix**: split the header match so `QMATRIX` sets a new `quad_is_full = true` flag (`QUADOBJ`/`QSECTION` set it `false`); in the content parser, when `quad_is_full && i_col < j_col`, skip the strict-upper mirror so each off-diagonal survives exactly once in the lower triangle. Single-triangle sections keep every entry (unchanged). **Latent-but-real**: no in-repo data uses QMATRIX (the `mm_published_optima` fixtures are all QUADOBJ, which is why they always passed), so this path had **zero** prior coverage; any user supplying a standard CPLEX/Maros-Mészáros QMATRIX file hit the bug. **Tests** (`src/tests/qps_unit.rs`): `parse_qps_qmatrix_full_matrix_does_not_double_off_diagonals` parses a QMATRIX `H = [[2,1],[1,2]]` (both `X1·X2` and `X2·X1` listed) and asserts the summed off-diagonal `H_21 == 1.0` (not 2.0) with diagonals intact; pre-fix it **FAILS** (`H_21 = 2`), post-fix passes. A paired `parse_qps_quadobj_single_triangle_keeps_off_diagonal` guards the QUADOBJ path against the fix regressing it. Full `pounce-qp` suite green (77 lib + 1 + 5 `mm_published_optima` integration, 0 failed). See `## M7 detail`. |
| M8 | l1penalty: augmented `x` passed to inner `eval_jac_g` | **FIXED** | **Mechanism confirmed by code inspection + reproduced by a failing test**: in `L1PenaltyBarrierTnlp` (`crates/pounce-l1penalty/src/wrapper.rs`) every forwarding method truncates the augmented variable vector to the inner's original `n` before calling the inner TNLP — `eval_f` (`&x[..n]`), `eval_grad_f` (`&x[..n]`), `eval_g` (`&x[..n]`), `eval_h` (`x.map(|xa| &xa[..n])`) — **except** `eval_jac_g`, which forwarded the full augmented slice `x` (length `n + 2·m_eq`) unchanged to both the `Structure` and `Values` inner calls (old lines 416, 445). The augmented variables append `m_eq` `p` and `m_eq` `n` slacks, so the inner saw `m_eq*2` extra trailing entries. **Why it matters / latent**: most inner `eval_jac_g` impls index `x[j]` for fixed `j < n` and are unharmed, so no in-repo test caught it — but any inner that validates `x.len()` (a reasonable defensive check) or iterates the slice (`x.iter()`) reads garbage/out-of-contract data. The inconsistency with the other four methods is itself a latent correctness hazard. **Fix**: compute `let inner_x = x.map(|xa| &xa[..n]);` once and pass `inner_x` to both inner `eval_jac_g` calls, mirroring `eval_h` exactly. The wrapper's own slack Jacobian entries (the `-1`/`+1` columns) are unchanged. **Test** (`wrapper.rs` tests): `jacobian_passes_inner_only_original_x` wraps a `LenSpy` inner TNLP (`n=2, m=1`) that records, via `Rc<Cell<usize>>`, the length of the `x` slice it receives in `eval_jac_g`; the test calls the wrapper's `eval_jac_g` with an augmented `x` of length 4 (`2 + 2·1`) and asserts the inner saw length **2**. Pre-fix the inner sees **4** (the assertion **FAILS**, verified by temporarily reverting `inner_x`→`x`); post-fix it sees 2. Full `pounce-l1penalty` suite green (11 tests) and the `pounce-algorithm` consumer green (245 + integration binaries, 0 failed). See `## M8 detail`. |
| M9 | restoration: silent zero-substitution on failed `DenseVector` downcasts | **FIXED** (scope corrected — sensitivity sites in the review do not exhibit the pattern) | **Mechanism confirmed by code inspection + reproduced by a failing test**: the restoration init/clone paths read outer-iterate blocks with `v.as_any().downcast_ref::<DenseVector>().map(|d| d.expanded_values()).unwrap_or_else(|| vec![0.0; dim])`. A failed downcast (a non-`DenseVector`, e.g. a compound block) silently substitutes **zeros** — seeding the restoration start point from a zero residual / zero multiplier with **no diagnostic**, masking the invariant violation. This is asymmetric with the *write* side, which already `.expect()`-panics on the same mismatch (`downcast_dense_mut`, `init.rs:475`). `expanded_values()` already handles the *homogeneous* DenseVector case correctly, so only a genuinely non-dense block triggers it. **Sites fixed (all in `pounce-restoration`)**: 7 inline reads in `init.rs` (c, d−s, s, z_l, z_u, v_l, v_u) plus the shared `expanded_dense_values` helpers in `resto_inner_solver.rs:775` and `resto_resto.rs:234`. **Scope correction**: the review also cited `pounce-sensitivity/src/solver.rs` and `convenience.rs` and `aug_resto_system_solver.rs:553`, but (a) a `grep` for the zero-fill pattern finds **none** in pounce-sensitivity (those line numbers now point to `IndexSchurData::from_parts` / the `SensResult` builder — unrelated; likely shifted by the M6 edit), and (b) `aug_resto_system_solver.rs:553` is `lr.get_diag()…unwrap_or_else(|| vec![0.0; n])` where the `Option` is a *legitimate* absence (a low-rank update with no diagonal → zero diagonal is correct), **not** a failed downcast — both excluded with rationale. **Fix**: introduce `expanded_dense_or_panic(v, what)` in `init.rs` (panics with a labelled message) and route all 7 inline sites through it; convert both `expanded_dense_values` helpers to panic (retaining `fallback_dim` only to size the diagnostic). Read and write sides are now symmetric — a non-dense block fails loudly. **Test** (`init.rs` tests): `expanded_dense_or_panic_panics_on_non_dense` builds a 1-block `CompoundVector` (not a `DenseVector`) and asserts the helper panics (`#[should_panic(expected = "must be a DenseVector")]`); `expanded_dense_or_panic_returns_values_for_dense` guards the happy path. **Pre-fix the panic test FAILS** ("test did not panic as expected" — the helper returns zeros), verified by temporarily restoring the silent `vec![0.0; v.dim()]` fallback. Full `pounce-restoration` suite green (105 lib + integration, 0 failed) and the `pounce-algorithm` consumer green (245 + integration, 0 failed). See `## M9 detail`. |
| M10 | Schur-update QP path: no inertia re-check after working-set drops; `O(m·nnz(A))` assembly per reset | **VERIFIED (by inspection) — doc corrected; behavioral fix DEFERRED** | **Asymmetry confirmed by code inspection.** The refactor path (`solve_general`/`solve_box_constrained`) calls `factorize_with_inertia_control` **every iteration** (`solver.rs:734`, `:238`), re-checking KKT inertia and applying a δ-shift on `WrongInertia`/`Singular`. The Schur path (`solve_general_schur`) runs inertia control **only inside `SchurState::reset`** (at init + every `max_schur_updates_before_refactor = 50` changes); the rank-2 SMW `apply_change` after a DROP (`solver.rs:1234`) does **not** re-check inertia. A drop enlarges the active-set null space and can expose negative curvature the cached factor never regularizes until the next reset, contradicting the doc claim "algorithmically identical to the refactor-per-iteration path" (`solver.rs:1137`). **Latent**: indefinite-reduced-Hessian only; `use_schur_updates` defaults `false` and *no production caller flips it* (the SQP driver feeds `HessianInertia::Psd`, for which the reduced Hessian is always PD and both paths are provably identical). **Not deterministically regression-testable**: two indefinite-QP probes — (a) `H = diag(-1,2)`, box `[-1,1]²`, drop into negative curvature; (b) same with `x₁` unbounded so the dropped direction is unbounded below — were run through *both* paths. **Both produced byte-identical results** (case a: both `Optimal` at `x=(-1,0)`; case b: both `MaxIter` at identical `x`). The active-set ratio-test re-add and the global-KKT-inertia gating (a single 1-D negative-curvature exposure often still matches `expected_neg`, so even the refactor path takes no shift) make constructed cases self-correct or diverge identically; I could not force a deterministic divergence to anchor a fail-first test. **Disposition mirrors M1**: documented, not silently fixed. **Verifiable correction applied**: the false "algorithmically identical" doc comment in `solver.rs` is rewritten to state the PD-only equivalence and spell out the indefinite-H inertia caveat (DROP vs ADD curvature argument). **Behavioral fix DEFERRED** (forcing `schur.reset()` unconditionally after every drop would restore parity, but absent a failing test and given the numerical delicacy / blast radius on the opt-in path, it is not applied here). **Perf sub-claim** (`O(m·nnz(A))` assembly in `build_k_max_triplet` per reset, `schur.rs`) is real but a performance characteristic, not a correctness bug, and not naturally regression-testable. `cargo test -p pounce-qp` green (77 + 1 + 5, 0 failed). See `## M10 detail`. |
| M11 | CLI QP extraction drops constraint terms folded into the nonlinear tree | **FIXED** | **Mechanism confirmed + reproduced by a failing test.** `extract_qp_with_map` (`crates/pounce-cli/src/qp_extract.rs`) built `A`/`G` from `prob.con_linear` **only**, ignoring `prob.con_nonlinear[row]`. But the classifier deliberately admits constraint rows whose nonlinear expression reduces to degree ≤ 1 (`dispatch.rs`), and AMPL/Pyomo fold a row's linear+constant terms into that nonlinear tree (cancelled quadratics, defined variables) — exactly as the *objective* path already handles via `analyze_quadratic_full` (`qp_extract.rs:80,98`) and as the **SOCP** extractor handles for constraints (`qp_extract.rs:355-396`, `nl_lin` + `const_shift`). So an LP/convex-QP with linear/constant terms inside a constraint's nonlinear tree got silently wrong constraints on the convex path: the folded coefficients vanished and the folded constant never shifted the bound. **Fix**: in the QP constraint loop, run `analyze_quadratic_full(&prob.con_nonlinear[row], n)` (empty Hessian for these linear rows), add the recovered `nl_lin` to the row coefficients, and shift the bound by the folded constant (`g_l−k ≤ row ≤ g_u−k`) — mirroring the SOCP path verbatim, including the `nonzeros()` filter so all-zero rows are not emitted. `con_nonlinear` is always parallel to `con_linear` (both length `m`, initialized to `Expr::Const(0.0)` per row at parse, `nl_reader.rs:295`), so the index is safe. **Test** (`qp_extract::tests::constraint_linear_terms_folded_in_tree_are_recovered`): `min x0 s.t. x0−3 ≥ 0` with the whole `x0−3` body in `con_nonlinear[0]` and `con_linear[0]` empty; asserts the solve is `Optimal` at `x0 = 3` and the recovered dual is finite. **Pre-fix the test FAILS** (`assert_eq!(sol.status, Optimal)` — the dropped constraint leaves a vacuous `0 ≤ 0` row and `min x0` is unbounded), confirmed by temporarily forcing the `nl_lin`/`const_shift` to `Default::default()` via an `if false` guard; post-fix it solves to `x0 = 3`. Full `pounce-cli` suite green (155 lib + all integration, 0 failed). See `## M11 detail`. |
| M12 | `DivergingIterates` mapped to AMPL code 401 ("limit") instead of the 300 ("unbounded") range | **FIXED** | **Mechanism confirmed + reproduced by a failing test.** `status_to_solve_result_num` (`crates/pounce-solve-report/src/lib.rs:453`) mapped `ApplicationReturnStatus::DivergingIterates → 401`. `DivergingIterates` is Ipopt's **unboundedness** signal (the iterates run off to infinity), so per the AMPL `solve_result_num` convention (300–399 = unbounded; 400–499 = limit) it belongs in the 300 range. This was internally inconsistent: the CLI's convex path maps the *same* unbounded condition (`QpStatus::DualInfeasible`) to **300** in its own numeric mapping (`main.rs:1276,1425`, with the range documented at `main.rs:1271-1272`), yet routes the NLP-side `DualInfeasible → DivergingIterates` (`main.rs:1165`) which then went through the 401 mapping — so the same physical outcome reported 300 on the convex path and 401 on the NLP path. Also matches upstream Ipopt's ASL driver (Diverging_Iterates → 300). **Fix**: one-line mapping change `DivergingIterates => 300`; the doc comment is extended to document the 300 "unbounded" bucket and why `DivergingIterates` lives there (not a 400 limit). **Test** (`tests::diverging_iterates_maps_to_unbounded_range`): asserts `DivergingIterates → 300`, and pins the surrounding buckets (`SolveSucceeded → 0`, `InfeasibleProblemDetected → 200`, `MaximumIterationsExceeded`/`SearchDirectionBecomesTooSmall → 400`, `RestorationFailed → 500`) so the range convention can't silently drift. **Pre-fix the test FAILS** (`left: 401, right: 300`), confirmed by reverting the mapping to `401`. No test anywhere hard-coded the old `401` (grep confirmed). `pounce-solve-report` (7) and `pounce-cli` suites green. See `## M12 detail`. |
| M13 | NLP-path presolve: `.sol` / JSON dual block carries the reduced kept-row count, not the original `.nl` `m` | **FIXED** | **Mechanism confirmed + reproduced end-to-end.** With `presolve=yes` on the general-NLP route, `PresolveTnlp` drops redundant rows and the solver works in a reduced (`m_out`) row space. The CLI captures the converged duals from **outside** that wrapper — the IPM `on_converged` hook (`main.rs:612`, via `pack_lambda_for_user`) and the active-set `CountingTnlp` fallback (`main.rs:950`) both see the reduced solution — so the `.sol` / JSON dual block had length `m_out`, shorter than the originating `.nl`'s `m`. AMPL/Pyomo read the dual block **positionally** against the `.nl`, so a short block mis-aligns or is rejected. **Reproduced** by running `lp_afiro.nl solver_selection=nlp presolve=yes` (drops 4 of 27 rows): pre-fix `.sol`/JSON `lambda` length was **23**, vs **27** for the `presolve=no` baseline; `dual_order.nl` (drops both rows) emitted a **zero-length** dual block. **Fix** (reuses existing machinery, no new dual math): `PresolveTnlp::finalize_solution` *already* lifts the duals back to the original row order with dropped-row multiplier recovery (the Phase-0 `recover_dropped_multipliers` path) before forwarding to the inner TNLP — it just wasn't surfaced. Added a `finalized_full_solution: Option<(Vec<Number>,Vec<Number>)>` capture on `PresolveTnlp` (stored at finalize, exposed via a getter); the CLI, when presolve dropped rows, swaps that full-length `lambda` into `nominal_capture` before the `.sol`/JSON writers run. Also: the `.sol` zero-fallback block and the JSON `problem.n_constraints` are now sized to the original `m` (`m_out + n_dropped_rows`), restoring the documented `lambda.len() == n_constraints` invariant. **Dropped-row duals**: redundant rows recover to a *valid alternative* certificate — exactly baseline for genuinely-slack rows (active-row duals match `lp_afiro` baseline tightly), and 0 where bound-tightening migrated the dual to a bound multiplier (e.g. `dual_order`); both satisfy KKT. The fix targets the **length/alignment** defect M13 names. **Test** (`tests/presolve_dual_length.rs::presolve_dual_block_keeps_original_nl_length`): runs `lp_afiro` through the NLP path with `presolve=no` then `=yes`, guards that presolve genuinely drops rows (parses the stdout summary), and asserts the presolved `lambda` length equals the baseline `m` **and** the reported `n_constraints`. **Pre-fix the test FAILS** (`presolve dual block length 23 != original .nl m 27`), confirmed by neutering the lambda swap with an `if false` guard. Mitigated in practice by presolve defaulting off. `pounce-presolve` (207 lib + 9 doc) and full `pounce-cli` (155 lib + all integration, 0 failed) suites green. See `## M13 detail`. |
| M14 | Any `--minima` tuning knob (`--seed`, `--patience`, `--dedup`, `--sobol`, …) silently switches the whole run into multistart mode | **FIXED** | **Mechanism confirmed + reproduced.** The `minima_num!` macro and the `--sobol`/`--no-sobol` arms in the CLI parser call `minima.get_or_insert_with(MinimaArgs::default)` to stash the knob value, which materializes `Some(MinimaArgs { method: Deflation, .. })` — and `main.rs:420` reroutes the *entire* run through multistart on any `Some(minima)`. So a lone tuning knob with no `--minima <method>`/`--multistart` silently enables global search (different output, `.sol` with zero duals). Help text says only `--minima`/`--multistart` enable it. **Reproduced**: `lp_afiro.nl --seed 42` (no method flag) prints `Searching for up to 10 minima via \`deflation\`…`. **Fix**: track an explicit-method flag (`minima_method_explicit`, set *only* by `--minima`/`--multistart`) and the first knob seen (`minima_knob`); after parsing, if a knob was given without an explicit method, return a clear error instead of silently entering multistart. **Verified post-fix**: lone `--seed 42` now errors `--seed is a --minima tuning knob and has no effect on its own; enable global search with --minima <method> or --multistart`; `--multistart --seed 42` still parses (method=Multistart, seed=42). **Test** (`cli::tests::lone_minima_knob_without_method_is_rejected` + `minima_knob_with_explicit_method_is_accepted`): the first asserts lone `--seed 42` and lone `--no-sobol` error and that the message names both the knob and `--minima`; the second asserts `--seed 7 --multistart` parses (order-independent) to method=Multistart, seed=7. **Pre-fix the rejection test FAILS** (lone `--seed 42` parses to `Some(MinimaArgs{method:Deflation,seed:42})`), confirmed by neutering the guard to `if false && !minima_method_explicit`. **Non-breaking**: every existing multistart test pairs its knobs with an explicit method. Full `pounce-cli` green (157 lib + all integration, 0 failed). See `## M14 detail`. |
| M15 | Real-AMPL driver conventions unsupported despite `-AMPL`: no `.nl`-appending for extensionless stubs, no `pounce_options` env var | **FIXED** | **Both facets confirmed + reproduced.** The `-AMPL` flag advertises "for Pyomo / AMPL drivers", and AMPL invokes a solver as `pounce mystub -AMPL` — passing the stub *without* `.nl` and conveying options through the `<solver>_options` env var (`pounce_options`). Pyomo worked (it passes a full `.nl` path and CLI `key=value` args); genuine AMPL did not. **Reproduced**: (1) `pounce mystub -AMPL` with `mystub.nl` present errored `could not read …/mystub: No such file or directory` (exit 2); (2) `pounce_options="max_iter=1" pounce model.nl` ignored the env var entirely. **Fix** — two small, additive changes: (a) `read_nl_file` (`crates/pounce-nl/src/nl_reader.rs`) now resolves an extensionless stub: if the path as given is missing but `<path>.nl` exists, read that (and name the `.col`/`.row` siblings off the resolved stem). An existing path is still read verbatim, so Pyomo / `--nl-file` / the second-positional form are untouched. A new `append_extension` helper *appends* `.nl` (AMPL semantics: `my.model` → `my.model.nl`), unlike `Path::with_extension` which would replace. (b) `main.rs` reads the `pounce_options` env var and merges its whitespace-separated `key=value` tokens (parsed by the new pure `cli::options_from_env`) *ahead of* the command-line `key=value` options, so an explicit CLI flag wins (`set_options` is applied last-wins). The `-AMPL` help text now documents both conventions. **Verified post-fix**: `pounce mystub -AMPL` solves to Optimal and writes `mystub.sol`; `pounce_options="max_iter=1"` caps iterations (Maximum Iterations Exceeded); a bogus env option exits 2 with `failed to set …`; CLI `max_iter=3000` overrides env `max_iter=1` (converges). **Tests**: `nl_reader::tests::{read_nl_file_resolves_extensionless_ampl_stub, read_nl_file_prefers_exact_path_over_nl_sibling, append_extension_appends_rather_than_replaces}`; `cli::tests::{options_from_env_parses_whitespace_separated_pairs, options_from_env_skips_non_kv_tokens_and_empty}`; integration `tests/ampl_driver_conventions.rs` (stub→`.nl`+`.sol`, env applied/rejected, CLI overrides env). **Fail-first**: neutering the stub fallback (`if true \|\| path.exists()`) fails the stub tests with `could not read …/mystub`; neutering the env merge fails the env tests (no `failed to set`, exit ≠ 2). Scope note: AMPL's rarer `keyword value` (space-separated) option spelling is intentionally not supported — it matches the existing CLI grammar, which has no `key value` form. `pounce-nl` (78) and full `pounce-cli` (159 lib + all integration, 0 failed) green. See `## M15 detail`. |
| M16 | Constraints and the full Jacobian are evaluated twice per iterate (no shared full-space cache below the c/d split) | **FIXED** | **Mechanism confirmed + reproduced.** In `OrigIpoptNlp` (`crates/pounce-nlp/src/orig_ipopt_nlp.rs`), `eval_c_internal` and `eval_d_internal` each independently called the user `eval_g` to fill a full-space `full_g`, then sliced their own rows out; likewise `eval_jac_c_internal`/`eval_jac_d_internal` each evaluated all `nnz_jac_g_full` entries via `eval_jac_g`. Because the filter line search needs both `c` and `d` (and both Jacobians) at each iterate, the dominant AD cost was paid **twice** — a ~2× tax on `.nl` problems. **Reproduced** with a counting `Hs071` TNLP: pre-fix, `eval_c(x)` then `eval_d(x)` at one iterate invoked the user `eval_g` **2×** (and `eval_jac_c`+`eval_jac_d` invoked `eval_jac_g` 2×). **Fix** (mirrors upstream's tagged `full_g_`/`jac_g_` buffers): added two shared, tag-keyed caches — `full_g_cache`/`full_jac_g_cache` (`Cache<Rc<Vec<Number>>>`, size 1) — and two private helpers `full_g(x)`/`full_jac_g(x)` that compute the full-space vector/Jacobian once per iterate and memoize on the input vector's tag. `eval_c`/`eval_d` now slice rows out of one shared `full_g(x)`; `eval_jac_c`/`eval_jac_d` slice one shared `full_jac_g(x)`. NaN-on-eval-failure and scaling/bound-subtraction semantics are unchanged (only the *source* of `full_g`/`full_vals` moved). Per-subsystem counters (`c_evals`/`d_evals`/`jac_c_evals`/`jac_d_evals`) still report one evaluation each — they count produced c/d vectors, which is legitimate — while the redundant *user* AD call is gone. **Verified post-fix**: the counting TNLP shows exactly **1** `eval_g` shared by `eval_c`+`eval_d` (and 1 `eval_jac_g` shared by both Jacobians); a genuinely new iterate (tag bumped) costs exactly one more; values unchanged (c=12, d=25 at the HS071 start). End-to-end `lp_afiro solver_selection=nlp` still converges to the known optimum (−464.753, Optimal). **Tests** (`orig_ipopt_nlp::tests`): `eval_c_and_eval_d_share_one_eval_g_per_iterate`, `eval_jac_c_and_eval_jac_d_share_one_eval_jac_g_per_iterate` (new `build_orig_nlp_counting` keeps a typed `Rc<RefCell<Hs071>>` handle to read the user-side call counters). **Pre-fix both FAIL** (`left: 2, right: 1`), confirmed by neutering the shared lookups with `.filter(|_| false)`. `pounce-nlp` (36, 0 failed), `pounce-algorithm`, and `pounce-cli` suites all green. See `## M16 detail`. |
| M17 | `eval_c_internal` re-fetches all bounds and makes four full-size scratch allocations per cache miss, in the line-search hot path | **FIXED** | **Mechanism confirmed + reproduced.** In `OrigIpoptNlp` (`crates/pounce-nlp/src/orig_ipopt_nlp.rs`), `eval_c_internal` formed the equality residual `c_i = g[g_idx] - g_l[g_idx]` by calling the user `get_bounds_info` on **every** cache-missing iterate — allocating four fresh full-size scratch vectors (`tmp_x_l`, `tmp_x_u`, `full_g_l`, `full_g_u`) each time — purely to read the *constant* equality RHS `g_l[g_idx]` (`g_l == g_u` for equalities). Since the filter line search evaluates `c` at every trial point, this per-trial bounds fetch + four allocations sat squarely in the hot path. Upstream Ipopt captures the RHS once as `c_rhs`. **Reproduced** with a counting `Hs071` TNLP (now also tallying `get_bounds_info_calls`): pre-fix, each fresh iterate added another `get_bounds_info` call (6 extra calls for 6 iterates above the 2-call construction baseline). **Fix**: capture the constant RHS once at construction — new field `c_rhs: Vec<Number>` (length `n_c`), computed in `OrigIpoptNlp::new` from the already-fetched `full_g_l` as `c_map.iter().map(|&g_idx| full_g_l[g_idx]).collect()`. `eval_c_internal` now forms `raw = full_g[g_idx] - self.c_rhs[i]`, dropping the per-iterate `get_bounds_info` call and all four scratch allocations. Scaling and the `full_g` source (M16's shared cache) are unchanged, so numerics are identical. **Verified post-fix**: counting TNLP shows `get_bounds_info_calls` stays at the construction baseline across 6 fresh-iterate `eval_c` calls; residual value unchanged (c = g1−40 = 12 at the HS071 start); end-to-end `lp_afiro solver_selection=nlp` still converges to −464.75314761311961 ("Optimal Solution Found"). **Test** (`orig_ipopt_nlp::tests::eval_c_does_not_refetch_bounds_per_iterate`): snapshots `get_bounds_info_calls` after construction, runs `eval_c` at six distinct iterates, asserts the count is unchanged (and that c=12). **Pre-fix FAILS** (`left: 8, right: 2`), confirmed by neutering the fix — temporarily restoring the per-iterate `get_bounds_info` fetch — then restored. `pounce-nlp` (37, 0 failed), `pounce-algorithm` (245), and full `pounce-cli` (159 lib + all integration, 0 failed) green. See `## M17 detail`. |
| M18 | Per-call allocations in the tape-AD gradient hot path (`forward` + `reverse` allocate per summand tape, ~10⁶ tapes per `eval_jac_g`) | **FIXED** | **Mechanism confirmed + reproduced.** In `crates/pounce-nl/src/nl_tape.rs`, `Tape::gradient_seed` calls `forward` (allocates a `Vec<f64>` of forward values, `:198`) and `reverse` (allocates `adj = vec![0.0; n]`, `:272`) on every invocation. The `.nl` front end (`nl_reader.rs`) deliberately emits one tiny `Tape` per additive summand, so `eval_jac_g` (`gradient_seed` per constraint summand) and `eval_grad_f` (per objective summand) drive these two small allocations millions of times on large models. The Hessian path already had the `forward_into` + reusable-scratch pattern; the gradient path did not. **Reproduced** with a counting global allocator: `gradient_seed` allocates on essentially every call (≥1000 allocations across 1000 calls on a sample tape). **Fix** (mirrors the Hessian scratch pattern): added `Tape::gradient_seed_into(x, seed, grad, vals, adj)` — a `pub` allocation-free variant that runs `forward_into` (existing) into a caller `vals` arena and a new private `reverse_into(vals, seed, grad, adj)` that zeroes the touched `[0,n)` slots of a caller `adj` arena instead of allocating one. `reverse` now delegates to `reverse_into` (behavior unchanged; `gradient_seed` and the FD-comparison tests still exercise it). The two hot-path callers in `nl_reader.rs` (`eval_grad_f`, `eval_jac_g`) now call `gradient_seed_into`, reusing the existing `vals_scratch`/`adj_scratch` arenas (already sized to `max_tape_n`, the max over all obj+con tapes, so always ≥ any single summand tape). **Verified post-fix**: the counting allocator shows `gradient_seed_into` performs **0** allocations across 1000 calls (vs ≥1000 for `gradient_seed`) and computes the identical gradient; end-to-end NLP solves unchanged — `convex_qp`→2.0, `tame`→0.0, `nonconvex_qp`→1.0, all "Optimal Solution Found". **Test** (`tests/tape_gradient_no_alloc.rs`): a counting `#[global_allocator]` (single test in its own integration binary so no sibling thread perturbs the counter) asserts `gradient_seed_into` == `gradient_seed` numerically, the baseline allocates ≥1000×, and the new path allocates 0×. **Pre-fix the 0-assertion FAILS** (`left: 1000, right: 0`), confirmed by neutering `gradient_seed_into` with a throwaway `vec!` per call, then restored. `pounce-nl` (78 lib + 1 new integration, 0 failed) and full `pounce-cli` (0 failed) green. See `## M18 detail`. |
| M19 | Initial duals from the `.nl` `d` segment parsed into `lambda0` but never used; `warm_start_init_point yes` silently warm-starts from zero multipliers | **FIXED** | **Mechanism confirmed + reachability traced.** The `.nl` reader (`crates/pounce-nl/src/nl_reader.rs`) parses the `d` segment into `NlProblem::lambda0` (`:458`), but `NlTnlp::get_starting_point` only copied `x0` into `sp.x` and ignored `sp.lambda` entirely — so the parsed constraint multipliers were discarded. The module header even claimed the `d` segment is "read and discarded". **Reachability**: the warm-start path is live — `OrigIpoptNlp::get_starting_point` (`crates/pounce-nlp/src/orig_ipopt_nlp.rs:1289`) requests `init_lambda: init_y_c \|\| init_y_d` and, on return, compresses `full_lambda` into the algorithm-side `y_c`/`y_d` via `c_map`/`d_map` with `obj`/constraint scaling (`:1320-1333`). The engine sets `init_lambda` when `warm_start_init_point yes`; with the TNLP leaving `sp.lambda` untouched, the warm start began from zero multipliers — defeating the point of supplying `lambda0`. **Fix**: when `sp.init_lambda` is set, `get_starting_point` now copies `self.prob.lambda0` into `sp.lambda` (the `.nl` `d` segment carries no bound multipliers, so `z_l`/`z_u` are left to engine defaults); the stale module-header doc comment is corrected to state both `x` and `d` are parsed and returned, the duals feeding a `warm_start_init_point` solve. **Verified post-fix**: the new test parses an equality-constrained `.nl` with `d1\n0 2.5`, asserts `lambda0 == [2.5]`, then drives `get_starting_point` with `init_lambda: true` (yields `lambda == [2.5]`) and `init_lambda: false` on a pre-filled buffer (left untouched at `[7.0]`). Cold-start paths are unaffected because they pass `init_lambda: false`; `lp_afiro solver_selection=nlp` still converges to −464.75314761311961 ("Optimal"). **Test** (`nl_reader::tests::get_starting_point_returns_nl_initial_duals`). **Pre-fix the warm-start branch FAILS** (`left: [0.0], right: [2.5]`), confirmed by neutering the copy with `if sp.init_lambda && false`, then restored. `pounce-nl` (79 lib + integration, 0 failed) and full `pounce-cli` (0 failed) green. See `## M19 detail`. |
| M20 | Silent tolerance relaxation: convex IPM breakdowns re-labeled `Optimal` at residuals far above `tol`, with no way for callers to detect the reduced accuracy | **FIXED** | **Mechanism confirmed + reproduced.** Both convex HSDE drivers accept a usable-but-inaccurate iterate when a factorization/back-solve breaks down (or the cap/a stalled step is hit) while the best KKT residual is already small: the symmetric `hsde.rs` accepts within `~1e3·tol` (`near_opt`, 4 break sites), and the non-symmetric `hsde_nonsym.rs` accepts within `~1e3·tol` (6 break sites) and, at exit, restores the best iterate if `best_res < √tol` (`:1176`). Each of these reported a bare `QpStatus::Optimal` — and since `QpStatus` had no reduced-accuracy variant and `QpSolution` carries no final-residual field, a residual sitting at `1e-5`/`1e-4` (default `tol=1e-8`) was indistinguishable from a genuinely converged solve. ECOS/Clarabel expose exactly this as a distinct `*_INACC` status. **Reproduced**: the exp/power-cone GP suites (`exp_cone_vs_nlp`, `cblib_cbf`, `cblib_vs_nlp` — `demb761`/`beck751`/`fang88`/`pow3`, log-sum-exp, entropy, near-boundary GP) reach their optima through the non-symmetric reduced-accuracy fallback, so they were *already* being reported as `Optimal` despite residuals only within `√tol` — the masquerade in the field. **Fix** (mirrors ECOS/Clarabel `*_INACC`): added `QpStatus::OptimalInaccurate` and a centralized `breakdown_status(near_opt)` (in `qp.rs`) that returns `OptimalInaccurate` when `near_opt` (else `NumericalFailure`); both drivers now call it instead of an inline `if near_opt { Optimal } else { NumericalFailure }`, and the non-symmetric best-iterate restoration sets `OptimalInaccurate`. The clean convergence test (`pres<tol && dres<tol && gap<tol → Optimal`) is untouched, so genuinely-converged solves still report `Optimal`. CLI surfacing: `convex_status_report` (extracted, shared by the QP/LP and SOCP report paths) maps `OptimalInaccurate → ("Solved to acceptable level (reduced accuracy).", ok=true, solve_result_num=100)` — the AMPL 100–199 reduced-accuracy band; `qp_status_to_ars` (CLI + `pounce_cblib`) maps it to `ApplicationReturnStatus::SolvedToAcceptableLevel` (the JSON report's `solve_result_num` 100); `pounce_cblib` treats it as a success exit code; the two `pounce-py` `status_str` maps emit `"optimal_inaccurate"`. Conservatively, sensitivity (`sensitivity.rs:91`) and SOS exactness (`sos.rs:474/498`) keep their strict `== Optimal` gate, so reduced-accuracy solutions are *not* used as exact certificates. **Verified post-fix**: well-conditioned `lp_afiro solver_selection=lp-ipm` and `convex_qp solver_selection=qp-ipm` still print "Optimal Solution Found." (`srn` 0) — the new path does not fire for them; the exp/power-cone suites now report `OptimalInaccurate` while their objective cross-checks against an independent NLP still hold. **Tests**: `qp::residual_tests::breakdown_status_marks_near_opt_as_inaccurate_not_optimal` (pins `breakdown_status(true)==OptimalInaccurate≠Optimal`, `(false)==NumericalFailure`); CLI `convex_status_tests::optimal_inaccurate_is_distinct_from_optimal` (pins `srn=100`, `ok=true`, message ≠ "Optimal…", and `→ SolvedToAcceptableLevel`, all distinct from `Optimal`). Existing conic tests updated to accept either usable status (clean `Optimal` or `OptimalInaccurate`) while keeping their objective checks. **Pre-fix FAILS** (`left: Optimal, right: OptimalInaccurate`), confirmed by neutering `breakdown_status` to return `Optimal`, then restored. `pounce-convex` (104 lib + integration, 0 failed), full `pounce-cli` (20 binaries, 0 failed), `pounce-py` build green. **Residual-field follow-up** (a `QpSolution.final_residual`) deferred — the distinct status already resolves "callers cannot detect it". See `## M20 detail`. |
| M21 | SOS flat-truncation exactness check is weaker than Curto–Fialkow for constrained problems; extracted atoms are never validated against the constraints, so `is_exact = true` can over-claim | **FIXED** | **Mechanism confirmed + concrete failing instance constructed** (the review left this "Uncertain"). `recover_from_moments` (`crates/pounce-convex/src/sos.rs`) certified exactness purely by flat truncation of the moment matrix `M_d` (`rank M_d = rank M_{d−1}`) and then extracted the atoms — it never looked at `prob.inequalities`/`prob.equalities`. For a *constrained* program `rank M_d = rank M_{d−1}` only certifies a representing measure on `ℝⁿ`; when some constraint `gᵢ` has `dg = ⌈deg gᵢ/2⌉ > 1` (degree > 2) the `d−1` window is a strictly *weaker* condition than the `d−dg` window Curto–Fialkow/Henrion–Lasserre require to pin the atoms to `K`, so a flat `M_d` can extract atoms outside the feasible set while `is_exact = true` reports the *unconstrained* bound as the exact constrained optimum. **Reproduced by running code**: `min (x+1)² s.t. x³ ≥ 0` (feasible `x ≥ 0`; true constrained min `1` at `x = 0`, unconstrained min `0` at `x = −1`) at order 2 reported `is_exact = true`, `lower_bound = 0`, single "minimizer" at **x ≈ −0.719 — infeasible** (`x³ ≈ −0.37 < 0`). Same pathology for `min (x+2)² s.t. x³ ≥ 0` (atom ≈ −0.63) and `min (x+1)² s.t. x⁵ ≥ 0` at order 3 (a spurious atom at x ≈ 318). **Fix**: validate every extracted atom against the constraints before keeping the certificate — `recover_from_moments` now takes `prob` and, after extraction, calls a new `PolyProblem::is_feasible(x, tol)` (each `gᵢ(x) ≥ −tol·(1+‖gᵢ‖(x))`, `|hⱼ(x)| ≤ tol·(1+‖hⱼ‖(x))`, a scale-invariant relative margin via new private `Polynomial::eval`/`eval_magnitude`); if any atom is infeasible the exactness certificate is withdrawn (`is_exact = false`, no minimizers). `lower_bound` is unchanged — it stays a valid lower bound (here `0 ≤ 1`). Unconstrained problems have empty constraint lists, so `is_feasible` is a no-op and their recovery is untouched. The facial-reduction re-solve also runs the validated recovery. The `SosSolution`/`is_exact` docs now state the constrained certificate requires feasible atoms. **Verified post-fix**: the failing instance now returns `is_exact = false`, `num_minimizers = 0`, `lower_bound ≤ 1`; all existing `sos_minimize` extraction tests (unconstrained — unique/two/three/four-atom, facial reduction, Goldstein–Price) still certify and extract unchanged. **Test** (`sos::tests::constrained_overclaim_rejected_when_atom_infeasible`). **Pre-fix FAILS** (`is_exact = true` with minimizer `[−0.719]`), confirmed by neutering the guard (`is_feasible(..) || true`), then restored. `pounce-convex` (105 lib + all integration, 0 failed) and `pounce-py` build green. See `## M21 detail`. |
| M22 | SOS SDP assembly iterates a `HashMap`, so the coefficient-matching row order — and hence the solver's floating-point results — is nondeterministic run-to-run | **FIXED** | **Mechanism confirmed + reproduced by running code.** `build_sos_sdp` (`crates/pounce-convex/src/sos.rs`) accumulates the coefficient-matching equalities in `by_mono: HashMap<Vec<usize>, Vec<(usize,f64)>>` and then emits one SDP row per entry by iterating `for (alpha, terms) in &by_mono`. Rust's `HashMap` is seeded per-instance (DoS-resistant `RandomState`), so the iteration order — and therefore the assignment of monomials to SDP rows — changes every run; the solver then walks a different (mathematically equivalent) row permutation each time, yielding run-to-run floating-point drift in the bound/minimizers (the SOS tests carry loose `1e-5` tolerances to absorb it). **Reproduced**: building the *same* problem twice in one process (each `HashMap` gets a distinct seed) gives different RHS row order and a different monomial→row map every run — `qp1.b == qp2.b` and `mi1.row_of == mi2.row_of` were both **false** across three separate processes. **Fix**: switch `by_mono` to a `BTreeMap<Vec<usize>, …>` (keys are exponent vectors, which are `Ord`), so the row-emitting iteration is in deterministic sorted-by-monomial order. One-line type change plus the import; `row_of`/`coeff_map` stay `HashMap` (lookup-only, order irrelevant — their row *values* are now deterministic because they derive from the ordered `by_mono` walk). No numerical or API change beyond fixing the ordering. **Verified post-fix**: the same twice-build probe now yields `qp1.b == qp2.b` **and** `mi1.row_of == mi2.row_of` (true); all SOS bound/extraction tests still pass. **Test** (`sos::tests::sdp_row_order_is_deterministic`): builds the SDP twice and asserts identical RHS order and monomial→row map. **Pre-fix FAILS** (`RHS row order differs between builds`), confirmed by reverting `by_mono` to `HashMap`, then restored. `pounce-convex` (106 lib + all integration, 0 failed) and `pounce-py` build green. See `## M22 detail`. |
| M23 | `PsdCone::kkt_block` is O(n⁵) per cone per IPM iteration (it applies the scaling operator to every svec unit vector); `lyapunov_solve` likewise uses O(n⁴) quadruple loops instead of two matmuls — dominant per-iteration cost for PSD/SOS moment SDPs | **FIXED** | **Cost confirmed by timing.** `kkt_block` (`crates/pounce-convex/src/cones/psd.rs`) built the `m×m` (`m = n(n+1)/2`) symmetric-Kronecker block `H = W ⊗ₛ W` by, for each of the `m = O(n²)` svec unit vectors `e_b`, calling `apply_scaling` (two O(n³) matmuls `W·smat(e_b)·W`) and extracting the column — O(n²·n³) = **O(n⁵)** per cone per iteration. `lyapunov_solve` formed `R̃ = QᵀRQ` and `D = QD̃Qᵀ` with explicit quadruple loops — **O(n⁴)** each instead of O(n³). For SOS moment SDPs (one large PSD cone, `n` = moment-matrix size) this is the dominant per-iteration work. **Reproduced by running code**: a timing probe over `n = 8,16,24,32` showed the old construction scaling steeply super-quartically (`n=16→0.00052s`, `n=24→0.0036s`, `n=32→0.014s`, ~factor-7 per `n`-doubling between the O(n⁴) and O(n⁵) regimes). **Fix**: replace the per-unit-vector loop with a **closed form** for the symmetric-Kronecker entries — column `b ↔ svec pair (p,q)`, `D = W·smat(e_b)·W` has `D_ij = W_ip W_jp` (`p=q`) or `(W_ip W_jq + W_iq W_jp)/√2` (`p>q`), and `H[a][b] = (i=j ? 1 : √2)·D_ij` for output pair `(i,j)` — building the lower triangle directly in **O(n⁴)** (one `O(1)` expression per of the `O(n⁴)` entries). `lyapunov_solve` rewritten to transpose `Q` once (column-major→row-major) and use the existing `matmul` for both congruences — **O(n³)**. No public API, cone layout (`ConeBlock::DenseLower`, svec √2 convention), or numerical contract changed. **Verified post-fix**: the same timing probe gives `n=32` **20.1× faster** (`0.014s → 0.0007s`), and the measured speedup grows ~linearly in `n` (1.2×→6.4×→13.1×→20.1× over `n=8,16,24,32`) — exactly the O(n) factor O(n⁵)→O(n⁴) predicts. **Tests**: `cones::psd::tests::kkt_block_matches_apply_scaling_reference` asserts the closed-form block reproduces — entry for entry, within `1e-9`, over `n = 1,2,3,5,8` — the reference built by applying `W⊗ₛW` to each unit vector (the previous construction); the existing `kkt_block_maps_z_to_s` (defining property `H·svec(z)=svec(s)`), `recover_ds_matches_block_and_rhs`, and `lyapunov_inverts_jordan` (`z∘(Arw(z)⁻¹r)=r`, guards the matmul rewrite) all still pass. **Pre-fix FAILS** (`[0][0]: … vs …`), confirmed by perturbing the closed-form entry by `1e-3`, then restored. `pounce-convex` (107 lib + all integration, 0 failed) and `pounce-py` build green. See `## M23 detail`. |
| M24 | Rows dropped as redundant *because of the bound-tightening they themselves implied* (e.g. `x ≥ 2` tightens `x_l = 2`, then is dropped) get `λ = 0`; if that bound is active the IPM reports the dual on a variable-bound multiplier (`z_l`/`z_u`) against a bound absent from the original problem | **DOCUMENTED** (verified; design-inherent, fix deferred) | **Mechanism confirmed + reproduced by running code.** Phase 2 (`crates/pounce-presolve/src/redundant.rs`) drops a linear row when its activity interval over the current box is `⊆ [lo, hi]`; a row like `x ≥ 2` first tightens `x_l = 2` in Phase 1, so its activity `[2, 10] ⊆ [2, +∞)` and Phase 2 drops it. The reduced solve then puts the dual on the (presolve-created) *variable bound* `x ≥ 2`, while the reinstated row keeps `λ = 0` — the dual is attributed to a bound that did not exist in the original `.nl`. **Reproduced**: a single-variable `min x s.t. x ≥ 2` (orig. box `[0,10]`) driven through `PresolveTnlp` reports `n_dropped_rows = 1`, `m: 1→0`, `x_l` tightened to `2`; finalizing the reduced optimum `x=2` with `z_l=1` gives recovered full-space `λ[0] = 0` (the dual stays on `z_l`, is *not* transferred to the row). **Scope**: the review notes "primal/objective unaffected; inherent to the design and worth documenting or fixing via dual transfer." Verified that the KKT certificate is intact — stationarity `∇f − Jᵀλ − z_l + z_u = 1 − 0 − 1 + 0 = 0` holds with the dual on `z_l`; only the *attribution* (bound multiplier vs. row `λ`) differs from a no-presolve solve. **Why documented, not fixed**: a faithful dual transfer needs Phase-1 *provenance* (which row implied which bound) — `bound_tighten::tighten_bounds` returns only counts (`TightenReport`), records no row→bound link, and the multi-variable case is genuinely ambiguous (a row binding at a box vertex maps to several bound multipliers). The existing `recover_dropped_multipliers` machinery covers only Phase-0 aux-eliminated rows (reduction-stack frames), not Phase-2 redundant rows. Adding provenance plumbing is a substantial, risky change the review itself ranks behind documenting; deferred as future work with an explicit target. **Action**: expanded the Phase 2 module-doc with a "Dual-attribution caveat (issue M24)" paragraph stating the limitation precisely (primal/objective/KKT unaffected, attribution differs, transfer needs untracked provenance), and added a **characterization test** `tests::dropped_row_dual_lands_on_bound_not_row` pinning the behavior: row dropped, `x_l→2`, full-space `λ[0]==0`, primal `x=2`, and KKT stationarity residual `≈0`. The `λ[0]==0` assertion is the explicit hook a future dual-transfer fix would flip to `λ[0]≈z_l`. Doc-only + test; no behavior change. `pounce-presolve` (208 lib + integration, 0 failed) and `pounce-py` build green. See `## M24 detail`. |
| M25 | A genuine Phase-1 infeasibility leaves crossed bounds `x_l > x_u` in the box handed to the IPM — the rollback guard only fires when the reduction stack is non-empty, so an infeasibility found with an empty stack reaches the solver as a degenerate problem | **FIXED** | **Mechanism confirmed + reproduced by running code.** `tighten_pass` (`crates/pounce-presolve/src/bound_tighten.rs:128-131`) returns the instant it derives `x_l[j] > x_u[j]`, leaving those crossed bounds written into `x_l`/`x_u`. In `lib.rs` the only restoration was the aux-rollback guard (`if tighten_report.infeasible && !reduction_stack.is_empty()`, `:609`), which re-runs Phase 1 on un-filtered rows — but it is gated on a *non-empty* reduction stack. When `auxiliary` is off (the default) the stack is empty, so a genuine Phase-1 infeasibility skips the guard entirely and the crossed box flows straight into `CachedBounds` (`:821`) → the IPM, which reports an invalid-problem error rather than a clean infeasibility verdict. (The FBBT path at `:679-695` already handles its own infeasibility correctly by restoring the pre-FBBT box; the Phase-1 path did not.) **Reproduced**: a single-variable TNLP with box `x ∈ [0,10]` and two contradictory linear rows `x ≥ 5`, `x ≤ 3` (auxiliary off) — after `get_nlp_info`, `tighten_report().infeasible == true` and `cached_bounds()` returns `x_l = [5.0]`, `x_u = [3.0]` (crossed). **Fix** (mirrors the aux rollback and FBBT handling): after the Phase-1 block, `if tighten_report.infeasible { x_l.copy_from_slice(&inner_x_l); x_u.copy_from_slice(&inner_x_u); }` — restore the pristine original inner box (snapshotted at `:472-473`, always a valid box; it is also the correct target after a rollback re-tighten that stays infeasible, since the rollback already restored to it before re-tightening). The IPM then runs on a well-posed box and certifies infeasibility itself; the `infeasible` flag is preserved and still surfaced via `tighten_report()` for diagnostics (a `tracing::warn!` records the discard). **Verified post-fix**: the same reproduction now returns `x_l = [0.0]`, `x_u = [10.0]` (valid, original box) while `tighten_report().infeasible` stays `true`. **Test** (`tests::phase1_infeasible_restores_valid_box_for_ipm`): asserts the box handed to the IPM is valid (`x_l ≤ x_u`, restored to `[0,10]`) and the infeasibility is still flagged. **Pre-fix FAILS** (`bounds handed to IPM must be valid, got x_l=5 > x_u=3`), confirmed by neutering the guard (`&& false`), then restored. The existing aux-rollback test (`phase0_via_tnlp_no_infeasible_with_default_bound_tightening`) — where the rollback *clears* the infeasibility — is unaffected (the new guard only fires when `infeasible` survives). `pounce-presolve` (209 lib + integration, 0 failed) and `pounce-py` build green. See `## M25 detail`. |
| M26 | `finalize_solution` densifies the *full* inner Jacobian — `vec![0.0; m_inner * n_inner]` whenever a reduction frame exists (≈80 GB at 100k×100k) — even though `recover_dropped_multipliers` reads only the `k` fixed-var columns | **FIXED** | **Mechanism confirmed by reading + running code.** In `lib.rs` the multiplier-recovery block allocated a dense `m_inner × n_inner` Jacobian (`jac_dense`) from the inner COO, then handed it to `frame.recover_dropped_multipliers` for every frame. But that method (`reduction_frame.rs`) indexes `jac_full_row_major` **only** at columns `i ∈ fixed_vars` — the k×k block matrix at `jac[dr*n_vars + i]` and the kept-row sum at `jac[r*n_vars + i]`, both with `i ∈ fixed_vars`, and **never any other column**. So the `(n_inner − k)` non-fixed columns of the dense block are materialized and never read: O(m·n) memory for an O(m·k) need (k = total distinct fixed vars across frames, typically a handful). **Verified by running code**: `tests::recover_only_reads_fixed_var_columns` fills every *non-fixed* column of the dense Jacobian with `NaN` and shows the recovered multipliers are **bit-for-bit identical** (`to_bits()`) to the clean run and stay finite — proving those columns are dead weight. **Fail-first**: poisoning a *fixed* column instead makes the recovery go `NaN` and the test fails (confirmed by a temporary edit, then restored). **Fix**: add `ReductionFrame::recover_dropped_multipliers_cols`, a column-compacted variant reading an `(n_full_rows × n_cols)` buffer via an `orig_to_compact` map; both it and the original `recover_dropped_multipliers` now delegate to a shared private `recover_core(grad_f, lambda_full, get: impl Fn(row,col)->Number)` so the math lives in one place (all 8 existing call sites + the doctest unchanged). `finalize_solution` now builds only the union of the frames' `fixed_vars` columns (`m_inner × n_cols`) instead of `m_inner × n_inner`. **Tests**: `recover_cols_matches_dense` asserts the compacted recovery reproduces the dense one bit-for-bit; `recover_cols_empty_frame` covers the zero-fixed-var (`n_cols == 0`) path. `pounce-presolve` (212 lib + integration + 9 doctests, 0 failed) and `pounce-py` build green. See `## M26 detail`. |
| M27 | Phase-0 block assembly rescans the *entire* COO Jacobian once per block row — O(block_rows × nnz) — in the C2 gate, `solve_linear_block`, `residual_norm_linear`, and `NonlinearBlock::jacobian`; quadratic in problem size for many small blocks | **FIXED** | **Cost confirmed by timing.** Phase-0 auxiliary elimination (`crates/pounce-presolve/src/auxiliary.rs`) assembles each square block by walking the full COO Jacobian (`for kk in 0..nnz { decode row; if row != r continue; … }`) once **per block row** — in the C2 acceptance gate, `solve_linear_block`, `residual_norm_linear`, and `NonlinearBlock::jacobian`. For the common case of many small blocks (e.g. n singleton rows, the diagonal-aux pattern) the total work is `O(total_block_rows × nnz) = O(n²)`. **Reproduced by running code**: a timing probe over the all-singleton diagonal pattern at `n = 500/1000/2000` showed clearly super-linear scaling — `1.11 / 2.32 / 7.03 ms` (each doubling of `n` more than doubling time, trending toward 4×). **Fix**: build a CSR row index **once** up front — `build_row_nnz(jac_irow, n_rows, one_based)` returns `(ptr, entries)` mapping each 0-based row to the list of its nnz positions `kk` in O(nnz); a `Copy` `RowNnz { ptr, entries }` view exposes `of_row(r) -> &[kk]`. The four hot loops now iterate `for &kk in row_nnz.of_row(r)` instead of scanning all nnz, threaded through the C2 gate, `solve_linear_block`, `residual_norm_linear`, and `NonlinearBlock::jacobian` — total Phase-0 assembly drops to O(nnz). Index decode (`decode_idx`) honors the one-/zero-based COO convention identically in the count and fill passes. No public API or numerical contract changed. **Verified post-fix**: the same probe is near-linear — `0.54 / 0.92 / 1.97 ms` at `n = 500/1000/2000`, a speedup that *grows* with `n` (2.1× → 3.6×), exactly the O(n)→O(1)-per-row improvement predicts. **Tests**: `build_row_nnz_groups_by_row_zero_based` and `build_row_nnz_honours_one_based_decode` pin the CSR builder (one-based ptr/entries identical to zero-based); `phase0_diagonal_many_singletons_correct` (n = 400) asserts all n vars are fixed to `r+1` and all rows dropped; `phase0_one_based_two_blocks_eliminated` is an end-to-end one-based pipeline asserting `fixed_vars == [0,1]`, values `5.0/7.0`, `dropped_rows == [0,1]` — identical to the pre-M27 result. **Pre-fix vs neuter**: a clean fail-first (restore with `cp`+`touch`, never `mv` — `mv` rolls the source mtime below the compiled binary's, so cargo silently reuses a stale half-neutered binary and fabricates phantom failures) confirmed `build_row_nnz_honours_one_based_decode` and `phase0_one_based_two_blocks_eliminated` FAIL on a decode-inconsistent neuter, then pass on restore. `pounce-presolve` (216 lib + integration + 9 doctests, 0 failed) and `pounce-py` build green. See `## M27 detail`. |
| M28 | FBBT allocates and scans O(n_vars) per constraint per sweep: `vec![Interval::ENTIRE; n_vars]` + a `0..n_vars` apply loop for every constraint — O(m·n) per sweep when each tape touches a handful of variables | **FIXED** | **Cost confirmed by timing.** `run_fbbt` (`crates/pounce-presolve/src/fbbt/orchestrator.rs:159-197`) allocated a fresh `vec![Interval::ENTIRE; n_vars]` *per constraint* and then ran a `for j in 0..n_vars` apply loop over it — even though a constraint's tape typically mentions only a handful of variables. Per sweep that is `O(m·n_vars)` regardless of sparsity. **Reproduced by running code**: a timing probe (m = n constraints, each a single-`Var(i)` tape over a wide non-tightening bound) showed clearly quadratic scaling — `0.62 / 2.25 / 8.86 ms` at `n = 1000/2000/4000` (each `n`-doubling ≈ 4×, the O(n²) signature). **Fix**: hoist the per-variable scratch out of the loops and touch only the variables a constraint actually mentions. A reused `tighten: Vec<Interval>` (length n_vars, allocated once), a `last_seen: Vec<usize>` stamp array, a `touched: Vec<usize>` list, and a monotonic per-constraint `stamp` implement a "first `Var(j)` slot overwrites, later slots intersect" discipline with **no per-constraint reset** — the apply step iterates only `touched`. Variables absent from the tape keep an `ENTIRE` interval and can never tighten or be empty, so iterating `touched` is exactly equivalent to the old `0..n_vars` scan. Per-constraint work drops to O(tape length); per sweep to O(nnz). No public API or numerical contract changed. **Verified post-fix**: the same probe is near-linear — `0.073 / 0.119 / 0.191 ms` at `n = 1000/2000/4000`, a speedup that *grows* with `n` (8.5× → 46×), exactly the O(n)→O(1)-per-constraint improvement predicts. **Test**: `duplicate_var_slots_intersect` pins the subtle part — a variable in two structurally distinct `Var(0)` slots of one tape (`x²+x=6`, squared slot first/tight, linear slot second/loose) must end with the INTERSECTION; in a single sweep (`max_iter=1`, essential — a fixed point washes the difference out) the correct intersection gives `x_hi ≈ √6 ≈ 2.449`. The existing 64 FBBT tests (coupled-constraint iteration, soundness fuzz, row-mask, infeasibility, caps) all still pass unchanged. **Pre-fix vs neuter**: making later slots *overwrite* instead of *intersect* makes `duplicate_var_slots_intersect` FAIL (`x_hi = 6.0`, the loose linear slot), confirmed then restored. `pounce-presolve` (217 lib + integration + 9 doctests, 0 failed) and `pounce-py` build green. See `## M28 detail`. |
| M29 | LICQ structural check duplicates and degrades an existing primitive: per-row `vec![false; n]` allocation (O(m·n)) and recursive augmenting paths (stack-overflow risk on long chains, e.g. discretized dynamics), while the crate already has an iterative Hopcroft–Karp in `matching.rs` | **FIXED** | **Both degradations confirmed by running code.** `bipartite_matching_rank` (`crates/pounce-presolve/src/licq.rs:72-110`) ran its own Hungarian-style matcher: a `vec![false; n]` `seen` array allocated **per row** (O(m·n) total just to zero scratch) and a **recursive** `try_augment` whose depth equals the augmenting-path length. The crate already ships an iterative, BFS-layered Hopcroft–Karp (`matching.rs::hopcroft_karp`, O(E·√V), König-cross-checked) operating on `EqualityIncidence` — the LICQ matcher was a second, weaker copy. **Reproduced by running code**: (a) a temporary recursion-depth counter on `try_augment` over a staircase chain (`row 0:{0}`, `row i:{i−1,i}`, final row → last column only) measured max depth `= m−1` exactly (999 / 3999 / 15999 at m = 1000/4000/16000) — linear in chain length, so a chain of tens of thousands of rows overflows a normal 2–8 MB stack; (b) a timing probe on the m = n diagonal showed the per-row allocation scaling super-linearly (`0.023 / 0.069 / 0.245 ms` at n = 1000/2000/4000, the O(m·n) signature). **Fix**: delete `try_augment` and rewrite `bipartite_matching_rank` to pack the `EqRow` list into a CSR `EqualityIncidence` (out-of-range columns dropped, columns sorted+deduped exactly as `from_probe`) and call `hopcroft_karp(&inc).size`. Hopcroft–Karp prunes failed searches in BFS (no DFS at all when no augmenting path exists) and bounds its DFS recursion depth to the BFS layer distance (O(√V)), removing both the per-row allocation and the deep recursion. `licq_check`'s public verdict semantics are unchanged. **Verified post-fix**: all 7 existing LICQ tests (over-determined, empty-row, duplicate-singleton, distinct-singleton, augmenting-path) pass unchanged. **New tests**: `long_chain_does_not_overflow_stack` (m = 50 000 rows over m−1 touched columns + a phantom column so `m ≤ n`; the exact chain that drove the old matcher to depth ≈ 50 000) completes on the default 2 MB test stack and returns `StructuralRank(m−1)`; `long_chain_full_rank` (m = 20 000, m columns, perfect matching) returns `Full`, guarding against the fix capping long augmenting paths short. `pounce-presolve` (219 lib + integration + 9 doctests, 0 failed), no new clippy warnings, and `pounce-py` build green. See `## M29 detail`. |
| M30 | python: `curve_fit` covariance never projects onto the active *general-constraint* nullspace — `active_mask` covers variable bounds only, so an active equality between parameters is returned as the unconstrained covariance while labeled `reduced_hessian(projected)`, overstating variances and dropping induced anti-correlations | **FIXED** | **Bug confirmed by running code.** `_covariance` (`python/pounce/_curve_fit.py:1542-1547`) and its streaming twin `_stream_covariance` (1108-1112) handled the active set with `free = ~active_mask` — projecting out only **bound**-active columns. With `m_con > 0` but no active bound the branch fired (`if m_con > 0 or active_mask.any()`), computed `free = all-True`, and returned the **unconstrained** `s2·pinv(M)` while labeling it `reduced_hessian(projected)`. **Reproduced by running code**: a weighted line fit (`f = a·x + b`, `M = JwᵀJw`) under an active equality `a + b = c` — calling `_covariance` directly and checking the variance along the constraint gradient. Pre-fix `A·pcov·Aᵀ = 0.318` (should be 0: the binding relation is known exactly) and `pcov` was bit-identical to the unconstrained inverse, with the induced anti-correlation absent; the correct projected covariance carries `A·pcov·Aᵀ = 0` and a `-0.065` off-diagonal. **Fix**: thread the constraint plumbing already on `_FitProblem` (`jac_combined`, `g_combined`, `cl`, `cu`) into both covariance functions and project onto the **joint** active-set nullspace. `_active_constraint_jac` selects the binding general-constraint rows (equalities `cl==cu` always bind; inequalities within `tol` of a finite bound); `_projected_covariance` stacks those with unit rows `eⱼ` for active bounds, takes an orthonormal nullspace basis `Z` (SVD), and returns `s2·Z·pinv(ZᵀMZ)·Zᵀ`. For a bounds-only active set `Z` is the free coordinate subspace and this reduces **exactly** to the old `cov[ix_(free,free)] = s2·pinv(M[free,free])` (the prior behavior preserved, verified by the existing bound tests). When `m_con > 0` but every inequality is slack and no bound binds, nothing is projected and the source is honestly reported as `jacobian`. **Also**: the `_covariance` docstring now states it is the first-order (delta-method) asymptotic covariance — `M` is the Gauss-Newton Hessian and the constraints are linearized at `popt`, omitting the curvature term `ΣλᵢHᵢ` (zero for linear constraints, higher-order otherwise) — resolving the "Gauss-Newton comment assumes linear constraints" note; the module docstring's "projection onto the active-constraint nullspace" claim is now actually true. **Tests** (`python/tests/test_curve_fit.py`): `test_active_equality_constraint_projects_covariance` (in-memory) and `test_streaming_active_equality_projects_covariance` (streaming twin) fit a line under an active `a+b=1` equality and assert `cov_source == "reduced_hessian(projected)"`, zero variance along the constraint gradient (`g·pcov·g < 1e-9`), a negative `pcov[0,1]`, and a match to the closed-form `Z·pinv(ZᵀMZ)·Zᵀ`. **Pre-fix both FAIL** (`g·pcov·g ≈ 1.6e-3`, the unconstrained variance) — confirmed before the fix; post-fix both PASS. Full `test_curve_fit.py` (44) green, and `test_sensitivity.py`/`test_minimize.py`/`test_minima.py` (37) unaffected. See `## M30 detail`. |
| M31 | python: the issue-#112 indefinite-`P` guard fires only on `solve_qp` — every other host QP entry point (`solve_qp_batch`, `solve_qp_multi_rhs`, `QpFactorization`, `QpSensitivity`, `solve_socp`) and the jax/torch differentiable layers skip the PSD check, so a nonconvex `P` is solved by the convex IPM and returns a silently-wrong `status="optimal"` (or a constructed handle / a corrupt backward pass) | **FIXED** | **Bug confirmed by running code** (`/tmp/m31_verify.py`): an indefinite `P = diag(1,-1)` with box bounds fed to all six host entry points — only `solve_qp` raised; the other five returned `status="optimal"` or constructed a usable handle. **Fix**: a shared `_maybe_check_psd(P, c, check_psd)` helper (honoring `check_psd ∈ {None=auto, True, False}` with the `_PSD_CHECK_AUTO_MAX_N=1500` auto threshold) is threaded into all six host entry points, each of which gained a `check_psd` parameter; the jax/torch host forwards (`_forward_solve`, `_forward_solve_batch`, `_forward_solve_socp`) gained a `_guard_psd` that runs the same eigenvalue screen before building the `_pounce.QpProblem`. **Tests**: `test_qp_host.py` gained 7 tests — five `*_rejects_indefinite_p` (one per previously-unguarded entry point, `pytest.raises(ValueError, match="positive semidefinite")`), `test_check_psd_false_bypasses_guard_everywhere`, and `test_psd_p_still_solves_on_all_entry_points`; `test_qp_jax.py`/`test_qp_torch.py` each gained `test_indefinite_p_rejected_in_{forward,batch_forward}` (jax wraps the host `ValueError` but the "semidefinite" message survives). **Pre-fix the five rejection tests FAIL** (the unguarded points return `optimal`) — confirmed by neutering the guard; post-fix all PASS. Full QP suite green: `test_qp.py`/`test_qp_host.py`/`test_qp_jax.py`/`test_qp_torch.py`/`test_qp_sensitivity.py`/`test_socp.py` (82 passed). See `## M31 detail`. |
| M32 | rust(pounce-py): the `intermediate` TNLP callback (`crates/pounce-py/src/tnlp_bridge.rs:364-374`) (a) coerces a non-`bool` return via `res.extract::<bool>().unwrap_or(true)`, so a cyipopt-valid falsy int `0` (meaning *stop*) fails strict bool extraction and is read as *continue* — silently ignoring the user's stop; and (b) maps any callback exception to `Err(_) => false` with **no logging** (unlike the eval callbacks), so a crashing callback masquerades as a silent `User_Requested_Stop` | **FIXED** | **Both bugs confirmed by running code** (`/tmp/m32_verify.py`, after a `maturin build` of the worktree): an `intermediate` returning `0` at `iter_count>=1` was **ignored** pre-fix — the solve ran all 8 IPM iterations to `Solve_Succeeded` (`x→3`) instead of stopping. **Fix**: replace `res.extract::<bool>().unwrap_or(true)` with `res.is_truthy()?` (Python truthiness, matching cyipopt: `False`/`0`/`0.0`/`[]` stop, truthy continues; `None`/no-return still continues via the existing `is_none()` branch), and replace `Err(_) => false` with `Err(e) => { tracing::error!(target: "pounce::py", "pounce-py: intermediate(): {e}"); false }` so a raising callback leaves a trace like `objective`/`gradient`/… (verified: post-fix log line `ERROR pounce::py: pounce-py: intermediate(): RuntimeError: boom...`). **Tests** (`python/tests/test_problem.py`): `test_intermediate_falsy_return_stops[0,False,0.0,[]]` (all must yield `User_Requested_Stop` and not reach `x*=3`), `test_intermediate_truthy_return_continues[1,True,0.5,[0]]` (→`Solve_Succeeded`), `test_intermediate_no_return_continues`, and `test_intermediate_exception_aborts_with_user_stop`. **Fail-first confirmed** by swapping the pre-fix `.so`: `[0]`, `[0.0]`, `[[]]` FAIL (`Solve_Succeeded`) while `[False]` already passed — exactly the `extract::<bool>` gap; post-fix all 14 pass. Broader solve-exercising suite green (53 passed); `cargo clippy -p pounce-py` clean of new warnings. See `## M32 detail`. |
| M33 | python(pyomo-pounce): the Pyomo plugin's `_default_executable` (`pyomo-pounce/pyomo_pounce/pounce_solver.py:35-36`) resolves the solver only via `shutil.which("pounce")` despite depending on `pounce-solver`, which bundles the binary at a deterministic path (`pounce/bin/pounce`, exposed by `pounce._cli._bundled_binary`). A non-activated-venv run (cron, IDE runner, Jupyter kernel) with `<venv>/bin` off PATH reports the solver unavailable, or PATH shadowing picks up a stale system binary | **FIXED** | **Bug confirmed by running code**: with a bundled binary present at the deterministic path but PATH stripped of `pounce` (simulating cron/Jupyter), `_default_executable()` returned `None` — the solver reported unavailable. **Fix**: prefer `pounce._cli._bundled_binary()` when it `is_file()` (a lazy import guarded by `try/except` so a missing `pounce-solver` degrades gracefully), and fall back to `shutil.which("pounce")` for system installs / local cargo dev builds. **Tests** (`pyomo-pounce/tests/test_pounce.py`): `test_default_executable_prefers_bundled` (bundled present + PATH stripped → returns bundled path), `test_default_executable_falls_back_to_path` (no bundled → returns the PATH binary), `test_default_executable_none_when_nowhere` (neither → `None`), all via `monkeypatch` of `pounce._cli._bundled_binary` and `PATH`. **Fail-first confirmed** by reverting the method to `return shutil.which("pounce")`: `prefers_bundled` FAILS (`None != bundled`) while the fallback/none tests pass; post-fix all 7 plugin tests pass (the 3 solve-smoke tests run against the on-PATH binary, no skips). See `## M33 detail`. |
| M34 | python: default auto-routing in `pounce.minimize` costs O(n²) user-function evaluations before the solve. On the `auto` path the LP/QP router (`classify_and_extract`) and the SOCP/QCQP router (`classify_and_extract_socp`) both FD-fit the *same* objective at an *identical* probe set (same `seed=0`), so the objective is finite-differenced twice; for a problem that ends up on the NLP path this is pure overhead, and it was undocumented (`python/pounce/_route.py`, `_minimize.py:425,447-468`) | **FIXED** | **Bug confirmed by running code**: counting `fun` calls through `minimize` on a quartic (NLP route, n=5), the routing overhead (auto-path calls minus nlp-forced-path calls) was 520 = exactly 2× a single router's 260 probe calls — the SOCP router re-probed every point the QP router had already evaluated. **Fix**: wrap the router callables (`fun`/`jac`/`hess`/`g_combined`/`jac_combined`) in one shared point-keyed cache (`_route._point_cache`, keyed on the point's float64 bytes) inside the `route_kw` both routers receive, so the second router's probes are cache hits; the NLP fallback still calls the *original* callables, so the actual solve is unaffected. Also documented the routing cost and the `solver_selection="nlp"` opt-out in the `minimize` docstring. **Test** (`python/tests/test_minimize_autoroute.py::test_auto_route_probes_objective_once_not_twice`): asserts the auto-path routing overhead equals one router's probe count, not two. **Fail-first confirmed** by reverting the `_point_cache` wrapping: overhead = 520 ≠ 260 → test FAILS; post-fix 74 routing/minimize tests pass. See `## M34 detail`. |
| M35 | rust(pounce-py): session-style solves hold the GIL for the whole IPM run. `PySolver::solve` (`crates/pounce-py/src/solver.rs:80`), `QpFactorization::solve` and `QpSensitivity::new` (`crates/pounce-py/src/qp.rs`) call the Rust solver without `py.allow_threads`, unlike `PyProblem::solve` and the one-shot QP/SOCP entry points. `Solver` is the workhorse under `curve_fit` and the jax/torch hosts, so concurrent solves on multiple Python threads serialize | **FIXED** | **Bug confirmed by running code**: the QP path is pure Rust (no Python callbacks), so a `QpSensitivity` solve held the GIL *continuously* — a background watcher thread stalled 23.6 ms ≈ the full 31 ms solve, and 8 `QpSensitivity` solves across 8 threads took as long as serial (ratio 0.97) on a 14-core box. **Fix**: wrap each solve in `py.allow_threads`. The QP sites are pure Rust but hold non-`Send` linear-solver trait objects, so a transparent `SendGuard` (the same trick `PyProblem::solve` uses for its `Rc`s) crosses the GIL-release boundary; the closure runs on the calling thread so it never actually moves between OS threads. The NLP `PySolver::solve` uses the identical `SendGuard` pattern as `PyProblem::solve` (every `tnlp_bridge.rs` callback re-acquires the GIL via `Python::with_gil`, so re-entrancy is safe). **Test** (`python/tests/test_qp_sensitivity.py::test_qp_solve_releases_the_gil`): asserts 8 threaded solves finish in < 0.75× serial (skips on < 4 cores). Post-fix the watcher stall dropped to 4.5 ms and the threaded ratio to 0.39 (~2.5× speedup). **Fail-first confirmed** by swapping the pre-M35 `.so`: ratio 1.01 → test FAILS; post-fix all 41 QP + 112 NLP-session/sensitivity/curve_fit tests pass (one pre-existing, unrelated `test_socp.py` exp-cone failure reproduces identically on the pre-M35 `.so`). See `## M35 detail`. |
| M36 | rust(studio-core): the report-reader's `InputDescriptor` mirror (`crates/pounce-studio-core/src/report.rs:142-154`) is missing the `CbfFile` variant that the writer (`crates/pounce-solve-report/src/lib.rs:185-204`) emits as `"kind": "cbf-file"` for `.cbf` conic instances. serde's internally-tagged enum hard-fails on the unknown tag, so the *entire* solve report is rejected — CBF solve reports can't be loaded at all | **FIXED** | **Bug confirmed by running code**: rewriting a good fixture's `fair_metadata.input` to `{"kind":"cbf-file","path":…,"size_bytes":…}` and loading it via `SolveReport::from_json_str` failed with serde `unknown variant 'cbf-file', expected one of 'nl-file', 'builtin', 'tnlp-direct'`. **Fix**: add the `CbfFile { path, size_bytes }` variant to studio-core's `InputDescriptor`, mirroring the writer (kebab-case `"cbf-file"`; `path: String` matching the reader's other variants). No production code matches the enum exhaustively, so the addition is self-contained. **Test** (`crates/pounce-studio-core/tests/fixtures.rs::loads_cbf_file_input_descriptor`): loads a cbf-file report and asserts it decodes to `InputDescriptor::CbfFile` with the right path/size. **Fail-first confirmed**: pre-fix a load-only form of the test failed with the serde unknown-variant error; post-fix all 13 studio-core tests pass. See `## M36 detail`. |
| M37 | rust(cinterface): library UB — the sensitivity C API feeds NULL straight into `slice::from_raw_parts`. `IpoptSolverParametricStep` (`crates/pounce-cinterface/src/solver.rs:339,347`) and `IpoptSolverReducedHessian` (`:383`) accept a legal `n_pins == 0` call (which is allowed to pass NULL `pin_indices`/`deltas` — there is nothing to point at), but build the slices with `from_raw_parts(pin_indices, 0)` unconditionally. `from_raw_parts` requires its pointer be non-null and aligned *even for empty slices*, so `from_raw_parts(NULL, 0)` is UB; recent rustc's `-C debug-assertions` precondition checks turn it into a process abort. The rest of the crate gates this correctly (`IpoptSolverSolve` uses `if n_us > 0 { from_raw_parts } else { &[] }`) | **FIXED** | **Bug confirmed by running code**: a converged-session solve followed by `IpoptSolverParametricStep(solver, 0, NULL, NULL, dx_out)` aborts with SIGABRT — `unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be aligned and non-null` (the session check sits *before* the bad `from_raw_parts`, so a solve is required to reach it). **Fix**: a local `slice_or_empty(ptr, len)` helper that returns `&[]` when `len == 0` and only calls `from_raw_parts` otherwise — mirroring the `n_us > 0` gate already used in `IpoptSolverSolve` — applied to all three sites (the two `ParametricStep` slices + the `ReducedHessian` pins). An empty pin set is a well-defined no-op (zero perturbation → Δx ≈ 0, 0×0 reduced Hessian), so both calls now return `TRUE`. **Test** (`crates/pounce-cinterface/src/solver.rs::zero_pins_with_null_pointers_is_not_ub`): solves the 1-D quad to a session, then calls both entry points with `n_pins=0` + NULL pointers and asserts `TRUE`. **Fail-first confirmed** by reverting the `ParametricStep` slices to bare `from_raw_parts`: the test aborts (signal 6, SIGABRT, the non-null precondition message); post-fix all 43 pounce-cinterface lib tests pass, clippy clean of new warnings. See `## M37 detail`. |
| M38 | release: no tag-vs-manifest version check in any release workflow. `.github/workflows/release-crates.yml`, `release-pounce.yml`, `release-pyomo-pounce.yml` key off the tag prefixes `v*` / `python-v*` / `pyomo-pounce-v*` but never confirm the tag's version matches the manifest it publishes. Tagging `v0.5.0` with manifests at 0.4.0 makes the crates publish a silent green no-op (`publish-crates.sh` skips every crate as "already published" at 0.4.0) and the PyPI publish ship the stale 0.4.0 wheel under the 0.5.0 release | **FIXED** | **Bug confirmed by running code**: there was no guard at all — `check-release-consistency.sh` checks the three *manifests* agree with each other but never against the *tag*. Added `scripts/check_tag_version.py <tag-or-ref>`, which strips the longest matching release prefix (`pyomo-pounce-v`/`python-v`/`v`, longest-first so the PyPI tags aren't misread as the bare crates `v`), reads the first top-of-line `version = "..."` from the routed manifest (Cargo `[workspace.package]` / the two `pyproject.toml`s — same extraction as the consistency script), and exits 2 on mismatch / 3 on an unrecognized tag / 4 on an unreadable manifest. **Verified live**: against the repo at 0.4.0, `check_tag_version.py refs/tags/v0.5.0` fails with exit 2 and a TAG/MANIFEST MISMATCH message (the exact M38 scenario nothing previously caught), while `v0.4.0`/`python-v0.4.0` pass and `pyomo-pounce-v1.0.0` correctly routes to the pyomo manifest. **Wiring**: `release-crates.yml` gains a guard step before `Publish crates`; `release-pounce.yml`/`release-pyomo-pounce.yml` gain a `verify-version` job that the build jobs `needs:`, so a mismatch fails before the multi-platform wheel matrix runs. All three gate on `github.event_name == 'push'`, so manual `workflow_dispatch` dry-runs (no tag) skip the check (no-op pass). **Test** (`scripts/tests/test_check_tag_version.py`, mirroring the sibling `test_check_dep_publishability.py` standalone-unittest convention): 18 cases over synthetic manifests (stable across version bumps) covering prefix routing, longest-prefix precedence, prerelease suffixes, and the M38 mismatch → exit 2; the three workflows parse and the `verify-version → build → publish` dependency graph is well-formed. 25 `scripts/tests` total green. See `## M38 detail`. |
| M39 | ci: `pounce-hsl` is on the crates.io publish list but compiled by zero CI jobs. `.github/workflows/ci.yml:63,66,69` (clippy/build/test) all pass `--exclude pounce-hsl` because the crate FFI-links the licensed `libcoinhsl` (absent from CI), so its first compile is the `cargo publish` verify build mid-release — and it is 5th of 19 in the publish order, so the four crates ahead of it (pounce-common/-linalg/-linsol/-feral) are already irreversibly published when it fails | **FIXED** | **Bug confirmed by running code**: with a deliberate type error appended to `crates/pounce-hsl/src/lib.rs`, the current CI build command `cargo build --workspace --exclude pounce-hsl` finished green (exit 0) — the error was completely invisible to CI. Root-caused the exclusion: `pounce-hsl/Cargo.toml` has `links = "coinhsl"` + a `build.rs`, but `build.rs` degrades gracefully when `COINHSL_DIR` is unset (emits a warning and returns, compiling a plain rlib with no link directives), so the crate *type-checks* fine without HSL — only *linking* (build/test of a final artifact) needs the library. **Fix**: add a `cargo check -p pounce-hsl --all-targets --verbose` step to the `test` job (after Test). `cargo check` type-checks without linking; `--all-targets` also covers the test modules (which the excluded test job never compiles either). Verified: against the injected error this step fails with `E0308` (exit 101) — catching exactly what the build step missed; with the error reverted it passes (exit 0), COINHSL_DIR unset, emitting only the benign warning. The publish list position (5/19) and the four-crates-ahead claim were confirmed from `scripts/publish-crates.sh`. **Test/verification**: the fail-first demonstration is the injected-error A/B above (current CI build green vs new check exit 101); the live repo `cargo check -p pounce-hsl --all-targets` is clean; `ci.yml` parses and the new step is present in the `test` job. CI-only change; no crate source touched (the temporary error was restored via `cp`+`touch`). See `## M39 detail`. |
| L1 | algorithm: the final iterate is never convergence-tested at the `max_iter` boundary. `IpoptAlgorithm::optimize`'s main loop (`crates/pounce-algorithm/src/ipopt_alg.rs:1651-1656`) increments `iter_count` and breaks with `Maximum_Iterations_Exceeded` *before* calling `iterate()` again, so the convergence check never runs on the iterate produced by the final permitted step. A solve converging on exactly the `max_iter`-th iterate reports `Maximum_Iterations_Exceeded` where upstream Ipopt — whose `CheckConvergence` runs at the top of the loop, convergence-first — reports success; the `MaxIterExceeded` branch in `conv_check/opt_error.rs:233` is consequently dead (`data.iter_count` can never reach `max_iter`) | **FIXED** | **Bug confirmed by running code**: HS071 converges to `Solve_Succeeded` at `iter=8` with a generous budget; re-solving with `max_iter=8` reported `MaximumIterationsExceeded` at `iter=7` — the loop broke before the converged 8th iterate was ever tested. **Root cause**: the outer loop carried its own `if iter_count >= self.max_iter { break MaxiterExceeded }` that short-circuited *before* the next `iterate()` call, while the real convergence test (component tolerances **then** the `iter >= max_iter` gate) lives inside `iterate()` → `check_convergence_with_state`. Because the break fired first, `data.iter_count` topped out at `max_iter - 1`, so the in-`iterate()` `MaxIterExceeded` branch (`opt_error.rs:233`) never executed. **Fix**: drop the premature break — bump the counter and loop, letting the next `iterate()` run its convergence check. Termination is still guaranteed: once `iter_count` reaches `max_iter`, `check_convergence_with_state` returns `Converged`/`ConvergedToAcceptable` or `MaxIterExceeded`, never `Continue`. This matches upstream's top-of-loop, convergence-first ordering and takes the same number of steps (`max_iter`), adding only the missing final-iterate check. **Test** (`crates/pounce-algorithm/tests/optimize_hs71.rs::hs071_converges_exactly_at_max_iter_boundary`): finds HS071's natural convergence iteration `k`, re-solves with `max_iter=k`, asserts success + objective ≈ 17.014017. **Fail-first confirmed**: pre-fix the test fails with `MaximumIterationsExceeded (max_iter = 8)`; post-fix all 16 `optimize_hs71` tests pass and the full pounce-algorithm suite is green (lib 245 + all integration tests, 0 failures). See `## L1 detail`. |
| L2 | algorithm: claim that the tiny-step *dual* test (`crates/pounce-algorithm/src/ipopt_alg.rs:1041-1042`) is absolute where upstream Ipopt is relative (`1/(1+‖y‖∞)` scaling), unlike the primal half (`detect_tiny_step`, 1152-1172), causing `STOP_AT_TINY_STEP` to under-fire on large-multiplier problems | **NOT A BUG** (premise refuted by upstream source) | **Premise checked against the actual upstream source and found false.** Fetched `coin-or/Ipopt` (stable/3.14) `src/Algorithm/IpBacktrackingLineSearch.cpp`: it sets `tiny_step_last_iteration_` via `Number delta_y_norm = Max(IpData().delta()->y_c()->Amax(), IpData().delta()->y_d()->Amax()); if (delta_y_norm < tiny_step_y_tol_) { ... }` — a **direct absolute comparison, no `1/(1+‖y‖∞)` scaling**. pounce's `let dy_amax = delta.y_c.amax().max(delta.y_d.amax()); self.tiny_step_last_iteration = dy_amax < self.tiny_step_y_tol;` is an exact, faithful port. The primal/dual asymmetry the review flags (primal relative per-component `|δxᵢ|/(1+|xᵢ|)`, dual absolute) is **present in upstream** and intentional — confirmed independently by the option help text for `tiny_step_y_tol`: *"the step in the y variables is smaller than this threshold"* (absolute), versus `tiny_step_tol`'s *"in relative terms for each component"* (primal). **No code change, no regression test**: the alleged bug does not exist; changing 1041-1042 to a relative form would *introduce* a divergence from upstream, not remove one. Recorded per the "document issues that cannot be verified" rule — here the issue is verifiable and refuted. See `## L2 detail`. |
| L3 | algorithm: the probing μ-oracle hard-codes its centering cap `sigma_max = 100.0` (`crates/pounce-algorithm/src/mu/adaptive.rs:685-691`) instead of forwarding the user-set `sigma_max` option, so a user-set `sigma_max` reaches only the quality-function oracle — unlike upstream, where the probing oracle reads the same option | **FIXED** | **Bug confirmed by running code + upstream source.** Fetched `coin-or/Ipopt` (stable/3.14) `src/Algorithm/IpProbingMuOracle.cpp`: it reads `options.GetNumericValue("sigma_max", sigma_max_, prefix)` in `InitializeImpl` and caps `sigma = Min(sigma, sigma_max_)` — so the probing oracle **is** user-configurable upstream (the registered option help saying "Only used if mu_oracle is quality-function" is itself slightly inaccurate; behavior is what matters). pounce's adaptive free-mode update constructed `ProbingMuOracle { sigma_max: 100.0, … }` (hard-coded), while the quality-function branch correctly forwarded `self.sigma_max` (`adaptive.rs:705`). **Reproduced**: solving HS071 with `mu_strategy=adaptive`, `mu_oracle=probing` took **10** iterations at the default `sigma_max=100` *and* at `sigma_max=1e-6` — byte-identical, i.e. the user value was ignored. **Fix**: forward `self.sigma_max` (one line, `adaptive.rs:686`), matching upstream; updated the field doc-comment (104-108) to note it now also feeds the probing oracle. The registered option help string is left verbatim (it is upstream's). **Test** (`crates/pounce-algorithm/tests/optimize_hs71.rs::hs071_probing_oracle_honors_user_sigma_max`): solves HS071 via the probing oracle at default `sigma_max` vs `sigma_max=1e-6` and asserts the iteration counts differ. **Fail-first confirmed**: pre-fix both runs take 10 iters → `assert_ne!` fails ("both runs took 10 iters"); post-fix default=10 vs 1e-6=8 (both still `Solve_Succeeded`), so the option now reshapes the μ trajectory. Full pounce-algorithm suite green (lib 245 + all integration, `optimize_hs71` now 17, 0 failures). See `## L3 detail`. |
| L4 | algorithm: `golden_section` can return an unevaluated `-100.0` sentinel endpoint when `qmax <= 0` (`src/mu/oracle/quality_function.rs:540-554` with 730, 741); also `>=` in `qf_ok` makes the default `qf_tol = 0.0` flat-stop dead | **PARTIAL — one facet fixed, one not-a-bug** | **Two facets; verified against upstream.** Fetched `coin-or/Ipopt` (stable/3.14) `src/Algorithm/IpQualityFunctionMuOracle.cpp::PerformGoldenSection`. **Facet 2 (`>=` makes flat-stop dead): NOT A BUG.** Upstream's loop condition is `(1. - Min/Max) >= qf_tol` — the *same* `>=` as pounce line 499. With the default `qf_tol = 0.0` the term `(1 - qmin/qmax) >= 0` is always true for any non-degenerate bracket, so the qf-tolerance never stops the loop *in either codebase*; that is upstream's intended behavior (the qf_tol stop is opt-in via a positive `quality_function_eps`), not a pounce regression. **Facet 1 (unevaluated sentinel return): REAL, fixed.** `pick_sigma` always passes one endpoint with the `-100.0` sentinel (search-up → `q_up=-100` at line 730; search-down → `q_lo=-100` at 741). Upstream never lands on a sentinel because its loop lacks a `qmax > 0` guard, so a sentinel state (large positive ratio) keeps the loop alive until the slot is overwritten, and its post-loop else-branch re-evaluates `if( q_up < 0. )` anyway. pounce **adds** `qmax > 0.0 &&` to `qf_ok` (line 499, to dodge a divide-by-zero when every sample ≤ 0); that guard can force `qf_ok = false` on the first pass while an endpoint still holds the sentinel, routing it into the `width_ok && !qf_ok` branch (540-554) — which, unlike pounce's *own* else-branch (561-572) and upstream, did **not** re-evaluate, so it returned the unevaluated `-100.0` endpoint as the spurious minimum. **Reproduced** by a focused unit test on the pure `golden_section`: `q(σ) = -σ` on the interior/lo points (all ≤ 0 ⇒ `qmax ≤ 0`) but `+50` at the upper endpoint; search-up call returns σ=3 (the sentinel endpoint, true q=50, the bracket *maximum*) pre-fix. **Fix**: in the `width_ok && !qf_ok` branch re-evaluate any unmoved sentinel endpoint (`sigma_lo==sigma_lo_in && q_lo<0` / `sigma_up==sigma_up_in && q_up<0`) before selecting the minimum, mirroring the else-branch and upstream; refreshed the stale doc-comment (524-530) that wrongly claimed the sentinel could never reach this branch. **Test** (`quality_function.rs::tests::golden_section_never_returns_unevaluated_sentinel`): asserts the result is `< sigma_up`. **Fail-first confirmed** (pre-fix returns σ=3; post-fix returns an interior σ≈2.24). Full pounce-algorithm suite green (lib 246 + all integration, 0 failures). See `## L4 detail`. |
| L5 | algorithm: `max_cpu_time` actually measures wall time — `src/conv_check/opt_error.rs:257` via `pounce_common::utils::cpu_time()`'s documented wallclock fallback | **FIXED** | **Bug confirmed by reading + running code; fix verified against upstream.** `pounce_common::utils::cpu_time()` was literally `wallclock_time()` (a documented "phase 4 will wire in a real CPU clock" stub), so the `max_cpu_time` gate at `opt_error.rs:257` (`timing.overall_alg.live_cpu_time() >= self.max_cpu_time`) bounded **wall** time, not CPU time — diverging from upstream, whose `max_cpu_time` bounds process CPU. **Upstream reference**: fetched `coin-or/Ipopt` (stable/3.14) `src/Common/IpUtils.cpp::CpuTime()` — on Unix it returns `getrusage(RUSAGE_SELF).ru_utime` (process **user** CPU time); on Windows it uses `clock()` (which on the MSVC runtime is itself elapsed real time). **Fix**: implement the Unix path with `libc::getrusage(RUSAGE_SELF)` returning `ru_utime` seconds, matching upstream exactly; keep the `wallclock_time()` fallback for non-Unix (faithful to upstream's Windows `clock()` ≈ wall behavior). Added `libc = "0.2"` to `[workspace.dependencies]` and a `[target.'cfg(unix)'.dependencies] libc` entry to pounce-common (Unix-only, so non-Unix targets pull nothing new); no change to the publish list / release-consistency guard. **Test** (`pounce-common::utils::tests::cpu_time_excludes_sleep_but_counts_compute`, `#[cfg(unix)]`): (1) sleeps 300 ms and asserts `wall_delta − cpu_delta > 0.1 s` (CPU must not accrue while blocked), and (2) runs a 50 M-iter busy loop and asserts `cpu_delta > 0` (clock is live, not constant-zero). **Fail-first confirmed** by temporarily reverting `cpu_time()` to the wallclock alias: it then reported "cpu_time advanced 0.310s across a 0.310s sleep … gap was only −0.000s" and the assertion fired; restored, it passes. Full pounce-common suite green (58) and pounce-algorithm green (lib 246 + all integration); `cargo check --workspace --exclude pounce-hsl` clean. See `## L5 detail`. |
| L6 | algorithm: dead/divergent duplicates of filter acceptance predicates — `src/line_search/filter_acceptor.rs:171-179` (no round-off slack, unlike the live path at 292-300) and 199-229 (parameterized `obj_max_inc` while the live path hard-codes 5.0) | **FIXED** | **Both divergences confirmed by reading + running code; unified.** Two near-duplicate copies of the filter sufficient-progress / iterate-acceptance logic had drifted from the live `check_acceptability` path. **(a) `is_sufficient_progress` (171-179)** used bare `<` where the live path (then 292-300) uses `compare_le` (a `<=` carrying `10·eps·|basval|` round-off slack); the helper was also **dead** (`grep` shows no caller — only `is_acceptable_to_current_iterate` is live, from `pounce-restoration/src/conv_check.rs:163`). On the round-off boundary (`phi_trial − phi == −gamma_phi·theta`, common near a solution where φ is flat and the descent is summation-noise-sized) the bare `<` rejects a step `compare_le` accepts — the same flat-objective failure mode documented on `armijo_holds`. **(b)** the live `check_acceptability` rapid-barrier-increase guard hard-coded `5.0`, while the parameterized `is_acceptable_to_current_iterate` (the restoration-live copy) reads an `obj_max_inc` argument — so the two paths would diverge for any non-default `obj_max_inc`. **Fix**: (1) rewrote `is_sufficient_progress` to use `compare_le` (now identical to the live OR-test) and made it the **single source of truth** — both `check_acceptability` and `is_acceptable_to_current_iterate` now delegate their sufficient-progress test to it; (2) added an `obj_max_inc` field to `FilterLsAcceptor` (default 5.0) and switched the live guard from the literal `5.0` to `self.obj_max_inc`, so the regular-phase and restoration paths share one cap. The live regular-phase behavior is **byte-identical** (it already used `compare_le` and 5.0 = the field default), so no integration regression. **Tests** (`filter_acceptor::tests`): `is_sufficient_progress_accepts_round_off_boundary_like_live_path` builds the φ-branch equality boundary and asserts the helper accepts it; `check_acceptability_honors_obj_max_inc_field` drives a ~1e7 barrier jump (log10≈7) and asserts Reject at the default cap 5.0 (threshold 6) but Accept once `obj_max_inc=10.0` (threshold 11). **Fail-first confirmed** by temporarily reverting both edits (bare `<` and literal `5.0`): both new tests fail; restored, both pass. Full pounce-algorithm green (lib **248** + all integration) and pounce-restoration green (105), confirming the dedup is regression-free. See `## L6 detail`. |
| L7 | algorithm: watchdog revert applies the current-direction fraction-to-boundary cap to the snapshot direction — `src/line_search/backtracking.rs:725-737`; the correct stored cap is `#[allow(dead_code)]`. Rescued by backtracking, but wastes evaluations post-watchdog | **NOT A BUG** (premise refuted by upstream source) + dead field removed | **Premise checked against the actual upstream source and found false.** Fetched `coin-or/Ipopt` (stable/3.14) `src/Algorithm/IpBacktrackingLineSearch.cpp`. In `FindAcceptableTrialPoint`, when the watchdog trial cap is exceeded the code does `StopWatchDog(actual_delta); skip_first_trial_point = true;`, and the next `DoBacktrackingLineSearch` executes `if( skip_first_trial_point ) { alpha_primal *= alpha_red_factor_; }` — it multiplies the **existing** `alpha_primal` (the *current* direction's `alpha_primal_max`, set fresh at the top of this outer iteration's call) by the reduction factor and does **NOT** recompute a fraction-to-the-boundary cap from the reverted snapshot delta. `StopWatchDog` only restores `actual_delta` to the snapshot (`actual_delta = watchdog_delta_->MakeNewContainer()`); it does not touch `alpha_primal`. pounce's `handle_watchdog_failure` re-runs `run_alpha_loop(&snap_delta, alpha_init, …, skip_first=true)`, which starts at `alpha_init * alpha_red_factor` (`backtracking.rs:842-843`) where `alpha_init` is the current direction's FTB cap — an **exact** match to upstream. The "correct stored cap" the review points to (`watchdog_alpha_primal_test`) is a misread: upstream's `watchdog_alpha_primal_test_` is the **acceptor's** frozen Armijo *test* step length (used inside `IpFilterLSAcceptor` when in watchdog), not a line-search restart cap, so there is no upstream behavior that would consume a snapshot FTB cap here. The "wastes evaluations when the snapshot direction has a tighter boundary" cost, to the extent it exists, is present in upstream too (both backtrack from the over-large start). **No behavioral change** is warranted; switching the restart to a snapshot-recomputed cap would *introduce* a divergence from upstream. **Cleanup done**: pounce's `watchdog_alpha_primal_test` field was genuinely dead (written in `start_watchdog`, never read; carried `#[allow(dead_code)]`). Removed the field, its initializer, and the `aff_step_alpha_primal_max` computation in `start_watchdog`, and added a comment at the revert site documenting the upstream-faithful `alpha_init` choice so the site is not re-flagged. **Verified by running code**: `cargo build -p pounce-algorithm` clean (no dead-code warning), full suite green (lib **248** + all integration, 0 failures) — the watchdog revert path is exercised by the HS/integration solves (e.g. PFIT3/PFIT4/scon1dls noted in the code comments), confirming the removal is regression-free. Recorded per the "document issues that cannot be verified" rule — here the issue is verifiable and refuted. See `## L7 detail`. |
| L8 | linsol: Ruiz scaler's 0/1-based auto-detection misclassifies a 0-based triplet whose index 0 carries no entries (`crates/pounce-linsol/src/ruiz.rs:117-129`); factors land on the wrong rows. Applied consistently, so result quality degrades rather than correctness; the only in-tree caller is safe (1-based) | **FIXED** | **Bug confirmed by reading + running code (latent: no live caller hits it).** `compute_sym_t_scaling_factors` auto-detected the index base with a **min-only** rule: `let offset = if min_idx >= 1 { 1 } else { 0 }`. A 0-based triplet whose row 0 is structurally empty has every index `>= 1`, so `min_idx >= 1` → it was treated as **1-based** and `airn[k] - 1`/`ajcn[k] - 1` shifted every entry down one row; the equilibration factors then landed on the wrong rows (row 0 received the factor meant for row 1, the true last row was never scaled). **Reproduced** with a focused unit test: `K = diag([0, 4, 9])` stored 0-based (entries on rows/cols 1,2; row 0 empty, `min_idx==1`, `max_idx==2==n-1`). Pre-fix the detector picked offset 1 and `s = [0.5, 0.333, 1.0]` — the factor for `K_11=4` leaked onto the empty row 0 and `K_22=9` was left unscaled. **Fix**: detect the base from **both** index extremes, which are individually decisive for an n×n matrix — a 1-based triplet never references index 0, and a 0-based triplet never references index n (valid range `[0, n-1]`). New rule: `min_idx == 0 ⇒ 0-based`; else `max_idx >= n ⇒ 1-based`; else `max_idx == n-1 ⇒ 0-based` (full 0-based coverage, the case the old rule botched); else fall back to the historical 1-based assumption. The in-tree 1-based caller (indices `1..=n`, `max_idx == n`) and the existing `fortran_index_style` / 0-based tests are unchanged. **Test** (`ruiz::tests::zero_based_with_empty_first_row_is_not_misread_as_fortran`): asserts row 0 keeps `d=1` and `K_11`,`K_22` equilibrate to ≈1. **Fail-first confirmed** by temporarily reverting to the min-only rule: the test fails with `empty row 0 must keep d=1, got 0.5`. Restored, full `pounce-linsol` suite green (18 + 1, 0 failures). See `## L8 detail`. |
| L9 | linsol: the KKT-dump diagnostic disables its one-shot via `unsafe std::env::remove_var("POUNCE_DBG_KKT_DUMP")` (`crates/pounce-linsol/src/t_sym_solver.rs:197-243`), which is unsound — `setenv`/`unsetenv` is not thread-safe and feral runs solves under rayon, so a concurrent env read can race the unset | **FIXED** | **Bug confirmed by reading the code + the threading model.** The dump block read `POUNCE_DBG_KKT_DUMP`, and after dumping called `unsafe { std::env::remove_var(...) }` to ensure a single dump. `std::env::{set_var,remove_var}` mutate the process environment via `setenv`/`unsetenv`, which glibc/musl do **not** make thread-safe against concurrent `getenv`; Rust 2024 marks them `unsafe` for exactly this reason. pounce-feral drives multiple solves in parallel through rayon, so one solver's `remove_var` can race another thread's env read (in this crate or any dependency that reads env, e.g. logging) — UB, not merely a lost dump. **Fix**: stop mutating the environment entirely. The env var is now **read-only** (gates whether dumping is requested); the one-shot guarantee moves to a lock-free atomic claim. Extracted a free fn `claim_kkt_dump(n_call, skip, &DUMPED) -> bool` that returns `false` while `n_call < skip` (the existing skip-N-calls knob) and otherwise `!dumped.swap(true, SeqCst)` — exactly one caller across all threads ever sees `true`. Statics are now `CALL_COUNT: AtomicUsize` + `DUMPED: AtomicBool` (the old `WARNED` flag folded in); no `unsafe`, no env writes. **Tests** (`t_sym_solver::tests`): `claim_kkt_dump_is_one_shot_after_skip` (sequential/deterministic — calls below `skip` return false, the first at/after `skip` returns true, all later return false) and `claim_kkt_dump_claims_exactly_once_under_concurrency` (32 threads + `Barrier`, asserts **exactly one** winner). **Fail-first confirmed** by making the helper non-one-shot (always-claim): both tests fail; restored, full `pounce-linsol` suite green (20 + 1, 0 failures). See `## L9 detail`. |
| L10 | hsl: 32-bit index arithmetic in MA57 sizing has no overflow guard (`crates/pounce-hsl/src/ma57.rs:263, 294-297, 434`) — `5*N + NE + max(N,NE) + 42` overflows i32 near ne ≈ 3×10⁸ and converts to an absurd allocation/abort instead of a clean `FatalError` | **FIXED** | **Bug confirmed by reading code + a provable arithmetic fact, and fixed with linked MA57 verification.** `Index = i32` (`pounce-common::types`). Three sizing sites computed lengths in i32: (a) symbolic `self.lkeep = 5*n + ne + n.max(ne) + 42` and `iwork = vec![0; (5*n) as usize]` (cpp:536) — for n=ne the leading term is `7·n`, so it exceeds `i32::MAX` (2.147×10⁹) once n ≳ 3.07×10⁸; in **release** the i32 sum wraps to a negative length that `as usize` turns into an enormous allocation, in **debug** it panics on overflow; (b) the `pre_alloc`-scaled suggested sizes `(info[8] as f64 * scale).ceil() as Index` — the float→int cast *saturates* to `i32::MAX` (Rust ≥1.45, so no wrap) but still yields an i32::MAX-element allocation; (c) backsolve `lwork = n * nrhs` — same i32 multiply overflow. **Fix**: extracted two pure helpers — `ma57_symbolic_sizes(n, ne) -> Option<(lkeep, liwork)>` (computes in i64, returns `None` when either exceeds `i32::MAX`) and `ma57_scaled_size(base, scale) -> Option<Index>` (validates the scaled length fits, never shrinks below MA57's own `base`); the backsolve widens `n*nrhs` to i64 with an `i32::MAX` check. Each out-of-range case now returns `ESymSolverStatus::FatalError` instead of allocating/aborting. **Tests** (`ma57::tests`): `ma57_symbolic_sizes_guards_i32_overflow` (exact small sizing; n=ne=3×10⁸ fits → `Some`; n=ne=3.5×10⁸, i.e. 2.45×10⁹, → `None`) and `ma57_scaled_size_guards_overflow_and_floors_at_base` (1.05× growth, `scale<1` floors at base, `Index::MAX-1` scaled up → `None`). **Verified by running code**: built+linked against a local CoinHSL (`COINHSL_DIR`, kept out of the repo for licensing) — full `pounce-hsl` suite green (12 lib + 3 integration). **Fail-first confirmed** by stripping both guards (return the i64→i32 `as` cast unconditionally): the overflow test fails (`is_none()` false) and the scaled test fails (`Some(2147483647)` vs `None`); restored, all green. **CI note**: `pounce-hsl` is `--exclude`d from the CI build/test/clippy jobs (needs proprietary HSL), so these tests run only where CoinHSL is installed — verified locally here. See `## L10 detail`. |
| L11 | perf: per-solve workspace allocations on the factor-once/solve-many hot path. MA57 backsolve allocates a fresh `vec![0.0; n*nrhs]` MA57C workspace every call (`crates/pounce-hsl/src/ma57.rs:457`), and feral's `backsolve` allocates an owned result `Vec` per solve (`crates/pounce-feral/src/lib.rs:559-577`); both run once per IPM iteration against a single factorization | **PARTIAL — ma57 fixed, feral blocked by external API** | **ma57 half — Bug confirmed by reading code + upstream; FIXED.** MA57 backsolve built a `let mut work: Vec<Number> = vec![0.0; lwork as usize];` on every call, where `lwork = n*nrhs`; in the IPM the matrix is factored once and back-solved every iteration, so this is a per-iteration heap allocation + zero-fill of pure scratch. Upstream (`coin-or/Ipopt` stable/3.14 `src/Algorithm/LinearSolvers/IpMa57TSolverInterface.cpp::Solve`) passes MA57C an **uninitialized** `new double[lwork]` and treats it as scratch (never reads it before MA57C writes), so the zero-fill is unnecessary and the buffer can be reused across solves. **Fix**: cache the workspace as a struct field `work: Vec<Number>` (init `Vec::new()`); backsolve does `self.work.resize(n*nrhs, 0.0)` (a no-op once large enough — no allocation in the solve-many hot path) and passes `self.work.as_mut_ptr()` to MA57C. Stale contents are fine because MA57C uses it as scratch. **Test** (`ma57::tests::backsolve_reuses_workspace_across_repeated_solves`): factors `A=[[2,1],[1,3]]` once (`multi_solve(true,…)`), captures `s.work.capacity()` after the first solve, then runs 3 more `multi_solve(false,…)` with different RHS asserting correct results AND `s.work.capacity()` unchanged (no realloc). **Verified by running code**: built+linked against local CoinHSL (`COINHSL_DIR`, kept out of the repo for licensing) — full pounce-hsl suite green (13 lib + 3 integration). **Fail-first confirmed** by reverting to the per-solve `vec!`: the capacity-stability assertion fails (`left: 0, right: 2`); restored, all green. **feral half — NOT FIXABLE in-tree (external API).** feral's `backsolve` calls `self.solver.solve(rhs)` / `solve_many` / `solve_refined` on the external pinned `feral::Solver`, each of which **returns an owned `Result<Vec<f64>>`** and is then `copy_from_slice`d into the caller's buffer. The allocation lives inside the external crate; `feral::Solver` exposes no in-place solve (`solve_into`) and its `last_factors` is private with no accessor, so the owned-`Vec` allocation cannot be removed without an upstream feral change (`solve_into` or a `factors()` accessor). Documented here per the "document issues that cannot be verified/fixed" rule; the in-tree (ma57) half is fixed. See `## L11 detail`. |
| L12 | feral: `FERAL_PIVTOL` breaks the documented `POUNCE_FERAL_*` env-var convention (`crates/pounce-feral/src/lib.rs:215-218`); `POUNCE_FERAL_PIVTOL` is silently ignored | **FIXED** | **Bug confirmed by running code.** `FeralConfig::from_env` reads six knobs under the `POUNCE_FERAL_*` prefix (`CASCADE_BREAK`, `FMA`, `REFINE`, `SINGULAR_PIVOT_FLOOR`, `ORDERING`, `SCALING`) but read the pivot threshold from the bare **`FERAL_PIVTOL`** — off-convention — and the `from_env` docstring didn't list pivtol at all, so a user following the documented `POUNCE_FERAL_*` convention sets `POUNCE_FERAL_PIVTOL` and it is silently ignored. **Reproduced** with a throwaway example calling `FeralConfig::from_env()`: `POUNCE_FERAL_PIVTOL=0.3` → `pivtol = 1e-8` (ignored), while `FERAL_PIVTOL=0.4` → `pivtol = 0.4` (legacy works). **Fix**: extracted a pure helper `resolve_pivtol_env(pounce, legacy) -> f64` that prefers `POUNCE_FERAL_PIVTOL` (the convention) and keeps the bare `FERAL_PIVTOL` only as a **deprecated legacy alias** (back-compat), falling through unparseable/unset values to the `1e-8` default. `from_env` now passes both env vars to it. Pure-helper design deliberately avoids mutating the process environment in the test (the rayon-parallel solves make env mutation a data race — the same hazard fixed in L9). Updated the `from_env` docstring (now lists `POUNCE_FERAL_PIVTOL`), the in-code comment at the `pivot_threshold` assignment, and the `feral_pivtol` OptionsList option help (`upstream_options.rs:1029`) to name `POUNCE_FERAL_PIVTOL` (preferred) + `FERAL_PIVTOL` (deprecated alias). **Test** (`pounce-feral` `tests::resolve_pivtol_env_honors_pounce_convention`): convention name read; legacy honored when convention unset; convention wins when both set; default with neither; unparseable falls through. **Fail-first confirmed** by reverting the helper to legacy-only (ignore the pounce arg): the test fails (`left: 1e-8, right: 0.3`); restored, full pounce-feral suite green (15 lib) and pounce-algorithm green (248 lib). `cargo fmt`/`clippy` (correctness/suspicious) clean on both crates. See `## L12 detail`. |
| L13 | qp/restoration: doc/code sign mismatches in restoration formulas (code right, docs wrong): `resto_nlp.rs:6-7` (`c − n + p` vs implemented `c + n − p`), `resto_resto.rs:16-21` (wrong quadratic for the stated root) | **FIXED** (docs corrected; code already right + matches upstream) | **Both premises confirmed by reading code + verifying the code against upstream, then correcting the docs.** **(1) Constraint signs**: `restoration_constraint_{c,d}` (`resto_nlp.rs:895,907`) and the `eval_c`/`eval_d` doc-comments implement `c_resto = c_orig + n_c − p_c` (and `d_orig + n_d − p_d`), but the module-level doc said `c(x) − n_c + p_c = 0` / `d(x) − n_d + p_d − s = 0` — slack signs swapped. The implemented `c + n − p` is **correct**: it matches upstream `IpRestoIpoptNLP` (`p_c = c(x) + n_c` ⇒ `c + n − p = 0`, verified by WebFetch) and the existing tests `constraint_{c,d}_combines_orig_n_p_with_correct_signs` already lock it. **(2) Quadratic**: the closed-form slack reset (`resto_resto.rs::compute_n_p`) computes `v = a + sqrt(a²+b)` with `a = mu/(2ρ) − 0.5·c`, `b = c·mu/(2ρ)` — **identical to upstream** `IpRestoRestoPhase.cpp::solve_quadratic` (verified by WebFetch quoting the literal body `v=a; v=v*v; v+=b; v=sqrt(v); v+=a` ⇒ `a + sqrt(a²+b)`). But the module doc stated this root solves `v² + 2·a·v − b = 0`, whose root is actually `−a + sqrt(...)`. The root `a + sqrt(a²+b)` solves `v² − **2·a·v** − b = 0` (substitute: `(v−a)² = a²+b`). Confirmed by first-principles derivation (minimize `ρ(2n+c) − μln n − μln(c+n)` ⇒ `n² + (c−2·half)n − b = 0`, linear coeff `= −2a`) and a c=0 sanity check (true minimizer `n = μ/ρ = 2·half`; code gives `half+half = 2half`, the doc's `−a+sqrt` form gives 0 — wrong). **The same sign error was mirrored in the `resto_resto` test name/comments** (`quadratic_root_satisfies_v2_plus_2av_minus_b_zero`) even though its *assertion* already used the correct `v*v − 2av − b`. **Fix (docs/labels only — no code change)**: corrected the constraint signs in `resto_nlp.rs` and the sibling `resto_alg_builder.rs` module doc; corrected the quadratic to `v² − 2·a·v − b = 0` in `resto_resto.rs` with a short derivation note; renamed the test to `quadratic_root_satisfies_v2_minus_2av_minus_b_zero` and added an assertion that the wrong `v² + 2av − b` form is clearly non-zero (`> 1e-4`) so the corrected sign is regression-locked both ways. **Verification**: full pounce-restoration suite green (105 lib + integration), incl. the renamed test and the pre-existing sign tests; `init.rs`'s quadratic was already correct (`n*n − 2.0*a*n − b`). The "fail-first" here is the provable doc-vs-upstream contradiction (the doc's `+2av`/`−n+p` forms do not hold for the code's actual, upstream-matching values). `cargo fmt`/`clippy` clean. See `## L13 detail`. |
| L14 | qp: the inertia-control retry loops recover from a linear-solver failure by substring-testing the error message for `"inertia"`/`"singular"` (`solver.rs:123,142`; `schur.rs:275,297`), a fragile case-sensitive match that silently misses the capitalized `Debug`-formatted `ESymSolverStatus` (`Singular`/`WrongInertia`) emitted by `LinearSolver::resolve`'s catch-all (`factor.rs:172`) — those failures propagate as unrecoverable instead of triggering a shift retry | **FIXED** | **Bug confirmed by reading + running code.** The §4.5 inertia-control loops in both the dense (`solver.rs`) and Schur (`schur.rs`) QP paths decide whether a `QpError::LinearSolverFailure(msg)` is recoverable (retry with a larger Hessian-diagonal shift) by `msg.contains("inertia") || msg.contains("singular")` — a **case-sensitive** substring test, duplicated at four sites. The factorize path produces lowercase messages (`"...inertia..."`), so those match. But `LinearSolver::resolve`'s catch-all (`factor.rs:172`) formats the backend status with `Debug`: `format!("resolve backend status: {other:?}")`, and `ESymSolverStatus`'s variants are **capitalized** (`Singular`, `WrongInertia`) — so a resolve-path singular/wrong-inertia failure yields `"resolve backend status: Singular"`, which `contains("singular")` **misses**. The recoverable failure then propagates as a hard error instead of triggering the shift retry that would rescue the solve. **Fix**: centralized the recoverability decision in one predicate `QpError::is_recoverable_factorization_failure()` (`error.rs`) that lowercases the message before testing (`m.contains("inertia") || m.contains("singular")`), and routed all four matchers through it (`solver.rs:123,142`; `schur.rs:275,297`), removing the duplicated inline substring tests. **Test** (`pounce-qp` `tests::refinement_unit::recoverable_factorization_failure_is_case_insensitive`): asserts the predicate accepts both the lowercase factorize-path messages **and** the capitalized resolve-path Debug strings (`"resolve backend status: Singular"`, `"...WrongInertia"`), and rejects non-recoverable failures (`"backend reported fatal error"`, `"resolve called before factorize"`) and non-`LinearSolverFailure` variants (`DimensionMismatch`). **Fail-first confirmed** by reverting the predicate to the case-sensitive `msg.contains(...)`: it fails on `"resolve backend status: Singular"` (`assertion failed: ...is_recoverable_factorization_failure()`); restored, full pounce-qp suite green (78 lib + 1 + 5 integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L14 detail`. |
| L15 | qp: `ElasticReformulation::original_inertia()` hardcodes `Psd` (`elastic.rs:169-175`), making the `Indefinite` arm of `as_qp`'s inertia match dead — an indefinite original problem is solved through the augmented elastic problem as if PSD; `solve_elastic` hard-calls `solve_general` (`solver.rs:1087`), ignoring `opts.use_schur_updates` | **FIXED** | **Both bugs confirmed by reading + running code (pounce-internal §4.3/§4.5 design, not an upstream-divergence).** **(1) Dead inertia arm.** `ElasticReformulation::build` discarded `qp.hessian_inertia`, and `original_inertia()` unconditionally returned `HessianInertia::Psd`. `as_qp` (`elastic.rs:162-165`) maps the original inertia onto the augmented problem with `Psd|Unknown => Psd`, `Indefinite => Indefinite` — but since `original_inertia()` could never return `Indefinite`, the augmented problem was *always* marked `Psd`, so an indefinite original `H` was solved as if PSD (skipping the §4.5 inertia-control assumption). **Fix**: `build` now captures `qp.hessian_inertia` into a new `orig_inertia` field and `original_inertia()` returns it; the augmented Hessian is block-diag(`H_orig`, 0) so it shares `H_orig`'s definiteness category (zero slack diagonals never introduce negative curvature), and the existing `as_qp` match now correctly propagates `Indefinite` while collapsing `Psd`/`Unknown` to `Psd`. **(2) `use_schur_updates` ignored.** The top-level `solve` dispatches between `solve_general_schur` (when `opts.use_schur_updates`) and `solve_general` (`solver.rs:1587-1591`), but `solve_elastic`'s recursive solve hard-called `solve_general`, so an infeasible problem solved with `use_schur_updates = true` silently fell back to the refactor path. **Fix**: `solve_elastic` now mirrors the same dispatch; both inner solvers bypass the `solve` feasibility audit, so the no-re-audit / no-recovery-loop property is preserved (comment updated). **Tests**: `elastic_unit::as_qp_propagates_original_hessian_inertia` (Indefinite original ⇒ augmented `Indefinite`; Psd/Unknown ⇒ `Psd`) and `analytical::l15_elastic_honors_use_schur_updates` (the `problem_5` infeasible QP solved with `use_schur_updates = true` returns the same minimal-l1 certificate **and** records `n_schur_updates > 0`, proving the Schur path ran inside the elastic recovery — the refactor path leaves it 0). **Fail-first confirmed** by reverting both edits (`original_inertia` → hardcoded `Psd`; dispatch → `solve_general` only): both tests fail (inertia `Indefinite != Psd`; `n_schur_updates == 0`). Restored, full pounce-qp suite green (80 lib + 1 + 5 integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L15 detail`. |
| L16 | sensitivity: `clamp_step_to_bounds` panics (index OOB) on non-dense bound vectors instead of the documented no-op (`boundcheck.rs:64-78,106-112`); and `dv.values()` (here + the `dense_to_vec` siblings in `solver.rs:408`, `convenience.rs:438`) trips `DenseVector::values`'s `!homogeneous` debug_assert where `expanded_values()` is the safe accessor | **FIXED** | **Both bugs confirmed by reading + running code (pounce sIPOPT port; the `values`/`expanded_values` homogeneous-value distinction is pounce-internal).** **(1) OOB panic on non-dense bounds.** `compressed_values` returns an empty `Vec` when the bound `dyn Vector` is not a `DenseVector` (documented contract: "silently no-ops"). But the clamp loops then indexed `bounds[compressed_i]` for every entry of the bound *expansion matrix* — so a non-dense `x_l`/`x_u` paired with a non-empty `px_l`/`px_u` panics `index out of bounds: the len is 0 but the index is 0` instead of no-opping. **Fix**: replaced both `bounds[compressed_i]` accesses with `bounds.get(compressed_i)` + `continue` on `None`, honoring the no-op contract (also covers a bounds slice shorter than the expansion). **(2) Homogeneous debug_assert.** `DenseVector::values()` carries `debug_assert!(self.initialized && !self.homogeneous)` (mirrors upstream's `DBG_ASSERT` in `DenseVector::Values() const`); a homogeneous bound vector — e.g. every lower bound 0, stored as a scalar with no materialized slice — makes `values()` panic in debug/test builds. Three sites used `dv.values().to_vec()`: `boundcheck.rs::compressed_values`, and the `dense_to_vec` helpers in `solver.rs` and `convenience.rs`. **Fix**: switched all three to `expanded_values()`, which materializes the scalar for a homogeneous vector and clones otherwise. **Tests** (`boundcheck::tests`): `clamp_handles_homogeneous_bounds_without_panicking` (homogeneous lower bound built via `Vector::set`; asserts no panic + correct single clamp) and `clamp_is_noop_on_non_dense_bounds` (a 1-block `CompoundVector` — the only other `dyn Vector` impl — as `x_l`; asserts 0 clamps, `dx` untouched). **Fail-first confirmed** by reverting both fixes: the homogeneous test panics at `dense_vector.rs:131` (`assertion failed: self.initialized && !self.homogeneous`) and the non-dense test panics at `boundcheck.rs` (`index out of bounds: the len is 0 but the index is 0`). Restored, full pounce-sensitivity suite green (45 lib + integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L16 detail`. |
| L17 | sensitivity: `IndexPCalculator::schur_matrix` drops the B-row sign and caches P columns by column index only, so a `−1` B row mis-signs its Schur entry and two A rows selecting the same column with opposite signs share one wrong-signed cached column (`p_calculator.rs:150-166,191-199`) | **FIXED** | **Both bugs confirmed by reading + running code, and verified against upstream sIPOPT (`SensIndexPCalculator.cpp`) — both behaviors *mirror* upstream but are only *reachable* in pounce because `from_parts`/`set_from_*` accept `−1` signs and duplicate columns, where production `IndexSchurData` is `+1`-only with unique columns.** **(1) B-row sign dropped.** Upstream `GetSchurMatrix` writes `S[i,j] = −P[B_colᵢ, A_colⱼ]` indexing `P` by `B`'s column index alone, never reading `B`'s ±1 factor. `schur_matrix` faithfully copied this (`let (b_idx_vec, _facs) = b.multiplying_row(i)?; … = -p_col[b_col]`), discarding `_facs`. So a `B` row carrying `−1` produced a Schur entry with the wrong sign. **Fix**: bind the factor (`b_facs`) and write `S[i,j] = −b_facs[0]·P[…]`. **(2) Duplicate-column cache conflation.** `compute_p` keyed the `p_cols` cache by column index only (`contains_key(&col)` / `insert(col, …)`) while the stored column *bakes in* A's sign (`K⁻¹(sign·e_col)`). Two A rows selecting the same column with opposite signs → the second hit `contains_key` on the first and silently reused the `+`-signed column for the `−` row. **Fix**: key the cache by `(col, sign)`; `schur_matrix` looks columns up by the same `(a_col, a_sign)` key. Public `p_columns()` return type changes `HashMap<Index,…>` → `HashMap<(Index,Index),…>` (only the test suite consumes it). **Tests** (`p_calculator::tests`): `schur_matrix_honors_negative_b_sign` (A col 0 +1, B col 1 −1 ⇒ `S[0,0] = −(−1)·P[1,0] = +½`, the buggy path yields −½) and `compute_p_distinguishes_same_column_opposite_signs` (A = col 1 twice with +1/−1 ⇒ both `(1,+1)` and `(1,−1)` cached as exact negatives; the buggy path caches only one). The contract test `adapter_compute_p_respects_negative_signs` (signed-column storage: `p_pos[i] == −p_neg[i]`) still passes — the fix preserves signed storage, it just stops two opposite-sign rows from aliasing. **Fail-first confirmed** by reverting each bug independently (drop `b_facs[0]`; collapse the cache key to column-only): the matching new test fails (`schur_matrix_honors_negative_b_sign` gets −½ not +½; `compute_p_distinguishes_same_column_opposite_signs`'s `(1,−1)` lookup is `None`) while the rest stay green. Restored, full pounce-sensitivity suite green (47 lib + 6 adapter + integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L17 detail`. |
| L18 | restoration: inner solver pins tolerances at `(1e-8, 1e-6, 15, 3000, 3000)` regardless of the user's outer `tol` (`resto_inner_solver.rs:251`), and `is_square_problem = n == m_eq` ignores inequalities (line 226), unlike IPOPT's `IsSquareProblem` | **FIXED (1 of 2; 2nd is not a bug)** | **Split verdict after running code + upstream cross-check.** **(1) Hardcoded resto tol — REAL, fixed.** `run_inner_resto` built the resto sub-solve's convergence check with `RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000)` — the inner-IPM stationarity `tol`, `acceptable_tol`, and `acceptable_iter` were literals, so a user `tol=1e-3` (or `1e-10`) still drove the restoration sub-NLP to `1e-8`. Upstream clones the outer `OptionsList` into the resto sub-options (`IpRestoMinC_1Nrm.cpp`), so the resto IPM *inherits* the user's `tol`/`acceptable_tol`/`acceptable_iter`. Verified the inner builder carries these: `IpoptApplication::algorithm_builder_from_options` sets `builder.conv_check.{tol,acceptable_tol,acceptable_iter}` from the `tol`/`acceptable_tol`/`acceptable_iter` options (`application.rs:1627,1648,1663`), and CLI/py/cinterface all hand that builder to `make_default_restoration_factory_provider`. **Fix**: extracted `build_resto_conv_check_adapter(&ConvCheckOptions)` that reads `conv.{tol,acceptable_tol,acceptable_iter}` and keeps the resto-phase iteration budgets as named consts `RESTO_INNER_MAX_ITERS`/`RESTO_MAX_SUCCESSIVE_ITERS` (= 3000, which mirror `IpRestoConvCheck.cpp:137,144` `maximum_iters`/`maximum_resto_iters` — resto-specific, *not* the outer `max_iter`); the callsite now passes `&inner_alg_builder.conv_check`. Added `RestoConvCheckAdapter::{inner_tol,inner_acceptable_tol,inner_acceptable_iter}` accessors. **(2) `is_square_problem` — NOT a bug.** The claim "ignores inequalities, unlike IPOPT's `IsSquareProblem`" is false: upstream `IpoptCalculatedQuantities::IsSquareProblem()` is literally `return (ip_data_->curr()->x()->Dim() == ip_data_->curr()->y_c()->Dim());` (`IpIpoptCalculatedQuantities.cpp:3732-3735`, fetched via the GitHub raw API) — i.e. `n == m_eq`, with no inequality term. pounce's `is_square_problem = n_orig == m_eq` matches upstream exactly; left unchanged. **Test** (`resto_inner_solver::tests`): `resto_conv_check_adapter_inherits_user_tolerances` builds a `ConvCheckOptions` with `tol=1e-3, acceptable_tol=1e-2, acceptable_iter=7` and asserts the adapter reports them. **Fail-first confirmed** by reverting the helper to the old `(1e-8, 1e-6, 15)` literals — the test fails (`1e-8 != 1e-3`, "outer tol must propagate"). Restored, full pounce-restoration suite green (106 lib + integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L18 detail`. |
| L19 | restoration: `min_c_1nrm` LSM branch mutates `data.curr` without restoring it only on the non-default `constr_mult_reset_threshold > 0` path (`min_c_1nrm.rs:397-407`) — a divergence trap between option settings | **FIXED** | **Bug confirmed by reading + running code (pounce port of `IpRestoMinC_1Nrm`).** In `recover`'s least-square-multiplier (LSM) block, the non-default `constr_mult_reset_threshold > 0` path sets `data.curr = recovered` (the just-staged trial container) so `EqMultCalculator::calculate_y_eq` evaluates ∇f/J_c/J_d at the recovered iterate — but it **never restored `curr`**, so this path returned with `curr` = the recovered iterate while the default (`threshold == 0`) path leaves `curr` untouched. Upstream `DefaultIterateInitializer::least_square_mults` ends with `CopyTrialToCurrent` + `AcceptTrialPoint`, so upstream's `curr` becomes the recovered iterate on *every* path; pounce instead defers that promotion to the caller (`IpoptAlgorithm`'s `Recovered` arm calls `accept_trial_point`, `curr ← trial`), so `recover` must leave `curr` exactly as found or the two option settings diverge mid-cycle — a latent trap for any read of `curr` between `recover`'s return and the caller's promotion (e.g. `adjust_variable_bounds_for_small_slacks`). **Not observable at solver-output level** because the caller unconditionally overwrites `curr` immediately after `Recovered`; the fix is verified directly at the `recover` boundary. **Fix**: snapshot `saved_curr = data.curr.clone()` before the temporary `curr = recovered`, then restore `data.curr = saved_curr` after the LSM step, so both option paths leave `curr` identical on return. **Test** (`min_c_1nrm::tests`): a direct `perform_restoration` fixture (2-var/1-eq/1-ineq `MockNlp` → non-square so the LSM branch is reachable; stub `EqMultCalculator` writing sub-threshold multipliers without touching `cq`/`aug_solver`; synthetic inner-solver hook returning a recovered `trial_x = [10,20]` ≠ `curr.x = [2,3]`) asserts `curr.x` is unchanged after the call on both the `threshold = 10` and `threshold = 0` paths. **Fail-first confirmed** by dropping the restore: `recover_restores_curr_on_constr_mult_reset_path` fails (`curr` leaks `[10,20]` vs expected `[2,3]`) while `recover_leaves_curr_unchanged_on_default_path` stays green. Restored, full pounce-restoration suite green (108 lib + integration). `cargo fmt`/`clippy` (correctness/suspicious) clean. See `## L19 detail`. |
| L20 | sensitivity: `IndexSchurData::set_from_flags` returns the wrong error variant (`AlreadyInitialized`) for an out-of-range flag value and leaves partially-populated `idx`/`val` state with `initialized == false` (`schur_data.rs:217`), so a caller retry double-appends rows | **FIXED** | **Bug confirmed by reading + running code (pounce port of `SensIndexSchurData`).** Two coupled defects: (1) an out-of-range flag (anything other than 0/1) returned `Err(SchurDataError::AlreadyInitialized)` — a misleading variant that names an unrelated failure mode; (2) the validation happened *mid-loop*, after earlier `f == 1` entries had already been pushed to `idx`/`val`, and the early return left `initialized == false`, so a caller that caught the error and retried with a corrected flag array would re-run the push loop and **append duplicate rows** on top of the partial state. **Fix**: added a dedicated `SchurDataError::InvalidFlag` variant (doc cites upstream `SensIndexSchurData.cpp:51-78`) and rewrote `set_from_flags` to validate the entire flag array up front (`flags.iter().any(|&f| f != 0 && f != 1)` → `Err(InvalidFlag)`) *before* any mutation — atomic, so a rejected call leaves the instance pristine for retry. **Tests** (`schur_data::tests`): `set_from_flags_rejects_invalid_flag_with_distinct_variant` (flags `[1,0,2,1]` → `Err(InvalidFlag)`, not `AlreadyInitialized`); `set_from_flags_invalid_flag_leaves_instance_pristine_for_retry` (flags `[1,0,5]` → `InvalidFlag`, asserts `!is_initialized()`, `nrows() == 0`, empty `col_indices`; then retry `[1,0,1]` yields `col_indices() == [0,2]` with no duplicate append). **Fail-first confirmed**: against the pre-fix code both new tests fail — the old loop returns `Err(AlreadyInitialized)` for the invalid-flag input and, with the early return removed, leaves `col_indices() == [0,3]` (partial) so the retry double-appends to `[0,3,0,2]`. Restored; full pounce-sensitivity suite green. `cargo fmt -p pounce-sensitivity` / `cargo clippy -p pounce-sensitivity --all-targets -- -D clippy::correctness -D clippy::suspicious` clean. See `## L20 detail`. |
| L21 | l1penalty: `L1PenaltyBarrierTnlp::new` discards the `bool` return of `get_starting_point` and `eval_g` with `let _ =` (`wrapper.rs:179-191`), so a failing seed callback silently seeds the `(p, n)` slacks from a zeroed `x_0`/violation instead of returning `None` as documented | **FIXED** | **Bug confirmed by reading + running code (pounce port of ripopt `L1PenaltyBarrierNlp`).** `new`'s doc-contract: "Calls … `get_starting_point`, and `eval_g` … If any of these fail, returns `None`." The code violated it — both calls were `let _ = …`, so a TNLP whose `get_starting_point`/`eval_g` returns `false` would proceed with `x0` left at its `vec![0.0; n]` initialization (and `g0` at zero), seeding every equality slack `(p_k, n_k)` from a bogus zero violation rather than rejecting the wrap. Both methods return `bool` (`pounce-nlp/src/tnlp.rs:175,184`). **Fix**: capture the `get_starting_point` result and `return None` if false; fold the `eval_g` call into `if m > 0 && !…eval_g(…) { return None; }`. Now `new` honors its documented "returns `None` if any of these fail" contract. **Tests** (`wrapper::tests`): a `SeedFails` mock (mirrors `EqOnly` but with `fail_starting_point`/`fail_eval_g` flags) drives `new_returns_none_when_starting_point_fails`, `new_returns_none_when_eval_g_fails`, and a control `new_succeeds_when_seed_callbacks_succeed` (both flags off → `Some`). **Fail-first confirmed**: reverting to `let _ =` makes both `*_returns_none_*` tests fail (`new` returns `Some`) while the control stays green. Restored; full pounce-l1penalty suite green (14 lib). `cargo fmt -p pounce-l1penalty` / `cargo clippy -p pounce-l1penalty --all-targets -- -D clippy::correctness -D clippy::suspicious` clean. See `## L21 detail`. |
| L22 | CLI: the `--cite <model>.nl` "wrong file" hint suggests producing a report with `pounce … --solve-report report.json`, but no `--solve-report` flag exists — `cli.rs` parses only `--json-output` (`main.rs:1700-1711`), so a user following the hint hits "unknown argument" | **FIXED** | **Bug confirmed by reading + running code.** When `--cite` is handed a `.nl` model instead of a solve-report JSON, `run_cite` prints a help hint telling the user to generate a report first. The hint named `--solve-report`, which the CLI arg parser does not accept (grep of `cli.rs` shows the report-output flag is `--json-output` at `cli.rs:520`; `--solve-report` appears nowhere). Following the hint literally produces an unknown-argument error. **Fix**: corrected the hint string (and its preceding comment) to `pounce {} --json-output report.json`. **Test** (`tests/cite_hint_flag.rs`, new integration test using `CARGO_BIN_EXE_pounce`): writes a temp `.nl`-extension file with non-JSON contents, runs `pounce --cite <file>.nl`, and asserts stderr **contains** `--json-output` and **does not contain** `--solve-report`. **Fail-first confirmed**: reverting the hint to `--solve-report` makes the test fail (stderr lacks `--json-output`). `cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --all-targets -- -D clippy::correctness -D clippy::suspicious` clean. See `## L22 detail`. |
| L23 | CLI: after a failed MC64 hypersensitivity scaling retry, `status` reverts to the original local-infeasibility verdict but the reported statistics still reflect the *retry* solve (`main.rs:883-899, 902`) — `app.statistics()` was read **after** the second `optimize_tnlp`, so a non-promoting retry pairs the original verdict with the failed retry's iteration count / objective | **FIXED** | **Bug confirmed by reading + running code.** The `feral_infeasibility_scaling_retry` guard re-solves once with MC64 on a local-infeasibility verdict; on failure it sets `status = InfeasibleProblemDetected` (original verdict) but the subsequent `let solve_stats = app.statistics()` returns the *retry* solve's stats (the retry's `optimize_tnlp` overwrote `app.statistics()`), so the summary/JSON report show the original verdict next to the failed retry's iterations/objective. On the promote path stats were correct only incidentally. **Fix**: snapshot `solve_stats = app.statistics()` immediately after the solve loop (the verdict-bearing solve), and after the retry set `(status, solve_stats)` together via a new pure helper `resolve_scaling_retry_outcome(retry_status, original_stats, retry_stats)` — promote ⇒ `(retry_status, retry_stats)`; otherwise ⇒ `(InfeasibleProblemDetected, original_stats)`. Status and stats now move in lockstep. Extracted `scaling_retry_promoted` for the promote predicate. **Tests** (`main.rs` `scaling_retry_tests`): `failed_retry_keeps_original_status_and_stats` (over `InfeasibleProblemDetected`/`MaximumIterationsExceeded`/`RestorationFailed` retry verdicts → status stays infeasible AND `iteration_count`/`final_objective` stay the original solve's `7`, not the retry's `42`); `promoted_retry_adopts_retry_status_and_stats` (`SolveSucceeded`/`SolvedToAcceptableLevel` → status + stats both the retry's `42`). **Fail-first confirmed**: modeling the pre-fix leak (else-branch returns `retry_stats`) fails the failed-retry test (`iteration_count` 42 ≠ 7). Restored; full pounce-cli bin tests green. `cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious` clean. See `## L23 detail`. |
| L24 | CLI: the `.nl` was fully re-parsed a second time purely to classify it for LP/QP dispatch (`main.rs:456-461` / the `nl_reader::read_nl_file` at the dispatch block), doubling parse time and peak memory on large models; the classification re-parse's error arm (`Err(_) => (ProblemClass::Nlp, None)`) silently fell back to NLP | **FIXED** | **Bug confirmed by reading + running code.** The `.nl` is parsed once to build `NlTnlp` (consuming `NlProblem`), then `read_nl_file(path)` ran a *second* full parse in the dispatch block just to call `classify_problem`. Every `.nl` solve paid two parses; large models paid double parse time + peak memory. The second parse's `Err(_)` arm discarded the error and defaulted to `ProblemClass::Nlp` (silent mis-route latent on a re-read failure). **Fix**: classify during the **first** parse — capture `nl_class = Some(classify_problem(&prob))` right before `prob` is moved into `NlTnlp::new`, and have the dispatch block read `class` from `nl_class` (no re-parse). The specialized convex solvers still need an owned `NlProblem` (the first one was consumed), so re-parse **once, lazily, only on the convex dispatch path** (LP/convex-QP/SOCP) — never for a general NLP solve — and on that re-parse a failure now surfaces (`eprintln!` + exit 2) instead of silently routing to NLP. Builtins (always NLP) never re-parse. **Tests/verification**: the existing end-to-end suites exercise the refactored path — `qp_dispatch_end_to_end::auto_routes_convex_qp_to_pounce_convex` (auto classifies `convex_qp.nl` → routes to pounce-convex, proving the captured `nl_class` drives routing), `nlp_path_still_solves_same_file`, plus `dispatch_routing` (21 tests total) all green. **Fail-first confirmed**: breaking the capture (`nl_class = None` → defaults to Nlp) makes `auto_routes_convex_qp_to_pounce_convex` fail (the convex QP misroutes to NLP, stdout lacks `pounce-convex`). The pure parse-count reduction and the re-parse error-surfacing path are verified by code reading (no deterministic fail-first seam for "parsed once not twice" or a first-succeeds/second-fails race). Restored; suites green. `cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious` clean. See `## L24 detail`. |
| L25 | CLI: a failed `.sol` write exited 0 on the NLP path (`main.rs:1076-1080`) but 2 on the convex LP/QP/SOCP path (`main.rs:1287`); under `-AMPL` a convex-path write failure made the process exit non-zero, so Pyomo/ASL raised `ApplicationError` and never read the (stale/missing) `.sol` | **ALREADY FIXED (by H4)** | **Verified by reading + git history; no separate fix needed.** The exact asymmetry L25 describes was eliminated by the earlier High-priority fix **H4** (`fix(cli): honor -AMPL exit-code contract on convex LP/QP/SOCP paths`, commit `ce49710`). Its message states it "drop[s] the convex paths' `.sol`-write-failure `exit 2` in favor of log-and-continue, matching the NLP path so the exit code uniformly follows the solve outcome." Confirmed by code reading: all three `.sol`-write sites — NLP (`main.rs:1169-1172`), convex-QP (`main.rs:1433-1435`), convex-SOCP (`main.rs:1574-1576`) — now `eprintln!` a warning and continue; none early-returns `ExitCode::from(2)` on a write failure. The convex paths' final exit routes through `convex_exit_code(ok, ampl)` (exits 0 when `ok \|\| ampl`), mirroring the NLP path's `_ if args.ampl => ExitCode::SUCCESS`. **Behavioral coverage**: H4's regression test `qp_dispatch_end_to_end::ampl_mode_honors_exit_code_contract_on_infeasible_convex_qp` runs the infeasible-QP fixture both ways (`-AMPL` exits 0 with srn 200 in the `.sol`; plain CLI exits non-zero), locking the `-AMPL` exit contract that L25 was concerned with. A dedicated `.sol`-write-failure test was not added — forcing a write failure hermetically/portably (unwritable path) is brittle, and the failure mode L25 flagged (distinct exit 2) no longer exists in any path. See `## L25 detail`. |
| L26 | CLI summary (`print.rs`): (a) the final-summary block prints the **same** number under both the `(scaled)` and `(unscaled)` columns for Dual infeasibility / Constraint violation / Complementarity / Overall NLP error (`print.rs:381-401`); (b) "Variable bound violation" is hardcoded `0.0` (`print.rs:391`); (c) the inequality bound-type breakdown didn't sum to `n_ineq` — a "free" inequality row (no finite bound) fell through to a no-op arm (`print.rs:87-89`) | **PARTIALLY FIXED** (c fixed + tested; a/b documented as solver-statistics limitations) | **All three verified by reading code.** **(c) FIXED:** a constraint row with `g_l=-inf, g_u=+inf` is classified inequality (`n_ineq += 1`) but its `(has_l,has_u)=(false,false)` match arm was `=> {}`, so `ineq_lower_only + ineq_both + ineq_upper_only < n_ineq` whenever a free row was present — exactly the "breakdown not sum to total" defect. The comment already said to "count it anyway under 'both' to keep the totals consistent," so the code was simply not doing what its comment claimed. **Fix**: `(false, false) => ineq_both += 1`, comment rewritten to match. **Test** (`print.rs` `inequality_tally_tests::free_inequality_row_keeps_breakdown_summing_to_total`): a mock TNLP with 3 inequality rows (lower-only, both, free) asserts `ineq_lower_only + ineq_both + ineq_upper_only == n_ineq == 3` (and the free row lands in "both": both=2). **Fail-first confirmed**: reverting to `=> {}` makes the sum 2 ≠ 3 (test fails). 160 pounce-cli lib tests green; `cargo fmt` / `cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious` clean. **(a)/(b) DOCUMENTED, not fixed — root cause is missing solver statistics, not a print bug:** `SolveStatistics` (`pounce-nlp/src/solve_statistics.rs:51-66`) carries a single value each for `final_dual_inf`/`final_constr_viol`/`final_compl`/`final_kkt_error` (populated once off the scaled-space `cq` cache at `application.rs:1347-1361`) and **no** variable-bound-violation field; only the objective is tracked in both spaces (`final_objective` + `final_scaled_objective`). So the print layer has only the scaled residual to show — printing it under both columns is the *display* symptom of the solver not computing unscaled residuals, and the `0.0` bound violation is a value the solver never measures (an interior-point iterate is bound-feasible by construction, so it is ~0 on a converged solve, but it is genuinely uncomputed for early-terminated solves). A faithful fix requires plumbing unscaled residuals + a bound-violation metric through `pounce-algorithm`'s finalize path into `SolveStatistics` — a solver-statistics change well beyond a CLI print remediation, deferred to a dedicated issue. Nothing parses this block (grep over `crates/`/`pyomo-pounce/`/`python/` finds no scraper), so the misleading columns are cosmetic, not a data-integrity break. See `## L26 detail`. |
| L27 | CLI module doc (`main.rs:9-12`) claimed "Exit status: 0 on `Solve_Succeeded`, non-zero otherwise" but the code (`main.rs:1183-1185`) also returns 0 for `SolvedToAcceptableLevel` — the doc understated the success set | **FIXED** | **Confirmed by reading code.** The NLP exit-code `match status { SolveSucceeded \| SolvedToAcceptableLevel => SUCCESS, … }` exits 0 for reduced-accuracy convergence too (consistent with `minimize()` parity #119 and Ipopt), but the module-level doc only named `Solve_Succeeded`. **Fix**: (1) corrected the module doc to name both `Solve_Succeeded` and `Solved_To_Acceptable_Level` as the success set; (2) extracted the inline exit-code match into a pure helper `nlp_exit_code(status, ampl)` (mirroring the existing `convex_exit_code(ok, ampl)`), itself composed from a testable predicate `nlp_solve_succeeded(status)` (mirrors L23's `scaling_retry_promoted`). This both removes the duplicated AMPL-contract comment and gives the doc-claimed behavior a regression lock. **Tests** (`main.rs` `nlp_exit_code_tests`): `acceptable_level_counts_as_success` (both `SolveSucceeded` and `SolvedToAcceptableLevel` → true — the crux of L27) and `non_convergent_statuses_are_not_success` (Infeasible/MaxIters/RestorationFailed/Diverging/MaxCpuTime/InternalError → false). `ExitCode` has no `PartialEq`, so the bool predicate is the testable seam. **Fail-first confirmed**: dropping `SolvedToAcceptableLevel` from `nlp_solve_succeeded` makes `acceptable_level_counts_as_success` fail (`assertion failed: nlp_solve_succeeded(A::SolvedToAcceptableLevel)`). Restored; 5 pounce-cli bin tests green. `cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious` clean. See `## L27 detail`. |
| L28 | `nl_hessian_program.rs` (`HessianProgram::compile`) is dead in-tree (only `pub mod` in `lib.rs:19`; zero callers of `compile`/`execute`) but `panic!`s on `Funcall`/min/max/conditional-logical/extra-transcendental ops (lines 456, 477, 594, 615, 766, 787 in `compile`; 1275, 1285, 1329, 1339 in the `reachable_to_output`/`depends_on_var` helpers) — would fire on arbitrary user `.nl` input if ever wired into dispatch | **FIXED** | **Confirmed by reading code + git.** The module declares `pub mod nl_hessian_program;` (compiled) but nothing calls `HessianProgram::compile`/`execute` (grep over `crates/`/`pyomo-pounce/`/`python/` finds only the `pub mod` line). All three compile sweeps (forward / forward-tangent / reverse-over-tangent) and the two dependence/reachability analyses `panic!`ed on opcodes the program path can't lower; the panic messages themselves say "use the Tape (build_with_externals) path instead", revealing the *intended* design is a graceful fall-back, not a crash. **Fix**: (1) `compile` now returns `Option<Self>` with an up-front guard `if !tape.ops.iter().all(program_supports_op) { return None; }` — a new free fn `program_supports_op` is the single source of truth for the supported set (smooth arithmetic + `sin`/`cos`); (2) all 10 `panic!`s became `unreachable!` — the guard filters unsupported ops before any sweep or helper runs, so they are statically unreachable from the public entry, and `unreachable!` now flags only an internal guard/sweep inconsistency (a programmer error), never user input; (3) both early returns wrapped in `Some(...)`; (4) the two in-tree test call sites updated to `.expect(...)`. **Test** (`nl_hessian_program::tests::unsupported_opcode_returns_none_instead_of_panicking`): a `tan(x0)` tape (→ unsupported `TapeOp::Tan`) makes `compile` return `None`; a `mul(x0,x1)` tape still returns `Some`. **Fail-first confirmed**: disabling the guard (`if false`) makes the tan tape reach the forward-sweep `unreachable!` and panic (`internal error: entered unreachable code: HessianProgram path does not yet support tan/...`) — exactly the L28 crash-on-user-input failure mode. Restored; 161 pounce-cli lib tests green (was 160). `cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --lib -- -D clippy::correctness -D clippy::suspicious` clean (remaining warnings are pre-existing `unwrap_used`/`expect_used` restriction-style, incl. the test `.expect()`s — not correctness/suspicious). See `## L28 detail`. |

| L29 | `Pow` first-order tangent disagrees with the reverse-mode gradient at base 0 (`nl_tape.rs` forward-tangent sites guard on `r != 0 && u != 0`; reverse guards only `rv != 0`) — Jacobian and Hessian-vector products use an inconsistent derivative model at `x = 0`, the `.nl` default start | **FIXED** | **Confirmed by reading code.** Three forward-tangent `Pow` sites — `forward_tangent` (was line 513), `hessian_directional` (736), free fn `fwd_tan_step` (2994) — guarded the base-derivative term `r * u^(r-1) * du` on `r != 0.0 && u != 0.0`, while the reverse adjoint sweep (`reverse_into` 343, free fn 2856) guards only `rv != 0.0`. At the `.nl` default start `x0 = 0` with `f = x0^x1`, `x1 = 1`: reverse gives `df/dx0 = 1*0^0 = 1`, forward-tangent gave `0` (term dropped) — a real Jacobian-vs-gradient disagreement. **Fix**: changed all three guards to `if r != 0.0` (dropping the spurious `&& u != 0.0`) so forward-tangent matches reverse exactly; `0.powf(r-1)` is `1` for `r=1`, `0` for `r>1`, `±inf` for `r<1` — identical to what reverse already produces. **Test** (`nl_tape::tests::pow_forward_tangent_matches_reverse_gradient_at_base_zero`): builds `pow(var(0), var(1))` (variable exponent stays a real `Pow`, not lowered to Mul/Sqrt), evaluates at `[0.0, 1.0]`, asserts forward-tangent `df/dx0 == reverse grad[0] == 1`. **Fail-first confirmed**: reverting the guard to `r != 0.0 && u != 0.0` makes forward-tangent return `0` while reverse stays `1`, failing the assert (`forward tangent df/dx0 = 0 must match reverse gradient 1`). Restored; 81 pounce-nl tests green. **Hessian (second-order) `Pow` base-0 sites left unchanged on purpose**: the forward-over-reverse cross-partial `∂²(u^r)/∂u∂r` at `u = 0` with a *variable* exponent is genuinely singular (`→ ±inf`); those sites already special-case integer `r ≥ 2` and base 0, and there is no finite value that both arms could consistently agree on — only the first-order tangent had a removable inconsistency. `cargo fmt -p pounce-nl` / `cargo clippy -p pounce-nl --all-targets -- -D clippy::correctness -D clippy::suspicious` clean (remaining warnings pre-existing `unwrap_used`/`expect_used` restriction-style). See `## L29 detail`. |

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

## H12 detail

- **Bug** (two independent defects in the Phase 1b FBBT call,
  `lib.rs:610-631`):
  1. **No Phase-0 row mask.** `run_fbbt` was handed `m_in` (the full inner
     row count) and `g_l_inner`/`g_u_inner` over the **aux-clamped** variable
     bounds — but Phase 0 may have dropped rows (recorded in
     `row_kept_inner`). Propagating a dropped row's interval against the
     clamped box can derive a contradiction that does not exist in the
     original problem — exactly the configuration the #53 review fixed for
     Phase 1 (filtered rows). FBBT got neither the filter nor the rollback.
  2. **No infeasibility handling.** `FbbtReport::infeasibility_witness`
     (`fbbt/orchestrator.rs:70-74`) documents that on infeasibility "the
     variable bounds … are undefined and must not be trusted" — FBBT can
     partially tighten several variables in a sweep before a later
     constraint proves the box empty, then return immediately. The call site
     stored `fbbt_report = Some(report)` and proceeded, feeding those
     corrupted bounds straight to the IPM. Genuine infeasibility was silently
     swallowed *and* the bounds were wrong.
- **Fix**:
  - `run_fbbt` (`fbbt/orchestrator.rs`) gained a `row_kept: Option<&[bool]>`
    parameter (length-asserted); the sweep `continue`s on any `!mask[i]`.
    `None` preserves the standalone/test behavior (consider every row).
  - The call site (`lib.rs`) passes `Some(&row_kept_inner)` — the same mask
    Phase 0/Phase 2 maintain — so dropped rows are never propagated.
  - Before FBBT the call site snapshots `x_l`/`x_u`; if
    `report.infeasibility_witness.is_some()` it restores the snapshot and
    logs a warning. Presolve has **no channel to certify infeasibility** to
    the solver (Phase 1's own infeasibility path likewise rolls back, not
    surfaces), so the correct conservative action is to discard FBBT's
    undefined bounds and let the IPM run on the valid pre-FBBT box and
    certify infeasibility itself. The report is still exposed via
    `fbbt_report()` for diagnostics.
- **Tests**:
  - `dropped_row_is_skipped_and_does_not_flag_infeasible`
    (`fbbt/orchestrator.rs`): a `Var(0)` row demanding `x = 5` over the box
    `[0,1]`. Control (`row_kept = None`) → `infeasibility_witness == Some(0)`;
    fixed (`Some(&[false, true])`, the row dropped) → `None`, box untouched.
  - `fbbt_infeasibility_discards_corrupted_bounds` (`lib.rs` integration):
    `FbbtPartialThenInfeasible` (1 var `x∈[0,1]`, two `g=x` nonlinear rows;
    row 0 tightens to `[0.3,0.7]`, row 1 demands `x=5`) + a `VarTapeProvider`.
    After presolve, `fbbt_report().infeasibility_witness == Some(1)` **and**
    the reduced box is the original `[0,1]`, not the corrupted `[0.3,0.7]`.
- **Verified by running code**: the integration test, run with the bound
  restore temporarily disabled, FAILED with `left: (0.3, 0.7)` (the corrupted
  partial tightening leaking to the IPM); with the fix it reads `(0.0, 1.0)`
  and PASSES. Full `pounce-presolve` suite green (207 lib + integration +
  9 doctests); `cargo fmt --check` clean; no build warnings.

## H13 detail

- **Bug** (`pounce-cinterface/src/solver.rs`): the session-style
  `IpoptSolverSolve` moves the configured `IpoptApplication` into a fresh
  `RustSolver` with
  `std::mem::replace(&mut info.problem.app, IpoptApplication::new())`,
  leaving a **default-initialised** app in `info.problem.app`. Nothing
  restored it. The struct doc claimed restoration happened "via the
  `app_template` field below", but no such field exists (grep-confirmed). On
  the **second** `IpoptSolverSolve` of the same handle:
  - every option set via `AddIpopt{Str,Num,Int}Option` (linear solver,
    tolerances, scaling, `max_iter`, …) had been silently replaced by
    defaults, and
  - the `feral_config_from_options(info.problem.app.options())` snapshot
    (`solver.rs:191`) read the already-blanked options.
  Repeated solves on one handle are the session API's whole purpose, so this
  is a silent wrong-result bug for any multi-solve caller.
- **Fix**: `OptionsList` derives `Clone` and holds the full option map plus
  the registry `Rc`. Clone it immediately before the `mem::replace`, then
  write it into the fresh blank app via `options_mut()`:
  ```rust
  let saved_options = info.problem.app.options().clone();
  let app = std::mem::replace(&mut info.problem.app, IpoptApplication::new());
  *info.problem.app.options_mut() = saved_options;
  ```
  The app moved into the solver keeps its own options for that solve; the
  handle keeps a faithful copy for the next one. The stale `app_template`
  doc comment on `IpoptSolverInfo::problem` was rewritten to describe the
  real clone-across-move behavior.
- **Scope note**: the review names options as the lost state; the fix
  restores the full `OptionsList`, which covers every `AddIpopt*Option` key
  and the derived `feral_config` snapshot. Per-solve app wiring (restoration
  provider, linsol sink) is already re-established each call and needs no
  preservation.
- **Test** (`solver.rs`, new `mod tests`):
  `options_survive_repeated_session_solves` builds the 1-D quadratic
  `f=(x−2)²` (the bridge tests' solvable problem), sets `max_iter = 7`,
  `IpoptCreateSolver` (consuming the problem), then solves **twice**,
  asserting `get_integer_value("max_iter")` still reads `7` after each solve.
- **Verified by running code**: with the restore line disabled the test
  FAILED after the first solve (`left: None`, the option blanked); with the
  fix it reads `Some(7)` after both solves and PASSES. Full
  `pounce-cinterface` suite green (42 tests); `cargo fmt --check` clean; no
  build warnings.

## H14 detail

- **Bug** (release tooling): the root `Cargo.toml:89` pins feral by git rev
  with **no `version =`**:
  ```toml
  feral = { git = "https://github.com/jkitchin/feral.git", rev = "11fb4b9…" }
  ```
  `pounce-feral` (crate **4 of 19** in `publish-crates.sh`'s topological
  order) depends on it (`feral.workspace = true`). `cargo publish` rewrites
  every path/git dep to a crates.io version requirement and refuses any dep
  that lacks one, so publishing `pounce-feral` hard-fails — *after* crates 1–3
  (`pounce-common`, `pounce-linalg`, `pounce-linsol`) are already live.
  crates.io versions are immutable, so a `vX.Y.Z` tag ships an irreversible,
  un-rollback-able **partial** release. Neither `check-release-consistency.sh`
  (versions / membership / topo order only) nor any CI job ran
  `cargo publish --dry-run`, so the guard reported the release safe.
- **Verified by running code**:
  ```
  $ cargo publish -p pounce-feral --dry-run
  error: failed to verify manifest …
    all dependencies must have a version requirement specified when publishing.
    dependency `feral` does not specify a version
  ```
  `cargo metadata` shows the dep as `req:"*"`, `source:"git+…"`. `feral` is on
  crates.io only at **0.10.0**, which predates the pinned MC64/scaling commits,
  so simply pinning `version="0.10.0"` would publish a crate that depends on
  *different code* than was built — the comment in `Cargo.toml` documents
  exactly this. The true root fix (feral cutting a release with the pinned
  commits) is **out of scope** for a code-review remediation.
- **Fix** (two layers, both runtime-verified):
  1. **Visibility** — new `scripts/check_dep_publishability.py` parses
     `cargo metadata` and flags any normal/build dependency of a publishable
     crate that is git-sourced or carries a wildcard/`*` (versionless)
     requirement; dev-deps and `publish = false` crates are exempt. Wired as
     **check #4** in `check-release-consistency.sh` (the guard CLAUDE.md
     documents as the pre-tag gate, run in CI on every PR). The blocker is no
     longer invisible: the guard now exits non-zero and names
     `pounce-feral -> feral` until feral is released and pinned.
  2. **Safety** — `publish-crates.sh` gained a **pre-flight** that runs the
     same scan against its `CRATES=(…)` list and aborts *before uploading
     crate 1*. This is the load-bearing fix: it converts the irreversible
     mid-batch failure into a clean no-op even if the guard is bypassed. The
     tag-triggered `release-crates.yml` inherits it (it invokes this script).
- **Tests** (`scripts/tests/test_check_dep_publishability.py`, 7 cases): runs
  the detector against **synthetic** `cargo metadata` documents, so they are
  independent of the live tree (which is itself blocked today). Cover:
  clean workspace → no blocker; git dep → flagged; wildcard `*` req → flagged;
  dev-dependency git dep → ignored; build-dependency git dep → flagged;
  `publish = false` crate's git dep → ignored; `restrict_to` scoping. All pass
  (`python3 scripts/tests/test_check_dep_publishability.py` → `Ran 7 tests … OK`).
- **Verification summary**: live guard now FAILS at check #4 (checks 1–3 still
  print OK, proving pre-fix the guard exited 0 — "looks safe but isn't");
  `publish-crates.sh --dry-run` ABORTS at pre-flight before any `cargo publish`;
  unit suite green.
- **Trade-off (flagged for the maintainer)**: because the guard runs on every
  PR (`ci.yml`), check #4 will keep CI red until `feral` cuts a crates.io
  release carrying the pinned commits and `Cargo.toml` is updated to
  `feral = { version = "X.Y.Z", git = …, rev = … }`. That red is intentional
  and honest — a crates.io release genuinely cannot succeed in the current
  state. If the team prefers the guard not gate unrelated PRs, demote check #4
  to a warning (drop the `fail=1`) while keeping the `publish-crates.sh`
  pre-flight as the hard gate; the harm-prevention is unaffected.

## H15 detail

- **Bug** (`python/pounce/_curve_fit.py`): `_solve_fit` — the single solve path
  behind `curve_fit`, `curve_fit_streaming`, and `curve_fit_minima` — computed
  ```python
  success = int(info["status"]) == 0
  ```
  Only `Solve_Succeeded` (0) counted; `Solved_To_Acceptable_Level` (1) — a
  converged solve where the iterate met the *acceptable* tolerance after the
  tight one stalled — was reported `success=False` despite returning a fully
  populated `popt`/`pcov`. Callers gating on `result.success` discard valid
  fits. The repo had already fixed exactly this class for `minimize`
  (gh #119, `_minimize.py:65` accepts `{0, 1}`) and the jax/torch paths accept
  both, so `curve_fit` was the lone straggler. It also lacked the
  `final_kkt_error` ≤ `acceptable_tol` fallback `minimize` applies
  (`_minimize.py:524-529`) for stall exits (e.g. tiny-step, status 3) that
  nonetheless land at an acceptable NLP error.
- **Verified by running code**: built the native extension into an isolated
  venv (`maturin develop`, 17 s incremental) and ran an exp-decay fit over the
  finite-difference path at a deliberately tight `tol=1e-12`,
  `acceptable_tol=1e-5`:
  ```
  status 1  success False  msg Solved_To_Acceptable_Level   popt [2.5 1.311 0.505]
  ```
  i.e. a verified optimum reported as a failure. (`tol=1e-9` converges fully →
  status 0, success True, confirming the tight tol is what forces the
  acceptable-level stall.)
- **Fix**: reuse the NLP `minimize` decision so the two entry points agree —
  import `_NLP_SUCCESS_STATUS` (`{0, 1}`) and `_DEFAULT_ACCEPTABLE_TOL` from
  `_minimize`, then
  ```python
  status_code = int(info["status"])
  acceptable_tol = float(user_opts.get("acceptable_tol", _DEFAULT_ACCEPTABLE_TOL))
  kkt_error = float(info.get("final_kkt_error", float("nan")))
  success = status_code in _NLP_SUCCESS_STATUS or (
      np.isfinite(kkt_error) and kkt_error <= acceptable_tol
  )
  ```
  Post-fix the same fit reports `status 1, success True`. `user_opts` (already
  built at `_curve_fit.py:702`) carries any caller-supplied `acceptable_tol`.
- **Tests** (`python/tests/test_curve_fit.py`):
  - `test_curve_fit_acceptable_level_reports_success` — e2e: the tight-`tol`
    FD fit above; asserts `res.status == 1` (the acceptable path actually
    fires) **and** `res.success is True` and `popt ≈ [2.5, 1.3, 0.5]`.
  - `test_curve_fit_success_mapping_matches_nlp_minimize` — pins that the rule
    reuses `_NLP_SUCCESS_STATUS` (0,1 success; 2 not), guarding against the
    two paths diverging again.
- **Verification summary**: with the fix reverted to the old one-liner the e2e
  test FAILS (`assert False is True`, `popt` valid — the exact bug); restored,
  both new tests PASS. Full `test_curve_fit.py` green (42), and
  `test_minima.py` + `test_minimize.py` green (30) — the streaming/minima
  routes and the `minimize` import are unaffected.

## M1 detail

- **Issue** (review M1): the convergence test compares the *internally scaled*
  residuals against the user-facing tolerances (`dual_inf_tol`,
  `constr_viol_tol`, `compl_inf_tol`), whereas upstream Ipopt tests the
  **unscaled** quantities. With `nlp_scaling_method` on (the default), a problem
  whose scaled residuals are below tolerance can have unscaled residuals well
  above it, so pounce can report `Solve_Succeeded` for a point the user's own
  `constr_viol_tol` would reject.
- **Verified by code inspection** (no fix shipped — see "why deferred"):
  - `conv_check/opt_error.rs:215-222` (`check_convergence_with_state`) and
    `:301-307` (`current_is_acceptable_with_state`) gate the per-component
    tolerances on the CQ accessors `curr_dual_infeasibility_max`,
    `curr_primal_infeasibility_max`, `curr_complementarity_max`, and `curr_f`.
  - Those accessors are the **scaled** ones (`ipopt_cq.rs:950-962, 1041-1047`).
    The CQ exposes **no** unscaled per-component accessor — only
    `unscaled_curr_f` exists (`ipopt_cq.rs:743`). So the unscaled comparison
    upstream performs is simply not expressible with today's CQ surface.
  - Scaling is on by default: `nlp_scaling_method` defaults to
    `gradient-based` (`upstream_options.rs:361`).
  - Direction of harm (`orig_ipopt_nlp.rs:897-916`, `row_max_to_scale`):
    `c_scaled = c_scale · c_orig` with `c_scale ≤ 1`. The user-space violation
    is `c_orig = c_scaled / c_scale ≥ c_scaled`, so a scaled residual that
    passes `constr_viol_tol` can correspond to an unscaled violation up to
    `1/c_scale` larger — pounce declares success while the real constraint
    violation exceeds the user's tolerance. (When `c_scale = 1`, i.e. scaling
    off or unit row, the two agree; the gap only opens as scaling shrinks rows.)
- **Why deferred, not fixed in this autonomous pass** — the correct fix is a
  cross-crate change to core convergence criteria, with non-trivial math, and
  carries high blast radius; it deserves a reference-validated review rather
  than an unattended edit:
  1. **Constraint violation** needs an unscaled accessor, which needs
     `c_scale`/`d_scale`. These live in `RefCell<Option<Vec<Number>>>` private
     to `OrigIpoptNlp`; the `IpoptNlp` trait exposes no constraint-scaling
     accessor. Exposing them means **new trait methods on every `IpoptNlp`
     implementor**, not a local patch.
  2. **Dual infeasibility and complementarity** cannot be recovered by a simple
     divide. x-scaling is identity in pounce, but objective scaling `df` is not,
     so the scaled Lagrangian gradient is
     `∇ₓL_scaled = df·∇f + Jᵀλ` versus the unscaled `∇f + Jᵀλ` — the `df`
     factor couples in and a naive `/df` corrupts the stationarity measure.
     Recovering the true unscaled quantities is exactly the NLPScalingObject
     unwind pounce **explicitly defers** (`orig_ipopt_nlp.rs:52-54`).
  3. This is the termination test itself: a wrong change silently flips
     `Success`/`failure` verdicts across the whole solver. It must be validated
     against upstream Ipopt on scaled problems, not shipped blind.
- **Scoped forward plan** (two PRs, each independently reviewable + testable):
  - **PR1 — constraint violation (mechanical, high value).** Add
    `unscaled_curr_primal_infeasibility_max` to the CQ, backed by new
    `IpoptNlp` trait methods exposing `c_scale`/`d_scale` (default impls return
    `None` ⇒ "no scaling" ⇒ identical to today for implementors that don't
    scale). Switch the `constr_viol_tol` gate in both convergence checks to the
    unscaled value, and the objective-change criterion to `unscaled_curr_f`
    (already available). **Test**: a small NLP with a deliberately ill-scaled
    constraint (row scale ≪ 1) whose *scaled* residual sits just under
    `constr_viol_tol` but whose *unscaled* residual is, say, 10× over — assert
    pounce now returns a non-success status (today it returns
    `Solve_Succeeded`). The test fails on `main` and passes after PR1.
  - **PR2 — dual-inf + complementarity (derivation-heavy).** Implement the
    `df`-coupled unscaled stationarity/complementarity recovery (the deferred
    NLPScalingObject unwind for these two terms), switch the remaining two
    gates, and validate termination verdicts against upstream Ipopt on a scaled
    reference problem set before merge.
- **No code changed for M1** — documented as VERIFIED — DEFERRED per the review
  workflow ("document issues that cannot be verified [here]"). The mechanism is
  confirmed; the fix is scoped above for a dedicated, reviewed change.

## M2 detail

- **Bug** (`crates/pounce-algorithm/src/ipopt_data.rs`): `accept_trial_point`
  promoted the staged trial unconditionally:
  ```rust
  pub fn accept_trial_point(&mut self) {
      self.curr = self.trial.take();
  }
  ```
  When no trial is staged, `self.trial.take()` is `None`, so this **nulls out
  `curr`**. Upstream `IpIpoptData::AcceptTrialPoint` `DBG_ASSERT`s a valid trial
  before promoting it, because upstream always runs a line search that stages
  one — so upstream never reaches this state.
- **Reachable path** — pounce has a documented bookkeeping-only `iterate()`
  mode (`ipopt_alg.rs:17-22`: "Without [NLP + search_dir], `iterate()` runs the
  bookkeeping pieces … and is exercised by structural unit tests"):
  1. Step 5 / search direction is gated on `if let (Some(nlp), Some(sd)) = …`
     (`ipopt_alg.rs:724-727`); without both, it is skipped and `data.delta`
     stays `None`.
  2. The line search is gated on `have_delta = self.data.borrow().delta.is_some()`
     (`ipopt_alg.rs:994-995`); with `delta == None` the whole block is skipped,
     so **no trial is staged**.
  3. `accept_trial_point()` is nevertheless called every iteration
     (`ipopt_alg.rs:1121`), so `curr` is set to `None`.
  4. The next iteration's CQ accessor `IpoptCq::curr_iv`
     (`ipopt_cq.rs:107-112`) does
     `let Some(iv) = …curr… else { unreachable!("curr iterate not set") }` —
     a panic.
- **Fix** — guard the promotion so an unstaged accept preserves `curr`:
  ```rust
  pub fn accept_trial_point(&mut self) {
      if let Some(trial) = self.trial.take() {
          self.curr = Some(trial);
      }
  }
  ```
  The normal solve path is byte-for-byte unchanged: after a line search `trial`
  is always `Some`, so it still promotes to `curr` and clears `trial` (the
  existing `accept_trial_point_promotes_trial_to_curr` test still passes). Only
  the previously-buggy `trial == None` case changes — from "destroy `curr`" to
  "leave `curr` intact".
- **Test** (`ipopt_data.rs` `#[cfg(test)] mod tests`):
  `accept_trial_point_preserves_curr_when_no_trial_staged` — sets `curr` via
  `set_curr(zero_iv())`, leaves `trial` unset, calls `accept_trial_point()`, and
  asserts `curr.is_some()` (and `trial.is_none()`).
- **Verification summary**: pre-fix the new test FAILS with
  `accept_trial_point() nulled curr with no trial staged` while the existing
  promote test passes; post-fix both pass. Full `pounce-algorithm` suite green
  (323 passed, 0 failed) — no regression in the normal-step path.

## M3 detail

- **Bug** (`crates/pounce-algorithm/src/eq_mult/least_square.rs`):
  `LeastSquareMults::calculate_y_eq` builds the least-squares-multiplier
  augmented system (`W=0`, `δx=δs=1`) and solved it with
  ```rust
  delta_c: 0.0, … delta_d: 0.0,
  ```
  then `aug_solver.solve(&coeffs, &aug_rhs, &mut sol, check_neg, num_eq)` with
  `check_neg = aug_solver.provides_inertia()` and `num_eq = n_c + n_d`
  (`least_square.rs:133-135`).
- **The sibling already worked this around.** The dual initializer in
  `init/default.rs:154-194` solves the *same* `W=0` augmented system, but sets
  `delta_c = delta_d = 1e-8` with an explicit comment (`init/default.rs:163-174`):
  pounce-feral's LDLᵀ mis-reports the inertia of an augmented system whose
  `(3,3)/(4,4)` block is structurally zero — "it counted 0 negative eigenvalues
  on `nuffield2_trap` where the true count is `n_c+n_d`, triggering
  `WrongInertia`." The `1e-8` gives the diagonal something nonzero to pivot on
  while leaving the solution numerically identical (the constraint Jacobian
  dominates the term). `least_square.rs` solves the identical structure but
  *omitted* this perturbation — duplicate logic that diverged.
- **Consequence**: when feral mis-reports the inertia, `calculate_y_eq` returns
  `false`; the caller `init/default.rs:387-390` treats that as "solver failed →
  leave at zero" and appends `"y0"` to the info string. The iterate then starts
  with `y_c = y_d = 0`, producing exactly the iter-0 `inf_du` blow-up the
  least-squares-multiplier step exists to prevent. Because the default
  `constr_mult_init_max = 1e3 > 0` (`init/default.rs:73`,
  `alg_builder.rs:256`), the LS path is active on every equality/inequality-
  constrained solve.
- **Fix**: mirror the sibling's perturbation — set `delta_c = delta_d = 1e-8`
  in `least_square.rs` (with a cross-reference comment instructing future edits
  to keep the two sites in sync), so the LS-multiplier solve survives feral's
  inertia mis-report identically to the dual initializer.
- **Why no synthetic fail-first test** (documented per the loop's "document
  issues that cannot be [fail-first] verified" clause): the failing-pre-fix
  trigger is feral's *data-dependent* inertia mis-report, which the sibling's
  own comment ties to one specific CUTEst matrix (`nuffield2_trap`) — a
  benchmark problem **not present in the repo**. The `pounce-algorithm`
  aug-solver unit harness drives `StdAugSystemSolver` with `DenseMock`, an exact
  LU oracle (`std_aug_system_solver.rs:1082`) that reports correct inertia
  regardless of `delta`, so it cannot reproduce the mis-report; and feral itself
  solves well-conditioned small structurally-zero-block systems correctly, so a
  synthetic matrix won't reliably trip it. A genuine fail-first test would
  require shipping the `nuffield2_trap` matrix. Notably the **sibling fix itself
  carries no synthetic fail-first unit test** (`init/default.rs` tests cover only
  `push_to_interior`); it was validated by integration solve — the same basis
  used here.
- **Verification by running**: with `constr_mult_init_max = 1e3` active by
  default, every constrained solve traverses `calculate_y_eq` during
  initialization. The constrained-problem integration tests `optimize_hs71`,
  `optimize_hs14`, and `hock_schittkowski_subset` — plus the full
  `pounce-algorithm` suite — stay green post-fix (323 passed, 0 failed),
  confirming the `1e-8` perturbation is numerically inert on every covered
  problem (no change to converged multipliers or solve outcomes). This is the
  strongest in-repo runtime evidence available; the data-dependent feral trigger
  is documented above for a future integration test if `nuffield2_trap` is added
  to the benchmark corpus.

## M4 detail

- **Bug** (`crates/pounce-linalg/src/eigen.rs`): `symmetric_eigen` runs cyclic
  Jacobi for up to `max_sweeps = 50`, `break`ing out of the sweep loop when the
  off-diagonal Frobenius mass `off` drops below `tol = 1e-28·‖A‖²_F`. The doc
  contract (`eigen.rs:32-35`) says it "Returns `true` on convergence … `false`
  if the iteration ran out of sweeps." But the old code fell through to a bare
  `true` (old `eigen.rs:153`) after the loop, so a matrix that exhausted all 50
  sweeps *without* converging was still reported as a success.
- **Why it matters**: callers branch on the boolean —
  `pounce-convex/src/cones/psd.rs:108,145,163,231` and
  `pounce-convex/src/sos.rs:615,672,717` — to decide whether to use the
  eigenpairs or take an error path. A false `true` feeds unconverged
  eigenvalues/eigenvectors into PSD cone projections and SOS decompositions.
  Latent in practice (cyclic Jacobi converges in a handful of sweeps for the
  small reduced-Hessian dimensions here), but a real correctness hole.
- **Fix**:
  1. Track `converged` (set `true` on the early `break`).
  2. After the loop, if `!converged`, recompute the off-diagonal mass once and
     set `converged = off < tol`. The per-sweep test runs at the *top* of each
     sweep, so it never observes the state produced by the final sweep; the
     post-loop recompute credits a run that converged on the last sweep and lets
     a genuinely stalled run report `false`.
  3. Extract/sort the eigenpairs unconditionally (unchanged), then
     `return converged` instead of `true`. Callers that ignore the bool keep
     getting best-effort values; callers that branch now see the truth.
- **Testability refactor**: the `false` path is essentially unreachable with
  real inputs (Jacobi always converges), so to exercise it the body moved into a
  private `fn symmetric_eigen_impl(a, n, evals, evecs, max_sweeps)`; the public
  `symmetric_eigen` delegates with `max_sweeps = 50`. Public signature and all
  callers are unchanged.
- **Tests** (`eigen.rs` `mod tests`):
  - `eigen_reports_false_when_sweeps_exhausted` — a coupled 4×4 symmetric matrix
    with `max_sweeps = 1` cannot converge in one cyclic sweep, so it must return
    `false`. **Pre-fix this FAILS** (the old code returned `true`).
  - `eigen_reports_true_when_converged` — the same matrix at `max_sweeps = 50`
    returns `true`, and an already-diagonal matrix returns `true` even at
    `max_sweeps = 1` (the top-of-sweep check fires before any rotation). Guards
    against the fix over-reporting `false`.
- **Verification summary**: pre-fix `eigen_reports_false_when_sweeps_exhausted`
  FAILS while the converged-path tests pass; post-fix all 8 `eigen` tests pass,
  and the full `pounce-linalg` plus `pounce-convex` consumer suites stay green
  (328 passed, 0 failed) — the existing convex PSD/SOS tests confirm the new
  verdict does not perturb the converged (normal) path.

## M5 detail

- **Bug** (`crates/pounce-qp/src/solver.rs`): the active-set QP solver's
  warm-start path can return `QpStatus::Optimal` at a point that violates a
  constraint — most sharply, an equality row the caller's working set left
  `Inactive`.
  - The inner loop of `solve_general` (and its Schur twin `solve_general_schur`)
    solves the EQP step system with the constraint block of the RHS zeroed
    (`rhs[n..] = 0`, lines 729-732): the step keeps `aᵢᵀp = 0` for every
    *active* row, i.e. it holds those rows at whatever residual the warm-start
    `x` already had. Nothing re-audits that residual — the cold path guarantees
    feasibility via `cold_general_initial`, but the warm-start path trusts the
    caller.
  - The `Optimal` return (lines 827-841 / 1259-1273) had no feasibility check,
    even though `QpStatus::Optimal`'s doc (`error.rs:8-9`) promises "KKT
    residual **and feasibility** within tolerance."
  - The ratio test skips equality rows entirely (`if qp.bl[i]==qp.bu[i] {
    continue; }`, lines 883-884 / 1299-1300), so an equality the caller left
    `Inactive` can never be picked up as a blocker and entered into the working
    set — it is silently never enforced.
- **Why it matters**: the solver is warm-started by the active-set SQP driver
  and by `solve_elastic`'s recursive call. A warm start whose `x` is infeasible
  (or whose working set omits an equality) converges to the unconstrained /
  wrong-working-set minimum and is reported `Optimal` — a *wrong answer*, not the
  "diverge or hit max_iter" the doc comment (lines 668-671) advertised. The doc
  itself names the missing piece: "Validation is deferred to a follow-up commit
  that adds an `OptimalityCheck` audit pass."
- **Fix**: add that audit pass.
  1. New free fn `point_is_feasible(qp, x, feas_tol)` — checks every general row
     **including equalities** (`bl`/`bu` against `aᵀx`) and every variable bound
     against `feas_tol`. Mirrors the inequality check already in
     `cold_general_initial` (lines 1000-1021), extended to cover equality rows.
  2. In the public `solve` — the single entry point that dispatches to both
     `solve_general` and `solve_general_schur` — capture the result and, when it
     claims `Optimal` but fails `point_is_feasible`, `return self.solve_elastic(qp,
     opts)`. This is the **same** recovery the cold path already uses when
     `cold_general_initial` returns an infeasible point (`None => return
     self.solve_elastic(...)`), so an infeasible warm start now reaches the true
     optimum (or a certified `Infeasible`) instead of a false `Optimal`.
- **Recursion safety**: `solve_elastic` recurses through `solve_general`
  *directly* (line ~1090), not through the public `solve`, so the augmented
  elastic problem is **not** re-audited. Its warm start is slack-feasible by
  construction (`reform.initial_seed` sets the slacks to absorb the residual
  exactly), and the active-set inner loop preserves feasibility, so even if the
  audit *were* reachable there it would pass. No infinite-recovery loop is
  possible.
- **Happy path unchanged**: feasible warm starts and cold solves (which
  `cold_general_initial` already guarantees feasible or routes to elastic) pass
  the audit and are returned verbatim. Confirmed by the existing warm-start and
  scaling tests staying green.
- **Scope note (equality ratio-test skip)**: the `continue`-on-`bl==bu` lines
  are left as-is. Making `Inactive` equalities enter through the ratio test is a
  deeper change to the active-set add logic with its own degeneracy/cycling
  considerations; the feasibility audit is the minimal, recursion-safe fix that
  closes the *observable* defect (the false `Optimal`) by routing any such case
  to the elastic solver, which enforces every row. The audit catches both
  failure modes (frozen active-row residual **and** never-entered equality).
- **Test** (`tests/analytical.rs`):
  `m5_warm_start_inactive_equality_is_not_a_false_optimal` — `min ½‖x‖² s.t.
  x₁+x₂=2`, no bounds; true optimum `(1,1)`. Warm-started at `(0,0)` with the
  single equality row marked `Inactive`. **Pre-fix FAILS**: the inner loop sees
  no active rows, computes `p = −Hx − g = 0`, declares KKT-stationarity, finds
  no active row to drop, and returns `Optimal` at `(0,0)` — residual
  `|x₁+x₂−2| = 2.0`. **Post-fix PASSES**: the audit flags the violation, elastic
  mode recovers `(1,1)`, status `Optimal`.
- **Verification summary**: pre-fix the new test FAILS (false `Optimal` at
  `(0,0)`); post-fix it PASSES. Full `pounce-qp` suite green (75 unit + 1 + 5
  integration, 0 failed); the `pounce-algorithm` QP consumer (active-set SQP +
  l1-elastic) green (245 unit + SQP/elastic integration, 0 failed) — the audit
  does not perturb any feasible-result path.

## M6 detail

- **Bug** (`crates/pounce-sensitivity/src/convenience.rs`): `SensSolve::run`
  installs an `on_converged` callback that performs the post-solve sensitivity
  work (parametric step, reduced Hessian, eigendecomposition) and writes its
  results into a side-channel `CallbackOut` (via `Rc<RefCell<_>>`). Every failure
  branch in that callback sets `outbox.error = Some(message)` and returns early:
  - no current iterate at convergence (line ~234);
  - a pinned index that is an inequality / not in the equality c-block
    (`full_g_to_c_block` → `None`, line ~296);
  - `PdSensBacksolver::new` failure (line ~311);
  - `IndexSchurData::from_parts` failure (line ~321);
  - `SensApplication::parametric_step` returning `false` (line ~339);
  - `compute_reduced_hessian` / `compute_reduced_hessian_eigen` returning
    `false` (lines ~364, ~372).
  The result builder (`SensResult { status, x: out.x.clone(), … }`) copied every
  `out.*` field **except** `error`, and `CallbackOut.error` was annotated
  `#[allow(dead_code)]` — so the diagnostic was written and immediately
  discarded.
- **Why it matters**: the callback only runs *after* the IPM solve converged, so
  `status` is `SolveSucceeded` (or `SolvedToAcceptableLevel`) regardless of the
  sensitivity outcome. On failure the requested outputs (`dx`, `dx_full`,
  `reduced_hessian`, …) are left `None` — which is *exactly* the same state as
  "the caller didn't request that computation." A caller doing
  `SensSolve::new(pins).with_deltas(dp).run(...)` and reading `result.dx` cannot
  tell a genuine sensitivity failure from a no-op. The review's framing: "a
  failed `parametric_step` yields `dx: None` with `status: SolveSucceeded`,
  indistinguishable from 'not requested'."
- **Fix**:
  1. Add `pub error: Option<String>` to `SensResult`, documented (type-level +
     field-level) as the dedicated channel for *sensitivity-stage* failures that
     `status` cannot express, and noting that callers must check it to
     distinguish failure from not-requested.
  2. Copy `error: out.error.clone()` in the result builder.
  3. Remove `#[allow(dead_code)]` from `CallbackOut.error` (now genuinely read).
  4. Update the two `SensResult` literals in `diff_handoff.rs` unit tests with
     `error: None`.
  5. **End-to-end surfacing**: the Python binding (`pounce-py/src/problem.rs`),
     the primary user-facing consumer, builds an `info` dict from the
     `SensResult`; it now sets `info["sens_error"]` (`Option<String>` →
     `None` / message). Previously the Python layer had no visibility into a
     sensitivity failure either.
- **Test** (`tests/convenience_api.rs`):
  `sens_solve_surfaces_sensitivity_stage_failure`. Reuses the known-good
  `ParametricTNLP` (`m = 4`, all equalities) and `make_app()` so the IPM solve
  reliably converges and fires `on_converged`; then pins an out-of-range index
  (`99`) so `full_g_to_c_block` returns `None` and the callback takes the
  "only equality constraints can be pinned" failure branch, writing `error`.
  Asserts the solve converged (`status` success) **and** `error.is_some()` **and**
  `dx.is_none()`. A paired happy-path solve (`pins = [2, 3]`, real deltas)
  asserts `error.is_none()` + `dx.is_some()`, guarding against the fix
  over-reporting. The out-of-range pin exercises the identical
  `outbox.error = Some(_)` → (previously discarded) plumbing as the
  `parametric_step` branch the review cited; both are closed by the same
  one-field propagation.
- **Verification summary**: pre-fix the new test FAILS (the swallowed error
  leaves `error == None` while `status == SolveSucceeded` and `dx == None`),
  confirmed by temporarily forcing `error: None` in the builder; post-fix it
  PASSES. Full `pounce-sensitivity` suite green (64 tests across 7 binaries, 0
  failed) and `pounce-py` compiles clean with the new `info["sens_error"]` key.

## M7 detail

- **Bug** (`crates/pounce-qp/src/qps.rs`): the QPS section dispatcher mapped all
  three quadratic-section headers to one state —
  `Some("QUADOBJ") | Some("QSECTION") | Some("QMATRIX") => section = Section::Quadobj`
  (old line 132) — but the conventions are **not** interchangeable:
  - `QUADOBJ` / `QSECTION` (Maros-Mészáros / CPLEX): the objective Hessian is
    given as a **single triangle** — each off-diagonal pair `H_ij` appears once.
  - `QMATRIX` (CPLEX full-matrix): the **entire symmetric matrix** is listed —
    both `(i,j)` and the mirror `(j,i)`, each carrying the same value.
  The `Section::Quadobj` content parser pushes every raw `(i_col, j_col, val)`
  triplet to `h_entries`. The later lower-triangle normalization,
  `let (lo, hi) = if i >= j { (j, i) } else { (i, j) }; h_irow.push(hi+1);
  h_jcol.push(lo+1); h_val.push(v);`, maps both QMATRIX mirror entries onto the
  **same** lower triplet `(hi, lo)`. pounce's evaluator **sums** all triplets at
  a position, so every off-diagonal ends up **2×** its file value. The diagonal
  (`i == j`) is listed once and is unaffected.
- **Why it matters**: a QMATRIX-format problem is silently solved with the wrong
  objective — `½xᵀHx` with all off-diagonals doubled — yielding a wrong optimum
  with no error. QMATRIX is a standard, widely-emitted convention (CPLEX `.qps`
  export, parts of the Maros-Mészáros distribution), so this is a correctness
  bug on real third-party input, not a contrived edge case.
- **Latent — zero prior coverage**: no `.qps` data in the repo uses QMATRIX
  (`grep -rln QMATRIX` over tests/benchmarks/data → empty). The
  `mm_published_optima` integration fixtures are all QUADOBJ, which is why the
  suite was green despite the defect — the QMATRIX branch was never exercised.
- **Fix**: split the header match — `QMATRIX` sets a new `quad_is_full = true`
  flag while `QUADOBJ` / `QSECTION` set it `false`. In the content parser, when
  `quad_is_full && i_col < j_col`, `continue` (skip the strict-upper mirror) so
  each off-diagonal survives exactly once and normalizes correctly; diagonal and
  lower entries are kept. Single-triangle sections keep every entry, so the
  already-correct QUADOBJ path is byte-for-byte unchanged. The fix is internal
  to `parse_qps` (signature/return type unchanged) — no downstream API impact.
- **Tests** (`src/tests/qps_unit.rs`), with a new `h_at(model, irow, jcol)`
  helper that sums the parsed triplets at a lower-triangle position (the
  *effective* H entry the solver sees):
  - `parse_qps_qmatrix_full_matrix_does_not_double_off_diagonals` — parses a
    QMATRIX file for `H = [[2, 1], [1, 2]]` (lists `X1·X1`, `X1·X2`, `X2·X1`,
    `X2·X2`) and asserts `H_21 == 1.0` (not 2.0) with `H_11 == H_22 == 2.0`.
    Pre-fix **FAILS** with `H_21 = 2`; post-fix passes.
  - `parse_qps_quadobj_single_triangle_keeps_off_diagonal` — the same `H` via a
    single `X1·X2 = 1` QUADOBJ entry; asserts `H_21 == 1.0`. Guards the
    single-triangle path against the QMATRIX fix regressing it (passes pre- and
    post-fix).
- **Verification summary**: fail-first confirmed by short-circuiting the new
  guard (`if false && quad_is_full && …`) — the QMATRIX test reports
  `H_21 = 2`, the QUADOBJ test stays correct. With the fix restored, full
  `pounce-qp` suite green (77 lib incl. the 2 new + 1 + 5 `mm_published_optima`
  integration, 0 failed).

## M8 detail

- **Bug** (`crates/pounce-l1penalty/src/wrapper.rs`): `L1PenaltyBarrierTnlp`
  lifts an inner TNLP with `n` original variables into an augmented problem with
  `n + 2·m_eq` variables (the extra `2·m_eq` are the `p`/`n` elastic slacks for
  each equality row). Every forwarding method strips the augmented tail before
  calling the inner:
  - `eval_f` → `self.inner…eval_f(&x[..n], …)` (line ~362)
  - `eval_grad_f` → `…eval_grad_f(&x[..n], …, &mut grad_f[..n])` (line ~375)
  - `eval_g` → `…eval_g(&x[..n], …)` (line ~388)
  - `eval_h` → `let inner_x = x.map(|xa| &xa[..n]); …eval_h(inner_x, …)` (~480)

  but `eval_jac_g` forwarded the **full augmented** `x: Option<&[Number]>`
  unchanged to both the `Structure` (line ~416) and `Values` (line ~445) inner
  calls. So the inner's `eval_jac_g` received a slice `2·m_eq` entries longer
  than its declared `n`.
- **Why it matters**: the slack columns of the augmented Jacobian are filled by
  the wrapper itself (the `-1`/`+1` entries at columns `n + k` / `n + m_eq + k`),
  so the inner only needs its own `n` variables. An inner that reads `x[j]` for a
  fixed `j < n` is unharmed — which is why every in-repo inner (and thus every
  existing test) passed despite the defect. But an inner that (a) validates
  `assert_eq!(x.len(), n)` defensively, or (b) iterates `x.iter()` /
  `x.iter().enumerate()`, sees `2·m_eq` extra trailing values (the slacks) and
  either panics or computes against out-of-contract data. The review flags it as
  "latent." Beyond the concrete failure, the inconsistency with the other four
  forwarders is itself a maintenance hazard.
- **Fix**: mirror `eval_h` — compute `let inner_x = x.map(|xa| &xa[..n]);` once at
  the top of `eval_jac_g` and pass `inner_x` (not `x`) to both inner calls. The
  `Structure` branch typically receives `x = None` (structure is x-independent),
  and `None.map(..) == None`, so that path is unaffected; the `Values` branch now
  hands the inner exactly its `n` original variables. No change to the wrapper's
  own slack-column entries.
- **Test** (`wrapper.rs` `#[cfg(test)] mod tests`):
  `jacobian_passes_inner_only_original_x`. Defines a `LenSpy` inner TNLP
  (`n = 2`, `m = 1`, one equality row) that records the length of the `x` slice
  it is handed in `eval_jac_g` into a shared `Rc<Cell<usize>>`. The test wraps it
  (`ρ = 1`, augmented `n = 4`), calls the wrapper's `eval_jac_g` in `Values` mode
  with `x = [0.4, 0.5, 0.2, 0.3]` (length 4), and asserts the inner saw length
  **2** — its original-variable count — not 4.
- **Verification summary**: pre-fix the inner sees length **4** and the test
  **FAILS** ("received x of length 4 but expected the 2 original vars only"),
  confirmed by temporarily reverting the `Values`-branch argument from `inner_x`
  back to `x`; post-fix it sees 2 and passes. Full `pounce-l1penalty` suite green
  (11 tests, 0 failed) and the sole consumer `pounce-algorithm` green (245 unit +
  all integration binaries, 0 failed).

## M9 detail

- **Bug** (`pounce-restoration`): the restoration entry/clone paths read outer
  iterate blocks with the pattern
  `v.as_any().downcast_ref::<DenseVector>().map(|d| d.expanded_values())
  .unwrap_or_else(|| vec![0.0; dim])`. The `Option` being unwrapped is the
  *downcast result*, so a failed downcast — a block that is **not** a
  `DenseVector` (e.g. a `CompoundVector`/homogeneous compound) — is silently
  replaced with a zero vector. The restoration NLP is then seeded from a zero
  residual / zero multiplier with **no diagnostic**, quietly corrupting the
  start point. `DenseVector::expanded_values()` already materializes the
  *homogeneous* dense case correctly (`vec![scalar; dim]`), so only a genuinely
  non-dense block triggers the fallback.
- **Why it matters**: the *write* side of the same initializer asserts the
  invariant loudly — `downcast_dense_mut` (`init.rs:475`,
  `resto_inner_solver.rs:802`, …) does `.expect("expected a DenseVector
  component")`. The read side silently swallowing the identical mismatch is
  inconsistent and strictly worse: a zeroed residual produces a plausible-looking
  but wrong restoration solve instead of a crash that pinpoints the bug.
- **Sites fixed (all in `pounce-restoration`)**:
  - `init.rs` — 7 inline reads: `c_vec`, `d_minus_s_vec`, `s`, `z_l`, `z_u`,
    `v_l`, `v_u` (the outer residuals and bound multipliers).
  - `resto_inner_solver.rs:775` and `resto_resto.rs:234` — the shared
    `expanded_dense_values(v, fallback_dim)` helper (one copy in each file),
    used by the dense-clone routines.
- **Scope correction vs. the review's citation**:
  - `pounce-sensitivity/src/solver.rs:380-388` and `convenience.rs:397-405`: a
    `grep` for the zero-fill pattern finds **no** occurrence anywhere in
    pounce-sensitivity. Those line numbers now point at
    `IndexSchurData::from_parts` (solver.rs) and the `SensResult { … }` builder
    (convenience.rs) — unrelated code, and the convenience.rs lines shifted when
    the M6 fix added the `error` field. No silent-downcast bug exists there.
  - `aug_resto_system_solver.rs:553`: `lr.get_diag().map(|d| orig_rows(…))
    .unwrap_or_else(|| vec![0.0; n])`. Here the `Option` is `get_diag()`'s own
    return — a low-rank update legitimately *may have no diagonal*, in which case
    a zero diagonal contribution is the correct value, not a masked failure.
    Excluded by design.
- **Fix**:
  1. `init.rs`: add `expanded_dense_or_panic(v: &dyn Vector, what: &str) ->
     Vec<Number>` that panics with a labelled message
     (`"…outer {what} must be a DenseVector…"`) on a failed downcast, and route
     all 7 inline sites through it (passing a human-readable block label). This
     also de-duplicates the 7 copies of the pattern.
  2. `resto_inner_solver.rs` / `resto_resto.rs`: change the two
     `expanded_dense_values` helper bodies from the zero-fill fallback to a
     `panic!` (keeping the `fallback_dim` parameter only to size the diagnostic
     message). All callers are unchanged.
  Read and write sides are now symmetric: a non-`DenseVector` block fails loudly
  in both directions.
- **Test** (`init.rs` `#[cfg(test)] mod tests`):
  - `expanded_dense_or_panic_panics_on_non_dense` — builds a minimal 1-block
    `CompoundVector` (via a `make_compound` helper; the compound does not
    downcast to `DenseVector`) and asserts the helper panics
    (`#[should_panic(expected = "must be a DenseVector")]`).
  - `expanded_dense_or_panic_returns_values_for_dense` — a real `DenseVector`
    round-trips `[1.0, -2.0, 3.5]`, guarding the happy path against the fix.
- **Verification summary**: pre-fix the `should_panic` test FAILS with "test did
  not panic as expected" (the helper returns `vec![0.0; dim]`), confirmed by
  temporarily restoring the silent `vec![0.0; v.dim()]` fallback in
  `expanded_dense_or_panic`; post-fix it panics and passes. Full
  `pounce-restoration` suite green (105 lib + all integration binaries, 0 failed)
  and the downstream `pounce-algorithm` consumer green (245 unit + all
  integration, 0 failed).

## M10 detail

- **Claim** (review §M10): the Schur-update QP path does no inertia re-check
  after working-set changes and assembles `K_max` in `O(m·nnz(A))` per reset, so
  the doc claim of being "algorithmically identical to the refactor-per-iteration
  path" (`solver.rs:1137`) does not hold for indefinite reduced Hessians after a
  drop.
- **Mechanism (confirmed by inspection)**: the refactor path runs
  `factorize_with_inertia_control` on **every** inner iteration —
  `solve_general` at `solver.rs:734`, `solve_box_constrained` at `:238` — so an
  indefinite reduced Hessian triggers a δ-shift (`H += δI` on the H-block) before
  the step is computed. The Schur path (`solve_general_schur`, `:1142`) factors
  `K_max` once via `SchurState::reset` (which *does* run the same δ-shift inertia
  control, `schur.rs:249`), then for each working-set change applies a rank-2 SMW
  update through `apply_change` (`schur.rs:318`) that **never re-checks inertia**.
  `reset` is only re-invoked when `needs_reset()` is true, i.e. after
  `max_schur_updates_before_refactor = 50` accumulated changes. Between resets a
  **drop** (`solver.rs:1234`, `going_active = false`) enlarges the active-set
  null space and can expose a negative-curvature direction that the cached factor
  leaves unregularized; an **add** (`:1334`, `going_active = true`) only shrinks
  the null space and cannot introduce new negative curvature. So only drops break
  parity, and only when the reduced Hessian is indefinite.
- **Why it is latent**: `QpOptions::use_schur_updates` defaults `false`
  (`options.rs:112`); a `grep` of the whole workspace finds it set `true` only in
  `pounce-qp`'s own parity tests — **no production caller flips it** (the SQP
  driver `sqp_alg.rs` keeps the default and feeds `HessianInertia::Psd`). For a
  PD reduced Hessian no shift is ever needed, so the two paths are provably
  identical; the gap exists only for indefinite `H` on the opt-in path.
- **Verification attempts (could not force a divergence)**: a scratch
  differential test put an indefinite QP through both paths
  (`use_schur_updates` false vs true) and compared `x`, `obj`, `status`:
    1. `H = diag(-1, 2)`, `g = (2, 0)`, box `[-1,1]²`, warm-started with both
       bounds `AtUpper` so the solver must drop `x₁` (the negative-curvature
       coordinate) into a now-indefinite reduced system. **Result: both paths
       `Optimal` at `x = (-1, 0)`, `obj = -2.5` — identical.**
    2. Same `H`/`g`, but `x₁` unbounded (`±∞`) and `x₂ ∈ [-1,1]`, so the dropped
       direction is unbounded below. **Result: both paths `MaxIter` at the same
       `x ≈ (-12.93, 1.0)`, same `obj` — identical.**
  In both, the unshifted Schur step and the refactor step coincide because (a)
  the ratio test immediately re-adds a blocking bound, self-correcting an ascent
  step, and (b) a single 1-D negative-curvature exposure frequently still yields
  a KKT inertia matching `expected_neg`, so even the refactor path takes **no**
  shift. Constructing a robust, deterministic divergence proved impractical —
  same conclusion as M1.
- **Disposition**: **VERIFIED by inspection, DEFERRED for behavior** (mirrors
  M1). The one *verifiable* defect — the overclaiming doc comment — **is fixed**:
  `solve_general_schur`'s doc now states the equivalence holds for PD reduced
  Hessians and spells out the indefinite-H inertia caveat (drop-vs-add curvature
  argument, the `reset`-only inertia control, and the latency on the opt-in
  path). The behavioral fix (force `schur.reset(...)` unconditionally after every
  drop, restoring per-change inertia control) is **not applied**: without a
  failing test to anchor it and given the numerical delicacy / blast radius of
  changing inertia handling on a path no production code exercises, the safe
  disposition is to document rather than perturb.
- **Perf sub-claim**: `build_k_max_triplet` iterating all of `A` per general
  slot is genuinely `O(m·nnz(A))` per reset, but that is a performance property,
  not a correctness bug, and is not naturally regression-testable; noted for a
  future optimization pass, not fixed here.
- **Tests**: no new test (no deterministic divergence to assert). The scratch
  differential probes were removed after confirming agreement. `cargo test -p
  pounce-qp` green (77 lib + 1 + 5 integration, 0 failed) with the doc change in
  place.

## M11 detail

- **Claim** (review §M11): `crates/pounce-cli/src/qp_extract.rs` builds `A`/`G`
  from `con_linear` only, while the classifier admits rows whose nonlinear
  expression reduces to degree ≤ 1 and the SOCP extractor handles them
  (`nl_lin` + `const_shift`). LP/QPs with linear/constant terms inside the
  nonlinear tree get silently wrong constraints on the convex path.
- **Verified**: the QP constraint loop (`extract_qp_with_map`, the `for (row,
  lin) in prob.con_linear.iter().enumerate()` block) consumed only `lin` and the
  raw bounds `g_l`/`g_u`; it never touched `prob.con_nonlinear[row]`. By
  contrast the *objective* in the same function already sums `obj_linear` +
  `analyze_quadratic_full(obj_nonlinear).1` (the tree linear part) at lines
  80/98, and `extract_socp_with_map` already folds the per-constraint tree
  `nl_lin` and shifts the bound by the tree constant at lines 355-396. So the QP
  path was the lone place that dropped a constraint's folded degree-≤1 terms.
- **Why it matters**: AMPL/Pyomo routinely emit a constraint body inside the
  nonlinear tree when it arose from a cancelled quadratic or a defined variable
  even though the result is linear (the classifier explicitly allows this:
  `dispatch.rs` admits a row if its nonlinear part is degree ≤ 1). For such a
  row, `con_linear` is empty (or partial) and the real coefficients/constant
  live in `con_nonlinear`. The old code emitted a constraint with the wrong (or
  missing) coefficients and an unshifted bound → a silently wrong convex solve.
- **Fix** (`qp_extract.rs`, QP constraint loop): for each row, compute
  `let (nl_lin, const_shift) = analyze_quadratic_full(&prob.con_nonlinear[row],
  n).map(|(_, l, k)| (l, k)).unwrap_or_default();`, accumulate `con_linear` +
  `nl_lin` into a dense `coef` vector, emit only the nonzero entries
  (`nonzeros()` closure, matching the SOCP path so all-zero rows are skipped),
  and shift every RHS by `const_shift`: equality `b.push(lo − k)`, upper
  `h.push(hi − k)`, lower `h.push(−(lo − k))`. For these linear rows
  `analyze_quadratic_full` returns an empty Hessian, so the quadratic part is
  correctly ignored; a genuinely quadratic constraint would have been routed to
  the SOCP path by the classifier, not here. Index safety: `con_nonlinear` is
  built parallel to `con_linear` (both length `m`, each row initialized to
  `Expr::Const(0.0)`) at parse time (`pounce-nl/src/nl_reader.rs:295`).
- **Test** (`qp_extract::tests::constraint_linear_terms_folded_in_tree_are_recovered`):
  `min x0  s.t.  x0 − 3 ≥ 0`, with the entire `x0 − 3` body placed in
  `con_nonlinear[0]` (as `Sub(Var0, Const3)`) and `con_linear[0]` left empty —
  the exact shape the bug mishandles. Asserts `qp.m_ineq() == 1`, the solve is
  `Optimal` at `x0 = 3` (1e-5), and the recovered dual is finite. **Fail-first**:
  temporarily forcing `(nl_lin, const_shift) = Default::default()` behind an
  `if false {…} else { Default::default() }` guard reproduces the bug — the
  constraint collapses to a vacuous `0 ≤ 0` row, `min x0` is unbounded, and
  `assert_eq!(sol.status, Optimal)` fails at `qp_extract.rs:934`. Restoring the
  fix makes it solve to `x0 = 3`. Mirrors the existing SOCP analogue
  `extract_and_solve_socp_folds_constraint_constant`.
- **Result**: full `pounce-cli` suite green (155 lib + all integration binaries,
  0 failed).

## M12 detail

- **Claim** (review §M12): `crates/pounce-solve-report/src/lib.rs:453` maps
  `DivergingIterates` to AMPL code 401 ("limit") instead of the 300-range
  ("unbounded"); upstream Ipopt's ASL driver maps it to 300 and the CLI's own
  convex path reports the same condition as 300 — an internal divergence.
- **Verified**: `status_to_solve_result_num` had `DivergingIterates => 401`. The
  AMPL `solve_result_num` convention (Gay 2005) buckets results by hundreds:
  0 solved, 100 warning/acceptable, 200 infeasible, **300 unbounded**, 400 limit
  reached, 500 failure. `DivergingIterates` is precisely Ipopt's *unboundedness*
  signal — the iterates diverge to infinity because the problem has no finite
  minimizer — so it is an unbounded (300) outcome, not a limit (400/401) outcome.
- **Internal inconsistency confirmed by reading the CLI**: in `pounce-cli`,
  `main.rs:1165` maps the convex solver's unbounded status `QpStatus::DualInfeasible
  → ApplicationReturnStatus::DivergingIterates` (comment: `// unbounded`). The
  convex path's *direct* numeric mapping at `main.rs:1276` and `:1425` reports
  `QpStatus::DualInfeasible => ("Problem is unbounded (dual infeasible).", false,
  300)`, and the range legend at `main.rs:1271-1272` states "300–399 unbounded,
  400–499 limit". So the same physical outcome (unbounded) emitted **300** when
  the convex path wrote the code directly, but **401** when it flowed through
  `ApplicationReturnStatus` → `status_to_solve_result_num`. AMPL/Pyomo readers
  key off the hundreds digit, so an unbounded model was mislabeled as a limit.
- **Fix** (`lib.rs`): change the match arm to `DivergingIterates => 300` and
  extend the function doc to (a) add "300s = unbounded" to the range legend and
  (b) explain that `DivergingIterates` is the unboundedness signal and belongs in
  300, matching upstream Ipopt's ASL driver and the CLI convex path. One-line
  behavioral change; no other arm touched.
- **Test** (`tests::diverging_iterates_maps_to_unbounded_range`): asserts
  `status_to_solve_result_num(DivergingIterates) == 300`, plus a guard set pinning
  `SolveSucceeded → 0`, `InfeasibleProblemDetected → 200`,
  `MaximumIterationsExceeded → 400`, `SearchDirectionBecomesTooSmall → 400`,
  `RestorationFailed → 500` so the bucket convention is locked against future
  drift. **Fail-first**: reverting the arm to `401` makes the first assertion
  fail (`left: 401, right: 300`); restoring `300` passes. A `grep` for `401`
  across the workspace found no test or caller hard-coding the old value, so the
  change is safe for downstream consumers (`pounce-cli` calls the function at
  `main.rs:1000,1088`).
- **Result**: `pounce-solve-report` suite green (7 tests, 0 failed); full
  `pounce-cli` suite green (no failures).

## M13 detail

- **Bug**: under `presolve=yes` on the general-NLP route, `PresolveTnlp`
  drops redundant constraint rows, so the solver operates on a reduced
  (`m_out`) row space. The CLI's solution capture sits **outside** the
  presolve wrapper:
  - the IPM `on_converged` hook builds `lambda` via
    `OrigIpoptNlp::pack_lambda_for_user(y_c, y_d)` (`main.rs:612`) — the
    `y_c`/`y_d` are reduced, so the result has length `m_out`;
  - the active-set SQP fallback reads `CountingTnlp::captured_solution`
    (`main.rs:950`), and `CountingTnlp` wraps *outside* presolve too.
  Both `.sol` (`main.rs` writer) and JSON (`SolutionInfo::lambda`) then
  carried `m_out` duals. AMPL/Pyomo read the `.sol` dual block
  positionally against the originating `.nl`'s `m` constraints, so a
  short block mis-aligns (wrong row → wrong dual) or is rejected.
- **Reachability / reproduction** (run, not just inspected):
  - `target/debug/pounce crates/pounce-cli/tests/fixtures/lp_afiro.nl
    solver_selection=nlp presolve=yes` → "dropped 4 redundant rows";
    pre-fix `.sol`/JSON `lambda` length **23**, vs **27** for
    `presolve=no`.
  - `dual_order.nl` (drops both of 2 rows) → pre-fix a **zero-length**
    dual block (`m_out = 0`) against a 2-constraint `.nl`.
- **Why the data was already available**: `PresolveTnlp::finalize_solution`
  (`crates/pounce-presolve/src/lib.rs`) reconstructs the inner-sized
  `lambda_full` by scattering kept-row duals through `rows_kept` **and**
  splicing recovered multipliers for dropped rows (the Phase-0
  `reduction_frame::recover_dropped_multipliers` walk), then forwards the
  full-length solution to `inner.finalize_solution`. The correct
  full-space dual vector was being computed and handed to the inner TNLP
  — the CLI just never read it, preferring its own reduced-space capture.
- **Fix**:
  - `pounce-presolve`: add a `finalized_full_solution:
    Option<(Vec<Number>, Vec<Number>)>` field to `PresolveTnlp`,
    populated at `finalize_solution` with `(sol.x.to_vec(),
    lambda_full.clone())` just before forwarding to the inner TNLP, and a
    public `finalized_full_solution()` getter.
  - `pounce-cli/src/main.rs`: after the active-set backfill, when
    `presolve_handle` is `Some` and `n_dropped_rows() > 0`, swap the
    captured `lambda` for `finalized_full_solution()`'s full-length
    vector (keeping the existing `x`). Also size the `.sol` zero-fallback
    block and the JSON `problem.n_constraints` to the original `m`
    (`m_out + n_dropped_rows`) so the documented `lambda.len() ==
    n_constraints` invariant holds.
- **Convention check**: the lambda forwarded into the TNLP stack at
  finalize is already `pack_lambda_for_user`'s output
  (`application.rs:2189`, `finalize_via_orig_nlp`) — c/d split inverted,
  scaling unwound, `.nl` row order — so `PresolveTnlp` lifts a vector in
  exactly the same user-facing convention the `on_converged` path uses;
  the only delta is the lift to `m_in` + dropped-row recovery. Verified
  empirically: post-fix `lp_afiro` active-row duals match the
  `presolve=no` baseline tightly.
- **Dropped-row dual values**: a recovered dual is a *valid alternative*
  KKT certificate. For genuinely-slack redundant rows it reproduces the
  baseline value; where bound-tightening presolve migrated the dual onto
  a bound multiplier (`dual_order`'s rows), the constraint dual is
  legitimately 0 (the "force" now lives in `z_l`/`z_u`). Both satisfy
  stationarity. M13 is specifically the **length/alignment** defect, and
  that is fully fixed.
- **Test**: `crates/pounce-cli/tests/presolve_dual_length.rs::
  presolve_dual_block_keeps_original_nl_length` runs `lp_afiro` through
  the NLP path (`solver_selection=nlp`) with `presolve=no` then `=yes`,
  guards that presolve genuinely dropped rows (parses the stdout
  "dropped N redundant rows" summary), and asserts the presolved
  `lambda` length equals the baseline `m` **and** the report's
  `n_constraints`. **Pre-fix it FAILS** ("presolve dual block length 23
  != original .nl m 27"), confirmed by neutering the lambda swap with an
  `if false` guard.
- **Scope note**: `solver_selection=nlp` is required — under the default
  `auto` route `lp_afiro` dispatches to the convex IPM, which has its own
  `.sol` path and never wraps `PresolveTnlp`. Presolve defaults off, so
  this never affected a default run.
- **Result**: `pounce-presolve` green (207 lib + 9 doc, 0 failed); full
  `pounce-cli` green (155 lib + all integration incl. the new test, 0
  failed).

## M14 detail

- **Issue**: any `--minima` tuning knob (`--seed`, `--patience`, `--dedup`,
  `--sobol`/`--no-sobol`, …) silently switches the whole run into multistart
  mode, even with no `--minima <method>` or `--multistart`. The help text
  advertises that only `--minima`/`--multistart` enable global search.
- **Mechanism** (`crates/pounce-cli/src/cli.rs`): the `minima_num!` macro and
  the `--sobol`/`--no-sobol` arms persist their value with
  `minima.get_or_insert_with(MinimaArgs::default)`. That insert materializes
  `Some(MinimaArgs { method: Deflation, .. })`. `main.rs:420` then reroutes the
  entire run through `pounce_cli::minima::run(...)` on *any* `Some(minima)`, so
  the lone knob silently enables global search — different console output and a
  `.sol` written with zero duals.
- **Reproduced**: `pounce lp_afiro.nl --seed 42` (no method flag) prints
  `Searching for up to 10 minima via \`deflation\`…`, i.e. it entered
  multistart purely from `--seed`.
- **Fix**: introduce two parser-local flags —
  - `minima_method_explicit` (bool), set *only* by the `--minima` and
    `--multistart` arms, and
  - `minima_knob: Option<&'static str>`, the first tuning knob seen (recorded
    in the `minima_num!` macro and the `--sobol`/`--no-sobol` arms).
  After parsing, in the `if !help && !version && !about && !cite` block, before
  the problem is required:
  ```rust
  if let Some(knob) = minima_knob {
      if !minima_method_explicit {
          return Err(format!(
              "{knob} is a --minima tuning knob and has no effect on its own; \
               enable global search with --minima <method> or --multistart"
          ));
      }
  }
  ```
- **Verified post-fix**: lone `--seed 42` now errors
  `--seed is a --minima tuning knob and has no effect on its own; enable
  global search with --minima <method> or --multistart`; `--multistart --seed 42`
  still parses (method=Multistart, seed=42) and `Searching … via \`multistart\``.
- **Tests** (`crates/pounce-cli/src/cli.rs`, `mod tests`):
  - `lone_minima_knob_without_method_is_rejected` — lone `--seed 42` and lone
    `--no-sobol` each error; the message names both the offending knob and
    `--minima`.
  - `minima_knob_with_explicit_method_is_accepted` — `--seed 7 --multistart`
    parses (order-independent) to `method = Multistart`, `seed = 7`.
- **Fail-first**: neutering the guard to `if false && !minima_method_explicit`
  makes the rejection test fail (lone `--seed 42` parses to
  `Some(MinimaArgs { method: Deflation, seed: 42 })`); restoring it passes.
- **Non-breaking**: every existing multistart test
  (`minima_method_and_shared_knobs`, `minima_strategy_knobs_are_optional_and_parsed`,
  `issue_103_mlsl_terminates`) pairs its knobs with an explicit method, so the
  new guard never trips on them.
- **Result**: full `pounce-cli` green (157 lib + all integration, 0 failed).

## M15 detail

- **Issue**: the `-AMPL` flag advertises "AMPL solver-protocol mode (for
  Pyomo / AMPL drivers)", but two real-AMPL invocation conventions were
  unsupported, so genuine AMPL (as opposed to Pyomo) could not drive pounce:
  1. **Extensionless stub** — AMPL runs a solver as `pounce mystub -AMPL`,
     passing the stub *without* the `.nl` extension and expecting `mystub.nl`
     to be read and `mystub.sol` written.
  2. **`pounce_options` env var** — AMPL conveys solver directives through a
     `<solver>_options` environment variable, not the command line.
  (Pyomo sidesteps both: it writes a full `<tmp>.nl` path and passes options
  as CLI `key=value` args, so it always worked.)
- **Repro (pre-fix)**:
  - `cp convex_qp.nl /tmp/mymodel.nl && pounce /tmp/mymodel -AMPL` →
    `pounce: failed to read /tmp/mymodel: could not read /tmp/mymodel: No such
    file or directory (os error 2)` (exit 2).
  - `pounce_options="max_iter=1" pounce /tmp/mymodel.nl` ran to normal
    convergence — the env var had no effect.
- **Fix (a) — stub resolution** (`crates/pounce-nl/src/nl_reader.rs`):
  `read_nl_file` resolves the path before reading: if `path.exists()` it is
  read verbatim; otherwise, if `append_extension(path, "nl")` exists, that is
  read instead. The `.col`/`.row` sibling-name lookups use the *resolved*
  path so they still hit `mystub.col` / `mystub.row`. New helper
  `append_extension` appends `.nl` to the full file name (AMPL convention:
  `my.model` → `my.model.nl`), as opposed to `Path::with_extension`, which
  would replace an existing extension. The fix is purely additive — an
  existing path is always read as-is, so `--nl-file`, the bare positional
  `.nl`, and the second-positional `.sol` form are unchanged. The `.sol`
  output path (`main.rs`) already derives correctly from the stub
  (`set_extension("sol")` on `mystub` → `mystub.sol`), so no change there.
- **Fix (b) — `pounce_options` env var** (`crates/pounce-cli/src/cli.rs`,
  `main.rs`): new pure `cli::options_from_env(&str) -> Vec<(String,String)>`
  splits the value on whitespace and parses each `key=value` token with the
  existing `parse_kv` (tokens without `=` are skipped). `main` reads
  `std::env::var("pounce_options")` after argv parsing and **prepends** the
  parsed pairs to `args.set_options`, so the command-line `key=value` options
  (pushed later, applied last-wins via `read_from_str`) override the env var —
  matching AMPL, where command-line options after the stub win. The `-AMPL`
  help text / `PATH` doc now describe both conventions.
- **Verified (post-fix)**:
  - `pounce /tmp/mymodel -AMPL solver_selection=nlp` → `Optimal Solution
    Found`, exit 0, and `/tmp/mymodel.sol` written next to the stub.
  - `pounce_options="max_iter=1" pounce …mymodel.nl solver_selection=nlp` →
    `Maximum Number of Iterations Exceeded` (env applied).
  - `pounce_options="bogus_opt=1" pounce …mymodel.nl` → exit 2,
    `pounce: failed to set bogus_opt=1: … OPTION_INVALID` (env read+applied).
  - `pounce_options="max_iter=1" pounce …mymodel.nl max_iter=3000` →
    `Optimal Solution Found` (CLI overrides env).
- **Tests**:
  - `nl_reader::tests::read_nl_file_resolves_extensionless_ampl_stub` — a
    stub with no extension resolves to `<stub>.nl`, and a sibling `.col`
    rides along.
  - `nl_reader::tests::read_nl_file_prefers_exact_path_over_nl_sibling` — an
    existing exact path is read verbatim even when a `<file>.nl` sibling
    exists (guards against silent redirection).
  - `nl_reader::tests::append_extension_appends_rather_than_replaces` — pins
    the append-vs-replace semantics (`my.model` → `my.model.nl`).
  - `cli::tests::options_from_env_parses_whitespace_separated_pairs` and
    `…_skips_non_kv_tokens_and_empty` — the env-string parser.
  - integration `tests/ampl_driver_conventions.rs`:
    `extensionless_stub_resolves_to_nl_and_writes_sol`,
    `pounce_options_env_var_is_applied` (bogus option → exit 2 + "failed to
    set"), `cli_key_value_overrides_pounce_options_env`.
- **Fail-first**: neutering the stub fallback to `if true || path.exists()`
  fails `read_nl_file_resolves_extensionless_ampl_stub` and the stub
  integration test (`could not read …/mystub`); gating the env merge off
  fails both env integration tests (no "failed to set", exit ≠ 2). Both
  restored after confirmation.
- **Scope note**: AMPL's rarer `keyword value` (space-separated, no `=`)
  option spelling is intentionally *not* supported — it matches the existing
  CLI grammar, which has no `key value` form either; such tokens are skipped
  rather than guessed at. The review item itself flagged uncertainty over
  whether genuine AMPL is in scope; since `-AMPL`'s own help text claims AMPL
  driver support, honoring these two well-defined conventions makes that
  claim true.
- **Result**: `pounce-nl` green (78, 0 failed); full `pounce-cli` green (159
  lib + all integration incl. the new test file, 0 failed).

## M16 detail

- **Issue**: `OrigIpoptNlp` (`crates/pounce-nlp/src/orig_ipopt_nlp.rs`) splits
  the user's combined constraint vector `g` into equalities (`c`) and
  inequalities (`d`). Each subsystem computed the *full* `g` (and the *full*
  Jacobian) independently:
  - `eval_c_internal` and `eval_d_internal` each called user `eval_g` into a
    fresh `full_g`, then sliced their rows (`c_map` / `d_map`).
  - `eval_jac_c_internal` and `eval_jac_d_internal` each called `eval_jac_g`
    over all `nnz_jac_g_full` entries, then sliced (`jac_c_entry_in_g` /
    `jac_d_entry_in_g`).
  The filter line search needs `c`, `d`, `jac_c`, and `jac_d` at every
  iterate, so the dominant AD cost was paid twice per iterate — roughly a 2×
  tax on `.nl` problems, plus inflated underlying-eval accounting. Upstream
  Ipopt avoids this with tagged `full_g_` / `jac_g_` buffers.
- **Reproduced**: a counting `Hs071` TNLP (it already tallies `eval_g_calls`
  and `eval_jac_g_value_calls`). Pre-fix, `eval_c(x)` followed by `eval_d(x)`
  at one iterate drove `eval_g_calls == 2`; `eval_jac_c(x)` + `eval_jac_d(x)`
  drove `eval_jac_g_value_calls == 2`.
- **Fix**: introduce shared, tag-keyed full-space caches and route both
  subsystems through them.
  - New fields `full_g_cache: RefCell<Cache<Rc<Vec<Number>>>>` and
    `full_jac_g_cache: RefCell<Cache<Rc<Vec<Number>>>>` (both `Cache::new(1)`,
    keyed on the input vector's tag — the same `get_1dep`/`add_1dep`
    mechanism the per-subsystem caches use).
  - New private helpers `full_g(&self, x)` and `full_jac_g(&self, x)`: return
    the cached buffer on a tag hit; otherwise lift `x`, call the user
    `eval_g` / `eval_jac_g` once, fill NaN on failure (unchanged
    line-search-backtrack contract), memoize, and return `Rc<Vec<Number>>`.
  - `eval_c_internal` / `eval_d_internal` now do `let full_g = self.full_g(x)`
    and slice as before; `eval_jac_c_internal` / `eval_jac_d_internal` now do
    `let full_vals = self.full_jac_g(x)`. The scaling, equality-bound
    subtraction (`full_g[g_idx] - full_g_l[g_idx]`), and row-mapping code are
    untouched — only the *source* of the buffer moved, so numerics are
    identical.
  - Per-subsystem counters (`c_evals`, `d_evals`, `jac_c_evals`,
    `jac_d_evals`) are deliberately left incrementing once per produced
    vector — they measure c/d production, which still happens once each; the
    saving is the redundant *user* AD call, now elided on the second
    subsystem via the shared-cache hit.
- **Why size-1 caches suffice**: within an iterate every `eval_*` is at the
  same `x` (same tag) → one compute, subsequent hits. When the solver moves
  to a new iterate, `x`'s tag bumps and the single slot is replaced — exactly
  how the existing `c_cache`/`jac_c_cache` behave.
- **Verified (post-fix)**:
  - counting TNLP: `eval_c` + `eval_d` ⇒ `eval_g_calls == 1`; a new iterate
    (x mutated) ⇒ one more (== 2). `eval_jac_c` + `eval_jac_d` ⇒
    `eval_jac_g_value_calls == 1`.
  - values unchanged: at the HS071 start c = g1−40 = 12, d = g0 = 25.
  - end-to-end `pounce lp_afiro.nl solver_selection=nlp` still converges to
    the known optimum (−464.753, "Optimal Solution Found").
- **Tests** (`orig_ipopt_nlp::tests`):
  - `eval_c_and_eval_d_share_one_eval_g_per_iterate`
  - `eval_jac_c_and_eval_jac_d_share_one_eval_jac_g_per_iterate`
  - helper `build_orig_nlp_counting` retains a typed `Rc<RefCell<Hs071>>`
    aliasing the adapter's `dyn TNLP`, so the test can read the user-side
    call counters (the adapter only exposes `dyn TNLP`).
- **Fail-first**: neutering the two shared lookups with
  `.filter(|_| false)` (forcing every call to recompute) makes both tests
  fail with `left: 2, right: 1`; restored after confirmation.
- **Result**: `pounce-nlp` green (36, 0 failed); downstream `pounce-algorithm`
  and full `pounce-cli` suites green (0 failed).

## M17 detail

- **Issue**: `OrigIpoptNlp::eval_c_internal`
  (`crates/pounce-nlp/src/orig_ipopt_nlp.rs`) forms the equality residual
  `c_i = g[g_idx] - g_l[g_idx]`. For equalities `g_l == g_u`, so the
  subtracted RHS is a problem constant — but on every cache-missing iterate
  the code re-fetched it by calling the user `get_bounds_info`, into four
  freshly-allocated full-size scratch vectors (`tmp_x_l`, `tmp_x_u`,
  `full_g_l`, `full_g_u`), discarding all but the `full_g_l[g_idx]` entries.
  The filter line search evaluates `c` at every trial step, so this bounds
  call + four allocations recurred per trial in the hot path. Upstream Ipopt
  captures the constant once as `c_rhs`.
- **Reproduced**: the counting `Hs071` TNLP gained a `get_bounds_info_calls`
  tally. Pre-fix, snapshotting the count after construction and then running
  `eval_c` at six distinct iterates drove six additional `get_bounds_info`
  calls (count climbed from the 2-call construction baseline to 8).
- **Fix**: capture the equality RHS once at construction.
  - New field `c_rhs: Vec<Number>` (length `n_c`), documented as upstream's
    `c_rhs`.
  - Computed in `OrigIpoptNlp::new` from the `full_g_l` array already fetched
    there for the bound vectors: `classification.c_map.iter().map(|&g_idx|
    full_g_l[g_idx as usize]).collect()` — no extra bounds call.
  - `eval_c_internal` now forms `raw = full_g[g_idx] - self.c_rhs[i]` and the
    per-iterate `get_bounds_info` call plus the four `vec![0.0; …]`
    allocations are gone. The `full_g` source (M16's shared tag-keyed cache)
    and the scaling logic are untouched, so the produced residual is
    bit-for-bit identical.
- **Why it's sound**: bounds are fixed for a problem's lifetime (the TNLP
  contract; upstream reads them once into `c_rhs`/`d_l`/`d_u` at setup). The
  inequality path (`eval_d_internal`) never subtracted a bound and already
  needed no bounds fetch, so only `eval_c` carried this cost.
- **Verified (post-fix)**:
  - counting TNLP: `get_bounds_info_calls` stays at the post-construction
    baseline across six fresh-iterate `eval_c` calls (no per-iterate refetch).
  - residual value unchanged: at the HS071 start c = g1−40 = 12.
  - end-to-end `pounce lp_afiro.nl solver_selection=nlp` still converges to
    −464.75314761311961 ("Optimal Solution Found") — identical to pre-fix.
- **Test** (`orig_ipopt_nlp::tests::eval_c_does_not_refetch_bounds_per_iterate`):
  snapshots `get_bounds_info_calls` after construction, evaluates `eval_c` at
  six distinct iterates, asserts the count is unchanged and that c == [12.0].
- **Fail-first**: temporarily restoring the old per-iterate `get_bounds_info`
  fetch inside `eval_c_internal` made the test fail with `left: 8, right: 2`;
  restored after confirmation.
- **Result**: `pounce-nlp` green (37, 0 failed); downstream `pounce-algorithm`
  (245) and full `pounce-cli` (159 lib + all integration) green (0 failed).

## M18 detail

- **Issue**: the tape-AD gradient sweep `Tape::gradient_seed`
  (`crates/pounce-nl/src/nl_tape.rs`) allocates twice per call:
  - `forward(x)` (`:198`) returns a fresh `Vec<f64>` of forward values.
  - `reverse(vals, seed, grad)` (`:272`) allocates `adj = vec![0.0; n]` for
    the adjoint accumulator.
  The `.nl` front end (`nl_reader.rs`) deliberately builds **one tiny `Tape`
  per additive summand** — on large models ~10⁶ of them — so a single
  `eval_jac_g` (which calls `gradient_seed` for every constraint summand) or
  `eval_grad_f` (every objective summand) turns those two small allocations
  into millions of heap hits. The Hessian path already avoids this with the
  `forward_into` + caller-scratch pattern (`vals_scratch`/`dot_scratch`/
  `adj_scratch`/`adj_dot_scratch`, sized to `max_tape_n`); the gradient path
  did not reuse them.
- **Reproduced**: a counting `#[global_allocator]` wrapping `System`. On a
  sample tape (`x0*x1 + exp(x0)*x1 + x0²`), `gradient_seed` allocates on
  essentially every call — ≥1000 allocations across 1000 calls.
- **Fix** (mirror the Hessian scratch pattern for first-order AD):
  - New `pub fn gradient_seed_into(&self, x, seed, grad, vals, adj)` on
    `Tape`: guards the `seed == 0 || ops empty` fast path like
    `gradient_seed`, then runs the existing `forward_into(x, vals)` and a new
    `reverse_into(vals, seed, grad, adj)`. `grad` is accumulated into (not
    zeroed); `vals`/`adj` are caller arenas of length ≥ `ops.len()`.
  - New private `fn reverse_into(&self, vals, seed, grad, adj)` holding the
    full reverse-sweep body. It zeroes only `adj[0..n]` (so a dirty arena is
    fine) and sets `adj[n-1] = seed` — no allocation.
  - `reverse` now allocates an `adj` once and delegates to `reverse_into`, so
    `gradient_seed` and every existing FD-comparison test still exercise the
    same code with identical behavior.
  - The two hot-path callers in `nl_reader.rs` switch to `gradient_seed_into`:
    - `eval_grad_f`: `t.gradient_seed_into(x, 1.0, grad, &mut self.vals_scratch,
      &mut self.adj_scratch)` per objective tape.
    - `eval_jac_g` (Values): same, accumulating into `self.scratch_row_grad`.
    Both reuse the pre-existing `vals_scratch`/`adj_scratch` arenas (sized to
    `max_tape_n` = max ops length over all obj+con tapes, so always ≥ any
    single summand tape). Disjoint `self` fields, so the borrow checker
    accepts simultaneous `&self.con_tapes` / `&mut self.scratch_*` access.
- **Why no numerics change**: `forward_into`/`reverse_into` are the
  byte-for-byte same arithmetic as `forward`/`reverse` — only the *storage*
  for the forward values and adjoints moved from a per-call `Vec` to a reused
  arena. The arena is fully (re)written each call (`forward_into` writes all
  `[0,n)`; `reverse_into` zeroes `adj[0,n)`), so no stale state leaks.
- **Verified (post-fix)**:
  - counting allocator: `gradient_seed_into` performs **0** allocations
    across 1000 calls; `gradient_seed` performs ≥1000; both yield the
    identical gradient.
  - end-to-end NLP solves (gradient path exercised via tapes) unchanged:
    `convex_qp` → 2.0, `tame` → 0.0, `nonconvex_qp` → 1.0, all "Optimal
    Solution Found".
- **Test** (`crates/pounce-nl/tests/tape_gradient_no_alloc.rs`): installs a
  counting global allocator; a single test (alone in its integration binary,
  so no sibling test thread perturbs the global counter inside the counting
  window) asserts (a) `gradient_seed_into == gradient_seed` numerically,
  (b) the baseline allocates ≥1000× (the harness genuinely observes
  allocations), and (c) `gradient_seed_into` allocates 0× across 1000 calls.
- **Fail-first**: neutering `gradient_seed_into` with a throwaway
  `vec![0.0; ops.len()]` per call made assertion (c) fail with
  `left: 1000, right: 0`; restored after confirmation.
- **Result**: `pounce-nl` green (78 lib + 1 new integration test, 0 failed);
  downstream full `pounce-cli` green (0 failed).

## M19 detail

- **Issue (review M19)**: the `.nl` `d` segment supplies initial constraint
  multipliers, which the reader parses into `NlProblem::lambda0`
  (`crates/pounce-nl/src/nl_reader.rs:458`), but `NlTnlp::get_starting_point`
  copied only `x0` into `sp.x` and ignored `sp.lambda`. So
  `warm_start_init_point yes` silently began from zero multipliers, discarding
  the parsed duals. The module header even described the `d` segment as "read
  and discarded".
- **Reachability** (traced, not assumed): the warm-start request reaches the
  TNLP. `OrigIpoptNlp::get_starting_point` (`orig_ipopt_nlp.rs:1289`) sets
  `init_lambda: init_y_c || init_y_d` on the `StartingPoint` it hands the
  adapter, and on return reads `full_lambda` back into the algorithm-side
  `y_c`/`y_d` through `c_map`/`d_map` with `obj_scal`/constraint scaling
  (`:1320-1333`). The engine sets `init_y_c`/`init_y_d` when the user requests
  `warm_start_init_point yes`. With `get_starting_point` leaving `sp.lambda`
  untouched, that whole path warm-started from zeros.
- **Fix**: in `NlTnlp::get_starting_point`, when `sp.init_lambda` is set, copy
  `self.prob.lambda0` into `sp.lambda`. The `.nl` `d` segment carries no bound
  multipliers, so `z_l`/`z_u` are deliberately left to the engine's defaults.
  Also corrected the stale module-header doc comment: both the initial primal
  (`x`) and the initial dual (`d`) segment are now described as parsed (into
  `x0`/`lambda0`) and returned by `get_starting_point`, the duals feeding a
  `warm_start_init_point` solve.
- **Why cold starts are safe**: the two cold-start callers in `OrigIpoptNlp`
  (`:739`, `:1894`) pass `init_lambda: false`, so the new copy is skipped and
  cold-start behavior is byte-for-byte unchanged. End-to-end
  `lp_afiro solver_selection=nlp` still converges to −464.75314761311961
  ("Optimal Solution Found").
- **Test** (`nl_reader::tests::get_starting_point_returns_nl_initial_duals`):
  builds an equality-constrained `.nl` whose `d` segment is `d1\n0 2.5`,
  asserts the reader produced `lambda0 == [2.5]`, then drives
  `get_starting_point` twice — with `init_lambda: true` (asserting
  `lambda == [2.5]`) and with `init_lambda: false` on a buffer pre-filled to
  `[7.0]` (asserting it is left untouched). Confirms both the warm-start copy
  and that the gate is honored.
- **Fail-first**: neutering the copy with `if sp.init_lambda && false` made the
  warm-start assertion fail with `left: [0.0], right: [2.5]`; restored after
  confirmation.
- **Result**: `pounce-nl` green (79 lib + integration, 0 failed); downstream
  full `pounce-cli` green (0 failed).

## M20 detail

- **Issue (review M20)**: the convex IPM's two HSDE drivers accept a usable
  iterate when the KKT factorization / a back-solve breaks down (or the cap or
  a non-positive step is hit) while the best residual reached is already small.
  The symmetric `hsde.rs` accepts within `~1e3·tol` (`near_opt`, four break
  sites: refactor, the constant-direction solve, the predictor solve, the
  corrector solve); the non-symmetric `hsde_nonsym.rs` accepts within
  `~1e3·tol` (six break sites) and, on exit, restores the best iterate when
  `best_res < √tol` (`:1176`). Every one of these reported a bare
  `QpStatus::Optimal`. With no reduced-accuracy variant on `QpStatus` and no
  final-residual field on `QpSolution`, a residual of `1e-5`/`1e-4`
  (default `tol = 1e-8`) was indistinguishable from a genuine convergence —
  exactly the "silent tolerance relaxation" the review names. ECOS/Clarabel
  expose this as a distinct `*_INACC` status.
- **Reproduced (real instances, not synthetic)**: the exp/power-cone geometric
  programs in `exp_cone_vs_nlp`, `cblib_cbf`, and `cblib_vs_nlp` —
  `demb761`/`beck751`/`fang88`, the synthetic power cone `pow3`, log-sum-exp,
  entropy maximization, and the near-boundary GP — all reach their optimum
  through the non-symmetric driver's reduced-accuracy fallback, so they were
  *already* being reported as a bare `Optimal` despite residuals only within
  `√tol`. Flipping their status assertions to require the new
  `OptimalInaccurate` shows them landing there (and *not* on a clean
  `Optimal`).
- **Fix** (mirrors ECOS/Clarabel `*_INACC` and Ipopt's "Solved To Acceptable
  Level"):
  - New `QpStatus::OptimalInaccurate` variant (documented as a usable solve
    whose residual sits above `tol`; callers needing full accuracy must treat
    it as *not* `Optimal`).
  - New centralized `pub(crate) fn breakdown_status(near_opt) -> QpStatus` in
    `qp.rs` returning `OptimalInaccurate` when `near_opt`, else
    `NumericalFailure`. Both drivers call it in place of the inline
    `if near_opt { Optimal } else { NumericalFailure }` (10 sites total), so
    the symmetric and non-symmetric drivers cannot drift apart and the rule is
    unit-testable. The non-symmetric best-iterate restoration sets
    `OptimalInaccurate` directly.
  - The clean convergence test (`pres<tol && dres<tol && gap<tol → Optimal`)
    is untouched — genuinely converged solves still report `Optimal`.
  - CLI surfacing: extracted `convex_status_report(status) -> (msg, ok, srn)`
    (shared by the QP/LP and SOCP report paths, replacing two identical inline
    matches) maps `OptimalInaccurate → ("Solved to acceptable level (reduced
    accuracy).", ok=true, srn=100)` — the AMPL 100–199 reduced-accuracy band.
    `qp_status_to_ars` (CLI `main.rs` and `pounce_cblib`) maps it to
    `ApplicationReturnStatus::SolvedToAcceptableLevel` (JSON-report
    `solve_result_num` 100 via `status_to_solve_result_num`); `pounce_cblib`'s
    exit-code check treats it as success. The two `pounce-py` `status_str`
    maps emit `"optimal_inaccurate"`.
  - Conservatively *excluded* from accuracy-critical consumers: sensitivity
    (`sensitivity.rs:91`) and SOS exactness certification (`sos.rs:474/498`)
    keep their strict `== Optimal` gate, so a reduced-accuracy solution is not
    used as an exact certificate.
- **Why well-conditioned solves are safe**: `lp_afiro solver_selection=lp-ipm`
  (obj −464.75314285, 15 iters) and `convex_qp solver_selection=qp-ipm`
  (obj 2.0, 1 iter) still print "Optimal Solution Found." (`srn` 0) — the
  reduced-accuracy path does not fire for them.
- **Tests**:
  - `pounce-convex` `qp::residual_tests::breakdown_status_marks_near_opt_as_inaccurate_not_optimal`:
    pins `breakdown_status(true) == OptimalInaccurate` (and `!= Optimal`),
    `breakdown_status(false) == NumericalFailure`.
  - `pounce-cli` `convex_status_tests::optimal_inaccurate_is_distinct_from_optimal`:
    pins `convex_status_report(OptimalInaccurate) == (.., ok=true, srn=100)`
    with a message containing "acceptable", distinct `srn`/message from
    `Optimal`, and `qp_status_to_ars(OptimalInaccurate) ==
    SolvedToAcceptableLevel`.
  - Existing exp/power-cone conic tests (`cblib_cbf`, `cblib_vs_nlp`,
    `exp_cone_vs_nlp`, and the two `hsde_nonsym` unit tests) updated to accept
    either usable status (`Optimal | OptimalInaccurate`) while keeping their
    objective cross-checks; the `exp_cone_vs_nlp` near-boundary safety sweep's
    allowed-status set gains `OptimalInaccurate`.
- **Fail-first**: neutering `breakdown_status` to return `Optimal` for
  `near_opt` made `breakdown_status_marks_near_opt_as_inaccurate_not_optimal`
  fail (`left: Optimal, right: OptimalInaccurate`); restored after
  confirmation.
- **Scope note / deferred**: the review also notes `QpSolution` carries no
  final-residual field. The distinct status already resolves the named defect
  ("callers cannot detect it"); a `QpSolution.final_residual` (so callers can
  read *how* inaccurate) is a separate additive enhancement, deferred.
- **Result**: `pounce-convex` green (104 lib + all integration, 0 failed);
  full `pounce-cli` green (20 binaries, 0 failed); `pounce-py` builds clean;
  workspace build clean.

## M21 detail

- **Issue (review M21)**: `crates/pounce-convex/src/sos.rs` certifies SOS
  exactness for `sos_minimize` by *flat truncation* of the moment matrix —
  `recover_from_moments` compares `rank M_d` with the rank on the
  degree-≤(d−1) sub-basis (`mi.basis0[i].iter().sum() < mi.d`, `:549-550`).
  The review flags that for a **constrained** program this is weaker than the
  Curto–Fialkow / Henrion–Lasserre condition: the sufficient stopping window
  is `rank M_d = rank M_{d−dg}` with `dg = max_i ⌈deg gᵢ/2⌉`, and the extracted
  atoms are never checked against the constraints, so `is_exact = true`
  ("provably the global minimum") can over-claim. The review marked it
  *Uncertain — no concrete failing instance constructed*.
- **Why the `d−1` window is the weaker test** (direction matters): the moment
  sub-matrices are nested, `rank M_{d−dg} ≤ rank M_{d−1} ≤ rank M_d`. So
  `rank M_d = rank M_{d−dg}` (the correct, atoms-in-`K` condition) *implies*
  `rank M_d = rank M_{d−1}` (the code's condition), but **not** conversely.
  When `dg > 1` the code accepts a strictly larger set of moment matrices as
  "flat" than the theorem licenses — exactly the regime that can yield atoms
  outside `K`. `dg > 1` requires a constraint of degree ≥ 3.
- **Concrete failing instance** (constructed by running code, closing the
  review's "Uncertain"): `min (x+1)² s.t. x³ ≥ 0`. The feasible set is
  `x ≥ 0`; the constrained minimum is `1` at `x = 0`, but the *unconstrained*
  minimum is `0` at `x = −1`. The single degree-3 constraint gives `dg = 2`,
  and at order 2 its localizing matrix is the lone scalar `L(x³) ≥ 0` — far too
  weak to pin the relaxation. Pre-fix, `sos_minimize(&prob, Some(2), …)`
  returned `status = Optimal`, `is_exact = true`, `lower_bound = 0`, and a
  single "minimizer" at **x ≈ −0.719**, which is **infeasible** (`x³ ≈ −0.37`).
  So the certificate simultaneously reported the wrong optimum (0 vs 1) and an
  infeasible optimizer. Two more confirmations: `min (x+2)² s.t. x³ ≥ 0` (atom
  ≈ −0.63, `x³ ≈ −0.26`) and `min (x+1)² s.t. x⁵ ≥ 0` at order 3 (a spurious
  atom at x ≈ 318).
- **Fix** (`crates/pounce-convex/src/sos.rs`): validate the extracted atoms
  against the feasible set before keeping the certificate, rather than
  widening the rank window (which would only convert over-claims into
  silent *under*-claims at low order and still not guarantee feasibility).
  - New private `Polynomial::eval(x)` and `Polynomial::eval_magnitude(x)` (the
    triangle-inequality bound `Σ|cᵢ|·∏|xₖ|^{eₖ}`), plus a shared
    `Polynomial::monomial(e, x)` helper.
  - New `PolyProblem::is_feasible(x, tol)`: `gᵢ(x) ≥ −tol·(1+‖gᵢ‖(x))` for
    every inequality and `|hⱼ(x)| ≤ tol·(1+‖hⱼ‖(x))` for every equality — a
    **scale-invariant relative** margin (normalized by each constraint's
    magnitude at `x`) so a binding constraint reading `gᵢ ≈ 0` is accepted
    within the ~1e-4 inaccuracy of moment-extracted atoms while a clear
    violation (a sizeable negative fraction of `‖gᵢ‖`) is caught. `FEAS_TOL =
    1e-4`.
  - `recover_from_moments` now takes `prob`; after extraction it withdraws the
    certificate (`is_exact = false`, clears `minimizers`, `num_minimizers = 0`)
    if any atom fails `is_feasible`. `lower_bound` is **unchanged** — it remains
    a valid lower bound (in the example `0 ≤ 1`).
  - Both call sites in `sos_minimize` (the central solve and the
    facial-reduction re-solve) pass `prob`, so the validation guards the
    facial-reduction path too. (Because the failing instance now reports
    `is_exact = false` from the first recovery, the facial-reduction branch
    fires and re-solves at the same order; at order 2 the relaxation is
    genuinely not tight, so it correctly stays `is_exact = false`.)
- **No effect on unconstrained recovery**: with no inequalities/equalities,
  `is_feasible` is vacuously true, so all existing `sos_minimize` extraction
  tests — `extract_unique_minimizer_1d/2d`, `extracts_two_global_minimizers`,
  `facial_reduction_recovers_nonunique_minimizers`,
  `facial_reduction_three_minimizers_degree_six`,
  `facial_reduction_four_minimizers_2d_order_three`,
  `goldstein_price_wide_coefficient_range` — still certify and extract
  unchanged. The constrained `sos_constrained_lower_bound` tests are also
  unaffected (they return only a bound and never enter recovery).
- **Test** (`crates/pounce-convex/src/sos.rs`,
  `sos::tests::constrained_overclaim_rejected_when_atom_infeasible`): runs the
  `min (x+1)² s.t. x³ ≥ 0` instance at order 2 and asserts `status = Optimal`,
  `!is_exact`, `num_minimizers = 0`, `minimizers` empty, and `lower_bound ≤ 1`
  (still a valid lower bound).
- **Fail-first**: neutering the guard (`prob.is_feasible(x, FEAS_TOL) || true`)
  makes the test fail with `is_exact = true` and minimizer `[−0.7189…]` —
  precisely the over-claim — then restored.
- **Result**: `pounce-convex` green (105 lib + all integration, 0 failed);
  `pounce-py` builds clean (its `sos_minimize` wrapper surfaces the corrected
  `is_exact`/`minimizers` unchanged in shape).
- **Scope note**: this hardens the *exactness certificate* — it never weakens
  the `lower_bound`, which is valid regardless of flatness. A user wanting the
  certified constrained optimum for such an instance should raise the
  relaxation `order` until the higher-order moment/localizing matrices make the
  relaxation tight (and the now-validated flat truncation fires on feasible
  atoms).

## M22 detail

- **Issue (review M22)**: in `build_sos_sdp`
  (`crates/pounce-convex/src/sos.rs`) the coefficient-matching equalities are
  accumulated in `by_mono: HashMap<Vec<usize>, Vec<(usize, f64)>>` (monomial →
  the SDP columns/coefficients contributing to it), and the SDP's equality
  rows are then emitted by iterating that map: `for (alpha, terms) in &by_mono
  { let row = b.len(); … row_of.insert(alpha.clone(), row); }`. Rust's
  `std::collections::HashMap` uses a per-instance random seed
  (`RandomState`), so the iteration order is **not** reproducible: the
  assignment of monomials to row indices changes run-to-run.
- **Why it matters**: the solver receives the same system under a different
  row permutation each run. The math is invariant, but the floating-point
  path is not, so the certified bound and extracted minimizers drift between
  runs at the solver's accuracy floor (and more, near ill-conditioning). The
  SOS tests carry loose `~1e-5` tolerances partly to absorb this jitter — a
  classic symptom rather than a fix.
- **Reproduced by running code**: Rust seeds each `HashMap` instance
  differently, so building the *same* problem twice **within one process**
  already exposes the nondeterminism. A probe built the degree-4 two-variable
  SDP twice and compared `qp.b` (the RHS, whose order reflects the row order)
  and `MomentInfo.row_of` (the monomial→row map). Across three separate
  processes, both comparisons were **false** every time:

  ```text
  n_rows=15 b_eq=false
  row_of_eq=false
  ```

- **Fix** (one type change + the import): switch `by_mono` from `HashMap` to
  `BTreeMap<Vec<usize>, Vec<(usize, f64)>>`. Exponent vectors (`Vec<usize>`)
  are `Ord`, so the row-emitting iteration now runs in deterministic
  sorted-by-monomial order. The `entry(..).or_default().push(..)` accumulation
  is identical on a `BTreeMap`. `coeff_map`/`rhs`/`row_of` stay `HashMap`:
  they are used only for point lookups (never iterated to decide ordering),
  and the row *values* now stored in `row_of` are themselves deterministic
  because they come from the ordered `by_mono` walk. No public API, cone
  layout, or numerical formula changed — only the row ordering is pinned.
- **Verified post-fix**: the identical twice-build probe now reports
  `b_eq=true` and `row_of_eq=true`; the full SOS bound/extraction suite still
  passes (bounds and minimizers unchanged within tolerance).
- **Test** (`crates/pounce-convex/src/sos.rs`,
  `sos::tests::sdp_row_order_is_deterministic`): builds the SDP for a fixed
  degree-4 two-variable polynomial twice and asserts `qp1.b == qp2.b` and
  `mi1.row_of == mi2.row_of` (with a guard that there are several rows, so a
  permutation is detectable).
- **Fail-first**: reverting `by_mono` to `HashMap` makes the test fail with
  `assertion left == right failed: RHS row order differs between builds`;
  restored to `BTreeMap` after confirmation.
- **Result**: `pounce-convex` green (106 lib + all integration, 0 failed);
  `pounce-py` builds clean. The loose SOS-test tolerances are left as-is —
  they remain appropriate for IPM accuracy — but the *cause* of run-to-run
  variation is removed, so results are now reproducible.

## M23 detail

- **Issue (review M23)**: `PsdCone::kkt_block`
  (`crates/pounce-convex/src/cones/psd.rs`) assembled the `(z,z)` KKT block —
  the symmetric Kronecker operator `H = W ⊗ₛ W` on svec space, an `m×m` SPD
  matrix with `m = dim = n(n+1)/2 = O(n²)` — by looping over every svec unit
  vector `e_b` and calling `apply_scaling(w, e_b, …)`, i.e. forming
  `W·smat(e_b)·W` with two O(n³) matmuls and reading off column `b`. That is
  `O(n²)` columns × `O(n³)` per column = **O(n⁵)** per cone per IPM iteration.
  Separately, `lyapunov_solve` (the Jordan/`Arw(z)⁻¹` solve behind
  `rhs_comp_term`/`recover_ds`) computed the two eigenbasis congruences
  `R̃ = QᵀRQ` and `D = QD̃Qᵀ` with explicit quadruple loops — **O(n⁴)** each,
  where two matmuls give O(n³).
- **Why it matters**: for SOS / Lasserre moment SDPs the relaxation is *one*
  large PSD cone whose order `n` is the moment-matrix size, so `kkt_block` is
  re-assembled every interior-point iteration and dominates the per-iteration
  cost; the O(n⁵)→O(n⁴) gap is a full factor of `n`.
- **Reproduced by running code**: a timing probe (release) over
  `n = 8,16,24,32` (`m` up to 528) measured the old per-unit-vector
  construction at `n=8→0.000022s`, `n=16→0.000520s`, `n=24→0.003581s`,
  `n=32→0.014077s` — i.e. ≈ factor-7 per `n`-doubling at the large end
  (between the O(n⁴)=16× and O(n⁵)=32× regimes, consistent with the matmul
  constant), confirming the steep super-quartic scaling.
- **Fix**:
  - `kkt_block` — derive the symmetric-Kronecker entries in **closed form**
    and write the lower triangle directly. Column `b` corresponds to the svec
    pair `(p,q)` with `p ≥ q`, for which `smat(e_b)` is `E_pp` (if `p=q`) or
    `(E_pq+E_qp)/√2` (if `p>q`); with `W` symmetric, `D = W·smat(e_b)·W` has
    `D_ij = W_ip W_jp` (`p=q`) or `(W_ip W_jq + W_iq W_jp)/√2` (`p>q`), and the
    svec row scaling gives `H[a][b] = (i=j ? 1 : √2)·D_ij` for the output pair
    `(i,j)`, `i ≥ j`. Each of the `O(n⁴)` lower-triangle entries is then one
    O(1) expression — **O(n⁴)** total, the NT scaling computed once.
  - `lyapunov_solve` — the eigensolver returns `Q` column-major
    (`q[c·n+i] = Q[i][c]`), so reading `q` as row-major *is* `Qᵀ`; transpose it
    once to get `Q` row-major, then both congruences are plain `matmul` calls
    (`R̃ = (q·R)·Q_rm`, `D = (Q_rm·D̃)·q`) — **O(n³)**.
  - No public API, `ConeBlock::DenseLower` layout, svec `√2` convention, or
    numerical contract changed.
- **Verified post-fix**: the same timing probe gives `n=8→0.000018s`,
  `n=16→0.000081s`, `n=24→0.000274s`, `n=32→0.000699s` — a **20.1×** speedup at
  `n=32`, and the speedup itself grows ~linearly in `n` (1.2×, 6.4×, 13.1×,
  20.1× over `n=8,16,24,32`), exactly the O(n) factor the O(n⁵)→O(n⁴)
  reduction predicts.
- **Test** (`crates/pounce-convex/src/cones/psd.rs`,
  `cones::psd::tests::kkt_block_matches_apply_scaling_reference`): for
  `n = 1,2,3,5,8` it builds the reference block by applying `W ⊗ₛ W` (via
  `apply_scaling`) to each svec unit vector — the *previous* O(n⁵)
  construction — and asserts the new closed-form `kkt_block` reproduces it
  entry-for-entry within `1e-9`. The defining-property test
  `kkt_block_maps_z_to_s` (`H·svec(z) = svec(s)`),
  `recover_ds_matches_block_and_rhs`, and `lyapunov_inverts_jordan`
  (`z ∘ (Arw(z)⁻¹ r) = r`, which guards the `lyapunov_solve` matmul rewrite)
  all continue to pass.
- **Fail-first**: perturbing the closed-form entry (`row_scale * d + 1e-3`)
  makes `kkt_block_matches_apply_scaling_reference` fail at `n=1 [0][0]`
  (`1.3539… vs 1.3529…`); restored after confirmation.
- **Result**: `pounce-convex` green (107 lib + all integration, 0 failed);
  `pounce-py` builds clean. The temporary timing probe was used only to
  measure the speedup and was removed before commit (no flaky wall-clock
  assertion is committed).

## M24 detail

- **Issue (review M24)**: Phase 2 redundant-row removal
  (`crates/pounce-presolve/src/redundant.rs`, `lib.rs:7-11`) drops a linear
  row whose activity interval over the current variable box is `⊆ [lo, hi]`.
  When the row was *itself* the cause of a Phase-1 bound tightening — e.g.
  `x ≥ 2` tightens `x_l = 2` — its activity becomes flush against its own
  bound (`[2, x_u] ⊆ [2, +∞)`), so Phase 2 sees it as redundant and drops it.
  At a solution where that bound is active, the interior-point method reports
  the dual on the *variable-bound* multiplier (`z_l`/`z_u`) — but that bound
  did not exist in the original `.nl` (it was a constraint *row*); the
  reinstated row keeps `λ = 0`. The dual is thus attributed differently than a
  no-presolve solve would attribute it.
- **Why it matters / scope**: this is a *dual-attribution* difference, not an
  optimality bug. The review notes "primal/objective unaffected; inherent to
  the design and worth documenting or fixing via dual transfer." Confirmed:
  the primal point, the objective, and KKT stationarity
  `∇f − Jᵀλ − z_l + z_u = 0` are all intact; only *where* the dual sits (bound
  multiplier vs. row `λ`) changes. Callers that read per-row constraint duals
  (sensitivity, `.sol` dual block) see `0` on such a dropped row and the mass
  on the variable's bound multiplier instead.
- **Reproduced by running code**: a single-variable TNLP `min x s.t. x ≥ 2`
  (original box `x ∈ [0, 10]`, the row marked `Linear`) driven through
  `PresolveTnlp` with `bound_tightening + redundant_constraint_removal`:
  - `get_nlp_info` returns `m = 0` (row dropped), `n_dropped_rows = 1`,
    `cached_bounds().x_l = [2.0]` (tightened by the row).
  - `finalize_solution` at the reduced optimum `x = 2` with `z_l = [1.0]`
    (stationarity `∇f − z_l = 1 − 1 = 0`) yields
    `finalized_full_solution()` lambda `= [0.0]` — the dropped row's
    multiplier is **not** recovered from the bound multiplier.
- **Why documented rather than fixed**: a faithful fix transfers the bound
  multiplier back onto the row's `λ` (and zeros the synthetic bound
  multiplier), but that requires Phase-1 **provenance** — a record of which
  row implied which bound — which is not tracked anywhere:
  `bound_tighten::tighten_bounds` returns only a `TightenReport` (counts, an
  `infeasible` flag), no row→bound link. The existing
  `recover_dropped_multipliers` solves a `k×k` system only for Phase-0
  aux-eliminated rows held on the reduction stack, not for Phase-2 redundant
  rows. The general case is also ambiguous: a multi-variable row binding at a
  box vertex distributes its dual across several active bound multipliers with
  no unique inverse. Adding provenance + a transfer step is a substantial,
  risky change that the review explicitly ranks behind documenting; it is
  deferred as future work.
- **Action taken** (doc-only + characterization test, no behavior change):
  - Expanded the Phase 2 entry in the crate module-doc
    (`crates/pounce-presolve/src/lib.rs`) with a **"Dual-attribution caveat
    (issue M24)"** paragraph stating the limitation precisely — primal,
    objective, and KKT stationarity unaffected; only the attribution differs;
    a faithful fix needs currently-untracked Phase-1 provenance and is left as
    future work.
  - Added `tests::dropped_row_dual_lands_on_bound_not_row`, which drives the
    reproduction above and asserts: the row is dropped (`n_dropped_rows == 1`,
    `m == 0`), `x_l` tightened to `2`, the full-space `λ[0] == 0`, the primal
    `x == 2`, and the KKT stationarity residual `≈ 0` with the dual on `z_l`.
    The `λ[0] == 0` assertion is the explicit hook a future dual-transfer fix
    would change to `λ[0] ≈ z_l`.
- **Result**: `pounce-presolve` green (208 lib + all integration, 0 failed);
  `pounce-py` builds clean. No public API or numerical behavior changed — the
  item is verified and documented, with the dual-transfer fix deferred.

## M25 detail

- **Issue (review M25)**: Phase 1 bound tightening can prove the feasible
  region empty and, in doing so, leave the variable box *crossed*
  (`x_l[j] > x_u[j]`). `tighten_pass`
  (`crates/pounce-presolve/src/bound_tighten.rs:128-131`) writes the tightened
  `x_l[j] = nl` / `x_u[j] = nh` and only *then* checks `x_l[j] > x_u[j] + tol`,
  returning immediately with `infeasible = true` and the crossed bounds in
  place. In `lib.rs` the sole restoration was the aux-rollback guard
  (`if tighten_report.infeasible && !reduction_stack.is_empty()`, `:609`):
  it restores the pre-Phase-0 box, undoes the row drops, clears the reduction
  stack, and re-runs Phase 1 on the un-filtered rows — but it is gated on a
  **non-empty** reduction stack.
- **Why it matters**: `auxiliary` is off by default, so the reduction stack is
  empty for the common case. A genuine Phase-1 infeasibility then skips the
  guard entirely, and the crossed box flows into `CachedBounds { x_l, x_u, … }`
  (`:821`) and on to the IPM via `get_bounds_info`. An interior-point method
  handed `x_l > x_u` cannot initialize a strictly-interior point and reports an
  invalid-problem / bad-bounds error — a confusing failure in place of the
  clean "infeasible" verdict the IPM would return if it were allowed to run on
  a valid box. (The FBBT path at `:679-695` already does the right thing for
  its own infeasibility: it snapshots the pre-FBBT box and restores it so the
  IPM can certify infeasibility itself. Phase 1 lacked the analogous step.)
- **Reproduced by running code**: a single-variable TNLP, box `x ∈ [0, 10]`,
  two contradictory linear rows `x ≥ 5` (row 0) and `x ≤ 3` (row 1), both
  marked `Linear`, `auxiliary = false`. After `get_nlp_info`:
  `tighten_report().infeasible == true`, and `cached_bounds()` returns
  `x_l = [5.0]`, `x_u = [3.0]` — crossed, exactly what would reach the IPM.
- **Fix** (mirrors the aux rollback and the FBBT handling): immediately after
  the Phase-1 block, restore the original inner box whenever the tighten still
  reports infeasibility:

  ```rust
  if tighten_report.infeasible {
      x_l.copy_from_slice(&inner_x_l);
      x_u.copy_from_slice(&inner_x_u);
      tracing::warn!(target: "pounce::presolve", "Phase 1 … crossed bounds … discarded …");
  }
  ```

  `inner_x_l`/`inner_x_u` are the pristine original bounds snapshotted at
  `:472-473` (only ever read afterward), so they are always a valid box. This
  is also the correct target for the *non-empty-stack* path: the aux rollback
  restores to those same inner bounds before re-tightening, so if the re-run
  stays infeasible, restoring to them again is exactly right (and the aux
  elimination has already been rolled back). The `infeasible` flag is left set
  and surfaced via `tighten_report()` for diagnostics. Phase 4 warm-z (which
  compares `x_l` to `inner_x_l`) and Phase 2 redundancy then run on the valid,
  un-tightened box — both conservative and correct.
- **Verified post-fix**: the same reproduction now returns `x_l = [0.0]`,
  `x_u = [10.0]` (the original box) while `tighten_report().infeasible`
  remains `true` — a valid box reaches the IPM, the infeasibility is still
  reported.
- **Test** (`tests::phase1_infeasible_restores_valid_box_for_ipm`): asserts
  `tighten_report().infeasible`, `x_l ≤ x_u`, and the box restored to
  `[0, 10]`.
- **Fail-first**: neutering the guard (`if tighten_report.infeasible && false`)
  makes the test fail with `bounds handed to IPM must be valid, got
  x_l=5 > x_u=3`; restored after confirmation.
- **Regression safety**: the existing aux-rollback test
  (`phase0_via_tnlp_no_infeasible_with_default_bound_tightening`), where the
  rollback re-tighten *clears* the infeasibility, is unaffected — the new guard
  fires only when `infeasible` survives the Phase-1 block.
- **Result**: `pounce-presolve` green (209 lib + all integration, 0 failed);
  `pounce-py` builds clean.

## M26 detail

- **Issue (review M26)**: `finalize_solution` densifies the full inner
  Jacobian. The review flags `vec![0.0; m_inner * n_inner]` allocated
  "whenever a reduction frame exists — 80 GB at 100k×100k", while
  `recover_dropped_multipliers` "only needs the k fixed columns".
- **Verification by reading code**: `recover_dropped_multipliers`
  (`crates/pounce-presolve/src/reduction_frame.rs`) builds a k×k block system
  and a kept-row correction, indexing the Jacobian at exactly two places —
  `jac_full_row_major[dr * n_vars + i]` (dropped rows `dr`, `i ∈ fixed_vars`)
  and `jac_full_row_major[r * n_vars + i]` (kept rows `r`, `i ∈ fixed_vars`).
  Both index columns are drawn from `fixed_vars`; no other column is ever
  touched. So with `k = |fixed_vars|`, only the `k` (across frames: the union
  of all frames' fixed vars) distinct columns are read — the remaining
  `n_inner − k` columns of the dense block are pure waste, O(m·n) memory for an
  O(m·k) need.
- **Verification by running code** (`tests::recover_only_reads_fixed_var_columns`):
  a 3-var/3-row frame with `fixed_vars = [0, 2]` (var 1 free). Recover once on
  a clean dense Jacobian; then poison **column 1** (the free var) in every row
  with `f64::NAN` and recover again. The two multiplier vectors are equal
  **bit-for-bit** (`to_bits()`), and every recovered entry is finite — a `NaN`
  in a read column would have propagated through the LU and been caught. This
  is the direct empirical proof that the non-fixed columns need not be
  materialized at all.
- **Fail-first**: temporarily poisoning a *fixed* column (`column 0`) instead
  makes the recovery produce `NaN` and the test fails
  (`recovered multiplier went NaN`); restored after confirmation. This shows
  the test genuinely detects column reads (it is not vacuously passing).
- **Fix** (no behavior change to the recovered values, only to the memory
  footprint):
  - Extract a private `ReductionFrame::recover_core(grad_f, lambda_full,
    get: impl Fn(usize, usize) -> Number)` holding the exact prior math; `get`
    abstracts the Jacobian layout and is invoked only at `col ∈ fixed_vars`.
  - `recover_dropped_multipliers` keeps its signature and asserts (so all 8
    existing call sites + the doctest are untouched) and delegates with the
    dense accessor `|row, col| jac[row * n_vars + col]`.
  - Add `recover_dropped_multipliers_cols(grad_f, jac_cols_row_major, n_cols,
    orig_to_compact, lambda_full)` reading an `(n_full_rows × n_cols)` compact
    buffer via `|row, col| jac_cols[row * n_cols + orig_to_compact[col]]`.
  - In `lib.rs` `finalize_solution`: replace the `m_inner × n_inner`
    densification with a single pass that (a) walks all frames to mark the
    union of `fixed_vars` columns and build `orig_to_compact` (sentinel
    `usize::MAX` for absent columns), (b) scatters the inner COO into a compact
    `m_inner × n_cols` buffer, skipping any column whose `orig_to_compact` is
    the sentinel, then (c) calls `recover_dropped_multipliers_cols` per frame.
    The per-frame z_l/z_u zeroing at `fixed_vars` is unchanged.
- **Tests**: `recover_cols_matches_dense` builds a dense Jacobian and its
  column-compacted form over `{0, 2}` and asserts the two recoveries agree
  bit-for-bit; `recover_cols_empty_frame` exercises the `k = 0`, `n_cols = 0`
  path (empty compact buffer is valid, returns an empty multiplier vector). The
  existing finalize tests (Phase-0 aux recovery, bound-multiplier zeroing) and
  the end-to-end pipeline fuzz all still pass, confirming the recovered
  multipliers are identical through the real `PresolveTnlp` path.
- **Result**: `pounce-presolve` green (212 lib + all integration + 9 doctests,
  0 failed); `pounce-py` builds clean.

## M27 detail

- **Bug**: Phase-0 auxiliary-equality elimination (`auxiliary.rs`) assembles
  each square block by scanning the *entire* COO Jacobian once per block row.
  The pattern `for kk in 0..nnz { let i = decode(jac_irow[kk]); if i != r_inner
  { continue; } … }` appears in four places — the C2 acceptance gate,
  `solve_linear_block`, `residual_norm_linear`, and `NonlinearBlock::jacobian`.
  Each is `O(nnz)` per block row, so total assembly is
  `O(total_block_rows × nnz)`. For the common many-small-blocks case
  (`n` singleton rows, the diagonal-aux pattern, `nnz = O(n)`) that is `O(n²)`.
- **Verified by running code** (timing probe, all-singleton diagonal pattern,
  `solve_block_max_size` large enough to keep each row its own block):

  | n    | pre-fix | post-fix | speedup |
  |------|---------|----------|---------|
  | 500  | 1.11 ms | 0.54 ms  | 2.1×    |
  | 1000 | 2.32 ms | 0.92 ms  | 2.5×    |
  | 2000 | 7.03 ms | 1.97 ms  | 3.6×    |

  Pre-fix each `n`-doubling more than doubles time (super-linear, trending to
  4× — the `O(n²)` signature); post-fix is near-linear and the speedup *grows*
  with `n`, exactly what removing an `O(n)` factor predicts.
- **Fix**: build a CSR row index **once**, before block assembly:
  - `decode_idx(raw, one_based) -> usize` centralizes the COO index decode
    (`raw - 1` when one-based, else `raw`), used identically in the count and
    fill passes so the convention can never drift between them.
  - `build_row_nnz(jac_irow, n_rows, one_based) -> (Vec<usize>, Vec<usize>)`
    builds CSR in `O(nnz)`: a counting pass fills `ptr`, a prefix sum, then a
    fill pass scatters each nnz position `kk` into `entries` grouped by its
    (decoded, 0-based) row. Out-of-range rows are skipped.
  - `RowNnz<'a> { ptr, entries }` (a `Copy` view) exposes
    `of_row(r) -> &[usize]` returning the nnz positions `kk` in row `r`.
  The four hot loops now iterate `for &kk in row_nnz.of_row(r)` — `O(deg(r))`
  per row, `O(nnz)` total. `solve_linear_block`, `residual_norm_linear`, and
  `solve_nonlinear_block` take `&RowNnz<'_>`; `NonlinearBlock` stores the
  `Copy` view (its `nnz`/`jac_irow` fields removed). No public API or numerical
  contract changed — block math and residual checks are byte-for-byte the same,
  only the iteration order over a row's nnz is now CSR-grouped.
- **Tests**:
  - `build_row_nnz_groups_by_row_zero_based` — pins `ptr`/`entries` for a small
    hand-checked COO.
  - `build_row_nnz_honours_one_based_decode` — asserts the one-based CSR
    (`ptr`,`entries`) is identical to the zero-based build of the same shape,
    proving the decode is applied consistently.
  - `phase0_diagonal_many_singletons_correct` (n = 400) — end-to-end: every var
    fixed to `r+1`, every row dropped.
  - `phase0_one_based_two_blocks_eliminated` — end-to-end one-based pipeline,
    asserts `fixed_vars == [0,1]`, values `5.0`/`7.0`, `dropped_rows == [0,1]`;
    bit-identical to the pre-M27 result.
- **Fail-first** (clean): temporarily making the count/fill passes decode
  inconsistently makes `build_row_nnz_honours_one_based_decode` and
  `phase0_one_based_two_blocks_eliminated` FAIL, then pass on restore.
  *Restore with `cp`+`touch`, never `mv`* — `mv` preserves the backup's older
  mtime, which can roll the source below the compiled binary's, so cargo skips
  recompilation and silently reuses a stale half-neutered binary, fabricating
  phantom failures (cost real debugging time on this issue; documented here as
  a standing lesson).
- **Result**: `pounce-presolve` green (216 lib + integration + 9 doctests,
  0 failed); `pounce-py` builds clean.

## M28 detail

- **Bug**: `run_fbbt` (`fbbt/orchestrator.rs`) aggregated per-variable
  tightening into a fresh `let mut tighten: Vec<Interval> = vec![Interval::ENTIRE;
  n_vars]` allocated **per constraint**, then applied it with a
  `for j in 0..n_vars` loop — also per constraint. A constraint's tape normally
  references only a handful of variables, so both the allocation and the apply
  scan are `O(n_vars)` of pure overhead per constraint. Per sweep that is
  `O(m · n_vars)`; the review flags it as quadratic on problems where each tape
  is sparse.
- **Verified by running code** (timing probe, `m = n` constraints, each a
  single-`Var(i)` tape, wide bound so nothing tightens — isolates the
  per-constraint `O(n_vars)` overhead):

  | n    | pre-fix | post-fix | speedup |
  |------|---------|----------|---------|
  | 1000 | 0.62 ms | 0.073 ms | 8.5×    |
  | 2000 | 2.25 ms | 0.119 ms | 19×     |
  | 4000 | 8.86 ms | 0.191 ms | 46×     |

  Pre-fix each `n`-doubling ≈ quadruples time (the `O(n²)` signature); post-fix
  is near-linear and the speedup *grows* with `n`, exactly what removing the
  `O(n_vars)`-per-constraint factor predicts.
- **Fix** — hoist the scratch out of the loops and touch only the variables a
  constraint mentions. Allocated **once**, before the sweep loop:
  - `tighten: Vec<Interval>` (length n_vars) — the running per-variable
    intersection within the current constraint.
  - `last_seen: Vec<usize>` — stamps which constraint last wrote each
    `tighten[j]`.
  - `touched: Vec<usize>` — the distinct variables the current constraint
    mentions.
  - `stamp: usize` — a monotonic per-constraint-visit counter.

  For each `Var(j)` slot: if `last_seen[j] == stamp` (already seen this
  constraint) intersect into `tighten[j]`, else overwrite `tighten[j]`, set
  `last_seen[j] = stamp`, and push `j` onto `touched`. This "first slot
  overwrites, later slots intersect" discipline needs **no per-constraint
  reset** of the n_vars scratch — the stamp invalidates stale entries lazily.
  The apply step iterates `for &j in &touched` instead of `0..n_vars`. A
  variable absent from the tape keeps `ENTIRE` and could never tighten or be
  empty, so the touched-only apply is exactly equivalent to the old full scan.
  Per-constraint work is now `O(tape length)`; per sweep `O(nnz)`. No public
  API or numerical contract changed.
- **Test**: `duplicate_var_slots_intersect` pins the only subtle behavior the
  rewrite touches — a variable appearing in two structurally distinct `Var(0)`
  slots of one tape must end with the INTERSECTION of both slots' reverse
  intervals. Tape `x² + x = 6` over `x ∈ [0, 10]` with the **squared** slot
  first (tight: `x ≤ √6 ≈ 2.449`) and the **linear** slot second (loose:
  `x ≤ 6`). With `max_iter = 1` (a single sweep — essential, since iterating to
  the fixed point makes all slot intervals coincide and washes the difference
  out) the correct intersection yields `x_hi ≈ 2.449`. The existing 64 FBBT
  tests (coupled-constraint iteration, soundness fuzz, `row_kept` masking,
  infeasibility detection, `max_iter`/`max_constraints` caps) pass unchanged.
- **Fail-first**: changing the later-slot branch from `intersect` to a plain
  overwrite makes `duplicate_var_slots_intersect` FAIL with `x_hi = 6.0` (the
  loose linear slot), confirmed then restored.
- **Result**: `pounce-presolve` green (217 lib + integration + 9 doctests,
  0 failed); `pounce-py` builds clean.

## M29 detail

- **Bug**: Phase-3 LICQ (`licq.rs`) computed its structural rank with a
  *second*, weaker bipartite matcher than the one the crate already ships.
  `bipartite_matching_rank` allocated a `vec![false; n]` `seen` array **per
  row** (O(m·n) total, just zeroing scratch) and augmented via a **recursive**
  `try_augment` whose call depth equals the augmenting-path length. Meanwhile
  `matching.rs::hopcroft_karp` is an iterative, BFS-layered Hopcroft–Karp
  (O(E·√V), cross-checked against brute force via König's theorem) operating on
  the CSR `EqualityIncidence` — exactly the matching LICQ needs.
- **Verified by running code** (two temporary probes on the old matcher):
  - *Recursion depth.* A `thread_local` max-depth counter wired into
    `try_augment`, run over a staircase chain (`row 0: {0}`, `row i: {i−1, i}`,
    final row reaching only the last column — augmenting the final row cascades
    down the whole chain), measured **max depth = m − 1 exactly**:

    | m (rows) | max recursion depth |
    |----------|---------------------|
    | 1 000    | 999                 |
    | 4 000    | 3 999               |
    | 16 000   | 15 999              |

    Linear in chain length — a chain of tens of thousands of rows (the
    discretized-dynamics shape the review names) recurses tens of thousands of
    frames deep and overflows a normal 2–8 MB stack.
  - *Per-row allocation.* On the m = n diagonal (`row i: {i}`), timing showed
    the `m × vec![false; n]` allocation scaling super-linearly:
    `0.023 / 0.069 / 0.245 ms` at n = 1000/2000/4000 — the O(m·n) signature.
- **Fix**: delete `try_augment` and rewrite `bipartite_matching_rank` to pack
  the `EqRow` list into a CSR `EqualityIncidence` (out-of-range columns dropped
  — preserving the old guard; per-row columns sorted + deduped exactly as
  `EqualityIncidence::from_probe`) and return `hopcroft_karp(&inc).size`.
  Hopcroft–Karp's BFS layering means a row with no augmenting path is rejected
  in BFS with **no DFS recursion at all**, and when DFS does run its depth is
  bounded by the BFS layer distance (O(√V) ≈ 224 for V = 50 000, vs the naive
  O(V)). `licq_check`'s public `LicqVerdict` semantics are unchanged — it still
  short-circuits `m > n → OverDetermined` and `empty row → EmptyRow` before
  matching, then maps `size == m → Full` else `StructuralRank(size)`.
- **Tests** (existing 7 LICQ tests — over-determined, empty-row,
  duplicate/distinct singletons, augmenting-path — all pass unchanged):
  - `long_chain_does_not_overflow_stack`: the m = 50 000 staircase chain that
    drove the old matcher to depth ≈ 50 000. To keep `m ≤ n` (so the
    over-determined short-circuit does not pre-empt the matcher) it declares one
    phantom extra column, leaving `n = m` with only m−1 columns touched; max
    matching = m−1 ⇒ `StructuralRank(m − 1)`. It completes on the default 2 MB
    test-thread stack — the regression guard against the recursive overflow.
  - `long_chain_full_rank`: m = 20 000 rows over m columns (perfect matching) ⇒
    `Full`, guarding against the new matcher capping a long augmenting path
    short.

## M30 detail

- **Bug** (`python/pounce/_curve_fit.py`): the constrained-covariance branch in
  both `_covariance` (the buggy block was lines 1542-1547) and the streaming
  twin `_stream_covariance` (1108-1112) read:
  ```python
  if m_con > 0 or (active_mask is not None and active_mask.any()):
      free = ~active_mask if active_mask is not None else np.ones(n, bool)
      cov = np.zeros((n, n))
      cov[np.ix_(free, free)] = s2 * np.linalg.pinv(M[np.ix_(free, free)])
      return cov, "reduced_hessian(projected)"
  ```
  `active_mask` (from `_active_bounds`) covers **variable bounds only**. So when
  `m_con > 0` but no bound is active, `free` is all-True and the function returns
  the **unconstrained** `s2·pinv(M)` — yet labels it `reduced_hessian(projected)`.
  An active equality between parameters is never projected out: variances are
  overstated and the induced anti-correlations are missing, directly
  contradicting the module docstring's headline ("correct under active bounds /
  constraints … the projection onto the active-constraint nullspace").
- **Verified by running code** (`/tmp/m30_verify.py`, loading the module in
  isolation): a weighted line fit `f(x) = a·x + b` with `M = JwᵀJw`, under an
  active equality `a + b = 3` (`A = [[1, 1]]`). Pre-fix `_covariance` returned a
  `pcov` bit-identical to the unconstrained `s2·pinv(M)`, with
  `A·pcov·Aᵀ = 0.318` — i.e. it claims the *exactly-known* combination `a + b`
  has appreciable variance — and no off-diagonal anti-correlation. The correct
  projected covariance has `A·pcov·Aᵀ = 0` and a `−0.065` off-diagonal. After
  threading the constraint Jacobian into the call, the same probe returns the
  projected covariance (`code == correct`, `A·pcov·Aᵀ = 0`).
- **Fix**: `_FitProblem` already carries `m_con`, `g_combined`, `jac_combined`,
  `cl`, `cu` (built by `_minimize._wrap_constraints`) but the covariance call
  site only passed `m_con`. Thread `jac_combined`/`g_combined`/`cl`/`cu` into
  both `_covariance` and `_stream_covariance` and replace the bounds-only
  projection with two helpers:
  - `_active_constraint_jac(popt, jac_combined, g_combined, cl, cu, tol=1e-6)`
    returns the rows of `jac_combined(popt)` that **bind**: an equality
    (`cl[i] == cu[i]`) always binds; an inequality binds when `g_combined(popt)`
    sits within `tol·max(1,|bound|)` of a finite `cl`/`cu`. This linearizes the
    (possibly nonlinear) constraints at `popt` — the first-order active set.
  - `_projected_covariance(M, s2, active_mask, A_gen, n)` stacks the active
    general rows `A_gen` with unit rows `eⱼ` for the active bounds, takes an
    orthonormal nullspace basis `Z` of the stack via SVD (rank-robust
    threshold), and returns `s2·Z·pinv(ZᵀMZ)·Zᵀ`. For an empty active set it
    returns `s2·pinv(M)`; for a fully-pinned active set it returns zeros.
  The caller reports `reduced_hessian(projected)` only when something actually
  binds (`_n_active > 0`), else `jacobian`.
- **Equivalence to the old bounds path**: for a bounds-only active set, `A` is a
  stack of unit rows `eⱼ`, so `Z` is precisely the free coordinate subspace and
  `s2·Z·pinv(ZᵀMZ)·Zᵀ` equals the old `cov[ix_(free,free)] = s2·pinv(M[free,free])`
  embedded with zero bound rows/cols. The existing `test_positivity_bound_active`
  and `test_streaming_active_bound_projects_covariance` confirm this is preserved.
- **Gauss-Newton / nonlinear-constraint note**: the `_covariance` docstring now
  states the result is the first-order (delta-method) asymptotic covariance — `M`
  is the Gauss-Newton Hessian and the constraints are linearized at `popt`, so it
  omits the constraint-curvature term `Σλᵢ∇²gᵢ` (identically zero for linear
  constraints, higher-order otherwise). This is the standard constrained-LS
  covariance and matches what the projection computes.
- **Tests** (`python/tests/test_curve_fit.py`):
  - `test_active_equality_constraint_projects_covariance` — fits a line under an
    active `a + b = 1` equality (analytic constraint jac) and asserts
    `cov_source == "reduced_hessian(projected)"`, `g·pcov·g < 1e-9` along the
    constraint gradient `g = [1,1]`, `pcov[0,1] < 0` (anti-correlation), and a
    match to the closed-form `s2·Z·pinv(ZᵀMZ)·Zᵀ`.
  - `test_streaming_active_equality_projects_covariance` — the streaming twin,
    asserting the streamed covariance matches the in-memory one and projects the
    same way.
  - **Pre-fix both FAIL** — `g·pcov·g ≈ 1.6e-3` (the unconstrained variance);
    confirmed before applying the fix, both PASS after. Run with
    `POUNCE_SKIP_EXT_STALE_CHECK=1 PYTHONPATH=$PWD/python` so `import pounce`
    binds the worktree's pure-Python module (the change is Python-only; the
    extension is unchanged).
  - Full `test_curve_fit.py` (44 passed) and `test_sensitivity.py` /
    `test_minimize.py` / `test_minima.py` (37 passed) green.
- **Result**: `pounce-presolve` green (219 lib + integration + 9 doctests,
  0 failed), no new clippy warnings; `pounce-py` builds clean.

## M31 detail

- **Issue** (`dev-notes/code-review-2026-06.md:468-473`,
  `python/pounce/qp.py:434-437`): the indefinite-`P` guard added under issue
  #112 (`_check_psd`, which raises `ValueError("... positive semidefinite ...")`
  when `λ_min(P) < -1e-8·max(scale,1)`) was wired into `solve_qp` only. The
  other five host QP entry points — `solve_qp_batch` (`qp.py:547`),
  `solve_qp_multi_rhs` (588), `QpFactorization` (626), `QpSensitivity` (679),
  and `solve_socp` (474) — never called it, and neither did the jax/torch
  differentiable layers' host forwards. A nonconvex (indefinite) `P` is outside
  the convex IPM's contract, so it converges to a meaningless KKT point and
  reports `status="optimal"` — a *silently wrong* answer. For the
  differentiable layers it is worse: the bogus primal feeds the OptNet implicit
  backward, corrupting gradients.
- **Verification (running code)**: `/tmp/m31_verify.py` builds
  `P = diag(1, -1)` (eigenvalues +1, −1) with `c = 0` and box bounds
  `[-1, 1]²` (the bounds keep the convex IPM from diverging so it returns a
  *concrete* status, exposing the silent-wrong behavior rather than a generic
  failure) and calls all six host entry points. Pre-fix: only `solve_qp`
  raised; `solve_qp_batch`/`solve_qp_multi_rhs`/`solve_socp` returned
  `status="optimal"` and `QpFactorization`/`QpSensitivity` constructed usable
  handles.
- **Fix** (`python/pounce/qp.py`): a single shared helper
  ```python
  def _maybe_check_psd(P, c, check_psd) -> None:
      if check_psd is False:
          return
      n = np.asarray(c, dtype=np.float64).ravel().shape[0]
      if check_psd or n <= _PSD_CHECK_AUTO_MAX_N:
          _check_psd(*_lower_triangle_coo(P, n), n)
  ```
  centralizes the policy: `check_psd=False` is the opt-out escape hatch (caller
  asserts PSD, or wants the nonconvex behavior, or is avoiding the O(n³) eig);
  `check_psd=True` forces the check at any size; `check_psd=None` (default) runs
  it automatically only when `n <= _PSD_CHECK_AUTO_MAX_N` (1500) to bound the
  eigenvalue cost on large QPs. `solve_qp` was refactored onto the helper, and
  each of the other five entry points gained a `check_psd` parameter and a call
  to it before building the `_pounce` problem (for `solve_qp_batch`, once per
  problem dict). The jax/torch layers (`pounce/jax/_qp.py`,
  `pounce/torch/_qp.py`) import `_PSD_CHECK_AUTO_MAX_N`/`_check_psd` and gained a
  local `_guard_psd(P, n)` that runs the same screen (size-gated) inside the
  host forwards `_forward_solve`, `_forward_solve_batch`, `_forward_solve_socp`
  — the points where a concrete numpy `P` is turned into a `_pounce.QpProblem`
  (jax via `pure_callback`, torch in the eager `autograd.Function.forward`).
- **Tests**:
  - `python/tests/test_qp_host.py` — five `test_*_rejects_indefinite_p` (one per
    previously-unguarded host entry point), each
    `pytest.raises(ValueError, match="positive semidefinite")`;
    `test_check_psd_false_bypasses_guard_everywhere` (the opt-out skips the
    guard on all five); `test_psd_p_still_solves_on_all_entry_points` (a
    genuine PSD `P` passes unscathed everywhere).
  - `python/tests/test_qp_jax.py` / `test_qp_torch.py` —
    `test_indefinite_p_rejected_in_forward` and
    `test_indefinite_p_rejected_in_batch_forward` (batched `c` is `(B, n)` and
    `h` is `(B, m)`); jax wraps the host `ValueError` in a runtime error but the
    `"semidefinite"` substring survives, so both layers match on it.
  - **Fail-first confirmed**: neutering the guard makes the five host rejection
    tests FAIL (the unguarded points return `status="optimal"`); restored, all
    pass.
  - Full QP suite green under `POUNCE_SKIP_EXT_STALE_CHECK=1 PYTHONPATH=$PWD/python`
    (Python-only change; extension unchanged): `test_qp.py`, `test_qp_host.py`,
    `test_qp_jax.py`, `test_qp_torch.py`, `test_qp_sensitivity.py`,
    `test_socp.py` — **82 passed**.

## M32 detail

- **Issue** (`dev-notes/code-review-2026-06.md:474-480`,
  `crates/pounce-py/src/tnlp_bridge.rs:364-374`): the optional `intermediate`
  TNLP callback maps the Python return value to the solver's continue/stop
  bool. Two defects:
  1. `Ok(Some(res.extract::<bool>().unwrap_or(true)))` — `extract::<bool>()`
     is strict (a Python `int` is *not* a `bool` even though `bool ⊂ int`),
     so a cyipopt-valid falsy `0` (the documented "stop" signal) fails
     extraction and `unwrap_or(true)` coerces it to **continue**. The user's
     stop request is silently dropped. Only an actual `False` stopped.
  2. `Err(_) => false` — a callback that *raises* is swallowed into a
     `false` (stop ⇒ `User_Requested_Stop`) with **no log**, unlike the eval
     callbacks (`objective`/`gradient`/`constraints`/`jacobian`/`hessian`),
     each of which `tracing::error!(target: "pounce::py", …)` on error. A
     crashing callback masquerades as a clean user stop.
- **Verification (running code)**: a `maturin build --release` of the worktree
  crate, the resulting `_pounce.abi3.so` extracted into `python/pounce/` and
  imported via `PYTHONPATH=$PWD/python` (the venv's editable install still
  points at the *main* repo, so it is untouched). `/tmp/m32_verify.py` solves
  `min (x-3)²` with an `intermediate` returning `0` at `iter_count>=1`:
  **pre-fix** → `Solve_Succeeded`, 8 iters, `x=3` (stop ignored); **post-fix**
  → `User_Requested_Stop`, 2 iters, `x≈7.6` (stopped early). The raising-
  callback case logs `ERROR pounce::py: pounce-py: intermediate(): RuntimeError:
  boom from intermediate` post-fix.
- **Fix** (`crates/pounce-py/src/tnlp_bridge.rs`):
  ```rust
  // was: Ok(Some(res.extract::<bool>().unwrap_or(true)))
  Ok(Some(res.is_truthy()?))           // cyipopt truthiness
  ...
  // was: Err(_) => false,
  Err(e) => {
      tracing::error!(target: "pounce::py", "pounce-py: intermediate(): {e}");
      false
  }
  ```
  `is_truthy()` makes `False`/`0`/`0.0`/`[]` stop and truthy continue; the
  pre-existing `res.is_none()` branch still maps a `None`/no-return to
  continue. The `Err` arm now logs (consistent with the eval callbacks) while
  preserving the stop-on-exception behavior.
- **Tests** (`python/tests/test_problem.py`):
  - `test_intermediate_falsy_return_stops[0, False, 0.0, []]` — each must abort
    with `User_Requested_Stop` and not reach `x*=3`.
  - `test_intermediate_truthy_return_continues[1, True, 0.5, [0]]` — truthy
    keeps iterating to `Solve_Succeeded` (`x≈3`).
  - `test_intermediate_no_return_continues` — a `None` return is not a stop.
  - `test_intermediate_exception_aborts_with_user_stop` — a raising callback
    aborts with `User_Requested_Stop` (the log line is verified manually; it
    routes through the Rust `tracing` subscriber, not visible to pytest).
  - **Fail-first confirmed** by swapping the pre-fix `.so` back in and running
    `-k intermediate`: `[0]`, `[0.0]`, `[[]]` FAIL (`Solve_Succeeded`) while
    `[False]` passes — precisely the `extract::<bool>` gap. Post-fix the
    restored extension passes all 14 `test_problem.py` tests.
- **Result**: `test_problem.py` (14) green; broader solve-exercising suite —
  `test_critical.py`/`test_warm_start.py`/`test_solver_session.py`/
  `test_sensitivity.py`/`test_minimize.py` — green (53 passed total). `cargo
  clippy -p pounce-py` shows only pre-existing warnings (none from the two
  changed lines). The rebuilt `.so` is a worktree-local build artifact
  (gitignored); only the Rust source, the test, and this doc are committed.

## M33 detail

- **Issue** (`dev-notes/code-review-2026-06.md:481-486`,
  `pyomo-pounce/pyomo_pounce/pounce_solver.py:35-36`): Pyomo's `ASL` base
  calls `_default_executable()` to locate the solver binary. The plugin
  implemented it as `return shutil.which("pounce")`. But the runtime
  dependency `pounce-solver` ships a per-platform wheel that drops the `pounce`
  binary at a deterministic path inside the package — `pounce/bin/pounce`,
  surfaced by `pounce._cli._bundled_binary()` (`python/pounce/_cli.py:22-24`)
  — and a `<venv>/bin/pounce` *console-script shim*. `shutil.which` finds only
  the shim, and only when `<venv>/bin` is on PATH. Runs that don't activate the
  venv (cron jobs, IDE test runners, Jupyter kernels launched from a different
  environment) have `<venv>/bin` off PATH and report the solver unavailable;
  worse, a stale system `pounce` earlier on PATH is silently preferred over the
  wheel's matching binary.
- **Verification (running code)**: from the worktree
  (`PYTHONPATH=$PWD/pyomo-pounce:$PWD/python`), monkeypatch
  `pounce._cli._bundled_binary` to a real temp executable and set
  `PATH=/usr/bin:/bin` (no `pounce`). Pre-fix `_default_executable()` →
  `None` even though the bundled binary exists; the solver is reported
  unavailable.
- **Fix** (`pyomo-pounce/pyomo_pounce/pounce_solver.py`):
  ```python
  def _default_executable(self):
      try:
          from pounce._cli import _bundled_binary
          bundled = _bundled_binary()
          if bundled.is_file():
              return str(bundled)
      except Exception:
          pass
      return shutil.which("pounce")
  ```
  The bundled binary (deterministic, PATH-independent, guaranteed to match the
  installed wheel) is preferred; the `try/except` keeps a missing
  `pounce-solver` (system install / local `cargo install` dev build) working
  via the PATH fallback.
- **Tests** (`pyomo-pounce/tests/test_pounce.py`, all hermetic via
  `monkeypatch`):
  - `test_default_executable_prefers_bundled` — bundled present, PATH stripped
    → returns the bundled path (the regression-sensitive case).
  - `test_default_executable_falls_back_to_path` — no bundled binary → returns
    the `pounce` found on PATH.
  - `test_default_executable_none_when_nowhere` — neither → `None` (the honest
    "unavailable" signal `available()` relies on).
  - **Fail-first confirmed** by reverting the method to `return
    shutil.which("pounce")`: `prefers_bundled` FAILS (`None != bundled`) while
    the other two pass; post-fix all pass.
- **Result**: `pyomo-pounce/tests/test_pounce.py` — **7 passed** (the 3
  solve-smoke tests ran against the on-PATH binary, no skips). Pure-Python
  plugin change; no extension rebuild involved.
## M34 detail

**Issue** (`python/pounce/_route.py`, `python/pounce/_minimize.py:425,447-468`):
`pounce.minimize` takes opaque callables, so to route an LP/convex-QP to the
dedicated convex solver it must *probe* the callables and finite-difference a
quadratic model. On the default `auto` path it runs **two** routers in
sequence — `classify_and_extract` (LP/QP) and, if that finds nothing,
`classify_and_extract_socp` (SOCP/QCQP). Both build their probe set from
`np.random.default_rng(seed=0)` and FD-fit the *same* objective, so they
evaluate the objective Hessian at an identical set of points. For a problem
that ultimately lands on the NLP solver (the common case for a genuine
nonlinear objective), every one of those evaluations is overhead, and it was
nowhere documented.

**Verification (running code)**: counting `fun` calls through `minimize` on a
quartic `Σ xᵢ⁴` (n=5, no analytic derivatives → FD, routes to NLP):

| path | `fun` calls |
|------|-------------|
| `auto` (both routers + NLP solve) | 491 |
| `solver_selection="nlp"` (no routing) | 231 |
| one router (`classify_and_extract`) in isolation | 260 |

Routing overhead = 491 − 231 = **260** post-fix, i.e. exactly one router's
probe count. Pre-fix the same measurement gave **520** (auto = 751), exactly
2× — the SOCP router re-probed from scratch every point the QP router had
already evaluated. (`/tmp/m34_verify.py` separately confirmed the two routers'
probe sets are identical: 608/608 points at n=8 with 608 shared.)

**Fix** (`python/pounce/_route.py` + `python/pounce/_minimize.py`): added a
small `_point_cache(f)` helper that wraps a callable so repeated evaluations at
the *same point* return a cached result, keyed on the point's exact float64
bytes (`np.asarray(x, float64).tobytes()`); `None` passes through unchanged and
cached values are returned as-is (never mutated). In `_minimize`, the
`route_kw` dict the routers receive now wraps `fun`/`jac`/`hess`/`g_combined`/
`jac_combined` in `_point_cache`, so the second router's probes are cache hits.
Crucially **only the router copies are cached** — the NLP fallback
(`_build_problem_obj`) still binds the *original* callables, so the actual
solve is byte-for-byte unaffected.

Also added a docstring paragraph to `minimize` documenting that auto-routing
costs `O(n²)` extra `fun` evaluations and that `solver_selection="nlp"` skips
routing entirely (for known-NLP problems or expensive `fun`).

**Test** (`python/tests/test_minimize_autoroute.py`):
`test_auto_route_probes_objective_once_not_twice` counts `fun` calls on the
auto path, the nlp-forced path, and one router in isolation, then asserts the
auto-path routing overhead (`auto − nlp`) equals a single router's probe count
— not double. **Fail-first confirmed** by reverting the `_point_cache`
wrapping in `route_kw`: overhead becomes 520 ≠ 260 and the test fails; with the
fix restored all **74** tests in `test_minimize_autoroute.py`,
`test_minimize_socp_autoroute.py`, `test_minimize.py`, and `test_curve_fit.py`
pass. Pure-Python change; no extension rebuild involved.

## M35 detail

**Issue** (`crates/pounce-py/src/solver.rs:80`, `crates/pounce-py/src/qp.rs`):
the session-style entry points ran the Rust IPM while still holding the Python
GIL. `PyProblem::solve` and the one-shot `solve_qp`/`solve_socp` already
release it with `py.allow_threads`, but `PySolver::solve` (the NLP `Solver`
session), `QpFactorization::solve`, and `QpSensitivity::new` did not. `Solver`
is the workhorse under `curve_fit` and the jax/torch hosts, so any code running
several of these on Python threads got no overlap — every solve serialized
behind the GIL.

**Verification (running code)**, `n=220` strictly-convex QP, 14-core box:

| metric | pre-fix | post-fix |
|--------|---------|----------|
| watcher-thread max stall during solves | 23.6 ms (≈ full 31 ms solve) | 4.5 ms |
| 8 `QpSensitivity` solves, threaded ÷ serial | 0.97 | 0.39 |

The QP path is pure Rust (no Python callbacks), so pre-fix a solve held the GIL
*continuously* and a background Python thread was completely starved for the
solve's duration; post-fix the watcher runs throughout and the eight solves
overlap across cores (~2.5× speedup).

**Fix**: wrap each solve body in `py.allow_threads`.
- `QpFactorization::solve` / `QpSensitivity::new` are pure Rust, but their
  factorization/sensitivity objects hold non-`Send` `dyn
  SparseSymLinearSolverInterface` / `dyn TSymScalingMethod` trait objects, so a
  plain `allow_threads` (which needs `Send`) doesn't typecheck. A transparent
  `SendGuard<T>` (`unsafe impl Send`, with method accessors that defeat the
  2021-edition disjoint-capture rule) carries the borrow / result across the
  boundary — the closure runs on the *calling* thread after
  `PyEval_SaveThread`, so the value never actually crosses OS threads. This is
  the same shim `PyProblem::solve` already uses for its `Rc<RefCell<…>>`.
  `QpSensitivity::new` gained a `py: Python<'_>` parameter and was restructured
  to return `(status, Option<payload>)` from the closure (no panic-on-`None`
  unwrap, so no new `clippy::expect_used`).
- `PySolver::solve` (NLP) uses the *identical* `SendGuard` pattern as
  `PyProblem::solve`: every TNLP callback in `tnlp_bridge.rs` re-acquires the
  GIL via `Python::with_gil` before touching Python, so re-entering the
  interpreter from the GIL-released solve is safe and serialized as usual.

**Test** (`python/tests/test_qp_sensitivity.py::test_qp_solve_releases_the_gil`):
runs 8 independent convex-QP solves serially and across 8 threads (best-of-2
each) and asserts threaded < 0.75 × serial; skips on < 4 cores. **Fail-first
confirmed** by swapping the pre-M35 `.so`: ratio 1.01 ⇒ test FAILS; with the
fix, ratio 0.39 ⇒ PASS.

**Result**: 41 QP tests (`test_qp.py`, `test_qp_sensitivity.py`,
`test_qp_host.py`) + 112 NLP-session / sensitivity / curve_fit / problem tests
pass; clippy reports no new warnings on the changed lines. The extension was
rebuilt via `maturin build --release` and the `.so` extracted into the
worktree. (One pre-existing, unrelated `test_socp.py::test_exp_cone_log_sum_exp_mixed`
failure reproduces identically on the pre-M35 `.so`, so it is not caused by
this change and is out of scope for M35.)

## M36 detail

**Issue** (`crates/pounce-studio-core/src/report.rs:142-154` vs writer
`crates/pounce-solve-report/src/lib.rs:185-204`): the solve-report *writer* and
the studio-core *reader* each define their own `InputDescriptor`, an
internally-tagged enum (`#[serde(tag = "kind", rename_all = "kebab-case")]`).
The writer has four variants — `NlFile`, **`CbfFile`** (`"kind": "cbf-file"`,
for a Conic Benchmark Format `.cbf` instance solved through the convex conic
driver), `Builtin`, `TnlpDirect` — but the reader mirror had only three,
missing `CbfFile`. Because serde's internally-tagged enums reject an unknown
tag for the *whole* value, any report produced from a `.cbf` input failed to
deserialize entirely: every studio-core consumer (the MCP `load_solve_report`,
`diagnose`, `inspect`, …) returned a hard error rather than the report.

**Verification (running code)**: starting from the known-good `rosenbrock.json`
fixture, rewrite `fair_metadata.input` to
`{"kind":"cbf-file","path":"/tmp/cblib/instance.cbf","size_bytes":4096}` and
load it through `SolveReport::from_json_str`. Pre-fix this returned
`Err(Json("unknown variant 'cbf-file', expected one of 'nl-file', 'builtin',
'tnlp-direct'"))` — the exact whole-report rejection.

**Fix** (`crates/pounce-studio-core/src/report.rs`): add the missing variant to
the reader, mirroring the writer:

```rust
/// A Conic Benchmark Format (`.cbf`) instance … Mirrors the writer's
/// `pounce_solve_report::InputDescriptor::CbfFile` (`"kind": "cbf-file"`).
CbfFile {
    path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
},
```

(`path: String` keeps the reader's existing convention — it uses `String` where
the writer uses `PathBuf`; the JSON is identical.) No production code in
studio-core pattern-matches `InputDescriptor` exhaustively (only test-fixture
*constructions* of `Builtin`/`TnlpDirect` exist), so the new variant needs no
other call-site changes.

**Test** (`crates/pounce-studio-core/tests/fixtures.rs`):
`loads_cbf_file_input_descriptor` performs the same fixture rewrite and asserts
the report loads and `fair_metadata.input` decodes to `InputDescriptor::CbfFile`
with the expected `path`/`size_bytes`. **Fail-first confirmed**: a load-only
form (`from_json_str(&src).is_ok()`, which compiles against the unfixed enum)
failed pre-fix with the serde unknown-variant error; post-fix the full test and
all 13 studio-core tests pass. clippy reports no new lib warnings (the test's
`unwrap`/`expect` follow the file's established convention). Pure-Rust crate; no
Python extension involved.

## M37 detail

**Issue** (`crates/pounce-cinterface/src/solver.rs`): the session-style
sensitivity C ABI exposes two entry points that take a pin set —
`IpoptSolverParametricStep(solver, n_pins, pin_indices, deltas, dx_out)` and
`IpoptSolverReducedHessian(solver, n_pins, pin_indices, obj_scal, hr_out)`. Both
deliberately treat `n_pins == 0` as legal: the null-guards only reject NULL
`pin_indices`/`deltas` *when `n_pins > 0`* (`if n_pins > 0 && (pin_indices.is_null()
|| deltas.is_null()) { return FALSE; }`), because an empty pin set has nothing to
point at, so a caller may pass NULL. But once past the solver/session guards, the
code built the slices unconditionally:

```rust
let pins_raw = std::slice::from_raw_parts(pin_indices, n_pins as usize); // :339, :383
let deltas_slice = std::slice::from_raw_parts(deltas, n_pins as usize);  // :347
```

`std::slice::from_raw_parts` documents that the pointer **must be non-null and
aligned even for a zero-length slice** (enum-layout niche optimizations rely on
references being non-null). So `from_raw_parts(NULL, 0)` is library UB. It is
silent on older toolchains, but recent rustc emits an `assert_unsafe_precondition!`
language-UB check under `-C debug-assertions` (the default for dev/test profiles),
which turns the call into a hard process abort. The rest of the crate already
gates this correctly — `IpoptSolverSolve` reads its primal buffer with
`let initial_x = if n_us > 0 { std::slice::from_raw_parts(x, n_us).to_vec() } else
{ Vec::new() };` — only these two sensitivity functions diverged.

**Verification (running code)**: the bad `from_raw_parts` sits *after* the
`info.session.as_ref()` guard, so a converged session is needed to reach it.
Create the 1-D quad (`f(x)=(x-2)²`), `IpoptCreateSolver`, `IpoptSolverSolve`
(→ `Solve_Succeeded`), then call
`IpoptSolverParametricStep(solver, 0, NULL, NULL, dx_out)`. Pre-fix the test
binary aborts:

```
unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be
aligned and non-null, and the total size of the slice not to exceed `isize::MAX`
... thread caused non-unwinding panic. aborting.
... (signal: 6, SIGABRT)
```

**Fix** (`crates/pounce-cinterface/src/solver.rs`): a small private helper that
mirrors the existing `n_us > 0` gate and dedupes the three sites —

```rust
/// Like `std::slice::from_raw_parts`, but yields an empty slice when
/// `len == 0` instead of dereferencing `ptr`. … `from_raw_parts(NULL, 0)`
/// is undefined behaviour and trips the non-null debug-assertion.
unsafe fn slice_or_empty<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 { &[] } else { std::slice::from_raw_parts(ptr, len) }
}
```

applied at all three call sites (`pins_raw`/`deltas_slice` in `ParametricStep`,
`pins_raw` in `ReducedHessian`). An empty pin set is a well-defined no-op: a zero
perturbation gives Δx ≈ 0 and a 0×0 reduced Hessian, so the underlying
`pounce_sensitivity::Solver` returns `Ok` and both C calls report `TRUE`.

**Test** (`crates/pounce-cinterface/src/solver.rs::zero_pins_with_null_pointers_is_not_ub`):
solves the quad to a session, then calls both entry points with `n_pins = 0` and
NULL `pin_indices`/`deltas` (valid `dx_out`/`hr_out` buffers), asserting each
returns `TRUE` — reaching the assertions at all proves no `from_raw_parts(NULL, 0)`
abort fired. **Fail-first confirmed** by reverting the two `ParametricStep` slices
to bare `from_raw_parts` (leaving the helper referenced by `ReducedHessian` so it
stays live): the test aborts with signal 6 / the non-null precondition message;
restoring the helper makes it pass. Full `pounce-cinterface` lib suite green (43
passed), `cargo clippy -p pounce-cinterface` clean of new warnings (the 3 reported
`needless_range_loop` warnings pre-date this change and live in `lib.rs`).
Pure-Rust crate; no Python extension rebuild needed.

## M38 detail

**Issue** (`.github/workflows/release-crates.yml`, `release-pounce.yml`,
`release-pyomo-pounce.yml`): POUNCE cuts a release by pushing a tag, and each
workflow keys off a distinct prefix —

| tag                     | workflow                  | manifest published                       |
|-------------------------|---------------------------|------------------------------------------|
| `v<X.Y.Z>`              | release-crates            | root `Cargo.toml` `[workspace.package]`  |
| `python-v<X.Y.Z>`       | release-pounce            | `python/pyproject.toml`                  |
| `pyomo-pounce-v<X.Y.Z>` | release-pyomo-pounce      | `pyomo-pounce/pyproject.toml`            |

None of them compared the *tag's* version against the *manifest's*. The
on-every-PR guard `scripts/check-release-consistency.sh` checks that the three
manifests agree **with each other**, but nothing checks that the tag you push
agrees with them. So a `v0.5.0` tag cut while the manifests still read 0.4.0:

* **crates.io** — `scripts/publish-crates.sh` is idempotent: it skips any crate
  already live at the workspace version. With the manifest at 0.4.0 (already
  published), every crate is skipped and the workflow ends green having
  published nothing — the 0.5.0 release silently never ships to crates.io.
* **PyPI** — the wheels/sdist are built from the 0.4.0 manifest, so the
  `pounce-solver`/`pyomo-pounce` 0.5.0 "release" publishes a 0.4.0 artifact (or
  collides with the already-published 0.4.0), shipping the wrong version.

**Verification (running code)**: there was no guard to demonstrate a "pre-fix"
failure against — the gap *is* the absence. So the new guard is exercised
directly. Against the live repo (all three manifests at 0.4.0):

```
$ python3 scripts/check_tag_version.py refs/tags/v0.5.0
check_tag_version: TAG/MANIFEST MISMATCH for crates.io workspace.
  tag 'v0.5.0' declares version 0.5.0
  Cargo.toml is at 0.4.0
  ...                                          # exit 2
$ python3 scripts/check_tag_version.py v0.4.0          # exit 0 (matches)
$ python3 scripts/check_tag_version.py pyomo-pounce-v1.0.0
  ... MISMATCH for pyomo-pounce (PyPI) ...     # exit 2, routed to the right manifest
```

The `pyomo-pounce-v…`/`python-v…` cases confirm the longest-prefix-first
dispatch — a `pyomo-pounce-v1.0.0` tag is not misread as a bare-`v` crates tag.

**Fix**: a new `scripts/check_tag_version.py` (pure functions `strip_ref`,
`parse_tag`, `manifest_version`, `check` for unit-testability, mirroring the
existing `scripts/check_dep_publishability.py`). It strips `refs/tags/`, matches
the tag against the longest release prefix, validates the remainder is an
`X.Y.Z` (optionally `-prerelease`/`+build`) version, reads the first
top-of-line `version = "..."` from the routed manifest (the same extraction
`check-release-consistency.sh` uses — anchored at column 0, so indented
dependency-table `version =` keys are ignored), and exits non-zero on
mismatch/unknown-tag/unreadable-manifest with a message naming both versions.

**Workflow wiring**:

* `release-crates.yml` — a `Verify tag matches manifest version` step before
  `Publish crates`, `if: github.event_name == 'push'`.
* `release-pounce.yml` / `release-pyomo-pounce.yml` — a standalone
  `verify-version` job that the `build-wheels`/`build-sdist` (resp. `build`)
  jobs declare in `needs:`, so a mismatch fails **before** the multi-platform
  wheel matrix runs rather than at publish time. The publish jobs already
  `needs:` the build jobs, so they are transitively gated.

All three gate on `github.event_name == 'push'`; a manual `workflow_dispatch`
(TestPyPI dry run / crates dry run) carries no release tag, so the step is
skipped and the job is a no-op pass — the dispatch path is unchanged.

**Test** (`scripts/tests/test_check_tag_version.py`): 18 unittest cases over a
synthetic repo tree built in a temp dir (so they don't depend on the live
manifest versions, which change every release): `strip_ref`, prefix routing,
longest-prefix precedence (`python-v` over `v`), prerelease-suffix acceptance,
rejection of non-version/unknown-prefix tags, the indented-dependency-version
trap, and `check()` end-to-end — matching tag passes (exit 0), the M38 mismatch
fails (exit 2), a python/pyomo tag validates against its own manifest even when
Cargo matches, unknown tag → 3, missing manifest → 4. The file follows the
sibling test's standalone convention (`python3 scripts/tests/test_*.py` /
`-m unittest`); neither is wired into `ci.yml`, so this one isn't either. Both
script-test files run clean together: `python3 -m unittest discover -s
scripts/tests` → 25 passed. All three workflow YAMLs parse and the
`verify-version → build → publish` dependency graph validates. CI/release-only
change; no Rust or Python package code touched.

## M39 detail

**Issue** (`.github/workflows/ci.yml:63,66,69`): the `test` job's Clippy, Build,
and Test steps all carry `--exclude pounce-hsl`:

```yaml
run: cargo clippy --workspace --exclude pounce-hsl --all-targets -- ...
run: cargo build  --workspace --exclude pounce-hsl --verbose
run: cargo test   --workspace --exclude pounce-hsl --verbose
```

so no CI job compiles `pounce-hsl` at all — yet it is on the crates.io publish
list (`scripts/publish-crates.sh`, **5th of 19**, after pounce-common,
pounce-linalg, pounce-linsol, pounce-feral). The exclusion is legitimate:
`crates/pounce-hsl/Cargo.toml` declares `links = "coinhsl"` and a `build.rs`,
and the crate FFI-wraps `libcoinhsl` (MA57), which is licensed and not present
on the CI runners — so a *link* of any final artifact (a `cargo build`/`cargo
test` binary) that pulls pounce-hsl in would fail. The cost of the blanket
exclusion: a plain type/syntax error in pounce-hsl source is caught by **no**
CI job. Its first compile is the verify build `cargo publish` runs by default,
mid-release; by the time the topological publish reaches it, the four crates
ahead are already live on crates.io and cannot be unpublished — a partial,
irreversible release triggered by a trivial compile error.

**Why a compile-check is safe without HSL**: `build.rs` is defensive — when
`COINHSL_DIR` is unset it prints a `cargo:warning` and returns *without*
emitting any `rustc-link-lib`/`rustc-link-search` directives, compiling
pounce-hsl as an ordinary rlib. Only a downstream crate selecting the `ma57`
path (which sets COINHSL_DIR) actually links the library. So `cargo check`
(which type-checks but never links) compiles pounce-hsl cleanly on a
library-less machine.

**Verification (running code)**:

```
# Inject a deliberate type error into crates/pounce-hsl/src/lib.rs, then:
$ cargo build --workspace --exclude pounce-hsl     # the CURRENT CI build
  ... Finished ... exit 0          # <-- error completely invisible to CI
$ cargo check -p pounce-hsl --all-targets          # the PROPOSED step
  error[E0308]: mismatched types ... exit 101       # <-- caught
```

with `COINHSL_DIR` unset in both (the CI condition). Reverting the error (via
`cp`+`touch`, not `mv`, to keep cargo's mtimes honest) returns the check to
exit 0 with only the benign "COINHSL_DIR not set" warning.

**Fix** (`.github/workflows/ci.yml`): add one step to the `test` job, after
Test, on the same runner that already has the toolchain + feral sibling:

```yaml
- name: Compile-check pounce-hsl (publishable, link-excluded above)
  run: cargo check -p pounce-hsl --all-targets --verbose
```

`--all-targets` extends the check to pounce-hsl's test modules (`ma57.rs`
`#[cfg(test)]`, etc.), which the excluded `cargo test` never compiles either —
so both the library and its tests are now type-checked on every PR without
needing an HSL install. Confirmed `cargo check -p pounce-hsl --all-targets`
passes on the clean tree (exit 0) and the workflow YAML parses with the new
step present. CI-only change; no crate source modified.

## L1 detail

- **Bug**: pounce splits the per-iteration convergence test (which lives at the
  *top* of `IpoptAlgorithm::iterate`, in `check_convergence_with_state`) from a
  *second*, redundant `max_iter` guard in the outer driver loop
  (`ipopt_alg.rs`, the `IterateOutcome::Continue` arm). The driver did:

  ```rust
  let mut iter_count = self.data.borrow().iter_count;
  iter_count += 1;
  if iter_count >= self.max_iter {
      break SolverReturn::MaxiterExceeded;   // <-- breaks BEFORE iterate()
  }
  self.data.borrow_mut().iter_count = iter_count;
  ```

  Because the break fired before the next `iterate()` call, the iterate
  produced by the final permitted step was never convergence-tested. Two
  consequences:
  1. A solve converging on *exactly* the `max_iter`-th iterate reported
     `Maximum_Iterations_Exceeded` instead of `Solve_Succeeded` — upstream
     Ipopt runs `CheckConvergence` (component tolerances first, then its own
     `iter_count >= max_iterations` test) at the top of the loop, so it catches
     that boundary convergence.
  2. `data.iter_count` could never be set to `max_iter` (the break intercepted
     the assignment), so the in-`iterate()` `MaxIterExceeded` branch at
     `conv_check/opt_error.rs:233` was effectively dead code.

- **Verification (run before any change)**: a fresh HS071 solve with the
  default budget converges to `Solve_Succeeded` at `iter=8`. Re-solving with
  `max_iter=8` reported `status=MaximumIterationsExceeded iter=7` — the loop
  broke at the boundary, one step short of testing the converged 8th iterate.

- **Fix**: remove the premature break. Bump `data.iter_count` and loop; the
  next `iterate()` runs its convergence check, which tests the component
  tolerances *before* the `iter_count >= max_iter` gate, so boundary
  convergence is reported as success and a genuine overrun still terminates via
  `opt_error.rs:233` (now live). Termination is guaranteed because
  `check_convergence_with_state` never returns `Continue` once
  `iter_count >= max_iter`. Behaviorally this matches upstream's
  top-of-loop / convergence-first ordering and takes the same `max_iter` steps;
  it only *adds* the previously-missing convergence test on the final iterate.

- **Test**: `optimize_hs71.rs::hs071_converges_exactly_at_max_iter_boundary`
  derives HS071's natural convergence iteration `k` from a generous-budget
  solve, then re-solves with `max_iter = k` and asserts
  `Solve_Succeeded`/`SolvedToAcceptableLevel` + objective ≈ 17.014017. Deriving
  `k` at runtime keeps the test robust to options/linear-backend drift.
  Fail-first: pre-fix it panics with
  `converging on the max_iter-th iterate must report success, got
  MaximumIterationsExceeded (max_iter = 8)`. Post-fix all 16 `optimize_hs71`
  tests pass and the full `pounce-algorithm` suite is green (lib 245 tests +
  every integration test, 0 failures), confirming the core loop change is
  regression-free.

## L2 detail

**Disposition: NOT A BUG. The review's premise is refuted by the upstream
source.**

- **Claim under review**: the dual half of the tiny-step test
  (`ipopt_alg.rs:1041-1042`) compares the dual step `Δy` to `tiny_step_y_tol`
  *absolutely*, whereas upstream Ipopt allegedly scales it by `1/(1+‖y‖∞)`
  (relative). The primal half (`detect_tiny_step`, 1152-1172) is relative
  (`|δxᵢ|/(1+|xᵢ|)`), so the review reads the dual side as an inconsistency
  that makes `STOP_AT_TINY_STEP` under-fire when the multipliers are large.

- **What pounce does** (`ipopt_alg.rs:1041-1042`):
  ```rust
  let dy_amax = delta.y_c.amax().max(delta.y_d.amax());
  self.tiny_step_last_iteration = dy_amax < self.tiny_step_y_tol;
  ```

- **What upstream actually does**: fetched `coin-or/Ipopt`, branch
  `stable/3.14`, `src/Algorithm/IpBacktrackingLineSearch.cpp`. The assignment
  of `tiny_step_last_iteration_` is:
  ```cpp
  Number delta_y_norm = Max(IpData().delta()->y_c()->Amax(),
                            IpData().delta()->y_d()->Amax());
  if( delta_y_norm < tiny_step_y_tol_ )
  {
     tiny_step_last_iteration_ = true;
  }
  else
  {
     tiny_step_last_iteration_ = false;
  }
  ```
  This is a **direct absolute comparison** — `Amax` of the dual step versus
  `tiny_step_y_tol_`, with **no division by `(1+‖y‖∞)`** and no other scaling.
  pounce is an exact, line-for-line port (same `Amax`, same `<` test, same
  tol). The `'T'`/`'t'` flag logic at 1034-1039 mirrors the same upstream
  block.

- **The primal/dual asymmetry is upstream's, and intentional**: upstream's
  `DetectTinyStep()` scales the *primal* step per component
  (`δx ./ (1+|x|)`, `δs ./ (1+|s|)`) and gates on the constraint violation
  `≤ 1e-4`; the *dual* step uses the bare `Amax` test above. pounce reproduces
  both halves faithfully. This is corroborated independently by the registered
  option help (`upstream_options.rs:411`): `tiny_step_y_tol` is described as
  *"the step in the y variables is smaller than this threshold"* (absolute),
  while `tiny_step_tol` is *"in relative terms for each component"* (the primal
  side). The two descriptions deliberately differ.

- **Conclusion**: there is nothing to fix. The `1/(1+‖y‖∞)` scaling the review
  attributes to upstream does not exist in the upstream code; applying it to
  `ipopt_alg.rs:1041-1042` would *introduce* a deviation from Ipopt, not remove
  one. No source change and no regression test were made. (Had the premise been
  true, a fail-first test would have constructed a large-‖y‖ iterate at a tiny
  primal step and asserted the `'T'`/`tiny_step_flag` path fires; that test is
  intentionally omitted because the behavior it would assert is *wrong*.)

## L3 detail

- **Bug**: in the adaptive μ-update's free-mode oracle dispatch
  (`mu/adaptive.rs`), the `MuOracleKind::Probing` arm built its oracle with a
  hard-coded cap:
  ```rust
  let mut oracle = ProbingMuOracle {
      sigma_max: 100.0,            // <-- ignores the user-set option
      mu_min: self.mu_min,
      ...
  };
  ```
  while the sibling `MuOracleKind::QualityFunction` arm forwarded
  `oracle.sigma_max = self.sigma_max;`. The probing oracle caps its centering
  parameter as `sigma = min((mu_aff/mu_curr)^3, sigma_max)`
  (`mu/oracle/probing.rs`), so the hard-coded 100 silently overrode any
  user-set `sigma_max` whenever `mu_oracle=probing`.

- **Upstream check** (the L2 lesson — verify, don't assume): fetched
  `coin-or/Ipopt` `stable/3.14`, `src/Algorithm/IpProbingMuOracle.cpp`. Its
  `InitializeImpl` does `options.GetNumericValue("sigma_max", sigma_max_,
  prefix);` and the μ computation does `sigma = Min(sigma, sigma_max_);`. So
  upstream's probing oracle **does** honor the user-set `sigma_max`. (The
  option's registered help text — `upstream_options.rs:742`, copied verbatim
  from upstream — claims it is "Only used if option mu_oracle is set to
  quality-function"; that text is inaccurate even in upstream, since
  `IpProbingMuOracle.cpp` reads it too. Behavior, not the help string, is the
  source of truth, so the fix matches upstream behavior and leaves the help
  string verbatim.)

- **Reproduced by running code**: HS071 with `mu_strategy=adaptive`,
  `mu_oracle=probing` solved in **10** iterations at the default
  `sigma_max=100` and in **10** iterations at `sigma_max=1e-6` — identical,
  proving the user value never reached the probing oracle.

- **Fix**: forward the option —
  ```rust
  sigma_max: self.sigma_max,
  ```
  (`adaptive.rs`, with an explanatory comment citing the upstream source) and
  refresh the `sigma_max` field doc-comment to note it now feeds both the
  quality-function and probing oracles. One-line behavioral change; the QF
  branch already did this, so the two arms are now symmetric.

- **Test**:
  `optimize_hs71.rs::hs071_probing_oracle_honors_user_sigma_max` runs HS071
  through the probing oracle twice — at the default `sigma_max` and at
  `sigma_max=1e-6` — and asserts the iteration counts differ (a tiny cap pins
  the centering parameter and reshapes the μ trajectory). Fail-first: pre-fix
  both runs take 10 iters and `assert_ne!` fires; post-fix default=10 vs
  1e-6=8, both still `Solve_Succeeded`. The full `pounce-algorithm` suite stays
  green (lib 245 + every integration test; `optimize_hs71` now 17 tests; 0
  failures), confirming the probing-path change is regression-free.

## L4 detail

**Issue (review L4):** "`golden_section` can return an unevaluated `-100.0`
sentinel endpoint when `qmax <= 0` (`src/mu/oracle/quality_function.rs:540-554`
with 730, 741); also `>=` in `qf_ok` makes the default `qf_tol = 0.0`
flat-stop dead."

This is two separate claims. I checked both against the real upstream source.

**Upstream reference.** Fetched `coin-or/Ipopt` (stable/3.14)
`src/Algorithm/IpQualityFunctionMuOracle.cpp::PerformGoldenSection`. Its loop
condition is
`while( (sigma_up - sigma_lo) >= sigma_tol*sigma_up && (1. - Min(q_lo,q_up,qmid1,qmid2)/Max(q_lo,q_up,qmid1,qmid2)) >= qf_tol && nsections < quality_function_max_section_steps_ )`.
Its post-loop selection has a qf_tol-stop branch
(`if( ... && (1.-Min/Max) < qf_tol ) { ... DBG_ASSERT(qf_min > -100.); }`) and
an else-branch that re-evaluates a sentinel endpoint:
`if( q_up < 0. ) { qtmp = CalculateQualityFunction(UnscaleSigma(sigma_up),...); } else { qtmp = q_up; }`.

**Facet 2 — `>=` in `qf_ok` makes flat-stop dead: NOT A BUG.** pounce
line 499 is `qf_ok = qmax > 0.0 && (1.0 - qmin/qmax) >= qf_tol`. Upstream uses
the *same* `>=` against `qf_tol`. With the default `qf_tol = 0.0`,
`(1 - qmin/qmax) >= 0` holds for any non-degenerate bracket in **both**
codebases, so the qf-tolerance stop simply never triggers by default — that is
upstream's design (the qf_tol stop is opt-in via a positive
`quality_function_eps`), not a pounce divergence. No change warranted.

**Facet 1 — unevaluated `-100.0` sentinel can be returned: REAL, fixed.**
`pick_sigma` always passes one endpoint as the `-100.0` sentinel: search-up
calls `golden_section(sigma_lo, sigma_up, qf_1, -100.0, ...)` (line 730),
search-down calls `golden_section(sigma_lo, sigma_up, -100.0, qf_1minus, ...)`
(line 741). The endpoint q is meant to be evaluated lazily and compared only
after the loop. Upstream never returns the sentinel because (a) it has no
`qmax > 0` guard, so a sentinel state yields a large positive
`(1 - qmin/qmax)` ratio that keeps the loop alive until the slot is
overwritten, and (b) its post-loop else-branch re-evaluates `if( q_up < 0. )`.

pounce **adds** `qmax > 0.0 &&` to `qf_ok` (line 499) — a guard against the
divide-by-zero / nonsense ratio that occurs when *every* sampled q is ≤ 0
(`qmax ≤ 0`). That guard is reasonable in isolation, but it forces
`qf_ok = false` on the very first pass of such a state, breaking the loop with
the sentinel still in place (`nsections = 0`, the sentinel endpoint never
moved). The state then lands in the `width_ok && !qf_ok` branch (540-554),
which — unlike pounce's *own* else-branch (561-572) and unlike upstream —
selected the raw minimum of `{q_lo, q_up, qmid1, qmid2}` **without**
re-evaluating the sentinel, so it returned the unevaluated `-100.0` endpoint
as the spurious minimizer.

**Reproduced (running code).** A focused unit test on the pure `golden_section`
function (no solver state needed): `q(σ) = -σ` on the interior and lower-bound
samples (all ≤ 0 ⇒ `qmax ≤ 0`, tripping the added guard) but `q(σ_up) = +50`
— i.e. the upper endpoint is the *worst* point in the bracket. A search-up
style call `golden_section(1.0, 3.0, q(1.0), -100.0, 1e-3, 0.0, 50, q)` returns
**σ = 3.0** pre-fix (the unevaluated sentinel endpoint, whose true q = 50, the
bracket maximum).

**Fix** (`quality_function.rs`, `width_ok && !qf_ok` branch): before selecting
the minimum, re-evaluate any endpoint that never moved during the loop and
still carries the sentinel —
`if sigma_lo == sigma_lo_in && q_lo < 0.0 { q_lo = q(sigma_lo); }` and the
symmetric `sigma_up` case — mirroring the else-branch and upstream's
`if( q_up < 0. )`. This makes `golden_section` provably never return an
unevaluated `-100.0`. Also refreshed the stale doc-comment (524-530) that had
asserted the sentinel could never reach this branch (it can, precisely because
of pounce's added `qmax > 0` guard).

**Test** (`quality_function.rs::tests::golden_section_never_returns_unevaluated_sentinel`):
asserts the returned σ is `< sigma_up`. **Fail-first confirmed**: pre-fix
returns σ = 3 (panics with "returned the unevaluated sentinel endpoint σ = 3,
true q there = 50"); post-fix returns the real interior minimizer σ ≈ 2.24 and
the test passes. Full `pounce-algorithm` suite green (lib **246** + every
integration test, 0 failures), so the re-evaluation does not perturb any
existing convergence path (the existing `golden_section_*` tests and the
HS071/adaptive-oracle integration tests all still pass).

## L5 detail

**Issue (review L5):** "`max_cpu_time` actually measures wall time —
`src/conv_check/opt_error.rs:257` via `pounce_common::utils::cpu_time()`'s
documented wallclock fallback."

**Confirmed (reading code).** `pounce_common::utils::cpu_time()` was a verbatim
alias for `wallclock_time()`:

```rust
pub fn cpu_time() -> Number {
    wallclock_time()
}
```

with a doc-comment acknowledging it as a stub ("std offers no portable CPU-time
API, so we fall back to wallclock … phase 4 will wire in a real path"). The
convergence check's time-budget gate
(`opt_error.rs:257`,
`timing.overall_alg.live_cpu_time() >= self.max_cpu_time`) therefore stopped on
**wall-clock** elapsed time, not CPU time. So on a problem that spends time
blocked (I/O, an external callback, the OS descheduling the process),
`max_cpu_time` fired early relative to upstream, whose `max_cpu_time` bounds
actual process CPU.

**Upstream reference.** Fetched `coin-or/Ipopt` (stable/3.14)
`src/Common/IpUtils.cpp::CpuTime()`:

* **Unix:** `getrusage(RUSAGE_SELF, &usage)` then
  `ru_utime.tv_sec + 1e-6 * ru_utime.tv_usec` — process **user** CPU time.
  (System time `ru_stime` is exposed separately as `SysTime()`.)
* **Windows (`_MSC_VER`/`__MSVCRT__`):** `clock() / CLOCKS_PER_SEC`. Note that
  on the MSVC runtime `clock()` returns elapsed *real* time, not CPU time — so
  upstream's own Windows path is already wall-ish.

**Fix** (`crates/pounce-common/src/utils.rs`): implement the Unix branch with
`libc::getrusage(RUSAGE_SELF)`, returning `ru_utime` in seconds — a direct port
of upstream's Unix path. Non-Unix targets keep `wallclock_time()`, which is
faithful to upstream's Windows `clock()` behavior. The `unsafe` is confined to
the `getrusage` FFI call (zeroed `rusage` in, return-code checked, degrade to
wallclock on the rare failure rather than panicking in a timing helper).

Dependency: added `libc = "0.2"` to root `[workspace.dependencies]` and a
`[target.'cfg(unix)'.dependencies] libc.workspace = true` to pounce-common, so
only Unix builds pull libc (Windows wheels are unaffected). libc was already in
`Cargo.lock` transitively. This does **not** touch the crates.io publish list
or topological order, so `scripts/check-release-consistency.sh` is unaffected
(pounce-common stays the first publishable crate; libc is an external dep).

**Reproduced / test** (`utils::tests::cpu_time_excludes_sleep_but_counts_compute`,
gated `#[cfg(unix)]`):

1. Record `cpu_time()` and `wallclock_time()`, sleep 300 ms, re-read both.
   Assert `wall_delta − cpu_delta > 0.1 s`: a sleeping thread accrues no user
   CPU, so a real CPU clock must lag wallclock by ~the full sleep. (Generous
   0.1 s margin absorbs any CPU burned by sibling test threads — pounce-common's
   suite is tiny, so real noise is well under that.)
2. Run a 50 M-iteration busy loop (`black_box`'d) and assert `cpu_delta > 0`,
   confirming the clock is live rather than a degenerate constant.

**Fail-first confirmed.** Temporarily reverting `cpu_time()` to
`return wallclock_time();` makes assertion (1) fire:
"cpu_time advanced 0.310s across a 0.310s sleep; it must measure CPU, not
wallclock (wall−cpu gap was only −0.000s)". With the `getrusage` fix in place
the test passes. Full `pounce-common` suite green (58 tests); `pounce-algorithm`
green (lib 246 + every integration test); `cargo check --workspace --exclude
pounce-hsl` clean. The `max_wall_time` gate at `opt_error.rs:260` was already
correct (`live_wallclock_time()`) and is unchanged, so the two budgets are now
distinct as upstream intends.

## L6 detail

**Issue (review L6):** "Dead/divergent duplicates of filter acceptance
predicates — `src/line_search/filter_acceptor.rs:171-179` (no round-off slack,
unlike the live path at 292-300) and 199-229 (parameterized `obj_max_inc` while
the live path hard-codes 5.0)."

The filter acceptor had three textual copies of the same Fletcher-Leyffer
sufficient-progress OR-test, and they had drifted apart:

1. **`is_sufficient_progress` (171-179)** — `phi_trial < phi - gamma_phi*theta
   || theta_trial < (1-gamma_theta)*theta`, a bare `<`.
2. **The live `check_acceptability` path (292-300)** — the same OR-test but via
   `compare_le` (the `10·eps·|basval|` round-off-tolerant `<=`).
3. **`is_acceptable_to_current_iterate` (199-229)** — again the same OR-test via
   `compare_le`, plus a rapid-barrier-increase guard parameterized on an
   `obj_max_inc` argument.

**Divergence (a): missing round-off slack + dead.** `grep` across the workspace
shows `is_sufficient_progress` has **no caller** — it is dead — while
`is_acceptable_to_current_iterate` is live (called from
`pounce-restoration/src/conv_check.rs:163`,
`RestoFilterConvCheck::test_orig_progress`). The dead helper's bare `<`
disagrees with the live `compare_le` exactly on the round-off boundary: near a
solution the barrier objective is flat, so `phi_trial - phi` is dominated by
floating-point summation noise (a tiny *positive* value even on a genuine
descent step) while `-gamma_phi·theta` is a tiny *negative* one. A bare `<`
then rejects a step that `compare_le` accepts — the same flat-objective stall
documented on `armijo_holds`. A future caller reaching for the public
`is_sufficient_progress` would silently get the slack-less, divergent behavior.

**Divergence (b): hard-coded 5.0 vs parameterized cap.** The live
`check_acceptability` rapid-increase guard hard-coded
`(phi_trial - phi).log10() <= 5.0 + basval`, while
`is_acceptable_to_current_iterate` takes `obj_max_inc` as an argument
(the restoration caller passes its own `obj_max_inc`, also defaulting to 5.0).
`obj_max_inc` is a registered upstream option
(`upstream_options.rs:492`, default 5.0). With the value hard-coded in the
regular-phase path, the two code paths would diverge for any non-default
`obj_max_inc`.

**Fix** (`crates/pounce-algorithm/src/line_search/filter_acceptor.rs`):

* Rewrote `is_sufficient_progress` to use `compare_le` for both branches —
  now textually identical to the live OR-test — and made it the **single
  source of truth**: `check_acceptability`'s `suff_progress_ok` and
  `is_acceptable_to_current_iterate`'s tail both now call it. Three copies
  collapse to one.
* Added an `obj_max_inc: Number` field to `FilterLsAcceptor` (default 5.0) and
  changed the live guard from the literal `5.0` to `self.obj_max_inc`, so the
  regular-phase and restoration progress tests read one configurable cap. (The
  env-gated `POUNCE_DBG_LS` diagnostic still computes `rapid_increase_ok` and
  `suff_progress_ok` separately, so its per-branch logging is preserved.)

Because the live regular-phase path already used `compare_le` and `5.0`
(= the new field default), its behavior is **byte-identical** after the
refactor — the change is a dedup, not a behavior change for the default
configuration.

**Tests** (`filter_acceptor::tests`):

* `is_sufficient_progress_accepts_round_off_boundary_like_live_path`: with the
  default acceptor, sets `theta = theta_trial = 1.0` (θ-branch firmly false)
  and `phi = 0`, `phi_trial = -gamma_phi*theta` so that `phi_trial - phi ==
  -gamma_phi*theta` *exactly* — the φ-branch equality boundary. The bare `<`
  rejects this; `compare_le`'s slack accepts it. Asserts the helper returns
  true.
* `check_acceptability_honors_obj_max_inc_field`: drives `check_acceptability`
  with `d_phi = 0` (out of the switching/Armijo branch), `theta_trial = 0.5`
  (< `theta_max`), and a `phi` jump of `1e7` (log10 ≈ 7). The θ-branch
  satisfies sufficient progress, so the decision turns purely on the
  rapid-increase guard: Reject at the default cap 5.0 (threshold 5+1=6), Accept
  once `obj_max_inc` is raised to 10.0 (threshold 11).

**Fail-first confirmed.** Temporarily reverting both edits — `is_sufficient_progress`
back to bare `<`, and the live guard back to the literal `5.0` — makes both new
tests fail (`17 passed; 2 failed`); with the fixes in place all 19
`filter_acceptor` tests pass. Full `pounce-algorithm` green (lib **248** + every
integration test) and `pounce-restoration` green (105), confirming the live
caller of `is_acceptable_to_current_iterate` is unaffected.

## L7 detail

**Issue (L7):** "Watchdog revert applies the current-direction fraction-to-boundary
cap to the snapshot direction — `src/line_search/backtracking.rs:725-737`; the
correct stored cap is `#[allow(dead_code)]`. Rescued by backtracking, but wastes
evaluations post-watchdog."

**Verdict: NOT A BUG (premise refuted by upstream source); dead parity field removed.**

### What the code does

On a `StopWatchDog` revert (`handle_watchdog_failure`, the
`evaluation_error || watchdog_trial_iter > watchdog_trial_iter_max` branch), pounce:

1. restores `curr` to the watchdog snapshot iterate,
2. re-runs the alpha loop on the **snapshot** direction `snap_delta`,
3. passes `alpha_init` as the cap with `skip_first = true`, so `run_alpha_loop`
   starts at `alpha_init * alpha_red_factor` (`backtracking.rs:842-843`).

`alpha_init` here is the **current** outer iteration's fraction-to-the-boundary
cap (`alpha_primal_max` for the direction that just failed), threaded down from
`find_acceptable_trial_point` → `run_filter_line_search` → `handle_watchdog_failure`.
It is **not** recomputed from `snap_delta`.

The review reads this as a bug: it argues the snapshot direction should use its
own FTB cap, which pounce stored at watchdog activation as
`watchdog_alpha_primal_test = aff_step_alpha_primal_max(delta, tau)` — a field
that was marked `#[allow(dead_code)]` and never read.

### Why it is not a bug — upstream does exactly the same

Fetched `coin-or/Ipopt` (stable/3.14)
`src/Algorithm/IpBacktrackingLineSearch.cpp`:

- `FindAcceptableTrialPoint`, watchdog-cap-exceeded branch:
  ```cpp
  if( evaluation_error || watchdog_trial_iter_ > watchdog_trial_iter_max_ )
  {
     StopWatchDog(actual_delta);
     skip_first_trial_point = true;
  }
  ```
- The next `DoBacktrackingLineSearch` with `skip_first_trial_point = true`:
  ```cpp
  if( skip_first_trial_point )
  {
     alpha_primal *= alpha_red_factor_;
  }
  ```
  i.e. it reduces the **existing** `alpha_primal` and does **not** recompute the
  FTB cap from `actual_delta`.
- `StopWatchDog` only swaps the direction back to the snapshot:
  ```cpp
  IpData().set_trial(old_trial);
  IpData().AcceptTrialPoint();
  actual_delta = watchdog_delta_->MakeNewContainer();
  IpData().SetHaveAffineDeltas(false);
  ```
  It never touches `alpha_primal`.

At the revert point `alpha_primal` holds the **current** outer iteration's
`alpha_primal_max` (set fresh at the top of this call's `DoBacktrackingLineSearch`;
in the watchdog window `alpha_min = alpha_primal_max`, so only the single full
step ran before the cap was exceeded). So upstream's restart =
`current_direction_alpha_primal_max * alpha_red_factor` applied to the reverted
snapshot delta — **byte-for-byte the same policy as pounce's
`alpha_init * alpha_red_factor`**.

### The "stored cap" is a misread

Upstream's `watchdog_alpha_primal_test_` lives in `IpFilterLSAcceptor` and is the
acceptor's frozen **Armijo test step length** used while in watchdog mode (the
`alpha_primal_test` fed to the switching/sufficient-decrease test), **not** a
line-search restart cap. Nothing in upstream consumes a snapshot FTB cap at the
revert site, so pounce should store no analogue here. The `aff_step_alpha_primal_max`
pounce stashed in `watchdog_alpha_primal_test` was simply unused scaffolding.

The "wastes evaluations" observation — if the snapshot direction's true FTB
boundary is tighter than `alpha_init`, the first reverted trial overshoots and is
rejected before backtracking finds an acceptable step — is real but is **upstream's
behavior too** (upstream also starts from the current direction's reduced alpha and
backtracks). It is not a pounce-introduced regression.

### Change made (cleanup only, no behavior change)

Removed the genuinely-dead `watchdog_alpha_primal_test` field, its initializer,
and the `aff_step_alpha_primal_max(delta, tau)` computation in `start_watchdog`
(eliminating the `#[allow(dead_code)]`), and added a comment at the revert site
explaining that passing `alpha_init` is the faithful upstream port — so a future
reviewer does not re-flag it. This matches the repo's recent dead-code-removal
pattern (#120, #121).

### Verification

`cargo build -p pounce-algorithm` is clean (no dead-code warning after removal).
Full `pounce-algorithm` suite green: **lib 248 + every integration test, 0
failures**. The watchdog revert path is exercised by the HS/integration solves
(the code comments cite PFIT3/PFIT4/scon1dls as problems that drive the
accept-anyway / revert branches), so the green suite confirms removing the dead
field changed no behavior. No new regression test was added because there is no
bug to pin — adding a snapshot-FTB recompute would *introduce* a divergence from
upstream, not remove one (same disposition as L2).

## L8 detail

**Issue (L8):** "Ruiz scaler's 0/1-based auto-detection misclassifies a 0-based
triplet whose index 0 carries no entries (`crates/pounce-linsol/src/ruiz.rs:117-129`);
factors land on the wrong rows. Applied consistently, so result quality degrades
rather than correctness; the only in-tree caller is safe (1-based)."

**Verdict: real (latent) bug — FIXED.**

### Root cause

`RuizTSymScalingMethod::compute_sym_t_scaling_factors` accepts triplets in
either 0-based or Fortran 1-based form and auto-detected the base with a
**min-only** heuristic:

```rust
let mut min_idx = airn[0];
for k in 0..nnz_us { /* min over airn[k], ajcn[k] */ }
let offset: Index = if min_idx >= 1 { 1 } else { 0 };
```

The detector's assumption is "any triplet whose smallest index is ≥ 1 is
1-based". That fails for a **0-based** triplet whose row 0 is structurally empty
(no entry references index 0): its `min_idx` is ≥ 1, so it is misread as 1-based,
and the subsequent `airn[k] - offset` / `ajcn[k] - offset` shift every entry down
one row. The Ruiz iteration then computes each row's ∞-norm on the wrong row, so
`d_i` factors are assigned to the wrong variables. It is applied consistently
(the same wrong offset is used to read and to write), so the factorization stays
*correct* — only the *quality* of the equilibration degrades.

The only in-tree caller routes 1-based Fortran triplets (`max_idx == n`), so the
defect is latent in production; it bites any 0-based caller with an empty leading
row (and is reachable from the public `TSymScalingMethod` trait API).

### Why min-only is wrong, and the fix

For an `n×n` matrix the two index conventions have disjoint *boundary* signals:

* a **1-based** triplet has indices in `[1, n]` — it can reference `n`, never `0`;
* a **0-based** triplet has indices in `[0, n-1]` — it can reference `0`, never `n`.

So each extreme is individually decisive when present. The new detection tracks
**both** `min_idx` and `max_idx`:

```rust
let offset: Index = if min_idx == 0 {
    0                        // references index 0 ⇒ unambiguously 0-based
} else if max_idx >= n {
    1                        // references index n ⇒ unambiguously 1-based
} else if max_idx == n - 1 {
    0                        // fills [.., n-1] ⇒ full 0-based n×n coverage
} else {
    1                        // truly ambiguous tiny submatrix ⇒ legacy 1-based
};
```

The third arm is the fix for the reported case: a 0-based triplet with an empty
row 0 still covers the last row (`max_idx == n - 1`), which a 1-based triplet of
an `n×n` system never does (its last row's diagonal would give `max_idx == n`).
Only a triplet that touches *neither* boundary (no index 0 and no index n−1/n,
i.e. a strict interior submatrix — not something the symmetric KKT scaler is fed,
since KKT systems carry a full diagonal) remains ambiguous, and there it keeps
the historical 1-based default.

### Verification

* **Test added** — `ruiz::tests::zero_based_with_empty_first_row_is_not_misread_as_fortran`:
  `K = diag([0, 4, 9])` stored 0-based (entries only on rows/cols 1 and 2;
  `min_idx == 1`, `max_idx == 2 == n − 1`). Asserts the empty row 0 keeps
  `d = 1` and that `K_11`, `K_22` equilibrate to ≈ 1.
* **Fail-first confirmed** — temporarily reverting the offset to the old
  `if min_idx >= 1 { 1 } else { 0 }` rule makes the new test fail with
  `empty row 0 must keep d=1, got 0.5` (the factor for row 1 leaked onto row 0,
  and row 2 was left unscaled). Restored, it passes.
* **No regression** — the existing `fortran_index_style` (1-based, `max_idx == n`)
  and 0-based tests (`equilibrates_diagonal_extremes`, `zero_row_keeps_unit_scale`,
  `fuzz_reduces_imbalance`) are unchanged. Full `pounce-linsol` suite green
  (18 lib + 1 integration, 0 failures).

## L9 detail

**Issue (review L9).** The KKT-dump diagnostic in
`crates/pounce-linsol/src/t_sym_solver.rs` enforced its "dump once" behavior by
calling `unsafe { std::env::remove_var("POUNCE_DBG_KKT_DUMP") }` after writing
the dump. Mutating the process environment (`setenv`/`unsetenv`) is not
thread-safe against concurrent `getenv`; Rust 2024 makes these `unsafe` for that
reason. pounce-feral schedules solves on a rayon pool, so this unset can run
concurrently with an environment read on another thread (here or in any
dependency that reads env), which is undefined behavior — the real hazard is the
data race, not a missed dump.

**Verification.** Read the dump block and traced the call path: the static
`CALL_COUNT`/`WARNED`/dumped state plus the `remove_var` disable were all
process-global, and the only synchronization for "dump exactly once" was the env
unset itself. Confirmed feral's parallel-solve model (rayon) makes concurrent
entry real.

**Fix.** Remove all environment mutation. `POUNCE_DBG_KKT_DUMP` (and the
skip-N-calls knob) are now read-only inputs; the one-shot guarantee is a
lock-free atomic claim:

```rust
fn claim_kkt_dump(n_call: usize, skip: usize, dumped: &std::sync::atomic::AtomicBool) -> bool {
    use std::sync::atomic::Ordering;
    if n_call < skip {
        return false;
    }
    !dumped.swap(true, Ordering::SeqCst)
}
```

Statics became `CALL_COUNT: AtomicUsize` and `DUMPED: AtomicBool` (the previous
`WARNED` flag folded into the same one-shot). No `unsafe`, no `set_var`/
`remove_var`. `swap(true)` is atomic, so across any number of threads exactly
one call observes the prior `false` and returns `true`.

**Tests** (`t_sym_solver::tests`):
- `claim_kkt_dump_is_one_shot_after_skip` — sequential and deterministic:
  with `skip=2`, calls 0 and 1 return `false`, call 2 returns `true`, calls 3
  and 4 return `false`.
- `claim_kkt_dump_claims_exactly_once_under_concurrency` — 32 threads released
  together by a `Barrier` all call `claim_kkt_dump(0, 0, &shared)`; the test
  asserts the winner count is exactly 1.

**Fail-first.** Temporarily made the helper always claim (drop the `swap`
one-shot): both tests fail (the sequential one on call 3 returning `true`, the
concurrency one on a winner count > 1). Restored; full `pounce-linsol` suite
green (20 lib + 1 integration, 0 failures). `cargo fmt -p pounce-linsol
--check` clean.

## L10 detail

**Issue (review L10).** MA57's symbolic-phase workspace sizing in
`crates/pounce-hsl/src/ma57.rs` did all index arithmetic in `Index` (`= i32`,
matching MA57's Fortran `INTEGER`), with no overflow guard:

- `self.lkeep = 5 * n + ne + n.max(ne) + 42;` (cpp:536) and
  `self.iwork = vec![0; (5 * n) as usize];` — for a matrix with `n ≈ ne`, the
  leading behavior is `7·n`, which exceeds `i32::MAX` (2 147 483 647) once
  `n ≳ 3.07×10⁸`. In a release build the i32 sum wraps to a negative value,
  and `negative_i32 as usize` becomes an astronomically large `usize`, so the
  `vec![…; len]` either aborts (OOM) or attempts an absurd allocation; in a
  debug build the multiply/add panics on overflow.
- `let suggested_lfact = (self.info[8] as Number * scale).ceil() as Index;` —
  `info[8]` is MA57's own suggested size (already an i32), grown by
  `ma57_pre_alloc` (default 1.05). The `f64 -> i32` cast *saturates* to
  `i32::MAX` (Rust ≥ 1.45), so it no longer wraps, but an i32::MAX-element
  `fact`/`ifact` allocation is still nonsense.
- backsolve `let lwork = n * nrhs;` — the same i32 multiply overflow for large
  `n*nrhs`.

The review notes this is inherited from the Fortran interface but "cheap to
guard," and asks for a clean `FatalError` rather than an allocation/abort.

**Verification.** The overflow is a provable arithmetic fact (i32 cannot hold
`7·3.5×10⁸ = 2.45×10⁹`). Because the crate links proprietary MA57, the tests
were run against a locally-installed CoinHSL via `COINHSL_DIR` (the library
lives outside the repo and is never committed, per its license); CI keeps
`pounce-hsl` in its `--exclude` list for the same reason.

**Fix.** Two pure, unit-testable helpers replace the inline arithmetic:

```rust
fn ma57_symbolic_sizes(n: Index, ne: Index) -> Option<(Index, Index)> {
    let (n64, ne64) = (n as i64, ne as i64);
    let lkeep = 5 * n64 + ne64 + n64.max(ne64) + 42;
    let liwork = 5 * n64;
    if lkeep > Index::MAX as i64 || liwork > Index::MAX as i64 {
        return None;
    }
    Some((lkeep as Index, liwork as Index))
}

fn ma57_scaled_size(base: Index, scale: Number) -> Option<Index> {
    let scaled = (base as Number * scale).ceil();
    if scaled > Index::MAX as Number {
        return None;
    }
    Some((scaled as Index).max(base))
}
```

`symbolic_factorization` calls `ma57_symbolic_sizes` (let-else → `FatalError`)
for `lkeep`/`iwork`, and `ma57_scaled_size` for the two suggested sizes;
`backsolve` widens `n*nrhs` to i64 and returns `FatalError` if it exceeds
`i32::MAX`. The on-the-happy-path behavior is byte-identical (same lengths for
in-range problems); only the out-of-range cases change from
overflow/abort to a clean `ESymSolverStatus::FatalError`.

**Tests** (`ma57::tests`):
- `ma57_symbolic_sizes_guards_i32_overflow` — exact small sizing
  `(5*N+NE+max+42, 5*N)`; `n=ne=3×10⁸` (`7·n = 2.1×10⁹ < i32::MAX`) → `Some`;
  `n=ne=3.5×10⁸` (`2.45×10⁹ > i32::MAX`) → `None`.
- `ma57_scaled_size_guards_overflow_and_floors_at_base` — `1.05×` growth;
  `scale<1` floors at `base` (never shrinks MA57's minimum); `Index::MAX-1`
  scaled up → `None`.

**Fail-first.** Temporarily stripped both `> Index::MAX` guards (returning the
`i64 -> i32 as` cast unconditionally): `ma57_symbolic_sizes_guards_i32_overflow`
fails (the `is_none()` assertion is false — the wrapped i32 is `Some`) and
`ma57_scaled_size_guards_overflow_and_floors_at_base` fails with
`left: Some(2147483647), right: None`. Restored; full `pounce-hsl` suite green
(12 lib + 3 integration, 0 failures), `cargo fmt -p pounce-hsl --check` clean,
`cargo clippy -p pounce-hsl` clean for correctness/suspicious.

## L11 detail

- **Bug (ma57)**: `Ma57SolverInterface::backsolve` (the inner of `multi_solve`)
  built a fresh `vec![0.0; lwork as usize]` (`lwork = n*nrhs`) MA57C real
  workspace on *every* call. In the IPM the KKT matrix is symbolically and
  numerically factored once per iteration and then back-solved several times
  (predictor/corrector, inertia-correction re-solves, iterative refinement), so
  this is a per-solve heap allocation + zero-fill of a buffer MA57C only ever
  writes to.
- **Upstream**: `coin-or/Ipopt` (stable/3.14)
  `src/Algorithm/LinearSolvers/IpMa57TSolverInterface.cpp::Solve` allocates the
  MA57C `WORK` as an *uninitialized* `new double[lwork]` and never reads it
  before MA57C populates it — it is pure scratch, so neither the zero-fill nor a
  per-solve fresh allocation is required; the buffer is safe to reuse.
- **Fix (ma57)**: hoisted the workspace to a struct field `work: Vec<Number>`
  (initialized `Vec::new()` in `with_options`). The backsolve now does
  `self.work.resize(lwork as usize, 0.0)` — a no-op once the buffer is large
  enough, so the factor-once/solve-many hot path allocates **zero** times after
  the first solve — and passes `self.work.as_mut_ptr()` to MA57C. Stale
  contents are harmless because MA57C treats it as scratch.
- **Test** (`ma57::tests::backsolve_reuses_workspace_across_repeated_solves`):
  factors `A=[[2,1],[1,3]]` once via `multi_solve(true,…)`, asserts
  `s.work.len() == n`, captures `s.work.capacity()`, then runs three further
  `multi_solve(false,…)` solves with distinct RHS, asserting each result is
  correct to 1e-10 (against the closed-form `A^-1`) **and** that
  `s.work.capacity()` never changes (no reallocation on same-size reuse).
- **Verification**: built + linked against a local CoinHSL (`COINHSL_DIR`, kept
  out of the repo for licensing) — full pounce-hsl suite green (13 lib + 3
  integration). **Fail-first**: reverting to the per-solve `vec![0.0; …]` makes
  the capacity-stability assertion fail (`left: 0, right: 2`); restored, all
  green. `cargo fmt -p pounce-hsl --check` and `cargo clippy -p pounce-hsl
  --all-targets` (correctness/suspicious) clean. pounce-hsl is `--exclude`d from
  the CI build/test/clippy jobs (needs proprietary HSL), so this test runs only
  where CoinHSL is installed — verified locally here.
- **feral half — cannot be fixed in-tree (external pinned API).**
  `pounce-feral`'s `backsolve` (`crates/pounce-feral/src/lib.rs:559-577`)
  dispatches to `self.solver.{solve,solve_many,solve_refined,solve_many_refined}`
  on the external `feral::Solver` (a git-pinned dependency). Each of those
  methods **returns an owned `Result<Vec<f64>>`**, which pounce then
  `copy_from_slice`s into the caller-provided output buffer. The allocation is
  performed *inside* the feral crate, not in pounce. `feral::Solver` exposes no
  in-place / solve-into-buffer API and its `last_factors` field is private with
  no accessor, so there is no way to back-solve into a reused buffer from
  pounce. Removing this allocation requires an upstream feral change (e.g. a
  `solve_into(&mut [f64])` method or a public `factors()` accessor). Recorded
  per the "document issues that cannot be verified/fixed" rule; the in-tree
  (ma57) half is fixed above.

## L12 detail

- **Bug**: `FeralConfig::from_env` (`crates/pounce-feral/src/lib.rs`) reads its
  six other knobs under the documented `POUNCE_FERAL_*` prefix
  (`POUNCE_FERAL_CASCADE_BREAK`, `_FMA`, `_REFINE`, `_SINGULAR_PIVOT_FLOOR`,
  `_ORDERING`, `_SCALING`) but read the Bunch-Kaufman pivot threshold from the
  bare **`FERAL_PIVTOL`** — off-convention — and the `from_env` doc-comment did
  not mention pivtol at all. A user following the documented convention sets
  `POUNCE_FERAL_PIVTOL` and it is silently ignored.
- **Distinct from `FERAL_PARALLEL`**: that bare-prefix var is *intentionally*
  legacy and documented as such (lib.rs comments at the `parallel` field /
  `with_config`). `FERAL_PIVTOL` was an undocumented inconsistency, not a
  deliberate legacy escape hatch.
- **Reproduced (running code)**: a throwaway `examples/` binary calling
  `FeralConfig::from_env()` printed `POUNCE_FERAL_PIVTOL=0.3 -> pivtol = 1e-8`
  (the convention name ignored) and `FERAL_PIVTOL=0.4 -> pivtol = 0.4` (legacy
  honored). Example removed after confirming.
- **Fix**: extracted a pure free fn
  `resolve_pivtol_env(pounce: Option<&str>, legacy: Option<&str>) -> f64` with
  precedence: `POUNCE_FERAL_PIVTOL` (the convention) > `FERAL_PIVTOL`
  (deprecated legacy alias, kept for back-compat) > `1e-8` default; an
  unset/unparseable source falls through to the next. `from_env` now calls it
  with both `std::env::var(...).ok().as_deref()` values. Keeping the parse
  logic in a pure helper means the regression test never mutates the process
  environment — important because pounce-feral drives solves on a rayon pool,
  so `set_var`/`remove_var` in a test would be the exact data-race UB fixed in
  L9.
- **Docs updated**: the `from_env` doc-comment now lists `POUNCE_FERAL_PIVTOL`
  (and notes the legacy alias); the in-code comment at
  `np.bk.pivot_threshold = cfg.pivtol` names both vars; the `feral_pivtol`
  OptionsList option help (`pounce-algorithm/src/upstream_options.rs:1029`) now
  reads "Falls back to the POUNCE_FERAL_PIVTOL environment variable (or its
  deprecated legacy alias FERAL_PIVTOL) when not set on the OptionsList."
- **Test** (`pounce-feral` `tests::resolve_pivtol_env_honors_pounce_convention`):
  asserts the convention name is read (`Some("0.3"), None → 0.3`), the legacy
  alias still works when the convention var is unset (`None, Some("0.4") →
  0.4`), the convention wins when both are set (`Some("0.3"), Some("0.4") →
  0.3`), the default holds with neither (`None, None → 1e-8`), and an
  unparseable convention value falls through to the legacy alias then the
  default.
- **Fail-first**: reverting the helper to legacy-only (ignore the `pounce` arg
  — the pre-fix behavior) fails the test with `left: 1e-8, right: 0.3`.
  Restored: full pounce-feral suite green (15 lib) and pounce-algorithm green
  (248 lib); `cargo fmt`/`clippy` (correctness/suspicious) clean on both
  crates.

## L13 detail

- **Two doc/code sign mismatches in the restoration formulas; in both the
  code is correct (and matches upstream Ipopt), the prose was wrong.**
- **Facet 1 — constraint slack signs.** `restoration_constraint_c` /
  `restoration_constraint_d` (`resto_nlp.rs:895,907`) and the `eval_c`/`eval_d`
  doc-comments implement `c_resto = c_orig(x) + n_c − p_c` and
  `d_resto = d_orig(x) + n_d − p_d`. The module-level doc (lines 6-7) instead
  wrote `c(x) − n_c + p_c = 0` and `d(x) − n_d + p_d − s = 0` — the slack
  signs swapped. The implemented `+n − p` is correct: upstream
  `IpRestoIpoptNLP` sets `p_c = c(x) + n_c` (so `c + n − p = 0`), verified by
  WebFetch, and the existing tests
  `constraint_{c,d}_combines_orig_n_p_with_correct_signs` already assert
  `c + n − p`. The identical wrong-sign prose also appeared in
  `resto_alg_builder.rs:11-12`'s problem statement.
- **Facet 2 — the closed-form quadratic.** `resto_resto.rs::compute_n_p`
  computes `a = mu/(2ρ) − 0.5·c_i`, `b = c_i·mu/(2ρ)`, `v = a + sqrt(a²+b)`,
  `n_i = v`, `p_i = c_i + v`. This is byte-identical to upstream
  `IpRestoRestoPhase.cpp::solve_quadratic`, whose literal body (quoted via
  WebFetch) is `v=a; v=v*v; v+=b; v=sqrt(v); v+=a` ⇒ `a + sqrt(a²+b)` with the
  same `a`,`b`. The module doc claimed this root solves `v² + 2·a·v − b = 0`,
  but the positive root of *that* quadratic is `−a + sqrt(a²+b)`. The root the
  code actually computes, `a + sqrt(a²+b)`, solves `v² − 2·a·v − b = 0`
  (substitute: `(v − a)² = a² + b`).
- **Independent confirmations of the correct sign**:
  - First-principles: minimizing the per-element resto barrier
    `φ(n) = ρ(2n + c) − μ·ln n − μ·ln(c+n)` gives
    `n² + (c − 2·half)·n − c·half = 0` with `half = mu/(2ρ)`; since
    `−2a = c − 2·half`, this is `n² − 2a·n − b = 0`.
  - c=0 sanity check: the true minimizer is `n = μ/ρ = 2·half`; the code's
    `a + sqrt(a²)` = `half + half` = `2·half` ✓, whereas the doc's implied
    `−a + sqrt(a²)` = `0` ✗.
  - The existing `resto_resto` test's *assertion* already used the correct
    `v*v − 2.0*a*v − b` (only its name/comments said `+2av`); `init.rs`'s
    `init_slack_pair_satisfies_quadratic_root` likewise asserts
    `n*n − 2.0*a*n − b ≈ 0`.
- **Fix (documentation/labels only — no production code changed)**:
  - `resto_nlp.rs:6-7`: constraints corrected to `c(x) + n_c − p_c = 0`,
    `d(x) + n_d − p_d − s = 0`, with a note tying them to the implementations.
  - `resto_alg_builder.rs:11-12`: same constraint-sign correction in the
    nested-resto problem statement.
  - `resto_resto.rs:16-22`: quadratic corrected to `v² − 2·a·v − b = 0` with a
    one-line derivation and an explicit note that upstream's `solve_quadratic`
    computes the same `a + sqrt(a²+b)`.
  - `resto_resto.rs` test renamed
    `quadratic_root_satisfies_v2_plus_2av_minus_b_zero` →
    `..._v2_minus_2av_minus_b_zero`; comments rewritten to the correct
    derivation; added an assertion that the wrong form `v² + 2av − b` is
    clearly non-zero (`> 1e-4`) so a future regression in either the code or
    the documented sign is caught.
- **Verification**: full pounce-restoration suite green (105 lib + all
  integration), including the renamed test and the pre-existing constraint-sign
  tests. The "fail-first" for a doc bug is the provable contradiction: the
  doc's `+2av` / `−n+p` forms do not hold for the code's actual,
  upstream-matching values (the new `wrong.abs() > 1e-4` assertion makes the
  `+2av` failure explicit). `cargo fmt -p pounce-restoration --check` and
  `clippy` (correctness/suspicious) clean.

## L14 detail

- **Bug**: the QP inertia-control retry loops gate "is this linear-solver
  failure recoverable by shifting the Hessian diagonal?" on a case-sensitive
  substring test, duplicated at four sites:
  - `crates/pounce-qp/src/solver.rs:123` (first inertia probe) and `:142`
    (shift-and-retry).
  - `crates/pounce-qp/src/schur.rs:275` and `:297` (the Schur-complement
    path's equivalents).
  Each read `QpError::LinearSolverFailure(msg)` and tested
  `msg.contains("inertia") || msg.contains("singular")`.

- **Root cause**: two error-message-producing paths with different casing.
  - The **factorize** path emits lowercase messages (e.g. inertia/singular
    descriptions), which the lowercase `contains` matched.
  - `LinearSolver::resolve`'s catch-all (`crates/pounce-qp/src/factor.rs:172`)
    formats the backend status with `Debug`:
    `Err(QpError::LinearSolverFailure(format!("resolve backend status:
    {other:?}")))`. `ESymSolverStatus`'s variants are **capitalized**
    (`Singular`, `WrongInertia`), so a resolve-path singular / wrong-inertia
    failure becomes `"resolve backend status: Singular"`, which
    `contains("singular")` does **not** match. The recoverable failure then
    escaped the retry loop as an unrecoverable hard error — the inertia-control
    shift that should have rescued the solve was never attempted.

- **Fix**: centralized the recoverability decision in a single predicate
  `QpError::is_recoverable_factorization_failure()`
  (`crates/pounce-qp/src/error.rs`) that lowercases the message first
  (`let m = msg.to_ascii_lowercase(); m.contains("inertia") ||
  m.contains("singular")`) and returns `false` for every non-
  `LinearSolverFailure` variant. Routed all four matchers through it,
  deleting the inline duplicated substring tests. This both fixes the casing
  bug at the resolve site and removes the four-way duplication so future
  recoverability tweaks happen in one place. No production control flow
  changed beyond the predicate result (the lowercase factorize-path messages
  still classify identically).

- **Test**: `crates/pounce-qp/src/tests/refinement_unit.rs` ::
  `recoverable_factorization_failure_is_case_insensitive` — asserts the
  predicate accepts the lowercase factorize-path messages AND the capitalized
  resolve-path Debug strings (`"resolve backend status: Singular"`,
  `"resolve backend status: WrongInertia"`), and rejects non-recoverable
  `LinearSolverFailure` messages (`"backend reported fatal error"`,
  `"resolve called before factorize"`) plus a non-`LinearSolverFailure`
  variant (`DimensionMismatch`).

- **Fail-first confirmed**: reverting the predicate body to the original
  case-sensitive `msg.contains("inertia") || msg.contains("singular")` makes
  the test fail on the capitalized resolve-path string
  (`assertion failed: QpError::LinearSolverFailure("resolve backend status:
  Singular".into()).is_recoverable_factorization_failure()`). Restored, the
  full pounce-qp suite is green (78 lib + 1 doc + 5 integration, 0 failures).
  `cargo fmt -p pounce-qp -- --check` clean; clippy
  (correctness/suspicious) clean.

## L15 detail

Two independent defects in the QP l1-elastic mode (§4.3) and its dispatch.
Both are pounce-internal design bugs (the elastic reformulation + Schur-update
machinery are pounce's own §4.3/§4.5 design — not present in upstream Ipopt),
so verification is by reading + running pounce code, not an upstream fetch.

### (1) Dead `Indefinite` inertia arm

- **Site**: `crates/pounce-qp/src/elastic.rs`.
  - `ElasticReformulation::build(qp, gamma)` constructed the augmented problem
    but **discarded** `qp.hessian_inertia`.
  - `original_inertia()` unconditionally returned `HessianInertia::Psd`.
  - `as_qp()` (lines 162-165) builds the augmented `QpProblem` with
    `hessian_inertia: match self.original_inertia() {
    HessianInertia::Psd | HessianInertia::Unknown => HessianInertia::Psd,
    HessianInertia::Indefinite => HessianInertia::Indefinite }`.
- **Bug**: because `original_inertia()` could never return `Indefinite`, the
  `Indefinite` match arm was **dead** and the augmented problem was *always*
  marked `Psd`. An indefinite original `H` was therefore solved through the
  augmented elastic problem as if it were PSD, telling the solver to skip the
  §4.5 inertia-control assumption it actually needs.
- **Fix**: added an `orig_inertia: HessianInertia` field to
  `ElasticReformulation`, set in `build` from `qp.hessian_inertia`;
  `original_inertia()` returns it. The augmented Hessian is
  block-diag(`H_orig`, 0), which shares `H_orig`'s definiteness category
  (appending zero slack diagonals cannot introduce negative curvature), so
  propagating the original inertia is correct: `Indefinite` now reaches the
  live arm, while `Psd`/`Unknown` still collapse to `Psd` (the augmented
  Hessian is PSD with explicit zero slack diagonals). No change to the
  common-case (`Unknown`→`Psd`) behavior.

### (2) `solve_elastic` ignored `opts.use_schur_updates`

- **Site**: `crates/pounce-qp/src/solver.rs`.
  - The top-level `solve` dispatches the general path between
    `solve_general_schur` (when `opts.use_schur_updates`) and `solve_general`
    (lines ~1587-1591).
  - `solve_elastic`'s recursive solve hard-called
    `self.solve_general(&qp_aug, Some(&ws), opts)` — ignoring the flag.
- **Bug**: an infeasible problem solved with `use_schur_updates = true`
  recovered through elastic mode but **silently fell back to the refactor
  path** for the augmented solve, so the opt-in Schur machinery was never
  exercised on the recovery solve.
- **Fix**: `solve_elastic` now mirrors the top-level dispatch
  (`if opts.use_schur_updates { solve_general_schur } else { solve_general }`).
  Both inner solvers recurse *directly*, bypassing the `solve` feasibility
  audit, so the original no-re-audit / no-recovery-loop property is preserved;
  updated the audit comment that documented the bypass to name both paths.

### Tests

- `crates/pounce-qp/src/tests/elastic_unit.rs` ::
  `as_qp_propagates_original_hessian_inertia` — builds reformulations from
  originals marked `Indefinite` / `Psd` / `Unknown` and asserts
  `as_qp().hessian_inertia` is `Indefinite` / `Psd` / `Psd` respectively (the
  `Indefinite` case is the formerly-dead arm).
- `crates/pounce-qp/src/tests/analytical.rs` ::
  `l15_elastic_honors_use_schur_updates` — the same infeasible QP as
  `problem_5_infeasibility_certified_by_elastic_mode`, solved with
  `use_schur_updates = true`. Asserts the same minimal-l1 certificate
  (`Infeasible`, `used_phase1`, `x ≈ 3.0`) **and** `stats.n_schur_updates > 0`,
  proving the Schur path ran inside the elastic recovery (the refactor path
  leaves the counter at 0).

### Fail-first

Reverting both edits — `original_inertia()` back to a hardcoded
`HessianInertia::Psd`, and the `solve_elastic` dispatch back to a bare
`solve_general` — makes both tests fail:
`as_qp_propagates_original_hessian_inertia` fails on the `Indefinite` case
(`Psd != Indefinite`) and `l15_elastic_honors_use_schur_updates` fails on
`n_schur_updates == 0`. Restored, the full pounce-qp suite is green (80 lib +
1 doc + 5 integration, 0 failures); `cargo fmt -p pounce-qp -- --check` clean;
clippy (correctness/suspicious) clean.

## L16 detail

Two defects in the parametric-sensitivity bound clamp (`pounce-sensitivity`,
a port of upstream sIPOPT's `SensStdStepCalculator::BoundCheck`). Both are
pounce-internal: the `DenseVector` homogeneous-value optimization and the
`values()`/`expanded_values()` accessor split are pounce's own, so verification
is by reading + running pounce code.

### (1) Index-OOB panic on non-dense bound vectors

- **Site**: `crates/pounce-sensitivity/src/boundcheck.rs`.
  - `compressed_values(v)` downcasts the bound `dyn Vector` to `DenseVector`;
    on failure it returns `Vec::new()`. The doc comment promises the
    boundcheck then "silently no-ops, matching upstream's behavior when bounds
    aren't represented as DenseVectors".
  - But `clamp_step_to_bounds`'s two loops iterate over the bound *expansion
    matrix*'s positions and index `bounds[compressed_i]` for each one.
- **Bug**: a non-dense `x_l`/`x_u` with a non-empty `px_l`/`px_u` makes
  `bounds` empty while `compressed_i` still ranges over the expansion entries,
  so `bounds[0]` panics (`index out of bounds: the len is 0 but the index is
  0`) — the opposite of the documented no-op.
- **Fix**: replaced both `let lo = bounds[compressed_i];` /
  `let hi = bounds[compressed_i];` with `bounds.get(compressed_i)` returning
  early (`continue`) on `None`. This honors the no-op contract for non-dense
  bounds and also defends against a bounds slice shorter than the expansion.

### (2) Homogeneous-vector debug_assert in `values()`

- **Site**: `boundcheck.rs::compressed_values`, plus the sibling `dense_to_vec`
  helpers in `crates/pounce-sensitivity/src/solver.rs:408` and
  `crates/pounce-sensitivity/src/convenience.rs:438`.
- **Bug**: `DenseVector::values()` is documented to "Panic if currently
  homogeneous" — it carries `debug_assert!(self.initialized &&
  !self.homogeneous)` (mirroring upstream's `DBG_ASSERT` in
  `DenseVector::Values() const`). A homogeneous bound vector (e.g. every lower
  bound 0, stored as a scalar with the dense storage freed) makes
  `dv.values().to_vec()` panic in debug/test builds. `expanded_values()` is
  the accessor that "always returns a fully-materialized slice", allocating the
  scalar fan-out when homogeneous — which is what the siblings should use.
- **Fix**: switched all three `dv.values().to_vec()` call sites to
  `dv.expanded_values()` (which already returns an owned `Vec`, so the
  `.to_vec()` drops out).

### Tests (`boundcheck::tests`)

- `clamp_handles_homogeneous_bounds_without_panicking` — a homogeneous lower
  bound built via `Vector::set(0.0)` (asserts `is_homogeneous()`), one
  violating and one non-violating coordinate; asserts no panic and exactly one
  correct clamp.
- `clamp_is_noop_on_non_dense_bounds` — a 1-block `CompoundVector` (the only
  other `dyn Vector` implementation) as `x_l`, with `px_l` selecting both
  variables and a deeply-violating `dx`; asserts 0 clamps and `dx` untouched
  (the documented no-op).

### Fail-first

Reverting both fixes (`expanded_values()` → `values().to_vec()`;
`bounds.get(compressed_i)`+`continue` → `bounds[compressed_i]`) makes both
tests fail:
- `clamp_handles_homogeneous_bounds_without_panicking` panics at
  `crates/pounce-linalg/src/dense_vector.rs:131` —
  `assertion failed: self.initialized && !self.homogeneous`.
- `clamp_is_noop_on_non_dense_bounds` panics at `boundcheck.rs` —
  `index out of bounds: the len is 0 but the index is 0`.

Restored, the full pounce-sensitivity suite is green (45 lib + integration, 0
failures); `cargo fmt -p pounce-sensitivity -- --check` clean; clippy
(correctness/suspicious) clean.

## L17 detail

**Issue (code-review-2026-06.md L17).** `IndexPCalculator::schur_matrix`
drops the B-row sign (mirrors upstream, but `from_parts` accepts −1 signs
making the wrong result reachable) and caches P columns by column index
only, so two A rows selecting the same column with opposite signs share one
cached column (`p_calculator.rs:150-166, 191-199`).

**Upstream cross-check.** Fetched
`contrib/sIPOPT/src/SensIndexPCalculator.cpp` from `coin-or/Ipopt`
(`stable/3.14`). Both pounce behaviors faithfully mirror upstream:

- `GetSchurMatrix` builds `S` by reading `P` at `B`'s *column indices*
  (`extractor.GetIntVector(...)`) and writing `S(i+1, col) = -(*(it->second))[i]`
  — it never multiplies by `B`'s ±1 value.
- `ComputeP` stashes each solved column in `data_A_`-keyed map `P_` using the
  *column index* as the key, with no consideration of the row's sign.

Upstream is safe because production `IndexSchurData` is built only from 0/1
flag arrays (`SetData_Flag`) or list+single-sign (`SetData_List`), so every
stored value is `+1` and column indices are unique. Pounce's `from_parts`
and `set_from_*` validate signs ∈ {+1, −1} (rejecting 0) but **allow −1 and
allow duplicate columns**, so the two latent bugs become reachable.

**Bug 1 — B-row sign dropped.** `schur_matrix` did
`let (b_idx_vec, _facs) = b.multiplying_row(i)?;` then
`dense_schur[j*n_b + i] = -p_col[b_col];`, discarding `_facs[0]` (the B row's
±1). For a B row carrying −1 the Schur entry came out with the wrong sign.
**Fix:** bind the factor and apply it —
`dense_schur[j*n_b + i] = -b_facs[0] * p_col[b_col];`.

**Bug 2 — duplicate-column cache conflation.** The stored P column bakes in
A's sign (`rhs[c_us] = signs[i]; solve → K⁻¹(sign·e_col)`), but the cache was
keyed by `col` alone: `if self.p_cols.contains_key(&col) { continue }` /
`self.p_cols.insert(col, p_col)`. So when A had the same column twice with
*opposite* signs, the second row saw the key already present and silently
reused the first row's (wrong-signed) column. **Fix:** key the cache by
`(col, sign)`; `schur_matrix` looks each A column up by the same
`(a_col, a_signs[j])` key. The public `p_columns()` accessor return type
changes `HashMap<Index, Vec<Number>>` → `HashMap<(Index, Index), Vec<Number>>`
(only the in-crate + integration test suites consume it; updated the three
`.get(&col)` callsites in `p_calculator.rs` tests and the three in
`tests/adapter_trait_pipeline.rs` to `(col, sign)` keys).

**Contract preserved.** `tests/adapter_trait_pipeline.rs::adapter_compute_p_respects_negative_signs`
asserts the signed-storage contract (`p_pos[i] == -p_neg[i]` for the same
column built once with +1 and once with −1). The fix keeps signed storage —
it only stops two opposite-sign rows *in the same A* from aliasing — so that
test still passes unchanged (apart from the tuple-key lookup).

**Tests** (`p_calculator::tests`):

- `schur_matrix_honors_negative_b_sign` — K = 3×3 tridiag(−1,2,−1), A picks
  col 0 (+1) so `P[:,0] = K⁻¹e_0 = (¾, ½, ¼)`, B picks col 1 with sign −1.
  Correct entry `S[0,0] = −b_sign·P[1,0] = −(−1)·½ = +½`; the buggy
  sign-dropping path yields −½.
- `compute_p_distinguishes_same_column_opposite_signs` — same K, A = col 1
  selected twice with +1 then −1. Asserts both `(1, +1)` and `(1, −1)` are
  cached and are exact negations (`K⁻¹e_1 = (½, 1, ½)`); the buggy col-only
  cache stores just one entry, so the `(1, −1)` lookup is `None`.

**Fail-first** confirmed by reverting each bug *independently* (so each new
test's failure is attributable to its own fix): dropping `b_facs[0]` fails
only `schur_matrix_honors_negative_b_sign` (−½ ≠ +½); collapsing the cache
key to column-only fails only `compute_p_distinguishes_same_column_opposite_signs`
(`(1, −1)` absent). Restored both; full pounce-sensitivity suite green
(47 lib + 6 adapter_trait_pipeline + the rest of the integration suites).
`cargo fmt -p pounce-sensitivity` / `cargo clippy -p pounce-sensitivity
--all-targets -- -D clippy::correctness -D clippy::suspicious` clean.

## L18 detail

**Issue (code-review-2026-06.md L18).** Restoration inner solver pins
tolerances at `(1e-8, 1e-6, 15, 3000, 3000)` regardless of the user's outer
`tol` (`resto_inner_solver.rs:251`), and `is_square_problem = n == m_eq`
ignores inequalities (line 226), unlike IPOPT's `IsSquareProblem`.

Two separate claims; verified independently.

### Claim 1 — hardcoded resto tolerances (REAL bug, fixed)

`run_inner_resto` built the restoration sub-solve's convergence check as

```rust
RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000)
```

The first three args are `tol` / `acceptable_tol` / `acceptable_iter` for the
inner IPM's `OptErrorConvCheck`. As literals, they ignored whatever `tol` the
user requested: a `tol=1e-3` run still drove the restoration sub-NLP to `1e-8`
stationarity (wasting inner iterations), and a `tol=1e-10` run exited the
sub-solve too early at `1e-8`.

**Upstream behavior.** `IpRestoMinC_1Nrm.cpp` clones the outer `OptionsList`
into `actual_resto_options` and only *overrides* `required_infeasibility_reduction`
(and a few `resto.`-prefixed knobs); it never overrides `tol`, so the
restoration IPM inherits the outer `tol` / `acceptable_tol` / `acceptable_iter`.

**Plumbing check.** The inner builder already carries the user's values:
`IpoptApplication::algorithm_builder_from_options` does
`builder.conv_check.tol = read_num("tol")`,
`builder.conv_check.acceptable_tol = read_num("acceptable_tol")`,
`builder.conv_check.acceptable_iter = read_int("acceptable_iter")`
(`application.rs` ~1627/1648/1663), and every embedder
(`pounce-cli/src/main.rs:265`, `pounce-py/src/problem.rs:661`,
`pounce-cinterface/src/{solver.rs:194,lib.rs:585}`) passes
`app.algorithm_builder_from_options()` as the `inner_alg_builder` to
`make_default_restoration_factory_provider`. So the user's tol was sitting on
`inner_alg_builder.conv_check` the whole time — just unread.

**Fix.** Extracted

```rust
fn build_resto_conv_check_adapter(conv: &ConvCheckOptions) -> RestoConvCheckAdapter {
    RestoConvCheckAdapter::new(
        conv.tol, conv.acceptable_tol, conv.acceptable_iter,
        RESTO_INNER_MAX_ITERS, RESTO_MAX_SUCCESSIVE_ITERS,
    )
}
```

with the two iteration budgets kept as named consts (`= 3000`), since those
mirror `IpRestoConvCheck.cpp:137,144` `maximum_iters` / `maximum_resto_iters`
— resto-phase-specific caps, *not* the user's outer `max_iter`. The callsite
now reads `&inner_alg_builder.conv_check`. Added read-only accessors
`RestoConvCheckAdapter::{inner_tol, inner_acceptable_tol, inner_acceptable_iter}`
so the mapping is unit-testable.

### Claim 2 — `is_square_problem` ignoring inequalities (NOT a bug)

The code is `let is_square_problem = n_orig == m_eq;`. The review says this
diverges from IPOPT's `IsSquareProblem`. It does not. Fetched the upstream
definition via the GitHub raw API
(`src/Algorithm/IpIpoptCalculatedQuantities.cpp:3732-3735`):

```cpp
bool IpoptCalculatedQuantities::IsSquareProblem() const
{
   return (ip_data_->curr()->x()->Dim() == ip_data_->curr()->y_c()->Dim());
}
```

`x()->Dim()` is the number of (original) variables and `y_c()->Dim()` is the
number of *equality* multipliers, so upstream is exactly `n == m_eq` and also
carries no inequality term. pounce matches upstream; left unchanged.

### Test + fail-first

`resto_inner_solver::tests::resto_conv_check_adapter_inherits_user_tolerances`
builds a `ConvCheckOptions { tol: 1e-3, acceptable_tol: 1e-2, acceptable_iter: 7 }`
and asserts `build_resto_conv_check_adapter` produces an adapter reporting
those values. Fail-first confirmed by reverting the helper body to the old
`RestoConvCheckAdapter::new(1e-8, 1e-6, 15, …)` literals: the test fails with
`1e-8 != 1e-3` ("outer tol must propagate"). Restored; full pounce-restoration
suite green (106 lib + integration). `cargo fmt -p pounce-restoration` /
`cargo clippy -p pounce-restoration --all-targets -- -D clippy::correctness
-D clippy::suspicious` clean.

## L19 detail

- **Location**: `crates/pounce-restoration/src/min_c_1nrm.rs`,
  `MinC1NormRestoration::perform_restoration` (the `recover` back-half),
  least-square-multiplier block at ~`397-427`.
- **Bug**: when `constr_mult_reset_threshold > 0` and the problem is
  non-square (`y_c.dim() != x.dim()`) with `total_eq_dim > 0`, the block
  repointed `data.curr` at the recovered trial container so
  `EqMultCalculator::calculate_y_eq` would evaluate the gradient/Jacobians
  at the recovered iterate (pounce's stand-in for upstream
  `CopyTrialToCurrent`), but it never restored `curr`. So this option path
  returned from `recover` with `curr == recovered`, whereas the default
  (`threshold == 0`) path leaves `curr` untouched.
- **Upstream**: `DefaultIterateInitializer::least_square_mults`
  (`IpDefaultIterateInitializer.cpp`) does `ip_data.CopyTrialToCurrent()`
  then ends with `set_trial(iterates); AcceptTrialPoint()`, so upstream's
  `curr` ends as the recovered iterate on *every* path. pounce factors that
  final promotion out to the caller: `IpoptAlgorithm`'s
  `RestorationOutcome::Recovered` arm (`ipopt_alg.rs:1388-1420`) runs
  `adjust_variable_bounds_for_small_slacks()` then `accept_trial_point()`
  (`curr ← trial`). So in pounce the contract for `recover` is to stage the
  recovered point on `data.trial` and leave `curr` as found; the temporary
  `curr = recovered` is an implementation detail of the LSM evaluation that
  must be undone.
- **Observability**: none at the solver-output level. The caller overwrites
  `curr` via `accept_trial_point` immediately after `Recovered`, and the
  only code reading `curr` in between (`adjust_variable_bounds_for_small_slacks`
  → `cq.adjusted_trial_bounds()`) reads `trial`, not `curr`. The divergence
  is a latent maintainability trap, not a numeric defect — so it is
  verified at the `recover` boundary directly rather than via a solve.
- **Fix**: save `let saved_curr = data.borrow().curr.clone();` before
  `data.borrow_mut().curr = Some(recovered);`, and after the LSM step
  restore `data.borrow_mut().curr = saved_curr;`. Both option-setting paths
  now leave `data.curr` identical on return. Comment block explains why the
  restore is required (deferred promotion to the caller).
- **Tests** (`min_c_1nrm::tests`): a direct-call fixture exercises the real
  `perform_restoration`:
  - `MockNlp` — 2 vars, 1 equality, 1 inequality (mirrors the
    `pounce_algorithm::ipopt_cq` test mock). Non-square (`y_c.dim() == 1 !=
    x.dim() == 2`) so the LSM branch is reachable. Only the bound/expansion
    accessors the slack computations touch are real; gradients/Jacobians/
    Hessian `unimplemented!()` (never reached).
  - `StubEqMult` — writes fixed sub-threshold multipliers and returns
    success without touching `cq`/`aug_solver`, so the LSM branch runs
    without a working augmented-system solve. `UnusedAug` panics if invoked.
  - The inner-solver hook returns a synthetic `RestoSolveResult` with
    `trial_x = [10, 20]` ≠ `curr.x = [2, 3]`, so a leaked `curr` is visible.
  - `recover_restores_curr_on_constr_mult_reset_path` (`threshold = 10`):
    asserts `curr.x == [2, 3]` after the call.
  - `recover_leaves_curr_unchanged_on_default_path` (`threshold = 0`, LSM
    branch skipped): pins the invariant the threshold>0 path must match.
- **Fail-first**: replacing the restore with `let _ = &saved_curr;` makes
  `recover_restores_curr_on_constr_mult_reset_path` fail with
  `left: [10.0, 20.0]  right: [2.0, 3.0]` ("recover must leave data.curr
  unchanged…"), while the default-path test stays green — exactly the
  between-options divergence. Restored; full pounce-restoration suite green
  (108 lib + integration). `cargo fmt -p pounce-restoration` /
  `cargo clippy -p pounce-restoration --all-targets -- -D clippy::correctness
  -D clippy::suspicious` clean.

## L20 detail

**Issue (L20).** `IndexSchurData::set_from_flags` returns the wrong error
variant and leaves partially-populated state on invalid flags
(`schur_data.rs:217`); a retry appends duplicates.

**Verification.** Pounce port of Ipopt's `SensIndexSchurData::SetSchurDataFromVector`
(`SensIndexSchurData.cpp:51-78`). `set_from_flags(flags, v)` builds the Schur
selection set: for each `flags[i] == 1` it pushes column index `i` (with sign
`+1`/`-1` from `v`) into `idx`/`val`. The contract is that flag entries are
`0` (skip) or `1` (select). The pre-fix loop handled the out-of-range case
inside the match as `_ => return Err(SchurDataError::AlreadyInitialized)`:

  - **Wrong variant.** `AlreadyInitialized` names "set_* called twice on the
    same instance" — nothing to do with a bad flag value. A caller inspecting
    the error to decide whether to retry would be misdirected.
  - **Partial mutation + non-atomic retry.** The check fired *mid-loop*, after
    any earlier `f == 1` entries had been pushed, and the early return left
    `self.initialized == false`. A caller that caught the error, fixed its
    flag vector, and called again would re-enter the push loop on top of the
    leftover `idx`/`val`, appending duplicate rows. Confirmed by running a
    fixture: flags `[1,0,5]` then retry `[1,0,1]` produced
    `col_indices() == [0,3,0,2]` instead of `[0,2]`.

**Fix** (`crates/pounce-sensitivity/src/schur_data.rs`). Added a dedicated
`SchurDataError::InvalidFlag` variant (doc cites the upstream line range and
notes it is distinct from `AlreadyInitialized` and leaves the instance
untouched). Rewrote `set_from_flags` to validate the whole array before any
mutation:

```rust
if flags.iter().any(|&f| f != 0 && f != 1) {
    return Err(SchurDataError::InvalidFlag);
}
let w: Index = if v > 0.0 { 1 } else { -1 };
for (i, &f) in flags.iter().enumerate() {
    if f == 1 {
        self.idx.push(i as Index);
        self.val.push(w);
    }
}
self.initialized = true;
Ok(())
```

This makes the operation atomic — a rejected call leaves `idx`/`val`/`initialized`
exactly as found, so a retry starts clean. Also corrected the
`AlreadyInitialized` doc (dropped the stale "or flags contained values other
than 0/1" clause) and updated the trait-level `set_from_flags` doc to mention
both error variants and the unchanged-on-error guarantee.

**Tests** (`schur_data::tests`):
- `set_from_flags_rejects_invalid_flag_with_distinct_variant` — flags
  `[1,0,2,1]` → `Err(SchurDataError::InvalidFlag)` (asserts the specific
  variant, not just `is_err()`).
- `set_from_flags_invalid_flag_leaves_instance_pristine_for_retry` — flags
  `[1,0,5]` → `InvalidFlag`; asserts `!is_initialized()`, `nrows() == 0`,
  empty `col_indices()`; then retry `[1,0,1]` yields `col_indices() == [0,2]`
  with no duplicate append.

**Fail-first confirmed.** Against the pre-fix code both new tests fail: the old
loop returns `Err(AlreadyInitialized)` for the invalid input, and (with the
early return removed to model the partial-state path) the retry double-appends
to `[0,3,0,2]`. Restored; full pounce-sensitivity suite green.
`cargo fmt -p pounce-sensitivity` / `cargo clippy -p pounce-sensitivity
--all-targets -- -D clippy::correctness -D clippy::suspicious` clean.

## L21 detail

**Issue (L21).** l1penalty `new()` ignores `get_starting_point`/`eval_g`
failures (`wrapper.rs:179-191`); slack seeds silently computed from zero data.

**Verification.** `L1PenaltyBarrierTnlp::new` (pounce port of ripopt's
`L1PenaltyBarrierNlp`) wraps an inner TNLP and seeds the ℓ₁-penalty slack pairs
`(p_k, n_k)` from the constraint violation `r_i = c_i(x_0) − g_i` at the user's
start point. Its own doc-comment states the contract: "Calls `inner.get_nlp_info`,
`get_bounds_info`, `get_starting_point`, and `eval_g` … If any of these fail,
returns `None` — the caller should not enable the wrapper for that TNLP."

`get_nlp_info` (`?`) and `get_bounds_info` (`if !ok { return None }`) were
honored, but the two seed calls were not:

```rust
let _ = inner.borrow_mut().get_starting_point(StartingPoint { … });
let mut g0 = vec![0.0; m];
if m > 0 {
    let _ = inner.borrow_mut().eval_g(&x0, true, &mut g0);
}
```

Both methods return `bool` (`pounce-nlp/src/tnlp.rs:175,184`). With the returns
discarded, a TNLP whose `get_starting_point` fails leaves `x0` at its
`vec![0.0; n]` initialization, and one whose `eval_g` fails leaves `g0` at
zero — so the violation `r_i` (and hence every `(p_k, n_k)` seed) is computed
from fabricated zero data instead of the wrap being rejected. Silent: the
wrapper would be returned as `Some(_)` and proceed to an inner solve with
garbage slack seeds.

**Fix** (`crates/pounce-l1penalty/src/wrapper.rs`). Honor the contract:

```rust
let sp_ok = inner.borrow_mut().get_starting_point(StartingPoint { … });
if !sp_ok {
    return None;
}
let mut g0 = vec![0.0; m];
if m > 0 && !inner.borrow_mut().eval_g(&x0, true, &mut g0) {
    return None;
}
```

**Tests** (`wrapper::tests`): a `SeedFails` mock mirrors `EqOnly`
(2 vars / 1 equality row) but carries `fail_starting_point` / `fail_eval_g`
flags so the chosen seed callback returns `false`:
- `new_returns_none_when_starting_point_fails` — `fail_starting_point` →
  `new(..)` is `None`.
- `new_returns_none_when_eval_g_fails` — `fail_eval_g` → `new(..)` is `None`.
- `new_succeeds_when_seed_callbacks_succeed` — control, both flags off →
  `new(..)` is `Some`, so the `None` results are attributable to the failures.

**Fail-first confirmed.** Reverting the two callbacks to `let _ = …` makes both
`*_returns_none_*` tests fail (`new` returns `Some` with zero-seeded slacks)
while the control stays green. Restored; full pounce-l1penalty suite green
(14 lib). `cargo fmt -p pounce-l1penalty` / `cargo clippy -p pounce-l1penalty
--all-targets -- -D clippy::correctness -D clippy::suspicious` clean.

## L22 detail

**Issue (L22).** Error hint names a nonexistent flag — `main.rs` suggests
`--solve-report`; the actual flag is `--json-output`.

**Verification.** `run_cite` (`crates/pounce-cli/src/main.rs`) handles
`pounce --cite <path>`. When `<path>` has a `.nl` extension and fails to parse
as a `SolveReport` JSON, it prints a help hint assuming the user passed the
model file by mistake:

```
pounce: --cite expects a solve-report JSON, not a model file. Run
`pounce <model>.nl --solve-report report.json` first, then
`pounce --cite report.json` …
```

But `--solve-report` is not a CLI flag. The arg parser in
`crates/pounce-cli/src/cli.rs` accepts `--json-output <path>` (line 520) as the
flag that writes the `pounce.solve-report/v1` JSON; a grep for `--solve-report`
across `pounce-cli/src` finds it only in this hint string. A user following the
hint verbatim (`pounce model.nl --solve-report report.json`) gets an
unknown-argument error, so the hint actively misleads.

**Fix** (`crates/pounce-cli/src/main.rs`). Changed the hint (and the comment
above it) to name the real flag:

```
Run `pounce {} --json-output report.json` first, then `pounce --cite report.json`
```

No behavior change beyond the user-facing string; the report path itself is
unaffected.

**Test** (`crates/pounce-cli/tests/cite_hint_flag.rs`, new). Hermetic
integration test via `CARGO_BIN_EXE_pounce`: writes a temp file with a `.nl`
extension and non-JSON contents, runs `pounce --cite <file>.nl`, and asserts the
stderr hint **contains** `--json-output` and **does not contain**
`--solve-report`.

**Fail-first confirmed.** Reverting the hint to `--solve-report report.json`
makes the test fail (stderr no longer contains `--json-output`). Restored.
`cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --all-targets
-- -D clippy::correctness -D clippy::suspicious` clean.

## L23 detail

**Issue (L23).** After a failed MC64 scaling retry, stats reflect the retry but
status reverts to the original verdict (`main.rs:883-899, 902`).

**Verification.** The `feral_infeasibility_scaling_retry` guard (on by default)
handles discs-class hypersensitivity: on a local-infeasibility verdict under a
non-MC64 effective scaling, it re-solves once with MC64 and promotes only if
MC64 succeeds. The control flow was:

```rust
let mut status = loop { app.optimize_tnlp(...) /* … */ };   // original solve
if scaling_retry_enabled && … && status == InfeasibleProblemDetected {
    app.options_mut().read_from_str("feral_scaling mc64\n", true);
    let retry_status = app.optimize_tnlp(...);              // SECOND solve
    if matches!(retry_status, SolveSucceeded | SolvedToAcceptableLevel) {
        status = retry_status;                              // promote
    } else {
        status = InfeasibleProblemDetected;                 // revert verdict
    }
}
let solve_stats = app.statistics();                         // <-- BUG: read here
```

`app.statistics()` returns the stats of the *most recent* `optimize_tnlp`
(`application.rs:428` clones `self.statistics`, which the retry overwrote). So
on the non-promoting branch `status` is the original verdict but `solve_stats`
is the failed retry's — the console summary and the JSON solve report then show
`InfeasibleProblemDetected` paired with the MC64 retry's iteration count and
objective, two different solves' data stitched together. (On the promote branch
the stats happened to match the verdict, so the bug only bites the failure
path — exactly the path the guard exists to handle.)

**Fix** (`crates/pounce-cli/src/main.rs`). Snapshot the verdict-bearing solve's
stats *before* the retry, and decide status + stats together:

- After the solve loop: `let mut solve_stats = app.statistics();`.
- After the retry: capture `let retry_stats = app.statistics();` and set
  `(status, solve_stats) = resolve_scaling_retry_outcome(retry_status,
  solve_stats, retry_stats);`.
- Removed the later `let solve_stats = app.statistics();` re-read.

Two new pure helpers make the decision testable:

```rust
fn scaling_retry_promoted(retry_status: ApplicationReturnStatus) -> bool {
    matches!(retry_status, SolveSucceeded | SolvedToAcceptableLevel)
}

fn resolve_scaling_retry_outcome(
    retry_status: ApplicationReturnStatus,
    original_stats: SolveStatistics,
    retry_stats: SolveStatistics,
) -> (ApplicationReturnStatus, SolveStatistics) {
    if scaling_retry_promoted(retry_status) {
        (retry_status, retry_stats)
    } else {
        (InfeasibleProblemDetected, original_stats)
    }
}
```

Now status and statistics always describe the same solve.

**Tests** (`main.rs` `scaling_retry_tests`, unit). An end-to-end reproduction
would need a problem that is infeasible under the default scaling *and* under
MC64, plus a way to distinguish the two solves' stats — fragile. Instead the
decision logic is extracted and tested directly with synthetic
`SolveStatistics` (original `iteration_count = 7`, retry `= 42`):
- `failed_retry_keeps_original_status_and_stats` — for each non-promoting retry
  verdict (`InfeasibleProblemDetected`, `MaximumIterationsExceeded`,
  `RestorationFailed`): status stays `InfeasibleProblemDetected` and stats stay
  the original (`iteration_count == 7`, `final_objective == 7.0`).
- `promoted_retry_adopts_retry_status_and_stats` — for `SolveSucceeded` /
  `SolvedToAcceptableLevel`: status becomes the retry's and stats become the
  retry's (`42`).

**Fail-first confirmed.** Modeling the original leak (the `else` branch returns
`retry_stats` instead of `original_stats`) fails
`failed_retry_keeps_original_status_and_stats` ("stats must stay the original
solve's, not the failed retry's", `iteration_count` 42 ≠ 7) while the promote
test stays green. Restored; full pounce-cli bin tests green.
`cargo fmt -p pounce-cli` / `cargo clippy -p pounce-cli --bin pounce
-- -D clippy::correctness -D clippy::suspicious` clean.

## L24 detail

**Issue (L24).** The `.nl` is fully re-parsed for classification
(`main.rs:456-461`) — doubles parse time/peak memory on large models; the error
case silently falls back to NLP.

**Verification.** In `run` (`crates/pounce-cli/src/main.rs`) the `.nl` is parsed
once at the "Load the problem" block (`nl_reader::read_nl_file`, then
`NlTnlp::new(prob)` *moves* the `NlProblem`). The LP/QP dispatch block then ran
a **second** full parse purely to classify:

```rust
let (class, reparsed) = match &args.problem {
    ProblemSource::NlFile(path) => match nl_reader::read_nl_file(path) {
        Ok(prob) => (classify_problem(&prob), Some(prob)),
        Err(_) => (ProblemClass::Nlp, None),   // <- silent NLP fallback
    },
    ProblemSource::Builtin(_) => (ProblemClass::Nlp, None),
};
```

Two costs: (1) every `.nl` solve parses the file twice — on a large model that
doubles parse wall-time and the transient peak memory of holding a second
`NlProblem`; (2) the `Err(_)` arm threw the parse error away and defaulted to
`ProblemClass::Nlp`, so a re-read failure would silently mis-route rather than
report.

**Fix** (`crates/pounce-cli/src/main.rs`).
- Capture the class from the *first* parse: a new outer
  `let mut nl_class: Option<ProblemClass> = None;`, set to
  `Some(pounce_cli::dispatch::classify_problem(&prob))` immediately before
  `NlTnlp::new(prob)` consumes `prob`. `classify_problem` is a pure read-only
  pass over the parsed problem (`dispatch.rs:174`), far cheaper than a parse.
- The dispatch block now reads `class` from `nl_class` (builtins → `Nlp`); the
  redundant `read_nl_file` is gone.
- The convex solvers (`run_convex_qp` / `run_convex_socp`) need an owned
  `NlProblem`, and the first one was moved into `NlTnlp`. So re-parse **once,
  lazily, only inside the convex dispatch branch** (`LpIpm`/`QpIpm`/`SocpIpm`),
  which only `.nl` inputs ever reach. A failure there now surfaces
  (`eprintln!` + `ExitCode::from(2)`) instead of falling through to NLP.

Net effect: the common general-NLP `.nl` solve now parses the file exactly once
(was twice); only the convex LP/QP/SOCP route — typically small problems —
re-parses, and it no longer swallows a parse error.

**Tests/verification.** The performance claim ("parsed once, not twice") has no
clean deterministic fail-first seam without instrumenting `read_nl_file`, and
the re-parse error path needs a first-succeeds/second-fails race — both are
verified by code reading. The behavioral correctness of the refactor (the
captured `nl_class` must still drive dispatch) is covered end-to-end by the
existing suites, all green after the change:
- `qp_dispatch_end_to_end::auto_routes_convex_qp_to_pounce_convex` — `auto` on
  `convex_qp.nl` classifies it as a convex QP and routes to pounce-convex.
- `qp_dispatch_end_to_end::nlp_path_still_solves_same_file`,
  `forced_qp_ipm_solves`, `forced_qp_active_set_solves_convex_qp`, etc.
- `dispatch_routing` (5 tests) — forced-mismatch rejection, `auto`/`nlp`
  no-regression. 21 dispatch/QP tests total pass.

**Fail-first confirmed.** Forcing the capture off (`nl_class = None`, so `class`
defaults to `Nlp`) makes `auto_routes_convex_qp_to_pounce_convex` fail — the
convex QP misroutes through the NLP path and stdout no longer contains
`pounce-convex`. This proves the dispatch now sources its class from the
first-parse capture. Restored; suites green. `cargo fmt -p pounce-cli` /
`cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness
-D clippy::suspicious` clean.

## L25 detail

**Issue (verbatim).** "Failed `.sol` write exits 0 on the NLP path
(`main.rs:1076-1080`) but 2 on the convex path (`main.rs:1287`); `-AMPL`
callers see a clean exit with a stale/missing `.sol`."

**Status: already fixed by H4 — no new fix in this iteration.**

The `.sol`-write-failure exit-code asymmetry L25 describes was removed by the
earlier High-priority issue **H4** (commit `ce49710`,
`fix(cli): honor -AMPL exit-code contract on convex LP/QP/SOCP paths`). At the
time L25 was authored, `run_convex_qp`/`run_convex_socp` early-returned a
distinct `exit 2` on a `.sol`-write failure while the NLP path only logged.
H4 unified all three paths.

**Current state (verified by reading the code at HEAD):**

- NLP path (`main.rs:1169-1172`):
  `match write_sol_file(...) { Ok => "wrote", Err(e) => "failed to write" }`
  — logs only, no early return.
- Convex LP/QP path (`main.rs:1430-1435`): `if let Err(e) = write_sol_file(...)
  { eprintln!("pounce: failed to write …") }` with an explicit comment:
  "Log a `.sol` write failure but do not early-return a distinct exit code: the
  NLP path … only logs, and under `-AMPL` the final exit must still follow the
  solve-outcome contract."
- Convex SOCP path (`main.rs:1571-1576`): identical log-and-continue.

A `grep` for `.sol`-write sites finds exactly these three; none returns
`ExitCode::from(2)` on a write error. The convex paths' terminal exit now flows
through `convex_exit_code(ok, ampl)` (0 when `ok || ampl`), the direct mirror of
the NLP path's `_ if args.ampl => ExitCode::SUCCESS`.

**Why no separate test was added.** H4 already shipped
`ampl_mode_honors_exit_code_contract_on_infeasible_convex_qp`, which runs the
infeasible-QP fixture under `-AMPL` (asserts exit 0 with the verdict, srn 200,
in the `.sol`) and plain (asserts non-zero) — this guards the exact `-AMPL`
contract L25 cared about. A test that forces a `.sol`-*write* failure
specifically (e.g. an unwritable path/dir) would be brittle and
platform-dependent (running as root, filesystem permission semantics), and the
failure mode L25 flagged — a path that exits 2 on write failure — no longer
exists anywhere in the binary, so there is nothing left to regress against on
that axis.

## L26 detail

**Issue (verbatim).** "Summary block prints identical numbers in the
'(scaled)/(unscaled)' columns and hardcodes variable bound violation 0.0
(`print.rs:368-401`); inequality tally comment/code mismatch (`print.rs:87-89`)
makes the breakdown not sum to the total."

Three distinct sub-issues; one is a clean correctness bug (fixed + tested), the
other two are display symptoms of a missing solver statistic (documented).

### (c) Inequality bound-type breakdown didn't sum to the total — FIXED

In `collect_stats` (`print.rs`), each non-equality row is bucketed by which
finite bounds it has:

```
(true, true)   => ineq_both += 1,
(true, false)  => ineq_lower_only += 1,
(false, true)  => ineq_upper_only += 1,
(false, false) => {}            // <-- BUG: free row counted in n_ineq, no bucket
```

A row with `g_l = -inf, g_u = +inf` (a fully open `.nl` range row) is not an
equality, so `n_ineq += 1` fires, but the `(false, false)` arm did nothing —
leaving `ineq_lower_only + ineq_both + ineq_upper_only < n_ineq`. The printed
breakdown (three lines under "Total number of inequality constraints") then did
not account for every inequality row. The arm's own comment already declared the
intended behavior ("count it anyway under 'both' to keep the totals
consistent"), so the code simply wasn't doing what it claimed.

**Fix:** `(false, false) => ineq_both += 1;` (comment rewritten to match).

**Test:** `print.rs::inequality_tally_tests::free_inequality_row_keeps_breakdown_summing_to_total`
— a mock `TNLP` (2 free vars; rows: lower-only `[0,+inf)`, both `[0,1]`, free
`(-inf,+inf)`; no equalities) asserts:
- `n_eq == 0`, `n_ineq == 3`;
- `ineq_lower_only + ineq_both + ineq_upper_only == n_ineq` (the headline
  invariant);
- the free row lands in "both": `lower_only == 1`, `upper_only == 0`,
  `both == 2`.

**Fail-first:** restoring `(false, false) => {}` makes the sum `2 != 3` and the
test fails (observed: "breakdown (1 lower + 1 both + 0 upper) must sum to
n_ineq=3 ... left: 2, right: 3"). Restored; 160 pounce-cli lib tests green;
`cargo fmt -p pounce-cli` and
`cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious`
clean (the 3 remaining clippy warnings are pre-existing `clippy::style`, not in
the changed code).

### (a) (scaled)/(unscaled) columns print the same number — DOCUMENTED

`print_summary` (`print.rs:368-401`) emits an Ipopt-style two-column block, but
only the objective row passes two distinct values
(`stats.final_scaled_objective`, `stats.final_objective`). The other four rows
pass the same field twice:

```
row("Dual infeasibility......", stats.final_dual_inf,   stats.final_dual_inf);
row("Constraint violation....", stats.final_constr_viol, stats.final_constr_viol);
row("Complementarity.........", stats.final_compl,      stats.final_compl);
row("Overall NLP error.......", stats.final_kkt_error,  stats.final_kkt_error);
```

**Root cause is in the solver, not the printer.** `SolveStatistics`
(`pounce-nlp/src/solve_statistics.rs:51-66`) defines exactly one field for each
of these residuals, and they are filled once off the scaled-space `cq` cache
(`application.rs:1347-1361`: `curr_dual_infeasibility_max()`,
`curr_primal_infeasibility_max()`, the compl max, `curr_nlp_error()`). pounce
never computes the *unscaled* counterparts. The printer has only one value to
show; duplicating it across both columns is the visible symptom. A faithful fix
would compute and store unscaled residuals alongside the scaled ones in
`SolveStatistics` (touching `pounce-algorithm`'s finalize path) — a
solver-statistics change, out of scope for a CLI-print remediation and deferred.

### (b) Variable bound violation hardcoded `0.0` — DOCUMENTED

`row("Variable bound violation", 0.0, 0.0)` (`print.rs:391`). pounce's
`SolveStatistics` has no bound-violation field, and the print site is not handed
the final `x` or the variable bounds, so the value is not derivable here without
both plumbing the metric through the solver statistics and passing `x`/bounds
into `print_summary`. For a *converged* interior-point iterate the true value is
~0 by construction (slacks stay strictly positive), so the constant is
approximately right at optimum but genuinely uncomputed for early-terminated
solves. Deferred to the same solver-statistics issue as (a).

### Scope / safety note

A grep for the summary block's text across `crates/`, `pyomo-pounce/`, and
`python/` finds no parser — the two-column residual block is human-facing only,
so the (a)/(b) display imprecision is cosmetic, not a data-integrity problem.
The (c) fix is the only behavioral correctness defect in L26 and is now closed
with a regression test.

## L27 detail

**Issue (verbatim).** "Module doc says exit 0 only on `Solve_Succeeded`; code
also (reasonably) returns 0 for `SolvedToAcceptableLevel` (`main.rs:9-12` vs
1090-1092)."

**Root cause.** The NLP path's exit-code logic exits 0 for both
`SolveSucceeded` and `SolvedToAcceptableLevel`:

```
match status {
    ApplicationReturnStatus::SolveSucceeded
    | ApplicationReturnStatus::SolvedToAcceptableLevel => ExitCode::SUCCESS,
    _ if args.ampl => ExitCode::SUCCESS,   // AMPL contract
    _ => ExitCode::from(1),
}
```

But the module-level doc comment (`main.rs:9`) said only:
"Exit status: 0 on `Solve_Succeeded`, non-zero otherwise." Treating
reduced-accuracy convergence as success is correct and matches `minimize()`
(#119) and Ipopt; the doc just understated it.

**Fix.**
1. Module doc corrected to name both `Solve_Succeeded` and
   `Solved_To_Acceptable_Level` as the exit-0 success set (AMPL clause
   unchanged).
2. The inline `match` was extracted into a pure helper
   `nlp_exit_code(status, ampl) -> ExitCode` (mirrors `convex_exit_code(ok,
   ampl)`), composed from a small testable predicate
   `nlp_solve_succeeded(status) -> bool`. `ExitCode` implements no `PartialEq`,
   so the bool predicate is the unit-testable seam (same pattern as L23's
   `scaling_retry_promoted`). The duplicated AMPL-contract comment now lives
   once, on the helper.

**Tests** (`main.rs` `nlp_exit_code_tests`):
- `acceptable_level_counts_as_success` — both `SolveSucceeded` and
  `SolvedToAcceptableLevel` return true (the L27 crux).
- `non_convergent_statuses_are_not_success` — `InfeasibleProblemDetected`,
  `MaximumIterationsExceeded`, `RestorationFailed`, `DivergingIterates`,
  `MaximumCpuTimeExceeded`, `InternalError` all return false.

**Fail-first.** Removing `SolvedToAcceptableLevel` from `nlp_solve_succeeded`
makes `acceptable_level_counts_as_success` fail
(`assertion failed: nlp_solve_succeeded(A::SolvedToAcceptableLevel)`). Restored.

5 pounce-cli bin tests green; `cargo fmt -p pounce-cli` and
`cargo clippy -p pounce-cli --bin pounce -- -D clippy::correctness -D clippy::suspicious`
clean (3 remaining warnings are pre-existing `clippy::style`).

## L28 detail

**Issue (verbatim).** "`nl_hessian_program.rs` is dead in-tree but panics on
`Funcall`/min/max/transcendental ops (lines 456, 477, 594) — would fire on user
input if ever wired in."

**Verification.** `crates/pounce-cli/src/nl_hessian_program.rs` is a 1500-line
precompiled-symbolic-Hessian optimization module. It is declared
`pub mod nl_hessian_program;` (`lib.rs:19`) so it compiles, but a grep across
`crates/`, `pyomo-pounce/`, and `python/` finds no caller of
`HessianProgram::compile` or `::execute` — only the module declaration. It is
genuinely dead, latent until wired into the Hessian dispatch path.

`HessianProgram::compile` lowers a `Tape` into a flat op program across three
sweeps (forward, per-`j` forward-tangent, per-`j` reverse-over-tangent), plus
two pre-analyses (`reachable_to_output`, `depends_on_var`). Each match-dispatches
`TapeOp` and `panic!`ed on any opcode it can't lower. The supported set
(intersection across all sweeps — the forward sweep is the most restrictive) is:
`Const, Var, Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt, Exp, Log, Log10, Sin,
Cos`. Everything else — `Funcall` (AMPL external functions), `Tan`/`Atan`/`Acos`/
the hyperbolics/`Asin`…/`Atan2`, and `Cmp`/`And`/`Or`/`Not`/`Select`/`Min`/`Max`
— hit a `panic!`. The panic messages all end "use the Tape (build_with_externals)
path instead", showing the *intended* contract was a graceful fall-back to the
interpreter path, with the panic a placeholder stub.

**Fix.**
1. `compile(tape, hess_map) -> Option<Self>` (was `-> Self`).
2. Up-front guard: `if !tape.ops.iter().all(program_supports_op) { return None; }`.
   New free fn `program_supports_op(&TapeOp) -> bool` is the single source of
   truth for the supported set, documented as such.
3. All 10 `panic!`s → `unreachable!`. The guard rejects unsupported tapes before
   any sweep or helper runs, so these arms are statically unreachable from the
   only public entry; `unreachable!` now signals an internal guard/sweep
   inconsistency (e.g. a future opcode added to the guard but not a sweep) — a
   programmer error surfaced in tests — never user input.
4. Both early returns wrapped in `Some(...)`; the two in-tree test call sites
   (`assert_program_matches_tape`, `slots_layout_matches_design`) updated to
   `.expect(...)`.

**Test:** `nl_hessian_program::tests::unsupported_opcode_returns_none_instead_of_panicking`
— a `tan(x0)` tape (lowers to the unsupported `TapeOp::Tan`) makes `compile`
return `None`; a supported `mul(x0,x1)` tape still returns `Some` (so the guard
rejects only genuinely-unsupported ops, not everything).

**Fail-first:** replacing the guard condition with `if false` lets the tan tape
flow into the forward sweep, which hits the (now) `unreachable!` arm and panics:
`internal error: entered unreachable code: HessianProgram path does not yet
support tan/atan/acos, …` — the exact crash-on-user-input failure mode L28
describes. Restored.

161 pounce-cli lib tests green (was 160). `cargo fmt -p pounce-cli` and
`cargo clippy -p pounce-cli --lib -- -D clippy::correctness -D clippy::suspicious`
clean. (The lib emits pre-existing `clippy::restriction` `unwrap_used`/
`expect_used` style warnings — including the two test `.expect()`s added here —
which are not in the gated `correctness`/`suspicious` groups.)

**Note.** This does not wire the module into dispatch (out of scope); it only
makes the dead path safe to wire in later — `compile` returning `None` is now
the documented "fall back to the Tape interpreter" signal a future caller
composes with.

## L29 detail

**Issue (L29).** The `.nl` reverse/forward AD engine in `crates/pounce-nl/src/nl_tape.rs`
modeled the first-order derivative of `Pow(u, r)` w.r.t. its base `u` differently in
forward-tangent mode than in reverse mode:

- **Reverse** (`reverse_into`, line 343; free fn `reverse` analog at 2856):
  `if rv != 0.0 { adj[base] += a * rv * lv.powf(rv - 1.0); }` — the base-derivative
  term is taken whenever the exponent is nonzero, *including at base `lv == 0`*.
- **Forward-tangent** (`forward_tangent` 513, `hessian_directional` 736, `fwd_tan_step`
  2994): `if r != 0.0 && u != 0.0 { result += r * u.powf(r - 1.0) * du; }` — the extra
  `&& u != 0.0` silently dropped the term whenever the base was exactly 0.

Because the `.nl` default starting point seeds unbounded variables at 0, this is exactly
the point evaluated on iteration 0. For `f = x0^x1` at `x = (0, 1)`:
- reverse `df/dx0 = 1 * 0^0 = 1`,
- forward-tangent `df/dx0 = 0` (term dropped),

so a Jacobian-vector product disagreed with the gradient of the same function at the
default start. Constant exponents are lowered to Mul/Sqrt chains and so never hit this
branch, which narrowed the exposure to genuine variable-exponent `Pow` nodes — but those
do occur (e.g. `x^y` couplings), and when they do the inconsistency is silent.

**Fix.** Drop the spurious `&& u != 0.0` from all three forward-tangent sites so the guard
becomes `if r != 0.0` — byte-identical to the reverse guard's intent. `0.0.powf(r - 1.0)`
is well defined and equals what reverse already computes: `1` at `r = 1`, `0` for `r > 1`,
`±inf` for `0 < r < 1` (true infinite slope of e.g. `sqrt` at 0). The two arms now agree
everywhere the first-order derivative is defined.

**Why the Hessian base-0 sites are intentionally untouched.** The forward-over-reverse
second-order `Pow` sites (methods at ~898 and ~1208, free fn at ~3148) compute the
cross-partial `∂²(u^r)/∂u∂r`. At `u = 0` with a *variable* exponent this is genuinely
singular (`r·u^(r-1)·ln(u)` → `±inf`), not a removable bookkeeping gap. Those sites
already special-case integer `r ≥ 2` and base 0; there is no finite value on which the two
second-order arms could be made to "agree," so the only real inconsistency — the
first-order tangent — is the one fixed here.

**Test.** `nl_tape::tests::pow_forward_tangent_matches_reverse_gradient_at_base_zero`
builds `pow(var(0), var(1))`, asserts the tape still contains a real `TapeOp::Pow`
(guarding against future lowering that would skip the branch), evaluates at `[0.0, 1.0]`,
and asserts forward-tangent `dot[output] == reverse grad[0] == 1`.

**Fail-first.** Reverting any forward-tangent guard back to `r != 0.0 && u != 0.0` makes
the forward tangent return `0` while reverse stays `1`; the test fails with
`forward tangent df/dx0 = 0 must match reverse gradient 1 at base 0`. Restored after
confirming. Full crate suite: 80 lib + 1 integration tests green. `cargo fmt` / gated
`cargo clippy` (`-D clippy::correctness -D clippy::suspicious`) clean.
