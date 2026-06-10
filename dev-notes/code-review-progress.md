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
