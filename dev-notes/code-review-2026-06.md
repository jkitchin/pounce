# Whole-repo code review — 2026-06-10

A read-only review of the full workspace at commit `8d6e981` for correctness
errors, silent bugs, inconsistencies, and performance issues. Every crate was
read by a dedicated reviewer (pounce-algorithm; linalg/linsol/hsl/feral;
presolve; convex; nl/nlp/common; qp/restoration/l1penalty/sensitivity; cli;
python/pounce-py/pyomo-pounce; cinterface/studio/tools/scripts/workflows),
with cross-crate contracts checked at the call sites. No code was changed.

Findings are ordered by severity. File references are to this checkout.
Severity rubric: **Critical** = silently wrong answers on plausible input in
shipped paths; **High** = wrong results/panics/broken integrations in
realistic use; **Medium** = conditional correctness issues, contract
violations, significant waste; **Low** = edge cases, doc/code drift,
latent hazards.

---

## Critical

### C1. presolve: Phase-2 redundancy mask misaligned after Phase 0 drops a linear row — can silently delete binding constraints

`crates/pounce-presolve/src/lib.rs:530-534` vs `:641-654`. `linear_rows` is
built **filtered** by the post-Phase-0 mask (`row_kept_inner`), but the loop
mapping `find_redundant_rows`' verdict mask back to inner rows advances the
mask iterator for **every** linear row, kept or not. If Phase 0 dropped any
linear row (its own tests show it routinely does), every kept linear row after
it receives its *predecessor's* verdict — a binding constraint can be dropped
and reinstated at postsolve with λ=0, i.e. a silent wrong answer. Also fires
after the Phase-0 rollback path (lines 556-582) because `linear_rows` is not
rebuilt, and `n_dropped_rows` can double-count. Triggered whenever
`presolve_auxiliary=yes` (since `redundant_constraint_removal` defaults on
within enabled presolve). Mitigation: presolve as a whole is default-off.
Fix direction: iterate kept linear rows only.

### C2. presolve: Phase-0 block elimination assumes non-block variables in a block's rows are constants — four ways that assumption is false

`crates/pounce-presolve/src/auxiliary.rs` (orchestrator), with structural
roots in `dulmage_mendelsohn.rs`, `components.rs`, `incidence.rs`.
`solve_linear_block` (auxiliary.rs:548-552) folds any out-of-block column into
the RHS at `x_running[j]`, and the residual check (auxiliary.rs:427-448)
evaluates at the same point, so it cannot catch a wrong constant. Dropped rows
are removed from the IPM's problem while depending on variables the IPM is
still free to move:

- **(a) Rejected earlier block.** When an earlier block is rejected
  (`OutOfBounds`, `CouplingDisallowed`, divergence — auxiliary.rs:340-344,
  411-423 just `continue`), later blocks referencing its still-free columns
  are still solved and their rows dropped. No dependency tracking exists.
- **(b) Square rows adjacent to Over-part columns.** The DM Over BFS
  (`dulmage_mendelsohn.rs:104-124`) can leave a Square row adjacent to an Over
  column; `SquareComponents::of_square_part` ignores the non-square edge
  (`components.rs:139-143`), so a block is eliminated with a forever-free
  variable baked in at its probe value. Coupling classification
  (`coupling.rs:88-104`) checks only `block.cols`, never the rows' other
  columns.
- **(c) Trivially-fixed variables solved at probe value.** PR-13 excludes
  `x_l==x_u` variables from incidence (auxiliary.rs:148-156), but `x_running`
  is initialized from `x_probe` (auxiliary.rs:231) and never clamped to the
  fixed values.
- **(d) Probe-zero Jacobian entries dropped from incidence.**
  `EqualityIncidence::from_probe` skips `vals[k] == 0.0` unconditionally
  (`incidence.rs:116-121`); for a nonlinear row a derivative that is zero at
  the probe (e.g. ∂x²/∂x at x=0) is not structurally zero, so a "square"
  block can omit a real dependency.

All four produce a feasibly-looking presolve and a final solution that
silently violates the dropped equalities. (b)–(d) need no rejection to
trigger. Phase 0 is opt-in (`presolve_auxiliary`), which is the only reason
this is not catastrophic today. Fix direction: per-component dependency
invalidation plus a hard gate "every non-block column in a block row is
already fixed".

---

## High

### H1. qp: inertia-shift regularization silently discarded — unbounded/indefinite QPs reported `Optimal` with δ-dependent garbage

`crates/pounce-qp/src/solver.rs:104-156` and all call sites (238, 384, 441,
566, 682, 943, 1569). `factorize_with_inertia_control` returns the final
shift δ but every caller drops it; stationarity is declared from the
*shifted* system `H+δI` and never re-verified. δ can grow to ~1e16. Concrete
failure: `min gᵀx`, `H = 0`, no constraints → empty KKT is singular, δ=1e-8,
result `x = −g/δ` with `status: Optimal`. Corroborating: `QpStatus::Unbounded`
and `QpStatus::NumericalError` are declared in `error.rs` but never
constructed anywhere in the crate — unbounded detection simply does not
exist; the regularizer masks it.

### H2. sensitivity: pin-row mapping omits `full_g_to_c_block` — silently wrong sensitivities on any problem with inequality constraints

`crates/pounce-sensitivity/src/convenience.rs:278-286` and
`solver.rs:310-319, 354-363` compute the KKT row of a pinned constraint as
`n_x + n_s + i` from the user's 0-based `g(x)` index. But the `y_c` block
contains only equality rows; the CLI driver does the translation correctly
via `nlp.full_g_to_c_block(full_ci)` with an explicit error for inequality
rows (`crates/pounce-cli/src/sens.rs:354-374` — duplicated logic that
diverged). With any inequality preceding a pinned equality,
`parametric_step` / `compute_reduced_hessian` return plausible-looking but
wrong numbers with no error. Tests pass only because fixtures are
equality-only.

### H3. cli: `.sol`/JSON constraint duals written in internal c/d-split order, unscaled — wrong for interleaved or scaled problems

`crates/pounce-cli/src/main.rs:602-624`. The `on_converged` hook builds
lambda as raw `y_c` expanded then `y_d` expanded, but pounce-nlp's
`c_map`/`d_map` interleave in original `.nl` row order
(`crates/pounce-nlp/src/tnlp_adapter.rs:383-405`). The correct path exists —
`OrigIpoptNlp::pack_lambda_for_user` scatters via the maps **and** applies
`c_scale`/`d_scale`/`obj_scale_factor`
(`crates/pounce-nlp/src/orig_ipopt_nlp.rs:1187-1221`) — but the CLI hook does
none of it, and the correct backfill (main.rs:934-938) only wins when the
nominal capture is empty (active-set route). Consequences: permuted dual
block in `.sol`/JSON for any `.nl` with interleaved eq/ineq rows (AMPL/Pyomo
read duals positionally); duals off by scale factors whenever the default
`gradient-based` scaling fires; silent zero-substitution on failed downcasts
(main.rs:613, 622). No test covers dual ordering.

### H4. cli: convex LP/QP/SOCP dispatch ignores the `-AMPL` exit-code contract

`crates/pounce-cli/src/main.rs:1347-1352` (`run_convex_qp`) and `:1485-1489`
(`run_convex_socp`) never receive `args.ampl` and return exit 1 for
infeasible/unbounded/limit outcomes. The NLP path documents and implements
the convention (exit 0 whenever a `.sol` was produced, main.rs:1090-1105)
precisely because Pyomo raises `ApplicationError` and never parses the `.sol`
on non-zero exit. Default routing (`solver_selection=auto`) sends every
LP/convex-QP/QCQP `.nl` down these paths, so `pounce model.nl -AMPL` on an
infeasible LP breaks the Pyomo integration. Also inconsistent: failed `.sol`
write exits 2 here (main.rs:1287) but 0 on the NLP path.

### H5. nl: external-function errors detected on the wrong channel — failed evals silently return garbage

`crates/pounce-nl/src/nl_external.rs:534, :574, :592-599`. The AMPL `funcadd`
ABI signals failure by the library **assigning** `al->Errmsg` to its own
string; the caller must re-read the pointer. `ExternalLibrary::eval` instead
pre-points `al.errmsg` at a caller-owned zeroed buffer and only checks
`errmsg_buf[0] != 0` — a conforming library that does
`al->Errmsg = "T out of range";` leaves the buffer untouched and the error is
invisible. This defeats the NaN-poisoning design in
`nl_tape.rs::ext_eval_or_nan` (lines 154-168, written exactly so the line
search backs off on out-of-domain IDAES Helmholtz evals); the IPM silently
consumes wrong f/∇f/∇²f. Fix: after the call, compare `al.errmsg` against the
original buffer pointer and read whichever is set.

### H6. qp: `select_blocker` EXPAND branch can panic on valid input

`crates/pounce-qp/src/solver.rs:1430-1471`. `alpha_min_relaxed` starts at 1.0
and pass 2 admits only candidates with `r_relaxed ≤ alpha_min_relaxed + tol`.
If every blocker has `r < 1` but `r + τ/|a·p| > 1 + tol` (small `|a·p|` just
above `feas_tol` makes `τ/|a·p|` large), pass 2 selects nothing and
`best.expect("non-empty candidates above")` panics. Narrow but reachable on
near-degenerate data, and EXPAND is the default anti-cycling mode. Fix:
initialize `alpha_min_relaxed` from candidates, or fall back to strict-min.

### H7. convex: dual-infeasibility certificate uses componentwise `Gd ≤ 0` — false `DualInfeasible` possible on SOC/PSD problems

`crates/pounce-convex/src/ipm.rs:2176-2193` (`detect_infeasibility_with`),
reached from both the direct driver (ipm.rs:1397) and the symmetric HSDE
driver (`hsde.rs:235`). The cone-aware wrapper (ipm.rs:2140-2149) fixes only
the *primal* certificate (`z ∈ K*` via `cone.in_dual_cone`); the
*dual*-infeasibility branch still validates the recession direction
componentwise. The correct condition for `Gx ⪯_K h` is `−Gd ∈ K`: e.g.
`−Gd = (0.1, 0.5)` passes componentwise but is not in the SOC. The doc
comment ("a positive result is a genuine proof and a false positive is
impossible", ipm.rs:2103-2106) is violated. Fix: test `−Gd ∈ K` with the
cone that already flows into the function.

### H8. convex: non-symmetric HSDE driver validates Farkas multipliers with the orthant-only test — wrong in both directions for exp/power (and SOC) blocks

`crates/pounce-convex/src/hsde_nonsym.rs:840` calls the componentwise default
(`ipm.rs:2121-2132`). The dual exp cone requires `u ≤ 0` (`exp.rs:110-122`),
so genuine exp Farkas certificates are *rejected* (infeasible exp-cone
problems degrade to `IterationLimit`) while an all-nonnegative `z` not in
`K_exp*` can be *accepted*, yielding a false `PrimalInfeasible`. The comment
at ipm.rs:2129-2131 claiming the default is "exact for the non-symmetric
Farkas paths" is incorrect. The block cones all expose `in_dual_cone`; the
symmetric driver was already fixed for the primal certificate — divergence
between the two drivers.

### H9. convex: `presolve_conic` protects only `SecondOrder` rows — unsound reductions / wrong `Infeasible` for PSD/exp/power rows

`crates/pounce-convex/src/presolve.rs:388-404`: the protected-row mask covers
only `ConeSpec::SecondOrder`, yet `reduced_cones` (presolve.rs:1436-1445)
passes `Exponential`/`Power`/`Psd` through, implying they are supported. An
exp-cone row with empty `G` row and `h < 0` (legal — `K_exp` contains points
with negative first coordinate) is declared `PrimalInfeasible` by the
empty-row check (presolve.rs:558-566); "redundant by activity" cone rows
(presolve.rs:581-592) are dropped, destroying the 3-row exp block / `svec`
layout. Latent today (only tests call it, with Nonneg+SOC), but it is public
API. Fix: protect all non-`Nonneg` rows or reject non-SOC cones explicitly.

### H10. presolve: postsolve does not zero `z_l`/`z_u` at aux-fixed variables — reported duals violate stationarity

`crates/pounce-presolve/src/lib.rs:1021-1033`. Phase 0 fixes variables by
clamping `x_l = x_u = v`, so the IPM produces large bound multipliers there.
`recover_dropped_multipliers` (`reduction_frame.rs:205-255`) correctly solves
`grad_f − Jᵀλ = 0` under the documented assumption `z_l = z_u = 0` at those
variables, but `finalize_solution` forwards `sol.z_l`/`sol.z_u` unchanged —
the bound-multiplier contribution is double-counted against the recovered λ.
Fix is one loop zeroing `z_l/z_u` at `frame.fixed_vars`.

### H11. presolve: objective coupling classified from the gradient at a single probe point

`crates/pounce-presolve/src/lib.rs:445-449`, `auxiliary.rs:203`,
`coupling.rs:46-52`. Any variable with zero gradient *at the starting point*
is treated as objective-free. For `f = (x−x₀)²` started at `x₀` (a common
warm-start pattern), a block containing `x` is misclassified `PureEquality`
and eliminated under the default `Safe` policy; for nonlinear blocks with
multiple roots, Newton picks a root with no regard to the objective —
silently suboptimal. `get_variables_linearity` exists on the TNLP trait and
is unused here; a linearity tag would make this safe.

### H12. presolve: FBBT lacks both the Phase-0 row mask and any infeasibility handling

`crates/pounce-presolve/src/lib.rs:592-611`. `run_fbbt` is given all rows
including those Phase 0 dropped, over aux-clamped bounds — the exact
configuration the #53 review fixed for Phase 1 (filtered rows + rollback);
FBBT got neither, so propagation can spuriously flag infeasibility. Worse,
`fbbt_report.infeasibility_witness` is never inspected anywhere despite
`FbbtReport` documenting that on infeasibility "the variable bounds … are
undefined and must not be trusted" (`fbbt/orchestrator.rs:70-74, 84-86`) —
genuine FBBT infeasibility is silently swallowed instead of short-circuiting.

### H13. cinterface: `IpoptSolverSolve` silently discards all user options after the first solve

`crates/pounce-cinterface/src/solver.rs:206` (doc at 46-55). Each call does
`mem::replace(&mut info.problem.app, IpoptApplication::new())` and moves the
app into the session — nothing restores it. The doc claims restoration "via
the `app_template` field below", but no such field exists (grep-verified).
On the second `IpoptSolverSolve` of the same handle, every option set via
`AddIpopt{Str,Num,Int}Option` (linear solver, tolerances, scaling) is
silently replaced by defaults — including the `feral_config_from_options`
snapshot at line 191, read from the already-blanked app. Multi-solve is the
session API's design center; this is a silent wrong-result bug.

### H14. release: crates.io automation currently guaranteed to fail mid-batch (irreversible partial publish), invisible to the consistency guard

Root `Cargo.toml:82-88`: `feral = { git = ..., rev = "11fb4b9..." }` has no
`version =`, and `cargo publish` refuses deps without a registry version (the
comment itself admits "this git pin blocks the crates.io publish"). Publish
order (`scripts/publish-crates.sh:42-62`) is common, linalg, linsol, then
**feral** — so a `vX.Y.Z` tag today publishes 3 of 19 crates and hard-fails
at crate 4, shipping a split release that cannot be rolled back (crates.io
versions are immutable). Neither `scripts/check-release-consistency.sh`
(checks versions/membership/topo order, not dep publishability) nor any CI
job runs `cargo publish --dry-run`. CLAUDE.md and the workflow header present
tag-push publishing as safe.

### H15. python: `curve_fit` reports `success=False` for `Solved_To_Acceptable_Level`

`python/pounce/_curve_fit.py:712`: `success = int(info["status"]) == 0`. The
repo fixed exactly this class in `minimize` (gh #119 —
`_minimize.py:65` accepts `{0, 1}`), and the jax/torch paths accept both
(`jax/_path.py:48`, `torch/_path.py:36`). `curve_fit`, `curve_fit_streaming`,
and `curve_fit_minima` all route through this line, so a fit that stalls at
acceptable tolerance is reported failed with fully-populated `popt`/`pcov`;
callers gating on `result.success` discard valid fits. Also lacks the
`final_kkt_error` fallback `minimize` applies (`_minimize.py:524-529`).

---

## Medium

### Algorithm core (pounce-algorithm)

- **M1. Convergence gates use internally *scaled* residuals where upstream
  uses unscaled.** `src/conv_check/opt_error.rs:215-222` (and
  `current_is_acceptable_with_state`, 295-307) gate `dual_inf_tol` /
  `constr_viol_tol` / `compl_inf_tol` / all `acceptable_*_tol` on the scaled
  CQ accessors (`src/ipopt_cq.rs:950-962, 1041-1047`); upstream
  `IpOptErrorConvCheck.cpp` deliberately uses the unscaled quantities for
  these per-component gates. Since gradient-based constraint scaling is
  implemented and default, pounce can declare `Success` while the user-space
  violation exceeds `constr_viol_tol` by the inverse scale factor (and may
  refuse termination on up-scaled problems). The `curr_f` fed to
  `acceptable_obj_change_tol` is likewise scaled.
- **M2. `accept_trial_point` silently destroys `curr` when no trial is
  staged.** `src/ipopt_data.rs:203-205` has no guard;
  `src/ipopt_alg.rs:1121` calls it unconditionally. On the documented
  no-search-dir bookkeeping mode (`have_delta == false`), `curr` becomes
  `None` and the next iteration hits `unreachable!` in `ipopt_cq.rs:107-112`.
  Upstream asserts the trial is non-null.
- **M3. `LeastSquareMults` lacks the δ_c/δ_d inertia workaround its sibling
  needed.** `src/eq_mult/least_square.rs:106-135` solves the same W=0
  structure that `src/init/default.rs:163-174` perturbs with `1e-8`
  specifically because feral mis-reports inertia on structurally-zero
  (3,3)/(4,4) blocks; here `delta = 0.0`, so the LS solve can spuriously fail
  and silently fall back to `y = 0` — the iter-0 `inf_du` blow-up this step
  exists to prevent. Duplicate logic that diverged.

### Linear algebra (pounce-linalg/linsol)

- **M4. `symmetric_eigen` reports success even on non-convergence.**
  `crates/pounce-linalg/src/eigen.rs:81-153`: doc promises `false` when the
  Jacobi iteration runs out of sweeps, but after 50 sweeps the function falls
  through and returns `true` unconditionally. Callers genuinely branch on it
  (`pounce-convex/src/cones/psd.rs:108,145,163,231`, `sos.rs:615,672,717`),
  so a stalling matrix would feed unconverged eigenpairs into PSD projections
  and SOS decompositions instead of the error path. Latent (Jacobi nearly
  always converges) but trivially fixable.

### QP / restoration / sensitivity

- **M5. QP warm-start can return `Optimal` at an infeasible point; unmarked
  equality rows are never enforced.** `crates/pounce-qp/src/solver.rs:616-624,
  649-655, 831-833`: active-row residuals are frozen by the zero-RHS step
  system and never audited; equality rows left `Inactive` by the caller's
  working set are skipped by the ratio test (`continue` at 831-833) and can
  never enter. The doc says infeasible warm starts "may diverge or hit
  max_iter"; the actual failure is a silent `Optimal`.
- **M6. `SensSolve` swallows sensitivity-stage failures.**
  `crates/pounce-sensitivity/src/convenience.rs:380-395`: failures write
  `outbox.error`, which is `#[allow(dead_code)]` and never copied into
  `SensResult` — a failed `parametric_step` yields `dx: None` with
  `status: SolveSucceeded`, indistinguishable from "not requested".
- **M7. QPS parser doubles Hessian off-diagonals for `QMATRIX` files.**
  `crates/pounce-qp/src/qps.rs:132`: `QMATRIX` (full-matrix convention) is
  treated as `QUADOBJ` (triangle convention); after lower-triangle
  normalization both mirror entries survive and the evaluators sum all
  triplets.
- **M8. l1penalty: augmented `x` passed to inner `eval_jac_g`.**
  `crates/pounce-l1penalty/src/wrapper.rs:443-451` forwards the
  length-`n_orig + 2·m_eq` slice unchanged (the `eval_h` path truncates
  correctly at 480-481). Latent: inner TNLPs that check `x.len()` or iterate
  the slice misbehave.
- **M9. Silent zero-substitution on failed downcasts in restoration/
  sensitivity init paths.** `crates/pounce-restoration/src/init.rs:182-193,
  216-221, 235-240` (also `resto_inner_solver.rs:775-780`,
  `resto_resto.rs:234-239`, `pounce-sensitivity/src/solver.rs:380-388`,
  `convenience.rs:397-405`): `downcast_ref::<DenseVector>() … unwrap_or(vec![0.0; …])`
  replaces residuals with zeros with no diagnostic if a vector is ever a
  compound/homogeneous block.
- **M10. Schur-update QP path: O(m·nnz(A)) assembly per reset and no inertia
  re-check after working-set changes.** `crates/pounce-qp/src/schur.rs:163-189,
  227-233`; the doc claim of being "algorithmically identical to the
  refactor-per-iteration path" (`solver.rs:1090-1095`) does not hold for
  indefinite reduced Hessians after a drop.

### CLI

- **M11. QP extraction drops constraint terms folded into the nonlinear
  tree.** `crates/pounce-cli/src/qp_extract.rs:109-145` builds `A`/`G` from
  `con_linear` only, while the classifier deliberately admits rows whose
  nonlinear expression reduces to degree ≤ 1 (`dispatch.rs:198, 240`) and the
  SOCP extractor handles them (`qp_extract.rs:356-396`, `nl_lin` +
  `const_shift`). LP/QPs with linear/constant terms inside the nonlinear tree
  (cancelled quadratics, defined variables) get silently wrong constraints on
  the convex path.
- **M12. `DivergingIterates` mapped to AMPL code 401 ("limit") instead of the
  300-range ("unbounded").** `crates/pounce-solve-report/src/lib.rs:453`;
  upstream Ipopt's ASL driver maps it to 300, and the CLI's own convex path
  reports the same condition as 300 (main.rs:1260, 1410) — internal
  divergence.
- **M13. With `presolve yes` and dropped rows, the `.sol` dual block has the
  wrong length.** `crates/pounce-cli/src/main.rs:604-624, 934-938,
  1059-1080`: both lambda sources are in the reduced row space (counting
  wraps outside presolve), so the `.sol` carries `m_out` duals against a `.nl`
  with `m` constraints — AMPL/Pyomo readers reject or misalign. Mitigated by
  presolve defaulting off.
- **M14. Any `--minima` strategy knob silently switches the run into
  multistart mode.** `crates/pounce-cli/src/cli.rs:427-438, 554-581`:
  `--seed`, `--patience`, `--dedup`, etc. lazily create `MinimaArgs`, which
  reroutes the entire run through multistart (different output, `.sol` with
  zero duals). Help text says only `--minima`/`--multistart` enable it.
- **M15. Real-AMPL driver conventions unsupported despite `-AMPL`.**
  `crates/pounce-cli/src/cli.rs:321-323`,
  `crates/pounce-nl/src/nl_reader.rs:257-264`: no `.nl`-appending for
  extensionless stubs, no `pounce_options` env var. Pyomo works; genuine AMPL
  does not. (Uncertain whether genuine AMPL is in scope; help text says
  "Pyomo / AMPL drivers".)

### NLP layer / .nl reader

- **M16. Constraints and the full Jacobian are evaluated twice per iterate.**
  `crates/pounce-nlp/src/orig_ipopt_nlp.rs:1380-1493, 1495-1576`:
  `eval_c`/`eval_d` each call user `eval_g`; `eval_jac_c`/`eval_jac_d` each
  evaluate all `nnz_jac_g_full` entries. No shared full-space cache below the
  c/d split (upstream keeps tagged `full_g_`/`jac_g_` buffers). Roughly a 2×
  tax on the dominant AD cost for `.nl` problems; also double-counts
  `c_evals`/`d_evals` in statistics.
- **M17. `eval_c_internal` re-fetches bounds and makes four full-size
  allocations per cache miss.** `orig_ipopt_nlp.rs:1416-1430`: the constant
  equality RHS should be captured once (upstream's `c_rhs`); this is in the
  line-search hot path.
- **M18. Per-call allocations in tape-AD hot paths.**
  `crates/pounce-nl/src/nl_tape.rs:198` (`forward`) and `:270-273`
  (`reverse`) allocate per summand tape per call; the design deliberately
  produces ~10⁶ tiny tapes on large models, so one `eval_jac_g` can perform
  millions of small allocations. The Hessian path already has the
  `forward_into` + scratch pattern to copy.
- **M19. Initial duals from the `.nl` `d` segment parsed but never used.**
  `crates/pounce-nl/src/nl_reader.rs:418-432` vs `:2197-2200`:
  `get_starting_point` ignores `lambda0`, so `warm_start_init_point yes`
  silently warm-starts from zero multipliers; the module header still says
  `d` is "read and discarded".

### Convex

- **M20. Silent tolerance relaxation: failures re-labeled `Optimal` at
  residuals far above `tol`.** `crates/pounce-convex/src/hsde.rs:179, 244-249`
  (and 254-261, 277-284, 321-328): factorization/back-solve failures with
  residual `< 1e3·tol` report `Optimal`;
  `hsde_nonsym.rs:1084-1111`: best iterate restored and reported `Optimal` if
  `best_res < √tol` (1e-4 at default 1e-8). ECOS/Clarabel expose this as a
  distinct `*_INACC` status; `QpStatus` has no such variant and `QpSolution`
  carries no final-residual field, so callers cannot detect it.
- **M21. SOS flat-truncation exactness check is weaker than Curto–Fialkow for
  constrained problems.** `crates/pounce-convex/src/sos.rs:546-560` uses
  `rank M_d = rank M_{d−1}`; with constraints of degree > 2 the sufficient
  condition is `rank M_d = rank M_{d−dg}`, and extracted atoms are never
  validated against the constraints — `is_exact = true` ("provably the global
  minimum", sos.rs:426-427) can over-claim. (Uncertain: no concrete failing
  instance constructed.)
- **M22. SOS SDP assembly iterates a `HashMap` — row order and hence
  floating-point results are nondeterministic run-to-run.**
  `crates/pounce-convex/src/sos.rs:361-371`; the test at 858-875 explicitly
  works around the resulting flakiness. A `BTreeMap` fixes it.
- **M23. `PsdCone::kkt_block` is O(n⁵) per cone per iteration.**
  `crates/pounce-convex/src/cones/psd.rs:428-453` applies the scaling
  operator to every unit vector; closed-form `W ⊗ₛ W` entries are O(n⁴).
  `lyapunov_solve` (psd.rs:317-348) similarly does O(n⁴) quadruple loops
  instead of two matmuls. Dominant per-iteration cost for SOS moment SDPs.

### Presolve (additional)

- **M24. Rows dropped as redundant due to bound tightening they themselves
  implied get λ=0 — wrong duals when that bound is active.**
  `crates/pounce-presolve/src/lib.rs:8-11`, `redundant.rs:27-34`: e.g. row
  `x ≥ 2` tightens `x_l = 2`, is dropped as redundant, and the IPM reports
  `z_l > 0` against a bound that does not exist in the original problem.
  Primal/objective unaffected; inherent to the design and worth documenting
  or fixing via dual transfer.
- **M25. Genuine Phase-1 infeasibility leaves `x_l > x_u` in the bounds
  handed to the IPM.** `lib.rs:556-582`: the rollback guard fires only when
  the reduction stack is non-empty; otherwise the IPM gets crossed bounds and
  reports an invalid-problem failure instead of a clean infeasibility verdict.
- **M26. `finalize_solution` densifies the full inner Jacobian.**
  `lib.rs:994-1008`: `vec![0.0; m_inner * n_inner]` whenever a reduction
  frame exists — 80 GB at 100k×100k. `recover_dropped_multipliers` only needs
  the k fixed columns.
- **M27. Quadratic scans in Phase-0 block assembly.**
  `auxiliary.rs:529-556, 717-741` scan the entire nnz array per block row
  (O(total_block_rows × nnz) — quadratic on the gas-network models that
  motivate the feature); `NonlinearBlock::jacobian` (auxiliary.rs:661-678)
  does `position()` per nnz per Newton iteration plus a full-problem
  `eval_jac_g` per iteration.
- **M28. FBBT allocates and scans O(n_vars) per constraint per sweep.**
  `fbbt/orchestrator.rs:142-180`: `vec![Interval::ENTIRE; n_vars]` plus a
  `0..n_vars` apply loop for every constraint — O(m·n) per sweep when each
  tape touches a handful of variables.
- **M29. LICQ structural check duplicates and degrades an existing
  primitive.** `licq.rs:72-110`: per-row `vec![false; n]` allocation and
  recursive augmenting paths (stack-overflow risk on long chains, e.g.
  discretized dynamics), while the crate already has an iterative
  Hopcroft–Karp in `matching.rs`. `licq_check` defaults on within enabled
  presolve.

### Python / bindings

- **M30. `curve_fit` covariance does not project onto the active
  general-constraint nullspace, contradicting its headline claim.**
  `python/pounce/_curve_fit.py:1526-1530` (streaming twin at 1091-1095):
  `active_mask` covers variable bounds only; an active equality between
  parameters is never projected out of `pcov`, overstating variances and
  missing induced anti-correlations, while the label says
  `"reduced_hessian(projected)"`. Relatedly the Gauss–Newton Hessian comment
  assumes linear constraints but the docstring advertises general ones.
- **M31. Issue-#112 PSD guard exists only on `solve_qp`.**
  `python/pounce/qp.py:434-437`: `solve_qp_batch` (547), `solve_qp_multi_rhs`
  (588), `QpFactorization` (626), `QpSensitivity` (679), `solve_socp` (474),
  and the differentiable `jax/_qp.py`/`torch/_qp.py` layers all skip
  `check_psd`, so an indefinite `P` reproduces the silently-wrong
  `"optimal"` — and in the OptNet layers feeds the backward pass too.
- **M32. `intermediate` callback exceptions swallowed with no trace;
  non-`bool` returns coerced to "continue".**
  `crates/pounce-py/src/tnlp_bridge.rs:364-374`: `Err(_) => false` (no
  logging, unlike the eval callbacks), so a crashing callback masquerades as
  `User_Requested_Stop`; `extract::<bool>().unwrap_or(true)` means a callback
  returning `0` (valid cyipopt truthiness) *continues* when the user asked to
  stop.
- **M33. Pyomo plugin finds the solver only via `PATH`.**
  `pyomo-pounce/pyomo_pounce/pounce_solver.py:35-36`: `shutil.which("pounce")`
  despite depending on the wheel that bundles the binary at a deterministic
  path (`python/pounce/_cli.py:22-24`). Non-activated-venv runs (cron, IDE
  runners, Jupyter kernels) report the solver unavailable or pick up a stale
  system binary.
- **M34. Default auto-routing costs O(n²) user-function evaluations before
  the solve starts.** `python/pounce/_route.py:188-192, 432-433`;
  `_minimize.py:425, 455-458`: with FD derivatives, the QP router computes
  two FD Hessians of an FD gradient (~8n² f-evals), then the SOCP router
  recomputes the probes from scratch (another ~8n²) plus constraint-Hessian
  probes. Tens of thousands of evaluations of overhead at n=50 on problems
  that route to the NLP solver anyway; undocumented.
- **M35. Session-style solves hold the GIL for the whole IPM run.**
  `crates/pounce-py/src/solver.rs:80`, `crates/pounce-py/src/qp.rs:486-490,
  522`: `PySolver::solve`, `QpFactorization::solve`, `QpSensitivity::new`
  don't use `allow_threads`, unlike `PyProblem::solve` and the one-shot QP
  entry points. `Solver` is the workhorse under `curve_fit` and the jax/torch
  hosts.

### Infrastructure

- **M36. studio-core's report mirror is missing the `CbfFile` input variant —
  CBF solve reports can't be loaded at all.**
  `crates/pounce-studio-core/src/report.rs:142-154` vs writer
  `crates/pounce-solve-report/src/lib.rs:185-204`: serde's internally-tagged
  enum hard-fails on the unknown `"kind": "cbf-file"`, rejecting the whole
  report despite a matching schema tag.
- **M37. Library UB: `slice::from_raw_parts(NULL, 0)` in the sensitivity C
  API.** `crates/pounce-cinterface/src/solver.rs:331, 339, 375`: legal
  `n_pins == 0` + NULL calls violate `from_raw_parts`' non-null requirement
  (trips debug assertions on recent Rust). The rest of the crate gates these
  correctly.
- **M38. No tag-vs-manifest version check in any release workflow.**
  `.github/workflows/release-crates.yml`, `release-pounce.yml`,
  `release-pyomo-pounce.yml`: tag `v0.5.0` with manifests at 0.4.0 makes the
  crates publish a silent green no-op (everything "already published") and
  PyPI publish the wrong version.
- **M39. `pounce-hsl` is compiled by zero CI jobs but is in the publish
  list.** `.github/workflows/ci.yml:63, 66, 69` exclude it; first compile is
  the `cargo publish` verify build mid-release, after earlier crates have
  already been irreversibly published. A `cargo check -p pounce-hsl` in CI
  closes this.

---

## Low

### Algorithm core

- **L1.** Final iterate never convergence-tested at the `max_iter` boundary —
  `src/ipopt_alg.rs:1651-1656` breaks before `iterate()` runs the check, so a
  solve converging on exactly the last step reports
  `Maximum_Iterations_Exceeded` where upstream reports `Success`; the
  `MaxIterExceeded` branch in `opt_error.rs:233` is effectively dead.
- **L2.** Tiny-step dual test is absolute where upstream is relative —
  `src/ipopt_alg.rs:1041-1042` vs upstream's `1/(1+‖y‖∞)` scaling; the primal
  half (1152-1172) does use the relative form. `STOP_AT_TINY_STEP` under-fires
  on large-multiplier problems.
- **L3.** Probing μ-oracle hard-codes `sigma_max = 100.0`
  (`src/mu/adaptive.rs:685-691`); user-set `sigma_max` affects only the
  quality-function oracle, unlike upstream.
- **L4.** `golden_section` can return an unevaluated `-100.0` sentinel
  endpoint when `qmax <= 0` (`src/mu/oracle/quality_function.rs:540-554` with
  730, 741); also `>=` in `qf_ok` makes the default `qf_tol = 0.0` flat-stop
  dead.
- **L5.** `max_cpu_time` actually measures wall time —
  `src/conv_check/opt_error.rs:257` via `pounce_common::utils::cpu_time()`'s
  documented wallclock fallback.
- **L6.** Dead/divergent duplicates of filter acceptance predicates —
  `src/line_search/filter_acceptor.rs:171-179` (no round-off slack, unlike
  the live path at 292-300) and 199-229 (parameterized `obj_max_inc` while
  the live path hard-codes 5.0).
- **L7.** Watchdog revert applies the current-direction fraction-to-boundary
  cap to the snapshot direction — `src/line_search/backtracking.rs:725-737`;
  the correct stored cap is `#[allow(dead_code)]`. Rescued by backtracking,
  but wastes evaluations post-watchdog.

### Linear algebra / FFI

- **L8.** Ruiz scaler's 0/1-based auto-detection misclassifies a 0-based
  triplet whose index 0 carries no entries
  (`crates/pounce-linsol/src/ruiz.rs:117-129`); factors land on the wrong
  rows. Applied consistently, so result quality degrades rather than
  correctness; the only in-tree caller is safe (1-based).
- **L9.** Deprecated KKT-dump block runs on every solve and uses
  `unsafe env::remove_var` (`crates/pounce-linsol/src/t_sym_solver.rs:197-243`)
  — unsound if any other thread reads the environment, and pounce-feral
  explicitly supports rayon-parallel outer solving (feral lib.rs:159-168).
- **L10.** 32-bit index arithmetic in MA57 sizing has no overflow guard
  (`crates/pounce-hsl/src/ma57.rs:263, 294-297, 434`) — `5n + ne + …`
  overflows i32 near ne ≈ 3×10⁸ and converts to an absurd allocation/abort
  instead of a clean `FatalError`. Inherited from the Fortran interface, but
  cheap to guard.
- **L11.** Per-solve allocations in the backsolve hot paths —
  `ma57.rs:434` (fresh `work` per solve) and
  `crates/pounce-feral/src/lib.rs:564-576` (owned `Vec` + copy per solve);
  dominant non-BLAS cost of the factor-once/solve-many use case.
- **L12.** `FERAL_PIVTOL` breaks the documented `POUNCE_FERAL_*` env-var
  convention (`crates/pounce-feral/src/lib.rs:215-218`);
  `POUNCE_FERAL_PIVTOL` is silently ignored.

### QP / restoration / sensitivity

- **L13.** Doc/code sign mismatches in restoration formulas (code right, docs
  wrong): `resto_nlp.rs:6-7` (`c − n + p` vs implemented `c + n − p`),
  `resto_resto.rs:16-21` (wrong quadratic for the stated root).
- **L14.** Error discrimination by message substring —
  `pounce-qp/src/solver.rs:123-124, 142-143`, `schur.rs:275-276, 297-298`
  retry on `contains("inertia")||contains("singular")`; `resolve`'s
  `"Singular"` (capitalized) would not match.
- **L15.** `ElasticReformulation::original_inertia()` hardcodes `Psd`
  (`elastic.rs:169-175`), making the `Indefinite` arm dead; `solve_elastic`
  ignores `use_schur_updates` (`solver.rs:1038`).
- **L16.** `boundcheck` can panic (index OOB) instead of the documented no-op
  on non-dense bound vectors
  (`pounce-sensitivity/src/boundcheck.rs:64-78, 106-112`); also
  `dv.values()` here and in `dense_to_vec` trips `DenseVector`'s
  homogeneous `debug_assert` where siblings use `expanded_values()`.
- **L17.** `IndexPCalculator::schur_matrix` drops the B-row sign (mirrors
  upstream, but `from_parts` accepts −1 signs making the wrong result
  reachable) and caches P columns by column index only, so two A rows
  selecting the same column with opposite signs share one cached column
  (`p_calculator.rs:150-166, 191-199`).
- **L18.** Restoration inner solver pins tolerances at
  `(1e-8, 1e-6, 15, 3000, 3000)` regardless of the user's outer `tol`
  (`resto_inner_solver.rs:251`), and `is_square_problem = n == m_eq` ignores
  inequalities (line 226), unlike IPOPT's `IsSquareProblem`.
- **L19.** `min_c_1nrm` LSM branch mutates `data.curr` without restoring it
  only on the non-default `constr_mult_reset_threshold > 0` path
  (`min_c_1nrm.rs:397-407`) — a divergence trap between option settings.
- **L20.** `IndexSchurData::set_from_flags` returns the wrong error variant
  and leaves partially-populated state on invalid flags
  (`schur_data.rs:217`); a retry appends duplicates.
- **L21.** l1penalty `new()` ignores `get_starting_point`/`eval_g` failures
  (`wrapper.rs:179-191`); slack seeds silently computed from zero data.

### CLI

- **L22.** Error hint names a nonexistent flag — `main.rs:1613-1621` suggests
  `--solve-report`; the actual flag is `--json-output`.
- **L23.** After a failed MC64 scaling retry, stats reflect the retry but
  status reverts to the original verdict (`main.rs:883-899, 902`).
- **L24.** The `.nl` is fully re-parsed for classification
  (`main.rs:456-461`) — doubles parse time/peak memory on large models; the
  error case silently falls back to NLP.
- **L25.** Failed `.sol` write exits 0 on the NLP path (`main.rs:1076-1080`)
  but 2 on the convex path (main.rs:1287); `-AMPL` callers see a clean exit
  with a stale/missing `.sol`.
- **L26.** Summary block prints identical numbers in the "(scaled)/(unscaled)"
  columns and hardcodes variable bound violation 0.0
  (`print.rs:368-401`); inequality tally comment/code mismatch
  (`print.rs:87-89`) makes the breakdown not sum to the total.
- **L27.** Module doc says exit 0 only on `Solve_Succeeded`; code also
  (reasonably) returns 0 for `SolvedToAcceptableLevel`
  (`main.rs:9-12` vs 1090-1092).
- **L28.** `nl_hessian_program.rs` is dead in-tree but panics on
  `Funcall`/min/max/transcendental ops (lines 456, 477, 594) — would fire on
  user input if ever wired in.

### .nl reader / NLP layer / common

- **L29.** `Pow` first-order tangent disagrees with the reverse-mode gradient
  at base 0 (`nl_tape.rs:469-481, 692-704, 2950-2962` guard on
  `r != 0 && u != 0`; reverse at 302-311 guards only `rv != 0`) — Jacobian
  and Hessian-vector products use inconsistent derivative models at `x = 0`,
  the `.nl` default start. Narrow exposure (constant exponents are lowered to
  Mul/Sqrt chains).
- **L30.** `compare_le` uses `10·eps·max(1, |BasVal|)` where upstream uses
  `10·eps·|BasVal|` (`pounce-common/src/utils.rs:24-27`); the doc comment
  asserts upstream has the `max`, which appears wrong — looser comparison for
  small filter values on the bit-equivalence path.
- **L31.** `strip_comment` can truncate AMPL string literals containing `#`
  (and trims declared-length `h<len>:` tokens) —
  `nl_reader.rs:1069-1074` with 852-873.
- **L32.** Malformed `J`/`G` indices panic later as slice OOB instead of a
  parse error, while `x`/`d` out-of-range entries are silently dropped —
  inconsistent strictness (`nl_reader.rs:379-383` vs 413-415, 427-429).
- **L33.** `relax_bounds` silently no-ops when a bound `Rc` is shared
  (`orig_ipopt_nlp.rs:638-649`) while `adjust_variable_bounds` treats the
  same condition as a hard invariant (`expect`, 1761-1769) — they should
  agree, and the loud version is safer.
- **L34.** `HybridTape` is dead code with a misleading panic in the
  promoted-CSE path (`nl_tape.rs:1861-2168, 2330-2331`).
- **L35.** `k`-segment line count assumed, not read (`nl_reader.rs:359-367`);
  a nonstandard count desynchronizes the line stream into a confusing
  downstream error.

### Convex

- **L36.** Stale module docs claim SOC reduced-system methods are
  `unimplemented!` (`cones/soc.rs:16-21`) and that only the orthant is
  implemented (`lib.rs:6-9`); both are fully implemented and production-wired.
- **L37.** Failure paths seed the dual inconsistently (`z = 1.0` vs `0.0`) —
  `ipm.rs:1116-1120`, `hsde.rs:504` vs `ipm.rs:350-357, 411-418`.
- **L38.** Equilibrated solves leave the per-iteration trace in scaled
  coordinates (`equilibrate.rs:254-278` unscales the solution only).
- **L39.** `QpSensitivity::build`/`reduced_hessian` re-scan all of `G` per
  active row (`sensitivity.rs:134-138, 273-277`; same pattern in postsolve
  `presolve.rs:1584-1590`).
- **L40.** `symmetric_eigen` return value ignored in `reduced_hessian`
  (`sensitivity.rs:300, 348`) — on non-convergence the rank/null-space would
  be silently wrong; everywhere else in the crate the return is checked.
- **L41.** Nonsym driver assembles SOC `(z,z)` blocks dense
  (`hsde_nonsym.rs:262-269`), unlike the symmetric driver's sparse
  diag+rank-1 form — O(m²) fill for large SOCs mixed with exp cones.
- **L42.** Documented sensitivity regularization default `1e-8` vs actual
  `reg: 1e-10` (`sensitivity.rs:30-31` vs `ipm.rs:135`).

### Presolve

- **L43.** Duplicate Jacobian entries handled inconsistently: assignment vs
  accumulation (`inequality_projection.rs:146-148, 213` vs 177, 215;
  `auxiliary.rs:549`) — under the summing convention a wrong `J_block` can
  admit an unsafe block via `all_implied = true`.
- **L44.** `Interval::mul` produces NaN endpoints on `0 × ∞` corners
  (`fbbt/interval.rs:148-159`), which `is_empty()` reads as EMPTY → spurious
  infeasibility in the reverse pass (currently masked by H12's ignored flag);
  `inverse_powint`'s `powf(1/n)` is not outward-rounded (`reverse.rs:223-224`).
- **L45.** `finalize_solution` ignores the `eval_g` return value
  (`lib.rs:931-933`); stale `g_full` forwarded on failure.
- **L46.** Metadata projection infers per-row-ness from vector length alone
  (`lib.rs:1151-1224`) — coincidental matches are silently subset/expanded.

### Python / bindings

- **L47.** `_wrap_constraints` probes user constraint functions at
  `np.zeros(n)` instead of `x0` (`_minimize.py:198-199`) — constraints
  undefined at the origin fail before the solve even with feasible `x0`;
  `jac_combined` (211) also re-evaluates `fn(x)` per FD call just for a row
  count it already has.
- **L48.** `minimize` silently drops information on specific routes
  (`_minimize.py:274, 292-305`): user `hess` ignored whenever constraints are
  present (no warning); convex routes forward only `tol`/`max_iter`, so
  `disp`/`print_level`/`acceptable_tol` are silently discarded and the
  "router is transparent" claim doesn't hold (different `info` keys too).
- **L49.** Strict-contiguity fast path turns valid non-contiguous float64
  arrays into errors instead of copying
  (`pounce-py/src/problem.rs:842-852`, `tnlp_bridge.rs:108-124`).
- **L50.** The KKT-fallback success heuristic can mark `User_Requested_Stop`
  as success (`_minimize.py:527-529`) — combined with M32, a *crashing*
  callback can yield `success=True`.

### Infrastructure

- **L51.** `GetIpoptCurrentViolations` bound-violation branches skip the
  length check their sibling branches perform
  (`pounce-cinterface/src/lib.rs:788-808`); a mismatch would panic inside
  `extern "C"` → abort. Also `nlp_constraint_violation`/`compl_g` are
  documented zero-filled placeholders returned with `TRUE`.
- **L52.** `target_triple` in solve reports is always `"unknown"` —
  `pounce-solve-report/src/lib.rs:405-411`: `option_env!("TARGET")` is never
  set for normal crate compilation (only build scripts see it); needs a
  `build.rs` re-export.
- **L53.** Stale `[patch.crates-io]` story: five "checkout feral sibling" CI
  steps, two manylinux bind-mounts (`ci.yml:21-24, 98-100, 178-183`), and
  `studio/skill/README.md:39` reference a patch that no longer exists.
- **L54.** `release-crates.yml:21` says "all 18 crates"; the list has 19.
- **L55.** `check-docs-consistency.sh`: friendlier failure message is dead
  code under `set -e` (the `rc=$?` after the python heredoc is unreachable).
- **L56.** No `catch_unwind` at the C FFI boundary
  (`pounce-cinterface/src/lib.rs`): any internal panic aborts the embedding
  process (upstream Ipopt returns `Internal_Error`). Related Stacked-Borrows
  aliasing nit on `GetIpoptCurrentIterate` during `IpoptSolve` (documented
  usage), almost certainly benign.

---

## Performance summary (cross-cutting)

The big-ticket items, in rough order of expected impact:

1. **2× constraint/Jacobian evaluation per iterate** in the NLP layer (M16) —
   affects every `.nl` solve with mixed eq/ineq constraints.
2. **Millions of small allocations per Jacobian eval** in the tape-AD path
   (M18) on summand-split models; the Hessian path already shows the fix.
3. **CQ-layer recomputation** (`pounce-algorithm/src/ipopt_cq.rs`):
   `curr_grad_lag_x` rebuilt ~6× per outer iteration (conv check ×3, monotone
   μ loop per reduction, QF oracle, search-dir RHS); dominates when
   factorization is cheap.
4. **O(n⁵) PSD `kkt_block`** (M23) — dominant cost for SOS moment SDPs;
   closed form is O(n⁴).
5. **Presolve Phase-0 quadratic scans** (M27) and FBBT O(m·n)-per-sweep
   (M28) — quadratic on the large models the features target.
6. **Python auto-router O(n²) function evaluations** before the solve (M34).
7. **Full-Jacobian densification in presolve postsolve** (M26) — memory, not
   time.
8. Smaller: per-solve backsolve allocations (L11), QF-oracle vector churn,
   CLI double `.nl` parse (L24), Schur-update QP O(m·nnz) assembly (M10).

---

## Areas reviewed and found clean

- **MA57 FFI** (`pounce-hsl`): ICNTL index mapping, buffer-size formulas,
  MA57ED grow loop, inertia via INFO(24), 1-based convention end-to-end —
  verified against the documented upstream idioms.
- **Triplet→CSR conversion** (`triplet_convert.rs`): hand-traced for
  triangular/full/duplicate/empty-row paths.
- **Core IPM machinery** (`pounce-algorithm`): perturbation-handler state
  machine, PD full-space solver residuals/refinement, augmented-system
  triplet signs, filter semantics, monotone and adaptive μ machinery, L-BFGS
  compact form, SOC and watchdog accept paths, iterate initializer — all
  faithful to upstream.
- **Derivative formulas** (`nl_tape.rs`): first- and second-order formulas
  for all ops verified, Hessian sparsity/coloring satisfies the
  Coleman–Moré condition; `.nl` opcode tables match ASL numbering.
- **Cone math** (`pounce-convex`): NT scaling, barrier derivatives (validated
  by FD + log-homogeneity), svec/smat isometry, HSDE τ-row algebra,
  equilibration round-trip including warm starts and Farkas certificates;
  the recent SOS coefficient equilibration correctly undoes the objective
  scale.
- **Restoration Schur algebra** (`aug_resto_system_solver.rs`) verified by
  hand; sensitivity core math validated against sIPOPT golden output.
- **Verification/infra**: `cli/verify.rs` (NaN-as-infinite-violation, both
  dual signs, FIPS-vectored SHA-256/HMAC), `nl_writer.rs` ASL block layout,
  `publish-crates.sh`/`check-release-consistency.sh` internals,
  observability guards, studio-core `iter_dump.rs` untrusted-input parsing,
  studio MCP server subprocess hygiene.
- **Version consistency**: 0.4.0 aligned across root `Cargo.toml`,
  `python/pyproject.toml`, `pyomo-pounce/pyproject.toml`, `CITATION.cff`,
  and CHANGELOG (note: CITATION.cff is the one surface
  `check-release-consistency.sh` does not guard).
- **No mutable default arguments** in the Python package; status-code
  plumbing Rust↔Python matches `ApplicationReturnStatus` exactly; the
  `SendGuard`/`allow_threads` pattern in `PyProblem::solve` is sound.

## Uncertain / could not verify from this checkout

- **U1.** The feral backend's correctness depends on `CscMatrix::from_triplets`
  (git dep, rev `11fb4b9`, not vendored) summing duplicate `(i,i)` entries —
  the KKT triplet always contains duplicates by construction
  (`std_aug_system_solver.rs:173-200` emits W's diagonal and the
  `D_x + δ_x` diagonal at the same coordinates). Almost certainly true given
  the test suite passes, but it is a load-bearing, unstated contract; the
  lower-triangle canonicalization (`pounce-feral/src/lib.rs:621-633`) can
  also create new collisions from mixed input. A `debug_assert` or doc note
  would be cheap insurance.
- **U2.** MC19 symmetrization mirrors only off-diagonals
  (`pounce-hsl/src/mc19.rs:58-72`); recollection of upstream
  `IpMc19TSymScalingMethod` is that it mirrors all entries. Harmless
  numerically if different, but contrary to the bit-equivalence goal; the
  `ref/Ipopt` tree the comments cite is absent from this checkout.
- **U3.** `SymTMatrix::compute_row_amax_impl` zero-fills even when
  `init=false` (`triplet.rs:583-588`), wiping prior contributions when it is
  the diagonal block of a `CompoundSymMatrix` (`compound_matrix.rs:588-600`).
  The comment asserts this mirrors an upstream bug bug-for-bug; if upstream
  does *not* zero-fill, this is a Medium correctness bug in row-norm scaling
  of compound systems.
