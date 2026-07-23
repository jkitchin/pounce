# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.


## [Unreleased]

### Fixed — `sos_minimize` certified a wrong minimizer as exact on Rosenbrock (#281)

- **`sos_minimize` reported `is_exact=True, num_minimizers=1` while returning a
  point that does not attain the certified bound.** On boxed Rosenbrock-2D
  (`f = (1-x)² + 100(y-x²)²`, unique global minimizer `(1,1)`, `f* = 0`) the
  moment relaxation is not flat at the true measure, so the SDP's first moments
  land in the flat "banana" valley at `≈(0.86, 0.74)` — `0.26` from `(1,1)` yet
  with `f ≈ 0.017–0.020`, close to the (correct) lower bound `≈0`. That point was
  handed back as an exact minimizer. The lower bound itself was always sound; the
  defect was the atom-objective consistency guard in `recover_from_moments`
  (`crates/pounce-convex/src/sos.rs`), whose tolerance `ATOM_OBJ_TOL = 1e-3` was
  too loose: Rosenbrock's flat valley makes a far-off point still read `f ≈ bound`,
  so the guard admitted it. The threshold is tightened to `1e-6` — measured to sit
  ~4× above the worst genuine extraction that asserts exactness (the rank-4
  `facial_reduction_four` case at `2.35e-7`) and ~10× below Rosenbrock's residual
  (`1.05e-5`), both deterministic. When it fires, `is_exact` is withdrawn and the
  still-valid lower bound is returned with **no** minimizers (the safe failure),
  rather than a confidently wrong point. Known-good extractions are unaffected:
  boxed Booth `(1,3)`, Dixon-Price n=2, the three-/six-hump camels, and the
  facial-reduction multi-atom cases all still certify and extract correctly. The
  enforced invariant is now `is_exact ⇒ f(extracted) ≈ lower_bound`.

### Fixed — best-acceptable fallback ranking degenerated to objective-only outside the feasibility band (#280)

- **The NLP best-acceptable fallback could prefer a *strictly more infeasible*
  point**, completing the #267/#270 fix. That fix ranked recorded acceptable
  points by a `(feasible_enough, objective)` key, but the key was a two-class
  *partition*, not an ordering: once **both** the incumbent and a recorded
  candidate sat outside `FEASIBLE_ENOUGH_CAP = 1e-2`, `a_ok == b_ok` and
  `ranks_better_within_band` fell through to a bare `a_obj < b_obj` — reading
  neither point's constraint violation, exactly the pre-#267 objective-only rule.
  Among two infeasible points it again picked the better *objective*, which can
  be *more* infeasible. With the opt-in dual-divergence guard on
  (`dual_diverging_streak=2`) and a widened `acceptable_constr_viol_tol=1e0`, the
  fallback on `deb7` discarded the incumbent at violation `5.292e-1` for a
  recorded point at `9.951e-1` (worse) to gain 36 % of objective, returning it
  under `solve_result_num=100`; `pounce verify` rejected that point. The ranking
  is now a **total order**: each violation is clamped *up* to the band before
  comparison (`viol.max(band)`), so points inside the band tie on feasibility and
  objective decides (unchanged), while outside the band the actual violation
  decides and a strictly-more-infeasible point can never win. Both the record and
  read sides (`record_best_acceptable`,
  `honour_best_acceptable_after_dual_guard`) route through the same
  `ranks_better_within_band` helper, so they cannot disagree. #270's headline
  (`autocorr_bern55-06` at obj `-2303.9999305`, viol `4.149e-5`) and stock-default
  behaviour (honest `MaximumIterationsExceeded` on non-convergence, no fallback
  under a success status) are unchanged. Config-gated: the guard is off by
  default, so stock-tolerance solves never hit this.

### Fixed — `solve_socp` panicked across the FFI boundary on a zero-dimension cone block (#278)

- **A zero-dimension cone block made `solve_socp` raise
  `pyo3_runtime.PanicException`** (a Rust panic crossing the FFI boundary,
  which Python cannot catch as a normal error) instead of a clean, catchable
  `ValueError`. The validator only checked that the cone dimensions *sum* to
  `rows(G)`, so a 0-dim block — contributing 0 rows — passed every documented
  check and then aborted inside a cone constructor: `SecondOrderCone::new(0)`
  hit an `assert!`, and `PsdCone` indexed `vals[0]` on an empty eigenvalue
  vector. It was reachable three ways, all ordinary user input: an explicit
  `("soc", 0)` / `("psd", 0)`; a **negative** dimension (silently saturated to
  `0` by `v.round() as usize`); and a **fractional** dimension below `0.5`
  (rounded to `0`). `parse_cones` now validates each cone's dimension at the
  Python boundary — rejecting non-finite, non-integer, negative, and
  below-minimum values (`soc`/`psd` need `≥ 1`; the empty-safe nonnegative
  orthant still permits `0`) — with a clear `ValueError` naming the offending
  cone's index, kind, and value. As defense in depth, the constructors no
  longer panic on a `0` reaching them from any path: `SecondOrderCone::new`
  drops its `assert!`, and `PsdCone::min_eig` / `max_step` short-circuit the
  degenerate `n = 0` block. A valid `("nonneg", 0)` block still solves
  unchanged, and every well-posed SOCP/SDP/exp/pow solve is unaffected.

### Fixed — convex `tol` / `max_iter` options were unvalidated (#277)

- **`solve_qp` / `solve_socp` (and the batch, multi-RHS, factorization, and
  sensitivity entry points) applied no validation to `tol`,** while every
  other pounce surface (`minimize`, the CLI, `sos_minimize`) rejects a
  non-positive or non-finite tolerance with `OPTION_INVALID`. Consequences: an
  unsatisfiable `tol` (`0`, `-1`, `NaN`, `Inf`) silently burned every
  iteration, and a huge finite `tol` (`1e300`) short-circuited at the
  interior-point *starting* iterate — the convex IPM tests `max KKT residual
  <= tol` at every iterate, so an O(1) tolerance "passes" immediately —
  returning `status="optimal"` after **0 iterations at a wrong point**
  (`x=(0,0)`, `kkt_error=1.0` on the issue's repro). That mislabel propagated
  through the facade: `minimize(solver_selection="qp-ipm", tol=1e300)` reported
  `success=True, nit=0`. Every convex entry point now rejects `tol <= 0`,
  non-finite `tol`, and `tol >= 1` with a clear `ValueError` naming the option
  and value. Capping at `1.0` (rather than accepting any positive `tol`)
  guarantees an accepted tolerance with an `"optimal"` result carries
  `kkt_error <= tol < 1`, i.e. a genuinely near-stationary point — a wrong
  point can never again be labeled `optimal`. A legitimate tight `tol` (e.g.
  `1e-8`) is untouched.
- **`solve_qp(max_iter=-5)` leaked a raw PyO3
  `OverflowError: can't convert negative int to unsigned`** from the `usize`
  binding. `max_iter` is now validated in Python — before it reaches the
  binding — on every convex entry point, so a negative, zero, or non-integer
  value raises a named `ValueError` instead.

### Fixed — integer options above `i32::MAX` silently truncated (#276)

- **`Problem.add_option` / `minimize(...)` wrapped out-of-range integer
  options instead of rejecting them.** The PyO3 binding converted an
  extracted `i64` with a bare `i as Index` (`Index = i32`) cast, which
  *wraps* rather than checks. So `max_iter = 2**32 + 3` silently truncated to
  `3` and ran exactly three iterations with no error or warning, while the CLI
  and Pyomo plugin rejected the same input — the surfaces disagreed. The cast
  is now a checked `i32::try_from`, so any integer option (`max_iter`,
  `acceptable_iter`, `print_level`, `max_soc`, … all share this one path)
  outside the signed-32-bit range raises a clear `ValueError` naming the
  option and quoting the value the user actually passed (not the truncated
  one). In-range values — including the `i32::MIN`/`i32::MAX` boundaries and
  legitimate negatives — still work unchanged.

### Fixed — non-finite inputs silently reported success (#275)

- **`lb = +inf` / `ub = -inf` were dropped as if the bound were absent.** The
  solver's presence test (`lb > -BOUND_INF`, `ub < BOUND_INF`) is
  sign-agnostic, so a bound no finite value can satisfy was discarded and
  `solve_qp` returned `status="optimal"` at a point violating it by an
  infinite margin. `solve_qp` / `solve_qp_batch` / `solve_qp_multi_rhs` and
  `minimize` now reject these spellings with a message naming the index.
  `±inf` on the *absent* side (lower `-inf`, upper `+inf`) remains the
  documented one-sided encoding, and a finite reversed box (`lb=1 > ub=0`) is
  still reported as `primal_infeasible` rather than raising.
- **A NaN or infinite `x0` reported `Solve_Succeeded` with `fun=nan` at
  iteration 0.** Every convergence test is a comparison against a tolerance
  and comparisons against NaN are False — including the ones that would have
  rejected the iterate — so the loop fell through to "converged". `minimize`
  now rejects a non-finite `x0`.

### Fixed — unbounded NLP reported in the AMPL "solved" family (#274)

- An unbounded NLP could be written to the `.sol` as
  `Solved_To_Acceptable_Level` with `solve_result_num = 100`, which lands in
  AMPL's *solved* range — so Pyomo reported `TerminationCondition.optimal` and
  **loaded the diverging iterate as an optimal solution**. `min -exp(x) s.t.
  x >= 0` returned `x ≈ 110.4`, `obj ≈ -8.8e47` under that label.
- Cause: the near-feasible restoration re-entry detector claimed acceptability
  from the *primal* residual alone. A feasible iterate can still be
  arbitrarily far from stationary, which is exactly what an unbounded
  objective looks like — the constraints stay satisfied (`inf_pr ≈ 1.7e-10`)
  while the iterates run off (`inf_du ≈ 8.8e+47`). The only guard was a
  non-finite check, and `-8.8e47` is finite.
- The detector now requires the point to pass the full acceptable-level
  triplet — including `acceptable_dual_inf_tol` — before reporting
  `Solved_To_Acceptable_Level`; otherwise it surfaces the same honest
  restoration-cycle status its sibling exits use. The CLI and the library API
  now agree on this model (`Error_In_Step_Computation`), where previously the
  library reported failure and the CLI reported success.
### Fixed — strictly convex QP falsely reported unbounded (#273)

- The convex IPM's dual-infeasibility (unboundedness) certificate tested
  `‖Pd‖ ≤ rtol·‖d‖` for the candidate recession direction `d`. Because `‖Pd‖`
  is itself proportional to `‖P‖·‖d‖`, the `‖d‖` cancelled and the test
  collapsed to `‖P‖ ≤ rtol` — a bare comparison of the Hessian's magnitude
  against the absolute constant `1e-10`, with no reference to `d` at all.
- Consequence: **any** strictly convex QP with a small enough Hessian was
  certified unbounded despite having a finite minimizer. `min -x + x²/(2M)
  s.t. x >= 0` (unique minimum `x* = M`) was reported unbounded for every
  `M >= 1e10`, terminating after 2 iterations twelve orders of magnitude short
  of the optimum, on a problem Ipopt and pounce's own NLP path both solve
  exactly.
- The residual bound is now scaled by `‖P‖`, restoring the intended meaning:
  a relative test for `d ∈ null(P)`. LPs (`P` empty, `Pd` exactly zero) and
  genuinely singular Hessians with `d` in the nullspace are unaffected, so
  real unboundedness is still detected.
### Fixed — `check_psd` validated a different matrix than the solver used (#279)

- With a `scipy.sparse` COO `P` containing **duplicate `(row, col)` entries**,
  the PSD guard reconstructed the Hessian by *assignment* (last duplicate
  wins) while the solver *sums* them, per the COO convention. The guard
  therefore validated a matrix that was never solved: an indefinite `P` passed
  `check_psd=True` and `solve_qp` returned `status="optimal"` at a saddle
  point. `coo_matrix(([2, 2, 1.5, 1.5], ([0, 1, 1, 1], [0, 1, 0, 0])))` is
  indefinite when summed (eigenvalues `[-1, 5]`) but positive definite under
  overwrite (`[0.5, 3.5]`); the identical **dense** matrix was always rejected
  correctly.
- Duplicate entries now accumulate, so sparse and dense inputs reach the same
  verdict. The mirror write is skipped on the diagonal so a duplicated
  diagonal entry is not counted twice.

### Fixed — constraint dual sign convention (#271, #272)

- **`.sol` / Pyomo `model.dual`** carried pounce's internal Lagrange
  multiplier instead of the AMPL *marginal* `d obj / d b`, so every dual came
  back negated relative to Ipopt, glpk, CBC and CONOPT (#271). Objectives and
  primal solutions were never affected. The `.sol` writer now performs the
  conversion. **This flips the sign of duals read through Pyomo and AMPL** —
  if you compensated for the old behavior in downstream code, remove that
  workaround.
- The Python API's `mult_g` and the JSON report's `solution.lambda` are
  unchanged: they keep the Lagrange-multiplier convention, which matches
  cyipopt. `marginal = −λ`; both conventions are now documented side by side
  in [Running Solves](docs/src/cli.md).
- **GAMS equation marginals on `maximizing` models** were inverted in both the
  pip link and the native C link (#272): the `obj_sign` factor already applied
  to the objective value and the variable marginals was missing from the
  multiplier conversion, which is now `pi = −obj_sign · λ`. `minimizing`
  models, objective values, variable marginals and status mapping were never
  affected, and the two links behaved identically, so install method made no
  difference.
- `pyomo_pounce.gradient()` on a `Constraint` target returned
  `d(λ)/d(param)`, which no longer matched `model.dual` once duals were
  converted; it now reports `d(dual)/d(param)`.

### Fixed — ambiguous 2-parameter tuple bounds and NaN bounds (#265)

- `curve_fit` / `curve_fit_minima` / `curve_fit_streaming`: at `n == 2`, a
  length-2 **tuple** of `(lo, hi)` pairs (e.g. `((0, 10), (0, 10))`) was
  silently read as scipy's `(lower, upper)` — the transposed box pinned both
  parameters and still reported `Solve_Succeeded` (#265, the mirror of #260).
  That shape is genuinely ambiguous and now **raises**, naming both unambiguous
  spellings; the list-of-pairs form `[(l0, u0), (l1, u1)]` and the
  tuple-of-arrays form `([l0, l1], [u0, u1])` are unchanged. A bare `None` side
  (`(None, 10.0)`) still means unbounded on that side.
- NaN bounds are now rejected in `minimize` and every `curve_fit` surface
  (previously they slipped past the reversed-bound check — `lb > ub` is `False`
  against `NaN` — and behaved as a silent "no bound"). `None` / ±inf remain the
  supported unbounded spellings.
- A degenerate covariance is no longer silent. A zero-width bound (`lo == hi`,
  pounce's "hold a parameter constant" idiom) fixes a parameter and reports its
  `perr` as `0`; a corner solution with every parameter on an active bound
  reports `pcov = 0` throughout. Both are intended, but previously came back
  with no signal — "infinite confidence in a wrong answer" in #265's words.
  `curve_fit` / `curve_fit_minima` / `curve_fit_streaming` now emit a
  `UserWarning` in each case (naming the pinned parameters for the zero-width
  case), so the perr of 0 reads as the constraint it is, not an estimated
  uncertainty. No numbers change.

### Changed — `dual_diverging_streak` is now **off by default** (#250 follow-up)

- **The dual-divergence guard is opt-in.** It shipped default-on at `15` to bound
  a reported emfl050 bad-warm-start grind. That justification did not survive
  being reproduced: the reported `11.7 s / iterations=0` measurement is
  caller-side JAX compilation (it follows call order, not the guard, in both
  orders and under both settings), and the build predating the guard solves both
  emfl050 instances to the same optimum in the same time.
- **What remained was basin luck, and knife-edge at that.** Across 1284 MINLPLib
  models the guard changes four outcomes, non-monotonically in its own threshold:
  `deb7`/`deb9` reach a better local optimum (104.95 -> 97.56) at *exactly* 15 and
  at no other value tried, while `pooling_rt2stp` turns from `Solve_Succeeded`
  into `Maximum_Iterations_Exceeded` at 10 and 15 only, solving cleanly at 0, 5,
  25 and 40.
- **The effect differs by host in _sign_.** Those `deb7`/`deb9` numbers are
  macOS/FERAL. On the Linux CI runner the same `dual_diverging_streak=15` makes
  `deb7` *worse*: 97.56 with the guard off, 127.87 with it on. It helps on one
  platform and hurts on the other, same source, same fixture. A heuristic whose
  sign depends on the host is not a property of the algorithm, which settles the
  question: it does not belong in the default path.
- **This also bounds what the fallback below can promise.** It guarantees a
  diverted run never returns worse than the best acceptable point *that run
  visited*; it cannot make the diversion no worse than not diverting, because
  that counterfactual solve never happened. On Linux the guard costs `deb7` 30 %
  of its objective and the fallback cannot recover it — 127.87 *is* the best
  acceptable point the diverted run reached.
- **The two sides are not commensurate.** The upside is a better local optimum on
  an already-solved nonconvex problem; the downside is a clean solve becoming a
  failure. A net-positive count on one corpus is not a reason to impose that on
  every user's problem, so the guard stays available and is no longer default.
- Set `dual_diverging_streak=15` to restore the previous behaviour. When enabled,
  the never-worse-off fallback below applies.
- Full-corpus effect of this release's changes, measured against the commit
  preceding them (`f5aea43`) over all 1284 models: **0 regressions**, 4
  improvements (`jit1`, `nvs04`, `heatexch_spec2`, `supplychainp1_030510` all
  move from a limit/failure status to solved), and one objective improvement
  (`st_e35` 21357.40 -> 21355.24, feasible to 1.8e-12).

### Fixed — the dual-divergence guard's diversion can no longer return a worse point (#250 follow-up)

- **The guard's bet is now non-destructive.** `dual_diverging_streak` (added for
  the emfl050 warm-start stall) routes a solve into restoration once the dual
  infeasibility has grown for that many consecutive iterations in an elevated
  regime. That is a bet, and on the MINLPLib corpus it is usually a good one — it
  rescues twice as many models as it harms — but nothing made *losing* it safe.
  POUNCE now records the best acceptable-quality iterate seen anywhere in the
  solve, and hands it back if a diverted run ends up worse. Applies whenever the
  guard is enabled; since the entry above it is no longer enabled by default.
- **Symptom it fixes.** On `autocorr_bern55-06` the guard fires at iteration 23;
  the diverted run reaches the true optimum (`-2304.0000278`, which Ipopt also
  finds) and holds it from iteration 57 to 86, but the dual residual sawtooths
  between `1e-8` and `2e-1` there, so it never strings together the
  `acceptable_iter` consecutive qualifying iterates that would stop the solve.
  It then entered restoration a second time, wandered into a worse basin, and
  returned `-2263.46` — 1.8 % worse, with an overall NLP error of **1.0**
  (feasible, but nowhere near a KKT point) under a "solved to acceptable level"
  status. The better point had already passed the acceptable test; it was
  overwritten only because `store_acceptable_point` keeps the latest rather than
  the best.
- **The firing threshold's meaning is unchanged** (only its default moved, per
  the entry above). Retuning it to spare this model was tried first and rejected:
  every setting that does so (>= 25) also loses the `deb7` / `deb9` / `deb8`
  rescues, which need exactly 15. No value separates the two classes, so the fix
  addresses the consequence rather than the trigger. Note that the rescues
  themselves are not regression-pinned and deliberately so — `deb7`'s response to
  the guard differs by host in sign (see above), so there is no cross-platform
  assertion to make about it.
- **Statuses that carry a fact of their own are preserved.** A restored point
  never relabels `MaxiterExceeded` / `CpuTimeExceeded` / `WallTimeExceeded` /
  `UserRequestedStop` — a caller polling for "did I run out of time" is not told
  "solved to acceptable level" merely because a better point was recoverable.
- The *use* of this bookkeeping is gated on the guard having actually fired (3 of
  500 corpus models), so every solve it never touches is bit-identical. The
  recording itself is deliberately not gated: the guard returns to the driver
  before the recording site on the iteration it fires, so gating the recording
  too would capture nothing at or before the diversion — exactly the case where a
  diversion wrecks a solve immediately. Recording costs one `f64` comparison per
  acceptable iterate, off already-computed quantities, and clones only on an
  improvement.

### Fixed — the best-acceptable fallback no longer trades feasibility for objective (#267)

- **The fallback now ranks by `(feasibility, objective)`, not objective alone.**
  The #250 fallback above chose among recorded acceptable points by scaled
  objective. Every candidate is *bounded* by `acceptable_constr_viol_tol`, but
  being bounded by the band is not the same as not trading feasibility *within*
  it, and the band is a user option. Widen it and a pure-objective argmax has no
  lower bound on the feasibility it will spend: it discards a nearly-feasible
  point for a lower-objective one that is grossly infeasible, then hands it back
  under a `Solved_To_Acceptable_Level` status.
- **Symptom it fixes.** On `autocorr_bern55-06` at
  `dual_diverging_streak=15 acceptable_constr_viol_tol=1e1 acceptable_tol=1e10
  acceptable_dual_inf_tol=1e30 acceptable_compl_inf_tol=1e10`, the guard-on solve
  returned objective `-2307.32` at a constraint violation of **9.94** — below the
  true optimum `-2304.0` precisely because the point is infeasible, and rejected
  by `pounce verify`. At the same tolerances the guard-*off* control returns a
  feasible point (`-2298.57`, violation `1.06e-4`), so the loose band alone is not
  the cause — it only widens what the fallback can exploit. The fix returns a
  feasible point in the guard-on case too.
- **How the ranking works.** Each recorded/returned point carries its unscaled
  max-norm constraint violation — the same quantity the `acceptable_constr_viol_tol`
  gate is defined against. The key is `(feasible_enough, objective)`: a point
  inside the feasibility band beats one outside it outright, and objective decides
  *only among points already inside the band*. `feasible_enough` uses the
  acceptable feasibility band **capped at its upstream default** (`1e-2`), so it
  never gets looser than a normal solve's — a grossly-infeasible low-objective
  iterate is simply not a candidate. At default or tighter tolerances this is
  behaviour-neutral: every recorded point already passed the
  `acceptable_constr_viol_tol` gate, so with that band at or below the cap every
  candidate is feasible-enough and objective alone decides, exactly as before. The
  cap only bites once the user loosens `acceptable_constr_viol_tol` past its
  default.
- **Symptom, resolved.** With the fix the guard-on solve above returns the
  diverted run's own near-optimal endpoint — objective `-2303.9999` at violation
  `1.13e-4`, which `pounce verify` accepts — instead of the infeasible `9.94`
  point a pure-objective ranking restored.
- **Scope.** Latent and config-gated, not a live default-path bug: the guard is
  off by default, and the trade needs `acceptable_constr_viol_tol` widened past
  its default. Every solve the guard never fires on is still bit-identical — the
  read stays gated on `dual_guard_fired`; only the recording cost grows by one
  already-computed norm per acceptable iterate.

### Fixed — false `Infeasible_Problem_Detected` on a feasible, aggressively scaled NLP

- **Rapid infeasibility detection now confirms its own claim before issuing it.**
  The detector fired on a violation gate plus a stationarity *surrogate*,
  `‖Jᵀc‖ / max(1, ‖c‖)`, against an absolute tolerance. That surrogate is not
  scale-invariant: under a row scaling `dc` the numerator carries `dc²` while the
  denominator clamps at 1, so an aggressive scaling drives it toward zero
  regardless of where the iterate actually is.
- **Symptom it fixes.** On Hock–Schittkowski 13 from `x₀ = (1e4, 1e4)` the
  starting Jacobian is ~3e8, gradient-based scaling picks `dc ≈ 3.3e-7`, and the
  surrogate read `5e-14` — far under the `1e-8` tolerance — at a point whose
  constraint violation was **0.51**, whose `‖∇θ‖` was 1.40, and where neither
  bound was active to block descent. POUNCE reported
  `Infeasible_Problem_Detected` on a *feasible* problem with `f* = 1`. It now
  converges to `0.98492872` in 29 iterations, matching Ipopt from the same start
  to 9 significant figures.
- **No tolerance fixes this, so the fix is not a retune.** Measured over 800
  corpus models plus targeted infeasible problems: the scaled surrogate is not
  separable on the targeted cases; measuring it *unscaled* needs a tolerance
  ≥ 1e-2 to fire at all, which introduces new false infeasibility on 3+ corpus
  models while still losing 2 correct detections; and a scale-invariant
  `‖Jᵀc‖ / ‖c‖²` is not separable either. A single absolute threshold on a
  surrogate cannot separate these cases.
- **What it does instead.** The surrogate is kept as a cheap pre-filter, and
  before the verdict is issued POUNCE probes for a materially less-violating
  point nearby — a few bound-clamped steps along `−∇θ`, comparing `θ` to `θ`.
  That is scale-free by construction and needs no calibration. The two regimes
  are far apart: near a genuine infeasible stationary point only ~0.07 % further
  descent exists, whereas at HS13's false verdict one step takes `θ` from 0.51 to
  zero.
- **Why it matters more than a wrong number.** A false *unbounded* verdict is
  loud and a driver can retry it; a false *infeasible* silently prunes a
  branch-and-bound node that may contain the optimum.
- Costs a handful of constraint evaluations, and only on solves that were about
  to terminate anyway. Genuine detection is preserved: across 1284 MINLPLib
  models the infeasible-verdict count is unchanged at 37, with **0** new false
  positives and **0** lost detections, and the sweep is otherwise bit-identical.

### Fixed — unreachable termination certificate on a strongly objective-scaled NLP (#257)

- **The dynamic barrier floor now expresses `compl_inf_tol` in the space μ
  actually lives in.** The floor is
  `min(tol, compl_inf_tol) / (barrier_tol_factor + 1)`, but its two terms are
  enforced in different spaces: `tol` is compared against the *scaled* NLP
  error, while `compl_inf_tol` is compared against the *unscaled*
  complementarity (pounce#173). Feeding the raw `compl_inf_tol` into a
  scaled-space floor put it `1/|obj_scaling_factor|` too high whenever the
  objective was scaled down, so on a strongly deflated objective μ bottomed out
  *above* the level the convergence test required and the strict certificate
  became unreachable — no matter how long the solve ran. The floor now uses
  `compl_inf_tol · |obj_scaling_factor|` (magnitude, so a maximization posed
  via `obj_scaling_factor = -1` is unaffected), falling back to the
  unconverted tolerance if the factor is absent or degenerate.
- **This removes a tolerance inversion**: because a smaller `tol` incidentally
  dragged the floor down, the failure appeared at *looser* tolerances and not
  at the default `1e-8`. Loosening `tol` could cost a user their certificate.
- **Symptom it fixes.** POUNCE would sit exactly on the optimum, with a scaled
  NLP error well under `tol`, unable to certify it; μ-at-floor plus the
  resulting vanishing step then exited
  `Search_Direction_Becomes_Too_Small`, which drivers commonly map onto
  unboundedness. On the branch-and-bound node subproblems discopt generates for
  MINLPLib `jit1` (`obj_scaling_factor = 1e-5`, `tol = 1e-7`) this hit **59 of
  59 node solves**, leaving the MINLP with no incumbent (`status=unknown`)
  unless the driver retried every node with Ipopt. Post-fix all nodes certify,
  POUNCE reproduces Ipopt's node optimum to the last digit
  (`173345.37683089852`), and `jit1` solves to `obj = 173982.61006345798`,
  `gap = 0` in 9 nodes — node-for-node identical to an Ipopt-driven search.
  Pinned by `crates/pounce-cli/tests/issue_257_jit1_node_certificate.rs`
  against `jit1_node.nl`, the failing node captured verbatim.

### Fixed — no spurious `Unbounded` on a bounded, ill-scaled NLP (#248)

- **The divergence guard no longer reports `DivergingIterates` (Ipopt's
  unboundedness verdict) purely on iterate magnitude.** `DivergingIterates`
  maps to the AMPL 300 "unbounded" range, but a large `max_i |x_i|` does not
  by itself prove unboundedness: under severe objective ill-scaling the
  normal-mode IPM can take a large excursion on a problem that is bounded
  below with a finite optimum (MINLPLib `jit1`), and if every variable is
  boxed the feasible region is bounded so unboundedness is structurally
  impossible. Both divergence guards (the running post-accept check and the
  restoration-failure fallback) now surface `DivergingIterates` only when the
  growth is consistent with an unbounded feasible region — some component
  past `diverging_iterates_tol` heading toward a side with no finite bound.
  When every large component is pinned by a finite bound (in particular, all
  variables boxed), the solve continues to its best iterate and returns a
  non-unbounded status (optimal / iteration limit) instead of a spurious
  `Unbounded`.
- **The verdict now also requires the divergence to *persist*.** For a
  genuinely free variable a low `diverging_iterates_tol` (the kind a branch-
  and-bound driver sets to abort runaway nodes) could still trip on `jit1`'s
  transient excursion — `max_i |x_i|` climbs to ~16 then recedes to the
  finite optimum near ~2.9. `DivergingIterates` is now surfaced only once a
  free, over-threshold iterate has kept *growing* for several consecutive
  iterations (a real recession ray grows geometrically; a settling iterate
  does not), or has blown past an absolute runaway backstop (`1e18`, at or
  below the default `1e20` threshold, so the default "fire the instant
  `|x|` crosses the threshold" behaviour is preserved). Verified end-to-end
  on the MINLPLib `jit1` `.nl`: at `diverging_iterates_tol=2` the published
  model (free variables) and a ±100-boxed variant both now reach the finite
  optimum (`obj ≈ 173345`), while a genuinely unbounded-below problem still
  reports `DivergingIterates`. Together these remove a non-rigorous fathom in
  discopt's branch-and-bound, which uses POUNCE as its per-node NLP backend.

### Fixed — no spurious `Unbounded` on an unbounded-box subproblem with a finite optimum (#252)

- **The divergence guard now also checks the objective trajectory, not just
  iterate growth.** #248's structural + growth-persistence gates cleared
  `jit1`'s boxed / free-variable *root* relaxation, but its spatial-B&B *node*
  subproblems carry variables with `ub = +∞` (integer-tightened boxes). Those
  pass the structural "heads toward an unbounded side" check, and under the
  linear tail's `1e7`-scale ill-scaling the transient excursion climbs past
  enough doublings to satisfy the growth streak too — so every one of jit1's
  59 nodes was reported `DivergingIterates` (an UNBOUNDED false negative;
  cyipopt/Ipopt find each node's finite optimum). A large, still-growing
  `max_i |x_i|` toward an open box side is *not* enough: the guard now requires
  the divergence to look like a genuine recession ray, whose per-step objective
  drop *keeps up* as `|x|` grows geometrically. An excursion converging to a
  finite optimum lowers `f` for a few steps but with a per-step drop that
  *decelerates* toward zero (settling onto the finite floor); it therefore no
  longer accumulates the streak and the solve converges to the optimum instead
  of reporting `Unbounded`. A genuinely unbounded-below objective
  (`f → −∞`), whose per-step drop grows without bound, still reports
  `DivergingIterates`, and the absolute runaway backstop (`1e18`) is unchanged.
  This lets discopt's branch-and-bound retire the load-bearing
  cyipopt-retry-on-UNBOUNDED guard it kept for jit1's nodes.

### Documentation — `PathFollower` gets a book page, figures, and torch docstrings

- **New book page `docs/src/path-following.md`**, linked from the Python API
  page and the TOC. `PathFollower` / `inverse_map_rhs` previously appeared in
  the book exactly once — a single cell in the JAX↔PyTorch parity table — so
  the whole path-following surface was effectively invisible to anyone
  reading the manual rather than the source. The page covers parameter
  continuation (`follow`), pseudo-arclength continuation through folds
  (`trace_arclength`), the inverse/uncertainty map, the `PathTrace` fields, a
  when-to-use-which table, and the equality-only / scalar-θ scope limits. Its
  numbers are measured, not asserted: the `monitor_tol` sweep table (10 → 3
  solves as the tolerance loosens from `1e-6` to `2e-2`, against 12 for
  re-solve-every-step) was generated by running the examples. The page
  carries the fold figure as `docs/src/images/path-following-fold.svg`,
  regenerated by the new `scripts/make-docs-figures.py` so it stays
  reproducible rather than being a pasted screenshot.
- **`docs/src/sensitivity.md` now points forward to it.** The sensitivity page
  ends at a single first-order perturbation off one factor, which is exactly
  where a reader starts wondering about repeated steps, active-set changes,
  and singular `∂x*/∂θ` — it said nothing about the machinery that handles
  them.
- **Notebook 14 is now illustrated** — four figures where it previously had
  only printed numbers: the traced parameter and solution paths coloured by
  path parameter; the `monitor_tol` accuracy-vs-solves trade-off curve
  against the re-solve-every-step baseline; the cubic fold S-curve with both
  turning points marked and `θ(s)` shown reversing (the reason parameter
  continuation stalls there); and the inverse map's recovered input path
  against its analytic value plus the loop-closure check.
- **`pounce.torch.PathFollower` is documented directly** rather than by
  reference. The class, `follow`, `trace_arclength`, and `PathTrace` had one-
  and two-line stubs pointing at the JAX versions, so `help()` on the torch
  class listed none of its nine constructor kwargs. Also restored
  `PathFollower` / `inverse_map_rhs` to the `pounce.jax` package docstring,
  which enumerated the frontend's layers but omitted them.

### Added — the last FERAL numerics env knob is now a registered option (#235)

- **`POUNCE_FERAL_MIN_PAR_FLOPS` is now the `feral_min_par_flops` option**,
  with the env var kept as a fallback — matching the pattern the other
  `feral_*` knobs already follow. FERAL's parallel-dispatch flop gate
  (feral#19) was previously reachable only through the environment: it was
  not a registered option, could not be set per solve, and left no trace in
  the solve report. It is registered as a lower-bounded
  **number** option (not integer) because the gate is a `u64` — an `i32`
  cannot hold large flop counts or the `u64::MAX` reject-all sentinel; the
  value is cast to `u64` with saturation. The four other numerics knobs the
  audit in #235 flagged (`feral_refine`, `feral_fma`, `feral_cascade_break`,
  `feral_singular_pivot_floor`) were already registered options.

### Added — the solve report records active environment overrides (#235)

- **The JSON solve report now captures solve-affecting environment
  variables** in `fair_metadata.environment`. A run that differs because
  `POUNCE_FERAL_PIVTOL` (or any `POUNCE_FERAL_*` knob, or the legacy
  `FERAL_PIVTOL` / `FERAL_PARALLEL`) was exported in a shell profile now
  says so, instead of differing silently between machines — closing the
  reproducibility gap the report is built to serve. Each entry is a
  `{ name, value }` pair; the block is omitted when nothing is set (the
  common case), and the `POUNCE_DBG_*` debug gates are deliberately not
  captured (they never change the result). Additive to
  `pounce.solve-report/v1` — older readers ignore the new field, and older
  reports deserialize unchanged.

### Changed — restoration debug gate reconciled onto one spelling (#235)

- **`POUNCE_DBG_RESTO` now also enables the augmented restoration-system
  stats** that previously answered only to `POUNCE_RESTO_DBG`. The two live
  spellings gated different output in different crates, so guessing the
  wrong one produced silence with no way to tell a spelling mistake from a
  code path that did not run. `POUNCE_DBG_RESTO` is the canonical name (the
  `POUNCE_DBG_*` convention every other gate follows); `POUNCE_RESTO_DBG` is
  retained as a deprecated alias.

### Docs — the previously source-only environment gates are documented (#235)

- **A new [Environment overrides](docs/src/options.md) section** tables the
  `POUNCE_FERAL_*` numerics fallbacks (each mapped to its option) and every
  `POUNCE_DBG_*` / diagnostic gate, noting which take a value, which tracing
  target each needs under `RUST_LOG`, and which print straight to stderr.
  `troubleshooting.md` points to it from the logging recipes.

### Fixed — failed initialization solves no longer poison variable values (Pyomo, #230)

- **A failed solve anywhere in the initialization pipeline now leaves
  variable values exactly as they were.** Previously `block_initialize`
  delegated its block loop to Pyomo's `solve_strongly_connected_components`,
  which loads whatever the subsystem solver returns: a diverged block with a
  warning-status result (infeasible, iteration limit) was written into
  `Var.value` and the loop continued, reporting success. On a 1525-equation
  air-separation model this turned a solvable steady state into a
  3000-iteration stall — the model's own build point solved fine, but the
  "initialized" point, containing a diverged 1500-variable block iterate,
  did not. The block loop is now in this module: each block's verdict is
  checked before its values are kept, a failed block restores its seed
  values, the loop stops there (later blocks typically feed on the failed
  one), and the failure names the block. `project_to_feasible` had
  the same defect — a diverged projection loaded its iterate, making the
  pipeline's "continuing with the unrepaired point" warning untrue — and now
  restores the pre-projection point on any non-optimal termination.

### Added — automatic specification repair: `block_repair_plan`, and repair inside `initialize`/`block_initialize` (Pyomo, #228)

- **A structurally broken specification now initializes anyway, and the report
  says what was repaired.** Some specifications are wrong by structure, not by
  starting point: hold every flow control of a distillation column at steady
  state and the drum levels are undetermined while the holdup balances turn
  redundant — square by count, singular in structure, unsolvable from any
  start. `pyomo_pounce.block_repair_plan(model, decision_candidates=...)`
  plans a valid specification from the variables you would like held: the
  subset a square system can hold (`decisions`), the ones the equalities
  claim and solve for instead (`pruned`, provably the minimum number), and
  the variables nothing can determine (`pinned`) — held at values of your
  choosing. Like `block_analyze` it is a plan, not an action: nothing fixed,
  read, or written, no values needed, and the defect lists
  (`loose_variables`, `redundant_constraints`) come back as uncapped
  component objects.
- **Pins are identified automatically, with no user input.** A variable is
  pinned when every one of its incidence edges is provably unusable: an
  equation `0 == f/g` cannot determine a variable appearing only in the
  denominator `g` (its sensitivity there vanishes at every solution), which
  is exactly the shape substituting `d/dt = 0` into a dynamic balance
  produces. Dropping those edges makes the identification canonical — on the
  569-equation double-column example the four drum levels come out
  identically under every matching order, where raw-incidence matching left
  an order-dependent (and numerically singular) choice.
- `initialize` and `block_initialize` run the same check on their `decisions`
  automatically (`repair="auto"`, the default). A square specification is
  used exactly as given (the shipped behavior); a broken one is repaired,
  with `report.repair` recording the plan (None when nothing was needed)
  and `n_pinned` counting pins separately from decisions. The repair is
  call-scoped like the decisions themselves, so it never alters the model's
  own specification. A pruned decision no longer needs a value (it gets
  solved for); a valueless pinned variable gets a bounds-aware seed that is
  never exactly zero (a pin lives in denominators). `repair="off"` is the
  strict path: decisions held exactly as given, non-square specifications
  reported instead of repaired.
- Pruning ties are deterministic and user-steerable: among candidates,
  earlier-listed ones are preferentially kept, so the `decision_candidates`
  listing order is an implicit priority. Fixing a variable removes it from
  the plan entirely.

### Added — `pyomo_pounce.block_analyze`, the analysis half of `block_initialize` on its own (Pyomo, #224)

- **The Dulmage-Mendelsohn partition is now available without solving
  anything.** `block_initialize` already computes the full partition of the
  equality system, but a caller could only see it through the initialization
  report, where the underconstrained/overconstrained lists are capped,
  name-only, and bundled behind the fill and block solves. `block_analyze(model,
  decisions=[...])` runs the same decision handling and the same decomposition
  and returns a `BlockAnalysisReport` carrying the **full** partition as the
  component objects themselves: the underconstrained and overconstrained
  subsystems (variables and constraints on both sides), the square part, and
  its block-triangular calculation order, with nothing capped. Convenience
  counts (`n_extra_degrees_of_freedom`, `n_extra_specifications`) say how far
  from square the system is.
- Analysis is purely structural, so decisions do not need values here (they do
  in `block_initialize`, whose solve must hold them at a concrete point), no
  values are read or written, and fixed flags are restored on the way out.
  This makes it a safe first pass on a badly specified model: diagnose, decide
  what to specify, then call `initialize`/`block_initialize` to do the work.
- `block_initialize` now delegates its partition step to `block_analyze`; its
  behavior and report are unchanged.

### Fixed — false `Solve_Succeeded` behind an extreme objective scale (#200)

- **POUNCE no longer certifies optimality at a point that is not a minimum.**
  Gradient-based objective scaling picks `df = nlp_scaling_max_gradient /
  max‖∇f‖`, floored at `nlp_scaling_min_value = 1e-8`. On a flat quartic the
  initial gradient is enormous, `df` pins at that floor, and — because a
  quartic's gradient vanishes *cubically* toward its minimum while `df` stays
  fixed — the scaled convergence test trips roughly 30% of the way in. The
  solver reported `Solve_Succeeded` at objective **248.88** (`quartc`) and
  **39.36** (`dqrtic`) when the true minimum of each is ~0, with an unscaled
  dual infeasibility of 0.84.

  | problem | before | after |
  |---|---|---|
  | `quartc` | `Solve_Succeeded`, obj 248.88 | `Solve_Succeeded`, obj **8.8e-07** |
  | `dqrtic` | `Solve_Succeeded`, obj 39.36 | `Solve_Succeeded`, obj **7.0e-07** |
  | `penalty1` | `Solve_Succeeded`, obj 6.44 | `Solve_Succeeded`, obj **0.0097** (true) |

- **This is a deliberate deviation from upstream Ipopt**, which was verified
  here to have the identical failure (`ipopt quartc.nl` → `Optimal Solution
  Found` at 248.88). The new `obj_scale_certificate_threshold` option (default
  `1e-4`, set `0` to disable) restores bit-for-bit upstream behaviour.

- **The mechanism tests the stop rather than predicting it.** When the objective
  scale is below the threshold and the unscaled KKT error is still above
  `acceptable_tol`, POUNCE refuses to terminate and keeps iterating: a constant
  objective scale cancels out of the Newton step and the line-search tests are
  scale-invariant, so the run follows the trajectory an unscaled run would take.
  If that reaches a better point, the stop was false. If it achieves nothing,
  the refused point is restored and reported with the status it would originally
  have had — so the mechanism is **never worse than not having it, by
  construction**, and no benchmark-fitted constant is needed to tell the two
  cases apart. Whether a stop is genuinely false cannot be read off the
  residuals: `meyer3` sits at the same 1e-8 scale floor as `quartc` while being
  genuinely converged. Measured across the 733-problem Vanderbei suite, all 16
  problems eligible for the veto keep their original status.

- **Fixed alongside: the console report hid the discrepancy.** The NLP summary
  passed the *scaled* residual to both the `(scaled)` and `(unscaled)` columns,
  so `quartc` printed dual infeasibility `8.38e-09` twice when the unscaled
  value is `8.38e-01` — a user auditing the suspicious certificate was shown a
  report that agreed with it. The unscaled statistics were already computed and
  already surfaced through the Python bindings; only the console dropped them.
  Upstream Ipopt prints these correctly, so this was a porting defect rather
  than a deviation.

### Fixed — a diverged conic solve could report `Optimal` with a `NaN` solution (#222)

- **`solve_socp_ipm` could return `QpStatus::Optimal` alongside `x = [NaN, NaN]`.**
  A caller checking the status — the documented way to know an answer is usable
  — was handed a garbage solution with no indication anything had gone wrong.
  Observed on the direct symmetric driver (`use_hsde: false`) on PSD programs;
  the underlying flaw was shared by both drivers.

- **Cause: `inf_norm` swallowed `NaN`.** `f64::max` is specified to *ignore*
  `NaN`, so the natural `fold(0.0, |m, x| m.max(x.abs()))` reports the ∞-norm of
  an all-`NaN` vector as a perfect `0.0`. Every convergence test in both drivers
  compares such a norm against `tol`, so a fully diverged iterate read as
  converged: on the reported instance the iterate went non-finite at iteration
  31 and the residuals computed from it came back `pinf = dinf = res = 0`, so
  `res < tol` passed and the solve declared success. `inf_norm` now
  short-circuits on `NaN`, making every such comparison correctly false.

- Two further guards, so the guarantee does not rest on one primitive: each
  driver now breaks with `NumericalFailure` as soon as its iterate goes
  non-finite, and a final pass demotes any `Optimal` / `OptimalInaccurate`
  verdict that is not backed by a finite solution and objective. Across a
  31,920-solve randomized sweep over both drivers, **no solve now reports
  success with a non-finite solution** (previously 32).

- The divergence itself is left as-is: the direct driver stalling or diverging
  on a degenerate face is known behaviour and precisely why the homogeneous
  self-dual embedding is the default (`use_hsde: true`, see `sos_opts`). HSDE
  solves the reported instance correctly. What was wrong was reporting that
  divergence as success, not the divergence.

- The randomized differential suite added in #221 asserted this property of the
  HSDE driver only, and had to skip the direct driver's cases as
  `reference-unusable`; the skip is gone and both drivers are now asserted.

### Added — weak-activity detection for QP sensitivity (#219)

- **`QpSensitivity` can now report whether its own precondition holds.** The
  first-order predictor `parametric_step` is exact only while the active set is
  unchanged, but nothing on the object let a caller check that. Two new
  properties expose what the object already knew: `active_indices` — which
  inequality rows and variable bounds are in the active set, by identity rather
  than the count implied by `kkt_dim` — and `weakly_active_indices`, the
  constraints at which **strict complementarity fails**. Both return an
  `ActiveSet` (`.inequalities` indexes rows of `G`, `.bounds` indexes
  variables), also exported at the package root.
- A weakly active constraint is binding in the primal while carrying a
  negligible multiplier. Classical post-optimal sensitivity (Fiacco) assumes
  this away; where it happens the perturbation changes the active set, so
  `dx/db` is a genuine *one-sided* derivative and the other direction has a
  different, equally valid value. Nothing previously returned was wrong — both
  branches are real derivatives — but a caller could not tell the situation
  apart from an ordinary one.
- The screen is deliberately tolerance-invariant, which `kkt_dim` is not. On the
  reported QP the two branches of `dx/db` are 33% apart and which one is
  reported turns on the solver's `tol`, an unrelated setting; `kkt_dim` flips
  4 → 3 across that sweep while the geometry never changes. The new flag stays
  on throughout, because it tests the multiplier and the slack *together* —
  at a degenerate optimum both collapse at ~`√tol`, while strict
  complementarity keeps one of them bounded away from zero.

### Fixed — the Lasserre/SOS hierarchy now converges on constrained programs (#218)

- **`sos_minimize` returns a real bound where it used to return `nan`.** On the
  reported benchmark (Lasserre, *SIAM J. Optim.* 11(3):796–817, Example 5) the
  hierarchy now tightens to the global optimum instead of stalling:

  | order | before | after |
  |---|---|---|
  | 2 | `optimal` `−7.000` (trivial box bound) | `optimal` `−7.000` |
  | 3 | `iteration_limit` `nan` | `optimal` `−6.667` |
  | 4 | `iteration_limit` `nan` | `optimal` `−5.5080139` (certified), **exact**, minimizer `(2.3295, 3.1785)` |

  True global minimum `−5.5080132716` at `(2.3295, 3.1783)`, verified
  independently here by a 400-start SLSQP sweep. Order 4 is certified exact by
  flat truncation, so the minimizer comes back with the bound, and the reported
  value is a *rigorous* lower bound (see below) rather than merely an accurate
  one.

- **Root cause: Mehrotra's centering parameter inverts on a degenerate face.**
  `σ = (μ_aff/μ)³` infers centering from how far the affine direction could
  travel. On a degenerate face that direction looks excellent while pointing
  almost straight out of the cone, so `σ` collapses toward zero — nearly no
  centering — exactly where centering is the only thing that helps. Order 4 of
  the reported problem pinned `σ` at 0.0218 while the step fell
  `4.0e-1 → 2.1e-2 → 9.7e-4 → … → 1e-281`, throttled by the PSD slack block
  alone, with `μ` frozen at 2.6e-3 and residuals at 4.2e-4 — nowhere near
  converged, simply stuck on the boundary. The corrector is now recomputed under
  an escalating `σ` when its step collapses, each retry one extra back-solve
  through the factorization already in hand. This is the PSD-cone counterpart of
  what the Gondzio correctors do on the orthant, where they are confined because
  a PSD block's complementarity product needs Jordan-algebra machinery.

  The same fix carries the **NETLIB GEN family** (`gen`, `gen1`, `gen4`) and
  `pilot87` from `Maximum_Iterations_Exceeded` to `Solved_To_Acceptable_Level`.

- **Regularization no longer ratchets on a conditioning symptom.** When
  iterative refinement could not reach its tolerance, the driver escalated both
  the `(z,z)` dynamic regularization *and* `δ_c`, the equality-block
  regularization — treating a cone-conditioning symptom as an
  equality-Jacobian rank defect. The KKT is inherently ill-conditioned in the
  `μ→0` endgame (the NT scaling's condition number blows up by design), so this
  fired there on healthy solves and drove `δ_c` to its `1e-1` ceiling, biasing
  the equality residual by `~δ_c·‖dy‖` and flooring `pres` permanently. Order 3
  of the reported problem stood at `pres` 8.6e-9 ✓, `dres` 1.2e-10 ✓, `gap`
  1.7e-8 — one step from converging — when the escalation pushed `pres` to
  2.7e-8, where it stayed. Inertia was correct at every try throughout, so no
  rank defect was ever present. Only the `(z,z)` regularization escalates on
  this signal now.

- **The bound is now certified, not merely converged.** A converged SDP reports
  a `γ` that is *accurate* but need not be a lower **bound**: on the reported
  problem at order 4 the raw value came back `2.2e-7` **above** the true
  minimum, which is the one genuinely unsound failure mode for this API and one
  no solver tolerance removes. `sos_minimize` now measures the miss instead of
  trusting the solve — it projects each Gram block onto the PSD cone, evaluates
  the residual `e` of the coefficient-matching system there, and reports
  `γ − Σ_α|e_α|`. Since every other term of the Putinar identity is nonnegative
  on the feasible set and `|u^α| ≤ 1` on the normalized box, that value is a
  true lower bound however the solve went. Order 4 moves from `−5.508013056`
  (invalid) to `−5.508013930` (valid, and within 1e-6 of the optimum). Costs one
  eigendecomposition per block — no extra solve.

  A new `certified` flag (Rust `SosSolution::certified`, Python
  `SosResult.certified`) reports whether this held. It requires the feasible set
  to lie in a box readable off the constraints — either `x ≥ l` / `x ≤ u` pairs
  or the `c − a·x² ≥ 0` idiom — and is `false` on an unbounded domain, where no
  finite correction exists (a residual with a negative leading coefficient is
  unbounded below). Adding explicit box constraints upgrades such a problem.

- **`tol` and `max_iter` are exposed** on `sos_minimize` (Python) and via
  `sos_minimize_opts` / the now-public `sos_opts` (Rust) — the escape hatch the
  report asked for, for a relaxation that will not converge. Loosening `tol`
  buys a weaker bound, never an invalid one: certification measures the actual
  residual rather than assuming convergence, so validity is preserved across
  the whole range.

- **A coarser bound is no longer discarded.** If the requested order does not
  converge, `sos_minimize` falls back through successively coarser orders and
  reports the first that does, via a new `order` field on the result (Rust
  `SosSolution::order`, Python `SosResult.order`) identifying which relaxation
  produced the bound. A lower-order bound is a valid bound on the same problem,
  so returning nothing in its place threw away a certificate already computed.

- **`PolyProblem::equilibrated` now normalizes the domain, not just the
  coefficients.** When box constraints pin a variable's range it is mapped onto
  `[−1, 1]` before the SDP is assembled. Coefficient equilibration (#124) left
  the domain alone, so a wide box made the moment matrix span decades by itself
  — over `x₁ ∈ [0,3]` the degree-8 moments span `3⁸ ≈ 6561` against 1, which no
  coefficient scaling touches. The change of variables is value- and
  minimizer-preserving, and recovered minimizers are mapped back to the caller's
  coordinates.

- Verified against the committed benchmark baselines: **371 NETLIB LPs** (4
  improvements above, no regressions — the three objective deltas are all on
  runs whose status is unchanged and non-converged or infeasible, where the
  objective is not a meaningful output) and **138 Maros–Mészáros QPs**
  (byte-identical statuses and objectives). The changed driver serves the
  symmetric cones only — exponential/power route to the separate non-symmetric
  driver, and the CBLIB tier is bit-identical to an unmodified tree — so SOC and
  PSD are covered by a new randomized differential suite
  (`tests/conic_hsde_vs_direct.rs`) that checks the HSDE driver against the
  untouched direct driver over **318 generated instances** with planted Slater
  points and compact feasible sets: no disagreement on any optimal value, no
  instance solved by the direct driver that HSDE failed, and 36 that only HSDE
  solved.

### Added — thread-scoped iteration capture from Rust (pounce-rs)

- **A Rust consumer embedding POUNCE can now record a solve's iteration
  trajectory with no direct `tracing`/`tracing-subscriber` dependencies and
  without touching the global subscriber.** `pounce-observability` gains
  thread-scoped helpers that bundle the collector-layer install:
  `with_iter_capture(|| nlp.solve())` runs a closure with capture active and
  returns its result alongside the recorded `IterRecord`s;
  `ScopedIterCapture::start()`/`.finish()` is the guard-shaped equivalent for
  solves that don't fit in a closure; and `collector_scope()` installs just
  the collector for the `IpoptApplication` path
  (`enable_iter_history()` + `statistics().iterations`), where the driver
  manages its own capture guard. All three activate the collector for the
  scope's lifetime and never touch the global subscriber: when
  `init_subscriber` already owns the global (which carries the collector)
  the scope is a no-op and capture composes with console/JSON logging;
  otherwise a collector-only thread-default is installed, which shadows the
  host's own subscriber on that thread for the duration — scope it tightly
  around the solve. Restoration sub-solve exclusion and nested /
  sequential capture semantics are unchanged, and the driver now propagates
  its iteration-history records to any enclosing capture on finish (new
  `extend_active_capture`), so wrapping a solve that has iteration history
  enabled in `with_iter_capture` yields the trajectory in both places
  instead of an empty outer buffer. `pounce-rs` re-exports the
  helpers plus `IterRecord`, `SolveStatistics`, `IterCaptureGuard`, and
  `init_subscriber` (and the whole `pounce_observability` crate), so the
  facade alone covers both iteration capture and console logging;
  `with_iter_capture`, `collector_scope`, `IterRecord`, and
  `SolveStatistics` join the prelude.

### Changed — richer `Nlp::solve()` result (pounce-rs)

- **`builder::Solution` now carries the full per-solve picture.** New fields:
  `g` (constraint values at the solution), `z_l`/`z_u` (bound multipliers),
  and `stats: SolveStatistics` — wall time
  (`stats.total_wallclock_time_secs`), `iteration_count`, evaluation counts,
  final scaled and unscaled infeasibilities, final barrier `mu`, and
  restoration counters, all filled on every solve with no new bookkeeping
  (the driver already computed them; the builder now reads
  `app.statistics()`). A new `.capture_iterations()` builder flag opts into
  the per-iteration trajectory (`stats.iterations`, one `IterRecord` per
  Newton iteration) by activating the thread-scoped collector around the
  solve — a no-op composing with the logs when `init_subscriber` owns the
  global subscriber, a host-subscriber-shadowing thread-default install
  otherwise. Only the interior-point engine emits the per-iteration event
  (an active-set SQP solve leaves `stats.iterations` empty while
  `iteration_count` still counts). The field
  additions are breaking only for code that exhaustively destructures
  `Solution` (pre-1.0).

### Changed — pyomo-pounce streams the engine's log under `tee=True`

- **`SolverFactory('pounce').solve(m, tee=True)` now streams the engine's own
  log — banner, problem statistics, iteration table, and end-of-run summary —
  live to `sys.stdout`, including in Jupyter.** The ~300-line Python
  reproduction of the CLI's blocks is gone: the solver core emits them (see
  below) and pyomo-pounce tails fd 1 to `sys.stdout` on a worker thread, so a
  long solve shows its iteration table as it runs rather than as one block at
  the end. The results object regained `solver.name` and the objective
  bounds, and `solver.time` now measures the solve alone (excluding stream and
  decode). Requires `pounce-solver >= 0.9.0`.

### Changed — the solver core emits its own console log

- **The problem-statistics and end-of-run summary blocks are now emitted by
  the solver core (`pounce-algorithm`), gated on `print_level`,** instead of
  by the CLI. Every frontend — the CLI, the Python bindings, and the C
  interface — gets the identical Ipopt-style log at `print_level >= 1`
  (`print_level 0` is silent). The console printers moved from `pounce-cli`
  to the shared `pounce-solve-report::console` crate as the single source of
  truth, and `IpoptNlp::eval_counts()` drains the NLP's per-evaluation
  counters into `SolveStatistics` so the summary's tallies come from the
  solver's own tracking. `pounce.print_banner()` is exposed for in-process
  frontends that print the up-front banner themselves.
- CLI stdout is byte-identical except: the summary's objective/gradient/
  constraint/Jacobian evaluation counts now report the solver's true count
  (one lower than before, which had included a frontend-side evaluation), and
  `print_level 0` now suppresses the statistics/summary blocks too (it
  previously silenced only the per-iteration table).

### Added — parameter covariance for estimation problems (Pyomo)

- **Parameter covariance for estimation problems, from one solve
  (Pyomo).** Declare the fitted variables (`declare_fitted(m.A, m.k)`,
  varargs, they stay free) and the residual container (`declare_residual(m.r)`,
  optional `group=` strings for heteroscedastic noise groups), solve
  ordinarily with `SolverFactory('pounce')`, then
  `pyomo_pounce.covariance(m)` returns the asymptotic covariance with
  no further information: `cov[m.A, m.k]`, `cov.std_err[m.A]`,
  `cov.correlation[m.A, m.k]`, `cov.matrix`, per-group `cov.sigma_sq`,
  and `cov.eigen()` for identifiability diagnosis. Multiple noise
  groups switch to the heteroscedastic sandwich covariance. Known
  variance (`sigma_sq=`, scalar or per-group) and a bare data count
  (`n_data=`) remain as alternatives to declared residuals. The
  objective must be a plain sum of squared residuals (the solve warns
  when the declared residuals do not reproduce it); the scaling
  (cov = 2 sigma^2 times the parameter block of the inverse KKT
  matrix) is pinned against the analytical linear-regression
  covariance in `tests/test_covariance.py`. All declarations also have
  explicit call-time forms on `solve()` (`sens_params=`, `fitted=`,
  `residuals=`). `covariance(..., hessian="gauss-newton")` reports the
  expected-information (Gauss-Newton) form instead of the default
  `hessian="lagrangian"` observed-information form (the exact reduced
  Hessian of the Lagrangian), from the same
  backsolves; the two agree for linear fits, and Gauss-Newton is
  structurally positive semidefinite and matches the scipy /
  `pounce.curve_fit` convention. A fitted parameter on an active bound
  is projected out (matching `curve_fit`): zero variance, covariance
  conditional on the bound, correlation entries reported as 0, plus the
  existing warning.

### Changed

- **Rust 2024 edition + resolver v3** (#204). The workspace now compiles on
  the Rust 2024 edition with Cargo's v3 (MSRV-aware) dependency resolver.
  This is a build-time change only: no public API, CLI, wheel, or numerical
  behavior changes. Building pounce from source now requires Rust ≥ 1.85 (the
  first toolchain to ship the 2024 edition); users installing the PyPI wheels
  or crates.io releases are unaffected. The migration is the mechanical
  `cargo fix --edition` output (`unsafe(no_mangle)` attributes, explicit
  `unsafe {}` blocks in `unsafe fn` bodies, closure captures) plus the 2024
  rustfmt style-edition reformatting; all `tail-expr-drop-order` sites were
  reviewed and are behavior-preserving (`RefCell`/`Mutex` guards dropped at
  end of scope with no observer of the relative order).

### Fixed — `solve_ivp(mass=M)` now projects inconsistent initial conditions (#215)

- **`pounce.ode.solve_ivp` with a singular mass matrix now projects `y0` onto
  the algebraic manifold before integrating, matching `solve_dae`.** Given a
  singular `M`, the mass path is an index-1 DAE, but it previously returned the
  user's `y0` verbatim in `res.y[:, 0]` even when its algebraic components
  violated `0 = f` — silently, with `res.success == True`. Since the whole
  point of the algebraic variables is that they are *determined* by the
  differential ones, a rough guess for them is the normal case, and the first
  column of the trajectory was left off the solution manifold. It now runs the
  same `consistent_initial_conditions` projection (the IDA `IDA_YA_YDP_INIT`
  computation) that `solve_dae(consistent="project")` already used, so both
  entry points to the index-1 DAE math agree. A new `consistent=` keyword
  (`"project"`, the default, or `"assume"`) opts out for callers relying on
  `res.y[:, 0]` echoing an already-consistent input. Non-singular (plain ODE)
  masses are unaffected.

### Added — opt-in on-manifold output projection for DAEs (#216)

- **`pounce.ode.solve_ivp` and `solve_dae` gain `project_output=False`.** Radau
  IIA is stiffly accurate, so a singular-mass DAE satisfies its algebraic
  constraints to round-off at every accepted step — but the dense-output
  polynomial only *interpolates* the constraint between steps, so intermediate
  `res.sol(t)` / `t_eval` points can sit slightly off the manifold. With
  `project_output=True` the algebraic components of each requested output point
  are Newton-polished back onto `0 = f_alg` (differential components held
  fixed), reusing the index-1 differential/algebraic split. It changes only
  what the caller reads — never the trajectory, step sequence, or error
  control. **Off by default, and skipped automatically when the algebraic rows
  are affine:** a linear conservation law (`sum(x) = 1`, atom / charge / site
  balance) is reproduced *exactly* by the degree-3 collocation output — the
  constraint cubic has four roots and is identically zero — so projection buys
  nothing there and the output is returned bit-for-bit unchanged. It matters
  only for a nonlinear algebraic constraint whose absolute interpolated
  residual is large enough to care about. New diagnostic
  `python/examples/dae_manifold_gap.py` (`manifold_gap()`) measures the gap on
  an arbitrary DAE, and notebook `27_dae_manifold_projection.ipynb` walks
  through both knobs.

## [0.8.0] - 2026-07-11

### Added — declared-parameter sensitivity for Pyomo (Python / Pyomo)

- **`pyomo_pounce` sensitivity interface.** Declare parameters while
  building the model (`declare_sens_param(m.p)` — a flag, no perturbed
  values), solve normally with `SolverFactory('pounce')`, then query:
  `gradient(m.x, wrt=m.p)` (exact dx*/dp; equality constraints give
  their multiplier's derivative), container/Jacobian access via the
  `Gradient` object, and `estimate(m, [(m.p, value)])` for
  first-order perturbed-solution estimates with bound clamping and an
  active-set warning. When declarations are present the solve runs
  in-process through the `pounce.Solver` session (`read_nl` + callback
  bridge) and keeps the converged KKT factorization; models without
  declarations use the CLI path unchanged.
- **Python `Solver` session additions**: `parametric_step_full`
  (full KKT-space step, exposing multiplier sensitivities) and
  `multiplier_rows` (map constraint indices to their `y_c` rows), with
  matching Rust methods `Solver::parametric_step_full` /
  `Solver::g_multiplier_rows` in `pounce-sensitivity`.

### Added — structure-aware KKT hooks (#180)

- **Caller-supplied KKT ordering** (item 1). A structure-aware presolve can
  now hand pounce a precomputed fill-reducing permutation for the KKT linear
  solver — a block-triangular / Schur ordering (Parker, Garcia & Bent,
  arXiv:2602.17968) or a tearing ordering from equation-oriented
  decomposition — that the built-in AMD/METIS pass cannot derive.
  Python: `Problem.set_ordering(perm)` / `get_ordering()` / `clear_ordering()`;
  Rust: `IpoptApplication::set_external_ordering(perm)`. `perm` is a 0-based
  new-to-old permutation whose length equals the augmented KKT dimension;
  FERAL validates it as a bijection and fails the factorization (never a wrong
  answer) on a bad permutation. Maps to FERAL's new `OrderingMethod::External`
  (feral#107); honored by the default FERAL backend only.
- **Per-solve linear-algebra / callback timing** (item 3). `Problem.solve`'s
  `info` dict now carries `info["wall_time"]` and an `info["timing"]`
  breakdown (overall total, the linear-algebra factorization-vs-back-solve
  split, and the per-callback objective / gradient / constraint / Jacobian /
  Lagrangian-Hessian eval time); `pounce.minimize` mirrors these as
  `res.wall_time` / `res.timing`. Lets a caller attribute a reduced-space
  solve's runtime (e.g. densified-Hessian eval cost) directly. The detailed
  breakdown is opt-in via `timing_statistics="yes"` (see #190 under Fixed);
  without it, `wall_time` / `overall_alg` are populated and the per-subsystem
  entries read `0.0`.
- **Block-triangular / Schur KKT solve** (item 2). A structure-aware presolve
  can hand pounce the reducible block of the KKT system; that block is
  Schur-complemented out and only the two diagonal blocks are factorized, with
  full-system inertia recovered a priori via Sylvester's law (Parker, Garcia &
  Bent, arXiv:2602.17968). Python: `Problem.set_kkt_schur_block(indices)` /
  `get_kkt_schur_block()` / `clear_kkt_schur_block()`; Rust:
  `IpoptApplication::set_kkt_schur_block(indices)`. `indices` are KKT-space
  (`x, slack, eq-dual, ineq-dual` block order). The Schur solver
  (`FeralSchurSolver` + `SchurAugSystemSolver`) uses only feral's stable
  factor/solve, and falls back to the standard full-space solver transparently
  when the partition is unsuitable (too large a fraction, malformed, or a
  singular diagonal block), so a stray hook never breaks a solve. Beneficial
  only when the Schur block is much smaller than the eliminated block; honored
  on the default feral + exact-Hessian path.

### Changed

- **feral 0.12.0 → 0.14.0.** 0.13.0 adds `OrderingMethod::External(Vec<usize>)`
  (feral#107), which backs the caller-supplied KKT ordering hook (#180); the
  enum is no longer `Copy` (the `External` arm carries a heap permutation), so
  pounce clones it where a copy was previously implicit. 0.14.0 is a pure-perf
  release targeting the IPM warm-refactor workload (one KKT pattern, thousands
  of factorizations): it splits the symbolic ordering races so only the winning
  candidate pays the expensive tail (feral#127), reuses the permute cache on the
  parallel numeric driver so a warm re-factor scatters in O(nnz) instead of
  rebuilding and re-sorting triplets every iteration (feral#124), and fuses the
  D-block solve into forward substitution for ~14–29% faster warm solves
  (feral#126), plus an opt-in tree-parallel sparse solve (feral#131) and
  analysis-time assembly maps (feral#125). Numerics are bit-identical; no
  breaking API change beyond the 0.13.0 `Copy` removal noted above.

### Fixed

- **A solved convex QCQP no longer reports `InternalError` / exit 1 (#209).**
  On the `.nl` → SOCP conic path (`solver_selection=socp`, and `auto` when it
  routes a QCQP there) POUNCE converged to the correct, feasible optimum and
  then reported failure, with an end-of-run summary showing a large
  `Constraint violation` for a point that is feasible to machine precision. Any
  driver that trusts the exit code or the status — a Pyomo/AMPL wrapper, a CI
  gate — read the solve as failed. Two independent defects:
  - The summary measured the second-order cone rows with the nonnegative
    orthant's per-row `Gx ≤ h` test. A converged SOC block legitimately has
    individual rows with `Gx > h` (only the cone membership `s₀ ≥ ‖s₁‖` must
    hold) and non-complementary rows (only the block product `⟨s, z⟩`
    vanishes), so the reported violation measured nothing to do with the
    quadratic constraint. Conic solves are now measured against their own
    cones (`QpSolution::kkt_residuals_conic`).
  - The conic driver's convergence test reads the *homogeneous* residuals,
    which carry the consistency of the internal slack `s` (`Gx + s − hτ`)
    alongside the real KKT quantities. That term is bookkeeping — `s` is never
    returned — and it floors out once μ reaches ~1e-16 and the Nesterov–Todd
    scaling's condition number explodes. On the reported QCQP it bottomed at
    1e-8, drifted back up to 1e-4, and the solve ground on to a factorization
    breakdown, all while the iterate itself was accurate to 1e-14. A solve
    that ends without a verdict of its own (breakdown or iteration limit) now
    takes one from the **true KKT error of the point it returns** — cone
    feasibility, stationarity, complementarity and `z ∈ K*`, measured on the
    un-homogenized iterate. Below `tol` that is `Optimal`; within `~1e3·tol`,
    `SolvedToAcceptableLevel`; beyond it the original verdict stands. The
    check runs only after the loop, so iterates and iteration counts are
    unchanged — only the verdict and the exit code differ.
- **`pounce.solve_socp` now reports final KKT `residuals`.** Conic solves
  previously returned `residuals=None` / `kkt_error=None` because only the
  orthant measure existed; they now carry the cone-aware residuals, so a conic
  solve's convergence is checkable from Python.
- **`pounce.minimize(..., args=...)` now works with convex routing.** The extra
  objective `args` were applied on the NLP path but not bound into the copies of
  `fun`/`jac`/`hess` handed to the LP/QP/SOCP routers, which probe them as bare
  `f(x)`. A parameterized convex objective therefore either silently never
  routed (`solver_selection=auto` fell back to NLP) or was wrongly rejected as
  "not convex" under a forced `solver_selection`. `args` are now bound into the
  router probes, so a parameterized convex QP/LP/QCQP routes to the specialized
  solver as expected.
- **Post-optimal requests are no longer silently dropped on the specialized
  solve paths (#196).** When an `.nl` declared the sIPOPT sensitivity suffixes
  (`sens_state_1` / `sens_state_value_1` / `sens_init_constr`) or the solve
  asked for a reduced Hessian (`--compute-red-hessian`), the request was
  honored only on the general NLP filter-IPM path. Three fast paths bypassed it
  without a word:
  - **Convex LP/QP/QCQP routing.** Under `solver_selection=auto`, a problem
    that classifies as LP / convex-QP was sent to the pounce-convex solver,
    which has no sensitivity / reduced-Hessian machinery, so no
    `sens_sol_state_1` was written. `auto` now declines the fast path and
    routes such a solve to the NLP path (which honors the request); an
    *explicit* convex `solver_selection` still runs convex but now warns that
    the request is skipped.
  - **`--minima` multistart.** The `--minima` early-return skipped the
    post-optimal step entirely; it now warns that sensitivity / reduced-Hessian
    is not available in a multistart search.
  - **`--minima` `.sol` duals.** The multistart `.sol` wrote a zero placeholder
    for the constraint multipliers; it now recovers the real base-problem duals
    at each reported minimum (via a clean re-solve, so a point accepted from an
    augmented penalty/tunnel solve still gets the base problem's multipliers).
  - **Python `pounce.minimize` convex route.** A user `callback` cannot fire on
    the convex/SOCP route (the solver consumes the extracted quadratic form and
    never calls back into Python); it is now surfaced in the dropped-options
    warning rather than silently ignored.
- **Registered-but-unread algorithmic tuning options are now honored (#191).**
  A range of options were registered but never read, so the solver always ran
  with the hard-coded defaults and any user-set value was silently dropped.
  Now wired through `AlgorithmBuilder`:
  - `kappa_sigma` (default `1e10`) — bounds how far the bound multipliers may
    deviate from their primal estimates via a clamp applied after every
    accepted step, including the documented `< 1` value that disables the
    correction.
  - `kappa_d` (default `1e-5`) — weight of the linear damping term for
    one-sided bounds in the barrier objective/gradient.
  - Filter switching / Armijo / margin constants for the filter line search:
    `eta_phi`, `theta_min_fact`, `theta_max_fact`, `gamma_phi`, `gamma_theta`,
    `s_phi`, `s_theta`, `alpha_min_frac`, `obj_max_inc`.
  - Second-order-correction constants: `max_soc` (incl. `0` to disable SOC),
    `kappa_soc`, `soc_method`.
  - Filter-reset heuristic: `max_filter_resets` (incl. `0` to disable),
    `filter_reset_trigger`.
  - Tiny-step and divergence guards on the algorithm: `tiny_step_tol`,
    `tiny_step_y_tol`, `diverging_iterates_tol`.
  - Inertia-correction / Jacobian-regularization constants on the
    perturbation handler: `max_hessian_perturbation`,
    `min_hessian_perturbation`, `first_hessian_perturbation`,
    `perturb_inc_fact_first`, `perturb_inc_fact`, `perturb_dec_fact`,
    `jacobian_regularization_value`, `jacobian_regularization_exponent`,
    `perturb_always_cd`.
  - Iterative-refinement constants on the KKT full-space solver:
    `min_refinement_steps`, `max_refinement_steps`, `residual_ratio_max`,
    `residual_ratio_singular`, `residual_improvement_factor`.
  - Restoration-phase constants: `bound_mult_reset_threshold`,
    `constr_mult_reset_threshold`, `resto_penalty_parameter`,
    `resto_proximity_weight`. The outer builder carries these (read from the
    options list) and propagates them into the restoration builder when the
    restoration factory is minted, so all frontends honor them without
    per-frontend plumbing.

  Every default equals the previously-hard-coded value, so runs that don't set
  these options are unchanged. The only options still not wired are ones whose
  underlying behavior is not yet implemented (e.g. `neg_curv_test_tol`'s
  non-zero branch, `expect_infeasible_problem`, `start_with_resto`,
  `alpha_for_y_tol`); wiring those would be misleading until the feature
  lands.
- **`timing_statistics=no` no longer runs the detailed timers every iteration
  (#190).** Every `TimedTask::start`/`end` pair calls `getrusage(RUSAGE_SELF)`
  (twice — once each), and the per-subsystem / per-callback timers wrap hot
  paths (each objective/gradient/constraint/Jacobian/Hessian evaluation, plus
  every solve phase). Upstream Ipopt gates these detailed timers on
  `timing_statistics` (default `no`), but pounce mirrored the timers without
  the gating — so the syscalls were paid unconditionally, measuring at 16–20%
  of busy CPU on fast-objective, high-iteration NLPs. The detailed timers are
  now disabled unless `timing_statistics yes` (or `print_timing_statistics
  yes`, which implies it) is set. `OverallAlgorithm` stays live regardless: it
  feeds the `max_cpu_time` convergence check and its total is always reported.
- **pip GAMS link honors `json_output` / `json_detail` (#187).** The pure-Python
  GAMS link parsed those option-file keys and then discarded them, so a
  `pounce.opt` requesting a `pounce.solve-report/v1` JSON was a silent no-op on
  the pip route even though `docs/src/gams.md` advertises it (only the native C
  link implemented it). `Problem.solve` now takes optional `report_path` /
  `report_detail` (`"summary"` | `"full"`) kwargs that emit the report through
  the **canonical Rust writer** (`pounce-solve-report`, the same schema/serializer
  as the CLI's `--json-output`) — no report format is reimplemented in Python.
  The GAMS link threads the two link options into that surface (Full detail
  enables the per-iteration trace the `pounce-studio`/MCP post-mortem tools
  consume), and the writer is now available to any Python caller.
- **Convex QP/LP path now honors `max_iter=0` (#186).** A Pyomo/AMPL solve
  auto-routed to the `pounce-convex` interior-point path reported *Optimal
  Solution Found* even with `max_iter=0`, because the routed problem was solved
  by presolve or a direct step that ignored the iteration cap — violating the
  AMPL/Ipopt contract that zero iterations cannot reach optimality (the NLP
  path already reported `MaximumIterationsExceeded`). The convex QP and SOCP
  dispatch now short-circuit to an iteration-limit result before any solve when
  `max_iter=0`, and `max_iter` is forwarded to the convex driver for the `=0`
  case (previously dropped). CI now runs `pytest pyomo-pounce/tests`
  end-to-end (not just an import smoke test), which is how this regressed
  silently.
- **`reaction_network` mode-aware dedup on flat eigenmodes (#183).** When the
  PES has a genuine zero (flat) Hessian eigenmode — rigid translation/rotation
  of any molecule, or an intrinsically flat coordinate — the minima dedup
  compared full-coordinate distance, so copies of the *same* basin displaced
  along the flat direction counted as distinct minima and exhausted the
  `n_states` budget before flooding reached other basins (a whole basin, and
  its connections, silently missed). `reaction_network` now deduplicates
  minima, saddles, and saddle→basin descent matches in the **non-null subspace**
  of the Hessian, quotienting out any eigenmode below `eig_tol`. Reduces
  exactly to the previous scaled-Euclidean metric when no null modes are
  present, so well-conditioned surfaces are unaffected. `find_saddles` gains an
  optional `distance` override for the same purpose.

## [0.7.0] - 2026-07-01

### Added — `pounce-rs` Rust facade crate (#168)

- **`pounce-rs`** is a single-crate facade for solving nonlinear programs from
  Rust. It re-exports the `TNLP` problem trait (`pounce-nlp`), the
  `IpoptApplication` driver (`pounce-algorithm`), and the supporting scalar
  types (`pounce-common`) in one place, plus a `prelude` — the Rust counterpart
  to the one-import `import pounce` Python API. Pins a single curated public
  surface. The 20th published crate.
- **Ergonomic builder API** in `pounce-rs` (argmin-style, per the #168
  discussion): implement the small `Problem` trait (only `objective` is
  required) and configure + solve with the `Nlp` builder
  (`Nlp::new(problem).var_bounds(..).constraint_bounds(..).solve()`).
  Unimplemented `gradient` / `jacobian` are finite-differenced and the Hessian
  defaults to limited-memory L-BFGS, so a simple problem stays small; the full
  `TNLP` trait remains for advanced use. Runnable HS071 + constrained-QP
  examples in the crate docs.

### Added — event detection (`pounce.ode.solve_ivp` / `solve_dae`)

- **SciPy-compatible `events=`** on both `solve_ivp` and `solve_dae`. Zero
  crossings of event functions `g(t, y)` are located during integration,
  root-found on each step's dense-output polynomial. Each event may carry
  `terminal` (`bool` or a positive `int` count — stops with `status=1`) and
  `direction` (`>0` rising, `<0` falling, `0` either); crossings are returned in
  `res.t_events` / `res.y_events`, and `args` are forwarded to events as in
  SciPy. Event times match `scipy.integrate.solve_ivp` to solver tolerance.
  (Resolves #165 item 4.)

### Added — state-dependent mass + higher-order differentiable DAE

- **`solve_ivp(mass=M(t, y))`** now accepts a callable mass (state/time-
  dependent `M(t, y) y' = f`), routed through the fully-implicit DAE engine as
  `F = M(t,y) y' − f`; the constant-array form is unchanged. (Resolves #165
  item 3.)
- **`pounce.jax.daeint` / `pounce.torch.daeint` default to BDF2** (`order=2`,
  L-stable, second-order) instead of backward Euler; pass `order=1` for BE.
  Same node-value collocation (one extra Jacobian subdiagonal), same IFT
  backward — validated as order-2 convergent with gradients matching finite
  differences.

### Added — fully-implicit DAEs (`pounce.ode.solve_dae`)

- **`pounce.ode.solve_dae(F, t_span, y0, yp0=None, ...)`** integrates a
  fully-implicit, index-1 DAE `F(t, y, y') = 0` with the same Radau IIA(5)
  collocation as `solve_ivp`, in residual form. A pounce extension —
  `scipy.integrate.solve_ivp` has no fully-implicit DAE solver. Reuses the
  whole stiff engine (sparse-LU stage solve + pattern reuse, stage predictor,
  adaptive control, dense output / `t_eval`). Index-1 only.
- **Consistent initial conditions** computed automatically
  (`consistent="project"`, the default): algebraic variables are detected from
  the sparsity of `∂F/∂y'`, then `(y0, y'0)` are Newton-projected onto
  `F(t0, y0, y'0) = 0` (the IDA `IDA_YA_YDP_INIT` computation), so an
  approximate `y0` and `yp0=None` are accepted. `consistent="assume"` uses a
  caller-supplied consistent `yp0` as-is.
- Optional analytic `jac(t, y, yp) -> (∂F/∂y, ∂F/∂y')`; both blocks are
  finite-differenced otherwise. Docs: `docs/src/dae.md`.
- **`pounce.jax.daeint` / `pounce.torch.daeint`** — differentiable fixed-mesh
  integration of `F(t, y, y', theta) = 0`, returning the trajectory
  differentiable w.r.t. `theta` and `y0` via the implicit-function theorem on a
  backward-Euler collocation (the FERAL sparse-LU back-solve mirrors
  `pounce.jax.odeint`). Gradients validated against finite differences.

### Changed

- **`pounce.ode.solve_ivp` no longer silently no-ops SciPy parameters**
  (gh #165): passing `vectorized=True` (ignored) or an unrecognized option now
  emits a `UserWarning` instead of vanishing. The fully-implicit `solve_dae`
  above also supersedes the "constant mass only" limitation for callers needing
  a general implicit form.
- **`pounce.minimize` routed LP/QP/SOCP results now report unbounded and
  infeasible outcomes in plain language** (gh #160). When the convex solver
  returns a dual- or primal-infeasibility certificate, the result `message` now
  reads "The problem appears unbounded …" / "… infeasible …" (status `3` /
  `2`, matching SciPy `linprog`) instead of the raw `dual_infeasible` /
  `primal_infeasible` string — so a downstream adapter can distinguish
  unboundedness from a generic iteration limit. The raw certificate is still
  available in `res.info["status"]`. (Note: the general NLP path —
  `solver_selection="nlp"`, the default — cannot certify LP unboundedness, the
  same as stock Ipopt; route linear/convex problems with
  `solver_selection="lp-ipm"` / `"auto"` to get the certificate.)


### Fixed — ODE/DAE Radau engine: dense LU, complex-split stage solve, exact Jacobian (#175)

- **`pounce.ode.solve_ivp` / `solve_dae` no longer crash with a `SingularBasis`
  error** on stiff/DAE problems whose stage matrices become ill-conditioned on
  the slow manifold (e.g. the Robertson index-1 DAE). The Radau IIA(5) stage and
  error operators now factor with a faer dense partial-pivoting LU (`DenseLU`)
  that — like LAPACK / SciPy's `Radau` — always completes (a singular matrix
  surfaces as `inf`/`nan` in the solve, which the step control already handles)
  instead of hard-failing. The stage solve is rewritten as the standard RADAU5
  **complex split** (one real + one complex shifted operator via the Butcher
  eigendecomposition), so it stays well-conditioned at a singular-Jacobian
  equilibrium.
- **The stage Jacobian now defaults to exact JAX forward-mode autodiff**
  (`jax.jacfwd`) when the RHS is JAX-traceable, falling back to an accurate
  central difference for opaque callables — replacing a noisy forward difference
  that inflated the step count ~45× near singular-Jacobian steady states.
  Robertson integrated to `t = 1e11` now matches SciPy's Radau step count.

### Changed — IPM status fidelity on ill-conditioned / scaled solves (#173)

- The interior-point solver no longer reports `Solve_Succeeded` when the
  **unscaled** KKT error remains large (untrustworthy duals) even though the
  scaled error looks converged. Convergence is now gated on the unscaled
  dual/primal/complementarity infeasibility, so a downstream consumer can trust
  an `Optimal` status. Adds unscaled-error accessors to the convergence check
  and extends the fidelity fix to the SQP and convex facades.

### Changed — feral linear-solver backend bumped to 0.12.0 (#177)

- Resolves the qap15 / mittelmann conic-KKT family end to end: (#91)
  `OrderingPreprocess::Auto` verifies fill instead of predicting, removing a
  misfiring MC64 `LdltCompress` trigger that inflated fill ~6× (qap15 factor
  15.4s → 0.77s); (#99) packed BLAS-3 dense trailing update (~8–10× on large
  dense fronts); (#102) fixes a latent re-entrant nested-rayon workspace-mutex
  deadlock the ordering change exposed; (#105) escalates the ordering to
  `LdltCompress` on pivot growth so factorization accuracy holds on late μ→0 IPM
  KKTs. qap15 now solves (was a 300s timeout) with no regressions across the
  mittelmann / LP / QP suites.

## [0.6.0] - 2026-06-20

### Performance — stiff ODE stepper (`pounce.ode`)

- **Stage predictor.** The adaptive Radau stepper now warm-starts each step's
  simplified-Newton stage solve by extrapolating the previous step's
  collocation polynomial (the standard RADAU5 predictor), instead of cold-
  starting from `K = 0`. This cuts the per-step Newton iterations: on Van der
  Pol (mu=1000) it drops `nfev` ~24% (≈23.7k → ≈18k) and wall-clock ~15%,
  bringing the stiff solve to near parity with `scipy.integrate.solve_ivp`.
  No change to accuracy or the public API.
- **Wider LU-reuse band.** The step-size controller now holds `h` (and so
  reuses the cached `(3n×3n)` factor) on growth up to 2× (was 1.2×). On large
  stiff systems where the dense factor dominates, this drops factorisations
  per step well below SciPy's and cuts wall-clock ~25–30% (e.g. a 100-state
  Brusselator), with no accuracy cost.
- **Reuse the LU pattern across refactors.** The stepper was rebuilding the
  `SparseLU` object — re-bucketing the `(3n)²` COO pattern and re-running
  FERAL's symbolic analysis — on *every* refactor, even though the sparsity
  pattern is fixed for the whole solve. The pattern object is now built once
  per solve and refactored in place (the binding already caches the symbolic),
  so each step pays only the numeric factorisation. This is a large-`n` win
  that grows with system size: **~4× faster on a 100-state Brusselator
  (318 → 79 ms) and ~7× on 300 states (2.83 s → 0.41 s)**, cutting the gap to
  `scipy.integrate.solve_ivp` from ~14–30× down to ~3–4×. Identical accuracy
  and step counts; no API change.

### Added — boundary value problems (`pounce.bvp`)

A `scipy.integrate.solve_bvp`-compatible boundary value problem solver, plus
differentiable JAX/PyTorch frontends:

- **`pounce.solve_bvp(fun, bc, x, y, p=None, ...)`** — drop-in for
  `scipy.integrate.solve_bvp`. Discretises the BVP with 4th-order
  Hermite–Simpson collocation on a fixed mesh and solves the square
  collocation root-find as a pounce feasibility NLP. Returns a SciPy-shaped
  bunch (`sol`, `x`, `y`, `yp`, `p`, `rms_residuals`, `niter`, `status`,
  `message`, `success`). Accuracy matches SciPy (same collocation scheme).
  The default `method="newton"` factors the exact **sparse** `N×N`
  collocation Jacobian (analytic per-node blocks from `fun_jac`/`bc_jac`,
  else a vectorised finite difference) with FERAL's unsymmetric sparse LU,
  using a **modified (frozen-Jacobian) Newton** that reuses the factor
  across steps and refactors only on stall — so it is **typically faster
  than `scipy.integrate.solve_bvp`** at equal mesh (≈0.6–1.0×), including
  large nonlinear problems. `method="ipm"` solves it as a pounce
  feasibility NLP.
  Adaptive mesh refinement is **on by default** (`adaptive=True`, like
  SciPy — a faithful port of SciPy's Lobatto residual estimator + refinement
  rule that reproduces its mesh sequence node-for-node); `adaptive=False`
  solves the given mesh as-is. The collocation system is solved to round-off
  independent of the mesh `tol` (the latter only gates refinement).
  `verbose` mirrors SciPy (1 = termination report, 2 = per-iteration
  progress). Result `status` codes: 0 converged, 1 max nodes, 2 singular
  Jacobian, 3 bc_tol unmet, 4 Newton non-convergence, 5 IPM acceptable-only.
- **`pounce._pounce.SparseLU`** — new PyO3 binding exposing FERAL's
  unsymmetric sparse LU (`factor` / `solve` / `solve_transpose`) for direct
  `A x = b` on general sparse matrices.
- **`pounce.solve_bvp_constrained`** — constrained / optimal-control BVPs
  (state & parameter bounds, inequality path constraints, optional
  objective over an under-determined system), solved with the interior-point
  method on the collocation NLP. This is unique to pounce —
  `scipy.integrate.solve_bvp` cannot express bounds, path constraints, or an
  objective.
- **`pounce.jax.solve_bvp` / `pounce.torch.solve_bvp`** — the same solve made
  differentiable w.r.t. a `theta` parameter threaded into `fun` / `bc`, via
  the implicit-function theorem on the collocation system. Supports
  gradients/Jacobians w.r.t. ODE/BC coefficients, boundary values, and the
  sensitivity of solved-for unknown parameters `p*`. The default
  `method="newton"` is the fast path (FERAL sparse-LU forward + sparse
  `R_zᵀ` backward, first-order). `method="ipm", second_order=True` wraps the
  solve in a `custom_jvp` that re-applies the implicit-function theorem,
  enabling `jax.grad(jax.grad(...))` / `jax.hessian` to arbitrary order.
- Docs: `docs/src/bvp.md`; worked accuracy/speed/differentiability comparison
  in `python/examples/bvp_scipy_compare.py`.

#### Scope and positioning

Honest framing of where this sits relative to other BVP solvers:

- **Algorithm class.** Fixed 4th-order Hermite–Simpson collocation — the same
  family as MATLAB `bvp4c` and `scipy.integrate.solve_bvp` (itself a
  bvp4c-style port). At equal mesh we match SciPy's accuracy and are
  typically a bit faster; this is "competitive with a widely-used production
  solver," **not** the numerical state of the art. Higher-order /
  variable-order collocation (COLNEW/COLSYS), 5th-order `bvp5c`, and
  deferred-correction / continuation codes (TWPBVP, ACDC) need fewer nodes
  per digit of accuracy and are more robust on stiff / singularly-perturbed
  boundary-layer problems.
- **Where it genuinely leads.** End-to-end **differentiability** of the
  solution (`∂y/∂θ`, Jacobians, second order) via implicit differentiation in
  JAX/PyTorch, and **integrated bound / path constraints and objectives**
  (optimal control) through the IPM — capabilities classical BVP solvers do
  not offer. (For heavy constrained optimal control, mature direct-collocation
  stacks such as CasADi and Pyomo.DAE + IPOPT remain more complete.)
- **Not yet covered:** variable/high-order collocation; continuation /
  deferred correction for stiff boundary layers; multipoint boundary
  conditions; DAEs; the singular term `S`; complex-valued problems. A
  credible "SOTA" claim would also require benchmarking against COLNEW /
  `bvp5c` / SciPy on a standard suite (e.g. the Cash–Mazzia test set) for
  accuracy-vs-nodes and robustness, not just speed-vs-SciPy.

### Added — stiff ODE / DAE initial value problems (`pounce.ode`)

A `scipy.integrate.solve_ivp`-compatible stiff solver, plus differentiable
JAX/PyTorch frontends:

- **`pounce.ode.solve_ivp(fun, t_span, y0, method="Radau", ...)`** — drop-in
  for `scipy.integrate.solve_ivp` with the implicit `Radau` method (3-stage
  Radau IIA, order 5, L-stable — the same RADAU5 of Hairer–Wanner that SciPy
  implements). Adaptive step control with the embedded order-3 error estimate
  and a simplified-Newton stage solve whose Jacobian is factored with FERAL's
  sparse LU. Tracks SciPy's `Radau` step-for-step on stiff problems (Van der
  Pol μ=1000: 1082 vs SciPy's 1188 steps, agreeing to ~7e-7) and returns a
  SciPy-shaped bunch (`t`, `y`, `sol`, `nfev`, `njev`, `nlu`, `status`,
  `message`, `success`). Supports `t_eval`, `dense_output`, `args`, `jac`,
  `first_step`, `max_step`, `rtol`/`atol`. Only `method="Radau"` is
  implemented (the stiff/DAE niche); other methods raise rather than silently
  substitute, and `events=` is not yet supported.
- **Index-1 DAEs** via a mass matrix: pass `mass=M` to integrate `M y' = f`.
  A **singular** `M` makes it an index-1 differential-algebraic equation —
  something `scipy.integrate.solve_ivp` cannot do. Validated on Robertson
  kinetics (conservation constraint held to round-off).
- **`pounce.jax.odeint` / `pounce.torch.odeint`** — differentiable
  fixed-mesh integration. An IVP on a fixed mesh is a BVP with
  `bc(ya, yb) = ya - y0`, so this reuses the Hermite–Simpson collocation and
  the same FERAL sparse-LU implicit-diff back-solve as `solve_bvp`. Returns
  the trajectory differentiably w.r.t. the ODE parameters `theta` **and** the
  initial condition `y0` (gradients exact for the discretisation; checked
  against analytic and finite differences).
- **Dict-subscriptable results.** `OdeResult` and `BVPResult` now support
  SciPy-`Bunch`-style item access (`res["y"]`, `"success" in res`,
  `res.keys()`, `res.get(...)`) alongside attribute access, for a tighter
  drop-in.
- Docs: `docs/src/ode.md`; worked stiff/DAE/differentiability comparison in
  `python/examples/ode_scipy_compare.py`.

### Added — GAMS solver link, now pip-installable (`pounce-solver[gams]`)

- **`pip install pounce-solver[gams]` + `pounce-gams register`** registers
  POUNCE as a GAMS NLP solver (`option nlp = pounce;`) with no compiler, no
  `sudo`, and nothing GAMS-owned redistributed — built on GAMS's own
  `gamsapi[core]` GMO/GEV bindings. The link wires GMO's numerical evaluators
  straight into the solver's cyipopt-style `Problem` callbacks (POUNCE is a
  local NLP solver, so no opcode translator is needed). Registration merges a
  per-user `gamsconfig.yaml` `solverConfig` entry, preserving other solvers and
  surviving GAMS upgrades. The native C link in `gams/` remains as the
  alternative route. Adds the `pounce-gams` / `pounce-gams-link` console scripts
  and the `[gams]` extra. Docs: `docs/src/gams.md`.

### Added — solver & `.nl` parser

- **`mu_strategy_fallback`** (opt-in, default off): on a
  `Solved_To_Acceptable_Level` or `Maximum_Iterations_Exceeded` exit, flip
  `mu_strategy` (adaptive↔monotone) once and re-solve, promoting the retry only
  if it reaches `Solve_Succeeded` (otherwise the original outcome is kept).
  Recovers genuine adaptive-μ stalls.
- **AMPL power opcodes** `o81` / `o82` / `o83` (`OP1POW` / `OP2POW` / `OPCPOW`)
  in the `.nl` reader. AMPL emits these as a hint that one operand is constant;
  they previously hit the unsupported-opcode fallthrough, so any `.nl` emitting
  them failed to parse. They lower to the existing negative-base-safe
  constant-power path.

### Fixed

- **`acceptable_iter=0` now disables acceptable-level termination** (restoring
  upstream Ipopt's `acceptable_iter_ > 0` guard) instead of firing on the very
  first acceptable iterate. The GAMS link also now defaults `acceptable_iter=0`,
  mirroring the GAMS–Ipopt link, which removes premature
  `Solved_To_Acceptable_Level` exits on several princetonlib models.
- **CLI:** honor `presolve` on the convex LP/QP path (#139); report reduced
  (post-fixed-variable-removal) dimensions in the solver banner (#140).


## [0.5.0] - 2026-06-14

### Added — broader `scipy.optimize.minimize` compatibility

`pounce.minimize` now covers much more of the SciPy surface, so it works as a
drop-in `method=` callable for `scipy.optimize.minimize` and ports existing
SciPy code with fewer changes:

- **`args=(...)`** — extra positional arguments forwarded to `fun` / `jac`.
- **`jac=True`** — `fun` returns `(value, gradient)` in one call; the pair is
  cached so the gradient is not recomputed.
- **`callback`** — invoked each iteration; both SciPy signatures are accepted
  (`callback(xk)` and `callback(intermediate_result)`).
- **scipy `Bounds` and `LinearConstraint` objects** — accepted alongside
  `(lo, hi)` pairs and constraint dicts; a `LinearConstraint` may carry a
  sparse `A`, which is honored. When all constraints are linear the objective
  Hessian is the Lagrangian Hessian, so an exact `hess` is used (no L-BFGS
  fallback).
- **scipy option spellings** as synonyms — `maxiter`→`max_iter`,
  `gtol`/`ftol`/`xtol`→`tol`, `disp`→`print_level`, `maxcor`→
  `limited_memory_max_history`; options may be passed as `**kwargs` (the legacy
  `options={…}` dict still works).
- The result is now a genuine `scipy.optimize.OptimizeResult` carrying the
  `nfev` / `njev` / `nhev` evaluation counters, with pounce extras under
  `res.info` and a back-compat shim so a key absent at the top level falls back
  to `res.info`.

**Changed:** the `solver_selection` default is now `"nlp"` (no structure
probing) — automatic LP/QP/QCQP routing is opt-in via `solver_selection="auto"`,
so a general NLP or an expensive `fun` pays no probe overhead. The `args`
argument is now the third positional parameter (matching SciPy), ahead of `jac`.

### Fixed — `obj_scaling_factor` was silently ignored (maximization diverged)

The `obj_scaling_factor` option was registered but never read: every solve
constructed the NLP with no-op scaling, so the documented behavior — a
constant multiplier on the objective, negative to **maximize** — was a silent
no-op and maximization problems diverged (the IPM minimized the unscaled
objective). The option value is now carried into `OrigIpoptNlp` on both the
IPM and SQP paths (`ConstObjScaling`), combining with gradient-based /
user scaling exactly as documented. Sensitivity analysis works under a
negative factor too: the natural-units correction from #128 below uses a
two-sided scaling with no square root, so `solve_with_sens` /
`Solver.reduced_hessian` return the declared problem's reduced Hessian for
maximization problems as well.

### Added — KKT regularization reported alongside sensitivity outputs

The IPM's inertia-correction perturbations are baked into the converged
factor in scaled space, so a regularized final factorization makes the
natural-units sensitivity outputs (covariance in particular) inexact and not
perfectly scaling-invariant. The final `(δ_x, δ_s, δ_c, δ_d)` are now
reported so workflows can check for the all-zero (exact) case:
`info["kkt_perturbations"]` and `Solver.kkt_perturbations` (Python),
`SensResult::kkt_perturbations` and `Solver::kkt_perturbations` (Rust).

### Fixed — sensitivity back-solves now return natural (unscaled) units (#128)

The reduced Hessian from `solve_with_sens(compute_reduced_hessian=True)` /
`Solver.reduced_hessian`, the parametric step `dx`, and the raw
`Solver.kkt_solve` were returned in the IPM's internally **scaled** space
whenever NLP scaling was active (the default
`nlp_scaling_method = "gradient-based"` fires when an objective gradient or a
constraint row exceeds 100 at the starting point). For a parameter-estimation
NLP this made `-inv(reduced_hessian)` differ from the true covariance by
`df / (dc_i·dc_j)` — the discretization-tracking "≈ nfe" fudge factor reported
in #128. The same scaled factor silently corrupted the factor-reuse VJP/JVP of
**both** differentiable frontends (`pounce.jax` `JaxProblem(factor_reuse=True)`
and `pounce.torch` `TorchProblem`) on badly-scaled problems.

The scaled primal-dual system is the two-sided diagonal scaling
`K_scaled = E·K_natural·F` (per-block: `E = (df, df/dd, dc, dd, df, df)` and
`F = (1, 1/dd, dc/df, dd/df, 1/df, dd/df)` over `x, s, y_c, y_d, z, v`), so
every held-factor back-solve now computes `K_natural⁻¹ = F·K_scaled⁻¹·E`: all
eight KKT blocks — including the bound-multiplier z/v rows in `dx_full` —
come back in the user's own units regardless of scaling method, and a
negative `obj_scaling_factor` is handled (no square root involved). The CLI
sIPOPT mode inherits the same correction: the `red_hessian` var-suffix output
is now natural-units where upstream sIPOPT prints a scaled value it warns
about. The pre-fix solver-space values and the factors stay accessible:
`info["reduced_hessian_scaled"]` / `info["obj_scaling_factor"]` /
`info["pin_g_scaling"]`, `Solver.reduced_hessian(..., scaled=True)`,
`Solver.kkt_solve(..., scaled=True)` / `kkt_solve_many(..., scaled=True)`,
the `Solver.nlp_scaling` dict, the C ABI's `IpoptSolverKktSolveScaled`, and
the matching Rust surfaces (`SensResult` fields,
`Solver::{compute_reduced_hessian_scaled, kkt_solve_scaled,
kkt_solve_many_scaled, nlp_scaling}`, `PdSensBacksolver::solve_scaled_space`).

Also fixed in the same change: `SensSolve` / `Solver` pin-constraint indices
are now mapped to KKT rows through the equality/inequality split
(`full_g_to_c_block`), so pins are selected correctly when inequality
constraints precede them in `g(x)` (previously the wrong row was used
silently; the CLI sIPOPT path already mapped correctly and now shares the
same helper). `pounce.curve_fit` no longer requires scaling to be off to
trust the converged factor for its covariance / data-sensitivity reads.

### Changed — C ABI: sensitivity entry points now return natural units (breaking)

Behavior change for C callers of the sensitivity ABI (`pounce-cinterface`).
`IpoptSolverReducedHessian`, `IpoptSolverParametricStep`, and
`IpoptSolverKktSolve` now return values in **natural (unscaled) units** as part
of the #128 fix above — previously, on a badly-scaled NLP (where the default
`gradient-based` method fires), they returned the IPM's internally scaled
values. A C caller that was compensating for the old behavior — e.g. passing a
non-`1.0` `obj_scal` to `IpoptSolverReducedHessian` to undo the `df / dc²`
factor by hand — will now get a doubly-corrected (wrong) result and must drop
that workaround; `obj_scal` is once again only the plain extra multiplier its
docs describe. Callers that want the old scaled values back have an escape
hatch **only** for the raw KKT solve: `IpoptSolverKktSolveScaled(..., scaled =
true)`. There is intentionally no scaled variant of `IpoptSolverReducedHessian`
or `IpoptSolverParametricStep` — the natural-units reduced Hessian and
parametric step are the only correct answers for a covariance / predictor read,
so the scaled forms are not re-exposed across the ABI (the Rust
`Solver::compute_reduced_hessian_scaled` remains for in-process calibrated
callers).

### Added — Batched NLP solving (`solve_nlp_batch`) (#126)

Solve N independent NLPs in parallel on a Rayon pool — the general-NLP
analog of `solve_qp_batch_parallel`, for parametric sweeps, multi-start,
MPC chains, and branch-and-bound node relaxations (each sibling node
differing only in tightened bounds).

- **Rust** — `pounce_algorithm::solve_nlp_batch` /
  `solve_nlp_batch_parallel`: one fully-equipped `IpoptApplication` per
  instance, built *inside* the worker via a `Sync` configure hook that
  receives the instance index (outer-parallel / inner-serial, like the
  QP batch; `install_serial_feral_backend` sets up the per-worker
  serial factor). Results return in input order with the captured
  final iterate, multipliers, and per-instance `SolveStatistics`.
- **`pounce-nl`** — CSE `Expr` sharing switched from `Rc` to `Arc`, so
  `NlProblem` / `NlTnlp` are `Send` and an owned evaluator can move to a
  worker. `NlTnlp` is now `Clone`, and `NlTnlp::variant` /
  `NlVariation` build per-instance bound / starting-point overrides on
  one parsed model (tapes are cheap to clone).
- **Python** — `pounce.solve_nlp_batch(problems, x0s=, options=,
  parallel=, warms=, share_structure=)`: native `NlProblem` inputs
  (from `read_nl` or the new `NlProblem.variant(...)`) solve in
  parallel with the GIL released. One `(x, info)` pair per input,
  `info` matching `Problem.solve`'s layout; `print_level` defaults to
  0 for the batch.
- **Phase 2: parallel callback batching** — callback-based
  `pounce.Problem` inputs also solve in parallel: each instance's
  bridge (Python callables + pre-resolved sparsity) moves to a rayon
  worker that owns the whole solve, re-acquiring the GIL transiently
  per `eval_*` callback. The GIL serializes only the Python share, so
  the speedup scales with the Rust/Python work ratio (~4x on 4 cores
  for an n=800 banded NLP with NumPy-vectorized callbacks; tiny
  callback-dominated problems won't speed up). Per-instance
  `add_option` settings are honored, with `options=` as a batch-level
  overlay; a raising callback degrades to that instance's failure
  without poisoning the batch.
- **Warm-started batches** — `solve_nlp_batch_parallel_warm` /
  `solve_nlp_batch_warm` (Rust) and `warms=` (Python, both input
  kinds): seed each instance from a previous result's iterate + duals
  and thread the converged barrier μ into `mu_init`
  (`warm_start_init_point=yes` forced; dimension mismatch falls back
  to a cold start). Re-solving a perturbed 24-instance `.nl` sweep
  warm cut total iterations 482 → 120.
- **Identical-sparsity structure sharing** — `FeralBackendPool` /
  `install_pooled_serial_feral_backend` (Rust) and
  `share_structure=True` (Python): opt-in per-worker backend pooling
  so FERAL's pattern-fingerprint symbolic cache (ordering + supernode
  structure) carries across batch instances instead of being rebuilt
  per instance. Always correct (pattern changes re-analyze); results
  are within solver tolerance of — not guaranteed bit-identical to —
  fresh-backend solves, which is why it is opt-in. Cross-*thread*
  symbolic sharing stays future work (needs the `BackendPool`
  ownership refactor documented in `dev-notes/backend-pool-resolve.md`
  or a feral-side symbolic export API).

### Added — PyTorch frontend for the differentiable solver (`pounce.torch`)

A PyTorch frontend mirroring `pounce.jax`: a solve is a
`torch.autograd.Function` you can drop inside a learned model and backprop
through, with the same constraint-satisfaction guarantee. This is a thin
adapter, not a second solver — the Rust IPM core and the
implicit-function-theorem backward are framework-agnostic; only the array
namespace differs. Because PyTorch is eager, the adapter is smaller than the
JAX one (no `pure_callback` / `ShapeDtypeStruct`, no host-callback registry or
single-thread executor pin), and float64 is requested per tensor rather than
via a global flag.

Surface (parity with `pounce.jax`):

- `from_torch` — build a `Problem` from `torch.func`-traced `f` / `g`
  (`grad` / `jacrev` / `jacfwd` / `hessian`; CPR colored AD for `sparse=`,
  via the shared detection/coloring helpers now in `pounce._ad_common`).
- `solve` / `solve_with_warm` — `autograd.Function` + KKT implicit-diff
  backward, with dual + barrier-μ warm-start threading.
- `vmap_solve` / `vmap_solve_parallel` — sequential / threadpool batches.
- `TorchProblem` — build-once handle with k_aug-style factor-reuse backward
  (`Solver.kkt_solve_many`), stacked block-diagonal batched solve, and the
  anchor / sensitivity / jvp_from_state / vjp_from_state / active_set_margin
  post-solve API.
- `solve_qp` / `solve_qp_batch` / `solve_socp` / `QpLayer` — OptNet-style
  differentiable conic layers (feasible-by-construction).
- `PathFollower` / `inverse_map_rhs` — predictor–corrector path following.

PyTorch is an optional dependency (`pip install pounce-solver[torch]`,
torch ≥ 2.2). See the [PyTorch integration guide](docs/src/python.md).

### Added — Convex QCQP auto-routes to the conic (SOCP) solver

The `auto` router now recognizes a convex **quadratically-constrained QP** and
sends it to the `pounce-convex` conic interior-point solver instead of the
general NLP path. Each convex-quadratic inequality `½xᵀHx + aᵀx + b ≤ 0`
(`H ⪰ 0`) is reformulated to one second-order cone (`H = FᵀF`, via a pivoted
rank-revealing Cholesky so a rank-deficient `H` yields the minimal cone), solved
alongside the QP objective and linear constraints, and its dual is mapped back
to a per-constraint multiplier. Works on both surfaces:

- **CLI** — a convex-QCQP `.nl`/Pyomo model routes automatically; force it with
  `solver_selection=socp` (errors if the problem is not a convex QCQP).
- **Python** — `minimize()` probes each constraint's Hessian at an anchor plus
  held-out points, validates the fitted quadratic before trusting it, and routes
  only when the feasible set is provably convex (a scipy `ineq` `g(x) ≥ 0` must
  be concave); otherwise it falls back to NLP. `options={"solver_selection":
  "socp"}` forces the conic path.

This closes the long-standing "conic solver: future" gap in the routing docs —
the conic solver shipped in 0.4.0 but was not reachable from either router for
quadratic *constraints*. See [LP / QP Solver Routing](docs/src/lp-qp-routing.md)
and [Choosing a Solver](docs/src/choosing-a-solver.md).

### Added — CLI knobs for the convex IPM and active-set QP solvers (#134)

Previously hard-coded solver defaults are now registered CLI options:

- **Convex IPM** (`solver_selection=lp-ipm` / `qp-ipm` / `socp`): `qp_tau`,
  `qp_reg`, `qp_infeas_tol`, `qp_hsde`, `qp_equilibrate`, `qp_crossover`.
  Each is forwarded only when explicitly set, so engine defaults are otherwise
  preserved.
- **Active-set QP** (`solver_selection=qp-active-set`): `sqp_qp_feas_tol`,
  `sqp_qp_opt_tol`, `sqp_qp_max_iter`, `sqp_qp_elastic_gamma`,
  `sqp_qp_anti_cycling` (`expand` / `bland` / `none`).

Both families are documented in `dev-notes/lp-qp-routing.md`. The LP crossover
default is now **off** (opt in with `qp_crossover=yes`): crossover-on regressed
the LP suite 3×–800× without reaching an exact vertex on the GEN family — the
machinery is unchanged, only the default flips.

### Fixed — `solver_selection=socp` was rejected; debugger silently no-op'd on `qp-active-set`

Two reachability gaps around the new convex solvers, found while confirming the
interactive debugger works on every backend:

- `solver_selection=socp` was a documented value (the conic IPM that a convex
  QCQP reaches under `auto`) but was missing from the option's registered
  allow-list, so forcing it failed with `Invalid value "socp"`. It is now
  accepted and routes a convex LP/QP/QCQP to the SOCP conic IPM.
- The interactive debugger (`--debug` / `--debug-script` / `--debug-json`) is a
  pdb-for-the-IPM. It engages correctly on `lp-ipm`, `qp-ipm`, and `socp`, but
  the active-set SQP engine (`qp-active-set`) has no such hook, so a debug
  request there used to run to completion without ever pausing — a silent
  no-op. It now prints an explicit note that the debugger is IPM-only and points
  at `qp-ipm` for interactive convex-QP debugging.

### Added — exact-vertex LP crossover (revised simplex)

`pounce-convex` gains a revised-simplex crossover that purifies a near-optimal
interior LP iterate to an exact optimal vertex, pivoting one variable at a time
on feral's unsymmetric sparse LU with Bland's anti-cycling rule (so it walks
through the highly degenerate NETLIB GEN vertices where the active-set bridge
stalls). It is tried first and accepted only when the KKT error does not
regress; on any breakdown it falls back to the legacy active-set bridge. Opt-in
via `qp_crossover=yes`.

### Fixed — convex LP/QP robustness on rank-deficient and large-scale data

Interior-point hardening in the dedicated convex (`lp-ipm` / `qp-ipm`) path:

- **Adaptive equality-block regularization (δ_c)** (#133). The
  equality-multiplier `(y,y)` block was frozen at a static value; on
  rank-deficient equality Jacobians that leaves a near-singular saddle and the
  solve plateaus to max_iter. δ_c now seeds from a μ-scaled base
  (`1e-8·μ^0.25`) and escalates on singular factorization / wrong KKT inertia /
  un-refinable direction probes, resetting each iteration so one hard iterate
  never inflates regularization for the rest of the solve. Regression-clean on
  NETLIB; drops the rank-deficient GEN constraint violation 9.3e-5 → 3.1e-8.
  (GEN still floors just above the 1e-8 tolerance, so #133 stays open.)
- **Scale-gated relative stopping + ratio-based infeasibility ray**. The
  absolute KKT stop is unreachable once the data scale pushes the
  finite-precision residual floor above tol — POWELL20 / BOYD1 / BOYD2 /
  QFORPLAN / QSHELL (scale 7e9–4e12) ran to max_iter despite being optimal to
  ~1e-9 relative. A Clarabel-style scale-relative residual now relaxes the
  absolute test, but only once roundoff sits below tol, so well-scaled problems
  are unaffected. Infeasibility now triggers on the ratio κ/τ→∞ rather than a
  bare τ floor, fixing a false `PrimalInfeasible` on feasible large-norm-x QPs.

### Fixed — active-set QP cycling on degenerate phase-1 (#133)

The active-set QP elastic phase-1 recovery (ℓ₁-infeasibility minimization,
γ=1e6) is inherently highly degenerate and could cycle under the default
anti-cycling rule (NETLIB `afiro` bailed at iteration 0). The phase-1 solve now
runs under Bland's rule (provably finite), and `solve_general` latches into
Bland after 50 consecutive non-improving iterations as a sticky,
scale-invariant anti-stall safety net.

### Changed — large sparse convex QPs are now recognized as convex (faster routing)

Problem classification certifies a coupled quadratic Hessian as PSD via feral's
sparse LDLᵀ inertia (~O(nnz·fill)) instead of a dense Jacobi eigensolve, so
large-but-sparse convex QPs (the CVXQP family, n≈1000) that previously fell back
to the general NLP solver are now sent to the dedicated convex path.

### Added — limited-memory update type & history honored on the IPM path (#131, #132)

`limited_memory_update_type` and `limited_memory_max_history` were registered
but read nowhere on the interior-point path (the updater was hard-wired to
Powell-damped BFGS). Both are now threaded through to the limited-memory
updater. The default is unchanged (`bfgs`, history 6 — bit-exact with Ipopt), so
there is no behavior change unless set; `sr1` (which can represent negative
curvature) is now selectable and rescues ill-conditioned nonconvex objectives
where damped BFGS hides indefiniteness from the inertia check.


## [0.4.0] — 2026-06-05

### Added — Convex / conic solver (`pounce-convex`; `solve_qp` / `solve_socp`)

POUNCE is no longer NLP-only: a new pure-Rust convex interior-point solver
(`pounce-convex`) handles **LP, convex QP, SOCP, and PSD / exp / power cones**,
solving each to a **global** optimum (a convex problem has no other kind). It
uses a homogeneous self-dual embedding (HSDE) — symmetric for the self-dual
cones and a non-symmetric driver for the exponential/power cones — over a
`Cone` abstraction (`nonneg`, `soc`, `psd`, `exp`, `power`, plus composite and
chordal decompositions for sparse SDPs). Convex solvers extract the constant
`P`, `A`, `c`, `b` data once at setup rather than re-evaluating per iteration,
and share the `pounce-linsol` / `pounce-linalg` factorization substrate with the
NLP path. Python entry points are typed (not SciPy-shaped, by necessity — a cone
program is *data*, not a callable): `solve_qp(P, c, A, b, G, h, lb, ub, …)`,
`solve_socp(…, cones=…)`, plus `solve_qp_batch` / `solve_qp_multi_rhs` for
batched factor reuse, and a reduced-Hessian sensitivity API. The CLI reads conic
instances from CBLIB / `.cbf` (including PSDCON / HCOORD / DCOORD SDP blocks).

### Fixed — Convex LP/QP reported objective dropped tree-folded constant

The convex LP/QP path (`solver_selection=lp-ipm` / `qp-ipm`) reported an
objective off by the objective's constant term whenever AMPL/Pyomo folded that
constant into the **nonlinear objective tree** (the `+9` of `(x-3)²`) rather
than the `.nl` linear-section constant. The quadratic-form extractor
(`analyze_quadratic_full`) discarded the degree-0 term — correct for the
*minimizer*, wrong for the *reported value* — so e.g. `HS21` reported `0.04`
instead of `−99.96` and `HS35` `−8.889` instead of `0.111`. The extractor now
returns that constant and the convex driver adds it to the reported objective
alongside `obj_constant`; the optimal point was always correct. Caught by a
head-to-head NLP-vs-convex run over the Maros-Mészáros QP and NETLIB LP suites
(`benchmarks/nl_compare_nlp_vs_convex.md`).

### Fixed — Convex LP/QP IPM stalled on badly-scaled NETLIB LPs

The static KKT regularization `δ` (added on the reduced KKT diagonal so the
LDLᵀ has a stable inertia) was `1e-8`, large enough to **floor the achievable
primal residual** at `δ·‖dy‖`: with a full Newton step `A·dx = −r_p + δ·dy`, so
on instances with large equality multipliers the primal infeasibility cannot
fall below `δ·‖dy‖`. On NETLIB `adlittle` (`‖dy‖ ≈ 4e8`) this froze `inf_pr`
near 4 and the LP IPM ran to its iteration cap, returning a wrong objective
(`439665` vs the published `225494.96`). Lowering the default `δ` to `1e-10` —
still strictly positive, so the system stays quasi-definite — clears the floor:
`adlittle` now converges in ~57 iterations to the optimum, `stocfor1` speeds up
(139 → 71 iters), and the rest of the LP/QP suites are unchanged (the QP suite
is bit-identical). The whole `1e-9‥1e-11` band converges the benchmark suites;
`1e-10` is centered in it.

Also: the convex IPM's opt-in iteration trace now records a **terminal record at
the converged iterate** (the NLP path's N+1 convention), so the trace always
ends at the optimum instead of at the last pre-step state — previously a solve
that converged in a single step left only the cold-start record in the trace.

### Added — SOS polynomial global optimization (`sos_minimize`)

`sos_minimize(objective, *, inequalities, equalities, …)` computes **certified
global** lower bounds for polynomial optimization via a sum-of-squares /
Lasserre relaxation (Putinar localizing multipliers for constraints), built on
the new PSD cone. When the relaxation is exact it extracts the global
minimizer(s) with an exactness certificate (multi-atom extraction without a
non-symmetric eig, plus facial reduction for degenerate solves).

### Added — Multi-backend interactive debugger (convex/conic IPM)

The interactive debugger was generalized over a `DebugState` trait so one REPL
drives the convex solver as well as the NLP loop. New backend: a
**convex/conic** debugger (`pounce_cblib --debug`, wired through the symmetric
and non-symmetric HSDE drivers), exposing the same checkpoints and commands as
the NLP path. This composes with the 0.4.0 debugger features below (quote-aware
tokenization, `ask` provider presets, `--debug-json` protocol, Ctrl-C escape
hatch).

### Added — `pounce.curve_fit` (Python)

A `scipy.optimize.curve_fit`-style nonlinear fitter on top of the
interior-point solver, returning much more than `(popt, pcov)`:

- parameter covariance, standard errors, and Student-t confidence intervals
  read pounce-natively from the converged factor's reduced Hessian
  (`pcov = 2·s²·inv(H_S) = s²·(JᵀJ)⁻¹`; matches scipy / `pycse.nlinfit`). The
  t-quantiles use scipy when present and an accurate scipy-free inverse-t
  (via the inverse regularized incomplete beta) otherwise, so the CIs are
  correct on a numpy-only install even for small samples;
- a smooth (C²) loss family — ordinary/weighted least squares plus robust
  Cauchy and a smooth pseudo-Huber, exposed under both `soft_l1` and `huber`
  (the same C² loss: a true piecewise Huber is only C¹, which the IPM can't
  use), with a sandwich covariance estimator (non-smooth L1/MAE is
  intentionally out of scope for the IPM);
- parameter constraints scipy can't express — positivity/negativity/ranges
  via `bounds`, and relations between parameters via `constraints=`; an active
  bound/constraint yields a covariance projected onto the free subspace;
- data sensitivity `dpopt/ddata` (∂params/∂data) from a single batched
  back-solve against the same factor (`Solver.kkt_solve_many`);
- a `CurveFitResult` with `predict()`, `confidence_band()` (both `confidence`
  and `prediction` kinds, heteroscedastic-aware), `correlation`, R²/χ²/dof,
  and `summary()`.

Derivatives resolve analytic `jac` → JAX autodiff (the default for
`jax.numpy` models) → a finite-difference fallback; exact derivatives let the
solve converge cleanly with scaling off, which is what makes the
factor-based covariance and sensitivity exact. Docs:
`docs/src/curve-fitting.md`; notebook `python/notebooks/18_curve_fit.ipynb`.

`p0` is now optional even without bounds: when omitted, the parameter count is
read from the model signature and the starting point is chosen data-drivenly
(a bound-aware, data-scale candidate sweep scored by the objective) instead of
defaulting to a flat vector of ones — so badly-scaled problems get a far better
seed, while `ones` (clipped into the bounds) is always among the scored
candidates so the choice is never worse than the old default.

### Added — `pounce.curve_fit_minima` (Python)

`curve_fit_minima` finds **multiple** parameter sets that each explain the
data, for the non-convex problems where one fit isn't the whole story
(peak-assignment ambiguity, frequency aliasing in sinusoids, amplitude/decay
trade-offs in sums of exponentials, sign/label symmetry, …).

- drives `pounce.find_minima` over the *very same* fitting objective as
  `curve_fit` — identical `sigma` weighting, robust `loss`, `f_scale`,
  `constraints`, and resolved Jacobian — so the enumerated minima are true
  optima of the actual fit, not a separate surrogate;
- reuses the model Jacobian as the search **gradient** and the Gauss-Newton
  matrix as the search **Hessian**, which sharpens the basin escapes and lets
  `find_minima` certify each point as a minimum (rejecting saddles);
- refines every distinct minimum into a full `CurveFitResult` (covariance,
  CIs, optional `dpopt/ddata`) and returns them ranked by SSE, best first;
- the `method`, `n_minima`, `max_solves`, `patience`, `dedup`, `seed`, and
  `find_minima_kw` arguments pass straight through to `find_minima`; finite
  `bounds` define the box it samples / repels within. Docs:
  `docs/src/curve-fitting.md`.

### Added — `pounce verify` subcommand + signed receipts

A `verify` subcommand that re-derives feasibility from the canonical `.nl`
rather than trusting a `.sol`'s status line or the solver/agent that produced
it — the trust anchor when pounce is a tool an agent calls: the agent
proposes a solution, a small deterministic checker disposes.

- `pounce verify <problem.nl> <claim.sol>` evaluates `g(x*)` and bounds
  against the canonical model, reporting the worst constraint/bound violation
  and (when the `.sol` carries duals) a bound-projected KKT stationarity
  residual. Exit 0 = VERIFIED, 20 = REJECTED, 2 = usage/IO. Feasibility
  gates; optimality is informational unless `--require-optimal`.
- The JSON receipt content-addresses both inputs by SHA-256 (zero new deps);
  with `POUNCE_VERIFY_KEY` set it signs the receipt with HMAC-SHA256 over a
  float-free preimage so any language can re-derive it.
- MCP `verify_solution` tool plus dependency-free `verify_sig` helpers and a
  stdlib reference signer service.

The check itself (recompute feasibility against the model + a content-addressed
receipt) is ready to use and needs no secrets; the signing / remote-authority
layer is an explicit proof of concept. Docs: `docs/src/verify.md`.

### Added — Debugger `load` / `sweep` / `multistart`

The interactive solver debugger gained three commands for seeding solves
from externally-computed points and for initialization-sensitivity
diagnostics:

- `load <file> [block]` — the inverse of `save`. Reads a block (default
  `x`) into the live iterate from either a `save` artifact (JSON; every
  block present is loaded) or a plain numeric file
  (comma/whitespace/newline-separated). The many-variable escape hatch:
  generate a start once (`numpy.savetxt`) and `load` it instead of typing
  it. A loaded `x` becomes the seed for the next step / `resolve`.
- `sweep <file>` — run one full solve per start in a file (one per line),
  then tabulate each terminal status / objective, count distinct minima
  (objectives clustered to a relative `1e-6`), and flag the best solve.
- `multistart <N> [rel]` — `N` solves from sampled restarts: each variable
  with a finite box `[x_Lᵢ, x_Uᵢ]` is drawn **uniformly in that box**;
  unbounded variables fall back to a relative jitter `±rel·(|xᵢ|+1)`
  around `x`. Start 0 is the unperturbed point; deterministic (fixed-seed
  PRNG), so runs reproduce. Backed by a new `DebugCtx::var_bounds()` that
  reconstructs full-length algorithm-space bounds (post-scaling, with `±∞`
  for absent bounds) from the NLP's reduced bound vectors + expansion
  matrices.

Tab completion now also covers **filesystem paths** (after
`load`/`sweep`/`save`/`source`, with a trailing `/` on directories) and
block names for `load`'s optional second argument — available both at the
REPL Tab key and via the programmatic `complete` command.

**Ctrl-C at the prompt** is now a working escape hatch: the first press
cancels the current input line (readline convention), a second in a row
stops the solve (a clean `UserRequestedStop`) — mirroring the running-mode
double-tap, so two Ctrl-Cs always exit whether running or paused.

And a little something for the 2am debugging sessions: an undocumented
`coffee` command at the prompt. ☕

Both sweep commands build on the existing re-solve machinery and keep each
solve's trajectory observable (breakpoints/events still fire inside a
sweep). JSON mode emits `sweep_result` per solve and a final
`sweep_summary`; `hello.capabilities` advertises `load` and `sweep`. For
automated global search with dedup and minimum certification, the Python
`find_minima` remains the production path. Docs: `docs/src/debugger.md`
(new "Multi-start and initialization sensitivity" section + scripting
examples).

### Added — Sparse (colored) AD for the JAX front-ends (`sparse=`)

`from_jax` and `JaxProblem` gained a `sparse=True` flag that computes the
constraint Jacobian and the Lagrangian Hessian with CPR-style colored AD
— one JVP/HVP per color (`k ≪ n` colors) scattered back to the detected
nonzeros — instead of materializing the dense matrix and slicing it
(pounce#83). Per-iteration derivative cost drops from `O(n)` to `O(k)`
AD passes on genuinely sparse problems; benchmarked on a banded family at
~560× (Jacobian) / ~200× (Hessian) per eval and 7.6× faster full solve
by `n=2000`. When the sparsity pattern is **value-independent** (any
composition of smooth pointwise ops) the reported structure, values, and
solutions are identical to the dense path; the differentiable backward is
unaffected. For **value-dependent** structure (`where` / `abs` / branches) a
random probe can miss a nonzero, and under compression a missed entry aliases
into a same-colored reported entry — silently wrong derivatives — so such
models should hand-specify the pattern via the `Problem` API or stay on the
dense path. Dense problems see a small bounded overhead, so the flag is opt-in.

- Forward/reverse mode selection (`jacfwd` when `n < m`, else `jacrev`)
  for the dense path / sparsity probe.
- Multi-probe sparsity detection (`n_probes=`, default 3 under
  `sparse=True`, 1 otherwise) unions several random probes to harden
  against value-dependent structure.
- Benchmark: `python/benchmarks/bench_sparse_ad_83.py`. Docs:
  `docs/src/python.md` (JAX integration → "Sparse Jacobian/Hessian
  compression").

### Added — Interactive solver debugger (`--debug` / `--debug-json`)

A "pdb for the interior-point loop." `pounce <problem> --debug` opens a
branded REPL that pauses the solve to inspect and **mutate** live state;
`--debug-json` speaks a newline-delimited JSON protocol so an LLM agent,
a script, or a visual debugger (VS Code DAP / webview) can drive it.
Full guide in `docs/src/debugger.md`. Zero effect on the solve when not
attached.

- **Checkpoints & stepping:** pauses at `iter_start`, the sub-iteration
  phases (`after_mu` / `after_search_dir` / `after_step`), `step_rejected`
  (line search gave up, before restoration), around restoration
  (`pre_/post_restoration_entry/exit`), and `terminated`.
  `step` / `stepi` / `continue` / `run N` / `stop-at <cp>` / `detach` /
  `quit`. The same debugger **steps into the restoration inner IPM**
  (pauses flagged `in_restoration`).
- **Breakpoints:** by iteration (`break N`, one-shot `tbreak N`),
  conditional with `&&`/`||`, on a solver **event** (`break on
  regularized|resto_entered|tiny_step|ls_rejected|mu_stalled|nan`), and
  **watchpoints** (`watchpoint x[3]`). `commands N …` auto-runs a list on
  hit.
- **Inspect:** `info`; `print` of blocks, search-direction blocks (`dx`),
  scalars (`mu obj inf_pr inf_du err compl iter`), `kkt` (inertia +
  regularization), and `active`; `watch`/`display`; `diff`.
- **Named-equation diagnostics:** `print residuals` labels primal/dual
  residuals with their original `.nl` constraint/variable names; `print
  equation <name|row>` renders the source algebra of a named constraint
  (by model name or `.nl` row index); `print rank` reports the SVD
  numerical rank of the equality Jacobian J_c and names the implicated
  rows. `diagnose` (alias `diag`) runs a panel of heuristics over the
  current iterate and emits a **named** health report — *"the worst
  constraint residual is c[mass_balance]"* rather than *"row 13 is
  infeasible"* — the live counterpart to the `pounce-studio` `diagnose`
  tool.
- **Mutate / what-if:** `set mu`, `set x[i]`, `set opt`; `goto`/`restart`
  (soft rewind) and `resolve` (re-solve from the current point).
- **Visualize:** `viz kkt`/`viz L`/`viz <block>` open via `pounce-dbg-viz`
  — an interactive Plotly viewer (spy/heatmap for the KKT matrix & LDLᵀ
  factor, bars for vectors); `save` dumps the iterate. `pip install
  'pounce-solver[viz]'`.
- **Attach & drive:** `--debug-on-error` (post-mortem), `--debug-on-
  interrupt` / Ctrl-C / in-band `{"cmd":"pause"}` (async pause),
  `--debug-script` / `source`, option discovery + Tab completion, `ask`
  (consult an LLM about the paused state; provider-selectable via
  `$POUNCE_DBG_LLM` = `claude` / `codex` / `gemini` / `llm` or a custom
  command template, default Claude Code), and a branded REPL banner
  reusing the project wordmark with a command cheat-sheet.
- **JSON protocol:** `hello` → `pause` → `result` (with `request_id`) →
  `progress` → `terminated`. Engine in `pounce-algorithm::debug`; front
  end in `pounce-cli::debug_repl`.
- **MCP live-debug proxy:** `pounce-studio` exposes the debugger over the
  Model Context Protocol (`debug_start` / `debug_command` / `debug_state`
  / `debug_sessions` / `debug_close`), proxying the `--debug-json`
  protocol so an MCP client can start, drive, and inspect a live solve.

### Added — `read_nl` / `NlProblem` (Python)

`pounce.read_nl(path)` loads an AMPL `.nl` file through pounce's own reader
and returns an `NlProblem` exposing the model's `objective`, `gradient`,
`hessian`, and constraint `jacobian` at any point — the same evaluation
pipeline the solver uses, available standalone for inspection, finite-
difference checks, or feeding another tool. Exported from `pounce`
(`read_nl`, `NlProblem` are in `__all__`).

### Added — Expanded `.nl` opcode coverage

The `.nl` reader now handles conditional/logical opcodes (`if-then-else`,
comparisons), the n-ary list reducers `o11` (MINLIST) / `o12` (MAXLIST), and
the remaining smooth transcendentals (inverse and hyperbolic trig). Models
that previously failed to load with an "unsupported opcode" error now parse,
with FD-verified first/second derivatives on the smooth interior.

> `min`/`max`/`if-then-else` are **non-smooth**: at a kink the gradient is a
> subgradient and the Hessian misses the kink curvature, so an iterate landing
> on or oscillating across the switch can stall the interior-point solve. The
> inverse-trig opcodes (`asin`/`acos`/`atanh`/`acosh`) have **bounded domains**
> whose derivatives blow up at the edge — bound such variables away from the
> boundary. The reader accepts these models; convergence is on you.

### Added — `pounce --cite` and `--minima`

- `pounce --cite [REPORT.json]` lists the citations to use for pounce (and,
  when a solve report is given, any method-specific references it triggered,
  e.g. the Byrd restoration paper). `--bibtex` emits ready-to-paste entries.
- `pounce <problem> --minima` runs the multistart global search from the CLI
  with full `find_minima` parity (method, `n_minima`, dedup, seed).

### Changed

- **Default solver trajectory** moved on several fronts as the interior-point
  method was brought closer to IPOPT. These change which iterates are visited
  (and, on a few problems, the iteration count) but not the math being solved:
  - the barrier parameter `μ` is now updated *inside* the monotone reduction
    loop, so the relaxed-complementarity error reflects the current `μ`. Net
    +2 problems reach Optimal on the internal `.nl` sweep, at a ~2.7% total
    iteration-count cost and a regression on `deconvb` / `gausselm`;
  - under the watchdog, the line search bypasses the acceptor's `alpha_min`
    floor (mirrors IPOPT) so the full-step watchdog trial actually runs;
  - the IPOPT safe-slack bound-adjustment mechanism (`slack_move`) is ported
    and active by default;
  - NLP gradient-based scaling now lifts fixed variables to their value before
    sampling, so the computed scale factors match the operating point.
- **Auto-retry on local infeasibility (default on).** New option
  `feral_infeasibility_scaling_retry` (default `yes`): when a solve ends in
  `Infeasible_Problem_Detected` under a non-MC64 effective scaling, pounce
  re-solves once with `feral_scaling=mc64` (main IPM and restoration sub-IPM).
  This rescues problems where a backward-stable scaling choice lands in a
  spurious infeasible basin under sensitive dependence (`discs.nl` is the
  canonical case); every individual solve along both trajectories is itself
  backward-stable, so an a-priori scaling router can't distinguish them. Set
  to `no` to restore the single-solve behavior.
- **New option `feral_scaling`** (default `auto`, mirrors `feral_ordering`):
  pins FERAL's diagonal KKT scaling strategy; also settable via the
  `POUNCE_FERAL_SCALING` env var.
- **Dependency:** `feral` pinned to crates.io `0.10.0` (was a git rev),
  bringing AMF ordering by default and MC64 inertia-guided scaling fallback.
- **Internal:** the `.nl` pipeline was extracted into a new leaf crate
  `pounce-nl` (re-exported from `pounce-cli`; no public API change).
- **`pounce-studio-mcp` → 0.1.0** (versioned independently of the `0.4.0`
  core): the MCP server graduated from its `0.0.1` spike to its first
  functional release — analyze / run / explain / citations tools, GAMS
  problem tools, a live debug-session proxy, and PyO3 backing via
  `pounce-studio-core`.

### Fixed

- **Windows build:** the debugger's `SIGINT`-to-break handler referenced
  `nix::sys` / `nix::libc`, which the (Unix-only) `nix` crate does not expose
  on Windows, breaking the `pounce-cli` build there. The POSIX handler is now
  `#[cfg(unix)]`-gated with a no-op `install()` stub elsewhere; the rustyline
  prompt's Ctrl-C double-tap remains the cross-platform escape hatch.
- **`.sol` banner no longer goes stale:** the `parse_sol` round-trip test
  fixture derived its `POUNCE <version>:` message from a hardcoded literal,
  which silently drifted on each release (it was still `0.3.1`). It now
  reads `CARGO_PKG_VERSION`, like the production writer always has, so the
  fixture self-updates and never needs a manual bump.
- **Restoration:** the limited-memory (L-BFGS) Hessian is now built in the
  iterates' native space, fixing a space mismatch on compound problems (#102);
  the cycle detector rolls back to the last acceptable point instead of
  erroring out when a usable iterate exists.
- **KKT:** the negative-eigenvalue cache is refreshed on `WrongInertia` /
  `Singular` outcomes (not only `Success`), matching IPOPT's inertia
  pass-through so δ_c regularization routing stays live near a singular KKT (#99).
- **`find_minima`:** the in-bounds test uses a bound-magnitude-relative
  tolerance so large-scale boxes aren't spuriously rejected (#101); MLSL is
  bounded by a sample budget so it always terminates instead of looping when
  its clustering filter rejects every sample (#103).
- **Bounds length is validated up front** across `minimize`, `find_minima`,
  `find_saddles`, `find_critical_points`, `reaction_network`, and `curve_fit`.
  A `bounds` list whose length didn't match the variable/parameter count used
  to fail silently — a too-short list left trailing variables unbounded, and in
  the sampling-based searches a length-1 box could *broadcast* across every
  dimension (sampling all of them from variable 0's interval). It now raises a
  clear `ValueError` immediately, like scipy; `curve_fit`'s scipy-style
  `(lo, hi)` tuple form is likewise checked so array sides must be scalar or
  length-`n_params`.
- **Input validation hardened** so imperfect-but-plausible arguments raise a
  clear `ValueError` up front instead of failing cryptically deep in the solve:
  - `minimize` / `find_minima` / `find_saddles` now **promote a scalar / 0-d
    `x0` to 1-D** (matching scipy), so `minimize(f, 1.5)` works instead of
    raising `iteration over a 0-d array`;
  - a **reversed bound** (`low > high`) is rejected instead of silently
    producing an infeasible box (a fixed `low == high` is still allowed);
  - **malformed constraint dicts** (not a dict, or missing `type` / `fun`, or a
    non-callable `fun`) raise a descriptive error instead of a bare `KeyError`;
  - `curve_fit` validates its data and weights: `xdata`/`ydata` length must
    match and be non-empty and finite, `sigma` must be positive and finite,
    `f_scale` must be positive and finite, and an explicit `p0` must have one
    start per model parameter — each previously surfaced as a `LinAlgError`,
    `ZeroDivisionError`, back-solve `RuntimeError`, broadcast error, or a
    silently wrong fit;
  - a model with **keyword-only parameters** (`f(x, *, a, b)`) — which
    `curve_fit` cannot call positionally as `f(x, *params)` — is rejected with
    a clear message instead of a downstream `TypeError`;
  - `CurveFitResult.confidence_band` checks that `x` has the same
    dimensionality as the fitted `xdata` and that a prediction-band `sigma` is
    scalar or matches `x`, replacing a cryptic einsum/broadcast error;
  - `find_minima` / `find_saddles` reject a sub-1 `n_minima` / `n_saddles` /
    `patience` / `max_solves`, and `find_saddles` rejects a Morse `index`
    outside `[1, n]` (which previously sliced the step vector wrong and found
    the wrong critical points).

## [0.3.0] — 2026-06-02

### Added — Multiple-minima & critical-point global search (PR #94)

`pounce.find_minima(fun, x0, n_minima=..., method=...)` returns several
distinct local minima from a single call. Methods: `flooding` /
`deflation` (add a repulsive Gaussian / pole "hump" — with analytic
gradient and Hessian — at each found minimum and re-solve), `multistart`,
`mlsl`, and `basinhopping`. Anisotropic auto bump widths and a
curvature-derived auto amplitude; Hessian-based saddle rejection; global
box restarts; bounds and constraints pass through `minimize` untouched.
The six-hump-camel demo recovers all six minima.

- `pounce.find_critical_points` / `pounce.find_saddles` — stationary
  points via the squared-gradient merit `½‖∇f‖²`, classified by Morse
  index from the Hessian eigenvalues (degenerate / non-Morse points are
  *flagged*, not mislabeled); eigenvector-following saddle search with
  box-clipped steps.
- `pounce.reaction_network` — minima, transition states, and barriers
  (Müller–Brown example).
- Robustness: non-finite candidates and objectives are rejected before
  acceptance; the de-duplication metric is the same per-dimension scaled
  distance across the minima and saddle routes.
- Examples (`gaussian_hump_minima.py`, `critical_points.py`,
  `reaction_barrier.py`), notebooks 15–17, docs (`find-minima.md`,
  `find-minima-choosing.md`), and 29 unit tests.

### Fixed

- **Acceptable-point termination now rejects a non-finite objective.** A
  near-feasible iterate whose objective evaluates to NaN/Inf (e.g. CUTE
  `himmelbj`) exits `Invalid_Number_Detected` instead of a spurious
  `Solved_To_Acceptable_Level` carrying a `nan` objective; the benchmark
  driver's objective scrape is null-safe to match.
- **No spurious `jacobian()` call on unconstrained problems**
  (`pounce-py`). `eval_jac_g` short-circuits when there are no constraint
  entries (mirroring the Hessian guard), so the unconstrained `minimize`
  facade — which legitimately omits `jacobian` — no longer logs an
  `AttributeError` at ERROR level on every iteration.

### Benchmarks & docs

- The benchmark suite now runs **single-threaded and sequential** by
  default (`OMP/VECLIB/RAYON=1`) so POUNCE and Ipopt solve times are
  directly comparable on one host; the ipopt-ma57 reference was
  regenerated and `BENCHMARK_REPORT.md` carries a threading/timing note
  (POUNCE's `faer`/`rayon` dense linear algebra is up to ~2× faster
  multi-threaded, so the single-threaded numbers are a controlled lower
  bound).
- Example notebooks re-executed against the current solver and corrected
  where the prose had drifted: warm-starting now demonstrates the
  `mu_init` + tight `warm_start_*_bound_push` tuning that actually cuts
  iterations (11 → 5 on HS071); the scaling example uses a `1e12`
  constraint where gradient scaling visibly helps (31 → 11 iters); the
  FBBT notebook shows the domain-safety and false-infeasibility wins.
- mdBook restyled in the POUNCE tiger / cream brand palette.

### Added — Inverse-map ODE recipe over a sensitivity RHS (pounce#91)

`pounce.jax.inverse_map_rhs(jp, dy_ds, *, output=None, x0=None)` builds
the right-hand side of the Alves–Kitchin–Lima inverse / uncertainty
mapping ODE (pounce#84, Eq. 3):

```
dθ/ds = (∂y/∂θ)^{-1} · dy/ds
```

where ``y = output(x*(θ), θ)`` is an output of the embedded optimizer.
POUNCE supplies the RHS; an off-the-shelf adaptive integrator (diffrax,
scipy) does the stepping — *no NLP inversion*.

- The inverse map is a **linear solve against** the total output
  sensitivity ``∂y/∂θ = (∂h/∂x) J + ∂h/∂θ`` (with ``J = ∂x*/∂θ`` from
  the held factor), not a Jacobian-vector product — so it wants the full
  ``J`` and ``jnp.linalg.solve``.
- The whole evaluation rides one `jax.pure_callback`, so the RHS is
  JAX-traceable and composes under `jax.jit` and diffrax (which
  jit-compiles the vector field).
- Worked example `python/examples/inverse_map_diffrax.py` integrates a
  closed output boundary with diffrax Dopri5 and round-trips back through
  the optimizer onto the boundary (~1e-7). `diffrax` is an optional
  extra (`pip install pounce[diffrax]`); the example falls back to RK4
  if it's absent.
- `inverse_map_rhs(..., warm=True)` warm-starts each inner solve from the
  previous evaluation's primal/duals/μ (pounce#86). Result-invariant (up
  to solver tolerance); a *modest* lever — ~1.4-1.7× fewer IPM iterations,
  ~1.3× wall-clock, roughly flat in problem size (interior-point
  warm-start ceiling + per-eval Jacobian-build overhead). Benchmark:
  `python/benchmarks/inverse_map_warm.py`. For a real speedup on a smooth
  map, prefer `PathFollower` (it skips solves, not just cheapens them).
- Switch to `PathFollower` when the path folds or the active set changes.
- Worked notebook `python/notebooks/14_path_following.ipynb` tours the
  whole family (sensitivity → margin → continuation → fold → inverse map).

Also fixes the build-once / stacked path for **unconstrained** problems
(``g=None``, ``m=0``): the constraint callbacks no longer dereference the
(``None``) constraint-Jacobian jit, so `JaxProblem.solve` /
`solve_with_jacobian` / the batched solves now work with no constraints.

### Added — Predictor–corrector path-following engine (pounce#90)

`pounce.jax.PathFollower` traces a solution path of a parametric NLP by
*composing* the post-solve sensitivity primitives instead of re-solving
at every step:

```python
from pounce.jax import PathFollower
pf = PathFollower(jp, monitor_tol=1e-6, ds0=0.05)
trace = pf.follow(theta_of_s, (0.0, 1.0), x0)   # parameter continuation
# trace.x, trace.theta, trace.s, trace.lam,
# trace.n_correctors, trace.n_accepts, trace.active_set_changes
```

- **predict** — extrapolate primal *and duals* along the held-factor
  sensitivity (`jvp_from_state(..., with_duals=True)`); **monitor**
  (no solve) — KKT residual + active-set margin (#89) at the predicted
  point; **correct** — only when the monitor trips, a warm-μ re-solve
  that also re-anchors the factor in one solve (`warm_anchor`, #86).
- Adaptive step size; detects and records active-set changes and
  re-anchors on the new active set.
- `PathFollower.trace_arclength(...)` — pseudo-arclength continuation for
  a scalar-parameter, equality/unconstrained family, tracing **past
  folds** where `∂x*/∂θ` is singular (parameter continuation cannot).
  Reports turning points. Bifurcation/branch-switching and
  inequality-active folds are out of scope for v1.
- On a linear-response NLP the predictor is exact, so the whole path is
  traced with **zero correctors** (one anchor solve vs one cold solve
  per step); nonlinear paths correct adaptively and still trace to
  tolerance.

New supporting public surface:

- `JaxProblem.warm_anchor(p, x0, *, duals=None, mu=None)` — a warm-started,
  μ-seeded re-solve that pins the converged factor and returns a `B=1`
  `AnchorState` (the corrector + anchor in one solve). Threads μ through
  the reusable build-once path (the #86 follow-up).
- `JaxProblem.jvp_from_state(..., with_duals=True)` /
  `batched_jvp_from_state(..., with_duals=True)` — also return the dual
  sensitivity `∂λ*/∂θ · dp` from the same held-factor back-solve.

### Added — Active-set-proximity monitor (pounce#89)

`JaxProblem.active_set_margin(state)` reports the distance to an
active-set change at the anchor point — the "predictor is about to
become invalid" signal for predictor–corrector path following. The
post-solve sensitivity is a derivative on a *fixed* active set; this
flags when a bound / inequality is about to cross its critical-region
boundary (where the sensitivity is discontinuous).

```python
r = jp.active_set_margin(state)
# r["margin"], r["min_mult"], r["min_slack"]  — each (B,)
```

- By complementarity: an **active** bound/inequality (multiplier `>
  active_tol`) is about to leave the set — its *multiplier* heads to
  zero; an **inactive** one is about to enter — its *slack* heads to
  zero. `min_mult` / `min_slack` track each; `margin = min(min_mult,
  min_slack)`.
- Equalities (`cl == cu`) are excluded (always active); `±inf` bounds
  and the slack side of a one-sided inequality drop out naturally.
  An unconstrained interior point returns `inf`.
- Pure-JAX reduction over state the `AnchorState` already holds — no
  solve, no back-solve. Pairs with the caller-side KKT-residual
  (smooth-drift) monitor: re-anchor when either trips.

### Added — Single-problem ergonomic sensitivity wrappers (pounce#88)

Thin un-batched wrappers over the `batched_*` post-solve sensitivity
methods, for the scalar / path-following user (one NLP at a time):

```python
x_star, (lam, zL, zU), J = jp.solve_with_jacobian(theta, x0)   # J: (n, p)
state = jp.anchor(theta, x0)             # un-batched point → B=1 state
J  = jp.sensitivity(state)               # (n, p) from the held factor
dx = jp.jvp_from_state(state, dtheta)    # J @ dtheta  -> (n,)
dp = jp.vjp_from_state(state, x_bar)     # J^T @ x_bar -> (p,)
```

- `solve_with_jacobian` / `sensitivity` / `jvp_from_state` /
  `vjp_from_state` accept and return un-batched shapes, delegating to
  the batched methods with `B=1` and squeezing — no new numerics.
- `anchor` now accepts a single un-batched point (`p_shape`) in addition
  to a batch (`(B,) + p_shape`); a single point yields a `B=1`
  `AnchorState`. The single-problem from-state wrappers reject a `B>1`
  state rather than silently mis-shaping.
- Implemented as `JaxProblem` methods (mirroring the `batched_*` names)
  rather than free functions, for consistency with the existing surface.

### Added — Exact post-solve sensitivity at a supplied point (pounce#87)

`JaxProblem.sensitivity_at(x_star, theta, duals, *, wrt_cols=None)`
returns the exact primal sensitivity `∂x*/∂θ` evaluated at a
caller-supplied primal-dual point, by re-assembling and factoring the
KKT system *there* — no IPM re-solve.

```python
J = jp.sensitivity_at(x_star, theta, (lam, zL, zU))   # (n, p_dim)
```

- **Re-factor, not reuse.** A held FERAL factor encodes the anchor
  point's `H` / `J`, so back-solving it at a moved `x_star` gives a
  first-order-stale sensitivity. `sensitivity_at` assembles the dense
  `(n+m)×(n+m)` KKT at the supplied point, which is exact there
  (assuming a KKT point for `theta`). The cheap-but-local reuse path
  stays as the predictor `batched_jvp_from_state`; this is its
  exact-refresh complement.
- Active set is read from the supplied bound multipliers `(zL, zU)`,
  exactly like the `custom_vjp` backward — the caller passes the duals
  the anchoring solve / `solve_with_warm` returned at this point.
- Pure-JAX, so itself differentiable (second-order sensitivities work);
  matches `jax.jacobian` over a fresh solve to ~1e-6 at every point
  along a swept path, including a binding bound.

This is the exact-refresh primitive for the inverse map, where `x*`
traces a known output boundary and the sensitivity must be evaluated at
the known point without paying a full re-solve per RK stage.

### Added — Barrier-μ warm start for predictor–corrector correctors (pounce#86)

The interior-point barrier parameter μ is now reported on every solve and
can be threaded into a warm-started re-solve, so a predictor–corrector
corrector resumes near the central path instead of re-walking the barrier
homotopy from the default initial μ.

- **`info["mu"]`** — every `Problem.solve` / `Solver.solve` /
  `solve_with_sens` info dict now carries the converged barrier parameter
  (`0.0` on the barrier-free SQP path).
- **`pounce.jax.solve_with_warm`** accepts a 4-element warm-state
  `(lam, zL, zU, mu)` that seeds `mu_init` / `warm_start_target_mu`, and
  returns the converged μ in a matching 4-tuple. The 3-tuple form is
  unchanged; passing `mu=None` inside a 4-tuple reports μ out without
  seeding it in. Differentiability w.r.t. `p` is preserved (the μ
  input/output are stop-gradient, like the duals).

On a small parametric NLP, seeding μ from the previous solve's converged
barrier cut a warm-started corrector from 5 interior-point iterations to
1 (same optimum). The `mu_init` / `warm_start_target_mu` algorithm
options already existed; this exposes the converged μ needed to drive
them along a path.

### Added — Post-solve Jacobian / sensitivity API from the held KKT factor (pounce#82)

`JaxProblem` now exposes a first-class post-solve sensitivity surface
that reuses the held FERAL stacked KKT factor instead of round-tripping
through `jax.vjp` / `jax.jacrev`:

```python
x_star, (lam, zL, zU), J, state = jp.batched_solve_with_jacobian(
    p_batch, x0,
    wrt_cols=slice(0, ny),   # optional parameter-column selection (1-D p)
    return_state=True,
)
dp_bar = jp.batched_vjp_from_state(state, x_bar)   # J^T @ x_bar
state.close()
```

- **`batched_solve_with_jacobian(...)`** returns the full per-block
  primal Jacobian `J` of shape `(B, n, p_dim)` (or `(B, n, len(wrt_cols))`)
  alongside `x_star` and the `(lam, zL, zU)` duals (same contract as
  `batched_solve_with_warm`). The Jacobian is assembled by evaluating the
  existing factor-reuse backward over the `n×n` identity output basis —
  one multi-RHS `Solver.kkt_solve_many` against the held LDLᵀ factor, no
  NLP re-solve.
- **`anchor(p_batch, x0, *, wrt_cols=None)`** solves once and pins the
  factor, returning an **`AnchorState`** handle for reuse across several
  post-solve sensitivity calls (linear-update pattern).
- **`batched_vjp_from_state(state, x_bar)`** is the public reverse-mode
  product `Jᵀ x̄` against a held factor.
- **`batched_jvp_from_state(state, dp)`** is the forward-mode product
  `J @ dp` — the cheap path for linear updates that never materialise the
  full `J`. It assembles the parameter-side RHS `[∂²L/∂x∂p · dp;
  ∂g/∂p · dp]` into the compound x- and constraint-blocks and back-solves
  once against the held factor. Accepts a reduced `dp` when the state was
  anchored with `wrt_cols`.
- **`AnchorState`** lifetime: works as a context manager
  (`with jp.anchor(...) as state:`) *and* supports explicit ownership
  (`state.close()`, `state.reanchor(...)`) for handles that outlive a
  lexical block. Pinned factors are exempt from the LRU but capped
  (`_pinned_capacity`, default 16) with a loud overflow error, and a
  `weakref` finalizer reclaims the factor if a handle is dropped without
  `close()`.

### Added — Structured logging + colored iteration table (pounce#71)

POUNCE now emits diagnostics through the
[`tracing`](https://docs.rs/tracing) ecosystem and renders the
per-iteration table in a tiger/rust branded color theme.

- **Colored iteration table.** Restoration lines take a background that
  varies by restoration kind (soft-stay → tan, soft-exit → amber, hard →
  deep rust); the row text shades smoothly from black toward red as the
  primal step length `alpha` shrinks (a visual stalling cue, shifted to
  cream → bright-yellow on the dark restoration backgrounds). Color is
  emitted only when stdout is a terminal — redirected output and
  `NO_COLOR` get plain text with identical column alignment.
- **Structured logs.** Solver-internal diagnostics, warnings, and
  developer instrumentation are now `tracing` events under namespaced
  targets (`pounce::algorithm`, `pounce::linsol`, `pounce::mu`,
  `pounce::sqp`, `pounce::linesearch`, `pounce::restoration`,
  `pounce::presolve`, `pounce::py`). Logs go to **stderr**; program
  output (iteration table, summary, `--dump`) stays on **stdout**.
- **Spans.** `solve`, `iteration`, `linear_solve`, and `restoration`
  spans tag nested events with context.
- **New environment variables:** `RUST_LOG` (verbosity / per-target
  filtering, default `info`), `POUNCE_LOG_FORMAT=text|json` (JSON sink on
  stderr, including the per-iteration `pounce::iteration` stream for
  Studio / CI), `NO_COLOR` / `CLICOLOR_FORCE` (color policy). Documented
  in `docs/src/options.md` and `docs/src/troubleshooting.md`.
- New `pounce-observability` crate (subscriber install + iteration
  collector) and a `pounce-common::style` palette module.
- A `log` → `tracing` bridge (`tracing_log::LogTracer`) so any remaining
  `log::*` call sites — chiefly transitive dependencies — surface through
  the subscriber and obey `RUST_LOG`.
- **Branded CLI header.** The `pounce` banner now renders a molten
  tiger/rust POUNCE logo (terminal-only; `NO_COLOR` / non-TTY get plain
  text).

### Changed

- Per-iteration JSON solve-report data is now sourced from the
  `pounce::iteration` tracing event (via an in-process collector layer)
  rather than an in-loop accumulation; the report contents are
  unchanged. Capturing iteration history requires the tracing subscriber
  installed by the CLI / Python / C frontends (or
  `pounce_observability::init_for_tests()` in tests).
- Bumped the `feral` linear-algebra dependency from 0.8.0 to 0.9.0.

### Removed

- Dropped the direct `log` crate dependency in favor of `tracing`.

### Added — Active-set SQP with working-set warm start (Phase 5b + 5c + 5d)

A new sequential-quadratic-programming driver sits alongside the
existing interior-point method, opt-in via a single option flip.
Designed for **warm-started NLP sequences** (MPC, parametric
continuation, homotopy sweeps), where the previous solve's active
set is a strong starting point.

**Tutorial:** `docs/src/active-set-sqp.md`.
**Python notebook:** `python/notebooks/06_sqp_parametric_continuation.ipynb`.
**C example:** `crates/pounce-cinterface/examples/sqp_warm_start.c`.
**GAMS example:** `gams/examples/parametric_sqp_warm_start.gms`.
**Design note:** `docs/src/active-set-sqp-warm-start.md`.

#### Algorithm selection (cross-cutting)

- New top-level option `algorithm`, values `interior-point`
  (default; existing IPM path) and `active-set-sqp` (new SQP driver).
  Settable through every interface — `add_option` in Rust /
  Python, `AddIpoptStrOption` in C, `pounce.opt` in GAMS — exactly
  like `linear_solver` already is.

#### SQP suboptions (`sqp_*` namespace)

`sqp_globalization` (`filter` | `l1-elastic`),
`sqp_hessian` (`exact` | `damped-bfgs` | `lbfgs`),
`sqp_max_iter`, `sqp_tol`, `sqp_constr_viol_tol`,
`sqp_dual_inf_tol`, `sqp_l1_penalty`, `sqp_l1_penalty_safety`,
`sqp_l1_penalty_max`, `sqp_bt_reduction`, `sqp_bt_min_alpha`,
`sqp_print_level`, `sqp_lbfgs_max_history`. Defaults mirror
`SqpOptions::default()`. Each is "only consulted when `algorithm`
is `active-set-sqp`"; the IPM path ignores them silently.

#### Python — `pounce.Problem`

New keyword argument and methods:

```python
prob.add_option("algorithm", "active-set-sqp")
x, info = prob.solve(x0, working_set=ws)
ws = info["working_set"]      # always present; None on the IPM path
ws = prob.get_working_set()
prob.set_working_set(ws)
prob.clear_working_set()
```

The `working_set` value is a 2-tuple `(bounds, constraints)` of
numpy int8 arrays with status codes 0..=3 (Inactive / AtLower /
AtUpper / Fixed-or-Equality). Module-level helper
`pounce.classify_working_set(x, x_l, x_u, g, g_l, g_u, lambda_g,
z_l, z_u, m_eq, ...)` classifies an IPM-converged iterate
into a WS suitable for `Problem.solve(working_set=…)`.

#### C ABI — four new entry points

```c
Bool IpoptGetWorkingSet(IpoptProblem, IpoptBoundStatus*, IpoptConsStatus*);
Bool IpoptSetWarmStartWorkingSet(IpoptProblem, const IpoptBoundStatus*, const IpoptConsStatus*);
Bool IpoptClearWarmStartWorkingSet(IpoptProblem);
enum ApplicationReturnStatus IpoptSolveWarmStart(
    IpoptProblem, ipnumber *x, *g, *obj_val, *mult_g, *mult_x_L, *mult_x_U,
    const IpoptBoundStatus *bound_in,
    const IpoptConsStatus  *cons_in,
    IpoptBoundStatus       *bound_out,
    IpoptConsStatus        *cons_out,
    UserDataPtr user_data);
```

Plus typedefs `IpoptBoundStatus`, `IpoptConsStatus` and the four
status constants `POUNCE_WS_INACTIVE` (= 0), `POUNCE_WS_AT_LOWER`
(= 1), `POUNCE_WS_AT_UPPER` (= 2), `POUNCE_WS_FIXED_OR_EQ` (= 3).
**No existing C entry-point signature changed** — cyipopt / JuMP /
AMPL clients link unchanged.

#### GAMS solver link

Two mechanisms ship in tandem:

- **§7.4(a) marginal-based reconstruction** (default, no
  configuration). The solver link reads variable and equation
  marginals (`x.m`, `con.m`) at the top of every `pouCallSolver`
  invocation and reconstructs the SQP working set automatically.
  Lossy at degenerate active sets — same idiom as CONOPT, IPOPT,
  KNITRO under GAMS.
- **§7.4(b) persistent state file** (opt-in via
  `sqp_state_file <path>` in `pounce.opt`). A small binary blob
  with FNV-1a checksum keyed by `(n, m, x_l, x_u, g_l, g_u)` so
  structural changes invalidate cleanly. Falls back to §7.4(a) on
  any read failure.

#### Sensitivity (`pounce-sensitivity`)

`SensResult` now carries the converged user-space multipliers
(`mult_g`, `mult_x_L`, `mult_x_U`) and constraint values (`g`),
so the parametric "predictor + SQP corrector" pattern is a single
`SensSolve::run` followed by one `classify_working_set` call.

#### Hessian sources

The `sqp_hessian` option selects between three implementations:

- `exact` — uses `eval_h`; pounce-qp's inertia control handles
  indefiniteness via diagonal-shift retry (§4.5).
- `damped-bfgs` — Powell-damped rank-2 BFGS, dense `n×n`,
  guaranteed PSD (Powell 1978).
- `lbfgs` — limited-memory BFGS with circular history, default
  6 pairs (matches IPOPT's `limited_memory_max_history`),
  materialized to dense Triplet at QP-solve time.

#### Globalizations

`sqp_globalization` selects the SQP outer-loop step-acceptance
test:

- `filter` (default) — Fletcher-Leyffer 2002 Pareto-frontier
  filter on `(constraint violation, objective)`. No penalty
  parameter; recommended general default.
- `l1-elastic` — Han-Powell merit `φ(x; ν) = f(x) + ν · violation(x)`
  with adaptive ν clamped by `sqp_l1_penalty_safety` /
  `sqp_l1_penalty_max`. SNOPT-style behaviour.

### Added — `feral_ordering` option (FERAL fill-reducing ordering)

User-facing knob for the FERAL backend's fill-reducing ordering. New
string option `feral_ordering` accepts `auto` (default; feral's
adaptive dispatcher — picks AMD / AMF / MetisND from cheap pattern
features), `auto_race` (runs symbolic factorization on AMD, MetisND,
ScotchND, KahipND and keeps the smallest factor_nnz; ~4× a single
symbolic pass, amortized across numeric refactorizations), and the
concrete methods `amd`, `amf`, `metis`, `scotch`, `kahip`. Settable
through every interface that consumes `pounce.opt` /
`OptionsList` — Rust, Python, C, GAMS, CLI — and also via the
`POUNCE_FERAL_ORDERING` environment variable for option-free
callers. Reuses the same explicit-set semantics as the other
`feral_*` options: leaving it unset keeps the `FeralConfig::from_env`
default (`Auto`).

The motivating case is `pinene_3200_0009`, where the cheap `Auto`
heuristic picks MetisND (88 s numeric) but AMD factors in 19.5 s on
the same matrix; `feral_ordering auto_race` measures both and lands
on the winner without per-problem manual tuning. See
`docs/src/options.md` "FERAL backend tuning" and
`docs/src/troubleshooting.md` for guidance.

### Added — AMPL imported (external) function support (issue #49)

`.nl` files that declare imported functions in their `F` segments
and call them via `f<id> <nargs>` tokens are now solved end-to-end.
Set `AMPLFUNC` to a newline-separated list of shared-library paths;
pounce loads each library via the standard AMPL `funcadd_ASL` ABI,
binds every referenced funcall id to a `(library, name)` pair, and
emits `TapeOp::Funcall` nodes that participate in full forward /
reverse / Hessian sweeps (first- and second-derivative requests
are issued back through the library on demand, with the packed
upper-triangular Hessian indexed as `hes[lo + hi*(hi+1)/2]`).

Tested against the IDAES `general_helmholtz_external.dylib`
fixture from the issue report — pounce reaches
`EXIT: Optimal Solution Found` on the 3-variable Helmholtz
problem. Without `AMPLFUNC` set, problems that need external
functions fail with a clear error naming the offending function
and pointing at `AMPLFUNC`.

Limitations: only the `Tape` (default) AD path supports external
functions. The `HybridTape` partial-separability path and the
JIT-style `HessianProgram` path panic on `TapeOp::Funcall` — both
are alternative routes not on `NlTnlp::new`'s critical path, so
the current production flow is unaffected.

### Added — Phase 5a `pounce-qp` crate

Standalone sparse parametric active-set QP solver. Drives the
SQP subproblem solves; also exposed as a standalone crate
(`pounce_qp::ParametricActiveSetSolver`). Implements
Gill-Murray-Saunders elastic mode (§4.3), full GMSW EXPAND
anti-cycling (§4.4), Bunch-Kaufman inertia control via
diagonal-shift retry (§4.5), iterative refinement (§4.7), and
Sherman-Morrison-Woodbury Schur-complement factor updates (§4.2,
opt-in via `QpOptions::use_schur_updates`).

### Added — In-repo regression fixtures

- `crates/pounce-algorithm/tests/hock_schittkowski_subset.rs` —
  10 HS problems with published closed-form optima.
- `crates/pounce-qp/tests/mm_published_optima.rs` —
  Maros-Mészáros-flavoured framework with 5 fixtures + reusable
  `compare_qps_to_published(text, x*, f*, …)` helper.
- `crates/pounce-algorithm/tests/parametric_sqp_corrector.rs` —
  IPM → classify_working_set → SQP corrector end-to-end.
- `crates/pounce-algorithm/tests/sqp_filter_vs_l1_elastic.rs` —
  parity between the two globalizations.

### Added — Auxiliary-equality preprocessing (Phase 0 presolve, issue #53)

A 14-PR series that scaffolds an opt-in *Phase 0* presolve pass:
detects block-triangular structure in the equality system, solves
the dependent blocks ahead of the IPM, and substitutes the
recovered variables back into the user TNLP. Targets gas-network,
power-flow, and process-design problems where a few hundred
algebraic state variables eliminate cleanly.

The algorithm and reference implementation are a port of
[ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32) by
**David Bernal Neira** ([@bernalde](https://github.com/bernalde)).
The ripopt work also vendored the
`tutorial_flow_density{,_perturbed}.nl` and `gaslib11_steady.nl`
fixtures we now use for end-to-end testing.

- Hopcroft-Karp incidence matching, Dulmage-Mendelsohn decomposition,
  Tarjan SCC → block-triangular form.
- Coupling classification (linear / nonlinear / inequality-coupled)
  plus a damped-Newton block solver with large-block fallback.
- Trivial-elimination pre-pass; inequality-coupled blocks handled
  by projection.
- Reduction-frame bookkeeping with full multiplier recovery so
  `final_zL` / `final_zU` round-trip back to the user space.
- Orchestrator wired into `PresolveTnlp`, gated by
  `presolve_auxiliary` (default off). Diagnostics surfaced via
  `presolve_auxiliary_diagnostics`.
- Design note: `dev-notes/auxiliary-equality-preprocessing.md`;
  user docs in `docs/src/auxiliary-presolve.md`.

### Added — FBBT (Feasibility-Based Bound Tightening, #62)

Three-commit landing of FBBT inside `pounce-presolve`:

- `pounce-presolve::interval` — outward-rounded interval arithmetic
  on `f64`, with `Interval::div` reciprocal endpoints rounded
  outward (fixes a subtle near-zero straddle case discovered in
  review).
- `ExpressionProvider` trait + forward pass walks each constraint
  expression and tightens variable bounds from the constraint's
  `g_l`/`g_u` envelope.
- Reverse propagation + orchestrator wired through `PresolveTnlp`
  end-to-end. New options: `presolve_fbbt` (master switch,
  default off), `fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints`.
- Docs: `docs/src/fbbt.md`; demo notebook
  `python/notebooks/08_fbbt.ipynb`.

### Added — Problem and KKT-system scaling (#61, f00c1f9)

End-to-end wiring of the upstream `nlp_scaling_*` and
`linear_system_scaling` option families:

- `nlp_scaling_method`: `none` / `user-scaling` (new — pulled from
  `set_problem_scaling` Python API or `SetIpoptProblemScaling`
  C API) / `gradient-based` (existing, now with target-gradient
  knobs `nlp_scaling_obj_target_gradient` and
  `nlp_scaling_constr_target_gradient`).
- `linear_system_scaling`: `none` / `mc19` / `ruiz` (iterative
  symmetric infinity-norm equilibration, new) / `slack-based`.
  Applied to the augmented system independent of NLP-level
  scaling.
- Python `Problem.set_problem_scaling(obj_scaling, x_scaling=None,
  g_scaling=None)` plus a worked example in
  `python/notebooks/07_scaling.ipynb`.
- Documentation: `docs/src/scaling.md`.

### Added — Mehrotra adaptive-μ defaults and init cascade (upstream parity)

- `mehrotra_algorithm` option routed through `PdSearchDirCalc`
  (previously parsed but inert).
- `adaptive_mu_globalization` cascade finished per upstream Ipopt;
  `bound_push` / `bound_frac` / `bound_mult_init_val` / `alpha_for_y`
  cascade from `mehrotra_algorithm yes`.
- `least_square_init_primal` implemented in
  `DefaultIterateInitializer`.
- `accept_every_trial_step` honored in the line search and
  cascaded from `mehrotra_algorithm` (matches upstream
  initialization behavior).

### Added — FERAL backend tunables and 0.8.0 bump

- `feral_pivtol` exposed as an `OptionsList` option with
  `FERAL_PIVTOL` environment-variable fallback.
- Tri-state `cascade_break` (#55): `auto` / `on` / `off`, inheriting
  the FERAL Phase B default unless explicitly set.
- Workspace bump to `feral 0.8.0`, which ships the SSIDS-aligned
  strict-zero-pivot inertia policy (feral gh#54 / pounce gh#52,
  *nuffield2_trap*). The temporary `[patch.crates-io]` block
  pointing at the local feral checkout has been removed.

### Added — `pounce-solve-report` crate + `IpoptWriteSolveReport` C API

- New publishable crate `pounce-solve-report` (first crates.io
  release) emits the machine-readable `pounce.solve-report/v1`
  JSON shared by the CLI, the C ABI, and the GAMS driver.
- C ABI: `IpoptWriteSolveReport(IpoptProblem, const char *path)`
  writes the report to disk after `IpoptSolve`.
- GAMS driver now emits `pounce.solve-report/v1` alongside the
  `.lst` so studio tooling can consume it directly.

### Added — Diagnostics dumps

- `--dump iterates:{summary,full}` (#68) — per-iteration trajectory
  artefacts the studio can replay. `summary` writes one JSON line
  per outer iteration; `full` adds the primal/dual vectors and
  KKT residuals.
- `--dump kkt:*+L` (#69) — augments the existing KKT-system dump
  with the LDLᵀ factor pattern (block structure, fill-in, pivot
  signs) for inertia post-mortems.
- `print_options_documentation yes` now actually walks the
  registered options and emits a categorized dump (previously a
  registered-but-inert toggle).

### Added — Studio Claude-skill and MCP GAMS tools

- `studio/skill/` — Claude-skill front-end as an alternative to the
  MCP server. Lighter-weight install path for users who just want
  the studio prompts and don't need an MCP runtime.
- `studio/mcp` — new GAMS problem tools (`run_gams_problem`,
  `analyze_gams_problem`, `parse_gams_listing`,
  `list_gams_examples`) plus an install script.

### Added — Parallel batched `pounce.jax.vmap_solve_parallel` + GIL release (pounce#74)

`pounce_py::Problem::solve` now releases the Python GIL across the
`optimize_tnlp` call (every TNLP callback was already
`Python::with_gil`-wrapped, so this is a localized
`py.allow_threads` block in `crates/pounce-py/src/problem.rs`).
That unlocks true concurrent IPM iteration across independent
`Problem` instances on different OS threads — Python-level
`f` / `g` callbacks still serialize on the GIL but the linear-algebra
heart of the solver runs in parallel.

`pounce.jax.vmap_solve_parallel` rides that change: a drop-in
replacement for `vmap_solve` that dispatches the batch over a
`ThreadPoolExecutor` of independent `Problem` instances. Forward
is parallel via the threadpool; backward is `jax.vmap` over the
per-element KKT solve (pure JAX, vectorizes naturally).

```python
from pounce.jax import vmap_solve_parallel

x_batch = vmap_solve_parallel(
    p_batch, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    workers=8,  # default: min(B, 8)
)
```

Microbench (`n=30`, `B=16`, nonlinear unconstrained, M1 8-core):
`vmap_solve` 1.00s → `vmap_solve_parallel(workers=8)` 0.37s
(~2.75×). Speedup grows with per-element solve cost. Numerically
identical to the sequential reference.

### Added — `pounce.jax.solve_with_warm` (pounce#74)

Companion to `pounce.jax.solve` that threads the previous solve's
dual triple `(mult_g, mult_x_L, mult_x_U)` into the next call via
IPOPT's `warm_start_init_point=yes` machinery.

```python
from pounce.jax import solve_with_warm

x_star, warm = solve_with_warm(
    p, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    warm_start=None,                # cold first call
)
for p_k in trajectory[1:]:
    x_star, warm = solve_with_warm(
        p_k, f=f, g=g, x0=x_star, n=n, m=m,
        lb=lb, ub=ub, cl=cl, cu=cu,
        warm_start=warm,            # threaded duals
    )
```

Differentiable w.r.t. `p` via the same implicit-function rule as
`solve`. Cotangents on the warm-state outputs and the warm-state
inputs are dropped (zero) — at the optimum the duals are a
function of `p` and the active set, not an independent input to
`dx*/dp`. `solve` itself is unchanged (non-breaking).

### Added — `pounce.jax.JaxProblem` build-once/solve-many handle (pounce#75)

Iterative outer loops (differentiable constrained layers in a
training step, parametric sweeps) were paying a ~45ms rebuild on
every call to `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` — re-JIT of `jax.grad`/`jacrev`/`hessian`, the
one-shot random sparsity probe, plus a fresh `pounce.Problem`
construction — versus a ~3ms underlying solve. On `n=5, m=6`
problems that's a ~14× wrapper overhead.

`JaxProblem` is a build-once handle: do the JIT and sparsity probe
in `__init__`, then expose `.solve(p, x0)`, `.solve_with_warm(p, x0,
warm)`, `.vmap_solve(p_batch, x0)`, and `.vmap_solve_parallel(p_batch,
x0, workers=)` as methods that reuse the prebuilt state across
calls. Each worker thread in `vmap_solve_parallel` keeps its own
cached `pounce.Problem` via `threading.local` so the build cost is
paid at most once per worker (typically `min(B, 8)` total) rather
than `B` times per batch.

```python
from pounce.jax import JaxProblem

jp = JaxProblem(
    f=f, g=g, n=n, m=m, p_example=p0,
    lb=lb, ub=ub, cl=cl, cu=cu,
    options={"tol": 1e-9, "print_level": 0},
)
for p_k in trajectory:
    x_star = jp.solve(p_k, x0=x_prev)
    x_prev = x_star
```

Microbench on the issue's `n=5, m=6` shape — 20 sequential solves at
different `p`:

```
top-level solve   (20 calls): 1.914s  → 95.7ms/solve
JaxProblem.solve  (20 calls): 0.136s  → 6.8ms/solve
speedup: 14.1x
```

Existing top-level `solve` / `vmap_solve` / `vmap_solve_parallel` /
`solve_with_warm` are unchanged (non-breaking) — `JaxProblem` is a
new surface for performance-sensitive iterative use.

### Added — `JaxProblem` factor-reuse backward (k_aug-style; pounce#76)

The `custom_vjp` backward of `JaxProblem.solve` /
`solve_with_warm` no longer assembles a dense
`(n+m) × (n+m)` KKT block in JAX and runs `jnp.linalg.solve` on it.
Instead it reuses the IPM's converged compound KKT factor through
`pounce.Solver.kkt_solve` — the same factor [k_aug] uses for
parametric sensitivity. Two wins:

* **Perf.** The dense back-solve is O((n+m)³) on every bwd call;
  reusing the held LDLᵀ factor makes it O(nnz(L)). For modest `n`
  the absolute savings are small; for `n+m` in the hundreds-to-
  thousands it dominates the bwd.
* **Correctness.** The compound block's bound-multiplier rows
  `(z_l, z_u)` already encode active-set behaviour — at convergence
  active bounds have unbounded `z` (forces `Δx_i = 0` in the
  back-solve), inactive bounds have `z ≈ 0` (leaves `Δx_i` free).
  Slack inequality rows in the user's `g` are handled the same way
  by `(v_l, v_u)`. The factor-reuse path therefore drops the
  explicit active-set masking the dense path does on `H` / `J` / `v`;
  accuracy is `O(μ)` at the IPM barrier parameter, well below `tol`
  after convergence.

Behaviour change: `JaxProblem(factor_reuse=True)` is the default. Set
`factor_reuse=False` for a verbatim fallback to the pre-#76 dense
backward (useful for higher-order differentiation, since the dense
backward stays JAX-traced and is itself differentiable).

Plumbing:

* `pounce.Solver` exposes a new `block_dims` getter returning the
  `(n_x, n_s, n_y_c, n_y_d, n_z_l, n_z_u, n_v_l, n_v_u)` layout of
  the compound KKT vector so the JAX bwd can pack a partial RHS
  (just the x-block) and unpack `u_x` / `u_y_c` / `u_y_d`.
* Each fwd registers its converged `Solver` in a bounded-LRU cache
  on the `JaxProblem` (default capacity 128, exposed as
  `clear_solver_cache()` for early eviction). LRU rather than
  pop-on-read because `jax.jacobian` calls the bwd N times per
  fwd; pop semantics would crash from the second direction onward.
* The back-solve `pure_callback` uses
  `vmap_method="sequential"` so `jax.jacobian` / `jax.vmap` of a
  loss-gradient correctly iterate one cotangent at a time across
  the impure host call.

The standalone `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` keep the dense backward for now.

[k_aug]: https://github.com/dthierry/k_aug

### Added — `JaxProblem.batched_solve` stacked block-diagonal solve (pounce#76 (A))

`JaxProblem.batched_solve(p_batch, x0)` runs one IPM solve over a
single NLP whose variables are `[x^(1); ...; x^(B)]`, constraints are
`concat(g(x^(k), p^(k)))`, and objective is `Σ_k f(x^(k), p^(k))`.
The Jacobian and Lagrangian Hessian are block-diagonal (no
cross-block coupling, since each block-`k` constraint touches only
the block-`k` slice of `X` and the objective is a pure sum), so the
IPM sees one big sparse problem but spends linear-system work
proportional to `B × (per-block factor cost)`.

Complementary to the existing batched surfaces:

* `vmap_solve` — sequential `jax.lax.map`, one solve per iterate.
* `vmap_solve_parallel` — B independent IPMs in a
  `ThreadPoolExecutor` (GIL released per solve). Wins when batch
  elements have very different convergence behaviour.
* `batched_solve` — one stacked IPM. Wins when blocks have similar
  convergence behaviour (shared barrier homotopy and shared
  symbolic factorisation amortise across the batch) and when B is
  large enough that the per-call Python overhead of B fwd
  dispatches becomes visible — one Rust crossing instead of B.

`custom_vjp`-wrapped: `jax.grad` / `jax.jacobian` through
`batched_solve` work end-to-end. The bwd vmaps the per-element
dense KKT back-solve, which is exact because the block-diagonal
coupling means `∂x^(k)*/∂p^(j) = 0` for `k ≠ j`.

Plumbing:

* `_StackedJaxNlp` lifts the per-block sparsity pattern (cached on
  the parent `JaxProblem` from the one-shot probe) to the stacked
  problem's block-diagonal pattern at construction time, so the
  per-solve `jacobianstructure` / `hessianstructure` callbacks are
  O(1).
* Stacked Problems are built per (thread, B) with a tiny LRU on
  the `JaxProblem` (cap 4) keyed by batch size — guards against
  cycling between a couple of sizes (e.g. eval batch ≠ train
  batch).
* Per-block bounds `lb`/`ub`/`cl`/`cu` are tiled across the batch;
  per-block bounds aren't exposed on this surface.

### Changed

- `pounce-qp::ParametricActiveSetSolver::solve_equality_plus_bounds`
  now falls through to `solve_elastic` when the equality-relaxed
  cold start violates a variable bound. Previously returned
  `UnsupportedFeature`.
- `optimize_sqp_tnlp` now populates `SolveStatistics`
  (`iteration_count`, `final_dual_inf`, `final_constr_viol`,
  `final_objective`) so `GetIpoptIterCount`, `info["iter_count"]`,
  etc. report SQP-side numbers on the SQP path.

### Fixed

- SQP `check_kkt` stationarity formula: was `∇f + Jᵀ λ_g + λ_x`,
  must be `∇f + Jᵀ λ_g − λ_x` (pounce-qp packs
  `λ_x = z_l − z_u = −λ_sat`). Latent — only triggered by problems
  with an active variable bound. Discovered on a 3-D simplex
  projection.
- `fix(mu): guard probing oracle against corrupted iterate (#58)`
  — the probing oracle no longer dereferences fields of an
  iterate that the line-search rejected mid-update.
- `fix(mu/probing)`: σ denominator uses `curr_avrg_compl`, not
  `data.curr_mu`, matching upstream.
- `fix(mu-oracle)`: allow inexact affine predictor solves to feed
  the quality-function oracle (upstream parity).
- `fix(l1-wrapper): use multi-pass restoration factory provider
  (#24)` — the ℓ₁ penalty wrapper now nests a restoration sub-IPM
  whose own restoration provider is the multi-pass factory,
  matching the outer IPM path.
- `fix(restoration)`: restoration sub-IPM inherits the outer
  `mu_strategy` rather than resetting to `monotone`.
- `fix(feral)`: zero-pivot factorizations on LP-shape KKT
  systems route to `Singular` instead of bubbling up as
  `Internal`.
- `fix(fbbt)`: outward-round reciprocal endpoints in
  `Interval::div` for the near-zero straddle case.
- `fix(presolve)`: auxiliary preprocessing + `presolve_bound_tightening`
  infeasibility paths (#60).
- `fix(init/ls)`: perturb `delta_c`/`delta_d` by 1e-8 in the
  least-squares-init augmented system to avoid exact rank
  deficiency.
- `fix(scaling)`: scale `d_l` / `d_u` in step with `d(x)` under
  gradient-based scaling.
- `fix(hsl)`: HSL build script is a no-op when `COINHSL_DIR` is
  unset, so `cargo build` works on machines without HSL
  installed even with the `ma57` feature off.
- `fix(benchmark-report)`: composite report now globs the newest
  `pounce_*.json` under `benchmarks/mittelmann/results/` instead
  of hard-coding `pounce_v0.1.0.json`.
- `fix(jax)`: `pounce.jax.solve` backward pass now respects the
  constraint active set, not just variable bounds. Slack inequality
  rows are dropped from the implicit-function-theorem KKT block via
  the same identity-augment trick used for active bounds; previously
  they were kept as equalities, silently returning the wrong
  `dx*/dp` whenever an inequality was inactive at the optimum
  (pounce#73).

### Docs

- `docs: adaptive-μ option tables, scaling worked example,
  troubleshooting guide` — `docs/src/options.md`,
  `docs/src/scaling.md`, `docs/src/troubleshooting.md` refreshed.
- FBBT reference page (`docs/src/fbbt.md`) and Pyomo demo
  notebook `python/notebooks/08_fbbt.ipynb` (#62).
- Scaling docs page (`docs/src/scaling.md`) + Python demo notebook
  `python/notebooks/07_scaling.ipynb` (#61).
- `studio/skill` README: corrected `POUNCE_BIN` claim,
  `inspect --json`, sibling-feral layout.
- README badges: PyPI version + downloads for `pounce-solver` and
  `pyomo-pounce`; Zenodo DOI
  `10.5281/zenodo.20387011` published.

### Compatibility

- All existing IPM users (`IpoptSolve`, `Problem.solve(x0=…)`,
  `option nlp = pounce` without `algorithm` set) continue to
  behave identically. Every Phase 5 addition is opt-in.
- The C ABI is strictly additive — four new symbols, no signature
  changes.
- The Python `Problem.solve` signature gained one optional kwarg
  (`working_set=None`); positional callers are unaffected.


### Algorithm-path isolation guarantees

The IPM and active-set SQP paths share the TNLP layer, options
registry, linear-solver backend, and `finalize_solution`, but are
otherwise isolated. Toggling `algorithm` is always safe:

- The default (`algorithm = interior-point`) runs zero Phase 5
  code. Users who never set `active-set-sqp` are unaffected.
- `sqp_*` options are silently ignored on the IPM path.
- IPM warm-start options (`warm_start_init_point`, `bound_push`,
  `bound_frac`, `slack_bound_push`, `mult_init_max`, `mu_init`,
  `mu_target`, …) are silently ignored on the SQP path.
- Warm-start payloads are path-local:
  `set_sqp_warm_start(SqpIterates)` /
  `Problem.solve(working_set=…)` / `IpoptSetWarmStartWorkingSet`
  feed the SQP loop only; `lagrange=` / `zl=` / `zu=` paired with
  `warm_start_init_point=yes` feed the IPM only.
- `info["working_set"]` is always present in the Python info
  dict but is `None` on the IPM path.
- Callers can flip between paths across solves on the same
  problem handle — the parametric corrector pattern in the
  tutorial uses this for cold IPM warm-up followed by an SQP
  corrector.

These guarantees are exercised by the test suite: see
`application_default_does_not_select_sqp`,
`application_sqp_warm_start_auto_clears_after_use`,
`application_sqp_warm_start_round_trip`, and
`test_get_working_set_returns_none_on_ipm_path` (Python).

## [0.2.0] — 2026-05-25

First tagged release. The `0.1.0` work-in-progress version was never
tagged; everything below summarizes the state of `main` as of this
release.

### Solver core

- **Full Ipopt-parity C ABI**: `CreateIpoptProblem`, `IpoptSolve`,
  `AddIpoptStrOption` / `AddIpoptNumOption` / `AddIpoptIntOption`,
  `OpenIpoptOutputFile`, `SetIpoptProblemScaling`,
  `SetIntermediateCallback`, `GetIpoptCurrentIterate`,
  `GetIpoptCurrentViolations`, plus a new `IpoptSolver` session
  handle (`IpoptSolverSolve`, `IpoptSolverResolve`,
  `IpoptSolverKktSolve`, `IpoptSolverParametricStep`).
- **Restoration phase** wired through `IpoptSolve` with the soft
  restoration line search; nested IPM honors the parent's
  `print_iter_output` gate.
- **Rapid infeasibility detection** in the main loop; convergence
  statuses certified against upstream Ipopt.
- **Option-parity (tier-A waves 1-4)**: convergence options
  (`tol`, `acceptable_tol`, etc.), mu/watchdog/output toggles,
  iteration-output flags, warm-start machinery,
  `fixed_variable_treatment`, `nlp_*_bound_inf`,
  `barrier_tol_factor`, `sigma_min` / `sigma_max` for the adaptive
  quality-function oracle.
- **Sensitivity (sIPOPT)**: Phase D landed — convenience API,
  eigendecomposition, fixed-variable lifting, boundcheck. New
  `Solver` session API on top: value-typed `Factorization` handle
  in `pounce-linsol` enables factor-once / solve-many; `Solver`
  exposes `kkt_solve`, `parametric_step`, and
  `compute_reduced_hessian` without callback shapes.
- **Presolve** crate (`pounce-presolve`) as an opt-in TNLP wrapper.

### Backends and bindings

- **Python** (`pounce-solver`): PyO3 bindings with a cyipopt-style
  `Problem` class and a scipy-style `minimize()` facade. The wheel
  bundles the `pounce` CLI executable.
- **Python session API** (`pounce.Solver`): pyclass that wraps the
  Rust `Solver`, enabling warm-start sequences (MPC / parametric /
  B&B) and many-RHS sensitivity workflows without the
  callback-based shape.
- **pyomo-pounce** (`pyomo-pounce`): Pyomo SolverFactory plugin
  that drives the `pounce` CLI on the user's PATH.
- **GAMS link**: native solver link (`libGamsPounce`) for GAMS;
  Jacobian eval skips dense memsets and pure-linear rows.
- **CLI**: bundled `pounce` binary writes AMPL `.sol` solution
  output; new `--about` prints version / build / features / paths;
  `--dump` writes per-iteration KKT artefacts; the sIPOPT
  sensitivity step is folded in.

### Linear-solver layer

- **Public `Factorization`** in `pounce-linsol`: factor once,
  back-solve many RHS, refactor with new values reusing the
  symbolic factor / AMD ordering.
- **MA57** backend (`pounce-hsl`) honors the `linear_solver`
  option default (`"ma57"`).
- **Feral** backend: cascade-break and FMA default off (opt-in via
  env); near-singular factorizations are flagged via an absolute
  pivot floor; explicit-zero stripping before KKT factor; skips
  refactor on same-matrix back-solve.

### Numerical robustness

- TNLP `eval_*` user-callback failures surface as NaN instead of
  panicking.
- Round-off-tolerant `Compare_le` in the Armijo line-search test.
- Unconstrained problems routed through the IPM (no degenerate
  paths).
- `push_x_into_interior` uses `dim()` (not `values().len()`),
  fixing a subtle off-by-one on partially-filled vectors.
- `OrigIpoptNlp::eval_h` always uses the `h_entry_in_full`
  mapping; closes the panic when an entire Hessian row sits on a
  fixed variable.

### Benchmarks

- **Composite report** (`make benchmark` →
  `benchmarks/BENCHMARK_REPORT.md`) covering 9 suites: CUTEst (727
  curated; 1542 full sweep), Mittelmann LP/QP, water-network
  design, gas-network, electrolyte, grid, CHO, large-scale, and
  the GAMS link.
- **Incremental per-suite targets**: `make benchmark-<suite>`
  skips when `results.json` is fresh; `make benchmark-<suite>-rerun`
  forces a rebuild.
- **MA57 baseline** integrated into the composite report.

### Studio & tooling

- **studio/mcp** MCP server (`pounce-studio-mcp`) with
  `analyze`, `run`, `explain`, `citations` tools and an embedded
  glossary; backed by `pounce-studio-core` via PyO3.
- **Linear-solver post-mortem** aggregated end-to-end and
  surfaced through the studio.

### Infrastructure

- CI workflow with format / clippy / build / test, plus
  wheel-smoke for `pounce-solver` and `pyomo-pounce`.
- mdbook documentation built and deployed to GitHub Pages via the
  new `docs.yml` workflow.
- Zenodo metadata (`.zenodo.json`) and `CITATION.cff` for
  archival on every GitHub Release.

[0.3.0]: https://github.com/jkitchin/pounce/releases/tag/v0.3.0
[0.2.0]: https://github.com/jkitchin/pounce/releases/tag/v0.2.0
