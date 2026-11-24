# Known limitations

Measured defects and accepted regressions that nobody is scheduled to work on.
Each one used to be an open issue; it lives here instead so the issue tracker
holds only planned work, while the measurements stay where the next person
can find them.

An entry records what is wrong, how big it is, what has already been tried
and failed, and how to tell whether a fix works. If you pick one up, open a
fresh issue that links back to the entry, and delete the entry when the fix
merges.

---

## Convex IPM: the largest Mittelmann LPs still need more memory than the factor (gh#1249)

gh#1249 found five Mittelmann LPs (`L1_sixm250obs`, `L1_sixm1000obs`,
`fhnw-binschedule1`, `Primal2_1000`, `L2CTA3D`) exhausting a 38.6 GB cap inside
FERAL before the first iteration. The ordering half is fixed in feral 0.25
(jkitchin/feral#246: monotone AMF scoring takes AMF's nnz(L) on fhnw from
1.39e10 to 1.84e8, and `Auto` sends n > 100 000 with average degree < 5 to
AMF). Pounce's stopgap, racing the orderings on convex KKTs of order >= 1M,
was dropped in gh#1265 after re-measuring under 0.25 (arm64, single-threaded,
`POUNCE_FERAL_REPORT_SYMBOLIC=1`):

| model | race | feral `Auto` |
|---|---|---|
| `fhnw-binschedule1` | AMF, peak 5.05 GB, 394 s | AMF, same prediction, 149 s |
| `Primal2_1000` | AMF, peak 7.03 GB, 74 s | AMF, same prediction, 5.8 s |
| `L2CTA3D` | analysis not done after 45 min | AMF, peak 427 GB, 71 s |

Neither fhnw run finished within 1 800 s (`Auto` reached 25 iterations).
`L2CTA3D` fits under no ordering on a 64 GB machine. The measurements below
predate 0.25 and the race's removal.

What is left, measured on a 15 GB / 4-core container with an RSS watchdog,
on a build based on `913be16` (after gh#1243's duplicate-copy fixes, before
gh#1248's `.nl` reader changes, so the baseline below may now be lower):

| model | ordering | result |
|---|---|---|
| `Primal2_1000` | race -> METIS (predicted peak 2.2 GB) | 37 iterations in 2400 s, peak RSS 12.4 GB, `Maximum_WallTime_Exceeded` |
| `Primal2_1000` | AMD (the old `Auto`, predicted 8.3 GB) | over 13.7 GB at 86 s, in the first numeric factor |
| `fhnw-binschedule1` | race -> AMD (predicted peak 6.8 GB) | symbolic done in 466 s (the race), then over 13.5 GB at 993 s in the numeric factor |

* **Memory outside the factor.** Primal2_1000 sits at ~6.5 GB RSS before its
  numeric factor whatever the ordering (measured with every factorization
  refused), against a predicted 2.2 GB factor. The candidates are the ones
  `dev-notes/kkt-bound-row-elimination.md` lists: `expand_bounds`' problem
  copy, the `.nl` names (0.5 GB on this model), presolve layers. fhnw's
  6.8 GB factor on top of the same kind of baseline did not fit in 13.5 GB;
  whether it fits in 38.6 GB is unmeasured.
* **The race's cost and reach.** feral 0.24's `AutoRace` ran
  `RACE_CANDIDATES = {Amd, MetisND}` one after the other and ranked them on
  `factor_nnz_estimate`; neither the race nor a candidate checks
  `max_wall_time` (pounce's `factor_deadline` is checked only after the
  whole analysis). feral 0.25 races {Amd, MetisND, ScotchND, Amf} at
  n >= 1M and computes MC64 once per race: about 1 007 s on fhnw (feral's
  measurement) against the 466-476 s below.
  * Time: 466-476 s on fhnw (AMD alone: 318 s; over the cpo3 corpus's 300 s
    limit), 120-130 s on Primal2 (AMD alone: 24 s).
  * `L2CTA3D` (order 14.2M) is **worse** under the race. METIS ran past the
    reviewer's timeout in the ordering sweep, while AMF (the old `auto`)
    finished its analysis and let the budget refuse the factor. Under the race
    every run on it waits out the METIS ordering before it can report or
    refuse anything. Opt out with `POUNCE_FERAL_ORDERING=auto` or `amf`.
  * `L1_sixm250obs`: the race picks METIS (187 GB predicted) over AMD
    (544 GB), but SCOTCH (158 GB) is not a candidate. Out of reach under every
    ordering anyway.
  * The fix wants a deadline check between candidates, or inside an ordering,
    in feral (feral#246).
* **The other three models**, measured in the gh#1250 review on a larger
  machine (single-threaded, `auto` unless stated):
  * `L1_sixm250obs` (order 1.03M) predicts 371 GB under `auto` (AMF), 544 GB
    under AMD, 187 GB under METIS and 158 GB under SCOTCH. It is out of reach
    under every ordering, so the race cannot rescue it. This is genuine fill,
    or at least fill no available ordering avoids.
  * `L1_sixm1000obs` (order 3.30M): the symbolic phase itself sits near 19 GB
    for 2.5 min, then jumps to 52.5 GB within ~3 s inside FERAL's
    `finish_and_copy`, before any prediction exists.
  * `L2CTA3D` (order 14.2M) predicts 265 GB under `auto` (AMF), and its
    analysis peaks near 25 GB. Under AMD it predicts nnz(L) 1.12G (~9 GB of
    factor) with a largest front of 10 100 rows, but 811 GB of contribution
    blocks. That figure is the postorder high-water mark (feral's
    `compute_peak_contrib` frees each child's block after allocating its
    parent's), not a sum, so it implies very many large sibling blocks
    pending at once. Whether that is real is unverified.

  The symbolic phase's own memory on these two (`finish_and_copy`,
  `compute_static_row_indices`) is FERAL's to fix; the pre-analysis cannot
  protect against it, because no prediction exists until the phase ends.

How to measure (from `benchmarks/lpopt/mps_to_nl.py` output): run with
`POUNCE_FERAL_REPORT_SYMBOLIC=1` for the prediction, and with
`POUNCE_FERAL_MAX_FACTOR_BYTES=1` to stop before the numeric factor, which
isolates the memory spent before it. A fix to the baseline shows as a lower
RSS at the refusal.

---

## Convex IPM: default static pivoting ends ns1116954 acceptable, and switching to delayed pivoting on demand costs other models

*Measured on `integrate-0.18-fixes` (parent `22ae155e`), MIPLIB ns1116954's
LP relaxation and the cpo3 benchmark set (`research/cpo3-benchmarks`) at
`tol=1e-8`, `max_wall_time=300`, `RAYON_NUM_THREADS=1`. Recorded 2026-11-21.*

### What is wrong

Since gh#1193 the convex IPM's FERAL backend factors with static pivoting
when `tol >= 1e-8` and at most 1% of columns are free
(`convex_ipm_static_pivoting`, `crates/pounce-feral/src/lib.rs`). On
ns1116954, at μ ≈ 1.7e-7, `res_p` jumps from 1e-13 to 2.9e-6, and no δ bump of
the HSDE regularization ladder (`hsde.rs`) brings it back below
`DYN_REG_RES_TOL`. From the next iteration on, the factor reports 30–50 more
negative pivots than a quasi-definite KKT can have, and each iteration pays
four ~22 s factorizations. The solve ends `Solved_To_Acceptable_Level` at
1.0e-6 after 12 iterations and 368 s. Before gh#1193 (e176b8fa) it was
`Solve_Succeeded` in 16 iterations and 277 s. `feral_static_pivoting=no`
reproduces that, digit for digit. Since 0.18.0 the deadline is checked
between refactorization retries, so `max_wall_time=300` is honoured: the
default now stops `Solved_To_Acceptable_Level` after 12 iterations at 305 s,
and `feral_static_pivoting=no` reaches `Solve_Succeeded` in 16 iterations
and 261 s.

### What was tried

The fix tried was to give up the *defaulted* static choice the first time a
refined solve stayed above `DYN_REG_RES_TOL`: switch to delayed pivoting once
and refactor at the same δ. It rescued ns1116954, but it also fired late in
the solve on other models, where the δ ladder recovers without it:

| model | μ at first trip | default | switched |
|---|---|---|---|
| neos-933966 | 4.9e-10 | 17 it, 41.8 s | 17 it, 73.9 s |
| netdiversion | 4.0e-11 | 30 it, 105 s | 31 it, 261 s |
| QPLIB_8602 | 4.0e-13 | 27 it, 125 s | 27 it, 204 s |
| atlanta-ip | 6.4e-10 | 124 it Optimal, 39 s | 200 it Acceptable, 300 s |

Requiring μ > 1e-8 before switching kept those four at their old numbers
and still rescued ns1116954, which trips at μ = 1.7e-7. A re-run of the
66 cpo3 models whose results had moved then showed slowdowns that are
from the switch, because `feral_static_pivoting=yes` restores the old
numbers:

| model | default | with the μ-gated switch |
|---|---|---|
| CRE-D | 49 it, 8.1 s | 54 it, 16.0 s |
| momentum1 | 83 it, 10.2 s | 82 it, 18.6 s |
| leo2 | 98 it, 43.0 s (34.9 s forced static) | 98 it, 57.4 s |

Each threshold was fitted to the handful of models measured at the time, so
the switch was dropped before 0.18.0.

### How to tell whether a fix works

- ns1116954 at `tol=1e-8` reports `Solve_Succeeded` within `max_wall_time=300`
  at default options.
- Every model in the two tables above keeps its default numbers. The
  switched/gated columns are the regression to rule out.
- The full cpo3 non-Mittelmann set does not lose a verified model or gain a
  >1.3× wall-time slowdown against the parent, run on an otherwise idle
  machine.
- The rmatr200-p5, waihi and turia cascades that static pivoting exists for
  stay unchanged.

---

## Convex IPM: the KKT verdict's rounding floor ignores row length, so long rows read as reduced accuracy

*Measured on `fix-boyd2-fallback-skip` (parent `22ae155e`), Maros–Mészáros
BOYD2 at `tol=1e-8`, `RAYON_NUM_THREADS=1`. Recorded 2026-11-20.*

### What is wrong

`QpSolution::kkt_residuals_above_floor` (`crates/pounce-convex/src/qp.rs`),
the verdict behind `Optimal` vs `OptimalInaccurate` (`demote_optimum_above_tol`
in `ipm.rs`), excuses each residual entry up to `k·mag`. Here `mag` is the
magnitude of the terms that entry is a sum of, and `k = REPORT_FLOOR_FACTOR
(4) · REL_FLOOR_KAPPA (64) · ε ≈ 5.7e-14`. `k` does not depend on how many
terms the entry sums. The rounding error of evaluating an `n`-term dot product
grows with `n`: about `√n·ε·mag` typically and `n·ε·mag` (γ_n) at worst. A long
row's honest rounding therefore reads as an unexcused residual.

### How big

On BOYD2 the convex arm stops after 77 iterations and 44.5 s at a point the
benchmark verifier passes: relative primal `5.4e-15`, dual `2.1e-12`, gap
`1.1e-10`. The verdict reads `1.96e-7` against `tol = 1e-8` and labels it
`Solved_To_Acceptable_Level`.

- **The binding entry** is equality row 0: `b = 1.6e7`, 10 992 nonzeros, term
  magnitude `3.2e7`. Its `|Ax − b|` is `2.0e-6`, which is `6.3e-14` of the
  term magnitude, just above the `5.7e-14` floor.
- **The rounding there** would be `√n·ε·mag ≈ 7.5e-7` typically and
  `n·ε·mag ≈ 7.8e-5` at worst.
- **The raw (unfloored) error** is `1.16e-3`. It sits on rows with
  `b ≈ 2e11`, `2.9e-15` relative, and the floor already excuses it.

The cost used to be 270 s. The `Acceptable` label sent BOYD2 to the
LP/QP → NLP fallback, which ran to `Maximum_WallTime_Exceeded`. Since
`convex_answer_meets_tol` (`crates/pounce-cli/src/main.rs`), an in-tol answer
like this one skips the fallback. What remains is the label: a well-solved
large QP reports `Solved_To_Acceptable_Level`.

### What was not tried, and why

This entry does not change the floor. A floor that grows with row length
(`k·nnz(row)` or `k·√nnz(row)`) changes the convex arm's verdict, and through
it every reroute gated on `OptimalInaccurate`. That is a convex trajectory and
verdict change. It needs the fixture sweep, `benchmarks/qp`, and the σ-path
and acceptable-band tests named in `CLAUDE.md`, as its own change.

The HSDE stopping test (`hsde.rs`, "Absolute test always governs") uses the
same `REL_FLOOR_KAPPA·ε·scale` cap. A fix to one without the other would let
the solver stop where the verdict demotes, or the reverse.

### How to tell whether a fix works

- BOYD2 at `tol=1e-8` reports `Solve_Succeeded` from `cvx-qp` with the same
  point; the verifier reads kkt `1.1e-10`.
- `benchmarks/qp` gains no `Optimal` that the verifier fails.
- `issue414_cost_normalized_false_optimal.rs`,
  `illconditioned_huge_scale.rs` and the gh#984 / gh#1180 demotion tests
  (QPILOTNO: a real `1.57e-7` violation on rows whose floor is `1e-11`) stay
  green. A floor loose enough to excuse BOYD2's row must not excuse QPILOTNO's.

---

## NLP: one uninterruptible factorization can overrun `max_wall_time`, and the predictive guard misroutes when it fires

*Measured on `22ae155e`, Maros–Mészáros BOYD2 forced onto the NLP arm by the
LP/QP fallback (`tol=1e-8 max_wall_time=300`, `RAYON_NUM_THREADS=1`). The
misroute is from reading the code, not from a run. Recorded 2026-11-20.*

### What is wrong

**The overrun.** A FERAL factorization cannot be interrupted
(`dev-notes/feral-factor-interrupt.md`). The between-step deadline checks of
pounce#244/#245 therefore let one factorization overshoot.

- The proactive guard of pounce#254 (`factor_overshoot_predicted` in
  `crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs`) engages only once
  a factorization has cost at least `FACTOR_OVERSHOOT_BUDGET_FRACTION = 0.5`
  of the whole budget.
- On BOYD2 the convex arm spent 44.5 s, and the NLP arm got the remaining
  ~255 s (`charge_wall_budget`; the budget is handed down correctly).
- Endgame factorizations took 36–40 s each. `sample` shows them in
  `feral::dense::schur_kernel::axpy_minus_unroll4_nofma`, a large dense front.
- At 36 s, below 0.5 × 255 s, the guard stayed off. Iteration 1020's
  factorization started with 29 s left and finished 11 s past the deadline.
- The solve ended at 312 s (a repeat: 324 s) against `max_wall_time=300`.

**The misroute.** When the guard *does* fire, the deadline has not been
crossed yet.

- `PdFullSpaceSolver::solve` returns `false`.
- `IpoptAlgorithm::iterate` (`ipopt_alg.rs`, after
  `compute_search_direction`) asks `deadline_status()`, which reads
  `Deadline::exceeded()` and returns `None`.
- The `!ok` branch then sends the iteration to restoration instead of
  `Maximum_WallTime_Exceeded`.
- Restoration builds its own KKT solver, with no factorization estimate yet.
  So it can start the very factorization the guard refused, or end
  `Restoration_Failed`.

### Why the fraction was not simply lowered

Lowering `FACTOR_OVERSHOOT_BUDGET_FRACTION` to about 0.1 would bound BOYD2's
overrun: 36 s ≥ 0.1 × 255 s, and 29 s left < 36 s, so the guard fires. But it
would also make the misroute above fire more often, turning time-limit exits
into restoration entries. The guard has to terminate cleanly before it is
engaged more widely.

### How to tell whether a fix works

- A fired guard ends the solve `Maximum_WallTime_Exceeded` (or
  `Maximum_CpuTime_Exceeded`) at the last accepted iterate, never in
  restoration. One way: have the guard mark the deadline as spent, so that
  `deadline_status()` sees it.
- A unit test exists for each branch: 36 s factor / 255 s budget / 29 s left
  fires, and the existing small-factor and ample-remaining cases do not.
- A timed NLP run on BOYD2 (`solver_selection=nlp tol=1e-8 max_wall_time=N`)
  ends by `N` plus less than one factorization.
- The fixture sweep is unaffected, because it sets no time limit.

---

## NLP: infeasibility fixtures slower after the line search kept upstream's `alpha_min` (gh#1217)

*Measured on `claude/eloquent-faraday-e3oeac` against `main` at `22ae155`,
`scripts/sweep-fixtures.sh`, both legs; Ipopt 3.14.19 + MUMPS (the CasADi
3.8.2 wheel) as the upstream reference. Recorded 2026-11-21.*

**The change.** The backtracking line search used to clamp the filter
acceptor's `alpha_min` (upstream's `CalculateAlphaMin`) from below at
`1e-12` and cap a sweep at 50 trials. Upstream has neither. Both are gone.
The acceptor's value is used as is, and a sweep gets the trials it needs to
reach it. That fixed gh#1217: pounce now matches Ipopt's 64 iterations line
for line on the maximum-entropy model, where the clamp had sent iteration 20
to restoration instead of taking Ipopt's `alpha = 5.2e-23` after 75 trials.

### What is worse

The sweep moves 5 of 302 lines. `dallass` (lbfgs) goes from
`Error_In_Step_Computation` to solved. The other four keep their verdict at
a higher cost:

| fixture | leg | `main` | branch |
|---|---|---|---|
| `issue_508_infeasible_gap_1em2` | lbfgs | infeasible, it=65, tot=300 | infeasible, it=419, tot=1131 |
| `issue_508_infeasible_gap_1em4` | exact | infeasible, it=441, tot=1828 | infeasible, it=569, tot=2502 |
| `circle_parabola_t1p5` | lbfgs | infeasible, it=154, tot=707 | infeasible, it=154, tot=815 |
| `cc:design_cent_3` | lbfgs | solved, inner tot=10967 | solved, inner tot=11031 |

Each line moves where a step below the old clamp is now accepted. On
`issue_508_infeasible_gap_1em2` (lbfgs) the first is iteration 66,
`7.63e-16f` after 18 trials, where `main` went to restoration. These are
infeasible models orbiting their least-violation point. The accepted
steps lengthen the orbit before the verdict.

`issue_508_infeasible_gap_1em4` also needed the second gh#1217 change.
Without it, that line reads `MaximumIterationsExceeded` at 3000. The gh#1000
square-certificate counter compared the violation with `constr_viol_tol`
strictly, and this model's least violation is exactly the default `1e-4`.

### Why it is accepted

These trajectories are already off upstream's before the line search
matters. On `issue_508_infeasible_gap_1em4`, pounce and Ipopt agree to
eight digits through iteration 33 and part at iteration 46 on rounding
(`inf_pr` 11.2 against 9.65 at a dual infeasibility near `1e16`). Ipopt
then reaches the verdict at its first converged restoration, after 78
iterations. On `1em2` Ipopt takes 83 (exact) and 84 (lbfgs). The old clamp
reached the same verdicts sooner only by happening to send these orbits to
restoration. Restoring the clamp would bring back gh#1217's failure.

### Update: the floor holds in the switching region again (gh#1226)

*Measured on `claude/eloquent-cerf-rt8q3r` against `main` at `04d4ea5`,
`scripts/sweep-fixtures.sh`, both legs, 304 lines. Recorded 2026-11-21.*

Dropping the clamp everywhere also dropped it where the acceptor's floor
comes from the `δ·θ^{s_θ}/(−∇φᵀd)^{s_φ}` term, which is applied only for
`θ ≤ θ_min` and falls with `θ` itself. A nearly feasible iterate with a
poor direction then accepts null steps until `max_iter`. discopt's
nonsmooth `max`/norm NLPs do exactly that (gh#1226;
`crates/pounce-algorithm/tests/issue_1226_nonsmooth_null_steps.rs`). The
clamp now holds for `θ ≤ θ_min` and nowhere else. gh#1217's iteration 20 is
at `θ = 4e3`, so it is unaffected. Against `04d4ea5`, 3 lines move:

| fixture | leg | `04d4ea5` | `θ ≤ θ_min` clamp |
|---|---|---|---|
| `issue_508_infeasible_gap_1em4` | exact | infeasible, it=569, tot=2502 | infeasible, it=441, tot=1828 (`22ae155`'s) |
| `cc:design_cent_3` | lbfgs | solved, inner tot=11031 | solved, inner tot=9550 |
| `dallass` | lbfgs | solved (`mu=monotone` re-solve), it=174 | `Error_In_Step_Computation`, it=243 (`22ae155`'s verdict) |

The two other lines in the table above, `issue_508_infeasible_gap_1em2`
(lbfgs) and `circle_parabola_t1p5` (lbfgs), do not move. Their sub-`1e-12`
steps are taken outside the switching region, and they remain open as
described above.

**`dallass` (lbfgs) is a regression against `04d4ea5`, and it is accepted.**
Run with `print_level=5`, the two binaries agree through iteration 162.
At iteration 163 `04d4ea5` accepts `alpha = 1.04e-13` after 32 trials
along `‖d‖ = 1.14e-3`. That step moves the iterate by about `1e-16`, i.e.
it does not move it at all, at `θ ≈ 7.6e-15`, inside the switching region.
The first attempt then went on to `Solved_To_Acceptable_Level` at 414
iterations, and the `mu_strategy_fallback` re-solve was promoted. With the
clamp, iteration 163's sweep stops at the `1e-12` floor (both binaries
take iteration 162 at `1.67e-12`, above it), restoration follows, and the
solve ends `Error_In_Step_Computation` at 243 with no fallback, as it did
on `22ae155`. The win came from accepting a null step: exactly the class
of step gh#1226 stops accepting. No floor that rejects gh#1226's
`~1e-14` moves can accept this `~1e-16` one.

### How to tell whether a fix works

A fix for the two remaining lbfgs lines shortens them with both statuses
unchanged and leaves `crates/pounce-cli/tests/issue_1217_max_entropy.rs` and
`crates/pounce-algorithm/tests/issue_1226_nonsmooth_null_steps.rs` green.
The first pins a step below `1e-12` taken after more than 50 trials outside
the switching region. The second pins the clamp inside it. A fix for
`dallass` (lbfgs) has to reach its answer without iteration 163's null
step. The likelier lever is the L-BFGS model or the `mu_strategy_fallback`
trigger, not the line-search floor.

---

## Handover: two declared-pair fixtures end acceptable after crossover step 4 seeds from step 3 (gh#1206 B1)

*Measured on `claude/optimistic-lamport-8vweis` against `main` at `e0b4442`,
`scripts/sweep-fixtures.sh <bin> <out> ipm_sqp_handover=yes`, both legs, with a
third binary that reverts B1 alone to attribute every line. Recorded
2026-11-20.*

**The change.** When crossover's step-3 QP solves but its point misses the
stopping tolerances, step 4 (the full active-set SQP) now starts from that
QP's optimal working set instead of the step-2 estimate. Step 4's first step
QP is step 3's QP again, so the estimate made it redo every pivot step 3 had
paid for — 591 s and 167 working-set changes on gh#1206's 90 k-variable
collocation model. Off the handover path nothing moves: the default sweep and
the `crossover=yes` sweep are empty (one `tot=` count on `cc:qpec_small`
under `crossover=yes`, 240 → 224).

### What is worse

Under `ipm_sqp_handover=yes`, B1 alone moves 12 of 300 lines. Two are worse in
status, both exact leg, both declared-pair (`cc:`) fixtures, and both reach a
slightly *lower* objective:

| fixture | `main` | branch |
|---|---|---|
| `cc:design_cent_3` | `SolveSucceeded`, it=12, tot=614, obj 3.723369071 | `SolvedToAcceptableLevel`, it=21, tot=134, obj 3.723366071 |
| `cc:opf_vv_lower_knee` | `SolveSucceeded`, it=3, tot=247, obj −0.3954687711 | `SolvedToAcceptableLevel`, it=64, tot=281, obj −0.3954730625 |

The rest are better or neutral: `cc:qpec_small` 13 → 1 iterations (tot 153 →
85) exact and tot 329 → 208 L-BFGS; L-BFGS `cc:mixed_pair` 39 → 19;
L-BFGS `cc:opf_vv_lower_knee` 61 → 39; L-BFGS `mpcc_worse_local_solution`
31 → 28; `csfi2` 30 → 37 (below).

### Mechanism

`cc:design_cent_3`, third lowered solve, handover at interior iteration 12:
step 3's QP is `Optimal` with one pivot; step 4 from the estimate repeats that
pivot and converges in 5 SQP iterations. From step 3's set the iterates agree
with `main`'s to the printed digits through SQP iteration 4, where `main`'s
step QP refactors with a rank test and returns `‖p‖ = 2.4e-8` while the
branch's takes its warm factorization with no refactor and no rank test and
returns `‖p‖ = 1.14`. The model is the lifted MPCC with multipliers near
`1e26` in its step QPs — exactly the LICQ-failing, degenerate active set
gh#1206 §1 describes — and the next QP hits its iteration cap in phase 1, so
the attempt is declined `ActiveSetNotConverged` and the complementarity
driver takes a different path. So the defect is the QP accepting an
unchecked factorization on a near-singular working set; B1 changes which
working set reaches it. `cc:opf_vv_lower_knee` moves earlier: a different
accepted handover point in an early lowered solve sends the driver down
another sequence.

`csfi2` (30 → 37, same point) is the gh#797 negative-curvature probe reading
roundoff: the point step 4 returns differs at `1e-15`, and the probe then
measures `dᵀ(W+Σ)d = −1.85e-19`, escapes, finds nothing, and restores it.

### How to tell whether a fix works

Rerun the `ipm_sqp_handover=yes` sweep: both lines back to `SolveSucceeded`
with the improvements above kept.

**Re-checked after B6/B7 landed (same branch, 2026-11-20): not fixed.** The
handover sweep of the final branch shows both lines unchanged
(`SolvedToAcceptableLevel`, it=21 and it=64). That is expected rather than
informative: B6's rank probe is off by default (it was size-gated to KKT
dimension ≥ 5000 when this was checked) and B7's
seeded phase 1 is opt-in (see the entry below), and these models are far
below the gate, so neither runs on them. An ungated B6/B7 run flipped
`cc:design_cent_3` back to `SolveSucceeded`, but it ran concurrently with
builds, which moves the wall-clock attempt budget, so it is not evidence. A fix belongs in `pounce-qp`'s handling of a
warm start whose working set is rank-deficient (gh#1206 B6/B7), not in
reverting B1, which on the model that reported it is the difference between
a step-4 first QP of 591 s and one of 23 s.

### Review (gh#1214, 2026-11-20): the two lines have different mechanisms

Both lines are unchanged by gh#1214 (its `ipm_sqp_handover=yes` sweep is
empty). Traced with `sqp_print_level=5`, `POUNCE_QP_TRACE=1` and
`RUST_LOG=pounce::handover=debug,pounce::sqp::qp=debug`, they are two
defects, not one, and neither is "an unchecked factorization" alone.

**`cc:design_cent_3`: a zero-model step taken at full length.** Step 4's
iterates 0–3 shrink the step `1.0e-4 → 1.2e-6 → 3.2e-7 → 2.5e-8` with the
working set unchanged and `θ` down to `4.4e-16`. Iterate 4's step QP, warm,
one factorization, no rank test, takes one Newton step on that same working
set (`ws = 0`) with `‖p‖∞ = 1.14` and a QP objective change of
**`+1.9e-15`**: the reduced Hessian is singular along `p` and the gradient
is orthogonal to it, so the QP's minimizers form a line and the solver
returned a point far along it instead of staying put. The NLP's curvature
turns that free step into `f = −3.7234 → −4.6024` and `θ = 4.4e-16 → 0.19`,
and the filter accepts it: Armijo on `f` holds against a predicted decrease
of ~0, and `θ` is far under `θ_max`. From there the next step QP's Newton
steps run to `obj ≈ −6e31` (`max-iter`, 320 refactors), the cold retry
stalls in phase 1, the attempt is declined `ActiveSetNotConverged`, and the
interior point finishes on the gh#884 downgrade (unscaled dual
infeasibility `1.54e-2`, multipliers `~3.7e9`). Two candidate fixes, neither
tried yet: in `pounce-qp`, treat a Newton step whose model decrease is at
round-off as a null step (`x` is already a minimizer on the working set, and
on a PSD reduced Hessian the multipliers carry over when `Hp ≈ 0`); or in
the SQP, refuse a step whose predicted decrease is ~0 while it raises `θ`,
i.e. require such a step to be `θ`-type. B1 only changed which working set
reached iterate 4: on `main` that QP refactored with a rank test and returned
`‖p‖ = 2.4e-8`, so `main` never met the zero-model step at all.

**`cc:opf_vv_lower_knee`: a degenerate tail, at a better point.** The
lowered solve that decides the verdict hands over at interior iteration 8
(error `3.65e-8`) at a biactive pair on a curve knee. Step 4 halves its step
each iteration (`1.8e-6 → 4.5e-7 → 2.3e-7 → 1.1e-7`) with QP multipliers
`5.7e6 → 1.5e6` — linear convergence at a point without strict
complementarity — and the attempt is declined. The interior point then crawls
the same halving sequence for 64 iterations and ends acceptable. Its
objective, `−0.3954730625`, is *lower* than `main`'s `−0.3954687711`, so this
line is a status regression at a better point; the earlier lowered solves,
where B1 moved which handover was accepted, are what send the driver there.
A fix would be the driver's or the handover's handling of a biactive point,
not B1's.

---

## Active-set QP: the gh#1214 back-solve economies run only in the crossover (`QpOptions::schur_lean_backsolves`)

*Measured on `claude/peaceful-dijkstra-lpkx32` against its parent (gh#1206's
branch at `09380ae` merged with `main` at `994121e`),
`scripts/sweep-fixtures.sh` on five legs, both Hessian legs each, with a
binary carrying one environment switch per economy to attribute each line.
Recorded 2026-11-20.*

**The change.** gh#1214 cut the Schur path's back-solves three ways:

1. a repeated right-hand side reuses `K_0⁻¹b` (every step after a drop);
2. a drop of a slot active at the last reset solves once, not twice
   (`K_0⁻¹v = ±e_p + v_p·K_0⁻¹e_p`, exact);
3. refinement stops once the residual is inside its own rounding error.

On a step-3 crossover QP of a synthetic 15 k-variable collocation model
(413 working-set changes, unchanged) they take the back-solves from 2 298 to
1 409 and the back-solve time from 19.7 s to 12.6 s. (1) is bit-identical and
runs everywhere. (2) and (3) round differently from the path they replace,
so they run only where they were measured: the crossover's step-3 QP and
step-4 step QPs, whose `crossover=yes` and `ipm_sqp_handover=yes` sweeps are
empty with them on.

### What is worse, ungated

With (2) and (3) on for every Schur QP, the default, `crossover=yes` and
`ipm_sqp_handover=yes` legs are empty and these move (both Hessian legs for
the convex lines):

| leg | fixture | parent | ungated |
|---|---|---|---|
| `solver_selection=qp-active-set` | `convex_qp_qgfrdxpn` | `SolveSucceeded`/3553 | `MaximumIterationsExceeded`/23477 |
| `solver_selection=qp-active-set` | `convex_qp_qscfxm1` | 4242 | 9402 |
| `solver_selection=qp-active-set` | `convex_qp_qforplan` | 12465 | 14976 |
| `solver_selection=qp-active-set` | `convex_qp_share1b` | 2299 | 2259 |
| `algorithm=active-set-sqp` (exact) | `qpnstair` | `SolveSucceeded`/1 | `MaximumIterationsExceeded`/0 |

QGFRDXPN is pinned by `issue_1076_qgfrdxpn_active_set_prune_cap.rs`.
`qpnstair` loses its first step QP, a degenerate ℓ1-elastic phase 1 of
~3 000 pivots and 85 rank tests.

### What was tried

Attribution: (1) alone moves nothing. QGFRDXPN fails with (2) alone
(22 830) and with (3) alone (45 021) and solves only with both off;
`qpnstair` solves with either off, along three different paths (2 350,
2 629 and 3 600 pivots). Neither economy is less accurate than what it
replaces — (2) is exact where the back-solve it skips is not — so this is
the sensitivity the QGFRDXPN entries below already record (its outcome is
non-monotone in the prune cap), now to rounding. Three floors for (3) were
tried with (2) on: the worst-case bound `γ_{k+1}` (shipped), and `c·u` of the
row scale with `c = 1` and `c = 2`. All three lose QGFRDXPN (23 477, 26 635,
8 837 iterations); `c = 1` with (2) off ends it `SolvedToAcceptableLevel`.

### How to tell whether a fix works

Turn `schur_lean_backsolves` on by default in `QpOptions::default()` and
rerun the `solver_selection=qp-active-set` and `algorithm=active-set-sqp`
sweeps: QGFRDXPN and `qpnstair` must stay `SolveSucceeded`. A fix belongs in
what makes those solves follow the last bit (the degenerate phase-1 cycling
the QGFRDXPN entries describe), not in the economies.

---

## Active-set QP: the gh#1206 H-diagonal padding is size-gated, and the rank probe and hint-seeded phase 1 are opt-in (gh#1206 B6, P2, B7)

*Measured on the gh#1206 pounce-qp branch against `b49e6d4`,
`scripts/sweep-fixtures.sh <bin> <out> solver_selection=qp-active-set`, both
legs, with a binary carrying one environment switch per item to attribute
each line. Recorded 2026-11-20.*

Three of gh#1206's QP items help the reporter's 90 k-variable step QP and
cost something on the fixture corpus, which has no QP within two orders of
that size. They ship gated, and the gate is what this entry records.

### What is worse, ungated

| item | fixture (both legs) | `b49e6d4` | ungated |
|---|---|---|---|
| B6 rank probe | `convex_qp_qgfrdxpn` | `SolveSucceeded`/3553 | `MaximumIterationsExceeded`/41395 |
| P2 H-diagonal padding | `convex_qp_qforplan` | `SolveSucceeded`/12465 | `MaximumIterationsExceeded`/13769 |
| B7 seeded phase 1 | `warm_start_pin_repair.rs`, horizon 40, step −0.2 | 14 pivots | 111 pivots |

Also moved, same status: `convex_qp_share1b` 2299 → 2435 (B6 and P2 both),
`feasible_x0_sentinel_bound` 6 → 8 (B6), `manyineq_concave_qp`
22727 → 11508 (B6 and P2, better).

Under `algorithm=active-set-sqp` the ungated build (B6 and P2 together, not
separated) moved 17 fixture-legs: worse `mpcc_worse_local_solution` (exact,
`SearchDirectionBecomesTooSmall`/2 → `MaximumIterationsExceeded`/0) and
`pooling_rt2stp` (exact, `MaximumIterationsExceeded`/3 →
`InfeasibleProblemDetected`/4 after four ladder rungs); better
`pooling_rt2stp` (L-BFGS, `MaximumIterationsExceeded`/200 →
`SearchDirectionBecomesTooSmall`/39), `jit1`/`jit1_boxed` (L-BFGS, 21 → 11,
obj 265486 → 240694), `ssebnln` (exact, 3 → 2); `ssebnln` L-BFGS 95 → 133.
Gated, that sweep is empty.

### Mechanism

- **B6.** A KKT singular unshifted and at the first `H` shift, which factors
  once `−δ_c` sits on its live constraint rows, has dependent rows; the probe
  reports it instead of climbing to `1e14`. QGFRDXPN's elastic phase 1 meets
  such sets, and the old ladder "factored" them at `δ ≈ 1e6` with the null
  direction masked (measured on a duplicated-row fixture: 9 factorizations to a
  `δ = 1e6` "success"). Reported, they go to the rank repair, and pruning at
  those vertices is the cycle gh#1076 describes. The counting prune's cap
  already keeps the old ladder once it stands down; the failure-repair prune
  has no such cap.
- **P2.** The explicit zeros add nothing to the matrix but are stored
  entries; FERAL's ordering and scaling read the stored pattern, so round-off
  moves, and QFORPLAN's outcome flips on any change of pivoting (gh#1076: it
  solves at counting caps 16, 20, 32, 40 and cycles at 24, 28, 48, 64).
- **B7.** Started at the hint's full vertex, the Bland-ruled phase 1 releases
  the hint's wrong bounds one lowest-index pivot at a time. Dropping Bland for
  the seeded phase reached a point `8e-8` below the cold optimum, i.e. not
  feasible to tolerance.

### B6 gated at 5000 aborted BOYD2; it is now off by default

B6 first shipped gated at KKT dimension 5000. The PR #1208 re-review ran
Maros-Meszaros under `solver_selection=qp-active-set`, `max_wall_time=150`.
BOYD2 (93 263 variables) then ran past the wall-time limit: still inside one
FERAL dense factorization (`packed_schur_tiles_nofma`) at 500 s on the
reviewer's machine, against `main`'s clean `MaximumWallTimeExceeded` at
150.7 s. On a 15 GB Linux container the same run aborted after about 100 s
with `memory allocation of 55729914368 bytes failed`, inside
`factorize_with_inertia_control`, in FERAL's `grow_pooled_front`. Bisected
with one environment switch per item: switching off P1, P2 or P7 alone still
aborts; switching off B6 alone, or all four, restores `main`'s behaviour.

The mechanism: the probe's `−δ_c` entries sit on constraint-row diagonals
that the KKT does not store, so the probe matrix has a new pattern. FERAL
re-analyses it, and on BOYD2 the new ordering produces a 55.7 GB front.
Nothing before the numeric factorization bounds the fill, and an allocation
failure aborts the process rather than returning an error. So B6 is now
opt-in: `QpOptions::dual_shift_probe_min_dim` defaults to `usize::MAX`.
Making it safe would mean either reusing the existing pattern (store explicit
zero diagonals on the constraint rows too, as P2 does for `H`) or checking
FERAL's predicted fill before the probe's numeric factorization.

### The gates

B6: `QpOptions::dual_shift_probe_min_dim`, compared against the **KKT
dimension** (`kkt.dim`), default `usize::MAX` (off). P2: `n + m ≥
kkt::H_DIAG_PAD_MIN_DIM` (5000, `SOFT_DROP_MIN_DIM`, the size from which the
soft-drop pattern of gh#1083 is on for the same symbolic-reuse reason). The
two measure different things: a K_max on the Schur path has dimension about
`2n + m`, so a probe gate of 5000 would engage there at `n + m ≈ 2500`. B7:
`QpOptions::elastic_seed_from_hint`, off. With the gates the
`qp-active-set` sweep no longer shows the two losses; nothing in the corpus
reaches the gated branches, so the sweep says nothing about them — the unit
tests (`rank_probe_unit.rs`, `kkt_unit.rs`,
`warm_start_failed_repair_seeds_elastic.rs`) reach them on purpose.

### How to tell whether a fix works

Lower the gates to 0 (and turn B7 on) and rerun the `qp-active-set` sweep:
QGFRDXPN and QFORPLAN must stay `SolveSucceeded` and `warm_start_pin_repair.rs`
must pass with the option on. For B6, also run Maros-Meszaros BOYD2 under
`solver_selection=qp-active-set max_wall_time=150`
(`benchmarks/qp/generate_nl.py BOYD2`; the `.mat` is fetchable from
`raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/main/data/`):
it must stop at the wall-time limit. For B6 the likely shape is a cap on
probe-triggered prunes per pass, as `COUNT_PRUNE_CAP` caps the counting
trigger; for B7, a phase 1 that releases a hint's wrong-signed bounds by
multiplier rather than by index.

---

## NLP: L-BFGS fixtures slower after restoration matched upstream (gh#1146)

*Measured on `claude/charming-carson-cz5psg` against `main` at `358ecbb`,
`scripts/sweep-fixtures.sh`, L-BFGS leg, on **two architectures**: x86_64
(Linux) by the PR, arm64 (Apple Silicon, macOS) in the PR #1148 review.
Recorded 2026-11-07.*

**The change.** Under `hessian_approximation=limited-memory`, restoration's
quasi-Newton update ran over the whole restoration primal
`[x | n_c | p_c | n_d | p_d]`. Upstream (`update_for_resto_`) confines it to
the `x` block and leaves the linear slack blocks at zero curvature. Pounce
now does the same (`LimMemQuasiNewtonUpdater::leading_block`). On the gh#1146
model under `nlp_scaling_method=none` that turns `Restoration_Failed` at
`x = 0.0429` into Ipopt's own trajectory and answer (`x = 0.44927`,
14 iterations, iterate for iterate).

### What is worse

No status moved on either architecture, and the exact leg is untouched on
both. L-BFGS lines, `main` → branch:

| fixture | x86_64 | arm64 |
|---|---|---|
| `square_flowsheet_resto` | 2834 → **314** | total **3402 → 6210**: first attempt now at the cap, rescued later by `square_system_lm` |
| `dallass` | 301 → 272 | **218 → 279** |
| `circle_parabola_t1p5` | 66 → 154 (206 → 553 with ladder) | same |
| `eigena2` | 239 → 263 | same |
| `infeasible_square_scaled_1em4` | 24 → 26 (90 → 115) | same |
| `cc:design_cent_3` (lowered-solve total) | 4724 → 10967 | 4724 → 8210 |
| `cc:infeasible_pair` | 36 → 29 it | 36 → 29 it, driver total 3233 → 3672 |

Better on both: `cresc4` solves on the first attempt (was the ladder,
3200 total), `circle_parabola_t0p5` and `discs` no longer need the ladder,
`eigmaxa` 19 → 16, `issue981_water_main` 25 → 21 (100 → 67),
`issue_508_infeasible_gap_1em4` 79 → 73 (550 → 369),
`degenerate_start_infeasible` total 69 → 43.

### What is understood, and what is not

**`circle_parabola_t1p5` and `eigena2` (both architectures).** Both had
already left Ipopt's path in the **main loop, before any restoration call**,
on `main` and on the branch alike. Measured against Ipopt 3.14.20
(`amplpy` `ampl_module_ipopt`), same options, x86_64:

- `circle_parabola_t1p5`: Ipopt takes 31 iterations. Pounce leaves its
  path at main-loop iteration 2 (`‖d‖ = 7.66` against Ipopt's `25.7`).
- `eigena2`: Ipopt takes 167 iterations and first enters restoration at
  iteration 132. Pounce leaves its path at main-loop iteration 6.

So restoration is entered from points Ipopt never visits, and its
iteration count there measures that earlier gap, not this change.

**`square_flowsheet_resto` and `dallass` (opposite directions on the two
architectures): rounding-chaotic, so a single run is not a measurement.**

- *It is not FERAL's NEON kernels.* Cross-compiled for
  `aarch64-unknown-linux-gnu` and run under `qemu-aarch64`, both binaries
  reproduce the **x86_64** numbers exactly: on `square_flowsheet_resto`,
  `main` 2834 and the branch 314 and 197. Thread count (`RAYON_NUM_THREADS`
  1–16) changes nothing either. What differs on the reviewer's machine is
  therefore macOS itself, most likely libm's `exp`/`log` in the last ulp.
- *`square_flowsheet_resto` has a singular start.* `cond(J(x0)) = 1.5e19`,
  measured by central differences on the residual in
  `crates/pounce-cli/tests/square_system_recovery.rs`. Every solver's first
  step is a regularized guess. Ipopt 3.14.20 takes `‖dx‖ = 7.6e7` and pounce
  `1.1e7`, where the exact `J⁻¹c` is `5.6e16`; neither satisfies `J·dx = −c`
  to better than 40·‖c‖. Pounce leaves Ipopt's path at iteration 1, on `main`
  and on the branch alike, before any restoration.
- *Measured as distributions.* Perturbing `bound_push` to
  `1e-2·(1 + k·1e-12)`, k = 0…23 (x86_64, L-BFGS):

  | fixture, options | `main` | branch |
  |---|---|---|
  | `square_flowsheet_resto`, default ladder, `square_system_recovery=no`, `feral_increase_quality_retry=no` | 10 succeed, 5 false verdicts (`Infeasible`/`Restoration_Failed`), 9 at the cap | 6 succeed, 1 false verdict, 17 at the cap |
  | `dallass`, defaults | 10 succeed (7 Optimal, 3 Acceptable), 14 `ErrorInStepComputation`; median 239.5 it | 8 succeed (7 + 1), 16 `ErrorInStepComputation`; median 243.5 it |

  On `dallass` the two distributions are the same within sampling noise.
  *One ulp in one derivative is enough to move it.* The gh#1151 follow-up
  rewrote `asin'` from `1/sqrt(1 - u*u)` to `1/sqrt((1-u)(1+u))`, which at
  `dallass`'s interior arguments changes the gradient by about an ulp, and
  its sweep moved this line `SolveSucceeded`/272 → `ErrorInStepComputation`/243.
  Scaling the *old* `asin'` by `1 ± f64::EPSILON` alone gives
  `ErrorInStepComputation` at 396 (+) and 220 (−); the exact leg stays at 28
  under both. A single `dallass` L-BFGS line in a sweep diff is therefore not
  evidence about the change that moved it.

  The same sweep's other moved lines are the same ulp, measured the same
  way. Each baseline variant scales one old form by `1 ± f64::EPSILON`; a
  clean `main` build gives every row's first column on every rerun.

  | line | `main` | branch | old `asin″` ×(1±ε) | old `sqrt(1-u*u)` ×(1+ε) | ×(1−ε) |
  |---|---|---|---|---|---|
  | `exact dallass` | 28, `q=0` | 27, `q=5` | 28, `q=0` (both) | 28, `q=0` | 28, **`q=5`** |
  | `exact cresc4` | 68 | 69 | 68 (both) | **69** | 68 |
  | `lbfgs cresc4` | 222 | 101 | 222 (both) | 81 | 124 |

  - `sqrt(1-u*u)` is the old `asin′`/`acos′` (`dallass` uses `asin`,
    `cresc4` `acos`).
  - One ulp in it, with nothing else changed, reproduces the `exact dallass`
    escalations and the `exact cresc4` count.
  - It spreads `lbfgs cresc4` over 81–222, so neither 222 nor 101 is
    evidence about the change.
  - **On arm64 this line draws a costly realisation** (0.17.0 release
    sweep, v0.16.0 → `92a33e6d`, both built on Apple Silicon). `lbfgs
    cresc4` goes from first-attempt `SolveSucceeded`/105 to a first attempt
    that ends `Restoration_Failed` at 996, rescued by
    `start_point_perturbation=1e-2`: 111 iterations, **1107 total**, objective
    `0.8718975252` → `0.8718975449`. That is the ladder this line left on
    arm64 under gh#1146 (3200 total there), now back. Same chaos as the x86
    spread above, not a separate defect, but it is the arm64 cost a user
    sees. A fix to the underlying sensitivity would show as this line, and
    the x86 one, holding a first-attempt solve under both `1 ± ε` nudges.
    The other arm64 moves in that sweep are `exact dallass` 28/`q=0` →
    27/`q=5` and `lbfgs dallass` `ErrorInStepComputation`/279 →
    `SolveSucceeded`/162.
  - Perturbing the old `asin″` at its three second-order sites moves
    nothing. The escalations come from the gradient, not from the more
    accurate Hessian term.
  On `square_flowsheet_resto` the branch trades false infeasibility verdicts
  for budget exits. That is the right direction for the verdict, but it also
  succeeds less often over these 24 draws. Ipopt's own L-BFGS is *not*
  chaotic on this model: all 12 perturbations end "Converged to a point of
  local infeasibility" at 42 iterations, a false verdict, since the model is
  feasible by construction.
- *Consequence for tests.* No test pins this fixture's L-BFGS outcome.
  `issue850_increase_quality_regression.rs` reports both runs.
  `square_system_recovery.rs` and `issue857_escalation_gated_quality_rung.rs`
  cap `max_iter` at 150: across the same 24 perturbations, with their
  options, no branch base solve succeeds before 197 iterations (277 with
  `square_system_recovery.rs`'s options), so the budget exit those tests
  need is stable.

**The same trade beyond the fixture corpus: Mittelmann's `qcqp…nc`.**
Measured in the PR #1148 review on arm64 (macOS), `hessian_approximation=limited-memory`,
120 s per problem, both binaries run concurrently:

| problem | `main` | branch |
|---|---|---|
| `qcqp1000-1nc` | `Restoration_Failed`, 611 it | `MaxIter`, 3000 |
| `qcqp500-3nc` | `Restoration_Failed`, 428 | `MaxIter`, 3000 |
| `qcqp750-2nc` | `Restoration_Failed`, 1026 | `MaxIter`, 3000 |
| `qcqp750-2c` | `Restoration_Failed`, 290 | **`Solve_Succeeded`**, 2987 |

- **Same shape as `square_flowsheet_resto`:** restoration no longer gives
  up early, and the outer loop then runs to the iteration cap. Each `nc` row
  is a failure before and after, but the new failure spends the whole budget.
- **Objective:** in all three, the objective at the cap is lower than at
  the old restoration exit.
- **Suite totals:** 31 solved before and after over the 47 problems. On the
  30 both solve, 8,996 → 9,020 iterations.
- **Not traced:** these models were not reproduced for this PR. The `.mod`
  sources need AMPL to translate, and Mittelmann's server could not be
  reached from the PR's environment. Whether the extra iterations buy a
  better point or only spend the budget is open.
- **How to tell:** if a later change makes these exit early again, check
  that it does so without losing `qcqp750-2c`.

**`cc:` totals (both architectures).** `cc:design_cent_3` and
`cc:infeasible_pair` go through the complementarity driver's lowered solves
and have no direct Ipopt comparison. Not traced.

### How to tell whether a fix works

- `circle_parabola_t1p5` and `eigena2` under
  `hessian_approximation=limited-memory` should match Ipopt's iteration log
  through their first restoration call.
- For `square_flowsheet_resto` and `dallass`, compare the perturbation
  distributions above, not single runs. A fix for `square_flowsheet_resto`
  belongs at its singular first step (pounce leaves Ipopt at iteration 1),
  not in restoration.
- Do not restore the whole-compound update to win any of these back. It is
  what made the gh#1146 model fail, and
  `crates/pounce-cinterface/tests/issue_1146_lbfgs_feasible_not_infeasible.rs`
  (`unscaled_lbfgs_restoration_recovers_as_upstream_does`) pins it.

---

## Active-set SQP: what the gh#1173 filter costs where an uphill step used to pay off (gh#1173)

*Measured on `claude/pounce-issue-1173-s5pdm9` at `cb8d7c7` against `main`
at `8632d06`: `scripts/sweep-fixtures.sh` under `algorithm=active-set-sqp`,
`crossover=yes`, `ipm_sqp_handover=yes` and the default; models transcribed
from Vanderbei's CUTE `.mod` files where named. The Vanderbei-corpus counts
quoted are the maintainer's run on PR #1190 at `42dd480d`, before the fixes
below that run prompted. Recorded 2026-11-18.*

**The change.** The SQP filter accepted any feasible trial from a feasible
iterate: with `θ_curr = 0`, "`θ_t ≤ (1 − γ_θ)·θ_curr`" holds for every
feasible trial, so nothing guarded `f`. It now follows Wächter-Biegler /
Ipopt: near feasibility a descent step is f-type (Armijo on `f`, and still
filter-acceptable), θ-progress needs `θ_curr > 0`, every comparison carries
`Compare_le`'s `10·ε·|φ|`, a full step below `10·ε·(1+|x|)` is taken
unjudged (two in a row fail the search), and an uphill exact-Hessian step at
an exactly feasible iterate is re-solved on `H + δI`. On the Vanderbei corpus
at `42dd480d` that took the forced SQP from 519 to 558 successes (exact) and
511 to 536 (L-BFGS), `InternalError` from 23 to 2 and 26 to 3, and moved
seven wrong successes onto the reference optimum.

### What is worse

Every entry below was, on `main`, a run rescued by an uphill step that
happened to land somewhere good. The new rule refuses those steps; none of
these is a wrong answer.

- **`mu_fallback_declined_bound_active`, exact, 7 → 21 iterations.** `main`
  goes `f` 1.73 → **223.7** → −2.10 → **76.6** → −3.0. The new run is
  monotone; its steps are short because the δ-shifted indefinite model
  under-steps.
- **`maratosb` (CUTE), limited-memory: `Solve_Succeeded` →
  `Maximum_Iterations_Exceeded` at the optimum `f = −1.0000000625`.**
  `x₁ + (x₁²+x₂²−1)²/1e-6` is conditioned ~1e6. `main` reached it through a
  step to `f = 1.2e9` whose curvature pair taught L-BFGS the valley; the
  Armijo steps that replace it (α ≈ 3e-8) never do, and stationarity
  oscillates around 1e-4. The exact-Hessian leg is unaffected.
- **`drcavty1` (CUTE, n = 10816), forced SQP and `ipm_sqp_handover=yes`:
  handover accepted at IPM iteration 136 and solved in 48 s on `main`;
  declined twice and out of wall time at 120 s now.** The handover SQP's
  iteration 2 is f-type and halves a full step that raised `f` from 1.22e-9
  to 1.34e-9. Two iterations later the warm-started step QP reports
  `warm start pins infeasible`, re-seeds from the interior point, and spends
  **28.8 s in 227 factorizations** until the handover's time limit declines
  the attempt. The filter decision is Ipopt's; the cost is the warm-start
  fallback of `pounce-qp` at this size, which `main`'s trajectory never
  reached. `drcav1lq` is reported the same way on the maintainer's corpus
  (not reproduced here: the transcription times out on both binaries).
- **`mpcc_worse_local_solution`, `crossover=yes`, limited-memory:
  `f = −84.33` → `−10.12`, both `Solve_Succeeded`.** The interior point
  returns a KKT point at −10.12 with `θ = 1e-8`. `main`'s crossover SQP took
  a full step that raised `f` (θ-progress), left that point, and converged at
  a different local minimum, −84.33, in 13 iterations. The step is f-type
  now; the run reaches −84.33 too but not within `crossover_max_iter`, so
  crossover is declined and the interior point's own answer stands. Crossover
  is a polish, so this is the intended outcome, but it is a different answer.
- **`jit1_node`, `ipm_sqp_handover=yes`, exact: objective 173345.38 →
  173346.10 (relative 4e-6), both `Solve_Succeeded`.** The handover is now
  accepted at its first attempt (IPM iteration 18), where `main` declined
  twice and accepted at 21. The SQP from there converges (scaled dual
  infeasibility 4.4e-10) to a neighbouring KKT point, the same one the
  forced L-BFGS SQP reaches.
- **`cc:design_cent_3`, `ipm_sqp_handover=yes`: lowered-solve iterations
  524 → 614**, same answer and verdict.
- **Vanderbei exact leg at `42dd480d`: `catena`, `hanging`, `kiwcresc`,
  `polak2`, `polak6`** stopped at iteration 0 with
  `Search_Direction_Becomes_Too_Small` and now run 175–200 iterations (to
  the wall time on `catena`) before failing the same way; `pfit2`/`pfit3`
  51–55 → 102–114, `powellbs` 36 → 93, `sineval` 2 → 42 where both succeed.
  Not re-measured on the final code, whose tiny-step rule ends stalled runs
  sooner (on the fixtures: `mpcc_scholtes4_biactive` 200 → 51).

### What has been tried

- Bitwise "trial equals `x`" as the stall test: missed `jit1_node` (its
  `α·p` is 2e-19, not zero) and, applied at `α = 1`, broke the zero-primal
  multiplier step `issue_986_sqp_working_set_fixed_vars` converges with.
  Replaced by the tiny-step rule.
- Armijo alone on f-type steps: `pooling_rt2stp` (limited-memory) took a
  step raising `θ` from 1.4e-3 to 0.85 that the filter rejected and never
  recovered (−3138 → −1881). Every trial is now also filter-checked.
- The slack on the f-type test only: `stancmin`'s round-off multiplier step
  has `∇fᵀp = +3e-15`, is h-type, and raised `φ` by one ulp.

### How to tell whether a fix works

- The candidate is Ipopt's watchdog (a bounded number of tentative full
  steps accepted without the filter, rolled back if the following
  iterations do not make progress): it is how Ipopt gets `main`'s good
  uphill steps without its bad ones. It must keep
  `issue_1173_filter_never_steps_uphill_on_an_unconstrained_model` and the
  issue's three cases (`sqp/tests.rs`) green, since those are uphill steps
  that went nowhere.
- `maratosb` (L-BFGS) and `drcavty1` (forced SQP, n = 10816) should solve
  again; `mu_fallback_declined_bound_active` (exact) should approach 7.
- For `drcavty1` alone, a fix to `pounce-qp`'s warm-start fallback (the
  28.8 s interior re-seed) would recover it without touching the filter.
- Re-run the Vanderbei forced-SQP legs: success counts must not fall below
  this change's.

---

## Active-set SQP: what the multiplier-scaled stopping test costs (gh#1191)

*Measured on `claude/eloquent-knuth-1ypxda` at `e6d8b03` against `main` at
`5463adb`: `scripts/sweep-fixtures.sh`, both legs, under the default options,
`algorithm=active-set-sqp`, `crossover=yes` and `ipm_sqp_handover=yes`.
Recorded 2026-11-19.*

**The change.** The SQP's stopping test compared the raw stationarity
residual with `sqp_tol`. It now divides the residual by Ipopt's multiplier
scale, `s_d = max(100, mean |λ|)/100`, first. The raw residual must still meet
`sqp_dual_inf_tol`. This was a deliberate choice: with large multipliers the
residual has a round-off floor above an absolute `1e-8`. `hs99exp`
(`s_d ≈ 91`) sat at 4e-8 to 1e-7, and certified only when noise happened to
dip below `tol`, which on the CUTE corpus `.nl` it never did.
`hs71_obj1e8` also certifies now, on both legs. A solve can stop only at the
same iterate or an earlier one.

### What is worse

The cost is accuracy where the SQP is a polish. Under
`ipm_sqp_handover=yes`, the polishing SQP now stops at an earlier, less tightly
converged iterate. All of these were `Solve_Succeeded` before and still are:

| fixture | `main` objective / dual inf | now |
|---|---|---|
| `deb7` (exact) | 97.55993795 / 7.5e-6 | 97.55998637 / 5.4e-7 |
| `mu_fallback_point_floor` (lbfgs) | −38315031.51 / 2.9e-11 | −38315032.24 / 5.9e-5 |
| `cc:opf_vv_lower_knee` (exact) | −0.3954730647 / 3e-15 | −0.3954687711 / 1.2e-6 |

The first two each improve one residual and give up the other. `deb7`'s
objective rises by 5e-7 relative, with a smaller dual infeasibility.
`mu_fallback_point_floor`'s objective falls, while its dual infeasibility
rises to 5.9e-5. `opf_vv_lower_knee` loses on both: its objective moves by
1e-5 relative, and its natural residual rises from 0 to 7.8e-11.

Under `crossover=yes` three objectives move, all still `Solve_Succeeded`:
- `deb7`: 97.55996651 → 97.55993795.
- `pooling_rt2stp` (lbfgs): −3273.954895 → −3273.954992.
- `mpcc_qpec_small_biactive`: 6.7e-74 → 5.3e-17.

### How to tell whether a fix works

Holding the handover's polish to the absolute test while keeping `s_d` for a
forced `algorithm=active-set-sqp` solve would recover the three handover
answers. It must keep `hs99exp` and `hs71_obj1e8` certifying on the forced
leg (`crates/pounce-cli/tests/issue1191_sqp_endgame_certifies.rs`). The three
handover lines above are the check.

---

## Active-set SQP: `ssebnln` (limited-memory) fails slower than it did before gh#1191

*First measured on `claude/eloquent-knuth-1ypxda` at `5b1724e` against `main`
at `5463adb`; updated on `main` at `9c0ab66`, after gh#1195's correction-QP
bounds (PR #1198). Forced `algorithm=active-set-sqp
hessian_approximation=limited-memory`, `RUST_LOG=pounce::sqp::qp=debug` for
the QP lines. Recorded 2026-11-19.*

**What is worse.** `ssebnln` fails before and after gh#1191
(`Search_Direction_Becomes_Too_Small`, f ≈ 1.617063e7; the exact leg solves
it in 3 iterations at 16170600). gh#1191 made it fail later:

| | iterations | cloud session |
|---|---|---|
| `main` before gh#1191 (`5463adb`) | 80 | 33 s |
| gh#1191, before gh#1195's bounds | 95 | 61 s |
| `main` with both (`9c0ab66`) | 95 | 38 s |

(On the maintainer's machine the middle row was 144 iterations and 384 s,
past `sweep-fixtures.sh`'s 300 s cap. The iteration counts differ between
machines, the mark of a round-off-sensitive path through a degenerate
endgame: stationarity there is 1e10 to 5e17, and the multipliers diverge.)

**Cause.** The trajectories agree to iteration 63. There the damped BFGS's
stale-scale shift (gh#1191 fix 2) fires with `sᵀy/sᵀBs = 12.8`, just over its
threshold of 10, because the diverging multipliers have moved, and again at
70. The new trajectory is 15 iterations longer, and it reaches two
second-order-correction QPs that cannot be had cheaply.

**What gh#1195 fixed.** Those two correction QPs ran to the step QP's budget
in every pass of the QP's elastic phase and its recovery stages: 3322 and
3251 refactors, 15.6 s and 12.6 s, then discarded. The correction QP now gets
one pass's iteration budget in total, and a pass that goes a tenth of it
without improving ends early (`soc_qp_options`). They stop after 791 and 712
refactors, 3.4 s and 3.1 s. The four correction QPs that come back usable on
this model are unchanged.

**What remains.** 38 s against 33 s, and the gap is the two capped
corrections: they still cost 6.5 s, and without them the 95 iterations take
about 31.5 s. Both end on the stall bound, after about 900 of their 2880
iterations; a tighter bound that would end them sooner would also cut usable
corrections elsewhere (`pooling_rt2stp`'s took 0.6 of the budget).

**How to tell whether a fix works.** `ssebnln` (limited-memory, forced SQP)
fails in 33 s or less on the cloud session, with the debug lines for its four
usable correction QPs still reading `optimal`, and the fixture corpus's other
usable corrections (`pooling_rt2stp`, `jit1_node`) still reading `optimal` too.
Fixing the SQP's endgame (why the multipliers diverge, so the trajectory never
asks for these corrections) would also do it.

---

## Active-set QP: working-set steps can climb on solve noise (gh#1195)

*Measured on `claude/pounce-issue-1195-owd2q0` against `main` at `9c0ab66`.
Recorded 2026-11-19.*

**What happens.** A working-set step should never raise the QP objective:
with the reduced Hessian made positive definite by a shift `δ ≥ 0`,
`∇q(x)ᵀp = −pᵀ(H + δI)p < 0`. In the elastic phase 1 the multipliers are of
the elastic penalty's size (10⁶ and up), and a KKT residual near round-off
times those multipliers can flip the sign of a computed step. Replaying the
229 step and correction QPs of `pooling_rt2stp`'s forced-SQP limited-memory
run (each from its own warm start, without gh#1195's correction bounds),
45 250 full steps raised the objective, most of them δ-shifted, many outside
phase 1. Taken, such a step moves `x` uphill, and a drop/re-add cycle that
moves `x` on every lap is one Bland's rule cannot end. Where the cost showed
— the SQP's second-order-correction QP — it is bounded instead (gh#1195,
PR #1198); the steps themselves are still taken.

**What was tried and dropped.** Reading a climbing step as a null step
(stationary on its working set, read the multipliers):

- *Every step.* `hs13_bigstart` (limited-memory, forced SQP) turned from
  success to `Search_Direction_Becomes_Too_Small`: at its cusp, two nearly
  parallel active rows in two variables, the climbing "step" is pure error,
  but the SQP had come to depend on taking it. `deb7` went `MaxIter` at
  iteration 1 until steps with a pin-repair right-hand side were excluded.
- *δ-shifted steps.* Their true descent `δ‖p‖²` can be smaller than the noise
  while the point is not stationary: refusing them cut
  `haverly_multistart_t3_brf0`'s elastic pass short of a proximal crawl that
  was still lowering the objective (14 → 22 SQP iterations).
- *Unshifted steps of the elastic phase, refactor path* — the one case where
  the arithmetic is exact. It left an unshifted fixed-point cycle (a bound
  dropped on a wrong-sign multiplier, re-added at `α = 0` by the shifted step
  after it; 10 795 working-set changes, 42 s on one `pooling_rt2stp` step QP)
  that needed the Schur path's cycle guard copied to the refactor path. With
  both: on 200 generated convex elastic QPs, 174 `Optimal` instead of 172 (3
  better, 1 worse), working-set changes −22%, time −16%;
  `manyineq_concave_qp` under `qp-active-set` 22727 → 6058 iterations; but
  `pooling_rt2stp` (limited-memory, forced SQP) `MaxIter`/200 at −3134.83 →
  `Search_Direction_Becomes_Too_Small`/128 at −3126.96, and the replay's
  climbing-step count barely moved (45 250 → 47 412), since most of those
  steps are shifted. Dropped as a small and uncertain gain, unmeasured on
  `benchmarks/qp`.
- *The same on the Schur-update path.* 3 of 300 generated QPs better, 3
  worse; Maros-Meszaros `QSCFXM1` 4242 → 9861 iterations on one correct
  refusal that reordered its pivots at a degenerate vertex.
- *Iterative refinement.* Every solve is already refined by the backend
  against the original matrix; what is left is conditioning.

**How to tell whether a fix works.** The climbing-step count on the
`pooling_rt2stp` replay falls well below 45 250, `haverly_multistart_t3_brf0`
(limited-memory, forced SQP) stays at 14 iterations, `hs13_bigstart` still
succeeds, and `benchmarks/qp` under `solver_selection=qp-active-set` does not
move against `main`.
---

## MPCC: four MacMPEC answers stay complementary only to ~7e-5

*Measured on `mpcc/cost-sweep-report` (after `df3fbd12`), MacMPEC through
the CLI. Recorded 2026-11-04.*

**Scope.** Answers whose branch re-solve (the "branch polish", see
`docs/src/complementarity.md`, "Complementary to `tol`, not to `sqrt(tol)`")
is refused, so the reported point keeps a nearly biactive pair at
`sqrt(τ)`. The report says so: `complementarity.natural_residual` and
`complementarity.polish.reason`, and `pounce-studio diagnose` warns
(`mpcc_loose_pairs`).

### What is wrong

| model | `natural_residual` | class | why the re-solve was refused |
|---|---|---|---|
| `pack-comp1-8` | 6.9e-5 | S | the branch is `Infeasible_Problem_Detected` |
| `pack-comp1p-8` | 6.9e-5 | S | the same |
| `pack-comp1p-32` | 7.0e-5 | W | the same |
| `incid-set2-32` | 6.4e-7 | S | the re-solved point certifies only W and a branch descends from it |

The objectives are at the best known to about 1e-8 relative
(`pack-comp1-8`: 0.6000000082 against 0.6), so the cost is the residual, not
the value.

### Already tried, did not work

On `pack-comp1-8`, 33 of its 49 pairs have `min(|G|, |H|)` above `10·tol`.
About 30 of them are strictly complementary, with sides of about 1e-6 and
1e-3 to 1e-2, so their product is just `τ`. Four are nearly biactive, with
sides within a factor of 30: pairs 9, 13, 17 and 46. Holding each pair's
other side, retried once from the same warm start when the first branch
fails:

- **Every loose pair.** This is wrong by construction: it pins the 1e-3
  sides of strictly complementary pairs. It is infeasible on all three
  `pack-comp1` models. On `opf_vv_lower_knee_held` it flips only that
  model's one loose pair, and the other branch is infeasible there too:
  solved directly, `s[1] = 0` with `r[1] >= 5e-5` reports local
  infeasibility.
- **Only the pairs whose sides are within 100× of each other.** Infeasible
  on all three.
- **Each of the four nearly biactive pairs alone.** Infeasible every time.

So the orientation of the nearly biactive pairs is not what makes the branch
infeasible. The retry was removed: it cost a solve on every refused polish
and gained nothing that was measured. Holding both sides of a pair is no
alternative, because that set lies inside both one-sided branches.

Untested: whether the branch NLP is truly infeasible or only locally so from
the warm start. A cold solve of the same branch, or a feasibility problem
with slacks on the held sides, would tell.

### How to tell whether a fix works

`make -C benchmarks macmpec-run` with `grep -l "keeping it"` over the logs.
The four models above should move to "re-solved on its branch", and nothing
else should move from at to worse. The `cc:` lines of
`scripts/sweep-fixtures.sh` must stay as they are except where the fix
intends them to change. `opf_vv_lower_knee_held.nl` (one branch infeasible)
and a both-branches-infeasible variant are the fixtures to test it against.
In the variant, `s[1] >= 5e-5` is added as well; the generator is in the
data root, `mpcc_opf.model.build` plus one `Constraint`.

---

## MPCC: what the coupled continuation costs where it falls back

*Measured on `mpcc/fast-continuation` (`5180695f` against `ecc2e4ff`, which
`complementarity_continuation=schedule` reproduces exactly). Recorded
2026-11-06. Tracked with the rest of pounce's MPCC cost in #1136.*

**Scope.** The default continuation (`complementarity_continuation =
coupled`, `docs/src/complementarity.md`) solves `τ = 1`, then one solve with
`τ` tied to `μ`, then the exact finish from three starts. Where its answer
does not pass the schedule's test, or the coupled solve reaches its budget,
the schedule resumes at `τ = 0.1` from the `τ = 1` solve, and what the coupled
attempt spent is the cost. Every case below falls back or loses an answer
the schedule found; none changes a status.

### What is worse

| case | schedule | coupled | why |
|---|---|---|---|
| fixture `ralph1`, lbfgs leg | 321 iterations | 584 | the coupled solve reaches the floor; all three finish attempts (100 each) miss; the schedule runs |
| fixture `opf_vv_lower_knee`, lbfgs leg | 554 | 804 | the same |
| OPF transmission (59), `mpcc-compare` | 240 s | 270 s (+12%), answers identical | on the IEEE 300 cases the coupled solve crawls toward the floor (about 3100 iterations unbudgeted, against about 900 for the whole schedule); its budget, `max(5 × first solve, 200)`, hands over |
| CALIPSO bunny-hop seed 1 | 462.605 | 463.195 (+0.13%) | both cheap routes reach the same exact answer, so the cold exact product, which finds 462.605 from this start, is skipped (`skips_cold_solve`); under the schedule the continuation's finish fails and the cold solve runs |

Elsewhere it is ahead: MacMPEC 158 at / 0 unsolved against 157 / 1 and
1149 s → 1000 s, `relu_nn` −23% wall, OPF and capacity −27% iterations,
CALIPSO ball-in-cup seed 1 better.

### Already tried, did not work

- **Each rung to `tol = κ·τ`** instead of coupling: −3% on `relu_nn`, and on
  MacMPEC 35 outcomes changed and `tap-15` turned `Infeasible_Problem_Detected`.
- **Coupling from `τ = 1`**: MacMPEC `bilevel1` 0 → 5 (the full `τ = 1` solve
  is what reaches the basin); net 154 at / 9 worse / 1 FAIL.
- **`τ = 1` first, the coupled solve warm at its `μ`**: crawls (`digits_128x3`
  251 iterations against 66 with `μ` restarted at `τ`).
- **A cleanup rung at `τ = 1e-8` before the finish**: converges in one
  iteration; the finish still fails. A monotone `μ` for the finish: no effect.
- **`κ < 1`** (`τ` reaches its floor earlier, so `μ ≪ τ` at the end): the
  finish still fails from the same endpoints.

The finish failures are the multipliers, not the point: the coupled and
scheduled endpoints agree to 1.4e-7 in `x`, but at degenerate pairs the
product-row multipliers differ (0.50 against 0.18 on rows at `G·H ≈ τ/2`).

### How to tell whether a fix works

Sweep against a baseline binary (`scripts/sweep-fixtures.sh`) and check the
two lbfgs `cc:` lines above; run `make -C benchmarks mpcc-compare` (the
transmission suite's wall time against `complementarity_continuation=schedule`)
and CALIPSO declared (`benchmarks/calipso`, the `declared` route, bunny-hop
seed 1). A fix should move those toward the schedule's numbers without
giving back MacMPEC's 158 / 0 or `relu_nn`'s time. The console line "the
coupled continuation did not reach an exact answer; resuming the schedule"
marks every fallback.

## MPCC: the exact-product stage is slow on dense degenerate QPECs

*Measured in the PR review of declared complementarity (jkitchin/pounce#1103),
on MacMPEC through the CLI. Recorded 2026-10-31.*

**Scope.** Models with declared complementarity pairs whose complementarity
rows are dense, under the default `complementarity_strategy=both` or `=exact`.
The qpecgen family in MacMPEC is the measured case. Models without pairs are
unaffected.

### What is wrong

The exact-product stage lowers each pair to `G·H = 0`, which has no strict
interior. On `qpec-100-2` (110 variables), run with
`complementarity_strategy=exact` to isolate the stage, it takes **561
iterations and 8.3 s**:

- the multipliers grow to about 1e12;
- it exits through the scale-relative dual-infeasibility floor (`dual_inf`
  612 against a dual scale of 1.6e12), not a converged KKT error;
- 5.0 s of the 8.3 s is Lagrangian Hessian evaluation, because qpecgen's
  complementarity rows are dense (11 020 Jacobian nonzeros on 110 variables).

Under the default `both`, the stage runs for every solve. `qpec-200-2` takes
**434 s** there, but that total also includes about 4 200 continuation
iterations, and the exact stage was not timed alone on it; how much of the
434 s is the exact stage is not measured. Only the `qpec-100-2` numbers
above isolate it.

### What has been tried

`complementarity_strategy=continuation` skips the stage and is the
workaround for a model known to behave this way. The stage is kept in the
default because on the Gate 0 corpus and the CALIPSO bunny-hop it finds
answers the continuation misses (see `docs/src/complementarity.md`).

**A relaxed exact stage (jkitchin/pounce#1121, 2026-11-01).** Solving
`G·H <= 1e-6` from the start, then a warm step to `1e-8` and the exact
finish, instead of `G·H = 0` cold. On the 28 MacMPEC models where the stage
costs or matters most (`complementarity_strategy=exact`) it takes 1,879
iterations against 25,712, converges on 26 against 20, and lands at or below
the published value on 23 against 19. `G·H <= 0` gains almost nothing (23,439
iterations): no interior either. A thinner `1e-8` from cold does worse than
`1e-6` (3,311 iterations, 20 at). Two things did not work:

- *Replacing* the cold solve: MacMPEC time −21%, but `water-net` moves from
  927 to 953 (the relaxed path settles on another branch) and two CALIPSO
  bunny-hop starts swap between 462.605 and 463.195. The two solves find
  different minima and neither dominates.
- Pinning each pair's smaller side at the relaxed point and solving the
  branch NLP (#1109's machinery): it does not recover `water-net`'s 927, and
  costs 266 extra iterations on `pack-comp1-32`.

So #1121 runs both and keeps the better certified answer. That is the
accepted cost recorded here: on the 167 public MacMPEC stubs, wall time
**+19%** (3,177 s to 3,767 s) and iterations **+37%** (78,431 to 107,563),
measured in clean sequential runs against `main` at `29ebd48`; CALIPSO's
declared route 170 s to 209 s. In exchange no answer is lost and three
improve (`dempe` 28.25 and `ex9.2.5` 5, both below their published values;
`pack-comp2p-8` reaches its published value). `pack-comp1p-32` still times
out at 300 s: the relaxed path alone solves it (0.704 against a published
0.653), and the cold solve's budget is what pushes the run over.

**Later (2026-11-02), and what remains.** Three rules cut MacMPEC's time
from 3,663 s to 2,528 s with no answer changed (see
`docs/src/complementarity.md`): a finish no longer replaces a better relaxed
point, the cold solve is skipped when the continuation and the relaxed path
agree, and a failed relaxation step is retried warm. Measured while finding
them: POUNCE's per-iteration cost is at parity with stock Ipopt + MUMPS on
the same lowered NLP (pack-comp1-32's `τ = 1` rung: 64 against 73 ms per
iteration), and stock Ipopt stalls on the cold `G·H = 0` solve the same way,
so the remaining time is in how many exact-product iterations the schedule
spends, not in the linear algebra. Three models timed out at 300 s when
this was written (the run at `5882e01`):

- `siouxfls`, `siouxfls1`: the first relaxation (`τ = 1`) alone takes about
  1,000 iterations and is solved twice (~110 s), and the `1e-7`/`1e-8` rungs
  struggle in restoration; the published value is already reached at
  `τ = 1e-6` (`siouxfls`) and `1e-7` (`siouxfls1`). With
  `complementarity_strategy=continuation max_cpu_time=900`, `siouxfls`
  reaches 208.258. An early stop when successive rungs stop moving the
  objective would trade product tightness for time.
- `pack-comp1p-32`: the continuation reaches 0.65304 (published 0.652979)
  in ~50 s; the exact stage's relaxed path and the cold solve take the rest.

The rerun after #1126 and gh#1128 (`benchmarks/macmpec/README.md`, Reference
result) moves this list: `pack-comp1p-32` and `siouxfls1` now finish inside
300 s, `siouxfls` still does not (676 s), and `qpec-200-2` joins it (335 s).

### How to tell whether a fix works

`qpec-100-2` under `complementarity_strategy=exact` should take a small
fraction of 561 iterations without changing its objective (time `qpec-200-2`
per stage first if it is to be the second witness), while the Gate 0 corpus
(`benchmarks/mpcc/declared.py`) stays at 60/60 cells at `f*`. (Neither model
is among the 167 public MacMPEC stubs #1121 measured; they are in the data
root's compiled set.) A cheaper schedule than #1121's two solves works if,
on the public MacMPEC stubs, it keeps 136 at / 12 better / 3 worse with
`water-net` at 927 and `dempe`, `ex9.2.5` below their published values, in
less than 3,767 s, and changes no CALIPSO declared-route answer. Integrated
approaches that tie the relaxation to the barrier parameter (relaxing the
products by an amount proportional to μ inside one solve) avoid the empty
interior and are the obvious direction. The MacMPEC suite and its harness are
in the benchmark data root (`macmpec/`).

---

## NLP: qcqp1500-1c hits the CPU limit at the default

*Not tracked. This entry is what remains of gh#1098, which was closed on
2026-11-01 when #1117 fixed `qcqp1500-1nc`. `qcqp1500-1c` has no known
path to a fix, so per the tracker's planned-work-only rule it has no
issue; its signature also differs from the crawl #1098 described. (#1098
also listed `waterx` reaching a different local minimum than Ipopt; that
is not a limitation for a local solver, so it is not recorded.) The 1nc crawl came
in with 49ab28aa (#1077), which put pounce on Ipopt's trajectory; Ipopt
crawls the same way. Recorded 2026-10-28, revised 2026-10-31 and
2026-11-01.*

**Scope.** The default exact-Hessian NLP path. The L-BFGS leg solves both
`qcqp1500` models.

### What was fixed

Mittelmann `qcqp1500-1nc` sat at μ ≈ 1.8e-6 for about 1200 iterations and
hit the 300 s CPU limit. The diagnosis is in
`dev-notes/qcqp1500-crawl-slack-perturbation.md`: inertia correction puts
`δ_s = δ_x` on every slack row, and on the thousands of inactive rows that
term is the whole curvature contribution, so successive regularized
directions disagree. Since 2026-11-01 the default is a row budget,
`δ_s = perturb_slack_fact · min(1, perturb_slack_max_rows / m_d) · δ_x`
with `perturb_slack_max_rows = 1000`, and `qcqp1500-1nc` solves in
128 iterations. Models with at most 1000 inequality rows are
untouched; `perturb_slack_max_rows=0` is upstream's rule exactly. The
Mittelmann before/after table is in the dev-note.
The budget's cost, Vanderbei `cresc132`, is the next entry.

### What is still wrong

- **`qcqp1500-1c`** still runs out the 300 s CPU limit, at the default and
  under every rule measured for 1nc. Its signature is different: the
  inertia correction never fires in the stretch it reaches (main and the
  budgeted default are identical through every row), so the slack
  perturbation has nothing to act on; the dual infeasibility stays near
  1e6–1e7 while `inf_pr` falls, with line searches of up to 17 trials, and
  it is around iteration 100–125 when the clock runs out.
  `ftb_centrality_floor=0.01` with
  `perturb_slack_fact=0.01` solves it (148 iterations) but turns the
  `discs` fixture into a false `InfeasibleProblemDetected`, so neither is
  on by default.

### What has been tried

- A fixed `perturb_slack_fact` (0.01–0.3): solves 1nc at 0.03–0.1, never
  1c; breaks `cresc4` / `discs` at 0.01 and 0.1; moves 22 CLI tests at 0.03.
- `(n/m_d)^p`, norm-based and per-row `Σ_s`-scaled rules for `δ_s`: each
  measured and rejected in the dev-note (wrong basins on `discs`, retries
  on `cresc4`, 2556 iterations on `pooling_rt2stp`).
- `ftb_centrality_floor` (γ > 0): solves 1c with a small factor, false
  infeasibility on `discs`; opt-in, default 0.

### How to tell whether a fix works

1. `qcqp1500-1c` solves within 300 s at the default, and `qcqp1500-1nc`
   stays at or near 128 iterations.
2. `discs` and `cresc4` keep their verdicts on the first attempt (no
   second-opinion rescue, `2nd=-` in the sweep).
3. `scripts/sweep-fixtures.sh` on both legs, diffing `2nd=` and `mu=`, and
   `benchmarks/grid` (15 of 22 models above the row cap) not worse than
   909 iterations.

### When to revisit

- The inertia-correction or `δ_s` handling changes.
- A user reports a large QCQP stalling at a fixed μ with `inf_pr` near 1e-3
  on a model with fewer than 1000 inequality rows — the budget does not
  reach it, and `perturb_slack_fact` is the knob.

---

## NLP: the slack row budget loses `cresc132` (and slows `laptime`)

*gh#1127. Measured on Apple silicon (14 cores) at #1126's head `6b669294`,
default options except the cap. Recorded 2026-11-01.*

**Scope.** The default exact-Hessian NLP path on models with more than
`perturb_slack_max_rows = 1000` inequality rows, which the gh#1098 budget
reaches. Complementarity models are not affected: the MPCC driver sets the
cap to 0 (#1126).

### What is wrong

Vanderbei `cresc132` (n = 6, m_d = 2654, factor 0.38) no longer solves
within 300 s at the default. With the cap off it solves as v0.14.0 does
(610 iterations including the adaptive re-solve, f = 0.85771). So in the
suites the budget gains `qcqp1500-1nc` (Mittelmann) and loses `cresc132`
(Vanderbei), one each.

The cap-off and cap-1000 iteration tables agree through iteration 21,
which includes the first restoration call. They diverge at 22, the first
iterate where regularization fires. From there the cap-1000 run falls back
into restoration and stays there.

### It is not a rule that is wrong, it is a model that is chaotic in `δ_s`

`cresc132` alone, by cap (factor = cap/2654):

| `perturb_slack_max_rows` | factor | result |
|---|---|---|
| 0 (upstream) | 1 | Solve_Succeeded, 610 it, only after the first attempt fails and the adaptive re-solve rescues it |
| 2500 | 0.94 | Solve_Succeeded, 247 it, 25 s |
| 2000 | 0.75 | **Infeasible_Problem_Detected**, 591 it (false) |
| 1000 (default) | 0.38 | CPU limit, stuck in restoration |
| 500 | 0.19 | **Infeasible_Problem_Detected**, 153 it (false) |

The response is not monotone in the factor. Even upstream's own rule solves
this model only on its second attempt.

`large_scale/laptime` (n = 58 014, m_d = 6000, factor 0.17) shows the same
pattern without a status change. Its iteration counts are 327 at cap 0, 331
at 500, **756** at 1000 and 284 at 2000. Cap 1000 lands in a different,
slightly lower local minimum, f = 65.29277 (the others reach 65.2947, as does
Ipopt): re-solved 2026-11-02 at `f1787277` it is a genuine KKT point (scaled
constraint violation 1.1e-10, dual infeasibility 7.1e-11), not a loose stop.
At cap 1000 it took 310 s of wall-clock time. That is past 300 s, but the
sweep's limit is on CPU time (`max_cpu_time`), so the run still ended
`SolveSucceeded`; a harness with a 300 s wall-clock limit would lose it.
The CPU limit is no wider a margin: the 0.15.0 sweep finished it at 756
iterations, while a standalone re-solve of the same binary under
`max_cpu_time=300` ran out at iteration 729 (244 s wall). Expect it to flip
to `Maximum_CpuTime_Exceeded` between otherwise identical runs.

### What the cap does on every model it reaches

All 87 non-LP/QP/MPCC models in the benchmark data with m_d > 1000 were run at
cap 1000 against cap 0 (300 s CPU limit, 5 at a time):

- **Better under the cap (iterations):** `qcqp1500-1nc` (timeout → 128),
  `pglib_case1888_rte` (grid 501 → 183, ECF 212 → 166),
  `pglib_case10000_goc` (polar 131 → 85, grid 99 → 87), the other 2000/500/179
  grid cases, `arki0003` 448 → 335, `qcqp1000-2nc` 289 → 184, `robot_b`
  271 → 180, `robot_c` 222 → 116, `svanberg` (both suites), `nql180`,
  `optmass`, `oet2` 63 → 34, `oet7` 297 → 189, and `NARX_CFy`, which lands in
  a different local minimum.
- **Worse:** `cresc132` (lost), `laptime` (327 → 756), `gausselm`
  (871 → 1311, a different local value −17.848 against −17.835), `robot_a`
  145 → 193, `arki0009` 328 → 332, and `ngone` 40 → 46 (it also reaches a
  lower local value, −0.63597 against −0.63328).
- Everything else is identical, including the 25 that route to a convex
  engine.

### What has been tried

- **Gating the budget on `n` or `m_d/n`** (the issue's suggestion). The
  losers sit at n = 6, 1495 and 58 014 and m_d/n = 442, 1.8 and 0.10.
  The winners sit in between (`qcqp1500-1nc` 1500 and 6.7, the grid cases
  0.4–2.4, `robot_b` 52, `oet2`/`oet7` n ≤ 7 and m_d/n ≈ 140–330). No threshold on
  either separates them. And `cresc132`'s table above shows that no factor
  rule can "keep" it, because neighbouring factors fail in both directions.
- **Other cap values.** `dev-notes/qcqp1500-crawl-slack-perturbation.md`
  measured 500/1000/2000 on the 19 Mittelmann models above the cap: every
  value solves `qcqp1500-1nc`, and 1000 has the lowest iteration sum. Both
  losers here fail or slow down at 1000 and recover at a nearby value, so
  moving the constant would trade them for other models' luck.

### How to tell whether a fix works

1. `cresc132` solves within 300 s at the default, *and* it keeps solving at
   caps either side of the default (to show it isn't another lucky value).
2. `qcqp1500-1nc` still solves, and the 87-model comparison above shows no
   new status losses. The model list and runner are reproducible from the
   description: `n`, `m`, `m_d` read from each `.nl` header, m_d > 1000.
3. The fixture sweep cannot see this (no fixture has more than 1100 rows,
   and only `manyineq_concave_qp` is above the cap), so use the Vanderbei
   and Mittelmann suites.

### Workaround

`perturb_slack_max_rows=0` restores upstream's rule for one model.

---

## Active-set QP: the per-pass prune cap is a tuned number (QSCFXM3, QSCSD6 slowdowns)

*Was gh#1081. Introduced by #1078 (gh#1076 item 3). Recorded 2026-10-27.*

**Scope.** Only `solver_selection=qp-active-set`. The default `auto` sends
convex QPs to the interior-point engine (`cvx-qp`), which never runs this code.

### What is wrong

`pounce-qp`'s Schur loop prunes an over-full active set to an independent
subset before it factors it. At a degenerate vertex the row the prune drops
is often a genuine blocker. The next step moves into it, it re-enters at
`α = 0`, and the prune drops it again. Left alone, that cycle runs to
`max_iter`. The fix in #1078 lets the counting prune run at most
`COUNT_PRUNE_CAP = 32` times per pass (`crates/pounce-qp/src/solver.rs`,
`schur_reset_rank_repaired`). The cap is measured, not derived, and the
outcome is not monotone in it:

- QGFRDXPN fails below 24.
- QFORPLAN solves at 16, 20, 32 and 40. It cycles to `MaxIter` at 24, 28,
  48, 64 and with no cap.

Both models are pinned by
`crates/pounce-cli/tests/issue_1076_qgfrdxpn_active_set_prune_cap.rs`, so a
change that pushes either across the window fails CI rather than shipping.

### How big

Small. The cap cost two Maros-Meszaros problems more than 2× in time; both
still solve. Measured on arm64, serial, base `ab59784a` vs #1078:

| problem | base | #1078 | ratio |
|---|---|---|---|
| QSCFXM3 | 229 s | 494 s | 2.16× |
| QSCSD6 | 19 s | 47 s | 2.41× |

QSHELL (1.45×) and QGROW22 (1.41×) also slowed. That comes from #1078's
phase-1 row-feasibility check, not the cap. Across the 39 problems both
versions solve, total wall time fell from 4087 s to 3415 s, and QGFRDXPN went
from failing to solving. The trade is a net gain.

**Wall-time exposure.** The convex arm has no deadline unless
`max_wall_time` is passed. On a 2.8 GHz Xeon cloud container (x86_64, idle,
default options) QSCFXM3 took **1119 s** (`SolveSucceeded`, 18 092
iterations). It would fail any `max_wall_time` below that. That container
runs about 3× slower than the arm64 machine above (QGFRDXPN: 26 s vs 6–9 s),
so read the number as "much slower hardware tips it over", not as an x86
penalty.

### Why each problem is slow

- **QSCFXM3:** its first phase-1 pass reaches the cap, stands down, and then
  runs unpruned for 21 310 working-set changes. Only the third attempt
  solves it; on base the second attempt did.
- **QSCSD6:** its first pass enters an exact prune cycle. The cycle guard
  ends it, and the solve then pays for the recovery path. Under the old rule
  that pass finished directly.

### Already tried, did not work

From #1078: per-slot limits, objective-moved conditions, ending the pass on
the first repeat, and caps from 8 to 64. QFORPLAN also fails at caps 24, 28,
48 and 64.

**Re-entry precedence** (2026-10-27, x86_64, iteration counts). When the
counting prune runs, slots it already dropped in this pass get
`PRECEDENCE_WEIGHT` in `independent_active_subset_sparse_weighted`, so the
probe drops a different member of the dependent set:

| problem | cap 32 (shipped) | old rule, no cap | re-entry, no cap | re-entry + cap 32 |
|---|---|---|---|---|
| QGFRDXPN | optimal, 3553 | — | optimal, 4151 | optimal, 4151 |
| QFORPLAN | optimal, 12465 | `MaxIter`, 12963 | optimal, 13445 | optimal, 13016 |
| QSCSD6 | optimal, 5874 | — | **`MaxWallTime` at 1800 s**, 7573 | optimal, 5874 |
| QSCFXM3 | optimal, 18092 | — | not run | optimal, 17605 |

Without the cap, the rule solves both pinned models, which no rule tried in
#1078 did. It loses QSCSD6: the cycle stops repeating exactly but does not
end. With the cap, it changes almost nothing. So protecting a re-entering
row is part of the answer but not all of it, and QSCSD6 is the model a
cap-free rule has to answer for.

### What a fix would look like

The cap stands in for a missing anti-cycling rule. The textbook candidates:

- a lexicographic or Bland-style choice of entering and dropped row;
- a stronger EXPAND (the loop already carries GMSW `expand_tol`);
- a small perturbation of the bounds at the vertex.

Any of these should end cycling whether it repeats exactly or not, which is
what QSCSD6 needs.

**Suggested first step:** trace QSCSD6 under re-entry precedence with no cap
(`POUNCE_QP_TRACE=1`) to see what the uncapped run spends its iterations on.

**Acceptance:**

1. QGFRDXPN, QFORPLAN, QSCSD6 and QSCFXM3 all solve with no cap.
   `benchmarks/qp/generate_nl.py <NAME>` builds them; run each with
   `solver_selection=qp-active-set`.
2. Then the full 138-problem `benchmarks/qp` run under
   `solver_selection=qp-active-set`, compared against the current tree for
   status and wall time.
3. Then `scripts/sweep-fixtures.sh` (both legs). This is a trajectory change
   to a core loop, and every line that moves needs an explanation.

### Also unguarded

The stationarity-prune cycle guard added in #1078 (fix 1) has unit tests for
its decision logic. No test fails if its call is removed from the loop,
because no small QP could be built that drives a real pass into that cycle.

### When to revisit

- The active-set QP becomes a default route for some class of model.
- The gh#1076 pins start failing after a FERAL, probe-pivoting or phase-1
  change.
- A user reports QSCFXM3-like timeouts under `qp-active-set`.

---

## Convex LP/QP: Farkas certificates are a tolerance at the far-feasible boundary

*Recorded 2026-11-01 with the fix that introduced `FARKAS_MARGIN_MIN`
(`crates/pounce-convex/src/ipm.rs`, `FarkasMargin`). Not tracked: no rule
can close it (see "What is wrong").*

**Scope.** The ratio Farkas test in `detect_infeasibility_with` (every
HSDE and direct-driver iteration of the convex QP/QCQP/conic arm) and the
free-column residual waiver in `active_set::certifies_primal_infeasible`.
The exact box proof (`farkas_box_proof`) and presolve verdicts are not
affected.

### What is wrong

An approximate ray `(y, z)` with `q = Aᵀy + Gᵀz ≠ 0` and
`β = bᵀy + hᵀz < 0` proves only that no feasible `x` has
`‖x‖₁ < |β|/‖q‖∞`. With `s = (|b|ᵀ|y| + |h|ᵀ|z|)/‖|A|ᵀ|y| + |G|ᵀ|z|‖∞`,
the data's natural scale, that radius equals `(ρ_h/ρ_g)·s`. The test
accepts at `ρ_h/ρ_g ≥ 1e4`. Two consequences follow, and no finite
threshold removes either:

1. **False certificates beyond `1e4·s`.** Take `g·(x0 + x1) ≤ −1` and
   `−g·(x0 + (1−ε)·x1) ≤ 0`, with no bounds. It is feasible at
   `x1 ≤ −1/(gε)`, its ray `z = (1, 1)` has radius `2/ε`, and it is
   certified infeasible by both engines when `ε ≤ 2e-4` and the absolute
   tests also pass (`gε ≤ 1e-10`). The unit test
   `margin_threshold_borderline` pins this boundary at `ε = 1e-4`
   (accepted) and `ε = 4e-4` (rejected).
2. **Lost certificates below `1e4·s`.** Integer `A` with an exact null
   vector, `Σh = −δ`, data near `1e6` and `δ = 1e-6` (a relative margin
   near `1e-13`): the IPM certifies 20 of 39 and active-set 18 of 39.
   The rest end `dual_infeasible`, `numerical_failure`,
   `iteration_limit`, `optimal_inaccurate` or, on active-set only,
   `optimal` (3 of 39). The feasible twins (`Σh = +δ`) show the same
   distribution, and the old test certified 14 of those 39 as infeasible,
   so at that margin the model's verdict lies below its data's rounding.

### Measurements (default options, before → after)

| corpus | before | after |
|---|---|---|
| netlib-infeas (29) | 29 infeasible | 29, same iteration counts |
| MIPLIB 2017 Infeas, nnz ≤ 150k (49) | 32 infeasible, 17 unbounded | same; `lr1dr04vc05v17a-t360_if` 36 → 37 iterations (first ray radius 2.0e3) |
| `benchmarks/qp` (138) | 137 optimal, BOYD2 timeout | identical, iteration counts included |
| fixture sweep (134 × 2 legs) | — | only `farkas_far_feasible_no_box` moves (Infeasible → Diverging) |
| synthetic, feasible (180 + 360) | 15 certified infeasible | 0 |
| synthetic, infeasible (360) | 360 / 360 (both engines) | IPM 341, active-set 339; every loss at `δ = 1e-6`, data `1e6` |

Ratio-path certificates on netlib-infeas have radius `2.4e8` (`pang`)
to `1.5e12`. Every false certificate measured had radius `0.16`–`2.4`.
`1e4` is the log-midpoint of that gap and the order of Clarabel's reduced
`AlmostPrimalInfeasible` tolerance (`5e-5`).

### What has been tried

- **K = 10…1e8.** Every `K ≥ 10` removes all 15 measured false
  certificates. On the infeasible family the IPM keeps 360/355/341/341/305/292
  at `K = 10, 1e2, 1e3, 1e4, 1e6, 1e8`, and netlib is unchanged up to
  `1e8`.
- **A componentwise `ρ_g = maxⱼ |qⱼ|/magⱼ`.** Rejected genuine netlib
  certificates (`ex72a`, `ex73a`, `ω ≈ 0.59`) because some columns have
  tiny magnitudes. The max-norm version is used instead.

### How to tell whether a fix works

A fix would need a status between certified and not, like Clarabel's
`AlmostPrimalInfeasible`, or an exact (rational or interval) check of the
ray. It works if `farkas_far_feasible_no_box.nl` stays uncertified, the
`two_row` borderline in `ipm.rs` with `ε = 1e-4` stops being certified,
and the netlib/MIPLIB counts above hold.

## Convex LP: a near-constant infeasible row needs equilibration (gh#1143)

`ε·x₀ ≤ −1` beside `x₀ ≥ 0` is infeasible for every `ε > 0`. Two paths no
longer reach a verdict on it at the smallest `ε`. Before gh#1143 they reported
it, wrongly, as unbounded (or rightly by accident, below); they now end
`NumericalFailure`:

| path | ε | before gh#1143 | now |
|---|---|---|---|
| `solve_qp_ipm`, default | 1e-14 | `DualInfeasible` (wrong) | `PrimalInfeasible`, 24 it |
| `solve_qp_ipm`, `equilibrate = false`, HSDE | 1e-14 | `DualInfeasible` (wrong) | `NumericalFailure`, 159 it |
| `solve_qp_ipm`, `equilibrate = false`, direct | 1e-14 … 1e-10 | `PrimalInfeasible`, 1 it (by accident) | `NumericalFailure`, ~100 it |
| `solve_socp_ipm` (`solver_selection=socp`), default, orthant or orthant + SOC | 1e-14 | `DualInfeasible` (wrong) | `NumericalFailure`, 155–159 it |
| `solve_socp_ipm`, same | 1e-10 | `DualInfeasible` (wrong) | `PrimalInfeasible`, 13 it |

What reaches the verdict through `solve_qp_ipm` is its gh#293 equilibrated
retry after a failed solve. The conic entry has no such retry, and Ruiz
equilibration is only sound per row on orthant rows, so adding one there is a
separate change to that arm's trajectory, not part of this fix.

Before gh#1143 the direct unequilibrated driver answered `PrimalInfeasible`
at iteration **1** — not by finding a Farkas ray, but by accepting the
non-diverging iterate `x ≈ (8.2, 0.06)` as a recession direction (row 0's
`(Gd)₀ = 8e-14` passed the absolute `1e-7·‖d‖` slack), issuing a false
`DualInfeasible`, and having the objective-free twin relabel it. The answer
was right by accident; the fix (a per-row recession slack, `min(1, ‖Gᵢ‖∞)`)
removes the false certificate and with it the accident.

### How to tell whether a fix works

In `crates/pounce-convex/tests/issue1143_tiny_row_false_unbounded.rs`,
`conic_entry_near_constant_infeasible_row_is_not_unbounded` can assert
`PrimalInfeasible` at `ε = 1e-14` too (it asserts only "not
`DualInfeasible`" there today), `near_constant_infeasible_row_is_primal_infeasible`
passes with `equilibrate: false` added to its option sets, and both
`*_satisfied_row_stays_unbounded` tests keep `DualInfeasible`.

---

## Python routing: a large constant term hides an LP/QP from `minimize` (gh#1169)

*Measured on `fix/1169-fd-route-noise`, `python/pounce/_route.py`, the test
QP `½xᵀdiag(1,2,3,4)x + q·x + off` on `[-1,1]⁴`, no `jac`. Recorded
2026-11-15.*

### What is wrong

With no `jac`, `minimize(solver_selection="auto")` classifies the objective
from values of `fun`. Rounding happens inside the user's `fun`, before the
router sees the value: `0.5·xᵀQx + q·x + 1e12` comes back rounded to about
`eps·1e12 ≈ 2e-4`, and every difference of `fun` inherits that error at any
step or stencil. Once the objective's variation across the probe region is a
few thousand units of that rounding, its shape cannot be read.

| offset | values' spread / `eps·|f|` | routes to |
|---|---|---|
| 1e10 | 5.2e5 | `qp-ipm` |
| 3e11 | 1.7e4 | `qp-ipm` |
| 1e12 | 5.2e3 | NLP |
| 1e13 | 5.2e2 | NLP |

The NLP fallback is limited by the same rounding: from `x0 = 1` at 1e12 it
ends 3.5 above the optimum (~16 ulp of `|f|`).

A second face of the same limit goes the other way. Curvature below the
rounding across the probe region reads as **linear**: `1e-7·|x|²` at offset
1e7 routes to `lp-ipm`, and a `1e-3`-scaled copy of the test QP does from
3e11. The LP gate allows for the differenced gradient's round-off (gh#1169,
commit 3), and its value check `f(a)+f(p)−2f(mid)` cannot see curvature that
small either.

The decline is not silent: when `fun`'s spread over the probes is within
`_HIDDEN_ULPS = 1e5` units of its rounding, `auto` warns and a forced
`lp-ipm` / `qp-ipm` / `qp-active-set` / `socp` error says so, both naming the
workaround.

### Workaround

Pass `jac` (the router then never differences `fun`), or subtract the
constant from `fun`: it changes neither the minimizer nor the multipliers.

### Already considered, not done

Fitting `P` from differences with steps the width of the box, and letting
the Hessian-constancy gate allow for rounding as the LP gate now does. At
1e12 the box-width second difference is ~1 against ~1e-4 of rounding, so the
test QP would route again, but `qp-ipm` would then solve a model whose `P`
is accurate to only `~eps·|f|/(h²·|P|)` — about 1e-4 relative here, and worse
as the offset grows — while the result reads as a clean convex solve. The
limit moves; it does not go away, and at 1e16 (`eps·|f| ≈ 1`) nothing is
left to fit.

### How to tell whether a fix works

`python/tests/test_issue_1169_fd_route_noise.py`: extend the QP and LP
offset sweeps past 1e12 and add the `1e-7·|x|²` / 1e7 case to the
small-curvature test as "must not be an LP". A fix must also keep
`test_small_curvature_at_large_offset_is_not_read_as_an_lp` and every
"not routed" test green, and report how accurate the routed `P` is, not only
that it routed.

---

## Convex HSDE: no early exit when it stalls at `DYN_REG_MAX` (gh#1219 §4.2)

*Measured on `claude/inspiring-edison-i30ua2` (`main` at `22ae155e`), by
instrumenting `crates/pounce-convex/src/hsde.rs` and running every
`pounce-convex` test binary in release. Recorded 2026-11-21.*

**The proposal.** gh#1219 §4.2 asked HSDE to bail out early, returning its
best iterate, when the primal residual stays above `DYN_REG_RES_TOL` with the
dynamic regularization already at `DYN_REG_MAX` and `μ` stops falling
(`μ_k ≥ 0.85 μ_{k−1}`) for several consecutive iterations, instead of running
corrupted, high-retry Newton steps to `max_iter`.

### Why it was not done

The signature is not specific to a hopeless solve. Counting, per solve, the
longest run of iterations that met all three conditions:

| test binary | longest run | iterations | final status | n |
|---|---|---|---|---|
| lib (`sos::tests::motzkin_uncertified_fallback_is_not_disguised_as_optimal`) | 16 | 600 | `IterationLimit` | 121 |
| `conic_hsde_vs_direct` | 5 | 25 | `Optimal` | 8 |
| `issue1034_unbounded_singular_qp_status` | 4 | 23 | `DualInfeasible` | 5 |
| `convex_cone_sensitivity` | 4 | 200 | `IterationLimit` | 4 |
| `convex_cone_sensitivity` | 3 | 101 | `Optimal` | 7 |
| `convex_cone_sensitivity` | 3 | 33 | `Optimal` | 7 |
| lib | 3 | 25 | `Optimal` | 383 |

A bail-out after 3–5 such iterations would have cut short four solves that go
on to `Optimal` and one verified unboundedness certificate. The single solve
it would shorten (the Motzkin SOS program, which is uncertifiable by design)
is a test that the uncertified fallback is *not* reported as optimal, and an
early exit would hand its iterate to the end-of-solve stash and
re-classification that test guards. In stock POUNCE the SQP seed that gh#1219
hit this from is also already capped at `IPM_SEED_MAX_ITER = 50`; the
reporter's long stalls came from their own full-space convexification.

### How to tell whether a fix works

A stall rule must stay silent on every row above that does not end
`IterationLimit` — rerun the instrumented count, not just the suite, since a
truncated solve that still lands within the loose tolerance band passes the
tests — and it must leave the Motzkin test's status alone. Then the convex
arm needs the fixture sweep and `benchmarks/qp` (CLAUDE.md), since every row
here is small.

---

## LP crossover: NETLIB PILOT spends the full bridge budget on a vertex it keeps declining (gh#1244)

*Measured at the gh#1244 fix against its parent `0c9dacc`, NETLIB `pilot`
(generated with `benchmarks/lp/generate_nl.py --netlib-only`),
`solver_selection=lp-ipm qp_crossover=yes`, release build, Linux x86_64.
Recorded 2026-11-22.*

### What is wrong

The revised simplex reaches a vertex (30 835 pivots) that the never-regress
gate declines, and the active-set bridge then spends all
`CROSSOVER_MAX_ITER` = 1000 working-set changes from the interior iterate
without a rescue, so the interior point is returned. Wall time with crossover
is 106 s against 4 s without; 63 s of the 106 s is the simplex, the other
~42 s is the bridge.

### Why it got slower

Before the gh#1244 fix the bridge inherited the convex IPM's static pivoting
(gh#1193), so its first factorization failed inertia control and it stopped
after 0 changes: 63 s. Turning static pivoting off on the bridge's backends
restored vertices on agg, grow15, grow22, nesm and the issue's LP (24–105 s
down to 0.06–0.7 s each, since the failing bridge used to burn its whole
budget on them) and moved nothing else among 98 LPs (89 NETLIB, the CLI's 8
`lp_*` fixtures, the issue's LP), except PILOT, which now runs its bridge to
the cap. Total sweep time went from 536 s to 395 s. So the +42 s on PILOT is
the cap working as designed (it is the 0.17.0 behaviour), not a new path.

### How to tell whether a fix works

`pilot` either accepts a vertex (simplex or bridge) or keeps the interior
point within a few seconds of the simplex's own 63 s, and the five rescues
above stay `vertex accepted`. A smaller cap or an early bail-out from an
l1-elastic phase 1 that makes no progress are the obvious candidates; neither
was tried.

## Convex presolve: dropping a degenerate declared bound slows the direct driver on `scaled_feasible_a` (gh#1246)

*Measured on `claude/busy-hypatia-jb79o6` after merging `main` at
`913be167` (gh#1243). Recorded 2026-11-23.*

**What moves.** The implied-bound pass (`relax_implied_bounds`, gh#1246
finding 2) drops one declared side of `scaled_feasible_a`: `ub = 5.0e5` on
column 1. Row 0 (`1e8·x₁ + … ≤ 5.3e8`, the rest of its activity at `−5e13`)
implies exactly that value. The proof holds to `1.4e-8`. The bound and the row
coincide, though, so the vertex is degenerate, and the direct driver's
trajectory is sensitive to which of the two constraints it carries:

| route | `main` | this branch |
|---|---|---|
| default (HSDE), convex arm alone (`solver_selection=qp-ipm`) | `Optimal` at 200, exactly the cap | **iteration limit at 200** (KKT `1.85e-13` relative, uncertified) |
| default, `qp_kkt_row_elim=no` | `Optimal` at 200 | `Optimal` at 179 |
| default `auto` route, as reported | `cvx-qp`, 200 | **`nlp`, 24 / 23** (gh#535 fallback; NLP answer `6.1e-17` against convex `3.0e-13`) |
| `qp_hsde=no` (direct) | 24 | **98** |
| `qp_hsde=no`, `qp_kkt_row_elim=no` | 24 | 24 |
| `qp_hsde=no`, `qp_presolve=no` | 24 | 24 |

Every reported answer is at the optimum `0`. On the default route the convex
arm now stops at the cap without certifying, and the fallback's NLP answer
reports. That answer is the more accurate of the two, but the total cost is
the convex attempt plus the NLP solve, and the sweep's engine column moves
`cvx-qp` → `nlp` on both legs. On `main` this model already certified with
no iterations to spare (gh#689's `the_default_route_reaches_the_same_optimum`
notes it needs about 3 596 as declared). Three things identify the trigger:
- It needs the declared pass. With only derived sides dropped, the direct
  driver takes 24 iterations.
- It needs gh#1243's bound-row folding, which is on by default; with folding
  off the count is 24 again.
- `scaled_feasible_b` does not move (29 iterations).

**Why it is not fixed.** No guard separates this case from the ones the pass
exists for:
- In x-units the implying row's scale is about the bound's own size
  (`5e13 / 1e8 = 5e5`), so a row-scale guard would not exclude it.
- Requiring a declared side to be *strictly* implied would also exclude the
  canonical case, `x ≤ 1` under `Σx = 1`, where the two likewise coincide.

So one deliberately pathological model costs more:
- the opt-in direct driver takes 98 iterations instead of 24;
- the default convex arm gives up at the cap and hands the model to the NLP
  arm, where `main` certified at the cap.

The reported answer is the optimum either way.

**How to tell whether a fix works.** `issue_689_direct_driver_scaled_feasible.rs`
asserts the direct driver's 60-iteration budget with `qp_presolve=no`. To
check a fix, run `scaled_feasible_a` at `qp_hsde=no` with presolve on, and
expect about 24 iterations. Then check that `solver_selection=qp-ipm` at
defaults certifies again (`Optimal`, at most 200 iterations). The sweep should
then show `scaled_feasible_a` back on `cvx-qp` on both legs.
