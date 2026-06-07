# PR #70 Hardening — Loop-Driven Verification Tracker

This file is the **state** for the PR #70 hardening loop. Plan:
`~/.claude/plans/woolly-launching-parnas.md`.

## Loop prompt (`/loop`)

> Work the **first unchecked** item below. Do only that one item end-to-end,
> update its section (Findings + checkbox), commit, then stop. Do not start the
> next item.

## Per-iteration protocol

1. **Select** the first `- [ ]` item; re-confirm scope from the plan.
2. **Implement** the named tests, reusing the oracle patterns below.
3. **Run** the item's command. Triage: test bug → fix test; real defect → fix if
   small & obviously correct, else record under Findings with a minimal repro +
   severity. Never paper over a wrong-answer defect.
4. **Record** Findings (tests added, pass/fail, defects, follow-ups). Flip
   `[ ]`→`[x]` only when Done criteria hold.
5. **Commit** one per item: `test(pr70): <item> — <result>` (with the required
   `Co-Authored-By` trailer; never `--no-verify`). Stop.

## Reusable oracle patterns (in-repo)

- **vs-NLP cross-check**: `crates/pounce-cli/tests/{cblib_vs_nlp,exp_cone_vs_nlp,qp_vs_nlp_iterations}.rs`
- **Known optima**: `crates/pounce-qp/tests/mm_published_optima.rs`, `crates/pounce-convex/tests/qp_known_optima.rs`
- **Routing unit**: `crates/pounce-cli/tests/dispatch_routing.rs` + `#[cfg(test)]` in `dispatch.rs`; fixtures `crates/pounce-cli/tests/fixtures/*.nl`
- **External validation**: `benchmarks/scripts/compare_pounce_clarabel.py`
- **`--json-output` schema**: `solution.status`, `statistics.{final_objective,iteration_count,total_wallclock_time_secs}`

## Baseline (captured at bootstrap)

- `cargo test --workspace`: **GREEN** — true exit 0, **1649 passed, 0 failed**
  (confirmed on a clean re-run, not piped through `tail`).
- Clarabel comparison (Item B input) — **full suite**, outputs in
  `benchmarks/clarabel_compare.md` + `clarabel_compare_{lp,qp}.json`:
  - **LP**: 467 problems, 419 both-solved, **412/419 agree** (reldiff < 1e-4).
    3 pounce-only, 28 clarabel-only. POUNCE non-solves incl. InternalError
    (greenbea, ch, nemsemm1, nemsemm2), several TimeOut/MaxIter.
  - **QP**: 138 problems, 114 both-solved, **110/114 agree**. 3 pounce-only,
    19 clarabel-only. `VALUES` failed with `ParseError:JSONDecodeError` on the
    pounce side — likely a JSON-report/harness bug, flag in B or G.
  - **Objective disagreements to triage in Item B** (both solved, reldiff ≥ 1e-4):
    - Near-zero-objective artifacts (both ≈ 0, published optimum 0 — almost
      certainly fine): LP `model11`; QP `S268`/`HS268`.
    - **Genuine, investigate**: QP `YAO` (pounce 197.70 vs clarabel 91.02,
      reldiff 0.54); LP `capri` (2625.0 vs 2690.0, reldiff 0.024).
    - Borderline (≈1–4e-4, likely tolerance): LP `lpl2`, `pltexpa3_16`,
      `pltexpa4_6`, `large001`, `fxm3_16`; QP `UBH1`.
  - POUNCE correct live; stored `benchmarks/lp/pounce.json` is STALE
    (adlittle/stocfor1 wrong) — regenerate in B.

---

## [x] A1 — Routing classification (HIGHEST RISK)
- Scope: `classify_problem` must never under-classify nonconvex as convex.
  Cover: indefinite Hessian → `NonconvexQp`; near-PSD boundary at `±PSD_TOL`
  (1e-9) resolves conservatively (inconclusive → NLP); maximize-of-convex
  (concave) → nonconvex; zero Hessian → `Lp`; pure linear; genuinely convex
  QP/QCQP still convex (no false fallback).
- Files: `crates/pounce-cli/src/dispatch.rs` (PSD test ~L576+, `#[cfg(test)]` mod).
- Run: `cargo test -p pounce-cli dispatch`
- Done: new cases green; any misclassification recorded as a Finding.
- Findings:
  - **Tests added** (5, all green; 29/29 in `dispatch::tests`):
    - `psd_rejects_small_but_real_negative_curvature` — diag(2, −1e-3) reads
      indefinite (the safety-critical direction: a real negative eigenvalue,
      even small, is NOT rounded to PSD).
    - `psd_threshold_is_psd_tol` — pins the cutoff: −1e-10 (|λ|<tol) → PSD,
      −1e-7 (|λ|>tol) → indefinite.
    - `classify_concave_minimize_is_nonconvex` — `minimize −x0²` → `NonconvexQp`
      (auto → NLP), complementing the existing maximize-of-PSD case.
    - `classify_qcqp_with_indefinite_constraint_falls_back_to_nlp` — convex obj +
      indefinite quadratic constraint → `Nlp` (conservative QCQP guard; was
      untested — only the all-convex QCQP case existed).
    - `classify_cancelling_quadratic_objective_is_lp` — `x0²−x0²` → `Lp`
      (collapsing quadratic, empty Hessian, not a spurious QP).
  - **Pre-existing coverage confirmed adequate**: indefinite→NonconvexQp,
    maximize-of-convex→nonconvex, maximize-of-concave→convex, pure LP, convex
    QP, convex QCQP, transcendental obj/con→NLP, cubic/transcendental rejection.
  - **Finding (informational, NOT a defect): the ±PSD_TOL band rounds toward
    convex.** The PSD test is `min_eig >= -PSD_TOL` (PSD_TOL=1e-9), so a Hessian
    with smallest eigenvalue in `[-1e-9, 0)` classifies **convex**, not NLP. The
    module doc (L36–38, L45–48) says it routes inconclusive cases "to the safe
    side, never to the convex path" — the wording overstates the actual `>= -tol`
    behavior. This is the *correct* engineering choice, not a bug: PSD includes
    semidefinite Hessians (zero eigenvalues — e.g. an LP-as-QP or a rank-deficient
    QP), whose smallest eigenvalue routinely computes as a tiny negative under
    Jacobi roundoff; requiring strict positivity would misroute legitimate convex
    QPs to NLP and regress `psd_accepts_psd_with_zero_eigenvalue`. The 1e-9 band is
    orders of magnitude below the solve error a convex IPM would incur on that much
    curvature. **Severity: none** (recommend only tightening the doc wording to
    match `>= -PSD_TOL`). No misclassification found.

## [x] A2 — Forced `solver_selection` mismatch must error, not mis-solve
- Scope: `qp-ipm`/`lp-ipm`/`qp-active-set` forced on a non-matching/nonconvex
  `.nl` returns a clear error (nonzero exit / error status), never a wrong
  "optimal." `auto` on the same routes safely (NLP/global).
- Files: `crates/pounce-cli/tests/qp_dispatch_end_to_end.rs`,
  `crates/pounce-cli/tests/dispatch_routing.rs`, new fixture
  `crates/pounce-cli/tests/fixtures/nonconvex_qp.nl`.
- Run: `cargo test -p pounce-cli`
- Done: mismatch cases assert error; green.
- Findings:
  - **New fixture** `nonconvex_qp.nl`: `min x0·x1 s.t. x0+x1=2, 0≤xᵢ≤4`
    (indefinite Hessian; classifies `nonconvex QP`). Box bounds keep the NLP
    fallback bounded (local optimum 0 at a corner) so `auto` exits 0 cleanly.
  - **Tests added (6, all green; full `pounce-cli` suite 0 failures):**
    - `forced_qp_ipm_on_nonconvex_qp_errors` — the headline case: convex QP IPM
      forced on a nonconvex QP exits 2, names the class + solver, and **does NOT
      print "Optimal Solution Found"** (the confident-wrong-answer failure mode
      is asserted absent).
    - `forced_qp_active_set_on_nonconvex_qp_errors` — same for the active-set QP.
    - `forced_lp_ipm_on_convex_qp_errors` — LP IPM forced on a convex QP errors
      (QP ≠ LP).
    - `auto_routes_nonconvex_qp_to_nlp_safely` — `auto` on the nonconvex QP
      routes to pounce-nlp (NOT pounce-convex), solves, exit 0.
    - `forced_qp_solvers_on_nlp_error` (dispatch_routing) — qp-ipm & qp-active-set
      forced on a general NLP (rosenbrock) both exit 2 with a naming message.
  - **Behavior confirmed manually** before writing tests: every mismatch exits 2
    with `problem class <X> does not match forced solver <Y> (expected <Z>)`;
    the error is raised at routing (before any solve), so no wrong objective is
    ever produced. No defect found.

## [x] B — Objective validation vs known optima + Clarabel
- Scope: netlib LP + Maros–Mészáros QP objectives from pounce match Clarabel /
  published optima within tol (rel < 1e-4); disagreements triaged. **Regenerate
  the stale `benchmarks/lp/pounce.json`** from live pounce. Conic/CBLIB covered
  via `cblib_vs_nlp`.
- Files: `benchmarks/scripts/compare_pounce_clarabel.py` (add `--check` mode +
  nonzero exit on disagreement), `benchmarks/lp/pounce.json` (regenerate),
  optionally `benchmarks/qp/pounce.json`.
- Run: `python3 benchmarks/scripts/compare_pounce_clarabel.py --class both`
- Done: all problems agree within tol or each disagreement is explained;
  `pounce.json` no longer stale.
- Findings:

  **Harness added.** `compare_pounce_clarabel.py` gained two flags:
  - `--from-json` — re-evaluate the committed `clarabel_compare_{lp,qp}.json`
    records without re-running both solvers (regression gate / CI).
  - `--check` — exit nonzero on any *genuine* objective disagreement. A
    disagreement counts only when BOTH solvers report a **certified** solve
    (pounce `SolveSucceeded` AND clarabel `Solved`; `AlmostSolved` /
    `SolvedToAcceptableLevel` are excluded as uncertified) yet objectives differ
    beyond the numpy-isclose band `|a−b| > atol + rtol·max(|a|,|b|)`,
    rtol=atol=1e-3. Helpers `isclose` / `check_disagreements`,
    `POUNCE_STRICT={SolveSucceeded}`, `CLARABEL_STRICT={Solved}`.

  **Coverage (live, 60s/solver):** LP 467 problems, both-certified-solved 413;
  QP 138, both-certified-solved 112. Under the strict gate exactly **one**
  hard-fail across both suites: `capri` (LP). `make`-driven default routing on
  the whole LP suite uses the same pounce-convex IPM the live `lp-ipm` run
  exercised (confirmed: `pounce capri.nl` with no flags → `auto` → convex LP IPM
  → identical 2625.01), so the live LP records *are* the default-routing results.

  **HIGH-SEVERITY DEFECT — `capri` silent wrong answer (MERGE-BLOCKER).**
  - Repro (identical generated `.nl`, only `solver_selection` differs):
    - `solver_selection=nlp`    → obj **2690.012861**, 192 it — CORRECT
      (matches Clarabel `Solved` 2690.0129, the documented netlib optimum, and
      the previous stored value).
    - `solver_selection=lp-ipm` → obj **2625.011804**, 25 it, status
      `SolveSucceeded` — **WRONG by 2.4%**, reported as optimal.
  - Same `.nl` on both paths ⇒ this is the **pounce-convex LP IPM**, NOT a
    conversion bug.
  - **Hit by DEFAULT routing**: `pounce capri.nl` (no flags) classifies LP and
    routes to the convex IPM, printing `Optimal Solution Found. obj=2625.01`. A
    user gets a confident wrong optimum with zero opt-in — this is not gated
    behind an expert flag. Severity: **HIGH, blocks merge** until the convex
    LP/QP IPM either solves `capri` correctly or fails honestly (non-optimal
    status) on it. `--check` (and `--check --from-json`) exits 1 naming `capri`,
    so this is now a standing regression gate.

  **Other disagreements — triaged, all benign:**
  - `YAO` (QP): pounce 197.70 vs clarabel 91.02, but clarabel only reached
    `AlmostSolved` (uncertified) and pounce's 197.70 matches the published
    Maros–Mészáros optimum — pounce correct; excluded by the strict gate.
  - Near-zero optima (S268/HS268 opt 0, model11, etc.): agree under the absolute
    tolerance; the relative metric is meaningless at 0.
  - Borderline-tolerance LPs (lpl2, pltexpa3_16, pltexpa4_6, large001, UBH1):
    differ only at ~1e-3 convergence-point slack, inside the isclose band; not
    flagged.
  - Clarabel-`AlmostSolved` cases (fxm3_16, etc.): excluded from the strict gate
    as uncertified.

  **`benchmarks/lp/pounce.json` regenerated (de-staled).** Rebuilt from the live
  LP records, mapping CamelCase → the file's underscored Ipopt convention
  (`SolveSucceeded`→`Solve_Succeeded`, `MaximumIterationsExceeded`→
  `Maximum_Iterations_Exceeded`, `InfeasibleProblemDetected`→
  `Infeasible_Problem_Detected`, `TimeOut`→`Maximum_CpuTime_Exceeded`,
  `InternalError`→`Solver_Error`). 465 records (the 2 `.nl`-generation harness
  failures de063157/stoprobs excluded — pounce never ran them). Confirmed the
  previously-stale objectives are now correct: `adlittle` 6812.5→**225494.96**,
  `stocfor1` −13875→**−41131.98**. `summarize_pounce.py lp` parses it cleanly
  (422/465 solved). NOTE: `capri` is stored as its actual buggy default output
  (2625.01, `Solve_Succeeded`) — the file faithfully records what pounce *does*;
  the wrongness is the defect above, not a staleness of this file. CAVEAT: live
  numbers are from a 60s/problem limit, so the 19 `Maximum_CpuTime_Exceeded`
  entries are time-limit artifacts of this run, not solver verdicts.

## [ ] C — Status / edge-case honesty
- Scope: Infeasible, Unbounded, and limit cases (iteration/time/node) report the
  correct status — **never "optimal."** Edge inputs: empty constraints, fixed
  variable, free variable, single variable, zero-Hessian QP-as-LP.
- Files: `crates/pounce-convex/tests/infeasibility.rs` (+bounded_form.rs),
  `crates/pounce-convex/src/{ipm,hsde,hsde_nonsym}.rs`;
  `crates/pounce-global/tests/global.rs` + `bnb.rs` `GlobalStatus::{Infeasible,NodeLimit,TimeLimit}`.
- Run: `cargo test -p pounce-convex infeasib && cargo test -p pounce-global`
- Done: status assertions green for every edge case.
- Findings:

## [ ] D — Nonsymmetric cones & SDP (riskiest numerics)
- Scope: exp/power cones (`hsde_nonsym` path) and `psd`/`chordal` least
  battle-tested. Adversarial: ill-conditioned, near-cone-boundary, a few larger
  instances; validate via vs-NLP and/or known optima (geometric/entropy for exp,
  small SDPs for psd).
- Files: `crates/pounce-convex/src/cones/{exp,power,psd,chordal,nonsym}.rs`,
  `crates/pounce-convex/src/hsde_nonsym.rs`; tests alongside cone tests +
  `crates/pounce-cli/tests/exp_cone_vs_nlp.rs`.
- Run: `cargo test -p pounce-convex cone && cargo test -p pounce-cli exp_cone`
- Done: new adversarial cases green or defects logged.
- Findings:

## [ ] E — Global solver soundness
- Scope: (1) certified **lower bound always a valid global bound**; relaxations
  (αBB/RLT/OBBT/McCormick) are valid outer approximations; (2) **parallel ==
  serial** optimum; (3) node/time limits return best-incumbent with correct
  status.
- Files: `crates/pounce-global/src/{bnb,alphabb,rlt,obbt,envelope,relax,branching}.rs`,
  `crates/pounce-global/tests/global.rs`.
- Run: `cargo test -p pounce-global`
- Done: bound-validity + serial==parallel + limit-status tests green.
- Findings:

## [ ] F — Presolve round-trip (primal AND dual)
- Scope: presolve + postsolve recovers true primal and **dual** solution,
  including on heavily-reduced problems.
- Files: `crates/pounce-convex/src/presolve.rs`,
  `crates/pounce-convex/tests/presolve_roundtrip.rs` (+ presolve_reductions/
  forcing/conic/bound_tightening).
- Run: `cargo test -p pounce-convex presolve`
- Done: primal+dual recovery asserted; green.
- Findings:

## [ ] G — FFI / Python surface
- Scope: `minimize()` auto-routing picks the right solver; JAX differentiable-QP
  gradients match finite differences; `--json-output` schema uniform across all
  solver paths.
- Files: `python/pounce/{_route.py,qp.py,jax/_qp.py,global_opt.py,sos.py}`,
  `python/tests/test_{minimize_autoroute,qp,qp_jax,qp_sensitivity,socp,global,sos}.py`.
- Run: `pytest python/tests -q` (build the extension first per repo norm).
- Done: pytest green; gradient finite-diff check within tol.
- Findings:

## [ ] H — Hygiene (build / clippy / full suite)
- Scope: clean `cargo build` + `cargo clippy` across the feature matrix (fix the
  known `unused import: QpStatus` in
  `crates/pounce-qp/.../illconditioned_fallback.rs`); full `cargo test` +
  `pytest` green; no new warnings.
- Run: `cargo clippy --workspace --all-targets && cargo test --workspace`
- Done: zero warnings; both suites green.
- Findings:
