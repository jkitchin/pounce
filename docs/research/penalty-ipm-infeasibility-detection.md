# Design note — penalty-IPM with rapid infeasibility detection

**Status: design / proposed. Not yet implemented.** This note is the
research → plan half of a research → plan → implement workflow; it is
written for review before any code lands.

## 1. What this is

SOTA restoration strategy #5 has two coupled ideas:

1. **Penalty-IPM** — fold the constraints into the objective via an
   exact penalty `f(x) + ν‖c(x)‖` and run the interior-point method on
   the penalized problem, with a *steering rule* that drives `ν`. The
   penalty term means infeasibility is always being minimized, so a
   dedicated feasibility-restoration sub-NLP is never needed.
2. **Rapid infeasibility detection** — a cheap per-iteration test that
   recognizes when the iterates are converging to a stationary point
   of the infeasibility measure `‖c(x)‖` with `‖c(x)‖` bounded away
   from zero, and exits immediately with `LocalInfeasibility` instead
   of grinding to `max_iter` or thrashing in restoration.

These are separable. The detection half is small, low-risk, and high
value *on its own*; the penalty-IPM half is a large algorithmic
change. This note recommends shipping detection first.

## 2. What pounce already has

- **A penalty line search.** `PenaltyLsAcceptor`
  (`crates/pounce-algorithm/src/line_search/penalty_acceptor.rs`,
  443 lines) already maintains a penalty parameter `ν` (`nu_init`,
  `nu_inc`, `nu_max`, `update_nu` at line 123) and an Armijo test on
  the penalty merit `M = φ + ν·θ` (line 219). It is selectable via
  `LineSearchChoice::Penalty`. This is *not* a full penalty-IPM — the
  KKT system is still the standard barrier system, only the
  globalization is penalty-based — but the `ν`-steering scaffolding
  exists.
- **`pounce-l1penalty` crate** — a 36-line ℓ₁-penalty helper.
- **Infeasibility detection only inside restoration.** The restoration
  sub-IPM flags `locally_infeasible` when it converges its own KKT but
  the original constraint violation stays above tolerance
  (`crates/pounce-restoration/src/min_c_1nrm.rs:224`), surfaced as
  `RestorationOutcome::LocallyInfeasible`
  (`crates/pounce-algorithm/src/restoration.rs:59`) →
  `SolverReturn::LocalInfeasibility` (`ipopt_alg.rs:1157`).
- **No infeasibility detection in the main loop.** The main
  convergence check `OptErrorConvCheck::check_convergence_with_state`
  (`crates/pounce-algorithm/src/conv_check/opt_error.rs:146`) tests
  only the optimality triplet (dual inf / primal inf / complementarity
  all under tolerance). It has no stationary-point-of-infeasibility
  test.
- **Cycle guardrails that work around the gap.** Because the main loop
  cannot detect infeasibility, `invoke_restoration`
  (`ipopt_alg.rs:921`) carries three hand-tuned cycle detectors
  (static cycle `:984`, slow-progress cycle `:1000`, near-feasible
  re-entry `:1028`) that exist specifically to stop infeasible
  problems from thrashing restoration to `max_iter`. Rapid
  infeasibility detection would make these guardrails largely
  redundant.

## 3. Rapid infeasibility detection — the proposed first deliverable

### 3.1 The test

At a point `x`, the gradient of the squared infeasibility
`½‖c(x)‖²` is `J(x)ᵀ c(x)`. The iterate is approaching an
**infeasible stationary point** when, simultaneously:

- the constraint violation is bounded away from zero:
  `θ = ‖c(x)‖ > κ · constr_viol_tol` (e.g. `κ = 100`);
- the infeasibility is stationary: the scaled gradient
  `‖J(x)ᵀ c(x)‖ / max(1, ‖c(x)‖)` is below a tight tolerance;
- this has held for `N` consecutive iterations (e.g. `N = 5`) — a
  streak counter, to avoid firing on a transient flat spot.

When all three hold, terminate with `SolverReturn::LocalInfeasibility`.

`J(x)ᵀc(x)` is exactly the dominant term of the restoration NLP's
objective gradient and is already computable from quantities pounce
maintains — `curr_jac_c`/`curr_jac_d` and `curr_c`/`curr_d_minus_s` in
`ipopt_cq.rs`. No new linear solve is required; this is a few
inner products per iteration.

### 3.2 Where it goes

- New per-component quantity in `ipopt_cq.rs`:
  `curr_infeasibility_stationarity()` →
  `‖Jᵀc‖ / max(1, ‖c‖)` (one `trans_mult` per Jacobian block, already
  the pattern used by `aff_step_alpha_primal_max`).
- New status variant `ConvergenceStatus::LocallyInfeasible` in
  `conv_check/trait.rs`, alongside the existing `CpuTimeExceeded` /
  `WallTimeExceeded` variants (added in the issue-15 Tier-A wave).
- New fields + streak counter in `OptErrorConvCheck`
  (`conv_check/opt_error.rs`): `infeas_stationarity_tol`,
  `infeas_max_streak`, and the gate logic in
  `check_convergence_with_state`.
- Main-loop mapping in `ipopt_alg.rs` — `LocallyInfeasible` →
  `SolverReturn::LocalInfeasibility`, the same return the restoration
  path already produces.
- Three new options registered in `upstream_options.rs` and read in
  `application.rs` (mirrors the watchdog / soft-resto option wiring).

### 3.3 Why this is worth doing first

- **Small and contained** — ~150 lines, no new linear algebra, no new
  globalization spine. Comparable in size to the soft-restoration
  phase that just landed.
- **Removes guesswork** — replaces three hand-tuned cycle detectors
  (`ipopt_alg.rs:984-1052`, each annotated with the specific CUTEst
  problems it was reverse-engineered against) with a principled test.
  Those detectors fire on `relative_distance ≤ 1e-10` heuristics that
  are fragile; an honest `Jᵀc`-stationarity test is not.
- **Faithful** — Ipopt has exactly this
  (`IpOptErrorConvCheck` / restoration `ConvergenceCheck` →
  `LOCALLY_INFEASIBLE`). pounce currently only has the restoration-side
  half.

## 4. The penalty-IPM half (larger, proposed second)

A true penalty-IPM modifies the **KKT system**, not just the line
search: the barrier problem becomes
`min f(x) + ν‖c(x)‖₁ + barrier`, the constraints enter as penalized
elastic variables, and a steering rule updates `ν` from the
predicted infeasibility reduction of the current step (Curtis–Nocedal–
Wächter 2010). Sketch:

- Reuse the elastic-variable reformulation already present in the
  restoration NLP (`resto_nlp.rs`, the 5-block `n_c / p_c / n_d / p_d`
  layout) but apply it to the *main* problem with a penalty objective.
- Replace the steering: instead of `PenaltyLsAcceptor::update_nu`'s
  merit-based bump, drive `ν` from the ratio of predicted infeasibility
  reduction at the penalty step vs. a pure-feasibility step (this is
  where the composite-step note's normal step would plug in — the two
  strategies share the normal/feasibility direction).
- The detection test from §3 becomes the natural termination signal:
  a penalty-IPM at a stationary point of the infeasibility *is* the
  rapid-infeasibility condition.

This is a multi-week change touching the KKT assembly, the mu update,
and the convergence check. It should not start until the detection
half (§3) is in and the composite-step note's Phase 1 is evaluated —
they share the feasibility direction.

## 5. Open questions for review

- **Default behavior.** Should rapid infeasibility detection be *on*
  by default? Recommendation: yes — it can only convert a `max_iter` /
  restoration-thrash outcome into an honest `LocalInfeasibility`, and
  it is gated behind a 5-iteration streak. But it must be validated
  that no currently-solving problem trips it (run the full CUTEst
  curated suite both ways before flipping the default).
- **Retiring the cycle detectors.** Once §3 lands, do we delete the
  three `invoke_restoration` cycle detectors, or leave them as a
  belt-and-suspenders fallback? Recommendation: leave them for one
  release, measure overlap, then delete.
- **Tolerances.** `infeas_stationarity_tol`, `κ`, `N` need tuning on
  the restoration-heavy subset (S365, S365MOD, PFIT1/4, the ACOPR
  family). This is the same tuning loop as the soft-resto threshold.
- **Penalty-IPM scope.** §4 is genuinely large. Confirm whether it is
  in scope at all, or whether the existing `PenaltyLsAcceptor` plus §3
  detection is considered sufficient coverage of strategy #5.

## 6. References

- Curtis, Nocedal, Wächter, "A penalty-interior-point algorithm for
  nonlinear constrained optimization", report (2010) / the penalty-IPM
  lineage.
- Byrd, Nocedal, Waltz, "Steering exact penalty methods for nonlinear
  programming", *Optim. Methods Softw.* 23 (2008).
- Nocedal, Öztoprak, Waltz, "An interior-point method for nonlinear
  optimization with a quasi-feasibility approach" — infeasibility
  detection in IPM.
- `ref/Ipopt/src/Algorithm/IpRestoConvCheck.cpp` — Ipopt's
  `LOCALLY_INFEASIBLE` test, the in-tree reference for §3.
