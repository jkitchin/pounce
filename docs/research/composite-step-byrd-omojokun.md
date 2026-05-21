# Design note — composite-step (Byrd–Omojokun) globalization

**Status: design / proposed. Not yet implemented.** This note is the
research → plan half of a research → plan → implement workflow; it is
written for review before any code lands.

## 1. What this is

The Byrd–Omojokun (BO) composite step splits the step computation into
two trust-region subproblems:

- a **normal step** `v` that reduces the linearized constraint
  infeasibility `‖cₖ + Jₖ v‖` inside a fraction (`ζ`, ~0.8) of the
  trust region `Δ`;
- a **tangential step** `u` lying (approximately) in the null space of
  `Jₖ` that reduces a quadratic model of the barrier objective /
  Lagrangian, subject to `‖v + u‖ ≤ Δ` and to not undoing the normal
  step's predicted feasibility gain.

The full step is `d = v + u`, accepted by a merit function or filter
with trust-region radius management. This is the globalization used by
KNITRO's interior-point algorithm and by Ipopt's inexact-step variant
(`IpInexactAlgorithm`), and is SOTA strategy #4 from the restoration
survey.

The value for pounce: BO never needs a *separate* feasibility
restoration sub-NLP. The normal step **is** the feasibility move, taken
every iteration in the same trust region as the optimality move. A
problem that currently triggers `invoke_restoration` (line-search
failure → spawn nested min-‖c‖₁ IPM) would instead just take a
normal-dominated composite step.

## 2. The architectural mismatch (read this first)

pounce is a **line-search** interior-point method. BO is natively a
**trust-region** method. They differ in the globalization spine, not
just a tactical knob:

- pounce computes one Newton direction per iteration
  (`PdSearchDirCalc::compute_search_direction`,
  `crates/pounce-algorithm/src/kkt/pd_search_dir_calc.rs:57`) and
  globalizes by backtracking α along it
  (`BacktrackingLineSearch`, `line_search/backtracking.rs`).
- BO computes *two* directions against a *radius* `Δ`, and globalizes
  by growing/shrinking `Δ` on a predicted/actual reduction ratio.

There is no trust-region radius anywhere in pounce today. Adopting BO
is therefore a **new globalization path**, not an edit to the existing
one. This note proposes it as an *opt-in alternative* selected by a new
`LineSearchChoice`-style enum value, leaving the filter line search as
the default and untouched.

## 3. What pounce already has that BO can reuse

| Need | Existing component | Location |
|---|---|---|
| Solve the augmented KKT system | `AugSystemSolver::solve` (4-block saddle point) | `kkt/aug_system_solver.rs:105` |
| Cached refactor / back-substitute | `AugSystemSolver::resolve` | `kkt/aug_system_solver.rs:126` |
| Multi-RHS solve (normal + tangential share a factor) | `AugSystemSolver::multi_solve` | `kkt/aug_system_solver.rs:143` |
| Inertia / degeneracy perturbation | `PerturbationHandler` | `kkt/perturbation_handler.rs:141,232,356` |
| Constraint RHS assembly from an iterate | `compute_soc_step` already builds `c`-RHS blocks | `kkt/pd_search_dir_calc.rs:292` |
| Fraction-to-the-boundary truncation | `aff_step_alpha_primal_max` / `aff_step_alpha_dual_max` | `ipopt_cq.rs:1223,1252` |
| Constraint-violation / barrier-obj quantities | `curr_constraint_violation`, `curr_barrier_obj` | `ipopt_cq.rs` |

The `multi_solve` interface is the key enabler: the normal-step and
tangential-step linear systems share the same KKT coefficient matrix
(only the RHS differs), so one factorization serves both.

## 4. New components required

1. **Trust-region state** — a radius `Δ`, accept/reject ratio
   thresholds (`η₁`, `η₂`), expand/contract factors. Small struct,
   owned by the new driver.
2. **Normal-step subproblem** — minimize `‖cₖ + Jₖ v‖²` s.t.
   `‖v‖ ≤ ζΔ`. Standard choice: a **dogleg** between the Cauchy point
   (steepest-descent on `½‖c+Jv‖²`) and the Newton/Gauss–Newton point.
   The Gauss–Newton point is one `AugSystemSolver::solve` with a
   pure-feasibility RHS (`rhs_x = 0`, `rhs_c = −c`, `rhs_d = −(d−s)`).
3. **Tangential-step subproblem** — minimize the barrier-objective
   quadratic model in the null space of `Jₖ`, `‖v + u‖ ≤ Δ`. In an IPM
   this is the *primal-dual* step with the constraint RHS replaced by
   the residual `Jₖ v` (so `u` does not move the linearized
   constraints), solved by a projected-CG / Steihaug truncated-CG
   iteration that stops at the trust-region boundary or on negative
   curvature.
4. **Composite-step acceptance** — a merit function
   `φ(x) = barrier_obj + ν·‖c‖` or a filter on `(‖c‖, barrier_obj)`,
   plus the predicted-reduction bookkeeping that drives `Δ`. The
   existing `FilterLsAcceptor` filter set could be reused for the
   pair-dominance test; the α-loop machinery cannot.
5. **A new driver** parallel to `BacktrackingLineSearch`, selected via
   a builder enum, wired in `alg_builder.rs::build_inner` and read from
   options in `application.rs`.

## 5. Proposed phasing

- **Phase 1 — normal step only.** Add the trust-region struct and the
  dogleg normal-step solver. Validate by feeding the normal step into
  the *existing* line search as an SOC-style feasibility correction
  when restoration would otherwise fire. Low risk, no new
  globalization spine; measurable on the restoration-heavy CUTEst
  subset (DECONVBNE, S365, HIMMELBJ, ACOPR14/30).
- **Phase 2 — tangential step + Steihaug-CG.** Add the projected
  truncated-CG tangential solve. Still feed `v + u` into the line
  search for acceptance.
- **Phase 3 — full trust-region globalization.** Replace the α-loop
  with `Δ`-management and predicted/actual-reduction acceptance,
  behind the new `LineSearchChoice::CompositeStep` (or a top-level
  `globalization` option). Default stays filter line search.

## 6. Open questions for review

- **Scope.** Phase 1 alone is a meaningful, low-risk win (it removes
  some restoration entries) and is ~1–2 weeks. Phases 2–3 are a much
  larger commitment (4–8 weeks, a second globalization path to
  maintain). Should we commit only to Phase 1 now and re-evaluate?
- **Tangential solve.** Steihaug-CG needs Hessian-vector products
  projected onto the null space of `Jₖ`. pounce assembles `W`
  explicitly for the direct solve; a projected-CG would need either
  (a) repeated `AugSystemSolver::solve` calls as the projection
  operator, or (b) a normal-equations reduction. (a) is simpler and
  reuses `resolve()`'s cached factor; (b) is faster but more code.
- **Interaction with the inexact-Hessian (L-BFGS) path.** BO's
  negative-curvature handling (Steihaug stops at the boundary) is
  cleaner than the perturbation handler's inertia correction — does BO
  *replace* `PerturbForWrongInertia` on this path, or coexist?
- **Default.** Recommendation: never change the default. Ship BO as
  opt-in; the filter line search remains the default globalization.

## 7. References

- Byrd, Gilbert, Nocedal, "A trust region method based on interior
  point techniques for nonlinear programming", *Math. Prog.* 89 (2000).
- Omojokun, "Trust region algorithms for optimization with nonlinear
  equality and inequality constraints", PhD thesis, CU Boulder (1989).
- Curtis, Nocedal, Wächter, "A matrix-free algorithm for equality
  constrained optimization problems with rank-deficient Jacobians",
  *SIAM J. Optim.* 20 (2009) — the inexact-step lineage in Ipopt.
- `ref/Ipopt/src/Algorithm/Inexact/` — Ipopt's inexact composite-step
  code, the closest in-tree reference implementation.
