# Research note — future-work roadmap: restoration SOTA & KNITRO parity

**Status: research note / roadmap.** This is the umbrella note above
the three design notes already in this directory
(`composite-step-byrd-omojokun.md`,
`penalty-ipm-infeasibility-detection.md`,
`interior-cg-matrix-free.md`). It frames *why* those exist, surveys
the rest of the landscape, and proposes an order of work. It is the research half of the research → plan → implement
workflow — written for review, not yet committing any code.

The primary sources are ingested in the project knowledge base
(`.crucible/`); citation keys appear in §7.

## 1. Two goals, one ladder

The user framed two levels of ambition:

- **Goal A — conservative.** Make sure pounce ships *state-of-the-art
  restoration strategies*. Restoration is the acknowledged soft spot
  of every filter-line-search interior-point method: when the line
  search fails, pounce abandons the objective and runs a nested
  min-‖c‖₁ solve, which can burn iterations or leave the solver
  feasible-but-stuck.
- **Goal B — ambitious.** Aim for *KNITRO parity* and tackle the five
  problem classes where the IPOPT-faithful design has documented blind
  spots.

The central observation of this note: **A is a strict subset of B.**
Two of the five SOTA restoration strategies (composite-step,
penalty-IPM) are *also* two of the five KNITRO-parity moves. Doing
Goal A well is the first two-fifths of Goal B. §4 makes the mapping
explicit; the rest of the note can be read as one ranked backlog.

**Where pounce stands today.** pounce is a faithful port of IPOPT's
interior-point filter line-search algorithm (citep wachter2006). In
KNITRO terms, pounce ≈ KNITRO's **Interior/Direct** algorithm and
nothing else. "KNITRO parity" therefore means, concretely, growing two
more algorithms next to it: an **Interior/CG**-style matrix-free path
and an **active-set/SQP** path. That is the honest scope — see §3.1.

## 2. Goal A — state-of-the-art restoration strategies

### 2.1 The five-rung ladder

Restoration strategies, ordered by how far each departs from the
single-line-search IPOPT spine. Rungs #4 and #5 have their own design
notes; the numbering here is the canonical one those notes refer to.

| # | Strategy                                                    | Departs from spine?     | pounce status                                  |
|---|-------------------------------------------------------------|-------------------------|------------------------------------------------|
| 1 | Classic feasibility-restoration phase (nested min-‖c‖₁ IPM) | no                      | **have** — full port                           |
| 2 | Step-level refinements: SOC, watchdog, filter               | no                      | **have**                                       |
| 3 | Inexact / funnel restoration with convergence theory        | partial                 | gap (low priority)                             |
| 4 | Composite-step trust region (Byrd–Omojokun)                 | yes — new globalization | design note; not implemented                   |
| 5 | Penalty-IPM + rapid infeasibility detection                 | yes — new KKT system    | detection **shipped**; penalty-IPM design note |

### 2.2 #1 — Classic feasibility-restoration phase — HAVE

IPOPT's restoration (citep wachter2006): when filter backtracking
drives α below α_min and every step is still rejected, abandon the
objective and solve an auxiliary NLP — minimize the ℓ₁ constraint
violation plus a proximity term ζ/2‖D_R(x − x_R)‖² — as a nested
interior-point solve, exiting as soon as the iterate is both more
feasible and filter-acceptable.

pounce ports this in full: the `pounce-restoration` crate (~6,700
lines) implements all three tiers — **soft restoration** (the
lightweight first resort, wired into `line_search/backtracking.rs`),
**MinC1Norm** (`min_c_1nrm.rs`, `resto_nlp.rs` — the full nested
feasibility NLP with its own augmented-system solver), and
**restoration-of-restoration** (`resto_resto.rs`, the fallback when
the restoration NLP itself stalls). This is the most battle-tested
restoration design in any open-source solver, and pounce has it.

### 2.3 #2 — Step-level refinements — HAVE

The refinements that *reduce how often restoration fires* without
touching the spine:

- **Second-order correction** — `compute_soc_step` in
  `kkt/pd_search_dir_calc.rs` recovers steps the filter would
  otherwise reject from linearization error.
- **Watchdog / non-monotone steps** — wired through `alg_builder.rs`,
  `upstream_options.rs`, `application.rs`.
- **Filter acceptance** itself — `line_search/filter_acceptor.rs`.

Nothing to do here; pounce is at parity with IPOPT.

### 2.4 #3 — Inexact / funnel restoration — GAP, low priority

The research-grade rung: replace the heuristic "restore, then resume"
with a framework that carries a *global convergence guarantee* and a
controlled feasibility/optimality trade-off — **inexact restoration**
(Martínez–Pilotta; Birgin–Martínez) or a **funnel** that caps allowed
infeasibility and shrinks it monotonically (Gould–Toint).

pounce's restoration is the classic heuristic. It works on the large
majority of problems, and the gap here is theoretical robustness on
adversarial cases, not a common practical failure. **Recommendation:
lowest priority of the five.** It is listed for completeness and
should not be scheduled until #4 and #5 are evaluated — at which point
a funnel may fall out naturally as the acceptance mechanism for the
composite step.

### 2.5 #4 — Composite-step trust region (Byrd–Omojokun)

Full design: `composite-step-byrd-omojokun.md`.

The Byrd–Omojokun composite step (citep byrd2000, citep byrd1999)
splits each step into a **normal step** that reduces linearized
infeasibility and a **tangential step** in the null space of the
Jacobian that reduces the barrier model, both inside one trust region.
The payoff for restoration: **there is no separate restoration phase
at all** — the normal step *is* the feasibility move, taken every
iteration. A problem that today triggers `invoke_restoration` would
instead just take a normal-dominated composite step.

Cost: BO is natively a *trust-region* method, and pounce has no
trust-region radius anywhere. It is a new globalization spine, opt-in
alongside the filter line search. The design note phases it: Phase 1
(normal step only, fed into the existing line search) is a ~1–2 week
low-risk win; Phases 2–3 (tangential Steihaug-CG, full TR
globalization) are a 4–8 week commitment.

### 2.6 #5 — Penalty-IPM + rapid infeasibility detection

Full design: `penalty-ipm-infeasibility-detection.md`.

Two coupled ideas (citep byrd2010, citep leyffer2003):

1. **Rapid infeasibility detection** — a cheap per-iteration test
   (‖Jᵀc‖ scaled, held over a streak of iterations) that recognizes
   convergence to a stationary point of the infeasibility with ‖c‖
   bounded away from zero, and exits with `LocalInfeasibility` instead
   of thrashing restoration to `max_iter`.
2. **Penalty-IPM** — fold the constraints into the objective via an
   exact penalty `f(x) + ν‖c(x)‖` with a steering rule for ν, so
   infeasibility is always being minimized and a dedicated restoration
   sub-NLP is never needed.

**Status update — the detection half has shipped.** The main-loop
test now exists: `ConvergenceStatus::LocallyInfeasible`
(`conv_check/trait.rs:34`) plus `infeas_stationarity_tol` /
`infeas_max_streak` in `conv_check/opt_error.rs`. The
`penalty-ipm-infeasibility-detection.md` note still labels §3 as "the
proposed first deliverable" and is now **stale on that point** — it
should be updated to mark detection done (see §6). The penalty-IPM
half — a genuine change to the KKT system — remains future work.

### 2.7 Restoration scorecard

| Rung | Have | Gap | Effort to close |
|---|---|---|---|
| #1 classic restoration | ✅ full 3-tier port | — | — |
| #2 SOC / watchdog / filter | ✅ | — | — |
| #3 inexact / funnel | ✗ | theoretical robustness | research-grade; defer |
| #4 composite-step | ✗ | no TR spine | Phase 1 ~1–2 wk; full 4–8 wk |
| #5 penalty-IPM | ◑ detection done | penalty KKT system | multi-week |

**Goal A verdict.** pounce already *is* state-of-the-art for an
IPOPT-faithful solver — it has the complete classic restoration plus
all step-level refinements, and now main-loop infeasibility
detection. "State of the art" in the *absolute* sense means rungs #4
and #5, and both are genuine departures from the IPOPT spine. They are
worth doing, but as the deliberate, opt-in extensions the two design
notes describe — not as tuning fixes.

## 3. Goal B — KNITRO parity: the five challenge areas

### 3.1 What "KNITRO parity" actually means

KNITRO is not one algorithm. It bundles (citep byrd2006, citep
waltz2006): **Interior/Direct** (interior point, direct factorization
— what pounce is today), **Interior/CG** (interior point with
projected conjugate gradient, matrix-free, for large/PDE problems),
and an **Active-Set/SLQP** algorithm — plus crossover between them,
multi-start, and MINLP. Parity with the *whole* package is a
multi-year program and is not the proposal here.

The tractable reading of "parity": close the five documented classes
where the pure Interior/Direct design fails. Each maps to one of
KNITRO's other capabilities. The five — call them **C1–C5** to keep
them distinct from the restoration rungs #1–#5:

### 3.2 C1 — Warm-started sequences of related problems — Tier 3

**Failure mode.** Interior-point methods warm-start badly: the barrier
pushes iterates to the interior, so a near-optimal point from a
previous solve sits near the boundary and cannot be exploited —
sometimes *slower* than a cold start. This is the documented reason
active-set SQP (SNOPT, filterSQP) is preferred for **MPC**
(re-solving a similar NLP every control step), **MINLP
branch-and-bound** (thousands of related node relaxations), and
**parametric / homotopy continuation**.

**What KNITRO does.** Its Active-Set/SLQP algorithm carries the
working set across solves.

**What pounce would need.** An active-set SQP path: a QP subproblem
solver, working-set machinery, and its own iterate state.
`IpoptData` / `IpoptCalculatedQuantities` are shaped around
primal-dual interior-point variables (slacks, barrier μ), not a
working set. This is a **new `AlgorithmStrategy` end to end** — Tier 3
(see §3.7), effectively a second solver sharing the model/derivative
and linear-algebra foundation but not the iteration skeleton. Highest
effort, and only worth it if warm-started workloads are a target.

### 3.3 C2 — Certifying infeasibility — Tier 1 (mostly done)

**Failure mode.** IPOPT detects infeasibility only indirectly —
restoration must converge to a nonzero stationary point of the
constraint violation. It is slow and sometimes fails to certify.
Documented: SQP-based detection (citep byrd2010) and one-phase
interior-point methods (citep hinder2018) certify infeasibility — and
KKT points — more reliably; in those benchmarks IPOPT failed
substantially more often at looser tolerances.

**Status.** Largely closed. Main-loop rapid infeasibility detection
has shipped (§2.6). What remains is the penalty-IPM half of rung #5,
which makes detection *structural* rather than a bolt-on test.

**Tier 1** for the part already done; the penalty-IPM completion is
the multi-week item from rung #5.

### 3.4 C3 — Degenerate / rank-deficient Jacobians — Tier 2

**Failure mode.** When LICQ/MFCQ fails, IPOPT leans on inertia
correction / regularization (δ_c·I on the (2,2) block). It works but
slows convergence and can trap restoration.

**What KNITRO does.** The Byrd–Omojokun composite step is documented
as robust under rank-deficient Jacobians — its normal step is a
well-defined least-squares move regardless of rank.

**What pounce would need.** Exactly restoration rung #4. **This is the
identity that unifies the two goals: C3 = #4.** Tier 2 — a new
`SearchDirCalculator` plus a parallel trust-region globalization path.

### 3.5 C4 — MPCCs / MPECs (complementarity constraints) — Tier 1

**Failure mode.** Complementarity constraints violate MFCQ at *every*
feasible point — a structural worst case for standard interior-point
methods.

**What KNITRO does.** Dedicated complementarity-constraint support
(its `KN_set_compcons` MPEC interface).

**What pounce would need.** A Scholtes-style relaxation reformulation
driven as an **outer loop in the modeling layer** — introduce the
relaxation parameter, drive it to zero, re-solve. This reuses the
entire existing solver unchanged, so it is **Tier 1**. Honest caveat:
penalty/interior-point methods on complementarity problems are not a
clean win — there is a documented result that the penalty
interior-point algorithm can converge to *nonstationary* points
(citep leyffer2003). C4 is "better, but still an active research
area," and any implementation must report that limitation.

### 3.6 C5 — Very large-scale / PDE-constrained — Tier 1 + Tier 2

**Failure mode.** IPOPT's direct sparse factorization of the KKT
system is the bottleneck when the matrix is too large to factorize or
has dense rows.

**What KNITRO does.** Interior/CG — projected (truncated) conjugate
gradient, matrix-free, never forms or factorizes the KKT matrix
(citep byrd1999, citep waltz2006).

**What pounce would need.** Two separable pieces:

- **Linear-algebra half — Tier 1.** A new `AugSystemSolver`
  implementation backed by a Krylov method. The trait boundary is
  already there; a quasi-exact iterative backend drops straight in
  next to the FERAL/MA57 direct backends.
- **Inexact-Newton half — Tier 2.** Accepting *inexact* steps
  correctly requires inexact-Newton acceptance tests that reach into
  the line search (the IPOPT `IpInexactAlgorithm` lineage). This is
  the medium part, and it shares the normal/tangential step machinery
  with composite-step #4.

Together these two pieces are the Interior/CG algorithm — full design
in `interior-cg-matrix-free.md`.

### 3.7 Effort tiers

From the earlier architecture review — pounce mirrors IPOPT's
strategy-object design with **14 traits**, one per algorithmic
component. The cost of a new path depends entirely on whether it keeps
the **primal-dual interior-point iteration skeleton**.

| Tier | Meaning                                                     | Items                                                                        |
|------|-------------------------------------------------------------|------------------------------------------------------------------------------|
| 1    | Plugs into an existing trait; high reuse                    | C4 MPCC relaxation; C5 iterative `AugSystemSolver`; C2 detection (done)      |
| 2    | A genuinely new strategy object; reuses model + linalg      | C3 composite-step (`SearchDirCalculator` + TR); C5 inexact-Newton acceptance |
| 3    | A different iteration skeleton; effectively a second solver | C1 active-set SQP                                                            |

**Always reused regardless of tier:** the NLP model + derivative layer
(`.nl` reader, CUTEst FFI), the sparse linear algebra and
factorization backends, scaling, options, output, KKT-error checks.
Even Tier 3 is "new algorithmic core on an existing platform," not
from scratch.

## 4. How the moves connect — and what each unlocks

| Restoration rung (Goal A) | KNITRO challenge (Goal B)      | Shared deliverable                            |
|---------------------------|--------------------------------|-----------------------------------------------|
| #2 SOC / watchdog         | —                              | (already done)                                |
| #5 detection half         | C2 infeasibility certification | shipped                                       |
| #5 penalty-IPM half       | C2 (structural completion)     | penalty KKT system                            |
| #4 composite-step         | C3 degenerate Jacobians        | normal + tangential step, TR spine            |
| —                         | C5 large-scale (linalg)        | iterative `AugSystemSolver`                   |
| —                         | C5 large-scale (inexact)       | inexact-Newton acceptance (shares #4's steps) |
| —                         | C4 MPCCs                       | relaxation outer loop (modeling layer)        |
| #3 funnel                 | —                              | acceptance mechanism for #4                   |
| —                         | C1 warm-starting               | active-set SQP (new solver)                   |

Reading the table top-to-bottom is also roughly the recommended order:
the shared deliverables cluster — composite-step #4 is the single
highest-leverage item because it closes restoration rung #4 *and*
challenge C3 *and* supplies the step machinery C5's inexact half
needs.

### 4.1 What each move unlocks — the prioritization input

The table above says which moves *overlap*. This one says which
*problems* each move makes solvable — the input to deciding order.
The cutest-full sweep now running will eventually replace the example
sets below with concrete per-class failure counts; until then the
examples are from the literature and the curated CUTEst subsets the
design notes already cite.

| Move | Problem class it unlocks | Concrete examples |
|---|---|---|
| composite-step #4 / C3 | degenerate / rank-deficient Jacobian; problems that thrash restoration | CUTEst DECONVBNE, S365, S365MOD, HIMMELBJ, PFIT*, the ACOPR family; over-modeled engineering models with redundant constraints |
| penalty-IPM #5 / C2 | infeasible & near-infeasible problems; restoration-thrash | infeasible CUTEst set; the seawater electrolyte case; over-constrained models |
| MPCC relaxation / C4 | complementarity constraints (MPEC / MPCC) | bilevel optimization, Stackelberg / equilibrium models, contact mechanics, control with friction or valve complementarity |
| Interior/CG / C5 | very large-scale; PDE-constrained; KKT too large or dense-row to factorize | PDE-constrained control / inverse problems; large AC OPF; network / multiperiod; the CUTEst large-n timeout tail |
| active-set / SQP / C1 | warm-started sequences of related NLPs | MPC (re-solve per control step); MINLP branch-and-bound node relaxations; parametric / homotopy continuation |
| funnel #3 | adversarial nonconvex problems (theoretical robustness) | rare in practice; no concrete target set |

**Prioritization read** — ranked by target problems unlocked per unit
effort. (This is the *impact* ranking; §5 turns it into an *execution*
order that additionally front-loads the cheap, independent Tier-1
items.)

1. **composite-step #4 / C3** — unlocks the largest *currently
   failing* class (restoration-thrash) and is the prerequisite for
   both Interior/CG and the penalty-IPM steering. First, unambiguously.
2. **Interior/CG / C5** — unlocks an entire size regime pounce cannot
   reach today; depends on composite-step. Second.
3. **penalty-IPM #5 / C2** — overlaps with already-shipped detection;
   the marginal class it adds is "infeasible problems that still
   thrash restoration." Third.
4. **MPCC / C4** — cheap (Tier 1, a modeling-layer loop) but unlocks a
   class not represented in this repo's workloads. Do it when an MPEC
   workload appears.
5. **active-set / SQP / C1** — largest effort; unlocks warm-starting.
   Gate on whether MPC / MINLP / continuation is a real target.
6. **funnel #3** — no concrete target set; documentation only.

**For this repo specifically.** The two benchmark families in
`benchmarks/` set the priority: the `grid` AC-OPF family scales into
Interior/CG territory and OPF is prone to near-degenerate binding
limits (composite-step); the `electrolyte` family produced the
seawater local-infeasibility case (penalty-IPM / detection). MPCC and
active-set / SQP unlock classes *not* present in the current
benchmarks — lower priority until such workloads are planned.

## 5. Recommended sequencing

**Phase 0 — status hygiene (hours).** Update
`penalty-ipm-infeasibility-detection.md` to mark §3 (rapid
infeasibility detection) as shipped, not proposed. Re-confirm the
CUTEst-full and Mittelmann benchmark numbers as the baseline every
later phase is measured against.

**Phase 1 — Tier-1 wins (weeks).** Highest robustness-per-effort,
no new globalization spine:
- C4 MPCC relaxation outer loop in the modeling layer.
- C5 iterative `AugSystemSolver` backend — `interior-cg-matrix-free.md`
  Phase 1 (the quasi-exact linear-algebra half only).
Both are insertable behind existing traits and individually shippable.

**Phase 2 — composite-step, Phase 1 (1–2 weeks).** The normal-step-only
slice of rung #4 / challenge C3: add the trust-region struct and a
dogleg normal-step solver, fed into the *existing* line search as an
SOC-style feasibility correction. Low risk, measurable on the
restoration-heavy CUTEst subset.

**Phase 3 — composite-step, Phases 2–3 (4–8 weeks).** Tangential
Steihaug-CG and full trust-region globalization, opt-in behind a new
globalization option. This closes C3 properly and produces the step
machinery Phase 5 reuses.

**Phase 4 — penalty-IPM (multi-week).** The penalty KKT system,
completing rung #5 / challenge C2. Shares the normal/feasibility
direction with the composite step, so it follows Phase 3.

**Phase 5 — stretch.** C5 inexact-Newton acceptance (reuses Phase 3's
steps); rung #3 funnel acceptance; and — only if warm-started
workloads are a confirmed target — the C1 active-set SQP path, the one
genuine second-solver commitment.

The default globalization (filter line search) never changes; every
new path is opt-in.

## 6. Open questions for review

- **Scope of "KNITRO parity."** Confirm it is read as "close C1–C5,"
  not "reproduce multi-start / MINLP / crossover." This note assumes
  the former.
- **Is C1 in scope at all?** The active-set SQP path is the only
  Tier-3 item and the only true second solver. It is worth a multi-week
  commitment *only* if warm-started workloads (MPC, MINLP nodes,
  continuation) are a target use case. If they are not, C1 should be
  dropped and the roadmap is entirely Tier 1–2.
- **Default behavior.** Recommendation throughout: never change the
  default. Every new path ships opt-in behind an option; the filter
  line search stays the default globalization.
- **Stale design note.** Should `penalty-ipm-infeasibility-detection.md`
  be edited now to reflect that detection shipped, or left and
  superseded by this roadmap? Recommendation: a one-line status edit
  (Phase 0).
- **#3 funnel.** Keep it on the ladder as a documented rung, or drop
  it as out of scope? Recommendation: keep as documentation, schedule
  nothing until Phase 3 is evaluated.

## 7. References

Primary sources, ingested in `.crucible/` (citation key in brackets):

- Wächter & Biegler, "On the implementation of an interior-point
  filter line-search algorithm for large-scale nonlinear
  programming," *Math. Prog.* 106 (2006). [wachter2006] — pounce's
  reference algorithm; the restoration phase.
- Byrd, Nocedal & Waltz, "Knitro: an integrated package for nonlinear
  optimization," *Large-Scale Nonlinear Optimization* (2006).
  [byrd2006] — the KNITRO algorithm bundle.
- Waltz, Morales, Nocedal & Orban, "An interior algorithm for
  nonlinear optimization that combines line search and trust region
  steps," *Math. Prog.* 107 (2006). [waltz2006] — KNITRO
  Interior/Direct + Interior/CG.
- Byrd, Hribar & Nocedal, "An interior point algorithm for
  large-scale nonlinear programming," *SIAM J. Optim.* 9 (1999).
  [byrd1999] — composite step + projected CG for large scale.
- Byrd, Gilbert & Nocedal, "A trust region method based on interior
  point techniques for nonlinear programming," *Math. Prog.* 89
  (2000). [byrd2000] — the trust-region interior-point foundation.
- Byrd, Curtis & Nocedal, "Infeasibility detection and SQP methods
  for nonlinear optimization," *SIAM J. Optim.* 20 (2010). [byrd2010]
  — reliable infeasibility certification.
- Hinder & Ye, "A one-phase interior point method for nonconvex
  optimization," arXiv:1801.03072 (2018). [hinder2018] — no
  restoration phase; documented IPOPT failure counts.
- Leyffer, "The penalty interior-point method fails to converge,"
  arXiv:math/0310357 (2003). [leyffer2003] — the nonstationary-point
  caveat for penalty-IP / MPCCs.
- Omojokun, "Trust region algorithms for optimization with nonlinear
  equality and inequality constraints," PhD thesis, CU Boulder (1989)
  — the composite step's origin; not digitized, not in `.crucible`.

In-tree references: `ref/Ipopt/src/Algorithm/Inexact/` (inexact
composite-step code); `ref/Ipopt/src/Algorithm/IpRestoConvCheck.cpp`
(the `LOCALLY_INFEASIBLE` test).
