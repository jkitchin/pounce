# Presolve for LP and convex QP: literature survey and gap analysis against pounce

*Written 2026-11-23 for gh#1246. Source read from the working tree on top of `0c9dacc`, which carries the uncommitted gh#1246 `relax_implied_bounds` change in `presolve.rs`. Line numbers
refer to that revision.*

Status markers used below:

- **[verified]**: read in a primary source this session (paper text, solver
  docs, source header, or README).
- **[recalled]**: stated from the cited paper's known content, but not re-read
  this session because the full text was paywalled or blocked.
- **[unverified]**: an inference or a figure I could not check.

---

## 1. Summary

**Where pounce stands.** `pounce-convex`'s presolve is a correct and carefully
guarded *primal-structural* presolve: empty/singleton rows, fixed and
free-column elimination, parallel rows, activity-based redundancy and forcing,
bound propagation, doubleton aggregation, and (since gh#1246) removal of
implied bound sides. Its postsolve is unusually rigorous: every layer is
validated as a KKT point of the original problem. What it lacks is almost the
whole **dual-side and column-side** half of the standard catalog. The only
dual-argument reduction is `DominatedColumn`. Despite its name, that rule is
sign-based *dual fixing* restricted to columns that appear in **no equality
row and not in `P`** (`presolve.rs:1940-1990`). Every textbook presolve
(Andersen & Andersen 1995; Gondzio 1997; Gould & Toint 2004; HiGHS; PaPILO;
the `presolve` crate) gets a large share of its column reductions from rules
pounce does not have: dominated columns via implied dual bounds, duplicate or
parallel columns, implied-free column substitution, and dependent equalities.
That gap is what OSA-60 exposes: cpo3 removes 94% of its columns and pounce
removes 1.

The 2.2x vs 1.15x figure is quoted from gh#1246. I could not reproduce it from
the cpo3 paper. arXiv:2611.18403 (Cummins, Santamaria, O'Donoghue,
13 Nov 2026) benchmarks cpo3 **with presolve and crossover disabled** [verified],
so the presolve speed-up must come from a separate measurement [unverified].

### Top 5 recommended additions for an IPM (ranked by payoff/effort)

| # | Addition | Why it pays for an IPM | Effort | Dual-postsolve difficulty |
|---|---|---|---|---|
| 1 | **Parallel / duplicate columns** (exact proportional `A`/`G` columns, proportional `P` columns). Fix the dominated one, or merge `x_j + r·x_k`. | It removes whole columns at no fill. It is the rule most likely behind OSA-60 (§4). pounce already has the hashing machinery (`dedup_rows`, `presolve.rs:2705`), so the column version is mostly a transpose. | Low-medium | **Easy** for the fixing variant: the fixed column's `z = c_k − a_kᵀy` is sign-correct by construction. **Moderate** for the merge variant: split the aggregate value back into the two boxes, both columns share `y`, and each `z` comes from its reduced cost. |
| 2 | **Dominated columns via implied dual bounds** (Andersen & Andersen 1995 §; Gould & Toint 2004 §3.3.3; HiGHS `DominatedCol`/`WeaklyDominatedCol`). This also lifts pounce's "no equality rows" restriction. | It fixes columns that sit in equality rows, which is the case pounce's current rule refuses outright. It also detects dual infeasibility cheaply. | Medium. It needs per-row dual bounds `[y_lo, y_hi]` from column singletons, then reduced-cost bounds per column. | **Easy.** The variable is fixed at a bound, and its reduced cost has the proven sign at *every* dual optimum, so `z_j = c_j + (Px)_j − a_jᵀy` needs no re-attribution. For QP the column must be absent from `P`, or `(Px)_j` must be bounded over the box (Gould & Toint). |
| 3 | **Implied-free column singletons and short-equality substitution** with a Markowitz/fill guard (PaPILO `Substitution`/`ColSingleton`/`Stuffing`; HiGHS `FreeColSubstitution`/`Aggregator`; MOSEK eliminator; cpo3 `implied_free_equalities`, `short_equalities`, `singleton_columns`). Today pounce substitutes only *truly* free singletons (`presolve.rs:1689`) and doubletons (`aggregate.rs`). | Each substitution removes a column **and** a row. Free and implied-free columns are the ones an IPM handles worst (no barrier, ill-conditioned KKT). | Medium. It needs implied bounds kept as internal state (item 4) and a fill limit (HiGHS/PaPILO default 10, cpo3 64). | **Easy** for a singleton: the consumed row's multiplier is unique, `y_r = −c_j/a_rj` (pounce already does this). For a general substitution it is **moderate**: a reverse triangular sweep, which pounce already has in `linear_eq_elim::recover_dropped_multipliers`. The implied-free argument gives `z_j = 0`. |
| 4 | **Keep implied bounds as presolve-internal state, not as reduced-problem bounds.** This is an architectural change: PaPILO/HiGHS "implied bounds", Gould & Toint's *loosest* mode, and Fourer & Gay 1994. Today pounce writes tightened bounds into the reduced box (`BoundTightening`, `presolve.rs:1992`) and later strips implied sides (`relax_implied_bounds`, `presolve.rs:1232`). | Gould & Toint measured **loosest** bounds giving a 12% average time gain vs 11% for tightest (LP: 14% vs 10%) with their IPM [verified]. It also removes the column-disjointness serialization that limits propagation to about one edge per round (`presolve.rs:209-227`), and it is a prerequisite for item 3. | Medium. Most of the change is deleting the `BoundTightening` re-attribution path, not adding code. | It **simplifies** postsolve. A bound the solver never saw carries no multiplier, so the `BoundTightening` re-attribution to the source row disappears. Forcing and fixing still record their reason rows. |
| 5 | **Linearly dependent equality rows** (Andersen 1995 "Finding all linearly dependent rows in large-scale LP"; Gondzio 1997; MOSEK `PRESOLVE_LINDEP_USE`; HiGHS `DependentEquations`; cpo3 `equality_dependencies` / `network_equalities`). | Dependent equalities make the normal equations or the KKT system singular. MOSEK's docs say outright that they "can cause numerical problems". pounce's regularized HSDE tolerates them, so the payoff is robustness more than speed [unverified for pounce]. | Medium-high. It needs a sparse LU/QR on the equality block. cpo3 bounds it with a 64-row scratch basis and 128-length rows, which is a cheap entry point. The network (zero-sum incidence) case is nearly free. | **Easy.** A consistent dependent row is dropped with `y = 0`, and the other rows' multipliers stay valid. Inconsistency means the problem is infeasible. |

Cheap extras that are not top 5:

- **Quadratic elimination / unconstrained separable columns.** Gould & Toint
  §3.4 cover a row-less column with diagonal `P`, which is fixed at its clipped
  1-D minimizer. cpo3's `quadratic_elimination` covers a free row-less column
  with `p_jj > 0`, substituted via a Schur complement. pounce drops row-less
  columns only when they are also absent from `P`, because `col_nnz` counts `P`
  entries (`presolve.rs:1606-1640`).
- **Column singletons in inequality rows.** These are slack-like columns, which
  PSLP and cpo3 handle.
- **The `dual_fixing` lock test on `Ax = b`.** It never fires there, which is
  why item 2 is needed.

---

## 2. Rule-by-rule table

pounce status: **have** / **restricted** / **missing**. File references are to
`crates/pounce-convex/src/` unless noted. "IPM relevance" is about what the
rule removes from the KKT system. "Dual" is the difficulty of recovering
`(y, z)` in postsolve.

| Rule | Key references | pounce status | IPM relevance | Dual postsolve |
|---|---|---|---|---|
| Empty rows | BMW 1975; A&A 1995; every solver | **have**. `presolve.rs:1643` (eq), `:1734` (ineq); substitution-emptied rows judged at `EMPTY_ROW_TOL` relative (`:681`, gh#496) | low (cleanup) | trivial (`y=0`) |
| Singleton rows → bound / fixing | BMW 1975; A&A 1995; HiGHS `SingletonRow`; cpo3 `singleton_rows` | **have**. Equality singletons fix (`:1643`); inequality singletons become bounds via tightening and are then dropped as redundant | high: removes rows; with item 4 the bound can be dropped too | easy (`y` from stationarity, or the bound multiplier handed back to the row) |
| Fixed columns (`l = u`) | all | **have**. Handled through the box/`fixed[]`; the planner also fixes equal-bound columns (`linear_eq_plan.rs` shape 1) | medium | easy |
| Empty columns (linear) | A&A 1995; HiGHS `EmptyCol` | **have**. `presolve.rs:2312` (also detects unbounded) | low | easy (`z = c`) |
| Row-less quadratic columns | Gould & Toint 2004 §3.4; cpo3 `quadratic_elimination` | **missing**. `col_nnz` includes `P` entries | medium on QPs with separable terms | easy (`z` from clipped minimizer / 0) |
| Parallel / duplicate rows | Tomlin & Welch 1986; A&A 1995; PaPILO `ParallelRows` (hashing); HiGHS; cpo3 `parallel_rows` | **have**. `presolve.rs:2396-2453`, `dedup_rows` `:2705`. Inequalities: positive multiples only; **opposite-sign pairs are not merged into a ranged row** because `QpProblem` has no ranged rows (`l ≤ a·x ≤ u` is split into two `G` rows in `pounce-cli/src/qp_extract.rs:295-320`) | medium | easy (kept row original; dropped `y=0`) |
| Activity-based redundant row | BMW 1975; A&A 1995; Gondzio 1997 | **have**. `presolve.rs:1750` | high: each `G` row is a slack + barrier pair | trivial |
| Infeasibility from activities | same | **have**. With re-derivation without speculative fixings (`presolve.rs:883`, gh#523) | — | — |
| Forcing rows | BMW 1975; A&A 1995; HiGHS `ForcingRow` | **have**. `presolve.rs:1802`. Restricted to **column-disjoint** forcing rows per round (lifted across rounds); tolerance guard `forcing_pin_is_tight` (`:775`) | high | hard in general (non-unique `y`); pounce emits a valid representative |
| Bound propagation (domain propagation) | BMW 1975; A&A 1995; PaPILO `Propagation`; cpo3 `bound_propagation` | **restricted**. `presolve.rs:1992`. One claim per (column, side) per round, so propagation is serialized (`:209-227`); `MAX_BOX_REFINEMENTS=12` (`:362`); derived bounds are **written into the reduced box** | medium: useful as *reasoning*, harmful as solver bounds (§3) | moderate: global bound recovery re-attributes active tightened bounds to source rows |
| Redundant / implied bound sides (final pass) | Fourer & Gay 1994; Gould & Toint 2004 *loosest* mode; cpo3 `redundant_bounds` | **have** (new, gh#1246). `relax_implied_bounds` `presolve.rs:1232`, coef-ratio guard `RELAX_MIN_COEF_RATIO=1e-3` `:1189` | **high**: each finite side is a barrier term and KKT row; OSA-60 went 232 966 → 465 930 sides before this pass (`:1203-1208`) | easy (identity; bound multipliers re-derived) |
| Dual fixing (locks) | A&A 1995; PaPILO `DualFix`; HiGHS `DualFixing`; cpo3 `dual_fixing` (incl. coupled-quadratic, P row 2–33 nnz) | **restricted**. This is what `DominatedColumn` (`presolve.rs:1940`) actually is: sign-definite `G` coefficients matching `sign(c)`, but only for columns **in no equality row and not in `P`**, and only in the `Catalog::Full` (speculative) pass | high on models whose columns are mostly in `≤` rows | easy (`z` = reduced cost, sign guaranteed) |
| Dominated columns (dual-bound / pairwise) | A&A 1995 (implied dual bounds from column singletons); Gould & Toint §3.3.3 ("weakly forcing dual constraint"); Gamrath, Koch, Martin et al. 2015 (pairwise domination, MIP) [recalled]; HiGHS `DominatedCol`, `WeaklyDominatedCol`; PaPILO `DomCol`; cpo3 `dominated_columns` (aggressive only) | **missing** | high | easy |
| Convex dominance | cpo3 `convex_dominance` (identical support, `[0,∞)` columns differing in one row; lower convex hull over (coef, cost)); I found no earlier primary source (it resembles column-domination for set-partitioning LPs, e.g. IBM RC22252) [unverified provenance] | **missing** | potentially very high on generated "many alternatives" LPs (§4) | easy (columns fixed at 0) |
| Parallel / duplicate columns | A&A 1995; PaPILO `ParallelCols`; HiGHS `ParallelRowsAndCols`; PSLP; cpo3 `parallel_columns` (requires `P[:,k] = r·P[:,j]`) | **missing** | high | easy (fix) / moderate (merge, split back into boxes) |
| Free column singleton | A&A 1995; Gould & Toint §3.3.1; HiGHS `FreeColSubstitution` | **have**. `presolve.rs:1689`; requires the column to be *declared* free, absent from `P` and `G` | high | easy (`y = −c/a`, unique) |
| Implied-free column singleton / implied-free substitution | A&A 1995; Gondzio 1997; MOSEK eliminator; PaPILO `ColSingleton` + `Stuffing`; cpo3 `implied_free_equalities`, `singleton_columns` | **missing**. Only declared-free columns qualify | **high**: OSA-like and network models are full of implied-free columns | easy (`y` unique, `z=0`) |
| Column singleton in an inequality | PSLP ("column singleton in inequality", dual); cpo3 `singleton_columns` | **missing** | medium | easy-moderate |
| Doubleton equality aggregation | A&A 1995; HiGHS `DoubletonEquation`; PaPILO `DoubletonEq` | **have**. `aggregate.rs` via `pounce-presolve` `linear_eq_plan.rs`; pivot on the larger coefficient when ratio > 4 (`linear_eq_plan.rs:~489`), so `|α| ≤ 4`; transfers bounds onto the survivor; `P' = MᵀPM`. **No fill limit on `P`** (cpo3 forbids net Hessian growth) | high | moderate (reverse sweep + bound re-attribution, `aggregate.rs:423`) |
| Short / general equality substitution (fill-limited) | Gould & Toint §3.3.2 (doubleton columns); PaPILO `Substitution` (Markowitz 0.01, maxfill 10); HiGHS `Aggregator` (maxfillin 10); cpo3 `short_equalities` (fill 64, row ≤ 8) | **missing** | medium-high (fewer rows/cols, but fill hurts factorization) | moderate (triangular sweep) |
| Linearly dependent equalities | Andersen 1995 (dependent rows); Gondzio 1997; Mészáros & Gondzio 2001 addendum; MOSEK lindep; HiGHS `DependentEquations`; cpo3 `equality_dependencies`, `network_equalities` | **missing**. `pounce-presolve` has a *structural* LICQ check only (`licq.rs`) | medium (robustness of factorization) | easy (`y=0` on dropped row) |
| Dependent free columns | HiGHS `DependentFreeCols`; Gondzio 1997 (dependent columns) | **missing** | medium (free columns are IPM-hostile) | moderate |
| Sparsification | Chang & McCormick 1993 (hierarchical sparsification) [recalled]; Gondzio 1997 / Gould & Toint §3.2.6 ("make A sparser"); Achterberg et al. 2020; PaPILO `Sparsify` (delayed); HiGHS `Sparsify`; cpo3 `sparsification` | **missing** | medium: fewer nonzeros in `AᵀDA`/KKT, but changes rows the user wrote | moderate (row combination is a linear map on `y`) |
| LP folding (colour refinement / equitable partitions) | Grohe, Kersting, Mladenov & Selman 2014 (ESA; arXiv:1307.5697); cpo3 `lp_folding` (LP only, runs first); extensions to convex (Zhu 2025) and MIP (van der Hulst 2026, arXiv:2603.12136) | **missing** | potentially huge on symmetric LPs, zero elsewhere | moderate (primal/dual lift by the partition map) |
| Bound shift (`x' = x − l`) | cpo3 (README names "bound shifts" but does not define it [verified]) | **missing** (not needed: pounce's IPM handles general boxes) | low | trivial |
| Dense-column handling | Gondzio 1997 | n/a in presolve (KKT-level concern) | medium for normal-equation IPMs; low for quasi-definite KKT | — |
| Dual propagation / dual inference | PaPILO `DualInfer`; PSLP dual propagation; cpo3 `dual_propagation` (aggressive) | **missing** | medium | easy (signs only) |
| Coefficient strengthening, probing, implied integer | Achterberg et al. 2020; PaPILO | n/a (MIP only) | — | — |
| Conic rows | cpo3 `cones`; Clarabel presolve (drops `NonnegativeCone` rows with `b ≥ ~infbound`, verified in `presolver.rs`) | **restricted**. `presolve_conic` (`presolve.rs:1475`) is a single pass, protects non-orthant rows, and does no aggregation | — | — |

NLP side (`crates/pounce-presolve`, lower priority): Andersen-style bound
tightening (Phase 1), redundant-row removal (Phase 2) with general dual
recovery (`dual_recovery.rs`), structural LICQ (Phase 3), linear-equality
elimination (`linear_eq_plan.rs`, shared with the convex aggregation), FBBT,
and BTF/Dulmage–Mendelsohn block solves. None of the dual-side column rules
above exist there either.

---

## 3. Architecture notes

### 3.1 Scheduling and fixpoint

- **pounce** reruns the whole catalog each round, then an aggregation layer.
  Each layer clones the problem it reduces (`presolve.rs:972-1067`), with
  `MAX_ROUNDS = 32` layers, a deadline check between layers (gh#1182), and
  `FixpointExit` reporting. There is no change tracking: every round touches
  every nonzero.
- **PaPILO** [verified, arXiv:2206.10709] sorts presolvers into *fast* (work
  O(changed nnz log)), *medium* (O(nnz log nnz)) and *exhaustive* (O(nnz²))
  classes. A class advances only when the previous one finds "too few"
  reductions, measured against `abortfac`. Sparsify is *delayed*: it is
  enabled only once the exhaustive class stalls. A transaction model lets
  presolvers run in parallel on one problem copy, with conflicts resolved at
  apply time.
- **cpo3 / `presolve` crate** [verified README] runs four phases:
  1. cleanup to stability;
  2. fast substitutions, repeated while nnz falls by more than 5%;
  3. medium (propagation, coupled dual fixing, parallel rows/cols);
  4. final passes (dependencies, sparsification, redundant bounds, dual
     propagation, network equalities, convex dominance).

  LP folding runs first. Budgets are a soft `time_limit` of 60 s plus a
  per-rule `WorkLimit`. `redundant_bounds` is skipped when the time limit was
  hit.
- **PSLP** (Cederberg & Boyd, arXiv:2604.23951) [verified] implements only
  fast and medium explorers: singleton/doubleton rows, redundant rows, column
  singletons in equalities and inequalities, locks, parallel rows/cols, and
  primal/dual propagation. It omits linear dependence, substitution,
  sparsification and **dominated columns**. It reports about 90% of Gurobi's
  nnz reduction at a fraction of the cost, and tracks dirty rows between
  rounds. Dependence detection is excluded because it needs a factorization.
- **Gould & Toint 2004** [verified] track rows/columns whose nnz changed so
  that the costly heuristics run only on those. pounce has no equivalent.

*Implication for pounce:* adding exhaustive-class rules (dominated columns,
dependent rows, sparsification) under the current "full catalog every round"
loop would multiply cost by the round count. Staging them (PaPILO-style
classes, or cpo3's "final passes") should come with them.

### 3.2 How implied bounds are used vs passed to the solver

This is the most consequential design difference.

- PaPILO and HiGHS keep *implied* column bounds alongside the declared ones.
  They use them for reasoning: implied-free detection, dominated columns,
  forcing. They change the problem's bounds only when doing so is itself the
  reduction. [PaPILO paper: activities are stored and updated per row,
  verified. The implied-bound bookkeeping is recalled from source, not
  re-read.]
- Gould & Toint 2004 made this an option and measured it. *Tightest* mode
  passes all derived bounds, *medium* passes only those known not to be
  redundant (after Fourer & Gay 1994), and *loosest* passes the loosest bounds
  that keep the problem equivalent. Results on 160 LP/QPs with GALAHAD QPB
  (an IPM), average gain over no presolve: tightest 11%, medium 2%, loosest
  12%; on LP: 10%, 9%, 14% [verified, §4 and §7.2]. They warn that tight
  bounds are "possibly more degenerate".
- pounce writes tightened sides into the reduced box. That choice forces the
  `BoundTightening` dual re-attribution, and the column-disjointness rule that
  re-attribution needs is what **serializes propagation** to about three
  tightenings per layer on `bore3d` (`presolve.rs:209-227`). gh#1246's
  `relax_implied_bounds` then removes implied sides at the end, which amounts
  to Gould & Toint's loosest mode bolted on afterwards. Moving to internal
  implied bounds would:
  - unserialize propagation;
  - delete the re-attribution;
  - make implied-free substitution (item 3) and dominated columns (item 2)
    expressible;
  - leave `relax_implied_bounds` handling only *declared* sides.

### 3.3 Tolerances and numerical safety guards

| Guard | pounce | PaPILO | HiGHS | cpo3 / `presolve` |
|---|---|---|---|---|
| Feasibility tol | `BOUND_FEAS_TOL = ACTIVITY_TOL = 1e-9`, absolute (`presolve.rs:656-660`); `EMPTY_ROW_TOL` relative | `numerics.feastol` (template `REAL`; exact rational mode available) | `primal_feasibility_tolerance` (unverified for presolve) | `feasibility = 1e-9` |
| Zero coefficient | exact `0.0` (`ZERO_TOL`) | `epsilon` | — | cancellation reject below `1e-10 ×` larger term |
| Parallel test | `PARALLEL_TOL = 1e-9` (`:2645`), hash then confirm | hashing + tolerance | — | `parallel = 1e-12`; **exact** proportionality required for columns |
| Huge bound in activities | none found: infinite contributions are counted (`:1292`, `:2096`) but finite huge bounds (e.g. 1e15) enter sums directly | `numerics.hugeval` (bounds above it are treated as infinite in activity reasoning; verified in `PresolveOptions.hpp`) | `kHighsInf`-based [unverified] | `huge_bound = 1e7` |
| Pivot / coefficient ratio | doubleton pivot on larger coefficient when ratio > 4 (`linear_eq_plan.rs`); `RELAX_MIN_COEF_RATIO = 1e-3` for bound removal (`:1189`); forcing pin guard `FORCING_PIN_TOL` (`:672`) | `substitution.markowitz_tolerance = 0.01` | — | doubleton ratio in `[1e-7, 1e7]` |
| Fill limit | **none** (aggregation is doubleton-only, so `A` fill is bounded, but `P' = MᵀPM` growth is not limited) | `maxfillinpersubstitution = 10`, `maxshiftperrow = 10` | `presolve_substitution_maxfillin = 10` | `substitution_fill = 64`; **no net Hessian growth** unless `allow_hessian_growth` |
| Propagation gain threshold | count-based budget `MAX_BOX_REFINEMENTS = 12` (deliberately not magnitude-based, gh#527) | `abortfac`-based | — | `max(1e4·feas, 0.01·|old|)` |
| Wrong-infeasibility guard | re-derive without speculative fixings (gh#523) | exact arithmetic option | — | Farkas multipliers returned with `Infeasible` |

pounce's guards are well reasoned where they exist. The two gaps that matter
once substitution and column rules arrive:

1. **No huge-value cutoff in activity reasoning.** PaPILO and cpo3 both have
   one. A bound of 1e10 on one column makes every implied bound from that row
   meaningless, and a tightening derived from it can be numerically garbage.
2. **No fill or Markowitz limit**, which general substitution needs.

### 3.4 QP-specific concerns (coupling through `P`)

- In QP the dual constraint for column `j` is `c_j + (Px)_j − a_jᵀy − z_j = 0`.
  Every dual-side argument (dual fixing, dominated columns, implied-free) has
  to bound `(Px)_j` over the box or require `P[:,j] = 0` (Gould & Toint §3.3,
  verified). pounce takes the second route everywhere (`presolve.rs:79-101`),
  which is sound but excludes every column with curvature. cpo3's
  coupled-quadratic `dual_fixing` takes the first route, bounding the
  derivative with `P` rows of 2–33 nnz.
- Parallel columns in QP need `P[:,k] = r·P[:,j]` as well (cpo3, verified).
- Substitution changes `P` to `TᵀPT`. That preserves PSD, which
  `aggregate.rs` already argues, but it can densify `P`. cpo3 forbids net
  Hessian growth by default, and pounce should adopt a growth limit before
  generalizing beyond doubletons.
- LP folding does not apply to QP in cpo3. Zhu 2025 (MIT PRIMES) extends
  colour refinement to smooth convex objectives [not read].
- Gould & Toint report QPs as "as amenable to presolving as linear ones". They
  measured a 13% average gain on QPs in tightest mode, but 0% (medium) and 8%
  (loosest) [verified].

### 3.5 Other solver notes

- **HiGHS** rule enum (`HConst.h`, verified): EmptyRow, SingletonRow,
  RedundantRow, EmptyCol, FixedCol, DominatedCol, ForcingRow, ForcingCol,
  FreeColSubstitution, DoubletonEquation, DependentEquations,
  DependentFreeCols, Aggregator, ParallelRowsAndCols, Sparsify, Probing,
  Enumeration, DualFixing, ZeroCostSingleton, ColStuffing, DualFixProbing,
  InitialSweep, FourierMotzkin, WeaklyDominatedCol. Each can be switched off
  via the `presolve_rule_off` bitmask.
  - Galabova's thesis (Edinburgh 2023, hdl 1842/39725) covers the
    primal/dual postsolve for these.
  - HiGHS's QP presolve coverage is not documented on the pages I read
    [unverified].
- **MOSEK** [verified docs]: the removal list is redundant constraints, fixed
  variables, linear dependencies, and (implied) free variables. The
  *eliminator* is singled out as the step that "can occasionally make a
  problem much harder", and the eliminator and lindep are named as the two
  most time-consuming steps.
- **Gurobi** [verified, parameter group only]: `Aggregate`, `AggFill`,
  `PreDepRow`, `PreDual`, `PreSparsify`, `DualReductions`. Their semantics
  were not read. The presolve paper for the MIP side is Achterberg et al.
  2020.
- **Clarabel**: its presolve only removes nonnegative-cone rows with infinite
  `b` (verified in source). No equivalent was found for OSQP, QPALM or PIQP;
  those solvers rely on scaling and regularization [unverified, not found in
  docs].
- **First-order / GPU**:
  - PDLP (Applegate et al., arXiv:2501.07018) lists presolve among its
    enhancements; its isolated effect was not checked.
  - PSLP is integrated in cuPDLPx, cuOpt and HPR-LP.
  - cuPSLP (Cederberg & Boyd, arXiv:2609.16182) is a GPU presolve, 11x and
    42x faster than PSLP on the Mittelmann and GAMS sets at similar reduction.

---

## 4. OSA-60: which rules explain cpo3's 94% column reduction (measured)

*This section was an inference in the first version of this note. It is now a
measurement, and the inference was partly wrong.*

**The model.** OSA-60 was decoded from Netlib's `emps` file with a Python port
of `emps.c`. The decoder was checked on `afiro`, which reproduces −464.7531.
- 232 966 columns and 10 280 rows (10 232 `≤`, 48 `≥`).
- Every column is `x ≥ 0` with a positive cost and six positive entries.
- 10 198 of the `≤` rows are `≤ 1`.
- No two columns share a support, so `parallel_columns` can do almost nothing
  (it accounts for 149 removals).

**Attribution.** A scratch probe (outside the repo, not committed) linked
the `presolve` 0.1.0 crate and switched rules off one at a time:

| setting | columns / rows after |
|---|---|
| defaults | 15 030 / 2 641 |
| `-convex_dominance` | 59 291 / 2 641 |
| `-lp_folding` | 57 634 / 10 209 (convex dominance fixes 174 610) |
| `-parallel_columns`, `-dual_fixing`, `-singleton_columns` | essentially unchanged |

So two rules do the work:
- **`convex_dominance`**: three columns with the same support that agree in
  every row but one, compared on a lower hull in (that coefficient, cost).
- **`lp_folding`**: an equitable-partition quotient of the LP, which uses its
  symmetry.

Supports become shared only *after* presolve drops the 71 rows its bounds make
redundant (37 `≥ 0` rows and 34 `≤ 89510.4` rows). Those rows are what made
every column distinct.

**What pounce now does.** `convex_dominance_once` in
`crates/pounce-convex/src/presolve.rs` implements the idea, not the crate's
code.
- It takes OSA-60 from 232 966 to 57 591 columns.
- The solve goes from 71.1 s / 89 iterations to 5.9 s / 37 iterations, at the
  same optimum 4.0440725e6.
- `lp_folding` is the remaining gap to 15 030.

**Dominated columns via implied dual bounds (recommendation #2).** This rule
also fires here: a quick count found 152 065 columns removable by it before any
row is dropped. It overlaps heavily with convex dominance on this model, so it
was not needed for OSA-60.

**The other models from the issue.** The same probe found:
- PDS-06: the crate's lead over pounce comes mainly from
  `implied_free_equalities` (5 614 substitutions), then `dual_fixing` (1 430),
  `parallel_columns` (575) and `singleton_columns` (335).
- KEN-18: the crate ends within 1.5% of pounce's column count.

## References

- A. L. Brearley, G. Mitra, H. P. Williams (1975). Analysis of mathematical
  programming problems prior to applying the simplex algorithm. *Math. Prog.*
  8:54–83. doi:10.1007/BF01580428. https://researchonline.lse.ac.uk/id/eprint/31582
- J. A. Tomlin, J. S. Welch (1986). Finding duplicate rows in a linear
  programming model. *Oper. Res. Lett.* 5(1):7–11 [recalled].
- R. Fourer, D. M. Gay (1994). Experience with a primal presolve algorithm. In
  *Large Scale Optimization*, Kluwer [recalled; cited by Gould & Toint].
- E. D. Andersen, K. D. Andersen (1995). Presolving in linear programming.
  *Math. Prog.* 71:221–245. doi:10.1007/BF01586000
- E. D. Andersen (1995). Finding all linearly dependent rows in large-scale
  linear programming. *Optim. Methods Softw.* 6:219–227 [recalled].
- J. Gondzio (1997). Presolve analysis of linear programs prior to applying an
  interior point method. *INFORMS J. Comput.* 9(1):73–91.
  doi:10.1287/ijoc.9.1.73. Addendum: Cs. Mészáros, J. Gondzio (2001),
  *INFORMS J. Comput.* 13(2):169–170 (corrects Props. 1–2).
  https://pubsonline.informs.org/doi/references/10.1287/ijoc.13.2.169.10519
- Cs. Mészáros, U. H. Suhl (2003). Advanced preprocessing techniques for linear
  and quadratic programming. *OR Spectrum* 25:575–595.
  doi:10.1007/s00291-003-0130-x
- N. Gould, Ph. L. Toint (2004). Preprocessing for quadratic programming.
  *Math. Prog. B* 100:95–132. doi:10.1007/s10107-003-0487-2.
  https://www.numerical.rl.ac.uk/media/people/nick-gould/GoulToin04_mp.pdf
  (read in full)
- G. Gamrath, T. Koch, A. Martin, M. Miltenberger, D. Weninger (2015).
  Progress in presolving for mixed integer programming. *Math. Prog. Comput.*
  7:367–398 [recalled].
- T. Achterberg, R. E. Bixby, Z. Gu, E. Rothberg, D. Weninger (2020). Presolve
  reductions in mixed integer programming. *INFORMS J. Comput.* 32(2):473–506.
  doi:10.1287/ijoc.2018.0857
- A. Gleixner, L. Gottwald, A. Hoen (2023). PaPILO: a parallel presolving
  library for integer and linear optimization with multiprecision support.
  *INFORMS J. Comput.* 35(6):1329–1341. doi:10.1287/ijoc.2022.0171;
  arXiv:2206.10709; https://github.com/scipopt/papilo
- PaPILO options header (`maxfillinpersubstitution`, `markowitz_tolerance`,
  `hugeval`):
  https://rust-digger.code-maven.com/src/scip-sys-0.1.16/headers/papilo/core/PresolveOptions.hpp
- I. Galabova (2023). *Presolve, crash and software engineering for HiGHS.*
  PhD thesis, Univ. of Edinburgh. https://era.ed.ac.uk/items/2b1d8a95-e422-44a4-b141-2fae99f87330
- HiGHS presolve rule enum:
  https://github.com/ERGO-Code/HiGHS/blob/master/highs/lp_data/HConst.h ;
  options: `HighsOptions.h` (`presolve_rule_off`,
  `presolve_substitution_maxfillin = 10`)
- M. Santamaria et al., `presolve` crate: https://github.com/matteosantama/presolve
- M. Cummins, M. Santamaria, B. O'Donoghue (2026). Strongly monotone central
  path equations for quadratic cone programs (cpo3). arXiv:2611.18403
- M. Grohe, K. Kersting, M. Mladenov, E. Selman (2014). Dimension reduction via
  colour refinement. ESA 2014; arXiv:1307.5697
- R. van der Hulst (2026). arXiv:2603.12136 (colour refinement for MIP) [not
  read beyond abstract snippet]
- D. Cederberg, S. Boyd (2026). Presolving for GPU-accelerated first-order LP
  solvers (PSLP). arXiv:2604.23951
- D. Cederberg, S. Boyd (2026). GPU-accelerated presolving for linear
  programming (cuPSLP). arXiv:2609.16182
- D. Applegate et al. (2025). PDLP: a practical first-order method for
  large-scale linear programming. arXiv:2501.07018
- MOSEK presolve docs: https://docs.mosek.com/latest/capi/presolver.html
- Gurobi presolve parameter group:
  https://docs.gurobi.com/projects/optimizer/en/13.0/concepts/parameters/groups.html
- Clarabel settings: https://clarabel.org/stable/api_settings/ ; presolver
  source: `Clarabel.rs/src/solver/implementations/default/presolver.rs`
- R. Schwan et al. (2023). PIQP. arXiv:2304.00290
- OSA-60 dimensions: https://sparse.tamu.edu/LPnetlib/lp_osa_60 ;
  https://www.netlib.org/lp/data/kennington/
- IBM RC22252 (duplicate columns in crew-scheduling LPs):
  https://dominoweb.draco.res.ibm.com/reports/RC22252.pdf
