---
description: "Adversarially test pounce: solve problems with known solutions across every solver family and report wrong answers (works inside adversary/; pounce repo only)"
---

# Adversary Agent: POUNCE Solver Correctness Testing

You are an adversary agent for **pounce**. Your job is to find optimization
problems with **known solutions** and test whether pounce solves them
correctly, across *every* solver family pounce ships. You must **NEVER modify
pounce source code** (`crates/*/src`, `python/pounce/**`, or any tests). You
operate entirely inside the `adversary/` directory.

Be skeptical of your own formulations: a derivative bug, a wrong cone
encoding, or a non-DCP oracle is far more likely than a genuine solver bug.
Confirm before you accuse.

## Arguments

$ARGUMENTS

Parse the arguments for two optional pieces of information:

1. **Count**: a bare integer (e.g. `3`, `5`) — how many problems to test this
   run. Default `1`.
2. **Family / topic guidance**: any non-numeric text steering selection. It may
   name a *solver family* (see the table below: `nlp`, `lp`, `qp`,
   `qp-active-set`, `socp`, `exp`, `power`, `sdp`, `sos`, `autoroute`, `batch`,
   `diff`, `sensitivity`) or a free-text topic (`degenerate`, `ill-conditioned`,
   `geometric programming`, …).

Examples:
- `/adversary` — 1 problem, autonomous family selection
- `/adversary 5` — 5 problems, balanced across families
- `/adversary socp` — 1 second-order-cone problem
- `/adversary 3 exp geometric programming` — 3 exponential-cone GP problems

When running multiple problems, run the full workflow (steps 1–7) for each
sequentially, picking a **different family each time** to keep balance (unless
topic guidance pins one). Print the summary table (step 9) at the end. The
branch sync (step 0) runs **once at session start** and the publish (step 10)
**once at session end** — not per problem.

## Solver families & their oracles

pounce is a *family* of solvers on one numerical backbone. Each family has a
preferred **correctness oracle** — an independent solver or check that does not
trust pounce. Pick the oracle from this table; never validate pounce against
pounce.

| Family          | pounce entry point                                  | Problem shape                                  | Primary oracle(s)                               |
|-----------------|-----------------------------------------------------|------------------------------------------------|-------------------------------------------------|
| `nlp`           | `pounce.minimize` / `Problem` / CLI on `.nl`        | general nonlinear program                      | **Ipopt** (Pyomo) + `pounce verify` + scipy     |
| `lp`            | `solve_qp(P=None,…)` / `minimize` auto              | linear program                                 | `scipy.optimize.linprog` / cvxpy / closed form  |
| `qp`            | `solve_qp` (convex IPM)                             | convex QP                                       | **cvxpy** / closed-form KKT / scipy             |
| `qp-active-set` | `minimize(options={"solver_selection":"qp-active-set"})` via CLI, or compare paths | convex QP, small | same as `qp`, **and** cross-check vs the IPM path |
| `socp`          | `solve_socp(cones=[("soc",d),…])`                  | second-order cone program                      | **cvxpy** (ECOS/Clarabel)                       |
| `exp`           | `solve_socp(cones=[("exp",3),…])`                  | GP / entropy / logsumexp / logistic            | **cvxpy** (ECOS/SCS)                            |
| `power`         | `solve_socp(cones=[("pow",α),…])`                  | p-norm / general geometric                     | **cvxpy** (SCS/Clarabel)                        |
| `sdp`           | `solve_socp(cones=[("psd",n),…])`                  | small dense semidefinite program               | **cvxpy** (SCS/Clarabel)                        |
| `sos`           | `pounce.sos_minimize`                               | polynomial global minimization (Lasserre)      | **known global optimum** + dense grid/multistart refutation |
| `autoroute`     | `pounce.minimize` (default `solver_selection="auto"`)| any of the above                              | **forced `solver_selection="nlp"`** must agree, AND the route must be the specialized one |
| `batch`         | `solve_qp_batch` / `solve_qp_multi_rhs`             | many QPs at once                               | per-item single `solve_qp`                      |
| `diff`          | `pounce.jax` / `pounce.torch` QP/SOCP layers        | differentiable layer `x*(θ)`                   | **finite-difference** `dx/dθ` + JAX↔Torch parity + `gradcheck`/`gradgradcheck` (float64) |
| `sensitivity`   | `QpSensitivity` / sIPOPT suffixes on `.nl`          | parametric `dx/dp`                             | **finite-difference re-solve**                  |

### Target selection (step 1 uses this)

Two signals, in priority order.

**1. Coverage ranking (primary).** `scripts/coverage-combined.sh` produces
`target/coverage-combined/core.txt`: numerical-core source files ranked by
**uncovered regions**. Uncovered numerical code is where a silently-wrong
answer can hide, so prefer a family whose implementation is poorly covered over
one that is merely under-represented in the log.

Use the existing report if it is fresh (regenerate if older than the last
substantive `crates/` change — the header line prints total coverage; the file
mtime tells you its age). Regenerating takes ~10 min:

```bash
scripts/coverage-combined.sh          # full
scripts/coverage-combined.sh --quick  # skip the slow pytest suite
```

⚠️ It leaves the extension built **with instrumentation** (slower; timing tests
may fail). Always restore afterwards: `cd python && maturin develop --release`.

**2. Log rotation (tie-break).** Read `adversary/log.org`, count tested problems
per family, and prefer the **least-tested** among families the coverage ranking
does not clearly discriminate. Keep all families moving.

#### Mapping uncovered files → the family that exercises them

| uncovered file | drive it with family |
|---|---|
| `pounce-nl/nl_tape.rs`, `nl_reader.rs` | `nlp` **via `.nl`** — CLI / pyomo / GAMS, *not* `pounce.minimize` (the facade uses Python callbacks and never touches the tape) |
| `pounce-convex/crossover.rs`, `simplex.rs` | `lp` |
| `pounce-convex/ipm.rs`, `hsde_nonsym.rs` | `qp`, `socp`, `exp`, `power`, `sdp` |
| `pounce-qp/solver.rs`, `pounce-algorithm/sqp/*` | `qp-active-set` |
| `pounce-algorithm/kkt/*`, `ipopt_alg.rs`, `mu/*` | `nlp` (KKT/barrier internals) |
| `pounce-restoration/*` | `nlp` — infeasible / badly-scaled / hard starts |
| `pounce-presolve/*` | `lp`, `qp` (redundant rows, fixed vars, dup columns) |
| `pounce-sensitivity/*` | `sensitivity`, `diff` |
| `pounce-sens-core/*` | `sensitivity`, `diff` — reached through both arms |
| `pounce-algorithm/batch.rs` | `batch` |
| `pounce-linalg/*` | any — reached through every solve |

#### How to read the report honestly

- **Rust-only coverage lies.** `cargo llvm-cov --workspace` runs only the Rust
  suite, so anything exercised solely through the Python extension or the CLI
  reads as cold. Under it `kkt/pd_full_space_solver.rs` shows 47.9% vs **78.7%**
  combined. Only use `coverage-combined.sh`; never rank from a bare
  `cargo llvm-cov` run.
- **0% is not automatically a gap.** Examples, benches, `*/bin/*`, `debug.rs`
  and the `iter_dump`/`iterate_dump` families are excluded from `core.txt` for
  exactly this reason. If something else reads 0%, find out *why* before
  believing it.
- **Coverage says WHERE to look, never WHETHER it is wrong.** The oracle
  requirement is unchanged: never validate pounce against pounce.
- **Covered ≠ verified, and green ≠ exercised.** A passing test can be vacuous.
  `set_kkt_schur_block()` shipped as a silent no-op for every default user while
  its test passed, because the Schur path falls back transparently and the test
  only compared against the full-space answer. Coverage of
  `kkt/schur_aug_system_solver.rs` — flat 0% — was the only thing that showed
  it. When a feature has a transparent fallback, assert that it *engaged*, not
  just that the answer is right.

### Attack the option space, not only the problem space

Everything above picks a *problem*. A whole class of defect is invisible to
that, because it needs a **non-default option** to surface — and the run that
found gh #508 was the one run that swept an option rather than choosing a new
model. Treat the option grid as a target in its own right, worth roughly one
run in four.

The shape: take a model whose answer you already know (a previous PASS from the
log is ideal — the answer is recorded and the formulation is already
adjudicated), then sweep the options around it and assert that the *verdict*
holds. Three probes, in order of yield:

1. **Kill-switch ablation.** Every POUNCE-only heuristic — one with no upstream
   Ipopt analogue — has an option that disables it: `infeas_stationarity_tol=0`
   for rapid infeasibility detection, `presolve=no`, `acceptable_iter=0`,
   `feral_ordering=…`. Run the model with each switch off and diff the verdict
   against stock. **A model where *disabling* a heuristic improves the verdict
   is a bug candidate, and the heuristic is the suspect.** This is also the
   cheapest possible discriminator once something looks wrong: it isolates the
   mechanism in one run, without an oracle and without reading any code.
   gh #505's mechanism was settled by exactly this and nothing else.

2. **Tolerance monotonicity, feasible direction.** Sweep each tolerance option
   independently over 4–6 orders on a model known feasible, and assert the
   verdict never crosses into the AMPL 200 (infeasible) or 500 (error) bands.
   Tightening may legitimately cost iterations or downgrade
   `Solved` → `Acceptable` → `MaxIter`; it must never manufacture an
   infeasibility. gh #505 is precisely this failure: `constr_viol_tol=1e-6` on a
   solved model produced `local infeasibility` at a point six orders inside
   `acceptable_tol`.

3. **Real driver option sets.** POUNCE's tests run at defaults; users do not.
   Harvest the option sets that real stacks pass unconditionally — Pyomo
   drivers, GAMS `option` blocks, the reporter scripts attached to filed issues
   — and run the corpus under them. gh #505 needed `constr_viol_tol=1e-6`,
   which the reporter's driver set on *every* solve, and which POUNCE's own
   fixtures had never once set on a feasible model.

Two properties make this arm strong. It needs **no oracle** — a verdict that
moves when only an option moved is an internal contradiction in POUNCE's own
semantics, which is the load-bearing evidence in both gh #505 and gh #508. And
it needs **no new formulation**, so the usual "your script is wrong more often
than the solver" risk largely disappears.

Background and the standing invariants this arm defends:
`dev-notes/termination-status-invariants.md`.

## Environment

- **venv:** `source /Users/jkitchin/projects/pounce/.venv-qa/bin/activate`
  (numpy, scipy, sympy, jax, torch present).
- **CLI binary:** `/Users/jkitchin/projects/pounce/target/release/pounce`
  (rebuild with `cargo build --release -p pounce-cli` only if stale; you do
  **not** modify its source).
- **Ipopt binary:** `/opt/homebrew/bin/ipopt` (already installed).
- **Oracles you may need to install into the venv on demand** (these are not
  pounce source, so installing them is allowed):
  - `pip install cvxpy` — the gold-standard convex/conic oracle (LP/QP/SOCP/
    exp/power/SDP). Install it the first time you test any convex family.
  - `pip install pyomo` — registers `SolverFactory('pounce')` (via the
    `pyomo-pounce` plugin) and `SolverFactory('ipopt')`; this is the NLP
    cross-validation path, identical in spirit to ripopt's adversary.
- If an oracle genuinely cannot be installed, fall back to a **known published
  optimum** and `pounce verify`, and say so in the report.

## Workflow

### 0. Sync from the shared branch
The **`adversary-runs` branch is the canonical `adversary/` state** — the daily
cloud routine (gh#395) and local runs both publish there. The local working
tree's `adversary/` is ignored via `.git/info/exclude` (so it never shows as
noise on `main`), which means local and branch can silently diverge. Reconcile
before selecting targets:

```bash
git fetch origin adversary-runs
git show origin/adversary-runs:adversary/log.org > /tmp/adv-branch-log.org
```

Compare the newest `** [YYYY-MM-DD]` entries in `/tmp/adv-branch-log.org` and
`adversary/log.org`:
- Branch has entries the local log lacks, local has none the branch lacks (the
  normal case — cloud runs happened since the last local run): overwrite the
  local `adversary/log.org` with the branch version.
- Both sides have unique entries (a local run that never published): splice the
  missing entries into the branch version, take the per-family **max** of the
  two coverage counts, and use that as the new local log.
Rotation and open-findings checks then run against the reconciled log.

### 1. Pick a target (coverage first, then the log)
Read `target/coverage-combined/core.txt` and `adversary/log.org`, and apply
**Target selection** above: let the coverage ranking choose the family, using
log counts only to break ties. State in the report which signal drove the
choice, and quote the coverage line you acted on (file, uncovered regions, %) —
so a later run can tell whether the number moved.

If no coverage report exists, or it predates the last substantive `crates/`
change, either regenerate it (`scripts/coverage-combined.sh`, ~10 min, and
**restore the extension afterwards**) or fall back to log rotation and say so.

Within the chosen family, pick a problem **class** appropriate to it (e.g. for
`nlp`: unconstrained, bound-, equality-, inequality-constrained,
nonlinear-equations, degenerate, large-sparse). When coverage drove the choice,
prefer a class that actually reaches the cold file — consult the mapping table,
and note that `.nl`-tape gaps require the CLI/pyomo path, not
`pounce.minimize`.

### 2. Select a problem with a KNOWN answer
Find a specific, small instance with a **published or analytically derivable**
optimum. Good sources by family:
- **nlp:** Hock-Schittkowski (HS001–HS120), CUTEst small instances, Nocedal &
  Wright, Biegler, Floudas, classic functions (Himmelblau, Beale, Powell,
  Wood, Rosenbrock-with-constraints), Broyden/Powell-singular nonlinear systems.
- **lp/qp:** textbook LP/QP with closed-form KKT; portfolio/Markowitz;
  least-squares with bounds; Maros-Mészáros QP set (small).
- **socp/exp/power/sdp:** Boyd & Vandenberghe *Convex Optimization* examples
  (robust LS, Chebyshev/`∞`-norm, GP, entropy maximization, max-volume
  ellipsoid, min eigenvalue / Lyapunov SDP), MOSEK modeling cookbook.
- **sos:** known global minima of polynomials (Motzkin-like, Goldstein-Price,
  six-hump camel, constrained polynomial programs from Lasserre/Parrilo).

Requirements: continuous (no integers); small (≤ ~50 vars / ≤ ~50 constraints
for non-SOS; SOS degree small enough to stay under a few seconds); solvable in
**< 10 s**; **not already in `adversary/log.org`**; not a near-duplicate of an
existing `python/tests/` fixture. Record the **exact source** (paper/book page,
equation, or URL) and the known optimum. Use web search if needed.

**Option-sweep runs invert this step.** If the target is the option space (see
"Attack the option space" above), do *not* find a new problem — reuse a model
already logged as PASS, or one infeasible by exact construction, and let the
option grid be the novelty. The known answer is already recorded, so the run
spends its budget on coverage of the option space instead of on re-adjudicating
a formulation.

### 3. Formulate as a runnable cross-check script
Write `adversary/runs/YYYY-MM-DD_<family>_<name>.py`. It must:
1. build the problem in the **pounce** entry point for that family;
2. build the **same** problem in the **oracle**;
3. solve both, compute objective/solution relative error and timings;
4. print a machine-greppable block ending in `VERDICT: …`.

Template (adapt per family — the convex/conic case shown; for `nlp` use
`minimize`/Pyomo, for `sos` use `sos_minimize`):

```python
"""Adversary cross-check: <problem name>
Family: <family>   Class: <class>
Source: <exact citation>
Known optimal: <value>
"""
import time, numpy as np

KNOWN_OPTIMAL = ...          # from the reference, or None if oracle-only

# --- pounce ---
from pounce import solve_qp           # or solve_socp / sos_minimize / minimize
t0 = time.perf_counter()
r = solve_qp(P=..., c=..., A=..., b=..., G=..., h=...)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = r.x, r.obj, r.status

# --- oracle (cvxpy shown; use linprog / Ipopt / known-optimum as per table) ---
import cvxpy as cp
x = cp.Variable(...)
prob = cp.Problem(cp.Minimize(...), [...])     # MUST be DCP / match exactly
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)                 # or ECOS / SCS
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = x.value, prob.value

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
obj_err = rel(obj_pounce, obj_oracle)
x_err   = float(np.linalg.norm(np.asarray(x_pounce) - np.asarray(x_oracle), np.inf))

print("=== pounce ===");  print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print("=== oracle ===");  print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
if KNOWN_OPTIMAL is not None:
    print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status in ("optimal",) or getattr(r, "success", False)) and obj_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
```

**Family-specific formulation notes:**
- **nlp:** prefer the Pyomo path — one model, solved by both `SolverFactory('pounce')`
  and `SolverFactory('ipopt')` (cross-validation + perf comparison in one shot).
  Additionally, write the model to `.nl` and run `pounce verify <nl> <sol>` as a
  solver-independent feasibility/KKT oracle. Hessian/gradient bugs in a
  hand-built `Problem` are the #1 false positive — if using the raw `Problem`
  API, verify derivatives by finite differences first.
- **autoroute:** solve once with default (`auto`) and once with
  `options={"solver_selection":"nlp"}`. They must agree to tolerance
  (routing-transparency contract). Also assert the auto path actually used the
  *specialized* solver (check `result` metadata / `solver` field), else the
  route silently fell back — that is itself a finding (`ROUTING_ERROR` only if
  the *answers* disagree; a conservative fall-through that still gets the right
  answer is "merely slower", **not** a bug — log it, don't file it).
- **socp/exp/power/sdp:** the cone encoding is the trap. Cross-check the cone
  dimensions and the `svec` layout for `psd` (lower triangle, column-major,
  off-diagonals ×√2). If pounce and cvxpy disagree, suspect your encoding
  before the solver.
- **sos:** `sos_minimize` returns a **lower bound** (`lower_bound`) and, when
  exact, recovers minimizers. PASS if the bound matches the known global min to
  tolerance (and, if `is_exact`, the recovered minimizer's objective matches).
  A loose bound at low `order` is **expected** (raise `order`); only a bound
  that **exceeds** the true global minimum (an invalid lower bound) is a
  `SOLVER_BUG`. Refute candidate minimizers with a dense grid / multistart
  scipy search.
- **diff:** check the layer's `dx/dθ` against a central finite difference of a
  re-solve (float64), and check JAX vs Torch agree. Use
  `torch.autograd.gradcheck`/`gradgradcheck`. A gradient that disagrees with
  finite differences beyond tolerance (with the forward solve correct) is a
  `GRADIENT_ERROR`.
- **sensitivity:** compare `QpSensitivity` (or the sIPOPT `sens_sol_state_1`
  suffix from the CLI) against a finite-difference re-solve `(x*(p+δ)−x*(p−δ))/2δ`.

### 4. Run it
```
source /Users/jkitchin/projects/pounce/.venv-qa/bin/activate
python adversary/runs/YYYY-MM-DD_<family>_<name>.py
```
If it errors, fix the **script** (never pounce). If results look wrong, verify
your formulation (derivatives / cone layout / DCP) before suspecting the solver.

### 5. Analyze & classify
Compute relative error vs the known optimum **and** vs the oracle. If they
disagree (rel error > `1e-4`):
1. **Re-verify the formulation** — finite-difference derivatives (nlp/diff),
   cone dimensions (conic), DCP correctness (cvxpy oracle), the reference value.
2. **Try alternative options** — tighter `tol`, larger `max_iter`, a different
   IPM/active-set path, a different cvxpy solver (ECOS↔SCS↔Clarabel).
3. **Run the independent oracle** — for nlp, `pounce verify` and Ipopt; for
   conic, a second cvxpy solver.
4. **Classify the verdict:**
   - `PASS` — pounce matches the oracle/known optimum to tolerance.
   - `FORMULATION_ERROR` — your script was wrong (derivatives, cone, DCP).
   - `REFERENCE_ERROR` — the published optimum was wrong.
   - `TOLERANCE` — within `1e-2` but not `1e-4` (note it; usually not filed).
   - `SOLVER_BUG` — wrong answer **confirmed** by an independent oracle + alt
     options + (for nlp) `pounce verify` rejecting the claimed point.
   - `SOLVER_LIMITATION` — pounce returns non-optimal / hits the iteration
     limit / reports infeasible-or-unbounded on a problem the oracle solves.
   - `PERFORMANCE_REGRESSION` — both solve, but pounce wall time > **3×** the
     oracle's **and** the absolute gap > **0.1 s** (ignore sub-0.1 s gaps).
   - `ROUTING_ERROR` (autoroute only) — auto vs forced-`nlp` answers disagree.
   - `GRADIENT_ERROR` (diff only) — forward solve correct but `dx/dθ` disagrees
     with finite differences / across JAX↔Torch beyond tolerance.

### 6. Write the run report
`adversary/runs/YYYY-MM-DD_<family>_<name>.org`:

```org
#+TITLE: Adversary Run: <Problem Name>
#+DATE: <YYYY-MM-DD>

* Problem
:PROPERTIES:
:FAMILY: <family>
:CLASS: <class>
:SOURCE: <exact citation with page/equation/URL>
:KNOWN_OPTIMAL: <value>
:N_VARIABLES: <count>
:N_CONSTRAINTS: <count or cone layout>
:SELECTED_BY: <coverage | log-rotation | topic-guidance>
:COVERAGE_TARGET: <file, uncovered regions, % — or "-" if not coverage-driven>
:END:

<mathematical formulation in LaTeX>

* Results
:PROPERTIES:
:STATUS: <PASS | FAIL | INCONCLUSIVE>
:END:

| Solver        | Status | Objective | Rel Err | Iters | Wall Time |
|---------------+--------+-----------+---------+-------+-----------|
| pounce        | ...    | ...       | ...     | ...   | ...       |
| <oracle>      | ...    | ...       | ...     | ...   | ...       |

Known optimal: <value>

<narrative>

* Cross-check / Verification
<oracle agreement; for nlp include `pounce verify` exit code and Ipopt output;
for diff include finite-difference vs analytic dx/dθ; for conic include the
second cvxpy solver>

* Verdict
<one of the verdicts above> — <one paragraph explaining the conclusion>

* Coverage effect
<only when coverage drove the target: the file's uncovered-region count before
and after this probe, per step 7. "no defect; raised nl_tape.rs 50.2% → 57.1%"
is a legitimate result. Omit this section entirely for log-driven runs.>
```

### 7. Update the log
Append to the `* Problem Index` of `adversary/log.org` (create it if missing):

```org
** [YYYY-MM-DD] <Problem Name> (<family>/<class>) - <PASS/FAIL>
:PROPERTIES:
:FAMILY: <family>
:SOURCE: <citation>
:KNOWN_OPTIMAL: <value>
:POUNCE_OBJECTIVE: <value>
:POUNCE_STATUS: <status>
:POUNCE_TIME: <seconds>
:ORACLE: <name>
:ORACLE_OBJECTIVE: <value>
:ORACLE_TIME: <seconds>
:VERDICT: <verdict>
:REPORT: [[file:runs/YYYY-MM-DD_<family>_<name>.org]]
:COVERAGE_TARGET: <file:uncovered/regions before, or "-" if log-driven>
:END:
```

Also refresh the `* Family coverage` table's `Tested` count and `Last run` for
the family you exercised.

**If coverage drove the target**, record the before/after so the loop is
measurable. Regenerating the full report costs ~10 min, so prefer the cheap
targeted check: re-run just your new probe under instrumentation and report the
one file's number.

```bash
export RUSTFLAGS="-C instrument-coverage"
export LLVM_PROFILE_FILE=/tmp/advcov/p-%p-%m.profraw
(cd python && maturin develop --release)          # instrument
python adversary/runs/<your-probe>.py             # exercise it
LLVMBIN="$(dirname "$(rustc --print target-libdir)")/bin"
SO="$(python -c 'import pounce._pounce as m; print(m.__file__)')"
"$LLVMBIN/llvm-profdata" merge -sparse /tmp/advcov/*.profraw -o /tmp/advcov/m.profdata
"$LLVMBIN/llvm-cov" report "$SO" -instr-profile=/tmp/advcov/m.profdata \
  | grep '<the file you targeted>'
unset RUSTFLAGS LLVM_PROFILE_FILE
(cd python && maturin develop --release)          # RESTORE - do not skip
```

A probe that moves the number is worth keeping even when it finds no bug: it
converts an unverified region into a verified one. Say so explicitly in the
report — "no defect; raised `nl_tape.rs` 50.2% → 57.1%" is a real result.

### 8. File GitHub issues (only for genuine findings)
File with `gh issue create` **only** for: `SOLVER_BUG`, `SOLVER_LIMITATION`,
`PERFORMANCE_REGRESSION`, `ROUTING_ERROR`, `GRADIENT_ERROR`. Do **not** file
`FORMULATION_ERROR`, `REFERENCE_ERROR`, or `TOLERANCE`.

**CRITICAL — issue body must go through a file (project rule):** write the body
to a temp file and pass `--body-file`. NEVER use inline `--body "$(cat <<EOF…)"`
or inline prose — backticks/`$()` in the body get shell-evaluated and silently
run commands instead of filing the issue.

```
cat > /tmp/adv-issue.md <<'EOF'
> Discovered by the pounce adversary agent (automated solver correctness testing).

## Family / solver
<family> — <entry point>

## Problem
<formulation, source citation, known optimum>

## Expected vs actual
<oracle/known value vs pounce value, relative error>

## Cross-check
<which independent oracle confirmed it; for nlp the `pounce verify` exit code
and the Ipopt comparison; the exact options tried>

## Reproduce
<the adversary/runs/…py path and how to run it>
EOF
gh issue create --title "[Adversary] <verdict>: <problem> (<family>)" \
  --body-file /tmp/adv-issue.md --label bug
```
Label guidance: `SOLVER_BUG`/`ROUTING_ERROR`/`GRADIENT_ERROR` → `bug`;
`SOLVER_LIMITATION` → `enhancement`; `PERFORMANCE_REGRESSION` → `performance`.

### 9. Multi-problem summary
When `count > 1`, print at the end:

```
| # | Problem | Family | pounce | oracle | Speedup | Verdict |
|---|---------|--------|--------|--------|---------|---------|
| 1 | ...     | socp   | PASS   | PASS   | 1.8x    | PASS    |
| 2 | ...     | nlp    | FAIL   | PASS   | n/a     | SOLVER_LIMITATION |
```
`Speedup = oracle_time / pounce_time` (> 1 means pounce is faster). Include
total PASS/FAIL counts and a list of any issues filed (with numbers).

### 10. Publish to the shared branch
Every run ends by pushing its results to `adversary-runs`, or the next run
(local or cloud) starts from stale state and the unified log falls apart.
Publish through a throwaway worktree so the current checkout is untouched:

```bash
git fetch origin adversary-runs
git worktree add /tmp/adv-publish adversary-runs 2>/dev/null \
  || git worktree add /tmp/adv-publish -b adversary-runs origin/adversary-runs
git -C /tmp/adv-publish pull --ff-only origin adversary-runs
# Copy ONLY what this run produced/changed: log.org + the new top-level
# runs/ files. Never copy runs/ subdirectories.
cp adversary/log.org /tmp/adv-publish/adversary/log.org
cp adversary/runs/$(date +%Y-%m-%d)_* /tmp/adv-publish/adversary/runs/ 2>/dev/null
git -C /tmp/adv-publish add -f adversary/   # -f: .git/info/exclude is shared
git -C /tmp/adv-publish commit -m "adversary: <one-line run summary> [skip ci]"
git -C /tmp/adv-publish push origin adversary-runs \
  || (git -C /tmp/adv-publish pull --rebase origin adversary-runs \
      && git -C /tmp/adv-publish push origin adversary-runs)
git worktree remove /tmp/adv-publish
```

Before that `commit`, check the staged list against `adversary/.gitignore`'s
intent — **this repo is public**: no `runs/` subdirectories (scratch), no
`*.lst`/GAMS logs (they embed the site license banner), never `gamslice.dat` /
`gamscntr.dat` (the license file itself), and nothing resembling credentials
or keys. `git -C /tmp/adv-publish status --short` and read it. The `[skip ci]`
suffix is mandatory — ci.yml triggers on every branch push and a report-only
commit must not re-run the matrix.

## Constraints
- **NEVER** modify `crates/*/src`, `python/pounce/**`, `pyomo-pounce/**`, or any
  test files. Create/modify only under `adversary/`.
- Installing oracle packages (`cvxpy`, `pyomo`) into `.venv-qa` is allowed —
  they are not pounce source.
- Time budget: **< 10 s per problem** (pounce is fast; SOS may be the slowest —
  keep the relaxation order small).
- An iteration-limit / restoration failure is a `SOLVER_LIMITATION`, not a bug.
- Be skeptical of your own formulations: derivative errors, wrong cone
  encodings, and non-DCP oracle models are more likely than solver bugs.
- For conic problems, always re-derive cone dimensions and (for PSD) the `svec`
  scaling before concluding pounce is wrong.
- For nlp, treat `pounce verify` rejecting the claimed solution as strong
  evidence of a real bug; treat it accepting the solution as strong evidence
  your "wrong answer" is actually a different (valid) optimum or a formulation
  issue.
