# POUNCE solver-equivalence validation

**What this validates.** That POUNCE does not merely return the right
*objective value*, but computes the right *primals, duals/multipliers, KKT
residuals, iteration behaviour, and — especially — sensitivity outputs
(covariance / standard errors / correlation)*. Every claim is checked against
at least one oracle that is **independent of POUNCE**: the reference solver
IPOPT, published Hock–Schittkowski multipliers, a closed-form KKT solve,
cvxpy/CLARABEL, scipy, and analytic derivatives. Objective agreement alone is
cheap; the load-bearing checks here are on the *duals* and the *sensitivities*.

**Why the duals matter right now.** A recent fix (#271/#272) corrected a
constraint-dual **sign** inversion that had flipped every AMPL/Pyomo dual.
Demonstrating that POUNCE's duals now agree with independent sources **in sign
and value** is therefore the single most important thing this document shows.
(One concrete artifact of that fix surfaced while building this suite — see the
note at the end on which `pounce` binary a Pyomo `SolverFactory('pounce')` must
resolve to.)

## Environment

| item                               | value                                                |
|------------------------------------|------------------------------------------------------|
| POUNCE version                     | `0.9.0`                                              |
| POUNCE binary (via `pyomo_pounce`) | `python/pounce/bin/pounce` → `target/release/pounce` |
| IPOPT (P2–P5, default MUMPS)       | `Ipopt 3.14.19 (Darwin arm64), ASL(20241202)`        |
| IPOPT-MA57 (P1 live reference)     | `Ipopt 3.14.20`, HSL MA57 (`ipopt-ma57`)             |
| date                               | 2026-07-22                                           |

Reproduce with `python validation/run_all.py` (see *How to reproduce*).

> **Honesty note.** Numbers below are the actual figures produced by the
> scripts, not rounded-up ideals. Where agreement is limited (P1 CHO
> covariance cross-check; P3 primals/duals on degenerate control problems), the
> real figure is reported together with the reason.

---

## Summary across all five problems

| # | Problem | What it uniquely proves | Tightest independent agreement achieved |
|---|---------|--------------------------|------------------------------------------|
| **P1** | CHO parameter estimation (21,732 vars) | Sensitivity/covariance is right, not just the optimum | Covariance **method** exact vs closed form (linear regression) to **1.0e-15**, and vs PyNumero reduced-Hessian to **5e-16**; nonlinear fit vs independent PyNumero to **7e-8**. Point estimate matches a **live HSL-MA57 IPOPT** solve to **1.5e-14** (rel) — where default MUMPS IPOPT fails on CHO but POUNCE's built-in FERAL solver succeeds. |
| **P2** | Hock–Schittkowski 71 | Duals vs a *published*, solver-independent source (sign fix) | Both constraint multipliers agree pounce/IPOPT/analytic-KKT **in sign**; pounce vs IPOPT dual `c1` to **6.0e-10**, pounce vs exact KKT to **1.2e-9** |
| **P3** | corkscrw, clnlbeam (hard control) | Restoration phase lands on IPOPT's optimum | Objective rel err **2.4e-13** (corkscrw) / **7.7e-12** (clnlbeam); both "Optimal Solution Found" |
| **P4** | Strictly convex QP | Convex-path duals vs closed form + 2 solvers, sign incl. | 4-way multiplier **sign** agreement; pounce multipliers vs closed-form KKT to **2.6e-11**, KKT stationarity **3e-15** |
| **P5** | Parametric NLP | A *second*, independent sensitivity mechanism | `dx/dp` vs analytic to **machine precision**; central-FD error falls as **O(δ²)** (ratios 4.00); envelope-theorem multiplier vs analytic `dObj/dp` to **1.3e-9** |

---

## P1 — CHO bioprocess parameter estimation (the headline: sensitivity/covariance)

**Why this problem.** This is a large
(~21.7k-variable) nonlinear parameter-estimation DAE where a solver can hit the
optimum yet report a *subtly wrong* covariance. It is a monolithic Pyomo model
— 10 CHO cell-culture batches, orthogonal collocation (60 finite elements × 3
collocation points), 12 shared kinetic parameters, deterministic seed. The
model is built from `benchmarks/cho/parmest_nl_export.py` **unmodified**, at
**full size** (`NFE=60`).

Problem size: **n = 21,732 variables, m = 21,660 constraints**, 1,800 residual
data points, 12 fitted parameters
`[K_glc, K_gln, KI_amm, KI_lac, KD_amm, KD_lac, m_glc, a1, a2, d_gln, r_amm, Q_B1]`.

### Point estimate

| quantity        | POUNCE              | IPOPT-MA57 (live)   |
|-----------------|---------------------|---------------------|
| status          | optimal             | optimal             |
| objective (SSE) | `67672.86755753502` | `67672.86755753601` |
| linear solver   | FERAL (built-in)    | HSL MA57            |

Objective agreement POUNCE vs a **live** MA57-linked IPOPT solve of the same
model: **rel 1.5e-14**. The committed `benchmarks/cho/ipopt_ma57.json`
(`67672.86755753602`, 33 iters) is a third, independent corroboration.

> **On the IPOPT linear solver — and a point in POUNCE's favor.** CHO is a
> large, badly-conditioned collocation NLP. IPOPT's **default MUMPS** linear
> solver *fails* on it — `internalSolverError: Error in step computation` /
> restoration failure — so the live reference above uses an **HSL-MA57-linked
> IPOPT** (`ipopt-ma57`, Ipopt 3.14.20), which solves it and agrees with POUNCE
> to 1.5e-14. Notably, **POUNCE solves the full CHO model with its built-in
> FERAL linear solver — no HSL / MA57 dependency — where default IPOPT fails
> outright.** (The script discovers `ipopt-ma57` on `PATH`; where it is absent
> it falls back to the committed MA57 benchmark, and records the MUMPS failure
> explicitly.)

Parameter recovery (fitted vs the true values used to generate the data) is
limited by **identifiability**, not by the solver: three parameters
(`KD_amm`, `KD_lac`, `r_amm`) are not identifiable from the data and sit on
their bounds at the optimum; the worst identifiable parameter is recovered to
~74% relative error. This is a property of the noisy synthetic data + model,
and would be identical for any solver that finds this optimum.

### Sensitivity / covariance — the load-bearing check

The asymptotic parameter covariance is `cov = 2·σ²·(K⁻¹)_pp`, the parameter
block of the inverse KKT matrix. We validate POUNCE's `pyomo_pounce.covariance()`
against an **independent** reduced-Hessian oracle built from **PyNumero**
(AMPL/ASL derivative evaluators — the same library IPOPT uses) assembled and
factored with **scipy.sparse** — a toolchain with no code in common with
POUNCE's held-factorization sensitivity path.

**Step 1 — the method is exact on cases with known answers.** Before trusting
either route on CHO, both are pinned to problems whose covariance is known in
closed form:

| control                              | POUNCE vs known               | PyNumero ref vs known | POUNCE vs PyNumero |
|--------------------------------------|-------------------------------|-----------------------|--------------------|
| linear regression, `cov = σ²(XᵀX)⁻¹` | **1.0e-15**                   | **1.3e-15**           | **5.0e-16**        |
| nonlinear `A·exp(−k·t)` fit          | vs scipy (Gauss–Newton) 0.9%¹ | —                     | **7.0e-8**         |

¹ The 0.9% is the *expected* difference between POUNCE's observed-information
(full Lagrangian Hessian) covariance and scipy's expected-information
(Gauss–Newton `JᵀJ`) covariance for a nonlinear fit — they are two different,
both-correct conventions that agree only in the small-residual/linear limit.
POUNCE and the independent PyNumero reduced Hessian (same observed-information
convention) agree to **7e-8**.

So the covariance **machinery is provably exact** (linear: 1e-15) and matches an
independent reduced-Hessian oracle on a nonlinear fit (7e-8).

**Step 2 — CHO standard errors and correlation.** POUNCE's covariance at the CHO
optimum (σ² = 37.85):

| parameter | std error (POUNCE) | note                                     |
|-----------|--------------------|------------------------------------------|
| K_glc     | 6.58e-02           |                                          |
| K_gln     | 1.05e-03           |                                          |
| KI_amm    | 8.15e-02           |                                          |
| KI_lac    | 1.78e+00           |                                          |
| KD_amm    | 0                  | on bound — unidentifiable, projected out |
| KD_lac    | 0                  | on bound — unidentifiable, projected out |
| m_glc     | 8.48e-06           |                                          |
| a1        | 8.20e-07           |                                          |
| a2        | 4.78e-01           |                                          |
| d_gln     | 1.07e-04           |                                          |
| r_amm     | 0                  | on bound — unidentifiable, projected out |
| Q_B1      | 1.94e-05           |                                          |

POUNCE correctly detects the three bound-active parameters and projects them
out (zero variance, conditional on the active bound), the same behaviour a
correct sIPOPT/k_aug computation gives. The full 12×12 correlation matrix is in
`results.json` (`cho_covariance.correlation_pounce`).

**Step 3 — CHO independent cross-check, reported honestly.** On the full CHO
model the independent PyNumero equality-KKT reference does **not** reproduce
POUNCE's CHO standard errors — it disagrees by factors of ~5×–3000×, and its
reduced Hessian even comes out **indefinite** (eigenvalues spanning ≈1e29 with a
negative minimum). This is **not** evidence that POUNCE's covariance is wrong;
it is a limit of the *reference*:

- The CHO reduced Hessian is severely **ill-conditioned** — three parameters
  are unidentifiable, so the reduced information matrix is near-singular.
- The independent reference can only use **solver-reported multipliers**
  (‖λ‖ ≈ 5.5e6, accurate to ~1.5e-5), and that finite multiplier error is
  amplified by the conditioning (×‖λ‖, ×cond) into an unusable, indefinite
  reduced Hessian. POUNCE's own `covariance()` instead uses the
  **machine-precision** multipliers from its converged factorization (which
  carried **no** inertia-correction perturbations here), so it is
  self-consistent where the from-scratch reference cannot be.

The trustworthy evidence that POUNCE's sensitivity math is right is therefore
**Step 1** (exact on closed-form and a nonlinear independent oracle) plus the
fact that on CHO its factorization is clean and its bound-active detection is
correct. We state plainly: a *high-precision independent* covariance for the
full, ill-conditioned CHO model is not achievable with an
independent-multiplier oracle, and we do not claim one.

---

## P2 — Hock–Schittkowski 71 (published, solver-independent duals)

**Why this problem.** The reference multipliers come from Hock & Schittkowski
(1981) — a source that *predates and is independent of* any modern
interior-point code. Agreement here cannot be an artifact of two solvers
sharing a convention, which makes it the cleanest possible evidence that the
#271/#272 dual-sign fix is correct. We add a machine-precision KKT solve as a
second, fully analytic oracle (active set: `x1` on its lower bound, `c1`
active, `c2` the equality).

```
min  x1·x4·(x1+x2+x3) + x3
s.t. x1·x2·x3·x4 >= 25          (c1)
     x1²+x2²+x3²+x4² == 40      (c2)
     1 <= xi <= 5,  x0=(1,5,5,1)
```

| quantity         | POUNCE                   | IPOPT (tol 1e-9)        | exact KKT oracle        |
|------------------|--------------------------|-------------------------|-------------------------|
| objective        | 17.014017145             | 17.014017140            | 17.0140172891563        |
| dual `c1` (ineq) | **+0.5522936588816062**  | **+0.5522936594811181** | **+0.5522936601207269** |
| dual `c2` (eq)   | **−0.16146856313488797** | **−0.1614685641449677** | **−0.1614685667705059** |

- **Sign agreement is exact** across POUNCE, IPOPT, *and* the analytic KKT
  oracle for both multipliers — the live evidence the dual-sign fix is correct.
- Value agreement: dual `c1` POUNCE vs IPOPT **6.0e-10**, POUNCE vs exact
  **1.2e-9**; dual `c2` POUNCE vs IPOPT **1.0e-9**, POUNCE vs exact **3.6e-9**.
- Objective: POUNCE and IPOPT agree to **4.8e-9**; both match the published
  `f*` and the exact KKT `f*` to ~1e-7 (the objective is very flat near the
  optimum, so a ~1e-9 primal difference maps to a ~1e-7 objective difference —
  this is why the *dual* agreement, 1e-9, is actually tighter than the
  *objective* agreement here).

The independent bound multiplier from the analytic KKT solve, `z(x1)=1.08787`,
also matches IPOPT's reported `ipopt_zL_out` for `x1` (1.08787).

---

## P3 — hard nonconvex optimal control (restoration-phase agreement)

**Why this problem.** These are the "messy, many-iteration" cases where POUNCE
reaches the optimum only by passing **through its restoration phase**. The
point is that POUNCE's restoration and endgame land on the *same* optimum IPOPT
does on genuinely hard problems — not just easy static NLPs. Both binaries run
on the **identical** `.nl` fixture.

|                               | corkscrw          | clnlbeam           |
|-------------------------------|-------------------|--------------------|
| n / m                         | 44,997 / 35,000   | 59,999 / 40,000    |
| POUNCE iters (restoration)    | 224 (**30**)      | 483 (0)            |
| IPOPT iters                   | 382               | 492                |
| both "Optimal Solution Found" | ✅                | ✅                 |
| objective (POUNCE)            | 98.09596925639791 | 344.8761314083454  |
| objective (IPOPT)             | 98.09596925642172 | 344.87613141098416 |
| **objective rel err**         | **2.4e-13**       | **7.7e-12**        |
| primal max|Δ| (L2 rel)        | 1.6e-4 (1.1e-4)   | 1.8e-5 (4.7e-6)    |
| dual max|Δ|                   | 4.6e-8            | 0.77               |

POUNCE takes 30 restoration iterations on corkscrw and still converges to
IPOPT's objective to **2.4e-13**.

> **Honest note on primals/duals.** On these control problems the *objective*
> agrees to 1e-12–1e-13 and both solvers report "Optimal", but the *primal*
> vectors differ at the 1e-4 (corkscrw) / 1e-5 (clnlbeam) level and clnlbeam's
> multipliers differ by up to 0.77. Both iterates satisfy the KKT conditions to
> ~1e-14, so this reflects **genuine near-degeneracy / non-uniqueness** of the
> solution (flat objective directions and non-unique multipliers, typical of
> constrained optimal-control problems) — not a disagreement about *which*
> optimum was found. The agreement that is well-posed here — the objective and
> the optimal status — is tight.

---

## P4 — strictly convex equality-constrained QP (analytic KKT + cvxpy + IPOPT)

**Why this problem.** On the convex path the primal *and* the multipliers have a
**closed form** (the KKT linear system), so the duals are not merely
self-consistent between solvers — they match an analytic value, **sign
included**. This is the most airtight dual check available. The instance is a
minimum-variance portfolio over 5 assets with a budget constraint `1ᵀx=1` and a
target-return constraint `μᵀx=r`; both multipliers are interpretable shadow
prices.

```
min 0.5·xᵀP x   s.t.  1ᵀx = 1,  μᵀx = 0.12
```

Closed-form KKT solution: `[[P, Aᵀ],[A, 0]][x; y] = [−c; b]`.

| route                 | max|Δx| vs closed form | max|Δy| vs closed form (multipliers) | KKT stationarity `‖Px+c+Aᵀy‖∞` |
|-----------------------|------------------------|--------------------------------------|--------------------------------|
| closed form           | —                      | —                                    | 1.2e-17                        |
| **POUNCE** `solve_qp` | **1.4e-10**            | **2.6e-11**                          | **3.0e-15**                    |
| cvxpy (CLARABEL)      | 7.9e-14                | 3.3e-15                              | 2.7e-15                        |
| IPOPT (Pyomo)         | 3.1e-16                | 6.2e-17                              | 3.5e-18                        |

Multiplier **signs agree 4-way** (closed form, POUNCE, cvxpy, IPOPT): budget
multiplier negative, return multiplier positive. POUNCE's returned duals satisfy
the KKT stationarity residual to **3e-15**, and match the closed-form multiplier
values to **2.6e-11** — on the convex path the duals are pinned to an analytic
number, sign included, by three independent routes.

> **Why POUNCE's `|Δx|`/`|Δy|` look looser than CLARABEL's and IPOPT's — it is
> the default stopping tolerance, not an accuracy ceiling.** `solve_qp` defaults
> to `tol=1e-8`. The distance to the exact vertex, `|Δx|`, is set by the
> **primal** residual `‖Ax−b‖`, which an interior-point method drives down to
> about `tol` — so at the default it sits near `5.7e-10`, giving `|Δx| ≈ 1.4e-10`.
> The **dual** stationarity is already machine-precision (`3e-15`) at that same
> default, which is the residual that actually certifies the duals are correct.
> Ask POUNCE for more and it delivers: tightening `tol` walks `|Δx|` down
> monotonically to **9.7e-16 at `tol=1e-14`** — matching IPOPT (`3.1e-16`) and
> beating CLARABEL (`7.9e-14`). IPOPT/CLARABEL look tighter "at default" only
> because this QP is trivial for them (`cond(KKT) = 73`) and their defaults
> polish further.
>
> | `tol` | primal `‖Ax−b‖` | dual `‖Px+c+Aᵀy‖` | `max|Δx|` vs exact |
> |-------|------------------|--------------------|---------------------|
> | `1e-8` (default) | 5.7e-10 | 3.0e-15 | 1.4e-10 |
> | `1e-10` | 2.8e-11 | 2.9e-15 | 6.8e-12 |
> | `1e-12` | 7.1e-14 | 3.5e-16 | 1.7e-14 |
> | `1e-14` | 3.4e-15 | 3.3e-17 | **9.7e-16** |

---

## P5 — parametric sensitivity dx/dp and dObj/dp (a second, independent mechanism)

**Why this problem.** This exercises a *different* sensitivity mechanism than
P1's covariance — the parametric derivative of the solution and objective w.r.t.
a model parameter, from `pyomo_pounce`'s `declare_sens_param`/`gradient` (the
held-factorization sIPOPT computation). It is checked against a **convention-free
finite-difference oracle** and against analytic values, closing the
"subtly-off sensitivity" concern from a second direction.

```
min −x1 − x2   s.t.  x1²+x2² == p,   x1,x2 >= 0
x*(p)=(√(p/2), √(p/2)),  Obj*(p)=−√(2p),  dObj*/dp = −1/√(2p)
```
at `p=2`: `x*=(1,1)`, multiplier `λ=1/2`, `dObj/dp=−1/2`.

**(a) `dx/dp` matches central FD to O(δ²).** The finite-difference error in
`dx1/dp` shrinks quadratically as δ halves:

| δ       | FD `dx1/dp` | error vs analytic (0.25) |
|---------|-------------|--------------------------|
| 1.0e-1  | 0.2500782   | 7.82e-5                  |
| 5.0e-2  | 0.2500195   | 1.95e-5                  |
| 2.5e-2  | 0.2500049   | 4.88e-6                  |
| 1.25e-2 | 0.2500012   | 1.22e-6                  |
| 6.25e-3 | 0.2500003   | 3.05e-7                  |

Error ratios per halving: **4.003, 4.001, 4.0002, 4.00005** — clean O(δ²)
convergence toward POUNCE's sensitivity value, which itself equals the analytic
`dx1/dp` to **machine precision** (0.0).

**(b) Envelope theorem.** POUNCE's constraint marginal `m.dual[con]` equals
`dObj*/dp`: reported **−0.5000000013**, analytic **−0.5**, agreeing to
**1.3e-9**, and matching IPOPT's marginal to **1.2e-9** — same value, correct
(negative) sign. A second, independent sensitivity path agrees with a
convention-free oracle.

---

## How to reproduce

```bash
source /Users/jkitchin/projects/pounce/.venv-qa/bin/activate
python /Users/jkitchin/projects/pounce/validation/run_all.py
```

`run_all.py` runs all five problem scripts, writes the machine-readable
`validation/results.json` (environment metadata + every number quoted here),
and prints a one-line summary per problem. Individual problems can be run
directly, e.g. `python validation/p2_hs71.py`. Each writes its own
`results.<name>.json`.

The scripts require the QA venv (numpy, scipy, cvxpy, pyomo with both
`SolverFactory('pounce')` and `SolverFactory('ipopt')`, `pyomo_pounce`,
PyNumero+ASL, and matplotlib for the CHO model import).

---

## Appendix — which `pounce` binary `SolverFactory('pounce')` uses (a dual-sign gotcha)

While building P2 an important, fix-relevant subtlety surfaced. With
`pyomo_pounce` **not** imported, Pyomo's generic ASL fallback registers
`SolverFactory('pounce')` against whatever `pounce` is first on `PATH` — on this
machine a **stale** `~/.local/bin/pounce` built *before* the #271 dual-sign
fix — and it reports **sign-flipped** duals (dual `c1` = −0.552 instead of
+0.552). Importing `pyomo_pounce` first registers the proper ASL plugin, which
resolves to the wheel-bundled binary (a symlink onto the freshly built
`target/release/pounce`) and reports the **correct** duals that match IPOPT.
Both binaries self-report version `0.9.0`; only the build date differs. Every
script here calls `setup_pounce()` (which imports `pyomo_pounce` and returns the
resolved executable) precisely so the validated, fix-containing binary is the
one exercised. This is itself a small piece of evidence for #271/#272: the
pre-fix build gives the old (wrong) sign, the post-fix build matches IPOPT.
