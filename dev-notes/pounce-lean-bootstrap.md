> ## SUPERSEDED — historical bootstrap guide
>
> The `pounce-lean` repository exists and is well past every milestone below:
> the reusable lemmas, the codegen, the fixtures, and the drift guard are all
> implemented, and the whole path is kernel-checked end to end by
> `scripts/check-lean-cert.sh`. Kept for its milestone reasoning.
>
> Note the module namespace here (`PounceLean.Examples.QP`) is not the one the
> emitter's fixtures use (`PounceLean.CertifyQP`); both sides now share the
> latter so the generated proofs are byte-identical artifacts.

# pounce-lean — bootstrap spec

**Purpose.** A self-contained guide to stand up the `pounce-lean` repository
locally and reach its first validated milestone: a Lean 4 + Mathlib project that
**kernel-checks a POUNCE solution certificate**. This document is deliberately
standalone — you can take it to a fresh directory and not need the POUNCE repo
open. (Companion design material, if you have the POUNCE repo: `dev-notes/lean-certificate.md`,
`dev-notes/lean-cert-schema-v1.md`, `dev-notes/lean-cert-example/`.)

---

## 1. What this repo is

`pounce-lean` turns a **certificate** emitted by POUNCE into a **kernel-checked
proof** that a returned `x*` is feasible and (where tractable) a minimum, over
**exact rational arithmetic** — no floating point in the trusted path.

POUNCE produces *data*; `pounce-lean` produces and *checks a proof*:

```
 .nl (canonical, hashed)
        │  POUNCE (separate repo): emits DATA, never a proof
        ▼
   cert.json   ──(schema: pounce.lean-cert/v1)──┐
        │                                         │  THIS REPO
        ▼                                         ▼
   codegen: cert.json → .lean      reusable lemmas (PSD-via-LDLᵀ, convex-KKT⟹global)
        │
        ▼
   lake build  →  verdict (exit 0 = proof checks, nonzero = it does not)
```

### The trust property that makes this work

The witnesses POUNCE supplies (duals, matrix factorizations, SOS data) **do not
need to be trusted.** Wrong witness data makes the proof *fail to typecheck* —
it can never make a false statement pass. So POUNCE can be adversarial and forge
nothing; `pounce-lean` only ever accepts what the Lean kernel accepts.

The unforgeability anchor is therefore the **Lean kernel**, not a secret key
(contrast POUNCE's HMAC receipts, which are only as strong as key isolation). A
consumer accepts a result **iff**:

1. `lake build` succeeds on the generated `.lean`, **and**
2. the proof's embedded `nl_sha256` literal equals the SHA-256 of the consumer's
   *own canonical* `.nl`, **and**
3. `statement_sha256` (recorded in the post-build receipt) equals the hash of the
   statement re-derived from that `.nl`.

`statement_sha256` and any signature are **not** in the certificate — they live
in the verification receipt produced *after* codegen, because the statement is
derived here, not by POUNCE.

---

## 2. Claim tiers (what "is a minimum" is allowed to mean)

The certificate names exactly one proven claim. Do not oversell — these are
wildly different strengths:

| Verdict | Means | Tractable for |
|---|---|---|
| `feasible` | `x*` satisfies all constraints/bounds (within a declared ε) | any algebraic (polynomial/rational) model |
| `local-min-strict` | KKT + second-order sufficient ⟹ strict local minimizer | smooth algebraic NLP |
| `global-min` | certified global minimizer | convex (LP / QP / convex NLP) **or** polynomial-via-SOS |

**Milestone 1 targets `global-min` on `qp-convex`** — the smallest slice that
exercises the whole pipeline (lossless rational `x*`, exact constraint eval,
KKT, PSD Hessian ⟹ global) with no SOS machinery and no equality-residual fuzz.

The enabling fact: **every f64 is exactly a dyadic rational `m·2^e`**, so
converting `x*`/`λ` to Lean `ℚ` is *lossless*. The only approximation is that
`x*` ≠ the true optimum; for the QP slice even that vanishes (the point is exact).

---

## 3. Prerequisites

- **elan** (Lean version manager): https://github.com/leanprover/elan
  ```sh
  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  ```
- **git**, and ~10 GB free (Mathlib's prebuilt cache is large).
- Do **not** hand-install a Lean toolchain; the `lean-toolchain` file pins it and
  elan fetches the right one.

---

## 4. Create the project

Use Mathlib's project template — it pins a matching toolchain + Mathlib revision
for you (do **not** guess version numbers by hand):

```sh
lake new pounce_lean math
cd pounce_lean
```

This scaffolds:

```
pounce_lean/
├─ lean-toolchain         # pins leanprover/lean4:vX.Y.Z (auto-selected)
├─ lakefile.toml          # requires mathlib
├─ lakefile or .lean      # (template-dependent)
├─ PounceLean.lean        # library root (imports submodules)
└─ PounceLean/
   └─ Basic.lean
```

**Immediately fetch the prebuilt Mathlib cache** — without this, the first build
compiles all of Mathlib (hours):

```sh
lake exe cache get
lake build          # should succeed on the empty scaffold
```

Record the pinned versions for the certificate's `toolchain` field:

```sh
cat lean-toolchain                     # → lean value
lake env lean --version
# Mathlib rev: look in lake-manifest.json for the mathlib entry's "rev"
```

> These two values backfill the `toolchain.lean` / `toolchain.mathlib`
> placeholders in the schema. The cert is *authored against* a toolchain;
> reproducing the proof requires the same one.

---

## 5. Target directory layout

```
pounce_lean/
├─ lean-toolchain
├─ lakefile.toml
├─ PounceLean.lean                 # root: imports the lib
├─ PounceLean/
│  ├─ Rational.lean                # cert ℚ helpers (parse {num,den}, inf sentinels)
│  ├─ PSD.lean                     # LEMMA: M = L·diag(D)·Lᵀ ∧ D ≥ 0 ⟹ xᵀMx ≥ 0
│  ├─ ConvexQP.lean                # THEOREM: convex + KKT + feasible ⟹ global min
│  └─ Examples/
│     └─ QP.lean                   # the worked example (Milestone 1 fixture)
├─ certs/
│  └─ qp.cert.json                 # input certificate (from POUNCE; here hand-written)
├─ codegen/                        # Milestone 2: cert.json → .lean (lang TBD; see §8)
└─ test/
   └─ fixtures/                    # golden (cert.json, expected .lean) pairs
```

---

## 6. Milestone 1 — compile the worked example (do this FIRST)

This is the highest-value de-risking step: it validates the *only* unproven
part of the architecture — that Lean actually closes the proof — using a
hand-written certificate, with **no codegen and no POUNCE emitter required**.

### 6a. The problem

A convex QP chosen so every quantity is exact:

```
minimize    f(x) = x₁² + x₂²              (= ½ xᵀQx with Q = diag(2,2))
subject to  x₁ + x₂ ≥ 1
```

Solution `x* = (1/2, 1/2)`, `f(x*) = 1/2`, dual `λ = 1`. The active constraint
holds exactly (`1/2 + 1/2 = 1`), so `tolerance = 0`.

### 6b. The input certificate — `certs/qp.cert.json`

```json
{
  "schema": "pounce.lean-cert/v1",
  "verdict": "global-min",
  "problem_class": "qp-convex",
  "tolerance": { "num": "0", "den": "1" },
  "binding": {
    "nl_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "sol_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "solver": "pounce 0.5.0"
  },
  "toolchain": { "lean": "leanprover/lean4:v4.x.0", "mathlib": "<git rev>" },
  "problem": {
    "n_vars": 2,
    "objective": {
      "kind": "quadratic",
      "half_quadratic": true,
      "Q": { "rows": 2, "cols": 2, "symmetric": true,
             "entries": [ {"i":0,"j":0,"val":{"num":"2","den":"1"}},
                          {"i":1,"j":1,"val":{"num":"2","den":"1"}} ] },
      "c": [ {"num":"0","den":"1"}, {"num":"0","den":"1"} ],
      "constant": {"num":"0","den":"1"}
    },
    "var_bounds": { "lower": ["-inf","-inf"], "upper": ["+inf","+inf"] },
    "constraints": [
      { "name": "c0",
        "coeffs": [ {"num":"1","den":"1"}, {"num":"1","den":"1"} ],
        "lower": {"num":"1","den":"1"}, "upper": "+inf" }
    ]
  },
  "candidate": {
    "x": [ {"num":"1","den":"2"}, {"num":"1","den":"2"} ],
    "objective": {"num":"1","den":"2"}
  },
  "witnesses": {
    "duals": [ {"num":"1","den":"1"} ],
    "hessian_psd": {
      "of": "Q",
      "L": { "rows": 2, "cols": 2, "unit_lower": true, "entries": [] },
      "D": [ {"num":"2","den":"1"}, {"num":"2","den":"1"} ]
    },
    "active_set": [ 0 ]
  }
}
```

### 6c. The proof — `PounceLean/Examples/QP.lean`

This is the **codegen target** — for Milestone 1 you hand-write it; Milestone 2
generates exactly this from the cert. It has not yet been run through Lean;
confirming it builds **is** Milestone 1.

```lean
/-
  POUNCE Lean certificate — worked example.
  Generated target for certs/qp.cert.json (pounce.lean-cert/v1,
  verdict = global-min, problem_class = qp-convex).

      minimize    f(x) = ½ xᵀQx,  Q = diag(2,2)   ⇒  f = x₁² + x₂²
      subject to  x₁ + x₂ ≥ 1
  Candidate:  x* = (1/2, 1/2),  f(x*) = 1/2,  dual λ = 1.
-/
import Mathlib

namespace PounceLean.Examples.QP

-- binding (cert.binding); embed so the theorem provably concerns these bytes
def nlSha256  : String := "0000000000000000000000000000000000000000000000000000000000000000"
def solSha256 : String := "0000000000000000000000000000000000000000000000000000000000000000"

/-- Objective, expanded from ½·xᵀQx with Q = diag(2,2). Over `ℚ`. -/
def f (x₁ x₂ : ℚ) : ℚ := x₁ ^ 2 + x₂ ^ 2

/-- Feasible set: the single linear constraint `1 ≤ x₁ + x₂`. -/
def Feasible (x₁ x₂ : ℚ) : Prop := 1 ≤ x₁ + x₂

-- candidate x* (cert.candidate.x), exact rationals
def xs₁ : ℚ := 1 / 2
def xs₂ : ℚ := 1 / 2

/-- Tier 1: the candidate is feasible (exactly: 1/2 + 1/2 = 1 ≥ 1). -/
theorem candidate_feasible : Feasible xs₁ xs₂ := by
  unfold Feasible xs₁ xs₂; norm_num

/-- Sanity: reported objective matches (cert.candidate.objective). -/
theorem candidate_objective : f xs₁ xs₂ = 1 / 2 := by
  unfold f xs₁ xs₂; norm_num

/--
  Tier 3 (global): `x*` is a global minimizer.

  Exact convex-QP argument:
    f(y) − f(x*) = ½ (y−x*)ᵀ Q (y−x*)     -- ≥ 0 by hessian_psd (Q ⪰ 0)
                 + ∇f(x*)·(y−x*)           -- ≥ 0 by KKT (duals) + feasibility
  For this instance the SOS witness is
    y₁²+y₂² − ½ = ½(y₁−y₂)² + ½(y₁+y₂−1)² + (y₁+y₂−1),
  all terms ≥ 0 once `1 ≤ y₁+y₂`. `nlinarith` should find this from the hints.
-/
theorem global_min :
    ∀ y₁ y₂ : ℚ, Feasible y₁ y₂ → f xs₁ xs₂ ≤ f y₁ y₂ := by
  intro y₁ y₂ hfeas
  unfold f xs₁ xs₂ Feasible at *
  nlinarith [sq_nonneg (y₁ - y₂), sq_nonneg (y₁ + y₂ - 1), hfeas]

end PounceLean.Examples.QP
```

Add `import PounceLean.Examples.QP` to `PounceLean.lean`, then:

```sh
lake build
```

**Success = `lake build` exits 0 with no `sorry` and no errors.** That validates
the riskiest claim in the entire design.

### 6d. If `nlinarith` balks

The SOS decomposition in the docstring is exact, so a manual fallback always
exists — replace the `nlinarith` line with the explicit identity:

```lean
  have key : y₁ ^ 2 + y₂ ^ 2 - (1/2 : ℚ)
      = (1/2) * (y₁ - y₂)^2 + (1/2) * (y₁ + y₂ - 1)^2 + (y₁ + y₂ - 1) := by ring
  nlinarith [sq_nonneg (y₁ - y₂), sq_nonneg (y₁ + y₂ - 1), hfeas, key]
```

(or finish from `key` with `linarith [sq_nonneg …, hfeas]`). The point of
Milestone 1 is to discover which tactic invocation actually lands and lock it in
as the codegen's emission pattern.

---

## 7. Milestone 1.5 — the reusable lemmas

Once the example builds, generalize the two hand-inlined facts so codegen for
arbitrary `n` doesn't re-derive them per problem:

- **`PounceLean/PSD.lean`** — from `hessian_psd` witness (`L`, `D`):
  `M = L · diag(D) · Lᵀ ∧ (∀ i, 0 ≤ D i) → ∀ v, 0 ≤ vᵀ M v`. The matrix identity
  is closed by `ring`/`norm_num`; nonnegativity follows from each `Dᵢ ≥ 0` and
  `sq_nonneg`. (Mathlib `Matrix`, `Matrix.PosSemidef` are the relevant pieces.)
- **`PounceLean/ConvexQP.lean`** — the general theorem: objective `½xᵀQx + cᵀx`
  with `Q ⪰ 0`, linear constraints, a KKT point `(x*, λ)` with feasibility +
  dual sign + complementarity ⟹ `x*` minimizes `f` over the feasible set. The
  per-problem `.lean` then just instantiates this with the cert's data.

---

## 8. Milestone 2 — the codegen (`cert.json → .lean`)

Reads a `pounce.lean-cert/v1` certificate and emits the `.lean`. Decision to make
(was an open question; recommend **(b)**):

- (a) write it in Lean itself (keeps everything one toolchain), or
- (b) write it in a scripting language (Python/Rust) — simpler string templating,
  no Lean metaprogramming. **Recommended** for v1.

Validation = run it on `certs/qp.cert.json` and confirm it reproduces the
**already-validated** `PounceLean/Examples/QP.lean` (golden-file diff), and that
the regenerated file still `lake build`s. Store `(cert.json, expected .lean)`
pairs under `test/fixtures/` as the regression suite.

Only after this does it make sense for POUNCE to build its `pounce certify`
emitter — against a schema now *proven* Lean-checkable.

---

## 9. Schema reference (`pounce.lean-cert/v1`) — essentials

Full field-level spec: `dev-notes/lean-cert-schema-v1.md` in the POUNCE repo. The
load-bearing rules:

- **No float anywhere.** Every number is `{ "num": "<int str>", "den": "<int str>" }`,
  reduced, `den > 0`. Big integers are decimal *strings* (JSON numbers overflow).
- **Bound sentinels** are strings `"-inf"` / `"+inf"`, not rationals.
- **Sparse matrix** = `{ "rows", "cols", "symmetric"?, "entries": [{i,j,val}] }`;
  symmetric stores the lower triangle.
- **`objective.half_quadratic`** flips the gradient: `true` ⇒ `f = ½xᵀQx + cᵀx + k`,
  `false` ⇒ `f = xᵀQx + cᵀx + k`. The codegen must honor it.
- **Constraints** are ranges `lower ≤ coeffs·x ≤ upper` (AMPL convention);
  equality is `lower == upper`; one-sided uses an `inf` sentinel.
- **Witnesses are untrusted** — bad data ⇒ proof fails, never falsely passes.
- **Versioning:** adding fields is non-breaking (tolerate unknowns); rename/remove
  bumps the major; never silently change a field's meaning. Pin on
  `schema starts_with "pounce.lean-cert/v1"`.

`nlp-poly` / `sos-poly` will later add an expression-tree problem encoding and
`sos: [{gram, multiplier_poly}]` witnesses — additive, not breaking.

---

## 10. CI (when the repo is real)

- `lake exe cache get && lake build` (the build *is* the test for proofs).
- A check that every `test/fixtures/*.cert.json` regenerates its expected `.lean`
  and that file builds — this is the **drift guard** against the POUNCE-side
  schema. Mirror of POUNCE's `scripts/check-release-consistency.sh` philosophy.
- Pin the GitHub Actions runner to the `lean-toolchain`; cache `~/.cache/mathlib`.

---

## 11. Milestone checklist

- [ ] **M1** — `lake new … math`, `lake exe cache get`, scaffold builds.
- [ ] **M1** — `Examples/QP.lean` (hand-written) `lake build`s clean; tactic
      invocation for `global_min` locked in. *(De-risks the whole design.)*
- [ ] Record pinned `lean` + `mathlib` versions; update cert `toolchain`.
- [ ] **M1.5** — `PSD.lean` + `ConvexQP.lean` reusable lemmas; `QP.lean`
      refactored to instantiate them.
- [ ] **M2** — codegen reproduces `QP.lean` from `qp.cert.json` (golden diff);
      fixtures + CI drift guard in place.
- [ ] Hand off: POUNCE builds `pounce certify` against the validated schema.

---

## 12. Open questions to settle as you go

1. **Tier-1 equality constraints:** declared-tolerance ε (exact arithmetic, ε
   stated — shippable now) vs. interval-Newton existence proof (rigorous, a real
   formalization effort). Decides what "verified feasible" *means*.
2. **Consumer ergonomics:** is requiring `lake build` acceptable, or also ship a
   prebuilt/attested verdict? (The latter reintroduces a trust-the-attester
   problem the proof was meant to remove.)
3. **Repo name:** `pounce-lean` vs `pounce-cert`.
