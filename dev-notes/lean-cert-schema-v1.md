# POUNCE Lean-certificate schema, v1 (DRAFT)

**Schema tag:** `pounce.lean-cert/v1`

> **Status: DRAFT / design-stage.** This schema is the contract between the
> POUNCE-side emitter (`pounce certify`, not yet implemented) and the external
> `pounce-lean` verification repo (not yet created). It lives in `dev-notes/`
> (not the published `docs/src/` book) precisely because the feature is not
> implemented yet — it will graduate to `docs/src/schema/lean-cert-v1.md` and be
> wired into the user-facing nav when `pounce certify` ships. Design rationale:
> `dev-notes/lean-certificate.md`.

This document defines the JSON certificate POUNCE emits for a solve so that
Lean 4 can independently produce a kernel-checked proof that the returned `x*`
is feasible and (where tractable) a minimum, over **exact rational
arithmetic** — no floating point in the trusted path.

## Producer / consumer split

* **POUNCE emits** this certificate: the problem encoded over ℚ, the candidate
  `x*`, and **untrusted witnesses** (duals, matrix factorizations, SOS data).
* **`pounce-lean` consumes** it: generates a `.lean` statement + proof, then
  `lake build` checks it against the Lean kernel.

The witnesses do **not** need to be trusted: wrong witness data makes the proof
fail to typecheck, never pass falsely. POUNCE can be adversarial and forge
nothing. See `dev-notes/lean-certificate.md` § Trust boundaries.

`statement_sha256` and any signature are **not** in this certificate — they
belong to the *verification receipt* produced after codegen, because the
statement is derived by `pounce-lean`, not by POUNCE.

## Versioning policy

Identical to the solve-report schema (`docs/src/schema/solve-report-v1.md`):
adding fields is non-breaking and consumers MUST tolerate unknown fields;
removing/renaming bumps the major (`v1` → `v2`); changing a field's semantics
without a rename is forbidden. Pin against the major (`schema` starts with
`pounce.lean-cert/v1`).

## Exact-rational encoding

There is **no float anywhere** in the certificate. Every numeric quantity is a
**rational** object:

```json
{ "num": "-7", "den": "2" }      // = -7/2
```

* `num`, `den` are **decimal integer strings** (arbitrary precision; JSON
  numbers cannot safely hold big integers).
* `den` > 0; the fraction is reduced (`gcd(|num|,den) = 1`); `0` is
  `{"num":"0","den":"1"}`.
* Because every f64 is exactly a dyadic rational `m·2^e`, the conversion from
  the solver's `x*`/`λ` is **lossless** — `den` is a power of two for values
  that came straight from f64, but the schema does not require that.

Bound sentinels use a string instead of a rational:

```json
"lower": "-inf"      // no lower bound;  "upper": "+inf"  → no upper bound
```

A **rational vector** is `[ rational, ... ]`. A **sparse rational matrix** is a
triplet list, each entry `{ "i": <row>, "j": <col>, "val": rational }`, plus an
explicit `rows`/`cols`. Symmetric matrices (e.g. a Hessian) store the lower
triangle only and set `"symmetric": true`.

## Top-level shape

```json
{
  "schema": "pounce.lean-cert/v1",
  "verdict": "global-min",
  "problem_class": "qp-convex",
  "tolerance": { "num": "0", "den": "1" },
  "binding": {
    "nl_sha256": "…64 hex…",
    "sol_sha256": "…64 hex…",
    "solver": "pounce 0.5.0"
  },
  "toolchain": {
    "lean": "leanprover/lean4:v4.x.0",
    "mathlib": "<git rev>"
  },
  "problem": { … },
  "candidate": { … },
  "witnesses": { … }
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `"pounce.lean-cert/v1"`. |
| `verdict` | enum | The single claim to be proven: `"feasible"`, `"local-min-strict"`, or `"global-min"`. |
| `problem_class` | enum | `"qp-convex"`, `"lp"`, `"nlp-poly"`, `"sos-poly"`, … — tells the codegen which proof template to instantiate. |
| `tolerance` | rational | ε for feasibility (`0` when feasibility is proven exactly, e.g. only linear/affine equalities). See § Tolerance. |
| `binding` | object | `nl_sha256`, `sol_sha256` (content-address the canonical problem and the claimed solution, exactly as `pounce verify` does), and the producing `solver`. |
| `toolchain` | object | Lean toolchain + Mathlib revision the certificate was authored against (reproducibility; not load-bearing for trust). |
| `problem` | object | The problem over ℚ — see § Problem encoding. |
| `candidate` | object | `x*` (and, when relevant, `objective`) over ℚ. |
| `witnesses` | object | Untrusted proof hints — see § Witnesses. |

## Problem encoding

For the v1 problem classes (`lp`, `qp-convex`), objective and constraints are at
most quadratic / linear and need no general expression tree:

```json
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
  "var_bounds": {
    "lower": [ "-inf", "-inf" ],
    "upper": [ "+inf", "+inf" ]
  },
  "constraints": [
    {
      "name": "c0",
      "coeffs": [ {"num":"1","den":"1"}, {"num":"1","den":"1"} ],
      "lower": { "num": "1", "den": "1" },
      "upper": "+inf"
    }
  ]
}
```

* `objective.kind` ∈ `{ "linear", "quadratic" }`. When `half_quadratic` is true
  (the POUNCE/QP convention), `f(x) = ½·xᵀQx + cᵀx + constant`; when false,
  `f(x) = xᵀQx + cᵀx + constant`. The codegen must honor this flag — it changes
  the KKT gradient.
* `var_bounds.lower/upper` are length-`n_vars` arrays of rationals or
  `"-inf"`/`"+inf"`.
* Each `constraints[k]` is a linear row `coeffs · x` with a range
  `lower ≤ coeffs·x ≤ upper` (matching the AMPL `g_l ≤ g(x) ≤ g_u` convention).
  An equality is `lower == upper`; a one-sided inequality uses an `inf` sentinel.

> **Forward compatibility.** `nlp-poly` / `sos-poly` will replace the quadratic
> objective and linear `constraints` with an expression-tree form
> (`{"op":"add","args":[…]}` over ℚ literals, variables, `pow`, `mul`). That is
> a v1 *addition* (new `kind`/`problem_class`), not a breaking change.

## Candidate

```json
"candidate": {
  "x": [ {"num":"1","den":"2"}, {"num":"1","den":"2"} ],
  "objective": { "num": "1", "den": "2" }
}
```

`x` is the candidate `x*` over ℚ (length `n_vars`). `objective` is `f(x*)`,
recomputed exactly by POUNCE and re-derived by Lean (informational; the proof
does not trust it).

## Witnesses (untrusted)

Per `verdict` / `problem_class`. For `global-min` on `qp-convex`:

```json
"witnesses": {
  "duals": [ {"num":"1","den":"1"} ],
  "hessian_psd": {
    "of": "Q",
    "L": { "rows": 2, "cols": 2, "unit_lower": true, "entries": [] },
    "D": [ {"num":"2","den":"1"}, {"num":"2","den":"1"} ]
  },
  "active_set": [ 0 ]
}
```

| Witness | Used by | Lean checks |
|---|---|---|
| `duals` | KKT stationarity | `∇f(x*) − Σλₖ·∇gₖ` is absorbed by bound multipliers (exact). One per constraint. |
| `hessian_psd` | convexity ⟹ global | `LDLᵀ` of the Lagrangian Hessian: identity `M = L·diag(D)·Lᵀ` (`ring`/`norm_num`) **and** `Dᵢ ≥ 0` entrywise. For a convex QP the Lagrangian Hessian is the constant `Q`. `unit_lower` `L` omits its implied unit diagonal. |
| `active_set` | complementarity | Indices of constraints treated as active (informational; the proof derives feasibility + complementarity directly). |

`local-min-strict` uses the same shape but `hessian_psd.of` is the **reduced**
Lagrangian Hessian on the active-constraint null space, with `D` strictly
positive. `sos-poly` adds `sos: [ { gram, multiplier_poly } … ]` — Gram matrices
(PSD, same `LDLᵀ` form) and the multiplier polynomials of the Positivstellensatz
identity.

## Tolerance

`tolerance` ε is the feasibility slack the proof is allowed for *equality*
constraints that a rational point cannot satisfy exactly:

* ε = `0` when feasibility is exact — all equalities are affine and `x*` lands on
  them exactly, or there are only inequalities with rational slack. The QP
  example below is exact.
* ε > `0` for nonlinear equalities: Lean proves `|gₖ(x*) − rhs| ≤ ε` *exactly*
  over ℚ, with ε named in the certificate. This is honest and exact arithmetic —
  strictly stronger than f64 fuzz — even though it is not a zero-residual proof.
* The rigorous interval-Newton existence variant (a *true* zero provably near
  `x*`) is a future schema addition, not part of v1.

## Worked example

A complete `qp-convex` / `global-min` certificate and the `.lean` it should
generate live in `dev-notes/lean-cert-example/` (`qp.cert.json`, `qp.lean`,
`README.md`). That is the end-to-end reference the `pounce-lean` codegen targets.
