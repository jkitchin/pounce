# POUNCE Lean-certificate schema, v1

**Schema tag:** `pounce.lean-cert/v1`

This document is the canonical reference for the exact-rational certificate
emitted by `pounce certify <problem.nl> <claim.sol>`. The certificate lets the
external `pounce-lean` repository (not yet public) produce a **kernel-checked
Lean 4 proof** that the returned `x*` is feasible and a **global** minimizer —
over exact rational arithmetic, with no floating point in the trusted path.

Implementation: the serde structs and the exact-rational emitter live in
[`crates/pounce-lean-cert/src/`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-lean-cert/src/lib.rs)
(`schema.rs`, `rational.rs`, `ldlt.rs`, `refine.rs`, `emit.rs`);
[`crates/pounce-cli/src/certify.rs`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-cli/src/certify.rs)
wires it to the CLI.

> **Status.** The `qp-convex` / `global-min` slice is **validated end-to-end**:
> `pounce certify` emits it, and `pounce-lean` kernel-checks it (reusable lemmas
> → codegen → `lake build`) with proofs resting only on Lean's standard axioms
> (`propext`, `Classical.choice`, `Quot.sound`; no `sorry`). Other verdicts and
> problem classes are additive future work.

## Two documents, two audiences — and one asymmetry

The schema is specified in two places, deliberately:

| Document | Audience | Scope |
|---|---|---|
| **this file** | the producer | what `pounce certify` emits, and why |
| `pounce-lean/docs/lean-cert-v1.md` (not yet public) | the consumer | what `codegen/gen_lean.py` *accepts*, and the theorems each field discharges |

They are not copies and neither is redundant. The consumer document is
authoritative for what will verify, because it describes the code that actually
reads the certificate.

> **The consumer accepts strictly more than the producer emits.** This is the one
> place the two sides do not line up, and it is load-bearing enough to state
> here:
>
> | Verdict | `pounce certify` emits | `pounce-lean` verifies |
> |---|---|---|
> | `global-min` | yes | yes |
> | `feasible` (ε-feasibility, and existence via `witnesses.feasible_witness.xhat`) | **no** | **yes** |
> | `infeasible` (Farkas witness `witnesses.farkas.y`) | **no** | **yes** |
> | `local-min-strict` | no | no |
>
> `emit.rs` hardcodes `verdict: "global-min"`, so **every `feasible` and
> `infeasible` certificate in existence is hand-written.** The consumer-side machinery for them
> (`candidate_eps_feasible`, `feasible_point_exists`, and the
> `feasible_witness` field) is implemented and tested there, and is described in
> § 9 of the consumer document — but nothing in POUNCE can produce input for it
> yet. Teaching the emitter to emit `feasible` is tracked as planned work; until
> then, treat § 9 of the consumer document as a specification the producer has
> not met.
>
> The `infeasible` case is cheap **to check** but not free **to emit**, and the
> difference is easy to get backwards. `QpStatus::PrimalInfeasible` is a unit
> variant carrying no payload; the Farkas certificate is the diverging dual
> iterate, which reaches a consumer as ordinary `.sol` duals.
>
> More importantly, that ray satisfies `Aᵀy = 0` only to a *relative*
> tolerance. On the `certify_infeasible` fixture `‖y‖ ≈ 2.3e10` with a residual
> of ~1.7e-11 relative — but converted losslessly to ℚ that residual is
> `−103801/262144`, which is not zero, so the Lean hypothesis is
> undischargeable. Copying the solver's duals would produce a certificate that
> can never verify.
>
> So emitting `infeasible` needs a `refine_farkas` analogous to `refine_kkt`:
> take the float ray as a hint for the certificate's support, then solve for an
> exact rational ray (here, `y = (1,1,1)`). New code, not a field copy. What
> remains true is that *checking* the result is trivial — one nonnegative vector
> and a linear identity, with no factorization or KKT system.

## How it differs from `pounce verify`

[`pounce verify`](../verify.md) re-evaluates `g(x*)` in **f64** and makes its
receipt unforgeable via SHA-256 content-addressing plus an optional HMAC. It is
candid that global optimality is *not* checkable that way and that the HMAC is
only as strong as key secrecy. `pounce certify` attacks both: the proof is over
exact ℚ (no float fuzz), certifies a **global** minimum for convex QPs, and its
unforgeability is the **Lean kernel** — there is no key. The SHA-256 hashes
remain, doing a *different* job: binding the proof to the exact problem bytes.

## Producer / consumer split

* **POUNCE emits** this certificate: the problem over ℚ, the candidate `x*`, and
  **untrusted witnesses** (duals, the `LDLᵀ` factorization).
* **`pounce-lean` consumes** it: generates a `.lean` statement + proof, then
  `lake build` checks it against the Lean kernel.

The witnesses do **not** need to be trusted: wrong witness data makes the proof
fail to typecheck, never pass falsely. POUNCE can be adversarial and forge
nothing — the worst failure mode is a certificate that does not verify. To make
that failure mode rare in practice, the emitter **self-checks every witness
exactly over ℚ before writing**, and refuses (exit 2) rather than emit a cert
that will not verify.

`statement_sha256` and any signature are **not** in this certificate — they
belong to the *verification receipt* produced after codegen, because the
statement is derived by `pounce-lean`. The certificate carries `binding.nl_sha256`
and `binding.sol_sha256` only.

## Versioning policy

Identical to the [solve-report schema](solve-report-v1.md): adding fields is
non-breaking and consumers MUST tolerate unknown fields; removing/renaming bumps
the major (`v1` → `v2`); changing a field's semantics without a rename is
forbidden. Pin on `schema` **starts-with** `"pounce.lean-cert/v1"`.

## Exact-rational encoding

There is **no float anywhere** in the certificate. Every numeric quantity is a
rational object:

```json
{ "num": "-7", "den": "2" }      // = -7/2
```

* `num`, `den` are **decimal integer strings** (arbitrary precision; JSON
  numbers cannot safely hold big integers).
* `den > 0`; the fraction is reduced (`gcd(|num|, den) = 1`); `0` is
  `{"num":"0","den":"1"}`.
* Every finite f64 is exactly a dyadic rational `m·2^e`, so the conversion of
  the solver's `x*`/coefficients is **lossless** — POUNCE does not round.

Bound slots that may be infinite use a string sentinel instead of a rational:

```json
"lower": "-inf"      // "upper": "+inf"
```

A **sparse matrix** is `{ "rows", "cols", "symmetric"?, "unit_lower"?, "entries": [{i,j,val}] }`.
A `symmetric` matrix stores the **lower triangle** only; a `unit_lower` matrix
carries strictly-below-diagonal entries and omits its implied unit diagonal.

## Top-level shape

```json
{
  "schema": "pounce.lean-cert/v1",
  "verdict": "global-min",
  "problem_class": "qp-convex",
  "tolerance": { "num": "0", "den": "1" },
  "binding":   { "nl_sha256": "…64 hex…", "sol_sha256": "…64 hex…", "solver": "pounce 0.9.0" },
  "toolchain": { "lean": "leanprover/lean4:v4.31.0", "mathlib": "<git rev>" },
  "problem":   { … },
  "candidate": { … },
  "witnesses": { … }
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `"pounce.lean-cert/v1"`. |
| `verdict` | enum | The single proven claim. v1 codegen: `"global-min"`. |
| `problem_class` | enum | v1 codegen: `"qp-convex"`. |
| `tolerance` | rational | Feasibility ε. `0` for the exact QP slice. |
| `binding` | object | `nl_sha256`, `sol_sha256` (content-address the canonical problem and claimed solution, exactly as `pounce verify` does), and the producing `solver`. |
| `toolchain` | object | The Lean toolchain + Mathlib revision the cert is authored against (a proof reproduces only under the same pin). |
| `problem` | object | The problem over ℚ — see below. |
| `candidate` | object | `x*` and its objective over ℚ. |
| `witnesses` | object | Untrusted proof hints — see below. |

## Problem encoding

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
  "var_bounds": { "lower": ["-inf","-inf"], "upper": ["+inf","+inf"] },
  "constraints": [
    { "name": "c0",
      "coeffs": [ {"num":"1","den":"1"}, {"num":"1","den":"1"} ],
      "lower": {"num":"1","den":"1"}, "upper": "+inf" }
  ]
}
```

* `half_quadratic` flips the quadratic scale: `true` ⇒ `f = ½·xᵀQx + cᵀx + k`
  (POUNCE's convention), `false` ⇒ `f = xᵀQx + cᵀx + k`. The codegen folds the
  factor of 2 into `Q`/`D` so the KKT gradient is consistent.
* Each `constraints[k]` is a linear row meaning `lower ≤ coeffs·x ≤ upper`
  (AMPL convention); a one-sided inequality uses an `inf` sentinel.

## Witnesses (untrusted)

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
| `duals` | KKT stationarity | exactly **one per constraint**, in order; the nonnegative multiplier of the normalized `A x ≥ b` row. |
| `hessian_psd` | convexity ⟹ global | `LDLᵀ` of `Q`: the identity `Q = L·diag(D)·Lᵀ` (`ring`/`norm_num`) **and** `Dᵢ ≥ 0` entrywise. `unit_lower` `L` omits its implied unit diagonal. |
| `active_set` | complementarity | indices of constraints treated as active (informational). |

## What the witnesses must satisfy

The codegen normalizes constraints to `A x ≥ b` (a `lower ≤ a·x` row → `(a, lower)`;
an `a·x ≤ upper` row → `(−a, −upper)`) and applies the theorem *convex-QP KKT
point ⟹ global minimizer*. With `M` the Hessian-of-record (`= Q` if
`half_quadratic`, else `2Q`), the `(x*, λ)` in the certificate satisfy, **exactly
over ℚ**:

* **stationarity** `M x* + c = Aᵀ λ`
* **dual feasibility** `λ ≥ 0`
* **complementarity** `λᵢ · ((A x*)ᵢ − bᵢ) = 0`

POUNCE solves in f64, so the float `x̃` is feasible/stationary only approximately
and is *not* the exact optimizer. `pounce certify` therefore performs an **exact
rational active-set refinement**: it takes the float active set, solves the KKT
system exactly over ℚ for the true rational `(x*, λ)`, and verifies dual
feasibility and that the inactive rows hold — refusing if the guess was wrong.

## Why there is no bound-multiplier witness

The schema carries no `z_L` / `z_U`. This is deliberate and has been re-examined
three times; recording the reasoning here so it is not re-litigated a fourth.

A variable bound `xᵢ ≥ lᵢ` **is** the linear row `eᵢᵀx ≥ lᵢ`. The emitter folds
every finite bound into `constraints` (`var{i}_lb`, `var{i}_ub`, or an equality
`var{i}_fix` when fixed) and emits `var_bounds` as all-infinite sentinels. A
bound's multiplier is then an ordinary entry of `witnesses.duals`, and the Lean
theorem — stated over an arbitrary `A x ≥ b` — never learns the row came from a
bound.

The recurring doubt is whether this survives *nonlinear* constraints. It does:
bounds are linear rows unconditionally, whatever the rest of the problem looks
like. What genuinely does not survive is the encoding of the other rows
`gⱼ(x) ≥ 0`, which is the `nlp-poly` problem-encoding gap — a different issue
that has been mistaken for this one.

Second-order certificates do not need multipliers at all: the consumer takes
stationarity in the reduced form `Zᵀ(Hx* + c) = 0`, which mentions neither `λ`
nor `z_L`, `z_U`.

Pinned by `an_active_variable_bound_yields_an_ordinary_constraint_dual` in
`crates/pounce-lean-cert/tests/emit_reference.rs`, which certifies a problem
whose optimum lies *on* a bound (so the multiplier is provably nonzero) and
asserts no bound-multiplier field appears.

## `nullspace` — the `local-min-strict` witness (emitted, not yet consumed)

Tier 2 asserts positive-definiteness of the **reduced** Hessian `ZᵀHZ` on the
active-constraint null space. Witnesses are untrusted, so the consumer must
verify the reduction rather than take it on faith, and that needs `Z` plus two
pieces of auxiliary data.

```json
"nullspace": {
  "Z": { "rows": 3, "cols": 2, "entries": [ … ] },
  "identity_rows": [ 1, 2 ],
  "spanning": {
    "rows": [ 0 ],
    "cols": [ 0 ],
    "inverse": { "rows": 1, "cols": 1, "entries": [ … ] }
  }
}
```

| Field | Consumer checks | Establishes |
|---|---|---|
| `Z` | `A_active · Z = 0` | `range(Z) ⊆ ker(A_active)` |
| `identity_rows` | `Z[identity_rows, :] = I` | `Z` has full column rank, so `dim range(Z) = k` |
| `spanning` | `A_active[rows, cols] · inverse = I`, `\|rows\| = \|cols\| = n − k` | `rank(A_active) ≥ n − k`, so `dim ker(A_active) ≤ k` |

**All three are required, and the third is the one that is easy to miss.** The
first two are necessary and read naturally as sufficient. They are not: together
they bound the spanned dimension only from *below*. A `Z` spanning a strict
subspace of the null space passes both, and if the direction it omits carries
negative curvature, then `ZᵀHZ ≻ 0` holds at a genuine **saddle point** and the
verdict is wrong. Under-reporting the null space is the dangerous direction;
over-reporting merely fails the first check. Adding `spanning` closes it —
`dim ker ≤ k` with `dim range(Z) = k` forces `range(Z) = ker(A_active)` exactly.

This was not caught by inspection. It surfaced when the consumer-side Lean proof
of second-order sufficiency would not close without a spanning hypothesis.

Both rank facts could be had from a determinant or an elimination — precisely
the `O(n³)` rational-matrix decision procedure that is already too slow in the
kernel for dense Hessians. Both are instead reduced to one matrix product by
shipping a byproduct the emitter's elimination already computed. Neither field
adds trust: the consumer verifies each claim, and forged data fails.

Produced by `pounce-lean-cert`'s `nullspace` module, which runs all three checks
before emitting, so a certificate that would not verify is never written.

> **Status.** The producer computes this block; the codegen does **not** read it
> yet and `local-min-strict` still exits 2. Documented here so the shape is
> settled before anything depends on it — not as a live capability.

## Supported slice (v1)

`problem_class = qp-convex`, `verdict = global-min`, quadratic objective
(`half_quadratic` honored), linear constraints (one-sided, **two-sided ranges**,
or **equalities**), **variable bounds** (one-sided, box, or fixed), convex (PSD)
Hessian.

Each cert constraint is routed by its `lower`/`upper`, exactly as the codegen
re-derives it:

* **inequality** (one finite bound) → an `A x ≥ b` row with a multiplier `λ ≥ 0`
  (an `a·x ≤ u` row is normalized to `−a·x ≥ −u`);
* **equality** (`lower == upper`) → an `E x = d` row with a **free-sign**
  multiplier `μ`, discharged by the `global_min_of_kkt_eq` theorem; `x*` must
  satisfy it exactly over ℚ;
* a **two-sided range** (`lower ≠ upper`, both finite) is split by the emitter
  into two one-sided rows `{c}_lo` / `{c}_hi` (at most one active, so
  non-degenerate) — the cert never carries a two-sided row.

Variable bounds fold the same way: `xᵢ ≥ lᵢ` → `var{i}_lb`, `xᵢ ≤ uᵢ` →
`var{i}_ub`, a fixed `xᵢ = v` → an equality `var{i}_fix`. Consequently `var_bounds`
is always emitted as the infinite sentinels in v1; bounds live in `constraints`.

Outside this slice `pounce certify` **exits 2** rather than emit an unsound
certificate: non-quadratic objectives, indefinite `Q`, maximize objectives, and
the `feasible` / `local-min-strict` verdicts are additive future work.

## Consumer acceptance

A result is accepted **iff all three** hold:

1. **`pounce cert-verify <problem.nl> <cert.json>`** — re-derives the problem
   from the consumer's *own* `.nl` (the trusted, deterministic Frontend) and
   checks it equals `cert.problem`, plus `binding.nl_sha256` matches. This rules
   out a certificate that proves an *easier* problem under the real `.nl`'s hash
   — the hash binding alone is necessary but not sufficient.
2. **`lake build`** of the generated `.lean` succeeds under the consumer's *own*
   pinned Lean/Mathlib (not the cert's suggested toolchain).
3. **Axiom audit** — `#print axioms …global_min` lists only Lean's standard
   axioms `{propext, Classical.choice, Quot.sound}` and no `sorryAx`. `lake
   build` exits 0 even on a `sorry` (it only warns), so the exit code alone is
   not sufficient; the axiom set is the real gate.

## Drift guard

`scripts/check-lean-cert.sh` (run in CI) regenerates the golden certificate from
a committed `.nl`/`.sol` fixture and diffs it byte-for-byte — the emitter is
deterministic, so any change is real drift — then runs `cert-verify` to confirm
the cert binds to its `.nl`. With `POUNCE_LEAN_DIR` set it also diffs the golden
generated `.lean`, and with `LAKE_BUILD=1` it `lake build`s the proof **and runs
the axiom audit**. The heavy `lake build` proper lives in `pounce-lean`'s own CI,
keeping the Mathlib toolchain off POUNCE's critical path.

## Worked example

The committed fixture `crates/pounce-cli/tests/fixtures/certify_qp.{nl,sol}` is
the canonical convex QP

```
minimize    f(x) = x₁² + x₂²        (= ½ xᵀQx, Q = diag(2,2))
subject to  x₁ + x₂ ≥ 1
x* = (1/2, 1/2),  f(x*) = 1/2,  dual λ = 1,  tolerance = 0
```

`certify_qp.cert.json` is the emitted certificate and `certify_qp.expected.lean`
the proof `pounce-lean` generates from it.
