# POUNCE Lean-certificate schema, v1

**Schema tag:** `pounce.lean-cert/v1`

This document is the canonical reference for the exact-rational certificate
emitted by `pounce certify <problem.nl> <claim.sol>`. For how to *use* the
command — what it proves, and how a certificate is accepted — start with
[Certifying Solutions (Lean)](../certify.md); this page is the field-by-field
format specification. The certificate lets the
external `pounce-lean` repository (not yet public) produce a **kernel-checked
Lean 4 proof** of what the solve actually established — that `x*` is feasible
and a **global** minimizer, that the problem is infeasible or unbounded, or that
a nonconvex objective is bounded below by an exact `γ` — over rational
arithmetic, with no floating point in the trusted path.

Implementation: the serde structs and the exact-rational emitter live in
[`crates/pounce-lean-cert/src/`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-lean-cert/src/lib.rs)
(`schema.rs`, `rational.rs`, `ldlt.rs`, `refine.rs`, `emit.rs`);
[`crates/pounce-cli/src/certify.rs`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-cli/src/certify.rs)
wires it to the CLI.

> **Status.** Every verdict the consumer accepts is now **validated
> end-to-end** — `global-min` (from `qp-convex` via KKT, *and* from `sos-poly`
> via an attained bound), `infeasible`, `unbounded`, `global-lower-bound`
> (`sos-poly`), `local-lower-bound` and `local-min` (`sos-poly` restricted to a
> ball), and `feasible`: `pounce certify` emits each, and `pounce-lean`
> kernel-checks it (reusable lemmas → codegen → `lake build`) with proofs
> resting only on Lean's standard axioms (`propext`, `Classical.choice`,
> `Quot.sound`; no `sorry`). Each has a fixture in
> `crates/pounce-cli/tests/fixtures/` that `scripts/check-lean-cert.sh`
> regenerates, rebuilds, and audits.

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
> | `global-min` (KKT + PSD Hessian) | yes | yes |
> | `global-min` (SOS bound + attaining point) | yes | yes |
> | `infeasible` (Farkas witness `witnesses.farkas.y`) | yes | yes |
> | `unbounded` (recession witness `witnesses.recession`) | yes | yes |
> | `global-lower-bound` (SOS witness `witnesses.sos`) | yes | yes |
> | either SOS verdict with Putinar multipliers (`problem.poly_constraints`) | yes | yes |
> | `feasible` (ε-feasibility, and existence via `witnesses.feasible_witness.xhat`) | yes (`--feasible`) | yes |
> | `local-min` / `local-lower-bound` (Putinar plus a ball, `problem.neighborhood`) | yes (`--local`) | yes |
> | `local-min-strict` | no | no |
>
> Local optimality closed **without** the second-order theory that row had been
> waiting on. The plan had been KKT plus a reduced-Hessian `LDLᵀ`, which needs
> an SOSC theorem Mathlib does not have. But a neighborhood is a polynomial
> inequality — `r² − ‖x − c‖² ≥ 0` — so adjoining it to the Putinar family
> proves minimality over the ball using machinery that already existed and a
> sign argument two lines long. No constraint qualification, no Taylor
> remainder, no implicit function theorem.
>
> `local-min-strict` remains open, and is now a genuinely smaller gap: what is
> missing is only *strictness*, since nothing above rules out a tie elsewhere in
> the ball.
>
> `feasible` was the last row where the consumer ran ahead of the producer, and
> it closed differently from the others — it is the one verdict the emitter will
> not choose on its own. `pounce certify --feasible` is an explicit request,
> because a failed optimality certificate is a result worth seeing rather than
> one to quietly replace with a weaker claim.
>
> None of the closed rows was a field copy, and it is easy to assume otherwise.
> `QpStatus::PrimalInfeasible` is a unit variant carrying no payload, and the
> float Farkas ray satisfies `Aᵀy = 0` only to a *relative* tolerance: on the
> `certify_infeasible` fixture `‖y‖ ≈ 2.3e10` with a ~1.7e-11 relative residual,
> which converted losslessly to ℚ is `−103801/262144` — not zero, so the Lean
> hypothesis would be undischargeable. Each verdict therefore needed a
> refinement step that treats the float object as a *hint* for the exact one's
> support: `refine_kkt` for `global-min`, `refine_farkas` for `infeasible`
> (which recovers `y = (1,1,1)` on that fixture), `round_gram` for the SOS
> Gram matrix, and `refine_recession` for `unbounded` with a nonzero `Q`.
> Float proposes; exact arithmetic decides.
>
> The `unbounded` row is worth its own sentence, because for a while it looked
> like the exception. An LP recession direction needs `A d ≥ 0` and `c·d < 0` —
> both **inequalities**, and an inequality satisfied with margin survives the
> f64→ℚ conversion with its sign intact, so the solver's diverging iterate was
> already an exact witness, verbatim. A nonzero `Q` adds `Q d = 0`, and that is
> an equality again. So the split was never LP-vs-QP: it is
> equality-vs-inequality, which is the same distinction that made every other
> row need refinement.

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
| `verdict` | enum | The single proven claim: `"global-min"`, `"feasible"`, `"infeasible"`, `"unbounded"`, `"global-lower-bound"`, `"local-min"`, or `"local-lower-bound"`. `global-min` is reached by two different theorems; `problem_class` says which. The `local-` pair is the SOS pair restricted to `problem.neighborhood`, and the two must agree — a `global-` verdict carrying a neighborhood is refused. |
| `problem_class` | enum | `"qp-convex"` or `"sos-poly"`. A **shape** discriminator — which half of `problem` is populated, and so which theorem the codegen routes to — *not* a convexity claim. `feasible` ships `qp-convex` on an indefinite `Q`, and so does `unbounded`, because neither verdict needs convexity. Only `global-min` on the KKT path does, and there the PSD claim is carried by `witnesses.hessian_psd` and checked by Lean. |
| `tolerance` | rational | Feasibility ε. `0` for every verdict except `feasible`, whose whole subject is a point that misses feasibility — there it is an exact rational bound on the residual, computed over ℚ and rounded **up** to one significant digit, never a solver setting copied across. |
| `bound` | rational | `sos-poly` only: the γ proven to satisfy `γ ≤ p(x)` — for all `x`, or, when `problem.poly_constraints` is present, for all *feasible* `x`. Present under both SOS verdicts — when the verdict is `global-min` it equals `candidate.objective`, and that equality is the whole difference. |
| `binding` | object | `nl_sha256`, `sol_sha256` (content-address the canonical problem and claimed solution, exactly as `pounce verify` does), and the producing `solver`. |
| `toolchain` | object | The Lean toolchain + Mathlib revision the cert is authored against (a proof reproduces only under the same pin). |
| `problem` | object | The problem over ℚ — see below. |
| `candidate` | object | `x*` and its objective over ℚ. **Absent** whenever the verdict names no point (`infeasible`, `unbounded`, `global-lower-bound`) — the key is omitted, not nulled, so a consumer cannot mistake a bound for a claimed minimizer. For `feasible` it is the solver's float verbatim, not a refined point: that point is what the verdict is *about*. |
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
| `farkas.y` | `infeasible` | `y ≥ 0`, `Aᵀ y = 0`, `bᵀ y > 0` — refined exactly, since the middle condition is an equality. |
| `recession.x0`, `recession.d` | `unbounded` | `x0` feasible, and `Q d = 0`, `A d ≥ 0`, `c·d < 0`. **Both** are required: a direction alone cannot distinguish an unbounded problem from an empty feasible set, where such a `d` also exists and there is nothing to travel from. `d` is normalized so its largest-magnitude entry is `±1` (all three conditions are homogeneous, so the scale is free). |
| `feasible_witness.xhat` | `feasible` | exactly feasible, and within `tolerance` of `candidate.x` in ∞-norm. |

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

## The SOS slice: `global-lower-bound` and `global-min`

A nonconvex polynomial has no KKT-based global argument — a local solve returns
one basin and can say nothing about the others. What it *can* support is a
**bound**, via a sum-of-squares identity: exhibit a monomial basis `m(x)`, a
rational `γ`, and a PSD Gram matrix `G` with

    p(x) − γ = m(x)ᵀ G m(x)

Since the right side is a nonnegative quadratic form, `γ ≤ p(x)` for **every**
real `x`. The certificate carries a polynomial objective instead of a quadratic
one, and the bound in place of a candidate:

```json
"problem": {
  "n_vars": 1,
  "polynomial": { "terms": [ { "exponents": [0], "coeff": {"num":"2","den":"1"} },
                             { "exponents": [2], "coeff": {"num":"-2","den":"1"} },
                             { "exponents": [4], "coeff": {"num":"1","den":"1"} } ] }
},
"bound": {"num":"1","den":"1"},
"witnesses": {
  "sos": [ { "monomials": [[0],[1],[2]],
             "gram": { "rows": 3, "cols": 3, "symmetric": true, "entries": [ … ] },
             "L": { "rows": 3, "cols": 3, "unit_lower": true, "entries": [ … ] },
             "D": [ {"num":"1","den":"1"}, {"num":"0","den":"1"}, {"num":"0","den":"1"} ] } ]
}
```

Each `monomials[k]` is an **exponent vector** of length `n_vars`, matching
`polynomial.terms[*].exponents`; `witnesses.sos[b]` is one PSD block. There is
no `objective`, `var_bounds`, or `constraints` key — the keys are absent,
because the QP encoding does not apply and the objective is the polynomial.
(A constrained problem carries its feasible set in `problem.poly_constraints`
instead; see below.) Lean discharges two independent obligations, and both must
hold:

| Witness | Lean checks |
|---|---|
| `gram` + `monomials` | the identity `p − γ = m(x)ᵀ G m(x)`, closed by `ring` — a polynomial identity, so no `n²` case split |
| `L`, `D` | `G ⪰ 0` via the exact `LDLᵀ`: `G = L·diag(D)·Lᵀ` **and** `Dᵢ ≥ 0` entrywise |

Two obligations means two ways to forge, so the drift guard carries two negative
fixtures: `certify_sos_forged_bound` inflates γ (the identity stops closing) and
`certify_sos_forged_psd` swaps in an indefinite `G` that still satisfies the
identity (the nonnegativity goal fails). Both must fail `lake build`.

### When the bound is attained: `global-min`

A bound plus a point that *reaches* it is a global minimum. If some `x₀` has
`p(x₀) = γ`, then `γ ≤ p(x)` everywhere makes `x₀` a global minimizer — of a
polynomial that need not be convex, where no KKT argument applies at all. The
certificate then carries a `candidate` alongside the `bound`, the verdict
becomes `global-min`, and Lean discharges one extra equation:

| Witness | Lean checks |
|---|---|
| `candidate.x` | `p(x₀) = γ`, by `norm_num` over ℚ |

The emitter looks for such an `x₀` by snapping the local solve's `x*` to a short
ladder of rational grids and evaluating `p` **exactly**; equality is required,
never closeness. So the two verdicts differ by whether a rational minimizer
exists and was found, and the fixtures are a matched pair:

| Fixture | Polynomial | Minimizer | Verdict |
|---|---|---|---|
| `certify_sos` | `x⁴ − 2x² + 2` | `±1` | `global-min`, γ = 1 |
| `certify_sos_bound` | `x⁴ − 3x² + 2` | `±√(3/2)` | `global-lower-bound`, γ = −1/4 |

The second is not a shortcoming to be fixed: a certificate over ℚ cannot exhibit
an irrational minimizer, and `1.2247…` is not one. Publishing the bound alone is
the correct answer, and it is still a global statement about every real `x`.

Attainment is a third independent obligation, so it gets a third negative
fixture: `certify_sos_forged_candidate` keeps the genuine bound, Gram, and
identity and corrupts only the exhibited point. Everything else about it checks
out, which is exactly why the `p(x₀) = γ` goal has to be the thing that catches
it — and it must fail `lake build`.

The SDP that finds `G` runs in f64 and returns a bound like `1 − 1e-9`, which is
not a rational the identity can close on. `round_gram` searches a small ladder
of candidate `γ` values and denominator grids, solving the coefficient-matching
system exactly over ℚ at each; the emitter then factors the result, serializes
it, and **re-derives both obligations from the serialized sparse form** before
writing. A polynomial that is nonnegative but not SOS (Motzkin's
`x⁴y² + x²y⁴ − 3x²y² + 1` at its tight bound) is refused rather than
approximated.

### Constraints: `problem.poly_constraints` and Putinar multipliers

An objective can be unbounded below on ℝⁿ and still have a perfectly good bound
on its feasible set — `x³ − 3x` on `[0, 3]` is the small example. There, no γ
makes `p − γ` a sum of squares, so the identity above does not exist at all.
What does exist is a **Putinar** identity, one sum-of-squares multiplier per
constraint:

    p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x),   feasible set: gₖ(x) ≥ 0

The certificate then carries the feasible set alongside the objective, and each
localizing block names the constraint it multiplies:

```json
"problem": {
  "n_vars": 1,
  "polynomial":       { "terms": [ … ] },
  "poly_constraints": [ { "terms": [ … ] },     // g₀ = x
                        { "terms": [ … ] } ]    // g₁ = 3 − x
},
"witnesses": {
  "sos": [ { "monomials": [[0],[1],[2]], "gram": …, "L": …, "D": … },
           { "multiplier": 0, "monomials": [[0],[1]], "gram": …, "L": …, "D": … },
           { "multiplier": 1, "monomials": [[0],[1]], "gram": …, "L": …, "D": … } ]
}
```

Each entry of `poly_constraints` is a `gₖ(x) ≥ 0` term list in the same
encoding as `polynomial`. Finite **variable bounds** are folded in as ordinary
rows (`xⱼ − lⱼ ≥ 0`, `uⱼ − xⱼ ≥ 0`) rather than a separate `var_bounds` block —
which is precisely what lets a box give a bound where ℝⁿ gives none.

`witnesses.sos[b].multiplier` is the **index into `poly_constraints`** of the
constraint that block multiplies. It is **absent** for `σ₀`, whose multiplier is
the constant `1`. So the unconstrained shape is not a special case in the
machinery — it is the Putinar shape with exactly one block and no constraints —
but it *is* a different claim, and the schema keeps the two apart in the one way
that matters: the constraints are in the `problem`, not the witnesses, so
`cert-verify` re-derives them from the consumer's `.nl` like everything else.

The codegen routes on **`poly_constraints` being present**, not on the verdict
(`global-lower-bound` and `global-min` occur in both shapes). A consumer that
ignored the field would read a bound on a box as a bound on all of ℝⁿ — the one
misreading this schema must make impossible. Every constraint must have exactly
one localizing block and no block may claim two, or the certificate is refused.

PSD-ness is checked **per block**, and the joint identity ties the blocks
together — the linear system matching `p − γ` runs over all blocks at once,
which is why a bad block cannot be compensated for elsewhere without being
caught. Two negative fixtures cover what the unconstrained ones cannot:
`certify_sos_box_forged_localizing` moves mass into σ₀ so the identity still
closes exactly while a *localizing* block goes indefinite, and
`certify_sos_box_forged_infeasible_min` exhibits `x = −2`, the other root of
`p − γ`: genuine bound, genuine Gram, exact attainment, simply not in `[0, 3]`.

That second one is the new obligation the constrained path brings. On the
unconstrained slice attainment alone makes `x₀` a minimizer; here a point
outside the feasible set can beat every point inside it, so the generated
theorem proves `gₖ(x₀) ≥ 0` for every `k` *and* `p(x₀) = γ`, and states both:

    (∀ i, 0 ≤ g i xstar) ∧ ∀ x, (∀ i, 0 ≤ g i x) → p xstar ≤ p x

**Equality constraints are refused.** `h(x) = 0` needs a sign-unrestricted
multiplier, not an SOS one, and splitting it into `h ≥ 0` and `−h ≥ 0` leaves
the feasible set with empty interior — where Putinar's theorem stops
guaranteeing a certificate exists at any degree. Refusing beats searching a
relaxation ladder that may have nothing on it.

### `problem.neighborhood`: local optimality, at no extra theory

At a local minimum that is not global, every claim above is *false*, so no
certificate for it exists at any relaxation order. `pounce certify --local`
narrows the claim to a ball instead:

```json
"problem": {
  "n_vars": 1,
  "polynomial":   { "terms": [ … ] },
  "neighborhood": { "center": [ {"num":"1","den":"1"} ],
                    "radius_sq": {"num":"1","den":"1"} }
}
```

This needs no new machinery, because a ball is a polynomial inequality:

    r² − ‖x − c‖² ≥ 0

so it joins the `gₖ` family as one more multiplier and the identity becomes

    p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x) + σ_B(x)·(r² − ‖x − c‖²)

with the proved statement guarded by `sqdist x center ≤ rsq` alongside
feasibility. That is local minimality stated directly — no second-order
sufficient conditions, no constraint qualification, no Taylor remainder.

Four things the encoding is deliberate about:

* **`radius_sq`, not `radius`.** ℚ is not closed under square roots, so a
  radius of `√2` has no exact representation while `r² = 2` does. The theorem
  mentions the square throughout and never takes a root. A non-positive
  `radius_sq` is refused by both emitter and codegen: the claim would be
  vacuous, and a module that builds and means nothing is worse than an error.
* **Structural, not a `poly_constraints` entry.** Storing the ball as a term
  list would put it in the certificate twice in two spellings nothing forces to
  agree, and would blur `poly_constraints`, whose honest meaning is *the
  problem's* feasible set — the ball is the certificate's own choice, not
  something the `.nl` says. Its `multiplier` index is one past the last
  constraint, so the multiplier family reads: constraints, then the ball.
* **`cert-verify` carries it across rather than re-deriving it.** Everything
  else in `problem` is rebuilt from the consumer's `.nl` and compared; the
  neighborhood cannot be, because no `.nl` determines it. It is copied
  unchanged, which is what makes the comparison of the rest exact.
* **The candidate needs a third obligation.** `p(x₀) = γ` does not put `x₀` in
  the ball: `x⁴ − 2x² + 2` attains its minimum at both `±1`. So the generated
  theorem proves `‖x₀ − c‖² ≤ r²` too, and states it:

      ((∀ i, 0 ≤ g i xstar) ∧ sqdist xstar center ≤ rsq) ∧
        ∀ x, (∀ i, 0 ≤ g i x) → sqdist x center ≤ rsq → p xstar ≤ p x

  `certify_sos_local_forged_outside_ball` is exactly that forgery — every
  witness genuine, `x₀ = −1` attaining exactly, two units outside a ball of
  radius one — and `certify_sos_local_forged_ball_psd` rewrites `σ_B` over a
  longer monomial basis so it still evaluates correctly while the Gram goes
  indefinite. Both must fail `lake build`, and `scripts/check-lean-cert.sh`
  checks that they do.

The centre is `x*` snapped to a coarse rational grid chosen to keep the true
solution well inside the ball — a certificate is about the ball it names, so
that ball must be one a rational centre describes exactly. The verdicts are
`local-min` and `local-lower-bound`, and the codegen refuses a certificate whose
verdict and shape disagree (a `global-min` carrying a neighborhood would
overclaim on read).

Routing between the QP and polynomial slices is on **degree**, not on
QP-extraction failure — a convex QP must never silently downgrade from
`global-min` to a mere bound. The degree test is on the *objective*, so a linear
objective under polynomial constraints stays on the QP path, where it is a
linear program and nothing about SOS improves on that.

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
certificate — except that a higher-degree polynomial objective is routed to the
SOS slice above instead of refused (with its polynomial inequality constraints,
if any, becoming Putinar multipliers), that `--feasible` opts into the weaker
`feasible` verdict, which needs no convexity (indefinite `Q` is fine — it
certifies only where the point sits, not that it is optimal), and that
`unbounded` likewise accepts any `Q` (it needs `Q` flat along `d`, never convex)
as long as the constraints are all inequalities, and that `--local` narrows the
SOS claim to a ball (`local-min` / `local-lower-bound`), which is what makes a
certificate possible at a local minimum that is not global.
Maximize objectives, *equality* constraints on a nonconvex polynomial, and the
*strict* local verdict `local-min-strict` remain additive future work.

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
