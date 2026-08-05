# Certifying Solutions (Lean)

```sh
pounce certify <problem.nl> <claim.sol> [OPTIONS]
pounce cert-verify <problem.nl> <cert.json>
```

[`pounce verify`](verify.md) re-evaluates the constraints in f64 and answers
*"is this point feasible?"* — and is candid that **global optimality is not
checkable that way**. `pounce certify` answers a different question: it emits
an exact-rational certificate that an external Lean 4 development turns into a
**machine-checked proof**, verified by the Lean kernel over ℚ with no floating
point anywhere in the trusted path.

The two are complements, not alternatives. `verify` runs anywhere, on any
problem, in milliseconds. `certify` covers a narrow slice of problems and needs
a Lean toolchain, but what it produces is a proof rather than a check.

> **Status.** The producer half — everything in this repository — is complete
> and tested for the slices below. The consumer half (`pounce-lean`: codegen
> plus the reusable Lean lemmas) **is not yet public**, so today you can emit
> and bind certificates, but you cannot run the final `lake build` step
> yourself. Certificates are plain JSON and stable under the `pounce.lean-cert/v1`
> schema tag, so they remain useful — and checkable later — regardless.

## Why a certificate instead of a check

Verification is only as good as what it *can't* be talked out of. Three
properties do the work here:

**The witnesses are untrusted.** A certificate carries proof hints — duals, a
Gram matrix, an `LDLᵀ` factorization — but nothing about them is believed.
Wrong witness data makes the generated Lean fail to typecheck; it can never
make a false statement pass. POUNCE could be adversarial and forge nothing.
The worst failure mode is a certificate that does not verify.

**There is no key.** `pounce verify` can sign its receipt with an HMAC, which
is exactly as trustworthy as the secrecy of the key. A Lean proof has no such
dependency: the trust anchor is the kernel, which the consumer runs, using
*their* pinned Lean and Mathlib.

**Nothing is approximate.** Every certificate carries `tolerance = 0`. The f64
solve is treated as a *heuristic that proposes* — an active set, a Gram matrix,
a bound — and everything emitted is recomputed exactly over ℚ. If the float
guess was wrong, the emitter refuses rather than shipping a certificate that
cannot verify.

That last point is the one that surprises people, so it is worth being concrete.
On the `certify_infeasible` fixture the solver's Farkas ray has norm ≈ 2.3e10
and satisfies `Aᵀy = 0` to a relative residual of ~1.7e-11 — excellent for
floating point. Converted losslessly to ℚ, that residual is `−103801/262144`,
which is not zero, so Lean cannot discharge the hypothesis. Copying the
solver's numbers would produce a certificate that always fails. POUNCE instead
uses the float ray as a hint for the *support* of the exact one and solves for
`y = (1, 1, 1)`.

## What it can prove

Four verdicts, chosen automatically from the `.nl` — you do not select one:

| Verdict | Slice | The proven statement |
|---|---|---|
| `global-min` | convex QP | `x*` is feasible and no feasible point has a lower objective |
| `infeasible` | LP / QP | no feasible point exists (Farkas witness) |
| `unbounded` | LP (`Q = 0`) | the objective decreases without bound along a recession direction |
| `global-lower-bound` | unconstrained polynomial, degree > 2 | `γ ≤ p(x)` for **every** real `x` |
| `global-min` | unconstrained polynomial, degree > 2 | that bound is *attained* at an exhibited `x₀`, so `p(x₀) ≤ p(x)` everywhere |

The last two rows are the interesting ones, because they cover the case where a
solver's usual answer is worth very little. A nonconvex polynomial has many
basins; a local solve returns one and can say nothing about the others, and no
KKT argument fixes that. What *can* be established is a bound, via a
sum-of-squares identity — exhibit a monomial basis `m(x)`, a rational `γ`, and
a positive-semidefinite Gram matrix `G` with

```text
p(x) − γ = m(x)ᵀ G m(x)
```

The right-hand side is a nonnegative quadratic form for every `x`, so the
bound holds globally. Lean closes the identity with `ring` and discharges
`G ⪰ 0` from an exact `LDLᵀ` — two independent obligations, both required.

If some rational point *reaches* γ, the claim strengthens from a bound to a
minimum, and the extra obligation is a single equation: `p(x₀) = γ` over ℚ. For
`x⁴ − 2x² + 2` that point is `x = 1`, and `pounce certify` finds it by snapping
the local solve's iterate to a short ladder of rational grids and evaluating `p`
**exactly** — equality or nothing, never closeness.

Not every polynomial has one. `x⁴ − 3x² + 2` minimizes at `±√(3/2)`, which no
rational point reaches, so its certificate stops at `γ = −1/4` and says so. That
is the honest answer rather than a limitation to route around: a certificate
over ℚ cannot exhibit an irrational minimizer, and the bound it does prove still
holds for every real `x`.

Anything outside these slices **exits 2** rather than emit something unsound:

```console
$ pounce certify certify_maximize.nl certify_maximize.sol
pounce certify: certify supports minimize objectives only (v1)
$ echo $?
2
```

Maximize objectives, indefinite `Q`, non-polynomial objectives, and
constrained higher-degree problems are all refused today.

## Emitting a certificate

Solve first, then certify the result. The `.sol` is an input: `certify` does
not rerun the solver, it certifies the claim.

```console
$ pounce nonconvex.nl
$ pounce certify nonconvex.nl nonconvex.sol -o nonconvex.cert.json
```

For `x⁴ − 2x² + 2` — global minima at both `+1` and `−1`, so the local solve
lands in one basin arbitrarily — the certificate opens:

```json
{
  "schema": "pounce.lean-cert/v1",
  "verdict": "global-min",
  "problem_class": "sos-poly",
  "tolerance": { "num": "0", "den": "1" },
  "bound":     { "num": "1", "den": "1" },
  "binding": {
    "nl_sha256":  "69ee731038777d1c5b26b0ada4191a93d11dd91b10d427a0e7a16d2da01946d8",
    "sol_sha256": "ff379f9e3437895a5af918b6e01e71a6c7742795bc710689f10a05ee4bb9ff49",
    "solver": "pounce 0.9.0"
  },
  "toolchain": { "lean": "leanprover/lean4:v4.31.0", "mathlib": "fabf563a…" },
  "candidate": { "x": [ { "num": "1", "den": "1" } ],
                 "objective": { "num": "1", "den": "1" } }
}
```

Three details are load-bearing. The SDP that finds `G` returns a bound near
`1 − 1e-9`; the certificate says exactly `1/1`. The iterate is near `1 − 4e-10`;
the candidate is exactly `1/1`. And `candidate.objective` equals `bound` — that
equality is the entire difference between this and a `global-lower-bound`
certificate, which omits the `candidate` key rather than nulling it, so a
consumer cannot mistake a bound for a claimed minimizer.

The full field-by-field reference is the
[Lean Certificate Schema v1](schema/lean-cert-v1.md).

## Binding a certificate to your problem

A hash match alone is not enough. A certificate could carry the right
`nl_sha256` while its `problem` block describes something *easier* — drop a
constraint and the resulting proof is perfectly valid, just about a different
problem. `cert-verify` closes that hole by re-deriving the problem from **your**
`.nl` using the trusted, deterministic frontend and comparing:

```console
$ pounce cert-verify nonconvex.nl nonconvex.cert.json
cert-verify: OK — certificate matches this .nl
```

```console
$ pounce cert-verify other.nl nonconvex.cert.json
cert-verify: REJECT — binding.nl_sha256 does not match this .nl
         cert: 69ee731038777d1c5b26b0ada4191a93d11dd91b10d427a0e7a16d2da01946d8
         .nl : ba8f872e8e6881259b04c6c5999ee211bd2e75c2f977e6ff853b6f18e701f10b
```

Exit `0` on match, `2` otherwise. This step does not run Lean.

Producer and consumer share the same re-derivation code rather than
reimplementing it, so a mismatch always means real drift — never two
implementations disagreeing about what the `.nl` says.

## The acceptance rule

A certificate is accepted **iff all three** hold. Any one alone is insufficient:

1. **`pounce cert-verify`** passes — the proof is about *this* problem.
2. **`lake build`** of the generated Lean succeeds, under the consumer's own
   pinned toolchain, not the one the certificate suggests.
3. **The axiom audit** — `#print axioms` lists only `propext`,
   `Classical.choice`, `Quot.sound`, and no `sorryAx`. Without this, a proof
   could lean on an admitted lemma and still build.

Step 3 is why the toolchain pin is recorded but not trusted: a proof reproduces
only under a matching Lean/Mathlib, and the consumer must be the one to decide
which that is.

## How it is tested

`scripts/check-lean-cert.sh` runs three layers on every fixture, and the third
is the one that tests the actual soundness claim:

| Layer | Checks | Needs Lean? |
|---|---|---|
| 1 | certificates regenerate byte-identically; `cert-verify` binds each to its `.nl` | no |
| 2 | codegen reproduces the golden `.lean`; `lake build` succeeds; axioms audited | yes |
| 3 | **deliberately corrupted certificates must FAIL to build** | yes |

Layer 3 carries one forgery per obligation a verdict rests on — a corrupted KKT
dual, an indefinite Hessian, a broken Farkas ray, an inflated SOS bound, an
indefinite Gram matrix that still satisfies the SOS identity, and a minimizer
that does not attain a bound which is itself genuine. A test suite where
everything passes proves only that valid inputs work; these fixtures are the
ones that would catch a proof that accepts too much.

## Limits

* **Feasibility is checkable; the model is not.** Like `verify`, this certifies
  a statement about *a given* model. Whether that model is the right one is
  owned outside POUNCE.
* **The slice is narrow by construction.** Every extension is real
  mathematics — the SOS route needed exact Gram rounding, not plumbing — so
  refusal is the honest default rather than a limitation to work around.
* **`unbounded` covers LP only** (`Q = 0`); a nonzero Hessian needs a
  recession-direction refinement that does not exist yet.
* **The SOS route is unconstrained-only.** The theorem quantifies over all of
  ℝⁿ, so a constrained problem would get a statement that is true but about the
  wrong problem. Routing is decided by objective *degree*, never by a QP
  extraction failing, so a convex QP can never silently downgrade from
  `global-min` to a mere bound.
* **`feasible` is emitted by nobody.** The consumer accepts an ε-feasibility
  verdict with an existence witness; POUNCE cannot construct one, so every such
  certificate in existence is hand-written.
