# pounce-lean-cert

Exact-rational [`pounce.lean-cert/v1`](../../docs/src/schema/lean-cert-v1.md)
certificate emitter for convex-QP / `global-min` solves.

POUNCE solves in `f64`; this crate converts a solve into an **exact-rational**
certificate (`{num, den}` integer strings, no floats) that the external
[`pounce-lean`](https://github.com/jkitchin/pounce-lean) repo turns into a
kernel-checked Lean 4 proof that the returned `x*` is the global minimizer — with
no floating point in the trusted path.

The emitted witnesses (KKT duals, the `LDLᵀ` PSD factorization) are *untrusted*:
wrong data only makes the Lean proof fail to typecheck, never pass falsely. The
emitter self-checks every witness **exactly over ℚ** before writing and errors
out rather than emit a certificate that will not verify.

## Exact refinement — the float `x*` is not what gets certified

POUNCE returns an `x*` that is feasible only to a tolerance (typically ~1e-9
off). Certifying *that* point exactly would fail: converted losslessly to ℚ, its
constraint violation is exactly representable and exactly nonzero.

So the emitter does not certify it. It takes the float active set as a **guess**,
solves the resulting KKT system **exactly over ℚ** (`refine.rs`), and certifies
the exact point that comes out. The float solve is demoted to a heuristic that
proposes an active set. This is why emitted certificates carry `tolerance = 0`.

## Scope (v1)

The authoritative statement is
[`docs/src/schema/lean-cert-v1.md`](../../docs/src/schema/lean-cert-v1.md)
§ "Supported slice (v1)"; this is a summary of it.

Emitted: `problem_class = qp-convex`, `verdict = global-min`, quadratic
objective (`half_quadratic` honored), convex (PSD) Hessian, and linear
constraints in any of these forms:

| Input | Encoding |
|---|---|
| inequality, either orientation | `A x ≥ b` row, multiplier `λ ≥ 0` (`a·x ≤ u` normalized to `−a·x ≥ −u`) |
| equality (`lower == upper`) | `E x = d` row, **free-sign** multiplier `μ` |
| two-sided range (both finite, unequal) | **split** into two one-sided rows `{c}_lo` / `{c}_hi` |
| variable bound `xᵢ ≥ lᵢ` / `xᵢ ≤ uᵢ` | **folded** to a row `var{i}_lb` / `var{i}_ub` |
| fixed variable `xᵢ = v` | folded to an equality row `var{i}_fix` |

> **Do not read the `var_bounds` field as the capability.** Because bounds fold
> into `constraints`, `var_bounds` is *always* emitted as the `±inf` sentinels —
> even for a box-constrained problem. Finite variable bounds are fully
> supported; they just do not appear there. (An earlier version of this file
> said "infinite variable bounds", conflating the encoding with the scope, and
> that error cost real planning time.)

Everything else **exits 2** rather than emitting an unsound certificate:
non-quadratic objectives, indefinite `Q`, maximize objectives, and the
`feasible` / `local-min-strict` verdicts, which are additive future work.

## Validation

Validated against Lean `leanprover/lean4:v4.31.0` + Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

`scripts/check-lean-cert.sh` drives the whole path for four fixtures
(`certify_qp`, `certify_box`, `certify_range`, `certify_eq`) — solve, emit,
diff against the golden cert, bind cert to `.nl`, and with
`POUNCE_LEAN_DIR=… LAKE_BUILD=1` also generate the Lean and kernel-check it,
auditing that the axioms are exactly `{propext, Classical.choice, Quot.sound}`
with no `sorry`.
