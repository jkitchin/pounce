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

## Scope (v1)

Emits **only** the validated slice: `problem_class = qp-convex`,
`verdict = global-min`, quadratic objective, one-sided linear inequality
constraints, infinite variable bounds, convex (PSD) Hessian. Anything else is
rejected — the crate never emits an unsound certificate.

Validated against Lean `leanprover/lean4:v4.31.0` + Mathlib
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.
