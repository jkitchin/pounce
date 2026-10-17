# pounce-rs

[![crates.io](https://img.shields.io/crates/v/pounce-rs.svg)](https://crates.io/crates/pounce-rs) [![CI](https://github.com/jkitchin/pounce/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/pounce/actions/workflows/ci.yml) [![docs.rs](https://img.shields.io/docsrs/pounce-rs)](https://docs.rs/pounce-rs)

A single-crate entry point for solving optimization problems with
[POUNCE](https://github.com/jkitchin/pounce) in Rust. For nonlinear programs —
the default build — it provides two APIs:

- a high-level builder API (`Problem` + `Nlp`) for the common case, where only the objective is required and everything else is optional; and
- the low-level `TNLP` trait, re-exported for full control over Hessians, sparsity patterns, scaling, and other advanced features.

Both APIs are backed by the same pure-Rust interior-point solver. POUNCE's
other solver paths — convex/LP/QP, active-set QP, and sensitivity analysis —
are behind [feature flags](#feature-flags-beyond-the-nlp-path).

## Install

```sh
cargo add pounce-rs
```

or add it to `Cargo.toml`:

```toml
[dependencies]
pounce-rs = "0.8"
```

## Quick start

Implement `Problem` (only `objective` is required), then configure and solve
with the `Nlp` builder:

```rust
use pounce_rs::prelude::*;

// min (x0-1)^2 + (x1-2)^2  s.t.  x0 + x1 == 3,  0 <= xi <= 5
struct P;
impl Problem for P {
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }
    fn n_constraints(&self) -> usize { 1 }
    fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
}

let sol = Nlp::new(P)                     // variable count inferred below
    .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
    .constraint_bounds(&[3.0], &[3.0])    // equality: lower == upper
    .x0(&[0.0, 0.0])
    .option_num("tol", 1e-10)
    .solve();

assert!(sol.success);
assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
```

For nonlinear bound tightening, expose each constraint as an `FbbtTape` and
enable the existing presolve options:

`presolve_fbbt=yes` is inactive unless `presolve=yes` is also set.

```rust
impl Problem for P {
    fn constraint_expression(&self, _i: usize) -> Option<FbbtTape> {
        Some(FbbtTape { ops: vec![
            FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Add(0, 1),
        ] })
    }
}

let sol = Nlp::new(P)
    .var_bounds(&[0.0, 0.0], &[10.0, 10.0])
    .constraint_bounds(&[3.0], &[3.0])
    .option_str("presolve", "yes")
    .option_str("presolve_fbbt", "yes")
    .solve();
assert!(sol.fbbt_report.is_some());
```

`constraint_expression(i)` must exactly restate the value written by
`constraints()` for row `i`; otherwise FBBT can cut off the true optimum.
Every slot in the tape must also influence its root's value: a dead slot, or
one masked by a `PowInt(_, 0)`, carries no information about the constraint
(gh #877). `try_solve` compares the two forms at the starting point and box
midpoint and reports sampled mismatches as `NlpError::InvalidFbbtTape`. This is
a smoke check, not a proof of equivalence — two points, compared to a relative
tolerance of ~1.5e-8 — so generate both forms from one source when possible.

Anything you don't implement is provided automatically. Missing gradients and
Jacobians are approximated with finite differences, while the Hessian defaults
to a limited-memory (L-BFGS) approximation. This keeps simple problems concise
without sacrificing access to exact derivatives when needed.

Solver options use the same names as upstream Ipopt
(`option_num`, `option_int`, `option_str`). Names, value types, ranges, and
choices are validated against the option registry. A rejected option is
never applied silently: `solve` panics, and `try_solve` returns
`Err(NlpError::InvalidOption { .. })` naming the option and the reason.

```rust
let sol = Nlp::new(P)
    .x0(&[0.0, 0.0])
    .option_str("mu_strategy", "adaptive")
    .try_solve()?;                        // Result<Solution, NlpError>
```

## Result

`Nlp::solve` returns a `Solution` containing

- `success` and the full `status`
- the optimal point `x`
- the objective value
- constraint multipliers (`multipliers`)
- constraint values (`g`)
- bound multipliers (`z_l` and `z_u`)
- solve statistics (`stats`): wall time, iteration count, evaluation counts,
  and final infeasibilities
- optional FBBT diagnostics (`fbbt_report`)

The vector fields remain empty if the solve aborts before finalization. Opt in
to the full per-iteration trajectory (`stats.iterations`) with
`.capture_iterations()` on the builder.

## Full control: the `TNLP` trait

For problems that need an exact Hessian, custom Jacobian/Hessian sparsity, or
NLP scaling, implement the re-exported `TNLP` trait directly and drive it with
`IpoptApplication`. The whole surface is reachable through the prelude.

See the [crate docs on docs.rs](https://docs.rs/pounce-rs) for a complete HS071
`TNLP` walkthrough.

## Feature flags: beyond the NLP path

The default build is the NLP path only. POUNCE's other solver families live in
their own modules behind features — separate modules because the two QP
families both name their types `QpProblem` / `QpSolution` / `QpStatus`, so a
flat surface could not carry both:

| feature | module | what it covers |
|---|---|---|
| `convex` | `pounce_rs::convex` | LP, convex QP, SOCP / exponential / power / PSD cones, SOS; batched and warm-started solves; symbolic-factorization reuse; QP sensitivity and reduced Hessian |
| `qp` | `pounce_rs::qp`, `pounce_rs::sqp` | sparse **parametric active-set** QP — the SQP / MPC / continuation engine, indefinite Hessians allowed — plus the SQP working-set warm-start contract |
| `sensitivity` | `pounce_rs::sensitivity` | sIPOPT-style NLP sensitivity: `∂x*/∂p` predictors, parametric warm starts, reduced Hessian |
| `minima` | `pounce_rs::minima` | multistart / find-minima global search: several distinct local minima of one NLP, six strategies (the search `pounce --minima` runs) |
| `full` | — | all four |

```toml
[dependencies]
pounce-rs = { version = "0.9", features = ["convex", "sensitivity"] }
```

`convex` and `qp` also bring in `pounce_rs::linsol`, whose `backend()` supplies
the sparse symmetric factorization those entry points take as an argument.

```rust
use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_batch_parallel};
use pounce_rs::linsol::serial_backend;

// A batch of box-constrained QPs, one per rayon worker.
let sols = solve_qp_batch_parallel(&probs, &QpOptions::default(), serial_backend);
assert!(sols.iter().all(|s| s.status == QpStatus::Optimal));
```

Enabling a feature widens what this crate *exports*; it is close to free at
build time, because the default NLP path already pulls `pounce-qp`,
`pounce-linsol`, and `pounce-feral` transitively. Only `convex`,
`sensitivity` and `minima` add crates to compile.

## License

EPL-2.0.
