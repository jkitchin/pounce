# pounce-rs

[![crates.io](https://img.shields.io/crates/v/pounce-rs.svg)](https://crates.io/crates/pounce-rs) [![CI](https://github.com/jkitchin/pounce/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/pounce/actions/workflows/ci.yml) [![docs.rs](https://img.shields.io/docsrs/pounce-rs)](https://docs.rs/pounce-rs)

A single-crate entry point for solving nonlinear programs with
[POUNCE](https://github.com/jkitchin/pounce) in Rust. It provides two APIs:

- a high-level builder API (`Problem` + `Nlp`) for the common case, where only the objective is required and everything else is optional; and
- the low-level `TNLP` trait, re-exported for full control over Hessians, sparsity patterns, scaling, and other advanced features.

Both APIs are backed by the same pure-Rust interior-point solver.

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
use pounce_rs::builder::{Problem, Nlp};

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

Anything you don't implement is provided automatically. Missing gradients and
Jacobians are approximated with finite differences, while the Hessian defaults
to a limited-memory (L-BFGS) approximation. This keeps simple problems concise
without sacrificing access to exact derivatives when needed.

Solver options use the same names as upstream Ipopt
(`option_num`, `option_int`, `option_str`).

## Result

`Nlp::solve` returns a `Solution` containing

- `success` and the full `status`
- the optimal point `x`
- the objective value
- constraint multipliers (`multipliers`)
- constraint values (`g`)
- bound multipliers (`z_l` and `z_u`)

The vector fields remain empty if the solve aborts before finalization.

## Full control: the `TNLP` trait

For problems that need an exact Hessian, custom Jacobian/Hessian sparsity, or
NLP scaling, implement the re-exported `TNLP` trait directly and drive it with
`IpoptApplication`. The whole surface is reachable through the prelude.

See the [crate docs on docs.rs](https://docs.rs/pounce-rs) for a complete HS071
`TNLP` walkthrough.

## License

EPL-2.0.
