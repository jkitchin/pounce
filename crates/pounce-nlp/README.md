# pounce-nlp

NLP-side glue for POUNCE. Port of Ipopt's `src/Interfaces/`. This is
the crate Rust users implement against when writing custom NLP
problems.

## What's in it

- The user-facing [`TNLP`] trait for problem definition (port of
  `IpTNLP.{hpp,cpp}`).
- Return-code enums [`ApplicationReturnStatus`] and [`AlgorithmMode`]
  (port of `IpReturnCodes_inc.h`).
- `SolverReturn` (algorithm-side, port of `IpAlgTypes.hpp`).
- [`SolveStatistics`] per-solve counters.
- `TNLPAdapter` and `OrigIpoptNlp`, the bound/constraint splitter
  chain that feeds the algorithm-side IPM.

The user-facing `IpoptApplication` lives in
[`pounce-algorithm`](../pounce-algorithm) (`optimize_tnlp` orchestrates
the algorithm), so `pounce-nlp` itself stays free of algorithm-side
imports. Wire-up direction is `pounce-algorithm → pounce-nlp`.

## Implementing a `TNLP`

The required methods describe the problem; defaults supply the
"do-nothing" behaviour for everything else. A minimal solver call:

```rust,no_run
use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use pounce_common::types::Number;

/// min (x[0]-3)^2 + (x[1]-4)^2, unconstrained. Optimum (3, 4).
struct Quadratic;

impl TNLP for Quadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2, m: 0, nnz_jac_g: 0, nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -1e19);
        b.x_u.iter_mut().for_each(|v| *v =  1e19);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 4.0);
        true
    }
    fn eval_g(&mut self, _: &[Number], _: bool, _: &mut [Number]) -> bool { true }
    fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, _: SparsityRequest<'_>) -> bool { true }
    fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
}

let mut app = IpoptApplication::new();
app.initialize().unwrap();
let status = app.optimize_tnlp(Rc::new(RefCell::new(Quadratic)));
assert_eq!(status as i32, 0); // Solve_Succeeded
```

A second built-in example (`rosenbrock`) ships in
[`pounce-cli`'s `builtin` module](../pounce-cli/src/builtin.rs).

## Sparsity convention

`SparsityRequest` carries either an `(irow, jcol)` structure call or a
`vals` values call. POUNCE accepts both 0-based (`IndexStyle::C`) and
1-based (`IndexStyle::Fortran`) triplets — same as upstream.

## License

EPL-2.0.

[`TNLP`]: src/tnlp.rs
[`ApplicationReturnStatus`]: src/return_codes.rs
[`AlgorithmMode`]: src/return_codes.rs
[`SolveStatistics`]: src/solve_statistics.rs
