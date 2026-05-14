# pounce-sensitivity

Sensitivity analysis, parametric NLP warm-start, and reduced-Hessian
computation for POUNCE. Port of upstream Ipopt's
[`contrib/sIPOPT/`][upstream-src].

## Status

Internal crate; **Phase A** (this commit) — `SchurData` and `PCalculator`
trait surface plus the `Index`-flavored implementations. Pure data-shuttling;
no IPM interaction yet. The numerical machinery (`SensStepCalc`,
`DenseGenSchurDriver`, `ReducedHessianCalculator`, `SensApplication`)
lands in Phases B–E per the plan in [pounce#7][issue].

## Algorithmic reference

> Pirnay, H., López-Negrete, R., and Biegler, L.T. (2012).
> *Optimal sensitivity based on IPOPT.*
> Mathematical Programming Computation, **4**(4), 307–331.
> DOI: [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).

DOI verification (Crossref, 2026-05-14): resolves to Springer; title,
authors, volume/issue/pages all confirmed.

## Upstream source

The port follows the upstream sIPOPT contrib at
[`ref/Ipopt/contrib/sIPOPT/src/`][upstream-src]
in this repo (EPL-2.0, © Hans Pirnay 2009–2011 per the headers).
Phase-A files mapped:

| pounce-sensitivity                        | upstream                                                                                          |
|-------------------------------------------|---------------------------------------------------------------------------------------------------|
| `src/schur_data.rs` — `SchurData` trait   | [`SensSchurData.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp) (lines 17–177)        |
| `src/schur_data/index.rs` — `IndexSchurData` | [`SensIndexSchurData.{hpp,cpp}`](../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)       |
| `src/p_calculator.rs` — `PCalculator` trait | [`SensPCalculator.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp) (lines 17–133)   |
| `src/backsolver.rs` — `SensBacksolver` trait | [`SensBacksolver.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensBacksolver.hpp)                    |

Every public item in this crate documents the upstream symbol it mirrors,
with line numbers when they're stable.

## License

EPL-2.0, matching upstream Ipopt and the sIPOPT contrib.

[upstream-src]: ../../ref/Ipopt/contrib/sIPOPT/src/
[issue]: https://github.com/jkitchin/pounce/issues/7
