# pounce-common

Common primitives used across the POUNCE workspace. Port of Ipopt's
`src/Common/`.

This is an internal crate — most users interact with POUNCE through
[`pounce-cli`](../pounce-cli), [`pounce-nlp`](../pounce-nlp), or
[`pounce-cinterface`](../pounce-cinterface). The types re-exported
here surface in those public APIs.

## What's in it

| Module          | Purpose                                                                  | Ipopt counterpart           |
|-----------------|--------------------------------------------------------------------------|-----------------------------|
| `types`         | `Number = f64`, `Index = i32`, `NLP_{LOWER,UPPER}_BOUND_INF`             | `IpTypes.hpp`               |
| `exception`     | `SolverException` + `ExceptionKind` — no `panic!` in the algorithm path  | `IpException.hpp`           |
| `journalist`    | `Journalist`, `Journal`, `FileJournal`, `StringJournal` — log routing    | `IpJournalist.{hpp,cpp}`    |
| `options_list`  | `OptionsList` — typed get/set with default fallback                      | `IpOptionsList.{hpp,cpp}`   |
| `reg_options`   | `RegisteredOptions` — option registry with types, ranges, defaults       | `IpRegOptions.{hpp,cpp}`    |
| `tagged`        | `TaggedObject` + `TaggedCell` — change-tracking for cached quantities    | `IpTaggedObject.hpp`        |
| `cached`        | `Cache<T>` — tagged-key lookup for `IpoptCalculatedQuantities`           | `IpCachedResults.hpp`       |
| `timing`        | `TimedTask` — wall-clock accumulator                                     | `IpTimingStatistics.{hpp,cpp}` |
| `utils`         | small helpers (string parsing, etc.)                                     | `IpUtils.{hpp,cpp}`         |
| `diagnostics`   | `DiagCategory` — per-iteration dump category flags                       | (pounce-only)               |
| `style`         | tiger/rust color palette for the CLI banner + iteration table            | (pounce-only)               |

## Conventions

- `Index` is `i32` to match upstream's `ipindex`. Sizes that exceed `i32`
  are not supported (mirrors upstream).
- `Number` is `f64`. No generics over scalar type.
- The `tagged` / `cached` pair is the workhorse for invalidation: when
  any iterate component changes, its `Tag` bumps and downstream caches
  miss automatically. This is how the algorithm avoids re-evaluating
  callbacks unnecessarily.

## License

EPL-2.0, matching upstream Ipopt.
