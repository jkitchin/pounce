# Upstream source and attribution

The Julia wrapper is MIT licensed. POUNCE's separately built Rust library
remains EPL-2.0.

The following files derive from
[Ipopt.jl commit 592aa741adaad85377c72aafa9bc08fdf0cc908e](https://github.com/jump-dev/Ipopt.jl/tree/592aa741adaad85377c72aafa9bc08fdf0cc908e),
copyright (c) 2013 Iain Dunning, Miles Lubin, and contributors:

| Pounce.jl file | Ipopt.jl source |
|---|---|
| `src/MOI_wrapper.jl` | `ext/IpoptMathOptInterfaceExt/MOI_wrapper.jl` |
| Callback API documentation in `src/MOI_callbacks.jl` | `ext/IpoptMathOptInterfaceExt/MOI_wrapper.jl` |
| `src/utils.jl` | `ext/IpoptMathOptInterfaceExt/utils.jl` |
| `src/C_wrapper.jl` | `src/C_wrapper.jl` |
| `test/MOI_wrapper.jl` | `test/MOI_wrapper.jl` |
| HS071 fixture in `test/cinterface.jl` | `test/C_wrapper.jl` |

The original MIT permission notice is reproduced in LICENSE.md.

Adaptations include POUNCE-specific library loading and solver identification,
one-byte callback booleans matching pounce.h, callback exception containment,
explicit GC rooting, buffer-length checks, idempotent cleanup, result validity
after failures, and live iterate inspection. Native solver-library tests for
SPRAL/HSL and the Ipopt extension alias are not applicable and are omitted.
POUNCE-specific tests exercise the new boundary and JuMP integration.
