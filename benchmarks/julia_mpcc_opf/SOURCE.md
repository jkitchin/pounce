# MPCCModels balanced AC OPF examples

- Source: https://github.com/frederikgeth/MPCCModels.jl
- Pinned revision: `19a4b275c8ebb3fb5f7a051cfe36c45ad1a948de` (package 0.1.0).
- Models: `examples/opf/`, balanced AC networks with piecewise volt-var and
  volt-watt controller laws; four scenarios selected by name in `run.jl`.
- Attribution/citation: Frederik Geth, **MPCCModels.jl balanced AC OPF examples**,
  revision above; MPCCModels.jl package by Anton E. Pozharskiy, Francois Pacaud,
  Armin Nurkanovic and contributors.
- License: MIT, as supplied in upstream `LICENSE` at that revision.
- The harness loads the original model and its independent residual/physical-law
  checks from the pinned checkout. No model data or upstream code is copied.
- Changes: the NLP callbacks are adapted to an MOI NLPBlock, and the original
  nonnegative variable pairs are declared using `MOI.Complements`. Both exact
  and limited-memory Hessian modes are tested with the original starts.
