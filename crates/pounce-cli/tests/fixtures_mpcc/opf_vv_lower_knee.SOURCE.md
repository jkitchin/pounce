# `opf_vv_lower_knee.nl`, `opf_vv_lower_knee_held.nl`

The two-bus balanced AC OPF case `vv_lower_knee` from Frederik Geth's
MPCCModels.jl PR #13 (<https://github.com/madsuite-org/MPCCModels.jl/pull/13>,
`examples/opf/`, pinned at `frederikgeth/MPCCModels.jl@19a4b275`). One inverter's
volt-var curve has a knot at V = 0.94, and the optimum sits exactly on it, so
that knot's hinge pair `0 ≤ r ⟂ s ≥ 0` is biactive at the answer.

The `.nl` files are written by Pyomo's `mpec.nl` from the port in the pounce
benchmark data root (`mpcc_opf/`, see its `SOURCE.md`). The `.col` files are
Pyomo's column names, in `.nl` order.

`opf_vv_lower_knee_held.nl` is a variant made for pounce's tests. It is
`opf_vv_lower_knee.nl` plus the one row `r[1] ≥ 5e-5`. The near-biactive point
the driver's stages stop at (`r[1] = 7.8e-5`, `s[1] = 1.3e-4`) satisfies that
row, but the branch holding `r[1]` at zero does not.

Licences:

- MPCCModels.jl is MIT, Copyright (c) 2026 Anton E. Pozharskiy, Francois
  Pacaud, Armin Nurkanovic.
- The hinge formulation comes from BMOPFTools.jl PR #383, BSD-3-Clause,
  Copyright 2026 Frederik Geth @ The University of Queensland.

Both permit redistribution with their notices, and the full texts are here:
`opf_vv_lower_knee.LICENSE.MPCCModels.jl` and
`opf_vv_lower_knee.LICENSE.BMOPFTools.jl`.
