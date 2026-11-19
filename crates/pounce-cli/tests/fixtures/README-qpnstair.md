# `qpnstair.nl` (gh#1196, gh#1202)

Vanderbei's CUTE model `qpnstair` (467 variables, 356 rows, nonconvex
diagonal quadratic objective, classification QLR2-MN-467-356): the STAIR
LP with Gould's class-4 nonconvex Hessian.

**Why it is here.** It is the model on which presolve's derived-bound dual
recovery (`pounce_presolve::dual_recovery`, M24) first hung for 838 s
(gh#1196), and then, with a work budget in place, rolled back and reported
the reduced problem's duals (gh#1202). With `presolve=yes` it reaches the
recovery's proximal-point loop on a 499-multiplier component with
non-unique multipliers, whose residual floors at about `2e-9` while the
multipliers are about `9e4`. That is the branch the loop's stopping rule
has to get right, and no synthetic fixture in the repository reaches it.
`crates/pounce-cli/tests/issue_1202_derived_bound_dual_recovery.rs` pins it.

**Provenance.** Converted from the AMPL source

    https://vanderbei.princeton.edu/ampl/nlmodels/cute/qpnstair.mod
    sha256 38306b215f957146e2f827003c0177c387a9dc71b81523d1ade0d120d9e90586

with Pyomo's `.nl` writer (no AMPL licence is needed): every `var`
declaration becomes a `pyomo.environ.Var` with its bounds and start, the
`minimize` and each `subject to` body is evaluated as a Python expression
over those variables (the file uses only `+ - *` and numeric literals), and
the model is written with `format='nl'`. Pyomo keeps the 82 fixed
variables (`l = u`) as variables, so `n = 467`; pounce removes them
itself, which is the `n = 385` the issue quotes.

The source carries this notice, which its terms ask to be reproduced:

    AMPL Model by Hande Y. Benson

    Copyright (C) 2001 Princeton University
    All Rights Reserved

    Permission to use, copy, modify, and distribute this software and
    its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that the copyright notice and this
    permission notice appear in all supporting documentation.
