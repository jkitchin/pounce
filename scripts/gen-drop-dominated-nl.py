#!/usr/bin/env python3
"""Generate a large smooth NLP whose warm SQP step QPs are drop-dominated (gh #1091).

Why this exists
---------------
The Schur-update rule for warm step QPs (`SqpOptions::step_qp_schur_updates`)
acts only from `n + m >= 5000`, on QPs that mostly *drop* rows. Nothing in the
fixture corpus is that large, and the Maros-Meszaros set cannot produce the
case either: at that size `algorithm=active-set-sqp` hands a model to the
interior point first, which finishes almost every one before the SQP solves a
warm QP. This model can, and it is how the rule was measured.

    min  sum_i exp(x_i) - a_i x_i + (rho/2) sum_i (x_i - x_{i+1})^2
    s.t. 0 <= x_i <= 1,   sum_i x_i <= 0.3 n

Started at `x = 0`, every lower bound is active; at the optimum only the
variables with small `a_i` stay there, so the working set shrinks over the
outer iterations and each warm step QP is mostly drops. The `exp` keeps it off
the convex QP route under `solver_selection=auto`.

Usage
-----
    python3 scripts/gen-drop-dominated-nl.py 6000 /tmp/drops6000.nl
    RUST_LOG=pounce::sqp::qp=debug pounce /tmp/drops6000.nl \\
        algorithm=active-set-sqp ipm_sqp_handover=no \\
        hessian_approximation=limited-memory [sqp_qp_use_schur_updates=yes]

`ipm_sqp_handover=no` asks for the cold active-set SQP, so the warm step QPs
are the SQP's own. The `pounce::sqp::qp` debug line reports, per QP, the
working-set changes, refactors, Schur updates and the time split. Requires
Pyomo, for its `.nl` writer.
"""
import argparse
import random

import pyomo.environ as pe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("n", type=int, help="number of variables")
    p.add_argument("out", help="output .nl path")
    p.add_argument("--seed", type=int, default=1091)
    p.add_argument("--rho", type=float, default=2.0, help="coupling weight")
    args = p.parse_args()

    rng = random.Random(args.seed)
    n = args.n
    a = [rng.uniform(0.2, 3.0) for _ in range(n)]

    m = pe.ConcreteModel()
    m.I = pe.RangeSet(0, n - 1)
    m.x = pe.Var(m.I, bounds=(0.0, 1.0), initialize=0.0)
    m.obj = pe.Objective(
        expr=sum(pe.exp(m.x[i]) - a[i] * m.x[i] for i in m.I)
        + 0.5 * args.rho * sum((m.x[i] - m.x[i + 1]) ** 2 for i in range(n - 1))
    )
    m.cap = pe.Constraint(expr=sum(m.x[i] for i in m.I) <= 0.3 * n)
    m.write(args.out, format="nl")


if __name__ == "__main__":
    main()
