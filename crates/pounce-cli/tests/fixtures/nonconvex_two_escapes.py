"""Regenerate `nonconvex_two_escapes.nl` (gh #805).

    python3 nonconvex_two_escapes.py nonconvex_two_escapes.nl

Needs only Pyomo (its `.nl` writer is native -- no AMPL involved), the same
way `cresc4.py` regenerates `cresc4.nl`.

WHY THIS FIXTURE EXISTS. gh #797 added `neg_curv_escapes`, and the two
fixtures it shipped with -- `nonconvex_qp.nl` and `nonconvex_qp_ineq.nl` --
place exactly ONE escape each: the escape lands on a point whose reduced
Hessian is positive definite, so the probe declines there and a second escape
never happens. Everything the mechanism does with more than one escape (which
certificate the floor holds, what a lost bet hands back after two of them) was
therefore unreachable from a test, which is how gh #805 -- the floor being
REPLACED by each escape rather than ranked -- got in and stayed in. This model
places two escapes and separates the three answers.

THE MODEL

    min  0.225*x0^4 - 0.45*x0^2 + (1000 - 1000.45*x0^2)*x1^2
    s.t. -2 <= x0 <= 2,   -1.5 <= x1 <= 1.5,   start (0, 1)

It is even in `x1`, so `dF/dx1 = 2*(1000 - 1000.45*x0^2)*x1` vanishes on the
whole line `x1 = 0` and the cross term `d2F/dx0dx1 = -4*1000.45*x0*x1`
vanishes with it. An iterate sitting at `x1 = 0` therefore has a zero gradient
and a decoupled Newton row in that coordinate, and stays there -- exactly the
symmetry that makes `nonconvex_qp.nl` converge onto its constrained maximum,
used here to hold the solve on a ridge for long enough to certify it.

Three points matter, and the coefficients are chosen so each is stationary
with the curvature named next to it:

    A = (0, 0)          f =      0        W = diag(-0.9, +2000)
    B = (+-1, 0)        f =     -0.225    W = diag(+1.8,    -0.9)
    C = (+-2, +-1.5)    f =  -6752.25     the global minimum, at the corner

A is a maximum along `x0` and B is a saddle whose negative direction is `x1`
-- a *different* coordinate, which is the point: the second escape has to find
curvature the first one did not leave behind. So the reported answer is a
strict ladder in the option:

    neg_curv_escapes = 0  ->  f =     0        (A, the pre-#797 answer)
    neg_curv_escapes = 1  ->  f =    -0.225    (B)
    neg_curv_escapes = 2  ->  f = -6752.25     (C)
    neg_curv_escapes = 3  ->  f = -6752.25     (C is a genuine minimum; the
                                                probe declines and the third
                                                escape is never placed)

WHY THE COEFFICIENTS ARE WHAT THEY ARE -- read this before editing them. The
escape direction is recovered by three inverse-iteration back-solves against
`(W + Sigma + delta_x I)`, so it is not exactly the `x0` axis: it carries an
`x1` component of order `((lambda_2 + delta)/(lambda_min + delta))^-3`. That
component is a seed, and it GROWS during the continuation as soon as `x1`'s
curvature turns negative -- past `|x0| = sqrt(1000/1000.45)`, i.e. just short
of B. If the seed is bigger than the convergence test can ignore, the solve
walks off B and slides all the way to C on the first escape, and there is no
second escape to observe.

`1000` is what buys the margin. It sets `lambda_2 = 2000` at A, so the seed
lands near `1e-10` and B certifies with `inf_du` well under `tol`. Measured
with `100` in its place the seed left `inf_du = 1.26e-8` at B against
`tol = 1e-8` -- one iteration short -- and the run walked off the ridge. Do
not lower it. The `.45` in `1000.45` and in `0.45` sets the curvature at B to
`-0.9`, which is just under the `delta_x = 1` rung of the probe's ladder and
so gives the probe its own amplification there; a curvature landing just above
a rung gives a ratio near one and the probe declines for want of separation.
"""

import sys

from pyomo.environ import ConcreteModel, Objective, Var, minimize


def build():
    m = ConcreteModel()
    m.x0 = Var(bounds=(-2.0, 2.0), initialize=0.0)
    m.x1 = Var(bounds=(-1.5, 1.5), initialize=1.0)
    m.obj = Objective(
        expr=0.225 * m.x0**4
        - 0.45 * m.x0**2
        + (1000.0 - 1000.45 * m.x0**2) * m.x1**2,
        sense=minimize,
    )
    return m


if __name__ == "__main__":
    build().write(sys.argv[1], io_options={"symbolic_solver_labels": False})
