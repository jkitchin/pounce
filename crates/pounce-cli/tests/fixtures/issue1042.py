"""Regenerate the gh #1042 non-finite-Hessian hand-back fixture.

    python3 issue1042.py            # writes the .nl next to this script

Needs only Pyomo (its `.nl` writer is native — no AMPL involved).

* `issue1042_nonfinite_w_handback.nl` —

      min (x - 3)^2 + |y|^2 * |-y|^0.85   s.t.  x^2 <= 1,  x >= 0

  from `(x, y) = (1, 0)`. The objective is the `dallas*` Hazen-Williams
  term written the way AMPL emits it, so its second derivative in `y` is
  `0 * inf` at `y = 0`. The second factor's operand is `-y`, not `y`, on
  purpose: the `.nl` reader merges `|e|^p * |e|^q` into `|e|^(p+q)` when the
  two operands are the same expression (gh #1042 R3), and the merged form has
  a finite Hessian at 0, which would solve this model in 5 iterations without
  ever entering restoration. `|-y|` evaluates identically and declines the
  merge, the same device as `tests/data/dallass_kink_unmerged.nl`.

  At `y = 0` the outer factorization rejects `W` before any step is
  tried, and restoration is entered on a non-finite Hessian.
  `y` appears in no constraint, so restoration's proximity term leaves it
  at exactly 0, and the start is already feasible, so restoration cannot
  cut the violation by `kappa_resto` and hands back a point it never
  released. The outer then re-enters restoration from that very point —
  the gh #1042 shape — but on a non-finite `W` again, which is the
  exemption: the entry says nothing about the outer being stuck.
"""

import os

from pyomo.environ import ConcreteModel, Constraint, Objective, Var

HERE = os.path.dirname(os.path.abspath(__file__))


def nonfinite_w_handback():
    m = ConcreteModel()
    m.x = Var(initialize=1.0, bounds=(0, None))
    m.y = Var(initialize=0.0)
    m.obj = Objective(expr=(m.x - 3) ** 2 + abs(m.y) ** 2 * abs(-m.y) ** 0.85)
    m.c = Constraint(expr=m.x ** 2 <= 1)
    return m


if __name__ == "__main__":
    nonfinite_w_handback().write(
        os.path.join(HERE, "issue1042_nonfinite_w_handback.nl"),
        io_options={"symbolic_solver_labels": False},
    )
