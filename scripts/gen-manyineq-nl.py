#!/usr/bin/env python3
"""Generate a small nonconvex QP with many more linear inequalities than
variables (gh#1098).

Why this exists
---------------
`perturb_slack_max_rows` branches on the number of inequality rows `m_d`:
at or below the cap the inertia correction is upstream Ipopt's (`δ_s = δ_x`),
above it the slack-block perturbation decays as `cap / m_d`. No fixture in
`crates/pounce-cli/tests/fixtures` had more than 351 inequality rows against
a default cap of 1000, so the fixture sweep could not see the second branch
at all. This writes the Mittelmann `qcqp1500-1nc` shape in miniature —
few variables, a handful of equalities, many inequalities most of which are
inactive at the solution — so that branch has a fixture and a sweep line.

`manyineq_concave_qp.nl` in that directory was produced by

    scripts/gen-manyineq-nl.py 30 6 1100 4 \
        crates/pounce-cli/tests/fixtures/manyineq_concave_qp.nl neg

A generated instance is not a Mittelmann instance; nothing measured on one
may be reported as if it were.

What it writes
--------------
* `n` free variables with `x0 = 0`;
* an objective `½xᵀQx + cᵀx` with a random sparse symmetric `Q`. With the
  trailing `neg` argument the diagonal is drawn from `U(-2, -0.5)`, so the
  objective is concave and the inertia correction fires on every iteration
  (the regime the slack perturbation acts in); without it the diagonal is
  `U(-2, 2)` and the correction may never be needed;
* `m_eq` sparse linear equalities through a reference point `x_ref`;
* `m_ineq` linear inequalities: `2n` box rows `|x_i| <= 3` written as rows
  (so they count as inequality rows, not bounds) and the rest sparse random
  rows `aᵀx <= aᵀx_ref + slack`, so `x_ref` is strictly feasible.

Written through Pyomo's `.nl` writer so the file is reproducible from this
script and the seed alone.

usage: gen-manyineq-nl.py <n> <m_eq> <m_ineq> <seed> <out.nl> [neg]
"""
import random
import sys

import pyomo.environ as pyo

n, m_eq, m_ineq, seed, out = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
concave = len(sys.argv) > 6
rng = random.Random(seed)
m = pyo.ConcreteModel()
m.I = pyo.RangeSet(0, n - 1)
m.x = pyo.Var(m.I, initialize=0.0)
# reference point: interior of the polytope
xref = [rng.uniform(-1, 1) for _ in range(n)]
Q = [[0.0] * n for _ in range(n)]
for i in range(n):
    Q[i][i] = rng.uniform(-2, -0.5) if concave else rng.uniform(-2, 2)
    for j in range(i + 1, n):
        if rng.random() < 0.2:
            v = rng.uniform(-1, 1)
            Q[i][j] = v
            Q[j][i] = v
c = [rng.uniform(-1, 1) for _ in range(n)]
m.obj = pyo.Objective(
    expr=0.5
    * sum(
        Q[i][j] * m.x[i] * m.x[j]
        for i in range(n)
        for j in range(n)
        if Q[i][j] != 0.0
    )
    + sum(c[i] * m.x[i] for i in range(n)),
    sense=pyo.minimize,
)
m.eq = pyo.ConstraintList()
for _ in range(m_eq):
    a = [rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0 for _ in range(n)]
    if not any(a):
        a[rng.randrange(n)] = 1.0
    m.eq.add(
        sum(a[i] * m.x[i] for i in range(n) if a[i] != 0.0)
        == sum(a[i] * xref[i] for i in range(n))
    )
m.ineq = pyo.ConstraintList()
# 2n box rows as inequalities so the region is bounded (they count as d-rows)
for i in range(n):
    m.ineq.add(m.x[i] <= 3.0)
    m.ineq.add(-m.x[i] <= 3.0)
for _ in range(m_ineq - 2 * n):
    a = [rng.uniform(-1, 1) if rng.random() < 0.1 else 0.0 for _ in range(n)]
    if not any(a):
        a[rng.randrange(n)] = 1.0
    slack = rng.uniform(0.05, 2.0)
    m.ineq.add(
        sum(a[i] * m.x[i] for i in range(n) if a[i] != 0.0)
        <= sum(a[i] * xref[i] for i in range(n)) + slack
    )
m.write(out, io_options={"symbolic_solver_labels": False})
print("wrote", out, "n=%d m_eq=%d m_ineq=%d" % (n, m_eq, m_ineq))
