#!/usr/bin/env python3
"""Generate the large-scale synthetic NLP suite as AMPL ``.nl`` files.

These five problems were originally hand-written as Rust ``TNLP`` structs
(the retired ``pounce-large-scale`` crate). They are large, sparse,
parameterised NLPs that stress the sparse linear-algebra path and
workspace sizing of both POUNCE and Ipopt. This script reproduces the
exact same math in Pyomo and emits one ``.nl`` per problem so the suite
runs through the standard ``.nl`` driver
(``benchmarks/scripts/run_nl_bench.sh``) like every other suite — no
compiled Rust harness, no libipopt FFI.

The five problems (default sizes mirror the old Rust harness defaults):

  rosenbrock   Generalized/chained Rosenbrock (CUTE GENROSE), unconstrained,
               nonlinear, tridiagonal Hessian.            n = 2000
  bratu        1-D Bratu BVP, -u'' = λ e^u, 3-point stencil, feasibility
               (objective ≡ 0), nonlinear equality constraints.   n = 10000
  optcontrol   Discretised linear-quadratic optimal control; quadratic
               objective, linear dynamics constraints.    T = 50000  (n = 100001)
  poisson      2-D Poisson boundary control on a K×K grid; quadratic
               objective, linear 5-point-stencil constraints.  K = 200 (n = 80000)
  sparseqp     Convex sparse QP, tridiagonal Q, cyclic 3-term inequality
               rows, box bounds.                          n = 50000

Usage:
    python3 generate_nl.py                 # all problems, default sizes
    python3 generate_nl.py --scale 0.1     # 10% of every default size (quick)
    python3 generate_nl.py rosenbrock bratu # only the named problems
    python3 generate_nl.py --out-dir nl    # output directory (default: ./nl)

Per-problem sizes can also be overridden individually:
    python3 generate_nl.py --rosenbrock-n 500 --optcontrol-t 1000

The ``.nl`` files (and matching ``.row``/``.col`` name maps) land in
``--out-dir`` and are regenerated locally — they are not tracked in git.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    RangeSet,
    Reals,
    minimize,
    exp,
    sin,
)

# Default sizes — mirror the retired Rust harness
# (benchmarks/large_scale/src/bin/large_scale_suite.rs). Rosenbrock is
# capped lower because chained Rosenbrock is fundamentally O(n) Newton
# iterations regardless of solver.
DEFAULTS = {
    "rosenbrock_n": 2000,
    "bratu_n": 10000,
    "optcontrol_t": 50000,
    "poisson_k": 200,
    "sparseqp_n": 50000,
}


def build_rosenbrock(n: int) -> ConcreteModel:
    """min 1 + Σ_{i=1}^{n-1} [100 (x_{i+1} - x_i²)² + (1 - x_{i+1})²].

    Unconstrained. Canonical CUTE GENROSE start x_i = i/(n+1).
    """
    m = ConcreteModel(name=f"rosenbrock_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, initialize=lambda _m, i: i / (n + 1.0))

    def obj(m):
        return 1.0 + sum(
            100.0 * (m.x[i + 1] - m.x[i] ** 2) ** 2 + (1.0 - m.x[i + 1]) ** 2
            for i in range(1, n)
        )

    m.obj = Objective(rule=obj, sense=minimize)
    return m


def build_bratu(n: int) -> ConcreteModel:
    """Feasibility (obj ≡ 0) Bratu BVP -u'' = λ e^u on [0,1], u(0)=u(1)=0.

    Variables x_1..x_n; x_1 and x_n fixed to 0 (Dirichlet). Interior
    residual at i=2..n-1:  (-x_{i-1} + 2 x_i - x_{i+1})/h² - λ e^{x_i} = 0.
    """
    h = 1.0 / (n + 1.0)
    lam = 1.0
    m = ConcreteModel(name=f"bratu_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, initialize=0.0)
    # Dirichlet boundary conditions baked into bounds.
    m.x[1].fix(0.0)
    m.x[n].fix(0.0)

    m.Interior = RangeSet(2, n - 1)

    def residual(m, i):
        return (-m.x[i - 1] + 2.0 * m.x[i] - m.x[i + 1]) / (h * h) - lam * exp(m.x[i]) == 0

    m.pde = Constraint(m.Interior, rule=residual)
    m.obj = Objective(expr=0.0, sense=minimize)
    return m


def build_optcontrol(t: int) -> ConcreteModel:
    """Discretised LQ optimal control.

    min h Σ_{i=0}^{T} (y_i - 1)² + α h Σ_{i=0}^{T-1} u_i²
    s.t. y_0 = 0;  y_{i+1} = (1-h) y_i + h u_i,  i = 0..T-1.
    """
    h = 1.0 / t
    alpha = 0.01
    m = ConcreteModel(name=f"optcontrol_t{t}")
    m.Iy = RangeSet(0, t)
    m.Iu = RangeSet(0, t - 1)
    m.y = Var(m.Iy, domain=Reals, initialize=0.0)
    m.u = Var(m.Iu, domain=Reals, initialize=0.0)

    def obj(m):
        return h * sum((m.y[i] - 1.0) ** 2 for i in range(t + 1)) + alpha * h * sum(
            m.u[i] ** 2 for i in range(t)
        )

    m.obj = Objective(rule=obj, sense=minimize)

    m.y0 = Constraint(expr=m.y[0] == 0.0)

    def dynamics(m, i):
        return m.y[i + 1] - (1.0 - h) * m.y[i] - h * m.u[i] == 0

    m.dyn = Constraint(m.Iu, rule=dynamics)
    return m


def build_poisson(k: int) -> ConcreteModel:
    """2-D Poisson boundary control on a K×K interior grid.

    min Σ_{ij} ½ h² (u_{ij} - u_d)² + ½ α h² f_{ij}²
    s.t. (4 u_{ij} - neighbours)/h² - f_{ij} = 0  (5-point stencil, Dirichlet 0).
    u_d(x,y) = sin(πx) sin(πy), x=(i+1)h, y=(j+1)h, h = 1/(K+1).
    """
    h = 1.0 / (k + 1.0)
    alpha = 0.01
    m = ConcreteModel(name=f"poisson_k{k}")
    m.I = RangeSet(0, k - 1)
    m.J = RangeSet(0, k - 1)
    m.u = Var(m.I, m.J, domain=Reals, initialize=0.0)
    m.f = Var(m.I, m.J, domain=Reals, initialize=0.0)

    def u_desired(i, j):
        x = (i + 1.0) * h
        y = (j + 1.0) * h
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    def obj(m):
        return sum(
            0.5 * h * h * (m.u[i, j] - u_desired(i, j)) ** 2
            + 0.5 * alpha * h * h * m.f[i, j] ** 2
            for i in range(k)
            for j in range(k)
        )

    m.obj = Objective(rule=obj, sense=minimize)

    def pde(m, i, j):
        lap = 4.0 * m.u[i, j]
        if i > 0:
            lap -= m.u[i - 1, j]
        if i < k - 1:
            lap -= m.u[i + 1, j]
        if j > 0:
            lap -= m.u[i, j - 1]
        if j < k - 1:
            lap -= m.u[i, j + 1]
        return lap / (h * h) - m.f[i, j] == 0

    m.pde = Constraint(m.I, m.J, rule=pde)
    return m


def build_sparseqp(n: int) -> ConcreteModel:
    """Convex sparse QP with cyclic 3-term inequality rows and box bounds.

    min ½ xᵀQx - Σ x_i,  Q tridiagonal (4 on diagonal, -1 off-diagonal)
    s.t. x_j + x_{(j+1) mod n} + x_{(j+2) mod n} ≤ 2.5,  0 ≤ x_i ≤ 10.

    ½ xᵀQx expands to Σ 2 x_i² - Σ_{i=1}^{n-1} x_i x_{i+1}.
    """
    m = ConcreteModel(name=f"sparseqp_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, bounds=(0.0, 10.0), initialize=0.5)

    def obj(m):
        quad = sum(2.0 * m.x[i] ** 2 for i in range(1, n + 1))
        offdiag = sum(m.x[i] * m.x[i + 1] for i in range(1, n))
        linear = sum(m.x[i] for i in range(1, n + 1))
        return quad - offdiag - linear

    m.obj = Objective(rule=obj, sense=minimize)

    def threesum(m, j):
        # 0-based cyclic indices j, j+1, j+2 → 1-based variable keys.
        a = j
        b = (j % n) + 1
        c = ((j + 1) % n) + 1
        return m.x[a] + m.x[b] + m.x[c] <= 2.5

    m.row = Constraint(m.I, rule=threesum)
    return m


BUILDERS = {
    "rosenbrock": ("rosenbrock_n", build_rosenbrock),
    "bratu": ("bratu_n", build_bratu),
    "optcontrol": ("optcontrol_t", build_optcontrol),
    "poisson": ("poisson_k", build_poisson),
    "sparseqp": ("sparseqp_n", build_sparseqp),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("problems", nargs="*", choices=list(BUILDERS) + [],
                   help="problems to generate (default: all)")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "nl"),
                   help="output directory for .nl files (default: ./nl)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply every default size by this factor (e.g. 0.1)")
    for key, default in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=None,
                       help=f"override size for {key} (default {default})")
    args = p.parse_args(argv)

    selected = args.problems or list(BUILDERS)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in selected:
        size_key, builder = BUILDERS[name]
        override = getattr(args, size_key)
        if override is not None:
            size = override
        else:
            size = max(2, int(round(DEFAULTS[size_key] * args.scale)))
        model = builder(size)
        stub = os.path.join(args.out_dir, name)
        path = stub + ".nl"
        model.write(path, format="nl",
                    io_options={"symbolic_solver_labels": True})
        nvars = sum(1 for _ in model.component_data_objects(Var))
        ncons = sum(1 for _ in model.component_data_objects(Constraint))
        print(f"wrote {path}  ({size_key}={size}, vars={nvars}, cons={ncons})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
