"""Find saddle points and all critical points, two ways.

Extends the multiple-minima idea to *all* stationary points of f. A critical
point has grad f = 0; its Morse index (number of negative Hessian
eigenvalues) says what it is: 0 = minimum, 1 = transition state, n = maximum.

* Route A -- `find_critical_points`: enumerate the roots of grad f = 0 by
  minimizing the gradient-norm merit ||grad f||^2 with find_minima's
  deflation, then classify each by index. (pounce as a root-finder.)
* Route B -- `find_saddles`: eigenvector following (Cerjan-Miller / dimer
  spirit) walks uphill along the softest Hessian modes and downhill in the
  rest, landing directly on index-1 saddles.

Demo landscape (small, with a known answer):

    f(x, y) = (x^2 - 1)^2 + (y^2 - 1)^2

    4 minima  at (+-1, +-1)   f = 0   index 0
    4 saddles at (+-1, 0), (0, +-1)  f = 1   index 1
    1 maximum at (0, 0)        f = 2   index 2

Run:  python critical_points.py
"""

import os

os.environ.setdefault("RUST_LOG", "off")  # quiet the harmless solve log

import numpy as np

import pounce


def fun(z):
    x, y = z
    return (x * x - 1) ** 2 + (y * y - 1) ** 2


def grad(z):
    x, y = z
    return np.array([4 * x * (x * x - 1), 4 * y * (y * y - 1)])


def hess(z):
    x, y = z
    return np.array([[4 * (3 * x * x - 1), 0.0], [0.0, 4 * (3 * y * y - 1)]])


BOUNDS = [(-1.5, 1.5), (-1.5, 1.5)]
OPTS = {"print_level": 0, "tol": 1e-10}


def route_a():
    print("Route A -- find_critical_points (enumerate roots of grad f = 0)")
    r = pounce.find_critical_points(
        fun, [0.3, 0.4], grad=grad, hess=hess, bounds=BOUNDS,
        method="deflation", n_points=12, max_solves=200, patience=40,
        dedup=1e-2, seed=0, options=OPTS,
    )
    print(f"  {len(r)} critical points "
          f"({len(r.minima)} minima, {len(r.saddles)} saddles, "
          f"{len(r.maxima)} maxima); status={r.status}, solves={r.n_solves}")
    for p in r.points:
        print(f"    {p.kind:16s} ({p.x[0]:+.3f}, {p.x[1]:+.3f})  "
              f"f={p.f:.3f}  eig={np.round(p.eigvalues, 2)}")
    return r


def route_b():
    print("\nRoute B -- find_saddles (eigenvector following, index 1)")
    s = pounce.find_saddles(
        fun, [0.3, 0.4], grad=grad, hess=hess, bounds=BOUNDS,
        index=1, n_saddles=4, max_solves=100, patience=40, dedup=1e-2, seed=0,
    )
    print(f"  {len(s)} index-1 saddles; status={s.status}, solves={s.n_solves}")
    for p in s.points:
        print(f"    saddle ({p.x[0]:+.3f}, {p.x[1]:+.3f})  f={p.f:.3f}  "
              f"|grad|={p.grad_norm:.1e}")
    return s


def maybe_plot(r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(xs, xs)
    Z = (X**2 - 1) ** 2 + (Y**2 - 1) ** 2
    plt.figure(figsize=(5.2, 5))
    plt.contourf(X, Y, Z, levels=40, cmap="viridis")
    plt.colorbar(label="f")
    colors = {0: "white", 1: "red", 2: "black"}
    markers = {0: "o", 1: "^", 2: "s"}
    for p in r.points:
        plt.scatter(p.x[0], p.x[1], c=colors.get(p.index, "magenta"),
                    marker=markers.get(p.index, "x"), s=120,
                    edgecolors="k", zorder=5)
    plt.title("critical points: o min, ^ saddle, square max")
    plt.xlabel("x"); plt.ylabel("y")
    out = os.path.join(os.path.dirname(__file__), "critical_points.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


def main():
    r = route_a()
    route_b()
    maybe_plot(r)


if __name__ == "__main__":
    main()
