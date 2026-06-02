"""Reaction barriers on a molecular potential energy surface.

A chemical reaction is a walk on a potential energy surface (PES) from one
minimum (a stable molecular state) to another, over a mountain pass. That
pass is an **index-1 saddle point** — the *transition state* — and the
**reaction barrier** is its height above the reactant:

    barrier(reactant -> product) = E(transition state) - E(reactant).

So finding reaction barriers is exactly: (1) find the minima (stable
states), (2) find the index-1 saddles (transition states), (3) connect each
saddle to the two minima it joins, by sliding downhill from the pass along
its unstable direction into each adjacent valley.

This example uses the **Müller-Brown potential**, the standard 2-D benchmark
PES for reaction-path methods, which has three minima and two transition
states (Müller & Brown, *Theoret. Chim. Acta* **53**, 75-93 (1979),
doi:10.1007/BF00547608). It uses:

* ``pounce.find_minima``  -> the three stable states,
* ``pounce.find_saddles`` -> the two transition states (eigenvector
  following), and
* a short steepest-descent "intrinsic reaction coordinate" from each saddle
  to assemble the reaction network and barrier heights.

Run:  python reaction_barrier.py
"""

import os

os.environ.setdefault("RUST_LOG", "off")  # quiet the harmless solve log

import numpy as np

import pounce

# --- Müller-Brown potential and its analytic derivatives -------------------
_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def V(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    return float(np.sum(_A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)))


def grad(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    return np.array([np.sum(e * (2 * _a * dx + _b * dy)),
                     np.sum(e * (_b * dx + 2 * _c * dy))])


def hess(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    px, py = 2 * _a * dx + _b * dy, _b * dx + 2 * _c * dy
    hxx = np.sum(e * (px * px + 2 * _a))
    hyy = np.sum(e * (py * py + 2 * _c))
    hxy = np.sum(e * (px * py + _b))
    return np.array([[hxx, hxy], [hxy, hyy]])


BOUNDS = [(-1.5, 1.2), (-0.5, 2.2)]
OPTS = {"print_level": 0, "tol": 1e-8}


# --- steepest-descent path from a point down to a minimum ------------------
def descend(x_start, minima, ds=0.01, max_steps=3000, reach=0.05):
    """Follow the normalized negative gradient (an approximate intrinsic
    reaction coordinate) until the path reaches one of ``minima``; snap to it.

    (Fixed-length steepest-descent steps oscillate around these steep minima
    rather than converging by gradient norm, so we terminate on proximity to
    a known minimum instead.)"""
    x = np.asarray(x_start, float).copy()
    path = [x.copy()]
    for _ in range(max_steps):
        g = grad(x)
        ng = np.linalg.norm(g)
        if ng < 1e-9:
            break
        x = x - ds * g / ng
        path.append(x.copy())
        j = int(np.argmin([np.linalg.norm(x - m) for m in minima]))
        if np.linalg.norm(x - minima[j]) < reach:
            path.append(minima[j].copy())
            return np.array(path), minima[j]
    return np.array(path), x


def main():
    # 1. Stable states = minima.
    states = pounce.find_minima(
        V, [-0.5, 1.4], method="flooding", jac=grad, hess=hess, bounds=BOUNDS,
        n_minima=3, max_solves=120, patience=40, dedup=1e-2, seed=0,
        strategy_kw={"sigma": 0.4, "amplitude": 150.0}, options=OPTS,
    )
    minima = [np.asarray(x, float) for x in states.minima]
    Emin = list(states.values)

    # 2. Transition states = index-1 saddles.
    ts = pounce.find_saddles(
        V, [0.0, 0.5], grad=grad, hess=hess, bounds=BOUNDS, index=1,
        n_saddles=2, max_solves=120, patience=50, dedup=1e-2, seed=0,
        max_step=0.05, grad_tol=1e-5,
    )

    print(f"Found {len(minima)} stable states and {len(ts)} transition states.\n")
    print("Stable states (minima):")
    for i, (x, e) in enumerate(zip(minima, Emin)):
        tag = "  <- global" if i == 0 else ""
        print(f"  state {i}: ({x[0]:+.4f}, {x[1]:+.4f})  E = {e:8.2f}{tag}")

    # 3. Connect each transition state to the two minima it joins.
    def nearest_state(x):
        return int(np.argmin([np.linalg.norm(x - m) for m in minima]))

    print("\nTransition states and reaction barriers:")
    connections = []
    for p in ts.points:
        xs = p.x
        w, U = np.linalg.eigh(hess(xs))   # softest eigenvector = unstable mode
        v = U[:, 0]
        sides = []
        for sgn in (+1.0, -1.0):
            seg, end = descend(xs + 0.02 * sgn * v, minima)
            sides.append((nearest_state(end), seg))
        (i, seg_i), (j, seg_j) = sides
        connections.append((p, i, seg_i, j, seg_j))
        Ets = p.f
        print(f"  TS ({xs[0]:+.4f}, {xs[1]:+.4f})  E = {Ets:8.2f}  "
              f"connects state {i} <-> state {j}")
        print(f"      barrier  {i}->{j} = {Ets - Emin[i]:7.2f}     "
              f"{j}->{i} = {Ets - Emin[j]:7.2f}")

    # The shared state is the reaction intermediate.
    counts = {}
    for _, i, _, j, _ in connections:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    hub = max(counts, key=counts.get)
    print(f"\nReaction network: state {[k for k in counts if k!=hub][0]} "
          f"<=> state {hub} (intermediate) <=> "
          f"state {[k for k in counts if k!=hub][-1]}")

    _maybe_plot(minima, Emin, connections, hub)


def _maybe_plot(minima, Emin, connections, hub):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # --- (1) PES map with states, transition states, and MEPs ---
    xs = np.linspace(-1.5, 1.2, 400)
    ys = np.linspace(-0.5, 2.2, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for k in range(4):
        Z += _A[k] * np.exp(_a[k] * (X - _x0[k])**2 + _b[k] * (X - _x0[k]) * (Y - _y0[k])
                            + _c[k] * (Y - _y0[k])**2)
    Z = np.clip(Z, -150, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax1.contour(X, Y, Z, levels=20, colors="k", alpha=0.15, linewidths=0.5)
    for p, i, seg_i, j, seg_j in connections:
        for seg in (seg_i, seg_j):
            ax1.plot(seg[:, 0], seg[:, 1], "w-", lw=2)
        ax1.scatter(*p.x, c="red", marker="^", s=160, edgecolors="k", zorder=6)
    for idx, m in enumerate(minima):
        ax1.scatter(*m, c="white", marker="o", s=140, edgecolors="k", zorder=6)
        ax1.annotate(f"{idx}", m, color="k", fontweight="bold",
                     ha="center", va="center", zorder=7)
    ax1.set_title("Müller-Brown PES: o states, ^ transition states, — MEP")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    # --- (2) reaction-coordinate energy profile across the network ---
    # Order: end -> TS -> intermediate -> TS -> other end.
    ends = [k for k in (set(c[1] for c in connections) | set(c[3] for c in connections))
            if k != hub]
    # Build a continuous path end0 -> hub -> end1 from the descent segments.
    def seg_between(state_a, state_b):
        for p, i, seg_i, j, seg_j in connections:
            if {i, j} == {state_a, state_b}:
                # state_a -> (down-path reversed) -> TS -> (down-path) -> state_b
                up = seg_i if i == state_a else seg_j
                dn = seg_j if j == state_b else seg_i
                return np.vstack([up[::-1], p.x[None, :], dn])
        return None

    path = seg_between(ends[0], hub)
    tail = seg_between(hub, ends[1])
    full = np.vstack([path, tail[1:]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))])
    energies = np.array([V(p) for p in full])
    ax2.plot(s, energies, "b-", lw=2)
    # Mark the stationary points along the coordinate.
    for label, e in [(f"state {ends[0]}", Emin[ends[0]]),
                     (f"state {hub}", Emin[hub]),
                     (f"state {ends[1]}", Emin[ends[1]])]:
        k = int(np.argmin(np.abs(energies - e)))
        ax2.scatter(s[k], energies[k], c="white", edgecolors="b", s=80, zorder=5)
        ax2.annotate(label, (s[k], energies[k]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8)
    for p, *_ in connections:
        k = int(np.argmin(np.abs(energies - p.f)))
        ax2.scatter(s[k], energies[k], c="red", marker="^", s=90, zorder=5)
    ax2.set_title("Reaction-coordinate energy profile")
    ax2.set_xlabel("reaction coordinate (arc length)")
    ax2.set_ylabel("energy")
    ax2.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "reaction_barrier.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
