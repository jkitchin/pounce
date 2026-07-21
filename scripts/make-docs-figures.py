#!/usr/bin/env python3
"""Regenerate the matplotlib-produced figures in `docs/src/images/`.

Book figures are committed as SVG (the rest of `docs/src/images/` is
hand-authored SVG), so this script exists to keep the generated ones
reproducible rather than being one-off screenshots. Run it after changing
anything the figures illustrate:

    python3 scripts/make-docs-figures.py

Requires the Python frontend and its JAX extra (`pip install -e
'python[jax]'`). Figures are drawn on an opaque white backdrop so they
read on both the light and dark (navy) book themes — the same convention
`solver-landscape.svg` follows.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pounce.jax import JaxProblem, PathFollower  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "src" / "images"

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def path_following_fold() -> None:
    """The cubic fold traced by pseudo-arclength continuation.

    Stationarity of ``f = x⁴/4 − x²/2 − θx`` is ``θ = x³ − x``, which
    folds at ``x = ∓1/√3`` (``θ = ±2/(3√3) ≈ ±0.3849``) — the canonical
    case where parameter continuation stalls and arclength does not.
    """
    def f_cubic(x, p):
        th = p[0]
        return x[0] ** 4 / 4.0 - x[0] ** 2 / 2.0 - th * x[0]

    jp = JaxProblem(
        f=f_cubic, g=None, n=1, m=0, p_example=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    trc = PathFollower(jp).trace_arclength(
        jnp.array([-1.3]), -0.4, ds=0.05, n_steps=120,
    )

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))

    xg = np.linspace(-1.6, 1.6, 400)
    ax[0].plot(xg ** 3 - xg, xg, "-", color="0.85", lw=3,
               label="θ = x³ − x  (exact)")
    sc = ax[0].scatter(trc.theta, trc.x[:, 0], c=trc.s, cmap="viridis",
                       s=10, zorder=3)
    for tp in trc.turning_points:
        ax[0].axvline(tp, color="C3", ls=":", lw=1)
        ax[0].plot([tp], [np.sign(-tp) / np.sqrt(3)], "*", color="C3",
                   ms=14, zorder=4)
    ax[0].set(xlabel="θ", ylabel="x*", title="arclength trace through both folds")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=ax[0], label="arclength s")

    ax[1].plot(trc.s, trc.theta, lw=2)
    for tp in trc.turning_points:
        ax[1].axhline(tp, color="C3", ls=":", lw=1)
    ax[1].set(xlabel="arclength s", ylabel="θ",
              title="θ reverses in s — that is the fold")

    fig.tight_layout()
    dest = OUT / "path-following-fold.svg"
    fig.savefig(dest, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest.relative_to(OUT.parents[2])}  "
          f"(turning points: {[round(t, 4) for t in trc.turning_points]})")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path_following_fold()
