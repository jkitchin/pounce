"""Nonlinear curve fitting with pounce, the scipy way — only richer.

`pounce.curve_fit` mirrors :func:`scipy.optimize.curve_fit` (fit a model
``f(x, *params)`` to data), but because pounce is a constrained interior-point
solver that keeps its converged factorization, it returns a lot more:

* parameter **covariance, standard errors, and confidence intervals**
  (the covariance comes straight from the inverse-Hessian block of the
  converged KKT factor — pounce's *reduced Hessian*),
* **robust losses** (Huber, Cauchy, soft-L1) that shrug off outliers,
* **parameter constraints** — positivity, ranges, or relations between
  parameters — which scipy's ``curve_fit`` cannot express, and
* **data sensitivity** ``dpopt/ddata``: how each fitted parameter responds to
  a perturbation of every data point, a single batched back-solve against the
  same factor.

Write the model with ``jax.numpy`` and pounce differentiates it exactly (no
finite differences), which is what makes the covariance and sensitivity sharp.

Run:  python curve_fit_demo.py
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import jax.numpy as jnp

import pounce


def model(x, a, b, c):
    """Exponential decay to an offset:  a*exp(-b*x) + c."""
    return a * jnp.exp(-b * x) + c


def model_np(x, a, b, c):
    return a * np.exp(-b * x) + c


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 5.0, 40)
    a_true, b_true, c_true = 3.0, 0.9, 0.5
    sigma = 0.04 + 0.02 * x  # heteroscedastic noise
    y = model_np(x, a_true, b_true, c_true) + rng.normal(0.0, sigma)

    # ---- 1. ordinary weighted fit with confidence intervals -----------
    res = pounce.curve_fit(model, x, y, p0=[1, 1, 0], sigma=sigma, sensitivity=True)
    print(res.summary())
    print(f"\n  truth      = [{a_true}, {b_true}, {c_true}]")
    print(f"  covariance from: {res.cov_source}")

    # ---- 2. data sensitivity dpopt/ddata ------------------------------
    # Which data point most influences the decay rate b?
    db = res.dpopt_ddata[1]  # row for parameter b
    i = int(np.argmax(np.abs(db)))
    print(f"\n  most influential point for b: x={x[i]:.2f}  (dB/dy = {db[i]:+.3f})")

    # ---- 3. robust fit on outlier-contaminated data -------------------
    yo = y.copy()
    yo[[5, 18, 31]] += np.array([1.5, -1.2, 1.8])  # three bad points
    fit_sse = pounce.curve_fit(model, x, yo, p0=[1, 1, 0])
    fit_rob = pounce.curve_fit(model, x, yo, p0=[1, 1, 0], loss="huber", f_scale=0.1)
    print("\n  with outliers:")
    print(f"    sse   -> a,b,c = {np.round(fit_sse.popt, 3)}")
    print(f"    huber -> a,b,c = {np.round(fit_rob.popt, 3)}   (closer to truth)")

    # ---- 4. constrained fit: enforce a >= 0 and c in [0, 1] -----------
    fit_con = pounce.curve_fit(
        model, x, y, p0=[1, 1, 0.2],
        bounds=[(0.0, np.inf), (None, None), (0.0, 1.0)],
    )
    print("\n  constrained (a>=0, 0<=c<=1):")
    print(f"    a,b,c = {np.round(fit_con.popt, 3)}  active={fit_con.active_mask.tolist()}")


if __name__ == "__main__":
    main()
