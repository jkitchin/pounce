# Curve Fitting

`pounce.curve_fit` fits a model `f(x, *params)` to data — the same call shape
as [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) —
but returns a much richer result and adds capabilities scipy's fitter does not
have. It runs on pounce's interior-point solver, so it inherits **parameter
constraints**, and because the solver keeps its converged factorization it can
hand back the **parameter covariance** (from the *reduced Hessian*) and the
**data sensitivity** `∂params/∂data` essentially for free.

```python
import numpy as np
import jax.numpy as jnp
import pounce

def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c       # write the model with jax.numpy

x = np.linspace(0.2, 5, 40)
y = 3.0 * np.exp(-0.9 * x) + 0.5 + 0.05 * np.random.default_rng(0).normal(size=x.size)

res = pounce.curve_fit(model, x, y, p0=[1, 1, 0])
print(res.summary())
res.popt          # fitted parameters
res.pcov          # covariance matrix
res.perr          # standard errors  = sqrt(diag(pcov))
res.ci            # (n, 2) confidence intervals at `alpha`
```

## How it differs from `scipy.optimize.curve_fit`

| | scipy.curve_fit | pounce.curve_fit |
|---|---|---|
| Least-squares fit + `pcov` | ✅ | ✅ |
| Weighted (`sigma`, `absolute_sigma`) | ✅ | ✅ |
| Box bounds on parameters | ✅ | ✅ |
| **Relations between parameters** (e.g. `a + b ≤ 1`) | ❌ | ✅ |
| **Robust losses** with covariance | partial | ✅ (sandwich) |
| **Confidence intervals / goodness-of-fit** in the result | ❌ | ✅ |
| **Data sensitivity** `∂params/∂data` | ❌ | ✅ |
| Exact derivatives via JAX | ❌ | ✅ |

The statistics follow the same conventions as scipy and
[`pycse.nlinfit`](https://kitchingroup.cheme.cmu.edu/pycse/): the covariance is
`s² · (JᵀJ)⁻¹` with `s² = SSE/(m − n)` (the reduced χ²) unless
`absolute_sigma=True`, and confidence intervals use the Student-t quantile
`popt ± t_{dof,1−α/2} · perr`.

> **Already modelling in Pyomo?** `pyomo_pounce.covariance` computes the
> parameter covariance for an estimation model written directly in Pyomo —
> residuals as constraints, arbitrary surrounding structure — instead of a
> `f(x, *params)` callable, using the same scale-and-invert-the-reduced-Hessian
> recipe read from the same held KKT factor. One caveat for **nonlinear**
> models: `curve_fit` here reports the **Gauss-Newton** covariance
> `2·s²·(JᵀJ)⁻¹` (the scipy/`nls` convention, always ≥ 0), whereas
> `covariance()` reports the **observed-information** covariance from the exact
> Hessian; they match for linear models and in the small-residual limit and
> differ by a few percent on a strongly-curved fit. See
> [Parameter covariance and identifiability](sensitivity.md). Use
> `curve_fit` when the fit is naturally a model-plus-data call (or when you
> want scipy-matching numbers); use `covariance()` to interrogate a Pyomo
> model you already have.

## Derivatives: prefer JAX

Accurate derivatives are what make the covariance and sensitivity sharp — and
they let the solver converge in a couple of iterations so the pounce-native
factor route is available. The Jacobian `∂f/∂p` is resolved in this order:

1. an analytic `jac=<callable>` returning `(len(x), n_params)`,
2. **JAX autodiff** (the default when the model is written with `jax.numpy`),
3. a finite-difference fallback (used only if neither of the above applies;
   it emits a warning and the covariance falls back to the Jacobian form).

```python
res = pounce.curve_fit(model, x, y, p0=[1, 1, 0])           # JAX (model uses jnp)
res = pounce.curve_fit(model, x, y, p0=[1, 1, 0], jac=myjac) # analytic
res = pounce.curve_fit(model_np, x, y, p0=[1, 1, 0])         # numpy model -> FD (warns)
```

## Loss functions

Only **smooth (C²)** losses are supported, because the underlying solver is an
interior-point method. Non-smooth L1/MAE is intentionally out of scope; use a
robust loss instead.

| `loss` | use |
|---|---|
| `"sse"` (default), `"chi2"` | ordinary / weighted least squares |
| `"soft_l1"` = `"huber"` | smooth pseudo-Huber, downweights outliers |
| `"cauchy"` | strong outlier rejection |

`"huber"` and `"soft_l1"` are the **same** smooth (C²) pseudo-Huber loss: a
true piecewise Huber is only C¹ (its curvature jumps at the knee), which the
interior-point solver can't use, so both names map to the C² form.

```python
res = pounce.curve_fit(model, x, y, p0=[1, 1, 0], loss="huber", f_scale=0.1)
res.cov_source        # "sandwich"  (robust covariance estimator)
```

## Parameter constraints

Box bounds express **positivity / negativity / ranges**; `constraints=`
expresses **relations between parameters** using the scipy-style dict format.

```python
# positivity, ranges
pounce.curve_fit(model, x, y, p0=[1, 1, 0.2],
                 bounds=[(0, np.inf), (None, None), (0, 1)])

# a relation: require a + b <= 1   (ineq g(p) >= 0)
cons = [{"type": "ineq", "fun": lambda p: 1.0 - (p[0] + p[1])}]
pounce.curve_fit(model, x, y, p0=[0.4, 0.4, 0], constraints=cons)
```

When a bound or constraint is **active** at the optimum, the covariance is
projected onto the active-constraint nullspace (pounce's reduced Hessian does
exactly this), and the affected parameter is flagged in `res.active_mask` with
an effectively degenerate confidence interval. `res.cov_source` reports
`"reduced_hessian(projected)"` in that case.

## Data sensitivity: `∂params/∂data`

Pass `sensitivity=True` to get `res.dpopt_ddata`, an `(n_params, n_data)`
matrix whose entry `[j, i]` is how fitted parameter `j` moves when data point
`y_i` is perturbed. This is the implicit-function-theorem influence
`∂p*/∂y_i = 2 wᵢ² · H_S⁻¹ gᵢ`, computed as a single batched back-solve against
the converged factor (`Solver.kkt_solve_many`).

```python
res = pounce.curve_fit(model, x, y, p0=[1, 1, 0], sensitivity=True)
db = res.dpopt_ddata[1]              # sensitivity of parameter b
i = int(np.abs(db).argmax())         # most influential point for b
print("most influential x:", x[i])
```

## The result object

`CurveFitResult` carries everything in one place and supports dict-style access
(`res["popt"]`).

| field | meaning |
|---|---|
| `popt`, `pcov`, `perr`, `ci` | parameters, covariance, std errors, confidence intervals |
| `correlation` | normalized covariance |
| `residuals`, `sse`, `rmse`, `mae` | fit residuals and error norms |
| `r_squared`, `adj_r_squared` | coefficient(s) of determination |
| `chi_square`, `reduced_chi_square`, `dof` | χ² statistics and degrees of freedom |
| `param_names` | parameter names inferred from the model signature |
| `active_mask` | which parameters sit on a bound |
| `cov_source` | how the covariance was computed |
| `dpopt_ddata` | data sensitivity (if requested) |
| `optimize_result` | the raw solver info dict |

Methods: `res.predict(xnew)`, `res.confidence_band(...)` (see below), and
`res.summary()` (a formatted report).

## Confidence vs prediction bands

`res.confidence_band(x, kind=..., sigma=...)` returns `(yhat, lower, upper)`,
but there are **two different bands** and they answer different questions.

- **Confidence band** (`kind="confidence"`, the default) — uncertainty in the
  **fitted curve** itself, i.e. where the true mean `E[y | x]` lies. Its
  variance is `gᵀ Σ g` (delta method, `g = ∂f/∂p`, `Σ = pcov`). It is *narrow*,
  it **shrinks toward zero** as you collect more data, and **most data points
  fall outside it** — that is correct, not a miscalibration.

- **Prediction band** (`kind="prediction"`) — uncertainty in a **new
  observation** `y = f(x) + ε`. It adds the observation-noise variance:
  `gᵀ Σ g + σ²(x)`. This is the band that contains about `1 − alpha` of the
  *data*; it does **not** shrink to zero, it floors at the noise level.

Both use the Student-t quantile `t_{dof, 1−α/2}` (not the normal `z`), so the
degrees of freedom are accounted for.

```python
yhat, lo, hi = res.confidence_band(xx)                       # band on the curve
yhat, lo, hi = res.confidence_band(xx, kind="prediction")    # band on new data
```

For the prediction band the noise level `σ(x)` is taken from the fit: the
`sigma` weights you supplied, scaled by the fitted variance `s²` (so a
heteroscedastic fit gives a **heteroscedastic** band — wider where the noise is
larger), or the homoscedastic level `√s²` if the fit was unweighted. Pass an
explicit `sigma=` (scalar or array over `x`) to override it, e.g. for new `x`
where you know the measurement noise.

> Rule of thumb: use the **confidence** band to show how well the *model* is
> pinned down; use the **prediction** band to show where the *next measurement*
> will land. If "~95% of my points should be inside," you want the prediction
> band.

## Out-of-core data: `curve_fit_streaming`

When the dataset is too large to hold in memory, `pounce.curve_fit_streaming`
fits **exactly the same model and objective** as `curve_fit`, but reads the data
in **mini-batches** instead of as in-memory arrays. The solver's objective,
gradient, and Gauss-Newton Hessian are all *additive sums over data points*, so
streaming and accumulating them produces the **identical** fit — only one batch
(plus an `n_params × n_params` matrix) is ever resident.

Instead of `xdata, ydata` you pass a `data_source`: a **zero-argument callable**
(a factory) that returns a *fresh* iterator of `(x_batch, y_batch)` — or
`(x_batch, y_batch, sigma_batch)` — tuples. It is called once per solver pass,
so it must yield the full dataset every time (re-open the file, re-slice the
mmap, …); a one-shot iterator is rejected.

```python
import numpy as np
import pounce

# 50M points living on disk — re-read in 100k-row batches each pass
x_mm = np.load("x.npy", mmap_mode="r")
y_mm = np.load("y.npy", mmap_mode="r")
BATCH = 100_000

def data_source():                       # fresh iterator every call
    for i in range(0, x_mm.shape[0], BATCH):
        yield x_mm[i : i + BATCH], y_mm[i : i + BATCH]

res = pounce.curve_fit_streaming(model, data_source, p0=[1, 1, 0])
print(res.summary())
res.popt, res.pcov, res.perr             # identical to the in-memory fit
```

Notes and trade-offs:

- **Re-readable, not one-shot.** Each solver iteration (~10–50) makes one pass
  over `data_source`, so it must replay the whole dataset on every call. Uniform
  batch sizes avoid an extra JAX retrace on a smaller final batch.
- **Provide `p0`.** The data-driven seed `curve_fit` uses needs a full in-memory
  pass, so give a starting vector. With only `n_params` the seed falls back to
  ones clipped into `bounds`. If the model signature doesn't name the parameters
  and you omit `p0`, pass `n_params=`.
- **What you get back is the same** — all scalar diagnostics (SSE, χ², R², dof)
  and the full covariance / standard errors / confidence intervals are computed
  and are bit-for-bit the in-memory result. Everything else carries over too:
  weighted fits (`sigma` batches), robust `loss` (the sandwich covariance is
  accumulated over batches), `bounds`, and `constraints` (active sets project the
  covariance exactly as in the in-memory fit).
- **What is omitted** — the two `O(n_data)` outputs are *not* returned:
  `res.residuals` and the data sensitivity `res.dpopt_ddata` are both `None`
  (they are the size of the data and would defeat the purpose). `confidence_band`
  still works for new `x`, but uses a homoscedastic noise level since the
  per-point `sigma` is not retained.

## Multiple parameter sets: `curve_fit_minima`

Nonlinear least squares is generally non-convex, so the objective `curve_fit`
minimizes can have **several local minima** — distinct parameter sets that each
explain the data (peak-assignment ambiguity, frequency aliasing in sinusoids,
amplitude/decay trade-offs in sums of exponentials, sign/label symmetry, …).
`pounce.curve_fit_minima` drives [`find_minima`](find-minima.md) over *exactly*
the same objective — same `sigma` weighting, robust `loss`, `f_scale`,
`constraints`, and resolved Jacobian — to enumerate those minima, then refines
each into a full `CurveFitResult`:

```python
fits = pounce.curve_fit_minima(
    model, x, y,
    bounds=[(0, 3), (-10, 10), (0.1, 2.5)],  # finite bounds = the search box
    method="multistart",   # or "deflation" | "flooding" | "mlsl" | ...
    n_minima=5,
    seed=0,
)

for r in fits:               # ranked best (lowest SSE) first
    print(r.popt, r.sse, r.r_squared)
fits[0].summary()            # each is a full CurveFitResult
```

It reuses everything `curve_fit` does: the data-driven seed becomes the
search's starting point, the model Jacobian is reused as the search **gradient**
and the Gauss-Newton matrix as the search **Hessian** — which sharpens the basin
escapes and lets `find_minima` certify each point as a true minimum (rejecting
saddles) before recording it. The returned list is ranked by SSE and may contain
fewer than `n_minima` entries when the landscape has fewer minima.

> Finite `bounds` are strongly recommended — they define the box the search
> samples / repels within. With the default unbounded box the search degrades to
> jittered restarts around the seed. The `method`, `n_minima`, `max_solves`,
> `patience`, `dedup`, and `seed` arguments pass straight through to
> `find_minima`; see [Finding Multiple Minima](find-minima.md) and
> [Choosing a Method](find-minima-choosing.md).

See `python/examples/curve_fit_demo.py` and the
[`22_curve_fit.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/22_curve_fit.ipynb)
and
[`23_curve_fit_minima.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/23_curve_fit_minima.ipynb)
notebooks for complete, runnable walkthroughs.
