# pounce — Python interface

`pounce` is a Python wrapper around POUNCE, a pure-Rust port of the
[Ipopt](https://github.com/coin-or/Ipopt) interior-point nonlinear
programming solver. The Python surface area is intentionally
cyipopt-compatible: code written for cyipopt typically runs against
pounce by changing only the import.

## Install (development)

```sh
# from the repo root:
cd python
pip install maturin
maturin develop --release            # builds the native extension into your venv
# optional extras:
pip install -e .[jax]                # jax integration
pip install -e .[dev]                # tests + jax + scipy
```

## Quick start (cyipopt-style)

```python
import numpy as np
import pounce

class HS071:
    def objective(self, x):
        return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    def gradient(self, x):
        return np.array([
            x[0]*x[3] + x[3]*(x[0]+x[1]+x[2]),
            x[0]*x[3],
            x[0]*x[3] + 1.0,
            x[0]*(x[0]+x[1]+x[2]),
        ])
    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])
    def jacobianstructure(self):
        return (np.repeat([0,1], 4), np.tile([0,1,2,3], 2))
    def jacobian(self, x):
        return np.array([
            x[1]*x[2]*x[3], x[0]*x[2]*x[3], x[0]*x[1]*x[3], x[0]*x[1]*x[2],
            2*x[0], 2*x[1], 2*x[2], 2*x[3],
        ])

prob = pounce.Problem(
    n=4, m=2,
    problem_obj=HS071(),
    lb=[1]*4, ub=[5]*4,
    cl=[25, 40], cu=[2e19, 40],
)
prob.add_option('tol', 1e-8)
x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
print(info['status_msg'], info['obj_val'], x)
```

## scipy.optimize-style

```python
from pounce import minimize
res = minimize(lambda x: (x-1)**2 @ (x-1) + 1, x0=np.zeros(5))
print(res.fun, res.x)
```

## JAX integration

```python
import jax, jax.numpy as jnp
from pounce.jax import from_jax

def f(x): return jnp.sum((x-1)**2)
def g(x): return jnp.stack([jnp.sum(x) - 5.0])

prob = from_jax(f, g, n=4, m=1, lb=jnp.zeros(4), ub=jnp.full(4, 10.0),
                cl=jnp.zeros(1), cu=jnp.zeros(1))
x, info = prob.solve(x0=jnp.ones(4))
```

Differentiate through the solver:

```python
from pounce.jax import solve as psolve

def loss(p):
    x_star = psolve(p, f=..., g=..., n=4, m=1, ...)
    return jnp.sum(x_star ** 2)

dloss_dp = jax.grad(loss)(jnp.ones(4))
```
