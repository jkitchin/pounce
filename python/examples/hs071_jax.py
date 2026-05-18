"""HS071 via JAX AD — no hand-coded gradient / Jacobian / Hessian."""

import jax.numpy as jnp
import numpy as np

from pounce.jax import from_jax


def f(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def g(x):
    return jnp.stack([jnp.prod(x), jnp.dot(x, x)])


def main():
    prob = from_jax(
        f, g,
        n=4, m=2,
        lb=np.array([1.0] * 4),
        ub=np.array([5.0] * 4),
        cl=np.array([25.0, 40.0]),
        cu=np.array([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    print(f"status: {info['status_msg']}")
    print(f"obj:    {info['obj_val']:.6f}")
    print(f"x:      {x}")


if __name__ == "__main__":
    main()
