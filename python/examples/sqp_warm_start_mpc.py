"""SQP working-set warm start across a parametric NLP sweep.

This is a minimal-fixture demonstration of pounce's Phase 5c §7.3
API surface:

   x*(p) = argmin ½‖x − p‖²  s.t.  x ≥ 0,  ∑ x_i = 1

(simplex projection — the canonical "active set rotates with p"
test case). At every step we run two solves:

  1. **Cold** — `Problem.solve(x0)` with no warm-start data.
  2. **Warm** — `Problem.solve(x0, working_set=ws)` where `ws`
     is the (bounds, constraints) tuple from the previous step's
     `info["working_set"]`.

For a convex quadratic objective + linear constraints both modes
land at the optimum in **one SQP outer iteration** — the QP
subproblem IS the original problem. The interesting warm-start
speedup happens *inside* the QP solver (fewer active-set
changes), which the per-solve iter-count column doesn't surface.
This example is therefore an API-correctness demo, not a
performance benchmark. For the latter, point at a sequence of
non-trivially nonlinear NLPs (§8.5 of the design note).

Run with:

    python python/examples/sqp_warm_start_mpc.py

Requires `pip install pounce` (or `maturin develop` against the
workspace).
"""

import numpy as np

import pounce


class SimplexProjection:
    """min ½ ‖x − p‖² s.t. x ≥ 0, sum(x) = 1.

    With n variables and 1 equality constraint. The bound x ≥ 0
    is the warm-start-relevant active set; the equality always
    binds.
    """

    def __init__(self, n):
        self.n = n
        self.p = np.zeros(n)

    def set_parameter(self, p):
        self.p = np.asarray(p, dtype=np.float64)

    def objective(self, x):
        d = x - self.p
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self.p

    def constraints(self, x):
        return np.array([float(np.sum(x))])

    def jacobianstructure(self):
        return (np.zeros(self.n, dtype=np.int64),
                np.arange(self.n, dtype=np.int64))

    def jacobian(self, x):
        return np.ones(self.n)

    def hessianstructure(self):
        # Diagonal Hessian.
        idx = np.arange(self.n, dtype=np.int64)
        return (idx, idx)

    def hessian(self, x, lagrange, obj_factor):
        return np.full(self.n, obj_factor)


def make_problem(prob_obj, n):
    p = pounce.Problem(
        n=n, m=1, problem_obj=prob_obj,
        lb=[0.0] * n, ub=[1e20] * n,
        cl=[1.0], cu=[1.0],
    )
    p.add_option("algorithm", "active-set-sqp")
    p.add_option("print_level", 0)
    p.add_option("sqp_tol", 1e-9)
    return p


def main():
    np.random.seed(0)
    n = 8
    n_steps = 20

    # Rotating parameter — a smooth path around the simplex centre.
    centre = np.full(n, 1.0 / n)
    direction = np.random.randn(n)
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    radius = 0.2

    cold_iters = []
    warm_iters = []
    last_ws = None

    print(f"{'step':>4} {'cold iters':>10} {'warm iters':>10}  "
          f"{'‖x* − x*_warm‖':>16}")
    print("-" * 50)

    for k in range(n_steps):
        theta = 2.0 * np.pi * k / n_steps
        p_k = centre + radius * np.cos(theta) * direction

        # --- Cold solve (no warm-start). ---
        cold_obj = SimplexProjection(n)
        cold_obj.set_parameter(p_k)
        cold_prob = make_problem(cold_obj, n)
        x_cold, info_cold = cold_prob.solve(x0=centre.copy())
        assert info_cold["status_msg"] == "Solve_Succeeded", \
            f"cold solve failed at step {k}: {info_cold['status_msg']}"
        cold_iters.append(info_cold["iter_count"])

        # --- Warm solve (carrying working set from previous step). ---
        warm_obj = SimplexProjection(n)
        warm_obj.set_parameter(p_k)
        warm_prob = make_problem(warm_obj, n)
        kwargs = {}
        if last_ws is not None:
            kwargs["working_set"] = last_ws
        x_warm, info_warm = warm_prob.solve(x0=centre.copy(), **kwargs)
        assert info_warm["status_msg"] == "Solve_Succeeded", \
            f"warm solve failed at step {k}: {info_warm['status_msg']}"
        warm_iters.append(info_warm["iter_count"])
        last_ws = info_warm["working_set"]

        dx = float(np.linalg.norm(x_warm - x_cold))
        print(f"{k:>4} {cold_iters[-1]:>10} {warm_iters[-1]:>10}  "
              f"{dx:>16.3e}")

    print("-" * 50)
    print(f"mean iter count: cold = {np.mean(cold_iters):.2f}, "
          f"warm = {np.mean(warm_iters):.2f}  "
          f"(speedup = {np.mean(cold_iters) / max(np.mean(warm_iters), 1e-9):.2f}x)")


if __name__ == "__main__":
    main()
