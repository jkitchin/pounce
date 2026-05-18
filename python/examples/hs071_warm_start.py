"""HS071 solved cold, then warm-started from the cold solution.

Re-running the same NLP from a primal point close to the optimum
saves iterations. Enable `warm_start_init_point=yes` to also forward
the dual seeds (`lagrange`, `zl`, `zu`) through to the iterate
initializer.
"""

import numpy as np

import pounce


class HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array([
            x[1] * x[2] * x[3], x[0] * x[2] * x[3],
            x[0] * x[1] * x[3], x[0] * x[1] * x[2],
            2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3],
        ])


def make_problem():
    p = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    return p


def main():
    cold_x, cold_info = make_problem().solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    print(f"cold: status={cold_info['status_msg']}, iters={cold_info['iter_count']}")

    warm_prob = make_problem()
    warm_prob.add_option("warm_start_init_point", "yes")
    warm_x, warm_info = warm_prob.solve(
        x0=cold_x,
        lagrange=np.asarray(cold_info["mult_g"]),
        zl=np.asarray(cold_info["mult_x_L"]),
        zu=np.asarray(cold_info["mult_x_U"]),
    )
    print(f"warm: status={warm_info['status_msg']}, iters={warm_info['iter_count']}")
    print(f"      Δobj   = {abs(warm_info['obj_val'] - cold_info['obj_val']):.2e}")
    print(f"      Δx     = {np.max(np.abs(warm_x - cold_x)):.2e}")


if __name__ == "__main__":
    main()
