"""HS071 — cyipopt's canonical example, ported to pounce.

    min   x1 x4 (x1 + x2 + x3) + x3
    s.t.  x1 x2 x3 x4 >= 25
          x1^2 + x2^2 + x3^2 + x4^2 = 40
          1 <= xi <= 5

The optimal solution is approximately
    x* ≈ (1.0000, 4.7430, 3.8211, 1.3794)
    f* ≈ 17.0140.
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
            x[1] * x[2] * x[3],
            x[0] * x[2] * x[3],
            x[0] * x[1] * x[3],
            x[0] * x[1] * x[2],
            2 * x[0],
            2 * x[1],
            2 * x[2],
            2 * x[3],
        ])

    def hessianstructure(self):
        # Lower triangle.
        return (
            np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
            np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
        )

    def hessian(self, x, lam, obj_factor):
        H_obj = np.zeros(10)
        # d²f / dx_i dx_j  on the lower triangle, in (i, j) order:
        # (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)
        H_obj[0] = 2 * x[3]
        H_obj[1] = x[3]
        H_obj[3] = x[3]
        H_obj[6] = 2 * x[0] + x[1] + x[2]
        H_obj[7] = x[0]
        H_obj[8] = x[0]
        H_obj[9] = 0.0

        H_c1 = np.zeros(10)
        H_c1[1] = x[2] * x[3]
        H_c1[3] = x[1] * x[3]
        H_c1[4] = x[0] * x[3]
        H_c1[6] = x[1] * x[2]
        H_c1[7] = x[0] * x[2]
        H_c1[8] = x[0] * x[1]

        H_c2 = np.zeros(10)
        H_c2[0] = 2.0
        H_c2[2] = 2.0
        H_c2[5] = 2.0
        H_c2[9] = 2.0

        return obj_factor * H_obj + lam[0] * H_c1 + lam[1] * H_c2


def main():
    prob = pounce.Problem(
        n=4,
        m=2,
        problem_obj=HS071(),
        lb=[1.0] * 4,
        ub=[5.0] * 4,
        cl=[25.0, 40.0],
        cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)

    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    print(f"status: {info['status_msg']}")
    print(f"obj:    {info['obj_val']:.6f}")
    print(f"x:      {x}")
    print(f"iters:  {info['iter_count']}")


if __name__ == "__main__":
    main()
