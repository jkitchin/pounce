"""Adversary cross-check: GAMS link marginal (dual) sign convention.

Family: nlp (GAMS link surface)   Class: duals / multipliers / sign invariants
Source: analytic LP shadow prices + the GAMS marginal convention, established
        empirically with GAMS 53.2.0 (CPLEX/LP) in this run's .org report.

Drives BOTH sides of the pip link's translation layer with an in-memory fake
GmoView (no GAMS license needed), exactly the way python/tests/test_gams_link.py
does, and then reproduces -- verbatim -- the sign arithmetic that
pounce.gams.link._write_solution applies before handing values to GMO:

    pi        = -mult_g                                (link.py:361-365)
    var_marg  = (mult_x_L - mult_x_U), negated if max  (link.py:370-377)
    obj       = obj_sign * obj_val                     (link.py:353-354)

The native C link does the identical arithmetic at gams/gams_pounce.c:1096-1118
and :1081, so this harness characterises both links.

Test LP (a textbook resource-allocation LP), run in BOTH senses:

    MAX form:  max  3 x1 + 5 x2 + 1 x3    s.t.  x1 + 2 x2 + x3 <= 10   (r1)
                                                3 x1 + 2 x2 + x3 <= 12 (r2)
                                                x >= 0
      optimum x = (1, 4.5, 0), obj = 25.5
      analytic shadow prices  dObj/d(rhs):  r1 = +2.25, r2 = +0.25
      analytic reduced cost   x3:           -1.5

    MIN form:  min  3 y1 + 5 y2           s.t.  y1 + 2 y2 >= 10   (s1)
                                                3 y1 + 2 y2 >= 12 (s2)
      optimum y = (1, 4.5), obj = 25.5
      analytic shadow prices  dObj/d(rhs):  s1 = +2.25, s2 = +0.25

GAMS's marginal convention (verified live, see the .org report) is
dObj/d(rhs) in the model's OWN sense: GAMS/CPLEX reports +2.25 / +0.25 for
BOTH of these models, and -1.5 for x3 in the max model.

Never modifies pounce source; imports it read-only.
"""

import numpy as np

from pounce.gams.gmo_translate import POUNCE_INF, problem_from_gmo

# ── analytic oracle ──────────────────────────────────────────────────────────
MAX_SHADOW = {"r1": 2.25, "r2": 0.25}   # dObj/d(rhs), GAMS convention
MAX_REDCOST_X3 = -1.5
MIN_SHADOW = {"s1": 2.25, "s2": 0.25}


class LPView:
    """Fake GmoView for a 3-var / 2-row LP, in either objective sense.

    Same shape as the HS071View fake in python/tests/test_gams_link.py: values
    are returned in the model's NATIVE sense; the max/min flip is the
    translator's job.
    """

    def __init__(self, maximize: bool):
        self._max = maximize
        if maximize:
            # max 3x1+5x2+1x3 s.t. Ax <= b
            self._c = np.array([3.0, 5.0, 1.0])
            self._A = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
            self._cl = [-POUNCE_INF, -POUNCE_INF]
            self._cu = [10.0, 12.0]
            self._n = 3
        else:
            # min 3y1+5y2 s.t. Ay >= b
            self._c = np.array([3.0, 5.0])
            self._A = np.array([[1.0, 2.0], [3.0, 2.0]])
            self._cl = [10.0, 12.0]
            self._cu = [POUNCE_INF, POUNCE_INF]
            self._n = 2

    def name(self):
        return "adversary_lp"

    def num_vars(self):
        return self._n

    def num_cons(self):
        return 2

    def maximize(self):
        return self._max

    def has_hessian(self):
        return True  # LP: Hessian is structurally empty

    def var_lower(self):
        return [0.0] * self._n

    def var_upper(self):
        return [POUNCE_INF] * self._n

    def var_init(self):
        return [0.5] * self._n

    def con_lower(self):
        return self._cl

    def con_upper(self):
        return self._cu

    def jac_structure(self):
        rows = [i for i in range(2) for _ in range(self._n)]
        cols = list(range(self._n)) * 2
        return rows, cols

    def hess_structure(self):
        return [], []

    def eval_obj(self, x):
        return float(self._c @ x)

    def eval_grad_obj(self, x):
        return self._c.tolist()

    def eval_cons(self, x):
        return (self._A @ x).tolist()

    def eval_jac(self, x):
        return self._A.reshape(-1).tolist()

    def hess_lag_value(self, x, lam, obj_weight, con_weight):
        return []


def run(maximize: bool):
    import pounce

    view = LPView(maximize)
    gp = problem_from_gmo(view)
    prob = pounce.Problem(
        n=gp.n, m=gp.m, problem_obj=gp.problem_obj,
        lb=gp.lb, ub=gp.ub, cl=gp.cl, cu=gp.cu,
    )
    prob.add_option("acceptable_iter", 0)
    prob.add_option("tol", 1e-10)
    x, info = prob.solve(x0=gp.x0)

    obj_sign = gp.obj_sign
    # ── verbatim reproduction of link.py:353-377 / gams_pounce.c:1081-1118 ──
    gams_obj = obj_sign * float(info["obj_val"])
    pi = -np.asarray(info["mult_g"], dtype=float)
    var_marg = (np.asarray(info["mult_x_L"], dtype=float)
                - np.asarray(info["mult_x_U"], dtype=float))
    if obj_sign < 0.0:
        var_marg = -var_marg
    return x, gams_obj, pi, var_marg, info


def main():
    print("=" * 74)
    print("MAXIMIZE model  (analytic GAMS marginals: r1=+2.25 r2=+0.25 x3.m=-1.5)")
    print("=" * 74)
    x, obj, pi, vm, info = run(maximize=True)
    print(f"status           = {info['status_msg']}")
    print(f"x                = {np.round(x, 6)}   (expect [1, 4.5, 0])")
    print(f"objective -> GAMS= {obj:+.6f}          (expect +25.5)")
    print(f"equ marginals    = {np.round(pi, 6)}   (expect [+2.25, +0.25])")
    print(f"var marginals    = {np.round(vm, 6)}   (expect [0, 0, -1.5])")
    max_pi_ok = np.allclose(pi, [MAX_SHADOW["r1"], MAX_SHADOW["r2"]], atol=1e-5)
    max_vm_ok = np.allclose(vm, [0.0, 0.0, MAX_REDCOST_X3], atol=1e-5)
    max_obj_ok = abs(obj - 25.5) < 1e-5
    print(f"  -> equ marginal sign OK? {max_pi_ok}")
    print(f"  -> var marginal sign OK? {max_vm_ok}")
    print(f"  -> objective sign OK?    {max_obj_ok}")

    print()
    print("=" * 74)
    print("MINIMIZE model  (analytic GAMS marginals: s1=+2.25 s2=+0.25)")
    print("=" * 74)
    x, obj, pi, vm, info = run(maximize=False)
    print(f"status           = {info['status_msg']}")
    print(f"x                = {np.round(x, 6)}   (expect [1, 4.5])")
    print(f"objective -> GAMS= {obj:+.6f}          (expect +25.5)")
    print(f"equ marginals    = {np.round(pi, 6)}   (expect [+2.25, +0.25])")
    print(f"var marginals    = {np.round(vm, 6)}   (expect [0, 0])")
    min_pi_ok = np.allclose(pi, [MIN_SHADOW["s1"], MIN_SHADOW["s2"]], atol=1e-5)
    min_vm_ok = np.allclose(vm, [0.0, 0.0], atol=1e-5)
    min_obj_ok = abs(obj - 25.5) < 1e-5
    print(f"  -> equ marginal sign OK? {min_pi_ok}")
    print(f"  -> var marginal sign OK? {min_vm_ok}")
    print(f"  -> objective sign OK?    {min_obj_ok}")

    print()
    ok = all([max_pi_ok, max_vm_ok, max_obj_ok, min_pi_ok, min_vm_ok, min_obj_ok])
    if ok:
        print("VERDICT: PASS")
    else:
        bad = []
        if not max_pi_ok:
            bad.append("MAX equation marginals sign-inverted")
        if not max_vm_ok:
            bad.append("MAX variable marginals wrong")
        if not max_obj_ok:
            bad.append("MAX objective wrong")
        if not min_pi_ok:
            bad.append("MIN equation marginals wrong")
        if not min_vm_ok:
            bad.append("MIN variable marginals wrong")
        if not min_obj_ok:
            bad.append("MIN objective wrong")
        print("VERDICT: SOLVER_BUG (" + "; ".join(bad) + ")")


if __name__ == "__main__":
    main()
