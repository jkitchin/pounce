"""Adversary cross-check: L2-norm (group-lasso) proximal operator as an
SOCP with a QUADRATIC objective term combined with the SOC cone.
Family: socp   Class: P + cone (quadratic objective term routed through
solve_socp's non-symmetric HSDE driver alongside a plain SOC block) --
distinct from prior socp probes, which all used c-only (linear) objectives.
Source: Parikh & Boyd, "Proximal Algorithms" (2014), sec 6.5.3 (block/
group soft-thresholding); Boyd, Parikh, Chu, Peleato, Eckstein, ADMM
monograph (2011) sec 6.2. The proximal operator of lambda*||.||_2 is

    prox_{lambda||.||_2}(b) = b * max(0, 1 - lambda/||b||_2)

which is exactly the solution of

    minimize 0.5*||x - b||_2^2 + lambda*||x||_2

Reformulated (dropping the constant 0.5*||b||_2^2, which both solvers
below also drop, so it cancels out of every objective comparison) as an
SOCP over z = (x_1..x_n, t):

    minimize    0.5*x'x - b'x + lambda*t
    subject to  (t, x) in SOC   [t >= ||x||_2]

P = blkdiag(I_n, 0), c = [-b; lambda], and the cone constraint is encoded
via G = -Perm, h = 0 where Perm moves the last coordinate (t) to the
front: s = h - G z = Perm z = (t, x).

Two cases in one run:
  (a) lambda = 0.3*||b||_2  -> shrinkage active, x* != 0 (interior optimum)
  (b) lambda = 1.5*||b||_2  -> x* = 0 exactly, t* = 0: the SOC constraint
      sits at the APEX of the cone (a degenerate boundary case), the kind
      of corner the family notes call out as the encoding trap.
"""
import time
import numpy as np

np.random.seed(7)
n = 5
b = np.random.uniform(-2.0, 2.0, n)
b_norm = np.linalg.norm(b)

from pounce import solve_socp
import cvxpy as cp


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


def run_case(label, lam):
    x_star = b * max(0.0, 1.0 - lam / b_norm)
    t_star = float(np.linalg.norm(x_star))
    # shifted objective (constant 0.5*b'b dropped), evaluated at the
    # closed-form optimum
    known_shifted_obj = 0.5 * x_star @ x_star - b @ x_star + lam * t_star

    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = np.eye(n)
    c = np.concatenate([-b, [lam]])

    Perm = np.zeros((n + 1, n + 1))
    Perm[0, n] = 1.0
    for i in range(1, n + 1):
        Perm[i, i - 1] = 1.0
    G = -Perm
    h = np.zeros(n + 1)

    t0 = time.perf_counter()
    r = solve_socp(P=P, c=c, G=G, h=h, cones=[("soc", n + 1)])
    t_pounce = time.perf_counter() - t0
    z = np.asarray(r.x)
    x_pounce, t_pounce_val = z[:n], z[n]
    obj_pounce = r.obj
    status = r.status

    xv = cp.Variable(n)
    tv = cp.Variable()
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(xv) - b @ xv + lam * tv),
        [cp.SOC(tv, xv)],
    )
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    t_oracle = time.perf_counter() - t0
    x_oracle, obj_oracle = xv.value, prob.value

    obj_err = rel(obj_pounce, obj_oracle)
    known_err = rel(obj_pounce, known_shifted_obj)
    x_err = float(np.linalg.norm(x_pounce - x_star, np.inf))
    x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

    print(f"=== case {label}: lambda={lam:.6f} (||b||={b_norm:.6f}) ===")
    print(f"pounce: status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
    print(f"oracle: obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
    print(f"closed_form: obj={known_shifted_obj:.10e} x*={x_star}")
    print(f"obj_err_vs_oracle={obj_err:.2e} obj_err_vs_known={known_err:.2e} "
          f"x_inf_err_vs_known={x_err:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

    ok = status == "optimal" and obj_err < 1e-4 and known_err < 1e-4 and x_err < 1e-4
    print(f"case {label} verdict: {'PASS' if ok else 'FAIL'}")
    return ok


ok_a = run_case("a (shrinkage active)", 0.3 * b_norm)
ok_b = run_case("b (boundary, x*=0)", 1.5 * b_norm)

ok = ok_a and ok_b
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (case_a={ok_a}, case_b={ok_b})")
