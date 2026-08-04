"""Adversary cross-check: auto-routing on a separable box-constrained (diagonal) QP
Family: autoroute   Class: pure box QP -- diagonal P, NO general linear
    constraints at all (bounds only). Fresh class for autoroute -- prior
    autoroute runs used QP (convex_qp, eqcon_qp, indefinite-box-qp REFUSAL,
    boxed-qp, nonunique-optimal-face), QCQP->socp, LP->lp-ipm, and NLP-only
    traps; none exercised a plain CONVEX box QP with lb<=x<=ub and no A/G
    rows at all, so the router's choice between the specialized box-clip
    path, qp-ipm, and qp-active-set on a trivial-shape problem is untested.
Source: constructed problem, separable per-coordinate closed form. For
    diagonal P=diag(d), minimize 0.5*d_i*x_i^2 + c_i*x_i s.t. lb_i<=x_i<=ub_i,
    each coordinate is independent: the unconstrained minimizer is
    x_i_unc = -c_i/d_i, and the constrained optimum is the projection
    x_i* = clip(x_i_unc, lb_i, ub_i). This closed-form projection is
    independent of both pounce and its own IPM/active-set code paths.

Data (n=5): d = (2,3,1,4,2), c = (-6,3,-1,20,-8), lb=(-1,-1,-1,-1,-1),
ub=(5,5,5,5,5). Unconstrained minimizers: -c/d = (3, -1, 1, -5, 4).
Clipped to [-1,5]: x* = (3, -1, 1, -1, 4) -- coordinates 0 and 4 interior,
coordinate 2 interior, coordinate 1 exactly at lb (unconstrained minimizer
already equals lb), coordinate 3 clipped at lb (-5 -> -1). A genuine MIXED
active/inactive case (2 of 5 bound-active) for the router to handle correctly.
"""
import time
import numpy as np
import pounce

d = np.array([2.0, 3.0, 1.0, 4.0, 2.0])
cc = np.array([-6.0, 3.0, -1.0, 20.0, -8.0])
lb = np.full(5, -1.0)
ub = np.full(5, 5.0)
n = 5

x_unc = -cc / d
X_STAR = np.clip(x_unc, lb, ub)
KNOWN_OPTIMAL = float(np.sum(0.5 * d * X_STAR ** 2 + cc * X_STAR))
active_lb = np.isclose(X_STAR, lb)
active_ub = np.isclose(X_STAR, ub)
print(f"closed-form: x_unc={x_unc} x*={X_STAR} obj={KNOWN_OPTIMAL:.10f} "
      f"active_lb={active_lb} active_ub={active_ub}")

fun = lambda x: float(0.5 * np.sum(d * x ** 2) + cc @ x)
jac = lambda x: d * x + cc
bounds = list(zip(lb, ub))
x0 = np.zeros(n)

t0 = time.perf_counter()
r_auto = pounce.minimize(fun, x0=x0, jac=jac, bounds=bounds, solver_selection="auto")
t_auto = time.perf_counter() - t0

t0 = time.perf_counter()
r_nlp = pounce.minimize(fun, x0=x0, jac=jac, bounds=bounds, solver_selection="nlp")
t_nlp = time.perf_counter() - t0

routed_solver = r_auto.info.get("solver") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "solver", None)
problem_class = r_auto.info.get("problem_class") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "problem_class", None)

x_auto = np.asarray(r_auto.x, float)
x_nlp = np.asarray(r_nlp.x, float)

# --- cross-check: same box QP via solve_qp (independent QP entry point) ---
P = np.diag(d)
r_qp = pounce.solve_qp(P=P, c=cc, lb=lb, ub=ub)
x_qp = np.asarray(r_qp.x)


def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))


auto_vs_nlp = float(np.linalg.norm(x_auto - x_nlp, np.inf))
auto_vs_known = float(np.linalg.norm(x_auto - X_STAR, np.inf))
auto_vs_qp = float(np.linalg.norm(x_auto - x_qp, np.inf))
obj_err_auto = rel(r_auto.fun, KNOWN_OPTIMAL)
obj_err_nlp = rel(r_nlp.fun, KNOWN_OPTIMAL)

print("=== pounce.minimize (solver_selection='auto') ===")
print(f"routed_solver={routed_solver} problem_class={problem_class}")
print(f"success={r_auto.success} obj={r_auto.fun:.10e} x={x_auto} t={t_auto:.4f}s")
print("=== pounce.minimize (solver_selection='nlp', forced) ===")
print(f"success={r_nlp.success} obj={r_nlp.fun:.10e} x={x_nlp} t={t_nlp:.4f}s")
print("=== pounce.solve_qp (independent QP entry point) ===")
print(f"status={r_qp.status} obj={r_qp.obj:.10e} x={x_qp}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"auto_vs_nlp_inf={auto_vs_nlp:.2e} auto_vs_known_inf={auto_vs_known:.2e} "
      f"auto_vs_qp_inf={auto_vs_qp:.2e} obj_err_auto={obj_err_auto:.2e} obj_err_nlp={obj_err_nlp:.2e}")

routed_to_qp = problem_class in ("qp", "convex_qp") or (routed_solver in ("qp-ipm", "qp-active-set"))
ok = (r_auto.success and r_nlp.success and auto_vs_nlp < 1e-5 and auto_vs_known < 1e-4
      and obj_err_auto < 1e-6 and obj_err_nlp < 1e-6 and routed_to_qp)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (auto_vs_nlp={auto_vs_nlp:.2e}, auto_vs_known={auto_vs_known:.2e}, "
      f"obj_err_auto={obj_err_auto:.2e}, routed_solver={routed_solver}, problem_class={problem_class})")
if not routed_to_qp:
    print(f"NOTE: auto-route did not select the specialized QP path (routed_solver="
          f"{routed_solver}, problem_class={problem_class}) -- if the answer still "
          f"matches the closed form this is a performance/routing note, not a correctness bug.")
