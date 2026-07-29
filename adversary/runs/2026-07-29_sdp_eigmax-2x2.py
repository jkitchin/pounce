"""Adversary cross-check: maximize the min-eigenvalue of an affine matrix pencil
Family: sdp   Class: eigenvalue optimization SDP (Boyd & Vandenberghe sec 3.1.5/4.6.3)
Source: for a 2x2 symmetric affine pencil A(x) = A0 + x*A1, lambda_min(A(x)) is
        concave in x (pointwise min of v^T A(x) v over unit v, each linear in x).
        Maximizing it is the classic SDP epigraph:
            maximize t   s.t.  A(x) - t*I  (psd)
        A0=[[2,1],[1,3]], A1=[[1,0],[0,-1]]  =>  A(x)=[[2+x,1],[1,3-x]].
        Eigenvalues of a 2x2 symmetric matrix [[a,b],[b,d]]:
            lambda_min = (a+d)/2 - sqrt(((a-d)/2)^2 + b^2)
        Here a+d = 5 (constant in x!), (a-d)/2 = x-0.5, b=1, so
            f(x) = 2.5 - sqrt((x-0.5)^2 + 1)
        which is maximized exactly where the sqrt term is minimized: x*=0.5,
        giving t* = f(0.5) = 2.5 - 1 = 1.5.
Known optimal: x* = 0.5, t* = 1.5
"""
import time
import numpy as np

A0 = np.array([[2.0, 1.0], [1.0, 3.0]])
A1 = np.array([[1.0, 0.0], [0.0, -1.0]])

KNOWN_X = 0.5
KNOWN_T = 1.5


def Amat(x):
    return A0 + x * A1


# sanity: closed form matches direct eigendecomposition on a small grid
grid = np.linspace(-3, 3, 20001)
lmin = np.array([np.linalg.eigvalsh(Amat(x))[0] for x in grid])
grid_argmax = grid[np.argmax(lmin)]
assert abs(grid_argmax - KNOWN_X) < 2e-4, f"grid check failed: argmax={grid_argmax}"
assert abs(lmin.max() - KNOWN_T) < 1e-3

# --- pounce: solve_socp with a single 2x2 psd cone, variables (x, t) ---
from pounce import solve_socp

SQRT2 = np.sqrt(2.0)
c = np.array([0.0, -1.0])  # minimize -t == maximize t
G = np.array([
    [-1.0, 1.0],   # s0 = 2 + x - t
    [0.0, 0.0],    # s1 = sqrt(2) * A(x)[1,0]  (constant, off-diag is x-independent)
    [1.0, 1.0],    # s2 = 3 - x - t
])
h = np.array([2.0, SQRT2, 3.0])
cones = [("psd", 2)]

t0 = time.perf_counter()
res = solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce, t_pounce_val = res.x[0], res.x[1]
status = res.status

# verify the returned point is genuinely psd-feasible (not just objective-close)
X = Amat(x_pounce) - t_pounce_val * np.eye(2)
eigs = np.linalg.eigvalsh(X)
min_eig_slack = eigs.min()

# --- oracle 1: cvxpy SDP, independent PSD-cone construction ---
import cvxpy as cp

xv = cp.Variable()
tv = cp.Variable()
A_expr = A0 + xv * A1
constraints = [A_expr - tv * np.eye(2) >> 0]
prob = cp.Problem(cp.Maximize(tv), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
x_cvx, t_cvx_val = xv.value, tv.value

# --- oracle 2: scipy scalar minimization of -f(x) (independent, closed-form f) ---
from scipy.optimize import minimize_scalar

def neg_f(x):
    return -(2.5 - np.sqrt((x - 0.5) ** 2 + 1))

t0 = time.perf_counter()
res_scipy = minimize_scalar(neg_f, bracket=(-2, 0.5, 3), method="brent",
                             options={"xtol": 1e-14})
t_scipy = time.perf_counter() - t0
x_scipy = res_scipy.x
t_scipy_val = -res_scipy.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


x_err_known = abs(x_pounce - KNOWN_X)
t_err_known = rel(t_pounce_val, KNOWN_T)
x_err_cvx = abs(x_pounce - x_cvx)
t_err_cvx = rel(t_pounce_val, t_cvx_val)
x_err_scipy = abs(x_pounce - x_scipy)
t_err_scipy = rel(t_pounce_val, t_scipy_val)

print("=== pounce (solve_socp, psd cone n=2) ===")
print(f"status={status} x={x_pounce:.10f} t={t_pounce_val:.10f} t_solve={t_pounce:.4f}s")
print(f"  feasibility check: eig(A(x)-tI)={eigs} (min={min_eig_slack:.2e}, must be >= ~0)")
print("=== oracle: cvxpy CLARABEL (native PSD cone via '>>' operator) ===")
print(f"x={x_cvx:.10f} t={t_cvx_val:.10f} t_solve={t_cvx:.4f}s")
print("=== oracle: scipy Brent (closed-form eigenvalue expression) ===")
print(f"x={x_scipy:.10f} t={t_scipy_val:.10f} t_solve={t_scipy:.4f}s")
print(f"known_optimal: x*={KNOWN_X} t*={KNOWN_T}")
print(f"x_err_vs_known={x_err_known:.2e} t_err_vs_known={t_err_known:.2e}")
print(f"x_err_vs_cvxpy={x_err_cvx:.2e} t_err_vs_cvxpy={t_err_cvx:.2e}")
print(f"x_err_vs_scipy={x_err_scipy:.2e} t_err_vs_scipy={t_err_scipy:.2e}")

ok = (
    status == "optimal"
    and min_eig_slack > -1e-6
    and x_err_known < 1e-4
    and t_err_known < 1e-6
    and t_err_cvx < 1e-6
    and t_err_scipy < 1e-6
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
