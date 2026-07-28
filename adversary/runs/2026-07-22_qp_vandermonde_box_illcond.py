"""Adversary cross-check: box-constrained Vandermonde least-squares QP with a
Hessian of condition number ~3.6e10.

Family: qp   Class: box-constrained convex QP, extremely ill-conditioned Hessian
Dimension: adversary iteration "ill-conditioning & bad scaling"

PROBLEM
-------
Fit the coefficients a in R^7 of a degree-6 polynomial p(s) = sum_j a_j s^j to
data (t_i, y_i) at the 7 dyadic nodes t_i = i/8, i = 1..7, subject to bounds on
the coefficients:

    minimize_a   || V a - y ||_2^2        subject to   -10 <= a_j <= 10

with V the 7x7 Vandermonde matrix V_ij = t_i^j (increasing powers).  In pounce's
solve_qp form (objective 0.5 x'Px + c'x) this is

    P = 2 V'V,      c = -2 V'y,      lb = -10,  ub = +10

(the constant ||y||^2 is dropped by both pounce and the cvxpy model below, so
the two objectives are directly comparable; the full least-squares residual is
reported separately).

WHY THIS IS AN ILL-CONDITIONING TEST
------------------------------------
V is a Vandermonde matrix on equispaced nodes, the textbook exponentially
ill-conditioned matrix (cond_2(V) ~ 1.9e5 here); forming the normal-equation
Hessian squares that: cond_2(P) ~ 3.6e10, with eigenvalues spanning
~6e-10 .. ~21.  P is still symmetric positive definite, so the box-constrained
QP is strictly convex and has a UNIQUE optimum.  The unconstrained minimizer has
coefficients as large as ~784 in magnitude, so the box is strongly active and
the optimal active set is nontrivial.  This is exactly the regime where an IPM's
normal-equation / KKT factorization is stressed.

Reference for the construction: the Vandermonde matrix on equispaced nodes is
the canonical exponentially ill-conditioned matrix -- N. J. Higham, "Accuracy and
Stability of Numerical Algorithms", 2nd ed., SIAM 2002, Section 22.1 ("Vandermonde
systems"), and Gautschi, "How (un)stable are Vandermonde systems?", 1990.  The
box-constrained-coefficient least-squares form is the classic bounded-variable
least squares (BVLS) problem, Stark & Parker, BIT 35 (1995) 186-196.

EXACTNESS OF THE DATA (this is the crux)
----------------------------------------
For an ill-conditioned QP it is meaningless to compare pounce's answer against
the optimum of some *nearby* problem: a 1-ulp change in c moves the solution by
O(cond * eps) ~ 1e-5.  So all problem data here is chosen to be EXACTLY
representable in IEEE double:
  - nodes t_i = i/8 are dyadic, so t_i^j = i^j / 8^j is exact (i^6 <= 117649),
  - y_i = m_i/16 is dyadic,
  - every entry of P = 2 V'V and c = -2 V'y is a dyadic rational whose numerator
    fits well inside 2^53, so the float64 matrices are EXACT.
The script asserts this (float64 P, c == exact Fraction P, c).  Therefore the
exact rational optimum computed below is the optimum of *precisely* the data
pounce is handed -- no perturbation, no "nearby problem" excuse.

ORACLE 1 (primary): EXACT closed-form KKT in rational arithmetic.
For a box-constrained strictly convex QP the KKT conditions are necessary and
sufficient.  With n=7 there are 3^7 = 2187 candidate active-set assignments
(each variable at lb, free, or at ub).  We enumerate all of them in exact
Fraction arithmetic: for each assignment solve the reduced stationarity system
P_FF x_F = -(c_F + P_F,B x_B) exactly by fraction-free Gaussian elimination,
then check primal feasibility on the free block and the multiplier sign
conditions (grad_i >= 0 at lb, grad_i <= 0 at ub).  Exactly one assignment
satisfies all conditions (strict convexity => unique optimum), and it is
certified with zero floating-point arithmetic.  No iterative solver, no
conditioning error can enter.

ORACLE 2: cvxpy / CLARABEL and cvxpy / SCS (independent numerical solvers).

Budget: < 10 s.
"""

import time
from fractions import Fraction as F
from itertools import product

import numpy as np

N = 7
LO, HI = -10.0, 10.0

# ---------------------------------------------------------------- problem data
k = np.arange(1, N + 1)
t = k / 8.0                                   # dyadic nodes, exact in float64
V = np.vander(t, N, increasing=True)          # V_ij = t_i^j
y = np.round(16 * np.sin(3 * t)) / 16.0       # dyadic data values, exact

P = 2.0 * (V.T @ V)
P = (P + P.T) / 2.0                           # exact no-op, guards symmetry
c = -2.0 * (V.T @ y)
lb = np.full(N, LO)
ub = np.full(N, HI)
CONST = float(y @ y)                          # dropped constant ||y||^2

# ------------------------------------------------- assert the data is EXACT
Ve = [[F(int(ki), 8) ** j for j in range(N)] for ki in k]
ye = [F(int(round(16 * v)), 16) for v in y]
Pe = [[2 * sum(Ve[i][a] * Ve[i][b] for i in range(N)) for b in range(N)]
      for a in range(N)]
ce = [-2 * sum(Ve[i][a] * ye[i] for i in range(N)) for a in range(N)]
assert all(F(P[a, b]) == Pe[a][b] for a in range(N) for b in range(N)), \
    "P is not exactly representable in float64"
assert all(F(c[a]) == ce[a] for a in range(N)), \
    "c is not exactly representable in float64"
assert np.array_equal(P, P.T)

evals = np.linalg.eigvalsh(P)
COND = evals[-1] / evals[0]
assert evals[0] > 0, "P must be positive definite"

LOe, HIe = F(LO), F(HI)


# ---------------------------------------------- ORACLE 1: exact rational KKT
def exact_solve(Amat, rhs):
    """Exact Gaussian elimination with partial (nonzero) pivoting on Fractions."""
    m = len(rhs)
    A = [row[:] + [rhs[i]] for i, row in enumerate(Amat)]
    for col in range(m):
        piv = next((r for r in range(col, m) if A[r][col] != 0), None)
        if piv is None:
            return None                      # singular (cannot happen: P_FF PD)
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(m):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [A[r][j] - f * A[col][j] for j in range(m + 1)]
    return [A[i][m] for i in range(m)]


def exact_box_qp():
    """Enumerate all 3^N active sets; return the unique KKT point (exact)."""
    hits = []
    for assign in product((0, 1, 2), repeat=N):   # 0=at lb, 1=free, 2=at ub
        free = [i for i in range(N) if assign[i] == 1]
        xb = {i: (LOe if assign[i] == 0 else HIe)
              for i in range(N) if assign[i] != 1}
        if free:
            Aff = [[Pe[i][j] for j in free] for i in free]
            rhs = [-(ce[i] + sum(Pe[i][j] * xb[j] for j in xb)) for i in free]
            sol = exact_solve(Aff, rhs)
            if sol is None:
                continue
            if any(v < LOe or v > HIe for v in sol):
                continue
            x = [None] * N
            for idx, i in enumerate(free):
                x[i] = sol[idx]
            for i, v in xb.items():
                x[i] = v
        else:
            x = [xb[i] for i in range(N)]
        # multiplier signs from the gradient g = P x + c
        ok = True
        for i in range(N):
            if assign[i] == 1:
                continue
            g = sum(Pe[i][j] * x[j] for j in range(N)) + ce[i]
            if assign[i] == 0 and g < 0:
                ok = False
                break
            if assign[i] == 2 and g > 0:
                ok = False
                break
        if ok:
            hits.append((assign, x))
    return hits


t0 = time.perf_counter()
hits = exact_box_qp()
t_exact = time.perf_counter() - t0
assert len(hits) == 1, f"expected a unique KKT point, got {len(hits)}"
ASSIGN, X_EXACT = hits[0]
OBJ_EXACT = (F(1, 2) * sum(X_EXACT[i] * sum(Pe[i][j] * X_EXACT[j]
                                            for j in range(N))
                           for i in range(N))
             + sum(ce[i] * X_EXACT[i] for i in range(N)))
X_STAR = np.array([float(v) for v in X_EXACT])
KNOWN_OPTIMAL = float(OBJ_EXACT)
ACTIVE_LO = [i for i in range(N) if ASSIGN[i] == 0]
ACTIVE_HI = [i for i in range(N) if ASSIGN[i] == 2]
FREE = [i for i in range(N) if ASSIGN[i] == 1]

# ------------------------------------------------------------------- pounce
import pounce  # noqa: E402

t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x, dtype=float)
obj_pounce = float(r.obj)
status = str(r.status)
iters = getattr(r, "iterations", getattr(r, "iters", None))

# tolerance sweep: does pounce keep converging on an ill-conditioned QP?
sweep = []
for kw in (dict(), dict(tol=1e-12), dict(tol=1e-14, max_iter=500)):
    t0 = time.perf_counter()
    rr = pounce.solve_qp(P=P, c=c, lb=lb, ub=ub, **kw)
    dt = time.perf_counter() - t0
    sweep.append((kw, str(rr.status), float(rr.obj),
                  np.asarray(rr.x, dtype=float), dt))

# ------------------------------------------------------ oracle 2: cvxpy x2
import cvxpy as cp  # noqa: E402

cvx = {}
for name, solver, kw in (
    ("CLARABEL", cp.CLARABEL, {}),
    ("SCS", cp.SCS, dict(eps=1e-11, max_iters=200000)),
):
    xv = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + c @ xv),
        [xv >= lb, xv <= ub],
    )
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, **kw)
        cvx[name] = (np.asarray(xv.value, dtype=float), float(prob.value),
                     time.perf_counter() - t0, prob.status)
    except Exception as e:                    # pragma: no cover
        cvx[name] = (None, float("nan"), time.perf_counter() - t0, f"error: {e}")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
x_err_known = float(np.max(np.abs(x_pounce - X_STAR)))

print("=== problem ===")
print(f"n={N}  box=[{LO},{HI}]  cond2(P)={COND:.4e}  "
      f"eig(P) in [{evals[0]:.3e}, {evals[-1]:.3e}]")
print(f"data exactly representable in float64: True   ||y||^2 const={CONST:.6f}")
print(f"exact active set: at_lb={ACTIVE_LO} free={FREE} at_ub={ACTIVE_HI}")
print(f"exact rational oracle time={t_exact:.3f}s (3^{N}=2187 active sets, "
      f"unique KKT point)")
print("x* (exact, as float) =", np.array2string(X_STAR, precision=12))
print(f"KNOWN_OPTIMAL (0.5x'Px+c'x) = {KNOWN_OPTIMAL:.16e}")
print(f"  full LS residual ||Va-y||^2 = {KNOWN_OPTIMAL + CONST:.16e}")

print("=== pounce ===")
print(f"status={status} iters={iters} obj={obj_pounce:.16e} t={t_pounce:.4f}s")
print("x =", np.array2string(x_pounce, precision=12))
print(f"rel_err_obj_vs_exact={obj_err_known:.3e}  x_inf_err_vs_exact={x_err_known:.3e}")

print("--- pounce tolerance sweep (vs exact rational optimum) ---")
for kw, st, ob, xx, dt in sweep:
    print(f"  opts={kw or '{default}'}: status={st} obj={ob:.16e} "
          f"rel_err={rel(ob, KNOWN_OPTIMAL):.2e} "
          f"x_inf_err={np.max(np.abs(xx - X_STAR)):.2e} t={dt:.4f}s")

print("=== oracles ===")
for name, (xo, oo, to, st) in cvx.items():
    if xo is None:
        print(f"{name}: {st} t={to:.4f}s")
        continue
    print(f"{name}: status={st} obj={oo:.16e} t={to:.4f}s "
          f"rel_err_obj_vs_exact={rel(oo, KNOWN_OPTIMAL):.3e} "
          f"x_inf_err_vs_exact={np.max(np.abs(xo - X_STAR)):.3e}")

# feasibility / stationarity residual of the pounce point (float, informational)
g = P @ x_pounce + c
viol = float(max(np.max(x_pounce - ub), np.max(lb - x_pounce), 0.0))
print(f"pounce primal box violation = {viol:.3e}")

# --------------------------------------------------------------- verdict
ok_status = status.lower().startswith("optimal") or getattr(r, "success", False)
ok = ok_status and obj_err_known < 1e-4 and viol < 1e-8
print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
      f"(status={status}, obj_rel_err={obj_err_known:.3e}, "
      f"x_inf_err={x_err_known:.3e}, box_viol={viol:.3e})")
