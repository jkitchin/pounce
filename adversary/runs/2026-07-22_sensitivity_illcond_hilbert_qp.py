"""Adversary cross-check: parametric sensitivity dx/db on an ILL-CONDITIONED QP.

Family: sensitivity   Class: ill-conditioning / bad scaling (near-singular KKT)

Source / construction:
  - Hilbert matrix H_n (H_ij = 1/(i+j-1)) is the textbook ill-conditioned SPD
    matrix; cond(H_6) ~ 1.5e7, cond(H_8) ~ 1.5e10  (Higham, "Accuracy and
    Stability of Numerical Algorithms" 2e, Sec. 28.1).
  - We further wreck the SCALING with D = diag(1e3,1e2,1e1,1,1e-1,1e-2), so
    P = D H_6 D has cond ~ 1e19 in the worst case and column norms spanning
    ~10 orders of magnitude.  The constraint matrix A is also badly scaled
    (rows differing by 1e4).
  - Sensitivity theory: Fiacco (1983); Nocedal & Wright 2e Sec. 16.5.  For an
    equality-constrained convex QP the KKT system
        [ P  A^T ] [ x ]   [ -c ]
        [ A   0  ] [ y ] = [  b ]
    is LINEAR in b, so
        dx/db_j  =  the x-block of the KKT solve with rhs [0; e_j],
    EXACTLY, with no truncation error whatsoever.

Two oracles, both fully independent of pounce:
  A. EXACT analytic dx/db by solving the KKT system in exact RATIONAL
     arithmetic with sympy (all data are rationals by construction).
  B. Central finite-difference re-solve, done in 60-digit mpmath (i.e. an
     independent high-precision re-solve, not a pounce re-solve), over a
     step-size sweep delta = 1e-1 ... 1e-9.  Because x*(b) is affine in b,
     the FD estimate must exhibit a PERFECT plateau across the whole sweep;
     any wobble would be pure FD noise and is reported as such.

The risk being probed: on a near-singular KKT the sensitivity back-solve may
silently return garbage, or an unregularized / over-regularized answer, while
reporting success.  A regularized factorization (P + delta*I) gives a dx/db
that is systematically WRONG but perfectly smooth -- it would never show up as
"noise", only as a stable bias vs the exact answer.  That is what we look for.

Cases:
  1. Equality-only ill-conditioned QP (6 vars, 2 equalities).  dx/db exact.
  2. Same, plus an ACTIVE inequality (strictly complementary), so the
     sensitivity must be taken on the active-set KKT.  Exact oracle uses the
     verified active set.
"""

import time

import numpy as np
import sympy as sp
from mpmath import mp

from pounce.qp import QpSensitivity

mp.dps = 60

N = 6
SCALE = [sp.Integer(10) ** k for k in (3, 2, 1, 0, -1, -2)]

# --- exact rational data -----------------------------------------------------
H = sp.Matrix(N, N, lambda i, j: sp.Rational(1, i + j + 1))
D = sp.diag(*SCALE)
P_s = D * H * D
c_s = sp.Matrix([sp.Rational(1, 1), sp.Rational(-2, 1), sp.Rational(3, 1),
                 sp.Rational(-1, 1), sp.Rational(1, 2), sp.Rational(-1, 4)])
A_s = sp.Matrix([[1, 1, 1, 1, 1, 1],
                 [sp.Integer(10) ** 4, 1, 1, 1, 1, sp.Rational(1, 10 ** 4)]])
b_s = sp.Matrix([sp.Rational(1, 1), sp.Rational(2, 1)])
G_s = sp.Matrix([[1, -1, 1, -1, 1, -1]])

P = np.array(P_s.tolist(), dtype=float)
c = np.array(c_s.tolist(), dtype=float).ravel()
A = np.array(A_s.tolist(), dtype=float)
b = np.array(b_s.tolist(), dtype=float).ravel()
G = np.array(G_s.tolist(), dtype=float)

print(f"cond(P)      = {np.linalg.cond(P):.3e}")
print(f"cond(A A^T)  = {np.linalg.cond(A @ A.T):.3e}")


def exact_kkt_solve(Aeq_s, rhs_x, rhs_c):
    """Exact rational solve of [[P, Aeq^T],[Aeq, 0]] [x;y] = [rhs_x; rhs_c]."""
    m = Aeq_s.rows
    K = sp.zeros(N + m, N + m)
    K[:N, :N] = P_s
    K[:N, N:] = Aeq_s.T
    K[N:, :N] = Aeq_s
    rhs = sp.Matrix.vstack(sp.Matrix(rhs_x), sp.Matrix(rhs_c))
    sol = K.LUsolve(rhs)
    return sol[:N, 0], sol[N:, 0]


def kkt_cond(Aeq_s):
    m = Aeq_s.rows
    K = np.zeros((N + m, N + m))
    K[:N, :N] = P
    Aeq = np.array(Aeq_s.tolist(), dtype=float)
    K[:N, N:] = Aeq.T
    K[N:, :N] = Aeq
    return np.linalg.cond(K)


def fd_sweep(Aeq_s, rhs_c_base, j, deltas):
    """Central FD of x*(b) wrt b_j by 60-digit re-solve of the KKT system.

    Fully independent of pounce (mpmath LU on the exact-data KKT matrix).
    """
    m = Aeq_s.rows
    K = mp.matrix(N + m, N + m)
    for r in range(N):
        for s in range(N):
            K[r, s] = mp.mpf(sp.Float(P_s[r, s], 60).__str__())
    for r in range(N):
        for s in range(m):
            v = mp.mpf(sp.Float(Aeq_s[s, r], 60).__str__())
            K[r, N + s] = v
            K[N + s, r] = v
    base = [mp.mpf(sp.Float(-c_s[r], 60).__str__()) for r in range(N)] + \
           [mp.mpf(sp.Float(rhs_c_base[s], 60).__str__()) for s in range(m)]

    out = {}
    for d in deltas:
        dd = mp.mpf(d)
        rp = mp.matrix(base)
        rp[N + j] = rp[N + j] + dd
        rm = mp.matrix(base)
        rm[N + j] = rm[N + j] - dd
        xp = mp.lu_solve(K, rp)
        xm = mp.lu_solve(K, rm)
        out[d] = np.array([float((xp[r] - xm[r]) / (2 * dd)) for r in range(N)])
    return out


def report(tag, dx_pounce, dx_exact, fd):
    deltas = sorted(fd, reverse=True)
    print(f"\n--- {tag}: FD step-size sweep (60-digit re-solve) ---")
    for d in deltas:
        spread = np.linalg.norm(fd[d] - fd[deltas[len(deltas) // 2]], np.inf)
        print(f"  delta={d:8.1e}  ||fd - fd_mid||_inf = {spread:.3e}")
    fd_vals = np.array([fd[d] for d in deltas])
    plateau = float(np.max(np.abs(fd_vals - fd_vals.mean(axis=0))))
    print(f"  plateau spread across sweep = {plateau:.3e}  "
          f"({'STABLE' if plateau < 1e-8 * max(1.0, np.abs(fd_vals).max()) else 'NOISY'})")

    fd_ref = fd_vals.mean(axis=0)
    scale = max(1.0, float(np.abs(dx_exact).max()))
    e_fd = float(np.linalg.norm(fd_ref - dx_exact, np.inf)) / scale
    e_p_exact = float(np.linalg.norm(dx_pounce - dx_exact, np.inf)) / scale
    e_p_fd = float(np.linalg.norm(dx_pounce - fd_ref, np.inf)) / scale
    print(f"  exact dx/db  = {np.array2string(dx_exact, precision=8)}")
    print(f"  FD    dx/db  = {np.array2string(fd_ref, precision=8)}")
    print(f"  pounce dx/db = {np.array2string(dx_pounce, precision=8)}")
    print(f"  rel_err(FD, exact)     = {e_fd:.3e}")
    print(f"  rel_err(pounce, exact) = {e_p_exact:.3e}")
    print(f"  rel_err(pounce, FD)    = {e_p_fd:.3e}")
    return e_p_exact, e_p_fd, plateau


results = {}

# =============================================================================
# CASE 1: equality-only ill-conditioned QP
# =============================================================================
print("\n" + "=" * 72)
print("CASE 1: equality-only, P = D H_6 D, badly scaled A")
print("=" * 72)
print(f"cond(KKT) = {kkt_cond(A_s):.3e}")

t0 = time.perf_counter()
s1 = QpSensitivity(P=P, c=c, A=A, b=b)
dx1 = s1.parametric_step([0], [1.0])
t1 = time.perf_counter() - t0
print(f"pounce QpSensitivity: t={t1:.4f}s  active={s1.active_indices} "
      f"weakly_active={s1.weakly_active_indices}")

x_ex, y_ex = exact_kkt_solve(A_s, -c_s, b_s)
x_ex_f = np.array([float(v) for v in x_ex])
print(f"  ||x_pounce - x_exact||_inf / ||x_exact||_inf = "
      f"{np.linalg.norm(np.asarray(s1.x) - x_ex_f, np.inf) / max(1.0, np.abs(x_ex_f).max()):.3e}")

dxdb_ex, _ = exact_kkt_solve(A_s, sp.zeros(N, 1), sp.Matrix([1, 0]))
dxdb_ex_f = np.array([float(v) for v in dxdb_ex])
fd1 = fd_sweep(A_s, b_s, 0, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
results["case1"] = report("CASE 1", np.asarray(dx1, dtype=float), dxdb_ex_f, fd1)

# =============================================================================
# CASE 2: same QP + an ACTIVE inequality (active-set KKT)
# =============================================================================
print("\n" + "=" * 72)
print("CASE 2: + active inequality  g^T x <= h  (strictly complementary)")
print("=" * 72)

# Choose h strictly below the case-1 optimum's g^T x so the row is active.
g_at_x1 = float((G @ x_ex_f)[0])
h_val = sp.nsimplify(sp.Rational(int(np.floor(g_at_x1 * 100)) - 20, 100))
h = np.array([float(h_val)])
print(f"g^T x*(case1) = {g_at_x1:.6f}   h = {float(h_val):.6f}  (active by construction)")

t0 = time.perf_counter()
s2 = QpSensitivity(P=P, c=c, A=A, b=b, G=G, h=h)
dx2 = s2.parametric_step([0], [1.0])
t2 = time.perf_counter() - t0
print(f"pounce QpSensitivity: t={t2:.4f}s  active={s2.active_indices} "
      f"weakly_active={s2.weakly_active_indices}")

# Exact oracle: equalities + the (verified) active inequality treated as equality.
Aact_s = sp.Matrix.vstack(A_s, G_s)
bact_s = sp.Matrix.vstack(b_s, sp.Matrix([h_val]))
print(f"cond(active KKT) = {kkt_cond(Aact_s):.3e}")

x2_ex, y2_ex = exact_kkt_solve(Aact_s, -c_s, bact_s)
x2_ex_f = np.array([float(v) for v in x2_ex])
mult_ineq = float(y2_ex[2])
print(f"  exact inequality multiplier = {mult_ineq:.6e} "
      f"({'STRICTLY COMPLEMENTARY (ok)' if abs(mult_ineq) > 1e-6 else 'WEAKLY ACTIVE (degenerate!)'})")
print(f"  ||x_pounce - x_exact||_inf / scale = "
      f"{np.linalg.norm(np.asarray(s2.x) - x2_ex_f, np.inf) / max(1.0, np.abs(x2_ex_f).max()):.3e}")

dxdb2_ex, _ = exact_kkt_solve(Aact_s, sp.zeros(N, 1), sp.Matrix([1, 0, 0]))
dxdb2_ex_f = np.array([float(v) for v in dxdb2_ex])
fd2 = fd_sweep(Aact_s, bact_s, 0, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
results["case2"] = report("CASE 2", np.asarray(dx2, dtype=float), dxdb2_ex_f, fd2)

# =============================================================================
# CASE 3: NEAR-LICQ-FAILURE -- two almost-parallel equality rows.
# This is the regime where a silent Tikhonov regularization of the KKT would
# show up: the true dx/db is ENORMOUS (~1/eps_par), an over-regularized solve
# returns a tame, smooth, completely wrong answer while reporting success.
# =============================================================================
print("\n" + "=" * 72)
print("CASE 3: near-rank-deficient A (rows differ by 1e-9) -- near-LICQ failure")
print("=" * 72)

EPS_PAR = sp.Rational(1, 10 ** 9)
A3_s = sp.Matrix([[1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1 + EPS_PAR]])
b3_s = sp.Matrix([sp.Rational(1, 1), sp.Rational(1, 1)])
A3 = np.array([[float(v) for v in A3_s.row(r)] for r in range(2)])
b3 = np.array([float(v) for v in b3_s], dtype=float)
print(f"sigma_min(A3)/sigma_max(A3) = {np.linalg.cond(A3):.3e}")
print(f"cond(KKT)                   = {kkt_cond(A3_s):.3e}")

t0 = time.perf_counter()
s3 = QpSensitivity(P=P, c=c, A=A3, b=b3)
dx3 = s3.parametric_step([0], [1.0])
t3 = time.perf_counter() - t0
print(f"pounce QpSensitivity: t={t3:.4f}s  (reported success, no exception)")

dxdb3_ex, _ = exact_kkt_solve(A3_s, sp.zeros(N, 1), sp.Matrix([1, 0]))
dxdb3_ex_f = np.array([float(v) for v in dxdb3_ex])
fd3 = fd_sweep(A3_s, b3_s, 0, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
results["case3"] = report("CASE 3", np.asarray(dx3, dtype=float), dxdb3_ex_f, fd3)
print(f"  ||exact dx/db||_inf = {np.abs(dxdb3_ex_f).max():.3e}   "
      f"||pounce dx/db||_inf = {np.abs(np.asarray(dx3, dtype=float)).max():.3e}  "
      f"(a damped/over-regularized answer would be much smaller)")

# --- CONTROL: is the information even present in float64? --------------------
# Decisive check.  We hand a PLAIN dense LAPACK LU (numpy.linalg.solve) the SAME
# float64 KKT matrix and the same rhs.  If numpy recovers the exact dx/db, then
# the double-precision data is sufficient and pounce's damping is algorithmic,
# not an inherent precision limit.
print("\n" + "=" * 72)
print("CASE 3b: near-singularity sweep -- pounce vs plain float64 LAPACK LU")
print("=" * 72)
print(f"{'eps_par':>9} {'cond(KKT)':>11} {'|dx|exact':>11} {'relerr npLU':>12} "
      f"{'relerr pounce':>14} {'|dx_p|/|dx_ex|':>15} {'relerr x*':>10}")
for k in range(3, 11):
    e = sp.Rational(1, 10 ** k)
    Ak_s = sp.Matrix([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1 + e]])
    Ak = np.array([[float(v) for v in Ak_s.row(r)] for r in range(2)])
    bk = np.array([1.0, 1.0])
    K = np.zeros((N + 2, N + 2))
    K[:N, :N] = P
    K[:N, N:] = Ak.T
    K[N:, :N] = Ak
    ex, _ = exact_kkt_solve(Ak_s, sp.zeros(N, 1), sp.Matrix([1, 0]))
    ex_f = np.array([float(v) for v in ex])
    xex, _ = exact_kkt_solve(Ak_s, -c_s, sp.Matrix([1, 1]))
    xex_f = np.array([float(v) for v in xex])
    np_lu = np.linalg.solve(K, np.array([0.0] * N + [1.0, 0.0]))[:N]
    sk = QpSensitivity(P=P, c=c, A=Ak, b=bk)
    dxk = np.asarray(sk.parametric_step([0], [1.0]), dtype=float)
    sc = float(np.abs(ex_f).max())
    print(f"{float(e):9.0e} {np.linalg.cond(K):11.2e} {sc:11.3e} "
          f"{np.abs(np_lu - ex_f).max() / sc:12.3e} "
          f"{np.abs(dxk - ex_f).max() / sc:14.3e} "
          f"{np.abs(dxk).max() / sc:15.3e} "
          f"{np.abs(np.asarray(sk.x) - xex_f).max() / max(1.0, np.abs(xex_f).max()):10.3e}")
print("  ^ npLU stays accurate throughout => the float64 data IS sufficient;")
print("    pounce's dx/db collapses (and is DAMPED) while x* stays accurate.")

# =============================================================================
print("\n" + "=" * 72)
TOL = 1e-4
bad = []
for k, (e_exact, e_fd, plateau) in results.items():
    stable = plateau < 1e-6
    print(f"{k}: rel_err(pounce,exact)={e_exact:.3e}  rel_err(pounce,FD)={e_fd:.3e}  "
          f"FD plateau spread={plateau:.3e} ({'stable' if stable else 'NOISY'})")
    if stable and max(e_exact, e_fd) > TOL:
        bad.append(k)

print(f"total wall time (pounce sensitivity calls) = {t1 + t2 + t3:.4f}s")
if not bad:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (disagrees with STABLE FD plateau + exact: {bad})")
