"""Adversary cross-check: nearest correlation matrix (Higham) -- RANK-DEFICIENT optimum.
Family: sdp   Class: projection onto the elliptope; optimum on the boundary,
                     X* singular (rank 2 of 3) => degenerate / non-strictly-complementary

Problem (N. J. Higham, "Computing the nearest correlation matrix -- a problem
from finance", IMA J. Numer. Anal. 22(3):329-343, 2002, sec. 1-3; also the
MOSEK Modeling Cookbook v3.3 sec. 6.2.x "nearest correlation matrix"):

    minimize    || X - A ||_F
    subject to  diag(X) = 1
                X >= 0   (PSD)

with the "equicorrelation" input (a classic pathological correlation estimate:
symmetric, unit diagonal, but NOT positive semidefinite)

    A = I + rho (J - I),  rho = -0.8,  n = 3
      = [[ 1.0, -0.8, -0.8],
         [-0.8,  1.0, -0.8],
         [-0.8, -0.8,  1.0]]
    eig(A) = {1 + 2 rho, 1 - rho, 1 - rho} = {-0.6, 1.8, 1.8}  -> indefinite.

KNOWN OPTIMAL -- exact, derived analytically (no solver):
  The feasible set (the elliptope E_3 = {X >= 0, diag X = 1}) and the objective
  are invariant under simultaneous row/column permutation X -> P X P'.  A is
  permutation invariant.  The Frobenius projection onto a closed convex set is
  UNIQUE, hence the minimizer must itself be permutation invariant, i.e.
      X = I + t (J - I)   for a scalar t.
  For that family eig(X) = {1 + 2t, 1 - t, 1 - t}, so X >= 0  <=>  t in [-1/2, 1],
  and ||X - A||_F^2 = 6 (t - rho)^2 (six off-diagonal entries).  Minimizing
  6(t + 0.8)^2 over t in [-1/2, 1] gives the clamped value

      t* = -1/2,   X* = I - (1/2)(J - I),   eig(X*) = {0, 3/2, 3/2},

  so X* is SINGULAR (rank 2) -- the optimum sits on the boundary of the PSD cone
  with a zero eigenvalue, which is exactly the degenerate case this run targets
  (the dual is rank 1, no strictly complementary pair, so the central path
  approaches a non-interior limit).  The optimal value is

      OPT = ||X* - A||_F = sqrt(6) * |t* - rho| = sqrt(6) * 0.3
          = 0.7348469228349534.

Cone encoding -- RE-DERIVED here, do not trust memory:
  pounce ("psd", n) slack block is svec(S): lower triangle, COLUMN BY COLUMN,
  off-diagonals scaled by sqrt(2), length n(n+1)/2.  For n = 3 with
      X = [[ 1, u0, u1],
           [u0,  1, u2],
           [u1, u2,  1]]      (u0 = X10, u1 = X20, u2 = X21)
  column 0 gives (X00, X10, X20), column 1 gives (X11, X21), column 2 gives X22:
      svec(X) = [X00, s2*X10, s2*X20, X11, s2*X21, X22]
              = [1,   s2*u0,  s2*u1,  1,   s2*u2,  1],      s2 = sqrt(2).
  This scaling makes <X,Y> = svec(X).svec(Y); it is verified numerically at the
  bottom of this file against explicit random symmetric matrices BEFORE any
  conclusion is drawn about pounce.

  The unit-diagonal equalities are substituted out (diag fixed at 1), so the
  decision vector is z = (T, u0, u1, u2), 4 variables:
      objective   min T
      SOC(4)      T >= || s2*(u - a) ||_2   where a = (rho, rho, rho)
                  since ||X - A||_F^2 = sum over the 6 off-diagonal cells
                  = 2 * sum_i (u_i - rho)^2  (diagonals match exactly).
      PSD(3)      smat(svec(X(u))) >= 0.
  Slack convention s = h - G z for both cones.
"""
import time
import numpy as np

np.set_printoptions(precision=8, suppress=True)
s2 = np.sqrt(2.0)
n = 3
rho = -0.8
A = np.full((n, n), rho)
np.fill_diagonal(A, 1.0)
assert np.linalg.eigvalsh(A)[0] < 0, "A should be indefinite (a genuine NCM input)"

KNOWN_OPTIMAL = float(np.sqrt(6.0) * 0.3)
X_star = np.full((n, n), -0.5)
np.fill_diagonal(X_star, 1.0)
assert abs(np.linalg.norm(X_star - A, 'fro') - KNOWN_OPTIMAL) < 1e-14
assert abs(np.linalg.eigvalsh(X_star)[0]) < 1e-14, "X* must be singular (rank deficient)"

# ---------------- svec layout verification (must pass before judging pounce) ----


def svec(M):
    """lower triangle, column by column, off-diagonals * sqrt(2)."""
    out = []
    for j in range(M.shape[0]):
        for i in range(j, M.shape[0]):
            out.append(M[i, j] if i == j else s2 * M[i, j])
    return np.array(out)


rng = np.random.default_rng(0)
for _ in range(5):
    M1 = rng.normal(size=(n, n)); M1 = M1 + M1.T
    M2 = rng.normal(size=(n, n)); M2 = M2 + M2.T
    assert abs(svec(M1) @ svec(M2) - np.trace(M1 @ M2)) < 1e-10, "svec inner product broken"
    assert abs(np.linalg.norm(svec(M1)) - np.linalg.norm(M1, 'fro')) < 1e-10
print("svec layout check: <X,Y> = svec(X).svec(Y) and ||svec(X)||=||X||_F  OK")

# ---------------- pounce ----------------
# z = (T, u0, u1, u2)
nz = 4
c = np.array([1.0, 0.0, 0.0, 0.0])

# SOC(4): s0 = T ; s_{1..3} = s2*(u_i - rho)
G_soc = np.zeros((4, nz))
h_soc = np.zeros(4)
G_soc[0, 0] = -1.0                      # s0 = 0 - (-1)*T = T
for i in range(3):
    G_soc[1 + i, 1 + i] = -s2           # s = h - Gz = h + s2*u_i
    h_soc[1 + i] = -s2 * rho            # -> s2*u_i - s2*rho
cone_soc = ("soc", 4)

# PSD(3): svec(X) = [1, s2*u0, s2*u1, 1, s2*u2, 1]
G_psd = np.zeros((6, nz))
h_psd = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
G_psd[1, 1] = -s2      # s2*u0  (entry X10)
G_psd[2, 2] = -s2      # s2*u1  (entry X20)
G_psd[4, 3] = -s2      # s2*u2  (entry X21)
cone_psd = ("psd", 3)

G = np.vstack([G_soc, G_psd])
h = np.concatenate([h_soc, h_psd])
cones = [cone_soc, cone_psd]

import pounce
t0 = time.perf_counter()
res = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z = np.asarray(res.x, dtype=float)
obj_pounce = float(z[0])
status = res.status
u = z[1:]
X_p = np.full((n, n), 0.0)
X_p[1, 0] = X_p[0, 1] = u[0]
X_p[2, 0] = X_p[0, 2] = u[1]
X_p[2, 1] = X_p[1, 2] = u[2]
np.fill_diagonal(X_p, 1.0)
fro_p = float(np.linalg.norm(X_p - A, 'fro'))
eigs_p = np.linalg.eigvalsh(X_p)

# ---------------- oracle: cvxpy, three solvers ----------------
import cvxpy as cp


def solve_cvxpy(solver):
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, cp.diag(X) == np.ones(n)]
    prob = cp.Problem(cp.Minimize(cp.norm(X - A, 'fro')), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return float(prob.value), time.perf_counter() - t0, np.asarray(X.value)


obj_cla, t_cla, X_cla = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, X_scs = solve_cvxpy(cp.SCS)


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


print("=== analytic reference (permutation-symmetry projection) ===")
print(f"A =\n{A}\neig(A) = {np.linalg.eigvalsh(A)}")
print(f"X* =\n{X_star}\neig(X*) = {np.linalg.eigvalsh(X_star)}  (rank {np.linalg.matrix_rank(X_star)})")
print(f"known_optimal = sqrt(6)*0.3 = {KNOWN_OPTIMAL:.10e}")
print("=== pounce (solve_socp, SOC(4) + PSD(3)) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"X_pounce =\n{X_p}")
print(f"||X_pounce - A||_F = {fro_p:.12f}   eig(X_pounce) = {eigs_p}")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj={obj_cla:.10e} t={t_cla:.4f}s\nX=\n{X_cla}")
print("=== oracle cvxpy/SCS ===")
print(f"obj={obj_scs:.10e} t={t_scs:.4f}s\nX=\n{X_scs}")
print(f"rel_err pounce vs known    = {rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"rel_err pounce vs CLARABEL = {rel(obj_pounce, obj_cla):.2e}")
print(f"rel_err pounce vs SCS      = {rel(obj_pounce, obj_scs):.2e}")
print(f"X_inf_err vs X*            = {np.max(np.abs(X_p - X_star)):.2e}")
print(f"epigraph gap T - ||X-A||_F = {obj_pounce - fro_p:.2e}")

ok = ((status == "optimal") or getattr(res, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and rel(obj_pounce, obj_cla) < 1e-4 \
    and rel(obj_pounce, obj_scs) < 1e-4 \
    and eigs_p[0] > -1e-6 \
    and np.max(np.abs(X_p - X_star)) < 1e-4
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"min_eig={eigs_p[0]:.2e}, X_err={np.max(np.abs(X_p - X_star)):.2e})")
