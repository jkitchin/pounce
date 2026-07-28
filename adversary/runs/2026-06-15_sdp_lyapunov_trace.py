"""Adversary cross-check: Lyapunov stability SDP (min Tr(P) s.t. A'P+PA <= -I, P>=0)
Family: sdp   Class: two-block dense semidefinite program (2 coupled 2x2 PSD cones)

Problem (continuous-time Lyapunov decay certificate):
    minimize    Tr(P)
    subject to  -(A^T P + P A) - I  >= 0      (2x2 PSD)
                 P                  >= 0       (2x2 PSD)
    over symmetric P in S^2.

This is the canonical "find a quadratic Lyapunov function V(x)=x^T P x" SDP from
control theory: for a Hurwitz (stable) A, V decreases along trajectories iff
A^T P + P A < 0.  Adding the normalizing constraint A^T P + P A <= -I and the
objective Tr(P) gives a bounded convex SDP with TWO coupled PSD cones in the same
matrix variable P -- distinct from every prior sdp run (single-cone max-eig,
min-trace 2x2 Schur, max-cut K3, trace-1 min-eig).

KNOWN OPTIMAL (closed form via the Lyapunov equation):
    For a Hurwitz A, the unconstrained problem min Tr(P) s.t. A^T P + P A <= -I,
    P>=0 has its first constraint ACTIVE at the optimum (decreasing Tr(P) pushes
    A^T P + P A up until it hits -I), so the optimizer solves the continuous
    Lyapunov equation
        A^T P + P A = -I,   P = integral_0^inf exp(A^T t) exp(A t) dt  > 0,
    which is exactly the observability/controllability Gramian.  Then
        optimum = Tr(P*),
    with P* obtained independently from scipy.linalg.solve_continuous_lyapunov.
    (A>0 here is automatic: P* is a Gramian of a stable system => PD, so the
    second cone is inactive but kept to exercise the 2-cone path.)
SOURCE: Boyd, El Ghaoui, Feron, Balakrishnan, "Linear Matrix Inequalities in
    System and Control Theory", SIAM 1994, Ch.5 (Lyapunov stability LMIs);
    standard control-theory result on the Lyapunov equation Gramian.

A chosen Hurwitz (eigenvalues -1, -2, both stable):
    A = [[-1, 1],
         [ 0, -2]]

svec layout (pounce, confirmed from proven runs): lower triangle, column-major,
off-diagonals * sqrt(2):
    2x2 -> svec(M) = [M00, s2*M10, M11], s2 = sqrt(2).
    Inner product <X,Y> = svec(X).svec(Y), so any linear map on a symmetric
    matrix is encoded row-by-row in this basis.
"""
import time
import numpy as np
import scipy.linalg as sla

s2 = np.sqrt(2.0)

A = np.array([[-1.0, 1.0],
              [0.0, -2.0]])
assert np.all(np.real(np.linalg.eigvals(A)) < 0), "A must be Hurwitz"

# --- closed-form reference: solve A^T P + P A = -I  ->  Tr(P*) ---
# scipy solves A X + X A^H = Q.  We need A^T P + P A = -I, i.e. with A->A^T, Q=-I.
P_star = sla.solve_continuous_lyapunov(A.T, -np.eye(2))
P_star = 0.5 * (P_star + P_star.T)
KNOWN_OPTIMAL = float(np.trace(P_star))

# Decision variable: v = [P00, P10, P11]  (symmetric 2x2, 3 free entries)
# index map: 0:P00 1:P10 2:P11

# Objective Tr(P) = P00 + P11 = v0 + v2
c = np.array([1.0, 0.0, 1.0])

# ---- Build the two PSD-cone affine maps  s = h - G v  (each s must be PSD-svec) ----
# Cone 1: M1 = -(A^T P + P A) - I  >= 0.
# Compute A^T P + P A as a linear map of (P00,P10,P11).
# Let P = [[p0, p1],[p1, p2]].  A = [[a,b],[c,d]].
a, b = A[0, 0], A[0, 1]
cc, d = A[1, 0], A[1, 1]
# A^T P:
#   A^T = [[a, c],[b, d]]
#   (A^T P) = [[a*p0+c*p1, a*p1+c*p2],[b*p0+d*p1, b*p1+d*p2]]
# P A = (A^T P)^T for the symmetric combination; S = A^T P + P A is symmetric.
# S00 = 2*(a*p0 + c*p1)
# S11 = 2*(b*p1 + d*p2)
# S10 = S01 = b*p0 + d*p1 + a*p1 + c*p2 = b*p0 + (a+d)*p1 + c*p2
# (derivation: S = A^T P + P A; entry (i,j)=sum_k A_ki P_kj + P_ik A_kj.)

# Linear coefficients of S entries wrt (p0,p1,p2):
S00 = np.array([2 * a, 2 * cc, 0.0])
S10 = np.array([b, (a + d), cc])
S11 = np.array([0.0, 2 * b, 2 * d])

# Cone-1 matrix M1 = -S - I.  Its svec = [M1_00, s2*M1_10, M1_11].
# M1_00 = -S00 - 1 ;  M1_10 = -S10 ;  M1_11 = -S11 - 1
# s = h - G v   =>   row = (constant h_row) + (-G_row) . v
# For M1_00:  -S00.v - 1   -> h=-1, (-G_row)=-S00 => G_row = S00
# For s2*M1_10: s2*(-S10.v) -> h=0,  G_row = s2*S10
# For M1_11:  -S11.v - 1   -> h=-1, G_row = S11
G1 = np.vstack([S00,
                s2 * S10,
                S11])
h1 = np.array([-1.0, 0.0, -1.0])

# Cone 2: M2 = P >= 0.  svec(P) = [p0, s2*p1, p2] = [v0, s2*v1, v2].
# s = h - G v, h=0, G maps v -> -svec(P).
G2 = np.array([[1.0, 0.0, 0.0],
               [0.0, s2, 0.0],
               [0.0, 0.0, 1.0]])
G2 = -G2  # so that s = -G2 v = svec(P)
h2 = np.zeros(3)

# Stack the two cones (pounce reads cones in order).
G = np.vstack([G1, G2])
h = np.concatenate([h1, h2])
cones = [("psd", 2), ("psd", 2)]

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, dtype=float)
obj_pounce = float(c @ v)
status = r.status
P_pounce = np.array([[v[0], v[1]],
                     [v[1], v[2]]])

# sanity: reconstruct slacks and verify both cones are PSD
S_p = A.T @ P_pounce + P_pounce @ A
M1_p = -S_p - np.eye(2)
eig_M1 = float(np.linalg.eigvalsh(M1_p)[0])
eig_P = float(np.linalg.eigvalsh(P_pounce)[0])

# ---- Oracle: cvxpy, two solvers ----
import cvxpy as cp


def solve_cvxpy(solver):
    P = cp.Variable((2, 2), symmetric=True)
    cons = [-(A.T @ P + P @ A) - np.eye(2) >> 0, P >> 0]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, P.value


obj_scs, t_scs, P_scs = solve_cvxpy(cp.SCS)
obj_cla, t_cla, P_cla = solve_cvxpy(cp.CLARABEL)


def rel(x, y):
    return abs(x - y) / max(1.0, abs(y))


print("=== reference (Lyapunov equation A^T P + P A = -I) ===")
print(f"P* =\n{P_star}")
print(f"Tr(P*) = known_optimal = {KNOWN_OPTIMAL:.10e}")
print("=== pounce (2-block PSD cone IPM) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"P_pounce =\n{P_pounce}")
print(f"min eig(-(A'P+PA)-I) = {eig_M1:.3e}  (>=0 feasible);  min eig(P) = {eig_P:.3e}")
print("=== oracle cvxpy/SCS ===")
print(f"obj={obj_scs:.10e} t={t_scs:.4f}s")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj={obj_cla:.10e} t={t_cla:.4f}s")
print(f"rel_err pounce vs known(Tr P*) = {rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"rel_err pounce vs SCS          = {rel(obj_pounce, obj_scs):.2e}")
print(f"rel_err pounce vs CLARABEL     = {rel(obj_pounce, obj_cla):.2e}")
print(f"||P_pounce - P*||_F            = {np.linalg.norm(P_pounce - P_star):.2e}")

ok = ((status == "optimal") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and rel(obj_pounce, obj_cla) < 1e-4 \
    and eig_M1 > -1e-6 \
    and eig_P > -1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"eig_M1={eig_M1:.2e}, eig_P={eig_P:.2e})")
