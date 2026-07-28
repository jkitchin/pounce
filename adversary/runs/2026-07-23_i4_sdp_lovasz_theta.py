"""Adversary i4: LOVASZ THETA of the 5-cycle C5.
Family: sdp   Class: dense PSD cone with many linear (edge) equalities.

theta(G) = max <J, X>  s.t.  Tr(X) = 1,  X_ij = 0 for (i,j) in E,  X >= 0.
For C5 (5-cycle), theta(C5) = sqrt(5) (Lovasz's famous result).
SOURCE: L. Lovasz, "On the Shannon capacity of a graph", IEEE Trans. Inf.
Theory 25(1), 1979; theta(C5) = sqrt(5) ~ 2.2360679...
DISTINCT from logged SDPs (max-eig, min-trace 2x2, maxcut triangle, min-eig,
Lyapunov, nearest-correlation): this is a 5x5 PSD with 5 off-diagonal zero
equalities plus a trace equality, objective = sum of all entries.

Decision variables = the 15 svec entries of X directly (x == svec(X)).
Then s = h - G x = x lies in psd cone with G = -I_15, h = 0.
Objective <J,X> = svec(J) . svec(X) = svec(J) . x. maximize -> minimize -svec(J).
svec (pounce): lower-tri, column-major, off-diagonals * sqrt(2).
"""
import time
import numpy as np

Nn = 5
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]   # C5
KNOWN_OPTIMAL = float(np.sqrt(5.0))
r2 = np.sqrt(2.0)

# svec index map, lower-tri column-major
idx = {}
k = 0
for j in range(Nn):
    for i in range(j, Nn):
        idx[(i, j)] = k; k += 1
svec_dim = k                                       # 15
def sidx(a, b):
    return idx[(max(a, b), min(a, b))]

# objective: -svec(J). svec(J): diag entries 1, off-diag entries sqrt(2)
svecJ = np.zeros(svec_dim)
for j in range(Nn):
    for i in range(j, Nn):
        svecJ[idx[(i, j)]] = 1.0 if i == j else r2
c = -svecJ

# s = h - G x in psd cone; take x = svec(X): G = -I, h = 0
G = -np.eye(svec_dim)
h = np.zeros(svec_dim)
cones = [("psd", Nn)]

# equalities:
Arows, brhs = [], []
# Tr(X) = 1  -> sum of diagonal svec entries = 1 (diag scale 1)
row = np.zeros(svec_dim)
for i in range(Nn):
    row[idx[(i, i)]] = 1.0
Arows.append(row); brhs.append(1.0)
# X_ij = 0 for edges  -> the svec entry (= sqrt(2) X_ij) = 0
for (a, b) in edges:
    row = np.zeros(svec_dim)
    row[sidx(a, b)] = 1.0
    Arows.append(row); brhs.append(0.0)
A = np.array(Arows); bvec = np.array(brhs)

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, float)
obj_pounce = float(-r.obj)                          # maximized <J,X>
status = str(r.status)

# reconstruct X and check PSD + constraints
X = np.zeros((Nn, Nn))
for j in range(Nn):
    for i in range(j, Nn):
        val = v[idx[(i, j)]] / (1.0 if i == j else r2)
        X[i, j] = val; X[j, i] = val
eig_min = float(np.linalg.eigvalsh(X)[0])
tr = float(np.trace(X))
edge_max = max(abs(X[a, b]) for (a, b) in edges)
JdotX = float(np.sum(X))

# oracle: cvxpy building the SAME SDP
import cvxpy as cp
def solve_cvxpy(solver):
    Xv = cp.Variable((Nn, Nn), symmetric=True)
    cons = [Xv >> 0, cp.trace(Xv) == 1]
    for (a, b) in edges:
        cons.append(Xv[a, b] == 0)
    prob = cp.Problem(cp.Maximize(cp.sum(Xv)), cons)
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), dt

val_cla, t_cla = solve_cvxpy(cp.CLARABEL)
val_scs, t_scs = solve_cvxpy(cp.SCS)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))

print("=== pounce (PSD-cone IPM, Lovasz theta C5) ===")
print(f"status={status} <J,X>={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"  Tr(X)={tr:.6f} min_eig(X)={eig_min:.3e} max|X_edge|={edge_max:.2e} sum(X)={JdotX:.8f}")
print(f"=== cvxpy/CLARABEL val={val_cla:.10e} t={t_cla:.4f}s")
print(f"=== cvxpy/SCS      val={val_scs:.10e} t={t_scs:.4f}s")
print(f"known theta(C5)=sqrt(5)={KNOWN_OPTIMAL:.10e}")
print(f"rel vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"vs_CLARABEL={rel(obj_pounce, val_cla):.2e} vs_SCS={rel(obj_pounce, val_scs):.2e}")

ok = (status in ("optimal", "optimal_inaccurate") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and eig_min > -1e-6 \
    and abs(tr - 1.0) < 1e-6 and edge_max < 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"eig_min={eig_min:.2e}, tr={tr:.4f}, edge={edge_max:.2e})")
