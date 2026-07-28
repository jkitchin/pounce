"""Adversary cross-check: sparse-matrix input & structure handling in solve_qp
Family: api (qp entry point)   Class: input-format / structure edge cases
Dimension: API contracts, option handling, input edge cases

Problem (n=4), a strictly convex equality-constrained QP with an analytic
(closed-form KKT) optimum:

    min  1/2 x'Px + c'x
    s.t. Ax = b            A = [1 1 1 1], b = [1]
         Gx <= h           x0 <= 0.8, -x3 <= 0.5   (both INACTIVE at the opt)

    P = [[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,5]]   (PD, tridiagonal)
    c = [-1,-2,-3,-4]

Because the inequalities are inactive, the optimum solves the KKT system
    [P A'; A 0][x; y] = [-c; b]
exactly -> closed-form oracle (solved in exact rationals with sympy when
available, else float64).

Sub-tests
  (a) every scipy.sparse format (csr,csc,coo,lil,dok,bsr) for P, A and G
  (b) explicit stored zeros (structural nonzero with value 0.0)
  (c) duplicate COO entries at the same (i,j)  -> must SUM (scipy convention)
  (d) unsorted CSR indices
  (e) completely empty sparse matrix (nnz = 0)
  (f) P stored as lower triangle only / upper triangle only  (convention!)
  (g) dense vs sparse agreement to machine precision
"""

import time
import numpy as np
import scipy.sparse as sp

from pounce import solve_qp

np.set_printoptions(precision=12, suppress=False)

# ----------------------------------------------------------------- problem ---
P_dense = np.array(
    [
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 5.0],
    ]
)
c_vec = np.array([-1.0, -2.0, -3.0, -4.0])
A_dense = np.array([[1.0, 1.0, 1.0, 1.0]])
b_vec = np.array([1.0])
G_dense = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]])
h_vec = np.array([0.8, 0.5])
n = 4

# ------------------------------------------------------- closed-form oracle ---
KKT = np.zeros((n + 1, n + 1))
KKT[:n, :n] = P_dense
KKT[:n, n:] = A_dense.T
KKT[n:, :n] = A_dense
rhs = np.concatenate([-c_vec, b_vec])
sol = np.linalg.solve(KKT, rhs)
X_STAR = sol[:n]
OBJ_STAR = 0.5 * X_STAR @ P_dense @ X_STAR + c_vec @ X_STAR

try:  # exact rational confirmation
    import sympy as sy

    Ks = sy.Matrix(KKT.tolist())
    rs = sy.Matrix(rhs.tolist())
    exact = Ks.solve(rs)
    X_EXACT = np.array([float(v) for v in exact[:n]])
    exact_err = float(np.max(np.abs(X_EXACT - X_STAR)))
except Exception as e:  # pragma: no cover
    exact_err = float("nan")
    print(f"(sympy unavailable: {e})")

assert np.all(G_dense @ X_STAR <= h_vec - 1e-9), "inequalities must be inactive"
print("=== closed form ===")
print(f"x*   = {X_STAR}")
print(f"obj* = {OBJ_STAR:.16e}")
print(f"sympy-exact vs float64 KKT max|dx| = {exact_err:.3e}")
print(f"G x* - h = {G_dense @ X_STAR - h_vec}  (both < 0 -> inactive)")

# --------------------------------------------------------- cvxpy 2nd oracle ---
OBJ_CVX = None
try:
    import cvxpy as cp

    xv = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_dense)) + c_vec @ xv),
        [A_dense @ xv == b_vec, G_dense @ xv <= h_vec],
    )
    prob.solve(solver=cp.CLARABEL)
    OBJ_CVX = prob.value
    print(f"cvxpy(CLARABEL) obj = {OBJ_CVX:.16e}  |dobj| = {abs(OBJ_CVX-OBJ_STAR):.3e}")
except Exception as e:
    print(f"(cvxpy unavailable/failed: {e})")

# ------------------------------------------------------------------ harness ---
RESULTS = []  # (label, status, obj, x_inf_err_vs_closed_form, note)


def run(label, P=None, A=None, G=None, expect="match", note="", **kw):
    """Solve and record. expect: 'match' | 'reject' | 'observe'."""
    t0 = time.perf_counter()
    try:
        r = solve_qp(
            P=P_dense if P is None else P,
            c=c_vec,
            A=A_dense if A is None else A,
            b=b_vec,
            G=G_dense if G is None else G,
            h=h_vec,
            tol=1e-12,
            **kw,
        )
        dt = time.perf_counter() - t0
        x = np.asarray(r.x, dtype=float)
        xe = float(np.max(np.abs(x - X_STAR)))
        oe = abs(r.obj - OBJ_STAR)
        RESULTS.append((label, r.status, r.obj, xe, oe, expect, note, dt, None))
        flag = "OK " if (xe < 1e-8 and r.status == "optimal") else "DIFF"
        print(
            f"  [{flag}] {label:<34s} status={r.status:<10s} "
            f"obj={r.obj:.16e} x_err={xe:.3e} t={dt:.3f}s {note}"
        )
        return r
    except Exception as e:
        dt = time.perf_counter() - t0
        RESULTS.append(
            (label, "RAISED", None, None, None, expect, note, dt, f"{type(e).__name__}: {e}")
        )
        print(f"  [RAISE] {label:<34s} {type(e).__name__}: {e}")
        return None


FORMATS = {
    "csr": sp.csr_matrix,
    "csc": sp.csc_matrix,
    "coo": sp.coo_matrix,
    "lil": sp.lil_matrix,
    "dok": sp.dok_matrix,
    "bsr": sp.bsr_matrix,
}

print("\n=== (g) dense baseline ===")
run("dense (baseline)")

print("\n=== (a) all sparse formats, P/A/G together ===")
for name, ctor in FORMATS.items():
    run(
        f"a: {name} for P,A,G",
        P=ctor(P_dense),
        A=ctor(A_dense),
        G=ctor(G_dense),
    )

print("\n=== (a2) each matrix sparse in isolation (csr) ===")
run("a2: P sparse only", P=sp.csr_matrix(P_dense))
run("a2: A sparse only", A=sp.csr_matrix(A_dense))
run("a2: G sparse only", G=sp.csr_matrix(G_dense))

print("\n=== (b) explicit stored zeros ===")
# P with explicit structural zeros at (0,2),(2,0),(0,3),(3,0)
Pz = sp.coo_matrix(
    (
        list(P_dense[np.nonzero(P_dense)]) + [0.0, 0.0, 0.0, 0.0],
        (
            list(np.nonzero(P_dense)[0]) + [0, 2, 0, 3],
            list(np.nonzero(P_dense)[1]) + [2, 0, 3, 0],
        ),
    ),
    shape=(n, n),
).tocsr()
Pz_chk = sp.csr_matrix(P_dense)
print(f"  P explicit-zero nnz={Pz.nnz} vs clean nnz={Pz_chk.nnz}")
run("b: P w/ explicit zeros", P=Pz)

Az = sp.coo_matrix(
    ([1.0, 1.0, 1.0, 1.0, 0.0], ([0, 0, 0, 0, 0], [0, 1, 2, 3, 1])), shape=(1, n)
)
# note: that would be a duplicate; use a genuinely-zero-only extra slot instead
Az = sp.csr_matrix(A_dense)
Az.data[1] = 1.0
Az = sp.coo_matrix(
    ([1.0, 1.0, 1.0, 1.0], ([0, 0, 0, 0], [0, 1, 2, 3])), shape=(1, n)
).tocsr()
# force an explicit zero into G (structural nonzero valued 0)
Gz = sp.coo_matrix(
    (
        [1.0, -1.0, 0.0, 0.0],
        ([0, 1, 0, 1], [0, 3, 2, 1]),
    ),
    shape=(2, n),
).tocsr()
print(f"  G explicit-zero nnz={Gz.nnz} vs clean nnz={sp.csr_matrix(G_dense).nnz}")
run("b: G w/ explicit zeros", G=Gz)

print("\n=== (c) duplicate COO entries (scipy convention: SUM) ===")
# P: split the (1,0)/(0,1) entry value 1.0 into 0.4+0.6 duplicates
rp, cp_, vp = [], [], []
for i in range(n):
    for j in range(n):
        if P_dense[i, j] != 0.0:
            if (i, j) in ((1, 0), (0, 1)):
                rp += [i, i]
                cp_ += [j, j]
                vp += [0.4, 0.6]
            else:
                rp.append(i)
                cp_.append(j)
                vp.append(P_dense[i, j])
P_dup = sp.coo_matrix((vp, (rp, cp_)), shape=(n, n))
print(f"  P_dup nnz(stored)={P_dup.nnz}  sum-check max|dense-P|="
      f"{np.max(np.abs(P_dup.toarray()-P_dense)):.3e}")
run("c: P duplicate COO (sum=1.0)", P=P_dup)

A_dup = sp.coo_matrix(
    ([0.25, 0.75, 1.0, 1.0, 1.0], ([0, 0, 0, 0, 0], [0, 0, 1, 2, 3])), shape=(1, n)
)
print(f"  A_dup toarray = {A_dup.toarray()}")
run("c: A duplicate COO (sum=1.0)", A=A_dup)

G_dup = sp.coo_matrix(
    ([0.5, 0.5, -1.0], ([0, 0, 1], [0, 0, 3])), shape=(2, n)
)
print(f"  G_dup toarray = {G_dup.toarray().tolist()}")
run("c: G duplicate COO (sum=1.0)", G=G_dup)

print("\n=== (d) unsorted CSR indices ===")


def unsort_csr(M):
    """Return an equal CSR matrix with column indices permuted within rows."""
    M = sp.csr_matrix(M).copy()
    M.sort_indices()
    for i in range(M.shape[0]):
        s, e = M.indptr[i], M.indptr[i + 1]
        if e - s > 1:
            M.indices[s:e] = M.indices[s:e][::-1]
            M.data[s:e] = M.data[s:e][::-1]
    M.has_sorted_indices = False
    return M


Pu, Au, Gu = unsort_csr(P_dense), unsort_csr(A_dense), unsort_csr(G_dense)
print(
    f"  has_sorted_indices: P={Pu.has_sorted_indices} A={Au.has_sorted_indices} "
    f"G={Gu.has_sorted_indices}; densify-equal: "
    f"{np.allclose(Pu.toarray(), P_dense)} {np.allclose(Au.toarray(), A_dense)} "
    f"{np.allclose(Gu.toarray(), G_dense)}"
)
run("d: unsorted CSR P", P=Pu)
run("d: unsorted CSR A", A=Au)
run("d: unsorted CSR G", G=Gu)
run("d: unsorted CSR P,A,G", P=Pu, A=Au, G=Gu)

print("\n=== (e) empty sparse matrices (nnz=0) ===")
# G empty (0 rows) with empty h -> equality-only QP, same optimum
r_e1 = None
t0 = time.perf_counter()
try:
    r_e1 = solve_qp(
        P=sp.csr_matrix(P_dense),
        c=c_vec,
        A=sp.csr_matrix(A_dense),
        b=b_vec,
        G=sp.csr_matrix((0, n)),
        h=np.zeros(0),
        tol=1e-12,
    )
    xe = float(np.max(np.abs(np.asarray(r_e1.x) - X_STAR)))
    print(
        f"  [{'OK ' if xe<1e-8 else 'DIFF'}] e: G 0-row empty sparse       "
        f"status={r_e1.status} obj={r_e1.obj:.16e} x_err={xe:.3e}"
    )
    RESULTS.append(("e: G 0-row empty", r_e1.status, r_e1.obj, xe,
                    abs(r_e1.obj - OBJ_STAR), "match", "", 0.0, None))
except Exception as e:
    print(f"  [RAISE] e: G 0-row empty sparse -> {type(e).__name__}: {e}")
    RESULTS.append(("e: G 0-row empty", "RAISED", None, None, None, "match", "",
                    0.0, f"{type(e).__name__}: {e}"))

# G all-structural-zero (2 rows, nnz=0): rows 0<=h are trivially satisfied
run("e: G all-zeros nnz=0", G=sp.csr_matrix((2, n)), expect="observe",
    note="(0 <= h trivially true)")

# P all zeros nnz=0 -> LP on a bounded feasible set (separate closed form)
t0 = time.perf_counter()
try:
    rlp_s = solve_qp(P=sp.csr_matrix((n, n)), c=c_vec, A=A_dense, b=b_vec,
                     lb=np.zeros(n), ub=np.ones(n), tol=1e-12)
    rlp_d = solve_qp(P=np.zeros((n, n)), c=c_vec, A=A_dense, b=b_vec,
                     lb=np.zeros(n), ub=np.ones(n), tol=1e-12)
    rlp_n = solve_qp(P=None, c=c_vec, A=A_dense, b=b_vec,
                     lb=np.zeros(n), ub=np.ones(n), tol=1e-12)
    # closed form: minimize c'x on simplex -> all mass on most negative c
    lp_star = -4.0
    print(
        f"  LP (P=empty-sparse) obj={rlp_s.obj:.16e} status={rlp_s.status}\n"
        f"  LP (P=dense zeros)  obj={rlp_d.obj:.16e} status={rlp_d.status}\n"
        f"  LP (P=None)         obj={rlp_n.obj:.16e} status={rlp_n.status}\n"
        f"  closed form         obj={lp_star:.16e}"
    )
    RESULTS.append(("e: P empty-sparse LP", rlp_s.status, rlp_s.obj, None,
                    abs(rlp_s.obj - lp_star), "match", "vs LP closed form",
                    0.0, None))
    RESULTS.append(("e: P=None LP", rlp_n.status, rlp_n.obj, None,
                    abs(rlp_n.obj - lp_star), "match", "vs LP closed form",
                    0.0, None))
except Exception as e:
    print(f"  [RAISE] e: empty P LP -> {type(e).__name__}: {e}")

print("\n=== (f) triangle convention for P ===")
print("  Documented (docs/src/convex-solver.md:59, qp.py:456): "
      "'P (lower triangle used, assumed symmetric)'")
P_low = sp.csr_matrix(np.tril(P_dense))
P_upp = sp.csr_matrix(np.triu(P_dense))
P_full = sp.csr_matrix(P_dense)
run("f: P lower triangle only", P=P_low)
run("f: P upper triangle only", P=P_upp, expect="observe",
    note="<- violates 'assumed symmetric'")
run("f: P full symmetric", P=P_full)
# what would 'upper only' correspond to mathematically?
P_upp_as_read = np.tril(P_upp.toarray())
P_upp_sym = P_upp_as_read + np.tril(P_upp_as_read, -1).T
print(f"  P as the solver reads 'upper-only' (mirror lower tri) =\n{P_upp_sym}")
KKT2 = np.zeros((n + 1, n + 1))
KKT2[:n, :n] = P_upp_sym
KKT2[:n, n:] = A_dense.T
KKT2[n:, :n] = A_dense
try:
    s2 = np.linalg.solve(KKT2, rhs)
    print(f"  -> predicted obj if P_diag-only used: "
          f"{0.5*s2[:n]@P_upp_sym@s2[:n] + c_vec@s2[:n]:.16e}")
except Exception as e:
    print(f"  (singular: {e})")

print("\n=== (c2) PSD-check consistency with duplicate COO ===")
# _min_eig_lower_coo assigns (not accumulates) duplicates -> the PSD guard may
# see a different matrix than the solver. Build a P that is PD when duplicates
# are SUMMED but indefinite when they are OVERWRITTEN (and vice versa).
# summed: off-diag(1,0) = 1.5+1.5 = 3.0 with diag 2,2 -> indefinite (eig 2-3<0)
# overwritten (last wins): off-diag = 1.5 -> PD (eig 2-1.5 = 0.5 > 0)
P_trap = sp.coo_matrix(
    ([2.0, 2.0, 1.5, 1.5], ([0, 1, 1, 1], [0, 1, 0, 0])), shape=(2, 2)
)
print(f"  P_trap densified (scipy sums dups) =\n{P_trap.toarray()}")
print(f"  eig(sum-convention symmetrized) = "
      f"{np.linalg.eigvalsh(np.tril(P_trap.toarray())+np.tril(P_trap.toarray(),-1).T)}")
try:
    rt = solve_qp(P=P_trap, c=np.array([-1.0, -1.0]), lb=-np.ones(2) * 10,
                  ub=np.ones(2) * 10, tol=1e-10)
    print(f"  solve_qp accepted P_trap: status={rt.status} obj={rt.obj:.6e} "
          f"x={np.asarray(rt.x)}")
    print("  -> the PSD guard did NOT reject an indefinite (sum-convention) P")
    RESULTS.append(("c2: PSD guard vs COO dups", rt.status, rt.obj, None, None,
                    "observe", "indefinite under sum convention", 0.0, None))
except Exception as e:
    print(f"  solve_qp rejected P_trap: {type(e).__name__}: {e}")
    RESULTS.append(("c2: PSD guard vs COO dups", "RAISED", None, None, None,
                    "observe", "", 0.0, f"{type(e).__name__}: {e}"))

# ------------------------------------------------------------------ summary ---
print("\n=== SUMMARY (vs closed-form x*, obj*) ===")
print(f"{'case':<34s} {'status':<12s} {'obj_err':>12s} {'x_inf_err':>12s}")
bad = []
for (lab, st, obj, xe, oe, exp, note, dt, err) in RESULTS:
    if err:
        print(f"{lab:<34s} {'RAISED':<12s} {err[:50]}")
        if exp == "match":
            bad.append((lab, err))
        continue
    xs = "n/a" if xe is None else f"{xe:.3e}"
    os_ = "n/a" if oe is None else f"{oe:.3e}"
    print(f"{lab:<34s} {st:<12s} {os_:>12s} {xs:>12s}   {note}")
    if exp == "match":
        if st != "optimal" or (xe is not None and xe > 1e-8) or (oe is not None and oe > 1e-9):
            bad.append((lab, f"status={st} obj_err={os_} x_err={xs}"))

print()
if bad:
    print("MISMATCHES:")
    for lab, why in bad:
        print(f"  - {lab}: {why}")
    print(f"VERDICT: FAIL ({len(bad)} mismatching case(s))")
else:
    print("VERDICT: PASS (all accepted sparse forms match the closed form / dense)")
