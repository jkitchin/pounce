"""Adversary cross-check: envelope theorem / implicit-function consistency of
the differentiable QP layer's gradients against the reported KKT multipliers.

Family: diff (+ duals/sensitivity)   Class: strictly-complementary convex QP
Source: analytic. Envelope theorem (Fiacco, *Introduction to Sensitivity and
Stability Analysis in Nonlinear Programming*, 1983, Thm 3.2.2) and the
implicit function theorem applied to the KKT system (Amos & Kolter, OptNet,
ICML 2017, eq. 6-8).  Also Boyd & Vandenberghe, *Convex Optimization*, §5.6.3
("local sensitivity"): dp*/du_i = -lambda_i*, dp*/dv_i = -nu_i*.

Oracle:  (i) the analytic envelope identity  d obj*/dh = -z,  d obj*/db = -y
         (ii) an INDEPENDENT implicit-differentiation reconstruction: the
              reduced KKT matrix assembled here in numpy from the reported
              multipliers and the strictly-active set, solved with numpy
         (iii) central finite differences in float64 with a step sweep.

The QP is constructed so that strict complementarity holds: every inequality
is either strictly active (z_i > 0, slack ~ 0) or strictly inactive
(slack > 0, z_i ~ 0), with a healthy margin.  Without that, x*(theta) is not
differentiable and any "mismatch" would be a false positive.
"""

import time

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# --------------------------------------------------------------------------
# Problem:  min 1/2 x'Px + c'x   s.t.  Ax = b,  Gx <= h    (n = 4)
# --------------------------------------------------------------------------
rng = np.random.default_rng(20260722)
n = 4
L = rng.normal(size=(n, n))
P = L @ L.T + 1.5 * np.eye(n)  # SPD, well conditioned
P = 0.5 * (P + P.T)
c = np.array([-1.0, 0.7, -2.0, 0.3])

A = np.array([[1.0, 1.0, 1.0, 1.0]])  # one equality
b = np.array([1.0])

# Four inequalities.  Rows 0 and 1 are made strictly active by construction,
# rows 2 and 3 are slack.  We pick a target point, then set h to make it so.
G = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, -0.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0, 1.0],
    ]
)
h = np.array([0.15, -0.30, 5.0, 4.0])

m_eq, m_in = A.shape[0], G.shape[0]

TOL = 1e-11


def qp_kwargs():
    return dict(P=P, c=c, A=A, b=b, G=G, h=h)


# ==========================================================================
# 1. Host solve — reported primal + duals, and strict-complementarity audit
# ==========================================================================
from pounce.qp import solve_qp as host_solve_qp  # noqa: E402

t0 = time.perf_counter()
res = host_solve_qp(**qp_kwargs(), tol=TOL)
t_host = time.perf_counter() - t0
x_star, y_star, z_star = np.asarray(res.x), np.asarray(res.y), np.asarray(res.z)
slack = h - G @ x_star

print("=== host solve_qp ===")
print(f"status={res.status} obj={res.obj:.12e} iters={res.iters} t={t_host:.4f}s")
print(f"x = {x_star}")
print(f"y = {y_star}")
print(f"z = {z_star}")
print(f"slack (h-Gx) = {slack}")
print(f"eq resid      = {A @ x_star - b}")

stat = P @ x_star + c + A.T @ y_star + G.T @ z_star
print(f"stationarity ||Px+c+A'y+G'z||_inf = {np.abs(stat).max():.3e}")
stat_flip = P @ x_star + c - A.T @ y_star + G.T @ z_star
print(f"  (with -A'y instead)             = {np.abs(stat_flip).max():.3e}")
SIGN_Y = 1.0 if np.abs(stat).max() < np.abs(stat_flip).max() else -1.0
print(f"  => y enters stationarity with sign {SIGN_Y:+.0f}")

# strict complementarity: |z_i| and slack_i must not both be near zero
ACT = z_star > 1e-6
strict_ok = True
for i in range(m_in):
    if ACT[i]:
        ok = z_star[i] > 1e-4 and abs(slack[i]) < 1e-7
    else:
        ok = slack[i] > 1e-4 and z_star[i] < 1e-7
    strict_ok &= ok
    print(
        f"  ineq {i}: z={z_star[i]:.6e} slack={slack[i]:.6e} "
        f"{'ACTIVE' if ACT[i] else 'inactive'} strict={'yes' if ok else 'NO'}"
    )
print(f"strict complementarity: {'OK' if strict_ok else 'VIOLATED'}")
n_act = int(ACT.sum())
assert strict_ok, "problem is degenerate; redesign before testing derivatives"
assert 0 < n_act < m_in, "want a mix of active and inactive rows"

# ==========================================================================
# 2. Independent implicit-diff oracle: assemble the reduced KKT matrix here
# ==========================================================================
Ga = G[ACT]
K = np.zeros((n + m_eq + n_act, n + m_eq + n_act))
K[:n, :n] = P
K[:n, n : n + m_eq] = A.T
K[:n, n + m_eq :] = Ga.T
K[n : n + m_eq, :n] = A
K[n + m_eq :, :n] = Ga
print(f"\nKKT cond = {np.linalg.cond(K):.3e}")


def ift_dx(dP=None, dc=None, dA=None, db=None, dG=None, dh=None):
    """dx from differentiating the KKT system, built from scratch in numpy.

    Rows:  P dx + A' dy + Ga' dza = -(dP x + dc + dA' y + dG' z)
           A dx = db - dA x
           Ga dx = dh_a - dGa x
    """
    r1 = np.zeros(n)
    if dP is not None:
        r1 -= dP @ x_star
    if dc is not None:
        r1 -= dc
    if dA is not None:
        r1 -= dA.T @ y_star
    if dG is not None:
        r1 -= dG.T @ z_star
    r2 = np.zeros(m_eq)
    if db is not None:
        r2 = r2 + db
    if dA is not None:
        r2 = r2 - dA @ x_star
    r3 = np.zeros(n_act)
    if dh is not None:
        r3 = r3 + np.asarray(dh)[ACT]
    if dG is not None:
        r3 = r3 - dG[ACT] @ x_star
    sol = np.linalg.solve(K, np.concatenate([r1, r2, r3]))
    return sol[:n], sol[n : n + m_eq], sol[n + m_eq :]


# ==========================================================================
# 3. Finite-difference oracle (central, float64, step sweep)
# ==========================================================================
def fd_scalar(f, base, steps=(1e-4, 1e-5, 1e-6)):
    """Central-difference gradient of scalar f(param_array) with step sweep.

    Returns the gradient at the step whose neighbour-to-neighbour change is
    smallest (Richardson-ish plateau pick), plus the spread across steps.
    """
    grads = []
    flat = base.ravel()
    for s in steps:
        g = np.zeros_like(flat)
        for i in range(flat.size):
            up, dn = flat.copy(), flat.copy()
            up[i] += s
            dn[i] -= s
            g[i] = (f(up.reshape(base.shape)) - f(dn.reshape(base.shape))) / (2 * s)
        grads.append(g)
    mid = grads[1]
    spread = max(np.abs(grads[0] - mid).max(), np.abs(grads[2] - mid).max())
    return mid.reshape(base.shape), spread


def obj_of_h(hv):
    r = host_solve_qp(P=P, c=c, A=A, b=b, G=G, h=hv, tol=TOL)
    return float(r.obj)


def obj_of_b(bv):
    r = host_solve_qp(P=P, c=c, A=A, b=bv, G=G, h=h, tol=TOL)
    return float(r.obj)


# ==========================================================================
# 4. JAX layer
# ==========================================================================
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pounce.jax as pj  # noqa: E402

Pj, cj, Aj, bj, Gj, hj = (jnp.asarray(v) for v in (P, c, A, b, G, h))

w = jnp.asarray([0.9, -1.3, 0.4, 2.1])  # scalar-loss weights for dx/dtheta


def jx(Pv, cv, Gv, hv, Av, bv):
    return pj.solve_qp(P=Pv, c=cv, G=Gv, h=hv, A=Av, b=bv, tol=TOL)


def jobj(Pv, cv, Gv, hv, Av, bv):
    xv = jx(Pv, cv, Gv, hv, Av, bv)
    return 0.5 * xv @ (Pv @ xv) + cv @ xv


def jloss(Pv, cv, Gv, hv, Av, bv):
    return w @ jx(Pv, cv, Gv, hv, Av, bv)


t0 = time.perf_counter()
x_jax = np.asarray(jx(Pj, cj, Gj, hj, Aj, bj))
gobj_j = jax.grad(jobj, argnums=(0, 1, 2, 3, 4, 5))(Pj, cj, Gj, hj, Aj, bj)
gloss_j = jax.grad(jloss, argnums=(0, 1, 2, 3, 4, 5))(Pj, cj, Gj, hj, Aj, bj)
t_jax = time.perf_counter() - t0
gobj_j = [np.asarray(g) for g in gobj_j]
gloss_j = [np.asarray(g) for g in gloss_j]

# ==========================================================================
# 5. Torch layer
# ==========================================================================
import torch  # noqa: E402
import pounce.torch as pt  # noqa: E402

torch.set_default_dtype(torch.float64)


def torch_grads(fn):
    tP = torch.tensor(P, requires_grad=True)
    tc = torch.tensor(c, requires_grad=True)
    tG = torch.tensor(G, requires_grad=True)
    th = torch.tensor(h, requires_grad=True)
    tA = torch.tensor(A, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    out = fn(tP, tc, tG, th, tA, tb)
    out.backward()
    return [
        v.grad.detach().numpy().copy() for v in (tP, tc, tG, th, tA, tb)
    ], out.item()


def tx(tP, tc, tG, th, tA, tb):
    return pt.solve_qp(P=tP, c=tc, G=tG, h=th, A=tA, b=tb, tol=TOL)


t0 = time.perf_counter()
gobj_t, obj_t = torch_grads(
    lambda *a: (lambda xv: 0.5 * xv @ (a[0] @ xv) + a[1] @ xv)(tx(*a))
)
gloss_t, loss_t = torch_grads(
    lambda *a: torch.tensor(np.asarray(w)).to(torch.float64) @ tx(*a)
)
t_torch = time.perf_counter() - t0
x_torch = tx(*[torch.tensor(v) for v in (P, c, G, h, A, b)]).detach().numpy()

# ==========================================================================
# 6. Compare everything
# ==========================================================================
IDX = {"P": 0, "c": 1, "G": 2, "h": 3, "A": 4, "b": 5}
fails = []


def chk(name, got, ref, tol=1e-6, note=""):
    err = float(np.abs(np.asarray(got) - np.asarray(ref)).max())
    scale = max(1.0, float(np.abs(np.asarray(ref)).max()))
    rel = err / scale
    ok = rel < tol
    if not ok:
        fails.append((name, rel))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<46} abs={err:.3e} rel={rel:.3e} {note}")
    return ok


print("\n=== forward parity ===")
chk("x  host vs jax", x_jax, x_star, 1e-8)
chk("x  host vs torch", x_torch, x_star, 1e-8)

print("\n=== (a) envelope: d obj*/dh  ==  -z ===")
fd_h, sp_h = fd_scalar(obj_of_h, h)
print(f"  FD step-sweep spread = {sp_h:.2e}")
print(f"  reported -z   = {-z_star}")
print(f"  FD  dobj/dh   = {fd_h}")
print(f"  JAX dobj/dh   = {gobj_j[IDX['h']]}")
print(f"  TCH dobj/dh   = {gobj_t[IDX['h']]}")
chk("FD  dobj*/dh vs -z", fd_h, -z_star, 1e-5)
chk("JAX dobj*/dh vs -z", gobj_j[IDX["h"]], -z_star, 1e-7)
chk("TCH dobj*/dh vs -z", gobj_t[IDX["h"]], -z_star, 1e-7)

print("\n=== (b) envelope: d obj*/db  ==  -y (sign per stationarity) ===")
fd_b, sp_b = fd_scalar(obj_of_b, b)
print(f"  FD step-sweep spread = {sp_b:.2e}")
print(f"  reported -y   = {-y_star}   (stationarity sign of y: {SIGN_Y:+.0f})")
print(f"  FD  dobj/db   = {fd_b}")
print(f"  JAX dobj/db   = {gobj_j[IDX['b']]}")
print(f"  TCH dobj/db   = {gobj_t[IDX['b']]}")
chk("FD  dobj*/db vs -y*sign", fd_b, -SIGN_Y * y_star, 1e-5)
chk("JAX dobj*/db vs -y*sign", gobj_j[IDX["b"]], -SIGN_Y * y_star, 1e-7)
chk("TCH dobj*/db vs -y*sign", gobj_t[IDX["b"]], -SIGN_Y * y_star, 1e-7)

print("\n=== (c) dx/dtheta vs independent KKT implicit-diff reconstruction ===")
# Build the reference gradients of loss = w'x by IFT, using the reported
# multipliers and the KKT matrix assembled above.
wn = np.asarray(w)

# d loss/dc : dx = -K^{-1}[I;0;0]  ->  dloss/dc_j = w' dx/dc_j
ref_dc = np.array([ift_dx(dc=np.eye(n)[j])[0] @ wn for j in range(n)])
# d loss/dh : only active rows carry gradient
ref_dh = np.zeros(m_in)
for i in range(m_in):
    if ACT[i]:
        e = np.zeros(m_in)
        e[i] = 1.0
        ref_dh[i] = ift_dx(dh=e)[0] @ wn
# d loss/db
ref_db = np.array([ift_dx(db=np.eye(m_eq)[j])[0] @ wn for j in range(m_eq)])
# d loss/dG (full matrix)
ref_dG = np.zeros_like(G)
for i in range(m_in):
    for j in range(n):
        E = np.zeros_like(G)
        E[i, j] = 1.0
        ref_dG[i, j] = ift_dx(dG=E)[0] @ wn
# d loss/dA
ref_dA = np.zeros_like(A)
for i in range(m_eq):
    for j in range(n):
        E = np.zeros_like(A)
        E[i, j] = 1.0
        ref_dA[i, j] = ift_dx(dA=E)[0] @ wn

chk("JAX dloss/dc vs IFT", gloss_j[IDX["c"]], ref_dc, 1e-7)
chk("TCH dloss/dc vs IFT", gloss_t[IDX["c"]], ref_dc, 1e-7)
chk("JAX dloss/dh vs IFT", gloss_j[IDX["h"]], ref_dh, 1e-7)
chk("TCH dloss/dh vs IFT", gloss_t[IDX["h"]], ref_dh, 1e-7)
chk("JAX dloss/db vs IFT", gloss_j[IDX["b"]], ref_db, 1e-7)
chk("TCH dloss/db vs IFT", gloss_t[IDX["b"]], ref_db, 1e-7)
chk("JAX dloss/dG vs IFT", gloss_j[IDX["G"]], ref_dG, 1e-7)
chk("TCH dloss/dG vs IFT", gloss_t[IDX["G"]], ref_dG, 1e-7)
chk("JAX dloss/dA vs IFT", gloss_j[IDX["A"]], ref_dA, 1e-7)
chk("TCH dloss/dA vs IFT", gloss_t[IDX["A"]], ref_dA, 1e-7)


# FD confirmation on c and h (loss = w'x)
def loss_of(param, val):
    kw = qp_kwargs()
    kw[param] = val
    r = host_solve_qp(**kw, tol=TOL)
    return float(wn @ np.asarray(r.x))


fd_lc, sp_lc = fd_scalar(lambda v: loss_of("c", v), c)
fd_lh, sp_lh = fd_scalar(lambda v: loss_of("h", v), h)
print(f"  FD spreads: c {sp_lc:.2e}  h {sp_lh:.2e}")
chk("FD  dloss/dc vs IFT", fd_lc, ref_dc, 1e-5)
chk("FD  dloss/dh vs IFT", fd_lh, ref_dh, 1e-5)

print("\n=== (d) gradient w.r.t. P vs analytic implicit-diff (symmetrized) ===")
# The layer documents dP as the SYMMETRIC gradient, so the IFT reference must
# be symmetrized too, and the FD check must perturb P symmetrically.
ref_dP = np.zeros_like(P)
for i in range(n):
    for j in range(n):
        E = np.zeros_like(P)
        E[i, j] = 1.0
        ref_dP[i, j] = ift_dx(dP=E)[0] @ wn
ref_dP_sym = 0.5 * (ref_dP + ref_dP.T)


def fd_dP_sym():
    """Central FD with a SYMMETRIC perturbation E_ij + E_ji (halved on the
    diagonal), giving exactly the symmetric-gradient convention."""
    s = 1e-6
    out = np.zeros_like(P)
    for i in range(n):
        for j in range(i, n):
            E = np.zeros_like(P)
            E[i, j] += 1.0
            E[j, i] += 1.0
            if i == j:
                E *= 0.5
            up = host_solve_qp(P=P + s * E, c=c, A=A, b=b, G=G, h=h, tol=TOL)
            dn = host_solve_qp(P=P - s * E, c=c, A=A, b=b, G=G, h=h, tol=TOL)
            d = (wn @ np.asarray(up.x) - wn @ np.asarray(dn.x)) / (2 * s)
            if i == j:
                out[i, i] = d
            else:
                out[i, j] = out[j, i] = 0.5 * d
    return out


fdP = fd_dP_sym()
chk("JAX dloss/dP vs IFT(sym)", gloss_j[IDX["P"]], ref_dP_sym, 1e-7)
chk("TCH dloss/dP vs IFT(sym)", gloss_t[IDX["P"]], ref_dP_sym, 1e-7)
chk("FD  dloss/dP vs IFT(sym)", fdP, ref_dP_sym, 1e-5)

# also the objective-gradient wrt P: obj* = 1/2 x'Px + c'x, explicit + implicit
# d obj*/dP = 1/2 x x' (explicit, symmetrized) + (Px+c)' dx/dP; at the optimum
# the implicit part is NOT zero unless the constraints are absent, so use IFT.
ref_objP = np.zeros_like(P)
g_obj = P @ x_star + c
for i in range(n):
    for j in range(n):
        E = np.zeros_like(P)
        E[i, j] = 1.0
        ref_objP[i, j] = 0.5 * x_star[i] * x_star[j] + g_obj @ ift_dx(dP=E)[0]
ref_objP = 0.5 * (ref_objP + ref_objP.T)
chk("JAX dobj*/dP vs IFT(sym)", gobj_j[IDX["P"]], ref_objP, 1e-7)
chk("TCH dobj*/dP vs IFT(sym)", gobj_t[IDX["P"]], ref_objP, 1e-7)

print("\n=== (e) JAX <-> Torch parity (all six gradients, both losses) ===")
for k, i in IDX.items():
    chk(f"obj-grad  JAX vs Torch  d/d{k}", gobj_j[i], gobj_t[i], 1e-9)
for k, i in IDX.items():
    chk(f"loss-grad JAX vs Torch  d/d{k}", gloss_j[i], gloss_t[i], 1e-9)

print("\n=== extra: torch.autograd.gradcheck on the layer ===")
# NOTE: gradcheck perturbs each entry of P INDEPENDENTLY (asymmetrically),
# which is incompatible with the layer's documented symmetric-gradient
# convention for P.  Checking raw P here is a known false positive; the
# correct check routes P through an explicit symmetrizer so the chain rule
# makes both conventions agree.  We run both to demonstrate exactly that.
gc_kw = dict(eps=1e-6, atol=1e-5, rtol=1e-3, nondet_tol=1e-8)


def run_gradcheck(label, fn, args, expect_fail=False):
    try:
        ok = torch.autograd.gradcheck(fn, args, **gc_kw)
        print(f"  {label}: {ok}")
        if expect_fail:
            print(f"    (unexpectedly PASSED -- reconsider the analysis)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: raised {type(e).__name__}")
        if expect_fail:
            print("    (expected: raw-P perturbation is asymmetric by design)")
        else:
            fails.append((label, float("nan")))
        return False


# (1) no P at all -- c, h, b, and the matrices G, A
run_gradcheck(
    "gradcheck(c,h,b,G,A)",
    lambda cv, hv, bv, Gv, Av: pt.solve_qp(
        P=torch.tensor(P), c=cv, G=Gv, h=hv, A=Av, b=bv, tol=TOL
    ),
    (
        torch.tensor(c, requires_grad=True),
        torch.tensor(h, requires_grad=True),
        torch.tensor(b, requires_grad=True),
        torch.tensor(G, requires_grad=True),
        torch.tensor(A, requires_grad=True),
    ),
)

# (2) P routed through an explicit symmetrizer -- the documented usage
run_gradcheck(
    "gradcheck(sym(M),c,h,b)",
    lambda Mv, cv, hv, bv: pt.solve_qp(
        P=0.5 * (Mv + Mv.T),
        c=cv,
        G=torch.tensor(G),
        h=hv,
        A=torch.tensor(A),
        b=bv,
        tol=TOL,
    ),
    (
        torch.tensor(P, requires_grad=True),
        torch.tensor(c, requires_grad=True),
        torch.tensor(h, requires_grad=True),
        torch.tensor(b, requires_grad=True),
    ),
)

# (3) raw P -- documented to fail (asymmetric FD vs symmetric gradient)
run_gradcheck(
    "gradcheck(raw P) [expected fail]",
    lambda Pv: pt.solve_qp(
        P=Pv,
        c=torch.tensor(c),
        G=torch.tensor(G),
        h=torch.tensor(h),
        A=torch.tensor(A),
        b=torch.tensor(b),
        tol=TOL,
    ),
    (torch.tensor(P, requires_grad=True),),
    expect_fail=True,
)

# gradgradcheck on the non-P arguments
try:
    ok2 = torch.autograd.gradgradcheck(
        lambda cv, hv, bv: pt.solve_qp(
            P=torch.tensor(P),
            c=cv,
            G=torch.tensor(G),
            h=hv,
            A=torch.tensor(A),
            b=bv,
            tol=TOL,
        ),
        (
            torch.tensor(c, requires_grad=True),
            torch.tensor(h, requires_grad=True),
            torch.tensor(b, requires_grad=True),
        ),
        **gc_kw,
    )
    print(f"  gradgradcheck(c,h,b): {ok2}")
except Exception as e:  # noqa: BLE001
    print(f"  gradgradcheck(c,h,b): raised {type(e).__name__}: {str(e)[:200]}")
    print("    (second-order through a one-shot implicit VJP is not expected)")

print(f"\ntimings: host={t_host:.3f}s jax={t_jax:.3f}s torch={t_torch:.3f}s")
if fails:
    print(f"failures ({len(fails)}): {[f[0] for f in fails]}")
    print("VERDICT: FAIL")
else:
    print("VERDICT: PASS")
