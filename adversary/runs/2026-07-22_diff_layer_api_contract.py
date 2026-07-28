"""Adversary cross-check: differentiable-layer FRAMEWORK INTEGRATION contract.

Family: diff   Class: API contract / batching / jit / dtype / device / failure
Entry points: pounce.jax.solve_qp / solve_qp_batch, pounce.torch.solve_qp /
              solve_qp_batch, QpLayer.
Oracles:
  (a) loop-of-single-solves         (b) non-jit reference
  (c)/(d) documented-behaviour audit (must be *clear*, never silently wrong)
  (e) status contract: never a silently-successful NaN
  (f) torch.autograd.gradcheck / gradgradcheck in float64
  (g) cross-framework parity JAX <-> Torch

Test problem (strictly convex QP, unique solution, active inequality):
    min  1/2 x'Px + c(theta)'x   s.t.  Gx <= h,  Ax = b
    n = 3, 3 inequalities (incl. one active), 1 equality.
"""
import sys
import time
import traceback

import numpy as np

FAIL = []
NOTE = []


def record(name, ok, detail=""):
    tag = "ok " if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(f"{name}: {detail}")


def note(msg):
    print(f"  [note] {msg}")
    NOTE.append(msg)


# ---------------------------------------------------------------- problem ---
P_np = np.array([[3.0, 0.5, 0.2],
                 [0.5, 2.0, 0.3],
                 [0.2, 0.3, 4.0]])
G_np = np.array([[1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0],
                 [-1.0, 0.0, 0.5],
                 [-1.0, -1.0, 0.0]])
h_np = np.array([0.05, 1.0, 2.0, 5.0])
A_np = np.array([[1.0, -1.0, 1.0]])
b_np = np.array([0.4])


def c_of(theta):
    """Smooth c(theta), theta a length-2 vector."""
    t0, t1 = theta[0], theta[1]
    return np.array([-1.0 + 2.0 * t0, 0.5 - t1, 0.3 * t0 + 0.7 * t1])


THETAS = np.array([[0.10, 0.20],
                   [0.35, -0.15],
                   [-0.25, 0.40],
                   [0.50, 0.50]])

import pounce.jax as pjax      # noqa: E402
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402
import pounce.torch as ptorch  # noqa: E402
import torch                   # noqa: E402


def c_jnp(theta):
    return jnp.array([-1.0 + 2.0 * theta[0],
                      0.5 - theta[1],
                      0.3 * theta[0] + 0.7 * theta[1]])


def c_torch(theta):
    return torch.stack([-1.0 + 2.0 * theta[0],
                        0.5 - theta[1],
                        0.3 * theta[0] + 0.7 * theta[1]])


Pj, Gj, hj, Aj, bj = (jnp.asarray(v) for v in (P_np, G_np, h_np, A_np, b_np))
Pt, Gt, ht, At, bt = (torch.tensor(v, dtype=torch.float64)
                      for v in (P_np, G_np, h_np, A_np, b_np))


def jax_solve(theta):
    return pjax.solve_qp(P=Pj, c=c_jnp(theta), G=Gj, h=hj, A=Aj, b=bj)


def torch_solve(theta):
    return ptorch.solve_qp(P=Pt, c=c_torch(theta), G=Gt, h=ht, A=At, b=bt)


def jax_loss(theta):
    x = jax_solve(theta)
    w = jnp.array([1.0, -2.0, 0.5])
    return jnp.sum(w * x ** 2)


def torch_loss(theta):
    x = torch_solve(theta)
    w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    return (w * x ** 2).sum()


print("=" * 72)
print("SANITY: forward solve is correct (cvxpy oracle)")
print("=" * 72)
x0_j = np.asarray(jax_solve(jnp.asarray(THETAS[0])))
try:
    import cvxpy as cp
    xv = cp.Variable(3)
    c0 = c_of(THETAS[0])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_np)) + c0 @ xv),
                      [G_np @ xv <= h_np, A_np @ xv == b_np])
    prob.solve(solver=cp.CLARABEL)
    err = float(np.max(np.abs(x0_j - xv.value)))
    record("forward vs cvxpy", err < 1e-7, f"inf_err={err:.2e} x={x0_j}")
    n_active = int(np.sum(G_np @ x0_j > h_np - 1e-6))
    record("problem has an active inequality", n_active >= 1, f"active={n_active}")
except Exception as e:  # pragma: no cover
    note(f"cvxpy oracle unavailable: {e}")

# --------------------------------------------------------------- (a) vmap ---
print()
print("=" * 72)
print("(a) BATCHING: jax.vmap and torch batch vs loop-of-single-solves")
print("=" * 72)

loop_x = np.stack([np.asarray(jax_solve(jnp.asarray(t))) for t in THETAS])
loop_g = np.stack([np.asarray(jax.grad(jax_loss)(jnp.asarray(t))) for t in THETAS])

try:
    t0 = time.perf_counter()
    vm_x = np.asarray(jax.vmap(jax_solve)(jnp.asarray(THETAS)))
    dt = time.perf_counter() - t0
    e = float(np.max(np.abs(vm_x - loop_x)))
    record("jax.vmap forward == loop (exact)", e == 0.0, f"max|d|={e:.2e} t={dt:.3f}s")
except Exception as e:
    record("jax.vmap forward", False, f"{type(e).__name__}: {e}")

try:
    vm_g = np.asarray(jax.vmap(jax.grad(jax_loss))(jnp.asarray(THETAS)))
    e = float(np.max(np.abs(vm_g - loop_g)))
    record("jax.vmap(grad) == loop (exact)", e == 0.0, f"max|d|={e:.2e}")
except Exception as e:
    record("jax.vmap(grad)", False, f"{type(e).__name__}: {e}")

# jax batch API
try:
    cs = jnp.stack([c_jnp(jnp.asarray(t)) for t in THETAS])
    bx = np.asarray(pjax.solve_qp_batch(P=Pj, c=cs, G=Gj, h=hj, A=Aj, b=bj))
    e = float(np.max(np.abs(bx - loop_x)))
    record("pounce.jax.solve_qp_batch == loop", e < 1e-9, f"max|d|={e:.2e}")
except Exception as e:
    record("pounce.jax.solve_qp_batch", False, f"{type(e).__name__}: {e}")

# torch batch API
loop_xt = np.stack([torch_solve(torch.tensor(t, dtype=torch.float64)).detach().numpy()
                    for t in THETAS])
try:
    cst = torch.stack([c_torch(torch.tensor(t, dtype=torch.float64)) for t in THETAS])
    bxt = ptorch.solve_qp_batch(P=Pt, c=cst, G=Gt, h=ht, A=At, b=bt).detach().numpy()
    e = float(np.max(np.abs(bxt - loop_xt)))
    record("pounce.torch.solve_qp_batch == loop", e < 1e-9, f"max|d|={e:.2e}")
except Exception as e:
    record("pounce.torch.solve_qp_batch", False, f"{type(e).__name__}: {e}")

# batched-gradient parity (torch): sum of per-row losses
def torch_batch_loss(thetas_t):
    cs = torch.stack([c_torch(thetas_t[i]) for i in range(thetas_t.shape[0])])
    xs = ptorch.solve_qp_batch(P=Pt, c=cs, G=Gt, h=ht, A=At, b=bt)
    w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    return (w * xs ** 2).sum()


try:
    th = torch.tensor(THETAS, dtype=torch.float64, requires_grad=True)
    torch_batch_loss(th).backward()
    gb = th.grad.detach().numpy()
    gl = []
    for t in THETAS:
        tt = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        torch_loss(tt).backward()
        gl.append(tt.grad.detach().numpy())
    gl = np.stack(gl)
    e = float(np.max(np.abs(gb - gl)))
    record("torch batch grad == loop grad", e < 1e-9, f"max|d|={e:.2e}")
except Exception as e:
    record("torch batch grad", False, f"{type(e).__name__}: {e}")

# torch vmap_solve helper on the QP layer via python loop is trivially fine;
# test torch.func.vmap over the *single* layer (the interesting contract)
try:
    out = torch.func.vmap(torch_solve)(torch.tensor(THETAS, dtype=torch.float64))
    e = float(np.max(np.abs(out.detach().numpy() - loop_xt)))
    record("torch.func.vmap over solve_qp == loop", e < 1e-9, f"max|d|={e:.2e}")
except Exception as e:
    note(f"torch.func.vmap over solve_qp raises (expected; eager host solve): "
         f"{type(e).__name__}: {str(e)[:160]}")

# ----------------------------------------------------------------- (b) jit ---
print()
print("=" * 72)
print("(b) JIT: jax.jit forward+grad vs eager")
print("=" * 72)
try:
    jf = jax.jit(jax_solve)
    xj = np.asarray(jf(jnp.asarray(THETAS[0])))
    e = float(np.max(np.abs(xj - loop_x[0])))
    record("jax.jit forward == eager (exact)", e == 0.0, f"max|d|={e:.2e}")
except Exception as e:
    record("jax.jit forward", False, f"{type(e).__name__}: {e}")

try:
    jg = jax.jit(jax.grad(jax_loss))
    gj = np.asarray(jg(jnp.asarray(THETAS[0])))
    e = float(np.max(np.abs(gj - loop_g[0])))
    record("jax.jit(grad) == eager (exact)", e == 0.0, f"max|d|={e:.2e}")
except Exception as e:
    record("jax.jit(grad)", False, f"{type(e).__name__}: {e}")

try:
    jvg = jax.jit(jax.vmap(jax.grad(jax_loss)))
    g2 = np.asarray(jvg(jnp.asarray(THETAS)))
    e = float(np.max(np.abs(g2 - loop_g)))
    record("jit(vmap(grad)) == loop (exact)", e == 0.0, f"max|d|={e:.2e}")
except Exception as e:
    record("jit(vmap(grad))", False, f"{type(e).__name__}: {e}")

# --------------------------------------------------------------- (c) dtype ---
print()
print("=" * 72)
print("(c) DTYPE: float32 inputs")
print("=" * 72)
try:
    P32 = torch.tensor(P_np, dtype=torch.float32)
    c32 = torch.tensor(c_of(THETAS[0]), dtype=torch.float32, requires_grad=True)
    x32 = ptorch.solve_qp(P=P32, c=c32,
                          G=torch.tensor(G_np, dtype=torch.float32),
                          h=torch.tensor(h_np, dtype=torch.float32),
                          A=torch.tensor(A_np, dtype=torch.float32),
                          b=torch.tensor(b_np, dtype=torch.float32))
    e = float(np.max(np.abs(x32.detach().numpy() - loop_x[0])))
    note(f"torch float32 ACCEPTED (silent upcast to float64): out.dtype={x32.dtype}, "
         f"err_vs_f64={e:.2e}")
    x32.sum().backward()
    note(f"torch float32 grad dtype={c32.grad.dtype} (input was float32)")
    record("torch float32 grad dtype matches input", c32.grad.dtype == torch.float32,
           str(c32.grad.dtype))
    record("torch float32 accuracy consistent with f32 input rounding",
           e < 1e-6, f"err={e:.2e}")
except Exception as e:
    note(f"torch float32 REJECTED: {type(e).__name__}: {str(e)[:200]}")

try:
    x32j = pjax.solve_qp(P=jnp.asarray(P_np, dtype=jnp.float32),
                         c=jnp.asarray(c_of(THETAS[0]), dtype=jnp.float32),
                         G=jnp.asarray(G_np, dtype=jnp.float32),
                         h=jnp.asarray(h_np, dtype=jnp.float32),
                         A=jnp.asarray(A_np, dtype=jnp.float32),
                         b=jnp.asarray(b_np, dtype=jnp.float32))
    e = float(np.max(np.abs(np.asarray(x32j) - loop_x[0])))
    note(f"jax float32 ACCEPTED (silent upcast): out.dtype={x32j.dtype} err={e:.2e}")
    record("jax float32 accuracy consistent with f32 input rounding",
           e < 1e-6, f"err={e:.2e}")
except Exception as e:
    note(f"jax float32 REJECTED: {type(e).__name__}: {str(e)[:200]}")

# -------------------------------------------------------------- (d) device ---
print()
print("=" * 72)
print("(d) DEVICE: MPS tensors into the torch layer")
print("=" * 72)
if torch.backends.mps.is_available():
    # MPS has no float64; feed float32 MPS tensors (the realistic user case).
    try:
        dev = torch.device("mps")
        Pm = torch.tensor(P_np, dtype=torch.float32, device=dev)
        cm = torch.tensor(c_of(THETAS[0]), dtype=torch.float32, device=dev,
                          requires_grad=True)
        Gm = torch.tensor(G_np, dtype=torch.float32, device=dev)
        hm = torch.tensor(h_np, dtype=torch.float32, device=dev)
        Am = torch.tensor(A_np, dtype=torch.float32, device=dev)
        bm = torch.tensor(b_np, dtype=torch.float32, device=dev)
        xm = ptorch.solve_qp(P=Pm, c=cm, G=Gm, h=hm, A=Am, b=bm)
        e = float(np.max(np.abs(xm.detach().cpu().numpy() - loop_x[0])))
        note(f"MPS forward SUCCEEDED: out.device={xm.device} out.dtype={xm.dtype} "
             f"err={e:.2e}")
        record("MPS forward numerically correct", e < 1e-9, f"err={e:.2e}")
        record("MPS output device == input device", xm.device.type == "mps",
               f"in=mps out={xm.device}")
        try:
            xm.sum().backward()
            note(f"MPS backward SUCCEEDED: grad.device={cm.grad.device} "
                 f"grad={cm.grad.detach().cpu().numpy()}")
        except Exception as be:
            note(f"MPS backward RAISED: {type(be).__name__}: {str(be)[:250]}")
            record("MPS backward: clear error or works (not silent garbage)",
                   True, "raised, which is acceptable")
    except Exception as e:
        note(f"MPS forward RAISED: {type(e).__name__}: {str(e)[:250]}")
        record("MPS: clear error (acceptable) rather than wrong result", True, "raised")
else:
    note("MPS unavailable; skipped")

# ------------------------------------------------------- (e) failed solves ---
print()
print("=" * 72)
print("(e) GRADIENT THROUGH A FAILED SOLVE (infeasible / iteration limit)")
print("=" * 72)

# Infeasible: Ax=b with contradictory equalities.
A_inf = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
b_inf = np.array([0.0, 1.0])


def probe_failure(label, fn):
    try:
        out = fn()
        arr = np.asarray(out)
        has_nan = bool(np.any(~np.isfinite(arr)))
        if has_nan:
            record(f"{label}: NaN returned WITHOUT error", False,
                   f"SILENT NaN: {arr}")
        else:
            record(f"{label}: returned finite values without error", False,
                   f"silently 'succeeded' with {arr}")
    except Exception as e:
        msg = str(e)
        clear = ("status" in msg) or ("optimal" in msg) or ("infeasible" in msg.lower())
        record(f"{label}: raises a clear error", clear,
               f"{type(e).__name__}: {msg[:180]}")


probe_failure("torch infeasible forward",
              lambda: ptorch.solve_qp(P=Pt, c=c_torch(torch.tensor(THETAS[0])),
                                      G=Gt, h=ht,
                                      A=torch.tensor(A_inf, dtype=torch.float64),
                                      b=torch.tensor(b_inf, dtype=torch.float64)
                                      ).detach())
probe_failure("jax infeasible forward",
              lambda: pjax.solve_qp(P=Pj, c=c_jnp(jnp.asarray(THETAS[0])),
                                    G=Gj, h=hj,
                                    A=jnp.asarray(A_inf), b=jnp.asarray(b_inf)))
probe_failure("jax infeasible under jit",
              lambda: jax.jit(lambda cc: pjax.solve_qp(
                  P=Pj, c=cc, G=Gj, h=hj,
                  A=jnp.asarray(A_inf), b=jnp.asarray(b_inf)))(
                      c_jnp(jnp.asarray(THETAS[0]))))

# iteration limit
probe_failure("torch max_iter=1",
              lambda: ptorch.solve_qp(P=Pt, c=c_torch(torch.tensor(THETAS[0])),
                                      G=Gt, h=ht, A=At, b=bt, max_iter=1).detach())
probe_failure("jax max_iter=1",
              lambda: pjax.solve_qp(P=Pj, c=c_jnp(jnp.asarray(THETAS[0])),
                                    G=Gj, h=hj, A=Aj, b=bj, max_iter=1))

# gradient through a failed solve (must not silently produce NaN grads)
def grad_through_failure(label, fn):
    try:
        g = fn()
        arr = np.asarray(g)
        if np.any(~np.isfinite(arr)):
            record(f"{label}: SILENT NaN GRADIENT", False, str(arr))
        else:
            record(f"{label}: finite grad from a failed solve", False, str(arr))
    except Exception as e:
        record(f"{label}: raises", True, f"{type(e).__name__}: {str(e)[:140]}")


def _t_infeas_grad():
    cc = torch.tensor(c_of(THETAS[0]), dtype=torch.float64, requires_grad=True)
    x = ptorch.solve_qp(P=Pt, c=cc, G=Gt, h=ht,
                        A=torch.tensor(A_inf, dtype=torch.float64),
                        b=torch.tensor(b_inf, dtype=torch.float64))
    x.sum().backward()
    return cc.grad.numpy()


grad_through_failure("torch grad through infeasible", _t_infeas_grad)
grad_through_failure(
    "jax grad through infeasible",
    lambda: jax.grad(lambda cc: jnp.sum(pjax.solve_qp(
        P=Pj, c=cc, G=Gj, h=hj, A=jnp.asarray(A_inf), b=jnp.asarray(b_inf))))(
            c_jnp(jnp.asarray(THETAS[0]))))

# batch containing one failing row: must not silently return good rows + junk
def _batch_one_bad():
    cs = torch.stack([c_torch(torch.tensor(t, dtype=torch.float64)) for t in THETAS])
    hs = torch.stack([ht] * len(THETAS)).clone()
    # rows 0 and 3 of G are  (x1+x2)  and  -(x1+x2):  x1+x2 <= -50 and
    # x1+x2 >= 50 simultaneously => strictly primal infeasible.
    hs[2] = torch.tensor([-50.0, 1.0, 2.0, -50.0], dtype=torch.float64)
    return ptorch.solve_qp_batch(P=Pt, c=cs, G=Gt, h=hs, A=At, b=bt).detach()


probe_failure("torch batch with one infeasible row", _batch_one_bad)


def _jbatch_one_bad():
    cs = jnp.stack([c_jnp(jnp.asarray(t)) for t in THETAS])
    hs = jnp.stack([hj] * len(THETAS))
    hs = hs.at[2].set(jnp.array([-50.0, 1.0, 2.0, -50.0]))
    return pjax.solve_qp_batch(P=Pj, c=cs, G=Gj, h=hs, A=Aj, b=bj)


probe_failure("jax batch with one infeasible row", _jbatch_one_bad)

# lb/ub are documented as NON-differentiable (folded in as constants).
try:
    lbv = torch.tensor([-5.0, -5.0, -5.0], dtype=torch.float64, requires_grad=True)
    xx = ptorch.solve_qp(P=Pt, c=c_torch(torch.tensor(THETAS[0], dtype=torch.float64)),
                         G=Gt, h=ht, A=At, b=bt, lb=lbv)
    xx.sum().backward()
    note(f"torch lb requires_grad -> lb.grad={lbv.grad} (documented: no gradient "
         f"flows to lb/ub); silently None/zero rather than an error")
except Exception as e:
    note(f"torch lb requires_grad raises: {type(e).__name__}: {str(e)[:160]}")


# -------------------------------------------------------- (f) second order ---
print()
print("=" * 72)
print("(f) SECOND ORDER: torch.autograd.gradcheck / gradgradcheck (float64)")
print("=" * 72)

# Use a strictly-feasible-interior problem so the active set is locally
# constant (the layer is only twice-differentiable where the active set is).
Ps = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=torch.float64)
Gs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
hs_ = torch.tensor([5.0, 5.0], dtype=torch.float64)


def f_c(c):
    return ptorch.solve_qp(P=Ps, c=c, G=Gs, h=hs_)


c_in = torch.tensor([-0.7, 0.4], dtype=torch.float64, requires_grad=True)
try:
    ok = torch.autograd.gradcheck(f_c, (c_in,), eps=1e-6, atol=1e-6, rtol=1e-4)
    record("gradcheck(dx/dc)", bool(ok))
except Exception as e:
    record("gradcheck(dx/dc)", False, f"{type(e).__name__}: {str(e)[:300]}")

try:
    ok = torch.autograd.gradgradcheck(f_c, (c_in,), eps=1e-6, atol=1e-5, rtol=1e-3)
    record("gradgradcheck(dx/dc)", bool(ok))
except Exception as e:
    record("gradgradcheck(dx/dc)", False, f"{type(e).__name__}: {str(e)[:400]}")

# also with an active inequality (P, h differentiable)
h_in = torch.tensor(h_np, dtype=torch.float64, requires_grad=True)


def f_h(hh):
    return ptorch.solve_qp(P=Pt, c=torch.tensor(c_of(THETAS[0]), dtype=torch.float64),
                           G=Gt, h=hh, A=At, b=bt)


try:
    ok = torch.autograd.gradcheck(f_h, (h_in,), eps=1e-6, atol=1e-6, rtol=1e-4)
    record("gradcheck(dx/dh) with active constraint", bool(ok))
except Exception as e:
    record("gradcheck(dx/dh) with active constraint", False,
           f"{type(e).__name__}: {str(e)[:300]}")

try:
    ok = torch.autograd.gradgradcheck(f_h, (h_in,), eps=1e-6, atol=1e-4, rtol=1e-2)
    record("gradgradcheck(dx/dh) with active constraint", bool(ok))
except Exception as e:
    record("gradgradcheck(dx/dh) with active constraint", False,
           f"{type(e).__name__}: {str(e)[:400]}")

# ---------------------------------------------------------- (g) FD + parity ---
print()
print("=" * 72)
print("(g) CROSS-FRAMEWORK PARITY + central finite differences (float64)")
print("=" * 72)


def fd_grad(theta, eps=1e-6):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        lp = float(jax_loss(jnp.asarray(tp)))
        lm = float(jax_loss(jnp.asarray(tm)))
        g[i] = (lp - lm) / (2 * eps)
    return g


for k, th_np in enumerate(THETAS):
    gj = np.asarray(jax.grad(jax_loss)(jnp.asarray(th_np)))
    tt = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
    torch_loss(tt).backward()
    gt = tt.grad.detach().numpy()
    gfd = fd_grad(th_np)
    e_par = float(np.max(np.abs(gj - gt)))
    e_fd = float(np.max(np.abs(gj - gfd)) / max(1.0, np.max(np.abs(gfd))))
    record(f"theta[{k}] JAX<->Torch grad parity", e_par < 1e-9, f"max|d|={e_par:.2e}")
    record(f"theta[{k}] grad vs central FD", e_fd < 1e-5,
           f"rel={e_fd:.2e} ad={gj} fd={gfd}")
    xj_ = np.asarray(jax_solve(jnp.asarray(th_np)))
    xt_ = torch_solve(torch.tensor(th_np, dtype=torch.float64)).detach().numpy()
    record(f"theta[{k}] JAX<->Torch forward parity",
           float(np.max(np.abs(xj_ - xt_))) < 1e-10,
           f"max|d|={float(np.max(np.abs(xj_ - xt_))):.2e}")

# ------------------------------------------------------------------ report ---
print()
print("=" * 72)
print(f"FAILURES: {len(FAIL)}")
for f in FAIL:
    print("  - " + f)
print(f"NOTES: {len(NOTE)}")
for n_ in NOTE:
    print("  * " + n_)
print("VERDICT: PASS" if not FAIL else f"VERDICT: FAIL ({len(FAIL)} contract breaks)")
