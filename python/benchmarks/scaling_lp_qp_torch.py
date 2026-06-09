"""Scaling sweep for the LP/QP/conic stack and the PyTorch differentiable layers.

This is a hands-on QA harness (companion to ``dev-notes/qa-lp-qp-torch.md``),
**not** a CI benchmark — it is meant to be run by a human to find the cliff:
where each solver gets slow, where memory grows, and to confirm failures are
graceful (a status, not a panic/hang). Five sweeps, each independently
selectable:

  * ``qp``    convex-QP interior point (``pounce.qp.solve_qp``): n sweep, dense
              vs scipy-sparse ``P``, with a box and a modest inequality block.
  * ``socp``  second-order cone program (``pounce.qp.solve_socp``): scaling in
              cone *count* (many small cones) and cone *size* (one big cone).
  * ``sos``   SOS/Lasserre (``pounce.sos.sos_minimize``): degree/order sweep —
              expected to blow up combinatorially; we map the practical ceiling.
  * ``torch`` differentiable backward (``pounce.torch.solve_qp``): the VJP
              assembles a *dense* KKT and uses ``torch.linalg.solve`` (flagged
              as a follow-up in ``torch/_qp.py``), so dx/dp time grows ~n³ —
              this quantifies that known risk.
  * ``batch`` ``solve_qp_batch`` vs ``vmap_solve`` vs ``vmap_solve_parallel``:
              batch-size scaling; confirms the threadpool path actually wins.

Run (repo root, a Python with pounce[torch] + scipy installed):
    python python/benchmarks/scaling_lp_qp_torch.py                # all sweeps
    python python/benchmarks/scaling_lp_qp_torch.py --only qp socp
    python python/benchmarks/scaling_lp_qp_torch.py --json out.json
    python python/benchmarks/scaling_lp_qp_torch.py --max-n 2000   # cap the QP sweep

Times are best-of-``--repeat`` wall seconds. Peak RAM is sampled with
``resource.getrusage`` (maxrss) as a coarse high-water mark. Nothing here is
asserted — it prints tables; the report cites the numbers.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time

import numpy as np


def _maxrss_mb() -> float:
    """Process peak RSS in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _best(fn, repeat: int):
    """best-of-repeat wall seconds for fn(); returns (seconds, result)."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


# --------------------------------------------------------------------------
# QP: convex IPM, dense vs sparse P, n sweep
# --------------------------------------------------------------------------
def sweep_qp(ns, repeat, rng):
    from scipy import sparse
    from pounce.qp import solve_qp

    print("\n== convex QP (solve_qp): n sweep, box + ineq, dense vs sparse P ==")
    print(f"{'n':>6} {'m_ineq':>7} | {'dense_s':>9} {'iters':>6} {'status':>10} "
          f"| {'sparse_s':>9} {'iters':>6} | {'maxrss_MB':>10}")
    rows = []
    for n in ns:
        m = max(1, n // 10)
        # Diagonally dominant SPD P (cheap to build at scale), linear term,
        # a thin inequality block Gx <= h kept loose, and a finite box.
        d = rng.uniform(1.0, 3.0, size=n)
        Pd = np.diag(d)
        Ps = sparse.diags(d).tocsc()
        c = rng.standard_normal(n)
        G = rng.standard_normal((m, n))
        h = np.abs(rng.standard_normal(m)) + n  # loose
        lb = -5 * np.ones(n)
        ub = 5 * np.ones(n)

        try:
            t_d, r_d = _best(lambda: solve_qp(Pd, c, G=G, h=h, lb=lb, ub=ub,
                                              tol=1e-8), repeat)
            st, it = r_d.status, getattr(r_d, "iterations", getattr(r_d, "iters", -1))
        except Exception as e:  # graceful-failure probe
            t_d, st, it = float("nan"), f"EXC:{type(e).__name__}", -1
        try:
            t_s, r_s = _best(lambda: solve_qp(Ps, c, G=G, h=h, lb=lb, ub=ub,
                                              tol=1e-8), repeat)
            it_s = getattr(r_s, "iterations", getattr(r_s, "iters", -1))
        except Exception as e:
            t_s, it_s = float("nan"), -1
        mb = _maxrss_mb()
        print(f"{n:>6} {m:>7} | {t_d:>9.4f} {it:>6} {st:>10} "
              f"| {t_s:>9.4f} {it_s:>6} | {mb:>10.1f}")
        rows.append(dict(n=n, m=m, dense_s=t_d, sparse_s=t_s, iters=it,
                         status=st, maxrss_mb=mb))
    return rows


# --------------------------------------------------------------------------
# SOCP: scale cone count and cone size
# --------------------------------------------------------------------------
def sweep_socp(repeat, rng):
    from pounce.qp import solve_socp

    print("\n== SOCP (solve_socp): many small cones vs one big cone ==")
    print(f"{'mode':>10} {'n':>6} {'cones':>6} {'rows':>7} | {'solve_s':>9} "
          f"{'status':>10} {'maxrss_MB':>10}")
    rows = []

    # (a) many small cones: k cones of fixed size 3, n = k vars, obj min sum.
    for k in (8, 32, 128, 512, 2048):
        n = k
        c = np.ones(n)
        # each var bounded in its own little SOC: (1; x_i something)
        G_blocks, h_blocks, cones = [], [], []
        for i in range(k):
            row0 = np.zeros(n)
            rowi = np.zeros(n); rowi[i] = -1.0
            G_blocks += [row0, rowi]
            h_blocks += [1.0, 0.0]
            cones.append(("soc", 2))
        G = np.array(G_blocks); h = np.array(h_blocks)
        try:
            t, r = _best(lambda: solve_socp(P=None, c=c, G=G, h=h, cones=cones),
                         repeat)
            st = r.status
        except Exception as e:
            t, st = float("nan"), f"EXC:{type(e).__name__}"
        print(f"{'many':>10} {n:>6} {k:>6} {len(h):>7} | {t:>9.4f} {st:>10} "
              f"{_maxrss_mb():>10.1f}")
        rows.append(dict(mode="many", n=n, cones=k, rows=len(h), solve_s=t, status=st))

    # (b) one big cone: ||x|| <= 1, n grows.
    for n in (8, 64, 256, 1024, 4096):
        c = rng.standard_normal(n)
        G = np.vstack([np.zeros((1, n)), -np.eye(n)])
        h = np.concatenate([[1.0], np.zeros(n)])
        cones = [("soc", n + 1)]
        try:
            t, r = _best(lambda: solve_socp(P=None, c=c, G=G, h=h, cones=cones),
                         repeat)
            st = r.status
        except Exception as e:
            t, st = float("nan"), f"EXC:{type(e).__name__}"
        print(f"{'one-big':>10} {n:>6} {1:>6} {len(h):>7} | {t:>9.4f} {st:>10} "
              f"{_maxrss_mb():>10.1f}")
        rows.append(dict(mode="one-big", n=n, cones=1, rows=len(h), solve_s=t, status=st))
    return rows


# --------------------------------------------------------------------------
# SOS: degree (n_vars) x order sweep -> combinatorial blow-up
# --------------------------------------------------------------------------
def sweep_sos(repeat):
    from pounce.sos import sos_minimize

    print("\n== SOS/Lasserre (sos_minimize): n_vars x order, the practical ceiling ==")
    print(f"{'n_vars':>7} {'order':>6} | {'solve_s':>9} {'lower_bound':>13} {'status':>14}")
    rows = []
    for nv in (1, 2, 3, 4, 5):
        # convex quadratic bowl in nv vars: sum (x_i - 1)^2 + 1, min = 1.
        obj = {}
        for i in range(nv):
            e = tuple(2 if j == i else 0 for j in range(nv)); obj[e] = obj.get(e, 0) + 1.0
            e1 = tuple(1 if j == i else 0 for j in range(nv)); obj[e1] = obj.get(e1, 0) - 2.0
        obj[tuple(0 for _ in range(nv))] = float(nv) + 1.0
        for order in (1, 2, 3):
            try:
                t, r = _best(lambda: sos_minimize(obj, order=order), repeat)
                lb = getattr(r, "lower_bound", float("nan"))
                st = getattr(r, "status", "ok")
            except Exception as e:
                t, lb, st = float("nan"), float("nan"), f"EXC:{type(e).__name__}"
            print(f"{nv:>7} {order:>6} | {t:>9.4f} {lb:>13.6f} {str(st):>14}")
            rows.append(dict(n_vars=nv, order=order, solve_s=t, lower_bound=lb,
                             status=str(st)))
    return rows


# --------------------------------------------------------------------------
# Torch differentiable backward: dense-KKT VJP cost vs n
# --------------------------------------------------------------------------
def sweep_torch(ns, repeat):
    import torch
    torch.set_default_dtype(torch.float64)
    from pounce.torch import solve_qp

    print("\n== torch solve_qp: forward vs backward (dense KKT VJP) cost vs n ==")
    print(f"{'n':>6} {'m_ineq':>7} | {'fwd_s':>9} {'bwd_s':>9} {'bwd/fwd':>8} "
          f"{'maxrss_MB':>10}")
    rows = []
    for n in ns:
        m = max(1, n // 10)
        d = torch.rand(n) + 1.0
        P = torch.diag(d)
        G = torch.randn(m, n)
        h = torch.abs(torch.randn(m)) + n

        def fwd():
            c = torch.randn(n, requires_grad=True)
            return solve_qp(P=P, c=c, G=G, h=h), c

        # forward only
        t_f, _ = _best(lambda: solve_qp(P=P, c=torch.randn(n), G=G, h=h), repeat)

        # forward + backward
        def fb():
            c = torch.randn(n, requires_grad=True)
            x = solve_qp(P=P, c=c, G=G, h=h)
            x.sum().backward()
            return c.grad
        try:
            t_fb, _ = _best(fb, repeat)
            ratio = t_fb / t_f if t_f > 0 else float("nan")
            print(f"{n:>6} {m:>7} | {t_f:>9.4f} {t_fb:>9.4f} {ratio:>8.2f} "
                  f"{_maxrss_mb():>10.1f}")
            rows.append(dict(n=n, m=m, fwd_s=t_f, bwd_s=t_fb, ratio=ratio))
        except Exception as e:
            print(f"{n:>6} {m:>7} | backward EXC: {type(e).__name__}: {e}")
            rows.append(dict(n=n, m=m, fwd_s=t_f, bwd_s=float("nan"),
                             error=type(e).__name__))
    return rows


# --------------------------------------------------------------------------
# Batch: solve_qp_batch vs vmap_solve vs vmap_solve_parallel
# --------------------------------------------------------------------------
def sweep_batch(repeat):
    import torch
    torch.set_default_dtype(torch.float64)
    from pounce.torch import solve_qp_batch, vmap_solve, vmap_solve_parallel

    print("\n== batch QP: solve_qp_batch vs vmap_solve vs vmap_solve_parallel ==")
    print(f"{'batch':>6} | {'qp_batch_s':>11} {'vmap_s':>9} {'vmap_par_s':>11} "
          f"{'par_speedup':>11}")
    rows = []
    n = 8
    P = torch.eye(n)
    G = torch.randn(3, n)
    h = torch.abs(torch.randn(3)) + n

    # vmap path: an unconstrained quadratic param'd by the target p.
    def f(x, p):
        return torch.sum((x - p) ** 2)

    for B in (4, 16, 64, 256):
        cb = torch.randn(B, n)
        t_qpb, _ = _best(lambda: solve_qp_batch(P=P, c=cb, G=G, h=h), repeat)

        pb = torch.randn(B, n)
        try:
            t_vm, _ = _best(lambda: vmap_solve(pb, f=f, x0=torch.zeros(n), n=n,
                                               options={"print_level": 0}), repeat)
        except Exception as e:
            t_vm = float("nan")
        try:
            t_vp, _ = _best(lambda: vmap_solve_parallel(pb, f=f, x0=torch.zeros(n),
                                                        n=n,
                                                        options={"print_level": 0}),
                            repeat)
        except Exception as e:
            t_vp = float("nan")
        sp = t_vm / t_vp if (t_vp and t_vp == t_vp and t_vp > 0) else float("nan")
        print(f"{B:>6} | {t_qpb:>11.4f} {t_vm:>9.4f} {t_vp:>11.4f} {sp:>11.2f}")
        rows.append(dict(batch=B, qp_batch_s=t_qpb, vmap_s=t_vm,
                         vmap_par_s=t_vp, par_speedup=sp))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+",
                    choices=["qp", "socp", "sos", "torch", "batch"],
                    help="run only these sweeps (default: all)")
    ap.add_argument("--max-n", type=int, default=10000,
                    help="cap n for the qp/torch sweeps")
    ap.add_argument("--repeat", type=int, default=3, help="best-of-N timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write all rows to this JSON file")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    only = set(args.only) if args.only else {"qp", "socp", "sos", "torch", "batch"}
    qp_ns = [n for n in (10, 100, 1000, 5000, 10000) if n <= args.max_n]
    torch_ns = [n for n in (10, 50, 100, 250, 500, 1000) if n <= args.max_n]

    out = {}
    if "qp" in only:
        out["qp"] = sweep_qp(qp_ns, args.repeat, rng)
    if "socp" in only:
        out["socp"] = sweep_socp(args.repeat, rng)
    if "sos" in only:
        out["sos"] = sweep_sos(args.repeat)
    if "torch" in only:
        out["torch"] = sweep_torch(torch_ns, args.repeat)
    if "batch" in only:
        out["batch"] = sweep_batch(args.repeat)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
