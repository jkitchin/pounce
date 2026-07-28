"""Adversary cross-check: non-finite / malformed numeric input into public APIs.

Family: api (contracts, option handling, input edge cases)
Class:  robustness / input validation
Source: N/A -- contract test, not a published problem.

Contract under test
-------------------
 * +/-Inf in BOUNDS is LEGITIMATE (unbounded in that direction) and must be
   handled correctly -- answer must match a finite-but-huge-bound reference.
 * NaN anywhere is ALWAYS invalid -> clear Python exception, never a result.
 * +/-Inf in P/A/G/c/b/h is invalid -> clear Python exception.
 * Odd-but-valid numpy inputs (int dtype, float32, F-order, transposed view,
   non-contiguous slice, read-only) must give the SAME answer as the base.
 * Bad shapes -> clear exception. x0 with NaN/Inf -> clear exception.

Classification per case:
  PASS      - clean Python exception (for invalid input), or correct answer
              (for valid input)
  BUG       - status 'optimal' with NaN/Inf in x, or a silently wrong answer,
              or an interpreter crash (segfault/abort), or a hang.
  PANIC     - Rust panic surfaced as pyo3 PanicException (noted, sub-bug)

Each case runs in its OWN subprocess so a crash is detected rather than
killing the harness.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.abspath(__file__)
TOL = 1e-6

# --------------------------------------------------------------------------
# Base problem (strictly convex QP, unique solution)
#   min 0.5 x'Px + c'x   s.t. Ax=b, Gx<=h, lb<=x<=ub
# --------------------------------------------------------------------------


def base():
    P = np.array(
        [[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 5.0]]
    )
    c = np.array([-1.0, -2.0, -3.0, -4.0])
    A = np.array([[1.0, 1.0, 1.0, 1.0]])
    b = np.array([2.0])
    G = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0]])
    h = np.array([0.5, 0.3])
    lb = np.array([-5.0, -5.0, -5.0, -5.0])
    ub = np.array([5.0, 5.0, 5.0, 5.0])
    return dict(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)


# --------------------------------------------------------------------------
# Case definitions.  Each returns (kwargs, expect) where expect is one of
#   "error"          -> must raise a clean Python Exception
#   ("value", obj)   -> must return status optimal with objective ~= obj
#   ("match_ref",)   -> must match the reference solve of the same kwargs family
# --------------------------------------------------------------------------

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn

    return deco


# ---------- baseline ----------
@case("baseline")
def _baseline():
    return base(), "ok"


# ---------- legitimate infinite bounds ----------
@case("bounds_inf_both")
def _c():
    k = base()
    k["lb"] = np.full(4, -np.inf)
    k["ub"] = np.full(4, np.inf)
    return k, "ok"


@case("bounds_huge_both_REF")
def _c():
    k = base()
    k["lb"] = np.full(4, -1e12)
    k["ub"] = np.full(4, 1e12)
    return k, "ok"


@case("bounds_none_REF")
def _c():
    k = base()
    k["lb"] = None
    k["ub"] = None
    return k, "ok"


@case("bounds_inf_mixed")
def _c():
    k = base()
    k["lb"] = np.array([-np.inf, -1.0, -np.inf, 0.0])
    k["ub"] = np.array([np.inf, np.inf, 2.0, np.inf])
    return k, "ok"


@case("bounds_huge_mixed_REF")
def _c():
    k = base()
    k["lb"] = np.array([-1e12, -1.0, -1e12, 0.0])
    k["ub"] = np.array([1e12, 1e12, 2.0, 1e12])
    return k, "ok"


# ---------- NaN everywhere (all must error) ----------
for _slot, _idx in [
    ("P", (0, 0)),
    ("P", (1, 2)),
    ("c", (2,)),
    ("A", (0, 1)),
    ("b", (0,)),
    ("G", (1, 3)),
    ("h", (0,)),
    ("lb", (2,)),
    ("ub", (3,)),
]:
    def _mk(slot=_slot, idx=_idx):
        def _c():
            k = base()
            k[slot] = k[slot].copy()
            k[slot][idx] = np.nan
            return k, "error"

        return _c

    CASES["nan_%s_%s" % (_slot, "".join(map(str, _idx)))] = _mk()

# ---------- Inf in P/A/G/c/b/h (invalid -> must error) ----------
for _slot, _idx, _val in [
    ("P", (0, 0), np.inf),
    ("P", (1, 2), -np.inf),
    ("c", (2,), np.inf),
    ("c", (0,), -np.inf),
    ("A", (0, 1), np.inf),
    ("b", (0,), np.inf),
    ("G", (1, 3), -np.inf),
    ("h", (0,), np.inf),
]:
    def _mk(slot=_slot, idx=_idx, val=_val):
        def _c():
            k = base()
            k[slot] = k[slot].copy()
            k[slot][idx] = val
            return k, "error"

        return _c

    _sgn = "p" if _val > 0 else "m"
    CASES["inf%s_%s_%s" % (_sgn, _slot, "".join(map(str, _idx)))] = _mk()


# ---------- inverted / degenerate bounds ----------
@case("bounds_lb_gt_ub")
def _c():
    k = base()
    k["lb"] = np.array([1.0, 1.0, 1.0, 1.0])
    k["ub"] = np.array([0.0, 0.0, 0.0, 0.0])
    return k, "error_or_infeasible"


@case("bounds_lb_plusinf")
def _c():
    k = base()
    k["lb"] = np.full(4, np.inf)
    return k, "error_or_infeasible"


@case("bounds_ub_minusinf")
def _c():
    k = base()
    k["ub"] = np.full(4, -np.inf)
    return k, "error_or_infeasible"


# ---------- dtype / layout / view oddities (must match baseline) ----------
@case("dtype_float32")
def _c():
    k = base()
    for s in ("P", "c", "A", "b", "G", "h", "lb", "ub"):
        k[s] = k[s].astype(np.float32)
    return k, "ok_loose"


@case("dtype_int")
def _c():
    # integer-valued problem so int dtype is lossless
    k = base()
    k["P"] = k["P"].astype(np.int64)
    k["c"] = k["c"].astype(np.int64)
    k["A"] = k["A"].astype(np.int64)
    k["b"] = k["b"].astype(np.int64)
    k["G"] = k["G"].astype(np.int64)
    k["h"] = np.array([0, 0])  # integer h
    k["lb"] = k["lb"].astype(np.int64)
    k["ub"] = k["ub"].astype(np.int64)
    return k, "ok_int"


@case("layout_fortran")
def _c():
    k = base()
    k["P"] = np.asfortranarray(k["P"])
    k["A"] = np.asfortranarray(k["A"])
    k["G"] = np.asfortranarray(k["G"])
    return k, "ok"


@case("layout_transposed_view")
def _c():
    k = base()
    # G stored transposed; pass the .T view (same values, F-strided)
    k["G"] = np.ascontiguousarray(k["G"].T).T
    k["A"] = np.ascontiguousarray(k["A"].T).T
    k["P"] = np.ascontiguousarray(k["P"].T).T
    return k, "ok"


@case("layout_noncontig_slice")
def _c():
    k = base()
    big = np.zeros((4, 8))
    big[:, ::2] = k["P"]
    k["P"] = big[:, ::2]
    bigc = np.zeros(8)
    bigc[::2] = k["c"]
    k["c"] = bigc[::2]
    bigG = np.zeros((2, 8))
    bigG[:, ::2] = k["G"]
    k["G"] = bigG[:, ::2]
    return k, "ok"


@case("readonly_arrays")
def _c():
    k = base()
    for s in ("P", "c", "A", "b", "G", "h", "lb", "ub"):
        v = k[s].copy()
        v.setflags(write=False)
        k[s] = v
    return k, "ok"


@case("negative_stride")
def _c():
    k = base()
    k["c"] = k["c"][::-1][::-1]  # double reverse -> same values, stride +1 view
    kk = base()
    kk_c = kk["c"][::-1]
    k["c"] = kk_c[::-1]
    return k, "ok"


# ---------- shape mismatches ----------
@case("shape_c_too_short")
def _c():
    k = base()
    k["c"] = k["c"][:3]
    return k, "error"


@case("shape_P_nonsquare")
def _c():
    k = base()
    k["P"] = k["P"][:3, :]
    return k, "error"


@case("shape_b_too_long")
def _c():
    k = base()
    k["b"] = np.array([2.0, 3.0])
    return k, "error"


@case("shape_A_wrong_cols")
def _c():
    k = base()
    k["A"] = np.ones((1, 6))
    return k, "error"


@case("shape_h_mismatch")
def _c():
    k = base()
    k["h"] = np.array([0.5])
    return k, "error"


@case("shape_lb_too_short")
def _c():
    k = base()
    k["lb"] = k["lb"][:2]
    return k, "error"


@case("shape_P_3d")
def _c():
    k = base()
    k["P"] = np.zeros((2, 2, 2))
    return k, "error"


# --------------------------------------------------------------------------
# child: run one case
# --------------------------------------------------------------------------


def run_child(name):
    from pounce import solve_qp

    kwargs, expect = CASES[name]()
    out = {"name": name, "expect": expect}
    t0 = time.perf_counter()
    try:
        r = solve_qp(**kwargs)
        out["t"] = time.perf_counter() - t0
        x = np.asarray(r.x, dtype=float) if r.x is not None else None
        out["status"] = str(r.status)
        out["obj"] = None if r.obj is None else float(r.obj)
        out["x"] = None if x is None else x.tolist()
        out["x_nonfinite"] = bool(x is not None and not np.all(np.isfinite(x)))
        out["obj_nonfinite"] = bool(out["obj"] is not None and not np.isfinite(out["obj"]))
        out["raised"] = None
    except Exception as e:  # noqa: BLE001
        out["t"] = time.perf_counter() - t0
        out["raised"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        out["panic"] = type(e).__name__ == "PanicException"
    except BaseException as e:  # pyo3 PanicException is BaseException in some builds
        out["t"] = time.perf_counter() - t0
        out["raised"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        out["panic"] = True
    print("__RESULT__" + json.dumps(out))


# --------------------------------------------------------------------------
# also probe x0 handling through pounce.minimize
# --------------------------------------------------------------------------

X0_CASES = {
    "x0_nan": np.array([1.0, np.nan]),
    "x0_posinf": np.array([np.inf, 1.0]),
    "x0_neginf": np.array([1.0, -np.inf]),
    "x0_wild": np.array([1e18, -1e18]),
    "x0_wrong_len": np.array([1.0, 2.0, 3.0]),
    "x0_int": np.array([1, 2]),
    "x0_ok": np.array([1.0, 2.0]),
}


def run_x0_child(name):
    from pounce import minimize

    x0 = X0_CASES[name]

    def f(x):
        return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)

    def g(x):
        return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])

    out = {"name": name, "expect": "n/a"}
    t0 = time.perf_counter()
    try:
        r = minimize(f, x0, jac=g)
        out["t"] = time.perf_counter() - t0
        x = np.asarray(r.x, dtype=float) if getattr(r, "x", None) is not None else None
        out["status"] = str(getattr(r, "status", getattr(r, "message", "?")))
        out["success"] = bool(getattr(r, "success", False))
        out["obj"] = None if getattr(r, "fun", None) is None else float(r.fun)
        out["x"] = None if x is None else x.tolist()
        out["x_nonfinite"] = bool(x is not None and not np.all(np.isfinite(x)))
        out["obj_nonfinite"] = bool(out["obj"] is not None and not np.isfinite(out["obj"]))
        out["raised"] = None
    except BaseException as e:  # noqa: BLE001
        out["t"] = time.perf_counter() - t0
        out["raised"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        out["panic"] = type(e).__name__ == "PanicException"
    print("__RESULT__" + json.dumps(out))


# --------------------------------------------------------------------------
# parent driver
# --------------------------------------------------------------------------


def drive(kind, names):
    results = {}
    for n in names:
        p = subprocess.run(
            [sys.executable, HERE, kind, n],
            capture_output=True,
            text=True,
            timeout=30,
        )
        line = [l for l in p.stdout.splitlines() if l.startswith("__RESULT__")]
        if not line:
            results[n] = {
                "name": n,
                "crashed": True,
                "returncode": p.returncode,
                "stderr": p.stderr[-400:],
            }
        else:
            results[n] = json.loads(line[0][len("__RESULT__") :])
            results[n]["crashed"] = False
            results[n]["stderr_tail"] = p.stderr[-200:] if p.returncode else ""
    return results


def cvxpy_reference(inf_bounds):
    import cvxpy as cp

    k = base()
    x = cp.Variable(4)
    cons = [k["A"] @ x == k["b"], k["G"] @ x <= k["h"]]
    if not inf_bounds:
        cons += [x >= k["lb"], x <= k["ub"]]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(k["P"])) + k["c"] @ x), cons)
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), np.asarray(x.value)


def cvxpy_reference_mixed():
    import cvxpy as cp

    k = base()
    x = cp.Variable(4)
    cons = [
        k["A"] @ x == k["b"],
        k["G"] @ x <= k["h"],
        x[1] >= -1.0,
        x[3] >= 0.0,
        x[2] <= 2.0,
    ]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(k["P"])) + k["c"] @ x), cons)
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), np.asarray(x.value)


def main():
    qp_names = list(CASES.keys())
    x0_names = list(X0_CASES.keys())
    qp = drive("qp", qp_names)
    x0 = drive("x0", x0_names)

    findings = []
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    # ---- oracle references ----
    obj_cvx_box, x_cvx_box = cvxpy_reference(inf_bounds=False)
    obj_cvx_free, x_cvx_free = cvxpy_reference(inf_bounds=True)
    obj_cvx_mixed, x_cvx_mixed = cvxpy_reference_mixed()

    emit("=== cvxpy (CLARABEL) references ===")
    emit("  boxed  [-5,5]     obj=%.12e" % obj_cvx_box)
    emit("  free   (no bnds)  obj=%.12e" % obj_cvx_free)
    emit("  mixed  inf/finite obj=%.12e" % obj_cvx_mixed)
    emit("")

    def obj_of(n):
        r = qp[n]
        return None if r.get("crashed") or r.get("obj") is None else r["obj"]

    def relerr(a, b):
        if a is None or b is None:
            return float("nan")
        return abs(a - b) / max(1.0, abs(b))

    # ---- 1. legitimate infinite bounds ----
    emit("=== 1. LEGITIMATE +/-Inf BOUNDS (must be accepted & correct) ===")
    for n, ref, reflabel in [
        ("bounds_inf_both", obj_cvx_free, "cvxpy-free"),
        ("bounds_huge_both_REF", obj_cvx_free, "cvxpy-free"),
        ("bounds_none_REF", obj_cvx_free, "cvxpy-free"),
        ("bounds_inf_mixed", obj_cvx_mixed, "cvxpy-mixed"),
        ("bounds_huge_mixed_REF", obj_cvx_mixed, "cvxpy-mixed"),
        ("baseline", obj_cvx_box, "cvxpy-box"),
    ]:
        r = qp[n]
        o = obj_of(n)
        e = relerr(o, ref)
        bad = r.get("crashed") or r.get("raised") or r.get("x_nonfinite") or not (e < 1e-6)
        emit(
            "  %-24s status=%-12s obj=%-22s relerr(%s)=%.2e  %s"
            % (
                n,
                r.get("status", "CRASH" if r.get("crashed") else "EXC"),
                ("%.12e" % o) if o is not None else str(r.get("raised"))[:40],
                reflabel,
                e,
                "FAIL" if bad else "ok",
            )
        )
        if bad:
            findings.append(("INF_BOUNDS", n, r))

    # inf-bounds vs huge-bounds agreement (pounce-internal consistency)
    for a, b in [("bounds_inf_both", "bounds_huge_both_REF"), ("bounds_inf_mixed", "bounds_huge_mixed_REF")]:
        oa, ob = obj_of(a), obj_of(b)
        if oa is not None and ob is not None:
            emit("  consistency %s vs %s: relerr=%.2e" % (a, b, relerr(oa, ob)))

    # ---- 2. NaN must error ----
    emit("")
    emit("=== 2. NaN INPUT (must raise; a returned solution is a BUG) ===")
    for n in sorted(x for x in qp if x.startswith("nan_")):
        r = qp[n]
        if r.get("crashed"):
            emit("  %-20s CRASH rc=%s" % (n, r["returncode"]))
            findings.append(("NAN_CRASH", n, r))
        elif r.get("raised"):
            tag = "PANIC" if r.get("panic") else "ok"
            emit("  %-20s raised %s  [%s]" % (n, r["raised"][:90], tag))
            if r.get("panic"):
                findings.append(("NAN_PANIC", n, r))
        else:
            emit(
                "  %-20s *** RETURNED status=%s obj=%s x_nonfinite=%s ***"
                % (n, r["status"], r["obj"], r["x_nonfinite"])
            )
            findings.append(("NAN_SILENT", n, r))

    # ---- 3. Inf in data must error ----
    emit("")
    emit("=== 3. +/-Inf in P/c/A/b/G/h (invalid; must raise) ===")
    for n in sorted(x for x in qp if x.startswith("infp_") or x.startswith("infm_")):
        r = qp[n]
        if r.get("crashed"):
            emit("  %-20s CRASH rc=%s" % (n, r["returncode"]))
            findings.append(("INF_CRASH", n, r))
        elif r.get("raised"):
            tag = "PANIC" if r.get("panic") else "ok"
            emit("  %-20s raised %s  [%s]" % (n, r["raised"][:90], tag))
            if r.get("panic"):
                findings.append(("INF_PANIC", n, r))
        else:
            emit(
                "  %-20s *** RETURNED status=%s obj=%s x_nonfinite=%s ***"
                % (n, r["status"], r["obj"], r["x_nonfinite"])
            )
            findings.append(("INF_SILENT", n, r))

    # ---- 4. degenerate bounds ----
    emit("")
    emit("=== 4. inverted / degenerate bounds (error OR infeasible, never 'optimal') ===")
    for n in ("bounds_lb_gt_ub", "bounds_lb_plusinf", "bounds_ub_minusinf"):
        r = qp[n]
        if r.get("crashed"):
            emit("  %-20s CRASH rc=%s" % (n, r["returncode"]))
            findings.append(("DEGEN_CRASH", n, r))
        elif r.get("raised"):
            emit("  %-20s raised %s" % (n, r["raised"][:90]))
        else:
            bad = "optimal" in r["status"].lower() and not r["x_nonfinite"]
            emit(
                "  %-20s status=%s obj=%s x_nonfinite=%s %s"
                % (n, r["status"], r["obj"], r["x_nonfinite"], "*** BUG" if bad else "ok")
            )
            if bad or r["x_nonfinite"]:
                findings.append(("DEGEN", n, r))

    # ---- 5. dtype / layout ----
    emit("")
    emit("=== 5. dtype / memory-layout views (must match baseline answer) ===")
    base_obj = obj_of("baseline")
    base_x = np.asarray(qp["baseline"]["x"]) if qp["baseline"].get("x") else None
    for n in (
        "dtype_float32",
        "dtype_int",
        "layout_fortran",
        "layout_transposed_view",
        "layout_noncontig_slice",
        "readonly_arrays",
        "negative_stride",
    ):
        r = qp[n]
        if r.get("crashed"):
            emit("  %-24s CRASH rc=%s %s" % (n, r["returncode"], r["stderr"][-120:]))
            findings.append(("LAYOUT_CRASH", n, r))
            continue
        if r.get("raised"):
            # a clean rejection of an unsupported dtype is acceptable
            emit("  %-24s raised %s  [rejected -> acceptable]" % (n, r["raised"][:80]))
            continue
        o = r["obj"]
        if n == "dtype_int":
            emit("  %-24s status=%s obj=%s (different problem: h=[0,0]) x=%s" % (n, r["status"], o, r["x"]))
            continue
        e = relerr(o, base_obj)
        tol = 1e-4 if n == "dtype_float32" else 1e-9
        xe = (
            float(np.max(np.abs(np.asarray(r["x"]) - base_x)))
            if (r.get("x") and base_x is not None)
            else float("nan")
        )
        bad = not (e < tol) or r["x_nonfinite"]
        emit(
            "  %-24s status=%-10s obj=%.12e relerr=%.2e xinf=%.2e %s"
            % (n, r["status"], o, e, xe, "*** MISMATCH" if bad else "ok")
        )
        if bad:
            findings.append(("LAYOUT_MISMATCH", n, r))

    # ---- 6. shapes ----
    emit("")
    emit("=== 6. mismatched shapes (must raise) ===")
    for n in sorted(x for x in qp if x.startswith("shape_")):
        r = qp[n]
        if r.get("crashed"):
            emit("  %-24s CRASH rc=%s" % (n, r["returncode"]))
            findings.append(("SHAPE_CRASH", n, r))
        elif r.get("raised"):
            tag = "PANIC" if r.get("panic") else "ok"
            emit("  %-24s raised %s [%s]" % (n, r["raised"][:90], tag))
            if r.get("panic"):
                findings.append(("SHAPE_PANIC", n, r))
        else:
            emit("  %-24s *** RETURNED status=%s obj=%s ***" % (n, r["status"], r["obj"]))
            findings.append(("SHAPE_SILENT", n, r))

    # ---- 7. x0 through minimize ----
    emit("")
    emit("=== 7. x0 handling via pounce.minimize (min (x0-1)^2+(x1-2)^2, opt=0) ===")
    for n in x0_names:
        r = x0[n]
        if r.get("crashed"):
            emit("  %-16s CRASH rc=%s %s" % (n, r["returncode"], r["stderr"][-150:]))
            findings.append(("X0_CRASH", n, r))
            continue
        if r.get("raised"):
            tag = "PANIC" if r.get("panic") else "ok"
            emit("  %-16s raised %s [%s]" % (n, r["raised"][:90], tag))
            if r.get("panic"):
                findings.append(("X0_PANIC", n, r))
            continue
        bad = False
        if n in ("x0_nan", "x0_posinf", "x0_neginf"):
            # must NOT claim success with a garbage/NaN answer
            bad = r["success"] and (r["x_nonfinite"] or r["obj_nonfinite"])
            if r["success"] and not r["x_nonfinite"]:
                bad = abs(r["obj"]) > 1e-6  # claims success but wrong answer
        elif n in ("x0_wild", "x0_ok", "x0_int"):
            bad = (not r["success"]) or r["x_nonfinite"] or abs(r["obj"]) > 1e-6
        emit(
            "  %-16s success=%-5s status=%-28s obj=%s x=%s %s"
            % (
                n,
                r["success"],
                r["status"][:28],
                ("%.6e" % r["obj"]) if r["obj"] is not None else None,
                np.round(np.asarray(r["x"]), 6).tolist() if r.get("x") else None,
                "*** FINDING" if bad else "ok",
            )
        )
        if bad:
            findings.append(("X0", n, r))

    # ---- summary ----
    emit("")
    emit("=== SUMMARY ===")
    emit("cases run: %d qp + %d x0" % (len(qp), len(x0)))
    if not findings:
        emit("no findings")
        emit("VERDICT: PASS")
    else:
        for kind, n, r in findings:
            emit("  FINDING %-16s %-24s %s" % (kind, n, json.dumps({k: r.get(k) for k in ("status", "obj", "x_nonfinite", "raised", "returncode")})[:220]))
        hard = [f for f in findings if f[0].endswith(("_SILENT", "_CRASH")) or f[0] in ("INF_BOUNDS", "LAYOUT_MISMATCH", "DEGEN")]
        emit("VERDICT: SOLVER_BUG" if hard else "VERDICT: FINDINGS_SOFT")

    with open(HERE.replace(".py", "_output.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "qp":
        run_child(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "x0":
        run_x0_child(sys.argv[2])
    else:
        main()
