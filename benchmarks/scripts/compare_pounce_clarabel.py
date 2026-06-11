#!/usr/bin/env python3
"""Compare POUNCE's convex LP/QP IPM against Clarabel on the LP (netlib +
Maros-Meszaros) and QP (Maros-Meszaros) benchmark suites.

POUNCE numbers are read from the canonical reports produced by the .nl runs
(``benchmarks/lp/pounce.json``, ``benchmarks/qp/pounce.json``). Clarabel is run
fresh here, in-process, on the *same* source problems and joined by name.

Clarabel has no model-file reader, so each instance is converted to matrices:

  QP (.mat)   : min 1/2 x'Px + q'x  s.t.  l <= Ax <= u          (+ const r)
  LP (e[mps]) : min c'x             s.t.  rl <= Ax <= ru, cl <= x <= cu

Two-sided rows / finite variable bounds become a ZeroCone (equalities) plus a
NonnegativeCone (one-sided inequalities), in that order.

LP sources are emps-compressed (Maros-Meszaros additionally gzipped); we build
the repo's ``benchmarks/lp/mps/emps.c`` decompressor and pipe through HiGHS.

Usage:
  python3 benchmarks/scripts/compare_pounce_clarabel.py [--class lp|qp|both]
                                                        [--limit N]
                                                        [--time-limit SECS]
                                                        [--from-json]
                                                        [--check]
Out:
  benchmarks/clarabel_compare_{lp,qp}.json   per-problem records
  benchmarks/clarabel_compare.md             side-by-side markdown report

--from-json   skip the live run; load the per-problem records from the existing
              benchmarks/clarabel_compare_{lp,qp}.json (regression gate / CI).
--check       exit nonzero if any *genuine* objective disagreement remains. A
              disagreement counts only when BOTH solvers report a hard solve
              (pounce SolveSucceeded AND clarabel Solved -- AlmostSolved and
              SolvedToAcceptableLevel are excluded as not-certified) yet their
              objectives differ by more than the numpy-isclose band
              |a-b| > atol + rtol*max(|a|,|b|) (rtol=atol=1e-3). This flags real
              wrong-answer bugs while tolerating convergence-point slack.
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import clarabel

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)

INF = 1e20
EMPS_SRC = os.path.join(BENCH, "lp", "mps", "emps.c")
EMPS_BIN = os.path.join(tempfile.gettempdir(), "pounce_emps")
POUNCE_BIN = os.path.join(ROOT, "target", "release", "pounce")
MPS_TO_NL = os.path.join(BENCH, "lp", "mps_to_nl.py")

# POUNCE statuses that count as a successful optimal solve. POUNCE is run LIVE
# (the committed pounce.json reports were found to be partially stale), so we
# read its --json-output: solution.status + statistics.{final_objective,
# iteration_count, total_wallclock_time_secs}.
POUNCE_OK = {"SolveSucceeded", "SolvedToAcceptableLevel"}
CLARABEL_OK = {"Solved", "AlmostSolved"}

# Lazily imported single-file .mat -> Pyomo model converter from generate_nl.py.
_qp_gen = None


def qp_gen():
    global _qp_gen
    if _qp_gen is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qp_generate_nl", os.path.join(BENCH, "qp", "generate_nl.py"))
        _qp_gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_qp_gen)
    return _qp_gen


# ----------------------------------------------------------------------------
# Matrix assembly: l <= Ax <= u (+ box) -> Clarabel (Zero then Nonneg cones).
# ----------------------------------------------------------------------------
def build_cones(A, lo, hi, P, q, eq_tol=1e-9):
    """Return (P, q, G, b, cones) for min 1/2 x'Px+q'x s.t. lo<=Ax<=hi.

    Variable bounds, if any, should already be folded into A/lo/hi by the
    caller (LP folds an identity block; QP has them inside A)."""
    A = A.tocsr()
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)

    fin_lo = lo > -INF
    fin_hi = hi < INF
    eq = fin_lo & fin_hi & (np.abs(hi - lo) <= eq_tol)
    only_hi = fin_hi & ~eq
    only_lo = fin_lo & ~eq

    blocks, rhs = [], []
    # ZeroCone block: equalities  Ax = lo.
    n_zero = int(eq.sum())
    if n_zero:
        blocks.append(A[eq])
        rhs.append(lo[eq])
    # NonnegativeCone block:  Ax <= hi  and  -Ax <= -lo.
    n_nn = 0
    if only_hi.any():
        blocks.append(A[only_hi])
        rhs.append(hi[only_hi])
        n_nn += int(only_hi.sum())
    if only_lo.any():
        blocks.append(-A[only_lo])
        rhs.append(-lo[only_lo])
        n_nn += int(only_lo.sum())

    if blocks:
        G = sp.vstack(blocks).tocsc()
        b = np.concatenate(rhs)
    else:
        G = sp.csc_matrix((0, A.shape[1]))
        b = np.zeros(0)

    cones = []
    if n_zero:
        cones.append(clarabel.ZeroConeT(n_zero))
    if n_nn:
        cones.append(clarabel.NonnegativeConeT(n_nn))
    return P, q, G, b, cones


def load_qp(path):
    """Maros-Meszaros .mat -> (P,q,G,b,cones, n,m, const_offset)."""
    m = sio.loadmat(path)
    P = sp.csc_matrix(m["P"]).astype(float)
    q = np.asarray(m["q"], float).ravel()
    A = sp.csc_matrix(m["A"]).astype(float)
    lo = np.asarray(m["l"], float).ravel()
    hi = np.asarray(m["u"], float).ravel()
    r = float(np.asarray(m.get("r", 0.0)).ravel()[0]) if "r" in m else 0.0
    n = P.shape[0]
    mcon = A.shape[0]
    P, q, G, b, cones = build_cones(A, lo, hi, P, q)
    return P, q, G, b, cones, n, mcon, r


def ensure_emps():
    if os.path.exists(EMPS_BIN):
        return
    r = subprocess.run(["cc", "-std=gnu89", "-O2", "-w", "-o", EMPS_BIN, EMPS_SRC],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(EMPS_BIN):
        raise RuntimeError(f"failed to build emps: {r.stderr[:300]}")


def load_lp(path):
    """netlib/Maros emps (maybe .gz) -> (P,q,G,b,cones, n,m, const_offset).

    P is the zero matrix (pure LP). Variable bounds are folded into A."""
    import highspy

    ensure_emps()
    # Decompress emps -> plain MPS.
    raw = gzip.open(path, "rb").read() if path.endswith(".gz") else open(path, "rb").read()
    dec = subprocess.run([EMPS_BIN], input=raw, capture_output=True)
    if dec.returncode != 0 or not dec.stdout:
        raise RuntimeError("emps decompress produced no output")
    with tempfile.NamedTemporaryFile("wb", suffix=".mps", delete=False) as tf:
        tf.write(dec.stdout)
        mps = tf.name
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(mps)
        lp = h.getLp()
        n, mcon = lp.num_col_, lp.num_row_
        c = np.array(lp.col_cost_, float)
        cl = np.array(lp.col_lower_, float)
        cu = np.array(lp.col_upper_, float)
        rl = np.array(lp.row_lower_, float)
        ru = np.array(lp.row_upper_, float)
        offset = float(getattr(lp, "offset_", 0.0))
        A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_,
                           lp.a_matrix_.start_), shape=(mcon, n))
        sense = getattr(lp, "sense_", None)
        # HiGHS: kMaximize flips; pounce/clarabel minimize. Normalize to min.
        if sense is not None and int(sense) == int(getattr(highspy.ObjSense, "kMaximize", 1)):
            c = -c
            offset = -offset
    finally:
        os.unlink(mps)

    # Fold variable bounds into the constraint block as an identity.
    I = sp.eye(n, format="csr")
    Afull = sp.vstack([A, I]).tocsr()
    lofull = np.concatenate([rl, cl])
    hifull = np.concatenate([ru, cu])
    P = sp.csc_matrix((n, n))
    P, q, G, b, cones = build_cones(Afull, lofull, hifull, P, c)
    return P, q, G, b, cones, n, mcon, offset


# ----------------------------------------------------------------------------
def solve_clarabel(P, q, G, b, cones, offset, time_limit):
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.time_limit = float(time_limit)
    t = time.perf_counter()
    try:
        sol = clarabel.DefaultSolver(P, q, G, b, cones, s).solve()
        wall = time.perf_counter() - t
        st = str(sol.status)
        obj = sol.obj_val + offset if st in CLARABEL_OK else None
        return {"status": st, "objective": obj,
                "iterations": int(sol.iterations),
                "solve_time": float(sol.solve_time), "wall": wall}
    except Exception as e:
        return {"status": f"Error:{type(e).__name__}", "objective": None,
                "iterations": None, "solve_time": None,
                "wall": time.perf_counter() - t}


def reldiff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1e-10)


# Strict objective-agreement gate for --check. Statuses that count as a
# *certified* solve for each solver (AlmostSolved / SolvedToAcceptableLevel are
# deliberately excluded: an uncertified point may legitimately differ).
POUNCE_STRICT = {"SolveSucceeded"}
CLARABEL_STRICT = {"Solved"}
CHECK_RTOL = 1e-3
CHECK_ATOL = 1e-3


def isclose(a, b, rtol=CHECK_RTOL, atol=CHECK_ATOL):
    """numpy-isclose style absolute+relative tolerance."""
    if a is None or b is None:
        return False
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def check_disagreements(rows):
    """Return the rows where both solvers certify a solve yet objectives differ
    beyond the isclose band -- the genuine wrong-answer set the gate fails on."""
    bad = []
    for r in rows:
        if (r["pounce"]["status"] in POUNCE_STRICT
                and r["clarabel"]["status"] in CLARABEL_STRICT
                and not isclose(r["pounce"]["objective"], r["clarabel"]["objective"])):
            bad.append(r)
    return bad


# ----------------------------------------------------------------------------
# POUNCE, run live on a freshly generated .nl (same problem Clarabel solves).
# ----------------------------------------------------------------------------
def gen_nl_lp(src_path, out_nl):
    """emps[.gz] source -> plain MPS -> .nl via the repo's mps_to_nl.py."""
    ensure_emps()
    raw = gzip.open(src_path, "rb").read() if src_path.endswith(".gz") else open(src_path, "rb").read()
    dec = subprocess.run([EMPS_BIN], input=raw, capture_output=True)
    if dec.returncode != 0 or not dec.stdout:
        raise RuntimeError("emps decompress produced no output")
    with tempfile.NamedTemporaryFile("wb", suffix=".mps", delete=False) as tf:
        tf.write(dec.stdout)
        mps = tf.name
    try:
        r = subprocess.run([sys.executable, MPS_TO_NL, mps, out_nl],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(out_nl):
            raise RuntimeError(f"mps_to_nl failed: {r.stderr[:200]}")
    finally:
        os.unlink(mps)


def gen_nl_qp(mat_path, out_nl):
    """Maros-Meszaros .mat -> .nl via generate_nl.build_model (the repo path)."""
    g = qp_gen()
    name = os.path.basename(mat_path)[:-4]
    P, q, r, C, lc, uc, lb, ub = g.load_qp(mat_path)
    model = g.build_model(name, P, q, r, C, lc, uc, lb, ub)
    model.write(out_nl, format="nl",
                io_options={"symbolic_solver_labels": False})


def run_pounce(nl_path, selection, time_limit):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    t = time.perf_counter()
    try:
        subprocess.run([POUNCE_BIN, nl_path, f"solver_selection={selection}",
                        "--json-output", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=time_limit)
    except subprocess.TimeoutExpired:
        return {"status": "TimeOut", "objective": None,
                "iterations": None, "solve_time": time_limit}
    wall = time.perf_counter() - t
    try:
        d = json.load(open(out))
        sol, stat = d.get("solution", {}), d.get("statistics", {})
        return {"status": sol.get("status"),
                "objective": stat.get("final_objective", sol.get("objective")),
                "iterations": stat.get("iteration_count"),
                "solve_time": stat.get("total_wallclock_time_secs", wall)}
    except Exception as e:
        return {"status": f"ParseError:{type(e).__name__}", "objective": None,
                "iterations": None, "solve_time": wall}
    finally:
        os.path.exists(out) and os.unlink(out)


# ----------------------------------------------------------------------------
def run_class(kind, limit, time_limit):
    """kind in {'lp','qp'}. Runs BOTH solvers live on each source problem and
    returns joined per-problem records."""
    if kind == "qp":
        srcs = sorted(glob.glob(os.path.join(BENCH, "qp", "data", "*.mat")),
                      key=os.path.getsize)
        name_of = lambda p: os.path.basename(p)[:-4]
        loader, gen_nl, selection = load_qp, gen_nl_qp, "qp-ipm"
    else:
        srcs = (sorted(glob.glob(os.path.join(BENCH, "lp", "data", "netlib", "*")))
                + sorted(glob.glob(os.path.join(BENCH, "lp", "data", "meszaros", "*"))))
        name_of = lambda p: os.path.basename(p).split(".")[0]
        loader, gen_nl, selection = load_lp, gen_nl_lp, "lp-ipm"
    if limit:
        srcs = srcs[:limit]

    rows = []
    print(f"\n=== {kind.upper()}  ({len(srcs)} problems, pounce={selection}) ===")
    print(f"{'problem':<16}{'p.status':>14}{'c.status':>14}"
          f"{'reldiff':>11}{'p.it':>6}{'c.it':>6}{'p.s':>9}{'c.s':>9}")
    for p in srcs:
        name = name_of(p)
        # POUNCE (live): generate .nl, solve.
        try:
            with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as tf:
                nl = tf.name
            gen_nl(p, nl)
            pr = run_pounce(nl, selection, time_limit)
            os.path.exists(nl) and os.unlink(nl)
        except Exception as e:
            pr = {"status": f"GenError:{type(e).__name__}", "objective": None,
                  "iterations": None, "solve_time": None}
        # Clarabel: load matrices, solve.
        try:
            P, q, G, b, cones, n, m, off = loader(p)
            cl = solve_clarabel(P, q, G, b, cones, off, time_limit)
        except Exception as e:
            cl = {"status": f"LoadError:{type(e).__name__}", "objective": None,
                  "iterations": None, "solve_time": None, "wall": None}
            n = m = None
        rd = reldiff(pr.get("objective"), cl["objective"])
        rows.append({"name": name, "n": n, "m": m,
                     "pounce": pr, "clarabel": cl, "reldiff": rd})
        fr = f"{rd:.1e}" if rd is not None else "n/a"
        ps, cs = pr.get("solve_time"), cl.get("solve_time")
        print(f"{name:<16}{str(pr.get('status'))[:13]:>14}{cl['status'][:13]:>14}"
              f"{fr:>11}{str(pr.get('iterations')):>6}{str(cl['iterations']):>6}"
              f"{(ps if ps is not None else float('nan')):>9.3f}"
              f"{(cs if cs is not None else float('nan')):>9.3f}")
    return rows


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def summarize(kind, rows):
    both = [r for r in rows
            if r["pounce"]["status"] in POUNCE_OK and r["clarabel"]["status"] in CLARABEL_OK]
    agree = [r for r in both if r["reldiff"] is not None and r["reldiff"] < 1e-4]
    p_only = [r for r in rows
              if r["pounce"]["status"] in POUNCE_OK and r["clarabel"]["status"] not in CLARABEL_OK]
    c_only = [r for r in rows
              if r["pounce"]["status"] not in POUNCE_OK and r["clarabel"]["status"] in CLARABEL_OK]
    speed = [r["pounce"]["solve_time"] / r["clarabel"]["solve_time"]
             for r in both
             if r["pounce"]["solve_time"] and r["clarabel"]["solve_time"]]
    gm = geomean(speed)
    out = [
        f"### {kind.upper()} — {len(rows)} problems",
        "",
        f"- Solved by **both**: {len(both)}",
        f"- Objective agreement (reldiff < 1e-4): **{len(agree)}/{len(both)}**",
        f"- POUNCE solved, Clarabel did not: {len(p_only)}",
        f"- Clarabel solved, POUNCE did not: {len(c_only)}",
    ]
    if gm:
        faster = "Clarabel faster" if gm > 1 else "POUNCE faster"
        out.append(f"- Geomean solve-time ratio pounce/clarabel: **{gm:.2f}×** "
                   f"({faster} on average, over {len(speed)} both-solved)")
    if p_only:
        out.append(f"- Clarabel non-solves: " +
                   ", ".join(f"{r['name']}({r['clarabel']['status']})" for r in p_only[:12]) +
                   (" …" if len(p_only) > 12 else ""))
    if c_only:
        out.append(f"- POUNCE non-solves: " +
                   ", ".join(f"{r['name']}({r['pounce']['status']})" for r in c_only[:12]) +
                   (" …" if len(c_only) > 12 else ""))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", choices=["lp", "qp", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per class (debug)")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--from-json", action="store_true",
                    help="load existing clarabel_compare_{kind}.json instead of "
                         "running both solvers live")
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero on any genuine objective disagreement "
                         "(strict-solved gate, isclose rtol=atol=1e-3)")
    args = ap.parse_args()

    kinds = ["lp", "qp"] if args.cls == "both" else [args.cls]
    md = ["# POUNCE vs Clarabel — convex LP/QP benchmark comparison", "",
          f"Both solvers run live on this machine, per-solver time limit "
          f"{args.time_limit:g}s. POUNCE: convex LP/QP IPM (`solver_selection="
          "{lp,qp}-ipm`) on a freshly generated `.nl`. Clarabel "
          f"{clarabel.__version__} (Python) on matrices from the same source "
          "(its backend may use multiple threads, so wall-time comparisons "
          "favor it on larger problems). Both minimize; objectives joined by "
          "problem name.",
          ""]
    all_bad = []
    for kind in kinds:
        json_path = os.path.join(BENCH, f"clarabel_compare_{kind}.json")
        if args.from_json:
            with open(json_path) as fh:
                rows = json.load(fh)
            print(f"\n=== {kind.upper()}  (loaded {len(rows)} records from "
                  f"{os.path.relpath(json_path, ROOT)}) ===")
        else:
            rows = run_class(kind, args.limit, args.time_limit)
            with open(json_path, "w") as fh:
                json.dump(rows, fh, indent=2)
        md.append(summarize(kind, rows))
        print("\n" + summarize(kind, rows))

        if args.check:
            bad = check_disagreements(rows)
            if bad:
                print(f"--check {kind.upper()}: {len(bad)} genuine "
                      f"disagreement(s) (both certified-solved, "
                      f"|Δobj| > {CHECK_ATOL}+{CHECK_RTOL}·max):")
                for r in bad:
                    print(f"  {r['name']:<16} pounce={r['pounce']['objective']!r} "
                          f"clarabel={r['clarabel']['objective']!r} "
                          f"reldiff={r['reldiff']}")
            else:
                print(f"--check {kind.upper()}: PASS "
                      f"(no certified-solve objective disagreements)")
            all_bad.extend((kind, r) for r in bad)

    if not args.from_json:
        with open(os.path.join(BENCH, "clarabel_compare.md"), "w") as fh:
            fh.write("\n".join(md))
        print(f"\nwrote {os.path.join(BENCH, 'clarabel_compare.md')}")

    if args.check and all_bad:
        print(f"\nFAIL: {len(all_bad)} genuine objective disagreement(s) across "
              f"{', '.join(sorted(set(k.upper() for k, _ in all_bad)))}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
