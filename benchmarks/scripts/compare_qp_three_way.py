#!/usr/bin/env python3
"""Three-way head-to-head on the Maros-Meszaros QP benchmark:

    pounce-convex QP-IPM   vs   Clarabel   vs   pounce NLP (filter-IPM)

graded against the published ground-truth optima
(``pounce-bench-data/qp/Maros-Meszaros-answers.json``, DOC 97/6).

Each .mat instance is solved by all three on the *same* source problem:
  - pounce QP-IPM and pounce NLP run live on a freshly generated .nl
    (solver_selection={qp-ipm,nlp}),
  - Clarabel runs in-process on the matrices.
Every objective is compared to the ground-truth OPT; a solve is "correct" when
|obj-opt| <= atol + rtol*max(|obj|,|opt|).

Reuses the assembly/runner helpers in compare_pounce_clarabel.py.

Usage:
  python3 benchmarks/scripts/compare_qp_three_way.py [--limit N]
        [--time-limit SECS] [--rtol R] [--atol A]
Out:
  benchmarks/qp_three_way.json   per-problem records
  benchmarks/qp_three_way.md     side-by-side report
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)

# Reuse the existing comparison module (matrix assembly, Clarabel runner,
# .nl generation, pounce runner).
_spec = importlib.util.spec_from_file_location(
    "cmp_pc", os.path.join(HERE, "compare_pounce_clarabel.py"))
cmp_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_pc)

GROUND_TRUTH = os.path.expanduser(
    "~/Dropbox/projects/pounce-bench-data/qp/Maros-Meszaros-answers.json")

POUNCE_OK = cmp_pc.POUNCE_OK            # {"SolveSucceeded","SolvedToAcceptableLevel"}
CLARABEL_OK = cmp_pc.CLARABEL_OK        # {"Solved","AlmostSolved"}


def load_ground_truth():
    with open(GROUND_TRUTH) as fh:
        doc = json.load(fh)
    # key by lowercase name (matches .mat basename.lower())
    return {k.lower(): v["opt"] for k, v in doc["problems"].items()}


def rel_err(obj, opt):
    if obj is None or opt is None:
        return None
    return abs(obj - opt) / max(abs(opt), abs(obj), 1e-10)


def correct(obj, opt, rtol, atol):
    if obj is None or opt is None:
        return False
    return abs(obj - opt) <= atol + rtol * max(abs(obj), abs(opt))


def run(limit, time_limit, rtol, atol):
    gt = load_ground_truth()
    srcs = sorted(glob.glob(os.path.join(BENCH, "qp", "data", "*.mat")),
                  key=os.path.getsize)
    if limit:
        srcs = srcs[:limit]

    rows = []
    hdr = (f"{'problem':<14}{'opt':>14} | "
           f"{'qp-ipm':>11}{'re':>9} | {'clarabel':>11}{'re':>9} | "
           f"{'nlp':>11}{'re':>9} | t(qp/cl/nlp)")
    print(f"=== QP three-way ({len(srcs)} problems, time-limit {time_limit:g}s, "
          f"correct: |Δ|<= {atol:g}+{rtol:g}·max) ===")
    print(hdr)
    print("-" * len(hdr))

    for p in srcs:
        name = os.path.basename(p)[:-4]
        key = name.lower()
        opt = gt.get(key)

        # pounce QP-IPM and NLP both run on one generated .nl.
        try:
            with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as tf:
                nl = tf.name
            cmp_pc.gen_nl_qp(p, nl)
            qp = cmp_pc.run_pounce(nl, "qp-ipm", time_limit)
            nlp = cmp_pc.run_pounce(nl, "nlp", time_limit)
            os.path.exists(nl) and os.unlink(nl)
        except Exception as e:
            err = {"status": f"GenError:{type(e).__name__}", "objective": None,
                   "iterations": None, "solve_time": None}
            qp, nlp = dict(err), dict(err)

        # Clarabel on matrices.
        try:
            P, q, G, b, cones, n, m, off = cmp_pc.load_qp(p)
            cl = cmp_pc.solve_clarabel(P, q, G, b, cones, off, time_limit)
        except Exception as e:
            cl = {"status": f"LoadError:{type(e).__name__}", "objective": None,
                  "iterations": None, "solve_time": None, "wall": None}
            n = m = None

        def grade(rec, ok_set):
            o = rec.get("objective")
            solved = rec.get("status") in ok_set
            re_ = rel_err(o, opt)
            return {
                "status": rec.get("status"),
                "objective": o,
                "iterations": rec.get("iterations"),
                "solve_time": rec.get("solve_time"),
                "rel_err": re_,
                "solved": solved,
                "correct": solved and correct(o, opt, rtol, atol),
            }

        row = {
            "name": name, "n": n, "m": m, "opt": opt,
            "qp_ipm": grade(qp, POUNCE_OK),
            "clarabel": grade(cl, CLARABEL_OK),
            "nlp": grade(nlp, POUNCE_OK),
        }
        rows.append(row)

        def cell(g):
            o = g["objective"]
            os_ = f"{o:.4e}" if o is not None else g["status"][:11]
            re_ = g["rel_err"]
            rs = f"{re_:.1e}" if re_ is not None else "  n/a"
            mark = "✓" if g["correct"] else ("·" if g["solved"] else "✗")
            return f"{os_:>11}{rs:>8}{mark}"

        ts = lambda g: g["solve_time"] if g["solve_time"] is not None else float("nan")
        opts = f"{opt:.4e}" if opt is not None else "n/a"
        print(f"{name:<14}{opts:>14} | {cell(row['qp_ipm'])} | "
              f"{cell(row['clarabel'])} | {cell(row['nlp'])} | "
              f"{ts(row['qp_ipm']):.2f}/{ts(row['clarabel']):.2f}/{ts(row['nlp']):.2f}")

    return rows


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def summarize(rows, rtol, atol):
    N = len(rows)
    have_gt = [r for r in rows if r["opt"] is not None]
    out = ["# POUNCE-QP vs Clarabel vs POUNCE-NLP — Maros-Meszaros QP benchmark",
           "",
           f"{N} problems; {len(have_gt)} with ground-truth optima "
           "(DOC 97/6, BPMPD reference). A solve is **correct** when "
           f"`|obj-opt| <= {atol:g} + {rtol:g}·max(|obj|,|opt|)`.",
           ""]

    def block(key, label):
        solved = [r for r in rows if r[key]["solved"]]
        cor = [r for r in have_gt if r[key]["correct"]]
        # wrong = solved (by its own status) but objective wrong vs ground truth
        wrong = [r for r in have_gt if r[key]["solved"] and not r[key]["correct"]]
        res = [r[key]["rel_err"] for r in cor if r[key]["rel_err"] is not None]
        med = sorted(res)[len(res) // 2] if res else None
        L = [f"### {label}",
             f"- Solved (own status): **{len(solved)}/{N}**",
             f"- Correct vs ground truth: **{len(cor)}/{len(have_gt)}**",
             f"- Solved-but-wrong (status OK, obj off): **{len(wrong)}**",
             (f"- Median rel-err on correct solves: {med:.1e}" if med is not None else ""),
             ]
        if wrong:
            L.append("- Wrong objectives: " + ", ".join(
                f"{r['name']}(re={r[key]['rel_err']:.1e})" for r in wrong[:20])
                + (" …" if len(wrong) > 20 else ""))
        L.append("")
        return "\n".join(x for x in L if x)

    out.append(block("qp_ipm", "pounce QP-IPM (solver_selection=qp-ipm)"))
    out.append(block("clarabel", "Clarabel"))
    out.append(block("nlp", "pounce NLP (solver_selection=nlp)"))

    # Speed: geomean over the set where ALL THREE produced a correct solve.
    allc = [r for r in have_gt
            if r["qp_ipm"]["correct"] and r["clarabel"]["correct"] and r["nlp"]["correct"]
            and r["qp_ipm"]["solve_time"] and r["clarabel"]["solve_time"]
            and r["nlp"]["solve_time"]]
    if allc:
        gq = geomean([r["qp_ipm"]["solve_time"] for r in allc])
        gc = geomean([r["clarabel"]["solve_time"] for r in allc])
        gn = geomean([r["nlp"]["solve_time"] for r in allc])
        out += [f"### Speed (geomean over {len(allc)} all-three-correct problems)",
                f"- pounce QP-IPM : {gq:.3f}s",
                f"- Clarabel      : {gc:.3f}s",
                f"- pounce NLP    : {gn:.3f}s",
                f"- QP-IPM vs Clarabel: {gq/gc:.2f}×  "
                f"(Clarabel {'faster' if gq>gc else 'slower'})",
                f"- QP-IPM vs NLP     : {gn/gq:.2f}×  "
                f"(QP-IPM {'faster' if gn>gq else 'slower'})",
                ""]

    # Where ground truth discriminates: pounce-QP correct but another solver wrong.
    disc = [r for r in have_gt if r["qp_ipm"]["correct"]
            and (not r["clarabel"]["correct"] or not r["nlp"]["correct"])]
    if disc:
        out.append("### Problems where pounce-QP is correct but another solver is not")
        out.append("")
        out.append("| problem | opt | clarabel | nlp |")
        out.append("|---|---|---|---|")
        for r in disc:
            def st(g):
                if g["correct"]:
                    return "✓"
                if g["solved"]:
                    return f"off re={g['rel_err']:.1e}" if g["rel_err"] is not None else "off"
                return g["status"]
            out.append(f"| {r['name']} | {r['opt']:.6g} | "
                       f"{st(r['clarabel'])} | {st(r['nlp'])} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    rows = run(args.limit, args.time_limit, args.rtol, args.atol)
    jpath = os.path.join(BENCH, "qp_three_way.json")
    with open(jpath, "w") as fh:
        json.dump(rows, fh, indent=2)
    md = summarize(rows, args.rtol, args.atol)
    mpath = os.path.join(BENCH, "qp_three_way.md")
    with open(mpath, "w") as fh:
        fh.write(md + "\n")
    print("\n" + md)
    print(f"\nwrote {os.path.relpath(jpath, ROOT)} and {os.path.relpath(mpath, ROOT)}")


if __name__ == "__main__":
    main()
