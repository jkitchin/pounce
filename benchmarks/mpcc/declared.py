"""The Gate 0 corpus through POUNCE's declared-complementarity entry point.

    python -m mpcc.declared            # from benchmarks/; full configuration
    python -m mpcc.declared --json out.json

Every cell of the ``--full`` configuration (every case, both scaling legs,
every named start) is solved twice on the same build:

``declared``           the source model handed to ``pounce.Problem`` with
                       its pairs declared (``complementarity=``) and no
                       option but ``print_level`` -- what a user writes;
                       POUNCE picks the formulation and certifies it.
``scholtes_then_ncp``  the route Gate 0 chose as supported, run by this
                       harness's own `runner.run_cell` with its pinned
                       options.

Scored the way `report.py` scores the route table: *solved* is a returned
point whose worst source residual is <= 1e-5; *at f\\** and *below f\\** are
against the branch-enumeration oracle's optimum in the manifest.

A pair ``0 <= G ⟂ H >= 0`` is declared as ``(row H >= 0, variable)``. When
``G`` is a plain variable with a zero lower bound (or none), that variable is
the pair's variable, as a user would write it; otherwise a slack ``s = G(x)``,
``s >= 0`` stands in, as AMPL does for an expression on that side.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional

import numpy as np

from . import routes as R
from . import manifest as M
from .cases import all_cases
from .report import _worst_source
from .runner import run_cell
from .spec import SCALINGS, MpccCase

BIG = 2e19


class DeclaredNlp:
    def __init__(self, case: MpccCase):
        n = case.n
        lb = np.where(np.isfinite(case.lb), case.lb, -BIG).astype(float)
        ub = np.where(np.isfinite(case.ub), case.ub, BIG).astype(float)
        var_of_pair: List[Optional[int]] = []
        for p in case.pairs:
            nz = np.flatnonzero(p.G.a)
            j = int(nz[0]) if nz.size == 1 else None
            plain = (
                j is not None
                and p.G.a[j] == 1.0
                and p.G.b == 0.0
                and ub[j] >= BIG
                and lb[j] in (0.0, -BIG)
            )
            var_of_pair.append(j if plain else None)
        slack_of = {}
        for i, j in enumerate(var_of_pair):
            if j is None:
                slack_of[i] = n + len(slack_of)
        self.case = case
        self.n_src = n
        self.n = n + len(slack_of)
        self.lb = np.concatenate([lb, np.zeros(len(slack_of))])
        self.ub = np.concatenate([ub, np.full(len(slack_of), BIG)])
        for j in var_of_pair:
            if j is not None:
                self.lb[j] = 0.0  # the declaration's own G >= 0
        # rows: source rows, then H_i >= 0, then G_i(x) - s_i = 0
        cl = [r.lo for r in case.rows]
        cu = [r.hi for r in case.rows]
        h0 = len(cl)
        cl += [0.0] * case.q
        cu += [BIG] * case.q
        cl += [0.0] * len(slack_of)
        cu += [0.0] * len(slack_of)
        self.cl = np.clip(np.array(cl, float), -BIG, BIG)
        self.cu = np.clip(np.array(cu, float), -BIG, BIG)
        self.m = len(cl)
        self.slack_of = slack_of
        self.complementarity = [
            (h0 + i, var_of_pair[i] if var_of_pair[i] is not None else slack_of[i])
            for i in range(case.q)
        ]

    def x0(self, x_src):
        x = np.zeros(self.n)
        x[: self.n_src] = x_src
        for i, s in self.slack_of.items():
            x[s] = self.case.pairs[i].G.value(np.asarray(x_src, float))
        return x

    def objective(self, x):
        return self.case.objective.value(np.asarray(x[: self.n_src]))

    def gradient(self, x):
        g = np.zeros(self.n)
        g[: self.n_src] = self.case.objective.grad(np.asarray(x[: self.n_src]))
        return g

    def constraints(self, x):
        xs = np.asarray(x[: self.n_src])
        out = [r.form.value(xs) for r in self.case.rows]
        out += [p.H.value(xs) for p in self.case.pairs]
        out += [self.case.pairs[i].G.value(xs) - x[s] for i, s in self.slack_of.items()]
        return np.array(out)

    def _jac_dense(self, x):
        xs = np.asarray(x[: self.n_src])
        J = np.zeros((self.m, self.n))
        k = 0
        for r in self.case.rows:
            J[k, : self.n_src] = r.form.grad(xs)
            k += 1
        for p in self.case.pairs:
            J[k, : self.n_src] = p.H.a
            k += 1
        for i, s in self.slack_of.items():
            J[k, : self.n_src] = self.case.pairs[i].G.a
            J[k, s] = -1.0
            k += 1
        return J

    def jacobianstructure(self):
        return np.repeat(np.arange(self.m), self.n), np.tile(np.arange(self.n), self.m)

    def jacobian(self, x):
        return self._jac_dense(x).ravel()

    def hessianstructure(self):
        return np.tril_indices(self.n)

    def hessian(self, x, lagrange, obj_factor):
        xs = np.asarray(x[: self.n_src])
        h = np.zeros((self.n, self.n))
        hs = obj_factor * self.case.objective.hess(xs)
        for lam, r in zip(lagrange, self.case.rows):
            if lam != 0.0:
                hs = hs + lam * r.form.hess(xs)
        h[: self.n_src, : self.n_src] = hs
        r, c = np.tril_indices(self.n)
        return h[r, c]


def solve_declared(case: MpccCase, x0) -> dict:
    import pounce

    nlp = DeclaredNlp(case)
    prob = pounce.Problem(
        n=nlp.n, m=nlp.m, problem_obj=nlp, lb=nlp.lb, ub=nlp.ub, cl=nlp.cl, cu=nlp.cu,
        complementarity=nlp.complementarity,
    )
    prob.add_option("print_level", 0)
    t0 = time.perf_counter()
    x, info = prob.solve(nlp.x0(x0))
    wall = time.perf_counter() - t0
    xs = np.asarray(x[: case.n])
    cc = info["complementarity"]
    cert = cc["certificate"] or {}
    return {
        "status": int(info["status"]),
        "status_msg": str(info["status_msg"]),
        "accepted_by": cc["accepted_by"],
        "class": cert.get("class"),
        "iters": int(cc["total_iterations"]),
        "solves": int(cc["n_solves"]),
        "wall": wall,
        "obj": float(case.objective.value(xs)),
        "source": case.source_feasibility(xs),
        "x": xs.tolist(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="write every cell here")
    args = ap.parse_args(argv)

    man = {c["name"]: c for c in M.load()["cases"]}
    cells = []
    for base in all_cases():
        for sc in SCALINGS:
            case = base.rescale(SCALINGS[sc](base.n)) if sc != "unit" else base
            for st in case.starts:
                d = solve_declared(case, case.starts[st])
                ref = run_cell(case, R.ROUTES["scholtes_then_ncp"], "none", st, sc,
                               SCALINGS[sc](base.n), capture_log=False)
                cells.append({
                    "case": base.name, "scaling": sc, "start": st,
                    "fstar": (man.get(base.name, {}).get("expected") or {}).get("obj"),
                    "declared": d,
                    "scholtes_then_ncp": {
                        "ok": ref.ok, "obj": ref.obj, "iters": ref.iters,
                        "status_msg": ref.status_msg,
                        "source": ref.source or {},
                    },
                })

    def score(rows, key):
        solved = at = below = 0
        iters = []
        for c in rows:
            r = c[key]
            ok = r["ok"] if key != "declared" else r["status"] in R.OK_STATUS
            worst = _worst_source(r) if r["source"] else float("inf")
            if ok and worst <= 1e-5:
                solved += 1
            iters.append(r["iters"])
            f = c["fstar"]
            if f is None or r["obj"] is None:
                continue
            if r["obj"] < f - 1e-6:
                below += 1
            elif abs(r["obj"] - f) <= 1e-5 * max(1.0, abs(f)):
                at += 1
        iters.sort()
        return solved, at, below, len(rows), iters[len(iters) // 2] if iters else 0

    feas = [c for c in cells if c["case"] != "infeasible_pair"]
    print("| route | solved | at f* | below f* | of | median iters |")
    print("|---|---:|---:|---:|---:|---:|")
    for key in ("declared", "scholtes_then_ncp"):
        s, a, b, n, mi = score(feas, key)
        print(f"| `{key}` | {s} | {a} | {b} | {n} | {mi} |")
    by = {}
    for c in feas:
        d = c["declared"]
        by.setdefault((d["accepted_by"], d["class"]), 0)
        by[(d["accepted_by"], d["class"])] += 1
    print("\ndeclared: answered by / certificate")
    for (stage, klass), k in sorted(by.items(), key=lambda t: -t[1]):
        print(f"  {stage!s:>14} {klass!s:>12}  {k}")
    print("\ncells where the two disagree (solved, or objective by > 1e-5):")
    for c in feas:
        d, r = c["declared"], c["scholtes_then_ncp"]
        dok = d["status"] in R.OK_STATUS and _worst_source(d) <= 1e-5
        rok = r["ok"] and r["source"] and _worst_source(r) <= 1e-5
        if dok != bool(rok) or (r["obj"] is not None and abs(d["obj"] - r["obj"]) > 1e-5 * max(1, abs(r["obj"]))):
            print(f"  {c['case']:<22} {c['scaling']:<5} {c['start']:<10} f*={c['fstar']}  "
                  f"declared {d['obj']:.6g} ({d['accepted_by']}, {d['class']}, {d['status_msg']})  "
                  f"scholtes_then_ncp {r['obj']} ({r['status_msg']})")
    inf = [c for c in cells if c["case"] == "infeasible_pair"]
    for c in inf:
        print(f"\ninfeasible_pair {c['scaling']} {c['start']}: declared {c['declared']['status_msg']}, "
              f"accepted_by {c['declared']['accepted_by']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(cells, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
