#!/usr/bin/env python3
"""Corrective N-1 AC security-constrained OPF, as a family of .nl instances.

Structured-KKT Phase 0c: the *arrowhead* benchmark. The model is a base-case
AC-OPF plus K contingency copies of the network, each with one branch out of
service. They share one small set of variables and are otherwise independent:

  global (shared by every block): real power of every generator except the
      reference one -- the base-case dispatch;
  block k (base case k = 0, contingencies k = 1..K): bus voltage magnitudes
      and angles, generator reactive power, the reference generator's real
      power, and (contingencies only) a corrective re-dispatch of every other
      generator within a ramp limit of RAMP x its capacity, with that block's
      power balance, voltage limits, and thermal limits on its in-service
      branches.

Corrective rather than preventive: with one dispatch shared by every
contingency (preventive), case118_ieee is jointly infeasible from K = 64 even
though every single contingency is feasible (Ipopt agrees). The re-dispatch
keeps each contingency individually recoverable without changing the topology:
blocks still couple only through the base dispatch.

So the KKT system is block-diagonal apart from the global dispatch columns --
the arrowhead topology where decomposition by structure is best established
(independent block factorizations, a Schur complement on the globals). K is
the block-count knob; files are named ..._K###.nl for sweep.py.

Polar formulation, pglib-opf cases (CC BY 4.0), downloaded on first use.
Contingency thermal limits are 1.3 x rate A (an emergency rating). Outages are
chosen first in case order among branches whose loss keeps the network
connected *and* whose single-contingency problem pounce solves (screened once
per case with --screen BIN and cached): an infeasible contingency makes every
larger instance infeasible, which measures the infeasibility detector rather
than the KKT solve. Written with symbolic labels, so the .row/.col files carry each
variable's and constraint's block index as the first index of its name.

Usage:
  gen_scopf.py OUTDIR CASE K [K ...] [--screen POUNCE_BIN]
  e.g.  gen_scopf.py nl case118_ieee 1 2 4 8 16 --screen target/release/pounce
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pyomo.environ as pyo

RAW = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"
EMERGENCY = 1.3
RAMP = 0.2


def _matrix(text, name):
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.replace(";", " ").split()])
    return np.array(rows)


def load_case(case, cache_dir):
    path = os.path.join(cache_dir, f"pglib_opf_{case}.m")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with urllib.request.urlopen(RAW + f"pglib_opf_{case}.m", timeout=300) as r:
            open(path, "wb").write(r.read())
    text = open(path).read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text).group(1))
    bus, gen, br, cost = (_matrix(text, k) for k in ("bus", "gen", "branch", "gencost"))
    gen_on = gen[:, 7] > 0
    br_on = br[:, 10] > 0
    return base, bus, gen[gen_on], cost[gen_on], br[br_on]


def connected_without(nb, edges, drop):
    parent = list(range(nb))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e, (f, t) in enumerate(edges):
        if e != drop:
            parent[find(f)] = find(t)
    return len({find(i) for i in range(nb)}) == 1


def candidate_outages(case, cache_dir):
    _, bus, _, _, br = load_case(case, cache_dir)
    idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    edges = [(idx[int(f)], idx[int(t)]) for f, t in br[:, :2]]
    return [e for e in range(len(br)) if connected_without(len(bus), edges, e)]


def screen_outages(case, cache_dir, pounce, need):
    """Non-islanding outages whose one-contingency SCOPF pounce solves.

    Cached in CACHE/<case>_feasible_outages.json; extended on demand."""
    path = os.path.join(cache_dir, f"{case}_feasible_outages.json")
    data = json.load(open(path)) if os.path.exists(path) else {"checked": [], "feasible": []}
    checked = set(data["checked"])
    for e in candidate_outages(case, cache_dir):
        if len(data["feasible"]) >= need:
            break
        if e in checked:
            continue
        m, _ = build(case, 1, cache_dir, outages=[e])
        with tempfile.TemporaryDirectory() as tmp:
            nl = os.path.join(tmp, "c.nl")
            rep = os.path.join(tmp, "r.json")
            m.write(nl, format="nl")
            subprocess.run([pounce, nl, os.path.join(tmp, "c.sol"), "--json-output", rep,
                            "print_level=0", "max_iter=500"], capture_output=True, timeout=600)
            ok = os.path.exists(rep) and json.load(open(rep))["solution"]["status"] in (
                "SolveSucceeded", "SolvedToAcceptableLevel")
        data["checked"].append(e)
        if ok:
            data["feasible"].append(e)
        print(f"  screen {case} outage {e}: {'ok' if ok else 'INFEASIBLE / failed'}", flush=True)
        json.dump(data, open(path, "w"))
    return data["feasible"][:need]


def build(case, K, cache_dir, outages=None):
    base, bus, gen, cost, br = load_case(case, cache_dir)
    nb, ng, nl = len(bus), len(gen), len(br)
    idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    ref = int(np.flatnonzero(bus[:, 1] == 3)[0])
    gbus = [idx[int(g)] for g in gen[:, 0]]
    ref_gen = next(g for g in range(ng) if gbus[g] == ref)
    edges = [(idx[int(f)], idx[int(t)]) for f, t in br[:, :2]]
    if outages is None:
        outages = [e for e in range(nl) if connected_without(nb, edges, e)][:K]
    outages = list(outages)[:K]
    if len(outages) < K:
        raise SystemExit(f"{case}: only {len(outages)} usable outages for K={K}")

    # branch admittances (standard pi model with tap and shift)
    r, x, b = br[:, 2], br[:, 3], br[:, 4]
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8])
    shift = np.deg2rad(br[:, 9])
    y = 1.0 / (r + 1j * x)
    g_, b_ = y.real, y.imag
    rate = np.where(br[:, 5] > 0, br[:, 5], 1e4) / base

    m = pyo.ConcreteModel(name=f"scopf_{case}_K{K}")
    m.B = pyo.RangeSet(0, K)                    # block 0 = base case
    m.N = pyo.RangeSet(0, nb - 1)
    m.G = pyo.RangeSet(0, ng - 1)
    m.L = pyo.RangeSet(0, nl - 1)
    shared = [g for g in range(ng) if g != ref_gen]
    m.S = pyo.Set(initialize=shared)

    pmin, pmax = gen[:, 9] / base, gen[:, 8] / base
    qmin, qmax = gen[:, 4] / base, gen[:, 3] / base
    # global: preventive dispatch of the non-reference generators
    m.pg = pyo.Var(m.S, bounds=lambda _, g: (pmin[g], pmax[g]),
                   initialize=lambda _, g: 0.5 * (pmin[g] + pmax[g]))
    # per block
    m.vm = pyo.Var(m.B, m.N, bounds=lambda _, k, i: (bus[i, 12], bus[i, 11]), initialize=1.0)
    m.va = pyo.Var(m.B, m.N, initialize=0.0)
    m.qg = pyo.Var(m.B, m.G, bounds=lambda _, k, g: (qmin[g], qmax[g]), initialize=0.0)
    m.pref = pyo.Var(m.B, bounds=(pmin[ref_gen], pmax[ref_gen]),
                     initialize=0.5 * (pmin[ref_gen] + pmax[ref_gen]))
    # corrective re-dispatch in each contingency (block 0 has none)
    m.C = pyo.RangeSet(1, K) if K > 0 else pyo.Set(initialize=[])
    m.dpg = pyo.Var(m.C, m.S, bounds=lambda _, k, g: (-RAMP * pmax[g], RAMP * pmax[g]),
                    initialize=0.0)
    m.dpg_box = pyo.ConstraintList()   # the re-dispatched output stays in its limits
    for k in m.C:
        for g in shared:
            m.dpg_box.add(pyo.inequality(pmin[g], m.pg[g] + m.dpg[k, g], pmax[g]))
    for k in m.B:
        m.va[k, ref].fix(0.0)

    out = {k: (outages[k - 1] if k > 0 else None) for k in m.B}

    def p_gen(k, g):
        if g == ref_gen:
            return m.pref[k]
        return m.pg[g] if k == 0 else m.pg[g] + m.dpg[k, g]

    def flows(k, e):
        f, t = edges[e]
        vf, vt = m.vm[k, f], m.vm[k, t]
        d = m.va[k, f] - m.va[k, t] - shift[e]
        tf = tap[e]
        pf = (g_[e] / tf**2) * vf**2 - (vf * vt / tf) * (g_[e] * pyo.cos(d) + b_[e] * pyo.sin(d))
        qf = -((b_[e] + b[e] / 2) / tf**2) * vf**2 - (vf * vt / tf) * (g_[e] * pyo.sin(d) - b_[e] * pyo.cos(d))
        pt = g_[e] * vt**2 - (vf * vt / tf) * (g_[e] * pyo.cos(-d) + b_[e] * pyo.sin(-d))
        qt = -(b_[e] + b[e] / 2) * vt**2 - (vf * vt / tf) * (g_[e] * pyo.sin(-d) - b_[e] * pyo.cos(-d))
        return pf, qf, pt, qt

    m.pbal = pyo.ConstraintList()
    m.qbal = pyo.ConstraintList()
    m.therm = pyo.ConstraintList()
    for k in m.B:
        p_inj = {i: -bus[i, 2] / base - bus[i, 4] / base * m.vm[k, i] ** 2 for i in range(nb)}
        q_inj = {i: -bus[i, 3] / base + bus[i, 5] / base * m.vm[k, i] ** 2 for i in range(nb)}
        for g in range(ng):
            p_inj[gbus[g]] = p_inj[gbus[g]] + p_gen(k, g)
            q_inj[gbus[g]] = q_inj[gbus[g]] + m.qg[k, g]
        p_out = {i: 0 for i in range(nb)}
        q_out = {i: 0 for i in range(nb)}
        limit = 1.0 if k == 0 else EMERGENCY
        for e in range(nl):
            if e == out[k]:
                continue
            f, t = edges[e]
            pf, qf, pt, qt = flows(k, e)
            p_out[f] += pf; q_out[f] += qf; p_out[t] += pt; q_out[t] += qt
            if rate[e] < 99:
                m.therm.add(pf**2 + qf**2 <= (limit * rate[e]) ** 2)
                m.therm.add(pt**2 + qt**2 <= (limit * rate[e]) ** 2)
        for i in range(nb):
            m.pbal.add(p_inj[i] == p_out[i])
            m.qbal.add(q_inj[i] == q_out[i])

    c2, c1, c0 = cost[:, 4] * base**2, cost[:, 5] * base, cost[:, 6]
    m.obj = pyo.Objective(expr=sum(
        c2[g] * p_gen(0, g) ** 2 + c1[g] * p_gen(0, g) + c0[g] for g in range(ng)))
    return m, dict(nb=nb, ng=ng, nl=nl, shared=len(shared), outages=outages)


def main(outdir, case, Ks, pounce=None):
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, "pglib")
    load_case(case, cache)
    outages = screen_outages(case, cache, pounce, max(Ks)) if pounce else None
    for K in Ks:
        m, info = build(case, K, cache, outages=outages)
        path = os.path.join(outdir, f"scopf_{case}_K{K:03d}.nl")
        m.write(path, format="nl", io_options={"symbolic_solver_labels": True})
        nv = sum(1 for v in m.component_data_objects(pyo.Var) if not v.fixed)
        print(f"K={K:3d}  blocks={K + 1:3d}  vars={nv:7d}  shared={info['shared']}  -> {path}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    pounce = None
    if "--screen" in args:
        i = args.index("--screen")
        pounce = args[i + 1]
        del args[i:i + 2]
    main(args[0], args[1], [int(k) for k in args[2:]], pounce)
