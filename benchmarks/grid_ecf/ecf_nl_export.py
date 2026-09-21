#!/usr/bin/env python3
"""Generate the AC-OPF equivalent-circuit (ECF) suite as AMPL ``.nl`` files.

What this is
------------
Jereminov, Pandey & Pileggi, *Equivalent Circuit Formulation for Solving AC
Optimal Power Flow* (CMU, 2018), reformulate AC-OPF in current/voltage/
admittance state variables instead of the usual polar voltage magnitude and
angle. Their claim is that the resulting model is dramatically more robust to
the starting point, because it carries **no trigonometry**: every constraint is
bilinear or quadratic.

The paper couples that formulation to a bespoke solver (ESCAPE: SPICE-style
per-variable limiting, diode-form complementarity, Tx-stepping homotopy) and
argues at length that a general-purpose primal-dual interior-point method is
the wrong tool. But the *formulation* and the *solver* are separable, and the
paper never runs the separation. This exporter does: it emits the ECF model as
an ordinary NLP so POUNCE (or Ipopt, or anything else reading ``.nl``) can
solve it directly.

The controlled comparison
-------------------------
``--form polar`` emits a matched polar baseline from the **same parser, the
same case data, the same objective, and the same thermal-limit convention**.
That control is the reason this script builds its own polar model rather than
reusing ``benchmarks/grid/``: the existing grid suite's ``.nl`` were written by
Egret, whose encoding carries extra branch-flow variables (191 variables for a
14-bus case, against 38 polar / 62 ECF here), so a formulation difference measured against it
would be confounded by an encoding difference.

Run both forms and the only thing that varies is the formulation.

The variables (ECF)
-------------------
Per bus ``i``:  ``vr[i]``, ``vi[i]`` (rectangular voltage), ``dsq[i]``
Per online generator ``g``:  ``G[g]``, ``B[g]``, ``pg[g]``, ``qg[g]``
Per branch end ``e`` (when ``--thermal current``):  ``isq[e]``

and the constraints are the paper's equations, numbered as published:

* (26)-(27)  KCL on the real and imaginary sub-circuits, in which the
  generator appears as its **GB macro-model** -- a *negative* conductance
  supplying real power in parallel with a susceptance carrying reactive
  power -- and the constant-power load as the current source (17)-(18).
* (30)  ``dsq[i] = vr[i]^2 + vi[i]^2``.  ``dsq`` replaces ``|V|^2``
  everywhere it would otherwise appear, which is what keeps the generator
  power relations bilinear rather than cubic; the voltage magnitude limits
  become the plain box ``Vmin^2 <= dsq <= Vmax^2`` (14).
* (28)-(29)  ``pg = -G * dsq`` and ``qg = B * dsq``.
* (31)  ``isq[e] = IR[e]^2 + II[e]^2`` with the box (19).

Two departures from the paper, both deliberate and both documented here
because they change what the numbers mean:

1. **Thermal limits.**  The paper replaces the apparent-power limit
   ``Pf^2 + Qf^2 <= Smax^2`` with a current limit ``|I|^2 <= (Smax/Vnom)^2``
   (19), arguing that a conductor's rating is physically a current rating.
   That is a different feasible set, so an ECF run under (19) and a polar run
   under the apparent-power limit are not solving the same problem and their
   objectives may legitimately differ wherever a limit binds.  ``--thermal``
   selects the convention and applies it to **both** forms, so the comparison
   can be run either way.

   The default is ``apparent``, not the paper's ``current``, for one reason:
   under ``apparent`` both forms reproduce the **published pglib optimal
   objectives**, so the suite is checkable against a reference outside this
   repo rather than only against itself.  ``--thermal current`` gives the
   paper's model literally, and the two forms still agree with each other
   under it -- they just no longer agree with the published number.
2. **Angle-difference limits.**  pglib carries ``angmin``/``angmax`` on every
   branch (typically +/-30 deg) and they bind on several cases.  The paper
   does not mention them.  Dropping them would make the ECF model a relaxation
   of the polar one and every objective comparison meaningless, so they are
   included, in the standard bilinear rectangular form: with
   ``w_r = vr_f*vr_t + vi_f*vi_t`` and ``w_i = vi_f*vr_t - vr_f*vi_t``
   (the real and imaginary parts of ``V_f * conj(V_t)``),
   ``tan(amin)*w_r <= w_i <= tan(amax)*w_r``, valid while ``|dtheta| < 90``.
   ``--no-angle-limits`` drops them from both forms if you want the paper's
   model literally.

The rectangular sign symmetry
-----------------------------
Every constraint above is odd under ``V -> -V`` (KCL is linear in V and the
load current (17)-(18) is homogeneous of degree 1), so the rectangular model
has a spurious mirror solution.  Fixing ``vi = 0`` at the reference bus is not
enough to kill it -- that leaves ``vr_ref = +/-|V|``.  The reference bus's
``vr`` is therefore boxed to ``[Vmin, Vmax]`` (strictly positive) rather than
to ``[-Vmax, Vmax]``, which pins the branch without excluding any physical
operating point.

Starting point
--------------
The same physical start in both forms, and a deliberately unhelpful one: flat
voltage (``|V| = 1``, all angles zero), ``pg`` at the midpoint of its box,
``qg = 0``.  The paper initializes from a DC-OPF solution; a flat start is the
harder test and is what makes a robustness claim mean anything.

Every one of those is clamped into its own box before use -- several pglib
cases carry buses with ``Vmin > 1`` and generators with ``Qmin > 0`` -- because
an initial point outside its bounds is violated *differently* by each
formulation, which would silently confound the comparison.  ``G`` and ``B``
are then initialized through (28)-(29) at that clamped point, so the start
satisfies those rows exactly.

``--start random --seed N`` replaces the flat start with a random one, and is
how the paper's robustness claim gets tested rather than argued about.  What
is randomized is the **physical** operating point -- a voltage magnitude and
angle per bus, a real and reactive output per generator, each drawn uniformly
from its own box -- which each formulation then maps into its own variables.
The same case and seed therefore give the two forms *the same physical start*,
which is the whole point: a robustness difference between them is then a
property of the coordinates, not of where each happened to be dropped.

Usage
-----
    python3 ecf_nl_export.py                      # ECF, all default cases
    python3 ecf_nl_export.py --form polar         # the matched baseline
    python3 ecf_nl_export.py --thermal apparent   # S-limits in both forms
    python3 ecf_nl_export.py case14_ieee case118_ieee
    python3 ecf_nl_export.py --start random --seed 3 --angle-spread 0.5 \
        --out-dir /tmp/ecf_s3                     # random start (needs --out-dir)
    python3 ecf_nl_export.py --no-fetch           # use cached .m only
    python3 ecf_nl_export.py --list               # print the default case list

The downloaded ``.m`` cases (under ``--data-dir``) and the generated ``.nl``
are regenerated locally and not tracked in git, like every other suite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
import zlib

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Reals,
    Var,
    cos,
    minimize,
    sin,
)

RAW_BASE = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"

#: The cases carried by ``benchmarks/grid/``, so the two suites line up.
DEFAULT_CASES = [
    "case3_lmbd",
    "case5_pjm",
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case73_ieee_rts",
    "case89_pegase",
    "case118_ieee",
    "case162_ieee_dtc",
    "case179_goc",
    "case240_pserc",
    "case300_ieee",
    "case500_goc",
    "case588_sdet",
    "case793_goc",
    "case1354_pegase",
    "case1888_rte",
    "case2000_goc",
    "case4661_sdet",
    "case10000_goc",
]

# MATPOWER column indices (MATPOWER manual, appendix B).
BUS_I, BUS_TYPE, PD, QD, GS, BS = 0, 1, 2, 3, 4, 5
VMAX, VMIN = 11, 12
GEN_BUS, QMAX, QMIN, GEN_STATUS, PMAX, PMIN = 0, 3, 4, 7, 8, 9
F_BUS, T_BUS, BR_R, BR_X, BR_B = 0, 1, 2, 3, 4
RATE_A, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX = 5, 8, 9, 10, 11, 12
COST_MODEL, NCOST, COST_C0 = 0, 3, 4

REF_BUS_TYPE = 3


# --------------------------------------------------------------------------
# MATPOWER .m parsing
# --------------------------------------------------------------------------
def parse_matpower(text: str) -> dict:
    """Parse the ``mpc.*`` matrices out of a MATPOWER case file.

    Only what an OPF needs: ``baseMVA``, ``bus``, ``gen``, ``branch``,
    ``gencost``.  Rows may carry trailing ``% comment`` text (pglib labels
    every generator with its fuel type), which is stripped before the split.
    """
    out: dict = {}

    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text)
    if not m:
        raise ValueError("no mpc.baseMVA in case file")
    out["baseMVA"] = float(m.group(1))

    for name in ("bus", "gen", "branch", "gencost"):
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\]\s*;", text, re.S)
        if not m:
            raise ValueError(f"no mpc.{name} matrix in case file")
        rows = []
        for line in m.group(1).splitlines():
            line = line.split("%")[0].strip().rstrip(";").strip()
            if not line:
                continue
            rows.append([float(t) for t in line.split()])
        if not rows:
            raise ValueError(f"mpc.{name} parsed empty")
        width = max(len(r) for r in rows)
        # gencost rows are ragged across cost models; pad rather than fail.
        out[name] = np.array([r + [0.0] * (width - len(r)) for r in rows])

    return out


class Network:
    """A MATPOWER case reduced to what both formulations need, in per-unit.

    Buses are renumbered to ``0..nb-1``; out-of-service generators and
    branches are dropped.  ``ybus`` folds in bus shunts and the full
    transformer pi-model (off-nominal tap ratio and phase shift), so both
    formulations see exactly the same network.
    """

    def __init__(self, name: str, mpc: dict):
        base = mpc["baseMVA"]
        bus, gen, branch, gencost = (
            mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]
        )

        self.name = name
        self.base = base
        self.nb = bus.shape[0]
        self.index = {int(bus[i, BUS_I]): i for i in range(self.nb)}

        self.pd = bus[:, PD] / base
        self.qd = bus[:, QD] / base
        self.vmax = bus[:, VMAX].copy()
        self.vmin = bus[:, VMIN].copy()

        ref = np.flatnonzero(bus[:, BUS_TYPE] == REF_BUS_TYPE)
        if len(ref) == 0:
            raise ValueError("case has no reference bus")
        self.ref = int(ref[0])

        # Generators, in service only, carrying their cost row alongside.
        if gencost.shape[0] < gen.shape[0]:
            raise ValueError("fewer gencost rows than gen rows")
        self.gens = []
        for j in range(gen.shape[0]):
            if gen[j, GEN_STATUS] <= 0:
                continue
            c = gencost[j]
            if c[COST_MODEL] != 2:
                raise ValueError(
                    f"generator {j}: cost model {c[COST_MODEL]:.0f} is not "
                    "polynomial (model 2); piecewise-linear costs are not "
                    "supported by this exporter"
                )
            ncost = int(c[NCOST])
            if ncost > 3:
                raise ValueError(
                    f"generator {j}: cost polynomial of degree {ncost - 1} "
                    "exceeds quadratic"
                )
            # MATPOWER stores coefficients highest-order first; normalise to
            # (c2, c1, c0) with Pg in MW.
            coef = list(c[COST_C0:COST_C0 + ncost])
            coef = [0.0] * (3 - ncost) + coef
            self.gens.append({
                "bus": self.index[int(gen[j, GEN_BUS])],
                "pmin": gen[j, PMIN] / base,
                "pmax": gen[j, PMAX] / base,
                "qmin": gen[j, QMIN] / base,
                "qmax": gen[j, QMAX] / base,
                "c2": coef[0], "c1": coef[1], "c0": coef[2],
            })
        if not self.gens:
            raise ValueError("case has no in-service generators")

        # Branches, in service only, with the MATPOWER pi-model.
        self.branches = []
        for e in range(branch.shape[0]):
            if branch[e, BR_STATUS] <= 0:
                continue
            f = self.index[int(branch[e, F_BUS])]
            t = self.index[int(branch[e, T_BUS])]
            y = 1.0 / complex(branch[e, BR_R], branch[e, BR_X])
            bc = branch[e, BR_B]
            tap = branch[e, TAP]
            tap = 1.0 if tap == 0.0 else tap          # 0 means nominal
            shift = math.radians(branch[e, SHIFT])
            T = tap * complex(math.cos(shift), math.sin(shift))
            ysh = y + 1j * bc / 2.0
            self.branches.append({
                "f": f, "t": t,
                "yff": ysh / (tap * tap),
                "yft": -y / T.conjugate(),
                "ytf": -y / T,
                "ytt": ysh,
                # rate 0 is MATPOWER for "no limit"
                "rate": (branch[e, RATE_A] / base
                         if branch[e, RATE_A] > 0 else None),
                "amin": math.radians(branch[e, ANGMIN]),
                "amax": math.radians(branch[e, ANGMAX]),
            })

        # Nodal admittance matrix, as a dict of (i, k) -> complex.
        Y: dict = {}

        def add(i, k, v):
            Y[(i, k)] = Y.get((i, k), 0j) + v

        for i in range(self.nb):
            sh = complex(bus[i, GS], bus[i, BS]) / base
            if sh != 0:
                add(i, i, sh)
        for br in self.branches:
            f, t = br["f"], br["t"]
            add(f, f, br["yff"])
            add(f, t, br["yft"])
            add(t, f, br["ytf"])
            add(t, t, br["ytt"])
        self.ybus = {k: v for k, v in Y.items() if v != 0}

        self.rows: dict = {}
        for (i, k) in self.ybus:
            self.rows.setdefault(i, []).append(k)

    @staticmethod
    def angle_restrictive(bound) -> bool:
        """True if one angle bound actually restricts anything.

        Checked **per bound**, not per branch, and this matters more than it
        looks.  MATPOWER spells "no limit" as +/-360 degrees, and the
        rectangular form of the constraint carries ``tan(bound)`` --
        ``tan(radians(360))`` is ~0, not infinity.  So a branch limited on one
        side only (``angmin = -360``, ``angmax = 30``) would, under a
        per-branch guard, emit a lower row with coefficient ~0 and silently
        pin the angle difference to ``dtheta >= 0``: a wrong answer with no
        symptom.  Only bounds strictly inside +/-90 degrees, where the
        rectangular form is valid at all, produce a row.

        Every branch in the 20 default pglib cases is +/-30, so this guard
        changes nothing there; it exists so that a case which is not cannot
        fail quietly.
        """
        return abs(bound) < math.pi / 2 - 1e-9

    def angle_rows(self):
        """(lower-limited, upper-limited) branch index lists."""
        lo = [e for e, br in enumerate(self.branches)
              if self.angle_restrictive(br["amin"])]
        hi = [e for e, br in enumerate(self.branches)
              if self.angle_restrictive(br["amax"])]
        return lo, hi

    def cost_expr(self, pg):
        """Generation cost in $/h, with ``pg`` the per-unit power variables."""
        base = self.base
        return sum(
            g["c2"] * (pg[j] * base) ** 2 + g["c1"] * (pg[j] * base) + g["c0"]
            for j, g in enumerate(self.gens)
        )

    def physical_start(self, rng=None, angle_spread=0.5):
        """The shared starting operating point: (vm, va, pg, qg).

        With ``rng`` None this is the flat start -- unit voltage clamped into
        each bus's box, zero angles, mid-box real power, zero reactive power.
        With an ``rng`` every quantity is drawn uniformly from its own box
        (angles from ``+/- angle_spread`` radians, the reference bus always at
        zero).  Both formulations consume this same tuple, so a seed names one
        physical point rather than one point per formulation.
        """
        if rng is None:
            vm = self.vm_start()
            va = [0.0] * self.nb
            pg = [0.5 * (g["pmin"] + g["pmax"]) for g in self.gens]
            qg = [min(max(0.0, g["qmin"]), g["qmax"]) for g in self.gens]
            return vm, va, pg, qg

        vm = [float(rng.uniform(self.vmin[i], self.vmax[i]))
              for i in range(self.nb)]
        va = [0.0 if i == self.ref
              else float(rng.uniform(-angle_spread, angle_spread))
              for i in range(self.nb)]
        pg = [float(rng.uniform(g["pmin"], g["pmax"])) for g in self.gens]
        qg = [float(rng.uniform(g["qmin"], g["qmax"])) for g in self.gens]
        return vm, va, pg, qg

    def pg_start(self):
        return [0.5 * (g["pmin"] + g["pmax"]) for g in self.gens]

    def qg_start(self):
        return [min(max(0.0, g["qmin"]), g["qmax"]) for g in self.gens]

    def vm_start(self):
        """Flat 1.0 pu, clamped into each bus's own magnitude box.

        Several pglib cases (case1888_rte among them) carry buses whose Vmin
        exceeds 1.0, so an unclamped flat start would hand the solver an
        initial point outside its own bounds -- and a *different* violation in
        each formulation, which would quietly break the shared-start control
        this exporter exists to maintain.
        """
        return [min(max(1.0, float(self.vmin[i])), float(self.vmax[i]))
                for i in range(self.nb)]


# --------------------------------------------------------------------------
# The equivalent-circuit formulation
# --------------------------------------------------------------------------
def build_ecf(net: Network, thermal: str, angle_limits: bool,
              start=None) -> ConcreteModel:
    """The paper's model: rectangular V, GB generators, dsq, current limits."""
    nb, ng = net.nb, len(net.gens)
    nbr = len(net.branches)
    m = ConcreteModel(name=f"{net.name}_ecf")
    B = range(nb)
    G = range(ng)

    vm0, va0, pg0, qg0 = start if start is not None else net.physical_start()
    vr0 = [vm0[i] * math.cos(va0[i]) for i in range(nb)]
    vi0 = [vm0[i] * math.sin(va0[i]) for i in range(nb)]
    dsq0 = [v * v for v in vm0]

    # Reference bus vr is boxed strictly positive to kill the V -> -V mirror.
    def vr_bounds(_m, i):
        if i == net.ref:
            return (float(net.vmin[i]), float(net.vmax[i]))
        return (-float(net.vmax[i]), float(net.vmax[i]))

    def vi_bounds(_m, i):
        if i == net.ref:
            return (0.0, 0.0)
        return (-float(net.vmax[i]), float(net.vmax[i]))

    m.vr = Var(B, domain=Reals, bounds=vr_bounds,
               initialize=lambda _m, i: vr0[i])
    m.vi = Var(B, domain=Reals, bounds=vi_bounds,
               initialize=lambda _m, i: vi0[i])
    m.dsq = Var(
        B, domain=Reals,
        bounds=lambda _m, i: (float(net.vmin[i]) ** 2, float(net.vmax[i]) ** 2),
        initialize=lambda _m, i: dsq0[i],
    )
    m.pg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: (net.gens[j]["pmin"], net.gens[j]["pmax"]),
        initialize=lambda _m, j: pg0[j],
    )
    m.qg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: (net.gens[j]["qmin"], net.gens[j]["qmax"]),
        initialize=lambda _m, j: qg0[j],
    )

    # G and B are tied to pg/qg by (28)-(29) through dsq, which lives in
    # [Vmin^2, Vmax^2]; those two boxes imply the admittance box below.
    # Interval arithmetic on p = -G*dsq over dsq in [lo, hi], both signs of p.
    def adm_box(lo_p, hi_p, lo_d, hi_d, sign):
        vals = []
        for p in (lo_p, hi_p):
            for d in (lo_d, hi_d):
                vals.append(sign * p / d)
        return min(vals), max(vals)

    def gb_bounds(j, sign):
        g = net.gens[j]
        i = g["bus"]
        lo_d = float(net.vmin[i]) ** 2
        hi_d = float(net.vmax[i]) ** 2
        if sign < 0:
            return adm_box(g["pmin"], g["pmax"], lo_d, hi_d, -1.0)
        return adm_box(g["qmin"], g["qmax"], lo_d, hi_d, 1.0)

    # Initialized THROUGH (28)-(29) at the shared start rather than
    # independently, so the starting point satisfies those two rows exactly
    # and lands inside the boxes derived from the same interval arithmetic.
    m.gg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: gb_bounds(j, -1),
        initialize=lambda _m, j: -pg0[j] / dsq0[net.gens[j]["bus"]],
    )
    m.bg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: gb_bounds(j, +1),
        initialize=lambda _m, j: qg0[j] / dsq0[net.gens[j]["bus"]],
    )

    gens_at = {}
    for j, g in enumerate(net.gens):
        gens_at.setdefault(g["bus"], []).append(j)

    # (26)-(27): KCL on the real and imaginary sub-circuits.
    def kcl_r(_m, i):
        e = sum(net.ybus[(i, k)].real * m.vr[k]
                - net.ybus[(i, k)].imag * m.vi[k] for k in net.rows.get(i, []))
        for j in gens_at.get(i, []):
            e += m.gg[j] * m.vr[i] - m.bg[j] * m.vi[i]
        if net.pd[i] != 0.0 or net.qd[i] != 0.0:
            e += (net.pd[i] * m.vr[i] + net.qd[i] * m.vi[i]) / m.dsq[i]
        return e == 0.0

    def kcl_i(_m, i):
        e = sum(net.ybus[(i, k)].real * m.vi[k]
                + net.ybus[(i, k)].imag * m.vr[k] for k in net.rows.get(i, []))
        for j in gens_at.get(i, []):
            e += m.gg[j] * m.vi[i] + m.bg[j] * m.vr[i]
        if net.pd[i] != 0.0 or net.qd[i] != 0.0:
            e += (net.pd[i] * m.vi[i] - net.qd[i] * m.vr[i]) / m.dsq[i]
        return e == 0.0

    m.kcl_real = Constraint(B, rule=kcl_r)
    m.kcl_imag = Constraint(B, rule=kcl_i)

    # (30) dsq = vr^2 + vi^2
    m.dsq_def = Constraint(
        B, rule=lambda _m, i: m.vr[i] ** 2 + m.vi[i] ** 2 - m.dsq[i] == 0.0
    )

    # (28)-(29) pg = -G*dsq, qg = B*dsq
    m.pg_def = Constraint(
        G, rule=lambda _m, j: m.pg[j] + m.gg[j] * m.dsq[net.gens[j]["bus"]] == 0.0
    )
    m.qg_def = Constraint(
        G, rule=lambda _m, j: m.qg[j] - m.bg[j] * m.dsq[net.gens[j]["bus"]] == 0.0
    )

    _add_thermal_rect(m, net, thermal, nbr)
    if angle_limits:
        _add_angle_rect(m, net)

    m.obj = Objective(expr=net.cost_expr(m.pg), sense=minimize)
    return m


def _branch_current_rect(m, net, br, end):
    """(I_re, I_im) at one end of a branch, from rectangular voltages."""
    f, t = br["f"], br["t"]
    if end == "f":
        a, z, ya, yz = f, t, br["yff"], br["yft"]
    else:
        a, z, ya, yz = t, f, br["ytt"], br["ytf"]
    re = (ya.real * m.vr[a] - ya.imag * m.vi[a]
          + yz.real * m.vr[z] - yz.imag * m.vi[z])
    im = (ya.real * m.vi[a] + ya.imag * m.vr[a]
          + yz.real * m.vi[z] + yz.imag * m.vr[z])
    return re, im


def _add_thermal_rect(m, net, thermal, nbr):
    """Branch thermal limits for the ECF model, in the chosen convention.

    ``current`` is the paper's (19)/(31): ``isq`` is a state variable with a
    box, defined by an equality.  ``apparent`` keeps the conventional
    ``Pf^2 + Qf^2 <= Smax^2`` inline, so the two forms can be compared under
    an identical feasible set.
    """
    ends = [(e, end) for e in range(nbr) for end in ("f", "t")
            if net.branches[e]["rate"] is not None]
    if not ends:
        return
    m.end_idx = list(range(len(ends)))
    emap = {k: ends[k] for k in m.end_idx}

    if thermal == "current":
        # Vnom = 1.0 pu, per the paper's eq (19).
        def isq_bounds(_m, k):
            e, _ = emap[k]
            return (0.0, float(net.branches[e]["rate"]) ** 2)

        m.isq = Var(m.end_idx, domain=Reals, bounds=isq_bounds, initialize=0.0)

        def isq_def(_m, k):
            e, end = emap[k]
            re, im = _branch_current_rect(m, net, net.branches[e], end)
            return re ** 2 + im ** 2 - m.isq[k] == 0.0

        m.isq_def = Constraint(m.end_idx, rule=isq_def)
    else:
        def s_limit(_m, k):
            e, end = emap[k]
            br = net.branches[e]
            a = br["f"] if end == "f" else br["t"]
            re, im = _branch_current_rect(m, net, br, end)
            # S = V conj(I): P = vr*Ire + vi*Iim, Q = vi*Ire - vr*Iim
            p = m.vr[a] * re + m.vi[a] * im
            q = m.vi[a] * re - m.vr[a] * im
            return p ** 2 + q ** 2 <= float(br["rate"]) ** 2

        m.s_limit = Constraint(m.end_idx, rule=s_limit)


def _add_angle_rect(m, net):
    """Angle-difference limits as bilinear rows on V_f * conj(V_t)."""
    lo_idx, hi_idx = net.angle_rows()

    def lo(_m, e):
        br = net.branches[e]
        f, t = br["f"], br["t"]
        wr = m.vr[f] * m.vr[t] + m.vi[f] * m.vi[t]
        wi = m.vi[f] * m.vr[t] - m.vr[f] * m.vi[t]
        return math.tan(br["amin"]) * wr - wi <= 0.0

    def hi(_m, e):
        br = net.branches[e]
        f, t = br["f"], br["t"]
        wr = m.vr[f] * m.vr[t] + m.vi[f] * m.vi[t]
        wi = m.vi[f] * m.vr[t] - m.vr[f] * m.vi[t]
        return wi - math.tan(br["amax"]) * wr <= 0.0

    if lo_idx:
        m.ang_lo_idx = lo_idx
        m.ang_lo = Constraint(m.ang_lo_idx, rule=lo)
    if hi_idx:
        m.ang_hi_idx = hi_idx
        m.ang_hi = Constraint(m.ang_hi_idx, rule=hi)


# --------------------------------------------------------------------------
# The matched polar baseline
# --------------------------------------------------------------------------
def build_polar(net: Network, thermal: str, angle_limits: bool,
                start=None) -> ConcreteModel:
    """Textbook polar AC-OPF from the same data, objective and limit choice."""
    nb, ng = net.nb, len(net.gens)
    m = ConcreteModel(name=f"{net.name}_polar")
    B = range(nb)
    G = range(ng)
    vm0, va0, pg0, qg0 = start if start is not None else net.physical_start()

    m.va = Var(
        B, domain=Reals,
        bounds=lambda _m, i: (0.0, 0.0) if i == net.ref else (-math.pi, math.pi),
        initialize=lambda _m, i: va0[i],
    )
    m.vm = Var(
        B, domain=Reals,
        bounds=lambda _m, i: (float(net.vmin[i]), float(net.vmax[i])),
        initialize=lambda _m, i: vm0[i],
    )
    m.pg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: (net.gens[j]["pmin"], net.gens[j]["pmax"]),
        initialize=lambda _m, j: pg0[j],
    )
    m.qg = Var(
        G, domain=Reals,
        bounds=lambda _m, j: (net.gens[j]["qmin"], net.gens[j]["qmax"]),
        initialize=lambda _m, j: qg0[j],
    )

    gens_at = {}
    for j, g in enumerate(net.gens):
        gens_at.setdefault(g["bus"], []).append(j)

    def pbal(_m, i):
        inj = sum(
            m.vm[k] * (net.ybus[(i, k)].real * cos(m.va[i] - m.va[k])
                       + net.ybus[(i, k)].imag * sin(m.va[i] - m.va[k]))
            for k in net.rows.get(i, [])
        )
        return (sum(m.pg[j] for j in gens_at.get(i, []))
                - net.pd[i] - m.vm[i] * inj == 0.0)

    def qbal(_m, i):
        inj = sum(
            m.vm[k] * (net.ybus[(i, k)].real * sin(m.va[i] - m.va[k])
                       - net.ybus[(i, k)].imag * cos(m.va[i] - m.va[k]))
            for k in net.rows.get(i, [])
        )
        return (sum(m.qg[j] for j in gens_at.get(i, []))
                - net.qd[i] - m.vm[i] * inj == 0.0)

    m.pbal = Constraint(B, rule=pbal)
    m.qbal = Constraint(B, rule=qbal)

    # Branch flows: reuse the rectangular admittance algebra with
    # vr = vm cos(va), vi = vm sin(va), so both forms carry the SAME pi-model
    # arithmetic and only the coordinates differ.
    def vrec(i):
        return m.vm[i] * cos(m.va[i]), m.vm[i] * sin(m.va[i])

    def current(br, end):
        f, t = br["f"], br["t"]
        if end == "f":
            a, z, ya, yz = f, t, br["yff"], br["yft"]
        else:
            a, z, ya, yz = t, f, br["ytt"], br["ytf"]
        vra, via = vrec(a)
        vrz, viz = vrec(z)
        re = ya.real * vra - ya.imag * via + yz.real * vrz - yz.imag * viz
        im = ya.real * via + ya.imag * vra + yz.real * viz + yz.imag * vrz
        return a, re, im

    ends = [(e, end) for e in range(len(net.branches)) for end in ("f", "t")
            if net.branches[e]["rate"] is not None]
    if ends:
        m.end_idx = list(range(len(ends)))
        emap = {k: ends[k] for k in m.end_idx}

        def limit(_m, k):
            e, end = emap[k]
            br = net.branches[e]
            a, re, im = current(br, end)
            if thermal == "current":
                return re ** 2 + im ** 2 <= float(br["rate"]) ** 2
            vra, via = vrec(a)
            p = vra * re + via * im
            q = via * re - vra * im
            return p ** 2 + q ** 2 <= float(br["rate"]) ** 2

        m.thermal = Constraint(m.end_idx, rule=limit)

    if angle_limits:
        # Same per-bound guard as the rectangular form, so both models carry
        # the same rows -- a branch limited on one side only must not become
        # two-sided here and one-sided there.
        lo_idx, hi_idx = net.angle_rows()
        idx = sorted(set(lo_idx) | set(hi_idx))
        if idx:
            lo_set, hi_set = set(lo_idx), set(hi_idx)

            def ang(_m, e):
                br = net.branches[e]
                d = m.va[br["f"]] - m.va[br["t"]]
                if e in lo_set and e in hi_set:
                    return (br["amin"], d, br["amax"])
                if e in lo_set:
                    return br["amin"] <= d
                return d <= br["amax"]

            m.ang_idx = idx
            m.ang = Constraint(m.ang_idx, rule=ang)

    m.obj = Objective(expr=net.cost_expr(m.pg), sense=minimize)
    return m


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def fetch_case(case: str, data_dir: str, allow_fetch: bool) -> str:
    path = os.path.join(data_dir, f"pglib_opf_{case}.m")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not allow_fetch:
        raise FileNotFoundError(path)
    url = RAW_BASE + f"pglib_opf_{case}.m"
    with urllib.request.urlopen(url, timeout=300) as resp:
        blob = resp.read()
    with open(path, "wb") as f:
        f.write(blob)
    return path


def default_out_dir(form: str) -> str:
    """``<bench root>/grid_{ecf,polar}/nl``, matching every other suite."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    try:
        from bench_data import bench_data_root

        root = str(bench_data_root())
    except Exception:  # noqa: BLE001 - fall back to a local dir
        return os.path.join(here, "nl" if form == "ecf" else "nl_polar")
    return os.path.join(root, f"grid_{form}", "nl")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("cases", nargs="*",
                   help="pglib case names without the pglib_opf_ prefix "
                        "(default: the 20 cases of benchmarks/grid/)")
    p.add_argument("--form", choices=("ecf", "polar"), default="ecf",
                   help="formulation to emit (default: ecf)")
    p.add_argument("--thermal", choices=("current", "apparent"),
                   default="apparent",
                   help="branch limit convention, applied to BOTH forms; "
                        "apparent (default) reproduces the published pglib "
                        "objectives, current is the paper's eq (19)")
    p.add_argument("--no-angle-limits", action="store_true",
                   help="drop the branch angle-difference limits")
    p.add_argument("--out-dir", default=None,
                   help="output directory for .nl "
                        "(default: <bench root>/grid_<form>/nl)")
    p.add_argument("--data-dir", default=os.path.join(here, "data"),
                   help="cache directory for downloaded .m (default: ./data)")
    p.add_argument("--start", choices=("flat", "random"), default="flat",
                   help="starting point: flat (default) or a random physical "
                        "operating point shared by both forms")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for --start random (default: 0)")
    p.add_argument("--angle-spread", type=float, default=0.5,
                   help="half-width in radians of the random angle draw "
                        "(default: 0.5, comparable to pglib's +/-30 deg "
                        "branch angle limits)")
    p.add_argument("--no-fetch", action="store_true",
                   help="do not download; use .m already in --data-dir")
    p.add_argument("--list", action="store_true",
                   help="print the default case list and exit")
    args = p.parse_args(argv)

    if args.list:
        for c in DEFAULT_CASES:
            print(c)
        return 0

    if args.start == "random" and args.out_dir is None:
        # A random-start set must not silently overwrite the flat-start suite:
        # the .nl differ only in their initial values, and nothing downstream
        # would show which had been solved.
        raise SystemExit(
            "--start random requires an explicit --out-dir (it would "
            "otherwise overwrite the flat-start suite, which differs only in "
            "initial values and is indistinguishable downstream)"
        )
    out_dir = args.out_dir or default_out_dir(args.form)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    cases = args.cases or DEFAULT_CASES
    build = build_ecf if args.form == "ecf" else build_polar
    print(f"form={args.form}  thermal={args.thermal}  "
          f"angle_limits={not args.no_angle_limits}  start={args.start}"
          + (f" seed={args.seed} spread={args.angle_spread}"
             if args.start == "random" else ""))
    print(f"out  {out_dir}")
    print(f"{len(cases)} case(s) to process")

    ok, failed, written = 0, [], []
    for case in cases:
        try:
            path = fetch_case(case, args.data_dir, not args.no_fetch)
            with open(path, "r") as f:
                mpc = parse_matpower(f.read())
            net = Network(case, mpc)
            # Seeded per case, not per run, so a case's start does not depend
            # on how many cases preceded it in the list -- otherwise
            # `--start random case118_ieee` and `--start random` would give
            # case118_ieee different starts at the same seed.
            rng = None
            if args.start == "random":
                # zlib.crc32, not hash(): Python randomizes str hashing per
                # process unless PYTHONHASHSEED is pinned, so hash() here
                # would make --seed silently non-reproducible across runs.
                rng = np.random.default_rng(
                    zlib.crc32(f"{args.seed}:{case}".encode()) & 0xFFFFFFFF
                )
            model = build(net, args.thermal, not args.no_angle_limits,
                          net.physical_start(rng, args.angle_spread))
            out = os.path.join(out_dir, f"pglib_{case}.nl")
            model.write(out, format="nl",
                        io_options={"symbolic_solver_labels": True})
            nv = sum(len(v) for v in model.component_objects(Var))
            nc = sum(len(c) for c in model.component_objects(Constraint))
            print(f"  wrote pglib_{case}.nl  "
                  f"(buses={net.nb}, gens={len(net.gens)}, "
                  f"branches={len(net.branches)}, n={nv}, m={nc})")
            written.append({"case": case, "file": f"pglib_{case}.nl",
                            "buses": net.nb, "gens": len(net.gens),
                            "branches": len(net.branches), "n": nv, "m": nc})
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((case, repr(exc)))
            print(f"  FAILED {case}: {exc!r}", file=sys.stderr)

    # Provenance: the .nl carry no record of which conventions produced them,
    # and the two --thermal choices give different optima on the same case, so
    # a results file without this cannot be interpreted.
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({
            "generator": "benchmarks/grid_ecf/ecf_nl_export.py",
            "source": RAW_BASE,
            "form": args.form,
            "thermal": args.thermal,
            "angle_limits": not args.no_angle_limits,
            "start": args.start,
            "seed": args.seed if args.start == "random" else None,
            "angle_spread": (args.angle_spread
                             if args.start == "random" else None),
            "problems": written,
        }, f, indent=2)

    print(f"\n{ok}/{len(cases)} converted; {len(failed)} failed")
    for case, err in failed:
        print(f"  - {case}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
