#!/usr/bin/env python3
"""Generate a .nl file for the Mittelmann robot_a/b/c models.

The .mod source (plato.asu.edu/ftp/ampl-nlp-source/robot_a.mod) declares

    var X{1..N};
    var SUM{i in 1..M}  = sum{j in 1..N} VI [i,j]*X[j];
    var SUM1{i in 1..M} = sum{j in 1..N} VI1[i,j]*X[j];
    var SUM2{i in 1..M} = sum{j in 1..N} VI2[i,j]*X[j];
    minimize obj: sum{i in 1..N} BB[i]*X[i];
    s.t. gx1..gx18{i in 1..M}: ... >= 0

SUM/SUM1/SUM2 are AMPL *defined* variables, so they land in the .nl as
common subexpressions (V segments) rather than as extra columns; that is
why the published size is n = 1001 rather than 1001 + 3*4001.

Constraint families gx1/gx2, gx7/gx8, gx13/gx14 are all lower bounds on
the *same* linear form SUM[i]:

    C11*SUM[i] -+ V11[i] >= 0   <=>   SUM[i] >= +-V11[i]/C11
    C21*SUM[i] -+ V21[i] >= 0   <=>   SUM[i] >= +-V21[i]/C21
    C31*SUM[i] -+ V31[i] >= 0   <=>   SUM[i] >= +-V31[i]/C31

so they collapse to the single row SUM[i] >= max(|V11|/C11, |V21|/C21,
|V31|/C31).  With that merge the model has 13*M = 52013 rows, matching the
published m for robot_a/b/c.  `--no-merge` emits all 18*M = 72018 rows
instead (used to confirm which variant the benchmark measured).

Usage:  gen_robot_nl.py robot_a.mod out.nl [--no-merge]
"""

from __future__ import annotations

import math
import re
import sys


# ----------------------------------------------------------------- model data


def read_data(path):
    """Pull the `data;` section params out of a robot_[abc].mod."""
    text = open(path).read()
    data = {}
    for m in re.finditer(r"param\s+(\w+)\s*:=\s*([-\d.eE+]+)\s*;", text):
        data[m.group(1)] = float(m.group(2))
    return data


def ampl_round(x):
    """AMPL's round(): half away from zero (Python's round() is banker's)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


class Robot:
    def __init__(self, mod_path):
        d = read_data(mod_path)
        self.N = N = int(d["N"])
        self.M = M = int(d["M"])
        self.C = {k: d[k] for k in d if re.fullmatch(r"C\d\d", k)}
        self.H = H = 1.0 / (N - 1)
        self.TP = [None] + [(i - 1) / (M - 1) for i in range(1, M + 1)]
        self.K = [None] + [
            int(ampl_round(self.TP[i] / H + 1e-6 + 1)) for i in range(1, M + 1)
        ]
        self.BB = [None] + [
            0.5 * H
            if i == 1
            else 23 * H / 24
            if i == 2
            else 0.5 * H
            if i == N
            else 23 * H / 24
            if i == N - 1
            else H
            for i in range(1, N + 1)
        ]
        self._basis()
        self._vparams()

    def _basis(self):
        """VI/VI1/VI2 rows: at most four nonzeros, at n-index K-1..K+2."""
        H, N = self.H, self.N
        self.VI, self.VI1, self.VI2 = [None], [None], [None]
        for j in range(1, self.M + 1):
            K, TP = self.K[j], self.TP[j]
            t1 = TP - (K - 1) * H  # "TP - (K-1)*H"
            t2 = K * H - TP  # "K*H - TP"
            row, row1, row2 = {}, {}, {}
            for i in range(K - 1, K + 3):
                if i < 1 or i > N:
                    continue
                case = K - i + 3  # 4,3,2,1 for i = K-1,K,K+1,K+2
                if case == 1:
                    v = t1**3 / 6 / H**3
                    v1 = t1**2 / 2.0 / H**3
                    v2 = t1 / H**3
                elif case == 2:
                    v = 1.0 / 6.0 + t1 * 0.5 / H + t1**2 * 0.5 / H**2 - t1**3 * 0.5 / H**3
                    v1 = 0.5 / H + t1 / H**2 - t1**2 * 1.50 / H**3
                    v2 = 1.0 / H**2 - t1 * 3.0 / H**3
                elif case == 3:
                    v = 1.0 / 6.0 + t2 * 0.5 / H + t2**2 * 0.5 / H**2 - t2**3 * 0.5 / H**3
                    v1 = -0.5 / H - t2 / H**2 + t2**2 * 1.50 / H**3
                    v2 = 1.0 / H**2 - t2 * 3.0 / H**3
                else:  # case == 4
                    v = t2**3 / 6.0 / H**3
                    v1 = -(t2**2) / 2.0 / H**3
                    v2 = t2 / H**3
                if v != 0.0:
                    row[i] = v
                if v1 != 0.0:
                    row1[i] = v1
                if v2 != 0.0:
                    row2[i] = v2
            self.VI.append(row)
            self.VI1.append(row1)
            self.VI2.append(row2)

    def _vparams(self):
        """V11..V33: the tracking-trajectory derivative data, per row."""
        self.V = {k: [None] for k in ("11", "12", "13", "21", "22", "23", "31", "32", "33")}
        for i in range(1, self.M + 1):
            t = self.TP[i]
            A = 30 * t**2 * ((t - 2) * t + 1)
            B = 60 * t * ((2 * t - 3) * t + 1)
            Cc = (360 * t - 360) * t + 60
            g = 4.7 * t**3 * ((6 * t - 15) * t + 10)
            cg, sg = math.cos(g), math.sin(g)
            self.V["11"].append(1.5 * A)
            self.V["12"].append(1.5 * B)
            self.V["13"].append(1.5 * Cc)
            self.V["21"].append(-0.5 * (cg * 4.7 * A))
            self.V["22"].append(-0.5 * (-sg * (4.7 * A) ** 2 + cg * 4.7 * B))
            self.V["23"].append(
                -0.5
                * (
                    -cg * (4.7 * A) ** 3
                    - sg * 3 * 4.7**2 * A * B
                    + cg * Cc * 4.7
                )
            )
            self.V["31"].append(-1.3 * A)
            self.V["32"].append(-1.3 * B)
            self.V["33"].append(-1.3 * Cc)


# ------------------------------------------------------------- .nl generation


def num(x):
    return repr(float(x))


def gen(mod_path, out_path, merge=True, x0=1.491400623321533):
    r = Robot(mod_path)
    N, M = r.N, r.M
    n = N  # columns: X[1..N] -> 0-based 0..N-1

    # CSE numbering: SUM[i] -> n+(i-1), SUM1[i] -> n+M+(i-1), SUM2 -> n+2M+(i-1)
    def cse_S(i):
        return n + (i - 1)

    def cse_S1(i):
        return n + M + (i - 1)

    def cse_S2(i):
        return n + 2 * M + (i - 1)

    groups = [("1", "11", "12", "13"), ("2", "21", "22", "23"), ("3", "31", "32", "33")]

    # ---- constraint list -------------------------------------------------
    # rows: (kind, group, i).  Nonlinear rows must precede linear rows.
    rows = []
    for g, _, _, _ in groups:
        for kind in ("cube_minus", "cube_plus", "quint_minus", "quint_plus"):
            for i in range(1, M + 1):
                rows.append((kind, g, i))
    if merge:
        for i in range(1, M + 1):
            rows.append(("lin_merged", None, i))
    else:
        # unmerged: six linear rows per i, in declaration order gx1,2,7,8,13,14
        for g, _, _, _ in groups:
            for sgn in ("-", "+"):
                for i in range(1, M + 1):
                    rows.append(("lin", (g, sgn), i))
    n_nl_rows = 12 * M
    m = len(rows)

    def support(kind, i):
        """Columns the row touches (0-based), i.e. union of referenced CSEs."""
        if kind in ("lin", "lin_merged"):
            s = set(r.VI[i])
        elif kind in ("cube_minus", "cube_plus"):
            s = set(r.VI[i]) | set(r.VI1[i])
        else:
            s = set(r.VI[i]) | set(r.VI1[i]) | set(r.VI2[i])
        return sorted(j - 1 for j in s)

    # ---- Jacobian: linear coefficients + counts --------------------------
    jac = []  # per row: list of (col0, coef)
    colcount = [0] * n
    for kind, g, i in rows:
        cols = support(kind, i)
        if kind == "lin_merged":
            coefs = {j - 1: v for j, v in r.VI[i].items()}
        elif kind == "lin":
            grp, sgn = g
            c = r.C["C" + grp + "1"]
            coefs = {j - 1: c * v for j, v in r.VI[i].items()}
        elif kind in ("cube_minus", "cube_plus"):
            # nonlinear part is C_a2*S^3; the -+(V_a2*S - V_a1*S1) part is linear
            V1 = r.V[g + "1"][i]
            V2 = r.V[g + "2"][i]
            s = -1.0 if kind == "cube_minus" else 1.0
            coefs = {}
            for j, v in r.VI[i].items():
                coefs[j - 1] = coefs.get(j - 1, 0.0) + s * V2 * v
            for j, v in r.VI1[i].items():
                coefs[j - 1] = coefs.get(j - 1, 0.0) - s * V1 * v
        else:
            coefs = {}  # purely nonlinear row
        entries = [(j, coefs.get(j, 0.0)) for j in cols]
        jac.append(entries)
        for j, _ in entries:
            colcount[j] += 1
    nzc = sum(len(e) for e in jac)

    out = []
    w = out.append

    # ---- header ----------------------------------------------------------
    w(f"g3 1 1 0\t# problem robot")
    w(f" {n} {m} 1 0 0 0\t# vars, constraints, objectives, ranges, eqns, lcons")
    w(f" {n_nl_rows} 0\t# nonlinear constraints, objectives")
    w(" 0 0\t# network constraints: nonlinear, linear")
    w(f" {n} 0 0\t# nonlinear vars in constraints, objectives, both")
    w(" 0 0 0 1\t# linear network variables; functions; arith, flags")
    w(" 0 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)")
    w(f" {nzc} {n}\t# nonzeros in Jacobian, obj. gradient")
    w(" 0 0\t# max name lengths: constraints, variables")
    w(f" 0 {3 * M} 0 0 0\t# common exprs: b,c,o,c1,o1")

    # ---- V segments (defined variables) ----------------------------------
    for rowdata, base in ((r.VI, 0), (r.VI1, M), (r.VI2, 2 * M)):
        for i in range(1, M + 1):
            ent = sorted(rowdata[i].items())
            w(f"V{n + base + (i - 1)} {len(ent)} 0")
            for j, v in ent:
                w(f"{j - 1} {num(v)}")
            w("n0")

    # ---- C segments ------------------------------------------------------
    def emit_cube(g, i):
        """C_a2 * SUM[i]^3"""
        c2 = r.C["C" + g + "2"]
        w("o2")
        w(f"n{num(c2)}")
        w("o5")
        w(f"v{cse_S(i)}")
        w("n3")

    def emit_quint(g, i, plus):
        """C_a3*S^5 -+ (V_a3*S^2 - 3 V_a2 S S1 + 3 V_a1 S1^2 + V_a1 S S2)"""
        c3 = r.C["C" + g + "3"]
        V1, V2, V3 = r.V[g + "1"][i], r.V[g + "2"][i], r.V[g + "3"][i]
        w("o0" if plus else "o1")
        w("o2")
        w(f"n{num(c3)}")
        w("o5")
        w(f"v{cse_S(i)}")
        w("n5")
        w("o54")
        w("4")
        # V3*S^2
        w("o2")
        w(f"n{num(V3)}")
        w("o5")
        w(f"v{cse_S(i)}")
        w("n2")
        # -3*V2*S*S1
        w("o2")
        w(f"n{num(-3.0 * V2)}")
        w("o2")
        w(f"v{cse_S(i)}")
        w(f"v{cse_S1(i)}")
        # 3*V1*S1^2
        w("o2")
        w(f"n{num(3.0 * V1)}")
        w("o5")
        w(f"v{cse_S1(i)}")
        w("n2")
        # V1*S*S2
        w("o2")
        w(f"n{num(V1)}")
        w("o2")
        w(f"v{cse_S(i)}")
        w(f"v{cse_S2(i)}")

    for ridx, (kind, g, i) in enumerate(rows):
        w(f"C{ridx}")
        if kind == "cube_minus" or kind == "cube_plus":
            emit_cube(g, i)
        elif kind == "quint_minus":
            emit_quint(g, i, plus=False)
        elif kind == "quint_plus":
            emit_quint(g, i, plus=True)
        else:
            w("n0")

    # ---- objective (linear) ---------------------------------------------
    w("O0 0")
    w("n0")

    # ---- initial point ---------------------------------------------------
    w(f"x{n}")
    xs = num(x0)
    for j in range(n):
        w(f"{j} {xs}")

    # ---- constraint bounds ----------------------------------------------
    w(f"r")
    for kind, g, i in rows:
        if kind == "lin_merged":
            lo = max(
                abs(r.V["11"][i]) / r.C["C11"],
                abs(r.V["21"][i]) / r.C["C21"],
                abs(r.V["31"][i]) / r.C["C31"],
            )
            w(f"2 {num(lo)}")
        elif kind == "lin":
            grp, sgn = g
            v = r.V[grp + "1"][i]
            w(f"2 {num(v if sgn == '-' else -v)}")
        else:
            w("2 0")

    # ---- variable bounds (all free) --------------------------------------
    w("b")
    for _ in range(n):
        w("3")

    # ---- k segment (cumulative column counts) ----------------------------
    w(f"k{n - 1}")
    acc = 0
    for j in range(n - 1):
        acc += colcount[j]
        w(str(acc))

    # ---- J segments ------------------------------------------------------
    for ridx, entries in enumerate(jac):
        w(f"J{ridx} {len(entries)}")
        for j, c in entries:
            w(f"{j} {num(c)}")

    # ---- G segment -------------------------------------------------------
    w(f"G0 {n}")
    for j in range(n):
        w(f"{j} {num(r.BB[j + 1])}")

    with open(out_path, "w") as f:
        f.write("\n".join(out))
        f.write("\n")
    sys.stderr.write(
        f"{out_path}: n={n} m={m} (nonlinear rows {n_nl_rows}) nzJ={nzc} cse={3 * M}\n"
    )


if __name__ == "__main__":
    merge = "--no-merge" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen(args[0], args[1], merge=merge)
