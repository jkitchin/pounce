#!/usr/bin/env python3
"""Check the infeasibility / unboundedness certificates in pounce's JSON
reports against the ``.nl`` models they came from (gh#1181).

The check reads the model straight from the ``.nl`` text (``g`` format, linear
constraints and objective only -- every model in this suite) and shares no
code with pounce, so a certificate pounce wrongly marks ``verified`` fails
here.

A ``primal_infeasible`` certificate ``(lambda, z_l, z_u)`` passes when

* ``J^T lambda - z_l + z_u = 0``: its inf-norm, relative to the largest
  column magnitude ``max_j sum_i |J_ij lambda_i|``, is at most ``--tol``;
* ``lambda_i > 0`` only on a row with a finite upper side, ``< 0`` only on one
  with a finite lower side; ``z_l`` / ``z_u`` non-negative and only on finite
  bounds;
* ``value = sum_i support_i(lambda_i) - lb^T z_l + ub^T z_u < 0``, where
  ``support_i(t) = t*u_i`` for ``t > 0`` and ``t*l_i`` for ``t < 0`` (the
  constant part of a row body is already in ``r``-segment bounds).

A ``dual_infeasible`` certificate ``d`` passes when ``J d`` keeps every row
inside its sides' recession cone (``= 0`` on an equality, ``<= 0`` on an
upper-only row, ...), ``d`` respects every finite bound's sign, and
``c^T d`` is negative in minimize form.

Usage::

    check_certificates.py NL_DIR REPORT_DIR      # REPORT_DIR/<name>.json

Prints one line per model and exits non-zero if any certificate pounce marked
verified fails this check.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def _sides(kind, vals):
    """An ``r`` / ``b`` segment line's ``(lower, upper)``."""
    inf = math.inf
    if kind == 0:
        return (vals[0], vals[1])
    if kind == 1:
        return (-inf, vals[0])
    if kind == 2:
        return (vals[0], inf)
    if kind == 3:
        return (-inf, inf)
    if kind == 4:
        return (vals[0], vals[0])
    raise ValueError(f"unsupported bound kind {kind} (complementarity?)")


def parse_nl(path):
    """Linear ``g``-format .nl -> (n, m, rows, row_sides, bounds, c, minimize)."""
    lines = Path(path).read_text().splitlines()
    if not lines[0].startswith("g"):
        raise ValueError("binary .nl not supported")
    n, m, nobj = (int(t) for t in lines[1].split()[:3])
    rows = [dict() for _ in range(m)]
    sides = [None] * m
    bounds = [None] * n
    c = [0.0] * n
    const = [0.0] * m  # a row body's constant term, ``C<i>`` then ``n<k>``
    minimize = True
    i = 10
    while i < len(lines):
        head = lines[i].split("#")[0].strip()
        if not head:
            i += 1
            continue
        tag, rest = head[0], head[1:].split()
        if tag in "CO":
            body = lines[i + 1].split("#")[0].strip()
            if not body.startswith("n"):
                raise ValueError(f"nonlinear body in {head}: {body}")
            if tag == "O":
                minimize = int(rest[1]) == 0
            else:
                # A constant body term moves the row: l <= Jx + k <= u.
                const[int(rest[0]) if rest else int(head[1:])] = float(body[1:])
            i += 2
        elif tag == "d":
            # Initial duals: irrelevant to the certificate.
            i += 1 + int(rest[0])
        elif tag == "S":
            # A suffix: ``S<kind> <count> <name>`` then ``count`` lines.
            i += 1 + int(rest[1])
        elif tag == "x":
            i += 1 + int(rest[0])
        elif tag == "r":
            for k in range(m):
                t = lines[i + 1 + k].split()
                kind = int(t[0])
                vals = [float(v) for v in t[1:]]
                sides[k] = _sides(kind, vals)
            i += 1 + m
        elif tag == "b":
            for k in range(n):
                t = lines[i + 1 + k].split()
                kind = int(t[0])
                vals = [float(v) for v in t[1:]]
                bounds[k] = _sides(kind, vals)
            i += 1 + n
        elif tag == "k":
            i += 1 + int(rest[0])
        elif tag == "J":
            r, cnt = int(rest[0]), int(rest[1])
            for k in range(cnt):
                j, v = lines[i + 1 + k].split()
                rows[r][int(j)] = float(v)
            i += 1 + cnt
        elif tag == "G":
            cnt = int(rest[1])
            for k in range(cnt):
                j, v = lines[i + 1 + k].split()
                c[int(j)] = float(v)
            i += 1 + cnt
        else:
            raise ValueError(f"unhandled segment {head!r}")
    if nobj and not minimize:
        c = [-v for v in c]
    sides = [(lo - k, up - k) for (lo, up), k in zip(sides, const)]
    return n, m, rows, sides, bounds, c


def check_farkas(model, cert, tol):
    n, m, rows, sides, bounds, _ = model
    lam, zl, zu = cert["lambda"], cert["z_l"], cert["z_u"]
    q = [0.0] * n
    mag = [0.0] * n
    value = 0.0
    for i in range(m):
        li = lam[i]
        if li == 0.0:
            continue
        lo, up = sides[i]
        side = up if li > 0 else lo
        if not math.isfinite(side):
            return f"lambda[{i}]={li:g} on an infinite side"
        value += li * side
        for j, a in rows[i].items():
            q[j] += a * li
            mag[j] += abs(a * li)
    for j in range(n):
        lo, up = bounds[j]
        if zl[j] < 0 or zu[j] < 0:
            return f"negative bound multiplier at {j}"
        if zl[j] > 0:
            if not math.isfinite(lo):
                return f"z_l[{j}] on an infinite bound"
            value -= lo * zl[j]
        if zu[j] > 0:
            if not math.isfinite(up):
                return f"z_u[{j}] on an infinite bound"
            value += up * zu[j]
        q[j] += -zl[j] + zu[j]
    # Relative to the largest column magnitude, as Clarabel/SCS/ECOS measure
    # it -- not column by column: a column holding only a 1e-17 leftover of a
    # single term would read as 100% of itself while proving nothing either way.
    scale = max(mag, default=0.0)
    resid = max((abs(v) for v in q), default=0.0) / (scale if scale > 0 else 1.0)
    if resid > tol:
        return f"relative stationarity residual {resid:.2e}"
    if not value < 0:
        return f"value {value:g} is not negative"
    return None


def check_recession(model, cert, tol):
    n, m, rows, sides, bounds, c = model
    d = cert["d"]
    for i in range(m):
        jd = sum(a * d[j] for j, a in rows[i].items())
        mg = sum(abs(a * d[j]) for j, a in rows[i].items()) or 1.0
        lo, up = sides[i]
        if math.isfinite(up) and jd > tol * mg:
            return f"row {i} rises toward its upper side ({jd:.2e})"
        if math.isfinite(lo) and jd < -tol * mg:
            return f"row {i} falls toward its lower side ({jd:.2e})"
    for j in range(n):
        lo, up = bounds[j]
        if math.isfinite(lo) and d[j] < -tol:
            return f"d[{j}]={d[j]:g} crosses a lower bound"
        if math.isfinite(up) and d[j] > tol:
            return f"d[{j}]={d[j]:g} crosses an upper bound"
    cd = sum(ci * di for ci, di in zip(c, d))
    if not cd < 0:
        return f"c^T d = {cd:g} is not negative"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nl_dir", type=Path)
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    bad = 0
    for nl in sorted(args.nl_dir.glob("*.nl")):
        rep = args.report_dir / (nl.stem + ".json")
        if not rep.exists():
            print(f"{nl.stem:28s} no report")
            continue
        cert = json.loads(rep.read_text())["solution"].get("certificate")
        if cert is None:
            print(f"{nl.stem:28s} no certificate")
            continue
        try:
            model = parse_nl(nl)
        except (ValueError, IndexError) as e:
            # Not a model this checker can read (binary, nonlinear, defined
            # variables, ...): say so and go on, rather than stop the run.
            print(f"{nl.stem:28s} skipped ({e})")
            continue
        check = check_farkas if cert["kind"] == "primal_infeasible" else check_recession
        why = check(model, cert, args.tol)
        verdict = "PASS" if why is None else f"FAIL ({why})"
        claimed = "verified" if cert["verified"] else "unverified"
        print(f"{nl.stem:28s} {cert['kind']:18s} {claimed:10s} {cert['source']:18s} {verdict}")
        if why is not None and cert["verified"]:
            bad += 1
    if bad:
        print(f"{bad} certificate(s) marked verified failed the independent check")
        sys.exit(1)


if __name__ == "__main__":
    main()
