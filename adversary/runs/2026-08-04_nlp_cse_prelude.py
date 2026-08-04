"""Adversary cross-check: shared-CSE (.nl `V` segment) constraint evaluation.

Family: nlp (via .nl / CLI)   Class: defined variables / common subexpressions
Target: PR #480 — `eval_g`/`eval_f` now evaluate through a problem-wide
        `HybridTape` with a shared CSE prelude (pounce-nl/src/nl_reader.rs
        `con_hybrid`, nl_tape.rs `hybrid_supported`).
Target: PR #481 — `inf_pr` now reports `curr_unscaled_nlp_constraint_violation_max`.

Oracle strategy — triangulation, because a hand-written `.nl` is at least as
likely to be wrong as the solver:

  1. `ipopt` 3.14.19 reads the *identical* `.nl`. Independent implementation.
  2. Every model is emitted TWICE — once with `V` segments (hybrid path) and
     once with the CSE bodies inlined (flat path), mathematically identical.
     pounce must agree with itself across the two encodings.
  3. `pounce verify` re-checks the claimed solution against the canonical `.nl`
     without trusting the solver that produced it.

A disagreement in (2) alone indicts the hybrid path. A disagreement in (1) for
*both* encodings indicts this script's `.nl`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"


# --------------------------------------------------------------- .nl emitter


class Nl:
    """Minimal `.nl` writer for tiny hand-built models.

    `cses` are (linear_terms, nonlinear_prefix_tokens) pairs; `cons` are
    (nonlinear_prefix_tokens, {var: linear_coef}, lower, upper) tuples with
    `None` for an absent bound.
    """

    def __init__(self, n, obj_lin, obj_nl=None):
        self.n = n
        self.cses: list[tuple[dict[int, float], list[str]]] = []
        self.cons: list[tuple[list[str], dict[int, float], float | None, float | None]] = []
        self.obj_lin = obj_lin
        self.obj_nl = obj_nl or ["n0"]
        self.x0 = [1.0] * n

    def cse(self, lin, nl=None):
        self.cses.append((lin, nl or ["n0"]))
        return self.n + len(self.cses) - 1

    def con(self, nl, lin, lo=None, hi=None):
        self.cons.append((nl, lin, lo, hi))

    def write(self, path):
        n, m = self.n, len(self.cons)
        nl_rows = sum(1 for c in self.cons if c[0] != ["n0"])
        # nonlinear constraints must come first
        order = [i for i, c in enumerate(self.cons) if c[0] != ["n0"]]
        order += [i for i, c in enumerate(self.cons) if c[0] == ["n0"]]
        cons = [self.cons[i] for i in order]

        # Jacobian pattern: every variable the row touches, linear or not.
        def row_vars(nlt, lin):
            vs = set(lin)
            for t in nlt:
                if t.startswith("v"):
                    k = int(t[1:])
                    if k < n:
                        vs.add(k)
                    else:  # CSE reference: pull in its own support
                        vs |= self._cse_support(k - n)
            return sorted(vs)

        jac = [(row_vars(c[0], c[1]), c[1]) for c in cons]
        nzc = sum(len(v) for v, _ in jac)
        colcount = [0] * n
        for vs, _ in jac:
            for j in vs:
                colcount[j] += 1

        n_rng = sum(1 for _, _, lo, hi in cons
                    if lo is not None and hi is not None and lo != hi)
        n_eqn = sum(1 for _, _, lo, hi in cons
                    if lo is not None and hi is not None and lo == hi)

        L = []
        w = L.append
        w("g3 1 1 0")
        w(f" {n} {m} 1 {n_rng} {n_eqn}")
        w(f" {nl_rows} {'1' if self.obj_nl != ['n0'] else '0'}")
        w(" 0 0")
        w(f" {n} {n if self.obj_nl != ['n0'] else 0} {n if self.obj_nl != ['n0'] else 0}")
        w(" 0 0 0 1")
        w(" 0 0 0 0 0")
        w(f" {nzc} {len(self.obj_lin)}")
        w(" 0 0")
        w(f" 0 {len(self.cses)} 0 0 0")
        for k, (lin, nlt) in enumerate(self.cses):
            w(f"V{n + k} {len(lin)} 0")
            for v, c in sorted(lin.items()):
                w(f"{v} {c!r}")
            L.extend(nlt)
        for i, (nlt, _, _, _) in enumerate(cons):
            w(f"C{i}")
            L.extend(nlt)
        w("O0 0")
        L.extend(self.obj_nl)
        w(f"x{n}")
        for j in range(n):
            w(f"{j} {self.x0[j]!r}")
        w("r")
        for _, _, lo, hi in cons:
            if lo is not None and hi is not None:
                w(f"0 {lo!r} {hi!r}" if lo != hi else f"4 {lo!r}")
            elif lo is not None:
                w(f"2 {lo!r}")
            elif hi is not None:
                w(f"1 {hi!r}")
            else:
                w("3")
        w("b")
        for _ in range(n):
            w("3")
        w(f"k{n - 1}")
        acc = 0
        for j in range(n - 1):
            acc += colcount[j]
            w(str(acc))
        for i, (vs, lin) in enumerate(jac):
            w(f"J{i} {len(vs)}")
            for j in vs:
                w(f"{j} {float(lin.get(j, 0.0))!r}")
        w(f"G0 {len(self.obj_lin)}")
        for j, c in sorted(self.obj_lin.items()):
            w(f"{j} {float(c)!r}")
        Path(path).write_text("\n".join(L) + "\n")

    def _cse_support(self, k):
        lin, nlt = self.cses[k]
        vs = set(lin)
        for t in nlt:
            if t.startswith("v"):
                q = int(t[1:])
                vs |= {q} if q < self.n else self._cse_support(q - self.n)
        return vs


# ------------------------------------------------------------------- runners


def run_pounce(nlpath, extra=()):
    sol = str(Path(nlpath).with_suffix(".sol"))
    js = str(Path(nlpath).with_suffix(".json"))
    p = subprocess.run(
        [POUNCE, nlpath, sol, "--json-output", js, *extra],
        capture_output=True, text=True, timeout=120,
    )
    keep = str(Path(nlpath).with_suffix("")) + ".pounce.sol"
    try:
        Path(keep).write_bytes(Path(sol).read_bytes())
    except FileNotFoundError:
        keep = sol
    out = {"stdout": p.stdout, "rc": p.returncode, "sol": keep}
    try:
        d = json.loads(Path(js).read_text())
        out["obj"] = d["solution"]["objective"]
        out["status"] = d["solution"]["status"]
        out["x"] = d["solution"]["x"]
        out["iters"] = d["statistics"]["iteration_count"]
    except Exception:
        out["obj"] = None
        out["status"] = f"NO_JSON(rc={p.returncode})"
    infpr = [ln.split()[2] for ln in p.stdout.splitlines()
             if ln[:4].strip().isdigit() and len(ln.split()) > 9]
    out["inf_pr"] = infpr
    return out


def run_ipopt(nlpath):
    stub = str(Path(nlpath).with_suffix(""))
    optf = Path(nlpath).parent / "ipopt.opt"
    optf.write_text("print_level 5\n")
    p = subprocess.run([IPOPT, stub, "-AMPL"], capture_output=True, text=True,
                       cwd=str(Path(nlpath).parent), timeout=120)
    obj = None
    for ln in p.stdout.splitlines():
        if ln.startswith("Objective..............."):
            obj = float(ln.split()[-1])
    infpr = [ln.split()[2] for ln in p.stdout.splitlines()
             if ln[:4].strip().isdigit() and len(ln.split()) > 9]
    ok = "EXIT: Optimal Solution Found." in p.stdout
    return {"obj": obj, "ok": ok, "stdout": p.stdout, "inf_pr": infpr}


def verify(nlpath, sol):
    p = subprocess.run([POUNCE, "verify", nlpath, sol], capture_output=True,
                       text=True, timeout=120)
    return p.returncode, p.stdout


# -------------------------------------------------------------------- models
# Each builder returns (cse_model, inlined_model) — identical mathematics,
# one using `V` segments and one with the bodies written out at each use.


def m_nested_cse():
    """V4 = (x0+x1)^2 is itself built from V3 = x0+x1, and both are shared.

    If the prelude is not emitted in topological order, or a promoted body is
    evaluated before its operand, this silently produces garbage.
        min x0^2+x1^2+x2^2  s.t.  V4 + x2^2 >= 1,  V4 - x2 >= 0,  V3 >= 0.5
    """
    a = Nl(3, {0: 0.0, 1: 0.0, 2: 0.0},
           ["o54", "3", "o5", "v0", "n2", "o5", "v1", "n2", "o5", "v2", "n2"])
    v3 = a.cse({0: 1.0, 1: 1.0})
    v4 = a.cse({}, ["o5", f"v{v3}", "n2"])
    a.con(["o0", f"v{v4}", "o5", "v2", "n2"], {}, lo=1.0)
    a.con(["o1", f"v{v4}", "v2"], {}, lo=0.0)
    a.con(["n0"], {0: 1.0, 1: 1.0}, lo=0.5)
    a.x0 = [0.4, 0.4, 0.3]

    b = Nl(3, {0: 0.0, 1: 0.0, 2: 0.0},
           ["o54", "3", "o5", "v0", "n2", "o5", "v1", "n2", "o5", "v2", "n2"])
    sq = ["o5", "o0", "v0", "v1", "n2"]          # (x0+x1)^2 inlined
    b.con(["o0", *sq, "o5", "v2", "n2"], {}, lo=1.0)
    b.con(["o1", *sq, "v2"], {}, lo=0.0)
    b.con(["n0"], {0: 1.0, 1: 1.0}, lo=0.5)
    b.x0 = [0.4, 0.4, 0.3]
    return a, b


def m_promotion_boundary():
    """A CSE referenced by exactly TWO summands (promoted, >=2) against the
    same mathematics with it referenced ONCE (inlined). The >=2 threshold is
    the branch that decides prelude vs inline; both must agree."""
    a = Nl(3, {2: 1.0})
    v3 = a.cse({0: 1.0, 1: 2.0})
    a.con(["o5", f"v{v3}", "n2"], {}, lo=1.0)     # ref 1
    a.con(["o5", f"v{v3}", "n3"], {}, lo=-8.0)    # ref 2 -> promoted
    a.con(["n0"], {2: 1.0}, lo=0.25)
    a.x0 = [0.6, 0.6, 0.6]

    b = Nl(3, {2: 1.0})
    v3b = b.cse({0: 1.0, 1: 2.0})
    body = ["o0", "v0", "o2", "n2.0", "v1"]       # x0 + 2*x1 inlined
    b.con(["o5", f"v{v3b}", "n2"], {}, lo=1.0)    # ref 1 only -> inlined
    b.con(["o5", *body, "n3"], {}, lo=-8.0)
    b.con(["n0"], {2: 1.0}, lo=0.25)
    b.x0 = [0.6, 0.6, 0.6]
    return a, b


def m_minlist_in_shared_cse():
    """An opcode the hybrid path REJECTS (o11 min-list), buried inside a CSE
    body that is shared by two constraints. `hybrid_supported` must descend
    through the `Cse` payload and disable the path for the whole block;
    otherwise `build_multi` panics and the solve dies."""
    a = Nl(3, {2: 1.0})
    v3 = a.cse({}, ["o11", "2", "v0", "v1"])      # min(x0, x1)
    a.con(["o5", f"v{v3}", "n2"], {}, lo=0.25)
    a.con(["o2", f"v{v3}", "v2"], {}, lo=0.1)
    a.con(["n0"], {2: 1.0}, lo=0.5, hi=3.0)
    a.x0 = [0.9, 1.1, 0.8]
    return a, None


def m_domain_error_in_shared_cse():
    """A domain error (log of a negative) reached THROUGH a shared CSE.

    `V3 = x0 - 2` is referenced by two constraints as `log(V3)`, and x0 starts
    at 1.0, so the very first `eval_g` produces NaN. The flat tape and the
    prelude must agree on that — if the hybrid path swallowed or reordered the
    NaN the solver would take a different branch (recover vs abort) than it
    did before PR #480.
    """
    a = Nl(3, {2: 1.0})
    v3 = a.cse({0: 1.0}, ["n-2.0"])
    a.con(["o43", f"v{v3}"], {}, lo=0.0)                  # log(V3) >= 0
    a.con(["o2", "n2.0", "o43", f"v{v3}"], {}, lo=-1.0)   # 2*log(V3) >= -1
    a.con(["n0"], {2: 1.0}, lo=0.25)
    a.x0 = [1.0, 0.5, 0.5]

    b = Nl(3, {2: 1.0})
    body = ["o0", "v0", "n-2.0"]
    b.con(["o43", *body], {}, lo=0.0)
    b.con(["o2", "n2.0", "o43", *body], {}, lo=-1.0)
    b.con(["n0"], {2: 1.0}, lo=0.25)
    b.x0 = [1.0, 0.5, 0.5]
    return a, b


def m_range_and_equality():
    """PR #481's new loop: a RANGE row (both bounds) and an EQUALITY row, so
    the `has_l`/`has_u` branches and the c-block both carry weight."""
    a = Nl(3, {2: 1.0})
    v3 = a.cse({0: 1.0, 1: 1.0})
    a.con(["o5", f"v{v3}", "n2"], {}, lo=1.0, hi=4.0)   # range on a CSE
    a.con(["o2", f"v{v3}", "v2"], {}, lo=0.5, hi=0.5)   # equality
    a.con(["n0"], {2: 1.0}, lo=0.2)
    a.x0 = [0.8, 0.8, 0.7]
    return a, None


# ---------------------------------------------------------------------- main


# Analytic optima. Each objective is a single variable pinned by its own
# lower bound, with the remaining rows satisfiable there:
#   nested_cse          min x0^2+x1^2+x2^2 with x0+x1 >= 0.5 and (x0+x1)^2 >= ...
#                       -> the sum-of-squares minimum on x0+x1 = 0.5 at
#                          x0=x1=0.25, x2=0 gives 0.5 once (x0+x1)^2+x2^2 >= 1
#                          is inactive... verified against a converged ipopt.
#   promotion_boundary  min x2 s.t. x2 >= 0.25                       -> 0.25
#   minlist             min x2 s.t. 0.5 <= x2 <= 3, min(x0,x1) = 0.5 -> 0.5
#   range_and_equality  min x2 s.t. (x0+x1)x2 = 0.5, 1 <= (x0+x1)^2 <= 4,
#                       x2 >= 0.2; largest feasible t = x0+x1 = 2 -> x2 = 0.25
ABORT_EXPECTED = {"domain_error_in_shared_cse"}

KNOWN = {
    "nested_cse": 0.49999999749608903,   # converged ipopt on the same .nl
    "promotion_boundary": 0.25,
    "minlist_in_shared_cse": 0.5,
    "range_and_equality": 0.25,
    # log(x0-2) >= 0 => x0 >= 3; objective is x2 pinned at its bound 0.25.
    "domain_error_in_shared_cse": 0.25,
}


def rel(a, b):
    if a is None or b is None:
        return float("inf")
    return abs(a - b) / max(1.0, abs(b))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="adv-cse-"))
    cases = [
        ("nested_cse", m_nested_cse),
        ("promotion_boundary", m_promotion_boundary),
        ("minlist_in_shared_cse", m_minlist_in_shared_cse),
        ("range_and_equality", m_range_and_equality),
        ("domain_error_in_shared_cse", m_domain_error_in_shared_cse),
    ]
    rows = []
    for name, build in cases:
        a, b = build()
        pa = tmp / f"{name}_cse.nl"
        a.write(pa)
        ra = run_pounce(str(pa))
        ia = run_ipopt(str(pa))
        vrc, vout = verify(str(pa), ra["sol"]) if ra.get("obj") is not None else (None, "")

        rb = rib = None
        if b is not None:
            pb = tmp / f"{name}_inlined.nl"
            b.write(pb)
            rb = run_pounce(str(pb))
            rib = run_ipopt(str(pb))

        print(f"\n=== {name} ===")
        print(f"  pounce(CSE)     status={ra['status']:<24} obj={ra['obj']}")
        print(f"  ipopt (CSE)     ok={ia['ok']:<5}                    obj={ia['obj']}")
        if rb is not None:
            print(f"  pounce(inlined) status={rb['status']:<24} obj={rb['obj']}")
            print(f"  ipopt (inlined) ok={rib['ok']:<5}                    obj={rib['obj']}")
        if vrc is not None:
            print(f"  pounce verify   rc={vrc} "
                  f"({'VERIFIED' if vrc == 0 else 'REJECTED'})")
        # inf_pr parity (PR #481) over the shared iterations
        k = min(len(ra["inf_pr"]), len(ia["inf_pr"]))

        def close(u, v):
            fu, fv = float(u), float(v)
            return abs(fu - fv) <= 1e-6 * max(1.0, abs(fu), abs(fv))

        mism = [i for i in range(k) if not close(ra["inf_pr"][i], ia["inf_pr"][i])]
        print(f"  inf_pr parity vs ipopt: {k - len(mism)}/{k} identical"
              + (f"  first mismatch @iter {mism[0]}: "
                 f"pounce={ra['inf_pr'][mism[0]]} ipopt={ia['inf_pr'][mism[0]]}"
                 if mism else ""))

        known = KNOWN[name]
        if name in ABORT_EXPECTED:
            # NaN at x0 by construction: the contract under test is that the
            # hybrid and flat paths abort *identically*, not that either solves.
            checks = {
                "aborts_on_nan": "InvalidNumber" in str(ra["status"]),
                "cse_and_inlined_abort_alike": rb is not None
                and ra["status"] == rb["status"],
            }
            for cname, ok in checks.items():
                print(f"    [{'PASS' if ok else 'FAIL'}] {cname}")
            rows.append((name, checks))
            continue
        checks = {
            "pounce_solved": ra["obj"] is not None and "Succeeded" in str(ra["status"]),
            "matches_known_optimum": rel(ra["obj"], known) < 1e-6,
            "verify_ok": vrc == 0,
        }
        if ia["ok"]:
            checks["matches_ipopt"] = rel(ra["obj"], ia["obj"]) < 1e-6
            # inf_pr parity is only meaningful when the two solvers walk the
            # same trajectory; when ipopt diverges the columns describe
            # different iterates.
            checks["inf_pr_matches_ipopt"] = len(mism) == 0 and k > 0
        else:
            print(f"  NOTE: oracle (ipopt) did not converge on this model "
                  f"(obj={ia['obj']}); objective/inf_pr parity vs ipopt is "
                  f"not meaningful here — scored against the analytic optimum "
                  f"{known} instead.")
            checks["inf_pr_at_x0_matches_ipopt"] = (
                k > 0 and close(ra["inf_pr"][0], ia["inf_pr"][0]))
        if rb is not None:
            checks["cse_vs_inlined_agree"] = rel(ra["obj"], rb["obj"]) < 1e-8
        rows.append((name, checks))
        for cname, ok in checks.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {cname}")

    print("\n=== SUMMARY ===")
    allok = True
    for name, checks in rows:
        bad = [c for c, ok in checks.items() if not ok]
        allok &= not bad
        print(f"  {name:<24} {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")
    print(f"\nartifacts: {tmp}")
    print("VERDICT: PASS" if allok else "VERDICT: FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
