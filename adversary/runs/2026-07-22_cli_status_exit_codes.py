"""Adversary cross-check: CLI / .nl status-reporting surface + `pounce verify` oracle.

Family: nlp (CLI / .nl surface)   Class: infeasibility / unboundedness / status reporting
Source: AMPL solve_result_num convention (AMPL Book, `solve_result_num` ranges:
        0-99 solved, 100-199 solved?, 200-299 infeasible, 300-399 unbounded,
        400-499 limit, 500-599 failure). Ipopt ApplicationReturnStatus as the
        independent oracle on identical .nl files.

Sub-tests:
 (a) CLI exit codes discriminate optimal / infeasible / unbounded / iter-limit?
 (b) .sol solve_result_num correct per status?
 (c) `pounce verify` REJECTS a corrupted .sol (constraint-violating perturbation)?
 (d) `pounce verify` ACCEPTS the genuine .sol?
 (e) library API status strings agree with the CLI statuses?
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = os.path.join(tempfile.gettempdir(), "adv_cli_status")
os.makedirs(WORK, exist_ok=True)

import pyomo.environ as pyo

# ---------------------------------------------------------------- instances


def m_optimal():
    """min x^2 + y^2  s.t. x + y = 1  ->  x=y=0.5, obj = 0.5 (analytic)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=3.0)
    m.y = pyo.Var(initialize=-2.0)
    m.o = pyo.Objective(expr=m.x**2 + m.y**2)
    m.c = pyo.Constraint(expr=m.x + m.y == 1.0)
    return m


def m_infeasible():
    """min x^2+y^2 s.t. x+y >= 3, x+y <= 1, x,y in [-10,10]. Empty feasible set."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-10, 10), initialize=0.0)
    m.y = pyo.Var(bounds=(-10, 10), initialize=0.0)
    m.o = pyo.Objective(expr=m.x**2 + m.y**2)
    m.c1 = pyo.Constraint(expr=m.x + m.y >= 3.0)
    m.c2 = pyo.Constraint(expr=m.x + m.y <= 1.0)
    return m


def m_infeasible_nl():
    """Nonlinear infeasibility: x^2 + y^2 <= 1 and x + y >= 5. Empty."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.5)
    m.y = pyo.Var(initialize=0.5)
    m.o = pyo.Objective(expr=m.x + m.y)
    m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1.0)
    m.c2 = pyo.Constraint(expr=m.x + m.y >= 5.0)
    return m


def m_unbounded():
    """min x + y s.t. x - y == 0, both free. Objective -> -inf along x=y."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.o = pyo.Objective(expr=m.x + m.y)
    m.c = pyo.Constraint(expr=m.x - m.y == 0.0)
    return m


def m_unbounded2():
    """min -exp(x) (unbounded below, smooth). Free x."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.o = pyo.Objective(expr=-pyo.exp(m.x))
    m.c = pyo.Constraint(expr=m.x >= 0.0)
    return m


def m_iterlimit():
    """Chained Rosenbrock, n=40 -- needs many iterations; we cap max_iter=3."""
    n = 40
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, initialize=lambda mm, i: -1.2 if i % 2 == 0 else 1.0)
    m.o = pyo.Objective(
        expr=sum(
            100.0 * (m.x[i + 1] - m.x[i] ** 2) ** 2 + (1.0 - m.x[i]) ** 2
            for i in range(n - 1)
        )
    )
    m.c = pyo.Constraint(expr=sum(m.x[i] for i in m.I) >= 1.0)
    return m


INSTANCES = [
    ("optimal", m_optimal, "solved", (0, 99), []),
    ("infeasible_lin", m_infeasible, "infeasible", (200, 299), []),
    ("infeasible_nl", m_infeasible_nl, "infeasible", (200, 299), []),
    ("unbounded_lin", m_unbounded, "unbounded", (300, 399), []),
    ("unbounded_exp", m_unbounded2, "unbounded", (300, 399), []),
    ("iterlimit", m_iterlimit, "limit", (400, 499), ["max_iter=3"]),
]


def write_nl(model, stub):
    """Write .nl, return (nl_path, var_order_names)."""
    nl = os.path.join(WORK, stub + ".nl")
    _, smap_id = model.write(nl, format="nl", io_options={"symbolic_solver_labels": False})
    return nl


def parse_sol(path):
    """Parse a .sol: return (message, solve_result_num, values list)."""
    with open(path) as f:
        txt = f.read()
    m = re.search(r"objno\s+\d+\s+(-?\d+)", txt)
    srn = int(m.group(1)) if m else None
    return txt, srn


def run(cmd, cwd=None, timeout=60):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0


results = []
print("=" * 78)
print("(a)+(b) CLI exit codes and .sol solve_result_num")
print("=" * 78)

for name, builder, expect_class, srn_range, extra in INSTANCES:
    model = builder()
    nl = write_nl(model, name)
    sol = os.path.join(WORK, name + ".sol")
    if os.path.exists(sol):
        os.remove(sol)

    rc, out, err, t = run([POUNCE, nl, sol] + extra + ["print_level=0"])
    srn = None
    solmsg = ""
    if os.path.exists(sol):
        txt, srn = parse_sol(sol)
        solmsg = txt.splitlines()[0].strip() if txt.strip() else ""

    # pounce status line from stdout
    mstat = re.search(r"(?i)status[: ]+([A-Za-z_ ]+)", out)
    status_txt = mstat.group(1).strip() if mstat else ""
    # last non-empty stdout lines carry the EXIT message
    tail = [l for l in out.splitlines() if l.strip()][-3:]

    # ipopt oracle on the same .nl
    isol = os.path.join(WORK, name + "_ip.sol")
    icmd = [IPOPT, nl, "-AMPL", "print_level=0"]
    if extra:
        icmd += extra
    irc, iout, ierr, it = run(icmd, cwd=WORK)
    # ipopt writes <stub>.sol next to the nl
    ipsol = os.path.join(WORK, name + ".sol")
    # careful: ipopt overwrote pounce's sol -> we already parsed it. re-read:
    itxt, isrn = parse_sol(ipsol) if os.path.exists(ipsol) else ("", None)
    imsg = itxt.splitlines()[0].strip() if itxt.strip() else ""

    in_range = srn is not None and srn_range[0] <= srn <= srn_range[1]
    results.append(
        dict(
            name=name,
            expect=expect_class,
            srn=srn,
            srn_ok=in_range,
            rc=rc,
            ipopt_rc=irc,
            ipopt_srn=isrn,
            sol_msg=solmsg,
            ipopt_msg=imsg,
            tail=tail,
            t=t,
        )
    )
    print(f"\n--- {name}  (expect {expect_class}, srn in {srn_range}) ---")
    print(f"  pounce exit={rc}  solve_result_num={srn}  {'OK' if in_range else 'MISMATCH'}  t={t:.2f}s")
    print(f"  pounce .sol msg : {solmsg[:90]}")
    print(f"  pounce tail     : {' | '.join(x.strip()[:80] for x in tail)}")
    print(f"  ipopt  exit={irc}  solve_result_num={isrn}")
    print(f"  ipopt  .sol msg : {imsg[:90]}")

print()
print("=" * 78)
print("(a) exit-code discrimination summary")
print("=" * 78)
codes = {r["name"]: r["rc"] for r in results}
print("  pounce exit codes:", codes)
print("  ipopt  exit codes:", {r["name"]: r["ipopt_rc"] for r in results})
distinct = len(set(codes[n] for n, _, c, _, _ in INSTANCES for _ in [0]))
by_class = {}
for r in results:
    by_class.setdefault(r["expect"], set()).add(r["rc"])
print("  exit code by expected class:", {k: sorted(v) for k, v in by_class.items()})
exit_discriminates = len({tuple(sorted(v)) for v in by_class.values()}) == len(by_class)
print(f"  exit codes discriminate all 4 classes: {exit_discriminates}")

# ------------------------------------------------------------------ (c)/(d)
print()
print("=" * 78)
print("(c)+(d) `pounce verify` on genuine vs corrupted .sol")
print("=" * 78)

verify_rows = []


def sol_values(path):
    """Return (header_lines, values) from an ASCII .sol."""
    with open(path) as f:
        lines = f.read().splitlines()
    # locate the numeric block: after the 'options' section there are counts,
    # then n_dual duals then n_var primals. Simplest robust approach: collect
    # all pure-float lines at the tail before any 'objno'/'suffix' marker.
    return lines


# Build a fresh, genuine solve for a constrained problem whose corrupted point
# will clearly violate a constraint.
vm = pyo.ConcreteModel()
vm.x = pyo.Var(initialize=3.0)
vm.y = pyo.Var(initialize=-2.0)
vm.o = pyo.Objective(expr=vm.x**2 + vm.y**2)
vm.c = pyo.Constraint(expr=vm.x + vm.y == 1.0)   # x=y=0.5, obj 0.5
vnl = write_nl(vm, "verify_case")
vsol = os.path.join(WORK, "verify_case.sol")
rc, out, err, t = run([POUNCE, vnl, vsol, "print_level=0"])
print(f"  solve for verify case: exit={rc}")
with open(vsol) as f:
    good = f.read()
print("  --- genuine .sol ---")
print("\n".join("    " + l for l in good.splitlines()))

# (d) verify genuine
rc_good, out_good, err_good, _ = run([POUNCE, "verify", vnl, vsol])
print(f"\n  (d) verify(genuine)   exit={rc_good}  (expect 0)")
print("\n".join("      " + l for l in out_good.splitlines()[:20]))

# (c) corrupt: perturb the LAST numeric line (a primal variable) by +1.0 so
# x + y = 2.0 != 1.0, a 1.0 violation of the equality constraint.
lines = good.splitlines()
num_idx = [i for i, l in enumerate(lines) if re.fullmatch(r"\s*-?[\d.eE+-]+\s*", l) and ("." in l or "e" in l.lower())]
print(f"\n  numeric-looking line indices: {num_idx}")
corrupt = list(lines)
target = num_idx[-1]
oldv = float(corrupt[target])
corrupt[target] = repr(oldv + 1.0)
csol = os.path.join(WORK, "verify_case_bad.sol")
with open(csol, "w") as f:
    f.write("\n".join(corrupt) + "\n")
print(f"  corrupted line {target}: {oldv} -> {oldv + 1.0}")

rc_bad, out_bad, err_bad, _ = run([POUNCE, "verify", vnl, csol])
print(f"\n  (c) verify(corrupted) exit={rc_bad}  (expect 20)")
print("\n".join("      " + l for l in out_bad.splitlines()[:20]))

# also corrupt the OTHER primal (first primal) to be safe
corrupt2 = list(lines)
if len(num_idx) >= 2:
    t2 = num_idx[-2]
    oldv2 = float(corrupt2[t2])
    corrupt2[t2] = repr(oldv2 - 5.0)
    csol2 = os.path.join(WORK, "verify_case_bad2.sol")
    with open(csol2, "w") as f:
        f.write("\n".join(corrupt2) + "\n")
    rc_bad2, out_bad2, _, _ = run([POUNCE, "verify", vnl, csol2])
    print(f"\n  (c2) verify(corrupted other var, {oldv2} -> {oldv2 - 5.0}) exit={rc_bad2} (expect 20)")
    print("\n".join("      " + l for l in out_bad2.splitlines()[:12]))
else:
    rc_bad2 = None

# bound-violating corruption on a bounded problem
bm = pyo.ConcreteModel()
bm.x = pyo.Var(bounds=(0, 1), initialize=0.5)
bm.o = pyo.Objective(expr=(bm.x - 0.3) ** 2)
bnl = write_nl(bm, "bound_case")
bsol = os.path.join(WORK, "bound_case.sol")
run([POUNCE, bnl, bsol, "print_level=0"])
with open(bsol) as f:
    blines = f.read().splitlines()
bnum = [i for i, l in enumerate(blines) if re.fullmatch(r"\s*-?[\d.eE+-]+\s*", l) and ("." in l or "e" in l.lower())]
bcorrupt = list(blines)
bcorrupt[bnum[-1]] = "5.0"   # way outside [0,1]
bbad = os.path.join(WORK, "bound_case_bad.sol")
with open(bbad, "w") as f:
    f.write("\n".join(bcorrupt) + "\n")
rc_bgood, o1, _, _ = run([POUNCE, "verify", bnl, bsol])
rc_bbad, o2, _, _ = run([POUNCE, "verify", bnl, bbad])
print(f"\n  (c3) bound case: verify(genuine) exit={rc_bgood} (expect 0), "
      f"verify(x=5 outside [0,1]) exit={rc_bbad} (expect 20)")
print("\n".join("      " + l for l in o2.splitlines()[:12]))

# nonlinear-constraint corruption
nm = pyo.ConcreteModel()
nm.x = pyo.Var(initialize=0.5)
nm.y = pyo.Var(initialize=0.5)
nm.o = pyo.Objective(expr=-(nm.x + nm.y))
nm.c = pyo.Constraint(expr=nm.x**2 + nm.y**2 <= 1.0)   # opt at (1/sqrt2,1/sqrt2)
nnl = write_nl(nm, "nlcon_case")
nsol = os.path.join(WORK, "nlcon_case.sol")
run([POUNCE, nnl, nsol, "print_level=0"])
with open(nsol) as f:
    nlines = f.read().splitlines()
nnum = [i for i, l in enumerate(nlines) if re.fullmatch(r"\s*-?[\d.eE+-]+\s*", l) and ("." in l or "e" in l.lower())]
ncorrupt = list(nlines)
ncorrupt[nnum[-1]] = "3.0"   # x^2+y^2 ~ 9.5 >> 1
nbad = os.path.join(WORK, "nlcon_case_bad.sol")
with open(nbad, "w") as f:
    f.write("\n".join(ncorrupt) + "\n")
rc_ngood, on1, _, _ = run([POUNCE, "verify", nnl, nsol])
rc_nbad, on2, _, _ = run([POUNCE, "verify", nnl, nbad])
print(f"\n  (c4) nonlinear ineq case: verify(genuine) exit={rc_ngood} (expect 0), "
      f"verify(corrupted) exit={rc_nbad} (expect 20)")
print("\n".join("      " + l for l in on2.splitlines()[:12]))

# (c5) tolerance-boundary sweep on the verifier: perturbations straddling
# --feas-tol 1e-6 must flip the verdict at the right place.
print("\n  (c5) verifier tolerance-boundary sweep (--feas-tol 1e-6):")
for delta in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2):
    cc = list(lines)
    cc[target] = repr(oldv + delta)
    pth = os.path.join(WORK, f"sweep_{delta:g}.sol")
    with open(pth, "w") as f:
        f.write("\n".join(cc) + "\n")
    rcx, outx, _, _ = run([POUNCE, "verify", vnl, pth])
    viol = re.search(r"max constraint violation:\s*(\S+)", outx)
    print(f"      delta={delta:8.0e}  exit={rcx:2d}  reported viol={viol.group(1) if viol else '?'}"
          f"   {'REJECT' if rcx == 20 else 'accept'}")

# (c6) the false-'solved' unbounded case: does --require-optimal catch it?
print("\n  (c6) `--require-optimal` on the unbounded_exp .sol pounce called solved:")
uesol = os.path.join(WORK, "unbounded_exp.sol")
uenl = os.path.join(WORK, "unbounded_exp.nl")
# re-solve with pounce (ipopt clobbered it above)
run([POUNCE, uenl, uesol, "print_level=0"])
rc_ue1, o_ue1, _, _ = run([POUNCE, "verify", uenl, uesol])
rc_ue2, o_ue2, _, _ = run([POUNCE, "verify", uenl, uesol, "--require-optimal"])
kkt = re.search(r"KKT stationarity residual:\s*(\S+)", o_ue2)
objl = re.search(r"objective at x\*:\s*(\S+)", o_ue2)
print(f"      verify plain            exit={rc_ue1} (feasible -> 0 is correct)")
print(f"      verify --require-optimal exit={rc_ue2} (expect 20)")
print(f"      objective at claimed x* = {objl.group(1) if objl else '?'}, "
      f"KKT residual = {kkt.group(1) if kkt else '?'}")

# ------------------------------------------------------------------ (e)
print()
print("=" * 78)
print("(e) library API status strings vs CLI statuses")
print("=" * 78)

import numpy as np
import pounce

lib_rows = []


def lib_solve(f, grad, x0, cons=None, lb=None, ub=None, **kw):
    return pounce.minimize(f, x0, jac=grad, bounds=None, constraints=cons, **kw)


cases = []

# optimal: min x^2+y^2 st x+y=1
r1 = pounce.minimize(
    lambda v: v[0] ** 2 + v[1] ** 2,
    np.array([3.0, -2.0]),
    jac=lambda v: np.array([2 * v[0], 2 * v[1]]),
    constraints=[{"type": "eq", "fun": lambda v: np.array([v[0] + v[1] - 1.0]),
                  "jac": lambda v: np.array([[1.0, 1.0]])}],
)
cases.append(("optimal", r1))

# infeasible
r2 = pounce.minimize(
    lambda v: v[0] ** 2 + v[1] ** 2,
    np.array([0.0, 0.0]),
    jac=lambda v: np.array([2 * v[0], 2 * v[1]]),
    bounds=[(-10, 10), (-10, 10)],
    constraints=[
        {"type": "ineq", "fun": lambda v: np.array([v[0] + v[1] - 3.0]),
         "jac": lambda v: np.array([[1.0, 1.0]])},
        {"type": "ineq", "fun": lambda v: np.array([1.0 - v[0] - v[1]]),
         "jac": lambda v: np.array([[-1.0, -1.0]])},
    ],
)
cases.append(("infeasible_lin", r2))

# unbounded
r3 = pounce.minimize(
    lambda v: v[0] + v[1],
    np.array([0.0, 0.0]),
    jac=lambda v: np.array([1.0, 1.0]),
    constraints=[{"type": "eq", "fun": lambda v: np.array([v[0] - v[1]]),
                  "jac": lambda v: np.array([[1.0, -1.0]])}],
)
cases.append(("unbounded_lin", r3))

# iteration limit
n = 40


def rosen(v):
    return float(sum(100.0 * (v[i + 1] - v[i] ** 2) ** 2 + (1 - v[i]) ** 2 for i in range(n - 1)))


def rosen_g(v):
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -400.0 * v[i] * (v[i + 1] - v[i] ** 2) - 2 * (1 - v[i])
        g[i + 1] += 200.0 * (v[i + 1] - v[i] ** 2)
    return g


x0 = np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
r4 = pounce.minimize(rosen, x0, jac=rosen_g,
                     constraints=[{"type": "ineq", "fun": lambda v: np.array([v.sum() - 1.0]),
                                   "jac": lambda v: np.ones((1, n))}],
                     options={"max_iter": 3})
cases.append(("iterlimit", r4))

# unbounded via exp: min -exp(x) s.t. x >= 0
r5 = pounce.minimize(
    lambda v: -float(np.exp(v[0])),
    np.array([1.0]),
    jac=lambda v: -np.exp(v),
    constraints=[{"type": "ineq", "fun": lambda v: np.array([v[0]]),
                  "jac": lambda v: np.array([[1.0]])}],
)
cases.append(("unbounded_exp", r5))

cli_by_name = {r["name"]: r for r in results}
for nm, r in cases:
    st = getattr(r, "status", None)
    msg = getattr(r, "message", "")
    succ = getattr(r, "success", None)
    cli = cli_by_name.get(nm, {})
    print(f"  {nm:16s} lib: success={succ} status={st!r} msg={str(msg)[:60]!r}")
    print(f"  {'':16s} cli: srn={cli.get('srn')} exit={cli.get('rc')} sol_msg={cli.get('sol_msg','')[:60]!r}")
    lib_rows.append((nm, succ, st, str(msg), cli.get("srn"), cli.get("rc")))

# ------------------------------------------------------------------ verdict
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
srn_fail = [r["name"] for r in results if not r["srn_ok"]]
print(f"  (a) exit codes discriminate 4 classes: {exit_discriminates} "
      f"(pounce codes: {sorted(set(codes.values()))})")
print(f"  (b) solve_result_num out-of-range instances: {srn_fail or 'none'}")
print(f"  (c) verify rejects corrupted: eq={rc_bad}, eq2={rc_bad2}, bound={rc_bbad}, nlineq={rc_nbad} (want 20)")
print(f"  (d) verify accepts genuine  : eq={rc_good}, bound={rc_bgood}, nlineq={rc_ngood} (want 0)")

crit_c = all(v == 20 for v in [rc_bad, rc_bbad, rc_nbad] + ([rc_bad2] if rc_bad2 is not None else []))
crit_d = all(v == 0 for v in [rc_good, rc_bgood, rc_ngood])

if not crit_c:
    print("VERDICT: SOLVER_BUG (pounce verify ACCEPTED a corrupted solution)")
elif not crit_d:
    print("VERDICT: SOLVER_BUG (pounce verify REJECTED a genuine solution)")
elif srn_fail:
    print(f"VERDICT: SOLVER_BUG (solve_result_num wrong for {srn_fail})")
else:
    print("VERDICT: PASS")

json.dump(
    dict(results=results, verify=dict(good=rc_good, bad=rc_bad, bad2=rc_bad2,
                                      bgood=rc_bgood, bbad=rc_bbad,
                                      ngood=rc_ngood, nbad=rc_nbad),
         lib=lib_rows),
    open(os.path.join(WORK, "summary.json"), "w"), indent=2, default=str)
print(f"\n(raw artifacts in {WORK})")
