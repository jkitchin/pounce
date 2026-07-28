"""Adversary cross-check: .nl / CLI input surface edge cases.

Family: nlp (parser / API contract)
Class:  input edge cases -- no objective, no constraints, maximization sign
        convention, fixed variables, free-floating variables, constant
        objective, domain errors at x0, corrupt .nl files.
Oracle: Ipopt 3.14 (ASL) run on the BYTE-IDENTICAL .nl file, plus
        `pounce verify` and analytic values.

Every case writes one .nl with Pyomo, then runs both binaries on that exact
file (same bytes, same .col ordering) and compares.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    maximize,
    minimize,
    log,
    sqrt,
    exp,
)

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = tempfile.mkdtemp(prefix="adv_nl_")

results = []  # (case, name, verdict, detail)


def record(case, name, verdict, detail):
    results.append((case, name, verdict, detail))
    print(f"  [{case}] {name}: {verdict} -- {detail}")


# ---------------------------------------------------------------- utilities
def write_nl(m, stub):
    """Write model to <stub>.nl with a .col file; return (nl, colnames)."""
    nl = os.path.join(WORK, stub + ".nl")
    m.write(nl, io_options={"symbolic_solver_labels": True})
    col = nl[:-3] + ".col"
    names = []
    if os.path.exists(col):
        with open(col) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    return nl, names


def parse_sol(path):
    """Minimal AMPL .sol parser -> (message, duals, primals, solve_result_num)."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    msg = []
    while i < len(lines) and lines[i].strip() != "Options":
        if lines[i].strip():
            msg.append(lines[i].strip())
        i += 1
    if i >= len(lines):
        raise ValueError("no Options line")
    i += 1
    nopts = int(lines[i])
    i += 1
    need_vbtol = False
    if nopts > 4:
        nopts -= 2
        need_vbtol = True
    z = []
    for _ in range(nopts + 4):
        z.append(int(lines[i]))
        i += 1
    if need_vbtol:
        i += 1
    m_ = z[nopts + 1]
    n_ = z[nopts + 3]
    duals = [float(lines[i + k]) for k in range(m_)]
    i += m_
    primals = [float(lines[i + k]) for k in range(n_)]
    i += n_
    objno = None
    while i < len(lines):
        if lines[i].startswith("objno"):
            objno = int(lines[i].split()[2])
            break
        i += 1
    return "\n".join(msg), duals, primals, objno


def run(binary, nl, extra=(), timeout=20):
    """Run pounce (`pounce in.nl out.sol`) or ipopt (`ipopt stub -AMPL`) on the
    SAME .nl bytes.  Ipopt's ASL driver only accepts the stub form, so the file
    is copied to a per-solver stub first (bytes unchanged, verified by hash)."""
    tag = os.path.basename(binary)
    if tag == "ipopt":
        stub = nl[:-3] + ".ipopt"
        shutil.copyfile(nl, stub + ".nl")
        assert (hashlib.sha256(open(nl, "rb").read()).digest()
                == hashlib.sha256(open(stub + ".nl", "rb").read()).digest())
        sol = stub + ".sol"
        if os.path.exists(sol):
            os.remove(sol)
        cmd = [binary, stub, "-AMPL", "print_level=0"] + list(extra)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.perf_counter() - t0
        out = {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
               "time": dt, "sol": sol if os.path.exists(sol) else None}
        if out["sol"]:
            try:
                out["msg"], out["duals"], out["x"], out["objno"] = parse_sol(sol)
            except Exception as e:  # noqa: BLE001
                out["parse_error"] = str(e)
        return out
    sol = nl[:-3] + f".{tag}.sol"
    if os.path.exists(sol):
        os.remove(sol)
    cmd = [binary, nl, sol] + list(extra)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.perf_counter() - t0
    out = {
        "rc": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "time": dt,
        "sol": sol if os.path.exists(sol) else None,
    }
    if out["sol"]:
        try:
            out["msg"], out["duals"], out["x"], out["objno"] = parse_sol(sol)
        except Exception as e:  # noqa: BLE001
            out["parse_error"] = str(e)
    return out


def objective_at(m, names, x):
    """Evaluate the model's (single) objective at the .sol primals."""
    if x is None:
        return None
    byname = {v.name: v for v in m.component_data_objects(Var, active=True)}
    for nm, val in zip(names, x):
        if nm in byname:
            byname[nm].set_value(val, skip_validation=True)
    objs = list(m.component_data_objects(Objective, active=True))
    if not objs:
        return None
    from pyomo.environ import value

    return value(objs[0])


def panicked(out):
    s = (out.get("stdout", "") or "") + (out.get("stderr", "") or "")
    if "panicked at" in s or "RUST_BACKTRACE" in s:
        return True
    # negative return code == killed by signal (SIGSEGV/SIGABRT)
    return out["rc"] < 0


def verify(nl, sol, feas_tol=1e-6):
    p = subprocess.run(
        [POUNCE, "verify", nl, sol, "--feas-tol", str(feas_tol)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return p.returncode, (p.stdout + p.stderr).strip()


print(f"work dir: {WORK}\n")

# ------------------------------------------------------- (a) NO OBJECTIVE
print("(a) .nl with NO objective (pure feasibility)")
m = ConcreteModel()
m.x = Var(bounds=(-10, 10), initialize=1.0)
m.y = Var(bounds=(-10, 10), initialize=0.5)
m.c1 = Constraint(expr=m.x ** 2 + m.y ** 2 == 4.0)
m.c2 = Constraint(expr=m.x - m.y == 0.0)
try:
    nl, names = write_nl(m, "a_noobj")
    pr = run(POUNCE, nl)
    ir = run(IPOPT, nl)
    if panicked(pr):
        record("a", "no-objective", "SOLVER_BUG", f"panic rc={pr['rc']}")
    elif pr.get("x") is None:
        record("a", "no-objective", "SOLVER_LIMITATION",
               f"rc={pr['rc']} no .sol; stderr={pr['stderr'][:200]}")
    else:
        xv = dict(zip(names, pr.get("x")))
        r1 = xv["x"] ** 2 + xv["y"] ** 2 - 4.0
        r2 = xv["x"] - xv["y"]
        vc, vout = verify(nl, pr["sol"])
        ok = abs(r1) < 1e-6 and abs(r2) < 1e-6 and vc == 0
        record("a", "no-objective", "PASS" if ok else "SOLVER_BUG",
               f"x={pr['x']} |c1|={abs(r1):.2e} |c2|={abs(r2):.2e} "
               f"objno={pr.get('objno')} verify_rc={vc} ipopt_rc={ir['rc']} "
               f"ipopt_x={ir.get('x')}")
except Exception as e:  # noqa: BLE001
    record("a", "no-objective", "SCRIPT_ERROR", repr(e))

# --------------------------------------------- (b) NO CONSTRAINTS, BOUNDS
print("(b) .nl with no constraints, only bounds")
m = ConcreteModel()
m.x = Var(bounds=(0.0, 1.0), initialize=0.5)
m.y = Var(bounds=(-5.0, 5.0), initialize=0.0)
m.o = Objective(expr=(m.x - 3.0) ** 2 + (m.y + 2.0) ** 2, sense=minimize)
KNOWN_B = (1.0 - 3.0) ** 2 + 0.0  # x*=1 (at ub), y*=-2 -> 4.0
try:
    nl, names = write_nl(m, "b_nocon")
    pr = run(POUNCE, nl)
    ir = run(IPOPT, nl)
    fp = objective_at(m, names, pr.get("x"))
    fi = objective_at(m, names, ir.get("x"))
    ok = abs(fp - KNOWN_B) < 1e-6 and abs(fp - fi) < 1e-6
    record("b", "bounds-only", "PASS" if ok else "SOLVER_BUG",
           f"pounce obj={fp:.10g} x={pr['x']} | ipopt obj={fi:.10g} "
           f"| known={KNOWN_B}")
except Exception as e:  # noqa: BLE001
    record("b", "bounds-only", "SCRIPT_ERROR", repr(e))

# ----------------------------------------------- (c) MAXIMIZATION SIGN ***
print("(c) MAXIMIZATION sign convention (highest-value test)")
# SAME objective expression f = (x-3)^2 + (y-4)^2 on the SAME feasible set
#   x + y = 5, 0 <= x,y <= 5.
# Parameterise the line by x=t, y=5-t, t in [0,5]:
#   f(t) = (t-3)^2 + (1-t)^2 = 2t^2 - 8t + 10, a convex parabola.
#   min at t=2  -> x=2, y=3, f = 1 + 1 = 2       (interior of the box)
#   max at t=0  -> x=0, y=5, f = 9 + 1 = 10      (corner; f(5)=4+16=20 -> t=5)
#   f(5) = 20 > f(0) = 10, so the MAX is at t=5: x=5, y=0, f = 20.
# min and max therefore land on *different* points with *different* values --
# a sense/sign error cannot hide.
MIN_F, MIN_X = 2.0, (2.0, 3.0)
MAX_F, MAX_X = 20.0, (5.0, 0.0)


def build_xy(sense):
    mm = ConcreteModel()
    mm.x = Var(bounds=(0, 5), initialize=1.0)
    mm.y = Var(bounds=(0, 5), initialize=1.0)
    mm.c = Constraint(expr=mm.x + mm.y == 5.0)
    mm.o = Objective(expr=(mm.x - 3.0) ** 2 + (mm.y - 4.0) ** 2, sense=sense)
    return mm


try:
    mmax = build_xy(maximize)
    mmin = build_xy(minimize)
    nlmax, nmax = write_nl(mmax, "c_max")
    nlmin, nmin = write_nl(mmin, "c_min")
    hmax = hashlib.sha256(open(nlmax, "rb").read()).hexdigest()[:12]
    hmin = hashlib.sha256(open(nlmin, "rb").read()).hexdigest()[:12]

    pmax, imax = run(POUNCE, nlmax), run(IPOPT, nlmax)
    pmin, imin = run(POUNCE, nlmin), run(IPOPT, nlmin)

    f = lambda mo, nms, r: objective_at(mo, nms, r["x"])
    fpmax, fimax = f(mmax, nmax, pmax), f(mmax, nmax, imax)
    fpmin, fimin = f(mmin, nmin, pmin), f(mmin, nmin, imin)

    detail = (f"MAX: pounce obj={fpmax:.10g} x={pmax['x']} | ipopt obj={fimax:.10g} "
              f"x={imax['x']}\n        MIN: pounce obj={fpmin:.10g} x={pmin['x']} "
              f"| ipopt obj={fimin:.10g} x={imin['x']}\n        "
              f"nl sha max={hmax} min={hmin}")
    ok = (abs(fpmax - MAX_F) < 1e-5 and abs(fpmax - fimax) < 1e-5
          and abs(fpmin - MIN_F) < 1e-5 and abs(fpmin - fimin) < 1e-5)
    detail += f"\n        expected MAX={MAX_F} at {MAX_X}, MIN={MIN_F} at {MIN_X}"
    record("c", "max-vs-min sign", "PASS" if ok else "SOLVER_BUG", detail)

    # second sign probe: a *negative* objective, where a dropped sign would
    # look plausible.  f = -(x-2)^2 on x in [-5,5]:
    #   max -> x = 2, f = 0 ;  min -> x = -5, f = -49  (f(5) = -9 > -49).
    for sense, tag, want_x, want_f in ((maximize, "max_neg_sq", 2.0, 0.0),
                                       (minimize, "min_neg_sq", -5.0, -49.0)):
        m2 = ConcreteModel()
        m2.x = Var(bounds=(-5, 5), initialize=0.0)
        m2.o = Objective(expr=-(m2.x - 2.0) ** 2, sense=sense)
        nl2, n2 = write_nl(m2, "c_" + tag)
        p2, i2 = run(POUNCE, nl2), run(IPOPT, nl2)
        fp2, fi2 = objective_at(m2, n2, p2.get("x")), objective_at(m2, n2, i2.get("x"))
        ok2 = abs(fp2 - want_f) < 1e-5 and abs(fp2 - fi2) < 1e-5
        record("c", tag, "PASS" if ok2 else "SOLVER_BUG",
               f"pounce obj={fp2:.10g} x={p2['x']} | ipopt obj={fi2:.10g} "
               f"x={i2['x']} | expected obj={want_f} (x={want_x} or -{want_x})")
except Exception as e:  # noqa: BLE001
    record("c", "max-vs-min sign", "SCRIPT_ERROR", repr(e))

# ------------------------------------------------------ (d) FIXED VARIABLE
print("(d) fixed variable (lb == ub)")
m = ConcreteModel()
m.x = Var(bounds=(2.0, 2.0), initialize=0.0)   # fixed by equal bounds
m.y = Var(bounds=(-10, 10), initialize=0.0)
m.c = Constraint(expr=m.x + m.y >= 1.0)
m.o = Objective(expr=(m.x - 1.0) ** 2 + (m.y - 3.0) ** 2, sense=minimize)
KNOWN_D = 1.0 + 0.0  # x=2 forced, y=3 -> 1.0
try:
    nl, names = write_nl(m, "d_fixed")
    pr, ir = run(POUNCE, nl), run(IPOPT, nl)
    if panicked(pr):
        record("d", "fixed-var", "SOLVER_BUG", f"panic rc={pr['rc']}")
    else:
        fp = objective_at(m, names, pr.get("x"))
        fi = objective_at(m, names, ir.get("x"))
        xv = dict(zip(names, pr.get("x")))
        vc, _ = verify(nl, pr["sol"])
        ok = abs(fp - KNOWN_D) < 1e-6 and abs(xv["x"] - 2.0) < 1e-9 and vc == 0
        record("d", "fixed-var", "PASS" if ok else "SOLVER_BUG",
               f"pounce obj={fp:.10g} x={pr['x']} | ipopt obj={fi:.10g} "
               f"x={ir['x']} | known={KNOWN_D} verify_rc={vc}")
except Exception as e:  # noqa: BLE001
    record("d", "fixed-var", "SCRIPT_ERROR", repr(e))

# ------------------------------------------------ (e) FREE-FLOATING VARIABLE
print("(e) variable in neither objective nor any constraint")
m = ConcreteModel()
m.x = Var(bounds=(-10, 10), initialize=0.0)
m.z = Var(bounds=(-1.0, 4.0), initialize=3.0)  # never used
m.o = Objective(expr=(m.x - 1.0) ** 2, sense=minimize)
m.c = Constraint(expr=m.x >= -5.0)
try:
    nl, names = write_nl(m, "e_float")
    z_in_nl = "z" in names
    if not z_in_nl:
        # Pyomo prunes the unused column; force it in with a 0-coefficient
        # nonlinear term that the writer cannot simplify away.
        m2 = ConcreteModel()
        m2.x = Var(bounds=(-10, 10), initialize=0.0)
        m2.z = Var(bounds=(-1.0, 4.0), initialize=3.0)
        m2.o = Objective(expr=(m2.x - 1.0) ** 2, sense=minimize)
        m2.c = Constraint(expr=m2.x + 0.0 * exp(m2.z) >= -5.0)
        nl, names = write_nl(m2, "e_float2")
        m = m2
        z_in_nl = "z" in names
    pr, ir = run(POUNCE, nl), run(IPOPT, nl)
    if panicked(pr):
        record("e", "free-floating-var", "SOLVER_BUG", f"panic rc={pr['rc']}")
    elif pr.get("x") is None:
        record("e", "free-floating-var", "SOLVER_LIMITATION",
               f"rc={pr['rc']} no .sol; {pr['stderr'][:200]}")
    else:
        fp = objective_at(m, names, pr.get("x"))
        fi = objective_at(m, names, ir.get("x"))
        xv = dict(zip(names, pr.get("x")))
        zval = xv.get("z")
        zok = zval is None or (-1.0 - 1e-9 <= zval <= 4.0 + 1e-9)
        ok = abs(fp - 0.0) < 1e-8 and abs(fp - fi) < 1e-8 and zok
        record("e", "free-floating-var (pyomo)",
               "PASS" if ok else "SOLVER_BUG",
               f"z present in .nl={z_in_nl} cols={names} pounce obj={fp:.3e} "
               f"x={pr['x']} z={zval} in-bounds={zok} | ipopt obj={fi:.3e} "
               f"x={ir['x']}")
except Exception as e:  # noqa: BLE001
    record("e", "free-floating-var (pyomo)", "SCRIPT_ERROR", repr(e))

# Pyomo *prunes* an unreferenced column, so the realistic-writer path can never
# emit a truly free-floating variable.  GAMS/AMPL can.  Hand-build the same .nl
# with a second column `z` (bounds [-1,4], x0=3) present in NO row and NOT in
# the objective gradient, and feed it to both solvers.
print("(e2) hand-built .nl with a genuinely free-floating column")
FLOAT_NL = """g3 1 1 0	# problem unknown
 2 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 1 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 1 	# nonzeros in Jacobian, obj. gradient
 1 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c
n0
O0 0	#o
o5	#^
o0	#+
v0	#x
n-1.0
n2
x2	# initial guess
0 0.0	#x
1 3.0	#z
r	#1 ranges (rhs's)
2 -5.0	#c
b	#2 bounds (on variables)
0 -10 10	#x
0 -1 4	#z
k1	#intermediate Jacobian column lengths
1
J0 1	#c
0 1
G0 1	#o
0 0
"""
try:
    path = os.path.join(WORK, "e2_float.nl")
    with open(path, "w") as fh:
        fh.write(FLOAT_NL)
    pr, ir = run(POUNCE, path), run(IPOPT, path)
    if panicked(pr):
        record("e2", "free-floating column", "SOLVER_BUG",
               f"panic rc={pr['rc']} {pr['stderr'][:300]}")
    elif pr.get("x") is None:
        record("e2", "free-floating column", "SOLVER_LIMITATION",
               f"rc={pr['rc']} no .sol; {(pr['stdout'] + pr['stderr'])[-400:]}")
    else:
        xv, zv = pr["x"][0], pr["x"][1]
        # true solution: min (x-1)^2 s.t. x >= -5, x in [-10,10] -> x = 1.
        # z is untouched by the model -> any value inside [-1, 4] is valid.
        vc, _ = verify(path, pr["sol"])
        ok = (abs(xv - 1.0) < 1e-6 and -1.0 - 1e-9 <= zv <= 4.0 + 1e-9
              and vc == 0)
        record("e2", "free-floating column",
               "PASS" if ok else "SOLVER_BUG",
               f"pounce x={xv:.10g} (want 1) z={zv:.10g} (must stay in [-1,4]) "
               f"objno={pr.get('objno')} verify_rc={vc} | ipopt rc={ir['rc']} "
               f"x={ir.get('x')}")
except Exception as e:  # noqa: BLE001
    record("e2", "free-floating column", "SCRIPT_ERROR", repr(e))

# ------------------------------------------------ (f) CONSTANT-ONLY OBJECTIVE
print("(f) constant-only objective")
m = ConcreteModel()
m.x = Var(bounds=(-10, 10), initialize=1.0)
m.c = Constraint(expr=m.x ** 2 == 9.0)
try:
    m.o = Objective(expr=7.0, sense=minimize)
    nl, names = write_nl(m, "f_const")
    pr, ir = run(POUNCE, nl), run(IPOPT, nl)
    if panicked(pr):
        record("f", "constant-objective", "SOLVER_BUG", f"panic rc={pr['rc']}")
    elif pr.get("x") is None:
        record("f", "constant-objective", "SOLVER_LIMITATION",
               f"rc={pr['rc']} no .sol; {pr['stderr'][:300]}")
    else:
        xv = dict(zip(names, pr.get("x")))
        resid = abs(xv["x"] ** 2 - 9.0)
        vc, _ = verify(nl, pr["sol"])
        ok = resid < 1e-6 and vc == 0
        record("f", "constant-objective", "PASS" if ok else "SOLVER_BUG",
               f"pounce x={pr['x']} |c|={resid:.2e} objno={pr.get('objno')} "
               f"verify_rc={vc} | ipopt rc={ir['rc']} x={ir.get('x')}")
except Exception as e:  # noqa: BLE001
    record("f", "constant-objective", "SCRIPT_ERROR", repr(e))

# -------------------------------------------- (g) DOMAIN ERRORS AT x0 = 0
print("(g) nonlinear domain errors at the starting point")
domain_cases = []


def dc(name, build, check):
    domain_cases.append((name, build, check))


def b_log():
    mm = ConcreteModel()
    mm.x = Var(bounds=(0.0, 10.0), initialize=0.0)   # log(0) = -inf at x0
    mm.o = Objective(expr=-log(mm.x) + mm.x, sense=minimize)
    return mm, 1.0  # x* = 1, obj = 1


def b_sqrt():
    mm = ConcreteModel()
    mm.x = Var(bounds=(0.0, 10.0), initialize=0.0)   # d/dx sqrt at 0 = inf
    mm.o = Objective(expr=(sqrt(mm.x) - 2.0) ** 2, sense=minimize)
    return mm, 0.0  # x* = 4, obj = 0


def b_div():
    mm = ConcreteModel()
    mm.x = Var(bounds=(0.0, 10.0), initialize=0.0)   # 1/0 at x0
    mm.o = Objective(expr=1.0 / mm.x + mm.x, sense=minimize)
    return mm, 2.0  # x* = 1, obj = 2


def b_powneg():
    mm = ConcreteModel()
    mm.x = Var(bounds=(-5.0, 5.0), initialize=-2.0)  # (-2)**0.5 -> NaN
    mm.o = Objective(expr=(mm.x) ** 0.5, sense=minimize)
    return mm, None


dc("log(x) at x0=0", b_log, None)
dc("sqrt(x) at x0=0", b_sqrt, None)
dc("1/x at x0=0", b_div, None)
dc("x**0.5, x0=-2", b_powneg, None)

for i, (name, build, _chk) in enumerate(domain_cases):
    try:
        mm, known = build()
        nl, names = write_nl(mm, f"g_{i}")
        pr = run(POUNCE, nl)
        ir = run(IPOPT, nl)
        cx = subprocess.run([POUNCE, "check-x0", nl], capture_output=True,
                            text=True, timeout=20)
        if panicked(pr):
            record("g", name, "SOLVER_BUG",
                   f"panic rc={pr['rc']} {pr['stderr'][:300]}")
            continue
        fp = objective_at(mm, names, pr.get("x")) if pr.get("x") else None
        fi = objective_at(mm, names, ir.get("x")) if ir.get("x") else None
        bad = fp is not None and (fp != fp or abs(fp) == float("inf"))
        if known is not None:
            ok = fp is not None and not bad and abs(fp - known) < 1e-5
            verdict = "PASS" if ok else "SOLVER_LIMITATION"
        else:
            # NaN domain: acceptable outcomes are a clear error or a clean
            # (finite, in-domain) answer.  Garbage/NaN objective is the bug.
            ok = (pr.get("x") is None) or (not bad)
            verdict = "PASS" if ok else "SOLVER_BUG"
        record("g", name, verdict,
               f"pounce rc={pr['rc']} objno={pr.get('objno')} obj={fp} "
               f"x={pr.get('x')} | ipopt rc={ir['rc']} obj={fi} "
               f"x={ir.get('x')} | known={known} | check-x0 rc={cx.returncode}")
    except Exception as e:  # noqa: BLE001
        record("g", name, "SCRIPT_ERROR", repr(e))

# --------------------------------------------- (h) TRUNCATED / CORRUPT .nl
print("(h) truncated / corrupted .nl")
m = ConcreteModel()
m.x = Var(bounds=(-10, 10), initialize=0.5)
m.y = Var(bounds=(-10, 10), initialize=0.5)
m.c = Constraint(expr=m.x ** 2 + m.y ** 2 <= 1.0)
m.o = Objective(expr=exp(m.x) + (m.y - 1.0) ** 2, sense=minimize)
good, _ = write_nl(m, "h_good")
raw = open(good, "rb").read()


def mangle_bounds(raw_bytes, bad=b"0 notanumber"):
    """Replace the first bound line after the 'b' section header."""
    lines = raw_bytes.split(b"\n")
    for i, ln in enumerate(lines):
        if ln.startswith(b"b\t") or ln.strip() == b"b":
            lines[i + 1] = bad
            return b"\n".join(lines)
    raise AssertionError("no bounds section found")


def drop_a_line(raw_bytes, marker):
    lines = raw_bytes.split(b"\n")
    for i, ln in enumerate(lines):
        if ln.startswith(marker):
            del lines[i]
            return b"\n".join(lines)
    raise AssertionError(f"marker {marker!r} not found")


corruptions = {
    "truncated_50pct": raw[: len(raw) // 2],
    "truncated_90pct": raw[: int(len(raw) * 0.9)],
    "header_only": raw[: raw.index(b"\n", raw.index(b"\n") + 1)],
    "empty": b"",
    "garbage": b"\x00\x01\x02not an nl file\xff\xfe\n" * 20,
    "bad_nvar_count": raw.replace(b"\n 2 1 1 0 0 ", b"\n 99 1 1 0 0 ", 1),
    "bad_opcode": raw.replace(b"\no0", b"\no999", 1),
    "bad_header_magic": b"Q3 1 1 0" + raw[8:],
    "nonnumeric_bound": mangle_bounds(raw),
    "nan_bound": mangle_bounds(raw, b"0 nan"),
    "missing_bound_line": drop_a_line(raw, b"r\t"),
    "truncated_1byte": raw[:1],
}
for name, data in corruptions.items():
    path = os.path.join(WORK, f"h_{name}.nl")
    assert data != raw, f"corruption {name} was a no-op"
    with open(path, "wb") as fh:
        fh.write(data)
    try:
        out = run(POUNCE, path, timeout=15)
    except subprocess.TimeoutExpired:
        record("h", name, "SOLVER_BUG", "TIMEOUT (hang on corrupt input)")
        continue
    iout = run(IPOPT, path, timeout=15)
    s = (out["stdout"] or "") + (out["stderr"] or "")
    if panicked(out):
        record("h", name, "SOLVER_BUG",
               f"PANIC/signal rc={out['rc']}: {s.strip()[:300]}")
    elif out["rc"] == 0:
        record("h", name, "SOLVER_BUG",
               f"exit 0 on corrupt input! x={out.get('x')} msg={s[:200]}")
    else:
        record("h", name, "PASS",
               f"rc={out['rc']} clean error: {s.strip().splitlines()[:2]} "
               f"| ipopt rc={iout['rc']}")

# ------------------------------------------------------------------ SUMMARY
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
bad = [r for r in results if r[2] not in ("PASS",)]
for c, n, v, d in results:
    print(f"{v:22s} ({c}) {n}")
print(f"\n{len(results) - len(bad)}/{len(results)} PASS")
if not bad:
    print("VERDICT: PASS")
else:
    worst = ("SCRIPT_ERROR", "SOLVER_BUG", "SOLVER_LIMITATION", "TOLERANCE")
    v = next((w for w in ("SOLVER_BUG", "SCRIPT_ERROR", "SOLVER_LIMITATION",
                          "TOLERANCE") if any(r[2] == w for r in bad)), "FAIL")
    print(f"VERDICT: {v}")
print(f"\nartifacts: {WORK}")
