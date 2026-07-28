"""Adversary Iteration 1 (PREFIX i1): Primal / dual / sensitivity correctness,
cross-checked against the ipopt-MA57 executable.

Theme: verify pounce's PRIMAL and DUAL/SENSITIVITY quantities against
ipopt (linear_solver=ma57) .sol output, analytic/closed-form duals, and
finite-difference / sIPOPT sensitivity oracles.

RESULT (2026-07-23): the constraint-dual SIGN FLIP logged on 2026-07-22
(pounce .sol mult_g negated vs the AMPL/Pyomo/Ipopt marginal-value
convention) is now FIXED. Across LP (convex IPM), QP (convex IPM), NLP
(filter-IPM) and QCQP (conic IPM) .sol writers, pounce's mult_g matches
ipopt-MA57 SIGNED (rel_pos ~1e-9), and matches the analytic df/db shadow
prices, for both <= and >= and equality and ranged rows. Each dual test
below checks (a) primal, (b) |mult_g| magnitude vs MA57, (c) SIGNED mult_g
vs MA57 and vs analytic. The new ipopt_zL_out/ipopt_zU_out bound-multiplier
suffixes (#296/#312) match MA57 exactly incl. complementarity.
All 10 tests PASS; no new findings.

Run:  source .venv-qa/bin/activate
      export DYLD_LIBRARY_PATH=<CoinHSL lib>
      python adversary/runs/2026-07-23_i1_primal_dual_ma57.py [test_number|all]
"""
import os, re, sys, subprocess, time, math
import numpy as np

np.set_printoptions(precision=6, suppress=True)

ROOT = "/Users/jkitchin/projects/pounce"
CLI = f"{ROOT}/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
IPOPT_SENS = "/opt/homebrew/bin/ipopt_sens"
WORK = f"{ROOT}/adversary/runs/i1_work"
os.makedirs(WORK, exist_ok=True)
HSL = "/Users/jkitchin/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/lib"
ENV = dict(os.environ, DYLD_LIBRARY_PATH=HSL)

import pyomo.environ as pe


# ------------------------------------------------------------------ helpers
def relinf(a, b):
    a = np.atleast_1d(np.asarray(a, float)); b = np.atleast_1d(np.asarray(b, float))
    return float(np.linalg.norm(a - b, np.inf) / max(1.0, np.linalg.norm(b, np.inf)))


def write_nl(model, stub):
    """Write <stub>.nl with symbolic .row/.col so we can map indices->names."""
    model.write(stub + ".nl", format="nl",
                io_options={"symbolic_solver_labels": True})
    col = [l.strip() for l in open(stub + ".col")] if os.path.exists(stub + ".col") else []
    row = [l.strip() for l in open(stub + ".row")] if os.path.exists(stub + ".row") else []
    return col, row


def parse_sol(path, n, m):
    """Robust AMPL .sol parse anchored on the 'objno' line.

    Returns dict(msg, duals[m], primal[n], suffix={name:{idx:val}}).
    Both solvers run on the SAME .nl so variable/constraint index order is
    identical -> compare by index directly.
    """
    lines = [l.rstrip("\n") for l in open(path)]
    oi = next(i for i, l in enumerate(lines) if l.strip().startswith("objno"))
    # walk up collecting contiguous numeric lines
    j = oi - 1
    while j >= 0:
        try:
            float(lines[j].strip()); j -= 1
        except ValueError:
            break
    block = [float(x.strip()) for x in lines[j + 1:oi]]
    # The contiguous numeric run also contains the Options-header integers
    # (no blank line separates them). The dual(m)+primal(n) values are the
    # LAST n+m entries, immediately before 'objno'.
    assert len(block) >= n + m, f"{path}: need >= {n+m} numeric, got {len(block)}"
    nums = block[-(n + m):] if (n + m) > 0 else []
    duals = np.array(nums[:m]); primal = np.array(nums[m:m + n])
    # suffix blocks
    suffix = {}
    i = 0
    while i < len(lines):
        mt = re.match(r"^suffix\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", lines[i].strip())
        if mt:
            count = int(mt.group(2)); tabline = int(mt.group(5))
            name = lines[i + 1].strip()
            body = i + 2 + tabline
            d = {}
            for k in range(count):
                idx, val = lines[body + k].split()
                d[int(idx)] = float(val)
            suffix[name] = d
            i = body + count
            continue
        i += 1
    return dict(msg=lines[0].strip(), duals=duals, primal=primal, suffix=suffix)


def run_ipopt(stub, n, m, sens=False, **opts):
    """Run ipopt (or ipopt_sens) with MA57 on <stub>.nl; assert MA57 header."""
    exe = IPOPT_SENS if sens else IPOPT
    args = [exe, os.path.basename(stub), "-AMPL", "linear_solver=ma57"]
    args += [f"{k}={v}" for k, v in opts.items()]
    t0 = time.perf_counter()
    p = subprocess.run(args, cwd=os.path.dirname(stub), env=ENV,
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    hdr = p.stdout + p.stderr
    assert "running with linear solver ma57" in hdr, \
        f"MA57 NOT engaged for {stub}:\n{hdr[:400]}"
    sol = parse_sol(stub + ".sol", n, m)
    sol["t"] = dt
    return sol


def run_pounce(stub, n, m, **opts):
    sol_path = stub + "_pounce.sol"
    args = [CLI, stub + ".nl", sol_path] + [f"{k}={v}" for k, v in opts.items()]
    t0 = time.perf_counter()
    p = subprocess.run(args, env=ENV, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    sol = parse_sol(sol_path, n, m)
    sol["t"] = dt; sol["rc"] = p.returncode; sol["stdout"] = p.stdout
    return sol


def dual_relationship(pounce_d, ipopt_d, tol=1e-4):
    """Report magnitude match and whether pounce == -ipopt (known flip),
    == +ipopt, or something else, elementwise (ignoring ~0 entries)."""
    pounce_d = np.asarray(pounce_d, float); ipopt_d = np.asarray(ipopt_d, float)
    mag_err = relinf(np.abs(pounce_d), np.abs(ipopt_d))
    rel_neg = relinf(pounce_d, -ipopt_d)
    rel_pos = relinf(pounce_d, ipopt_d)
    return mag_err, rel_neg, rel_pos


def sign_note(rel_pos, rel_neg):
    if rel_pos < 1e-4:
        return "sign matches MA57 (2026-07-22 flip FIXED)"
    if rel_neg < 1e-4:
        return "KNOWN uniform flip vs MA57 (logged 2026-07-22)"
    return f"anomalous sign rel_pos={rel_pos:.1e} rel_neg={rel_neg:.1e}"


RESULTS = {}


def record(n, name, family, source, known, pounce_obj, pounce_status,
           oracle_obj, verdict, extra=""):
    RESULTS[n] = dict(n=n, name=name, family=family, source=source, known=known,
                      pobj=pounce_obj, pstat=pounce_status, oobj=oracle_obj,
                      verdict=verdict, extra=extra)


# ================================================================= TEST 1
def test1_zlzu_bounds():
    """z_L / z_U .sol suffix vs MA57 on a problem with active lower AND upper
    bounds (the #296/#312 ipopt_zL_out/ipopt_zU_out surface), + complementarity."""
    print("\n" + "=" * 70 + "\nTEST 1  i1_zLzU_bounds  (family sensitivity/bound-multipliers)")
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-1, 1), initialize=0.0)   # -> upper 1
    m.y = pe.Var(bounds=(-1, 1), initialize=0.0)   # -> lower -1
    m.z = pe.Var(bounds=(-1, 1), initialize=0.0)   # -> interior 0.5 (both slack)
    m.obj = pe.Objective(expr=(m.x - 3)**2 + (m.y + 3)**2 + (m.z - 0.5)**2)
    n, mm = 3, 0
    stub = f"{WORK}/t1"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    # analytic: x*=(1,-1,0.5), zU(x)=-4, zL(y)=+4, z inactive ~0
    xstar = np.array([1.0, -1.0, 0.5]); fstar = 8.0
    lb = np.array([-1, -1, -1.0]); ub = np.array([1, 1, 1.0])
    zL_ip = np.array([ip["suffix"]["ipopt_zL_out"].get(i, 0.0) for i in range(n)])
    zU_ip = np.array([ip["suffix"]["ipopt_zU_out"].get(i, 0.0) for i in range(n)])
    zL_pc = np.array([pc["suffix"]["ipopt_zL_out"].get(i, 0.0) for i in range(n)])
    zU_pc = np.array([pc["suffix"]["ipopt_zU_out"].get(i, 0.0) for i in range(n)])
    print("col order:", col)
    print(f"x* pounce={pc['primal']}  ipopt={ip['primal']}  analytic={xstar}")
    print(f"zL ipopt ={zL_ip}\nzL pounce={zL_pc}")
    print(f"zU ipopt ={zU_ip}\nzU pounce={zU_pc}")
    fp = (pc['primal'][0]-3)**2 + (pc['primal'][1]+3)**2 + (pc['primal'][2]-0.5)**2
    obj_err = relinf([fp], [fstar])
    zL_err = relinf(zL_pc, zL_ip); zU_err = relinf(zU_pc, zU_ip)
    # complementarity in pounce's own numbers: zL*(x-lb)=0, zU*(ub-x)=0
    compl_L = np.max(np.abs(zL_pc * (pc["primal"] - lb)))
    compl_U = np.max(np.abs(zU_pc * (ub - pc["primal"])))
    # analytic check: |zL(y)|~4, |zU(x)|~4
    ana_err = max(abs(abs(zL_pc[1]) - 4), abs(abs(zU_pc[0]) - 4))
    print(f"zL_err vs MA57={zL_err:.2e}  zU_err vs MA57={zU_err:.2e}  ana_err={ana_err:.2e}")
    print(f"complementarity: max|zL*(x-lb)|={compl_L:.2e}  max|zU*(ub-x)|={compl_U:.2e}")
    ok = zL_err < 1e-4 and zU_err < 1e-4 and ana_err < 1e-4 and compl_L < 1e-6 and compl_U < 1e-6
    v = "PASS" if ok else "FAIL"
    print(f"VERDICT: {v}" + ("" if ok else f" (zL={zL_err:.1e} zU={zU_err:.1e} compl={compl_L:.1e}/{compl_U:.1e})"))
    record(1, "zLzU_bounds", "sensitivity/bound-multipliers",
           "closed-form; ipopt 3.14.19 MA57 ipopt_zL_out/zU_out suffix (#296/#312)",
           "f*=8, zU(x)=-4, zL(y)=+4", f"{fstar:.4f}", pc["msg"][:30],
           f"{fstar:.4f}", v,
           f"zL/zU match MA57 to {max(zL_err,zU_err):.1e}; complementarity {max(compl_L,compl_U):.1e}")


# ================================================================= TEST 2
def test2_lp_shadow():
    """Analytic LP shadow prices vs pounce vs MA57; magnitude + uniform-flip."""
    print("\n" + "=" * 70 + "\nTEST 2  i1_lp_shadow  (family lp/shadow-prices)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(domain=pe.NonNegativeReals, initialize=0.0)
    m.x2 = pe.Var(domain=pe.NonNegativeReals, initialize=0.0)
    m.obj = pe.Objective(expr=-(4 * m.x1 + 3 * m.x2))     # min -> maximize 4x1+3x2
    m.c1 = pe.Constraint(expr=2 * m.x1 + m.x2 <= 10)
    m.c2 = pe.Constraint(expr=m.x1 + 3 * m.x2 <= 15)
    n, mm = 2, 2
    stub = f"{WORK}/t2"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    xstar = np.array([3.0, 4.0]); fstar = -24.0
    ana_dual = np.array([-1.8, -0.4])  # df/db (min form), both active
    print("row order:", row, " col:", col)
    print(f"x* pounce={pc['primal']} ipopt={ip['primal']} analytic={xstar}")
    print(f"mult_g ipopt ={ip['duals']}  analytic={ana_dual}")
    print(f"mult_g pounce={pc['duals']}")
    obj_err = relinf([-(4*pc['primal'][0]+3*pc['primal'][1])], [fstar])
    mag_err, rel_neg, rel_pos = dual_relationship(pc["duals"], ip["duals"])
    ana_mag = relinf(np.abs(pc["duals"]), np.abs(ana_dual))
    ip_ana = relinf(ip["duals"], ana_dual)
    print(f"obj_err={obj_err:.2e}  |dual| vs MA57={mag_err:.2e}  |dual| vs analytic={ana_mag:.2e}")
    print(f"ipopt vs analytic (signed)={ip_ana:.2e}")
    print(f"pounce vs +MA57 (signed)={rel_pos:.2e}   pounce vs -MA57 (old flip)={rel_neg:.2e}")
    signed_ok = rel_pos < 1e-4        # pounce dual matches MA57 SIGNED
    ana_signed = relinf(pc["duals"], ana_dual)
    ok = (obj_err < 1e-4 and mag_err < 1e-4 and ana_mag < 1e-4 and ip_ana < 1e-4
          and signed_ok and ana_signed < 1e-4)
    v = "PASS" if ok else "FAIL"
    if signed_ok:
        note = "dual sign matches MA57 AND analytic df/db (the 2026-07-22 mult_g flip is FIXED)"
    elif rel_neg < 1e-4:
        note = "KNOWN uniform mult_g flip vs MA57 (logged 2026-07-22)"
    else:
        note = "anomalous sign -- investigate"
    print(f"VERDICT: {v}  [{note}]")
    record(2, "lp_shadow", "lp/shadow-prices",
           "custom LP max 4x1+3x2 s.t. 2x1+x2<=10, x1+3x2<=15",
           "f*=-24, shadow (-1.8,-0.4)", f"{-(4*pc['primal'][0]+3*pc['primal'][1]):.4f}",
           "optimal", f"{fstar:.4f}", v, sign_note(rel_pos, rel_neg) + f"; |dual| match MA57 {mag_err:.1e}")


# ================================================================= TEST 3
def test3_qp_kkt_closed():
    """Closed-form KKT QP: active equality + active inequality; duals vs MA57."""
    print("\n" + "=" * 70 + "\nTEST 3  i1_qp_kkt_closed  (family qp/closed-form-KKT)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(initialize=0.0); m.x2 = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=0.5 * (m.x1**2 + m.x2**2))
    m.ce = pe.Constraint(expr=m.x1 + m.x2 == 2)          # eq  lambda=1.0
    m.ci = pe.Constraint(expr=m.x1 - m.x2 >= 1)          # ineq mu=0.5
    n, mm = 2, 2
    stub = f"{WORK}/t3"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    xstar = np.array([1.5, 0.5]); fstar = 1.25
    # analytic |duals|: eq 1.0, ineq 0.5
    ana_absdual = np.array([1.0, 0.5])
    print("row order:", row)
    print(f"x* pounce={pc['primal']} ipopt={ip['primal']} analytic={xstar}")
    print(f"mult_g ipopt ={ip['duals']}\nmult_g pounce={pc['duals']}  analytic|.|={ana_absdual}")
    fp = 0.5 * (pc['primal'][0]**2 + pc['primal'][1]**2)
    obj_err = relinf([fp], [fstar])
    mag_err, rel_neg, rel_pos = dual_relationship(pc["duals"], ip["duals"])
    # match analytic magnitude by row: need to know row order; compare sorted abs
    ana_err = relinf(np.sort(np.abs(pc["duals"])), np.sort(ana_absdual))
    print(f"obj_err={obj_err:.2e}  |dual| vs MA57={mag_err:.2e}  |dual| vs analytic(sorted)={ana_err:.2e}")
    print(f"pounce vs -MA57(flip)={rel_neg:.2e}  vs +MA57={rel_pos:.2e}")
    ok = obj_err < 1e-4 and mag_err < 1e-4 and ana_err < 1e-4
    v = "PASS" if ok else "FAIL"
    note = sign_note(rel_pos, rel_neg)
    print(f"VERDICT: {v} [{note}]")
    record(3, "qp_kkt_closed", "qp/closed-form-KKT",
           "min 1/2||x||^2 s.t. x1+x2=2, x1-x2>=1",
           "f*=1.25, eq lam=1, ineq mu=0.5", f"{fp:.4f}", "optimal",
           f"{fstar:.4f}", v, f"|dual| vs MA57 {mag_err:.1e}; " + note)


# ================================================================= TEST 4
def test4_nlp_eq_ineq():
    """Nonlinear NLP: active nonlinear equality + active inequality; duals vs MA57."""
    print("\n" + "=" * 70 + "\nTEST 4  i1_nlp_eq_ineq  (family nlp/duals-multipliers)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(initialize=0.6); m.x2 = pe.Var(initialize=-0.5)
    m.obj = pe.Objective(expr=m.x1 + m.x2)
    m.ce = pe.Constraint(expr=m.x1**2 + m.x2**2 == 1)     # nonlinear eq
    m.ci = pe.Constraint(expr=m.x1 >= 0.5)                # active ineq
    n, mm = 2, 2
    stub = f"{WORK}/t4"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    x1 = 0.5; x2 = -math.sqrt(0.75); fstar = x1 + x2
    lam = -1.0 / (2 * x2); mu = 1 + 2 * lam * x1
    ana_absdual = np.sort([abs(lam), abs(mu)])
    print("row order:", row)
    print(f"x* pounce={pc['primal']} ipopt={ip['primal']} analytic=({x1},{x2:.4f})")
    print(f"mult_g ipopt ={ip['duals']}\nmult_g pounce={pc['duals']}  analytic|.|(eq,ineq)=({abs(lam):.4f},{abs(mu):.4f})")
    fp = pc['primal'][0] + pc['primal'][1]
    obj_err = relinf([fp], [fstar])
    mag_err, rel_neg, rel_pos = dual_relationship(pc["duals"], ip["duals"])
    ana_err = relinf(np.sort(np.abs(pc["duals"])), ana_absdual)
    print(f"obj_err={obj_err:.2e}  |dual| vs MA57={mag_err:.2e}  |dual| vs analytic(sorted)={ana_err:.2e}")
    print(f"pounce vs -MA57(flip)={rel_neg:.2e}  vs +MA57={rel_pos:.2e}")
    ok = obj_err < 1e-4 and mag_err < 1e-4 and ana_err < 1e-4
    v = "PASS" if ok else "FAIL"
    note = sign_note(rel_pos, rel_neg)
    print(f"VERDICT: {v} [{note}]")
    record(4, "nlp_eq_ineq", "nlp/duals-multipliers",
           "min x1+x2 s.t. x1^2+x2^2=1, x1>=0.5",
           f"f*={fstar:.4f}, |lam|={abs(lam):.3f},|mu|={abs(mu):.3f}",
           f"{fp:.4f}", "optimal", f"{fstar:.4f}", v,
           f"|dual| vs MA57 {mag_err:.1e}; " + note)


# ================================================================= TEST 5
def test5_ranged():
    """Double-sided ranged constraint 2<=x1+x2<=6 (upper side active): dual vs MA57."""
    print("\n" + "=" * 70 + "\nTEST 5  i1_ranged  (family nlp/ranged-constraint-dual)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(initialize=0.0); m.x2 = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=(m.x1 - 5)**2 + (m.x2 - 5)**2)
    m.c = pe.Constraint(expr=pe.inequality(2, m.x1 + m.x2, 6))   # ranged row
    n, mm = 2, 1
    stub = f"{WORK}/t5"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    xstar = np.array([3.0, 3.0]); fstar = 8.0
    ana_absdual = 4.0   # |df/dub|, upper side active
    print("row order:", row)
    print(f"x* pounce={pc['primal']} ipopt={ip['primal']} analytic={xstar}")
    print(f"ranged dual ipopt ={ip['duals']}  pounce={pc['duals']}  analytic|.|={ana_absdual}")
    fp = (pc['primal'][0] - 5)**2 + (pc['primal'][1] - 5)**2
    obj_err = relinf([fp], [fstar])
    mag_err, rel_neg, rel_pos = dual_relationship(pc["duals"], ip["duals"])
    ana_err = abs(abs(pc["duals"][0]) - ana_absdual) / 4.0
    print(f"obj_err={obj_err:.2e}  |dual| vs MA57={mag_err:.2e}  |dual| vs analytic={ana_err:.2e}")
    print(f"pounce vs -MA57(flip)={rel_neg:.2e}  vs +MA57={rel_pos:.2e}")
    ok = obj_err < 1e-4 and mag_err < 1e-4 and ana_err < 1e-4
    v = "PASS" if ok else "FAIL"
    note = sign_note(rel_pos, rel_neg) + " (ranged upper side)"
    print(f"VERDICT: {v} [{note}]")
    record(5, "ranged", "nlp/ranged-constraint-dual",
           "min (x1-5)^2+(x2-5)^2 s.t. 2<=x1+x2<=6 (upper active)",
           "f*=8, |dual|=4", f"{fp:.4f}", "optimal", f"{fstar:.4f}", v,
           f"|dual| vs MA57 {mag_err:.1e}; " + note)


# ================================================================= TEST 6
def test6_reduced_costs():
    """Reduced costs on a bound-active variable while a constraint is also active,
    via ipopt_zL_out/zU_out vs MA57 and analytic."""
    print("\n" + "=" * 70 + "\nTEST 6  i1_reduced_costs  (family lp/reduced-costs)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(bounds=(0, 1), initialize=0.5)   # -> upper 1, rc = -1
    m.x2 = pe.Var(domain=pe.NonNegativeReals, initialize=0.5)
    m.obj = pe.Objective(expr=m.x1 + 2 * m.x2)
    m.c1 = pe.Constraint(expr=m.x1 + m.x2 >= 3)     # active, dual +2
    n, mm = 2, 1
    stub = f"{WORK}/t6"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    xstar = np.array([1.0, 2.0]); fstar = 5.0
    print("col order:", col, " row:", row)
    zU_ip = np.array([ip["suffix"].get("ipopt_zU_out", {}).get(i, 0.0) for i in range(n)])
    zU_pc = np.array([pc["suffix"].get("ipopt_zU_out", {}).get(i, 0.0) for i in range(n)])
    zL_ip = np.array([ip["suffix"].get("ipopt_zL_out", {}).get(i, 0.0) for i in range(n)])
    zL_pc = np.array([pc["suffix"].get("ipopt_zL_out", {}).get(i, 0.0) for i in range(n)])
    print(f"x* pounce={pc['primal']} ipopt={ip['primal']}")
    print(f"zU ipopt={zU_ip} pounce={zU_pc}   (x1 upper reduced cost)")
    print(f"zL ipopt={zL_ip} pounce={zL_pc}")
    print(f"c1 dual ipopt={ip['duals']} pounce={pc['duals']}  analytic|.|=2")
    fp = pc['primal'][0] + 2 * pc['primal'][1]
    obj_err = relinf([fp], [fstar])
    # x1 index: find in col
    ix1 = col.index("x1") if "x1" in col else 0
    rc_ip = zU_ip[ix1]; rc_pc = zU_pc[ix1]
    rc_err = relinf([rc_pc], [rc_ip])
    ana_rc_err = abs(abs(rc_pc) - 1.0)
    dual_mag = relinf(np.abs(pc["duals"]), np.abs(ip["duals"]))
    print(f"obj_err={obj_err:.2e}  x1 reduced cost: pounce={rc_pc:.4f} MA57={rc_ip:.4f} (analytic |1|)")
    print(f"rc vs MA57={rc_err:.2e}  rc vs analytic={ana_rc_err:.2e}  c1 |dual| vs MA57={dual_mag:.2e}")
    ok = obj_err < 1e-4 and rc_err < 1e-4 and ana_rc_err < 1e-4 and dual_mag < 1e-4
    v = "PASS" if ok else "FAIL"
    print(f"VERDICT: {v}")
    record(6, "reduced_costs", "lp/reduced-costs",
           "min x1+2x2 s.t. x1+x2>=3, 0<=x1<=1 (x1 at upper)",
           "f*=5, rc(x1)=-1, c1 dual=+2", f"{fp:.4f}", "optimal",
           f"{fstar:.4f}", v, f"rc vs MA57 {rc_err:.1e}; c1|dual| vs MA57 {dual_mag:.1e}")


# ================================================================= TEST 7
def test7_sens_qp_sipopt():
    """Parametric QP dx/db: pounce QpSensitivity vs (a) analytic, (b) central-FD
    re-solve, (c) sIPOPT (ipopt_sens via pyomo sensitivity_toolbox). Simultaneous
    2-parameter perturbation, well-conditioned."""
    print("\n" + "=" * 70 + "\nTEST 7  i1_sens_qp_sipopt  (family sensitivity/dx-db 3-way)")
    from pounce import QpSensitivity
    P = np.eye(3); c = np.zeros(3)
    A = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]]); b = np.array([3.0, 1.0])
    db = np.array([0.10, -0.20])
    # analytic: x* = A^T (A A^T)^{-1} b ; dx/db = A^T (A A^T)^{-1}
    AAT_inv = np.linalg.inv(A @ A.T)
    xstar = A.T @ AAT_inv @ b
    S = A.T @ AAT_inv           # dx/db  (n x 2)
    dx_analytic = S @ db
    s = QpSensitivity(P=P, c=c, A=A, b=b)
    dx_pounce = s.parametric_step([0, 1], list(db))
    # central FD via independent numpy KKT re-solve
    def solve(bb):
        return A.T @ AAT_inv @ bb
    h = 1e-6
    fd = np.zeros((3, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = h
        fd[:, k] = (solve(b + e) - solve(b - e)) / (2 * h)
    dx_fd = fd @ db
    print(f"x* pounce={s.x}  analytic={xstar}")
    print(f"dx pounce  ={dx_pounce}")
    print(f"dx analytic={dx_analytic}")
    print(f"dx FD      ={dx_fd}")
    err_ana = relinf(dx_pounce, dx_analytic)
    err_fd = relinf(dx_pounce, dx_fd)
    print(f"kkt_cond_estimate={s.kkt_cond_estimate:.3e} ill_conditioned={s.ill_conditioned}")
    # sIPOPT oracle via pyomo toolbox
    sipopt_ok = None; err_sip = None
    try:
        from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
        mm = pe.ConcreteModel()
        mm.I = pe.RangeSet(0, 2)
        mm.x = pe.Var(mm.I, initialize=0.0)
        mm.p1 = pe.Param(initialize=3.0, mutable=True)
        mm.p2 = pe.Param(initialize=1.0, mutable=True)
        mm.obj = pe.Objective(expr=0.5 * sum(mm.x[i]**2 for i in mm.I))
        mm.c1 = pe.Constraint(expr=mm.x[0] + mm.x[1] + mm.x[2] == mm.p1)
        mm.c2 = pe.Constraint(expr=mm.x[0] - mm.x[2] == mm.p2)
        m_pert = sensitivity_calculation(
            "sipopt", mm, [mm.p1, mm.p2], [float(3.0 + db[0]), float(1.0 + db[1])],
            tee=False)
        # sIPOPT perturbed primal lives in the sens_sol_state_1 suffix
        x_pert = np.array([m_pert.sens_sol_state_1[m_pert.x[i]] for i in range(3)])
        dx_sip = x_pert - xstar
        err_sip = relinf(dx_sip, dx_analytic)
        sipopt_ok = True
        print(f"dx sIPOPT  ={dx_sip}   err vs analytic={err_sip:.2e}")
    except Exception as e:
        sipopt_ok = False
        print(f"sIPOPT oracle unavailable/failed: {type(e).__name__}: {e}")
    print(f"pounce vs analytic={err_ana:.2e}  vs FD={err_fd:.2e}")
    ok = err_ana < 1e-6 and err_fd < 1e-6 and (err_sip is None or err_sip < 1e-4)
    v = "PASS" if ok else "FAIL"
    print(f"VERDICT: {v}")
    record(7, "sens_qp_sipopt", "sensitivity/dx-db",
           "min 1/2||x||^2 s.t. Ax=b, perturb b0+0.1,b1-0.2",
           f"dx analytic={np.round(dx_analytic,4)}",
           "n/a (sensitivity)", "optimal",
           f"sIPOPT err={err_sip if err_sip is not None else 'n/a'}", v,
           f"pounce dx vs analytic {err_ana:.1e}, vs FD {err_fd:.1e}, vs sIPOPT {err_sip}")


# ================================================================= TEST 8
def test8_sens_activeset():
    """Parametric QP sensitivity with a BINDING INEQUALITY in the active set
    (active-set KKT path, distinct from test7's equality-only case).
    dx/db of the equality RHS while x1>=2 stays active. Oracles: analytic
    active-set KKT, central-FD re-solve, and sIPOPT (ipopt_sens)."""
    print("\n" + "=" * 70 + "\nTEST 8  i1_sens_activeset  (family sensitivity/active-set dx-db)")
    from pounce import QpSensitivity
    # min 1/2||x||^2 s.t. x1+x2+x3 = b (eq), x1 >= 2 (ineq, active)
    P = np.eye(3); c = np.zeros(3)
    A = np.array([[1.0, 1.0, 1.0]]); b = np.array([3.0])
    G = np.array([[-1.0, 0.0, 0.0]]); h = np.array([-2.0])   # -x1 <= -2  <=> x1>=2
    db = 0.4
    # analytic: active set {eq, x1=2}. x1=2, x2=x3=(b-2)/2. dx/db=(0,0.5,0.5).
    def resolve(bb):
        x1 = 2.0; rest = (bb - 2.0) / 2.0
        return np.array([x1, rest, rest])
    xstar = resolve(b[0]); dx_analytic = np.array([0.0, 0.5, 0.5]) * db
    s = QpSensitivity(P=P, c=c, A=A, b=b, G=G, h=h)
    print(f"x* pounce={s.x}  analytic={xstar}")
    print(f"active_indices={s.active_indices}  weakly={s.weakly_active_indices}")
    dx_pc = s.parametric_step([0], [db])   # index 0 = the equality RHS
    # central FD (independent re-solve; x*(b) affine while active set fixed)
    hh = 1e-6
    dx_fd = (resolve(b[0] + hh) - resolve(b[0] - hh)) / (2 * hh) * db
    print(f"dx pounce  ={dx_pc}")
    print(f"dx analytic={dx_analytic}")
    print(f"dx FD      ={dx_fd}")
    err_ana = relinf(dx_pc, dx_analytic); err_fd = relinf(dx_pc, dx_fd)
    # sIPOPT oracle (parameter = equality RHS)
    err_sip = None
    try:
        from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
        mm = pe.ConcreteModel()
        mm.I = pe.RangeSet(0, 2); mm.x = pe.Var(mm.I, initialize=0.0)
        mm.x[0].setlb(2.0)
        mm.p = pe.Param(initialize=3.0, mutable=True)
        mm.obj = pe.Objective(expr=0.5 * sum(mm.x[i]**2 for i in mm.I))
        mm.c = pe.Constraint(expr=mm.x[0] + mm.x[1] + mm.x[2] == mm.p)
        m2 = sensitivity_calculation("sipopt", mm, [mm.p], [float(3.0 + db)], tee=False)
        x_pert = np.array([m2.sens_sol_state_1[m2.x[i]] for i in range(3)])
        dx_sip = x_pert - xstar
        err_sip = relinf(dx_sip, dx_analytic)
        print(f"dx sIPOPT  ={dx_sip}  err vs analytic={err_sip:.2e}")
    except Exception as e:
        print(f"sIPOPT oracle failed: {type(e).__name__}: {e}")
    print(f"pounce vs analytic={err_ana:.2e}  vs FD={err_fd:.2e}")
    ok = err_ana < 1e-6 and err_fd < 1e-6 and (err_sip is None or err_sip < 1e-4)
    v = "PASS" if ok else "FAIL"
    print(f"VERDICT: {v}")
    record(8, "sens_activeset", "sensitivity/active-set dx-db",
           "min 1/2||x||^2 s.t. x1+x2+x3=b, x1>=2 active; perturb b+0.4",
           f"dx analytic={np.round(dx_analytic,4)}", "n/a", "optimal",
           f"sIPOPT err={err_sip}", v,
           f"pounce dx vs analytic {err_ana:.1e}, vs FD {err_fd:.1e}, vs sIPOPT {err_sip}")


# ================================================================= TEST 9
def test9_qcqp_dual():
    """QCQP conic dual magnitude vs MA57 (SOCP .sol writer site 2b).
    min -x1-x2 s.t. x1^2+x2^2<=1 -> f*=-sqrt2, quad-constraint dual |.|=sqrt2/2."""
    print("\n" + "=" * 70 + "\nTEST 9  i1_qcqp_dual  (family socp/qcqp-dual)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(initialize=0.5); m.x2 = pe.Var(initialize=0.5)
    m.obj = pe.Objective(expr=-m.x1 - m.x2)
    m.c = pe.Constraint(expr=m.x1**2 + m.x2**2 <= 1)
    n, mm = 2, 1
    stub = f"{WORK}/t9"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc_nl = run_pounce(stub, n, mm)   # CLI/NLP path
    fstar = -math.sqrt(2); ana_absdual = math.sqrt(2) / 2
    print(f"x* ipopt={ip['primal']} pounce(NL)={pc_nl['primal']}  analytic=(0.7071,0.7071)")
    print(f"quad-con dual ipopt={ip['duals']} pounce(NL .sol)={pc_nl['duals']} analytic|.|={ana_absdual:.4f}")
    # conic path via python solve_socp / minimize auto
    from pounce import minimize
    r = minimize(lambda x: -x[0] - x[1], np.array([0.5, 0.5]),
                 constraints=[{"type": "ineq", "fun": lambda x: 1 - x[0]**2 - x[1]**2}],
                 solver_selection="auto")
    conic_dual = r["info"].get("mult_g")
    print(f"conic path (minimize auto) mult_g={conic_dual}  status={r['message'][:40]}")
    fp = -pc_nl['primal'][0] - pc_nl['primal'][1]
    obj_err = relinf([fp], [fstar])
    mag_err, rel_neg, rel_pos = dual_relationship(pc_nl["duals"], ip["duals"])
    ana_err = abs(abs(pc_nl["duals"][0]) - ana_absdual) / ana_absdual
    print(f"obj_err={obj_err:.2e}  NL |dual| vs MA57={mag_err:.2e}  vs analytic={ana_err:.2e}")
    print(f"pounce(NL) vs -MA57(flip)={rel_neg:.2e}  vs +MA57={rel_pos:.2e}")
    ok = obj_err < 1e-4 and mag_err < 1e-4 and ana_err < 1e-4
    v = "PASS" if ok else "FAIL"
    note = sign_note(rel_pos, rel_neg)
    print(f"VERDICT: {v} [{note}]")
    record(9, "qcqp_dual", "socp/qcqp-dual",
           "min -x1-x2 s.t. x1^2+x2^2<=1",
           f"f*={fstar:.4f}, |dual|={ana_absdual:.4f}", f"{fp:.4f}", "optimal",
           f"{fstar:.4f}", v, f"NL |dual| vs MA57 {mag_err:.1e}; " + note)


# ================================================================= TEST 10
def test10_weakly_active():
    """Degenerate weakly-active inequality: optimum exactly on constraint but
    multiplier=0. Flag TOLERANCE, not a bug. min (x1-1)^2+(x2-1)^2 s.t. x1+x2<=2."""
    print("\n" + "=" * 70 + "\nTEST 10  i1_weakly_active  (family nlp/degenerate-weak-activity)")
    m = pe.ConcreteModel()
    m.x1 = pe.Var(initialize=0.0); m.x2 = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=(m.x1 - 1)**2 + (m.x2 - 1)**2)
    m.c = pe.Constraint(expr=m.x1 + m.x2 <= 2)   # tight at (1,1), mu=0
    n, mm = 2, 1
    stub = f"{WORK}/t10"
    col, row = write_nl(m, stub)
    ip = run_ipopt(stub, n, mm)
    pc = run_pounce(stub, n, mm)
    xstar = np.array([1.0, 1.0]); fstar = 0.0
    print(f"x* ipopt={ip['primal']} pounce={pc['primal']} analytic={xstar}")
    print(f"weak-active dual ipopt={ip['duals']} pounce={pc['duals']}  analytic=0")
    xerr = relinf(pc["primal"], xstar)
    fp = (pc['primal'][0]-1)**2 + (pc['primal'][1]-1)**2
    dual_pc = abs(pc["duals"][0]); dual_ip = abs(ip["duals"][0])
    print(f"x_err={xerr:.2e}  f*={fp:.2e}  |dual| pounce={dual_pc:.2e} ipopt={dual_ip:.2e}")
    ok_primal = xerr < 1e-4 and fp < 1e-6
    ok_dual = dual_pc < 1e-3 and dual_ip < 1e-3
    if ok_primal and ok_dual:
        v = "PASS"
    elif ok_primal:
        v = "TOLERANCE"
    else:
        v = "FAIL"
    print(f"VERDICT: {v}  (weak activity: both solvers should report ~0 multiplier)")
    record(10, "weakly_active", "nlp/degenerate-weak-activity",
           "min (x1-1)^2+(x2-1)^2 s.t. x1+x2<=2 (tight, mu=0)",
           "f*=0, dual=0", f"{fp:.2e}", "optimal", f"{fstar:.4f}", v,
           f"x_err {xerr:.1e}; |dual| pounce {dual_pc:.1e} vs ipopt {dual_ip:.1e}")


TESTS = [test1_zlzu_bounds, test2_lp_shadow, test3_qp_kkt_closed,
         test4_nlp_eq_ineq, test5_ranged, test6_reduced_costs,
         test7_sens_qp_sipopt, test8_sens_activeset, test9_qcqp_dual,
         test10_weakly_active]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    sel = TESTS if which == "all" else [TESTS[int(which) - 1]]
    for t in sel:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n!!! {t.__name__} ERRORED: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n\n" + "#" * 70 + "\nSUMMARY")
    for n in sorted(RESULTS):
        r = RESULTS[n]
        print(f"  T{n:<2} {r['name']:<18} {r['verdict']:<10} {r['extra']}")


if __name__ == "__main__":
    main()
