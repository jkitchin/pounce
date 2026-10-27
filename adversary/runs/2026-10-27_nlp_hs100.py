"""Adversary cross-check: Hock-Schittkowski problem 100
Family: nlp   Class: inequality-constrained, nonconvex polynomial (degree 6 objective)
Source: Hock & Schittkowski, Test Examples for Nonlinear Programming Codes,
        LNEMS 187 (1981), problem 100, p. 111.  x0 = (1,2,0,4,0,1,1).
        f* = 680.6300573,
        x* = (2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227).
Path: Pyomo model -> .nl -> pounce (SolverFactory, nl tape) and Ipopt; pounce verify.
Known optimal: 680.6300573
"""
import json, subprocess, time, numpy as np, pyomo.environ as pe
KNOWN = 680.6300573
KX = np.array([2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227])
def build():
    m = pe.ConcreteModel(); x0 = [1, 2, 0, 4, 0, 1, 1]
    m.x = pe.Var(range(1, 8), initialize=lambda m, i: x0[i - 1]); x = m.x
    m.obj = pe.Objective(expr=(x[1] - 10) ** 2 + 5 * (x[2] - 12) ** 2 + x[3] ** 4 + 3 * (x[4] - 11) ** 2
                         + 10 * x[5] ** 6 + 7 * x[6] ** 2 + x[7] ** 4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7])
    m.c1 = pe.Constraint(expr=127 - 2 * x[1] ** 2 - 3 * x[2] ** 4 - x[3] - 4 * x[4] ** 2 - 5 * x[5] >= 0)
    m.c2 = pe.Constraint(expr=282 - 7 * x[1] - 3 * x[2] - 10 * x[3] ** 2 - x[4] + x[5] >= 0)
    m.c3 = pe.Constraint(expr=196 - 23 * x[1] - x[2] ** 2 - 6 * x[6] ** 2 + 8 * x[7] >= 0)
    m.c4 = pe.Constraint(expr=-4 * x[1] ** 2 - x[2] ** 2 + 3 * x[1] * x[2] - 2 * x[3] ** 2 - 5 * x[6] + 11 * x[7] >= 0)
    return m
out = {}
for name in ("pounce", "ipopt"):
    m = build(); s = pe.SolverFactory(name)
    if name == "ipopt": s.options["tol"] = 1e-8
    t0 = time.perf_counter(); r = s.solve(m); t = time.perf_counter() - t0
    xv = np.array([pe.value(m.x[i]) for i in range(1, 8)])
    out[name] = (str(r.solver.termination_condition), pe.value(m.obj), xv, t)
d = "adversary/runs/scratch_1027"; m = build(); nl = f"{d}/hs100.nl"; m.write(nl, io_options={"symbolic_solver_labels": False})
js = f"{d}/hs100.json"; sol = f"{d}/hs100.sol"
t0 = time.perf_counter(); subprocess.run(["target/release/pounce", nl, sol, "--json-output", js], capture_output=True); tc = time.perf_counter() - t0
J = json.load(open(js)); it = J.get("statistics", {}).get("iterations", J.get("iterations"))
v = subprocess.run(["target/release/pounce", "verify", nl, sol], capture_output=True, text=True)
rel = lambda u, w: abs(u - w) / max(1, abs(w))
for k, (st, ob, xv, t) in out.items():
    print(f"{k:7s} term={st} obj={ob:.10f} err_known={rel(ob,KNOWN):.1e} x_inf_vs_HS={np.abs(xv-KX).max():.1e} t={t:.3f}s")
print(f"CLI status={J.get('status')} obj={J['solution'].get('objective')} engine={J['solution'].get('engine')} iters={it} t={tc:.3f}s; pounce verify exit={v.returncode}")
print(f"pounce-vs-ipopt obj err={rel(out['pounce'][1], out['ipopt'][1]):.1e} x_inf={np.abs(out['pounce'][2]-out['ipopt'][2]).max():.1e}")
ok = out["pounce"][0] == "optimal" and rel(out["pounce"][1], KNOWN) < 1e-6 and v.returncode == 0 and rel(out["pounce"][1], out["ipopt"][1]) < 1e-6
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
