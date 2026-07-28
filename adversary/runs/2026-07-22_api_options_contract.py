"""Adversary probe: THE OPTIONS DICTIONARY CONTRACT.

Family: api (contracts / option handling / input edge cases)
Surfaces: pounce.minimize, pounce.solve_qp, pounce.solve_socp, CLI, Pyomo plugin.

Oracle: the DOCUMENTED contract (docs/src/options.md, docs/src/python.md,
docstrings) plus measurable behavioral evidence (final KKT error, iteration
count, exit status).

Probes:
 (a) unknown option key -> rejected or silently ignored?
 (b) documented options take effect (tol, max_iter, solver_selection)
 (c) out-of-range values (tol=0/-1/1e300, max_iter=0/-5/1e12)
 (d) type errors (str where float expected, None, list)
 (e) CLI vs Python vs Pyomo agreement
"""

import os
import subprocess
import sys
import time
import warnings

import numpy as np

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
FINDINGS = []
BUDGET = 10.0


def record(tag, name, detail):
    FINDINGS.append((tag, name, detail))
    print(f"  [{tag}] {name}: {detail}")


def hdr(s):
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


# --------------------------------------------------------------------------
# Test problem: HS071 (Hock-Schittkowski 71). Known optimum f* = 17.0140173.
# --------------------------------------------------------------------------
def hs071():
    def fun(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def jac(x):
        return np.array(
            [
                x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    cons = [
        {
            "type": "ineq",
            "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0,
            "jac": lambda x: np.array(
                [
                    x[1] * x[2] * x[3],
                    x[0] * x[2] * x[3],
                    x[0] * x[1] * x[3],
                    x[0] * x[1] * x[2],
                ]
            ),
        },
        {
            "type": "eq",
            "fun": lambda x: np.sum(x**2) - 40.0,
            "jac": lambda x: 2.0 * x,
        },
    ]
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    bounds = [(1.0, 5.0)] * 4
    return fun, jac, cons, x0, bounds


HS071_OPT = 17.01401724556958


def run_min(**kw):
    """Run minimize on HS071 with extra options; return (res, warnings)."""
    import pounce

    fun, jac, cons, x0, bounds = hs071()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = pounce.minimize(
            fun, x0, jac=jac, bounds=bounds, constraints=cons, **kw
        )
    return res, [str(x.message) for x in w]


# --------------------------------------------------------------------------
def probe_a_unknown_keys():
    hdr("(a) UNKNOWN OPTION KEYS")
    import pounce

    # A typo of a REAL option: max_iters vs max_iter.
    cases = [
        ("max_iters", 1),           # typo of max_iter
        ("tolerance", 1e-12),       # typo of tol
        ("this_is_not_an_option", 3),
        ("solverselection", "auto"),
    ]
    for key, val in cases:
        try:
            res, warns = run_min(**{key: val})
            got = f"NO ERROR (success={res.success}, nit={res.nit}, warns={warns})"
            tag = "SILENT-IGNORE"
        except Exception as e:
            got = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
            tag = "REJECTED"
        record(tag, f"minimize({key}={val!r})", got)

    # Same via the legacy options= dict form
    try:
        res, warns = run_min(options={"max_iters": 1})
        record("SILENT-IGNORE", "minimize(options={'max_iters':1})",
               f"NO ERROR nit={res.nit}")
    except Exception as e:
        record("REJECTED", "minimize(options={'max_iters':1})",
               f"{type(e).__name__}: {str(e).splitlines()[0][:140]}")

    # Unknown kwarg on solve_qp / solve_socp
    P = np.eye(2)
    c = np.array([-1.0, -1.0])
    for fname, fn, extra in [
        ("solve_qp", pounce.solve_qp, dict(P=P, c=c)),
        ("solve_socp", pounce.solve_socp,
         dict(P=P, c=c, G=np.eye(2), h=np.ones(2), cones=[("nonneg", 2)])),
    ]:
        try:
            fn(max_iters=1, **extra)
            record("SILENT-IGNORE", f"{fname}(max_iters=1)", "NO ERROR")
        except TypeError as e:
            record("REJECTED", f"{fname}(max_iters=1)", f"TypeError: {e}")
        except Exception as e:
            record("OTHER", f"{fname}(max_iters=1)", f"{type(e).__name__}: {e}")

    # CLI unknown option on an .nl file
    nl = write_hs071_nl()
    for opt in ["max_iters=1", "not_an_option=3"]:
        p = subprocess.run([CLI, nl, opt], capture_output=True, text=True, timeout=BUDGET)
        tail = (p.stdout + p.stderr).strip().splitlines()
        tail = tail[-1] if tail else ""
        record("REJECTED" if p.returncode != 0 else "SILENT-IGNORE",
               f"CLI {opt}", f"rc={p.returncode} last={tail[:120]!r}")


# --------------------------------------------------------------------------
def probe_b_effect():
    hdr("(b) DOCUMENTED OPTIONS TAKE EFFECT")
    # tol: tighter tol => smaller final KKT error
    prev = None
    for tol in (1e-4, 1e-8, 1e-12):
        res, _ = run_min(tol=tol)
        kkt = float(res.info.get("final_kkt_error", float("nan")))
        print(f"  tol={tol:<8g} nit={res.nit:<4d} kkt={kkt:.3e} "
              f"f={res.fun:.12f} success={res.success}")
        if prev is not None and np.isfinite(kkt) and np.isfinite(prev):
            if kkt > prev * 10:
                record("SUSPECT", f"tol={tol}", "KKT error grew with tighter tol")
        prev = kkt
    res_loose, _ = run_min(tol=1e-4)
    res_tight, _ = run_min(tol=1e-12)
    k_loose = float(res_loose.info["final_kkt_error"])
    k_tight = float(res_tight.info["final_kkt_error"])
    if k_tight < k_loose:
        record("OK", "tol takes effect", f"{k_loose:.2e} -> {k_tight:.2e}")
    else:
        record("SOLVER_BUG?", "tol has no effect",
               f"loose kkt={k_loose:.2e} tight kkt={k_tight:.2e}")

    # max_iter=1 => iteration-limit status
    res, _ = run_min(max_iter=1)
    print(f"  max_iter=1 -> nit={res.nit} status={res.status} "
          f"success={res.success} msg={res.message!r}")
    if res.nit <= 1 and not res.success:
        record("OK", "max_iter=1", f"stopped at nit={res.nit}, status={res.status}")
    else:
        record("SOLVER_BUG?", "max_iter=1 not honored",
               f"nit={res.nit} success={res.success}")

    # scipy alias maxiter
    res, _ = run_min(maxiter=1)
    record("OK" if res.nit <= 1 else "SOLVER_BUG?", "maxiter=1 (scipy alias)",
           f"nit={res.nit} status={res.status}")

    # solver_selection routing
    import pounce
    P = np.array([[2.0, 0.0], [0.0, 2.0]])

    def qf(x):
        return float(x @ P @ x / 2 - np.array([1.0, 2.0]) @ x)

    def qg(x):
        return P @ x - np.array([1.0, 2.0])

    for sel in ("nlp", "auto", "qp-ipm"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = pounce.minimize(qf, np.zeros(2), jac=qg, solver_selection=sel)
        got = r.info.get("solver", "?")
        print(f"  solver_selection={sel:<10} -> info['solver']={got!r} "
              f"x={np.round(r.x, 8)} f={r.fun:.10f}")
    r_nlp = pounce.minimize(qf, np.zeros(2), jac=qg, solver_selection="nlp")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_qp = pounce.minimize(qf, np.zeros(2), jac=qg, solver_selection="qp-ipm")
    if r_nlp.info.get("solver") == r_qp.info.get("solver"):
        record("SOLVER_BUG?", "solver_selection did not route",
               f"both report {r_nlp.info.get('solver')!r}")
    else:
        record("OK", "solver_selection routes",
               f"nlp->{r_nlp.info.get('solver')!r} qp-ipm->{r_qp.info.get('solver')!r}")


# --------------------------------------------------------------------------
def probe_c_out_of_range():
    hdr("(c) OUT-OF-RANGE / NONSENSICAL VALUES")
    import pounce

    cases = [
        dict(tol=0.0), dict(tol=-1.0), dict(tol=1e300), dict(tol=float("inf")),
        dict(tol=float("nan")),
        dict(max_iter=0), dict(max_iter=-5), dict(max_iter=10**12),
        dict(max_iter=2**63), dict(print_level=99), dict(print_level=-3),
    ]
    for kw in cases:
        t0 = time.perf_counter()
        try:
            res, _ = run_min(**kw)
            dt = time.perf_counter() - t0
            out = (f"NO ERROR nit={res.nit} status={res.status} "
                   f"f={res.fun:.6g} success={res.success} t={dt:.2f}s")
            tag = "ACCEPTED"
            if dt > BUDGET * 0.8:
                tag = "SLOW"
        except Exception as e:
            dt = time.perf_counter() - t0
            out = f"{type(e).__name__}: {str(e).splitlines()[0][:130]} (t={dt:.2f}s)"
            tag = "REJECTED"
        record(tag, f"minimize({kw})", out)

    # solve_qp direct
    P = np.eye(2)
    c = np.array([-1.0, -1.0])
    for kw in [dict(tol=0.0), dict(tol=-1.0), dict(tol=1e300),
               dict(max_iter=0), dict(max_iter=-5), dict(max_iter=10**12)]:
        t0 = time.perf_counter()
        try:
            r = pounce.solve_qp(P=P, c=c, **kw)
            dt = time.perf_counter() - t0
            record("SLOW" if dt > BUDGET * 0.8 else "ACCEPTED",
                   f"solve_qp({kw})",
                   f"status={r.status} iters={r.iters} obj={r.obj:.6g} t={dt:.2f}s")
        except Exception as e:
            record("REJECTED", f"solve_qp({kw})",
                   f"{type(e).__name__}: {str(e)[:130]}")

    # solve_socp direct
    for kw in [dict(tol=-1.0), dict(max_iter=0), dict(max_iter=-5)]:
        try:
            r = pounce.solve_socp(P=P, c=c, G=np.eye(2), h=np.ones(2),
                                  cones=[("nonneg", 2)], **kw)
            record("ACCEPTED", f"solve_socp({kw})",
                   f"status={r.status} iters={r.iters} obj={r.obj:.6g}")
        except Exception as e:
            record("REJECTED", f"solve_socp({kw})",
                   f"{type(e).__name__}: {str(e)[:130]}")


# --------------------------------------------------------------------------
def probe_d_types():
    hdr("(d) OPTION TYPE ERRORS")
    import pounce

    cases = [
        dict(tol="tight"),
        dict(tol=None),
        dict(tol=[1e-8]),
        dict(max_iter="many"),
        dict(max_iter=3.7),
        dict(max_iter=None),
        dict(max_iter=[10]),
        dict(mu_strategy=3),
        dict(print_level="loud"),
        dict(solver_selection=None),
        dict(solver_selection=["auto"]),
    ]
    for kw in cases:
        try:
            res, _ = run_min(**kw)
            record("ACCEPTED", f"minimize({kw})",
                   f"NO ERROR nit={res.nit} f={res.fun:.6g} status={res.status}")
        except Exception as e:
            record("REJECTED", f"minimize({kw})",
                   f"{type(e).__name__}: {str(e).splitlines()[0][:130]}")

    P = np.eye(2)
    c = np.array([-1.0, -1.0])
    for kw in [dict(tol="tight"), dict(max_iter="many"), dict(max_iter=3.7),
               dict(tol=[1e-8])]:
        try:
            r = pounce.solve_qp(P=P, c=c, **kw)
            record("ACCEPTED", f"solve_qp({kw})",
                   f"status={r.status} iters={r.iters}")
        except Exception as e:
            record("REJECTED", f"solve_qp({kw})",
                   f"{type(e).__name__}: {str(e)[:130]}")


# --------------------------------------------------------------------------
NL_PATH = "/private/tmp/claude-501/-Users-jkitchin-projects-pounce/671a5f76-82be-4f1a-bac6-59f0cb187d8b/scratchpad/hs071.nl"


def write_hs071_nl():
    """Write HS071 to .nl via Pyomo (cached)."""
    if os.path.exists(NL_PATH):
        return NL_PATH
    import pyomo.environ as pyo

    m = build_pyomo_hs071()
    m.write(NL_PATH, io_options={"symbolic_solver_labels": False})
    return NL_PATH


def build_pyomo_hs071():
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(1, 4)
    init = {1: 1.0, 2: 5.0, 3: 5.0, 4: 1.0}
    m.x = pyo.Var(m.I, bounds=(1.0, 5.0), initialize=init)
    m.obj = pyo.Objective(
        expr=m.x[1] * m.x[4] * (m.x[1] + m.x[2] + m.x[3]) + m.x[3]
    )
    m.c1 = pyo.Constraint(expr=m.x[1] * m.x[2] * m.x[3] * m.x[4] >= 25.0)
    m.c2 = pyo.Constraint(
        expr=sum(m.x[i] ** 2 for i in m.I) == 40.0
    )
    return m


def probe_e_surfaces():
    hdr("(e) CLI vs PYTHON vs PYOMO AGREEMENT")
    nl = write_hs071_nl()

    # --- CLI ---
    for opts, label in [([], "defaults"), (["max_iter=1"], "max_iter=1"),
                        (["tol=1e-12"], "tol=1e-12"),
                        (["max_iter=0"], "max_iter=0"),
                        (["max_iter=-5"], "max_iter=-5"),
                        (["tol=0"], "tol=0"),
                        (["tol=-1"], "tol=-1")]:
        t0 = time.perf_counter()
        p = subprocess.run([CLI, nl, "print_level=0"] + opts,
                           capture_output=True, text=True, timeout=BUDGET)
        dt = time.perf_counter() - t0
        out = (p.stdout + p.stderr).strip().splitlines()
        last = " | ".join(out[-2:])[:150] if out else ""
        print(f"  CLI {label:<12} rc={p.returncode} t={dt:.2f}s :: {last!r}")

    # --- Pyomo plugin ---
    try:
        import pyomo.environ as pyo

        for label, opts in [("defaults", {}), ("max_iter=1", {"max_iter": 1}),
                            ("max_iters=1 (typo)", {"max_iters": 1}),
                            ("max_iter=0", {"max_iter": 0}),
                            ("max_iter=-5", {"max_iter": -5}),
                            ("tol=-1", {"tol": -1.0})]:
            m = build_pyomo_hs071()
            solver = pyo.SolverFactory("pounce")
            for k, v in opts.items():
                solver.options[k] = v
            try:
                r = solver.solve(m, tee=False)
                f = pyo.value(m.obj)
                print(f"  Pyomo {label:<20} term={r.solver.termination_condition} "
                      f"obj={f:.10f}")
                if label == "max_iters=1 (typo)":
                    record("SILENT-IGNORE" if abs(f - HS071_OPT) < 1e-6 else "REJECTED",
                           "pyomo typo max_iters=1",
                           f"solved to optimum anyway (obj={f:.8f})")
            except Exception as e:
                print(f"  Pyomo {label:<20} EXC {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:110]}")
                if label == "max_iters=1 (typo)":
                    record("REJECTED", "pyomo typo max_iters=1",
                           f"{type(e).__name__}")
    except ImportError as e:
        print(f"  pyomo unavailable: {e}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    probes = {
        "a": probe_a_unknown_keys,
        "b": probe_b_effect,
        "c": probe_c_out_of_range,
        "d": probe_d_types,
        "e": probe_e_surfaces,
    }
    which = sys.argv[1:] or list(probes)
    for k in which:
        try:
            probes[k]()
        except Exception:
            import traceback

            traceback.print_exc()
    hdr("FINDINGS")
    for tag, name, detail in FINDINGS:
        print(f"{tag:<15} {name:<45} {detail}")
