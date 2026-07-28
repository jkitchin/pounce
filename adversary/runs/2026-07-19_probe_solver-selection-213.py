"""Targeted adversary probe of the gh #213 fix (PR #214).

Contract under test (`pounce.minimize`):
  1. unrecognized `solver_selection` raises ValueError (was: silently -> NLP);
  2. `qp-active-set` actually dispatches to the active-set SQP engine and
     agrees with `options={"algorithm":"active-set-sqp"}`;
  3. matching is case-insensitive;
  4. auto / nlp / lp-ipm / qp-ipm / socp unchanged.

Engine oracle: the Rust log written to fd 1 at print_level=5. The filter-IPM
prints the iteration table header "iter      objective   inf_pr"; the
active-set SQP driver does not. fd-level capture (os.dup2) is required because
the log is written from Rust, not Python stdout.

Value-set oracle: the CLI (`crates/pounce-cli/src/dispatch.rs`
`SolverSelection::parse`), which accepts exactly
{auto, nlp, lp-ipm, qp-ipm, socp, qp-active-set}.
"""

import os
import tempfile
import warnings

import numpy as np

from pounce import minimize

FAILS = []
NOTES = []


def check(name, cond, detail=""):
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(f"{name} :: {detail}")


# --- test problem: convex QP (so every selector is legal for it) --------------
Q = np.array([[2.0, 0.5], [0.5, 3.0]])
c = np.array([-2.0, -3.0])
XSTAR = np.linalg.solve(Q, -c)
FSTAR = 0.5 * XSTAR @ Q @ XSTAR + c @ XSTAR


def f(x):
    return float(0.5 * x @ Q @ x + c @ x)


def gf(x):
    return Q @ x + c


def hf(x):
    return Q


X0 = np.zeros(2)

# --- a genuinely NONLINEAR problem (Rosenbrock), to probe the documented
# "library path does not class-validate qp-active-set" behaviour --------------
def rosen(x):
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def rosen_g(x):
    return np.array(
        [-400.0 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200.0 * (x[1] - x[0] ** 2)]
    )


def rosen_h(x):
    return np.array(
        [[1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]], [-400.0 * x[0], 200.0]]
    )


IPM_MARK = "iter      objective   inf_pr"


def solve_capture(fun=f, jac=gf, hess=hf, x0=X0, **kw):
    """Run minimize with the Rust log captured at fd level. Returns (res, log)."""
    saved = os.dup(1)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
    os.dup2(tf.fileno(), 1)
    try:
        r = minimize(fun, x0, jac=jac, hess=hess, print_level=5, **kw)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        tf.flush()
    with open(tf.name) as fh:
        log = fh.read()
    os.unlink(tf.name)
    return r, log


def engine(log):
    return "ipm" if IPM_MARK in log else "sqp"


print("### 1. rejection of unrecognized values")
BAD = [
    "qp_ipm",  # underscore typo (the #213 motivating case)
    "qp-activeset",
    "qpactiveset",
    "active-set-sqp",  # the *algorithm* name, not a selector
    "QP_ACTIVE_SET",
    " nlp",  # leading space
    "nlp ",  # trailing space
    " nlp ",
    "nlp\n",
    "nlp\t",
    " nlp",  # NBSP
    "ｎｌｐ",  # fullwidth (unicode lookalike)
    "ｑｐ－ｉｐｍ",
    "nlp​",  # zero-width space
    "nłp",  # latin l-with-stroke lookalike
    "",  # empty
    "ipopt",
    "socp;nlp",
    "auto,nlp",
]
for v in BAD:
    try:
        minimize(f, X0, jac=gf, hess=hf, solver_selection=v)
        check(f"reject {v!r}", False, "accepted silently")
    except ValueError as e:
        check(f"reject {v!r}", "not a valid selector" in str(e))
    except Exception as e:  # noqa: BLE001
        check(f"reject {v!r}", False, f"wrong exception {type(e).__name__}: {e}")

print("\n### 1b. non-string types")
NONSTR = [None, 0, 1, True, False, 3.5, b"nlp", ["nlp"], ("nlp",), {"nlp": 1}, object()]
for v in NONSTR:
    try:
        minimize(f, X0, jac=gf, hess=hf, solver_selection=v)
        check(f"reject non-str {v!r}", False, "accepted silently")
    except ValueError:
        check(f"reject non-str {v!r}", True)
    except Exception as e:  # noqa: BLE001
        check(f"reject non-str {v!r}", False, f"{type(e).__name__}: {e}")

# str-coercible lookalikes that DO reach a valid selector
class FakeSel:
    def __str__(self):
        return "qp-active-set"


import enum


class Sel(str, enum.Enum):
    QPAS = "qp-active-set"


for label, v in [
    ("numpy.str_", np.str_("QP-Active-Set")),
    ("object with __str__", FakeSel()),
    ("str-enum", Sel.QPAS),
]:
    try:
        r, log = solve_capture(solver_selection=v)
        NOTES.append(f"{label} -> accepted, engine={engine(log)}, f={r.fun:.6e}")
        print(f"[note] {label}: accepted (str-coerced), engine={engine(log)}")
    except ValueError as e:
        print(f"[note] {label}: rejected ({e})")
        NOTES.append(f"{label} -> rejected")

print("\n### 2. qp-active-set really runs the SQP engine")
r_default, log_default = solve_capture()
check("default (no selector) uses IPM", engine(log_default) == "ipm", engine(log_default))

r_qpas, log_qpas = solve_capture(solver_selection="qp-active-set")
check("solver_selection='qp-active-set' uses SQP", engine(log_qpas) == "sqp", engine(log_qpas))

r_alg, log_alg = solve_capture(algorithm="active-set-sqp")
check("algorithm='active-set-sqp' uses SQP", engine(log_alg) == "sqp", engine(log_alg))
check(
    "qp-active-set == algorithm=active-set-sqp (objective)",
    abs(float(r_qpas.fun) - float(r_alg.fun)) < 1e-12,
    f"{r_qpas.fun!r} vs {r_alg.fun!r}",
)
check(
    "qp-active-set == algorithm=active-set-sqp (x)",
    np.allclose(np.asarray(r_qpas.x), np.asarray(r_alg.x), atol=1e-12),
    f"{np.asarray(r_qpas.x)} vs {np.asarray(r_alg.x)}",
)
check(
    "qp-active-set hits the analytic QP optimum",
    abs(float(r_qpas.fun) - FSTAR) < 1e-8 and np.allclose(np.asarray(r_qpas.x), XSTAR, atol=1e-8),
    f"f={r_qpas.fun:.12e} vs {FSTAR:.12e}",
)

# conflicting algorithm= : documented that qp-active-set wins
r_conf, log_conf = solve_capture(solver_selection="qp-active-set", algorithm="interior-point")
check(
    "qp-active-set beats algorithm='interior-point' (documented precedence)",
    engine(log_conf) == "sqp",
    engine(log_conf),
)

print("\n### 3. case-insensitivity (library) — CLI is case-SENSITIVE (see report)")
for v in ["QP-ACTIVE-SET", "Qp-Active-Set", "qP-aCtIvE-sEt"]:
    r, log = solve_capture(solver_selection=v)
    check(f"{v!r} -> SQP", engine(log) == "sqp", engine(log))
for v in ["AUTO", "Nlp", "NLP"]:
    try:
        minimize(f, X0, jac=gf, hess=hf, solver_selection=v)
        check(f"{v!r} accepted", True)
    except ValueError as e:
        check(f"{v!r} accepted", False, str(e))

print("\n### 4. valid selectors unchanged")
r_auto = minimize(f, X0, jac=gf, hess=hf, solver_selection="auto")
check("auto solves the QP", abs(float(r_auto.fun) - FSTAR) < 1e-6, f"{r_auto.fun}")
check("auto took the specialized route", r_auto.info.get("solver") in ("qp", "lp", "socp", "qp-ipm", "lp-ipm"), repr(r_auto.info.get("solver")))
r_qpipm = minimize(f, X0, jac=gf, hess=hf, solver_selection="qp-ipm")
check("qp-ipm solves the QP", abs(float(r_qpipm.fun) - FSTAR) < 1e-6, f"{r_qpipm.fun}")
r_nlp = minimize(f, X0, jac=gf, hess=hf, solver_selection="nlp")
check("nlp solves the QP", abs(float(r_nlp.fun) - FSTAR) < 1e-6, f"{r_nlp.fun}")
try:
    minimize(f, X0, jac=gf, hess=hf, solver_selection="socp")
    print("[note] socp accepted a plain convex QP")
except ValueError as e:
    NOTES.append(
        "socp on a plain (unconstrained) convex QP is REJECTED by the library router "
        f"({str(e)[:80]}...), while the CLI documents socp as valid for a convex "
        "LP/QP/QCQP. Pre-existing library-vs-CLI difference, orthogonal to #213; it "
        "is a loud error, not a silent engine swap."
    )
    print("[note] socp rejected a plain convex QP (library router wants a QCQP)")

# socp on a genuine convex QCQP (ball constraint) must still work
r_socp = minimize(
    lambda x: float(-x[0] - x[1]),
    np.array([0.1, 0.1]),
    jac=lambda x: np.array([-1.0, -1.0]),
    constraints=[
        {  # scipy legacy form: fun(x) >= 0  <=>  ||x||^2 <= 1
            "type": "ineq",
            "fun": lambda x: np.array([1.0 - x @ x]),
            "jac": lambda x: np.array([[-2.0 * x[0], -2.0 * x[1]]]),
        }
    ],
    solver_selection="socp",
)
check(
    "socp solves a convex QCQP (min -x1-x2 s.t. ||x||<=1 -> -sqrt(2))",
    abs(float(r_socp.fun) + np.sqrt(2.0)) < 1e-6,
    f"{r_socp.fun}",
)
try:
    minimize(f, X0, jac=gf, hess=hf, solver_selection="lp-ipm")
    check("lp-ipm rejects a non-LP", False, "no error")
except ValueError as e:
    check("lp-ipm rejects a non-LP", "not detected as" in str(e), str(e)[:60])

print("\n### 5. warm_start interaction")
res0 = minimize(f, X0, jac=gf, hess=hf, solver_selection="nlp")
from pounce import WarmStart

ws = WarmStart.from_info(res0.x, res0.info)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r_ws, log_ws = solve_capture(solver_selection="qp-active-set", warm_start=ws)
msgs = [str(x.message) for x in w]
check(
    "warm_start + qp-active-set warns it is ignored",
    any("warm_start" in m and "ignored" in m for m in msgs),
    str(msgs),
)
check(
    "warm_start + qp-active-set actually runs the NLP/IPM engine (matches the warning)",
    engine(log_ws) == "ipm",
    engine(log_ws),
)
try:
    minimize(f, X0, jac=gf, hess=hf, solver_selection="qp_active_set", warm_start=ws)
    check("invalid selector still rejected when warm_start is given", False, "accepted")
except ValueError:
    check("invalid selector still rejected when warm_start is given", True)

print("\n### 6. kwarg vs options= dict")
try:
    minimize(f, X0, jac=gf, hess=hf, options={"solver_selection": "bogus"})
    check("invalid selector inside options= rejected", False, "accepted")
except ValueError:
    check("invalid selector inside options= rejected", True)

r_both, log_both = solve_capture(
    options={"solver_selection": "qp-active-set"}, solver_selection="nlp"
)
check(
    "kwarg wins over options= (documented): -> IPM",
    engine(log_both) == "ipm",
    engine(log_both),
)
r_both2, log_both2 = solve_capture(
    options={"solver_selection": "nlp"}, solver_selection="qp-active-set"
)
check("kwarg wins over options= (reverse): -> SQP", engine(log_both2) == "sqp", engine(log_both2))

# non-Mapping options= : silently dropped (pre-existing, orthogonal to #213)
try:
    r_nm, log_nm = solve_capture(options=[("solver_selection", "qp-active-set")])
    print(f"[note] non-Mapping options=list -> no error, engine={engine(log_nm)}")
    NOTES.append(
        f"non-Mapping options= (list of pairs) is silently discarded; engine={engine(log_nm)} "
        "(expected sqp if honored). Pre-existing hole, independent of #213."
    )
except Exception as e:  # noqa: BLE001
    print(f"[note] non-Mapping options=list -> {type(e).__name__}: {e}")
    NOTES.append(f"non-Mapping options= raises {type(e).__name__}")

print("\n### 7. qp-active-set on a NON-QP (library path deliberately does not class-validate)")
r_ros, log_ros = solve_capture(
    fun=rosen, jac=rosen_g, hess=rosen_h, x0=np.array([-1.2, 1.0]),
    solver_selection="qp-active-set",
)
check("SQP engine ran on Rosenbrock", engine(log_ros) == "sqp", engine(log_ros))
check(
    "Rosenbrock via qp-active-set reaches (1,1)",
    np.allclose(np.asarray(r_ros.x), [1.0, 1.0], atol=1e-5),
    f"x={np.asarray(r_ros.x)} f={r_ros.fun:.3e} status={r_ros.status}",
)
NOTES.append(
    f"qp-active-set on Rosenbrock (non-QP): status={r_ros.status} f={r_ros.fun:.3e} "
    f"x={np.asarray(r_ros.x)} — library path runs SQP without class validation "
    "(documented); the CLI would reject with a class mismatch."
)

print("\n" + "=" * 70)
for n in NOTES:
    print("NOTE:", n)
print("=" * 70)
print(f"failures={len(FAILS)}")
for x in FAILS:
    print("  -", x)
print("VERDICT: PASS" if not FAILS else "VERDICT: FAIL")
