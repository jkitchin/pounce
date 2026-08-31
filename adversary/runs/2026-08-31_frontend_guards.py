"""Adversary probe: input-validation and frontend guards merged this week.
Targets: PR #865 (qp: validate P's shape before the PSD guard reads it)
         PR #796 (jax QP layer: read lb/ub concretely so jnp bounds survive jit)
         PR #790 (nl-parser: reject a truncated .nl instead of solving a different model)
Each guard is attacked with inputs adjacent to the one the PR fixed.
"""
import numpy as np, warnings, traceback
warnings.simplefilter("ignore")
import pounce

fails = []
def check(name, fn, want):
    """want: 'raise' (must raise a clear error) or a callable verifying the result."""
    try:
        out = fn()
    except Exception as e:
        if want == 'raise':
            msg = str(e)
            ok = len(msg) > 0 and 'panic' not in msg.lower()
            print(f"  [{'ok' if ok else 'BAD'}] {name}: raised {type(e).__name__}: {msg[:80]}")
            if not ok: fails.append((name, 'unclear error', msg))
        else:
            print(f"  [BAD] {name}: unexpected {type(e).__name__}: {str(e)[:90]}")
            fails.append((name, 'unexpected raise', str(e)))
        return
    if want == 'raise':
        print(f"  [BAD] {name}: NO ERROR, returned {out}")
        fails.append((name, 'accepted invalid input', repr(out)[:120]))
    else:
        ok, detail = want(out)
        print(f"  [{'ok' if ok else 'BAD'}] {name}: {detail}")
        if not ok: fails.append((name, 'wrong result', detail))

print("=== PR #865: solve_qp P-shape validation ===")
c2 = np.array([-1.0, -1.0])
check("P non-square (2x3)",      lambda: pounce.solve_qp(P=np.ones((2,3)), c=c2), 'raise')
check("P 1-D",                   lambda: pounce.solve_qp(P=np.ones(2), c=c2), 'raise')
check("P 3-D",                   lambda: pounce.solve_qp(P=np.ones((2,2,2)), c=c2), 'raise')
check("P size != len(c)",        lambda: pounce.solve_qp(P=np.eye(3), c=c2), 'raise')
# NOTE: an empty (0x0) QP is trivially optimal, and `solve_qp` documents
# "P (lower triangle is used; assumed symmetric)" (python/pounce/qp.py:925),
# so an asymmetric P is a caller precondition violation, not a solver defect.
# Both were flagged by a first draft of this probe and are FORMULATION_ERROR.
check("P with NaN",              lambda: pounce.solve_qp(P=np.array([[np.nan,0.],[0.,1.]]), c=c2), 'raise')
check("P with inf",              lambda: pounce.solve_qp(P=np.array([[np.inf,0.],[0.,1.]]), c=c2), 'raise')
check("P valid (control)",       lambda: pounce.solve_qp(P=np.eye(2), c=c2),
      lambda r: (abs(r.obj + 1.0) < 1e-8, f"obj={r.obj:.8f} expect -1.0"))

print("\n=== PR #790: truncated .nl must be rejected, not silently re-modelled ===")
import subprocess, os
CLI = "./target/release/pounce"
src = open("/tmp/saddle_x0.nl","rb").read()
ref = subprocess.run([CLI, "/tmp/saddle_x0.nl", "-AMPL"], capture_output=True, text=True)
ref_obj = [l for l in ref.stdout.splitlines() if "Objective..." in l]
print(f"  intact file: {ref_obj[0].split(':')[1].split()[0] if ref_obj else '??'}")
bad_accept = 0
for frac in (0.30, 0.50, 0.70, 0.85, 0.95, 0.99):
    p = f"/tmp/trunc_{int(frac*100)}.nl"
    open(p,"wb").write(src[:int(len(src)*frac)])
    r = subprocess.run([CLI, p, "-AMPL"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    solved = "Optimal Solution Found" in out
    if solved:
        bad_accept += 1
        fails.append((f"truncated .nl @{frac:.0%}", "solved a truncated model", out[-200:]))
    print(f"  truncated to {frac:>4.0%}: exit={r.returncode} {'SOLVED IT (bad)' if solved else 'rejected'}")

print("\n=== PR #796: jax QP layer, jnp bounds under jit ===")
try:
    import jax, jax.numpy as jnp
    from pounce.jax import solve_qp as qp_layer
    ok_jax = True
except Exception as e:
    print(f"  [skip] jax layer unavailable: {type(e).__name__}: {str(e)[:80]}")
    ok_jax = False
if ok_jax:
    n = 3
    P = jnp.eye(n); c = jnp.array([-1., -2., -3.])
    lb = jnp.array([0., 0., 0.]); ub = jnp.array([0.5, 0.5, 0.5])
    # PR #796's case: jnp lb/ub built OUTSIDE the trace and closed over.
    def run(c_):
        return qp_layer(P=P, c=c_, lb=lb, ub=ub)
    try:
        eager = run(c)
        jitted = jax.jit(run)(c)
        d = float(jnp.max(jnp.abs(jnp.asarray(eager) - jnp.asarray(jitted))))
        print(f"  eager vs jit max|dx| = {d:.2e}")
        if d > 1e-8: fails.append(("jax jit parity", "eager != jit", f"{d:.2e}"))
        # bounds must actually bind
        print(f"  eager x = {np.asarray(eager)}  (expect clipping at 0.5)")
    except Exception as e:
        print(f"  [BAD] jax layer raised: {type(e).__name__}: {str(e)[:120]}")
        fails.append(("jax layer", "raised", str(e)[:150]))

print(f"\nfailures: {len(fails)}")
for f in fails: print(f"   {f[0]}: {f[1]} -- {f[2][:100]}")
print("VERDICT: PASS" if not fails else f"VERDICT: FAIL ({len(fails)} issues)")
