#!/usr/bin/env python3
"""Generating C for the whole solve, for a target with no Python on it.

`solver.generate()` normally emits the *model*. For an `nlpsol` it emits
the model **and** the solve: one `.c` file holding the oracle functions,
the option calls, and the loop that drives them. Compile it into your
firmware, your ROS node, your Simulink S-function, and neither CasADi
nor Python is present at run time.

What is present is `libpounce_cinterface`, which the generated file calls
through `pounce.h`. That is the same bargain CasADi's Ipopt plugin
strikes — its generated code includes `<coin-or/IpStdCInterface.h>` and
links libipopt — and it is what "self-contained" does *not* mean here.

The check that matters is at the bottom: the generated solve must land on
exactly the same point as the interpreted one, multipliers included.
"""

import os
import shutil
import subprocess
import tempfile

import casadi as ca

HERE = os.path.dirname(os.path.abspath(__file__))
POUNCE_INC = os.path.join(HERE, "..", "..", "crates", "pounce-cinterface", "include")
POUNCE_LIB = os.path.abspath(os.path.join(HERE, "..", "..", "target", "release"))

cc = shutil.which("cc") or shutil.which("gcc")
if cc is None:
    raise SystemExit("no C compiler found — skipping")

# A small NMPC-ish problem: two states, a horizon of controls, bounded.
N = 10
u = ca.MX.sym("u", N)
x0 = ca.MX.sym("x0", 2)

x = x0
cost = 0
for k in range(N):
    x = ca.vertcat(x[0] + 0.1 * x[1], x[1] + 0.1 * (u[k] - 0.2 * x[1]))
    cost += ca.sumsqr(x) + 0.01 * u[k] ** 2

nlp = {"x": u, "p": x0, "f": cost, "g": x}
opts = {
    "print_time": False,
    "pounce": {"print_level": 0, "tol": 1e-9},
}
solver = ca.nlpsol("mpc_step", "pounce", nlp, opts)

call = dict(x0=[0.0] * N, p=[1.0, 0.0], lbx=-1.0, ubx=1.0,
            lbg=[-5, -5], ubg=[5, 5])
interpreted = solver(**call)

with tempfile.TemporaryDirectory() as d:
    cwd = os.getcwd()
    os.chdir(d)
    try:
        solver.generate("mpc_step.c")
        size = os.path.getsize("mpc_step.c")
        print(f"generated mpc_step.c ({size/1024:.1f} kB)")

        # Exactly what a deployment build looks like. No CasADi anywhere on
        # this command line.
        cmd = [cc, "-O2", "-Wall", "-shared", "-fPIC", "-o", "mpc_step.so",
               "mpc_step.c", "-I", POUNCE_INC, "-L", POUNCE_LIB,
               "-lpounce_cinterface", "-Wl,-rpath," + POUNCE_LIB, "-lm"]
        print("compiling:", " ".join(cmd[:4]), "… -lpounce_cinterface")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # `external` loads the compiled entry point back as an ordinary
        # Function. Nothing in it knows what a plugin is any more.
        embedded = ca.external("mpc_step", os.path.join(d, "mpc_step.so"))
        generated = embedded(lam_x0=0, lam_g0=0, **call)
    finally:
        os.chdir(cwd)

print()
print(f"interpreted u0 = {float(interpreted['x'][0]):+.12f}")
print(f"generated   u0 = {float(generated['x'][0]):+.12f}")
for key in ("x", "f", "lam_x", "lam_g"):
    err = float(ca.norm_inf(ca.DM(generated[key]) - ca.DM(interpreted[key])))
    print(f"  max |Δ{key:<6s}| = {err:.3e}")

print()
print("Not carried into generated code, and refused by name if you ask:")
print("  iteration_callback (a CasADi Function, and CasADi is not there),")
print("  warm_start_from_previous (state between calls of one solver object),")
print("  convexify_strategy.")
print("What is carried: options, bounds, the nonlinear-variable subset for")
print("L-BFGS, and clip_inactive_lam — so the multipliers match above.")
