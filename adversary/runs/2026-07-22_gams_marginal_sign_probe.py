"""Adversary probe: what sign does GMO store for equation/variable marginals?

Reads a *kept* GAMS control file (from a second `solve` of a MAXIMIZING LP whose
first solve already produced marginals) and asks GMO directly, via gmoGetEquM /
gmoGetVarM, what it holds.  GAMS reported to the user:

    r1.m = +2.25   r2.m = +0.25   x3.m = -1.50   (maximizing model)

gmoSetSolution2 / gmoSetVarM write into the same storage gmoGetEquM /
gmoGetVarM read (the C link's warm-start path relies on exactly that
round-trip, gams/gams_pounce.c:1005-1006), so whatever sign shows up here is
the sign a solver link must hand back.

Never modifies pounce source.  Read-only against GMO.
"""

import sys

import gams.core.gev as gev
import gams.core.gmo as gmo

CNTR = sys.argv[1] if len(sys.argv) > 1 else "gams_scratch/scr/gamscntr.dat"
SYSDIR = "/Library/Frameworks/GAMS.framework/Versions/53/Resources"

gev_h = gev.new_gevHandle_tp()
rc, msg = gev.gevCreateD(gev_h, SYSDIR, 256)
assert rc, msg
gev.gevInitEnvironmentLegacy(gev_h, CNTR)

gmo_h = gmo.new_gmoHandle_tp()
rc, msg = gmo.gmoCreateD(gmo_h, SYSDIR, 256)
assert rc, msg
gmo.gmoRegisterEnvironment(gmo_h, gev.gevHandleToPtr(gev_h))
gmo.gmoLoadDataLegacy(gmo_h)

gmo.gmoObjStyleSet(gmo_h, gmo.gmoObjType_Fun)
gmo.gmoObjReformSet(gmo_h, 1)
gmo.gmoIndexBaseSet(gmo_h, 0)

n = int(gmo.gmoN(gmo_h))
m = int(gmo.gmoM(gmo_h))
is_max = gmo.gmoSense(gmo_h) == gmo.gmoObj_Max
print(f"n={n} m={m} sense={'MAX' if is_max else 'MIN'}")

equ_m = gmo.doubleArray(m)
gmo.gmoGetEquM(gmo_h, equ_m)
var_m = gmo.doubleArray(n)
gmo.gmoGetVarM(gmo_h, var_m)
var_l = gmo.doubleArray(n)
gmo.gmoGetVarL(gmo_h, var_l)

print("equation marginals as GMO stores them:",
      [round(float(equ_m[i]), 6) for i in range(m)])
print("variable  marginals as GMO stores them:",
      [round(float(var_m[j]), 6) for j in range(n)])
print("variable  levels                      :",
      [round(float(var_l[j]), 6) for j in range(n)])
print()
print("GAMS showed the user: r1.m=+2.25  r2.m=+0.25  x3.m=-1.50")
