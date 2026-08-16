#!/usr/bin/env python3
"""Watching a solve iterate by iterate, and stopping it early.

CasADi's `iteration_callback` is handed the full iterate — `x`, `f`, `g`,
`lam_x`, `lam_g` — once per iteration, and a nonzero return asks the
solver to stop.

This is worth calling out: with the bundled Ipopt plugin, a stock Ipopt
build cannot supply the iterate, and CasADi prints *"intermediate_callback
is disfunctional in your installation"* and passes the callback nothing
usable. POUNCE serves live iterates through its C API, so the callback
below sees real numbers with no special build.
"""

import casadi as ca

x = ca.MX.sym("x", 2)
f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
g = x[0] ** 2 + x[1] ** 2 - 1.5
nlp = {"x": x, "f": f, "g": g}

NX, NG, NP = 2, 1, 0


class Watcher(ca.Callback):
    """Record the trajectory; optionally stop once the objective is small."""

    def __init__(self, name, stop_below=None, opts={}):
        ca.Callback.__init__(self)
        self.stop_below = stop_below
        self.trajectory = []
        self.construct(name, opts)

    # `iteration_callback` is called with the nlpsol *outputs*.
    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        dims = {"f": 1, "x": NX, "g": NG, "lam_x": NX, "lam_g": NG, "lam_p": NP}
        n = dims.get(name, 0)
        return ca.Sparsity.dense(n, 1) if n else ca.Sparsity(0, 0)

    def eval(self, arg):
        named = dict(zip(ca.nlpsol_out(), arg))
        xk = named["x"].full().ravel()
        fk = float(named["f"])
        self.trajectory.append((xk.copy(), fk))
        print(f"  iter {len(self.trajectory) - 1:2d}: x = [{xk[0]:+.6f}, {xk[1]:+.6f}]  f = {fk:.6e}")
        # Returning nonzero requests termination (status User_Requested_Stop).
        if self.stop_below is not None and fk < self.stop_below:
            print("  -> objective below threshold, asking the solver to stop")
            return [1]
        return [0]


print("plain run:")
watcher = Watcher("watcher")
solver = ca.nlpsol("solver", "pounce", nlp, {
    "print_time": False,
    "iteration_callback": watcher,
    "pounce": {"print_level": 0},
})
sol = solver(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=0)
print(f"  status = {solver.stats()['return_status']}, "
      f"{len(watcher.trajectory)} callback fires")

print("\nearly stop at f < 0.05:")
stopper = Watcher("stopper", stop_below=0.05)
solver2 = ca.nlpsol("solver2", "pounce", nlp, {
    "print_time": False,
    "iteration_callback": stopper,
    "pounce": {"print_level": 0},
})
sol2 = solver2(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=0)
print(f"  status = {solver2.stats()['return_status']}")
print(f"  stopped at x = {sol2['x'].full().ravel()}")
