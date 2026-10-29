"""Name -> model registry for the CALIPSO corpus.

Each entry gives the model, its initial-point generator (a function of the
seed), the CALIPSO.jl example it was ported from, and how to map a
CALIPSO.jl solution record (``results_calipso.jsonl``) into this port's
variable vector, so the port can be evaluated at CALIPSO's own answer.
"""

import numpy as np


def _trajectory_mapper(mod):
    return mod.from_calipso


def _variables(rec):
    return np.asarray(rec["variables"], dtype=float)


def _drifting_from_calipso(rec):
    import drifting
    z = np.zeros(drifting.nz)
    for t in range(drifting.T):
        z[drifting.ix[t]] = rec["states"][t]
    for t in range(drifting.T - 1):
        z[drifting.iu[t]] = rec["actions"][t]
    return z


def get(name):
    """-> (model, initial_point, calipso_example, from_calipso)"""
    if name == "bunnyhop":
        import bunnyhop as m
        return m.model, m.initial_point, "contact_implicit/bunnyhop.jl", _variables
    if name.startswith("drifting_"):
        import drifting as m
        return (m.make_model(name.split("_", 1)[1]), m.initial_point,
                "contact_implicit/drifting.jl", _drifting_from_calipso)
    if name == "ballincup":
        import ballincup as m
        return m.model, m.initial_point, "contact_implicit/ball_in_cup.jl", m.from_calipso
    if name == "rocket":
        import rocket as m
        return m.model, m.initial_point, "state_triggered/rocket_landing.jl", m.from_calipso
    if name == "quadruped":
        import quadruped as m
        return m.model, m.initial_point, "contact_implicit/quadruped_gait.jl", m.from_calipso
    if name in ("complementarity", "maratos", "wachter"):
        import nonconvex
        model, x0 = nonconvex.MODELS[name]()
        return model, x0, f"nonconvex/{name}.jl", _variables
    raise KeyError(name)


#: Models with a complementarity block; the others are plain NLPs, on which
#: every complementarity route reduces to the same solve.
MPCC = ("bunnyhop", "drifting_norm", "drifting_squared", "ballincup", "rocket", "quadruped",
        "complementarity")
NLP = ("maratos", "wachter")
ALL = MPCC + NLP

#: Models whose initial point does not depend on the seed: the source has
#: no random part, so five seeds would be five identical runs.
DETERMINISTIC = ("rocket", "complementarity", "maratos", "wachter")

#: Models whose complementarity is between second-order cones, and so also
#: run the paper's Jordan-product rows as a route.
SOC = ("drifting_norm", "drifting_squared", "quadruped")

MPCC_ROUTES = ("defaults_eq", "direct", "ncp_eq", "ncp_eq_l1", "scholtes_then_ncp", "scholtes_then_ineq")
NLP_ROUTES = ("defaults_eq", "ncp_eq")


def jobs(solver="pounce", seeds=(0, 1, 2, 3, 4), names=ALL):
    """(model, route, seed) triples for one solver. ncp_eq_l1 needs a
    POUNCE-only option and is skipped for Ipopt."""
    out = []
    for name in names:
        routes = list(MPCC_ROUTES if name in MPCC else NLP_ROUTES)
        if name in SOC:
            routes.append("jordan_eq")
        if solver != "pounce" and "ncp_eq_l1" in routes:
            routes.remove("ncp_eq_l1")
        for seed in (seeds[:1] if name in DETERMINISTIC else seeds):
            out += [(name, r, seed) for r in routes]
    return out


if __name__ == "__main__":
    import sys

    solver = sys.argv[1] if len(sys.argv) > 1 else "pounce"
    names = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else ALL
    for job in jobs(solver, names=names):
        print(*job)
