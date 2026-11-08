"""Regenerate `cinterface_mpcc_python.json`, the Python side of
`tests/cinterface_mpcc_parity.rs` (gh#1147 S1-I8).

Each row solves one `fixtures_mpcc/` model through Python's
`Problem(..., complementarity=...)` on the `.nl` model's own evaluators
(`pounce.read_nl`, the `NlTnlp` the CLI and the C harness also use), from the
`.nl` start, with the row's options, and records what the parity test holds
the C API to: `solution.{status, objective, x}`,
`statistics.iteration_count` and the report's `complementarity` block minus
`seconds`.

Reals are copied as the report writer printed them (not re-printed by
Python's `json`), so the Rust test parses both sides from identical text and
compares to the bit.

Run from the repository root, with the extension built from this tree:

    python crates/pounce-cli/tests/common/gen_cinterface_mpcc_python.py
"""

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

import pounce

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures_mpcc"
OUT = HERE / "cinterface_mpcc_python.json"

# (row name, fixture, options) — the rows of the parity test that Python can
# express. The masked limited-memory rows cannot be: `Problem` has no
# nonlinear-variable declaration.
ROWS = [
    ("qpec_small", "qpec_small", []),
    ("ralph1", "ralph1", []),
    ("mixed_pair", "mixed_pair", []),
    ("design_cent_3", "design_cent_3", []),
    ("infeasible_pair", "infeasible_pair", []),
    ("opf_vv_lower_knee", "opf_vv_lower_knee", []),
    ("opf_vv_lower_knee_held", "opf_vv_lower_knee_held", []),
    ("qpec_small_exact", "qpec_small", ["complementarity_strategy=exact"]),
    ("qpec_small_continuation", "qpec_small", ["complementarity_strategy=continuation"]),
    ("qpec_small_schedule", "qpec_small", ["complementarity_continuation=schedule"]),
    ("qpec_small_lift_yes", "qpec_small", ["complementarity_lift=yes"]),
    ("qpec_small_lift_no", "qpec_small", ["complementarity_lift=no"]),
    ("qpec_small_lbfgs", "qpec_small", ["hessian_approximation=limited-memory"]),
    ("infeasible_pair_lbfgs_unmasked", "infeasible_pair", ["hessian_approximation=limited-memory"]),
    ("opf_vv_lower_knee_lbfgs_unmasked", "opf_vv_lower_knee", ["hessian_approximation=limited-memory"]),
    (
        "opf_vv_lower_knee_held_lbfgs_unmasked",
        "opf_vv_lower_knee_held",
        ["hessian_approximation=limited-memory"],
    ),
]

RAW = "@@raw@@"


class NlCallbacks:
    def __init__(self, nl):
        self.nl = nl

    def objective(self, x):
        return self.nl.objective(x)

    def gradient(self, x):
        return self.nl.gradient(x)

    def constraints(self, x):
        return self.nl.constraints(x)

    def jacobianstructure(self):
        return self.nl.jacobian_structure()

    def jacobian(self, x):
        return self.nl.jacobian(x)

    def hessianstructure(self):
        return self.nl.hessian_structure()

    def hessian(self, x, lagrange, obj_factor):
        return self.nl.hessian(x, lagrange, obj_factor)


def option_value(v):
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def solve(fixture, options, report_path):
    nl = pounce.read_nl(str(FIXTURES / f"{fixture}.nl"))
    prob = pounce.Problem(
        n=nl.n,
        m=nl.m,
        problem_obj=NlCallbacks(nl),
        lb=list(nl.x_l),
        ub=list(nl.x_u),
        cl=list(nl.g_l),
        cu=list(nl.g_u),
        complementarity=[(r, v) for (r, v, _flags) in nl.complementarity],
    )
    prob.add_option("print_level", 0)
    for o in options:
        k, v = o.split("=", 1)
        prob.add_option(k, option_value(v))
    prob.solve(np.asarray(nl.x0, dtype=float), report_path=str(report_path), report_detail="full")
    # Keep every real as the writer printed it.
    return json.loads(Path(report_path).read_text(), parse_float=lambda s: RAW + s)


def main():
    rows = {}
    with tempfile.TemporaryDirectory() as d:
        for name, fixture, options in ROWS:
            r = solve(fixture, options, Path(d) / f"{name}.json")
            block = dict(r["complementarity"])
            for s in block["solves"]:
                s.pop("seconds", None)
            rows[name] = {
                "fixture": fixture,
                "options": options,
                "status": r["solution"]["status"],
                "objective": r["solution"]["objective"],
                "x": r["solution"]["x"],
                "iteration_count": r["statistics"]["iteration_count"],
                "complementarity": block,
            }
            print(name, block["total_iterations"], file=sys.stderr)
    text = json.dumps({"generated_by": "pounce " + pounce.__version__, "rows": rows}, indent=1)
    text = re.sub('"' + re.escape(RAW) + '([^"]*)"', r"\1", text)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
