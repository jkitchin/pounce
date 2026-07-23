"""Run all five solver-equivalence validation problems and emit results.json.

    python validation/run_all.py

Each problem script writes results.<name>.json; this driver runs them in
order, gathers those payloads plus environment metadata into a single
machine-readable validation/results.json, and prints a one-line summary per
problem.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SCRIPTS = [
    ("p1_cho", "p1_cho.py"),
    ("p2_hs71", "p2_hs71.py"),
    ("p3_control", "p3_control.py"),
    ("p4_qp", "p4_qp.py"),
    ("p5_parametric", "p5_parametric.py"),
]


def env_meta():
    sys.path.insert(0, str(HERE))
    import _common as c
    return {
        "pounce_version": c.pounce_version(),
        "pounce_executable": c.setup_pounce(),
        "ipopt_version": c.ipopt_version(),
        "date": c.today(),
    }


def main():
    meta = env_meta()
    results = {"environment": meta, "problems": {}}
    for name, script in SCRIPTS:
        print(f"=== running {script} ===", flush=True)
        r = subprocess.run([PY, str(HERE / script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise SystemExit(f"{script} failed (rc={r.returncode})")
        payload = json.loads((HERE / f"results.{name}.json").read_text())
        results["problems"][name] = payload

    (HERE / "results.json").write_text(json.dumps(results, indent=2,
                                                  default=str))

    # one-line summaries
    print("\n================ SUMMARY ================")
    p1 = results["problems"]["p1_cho"]
    print("P1 CHO: pounce obj={:.10g} vs IPOPT-MA57 benchmark rel err {:.2e}; "
          "covariance method exact on controls "
          "(linear {:.1e}, nonlinear {:.1e})".format(
              p1["point_estimate"]["pounce_objective"],
              p1["point_estimate"]["obj_relerr_pounce_vs_benchmark"],
              p1["covariance_method_controls"]["linear_regression"][
                  "pounce_vs_pynumero_maxrel"],
              p1["covariance_method_controls"]["nonlinear_expfit"][
                  "pounce_vs_pynumero_maxrel"]))
    p2 = results["problems"]["p2_hs71"]["agreement"]
    print("P2 HS71: duals match pounce/ipopt/analytic in SIGN; "
          "dual_c1 pounce-vs-ipopt {:.2e}".format(
              p2["dual_c1_absdiff_pounce_vs_ipopt"]))
    for nm, c in results["problems"]["p3_control"]["cases"].items():
        print("P3 {}: obj rel err {:.2e}, {} restoration iters, "
              "both optimal={}".format(nm, c["obj_relerr"],
                                       c["pounce_restoration_iters"],
                                       c["both_optimal"]))
    p4 = results["problems"]["p4_qp"]
    print("P4 QP: 4-way multiplier signs agree={}, pounce y err vs closed "
          "form {:.2e}".format(
              p4["multiplier_signs"]["all_agree"],
              p4["routes"]["pounce"]["y_maxabs_vs_closed_form"]))
    p5 = results["problems"]["p5_parametric"]
    print("P5 parametric: dx/dp vs analytic {:.2e}, FD quad-convergence "
          "ratios {}".format(
              p5["agreement"]["dx1_dp_pounce_vs_analytic"],
              [round(r, 2) for r in
               p5["quadratic_convergence_ratios_err_dx1"]]))
    print("\nwrote", HERE / "results.json")


if __name__ == "__main__":
    main()
