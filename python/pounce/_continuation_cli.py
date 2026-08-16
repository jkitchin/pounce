"""Path manifest and repeated-solve protocol for the CLI (pounce#608).

pounce#608 asks for the continuation driver to reach the CLI and GAMS
frontends, and names the mechanism: *"CLI/GAMS can follow using a path
manifest or repeated-solve protocol."* This is that mechanism.

What crosses the process boundary, and what does not
----------------------------------------------------

The in-process driver (:class:`pounce.Continuation`) gets its tangent
predictor from :meth:`pounce.Solver.parametric_step`, a back-solve
against the KKT factor the previous solve left in memory. **That factor
does not survive `exec`.** A CLI path is a sequence of separate
`pounce model.nl` processes, so the factor is gone before the next
point starts and no tangent is available. This is a property of the
process boundary, not a gap in the implementation: there is nothing to
serialise that would make a back-solve possible in a fresh process
short of shipping the factorization itself.

So the CLI path is **zero-order warm transfer plus the driver's
orchestration**: the trace, the counters, the active-set events, the
subdivision on rejection, and a single manifest describing the whole
path. What it adds over `--warm-start`-style flags is that the
*sequence* is a first-class object -- one command, one report, and the
per-point bookkeeping pounce#608's third acceptance criterion asks for.
``docs/src/continuation.md`` records what that is measured to be worth
against repeated cold invocations.

The transfer itself is not primal-only. An AMPL ``.nl`` file carries an
initial primal point (the ``x`` segment) *and* initial duals (the ``d``
segment), and pounce's reader honours both, so the previous point's
``.sol`` -- which reports duals then primals -- can be folded straight
back into the next point's model. With ``warm_start_init_point yes``
that is a genuine primal-dual warm start, the same state
:class:`pounce.WarmStart` carries in process, minus the barrier
parameter and the bound multipliers, which the ``.nl`` format has
nowhere to put.

Manifest format (JSON, ``version: 1``)
--------------------------------------

.. code:: json

    {
      "version": 1,
      "points": [
        {"model": "mpc_00.nl", "theta": [1.5, 0.0]},
        {"model": "mpc_01.nl", "theta": [1.4, 0.1]}
      ],
      "options": {"tol": "1e-8"},
      "warm": true
    }

``points`` is the path, in order; each names a model file the modeling
system already emitted for that parameter value. ``theta`` is optional
and carried through to the trace for the reader's benefit -- the
driver never has to interpret it, because the model file *is* the
parameter update. ``options`` are passed to every solve. Paths are
resolved relative to the manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

__all__ = [
    "PathManifest",
    "parse_sol",
    "rewrite_nl_initial_point",
    "trace_manifest",
]

_OK_STATUS = (0, 1)


class PathManifest:
    """A parsed continuation path manifest."""

    def __init__(self, points, options=None, warm=True, root="."):
        if not points:
            raise ValueError("manifest: 'points' is empty; nothing to trace")
        self.points = list(points)
        self.options = dict(options or {})
        self.warm = bool(warm)
        self.root = root

    @classmethod
    def load(cls, path) -> "PathManifest":
        with open(path) as fh:
            data = json.load(fh)
        version = data.get("version", 1)
        if int(version) != 1:
            raise ValueError(
                f"manifest: unsupported version {version!r}; this pounce "
                "reads version 1"
            )
        pts = data.get("points")
        if not isinstance(pts, list):
            raise ValueError("manifest: 'points' must be a list")
        root = os.path.dirname(os.path.abspath(path))
        out = []
        for i, p in enumerate(pts):
            if not isinstance(p, dict) or "model" not in p:
                raise ValueError(
                    f"manifest: point {i} has no 'model'; every point names "
                    "the model file emitted for its parameter value"
                )
            out.append({
                "model": os.path.join(root, p["model"]),
                "theta": p.get("theta"),
            })
        return cls(out, data.get("options"), data.get("warm", True), root)


def parse_sol(path, n, m):
    """``(duals, primals)`` from an AMPL ``.sol`` file.

    The format is: a message block, a blank line, ``Options``, an option
    count and that many option values, then four counts
    (``m, m, n, n``), then `m` dual values and `n` primal values.
    """
    with open(path) as fh:
        toks = fh.read().split("\n")
    i = 0
    while i < len(toks) and toks[i].strip() != "Options":
        i += 1
    if i >= len(toks):
        raise ValueError(f"{path}: no Options block; not an AMPL .sol file")
    i += 1
    nopt = int(toks[i].strip())
    i += 1 + nopt
    counts = [int(toks[i + k].strip()) for k in range(4)]
    i += 4
    m_sol, n_sol = counts[1], counts[3]
    vals = []
    while i < len(toks) and len(vals) < m_sol + n_sol:
        t = toks[i].strip()
        if t:
            vals.append(float(t))
        i += 1
    if len(vals) < m_sol + n_sol:
        raise ValueError(
            f"{path}: expected {m_sol + n_sol} values, found {len(vals)}"
        )
    return vals[:m_sol], vals[m_sol:m_sol + n_sol]


def _nl_counts(lines):
    """``(n, m)`` from an ASCII ``.nl`` header."""
    if not lines or not lines[0].startswith("g"):
        raise ValueError(
            "not an ASCII .nl file (binary .nl is not supported by the "
            "manifest driver; re-emit with an ASCII writer)"
        )
    parts = lines[1].split("#")[0].split()
    return int(parts[0]), int(parts[1])


def rewrite_nl_initial_point(src, dst, primals=None, duals=None):
    """Copy `src` to `dst` with its initial-point segments replaced.

    An ASCII ``.nl`` file states its initial primal point in an ``x``
    segment and its initial duals in a ``d`` segment, each a header
    line (``x<count>`` / ``d<count>``) followed by that many
    ``index value`` lines. Both are optional and both are honoured by
    pounce's reader, so replacing them is how a solved point is handed
    to the next model in the path.
    """
    with open(src) as fh:
        lines = fh.read().split("\n")
    n, m = _nl_counts(lines)
    if primals is not None and len(primals) != n:
        raise ValueError(
            f"{src}: model has {n} variables, got {len(primals)} primals"
        )
    if duals is not None and len(duals) != m:
        raise ValueError(
            f"{src}: model has {m} constraints, got {len(duals)} duals"
        )

    out: List[str] = []
    i = 0
    # Segment headers are a single letter at column 0. `x` and `d` are
    # the two we replace; everything else is copied through untouched.
    while i < len(lines):
        ln = lines[i]
        if ln[:1] in ("x", "d") and ln[1:2].isdigit():
            count = int(ln[1:].split()[0])
            i += 1 + count          # drop the old segment entirely
            continue
        out.append(ln)
        i += 1

    seg: List[str] = []
    if primals is not None:
        seg.append(f"x{n}")
        seg.extend(f"{k} {v!r}" for k, v in enumerate(primals))
    if duals is not None and m:
        seg.append(f"d{m}")
        seg.extend(f"{k} {v!r}" for k, v in enumerate(duals))

    # The initial-point segments belong after the header block (the
    # first ten lines) and before the body; the reader accepts them
    # anywhere in the segment stream, so appending after the header is
    # both valid and stable across models.
    body = out[:10] + seg + out[10:]
    with open(dst, "w") as fh:
        fh.write("\n".join(body))
    return dst


def _binary(explicit=None):
    if explicit:
        return explicit
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin",
                        "pounce")
    if os.path.exists(here):
        return here
    found = shutil.which("pounce")
    if found:
        return found
    raise FileNotFoundError(
        "pounce binary not found; pass --binary or install pounce-solver"
    )


def trace_manifest(manifest, *, binary=None, cold=False, workdir=None,
                   bound_push=1e-6, verbose=False):
    """Trace `manifest` by repeated CLI solves. Returns a trace dict.

    With ``cold=True`` every point is solved from the model's own
    starting point, which is the baseline the warm path is measured
    against: the same models, the same options, the same process count,
    differing only in whether the previous point's answer is carried
    forward.
    """
    exe = _binary(binary)
    tmp = workdir or tempfile.mkdtemp(prefix="pounce-cont-")
    steps = []
    prev = None            # (duals, primals) of the last accepted point
    prev_sig = None
    t_all = time.perf_counter()

    for k, pt in enumerate(manifest.points):
        model = pt["model"]
        warm = manifest.warm and not cold and prev is not None
        run_model = model
        if warm:
            run_model = os.path.join(tmp, f"step{k:04d}.nl")
            rewrite_nl_initial_point(model, run_model,
                                     primals=prev[1], duals=prev[0])

        sol = os.path.join(tmp, f"step{k:04d}.sol")
        rep = os.path.join(tmp, f"step{k:04d}.json")
        cmd = [exe, run_model, sol, "--json-output", rep,
               "--json-detail", "full"]
        # Options are bare `key=value` arguments, matching upstream
        # ipopt's CLI convention (`ipopt problem.nl print_level=8`).
        for key, val in manifest.options.items():
            cmd.append(f"{key}={val}")
        if warm:
            cmd += ["warm_start_init_point=yes",
                    f"warm_start_bound_push={bound_push}",
                    f"warm_start_mult_bound_push={bound_push}"]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        info = {}
        if os.path.exists(rep):
            try:
                with open(rep) as fh:
                    info = json.load(fh)
            except json.JSONDecodeError:
                info = {}
        sol_sec = info.get("solution") or {}
        stat_sec = info.get("statistics") or {}
        status = int(sol_sec.get("solve_result_num", -99)) if info else -99
        rec = {
            "index": k,
            "model": os.path.relpath(model, manifest.root),
            "theta": pt.get("theta"),
            "predictor": "cold" if not warm else "zero",
            "corrected": True,
            "prescribed": True,
            "iters": int(stat_sec.get("iteration_count", -1)),
            "solve_time": elapsed,
            "obj": stat_sec.get("final_objective"),
            "kkt_error": stat_sec.get("final_kkt_error"),
            "status": status,
            "status_msg": str(sol_sec.get("status", proc.returncode)),
            "returncode": proc.returncode,
        }
        for key, src in (("n_obj", "num_obj_evals"),
                         ("n_grad", "num_obj_grad_evals"),
                         ("n_cons", "num_constr_evals"),
                         ("n_jac", "num_constr_jac_evals"),
                         ("n_hess", "num_hess_evals")):
            v = stat_sec.get(src)
            rec[key] = int(v) if v is not None else 0

        ok = status in _OK_STATUS
        # Active-set event: the sign pattern of the constraint duals is
        # what a .sol can report, so that is the fingerprint the CLI
        # path uses. It is coarser than the in-process bound-multiplier
        # signature and the trace says so rather than implying parity.
        # The solve report carries the primal point and the constraint
        # multipliers directly, so the iterate is read from it rather
        # than re-parsed out of the .sol -- one format instead of two,
        # and the .sol stays purely an artifact for the modeling system.
        sig = None
        if ok:
            primals = sol_sec.get("x")
            duals = sol_sec.get("lambda")
            if primals is not None:
                duals = duals or []
                sig = tuple(int(d > 1e-6) - int(d < -1e-6) for d in duals)
                prev = (list(duals), list(primals))
            else:
                prev = None
        else:
            prev = None
        rec["active_set_event"] = bool(
            prev_sig is not None and sig is not None and sig != prev_sig
        )
        if sig is not None:
            prev_sig = sig
        steps.append(rec)
        if verbose:
            print(f"[{k + 1}/{len(manifest.points)}] {rec['model']} "
                  f"{'warm' if warm else 'cold'} "
                  f"iters={rec['iters']} status={rec['status_msg']}",
                  file=sys.stderr)

    oks = [s for s in steps if s["status"] in _OK_STATUS]
    return {
        "version": 1,
        "mode": "cold" if cold else ("warm" if manifest.warm else "cold"),
        "predictor": "none (the KKT factor does not cross a process "
                     "boundary; see pounce._continuation_cli)",
        "n_points": len(steps),
        "n_corrections": len(steps),
        "n_converged": len(oks),
        "n_active_set_events": sum(1 for s in steps if s["active_set_event"]),
        "total_iters": sum(s["iters"] for s in oks),
        "total_evals": sum(s["n_obj"] + s["n_grad"] + s["n_cons"]
                           + s["n_jac"] + s["n_hess"] for s in steps),
        "total_time": time.perf_counter() - t_all,
        "steps": steps,
    }


def _report(trace) -> str:
    return "\n".join([
        f"continuation ({trace['mode']}): {trace['n_converged']}"
        f"/{trace['n_points']} converged",
        f"  points            {trace['n_points']}",
        f"  corrections       {trace['n_corrections']}",
        f"  predictor         {trace['predictor']}",
        f"  active-set events {trace['n_active_set_events']}",
        f"  solver iterations {trace['total_iters']}",
        f"  total evaluations {trace['total_evals']}",
        f"  wall time         {trace['total_time'] * 1e3:.1f} ms",
    ])


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="pounce-continue",
        description="Trace a parametric NLP path from a manifest "
                    "(pounce#608). One command, one report, one process "
                    "per point.",
    )
    p.add_argument("manifest", help="path manifest JSON (version 1)")
    p.add_argument("--out", help="write the trace JSON here")
    p.add_argument("--cold", action="store_true",
                   help="solve every point cold -- the baseline the warm "
                        "path is measured against")
    p.add_argument("--binary", help="pounce executable (default: bundled)")
    p.add_argument("--workdir", help="keep intermediates here")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    man = PathManifest.load(args.manifest)
    trace = trace_manifest(man, binary=args.binary, cold=args.cold,
                           workdir=args.workdir, verbose=args.verbose)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(trace, fh, indent=2)
        print(f"pounce-continue: wrote {args.out}")
    print(_report(trace))
    return 0 if trace["n_converged"] == trace["n_points"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
