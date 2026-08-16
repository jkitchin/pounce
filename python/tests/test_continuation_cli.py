"""Path manifest / repeated-solve protocol for the CLI (pounce#608).

pounce#608's scope note says CLI and GAMS "can follow using a path
manifest or repeated-solve protocol". These cover the manifest parser,
the `.nl` initial-point rewrite that carries a solved point into the
next model, and -- when a `pounce` binary and the CLI fixture corpus
are both present -- one end-to-end trace.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from pounce._continuation_cli import (
    PathManifest,
    parse_sol,
    rewrite_nl_initial_point,
    trace_manifest,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIXTURES = os.path.join(_ROOT, "crates", "pounce-cli", "tests", "fixtures")


def _works(path):
    """A path is only a usable CLI if it actually runs.

    `shutil.which("pounce")` finds the wheel's *console-script shim*,
    which is present whenever pounce-solver is installed but only
    execs a real binary when the wheel was built with `pounce/bin/`
    staged. CI builds the wheel straight from `maturin-action` with no
    staging step, so the shim is on PATH, exits 1, and every solve in
    a trace silently returns nothing — which is how this arrived as
    `assert 0 == 2` rather than as a skip.
    """
    if path is None:
        return False
    try:
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _binary():
    local = os.path.join(_ROOT, "target", "release", "pounce")
    if os.path.exists(local):
        return local
    bundled = os.path.join(_ROOT, "python", "pounce", "bin", "pounce")
    if os.path.exists(bundled):
        return bundled
    return shutil.which("pounce")


_CLI = _binary() if _works(_binary()) else None

needs_cli = pytest.mark.skipif(
    _CLI is None or not os.path.isdir(_FIXTURES),
    reason="needs a *working* pounce binary (a wheel built without "
           "pounce/bin/ staged leaves only a shim that exits 1) and the "
           "CLI fixture corpus",
)


# -- manifest ----------------------------------------------------------


def _write(tmp_path, obj, name="path.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_manifest_round_trips(tmp_path):
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "a.nl", "theta": [1.0]},
                   {"model": "b.nl"}],
        "options": {"tol": "1e-8"},
    }))
    assert len(man.points) == 2
    assert man.warm is True
    assert man.options == {"tol": "1e-8"}
    # Model paths resolve against the manifest, not the cwd, so a
    # manifest is movable with its models.
    assert os.path.dirname(man.points[0]["model"]) == str(tmp_path)
    assert man.points[0]["theta"] == [1.0]
    assert man.points[1]["theta"] is None


def test_manifest_rejects_a_future_version(tmp_path):
    with pytest.raises(ValueError, match="unsupported version"):
        PathManifest.load(_write(tmp_path, {"version": 2, "points": []}))


def test_manifest_rejects_a_point_without_a_model(tmp_path):
    with pytest.raises(ValueError, match="no 'model'"):
        PathManifest.load(_write(tmp_path, {
            "version": 1, "points": [{"theta": [1.0]}]}))


def test_manifest_rejects_an_empty_path(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        PathManifest.load(_write(tmp_path, {"version": 1, "points": []}))


# -- .nl initial-point rewrite ----------------------------------------


_NL = """g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs
 0 0	# network constraints
 0 2 0 	# nonlinear vars
 0 0 0 1	# linear network variables
 0 0 0 0 0 	# discrete variables
 2 2 	# nonzeros
 0 0	# max name lengths
 0 0 0 0 0	# common exprs
x2
0 0.0
1 0.0
C0
n0
"""


def _src(tmp_path, text=_NL):
    p = tmp_path / "m.nl"
    p.write_text(text)
    return str(p)


def test_rewrite_replaces_the_initial_point(tmp_path):
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.5, -2.5],
                             duals=[0.25])
    lines = open(dst).read().split("\n")
    heads = [ln for ln in lines if ln[:1] in ("x", "d") and ln[1:2].isdigit()]
    # Exactly one of each -- the old x segment is replaced, not appended
    # to, or the reader would see two starting points.
    assert heads == ["x2", "d1"]
    assert "0 1.5" in lines and "1 -2.5" in lines
    assert "0 0.25" in lines
    # The model body is untouched.
    assert "C0" in lines and "n0" in lines


def test_rewrite_omits_the_dual_segment_when_no_duals_are_given(tmp_path):
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0, 2.0])
    heads = [ln for ln in open(dst).read().split("\n")
             if ln[:1] in ("x", "d") and ln[1:2].isdigit()]
    assert heads == ["x2"]


def test_rewrite_checks_the_lengths_against_the_model(tmp_path):
    dst = str(tmp_path / "out.nl")
    with pytest.raises(ValueError, match="2 variables"):
        rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0])
    with pytest.raises(ValueError, match="1 constraints"):
        rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0, 2.0],
                                 duals=[0.1, 0.2])


def test_rewrite_rejects_a_binary_nl(tmp_path):
    p = tmp_path / "b.nl"
    p.write_text("b3 1 1 0\n 2 1 1 0 1\n")
    with pytest.raises(ValueError, match="ASCII"):
        rewrite_nl_initial_point(str(p), str(tmp_path / "o.nl"),
                                 primals=[1.0, 2.0])


# -- .sol parsing ------------------------------------------------------


def test_parse_sol_splits_duals_from_primals(tmp_path):
    p = tmp_path / "s.sol"
    p.write_text("POUNCE: done\n\nOptions\n3\n0\n1\n0\n2\n2\n3\n3\n"
                 "10.0\n11.0\n1.0\n2.0\n3.0\n")
    duals, primals = parse_sol(str(p), 3, 2)
    assert duals == [10.0, 11.0]
    assert primals == [1.0, 2.0, 3.0]


def test_parse_sol_rejects_a_non_sol_file(tmp_path):
    p = tmp_path / "s.sol"
    p.write_text("not a sol file\n")
    with pytest.raises(ValueError, match="Options"):
        parse_sol(str(p), 1, 1)


# -- end to end --------------------------------------------------------


@needs_cli
def test_trace_manifest_solves_every_point(tmp_path):
    """A two-point path over the same model is the degenerate
    continuation: the second point starts from the first's answer."""
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl", "theta": [0.0]},
                   {"model": "p1.nl", "theta": [0.0]}],
        "options": {"print_level": "0", "sb": "yes"},
    }))
    trace = trace_manifest(man, binary=_CLI, workdir=str(tmp_path))
    assert trace["n_points"] == 2
    assert trace["n_converged"] == 2
    assert all(s["status"] in (0, 1) for s in trace["steps"])
    # The predictor is reported as absent rather than silently implied:
    # the KKT factor does not cross a process boundary.
    assert "process boundary" in trace["predictor"]
    assert trace["steps"][0]["predictor"] == "cold"
    assert trace["steps"][1]["predictor"] == "zero"


@needs_cli
def test_cold_mode_does_not_transfer(tmp_path):
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl"}, {"model": "p1.nl"}],
        "options": {"print_level": "0", "sb": "yes"},
    }))
    trace = trace_manifest(man, binary=_CLI, cold=True,
                           workdir=str(tmp_path))
    assert trace["mode"] == "cold"
    assert [s["predictor"] for s in trace["steps"]] == ["cold", "cold"]


@needs_cli
def test_cli_entry_point_runs(tmp_path):
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    shutil.copy(src, tmp_path / "p0.nl")
    man = _write(tmp_path, {
        "version": 1, "points": [{"model": "p0.nl"}],
        "options": {"print_level": "0", "sb": "yes"},
    })
    out = str(tmp_path / "trace.json")
    proc = subprocess.run(
        [sys.executable, "-m", "pounce._continuation_cli", man,
         "--out", out, "--binary", _CLI],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    with open(out) as fh:
        trace = json.load(fh)
    assert trace["n_points"] == 1
