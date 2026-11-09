"""``pounce.diagnose``: the Python face of ``pounce diagnose`` (gh#1160).

Runs the CLI binary on committed ``.nl`` fixtures and checks the report is
parsed faithfully: a healthy model is clean, a model with contradictory rows
has a proof-level error, a point can be supplied in memory, and bad
arguments fail before the binary runs.
"""

from pathlib import Path

import pytest

import pounce

FIXTURES = (
    Path(__file__).resolve().parents[2] / "crates" / "pounce-cli" / "tests" / "fixtures"
)


def _have_binary():
    try:
        pounce._diagnose._binary()
        return True
    except FileNotFoundError:
        return False


needs_cli = pytest.mark.skipif(
    not FIXTURES.exists() or not _have_binary(),
    reason="needs a source checkout and the pounce CLI binary",
)


@needs_cli
def test_healthy_model_is_clean():
    r = pounce.diagnose(FIXTURES / "parametric.nl")
    assert r.verdict == "CLEAN"
    assert not r.has_errors
    assert r.exit_code == 0
    assert r.raw["schema"] == "pounce.diagnose/v1"
    assert {c["check"] for c in r.checks} >= {"bounds", "numerical_rank", "start_point"}
    assert "pounce diagnose" in str(r)


@needs_cli
def test_contradictory_rows_are_a_proof_level_error():
    r = pounce.diagnose(FIXTURES / "inconsistent_eq_qp.nl")
    assert r.has_errors
    assert r.exit_code == 22
    (f,) = r.by_code("inconsistent_parallel_rows")
    assert f["severity"] == "error"
    assert f["basis"] == "proof"


@needs_cli
def test_in_memory_point():
    r = pounce.diagnose(FIXTURES / "parametric.nl", x0=[1.0] * 5)
    assert r.point == "the point passed as x0"


@needs_cli
def test_extra_points_reach_rank_robustness():
    r = pounce.diagnose(
        FIXTURES / "parametric.nl", extra_points=[[1.0] * 5, [2.0] * 5]
    )
    points = r.raw["sections"]["rank_robustness"]["points"]
    assert len(points) == 3  # the main point and the two extras


def test_argument_errors_fail_fast(tmp_path):
    with pytest.raises(ValueError):
        pounce.diagnose(tmp_path / "m.nl", x0=[0.0], sol_file="a.sol")
    with pytest.raises(FileNotFoundError):
        pounce.diagnose(tmp_path / "missing.nl")
