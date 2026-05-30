"""Tests for the interactive debugger-artifact viewer (`pounce-dbg-viz`).

Loads the module by file path so it doesn't require the compiled
`pounce._pounce` extension; skips if plotly isn't installed.
"""

import importlib.util
import pathlib

import pytest

pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

_MOD = pathlib.Path(__file__).resolve().parent.parent / "pounce" / "_debug_viz.py"
_spec = importlib.util.spec_from_file_location("_pounce_debug_viz", _MOD)
dv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dv)


def test_kkt_artifact_builds_symmetric_spy():
    art = {
        "label": "kkt",
        "iter": 0,
        "n_pos": 2,
        "n_neg": 1,
        "expected_neg": 1,
        "inertia_correct": True,
        "delta_w": 0.0,
        "delta_c": 0.0,
        "status": "Success",
        "matrix": {
            "dim": 3,
            "irn": [1, 2, 3, 3],
            "jcn": [1, 2, 1, 3],
            "vals": [2.0, 2.0, 1.0, -1.0],
            "format": "triplet_1based_lower",
        },
    }
    fig = dv.figure_for(go, art)
    assert fig is not None and len(fig.data) == 1
    # Off-diagonal (3,1) is mirrored to (1,3): 4 stored + 1 mirror = 5.
    assert len(fig.data[0].x) == 5
    assert "KKT matrix" in fig.layout.title.text


def test_l_factor_adds_unit_diagonal():
    art = {
        "label": "L",
        "iter": 1,
        "n": 2,
        "perm": [0, 1],
        "l_irn": [2],
        "l_jcn": [1],
        "l_vals": [0.5],
        "format": "strict_lower_1based_permuted",
    }
    fig = dv.figure_for(go, art)
    assert fig is not None
    # one strict-lower entry + 2 implicit unit-diagonal markers.
    assert len(fig.data[0].x) == 3


def test_vector_block_builds_bar():
    fig = dv.figure_for(go, {"label": "x", "iter": 3, "values": [1.0, -2.0, 3.0]})
    assert fig is not None
    assert list(fig.data[0].y) == [1.0, -2.0, 3.0]


def test_save_artifact_plots_primal():
    art = {"iter": 4, "mu": 0.1, "iterate": {"x": [1.0, 2.0], "z_l": [0.1, 0.2]}}
    fig = dv.figure_for(go, art)
    assert fig is not None
    assert list(fig.data[0].y) == [1.0, 2.0]


def test_unknown_artifact_returns_none():
    assert dv.figure_for(go, {"label": "mystery"}) is None
