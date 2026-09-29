"""The cross-repository Gate-1 flash artifact has a packaged versioned schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest


def _schema() -> dict:
    resource = Path(__file__).parents[1] / "pounce/examples/flash_results_v2.schema.json"
    return json.loads(resource.read_text(encoding="utf-8"))


def _residual(value: float = 0.0, *, admitted_scale=None) -> dict:
    return {
        "value": value,
        "definition": "max absolute source-row violation",
        "admitted_scale": admitted_scale,
    }


def _point() -> dict:
    return {
        "beta": 0.5,
        "x": [0.25, 0.75],
        "y": [0.75, 0.25],
        "sum_x": 1.0,
        "sum_y": 1.0,
        "compressibility": {"liquid": 0.04, "vapor": 0.87},
        "root_branches": {
            "liquid": "three_real_roots",
            "vapor": "three_real_roots",
        },
        "nontrivial_branch": "component_0_above",
        "regime": "two_phase",
        "active_branches": {"vapor": ["slack_zero"], "liquid": ["slack_zero"]},
    }


def _source() -> dict:
    return {
        "balance": _residual(),
        "isofugacity": _residual(),
        "eos": _residual(),
        "root_selection": _residual(),
        "nontriviality": _residual(),
        "bound": _residual(),
        "sign": _residual(),
        "complementarity": _residual(1e-8, admitted_scale=1e-4),
    }


def _artifact() -> dict:
    artifact = {
        "schema": "pounce-flash-results/2",
        "issue": "jkitchin/discopt#1526; jkitchin/pounce#776 Gate 1",
        "stamp": {
            "repositories": {
                "pounce": {"commit": "abcdef0123456789", "describe": "abcdef0"},
                "discopt": {
                    "present": True,
                    "version": "0.9.1.dev0",
                    "commit": "0123456789abcdef",
                    "comparison_run": True,
                    "reason": "five-point comparison completed",
                },
            },
            "model_data_revision": "0123456789abcdef",
            "environment": {"python": "3.12"},
            "model": {
                "case": "ethane_n_butane_10bar",
                "components": ["ethane", "n-butane"],
                "feed_composition": [0.5, 0.5],
                "pressure_pa": 1e6,
                "temperatures_k": [300.0],
            },
            "started_utc": "2026-09-28T00:00:00Z",
            "wall_s": 1.0,
        },
        "config": {
            "base_options": {"tol": 1e-8},
            "supported_route": "scholtes_then_ncp",
            "tau_schedule": [1e-2, 1e-4, 1e-8],
        },
        "oracle": {
            "method": "Michelsen tangent-plane stability plus Rachford-Rice",
            "rows": [
                {
                    "temperature_k": 300.0,
                    "regime": "two_phase",
                    "beta": 0.5,
                    "sum_x": 1.0,
                    "sum_y": 1.0,
                }
            ],
        },
        "hysteresis": {
            "disagreements": {},
            "iterations": {},
            "failures": {},
            "cold_legs_agree": True,
            "path_dependent": False,
        },
        "legs": [],
        "comparison": {
            "state": "complete",
            "discopt_issue": "jkitchin/discopt#1526",
            "temperatures_k": [300.0],
            "methods": ["gdp", "sos1", "scholtes"],
            "records": [
                {
                    "temperature_k": 300.0,
                    "method": "scholtes",
                    "state": "local",
                    "status": "local_optimal",
                    "gap_certified": False,
                    "objective": 0.0,
                    "bound": None,
                    "gap": None,
                    "node_count": None,
                    "wall_s": 0.5,
                    "point": _point(),
                    "source": _source(),
                    "lowered": {"row": _residual()},
                    "oracle": {
                        "regime_match": True,
                        "biactive_branch_admissible": None,
                        "beta_error": 0.0,
                        "x_error": 0.0,
                        "y_error": 0.0,
                        "sum_x_error": 0.0,
                        "sum_y_error": 0.0,
                        "z_liquid_error": 0.0,
                        "z_vapor_error": 0.0,
                        "root_branches_match": True,
                    },
                    "error": None,
                }
            ],
            "reason": None,
        },
    }
    local = artifact["comparison"]["records"][0]
    for method in ("gdp", "sos1"):
        certified = copy.deepcopy(local)
        certified.update(
            method=method,
            state="certified",
            status="optimal",
            gap_certified=True,
            bound=0.0,
            gap=0.0,
            node_count=1,
        )
        artifact["comparison"]["records"].append(certified)
    return artifact


def _errors(artifact: dict) -> list[str]:
    validator = jsonschema.Draft7Validator(_schema())
    return [err.message for err in validator.iter_errors(artifact)]


def test_schema_is_valid_draft_7():
    jsonschema.Draft7Validator.check_schema(_schema())


def test_a_complete_cross_repository_artifact_validates():
    assert _errors(_artifact()) == []


@pytest.mark.parametrize("method", ["gdp", "sos1", "scholtes"])
def test_a_complete_comparison_requires_every_declared_method(method):
    artifact = _artifact()
    artifact["comparison"]["records"] = [
        row for row in artifact["comparison"]["records"] if row["method"] != method
    ]
    assert _errors(artifact), f"complete comparison accepted without {method}"


@pytest.mark.parametrize("oracle", [None, {}])
def test_a_successful_record_requires_complete_oracle_evidence(oracle):
    artifact = _artifact()
    artifact["comparison"]["records"][0]["oracle"] = oracle
    assert _errors(artifact)


def test_local_and_certified_states_are_tied_to_their_method_classes():
    artifact = _artifact()
    artifact["comparison"]["records"][0]["method"] = "gdp"
    assert _errors(artifact), "an exact GDP arm cannot be labeled local"

    artifact = _artifact()
    certified = next(
        row for row in artifact["comparison"]["records"] if row["state"] == "certified"
    )
    certified["method"] = "scholtes"
    assert _errors(artifact), "the Scholtes arm cannot claim a global certificate"


def test_a_local_result_cannot_carry_a_certified_bound():
    artifact = _artifact()
    record = artifact["comparison"]["records"][0]
    record["gap_certified"] = True
    record["bound"] = 0.0
    errors = _errors(artifact)
    assert errors
    assert any("False" in message or "None" in message for message in errors)


def test_every_source_residual_keeps_its_definition():
    artifact = _artifact()
    del artifact["comparison"]["records"][0]["source"]["complementarity"]["definition"]
    assert any("definition" in message for message in _errors(artifact))


@pytest.mark.parametrize("name", ["eos", "root_selection", "nontriviality"])
def test_algebraic_root_checks_cannot_be_omitted(name):
    artifact = _artifact()
    del artifact["comparison"]["records"][0]["source"][name]
    assert any(name in message for message in _errors(artifact))


@pytest.mark.parametrize("name", ["compressibility", "root_branches", "nontrivial_branch"])
def test_root_point_data_cannot_be_omitted(name):
    artifact = _artifact()
    del artifact["comparison"]["records"][0]["point"][name]
    assert any(name in message for message in _errors(artifact))


def test_lowered_residuals_are_not_mixed_into_source_residuals():
    artifact = _artifact()
    record = artifact["comparison"]["records"][0]
    record["source"]["lowered_row"] = record.pop("lowered")["row"]
    errors = _errors(artifact)
    assert any("lowered" in message for message in errors)


def test_version_2_still_checks_version_1_leg_records():
    artifact = _artifact()
    artifact["legs"] = [
        {
            "direction": "up",
            "start_mode": "cold",
            "route": "scholtes_then_ncp",
            "records": [{}],
        }
    ]
    errors = _errors(artifact)
    assert any("case" in message for message in errors)


def test_a_certificate_requires_the_reported_bound_and_gap():
    artifact = _artifact()
    record = next(row for row in artifact["comparison"]["records"] if row["method"] == "gdp")
    record["bound"] = None
    record["gap"] = None
    assert any("number" in message for message in _errors(artifact))


def test_each_phase_names_at_least_one_active_complementarity_branch():
    artifact = _artifact()
    artifact["comparison"]["records"][0]["point"]["active_branches"]["vapor"] = []
    assert any("non-empty" in message for message in _errors(artifact))


@pytest.mark.parametrize("state", ["not_run", "failed"])
def test_an_unrun_or_failed_record_cannot_smuggle_a_point(state):
    artifact = _artifact()
    artifact["comparison"]["records"][0]["state"] = state
    assert any("None" in message for message in _errors(artifact))


@pytest.mark.parametrize(
    "discopt_provenance",
    [
        {"present": True, "commit": "0123456789abcdef"},
        {"present": False},
        {"present": False, "commit": None},
    ],
    ids=["installed", "absent-without-commit", "absent-with-null-commit"],
)
def test_not_run_is_distinct_from_an_absent_version_1_comparison(discopt_provenance):
    artifact = _artifact()
    artifact["stamp"]["repositories"]["discopt"] = {
        **discopt_provenance,
        "comparison_run": False,
        "reason": "comparison environment unavailable",
    }
    artifact["comparison"] = {
        "state": "not_run",
        "discopt_issue": "jkitchin/discopt#1526",
        "temperatures_k": [250.0, 268.0, 300.0, 324.0, 350.0],
        "methods": ["gdp", "sos1", "scholtes"],
        "records": [],
        "reason": "comparison environment unavailable",
    }
    assert _errors(artifact) == []

    absent = copy.deepcopy(artifact)
    del absent["comparison"]
    assert any("comparison" in message for message in _errors(absent))


@pytest.mark.parametrize(
    "discopt_provenance",
    [
        {"present": False, "commit": "0123456789abcdef"},
        {"present": True},
        {"present": True, "commit": None},
        {"present": True, "commit": "abc"},
    ],
    ids=["absent", "missing-commit", "null-commit", "short-commit"],
)
def test_complete_comparison_requires_discopt_provenance(discopt_provenance):
    artifact = _artifact()
    artifact["stamp"]["repositories"]["discopt"] = {
        **discopt_provenance,
        "comparison_run": True,
        "reason": "comparison completed",
    }
    assert _errors(artifact)
