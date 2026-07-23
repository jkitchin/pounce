"""Tests for the binary-resolution hardening (gh #315).

The concern: ``SolverFactory('pounce')`` can be driven by a stale or unrelated
`pounce` binary, and a *version string* cannot tell two builds apart (a binary
from before and after a fix can both report the same ``X.Y.Z`` — precisely the
dual-sign fix, gh #271/#272, where a stale 0.9.0 binary returned flipped
duals). The plugin therefore discriminates on the git *commit* embedded in
``pounce --about``, warns when it falls back to a PATH binary, and exposes
``check_binary()`` to report exactly which executable will run and whether a
different one shadows it on PATH.
"""

import warnings

import pytest

import pyomo_pounce
from pyomo_pounce import pounce_solver as ps


# ── build-id discrimination (the mechanism version strings can't provide) ──


def test_build_id_parses_commit_from_about():
    """``_build_id`` extracts the git commit from ``pounce --about``."""
    bundled = ps._bundled_path()
    if bundled is None:
        pytest.skip("no bundled pounce binary in this environment")
    bid = ps._build_id(bundled)
    assert bid is not None and len(bid) >= 6
    # It is a hex commit, not a version string.
    int(bid, 16)


def test_build_id_none_on_missing_or_bad_executable():
    assert ps._build_id(None) is None
    assert ps._build_id("/definitely/no/such/pounce/binary") is None


# ── check_binary() ─────────────────────────────────────────────────────────


def test_check_binary_reports_resolved_and_bundled():
    info = pyomo_pounce.check_binary(verbose=False)
    assert set(
        [
            "resolved_executable",
            "resolved_build_id",
            "bundled_executable",
            "bundled_build_id",
            "using_bundled",
            "matches_bundled",
            "path_pounce_binaries",
            "shadowing_path_binaries",
        ]
    ).issubset(info)
    # When a bundled binary exists it must be the one that runs, and match.
    if info["bundled_executable"] is not None:
        assert info["using_bundled"] is True
        assert info["matches_bundled"] is True
        assert info["resolved_build_id"] == info["bundled_build_id"]


def test_check_binary_flags_a_shadowing_build(tmp_path, monkeypatch):
    """A *different-build* `pounce` earlier on PATH is reported as shadowing;
    an identical-build copy is not."""
    resolved = pyomo_pounce.check_binary(verbose=False)["resolved_executable"]
    if resolved is None:
        pytest.skip("no pounce executable resolvable")

    # A fake `pounce` on PATH whose `--about` reports a DIFFERENT commit.
    fake_dir = tmp_path / "stale"
    fake_dir.mkdir()
    fake = fake_dir / "pounce"
    fake.write_text(
        "#!/bin/sh\n"
        'echo "pounce 0.9.0 (commit deadbeef, built 2000-01-01T00:00:00Z)"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + __import__("os").environ["PATH"])

    info = pyomo_pounce.check_binary(verbose=False)
    shadow_ids = {b["build_id"] for b in info["shadowing_path_binaries"]}
    assert "deadbeef" in shadow_ids, info["path_pounce_binaries"]


# ── the fallback warning ────────────────────────────────────────────────────


def test_default_executable_warns_on_path_fallback(monkeypatch):
    """When no bundled binary exists and the plugin falls back to a PATH
    binary, it emits a one-time UserWarning naming the resolved executable."""
    # Force the "no bundled binary" branch.
    monkeypatch.setattr(ps, "_bundled_path", lambda: None)
    monkeypatch.setattr(ps.shutil, "which", lambda name: "/some/path/pounce")
    # Reset the one-time latch so the warning can fire in this test.
    monkeypatch.setattr(ps, "_fallback_warned", False)
    monkeypatch.setattr(ps, "_build_id", lambda exe: "abc123")

    from pyomo.opt import SolverFactory

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe = SolverFactory("pounce")._default_executable()
    assert exe == "/some/path/pounce"
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("no wheel-bundled" in m and "/some/path/pounce" in m for m in msgs)


def test_default_executable_no_warning_when_bundled_present():
    """The common case (bundled binary present) must be silent."""
    if ps._bundled_path() is None:
        pytest.skip("no bundled binary to exercise the quiet path")
    from pyomo.opt import SolverFactory

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SolverFactory("pounce")._default_executable()
    assert not [
        w for w in caught if "no wheel-bundled" in str(w.message)
    ]
