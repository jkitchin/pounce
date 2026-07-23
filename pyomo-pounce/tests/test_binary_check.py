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

import os
import warnings

import pytest

import pyomo_pounce
from pyomo_pounce import pounce_solver as ps


def _write_fake_pounce(path, about_first_line):
    """A minimal `pounce` whose `--about` prints the given first line."""
    path.write_text("#!/bin/sh\n" f'echo "{about_first_line}"\n')
    path.chmod(0o755)
    return path


# ── build-id discrimination (the mechanism version strings can't provide) ──


def test_build_id_parses_commit_from_about():
    """``_build_id`` extracts the git commit from ``pounce --about``."""
    bundled = ps._bundled_path()
    if bundled is None:
        pytest.skip("no bundled pounce binary in this environment")
    bid = ps._build_id(bundled)
    assert bid is not None and len(bid) >= 6
    # It is a hex commit (optionally carrying a `+dirty` marker for a build
    # with uncommitted changes), not a version string.
    int(bid.split("+")[0], 16)


def test_build_id_none_on_missing_or_bad_executable():
    assert ps._build_id(None) is None
    assert ps._build_id("/definitely/no/such/pounce/binary") is None


def test_build_id_keeps_dirty_suffix(tmp_path):
    """A build with uncommitted changes reports ``<commit>+dirty``; that
    marker is kept, so it is distinguished from the clean build at the same
    commit — a genuine "same commit, different bits" case."""
    clean = _write_fake_pounce(
        tmp_path / "clean",
        "pounce 0.9.0 (commit 96fc5890, built 2026-01-01T00:00:00Z)",
    )
    dirty = _write_fake_pounce(
        tmp_path / "dirty",
        "pounce 0.9.0 (commit 96fc5890+dirty, built 2026-01-01T00:00:00Z)",
    )
    assert ps._build_id(str(clean)) == "96fc5890"
    assert ps._build_id(str(dirty)) == "96fc5890+dirty"
    assert ps._build_id(str(clean)) != ps._build_id(str(dirty))


def test_build_id_treats_unknown_commit_as_unqueryable(tmp_path):
    """A binary built outside a git checkout reports ``commit unknown``. That
    must read as None, so two independent such builds never compare equal
    (and never look like a match to the bundled binary)."""
    exe = _write_fake_pounce(
        tmp_path / "pounce",
        "pounce 0.9.0 (commit unknown, built 2026-01-01T00:00:00Z)",
    )
    assert ps._build_id(str(exe)) is None


# ── PATH scan (the candidates ASL fallback would pick from) ─────────────────


def test_all_path_pounce_finds_binary(tmp_path, monkeypatch):
    exe = _write_fake_pounce(
        tmp_path / "pounce", "pounce 0.9.0 (commit abc123, built x)"
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    found = ps._all_path_pounce()
    assert any(os.path.realpath(f) == os.path.realpath(str(exe)) for f in found)


def test_all_path_pounce_resolves_via_which(monkeypatch):
    """The scan resolves each PATH entry through ``shutil.which`` with the
    platform's executable name, rather than a bare-``pounce`` filename test.
    The pre-fix code used ``os.path.join(d, "pounce")`` directly, so it never
    called ``which`` — and on Windows (name ``pounce.exe``) found nothing,
    silently reporting no shadowing."""
    calls = []
    monkeypatch.setattr(
        ps.shutil, "which", lambda name, path=None: calls.append(name) or None
    )
    monkeypatch.setenv("PATH", "/nonexistent-dir-xyz")
    ps._all_path_pounce()
    assert calls, "the scan must resolve PATH entries through shutil.which"
    # The name carries the platform's executable extension (`pounce.exe` on
    # Windows), so the shadowing scan is not silently empty there.
    expected = "pounce.exe" if os.name == "nt" else "pounce"
    assert all(c == expected for c in calls)


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
