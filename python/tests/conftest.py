"""Pytest configuration for the pounce Python test suite.

Build-hygiene guard against a **stale compiled extension**.

When the suite runs against an in-repo editable build — the compiled
extension ``python/pounce/_pounce*.so`` sitting next to the package source,
where ``maturin develop`` leaves it — this guard checks that the artifact is
not older than the Rust binding sources it was built from. A stale ``.so`` is
the single most confusing local failure mode: the Rust binding grows a new
keyword argument or function, but pytest imports the old artifact and the
tests die with cryptic ``TypeError: ... unexpected keyword argument`` errors
that read like real bugs rather than "you forgot to rebuild" (this exact
trap cost a debugging session — see dev-notes/pr70-hardening.md, Item H).

We deliberately *fail fast* with an actionable message rather than
auto-rebuilding: a rebuild needs the Rust toolchain and would make test runs
surprisingly slow and stateful. Wheel installs (site-packages) are
unaffected — there is no in-repo ``.so`` next to the sources to compare, so
the guard is skipped, and CI (which builds a fresh wheel every run, then
installs it) never trips it.

Set ``POUNCE_SKIP_EXT_STALE_CHECK=1`` to bypass.
"""

import os
from pathlib import Path

import pytest


def _newest_rust_mtime(crates_dir: Path) -> float:
    """Newest mtime among the workspace's Rust sources and crate manifests.

    The extension statically links the whole workspace, so an edit to *any*
    crate (not just ``pounce-py``) can change its behavior; comparing against
    all of ``crates/`` is the conservative choice. A false "stale" verdict is
    harmless — it just asks for a rebuild, which is cheap and always correct.
    """
    newest = 0.0
    for p in crates_dir.rglob("*"):
        if p.suffix == ".rs" or p.name == "Cargo.toml":
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return newest


def _check_extension_freshness() -> None:
    if os.environ.get("POUNCE_SKIP_EXT_STALE_CHECK"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    pkg_dir = repo_root / "python" / "pounce"
    crates_dir = repo_root / "crates"
    # Only meaningful for an in-repo source checkout that has the editable
    # extension built in place. A wheel install has no sibling Rust sources
    # (or no in-repo `.so`), so there is nothing to go stale — skip silently.
    if not crates_dir.is_dir():
        return
    built = sorted(pkg_dir.glob("_pounce*.so")) + sorted(pkg_dir.glob("_pounce*.pyd"))
    if not built:
        return
    so_mtime = max(p.stat().st_mtime for p in built)
    src_mtime = _newest_rust_mtime(crates_dir)
    if so_mtime < src_mtime:
        newest_so = max(built, key=lambda p: p.stat().st_mtime)
        raise pytest.UsageError(
            f"pounce compiled extension is STALE: {newest_so.name} is older "
            "than the Rust sources under crates/. Running pytest now would "
            "import the old binding and fail with confusing errors (e.g. "
            "'unexpected keyword argument'). Rebuild it first:\n"
            "    cd python && maturin develop    # rebuild in place, or\n"
            "    make python-test                # rebuild then run pytest\n"
            "(set POUNCE_SKIP_EXT_STALE_CHECK=1 to bypass this guard.)"
        )


def pytest_configure(config):  # noqa: ARG001 (pytest hook signature)
    _check_extension_freshness()
