"""Starting the `pounce` console script must not import the `pounce` package (gh #1086).

The console script used to be `pounce._cli:main`. Importing that imports the
`pounce` package first, and `import pounce` pulls in scipy and numpy through
the solver facades: about 175 ms before the shim `exec`s the Rust binary.
Pyomo's ASL layer runs the executable twice per solve, so every Pyomo solve
paid ~0.35 s of startup for a binary that finishes small models in
milliseconds (HS100: 0.36 s through `SolverFactory('pounce')` vs Ipopt's
0.011 s).

The entry point is now `_pounce_cli:main`, a package beside `pounce` that
loads `pounce/_cli.py` by path. These tests hold it to that by what gets
*imported*, not by wall-clock time, which would be flaky on CI runners: the
cost is entirely module imports, so "no `pounce`, no scipy, no numpy" is the
property that keeps it cheap.
"""

import subprocess
import sys
from pathlib import Path

_HEAVY = ("pounce", "scipy", "numpy")


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )


def test_loading_the_shim_imports_neither_pounce_nor_its_dependencies():
    code = (
        "import sys, _pounce_cli\n"
        "cli = _pounce_cli._load_cli()\n"
        "assert callable(cli.main) and callable(cli.resolve_binary)\n"
        f"print(sorted(m for m in {_HEAVY!r} if m in sys.modules))\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        "the console script's startup imported "
        f"{out.stdout.strip()}; every module here is paid on every CLI call, "
        "twice per Pyomo solve (gh #1086)"
    )


def test_the_shim_runs_the_cli_from_the_pounce_package_directory():
    # `_cli.py`'s `__file__` must still be `pounce/_cli.py`: the wheel's
    # bundled `pounce/bin/` and the checkout fallback (gh #816) both resolve
    # relative to it.
    code = (
        "import importlib.util, _pounce_cli\n"
        "cli = _pounce_cli._load_cli()\n"
        "spec = importlib.util.find_spec('pounce')\n"
        "print(cli.__file__)\n"
        "print(list(spec.submodule_search_locations)[0])\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    cli_file, pkg_dir = out.stdout.split()
    assert Path(cli_file).parent == Path(pkg_dir)
    assert Path(cli_file).name == "_cli.py"


def test_the_console_script_entry_point_is_the_light_one():
    # The installed metadata is what pip turns into the `pounce` executable;
    # pointing it back at `pounce._cli` would bring the cost back silently.
    from importlib.metadata import entry_points

    eps = entry_points()
    scripts = (
        eps.select(group="console_scripts")
        if hasattr(eps, "select")
        else eps.get("console_scripts", [])
    )
    pounce_ep = [ep for ep in scripts if ep.name == "pounce"]
    assert pounce_ep, "no `pounce` console script installed"
    assert pounce_ep[0].value == "_pounce_cli:main"


def test_the_version_probe_pyomo_runs_still_answers():
    # `pounce -v` through the light entry point: what Pyomo's ASL layer runs
    # before every solve. Skipped where no binary is resolvable (a bare
    # `maturin develop` with nothing built), which test_issue_816 covers.
    code = (
        "import sys, _pounce_cli\n"
        "cli = _pounce_cli._load_cli()\n"
        "print(cli.resolve_binary()[0] is not None)\n"
    )
    have = _run(code).stdout.strip() == "True"
    if not have:
        import pytest

        pytest.skip("no CLI binary resolvable in this environment")
    out = _run("import sys, _pounce_cli; sys.argv = ['pounce', '-v']; _pounce_cli.main()")
    assert out.returncode == 0, out.stderr
    assert "POUNCE" in out.stdout or "pounce" in out.stdout.lower()


def test_import_pounce_does_not_import_scipy():
    # The same cost one level down: `import pounce` imported scipy through
    # `_minimize` (~150 ms of its ~175). Every process that needs only part
    # of the package paid it, and some pay it per solve: the GAMS link is a
    # new process for every GAMS solve. The facades now load on first use.
    out = _run(
        "import sys, pounce\n"
        "print('scipy' in sys.modules)\n"
        "from pounce import minimize\n"
        "print('scipy' in sys.modules)\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "True"], (
        "scipy must load when a facade that needs it is used, and not before"
    )


def test_the_gams_link_import_path_does_not_import_scipy():
    out = _run(
        "import sys, pounce, pounce.gams.link\n"
        "print('scipy' in sys.modules)\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_every_public_name_still_resolves():
    # Laziness must not change the surface: each name in `__all__` resolves,
    # through attribute access and through `from pounce import *`.
    out = _run(
        "import pounce\n"
        "missing = [n for n in pounce.__all__ if not hasattr(pounce, n)]\n"
        "ns = {}\n"
        "exec('from pounce import *', ns)\n"
        "missing += [n for n in pounce.__all__ if n not in ns]\n"
        "import pounce.qp\n"
        "assert pounce.qp.solve_qp is pounce.solve_qp\n"
        "print(missing)\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
