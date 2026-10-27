"""Entry point of the `pounce` console script (gh #1086).

The shim's logic lives in `pounce/_cli.py`, but the console script cannot
import it as `pounce._cli`: Python imports a package before any of its
submodules, and `import pounce` pulls in scipy and numpy through the solver
facades — about 175 ms, measured with `python -X importtime`, all of it spent
before the shim `exec`s the Rust binary. Pyomo's ASL layer runs the
executable twice per solve (a `-v` version probe, then the solve), so a
seven-variable model that the binary finishes in 7 ms took 0.36 s through
`SolverFactory('pounce')`, against Ipopt's 0.011 s.

So this package sits beside `pounce` rather than inside it. It locates the
`pounce` package with `importlib.util.find_spec`, which finds a top-level
package without executing its `__init__`, and loads `_cli.py` from there by
path. `_cli.py` imports nothing but the standard library, and its `__file__`
is still `pounce/_cli.py`, so the wheel-bundled `pounce/bin/` and the
source-checkout fallback (gh #816) resolve exactly as they do when it is
imported normally.

Nothing here may import `pounce` or anything that does: the import time of
this module *is* the console script's overhead, and
`python/tests/test_issue_1086_cli_startup.py` holds it to that.
"""

import importlib.util
import os
import sys

__all__ = ["main"]


def _load_cli():
    spec = importlib.util.find_spec("pounce")
    if spec is None or not spec.submodule_search_locations:
        sys.stderr.write("pounce: the `pounce` Python package is not installed.\n")
        raise SystemExit(1)
    path = os.path.join(list(spec.submodule_search_locations)[0], "_cli.py")
    cli_spec = importlib.util.spec_from_file_location("_pounce_cli._impl", path)
    module = importlib.util.module_from_spec(cli_spec)
    cli_spec.loader.exec_module(module)
    return module


def main() -> int:
    return _load_cli().main()
