"""pounce-studio: an MCP server exposing POUNCE to agents (Claude Desktop,
Claude Code, any MCP client).

Install with the ``mcp`` extra and run the console script::

    pip install "pounce-solver[mcp]"
    pounce-studio-mcp            # stdio MCP server

The tools: solve a model (``run_problem``), analyse a solve report
(``diagnose``, ``convergence_trace``, ...), check a solution against the
canonical model (``verify_solution``), preflight a start point
(``check_x0``), and drive a live, steppable solve (``debug_start`` /
``debug_command`` / ``debug_close``). The report analysis is the Rust
``pounce-studio-core`` crate, mounted in this package's extension as
``pounce._pounce.studio``; the server itself is ``pounce.studio.server``.

This package does not import ``mcp`` (or the server) until ``main`` runs,
so ``import pounce`` stays free of it.
"""
from __future__ import annotations

import os
import sys


def _enter_workdir() -> None:
    """Honour ``POUNCE_WORKDIR``: resolve relative model paths against it.

    Claude Desktop starts MCP servers with an arbitrary working directory
    (``/`` on macOS), so a bare ``cstr.nl`` from the user would otherwise
    resolve nowhere. The ``.mcpb`` bundle sets this from its "workspace"
    setting; a manual config can set it in ``env``.
    """
    wd = os.environ.get("POUNCE_WORKDIR", "").strip()
    if not wd:
        return
    path = os.path.expanduser(wd)
    try:
        os.chdir(path)
    except OSError as e:
        sys.stderr.write(f"pounce-studio-mcp: cannot enter POUNCE_WORKDIR={wd!r}: {e}\n")


def main() -> None:
    """Console-script entry point: run the stdio MCP server."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "pounce-studio-mcp needs the `mcp` package, which is the "
            "`[mcp]` extra of pounce-solver (Python >= 3.10):\n"
            '    pip install "pounce-solver[mcp]"\n'
            "or, without installing anything:\n"
            '    uvx --from "pounce-solver[mcp]" pounce-studio-mcp\n'
        )
        raise SystemExit(1)
    from .server import main as _run

    _run()
