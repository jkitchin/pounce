"""The .mcpb manifest's tool list must match what the server registers.

Claude Desktop shows the manifest's ``tools`` before the server ever runs, so
a tool added to ``pounce.studio.server`` without a manifest entry is invisible
on the install page, and a removed one is advertised but missing.
"""
import asyncio
import json
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[3] / "studio" / "mcpb" / "manifest.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="needs a pounce source checkout")
def test_manifest_tools_match_server():
    from pounce.studio.server import mcp

    served = {t.name for t in asyncio.run(mcp.list_tools())}
    listed = {t["name"] for t in json.loads(MANIFEST.read_text())["tools"]}
    assert served - listed == set(), "tools missing from studio/mcpb/manifest.json"
    assert listed - served == set(), "manifest lists tools the server lacks"
