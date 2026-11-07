"""The studio tests need the server's `mcp` dependency (the `[mcp]` extra);
skip the directory cleanly in an environment that does not have it."""
import pytest

pytest.importorskip("mcp.server.fastmcp")
