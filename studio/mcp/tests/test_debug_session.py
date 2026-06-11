"""Tests for the live debug-session proxy tools (debug_start / debug_command
/ debug_state / debug_sessions / debug_close).

These spawn a real `pounce --debug-json` child, so they skip when the
binary isn't built. They exercise the protocol proxy against the
`rosenbrock` builtin, which converges in ~21 iterations.
"""
from __future__ import annotations

import pytest

from pounce_studio_mcp.server import (
    debug_close,
    debug_command,
    debug_sessions,
    debug_start,
    debug_state,
)


def _pounce_available() -> bool:
    from pounce_studio_mcp.server import _find_pounce_bin
    try:
        _find_pounce_bin()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _pounce_available(), reason="pounce binary not built"
)


@pytest.fixture
def session():
    """A started rosenbrock session, always closed at teardown."""
    st = debug_start(builtin="rosenbrock")
    sid = st["session_id"]
    try:
        yield st
    finally:
        try:
            debug_close(sid)
        except ValueError:
            pass  # already closed by the test


# ---- start / handshake ----------------------------------------------


def test_start_returns_self_describing_hello(session):
    hello = session["hello"]
    assert hello["protocol"] == "pounce-dbg/1"
    # the lists an agent feature-detects off
    assert "continue" in hello["commands"]
    assert "iter_start" in hello["checkpoints"]
    assert hello["metrics"][:1] == ["iter"]
    assert hello["capabilities"]["mutate_iterate"] is True
    # parked at iteration 0
    assert session["pause"]["iter"] == 0
    assert session["pause"]["checkpoint"] == "iter_start"


def test_start_requires_exactly_one_input():
    with pytest.raises(ValueError):
        debug_start()
    with pytest.raises(ValueError):
        debug_start(builtin="rosenbrock", nl_file="x.nl")


def test_start_missing_nl_file_rejected():
    with pytest.raises(FileNotFoundError):
        debug_start(nl_file="/no/such/model.nl")


def test_start_with_setup_commands():
    st = debug_start(builtin="rosenbrock", setup=["stop-at kkt"])
    try:
        assert st["setup_results"][0]["ok"] is True
    finally:
        debug_close(st["session_id"])


# ---- stepping / inspection ------------------------------------------


def test_nonflow_command_parks(session):
    sid = session["session_id"]
    out = debug_command(sid, "print", ["x"])
    assert out["outcome"] == "parked"
    assert out["ok"] is True
    assert out["result"]["data"]["values"] == pytest.approx([-1.2, 1.0])


def test_step_advances_one_iteration(session):
    sid = session["session_id"]
    out = debug_command(sid, "step")
    assert out["outcome"] == "paused"
    assert out["state"]["iter"] == 1


def test_goto_emits_only_result(session):
    sid = session["session_id"]
    debug_command(sid, "step")
    debug_command(sid, "step")
    out = debug_command(sid, "goto", ["0"])
    # goto parks (no new pause event) and reports the restored iter
    assert out["outcome"] == "parked"
    assert out["result"]["data"]["restored_iter"] == 0


# ---- termination model ----------------------------------------------


def test_continue_parks_at_terminal_checkpoint(session):
    sid = session["session_id"]
    out = debug_command(sid, "continue")
    assert out["outcome"] == "finished"
    assert out["finished"] is True
    assert out["state"]["status"] == "Success"
    assert out["progress"]["count"] >= 1


def test_post_mortem_inspection_at_terminal_checkpoint(session):
    sid = session["session_id"]
    debug_command(sid, "continue")  # → finished, parked at terminal cp
    # the final iterate stays inspectable
    pm = debug_command(sid, "print", ["x"])
    assert pm["outcome"] == "parked"
    assert pm["result"]["data"]["values"] == pytest.approx([1.0, 1.0], abs=1e-6)
    diag = debug_command(sid, "diagnose")
    assert diag["ok"] is True


def test_release_past_terminal_yields_terminated_event(session):
    sid = session["session_id"]
    debug_command(sid, "continue")          # finished
    out = debug_command(sid, "continue")    # release → terminated summary
    assert out["outcome"] == "terminated"
    assert out["terminated"]["status"] == "SolveSucceeded"
    assert out["terminated"]["iterations"] >= 1
    assert "obj" in out["terminated"]["evals"]
    # further commands are refused cleanly
    with pytest.raises(RuntimeError):
        debug_command(sid, "info")


# ---- breakpoints -----------------------------------------------------


def test_conditional_breakpoint_fires(session):
    sid = session["session_id"]
    r = debug_command(sid, "break if inf_du<1e-6")
    assert r["ok"] is True
    out = debug_command(sid, "continue")
    # stops at the breakpoint, not the terminal checkpoint
    assert out["outcome"] == "paused"
    assert out["state"]["reason"] == "inf_du<1e-6"


# ---- timeout / interrupt recovery -----------------------------------


def test_timeout_interrupts_and_session_stays_usable(session):
    sid = session["session_id"]
    out = debug_command(sid, "continue", timeout_seconds=0.0001)
    assert out["outcome"] == "interrupted"
    assert out["finished"] is False
    # the session recovers and runs to completion afterward
    out2 = debug_command(sid, "continue", timeout_seconds=60)
    assert out2["outcome"] == "finished"
    assert out2["state"]["status"] == "Success"


# ---- registry / lifecycle -------------------------------------------


def test_state_and_sessions_listing(session):
    sid = session["session_id"]
    debug_command(sid, "step")
    state = debug_state(sid)
    assert state["alive"] is True
    assert state["state"]["iter"] == 1
    listing = debug_sessions()
    assert any(s["session_id"] == sid for s in listing["sessions"])
    assert listing["max_sessions"] >= 1


def test_unknown_session_rejected():
    with pytest.raises(ValueError):
        debug_command("deadbeefdead", "info")
    with pytest.raises(ValueError):
        debug_state("deadbeefdead")
    with pytest.raises(ValueError):
        debug_close("deadbeefdead")


def test_close_reports_final_status(session):
    sid = session["session_id"]
    debug_command(sid, "continue")
    cl = debug_close(sid)
    assert cl["final_status"] in ("Success", "SolveSucceeded")
    # idempotent: second close is a clean error
    with pytest.raises(ValueError):
        debug_close(sid)
