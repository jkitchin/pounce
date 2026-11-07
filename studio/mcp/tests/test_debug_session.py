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


# ---- restoration-inner terminal pauses / close latency ---------------

# Waechter-Biegler (2000): min x1 s.t. x1^2 - x2 - 1 = 0, x1 - x3 - 0.5 = 0,
# x2, x3 >= 0, from (-2, 1, 1). Enters restoration and ends locally
# infeasible, so its run has `terminated` pauses from restoration's inner
# solve (final: false) before the real one.
_WB_NL = (
    "g3 1 1 0\n 3 2 1 0 2\n 1 0\n 0 0\n 1 0 0\n 0 0 0 1 0\n 0 0 0 0 0\n"
    " 4 1\n 0 0\n 0 0 0 0 0\nC0\no5\nv0\nn2.0\nC1\nn0\nO0 0\nn0\nr\n"
    "4 1.0\n4 0.5\nb\n3\n2 0.0\n2 0.0\nx3\n0 -2.0\n1 1.0\n2 1.0\nk2\n2\n3\n"
    "J0 2\n0 0.0\n1 -1.0\nJ1 2\n0 1.0\n2 -1.0\nG0 1\n0 1.0\n"
)


@pytest.fixture
def wb_nl(tmp_path):
    p = tmp_path / "wb.nl"
    p.write_text(_WB_NL)
    return str(p)


def test_restoration_terminal_pause_is_not_the_end(wb_nl):
    """`continue` must not report `finished` at a restoration sub-solve's
    terminal checkpoint: that read the inner solve's status as the result."""
    st = debug_start(nl_file=wb_nl)
    sid = st["session_id"]
    try:
        stops = []
        for _ in range(20):
            out = debug_command(sid, "continue", timeout_seconds=60)
            stops.append((out["outcome"], (out.get("state") or {}).get("in_restoration")))
            if out["outcome"] in ("finished", "terminated"):
                break
        assert stops[-1][0] == "finished", stops
        # every earlier stop was an inner restoration termination
        assert all(o == "paused" and resto for o, resto in stops[:-1]), stops
        assert len(stops) >= 2, "fixture no longer reaches restoration"
        final = debug_state(sid)["state"]
        assert final["final"] is True and final["in_restoration"] is False
        assert final["status"] == "LocalInfeasibility"
    finally:
        debug_close(sid)


def test_close_after_finish_exits_promptly(wb_nl):
    """Closing a session parked at the real end used to wait out the 5 s
    grace and SIGKILL the child (`quit` was ignored and follow-on solves
    re-parked on stdin)."""
    import time

    st = debug_start(nl_file=wb_nl)
    sid = st["session_id"]
    for _ in range(20):
        if debug_command(sid, "continue", timeout_seconds=60)["outcome"] != "paused":
            break
    t0 = time.monotonic()
    out = debug_close(sid)
    assert time.monotonic() - t0 < 3.0
    assert out["exit_code"] is not None and out["exit_code"] >= 0, out
