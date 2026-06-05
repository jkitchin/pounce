"""Tests for the `pounce verify` receipt-signature helpers.

`verify_sig` is dependency-free (no `mcp`, no `_native`), so these import
and run without the compiled extension. The end-to-end test additionally
shells out to the `pounce` binary to confirm the Rust signer and the Python
checker agree byte-for-byte; it skips when the binary isn't built.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from pounce_studio_mcp.verify_sig import check_signature, signing_preimage


def _sample_receipt() -> dict:
    return {
        "problem": {"sha256": "aa" * 32, "n_vars": 5, "n_cons": 4},
        "solution": {"sha256": "bb" * 32},
        "feasibility": {"feasible": True},
        "verified": True,
        "verdict": "VERIFIED",
    }


def test_preimage_is_exact_documented_format():
    pre = signing_preimage(_sample_receipt())
    expected = (
        "pounce-verify-receipt/v1\n"
        "verify_version=1\n"
        f"nl_sha256={'aa' * 32}\n"
        f"sol_sha256={'bb' * 32}\n"
        "n_vars=5\n"
        "n_cons=4\n"
        "feasible=true\n"
        "verified=true\n"
        "verdict=VERIFIED\n"
    ).encode("utf-8")
    assert pre == expected


def test_signature_roundtrip_and_tamper_resistance():
    receipt = _sample_receipt()
    key = "secret"
    sig = hmac.new(key.encode(), signing_preimage(receipt), hashlib.sha256).hexdigest()
    receipt["signature"] = sig

    assert check_signature(receipt, key) is True
    assert check_signature(receipt, "wrong-key") is False

    # Flipping any signed field invalidates the signature.
    tampered = dict(receipt)
    tampered["verified"] = False
    assert check_signature(tampered, key) is False


def test_missing_signature_returns_false():
    assert check_signature(_sample_receipt(), "secret") is False


# ---- end-to-end: the Rust signer and Python checker must agree --------


def _pounce_bin() -> str | None:
    env = os.environ.get("POUNCE_BIN")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    for parent in here.parents:
        for profile in ("release", "debug"):
            cand = parent / "target" / profile / "pounce"
            if cand.exists():
                return str(cand)
        if (parent / ".git").exists():
            break
    return None


def _fixture_nl() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "crates" / "pounce-cli" / "tests" / "fixtures" / "parametric.nl"
        if cand.exists():
            return cand
        if (parent / ".git").exists():
            break
    return None


@pytest.mark.skipif(_pounce_bin() is None, reason="pounce binary not built")
@pytest.mark.skipif(_fixture_nl() is None, reason="parametric.nl fixture not found")
def test_rust_signed_receipt_validates_in_python():
    binary = _pounce_bin()
    nl = _fixture_nl()
    key = "cross-language-secret"

    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / "x.sol"
        receipt = Path(d) / "receipt.json"
        # genuine solve
        subprocess.run([binary, str(nl), str(sol)], check=True, capture_output=True)
        # signed verification
        env = dict(os.environ, POUNCE_VERIFY_KEY=key)
        proc = subprocess.run(
            [binary, "verify", str(nl), str(sol), "--json-output", str(receipt)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        r = json.loads(receipt.read_text())

    assert r["verified"] is True
    assert r["signature_alg"] == "HMAC-SHA256"
    # The Rust-produced signature validates under the real key only.
    assert check_signature(r, key) is True
    assert check_signature(r, "agent-does-not-have-this") is False
