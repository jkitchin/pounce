"""Receipt-signature helpers for `pounce verify` (dependency-free).

These reproduce `pounce_cli::verify::signing_preimage` and validate the
HMAC-SHA256 signature on a verification receipt. Kept in their own module
(no `mcp`, no `_native` import) so a consumer can check a receipt without
the full MCP runtime, and so they are unit-testable on their own.

See `docs/src/verify.md` for the canonical preimage format and threat model.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any


def signing_preimage(receipt: dict[str, Any]) -> bytes:
    """Reconstruct the float-free HMAC preimage from a receipt's fields.

    Must match `pounce_cli::verify::signing_preimage` byte-for-byte: eight
    newline-joined `key=value` lines with a trailing newline, booleans
    lowercase, no floats.
    """
    prob = receipt.get("problem", {})
    feas = receipt.get("feasibility", {})

    def b(x: object) -> str:
        return "true" if bool(x) else "false"

    text = (
        "pounce-verify-receipt/v1\n"
        "verify_version=1\n"
        f"nl_sha256={prob.get('sha256')}\n"
        f"sol_sha256={receipt.get('solution', {}).get('sha256')}\n"
        f"n_vars={prob.get('n_vars')}\n"
        f"n_cons={prob.get('n_cons')}\n"
        f"feasible={b(feas.get('feasible'))}\n"
        f"verified={b(receipt.get('verified'))}\n"
        f"verdict={receipt.get('verdict')}\n"
    )
    return text.encode("utf-8")


def check_signature(receipt: dict[str, Any], key: str) -> bool:
    """Recompute HMAC-SHA256 over the preimage and compare to `signature`.

    Returns False if the receipt has no string `signature` field.
    """
    sig = receipt.get("signature")
    if not isinstance(sig, str):
        return False
    expect = hmac.new(
        key.encode("utf-8"), signing_preimage(receipt), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(sig, expect)
