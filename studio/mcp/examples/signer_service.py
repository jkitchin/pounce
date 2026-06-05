#!/usr/bin/env python3
"""Reference out-of-process signer for `pounce verify` receipts.

This is the trust boundary that makes a *signed* receipt mean something: it
holds the HMAC key in **its own** environment and never hands it to the
caller. An agent calls it to verify a solution; the agent's environment
never contains the key, so the agent cannot mint its own valid receipt — it
can only ask this service to verify a genuinely-feasible `x*`.

It is a **reference / teaching** implementation, intentionally tiny and
dependency-free (stdlib `http.server`). It is NOT hardened for production.
For a real deployment you still need, at minimum:

  * to run this as a SEPARATE user / container / host from the agent (if the
    agent has a shell on the same user/host it can read this process's
    environment and the whole scheme collapses — see docs/src/verify.md);
  * a real key source (a file readable only by this service's user, or a
    KMS/HSM that signs without exposing the key) rather than a plain env var;
  * authn/authz on the endpoint, TLS, request size limits, and rate limits;
  * a strict path policy — this sketch only allows files under a configured
    root (POUNCE_SIGNER_ROOT) to blunt path-traversal, nothing more.

Remember: the signature is only as strong as this boundary. When the
consumer can simply run `pounce verify` itself, prefer that — it needs no
key at all (see "The default: recompute" in docs/src/verify.md).

--------------------------------------------------------------------------
Run:
    POUNCE_VERIFY_KEY="$(cat /run/secrets/pounce_verify_key)" \
    POUNCE_SIGNER_ROOT=/srv/problems \
    python3 signer_service.py            # listens on 127.0.0.1:8723

Request (the agent, with NO key in its env):
    curl -s localhost:8723/verify -XPOST -H 'content-type: application/json' \
      -d '{"nl_path":"/srv/problems/p.nl","sol_path":"/srv/problems/p.sol"}'
    # → the signed receipt JSON (verdict + signature)

Verify the returned receipt (consumer holding the same key):
    from pounce_studio_mcp.verify_sig import check_signature
    assert check_signature(receipt, key) and receipt["verified"]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- configuration (all from THIS process's environment, never the request) ---

KEY = os.environ.get("POUNCE_VERIFY_KEY", "")
# Only files under this root may be verified (path-traversal guard). Set to
# "" to disable the check (NOT recommended).
ROOT = os.environ.get("POUNCE_SIGNER_ROOT", "")
BIND = os.environ.get("POUNCE_SIGNER_BIND", "127.0.0.1")
PORT = int(os.environ.get("POUNCE_SIGNER_PORT", "8723"))
TIMEOUT = float(os.environ.get("POUNCE_SIGNER_TIMEOUT", "120"))


def _find_pounce_bin() -> str:
    env = os.environ.get("POUNCE_BIN")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "target" / "release" / "pounce"
        if cand.exists():
            return str(cand)
    which = shutil.which("pounce")
    if which:
        return which
    raise FileNotFoundError("could not locate the pounce binary; set POUNCE_BIN")


def _resolve_allowed(p: str) -> Path:
    """Resolve `p` and confirm it lives under ROOT (when ROOT is set)."""
    path = Path(p).expanduser().resolve()
    if ROOT:
        root = Path(ROOT).expanduser().resolve()
        if root not in path.parents and path != root:
            raise PermissionError(f"path not under POUNCE_SIGNER_ROOT: {path}")
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    return path


def sign_verify(nl_path: str, sol_path: str, *, feas_tol: float = 1e-6,
                opt_tol: float = 1e-6, require_optimal: bool = False) -> dict:
    """Run `pounce verify` with the key in THIS env; return the signed receipt."""
    nl = _resolve_allowed(nl_path)
    sol = _resolve_allowed(sol_path)
    binary = _find_pounce_bin()

    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="signer-")
    os.close(fd)
    receipt_path = Path(tmp)
    try:
        argv = [
            binary, "verify", str(nl), str(sol),
            "--feas-tol", repr(feas_tol),
            "--opt-tol", repr(opt_tol),
            "--json-output", str(receipt_path),
        ]
        if require_optimal:
            argv.append("--require-optimal")
        # The KEY reaches the binary ONLY through this child's environment.
        # It is never returned, logged, or echoed.
        child_env = dict(os.environ)
        if KEY:
            child_env["POUNCE_VERIFY_KEY"] = KEY
        subprocess.run(argv, env=child_env, capture_output=True,
                       text=True, timeout=TIMEOUT)
        return json.loads(receipt_path.read_text())
    finally:
        receipt_path.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path != "/verify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            if n <= 0 or n > 1 << 16:
                raise ValueError("missing or oversized body")
            req = json.loads(self.rfile.read(n))
            receipt = sign_verify(
                req["nl_path"], req["sol_path"],
                feas_tol=float(req.get("feas_tol", 1e-6)),
                opt_tol=float(req.get("opt_tol", 1e-6)),
                require_optimal=bool(req.get("require_optimal", False)),
            )
            self._send(200, receipt)
        except (KeyError, ValueError, PermissionError, FileNotFoundError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — reference service: report, don't crash
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *_args) -> None:  # keep the key-adjacent path quiet
        pass


def main() -> None:
    if not KEY:
        print("WARNING: POUNCE_VERIFY_KEY is empty — receipts will be UNSIGNED. "
              "This service then adds nothing over the consumer running "
              "`pounce verify` directly.")
    srv = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"pounce signer listening on http://{BIND}:{PORT}/verify "
          f"(root={ROOT or 'ANY (unsafe)'}, signed={bool(KEY)})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
