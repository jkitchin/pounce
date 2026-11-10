"""Model doctor: check a model without solving it.

The Python face of ``pounce diagnose`` (see ``docs/src/diagnose.md``). It
runs the CLI binary that ships with this package on a ``.nl`` file and
returns the ``pounce.diagnose/v1`` report::

    report = pounce.diagnose("model.nl")
    if report.has_errors:
        for f in report.errors():
            print(f["code"], f["observation"])

Each finding is a dict with ``code``, ``severity`` (error / warning / info),
``basis`` (proof / structural / numerical / heuristic), ``confidence``,
``observation``, ``why_it_matters``, ``suggested_next_check``, ``affected``
(``variables`` and ``constraints``) and ``evidence``. A ``numerical`` finding
is about the one point named in ``report.point`` and nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["diagnose", "DiagnoseReport"]

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class DiagnoseReport:
    """Result of :func:`diagnose`: the parsed ``pounce.diagnose/v1`` JSON.

    ``str(report)`` is the CLI's text report. :attr:`raw` holds the full JSON
    (``sections`` carries the supporting tables, including the embedded
    ``check_x0`` report).
    """

    verdict: str
    point: str
    findings: List[Dict[str, Any]]
    checks: List[Dict[str, Any]]
    summary: Dict[str, int]
    raw: Dict[str, Any] = field(repr=False)
    text: str = field(default="", repr=False)
    exit_code: int = 0

    @property
    def has_errors(self) -> bool:
        return any(f.get("severity") == "error" for f in self.findings)

    def by_severity(self, severity: str) -> List[Dict[str, Any]]:
        return [f for f in self.findings if f.get("severity") == severity]

    def errors(self) -> List[Dict[str, Any]]:
        return self.by_severity("error")

    def warnings(self) -> List[Dict[str, Any]]:
        return self.by_severity("warning")

    def by_code(self, code: str) -> List[Dict[str, Any]]:
        return [f for f in self.findings if f.get("code") == code]

    def unavailable_checks(self) -> List[Dict[str, Any]]:
        """Checks that did not run, each with its ``reason``."""
        return [c for c in self.checks if c.get("status") == "unavailable"]

    def to_dict(self) -> Dict[str, Any]:
        return self.raw

    def __str__(self) -> str:
        return self.text or json.dumps(self.raw, indent=2)


def _binary() -> str:
    env = os.environ.get("POUNCE_BIN")
    if env:
        return env
    from ._cli import resolve_binary

    path, _origin = resolve_binary()
    if path is None:
        raise FileNotFoundError(
            "no pounce CLI binary found (set POUNCE_BIN, or build it with "
            "`cargo build --release -p pounce-cli`)"
        )
    return str(path)


def diagnose(
    nl_file: PathLike,
    *,
    x0: Optional[Any] = None,
    x0_file: Optional[PathLike] = None,
    sol_file: Optional[PathLike] = None,
    feas_tol: Optional[float] = None,
    rank_tol: Optional[float] = None,
    max_dependent_sets: Optional[int] = None,
    max_per_code: Optional[int] = None,
    extra_points: Optional[Sequence[Any]] = None,
    opt_tol: Optional[float] = None,
    weak_band: Optional[float] = None,
    seed: Optional[int] = None,
    convexity_samples: Optional[int] = None,
    convexity_tol: Optional[float] = None,
    timeout: Optional[float] = 300.0,
) -> DiagnoseReport:
    """Check a ``.nl`` model without solving it.

    Args:
        nl_file: the AMPL ``.nl`` problem.
        x0: a starting point (sequence of ``n`` floats) for the numerical
            checks. At most one of ``x0``, ``x0_file`` and ``sol_file``.
        x0_file: a file of ``n`` whitespace-separated values.
        sol_file: an AMPL ``.sol``; its primal values are the point, and
            its multipliers enable the solution checks (KKT audit, LICQ,
            MFCQ, reduced Hessian).
        feas_tol: slack before a row or bound counts as violated (1e-6).
        rank_tol: singular-value threshold relative to sigma_max
            (default ``max(m, n)*eps``).
        max_dependent_sets: minimal dependent sets to localize (5).
        max_per_code: findings kept per code (25).
        extra_points: further points (each a sequence of ``n`` floats) for
            the rank-robustness comparison.
        opt_tol: stationarity / zero-multiplier tolerance, relative to
            ``max(1, |grad f|)`` (1e-6).
        weak_band: near-active band, relative (1e-4).
        seed: convexity sampler seed (0); the same seed replays the same
            counterexamples.
        convexity_samples: sample pairs per function (64).
        convexity_tol: relative Jensen gap that counts as a counterexample
            (1e-9).
        timeout: seconds before the check is killed.

    Returns:
        A :class:`DiagnoseReport`.

    Raises:
        FileNotFoundError: the ``.nl`` or the binary is missing.
        RuntimeError: the CLI reported a usage or I/O error (exit code 2).
    """
    given = [a is not None for a in (x0, x0_file, sol_file)]
    if sum(given) > 1:
        raise ValueError("pass at most one of x0, x0_file and sol_file")
    nl = Path(nl_file).expanduser()
    if not nl.exists():
        raise FileNotFoundError(f"no such .nl file: {nl}")

    cleanup: List[Path] = []
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="pounce-diagnose-")
    os.close(fd)
    report_path = Path(tmp)
    cleanup.append(report_path)
    argv: List[str] = [_binary(), "diagnose", str(nl), "--json-output", str(report_path)]
    try:
        if x0 is not None:
            fd, xtmp = tempfile.mkstemp(suffix=".txt", prefix="pounce-diagnose-x0-")
            with os.fdopen(fd, "w") as fh:
                fh.write(" ".join(repr(float(v)) for v in x0))
            cleanup.append(Path(xtmp))
            argv += ["--x0-file", xtmp, "--point-label", "the point passed as x0"]
        elif x0_file is not None:
            argv += ["--x0-file", str(Path(x0_file).expanduser())]
        elif sol_file is not None:
            argv += ["--sol", str(Path(sol_file).expanduser())]
        for k, pt in enumerate(extra_points or []):
            fd, ptmp = tempfile.mkstemp(suffix=".txt", prefix=f"pounce-diagnose-p{k}-")
            with os.fdopen(fd, "w") as fh:
                fh.write(" ".join(repr(float(v)) for v in pt))
            cleanup.append(Path(ptmp))
            argv += ["--extra-point", ptmp]
        # Format by kind, not with repr(): under NumPy >= 2,
        # repr(np.float64(1e-6)) is 'np.float64(1e-06)', which the CLI
        # cannot parse.
        for flag, val in (
            ("--feas-tol", feas_tol),
            ("--rank-tol", rank_tol),
            ("--opt-tol", opt_tol),
            ("--weak-band", weak_band),
            ("--convexity-tol", convexity_tol),
        ):
            if val is not None:
                argv += [flag, repr(float(val))]
        for flag, val in (
            ("--max-dependent-sets", max_dependent_sets),
            ("--max-per-code", max_per_code),
            ("--seed", seed),
            ("--convexity-samples", convexity_samples),
        ):
            if val is not None:
                argv += [flag, str(int(val))]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if proc.returncode not in (0, 22):
            raise RuntimeError(
                f"pounce diagnose failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        raw = json.loads(report_path.read_text())
    finally:
        for p in cleanup:
            p.unlink(missing_ok=True)

    text = proc.stdout
    marker = "  report: "
    if marker in text:
        text = text[: text.rfind(marker)]
    return DiagnoseReport(
        verdict=raw.get("verdict", ""),
        point=raw.get("point", ""),
        findings=raw.get("findings", []),
        checks=raw.get("checks", []),
        summary=raw.get("summary", {}),
        raw=raw,
        text=text,
        exit_code=proc.returncode,
    )
