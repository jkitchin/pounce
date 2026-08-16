"""Model fingerprinting and replay-compatibility rules for ``WarmStart``.

A warm start is a *point in a specific model's variable space*, plus
multipliers in that model's constraint space. Replay it against a model
whose variables have been reordered, whose bounds have moved, or which
is simply a different model with the same dimensions, and the arrays are
still the right *shape* — so nothing downstream objects. What comes back
is a wrong answer, or the right answer down a much longer trajectory,
and in neither case does anything say so (pounce#607; the same class of
silence as gh#544).

This module carries the metadata that makes that detectable:

* :class:`ProblemSignature` — dimensions, variable/constraint ordering
  or stable IDs, sparsity signature, bound signature, scaling
  convention, algorithm/backend, and the model-defining option
  fingerprint, captured from a live :class:`pounce.Problem`.
* :func:`compare` — a facet-by-facet mismatch report.
* :class:`WarmStartCompatibilityError` /
  :class:`WarmStartCompatibilityWarning` — the strict / warn outcomes.

It is deliberately separate from ``_warm_start.py``: the fingerprinting
rules change for reasons that have nothing to do with the warm-start
object's own fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WARM_START_SCHEMA_VERSION",
    "ProblemSignature",
    "Mismatch",
    "WarmStartCompatibilityError",
    "WarmStartCompatibilityWarning",
    "WarmStartLegacyWarning",
    "compare",
    "COMPAT_MODES",
    "MODEL_OPTIONS",
]

#: Version of the on-disk ``WarmStart`` artifact schema.
#:
#: ``1`` is the implicit version of every archive written before
#: pounce#607: ``x`` / ``lagrange`` / ``zl`` / ``zu`` / ``ws_*`` arrays
#: and a 4-wide numeric ``_meta`` row, with no model metadata at all.
#: ``2`` adds ``_schema`` and the JSON ``_signature`` / ``_provenance``
#: blobs. Version 2 archives stay readable by a version-1 loader
#: (``np.load`` ignores keys it is not asked for), so the format change
#: is backward *and* forward compatible at the array level; what a
#: version-1 loader cannot do is check anything.
WARM_START_SCHEMA_VERSION = 2

#: The three compatibility modes, in decreasing order of safety.
#:
#: ``strict``  — any mismatch, and any facet that cannot be verified on a
#:               signed artifact, raises before the solver is entered.
#: ``warn``    — the same report is emitted as a warning and the solve
#:               proceeds.
#: ``unsafe``  — no checking at all. The escape hatch for a caller who
#:               knows the artifact transfers and does not want the cost
#:               or the noise.
COMPAT_MODES = ("strict", "warn", "unsafe")

#: Options that change *the model being solved* or its representation,
#: as opposed to how hard the solver works at it. Only these enter the
#: fingerprint: a warm start stays valid across a changed ``max_iter``
#: or ``print_level``, and must not across a changed
#: ``fixed_variable_treatment``.
#:
#: ``nlp_scaling_method`` / ``obj_scaling_factor`` /
#: ``nlp_scaling_max_gradient`` are fingerprinted separately, as the
#: "scaling convention" facet, because they decide what units the
#: captured multipliers are in.
MODEL_OPTIONS = (
    "bound_relax_factor",
    "fixed_variable_treatment",
    "hessian_approximation",
    "hessian_constant",
    "honor_original_bounds",
    "jac_c_constant",
    "jac_d_constant",
)

_SCALING_OPTIONS = (
    "nlp_scaling_method",
    "nlp_scaling_max_gradient",
    "obj_scaling_factor",
)

_ALGORITHM_OPTIONS = ("algorithm", "linear_solver", "sqp_hessian")


class WarmStartCompatibilityError(ValueError):
    """A warm start does not match the problem it is being replayed on.

    Raised before the solver is entered, under ``compat="strict"``.
    """


class WarmStartCompatibilityWarning(UserWarning):
    """``compat="warn"`` counterpart of
    :class:`WarmStartCompatibilityError`."""


class WarmStartLegacyWarning(UserWarning):
    """A schema-version-1 archive was replayed.

    Version-1 archives carry no model metadata, so only the facets that
    can be recovered from the arrays themselves (the dimensions) are
    checked. See :meth:`pounce.WarmStart.migrate`.
    """


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------


def _digest(*parts: Any) -> str:
    """A short, stable, platform-independent digest of `parts`.

    Floats go in as their exact IEEE-754 bytes (``repr`` round-trips but
    is locale-free only by luck), integers as int64, everything else via
    a canonical JSON encoding. Truncated to 16 hex chars — long enough
    that an accidental collision between two models is not a thing that
    happens, short enough to print in a mismatch report.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(str(a.dtype.str).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _sorted_pairs(pairs: Sequence[Tuple[str, Any]], keys: Sequence[str]) -> list:
    """`pairs` filtered to `keys`, last-write-wins, sorted by name.

    ``Problem`` records options as an append-only list, and applies them
    in order, so the *effective* value of a name is its last occurrence.
    """
    eff: Dict[str, Any] = {}
    for k, v in pairs:
        if k in keys:
            eff[k] = v
    return sorted(eff.items())


def _effective_options(problem) -> List[Tuple[str, Any]]:
    """The flat option list of `problem`, in the order the solver
    applies it (strings, then numbers, then integers — see
    ``PyProblem::prepare``)."""
    try:
        strs, nums, ints = problem.options_snapshot()
    except AttributeError:  # pragma: no cover - pre-#607 extension
        return []
    return [(k, v) for k, v in strs] + [(k, v) for k, v in nums] + [
        (k, v) for k, v in ints
    ]


def _structure_digest(problem) -> Optional[str]:
    """Digest of the model's *declared* sparsity.

    ``None`` when the model does not declare one (a dense-jacobian
    ``problem_obj``, or an object this build cannot reach), which makes
    the facet unverifiable rather than silently equal.
    """
    try:
        obj = problem.problem_obj
    except AttributeError:  # pragma: no cover - pre-#607 extension
        return None
    parts: List[Any] = []
    for name in ("jacobianstructure", "hessianstructure"):
        fn = getattr(obj, name, None)
        if fn is None:
            parts.append(None)
            continue
        try:
            rows, cols = fn()
        except Exception:  # noqa: BLE001 - a structure query must never
            # take down the fingerprint; an unreadable facet is reported
            # as unverifiable, which is what it is.
            return None
        parts.append(name)
        parts.append(np.asarray(rows, dtype=np.int64).ravel())
        parts.append(np.asarray(cols, dtype=np.int64).ravel())
    if all(p is None for p in parts):
        return None
    return _digest(*parts)


def _scaling_digest(problem, opts: Sequence[Tuple[str, Any]]) -> str:
    """Digest of the scaling convention: the scaling options plus any
    user scaling vectors installed via ``set_problem_scaling``.

    This is a facet in its own right because it decides what units the
    captured multipliers are expressed in — a warm start captured under
    ``gradient-based`` scaling is not a warm start for the same model
    under ``none``.
    """
    parts: List[Any] = [_sorted_pairs(opts, _SCALING_OPTIONS)]
    user = None
    try:
        user = problem.get_problem_scaling()
    except AttributeError:  # pragma: no cover - pre-#607 extension
        pass
    if user is None:
        parts.append(None)
    else:
        obj_s, x_s, g_s = user
        parts.append(float(obj_s))
        parts.append(None if x_s is None else np.asarray(x_s, dtype=float))
        parts.append(None if g_s is None else np.asarray(g_s, dtype=float))
    return _digest(*parts)


def _ids_tuple(ids, n: int, what: str) -> Optional[Tuple[str, ...]]:
    if ids is None:
        return None
    out = tuple(str(v) for v in ids)
    if len(out) != n:
        raise ValueError(
            f"{what}: expected {n} identifiers to match the problem's "
            f"dimension, got {len(out)}"
        )
    if len(set(out)) != len(out):
        raise ValueError(f"{what}: identifiers must be unique")
    return out


# ---------------------------------------------------------------------------
# the signature
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProblemSignature:
    """What a :class:`pounce.WarmStart` has to match to be replayable.

    Every field is either a value or ``None``. ``None`` means *not
    recorded* — an unverifiable facet, which is reported as a mismatch
    under ``compat="strict"`` rather than quietly passing.

    Attributes:
        n / m: Dimensions.
        var_ids / con_ids: Caller-supplied stable identifiers for the
            variables and constraints, when the caller has them. These
            are what make a *reordered* model recoverable rather than
            merely detectable — see :meth:`pounce.WarmStart.reindex`.
        bounds: Digest of ``(lb, ub, cl, cu)``.
        sparsity: Digest of the declared jacobian / Hessian structure.
        scaling: Digest of the scaling convention.
        algorithm: Digest of the algorithm / backend selection.
        model: Digest of the model-defining options (:data:`MODEL_OPTIONS`).
    """

    n: int
    m: int
    var_ids: Optional[Tuple[str, ...]] = None
    con_ids: Optional[Tuple[str, ...]] = None
    bounds: Optional[str] = None
    sparsity: Optional[str] = None
    scaling: Optional[str] = None
    algorithm: Optional[str] = None
    model: Optional[str] = None

    #: Facets compared by :func:`compare`, in report order. Dimensions
    #: come first: they are the one facet recoverable from a legacy
    #: artifact, and the one whose mismatch the native layer would
    #: otherwise report from deep inside solve preparation.
    FACETS = ("n", "m", "var_ids", "con_ids", "bounds", "sparsity",
              "scaling", "algorithm", "model")

    #: Facets compared only when *both* sides recorded them. Stable IDs
    #: are the transfer key, not a verification requirement: a live
    #: ``Problem`` has no idea what its variables are called, so a target
    #: signature almost never carries them, and treating that absence as
    #: unverifiable would make signing an artifact with IDs strictly
    #: worse than signing it without.
    OPTIONAL_FACETS = ("var_ids", "con_ids")

    # -- construction ---------------------------------------------------

    @classmethod
    def from_problem(cls, problem, var_ids=None, con_ids=None) -> "ProblemSignature":
        """Fingerprint a live :class:`pounce.Problem`.

        `var_ids` / `con_ids` are optional stable identifiers (any
        sequence of `n` / `m` unique values; they are stringified).
        Supplying them is what lets :meth:`pounce.WarmStart.reindex`
        transfer a warm start across a reordering or a horizon shift
        instead of only refusing it.
        """
        n, m = int(problem.n), int(problem.m)
        opts = _effective_options(problem)
        try:
            lb, ub, cl, cu = problem.get_bounds()
            bounds = _digest(
                np.asarray(lb, dtype=float), np.asarray(ub, dtype=float),
                np.asarray(cl, dtype=float), np.asarray(cu, dtype=float),
            )
        except AttributeError:  # pragma: no cover - pre-#607 extension
            bounds = None
        return cls(
            n=n,
            m=m,
            var_ids=_ids_tuple(var_ids, n, "var_ids"),
            con_ids=_ids_tuple(con_ids, m, "con_ids"),
            bounds=bounds,
            sparsity=_structure_digest(problem),
            scaling=_scaling_digest(problem, opts),
            algorithm=_digest(_sorted_pairs(opts, _ALGORITHM_OPTIONS)),
            model=_digest(_sorted_pairs(opts, MODEL_OPTIONS)),
        )

    # -- persistence ----------------------------------------------------

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        for k in ("var_ids", "con_ids"):
            if d[k] is not None:
                d[k] = list(d[k])
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProblemSignature":
        d = json.loads(text)
        for k in ("var_ids", "con_ids"):
            if d.get(k) is not None:
                d[k] = tuple(str(v) for v in d[k])
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            # A newer pounce wrote facets this build does not know about.
            # Dropping them silently would turn a stricter artifact into a
            # weaker one, so say so; the caller still gets a usable object.
            raise ValueError(
                "warm-start signature carries facets this build does not "
                f"understand ({', '.join(unknown)}); upgrade pounce, or "
                "re-capture the warm start"
            )
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass(frozen=True)
class Mismatch:
    """One facet on which an artifact and a target problem disagree."""

    facet: str
    captured: Any
    target: Any
    #: True when the disagreement is "one side did not record this",
    #: rather than two recorded values differing.
    unverifiable: bool = False

    def __str__(self) -> str:
        if self.facet in ("var_ids", "con_ids") and not self.unverifiable:
            return (
                f"{self.facet}: identifiers differ "
                f"(captured {_preview(self.captured)}, "
                f"target {_preview(self.target)})"
            )
        if self.unverifiable:
            side = "the artifact" if self.captured is None else "this problem"
            return (
                f"{self.facet}: not recorded by {side}, so it cannot be "
                "verified"
            )
        return f"{self.facet}: captured {self.captured!r}, target {self.target!r}"


def _preview(ids) -> str:
    if ids is None:
        return "none"
    ids = list(ids)
    head = ", ".join(ids[:4])
    return f"[{head}{', …' if len(ids) > 4 else ''}] ({len(ids)})"


def compare(captured: ProblemSignature, target: ProblemSignature) -> List[Mismatch]:
    """Facet-by-facet comparison, most structural facet first.

    A facet neither side recorded is skipped — there is nothing to
    disagree about. A facet exactly one side recorded is reported as
    *unverifiable*, which strict mode treats as a mismatch: a signed
    artifact whose bound signature cannot be checked against the target
    is exactly as unsafe as one whose bound signature differs. The
    exception is :data:`ProblemSignature.OPTIONAL_FACETS`, compared only
    when both sides have them.
    """
    out: List[Mismatch] = []
    for facet in ProblemSignature.FACETS:
        a = getattr(captured, facet)
        b = getattr(target, facet)
        if a is None and b is None:
            continue
        if a is None or b is None:
            if facet in ProblemSignature.OPTIONAL_FACETS:
                continue
            out.append(Mismatch(facet, a, b, unverifiable=True))
        elif a != b:
            out.append(Mismatch(facet, a, b))
    return out


def format_report(
    mismatches: Sequence[Mismatch],
    *,
    replay: str,
    schema_version: Optional[int],
    source: Optional[ProblemSignature] = None,
) -> str:
    """The human-facing mismatch report.

    Names every facet that disagrees, says which is which, and — this is
    the part that makes it worth printing — names the two ways forward
    (re-capture, or transfer) rather than only the way that is blocked.
    """
    kind = "mapped" if replay == "mapped" else "exact-structure"
    lines = [
        f"warm start is not compatible with this problem "
        f"({len(mismatches)} mismatch"
        f"{'es' if len(mismatches) != 1 else ''}, {kind} replay"
        + (f", schema v{schema_version}" if schema_version else "")
        + "):",
    ]
    lines += [f"  - {m}" for m in mismatches]
    if source is not None:
        lines.append(
            f"  (this artifact was transferred from a {source.n}x{source.m} "
            "problem; a mapped warm start is valid only for the problem it "
            "was mapped to)"
        )
    lines += [
        "resolve it by one of:",
        "  - re-capture against this problem: "
        "WarmStart.from_info(x, info, problem=prob)",
        "  - transfer it explicitly: ws.transfer(prob, mapper) or, with "
        "stable IDs on both sides, ws.reindex(prob)",
        "  - assert it transfers as-is: ws.migrate(prob) (re-signs the "
        "artifact against this problem)",
        "  - downgrade the check: compat='warn' or compat='unsafe'",
    ]
    return "\n".join(lines)
