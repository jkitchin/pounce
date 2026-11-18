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
  convention, algorithm/backend, the model-defining option
  fingerprint, and an order-sensitive *model probe*, captured from a
  live :class:`pounce.Problem`.
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
    "PROBE_RTOL",
    "ORDERING_UNVERIFIED_NOTE",
    "WarmStartOrderingUnverifiedWarning",
    "ordering_is_unverified",
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
    "nlp_scaling_untagged_constr",
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


class WarmStartOrderingUnverifiedWarning(UserWarning):
    """A warm start was accepted by a comparison that could not have
    seen a reordering of the variables.

    Not a mismatch — nothing disagreed. It says that the one structural
    change the comparison is blind to (pounce#621) was not ruled out,
    which a bare clean verdict would otherwise leave the caller to infer.
    Raised to an error with
    ``warnings.simplefilter("error", WarmStartOrderingUnverifiedWarning)``
    by callers who would rather refuse than replay unverified.
    """


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
        # Which rows were given a factor decides which ones
        # `nlp_scaling_untagged_constr` rescales. Appended only when it was
        # installed, so a digest taken without it is unchanged.
        given = getattr(problem, "get_problem_scaling_given", lambda: None)()
        if given is not None:
            parts.append(np.asarray(given, dtype=bool))
    return _digest(*parts)


# ---------------------------------------------------------------------------
# the model probe (pounce#621)
# ---------------------------------------------------------------------------
#
# Every digest above is computed from what the model *declares*. None of
# them can see a pure permutation of the variables: permute a model with
# a uniform box and a dense jacobian and the bound digest and the
# sparsity digest come out bit-identical, so a reordered replay is
# allowed through unless the caller supplied `var_ids` on both sides
# (pounce#621, split out of #607).
#
# The probe closes that by fingerprinting what the model *computes*
# rather than what it declares: evaluate it once at a fixed point and
# record an order-sensitive summary of the result. A permutation moves
# the numbers; nothing else about it needs naming.
#
# Two properties do the work, and both are deliberate:
#
# 1. The probe point varies with the variable index. A point that did
#    not (all-ones, say) would be a fixed point of every permutation, so
#    a model symmetric at that point would slip through. Varying it
#    breaks that symmetry for free.
#
# 2. What gets recorded is a small vector of order-weighted projections
#    compared with a *relative tolerance*, not a hash compared for
#    equality. A hash cannot be compared approximately, and a model
#    whose evaluation is not bitwise reproducible — a threaded AD
#    backend, a different BLAS, another platform — would then be refused
#    for reproducing itself to 15 digits instead of 17. Refusing a valid
#    replay is a worse failure than the one being fixed, so the
#    comparison has to have slack in it, and that rules a digest out.

#: Order-weighted projections taken of each probed block.
_PROBE_PROJECTIONS = 4

#: Relative tolerance for the probe comparison. A permutation of
#: distinct values moves a projection by a fraction of the block's own
#: magnitude; measured evaluation noise sits nine orders below that
#: (re-associating a model's internal sums moves it by 5e-18 relative).
PROBE_RTOL = 1e-9

#: What fraction of the largest block's scale floors a *near-zero*
#: block's tolerance. It has to be small: the floor exists to keep a
#: block that computes to ~0 out of cancellation of large terms from
#: being held to bit equality, and `PROBE_RTOL * _PROBE_FLOOR_FRAC`
#: (1e-15 of the largest scale) sits an order above that cancellation
#: noise. Making it 1.0 — which is what the pre-#659 code computed —
#: hands every block the *largest* block's tolerance, so an inert
#: additive constant on the objective silently switches reorder
#: detection off (gh#659).
_PROBE_FLOOR_FRAC = 1e-6

#: |bound| at or above this is not a bound. pounce/Ipopt spell infinity
#: 2e19; nothing smaller than 1e19 is a real bound.
_BOUND_INF = 1e19

#: Irrational-ish strides. ``frac((j+1)*a)`` is a low-discrepancy
#: sequence in [0,1), and ``*`` / ``floor`` are exact IEEE-754
#: operations — so the weights and the probe point are bit-identical on
#: every platform, which nothing routed through libm (``cos``, ``exp``)
#: can promise.
_PROBE_STRIDES = (
    0.6180339887498949,   # 1/phi
    0.4142135623730951,   # sqrt(2) - 1
    0.7320508075688772,   # sqrt(3) - 1
    0.2360679774997896,   # sqrt(5) - 2
)

#: The blocks evaluated at the probe point, in record order.
_PROBE_BLOCKS = ("objective", "gradient", "constraints", "jacobian")


def _frac(a: np.ndarray) -> np.ndarray:
    return a - np.floor(a)


def _probe_point(lb, ub) -> np.ndarray:
    """A deterministic point strictly inside the box, varying with index.

    Feasibility with respect to the *constraints* is neither sought nor
    needed — this is a fingerprint, not a solve. Staying inside the
    *bounds* is needed, because a model is entitled to be undefined
    outside its own box (``log(x)`` with ``lb=0``).
    """
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    j = np.arange(lb.size, dtype=float) + 1.0
    # in [0.25, 0.75): off-centre, index-dependent, and never at a bound
    t = 0.25 + 0.5 * _frac(j * _PROBE_STRIDES[0])
    lo = lb > -_BOUND_INF
    hi = ub < _BOUND_INF
    x = np.empty(lb.size, dtype=float)
    # Both bounds: the box centre, offset by a capped fraction of the
    # span. Capping keeps a [-1e18, 1e18] box from being probed at 1e17,
    # where a badly scaled model overflows and the probe is lost.
    # `0.5*lb + 0.5*ub` rather than `0.5*(lb+ub)` for the same reason.
    both = lo & hi
    x[both] = (0.5 * lb + 0.5 * ub + (t - 0.5) * np.minimum(ub - lb, 1.0))[both]
    only_lo = lo & ~hi
    x[only_lo] = (lb + t)[only_lo]
    only_hi = hi & ~lo
    x[only_hi] = (ub - t)[only_hi]
    free = ~lo & ~hi
    x[free] = (t - 0.5)[free]
    return x


def _sketch(values) -> List[float]:
    """`_PROBE_PROJECTIONS` order-weighted sums of `values`, plus a scale.

    Each projection weights entry `j` by a different low-discrepancy
    value in [-0.5, 0.5), so reordering the entries moves it. The
    trailing L1 norm is permutation-*invariant* and is what the
    comparison measures the projections against, which is what makes the
    tolerance relative to the model's own magnitude rather than to 1.0.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return [0.0] * (_PROBE_PROJECTIONS + 1)
    j = np.arange(v.size, dtype=float) + 1.0
    out = [float(np.dot(_frac(j * a) - 0.5, v)) for a in _PROBE_STRIDES]
    out.append(float(np.abs(v).sum()))
    return out


def _model_probe(problem) -> Optional[Tuple[float, ...]]:
    """Evaluate `problem`'s model once and sketch the result.

    ``None`` — an unrecorded, and therefore unverifiable, facet — when
    the model cannot be reached, will not evaluate at the probe point,
    or answers with something non-finite. Every one of those is a model
    this cannot fingerprint, and none of them is a reason to refuse a
    replay, so the facet is dropped rather than failed.
    """
    try:
        obj = problem.problem_obj
        lb, ub, _, _ = problem.get_bounds()
    except AttributeError:  # pragma: no cover - pre-#607 extension
        return None
    if obj is None:
        return None
    try:
        x = _probe_point(lb, ub)
        flat: List[float] = []
        for name in _PROBE_BLOCKS:
            fn = getattr(obj, name, None)
            flat += _sketch(() if fn is None else fn(x))
    except Exception:  # noqa: BLE001 - a model that will not evaluate at
        # an arbitrary interior point is a model this cannot fingerprint,
        # not a model whose warm start is invalid. Same contract as
        # `_structure_digest`: an unreadable facet reads as unverifiable.
        return None
    if not all(np.isfinite(flat)):
        return None
    return tuple(flat)


def _probe_agrees(a: Sequence[float], b: Sequence[float]) -> bool:
    """Do two probes describe the same model, to :data:`PROBE_RTOL`?

    Compared block by block against each block's own L1 scale, with
    `_PROBE_FLOOR_FRAC` of the largest block's scale as a floor — so a
    gradient that is identically zero is judged against the magnitude of
    the rest of the model rather than against nothing, which would
    demand bit equality of exactly the block least likely to reproduce
    bitwise.

    The floor is a *fraction* of the largest scale, not the largest
    scale itself. Flooring at the bare maximum is what gh#659 was: it
    made `max(scales[block], floor)` equal `floor` for every block, so
    each block was judged against the largest block's magnitude and the
    per-block scale this docstring describes never took effect. An inert
    additive constant on the objective — which changes no derivative and
    no solution — then raised the gradient/constraint/jacobian
    tolerances until a variable transposition slipped through the
    enforcing `check_compatible` gate.
    """
    if len(a) != len(b):
        return False
    stride = _PROBE_PROJECTIONS + 1
    scales = [max(abs(a[k]), abs(b[k]))
              for k in range(_PROBE_PROJECTIONS, len(a), stride)]
    floor = _PROBE_FLOOR_FRAC * max(scales) if scales else 0.0
    for block, off in enumerate(range(0, len(a), stride)):
        tol = PROBE_RTOL * max(scales[block], floor)
        for k in range(off, off + stride):
            if abs(a[k] - b[k]) > tol:
                return False
    return True


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
        probe: Order-sensitive sketch of the model evaluated once at a
            fixed interior point (pounce#621). This is the facet that
            sees a pure *reordering*, which no digest of the model's
            declarations can. Compared to a relative tolerance, not for
            equality.
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
    probe: Optional[Tuple[float, ...]] = None
    scaling: Optional[str] = None
    algorithm: Optional[str] = None
    model: Optional[str] = None

    #: Facets compared by :func:`compare`, in report order. Dimensions
    #: come first: they are the one facet recoverable from a legacy
    #: artifact, and the one whose mismatch the native layer would
    #: otherwise report from deep inside solve preparation.
    FACETS = ("n", "m", "var_ids", "con_ids", "bounds", "sparsity",
              "probe", "scaling", "algorithm", "model")

    #: Facets compared only when *both* sides recorded them.
    #:
    #: Stable IDs are the transfer key, not a verification requirement: a
    #: live ``Problem`` has no idea what its variables are called, so a
    #: target signature almost never carries them, and treating that
    #: absence as unverifiable would make signing an artifact with IDs
    #: strictly worse than signing it without.
    #:
    #: The probe is optional for a different reason: it is best-effort by
    #: construction. A model that will not evaluate at an arbitrary
    #: interior point, an artifact captured with ``probe=False``, and
    #: every artifact written before pounce#621 all carry ``None``, and
    #: none of them is evidence that the replay is wrong.
    OPTIONAL_FACETS = ("var_ids", "con_ids", "probe")

    # -- construction ---------------------------------------------------

    @classmethod
    def from_problem(
        cls, problem, var_ids=None, con_ids=None, probe=True
    ) -> "ProblemSignature":
        """Fingerprint a live :class:`pounce.Problem`.

        `var_ids` / `con_ids` are optional stable identifiers (any
        sequence of `n` / `m` unique values; they are stringified).
        Supplying them is what lets :meth:`pounce.WarmStart.reindex`
        transfer a warm start across a reordering or a horizon shift
        instead of only refusing it.

        `probe` evaluates the model once at a fixed interior point to
        record the order-sensitive :attr:`probe` facet (pounce#621).
        Pass ``False`` for a model whose evaluation is expensive enough
        to notice or has side effects; the cost is then zero on both
        sides, because a replay only probes the target when the artifact
        it is being checked against carries a probe of its own.
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
            probe=_model_probe(problem) if probe else None,
            scaling=_scaling_digest(problem, opts),
            algorithm=_digest(_sorted_pairs(opts, _ALGORITHM_OPTIONS)),
            model=_digest(_sorted_pairs(opts, MODEL_OPTIONS)),
        )

    # -- persistence ----------------------------------------------------

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        for k in ("var_ids", "con_ids", "probe"):
            if d[k] is not None:
                d[k] = list(d[k])
        # `repr`-based float encoding round-trips a double exactly, so a
        # probe survives save/load bit-for-bit and the tolerance is spent
        # on the model's own noise rather than on the serializer's.
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProblemSignature":
        d = json.loads(text)
        for k in ("var_ids", "con_ids"):
            if d.get(k) is not None:
                d[k] = tuple(str(v) for v in d[k])
        if d.get("probe") is not None:
            d["probe"] = tuple(float(v) for v in d["probe"])
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
        if self.facet == "probe" and not self.unverifiable:
            # The raw projections mean nothing to a reader, and the one
            # thing worth saying is what this facet is *for*: it is the
            # only one that moves under a pure reordering.
            return (
                "probe: this problem's model does not evaluate to the same "
                "numbers as the one the warm start was captured against "
                "(a reordering of the variables looks exactly like this; so "
                "does a different model of the same shape)"
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
        elif facet == "probe":
            # Floats out of a model evaluation: compared to a relative
            # tolerance, never for equality. See `_probe_agrees`.
            if not _probe_agrees(a, b):
                out.append(Mismatch(facet, a, b))
        elif a != b:
            out.append(Mismatch(facet, a, b))
    return out


def ordering_is_unverified(
    captured: ProblemSignature, target: ProblemSignature
) -> bool:
    """True when nothing in this comparison could have seen a reordering.

    Both routes to catching a permutation need something on *both*
    sides: stable IDs, or a model probe. When neither is present on both
    the structural digests are all that ran, and those are blind to it
    (pounce#621) — which is worth saying out loud rather than leaving the
    reader to infer from a clean report.
    """
    return not (
        (captured.var_ids is not None and target.var_ids is not None)
        or (captured.probe is not None and target.probe is not None)
    )


#: Said whenever a comparison could not have seen a reordering.
ORDERING_UNVERIFIED_NOTE = (
    "note: neither a model probe nor stable IDs were available on both "
    "sides, so a pure reordering of the variables would not have been "
    "caught here (pounce#621). Re-capture with "
    "WarmStart.from_info(x, info, problem=prob) on a build that probes, "
    "or pass var_ids= on both sides."
)


def format_report(
    mismatches: Sequence[Mismatch],
    *,
    replay: str,
    schema_version: Optional[int],
    source: Optional[ProblemSignature] = None,
    ordering_unverified: bool = False,
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
    if ordering_unverified:
        lines.append(f"  ({ORDERING_UNVERIFIED_NOTE})")
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
