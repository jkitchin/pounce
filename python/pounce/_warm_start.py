"""One warm-start object for every solve path.

POUNCE's warm-start machinery is spread over several knobs whose
interplay is easy to get wrong (see ``docs/src/initialization.md``):
``warm_start_init_point=yes`` alone saves nothing, because the default
``mu_init`` (0.1) re-walks the barrier schedule and the default
``warm_start_bound_push``/``_frac`` (1e-3) shove an at-the-bound
solution back off its bounds. :class:`WarmStart` packages the whole
recipe into a single argument::

    x, info = prob.solve(x0=x0)                     # cold solve
    ws = pounce.WarmStart.from_info(x, info)        # capture everything
    x2, info2 = prob.solve(warm_start=ws)           # warm re-solve

    ws.save("state.npz")                            # ... and across processes
    ws = pounce.WarmStart.load("state.npz")

A warm start is a point in *one* model's variable space. Pass
``problem=`` to :meth:`WarmStart.from_info` and the object also carries
a :class:`~pounce._warm_start_schema.ProblemSignature` — dimensions,
bound and sparsity signatures, scaling convention, algorithm, and the
model-defining options — which is checked before the solver is entered
and refuses a replay against a model that has moved underneath it
(pounce#607). :meth:`WarmStart.transfer` / :meth:`WarmStart.reindex` are
the explicit way across a reordering or a horizon shift.

Applying a warm start is **scoped**: the enabling options below are
installed for the duration of that one call and taken back afterwards,
including when the solve raises, so a warm solve never changes what the
next ordinary ``solve`` on the same :class:`Problem` does.

``warm_start=`` is accepted by :meth:`pounce.Problem.solve` and
:func:`pounce.minimize`. On the interior-point path it seeds the primal
and dual iterates and sets the five enabling options; on the active-set
SQP path (``algorithm=active-set-sqp``) it forwards the captured
working set, which is that path's warm-start payload.

Since pounce#606 the *quality* of the supplied point is the solver's
business rather than this object's: with the default
``recentering="residual"`` the interior-point initializer measures the
seeded iterate's primal, dual and complementarity residuals and picks
μ, the fills for any multiplier block the caller left out, and the
recentering strength from that measurement. The knobs here therefore
set a floor, not a verdict — ``mu_init`` is the barrier the caller
would *like*, and a stale point gets a looser one. What the solver
decided is reported back in ``info["warm_start"]``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable, Optional, Tuple

import numpy as np

from . import _pounce
from ._warm_start_schema import (
    COMPAT_MODES,
    WARM_START_SCHEMA_VERSION,
    ProblemSignature,
    WarmStartCompatibilityError,
    WarmStartCompatibilityWarning,
    WarmStartLegacyWarning,
    compare,
    format_report,
)

__all__ = [
    "WarmStart",
    "TransferContext",
    "ProblemSignature",
    "WarmStartCompatibilityError",
    "WarmStartCompatibilityWarning",
    "WarmStartLegacyWarning",
]


def _opt_array(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    return a if a.size else None


@dataclasses.dataclass
class WarmStart:
    """A captured solve state, usable as ``solve(warm_start=...)``.

    Attributes:
        x: Primal point (used as ``x0`` unless an explicit ``x0`` is
            passed to ``solve``).
        lagrange: Constraint multipliers (``info["mult_g"]``).
        zl / zu: Lower / upper bound multipliers.
        mu: Barrier parameter at capture (``info["mu"]``); seeds
            ``mu_init``. ``None`` or ``<= 0`` (e.g. captured from the
            SQP path) falls back to ``mu_init_fallback``.
        working_set: The SQP ``(bounds, constraints)`` working-set pair,
            forwarded on the ``algorithm=active-set-sqp`` path.
        bound_push: Value applied to the five ``warm_start_*_push`` /
            ``_frac`` options. The tight default (1e-9) keeps an
            at-the-bound solution essentially where it is; raise it if
            the next problem's solution may sit elsewhere.
        mu_init: Explicit ``mu_init`` override; default derives it from
            ``mu`` (clamped to ``[1e-9, 1e-1]``). Under the default
            ``recentering="residual"`` this is a **floor**, not a
            setting: the solver measures the supplied point and raises
            μ above it when the point cannot support that barrier
            (pounce#606). Pass ``warm_start_target_mu`` yourself to pin
            μ outright.
        mu_init_fallback: ``mu_init`` used when ``mu`` is unknown.
        recentering: ``warm_start_recentering`` (pounce#606).
            ``"residual"`` (the default) lets the solver measure the
            supplied point's primal/dual/complementarity residuals and
            derive μ, the unseeded bound-multiplier fills, and the
            equality-multiplier reconstruction from them.
            ``"none"`` restores the pre-#606 universal constants.
            ``None`` leaves the option unset (solver default).
        signature: The :class:`ProblemSignature` this state was captured
            against, or ``None`` for an unsigned state (which is what
            ``from_info`` produces when no ``problem=`` is given, and
            what every archive written before pounce#607 holds).
        compat: Replay compatibility mode — ``"strict"`` (default),
            ``"warn"``, or ``"unsafe"``. See
            :meth:`check_compatible`.
        replay: ``"exact"`` for a state captured directly from the
            problem being replayed on, ``"mapped"`` for one produced by
            :meth:`transfer` / :meth:`reindex`.
        source_signature: For a mapped state, the signature it was
            mapped *from*. Provenance only; not compared.
        schema_version: Artifact schema this state came from.
            :data:`~pounce._warm_start_schema.WARM_START_SCHEMA_VERSION`
            for anything this build created, ``1`` for a legacy archive.
        origin: ``"memory"`` for a state captured in this process,
            ``"file"`` for one read back by :meth:`load`. Only the
            second kind warns when it turns out to be unsigned: an
            in-memory state that the caller chose not to sign is the
            pre-#607 behaviour and stays silent, whereas an *artifact*
            with no verifiable metadata is the case the issue is about.
    """

    x: np.ndarray
    lagrange: Optional[np.ndarray] = None
    zl: Optional[np.ndarray] = None
    zu: Optional[np.ndarray] = None
    mu: Optional[float] = None
    working_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    bound_push: float = 1e-9
    mu_init: Optional[float] = None
    mu_init_fallback: float = 1e-6
    recentering: Optional[str] = "residual"
    signature: Optional[ProblemSignature] = None
    compat: str = "strict"
    replay: str = "exact"
    source_signature: Optional[ProblemSignature] = None
    schema_version: int = WARM_START_SCHEMA_VERSION
    origin: str = "memory"

    def __post_init__(self):
        if self.compat not in COMPAT_MODES:
            raise ValueError(
                f"WarmStart(compat={self.compat!r}): expected one of "
                + ", ".join(repr(m) for m in COMPAT_MODES)
            )
        if self.replay not in ("exact", "mapped"):
            raise ValueError(
                f"WarmStart(replay={self.replay!r}): expected 'exact' or 'mapped'"
            )
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.lagrange = _opt_array(self.lagrange)
        self.zl = _opt_array(self.zl)
        self.zu = _opt_array(self.zu)
        if self.working_set is not None:
            b, c = self.working_set
            self.working_set = (
                np.asarray(b, dtype=np.int8).ravel(),
                np.asarray(c, dtype=np.int8).ravel(),
            )

    # -- construction --------------------------------------------------

    @classmethod
    def from_info(
        cls, x, info, problem=None, var_ids=None, con_ids=None, **overrides
    ) -> "WarmStart":
        """Capture a warm start from a solve's ``(x, info)`` result.

        Works with :meth:`Problem.solve`'s ``info`` dict and with
        :func:`pounce.minimize`'s ``result.info``. Keyword overrides
        (e.g. ``bound_push=1e-6``) are forwarded to the constructor.

        Pass ``problem=`` — the :class:`Problem` that produced the
        result — to fingerprint the model as well, which is what lets a
        later replay be checked instead of merely attempted
        (pounce#607). ``var_ids`` / ``con_ids`` are optional stable
        identifiers for the variables and constraints; supply them and
        :meth:`reindex` can carry the state across a reordering or a
        horizon shift.
        """
        mu = info.get("mu")
        mu = float(mu) if mu is not None and float(mu) > 0.0 else None
        if problem is not None:
            overrides.setdefault(
                "signature",
                ProblemSignature.from_problem(problem, var_ids, con_ids),
            )
        elif var_ids is not None or con_ids is not None:
            raise TypeError(
                "WarmStart.from_info: var_ids / con_ids identify a model, so "
                "they need problem= as well"
            )
        return cls(
            x=np.asarray(x, dtype=float),
            lagrange=info.get("mult_g"),
            zl=info.get("mult_x_L"),
            zu=info.get("mult_x_U"),
            mu=mu,
            working_set=info.get("working_set"),
            **overrides,
        )

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        """Serialize to a NumPy ``.npz`` archive (portable across
        processes; the file-based analog of the GAMS ``sqp_state_file``)."""
        payload = {"x": self.x, "_meta": np.array(
            [self.mu if self.mu is not None else np.nan,
             self.bound_push,
             self.mu_init if self.mu_init is not None else np.nan,
             self.mu_init_fallback]
        )}
        # `recentering` is a string, so it rides in its own array
        # rather than in the numeric `_meta` row. Absent in archives
        # written before pounce#606; `load` treats that as the default.
        if self.recentering is not None:
            payload["_recentering"] = np.array(self.recentering)
        for key in ("lagrange", "zl", "zu"):
            v = getattr(self, key)
            if v is not None:
                payload[key] = v
        if self.working_set is not None:
            payload["ws_bounds"], payload["ws_constraints"] = self.working_set
        payload.update(self._schema_payload())
        np.savez(path, **payload)

    def _schema_payload(self) -> dict:
        """The schema-v2 keys (pounce#607), as their own block.

        Kept out of :meth:`save` so the two evolve independently: the
        array payload is the warm start, this is the metadata that says
        which model it belongs to. Everything here is a plain string or
        integer array, so archives stay ``allow_pickle=False``-loadable.
        """
        out = {"_schema": np.array(WARM_START_SCHEMA_VERSION, dtype=np.int64),
               "_compat": np.array(self.compat),
               "_replay": np.array(self.replay)}
        if self.signature is not None:
            out["_signature"] = np.array(self.signature.to_json())
        if self.source_signature is not None:
            out["_source_signature"] = np.array(self.source_signature.to_json())
        return out

    @staticmethod
    def _schema_fields(data) -> dict:
        """Inverse of :meth:`_schema_payload`.

        A version-1 archive has none of these keys. It loads as an
        unsigned, legacy-marked state rather than failing: refusing to
        read an artifact that a user wrote last week and that replays
        correctly on the model it came from would be a migration cost
        with no safety return. What version 1 *does* forfeit is the
        checking — :meth:`check_compatible` says so once, out loud, and
        points at :meth:`migrate`.
        """
        files = data.files
        if "_schema" not in files:
            return {"schema_version": 1, "origin": "file"}
        out = {
            "origin": "file",
            "schema_version": int(data["_schema"]),
            "compat": str(data["_compat"]),
            "replay": str(data["_replay"]),
        }
        if "_signature" in files:
            out["signature"] = ProblemSignature.from_json(str(data["_signature"]))
        if "_source_signature" in files:
            out["source_signature"] = ProblemSignature.from_json(
                str(data["_source_signature"])
            )
        return out

    @classmethod
    def load(cls, path, compat=None) -> "WarmStart":
        """Inverse of :meth:`save`.

        `compat` overrides the mode stored in the archive; pass
        ``"warn"`` or ``"unsafe"`` to relax the check on an artifact you
        know transfers.
        """
        with np.load(path, allow_pickle=False) as data:
            meta = data["_meta"]
            working_set = None
            if "ws_bounds" in data.files:
                working_set = (data["ws_bounds"], data["ws_constraints"])
            recentering = (
                str(data["_recentering"])
                if "_recentering" in data.files
                else None
            )
            return cls(
                x=data["x"],
                lagrange=data["lagrange"] if "lagrange" in data.files else None,
                zl=data["zl"] if "zl" in data.files else None,
                zu=data["zu"] if "zu" in data.files else None,
                mu=None if np.isnan(meta[0]) else float(meta[0]),
                working_set=working_set,
                bound_push=float(meta[1]),
                mu_init=None if np.isnan(meta[2]) else float(meta[2]),
                mu_init_fallback=float(meta[3]),
                recentering=recentering,
                **cls._schema_overrides(data, compat),
            )

    @classmethod
    def _schema_overrides(cls, data, compat) -> dict:
        fields = cls._schema_fields(data)
        if compat is not None:
            fields["compat"] = compat
        return fields

    # -- application ----------------------------------------------------

    def options(self) -> dict:
        """The enabling solver options this warm start implies.

        ``warm_start_init_point=yes`` makes the solver honor the seeds;
        ``mu_init`` skips the barrier walk-down; the tightened
        ``warm_start_*`` pushes keep an at-the-bound point where it is.
        All are ignored (harmlessly) on the SQP path.
        """
        if self.mu_init is not None:
            mu_init = self.mu_init
        elif self.mu is not None:
            mu_init = float(np.clip(self.mu, 1e-9, 1e-1))
        else:
            mu_init = self.mu_init_fallback
        p = self.bound_push
        opts = {
            "warm_start_init_point": "yes",
            "mu_init": mu_init,
            "warm_start_bound_push": p,
            "warm_start_bound_frac": p,
            "warm_start_slack_bound_push": p,
            "warm_start_slack_bound_frac": p,
            "warm_start_mult_bound_push": p,
        }
        if self.recentering is not None:
            opts["warm_start_recentering"] = self.recentering
        return opts

    def solve_kwargs(self) -> dict:
        """The seed keyword arguments for :meth:`Problem.solve`."""
        kw = {}
        if self.lagrange is not None:
            kw["lagrange"] = self.lagrange
        if self.zl is not None:
            kw["zl"] = self.zl
        if self.zu is not None:
            kw["zu"] = self.zu
        if self.working_set is not None:
            kw["working_set"] = self.working_set
        return kw

    # -- compatibility ---------------------------------------------------

    def check_compatible(
        self, problem, compat=None, var_ids=None, con_ids=None
    ) -> list:
        """Validate this state against `problem` and act on the verdict.

        Returns the list of
        :class:`~pounce._warm_start_schema.Mismatch` found (empty when
        the replay is clean). What happens when it is *not* empty is the
        mode's business:

        * ``strict`` (the default) raises
          :class:`WarmStartCompatibilityError` carrying the full report,
          before the solver is entered;
        * ``warn`` emits the same report as a
          :class:`WarmStartCompatibilityWarning` and proceeds;
        * ``unsafe`` skips the comparison entirely.

        Dimensions are checked against the *arrays* whether or not this
        state is signed, so even a legacy artifact cannot be replayed at
        the wrong size. Everything else needs a signature. An unsigned
        state read back from a *file* warns
        (:class:`WarmStartLegacyWarning`) that the rest is unverifiable
        and names :meth:`migrate`; an unsigned state built in this
        process is the pre-#607 usage and stays silent.

        `var_ids` / `con_ids` name `problem`'s variables and constraints
        in the vocabulary this state was captured with. Supply them to
        catch a **reordering**: a permutation is the one structural
        change a fingerprint cannot see on its own, because permuting a
        model with a uniform box and a dense jacobian leaves the bound
        and sparsity digests bit-identical. Ordering is knowledge only
        the caller has; the digests cover everything else.
        """
        mode = self.compat if compat is None else compat
        if mode not in COMPAT_MODES:
            raise ValueError(
                f"check_compatible(compat={mode!r}): expected one of "
                + ", ".join(repr(m) for m in COMPAT_MODES)
            )
        if mode == "unsafe":
            return []

        mismatches = self._mismatches(problem, var_ids, con_ids)
        if self.signature is None and self.origin == "file" and not mismatches:
            # A *persisted* artifact with nothing to check it against.
            # Refusing outright would break every archive written before
            # pounce#607 while it still replays correctly on the model it
            # came from — a migration cost with no safety return. Saying
            # so, with the two ways to fix it, is the middle ground the
            # issue's "legacy artifact migration" asks for.
            warnings.warn(
                f"warm start loaded from a schema v{self.schema_version} "
                "archive carries no model signature, so only its dimensions "
                "could be checked against this problem. Re-capture with "
                "WarmStart.from_info(x, info, problem=prob) and re-save, or "
                "call ws.migrate(prob) to sign this one against the problem "
                "you are replaying it on.",
                WarmStartLegacyWarning,
                stacklevel=3,
            )

        if mismatches:
            report = format_report(
                mismatches,
                replay=self.replay,
                schema_version=self.schema_version,
                source=self.source_signature,
            )
            if mode == "strict":
                raise WarmStartCompatibilityError(report)
            warnings.warn(report, WarmStartCompatibilityWarning, stacklevel=3)
        return mismatches

    def describe_compatibility(self, problem, var_ids=None, con_ids=None) -> str:
        """The mismatch report against `problem` as a string, without
        raising or warning. The dry run for a replay you are unsure of.
        """
        mismatches = self._mismatches(problem, var_ids, con_ids)
        if not mismatches:
            return "warm start is compatible with this problem"
        return format_report(
            mismatches,
            replay=self.replay,
            schema_version=self.schema_version,
            source=self.source_signature,
        )

    def _mismatches(self, problem, var_ids=None, con_ids=None) -> list:
        """The facets on which this state and `problem` disagree.

        A signed state is compared facet by facet. An unsigned one is
        compared on the only facets its own arrays witness — the
        dimensions — which is what keeps a legacy artifact from being
        replayed at the wrong size.
        """
        if self.signature is not None:
            return compare(
                self.signature,
                ProblemSignature.from_problem(problem, var_ids, con_ids),
            )
        if var_ids is not None or con_ids is not None:
            raise WarmStartCompatibilityError(
                "this warm start carries no model signature, so there is "
                "nothing to compare the supplied var_ids / con_ids against. "
                "Capture it with WarmStart.from_info(x, info, problem=prob, "
                "var_ids=…)."
            )
        arrays = self._array_signature()
        # When the state carries no constraint-space array at all there
        # is nothing for `m` to be wrong about, so do not manufacture an
        # unverifiable facet out of the target's `m`. (An unconstrained
        # solve reports an empty `mult_g`, which `_opt_array` normalizes
        # to None — that is the common case here, not an exotic one.)
        return compare(
            arrays,
            ProblemSignature(
                n=int(problem.n),
                m=int(problem.m) if arrays.m is not None else None,
            ),
        )

    def _array_signature(self) -> ProblemSignature:
        """The dimensions this state's own arrays imply.

        ``lagrange`` is the only witness to `m`; when it was not
        captured (an unconstrained solve, or a caller who dropped it)
        `m` is left unrecorded rather than guessed, and
        :func:`~pounce._warm_start_schema.compare` reports the facet as
        unverifiable rather than inventing agreement.
        """
        n = int(self.x.size)
        for name, arr in (("zl", self.zl), ("zu", self.zu)):
            if arr is not None and arr.size != n:
                raise ValueError(
                    f"WarmStart: x has {n} entries but {name} has {arr.size}; "
                    "the state is internally inconsistent"
                )
        m = None
        if self.lagrange is not None:
            m = int(self.lagrange.size)
        elif self.working_set is not None:
            m = int(self.working_set[1].size)
        return ProblemSignature(n=n, m=m)

    def migrate(self, problem, var_ids=None, con_ids=None) -> "WarmStart":
        """Re-sign this state against `problem`, as-is.

        The migration path for a legacy (schema v1) archive, and the
        escape hatch for a signed one whose model changed in a way the
        caller knows is immaterial. It is an **assertion**, not a
        conversion: no array is touched, so use it only when the arrays
        really do belong to this problem. When they need rearranging,
        that is :meth:`reindex` or :meth:`transfer`.
        """
        target = ProblemSignature.from_problem(problem, var_ids, con_ids)
        if int(problem.n) != self.x.size:
            raise WarmStartCompatibilityError(
                f"migrate: this problem has {int(problem.n)} variables but the "
                f"warm start's x has {self.x.size}; migrate re-signs a state, "
                "it does not resize one — use transfer() with a mapper"
            )
        if self.lagrange is not None and int(problem.m) != self.lagrange.size:
            raise WarmStartCompatibilityError(
                f"migrate: this problem has {int(problem.m)} constraints but "
                f"the warm start's lagrange has {self.lagrange.size}; use "
                "transfer() with a mapper"
            )
        return dataclasses.replace(
            self,
            signature=target,
            replay="exact",
            source_signature=self.signature,
            schema_version=WARM_START_SCHEMA_VERSION,
            origin="memory",
        )

    # -- transfer --------------------------------------------------------

    def transfer(
        self,
        problem,
        mapper: Optional[Callable[["TransferContext"], dict]] = None,
        var_ids=None,
        con_ids=None,
        fill_x=None,
    ) -> "WarmStart":
        """Map this state onto `problem`, producing a *mapped* replay.

        `mapper` is called with a :class:`TransferContext` and returns a
        dict of replacement arrays — any of ``x``, ``lagrange``, ``zl``,
        ``zu``, ``working_set``, ``mu``. Anything it leaves out is
        carried over unchanged, and every array it returns is
        length-checked against `problem` before the result is built, so
        a mapper with an off-by-one fails here rather than three
        function evaluations into the solve.

        This is the hook for a horizon shift or a changed
        discretization: the arrays are yours to rearrange, interpolate,
        or prolong. When both sides carry stable IDs and the only change
        is *which* variables are present and in what order,
        :meth:`reindex` writes the mapper for you.

        The result is signed against `problem` with ``replay="mapped"``
        and remembers where it came from, so replaying it on a *third*
        problem is refused exactly like an unmapped one.
        """
        target = ProblemSignature.from_problem(problem, var_ids, con_ids)
        n, m = target.n, target.m
        if mapper is None:
            mapper = _reindex_mapper(fill_x)
        elif fill_x is not None:
            raise TypeError(
                "transfer: fill_x is the default mapper's policy for "
                "variables the target has and the source does not; it means "
                "nothing alongside an explicit mapper"
            )
        ctx = TransferContext(source=self, target=target, problem=problem)
        payload = mapper(ctx)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError(
                "transfer: the mapper must return a dict of replacement "
                f"arrays (x / lagrange / zl / zu / working_set / mu), got "
                f"{type(payload).__name__}"
            )
        unknown = sorted(set(payload) - {"x", "lagrange", "zl", "zu",
                                         "working_set", "mu"})
        if unknown:
            raise TypeError(
                f"transfer: the mapper returned unknown keys {unknown}; "
                "expected any of x / lagrange / zl / zu / working_set / mu"
            )
        out = dataclasses.replace(
            self,
            x=payload.get("x", self.x),
            lagrange=payload.get("lagrange", self.lagrange),
            zl=payload.get("zl", self.zl),
            zu=payload.get("zu", self.zu),
            working_set=payload.get("working_set", self.working_set),
            mu=payload.get("mu", self.mu),
            signature=target,
            replay="mapped",
            source_signature=self.signature,
            schema_version=WARM_START_SCHEMA_VERSION,
            origin="memory",
        )
        _check_lengths(out, n, m)
        return out

    def reindex(self, problem, var_ids, con_ids=None, fill_x=None) -> "WarmStart":
        """Transfer by matching stable identifiers.

        `var_ids` / `con_ids` name `problem`'s variables and
        constraints in the same vocabulary this state was captured with
        (``from_info(..., problem=prob, var_ids=…)``). Entries the
        target shares with the source are carried across to their new
        positions; entries the target has and the source does not are
        *unseeded* — ``NaN`` in the multiplier blocks, which is the
        native warm-start contract for "you decide", and `fill_x`
        (default: zero clipped into the variable's box) in ``x``.

        That covers the two cases the issue names: a reordering, where
        the ID sets are equal, and a horizon shift, where they overlap.
        """
        if self.signature is None or self.signature.var_ids is None:
            raise WarmStartCompatibilityError(
                "reindex: this warm start carries no variable identifiers, so "
                "there is nothing to match the target's against. Capture it "
                "with WarmStart.from_info(x, info, problem=prob, "
                "var_ids=…), or use transfer() with an explicit mapper."
            )
        return self.transfer(
            problem, None, var_ids=var_ids, con_ids=con_ids, fill_x=fill_x
        )


@dataclasses.dataclass(frozen=True)
class TransferContext:
    """What a :meth:`WarmStart.transfer` mapper is given.

    Attributes:
        source: The :class:`WarmStart` being mapped (arrays included).
        target: The :class:`ProblemSignature` of the problem being
            mapped onto.
        problem: The target :class:`Problem` itself, for anything the
            signature does not carry (bounds, the model object).
    """

    source: "WarmStart"
    target: ProblemSignature
    problem: object

    def index_map(self, axis="var"):
        """Target-indexed array of source positions, ``-1`` where the
        target entry has no counterpart in the source.

        Requires stable IDs on both sides for `axis` (``"var"`` or
        ``"con"``); returns ``None`` when either side lacks them.
        """
        key = "var_ids" if axis == "var" else "con_ids"
        src = getattr(self.source.signature, key, None) if self.source.signature else None
        dst = getattr(self.target, key)
        if src is None or dst is None:
            return None
        where = {name: i for i, name in enumerate(src)}
        return np.array([where.get(name, -1) for name in dst], dtype=np.int64)

    def bounds(self):
        """``(lb, ub, cl, cu)`` of the target problem."""
        return self.problem.get_bounds()


def _gather(src, idx, fill):
    """`src[idx]`, with `fill` wherever `idx` is -1. `src` may be None."""
    if src is None:
        return None
    out = np.full(idx.size, fill, dtype=float)
    hit = idx >= 0
    out[hit] = np.asarray(src, dtype=float)[idx[hit]]
    return out


def _reindex_mapper(fill_x):
    """The default :meth:`WarmStart.transfer` mapper: match stable IDs.

    Unmatched multiplier entries become NaN, which the native warm-start
    initializer reads as "unseeded" and fills with its own resolved
    default — so a prolonged horizon does not carry a fabricated
    multiplier into the new block.
    """

    def mapper(ctx: TransferContext) -> dict:
        vmap = ctx.index_map("var")
        if vmap is None:
            raise WarmStartCompatibilityError(
                "transfer: no mapper was given and stable variable IDs are "
                "not present on both sides, so there is no way to know which "
                "of the target's variables the captured point refers to. "
                "Pass var_ids= for the target (and capture the source with "
                "var_ids=), or supply an explicit mapper."
            )
        lb, ub, _, _ = ctx.bounds()
        if fill_x is None:
            base = np.clip(0.0, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
        else:
            base = np.asarray(fill_x, dtype=float).ravel()
            if base.size == 1:
                base = np.full(vmap.size, float(base[0]))
            if base.size != vmap.size:
                raise ValueError(
                    f"transfer: fill_x has {base.size} entries, expected "
                    f"{vmap.size} (or a scalar)"
                )
        x = np.asarray(ctx.source.x, dtype=float)
        new_x = base.astype(float, copy=True)
        hit = vmap >= 0
        new_x[hit] = x[vmap[hit]]
        out = {
            "x": new_x,
            "zl": _gather(ctx.source.zl, vmap, np.nan),
            "zu": _gather(ctx.source.zu, vmap, np.nan),
        }
        cmap = ctx.index_map("con")
        if cmap is not None:
            out["lagrange"] = _gather(ctx.source.lagrange, cmap, np.nan)
        elif ctx.source.lagrange is not None and ctx.target.m != ctx.source.lagrange.size:
            raise WarmStartCompatibilityError(
                "transfer: the constraint count changed "
                f"({ctx.source.lagrange.size} -> {ctx.target.m}) but no "
                "constraint IDs were supplied, so the captured multipliers "
                "cannot be placed. Pass con_ids= on both sides, or supply an "
                "explicit mapper."
            )
        if ctx.source.working_set is not None:
            b, c = ctx.source.working_set
            wb = _gather(b, vmap, 0.0)
            wc = c if cmap is None else _gather(c, cmap, 0.0)
            out["working_set"] = (
                np.asarray(wb, dtype=np.int8),
                np.asarray(wc, dtype=np.int8),
            )
        return out

    return mapper


def _check_lengths(ws: "WarmStart", n: int, m: int) -> None:
    for name, arr, want in (
        ("x", ws.x, n),
        ("lagrange", ws.lagrange, m),
        ("zl", ws.zl, n),
        ("zu", ws.zu, n),
    ):
        if arr is not None and arr.size != want:
            raise ValueError(
                f"transfer: the mapped '{name}' has {arr.size} entries, but "
                f"the target problem wants {want}"
            )
    if ws.working_set is not None:
        b, c = ws.working_set
        if b.size != n or c.size != m:
            raise ValueError(
                f"transfer: the mapped working set is ({b.size}, {c.size}), "
                f"but the target problem wants ({n}, {m})"
            )


# ---------------------------------------------------------------------------
# Problem.solve(warm_start=...) — wrap the native method once at import.
# The pyo3 class cannot be subclassed, so the ergonomic entry point is a
# thin wrapper that translates a WarmStart into the native seed kwargs +
# enabling options and otherwise passes through unchanged.
# ---------------------------------------------------------------------------

_native_solve = _pounce.Problem.solve


def _solve_with_warm_start(
    self,
    x0=None,
    lagrange=None,
    zl=None,
    zu=None,
    working_set=None,
    warm_start: Optional[WarmStart] = None,
    compat: Optional[str] = None,
    var_ids=None,
    con_ids=None,
    **kwargs,
):
    # **kwargs forwards any other native solve keywords untouched
    # (e.g. report_path / report_detail), so this wrapper never lags
    # the native signature.
    if warm_start is None:
        for name, val in (("compat", compat), ("var_ids", var_ids),
                          ("con_ids", con_ids)):
            if val is not None:
                raise TypeError(
                    f"Problem.solve(): {name}= describes how to replay a warm "
                    "start and means nothing without warm_start="
                )
        if x0 is None:
            raise TypeError("Problem.solve() missing required argument: 'x0'")
        return _native_solve(
            self,
            x0=x0,
            lagrange=lagrange,
            zl=zl,
            zu=zu,
            working_set=working_set,
            **kwargs,
        )
    ws = warm_start
    # Refuse an incompatible replay here, before any solver state exists
    # (pounce#607). Doing it first also means the option overlay below is
    # never installed for a call that was never going to run.
    ws.check_compatible(self, compat=compat, var_ids=var_ids, con_ids=con_ids)
    kw = ws.solve_kwargs()
    # Explicit per-call seeds win over the WarmStart's captured ones.
    for key, val in (
        ("lagrange", lagrange),
        ("zl", zl),
        ("zu", zu),
        ("working_set", working_set),
    ):
        if val is not None:
            kw[key] = val
    # Scoped overlay: the enabling options belong to *this call*, not to
    # the Problem. `add_option` is append-only, so taking them back needs
    # the snapshot/restore pair — and the restore has to survive an
    # exception, or a warm solve that fails leaves the next ordinary
    # solve running someone else's mu_init (pounce#607). Nothing here
    # enumerates option names: whatever `ws.options()` returns is what
    # gets scoped, so an option added to that recipe later is covered by
    # construction.
    snapshot = self.options_snapshot()
    try:
        for k, v in ws.options().items():
            self.add_option(k, v)
        return _native_solve(self, x0=ws.x if x0 is None else x0, **kw, **kwargs)
    finally:
        self.restore_options(snapshot)


_solve_with_warm_start.__name__ = "solve"
_solve_with_warm_start.__qualname__ = "Problem.solve"
_solve_with_warm_start.__doc__ = (_native_solve.__doc__ or "") + (
    "\n\n"
    "warm_start : pounce.WarmStart, optional\n"
    "    A captured solve state (see ``WarmStart.from_info``). Applies the\n"
    "    enabling options (``warm_start_init_point=yes``, ``mu_init``, the\n"
    "    tightened ``warm_start_*`` pushes) as a **scoped overlay** — they\n"
    "    are installed for this call and taken back afterwards, including\n"
    "    when the solve raises, so they do not change what the next\n"
    "    ordinary ``solve`` on this Problem does. Seeds the duals, forwards\n"
    "    the SQP working set when present, and defaults ``x0`` to the\n"
    "    captured point. Explicit ``x0``/seed arguments override the\n"
    "    captured ones. A warm start carrying a model signature is checked\n"
    "    against this Problem first, and an incompatible one is refused\n"
    "    before the solver is entered.\n"
    "compat : {'strict', 'warn', 'unsafe'}, optional\n"
    "    Override the warm start's own compatibility mode for this call.\n"
    "var_ids, con_ids : sequence, optional\n"
    "    Stable identifiers for *this* Problem's variables / constraints,\n"
    "    in the vocabulary the warm start was captured with. Supply them to\n"
    "    catch a reordering, which the structural digests cannot see on\n"
    "    their own.\n"
)

_pounce.Problem.solve = _solve_with_warm_start
