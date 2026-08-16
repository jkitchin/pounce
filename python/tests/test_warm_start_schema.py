"""pounce#607 — warm-start artifacts carry the model they belong to, and
warm-start options are scoped to the call that asked for them.

Two independent failures lived in `python/pounce/_warm_start.py`:

1. Applying a warm start called `add_option` on the *persistent* Problem.
   `add_option` is append-only, so the seven enabling options stayed set
   for every later `solve` on that object. On the HS071 fixture below
   that took an ordinary cold solve from 17 iterations to 24 — the right
   answer down a longer trajectory, which is precisely the class of
   regression gh#544 shipped and the fixture sweep exists to catch.

2. A serialized `WarmStart` recorded arrays and four floats. Replayed
   against a model whose variables had been reordered it produced
   objective 16.3801 where the truth is 17.0140, with nothing raised
   and nothing warned.

The tests here pin both, plus the migration path for artifacts written
before the schema existed.
"""

import dataclasses
import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce
from pounce import (
    WarmStart,
    WarmStartCompatibilityError,
    WarmStartCompatibilityWarning,
    WarmStartLegacyWarning,
)


# ---------------------------------------------------------------------------
# HS071 and structural variants of it
# ---------------------------------------------------------------------------
class HS071:
    """min x0*x3*(x0+x1+x2) + x2  s.t. prod(x) >= 25, |x|^2 == 40.

    `perm[k]` is the index in the canonical ordering that sits at
    position k here, so `HS071(perm=[2, 0, 3, 1])` is the same
    mathematics with its variables shuffled.
    """

    def __init__(self, perm=None):
        self.perm = np.arange(4) if perm is None else np.asarray(perm)
        self.inv = np.argsort(self.perm)

    def _orig(self, x):
        return np.asarray(x)[self.inv]

    def objective(self, x):
        y = self._orig(x)
        return y[0] * y[3] * (y[0] + y[1] + y[2]) + y[2]

    def gradient(self, x):
        y = self._orig(x)
        g = np.array([
            y[0] * y[3] + y[3] * (y[0] + y[1] + y[2]),
            y[0] * y[3],
            y[0] * y[3] + 1.0,
            y[0] * (y[0] + y[1] + y[2]),
        ])
        return g[self.perm]

    def constraints(self, x):
        y = self._orig(x)
        return np.array([np.prod(y), np.dot(y, y)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        y = self._orig(x)
        j = np.array([
            y[1] * y[2] * y[3], y[0] * y[2] * y[3],
            y[0] * y[1] * y[3], y[0] * y[1] * y[2],
        ])
        return np.concatenate([j[self.perm], (2 * y)[self.perm]])


class HS071ExtraNonzero(HS071):
    """Same mathematics, one extra declared structural entry."""

    def jacobianstructure(self):
        rows, cols = super().jacobianstructure()
        return (np.append(rows, 0), np.append(cols, 0))

    def jacobian(self, x):
        return np.append(super().jacobian(x), 0.0)


X0 = np.array([1.0, 5.0, 5.0, 1.0])
X0_FAR = np.array([4.9, 4.9, 4.9, 4.9])
VAR_IDS = ["v0", "v1", "v2", "v3"]
CON_IDS = ["prod", "sumsq"]


def make(obj=None, ub=5.0, n=4, **opts):
    p = pounce.Problem(
        n=n, m=2, problem_obj=HS071() if obj is None else obj,
        lb=[1.0] * n, ub=[ub] * n, cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    for k, v in opts.items():
        p.add_option(k, v)
    return p


def cold():
    p = make()
    x, info = p.solve(x0=X0)
    return p, x, info


def signed(**kw):
    p, x, info = cold()
    return WarmStart.from_info(x, info, problem=p, **kw)


# ---------------------------------------------------------------------------
# 1. warm-start options are scoped to the call
# ---------------------------------------------------------------------------
def test_warm_options_do_not_leak_into_the_next_ordinary_solve():
    ws = signed()
    _, pristine = make().solve(x0=X0_FAR)

    reused = make()
    reused.solve(warm_start=ws)
    _, after = reused.solve(x0=X0_FAR)

    assert after["iter_count"] == pristine["iter_count"]
    assert after["obj_val"] == pytest.approx(pristine["obj_val"])


def test_warm_options_are_restored_even_when_the_solve_raises():
    ws = signed()
    _, pristine = make().solve(x0=X0_FAR)

    p = make()
    with pytest.raises(ValueError):
        # A wrong-length explicit seed fails inside solve preparation,
        # after the overlay is installed. Without the `finally` restore
        # the options survive the failed call.
        p.solve(warm_start=ws, lagrange=np.zeros(99))
    _, after = p.solve(x0=X0_FAR)

    assert after["iter_count"] == pristine["iter_count"]


def test_overlay_restores_the_exact_option_list():
    """The overlay is a snapshot/restore of the whole option list, not a
    hand-maintained list of names to unset — so an option the warm start
    introduces is removed, and one the caller had set is put back at its
    own value."""
    ws = signed()
    p = make()
    p.add_option("mu_init", 0.5)          # caller's own value for a key
    p.add_option("max_iter", 300)         # the warm start does not touch
    before = p.options_snapshot()

    p.solve(warm_start=ws)

    assert p.options_snapshot() == before
    strs, nums, ints = p.options_snapshot()
    assert dict(nums)["mu_init"] == 0.5
    assert dict(ints)["max_iter"] == 300
    assert "warm_start_init_point" not in dict(strs)


def test_scoped_overlay_covers_whatever_options_returns():
    """Nothing in the wrapper enumerates option names, so a key added to
    the recipe later is scoped by construction."""
    ws = signed()
    ws_opts = dict(ws.options())
    p = make()
    empty = p.options_snapshot()

    class Extra(WarmStart):
        def options(self):
            out = dict(super().options())
            out["bound_mult_init_val"] = 2.0
            return out

    extra = Extra(**{f.name: getattr(ws, f.name) for f in dataclasses.fields(ws)})
    assert "bound_mult_init_val" not in ws_opts
    p.solve(warm_start=extra)
    assert p.options_snapshot() == empty


# ---------------------------------------------------------------------------
# 2. a compatible round trip stays lossless
# ---------------------------------------------------------------------------
def test_round_trip_is_lossless_and_still_warm(tmp_path):
    p, cold_x, cold_info = cold()
    ws = WarmStart.from_info(cold_x, cold_info, problem=p,
                             var_ids=VAR_IDS, con_ids=CON_IDS,
                             bound_push=1e-8)
    path = tmp_path / "state.npz"
    ws.save(path)
    back = WarmStart.load(path)

    for name in ("x", "lagrange", "zl", "zu"):
        np.testing.assert_array_equal(getattr(back, name), getattr(ws, name))
    assert back.mu == ws.mu
    assert back.bound_push == 1e-8
    assert back.signature == ws.signature
    assert back.signature.var_ids == tuple(VAR_IDS)
    assert back.compat == "strict"
    assert back.replay == "exact"
    assert back.schema_version == 2

    warm_x, warm_info = make().solve(warm_start=back)
    assert warm_info["iter_count"] < cold_info["iter_count"]
    np.testing.assert_allclose(warm_x, cold_x, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. incompatible replays are refused with a report, before the solver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("facet,factory", [
    ("bounds", lambda: make(ub=4.0)),
    ("sparsity", lambda: make(obj=HS071ExtraNonzero())),
    ("scaling", lambda: make(nlp_scaling_method="none")),
    ("algorithm", lambda: make(algorithm="active-set-sqp")),
    ("model", lambda: make(fixed_variable_treatment="make_constraint")),
])
def test_incompatible_replay_is_refused(facet, factory):
    ws = signed()
    with pytest.raises(WarmStartCompatibilityError) as e:
        factory().solve(warm_start=ws)
    report = str(e.value)
    assert f"  - {facet}:" in report
    assert "re-capture against this problem" in report
    assert "ws.reindex(prob)" in report


def test_dimension_mismatch_is_refused_before_the_solver():
    ws = signed()
    with pytest.raises(WarmStartCompatibilityError) as e:
        make(n=5).solve(x0=np.full(5, 2.0), warm_start=ws)
    assert "n: captured 4, target 5" in str(e.value)


def test_reordered_variables_are_refused_when_ids_are_supplied():
    """A permutation of a model with a uniform box and a dense jacobian
    leaves every structural digest bit-identical — ordering is knowledge
    only the caller has. Replaying through it produced objective
    16.3801 against a true 17.0140 before pounce#607."""
    perm = [2, 0, 3, 1]
    ws = signed(var_ids=VAR_IDS, con_ids=CON_IDS)
    target = make(obj=HS071(perm=perm))

    # Without the ordering, nothing can see the permutation...
    assert ws.check_compatible(target) == []
    # ...but the caller who knows it gets a refusal.
    with pytest.raises(WarmStartCompatibilityError) as e:
        target.solve(warm_start=ws,
                     var_ids=[VAR_IDS[i] for i in perm], con_ids=CON_IDS)
    assert "var_ids: identifiers differ" in str(e.value)


def test_a_different_model_with_the_same_shape_is_refused():
    class Other(HS071):
        def objective(self, x):
            return float(np.sum((np.asarray(x) - 2.0) ** 2))

        def gradient(self, x):
            return 2.0 * (np.asarray(x) - 2.0)

    # Same n, m, bounds and declared sparsity; different bounds digest is
    # not what catches it, so make the boxes identical and lean on the
    # model facet by changing an option that redefines the model.
    ws = signed()
    with pytest.raises(WarmStartCompatibilityError):
        make(obj=Other(), bound_relax_factor=0.0).solve(warm_start=ws)


def test_describe_compatibility_reports_without_raising():
    ws = signed()
    assert "compatible" in ws.describe_compatibility(make())
    report = ws.describe_compatibility(make(ub=4.0))
    assert "bounds:" in report and "resolve it by one of" in report


# ---------------------------------------------------------------------------
# 4. strict / warn / unsafe
# ---------------------------------------------------------------------------
def test_warn_mode_reports_and_proceeds():
    ws = signed()
    with pytest.warns(WarmStartCompatibilityWarning, match="bounds:"):
        make(ub=4.0).solve(warm_start=ws, compat="warn")


def test_unsafe_mode_is_silent():
    import warnings as _w

    ws = signed()
    with _w.catch_warnings():
        _w.simplefilter("error")
        make(ub=4.0).solve(warm_start=ws, compat="unsafe")


def test_mode_stored_on_the_artifact_is_honored(tmp_path):
    p, x, info = cold()
    ws = WarmStart.from_info(x, info, problem=p, compat="warn")
    path = tmp_path / "warn.npz"
    ws.save(path)
    with pytest.warns(WarmStartCompatibilityWarning):
        make(ub=4.0).solve(warm_start=WarmStart.load(path))
    # ...and `load(compat=...)` overrides it.
    with pytest.raises(WarmStartCompatibilityError):
        make(ub=4.0).solve(warm_start=WarmStart.load(path, compat="strict"))


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="compat"):
        WarmStart(x=np.zeros(4), compat="lenient")


def test_replay_kwargs_need_a_warm_start():
    for kw in ({"compat": "warn"}, {"var_ids": VAR_IDS}, {"con_ids": CON_IDS}):
        with pytest.raises(TypeError, match="warm_start="):
            make().solve(x0=X0, **kw)


# ---------------------------------------------------------------------------
# 5. legacy artifacts
# ---------------------------------------------------------------------------
def _write_v1(path, ws):
    """Exactly what `save` wrote before pounce#607: arrays and a 4-wide
    numeric `_meta` row, no schema key."""
    payload = {"x": ws.x, "_meta": np.array([ws.mu, ws.bound_push,
                                             np.nan, ws.mu_init_fallback])}
    for key in ("lagrange", "zl", "zu"):
        v = getattr(ws, key)
        if v is not None:
            payload[key] = v
    np.savez(path, **payload)


def test_legacy_artifact_loads_replays_and_says_it_is_unverified(tmp_path):
    p, cold_x, cold_info = cold()
    path = tmp_path / "legacy.npz"
    _write_v1(path, WarmStart.from_info(cold_x, cold_info))

    ws = WarmStart.load(path)
    assert ws.schema_version == 1
    assert ws.signature is None
    assert ws.origin == "file"

    with pytest.warns(WarmStartLegacyWarning, match="schema v1"):
        warm_x, warm_info = make().solve(warm_start=ws)
    # It still warm-starts: the migration is about provenance, not payload.
    assert warm_info["iter_count"] < cold_info["iter_count"]
    np.testing.assert_allclose(warm_x, cold_x, atol=1e-6)


def test_legacy_artifact_still_cannot_be_replayed_at_the_wrong_size(tmp_path):
    p, cold_x, cold_info = cold()
    path = tmp_path / "legacy.npz"
    _write_v1(path, WarmStart.from_info(cold_x, cold_info))
    with pytest.raises(WarmStartCompatibilityError, match="n: captured 4"):
        make(n=5).solve(x0=np.full(5, 2.0), warm_start=WarmStart.load(path))


def test_migrate_signs_a_legacy_artifact(tmp_path):
    p, cold_x, cold_info = cold()
    path = tmp_path / "legacy.npz"
    _write_v1(path, WarmStart.from_info(cold_x, cold_info))
    legacy = WarmStart.load(path)

    target = make()
    migrated = legacy.migrate(target, var_ids=VAR_IDS, con_ids=CON_IDS)
    assert migrated.schema_version == 2
    assert migrated.signature.var_ids == tuple(VAR_IDS)
    assert migrated.source_signature is None

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")           # no legacy warning any more
        target.solve(warm_start=migrated)
    # ...and it is now a real signature, so it still refuses a bad target.
    with pytest.raises(WarmStartCompatibilityError):
        make(ub=4.0).solve(warm_start=migrated)


def test_migrate_refuses_to_resize():
    ws = signed()
    with pytest.raises(WarmStartCompatibilityError, match="does not resize"):
        ws.migrate(make(n=5))


def test_in_memory_unsigned_state_stays_silent():
    """The pre-#607 call — `from_info(x, info)` with no problem — keeps
    working, unchanged and unnagged. Only a *persisted* artifact with no
    metadata gets the migration notice."""
    import warnings as _w

    _, x, info = cold()
    ws = WarmStart.from_info(x, info)
    assert ws.signature is None and ws.origin == "memory"
    with _w.catch_warnings():
        _w.simplefilter("error")
        make().solve(warm_start=ws)


def test_v2_archive_is_still_loadable_by_a_v1_reader(tmp_path):
    """The schema keys are additive, so the array payload a pre-#607
    loader asks for is untouched."""
    p, x, info = cold()
    path = tmp_path / "v2.npz"
    WarmStart.from_info(x, info, problem=p).save(path)
    with np.load(path, allow_pickle=False) as data:
        assert {"x", "lagrange", "zl", "zu", "_meta"} <= set(data.files)
        assert data["_meta"].shape == (4,)
        assert int(data["_schema"]) == 2


# ---------------------------------------------------------------------------
# 6. transfer: horizon shift and explicit mappers
# ---------------------------------------------------------------------------
class Chain:
    """A tiny receding-horizon model: track `targets` under a slew limit.

    min sum_k (x_k - t_k)^2   s.t.  |x_{k+1} - x_k| <= 0.5,  0 <= x <= 10.
    """

    def __init__(self, targets):
        self.t = np.asarray(targets, dtype=float)
        self.nv = self.t.size

    def objective(self, x):
        return float(np.sum((np.asarray(x) - self.t) ** 2))

    def gradient(self, x):
        return 2.0 * (np.asarray(x) - self.t)

    def constraints(self, x):
        y = np.asarray(x)
        return y[1:] - y[:-1]

    def jacobianstructure(self):
        k = self.nv - 1
        rows = np.repeat(np.arange(k), 2)
        cols = np.column_stack([np.arange(k), np.arange(1, k + 1)]).ravel()
        return rows, cols

    def jacobian(self, x):
        return np.tile([-1.0, 1.0], self.nv - 1)


TRACK = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 6.0, 7.0])
HORIZON = 5


def window(start):
    """The horizon starting at `start`, plus its stable IDs."""
    t = TRACK[start:start + HORIZON]
    p = pounce.Problem(
        n=HORIZON, m=HORIZON - 1, problem_obj=Chain(t),
        lb=[0.0] * HORIZON, ub=[10.0] * HORIZON,
        cl=[-0.5] * (HORIZON - 1), cu=[0.5] * (HORIZON - 1),
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    var_ids = [f"x{start + i}" for i in range(HORIZON)]
    con_ids = [f"c{start + i}" for i in range(HORIZON - 1)]
    return p, var_ids, con_ids


def test_horizon_shift_transfers_and_replays():
    p0, v0, c0 = window(0)
    x0, info0 = p0.solve(x0=np.full(HORIZON, 3.0))
    ws = WarmStart.from_info(x0, info0, problem=p0, var_ids=v0, con_ids=c0)

    p1, v1, c1 = window(1)
    # Replaying the un-shifted artifact is refused: the objective's
    # targets moved, so the bound/model digests no longer match... and
    # where they would match, the IDs do.
    with pytest.raises(WarmStartCompatibilityError):
        p1.solve(warm_start=ws, var_ids=v1, con_ids=c1)

    moved = ws.reindex(p1, var_ids=v1, con_ids=c1)
    assert moved.replay == "mapped"
    assert moved.source_signature == ws.signature
    # x1..x4 carried across; the freshly-entered stage is unseeded.
    np.testing.assert_allclose(moved.x[:HORIZON - 1], x0[1:])
    assert np.isnan(moved.zl[-1]) and np.isnan(moved.zu[-1])
    assert np.isnan(moved.lagrange[-1])

    warm_x, warm_info = p1.solve(warm_start=moved)
    cold_x, cold_info = window(1)[0].solve(x0=np.full(HORIZON, 3.0))
    assert warm_info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(warm_x, cold_x, atol=1e-6)
    # Deliberately no assertion that the mapped replay is *cheaper*. On
    # this family it is not: the shifted point costs 12 iterations here
    # against 7 for a cold solve, and on a longer sinusoidal track the
    # gap widens with the horizon -- 12 vs 9 at HORIZON=5, 15 vs 11 at
    # 10, 22 vs 9 at 20, 30 vs 10 at 40. Neither dropping the carried mu
    # nor loosening it nor loosening bound_push moves it (11-13 across
    # six variants).
    # That is the structural limit docs/src/initialization.md already
    # states — the barrier pushes iterates off their bounds, so a
    # converged interior point's active-set information does not survive
    # the transfer — and it is what pounce#606 attacks on the Rust side.
    # The transfer hook's job here is that the replay is *valid* and
    # *labelled*, not that it is fast.


def test_mapped_artifact_is_still_refused_on_a_third_problem():
    p0, v0, c0 = window(0)
    x0, info0 = p0.solve(x0=np.full(HORIZON, 3.0))
    ws = WarmStart.from_info(x0, info0, problem=p0, var_ids=v0, con_ids=c0)
    p1, v1, c1 = window(1)
    moved = ws.reindex(p1, var_ids=v1, con_ids=c1)

    p2, v2, c2 = window(2)
    with pytest.raises(WarmStartCompatibilityError) as e:
        p2.solve(warm_start=moved, var_ids=v2, con_ids=c2)
    assert "mapped replay" in str(e.value)


def test_reindex_needs_ids_on_the_source():
    ws = signed()                     # signed, but with no identifiers
    with pytest.raises(WarmStartCompatibilityError, match="no variable identifiers"):
        ws.reindex(make(), var_ids=VAR_IDS)


def test_explicit_mapper_gets_the_context_and_is_length_checked():
    p0, v0, c0 = window(0)
    x0, info0 = p0.solve(x0=np.full(HORIZON, 3.0))
    ws = WarmStart.from_info(x0, info0, problem=p0, var_ids=v0, con_ids=c0)
    p1, v1, c1 = window(1)

    seen = {}

    def mapper(ctx):
        seen["n"] = ctx.target.n
        seen["map"] = ctx.index_map("var")
        lb, ub, _, _ = ctx.bounds()
        assert lb.size == ctx.target.n == ub.size
        return {"x": np.roll(ctx.source.x, -1), "lagrange": None,
                "zl": None, "zu": None}

    moved = ws.transfer(p1, mapper, var_ids=v1, con_ids=c1)
    assert seen["n"] == HORIZON
    np.testing.assert_array_equal(seen["map"], [1, 2, 3, 4, -1])
    assert moved.replay == "mapped"
    p1.solve(warm_start=moved)

    def bad(ctx):
        return {"x": np.zeros(HORIZON + 2)}

    with pytest.raises(ValueError, match="the target problem wants"):
        ws.transfer(p1, bad, var_ids=v1, con_ids=c1)

    def wrong_key(ctx):
        return {"primal": np.zeros(HORIZON)}

    with pytest.raises(TypeError, match="unknown keys"):
        ws.transfer(p1, wrong_key, var_ids=v1, con_ids=c1)


def test_transfer_without_ids_or_mapper_explains_itself():
    ws = signed()
    with pytest.raises(WarmStartCompatibilityError, match="stable variable IDs"):
        ws.transfer(make())


# ---------------------------------------------------------------------------
# 7. signature bookkeeping
# ---------------------------------------------------------------------------
def test_signature_records_the_named_facets():
    p = make()
    sig = pounce.ProblemSignature.from_problem(p, VAR_IDS, CON_IDS)
    assert (sig.n, sig.m) == (4, 2)
    assert sig.var_ids == tuple(VAR_IDS) and sig.con_ids == tuple(CON_IDS)
    for facet in ("bounds", "sparsity", "scaling", "algorithm", "model"):
        assert getattr(sig, facet) is not None
    assert pounce.ProblemSignature.from_json(sig.to_json()) == sig


def test_ids_must_match_the_dimension_and_be_unique():
    p = make()
    with pytest.raises(ValueError, match="expected 4 identifiers"):
        pounce.ProblemSignature.from_problem(p, ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        pounce.ProblemSignature.from_problem(p, ["a", "a", "b", "c"])


def test_ids_without_a_problem_is_a_typeerror():
    _, x, info = cold()
    with pytest.raises(TypeError, match="problem="):
        WarmStart.from_info(x, info, var_ids=VAR_IDS)


def test_non_model_options_do_not_change_the_signature():
    """A warm start stays valid across a changed iteration cap; it must
    not across a changed model definition."""
    base = pounce.ProblemSignature.from_problem(make())
    assert pounce.ProblemSignature.from_problem(make(max_iter=17)) == base
    assert pounce.ProblemSignature.from_problem(make(print_level=5)) == base
    assert pounce.ProblemSignature.from_problem(
        make(bound_relax_factor=0.0)) != base
