# Issue #265 — implementation plan

Fix for [#265](https://github.com/jkitchin/pounce/issues/265): PR #263
silently transposes a 2-parameter tuple pair-list box (the mirror image of
the #260 failure it fixed), and NaN bounds pass validation everywhere.

This plan was prepared against `5acbb10` (current `main`). Every behavior
claim below was re-verified empirically at that commit (§2), not taken from
the issue text.

## 1. History — why this is round three

| Round | What happened |
|---|---|
| #260 | `bounds=([0,1.6],[10,10])` (scipy's `(lower, upper)`) at n=2 was misread as a per-parameter pair list. Box silently transposed, `Solve_Succeeded`, wrong fit. |
| PR #263 | Dropped the `_is_pair_list` guard: a length-2 **tuple** is now *always* scipy's `(lower, upper)`. A pair list must be a **list**. Fixed #260's direction. |
| #265 | The same collision now fails in the mirror direction: at n=2, a pair list written as a **tuple** — `((0,10),(0,10))` or `((None,10.0),(0.0,None))` — is silently transposed. Bonus: the `None` form produces NaN bounds that pass the NaN-blind `lb > ub` check. |

The lesson from two adversary rounds: at n=2 the tuple-of-two-pairs shape is
**genuinely ambiguous**, and any silent resolution loses one direction. The
only fix that cannot be silently wrong is to make the ambiguous shape an
error, while keeping an unambiguous spelling for each reading.

## 2. Confirmed behavior at `5acbb10`

Reproduced without the compiled extension (stub-import recipe in §8.1):

```
_normalize_bound_arg(((0.0,10.0),(0.0,10.0)), 2)   -> [(0.0, 0.0), (10.0, 10.0)]   # transposed: both params pinned
_normalize_bound_arg(((None,10.0),(0.0,None)), 2)  -> [(nan, 0.0), (10.0, nan)]    # transposed + NaN
_normalize_bound_arg([(0.0,10.0),(0.0,10.0)], 2)   -> unchanged                    # list form correct
_normalize_bound_arg(([0.0,1.6],[10.0,10.0]), 2)   -> [(0.0,10.0), (1.6,10.0)]     # scipy form correct
_normalize_bound_arg(((0,10),(0,10),(0,10)), 3)    -> unchanged                    # n!=2 tuple pair list correct
_normalize_bound_arg((None,10.0), 1)               -> [(nan, 10.0)]                # NaN lower at n=1 (!)
_normalize_bounds([(float('nan'),10.0)], 1)        -> (array([nan]), array([10.])) # NaN passes validation
_minima_bounds(((None,10.0),(0.0,None)), 2)        -> [(None, 0.0), (10.0, None)]  # transposed for find_minima
```

Two defects, independently fixable, and **both are required** (either alone
leaves a hole — see the issue's first comment):

- **(A)** `_normalize_bound_arg` (`python/pounce/_curve_fit.py:1460`) silently
  resolves the ambiguous n=2 tuple shape.
- **(B)** `_normalize_bounds` (`python/pounce/_minimize.py:339`) accepts NaN
  bounds: `lb > ub` is False against NaN, so a malformed box sails through.
  Reachable directly: `pounce.minimize(..., bounds=[(float('nan'), 10.0)])`
  succeeds today.

Plus one **interaction discovered while preparing this plan**: at n=1,
`bounds=(None, 10.0)` goes down the scipy branch and `_to_array(None)` → NaN,
which today *silently behaves as unbounded-below* (accidentally the right
semantics). A naive fix (B) would turn that working call into a NaN error.
Fix (A) must therefore handle side-level `None` explicitly (§3.2) **before**
(B) lands, or in the same change.

## 3. Design — the disambiguation rule

Keep #263's convention (length-2 tuple = scipy `(lower, upper)`; pair list =
list). Layer three changes on top, all inside the scipy-tuple branch of
`_normalize_bound_arg` except (B):

### 3.1 (A) Raise on the genuinely ambiguous shape

At `n == 2`, inside the `isinstance(bounds, tuple) and len(bounds) == 2`
branch, **before** array conversion, raise `ValueError` if either element is
"pair-shaped":

```python
def _pair_shaped(el):
    # An element that reads as a pair-list entry rather than a scipy
    # lower/upper array: a length-2 tuple, or a sequence containing None
    # (scipy arrays cannot contain None; pair entries can).
    if isinstance(el, tuple) and len(el) == 2:
        return True
    if isinstance(el, (list, tuple)) and any(e is None for e in el):
        return True
    return False

if n == 2 and (_pair_shaped(lo) or _pair_shaped(hi)):
    raise ValueError(...)
```

The error message must name both readings and both disambiguating
spellings (adapt the issue's suggested wording):

```
bounds=((0.0, 10.0), (0.0, 10.0)) is ambiguous for a 2-parameter model: it
reads either as scipy's (lower, upper) — param 0 in [0.0, 0.0], param 1 in
[10.0, 10.0] — or as a per-parameter pair list — param 0 in [0.0, 10.0],
param 1 in [0.0, 10.0]. Pass a list [(l0, u0), (l1, u1)] for the pair-list
reading, or lists/arrays ([l0, l1], [u0, u1]) for the scipy reading.
```

(If an element contains `None`, computing the per-reading boxes for the
message may not be possible — fall back to the static two-spellings sentence.
Tests should match on a stable substring, e.g. `"ambiguous for a
2-parameter model"`.)

**Why inner *tuples* raise but inner *lists/arrays* stay scipy.** This is the
crux, so it is worth being explicit:

- `([0.0, 1.6], [10.0, 10.0])` — tuple of lists/arrays — is scipy's canonical
  spelling, exactly what #260's reporter passed and what #263's regression
  tests pin (`test_curve_fit.py:519/:551/:582` all use list elements; the
  issue's follow-up comment confirms they are expected to stay green). Raising
  on it would re-break scipy drop-in compatibility at n=2, i.e. undo #263.
- `((0.0, 10.0), (0.0, 10.0))` — tuple of tuples — is the pair-list spelling
  pounce itself used pre-#263 and the shape both adversary rounds abused.
  Nobody writes scipy bounds as a tuple of tuples of scalars *at exactly
  n=2* with colliding intent that we could safely guess.
- Elements containing `None` can never be a valid scipy array (scipy uses
  ±inf), so `((None, 10.0), (0.0, None))` and `([None, 10.0], [0.0, None])`
  both raise — the former via the tuple rule, the latter via the None rule.
  This kills the NaN-injection path at its source, with a message far better
  than a downstream NaN error.

The residual asymmetry — a user who writes a pair list as a tuple of *lists*
`([0, 10], [0, 10])` at n=2 still gets the scipy reading silently — is
irreducible: that spelling is byte-for-byte scipy's documented form, and the
docstring (since #263) says pair lists are lists. Document it; don't guess.

**Scope guard:** the raise applies only at `n == 2`. At n=1 and n≥3 a
length-2 tuple of length-2 sequences already fails the existing
length-vs-n check ("bounds lower has length 2 but the problem has N
parameter(s)"), and a length-n tuple pair list (`((0,10),(0,10),(0,10))` at
n=3) never enters the scipy branch (len ≠ 2) — verified correct today; pin
with tests.

### 3.2 Side-level `None` maps to ∓inf in the scipy branch

Still inside the scipy branch, before `_to_array`:

```python
if lo is None: lo = -np.inf
if hi is None: hi = np.inf
```

Rationale: preserves today's (accidental) n=1 `(None, 10.0)` = unbounded-below
behavior once NaN rejection (B) lands; `None` = "no bound" is pounce's
convention everywhere else; and at any n the pair-list reading of a bare
`(None, x)` tuple is invalid, so this is unambiguous. Note `_pair_shaped`
deliberately does *not* treat bare `None` as pair-shaped — `(None, None)`
and `(None, 10.0)` take the scipy reading, where both readings' semantics
coincide anyway.

After conversion, reject NaN inside scipy arrays with a targeted message
(catches `([None, 0], [10, 10])` at n≠2, and literal NaN at any n):

```python
for side, arr in (("lower", lo_a), ("upper", hi_a)):
    if np.isnan(arr).any():
        raise ValueError(
            f"bounds {side} contains NaN (or None inside an array); "
            f"use -inf/inf for an unbounded side, or pass a per-parameter "
            f"list of (lo, hi) pairs"
        )
```

### 3.3 (B) `_normalize_bounds` rejects NaN

In `python/pounce/_minimize.py:339`, immediately before the reversed-bound
check (both the `Bounds` path and the legacy pair path flow through there):

```python
for side, arr in (("lower", lb), ("upper", ub)):
    bad = np.where(np.isnan(arr))[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"bounds[{i}] has a NaN {side} bound; use -inf/inf "
            f"(or None in the pair-list form) for an unbounded side"
        )
```

Must be `np.isnan`, **not** `~np.isfinite`: ±inf is the one-sided encoding
this very function produces (`:356-357`) and must stay legal.

This is a deliberate behavior change: NaN was previously a silent
"no bound" sentinel on this path. In-repo grep shows nobody relies on it;
changelog entry required (§6).

### 3.4 (B′) `_minima_bounds` must not launder NaN into `None`

`_curve_fit.py:872` maps non-finite → `None` via `np.isfinite`, so a NaN pair
entry (`curve_fit_minima(..., bounds=[(float('nan'), 10)])`) silently becomes
"unbounded" — inconsistent with the minimize path after (B). Add an explicit
NaN check in the loop (raise, same message shape as 3.3) while keeping
±inf → `None`. Note `lo`/`hi` may be `None` here — guard before `np.isnan`.

## 4. Files to change

| File | Change |
|---|---|
| `python/pounce/_curve_fit.py:1460` `_normalize_bound_arg` | §3.1 raise, §3.2 None→∓inf + NaN check; rewrite docstring (it currently asserts "a length-2 tuple is always read the scipy way" — no longer unconditionally true). |
| `python/pounce/_curve_fit.py:872` `_minima_bounds` | §3.4 NaN rejection. |
| `python/pounce/_minimize.py:339` `_normalize_bounds` | §3.3 NaN rejection before the reversed check. |
| `python/pounce/_curve_fit.py:920-928` `curve_fit` docstring | Document: tuple = scipy form; list = pair list; ambiguous n=2 tuple-of-pairs raises; None sides OK. Mention #265. `curve_fit_minima` / `curve_fit_streaming` docstrings inherit by reference — check they don't restate the old rule. |
| `docs/src/python.md:297` (`minimize` bounds row) | Note NaN bounds now raise. |
| `docs/src/curve-fitting.md` | If it shows tuple-form bounds anywhere, align wording; its examples at `:110/:263` already use the list form — likely no change, but check. |
| `CHANGELOG.md` `## [Unreleased]` | New `### Fixed` entry (§6). |
| `python/tests/test_curve_fit.py`, `python/tests/test_minimize.py` (or wherever `_normalize_bounds` tests live — check) | §5. |

All three public surfaces (`curve_fit`, `curve_fit_minima`,
`curve_fit_streaming`) route through `_normalize_bound_arg`
(`_curve_fit.py:629`, `:876`, `:1212`), so no per-surface parsing changes —
only tests to prove each inherits the fix.

## 5. Test plan

Existing tests that MUST stay green unchanged (they encode #263's fix and
all use list elements inside the tuple, which stays the scipy reading):

- `test_curve_fit.py:519` `test_scipy_tuple_bounds_two_params_not_misread_as_pair_list`
- `test_curve_fit.py:551` `test_scipy_tuple_bounds_match_scipy_across_param_counts`
- `test_curve_fit.py:582` `test_scipy_tuple_bounds_reversed_looking_box_is_valid`

New tests — unit level, directly on `_normalize_bound_arg` (fast, no solver):

| Input (n) | Expected |
|---|---|
| `((0.0,10.0),(0.0,10.0))` (2) | raises, message contains "ambiguous for a 2-parameter model" |
| `((None,10.0),(0.0,None))` (2) | raises (ambiguous) |
| `([None,10.0],[0.0,None])` (2) | raises (None inside array → the §3.2 message or ambiguity message; pick one and assert it) |
| `((0.0,10.0), None)` (2) | raises (element 0 pair-shaped) |
| `[(0.0,10.0),(0.0,10.0)]` (2) | pair list, unchanged |
| `[(None,10.0),(0.0,None)]` (2) | pair list, unchanged |
| `([0.0,1.6],[10.0,10.0])` (2) | scipy: `[(0.0,10.0),(1.6,10.0)]` |
| `(np.array([0.0,1.6]), np.array([10.0,10.0]))` (2) | scipy, same as above |
| `(0, 10)` (1, 2, 3) | scipy scalar broadcast |
| `(None, 10.0)` (1) | `[(-inf, 10.0)]` — the §3.2 regression guard |
| `(None, 10.0)` (2) | scipy: both params `(-inf, 10.0)` |
| `((0,10),(0,10),(0,10))` (3) | pair list, unchanged (n≠2 pin) |
| `((0,10),)` (1) | pair list, unchanged (len≠2 tuple) |
| `([0,0,0],[10,10,10])` (3) | scipy, unchanged |
| `(nan, 10.0)` (2) | raises (NaN in scipy branch) |

New tests — end-to-end, one per public surface, on the issue's model
(`f(x; A, k) = A·exp(-k·x)`, noiseless data at A=2, k=1, deterministic):

- `curve_fit` with `bounds=((0.0,10.0),(0.0,10.0))` → `pytest.raises(ValueError, match="ambiguous")`.
- `curve_fit` with `bounds=[(None,10.0),(0.0,None)]` (list spelling) →
  `popt ≈ (2.0, 1.0)`; cross-check against
  `scipy.optimize.curve_fit(..., bounds=([-np.inf,0.0],[10.0,np.inf]))`.
- `curve_fit_minima` with the ambiguous tuple → raises (covers `_minima_bounds`);
  with the list spelling → best minimum ≈ (2, 1) and `perr` not identically 0.
- `curve_fit_streaming` with the ambiguous tuple → raises; with
  `bounds=([0.0,0.0],[10.0,10.0])` → ≈ (2, 1).

New tests — `_normalize_bounds` (B):

- `[(float('nan'), 10.0)]` → raises "NaN".
- `scipy.optimize.Bounds(np.nan, 10.0)` → raises.
- `[(-np.inf, np.inf)]` and `[(None, None)]` → still pass, lb/ub = ∓inf.
- End-to-end: `pounce.minimize(lambda v: (v[0]-3.0)**2, x0=[0.0],
  bounds=[(float('nan'), 10.0)])` → raises (today it returns success).
- `curve_fit_minima(..., bounds=[(float('nan'), 10.0), (0.0, 10.0)])` → raises (§3.4).

## 6. Changelog entry (sketch)

Under `## [Unreleased]`, `### Fixed`:

- `curve_fit`/`curve_fit_minima`/`curve_fit_streaming`: at n=2, a length-2
  **tuple** of `(lo, hi)` pairs (e.g. `((0,10),(0,10))`) was silently read as
  scipy's `(lower, upper)` — the transposed box pinned both parameters and
  still reported `Solve_Succeeded` (#265, the mirror of #260). The ambiguous
  shape now raises with both unambiguous spellings; list-of-pairs and
  tuple-of-arrays forms are unchanged.
- NaN bounds are now rejected in `minimize` and all `curve_fit` surfaces
  (previously they silently passed the reversed-bound check and behaved as
  "no bound"). `None`/±inf remain the supported unbounded spellings.

## 7. Implementation order

1. §3.2 (None→∓inf + scipy-branch NaN check) and §3.1 (ambiguity raise) in
   `_normalize_bound_arg`, together — one commit.
2. §3.3 + §3.4 NaN rejection — second commit (independent; keep separable
   for review).
3. Tests + docs + changelog alongside each.

## 8. Verification protocol

### 8.1 Fast loop, no Rust build

The normalization layer is pure Python. Verify the full §5 unit matrix
without building `pounce._pounce` (numpy + scipy required):

```python
# run from python/
import sys, types, importlib.util
pkg = types.ModuleType('pounce'); pkg.__path__ = ['pounce']; sys.modules['pounce'] = pkg
stub = types.ModuleType('pounce._pounce')
for name in ('Solver', 'Problem'):
    setattr(stub, name, type(name, (), {}))
sys.modules['pounce._pounce'] = stub
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
m  = load('pounce._minimize',  'pounce/_minimize.py')
cf = load('pounce._curve_fit', 'pounce/_curve_fit.py')
# ... assert the §5 unit matrix against cf._normalize_bound_arg / m._normalize_bounds
```

(This recipe was used to produce §2; it works at `5acbb10`. If the stub
list drifts, add whatever names the import error asks for.)

### 8.2 Full suite

```bash
cd python && maturin develop --release   # or: pip install -e . in a venv
python -m pytest tests/test_curve_fit.py tests/test_minimize.py -q
python -m pytest tests -q                # full suite, mirrors CI's python-test job
```

Expected: currently 41 tests in `test_curve_fit.py` (49 asserted in the
issue includes parametrized expansion) — all green plus the new ones. CI
(`ci.yml` `python-test`) runs `pytest python/tests -q` on the built wheel.

### 8.3 Issue reproduction

The issue's `adversary/runs/...` scripts are NOT in this repo — don't look
for them. Re-derive the check from the issue table:

```python
import numpy as np, pounce, scipy.optimize as sopt
x = np.linspace(0, 3, 60); y = 2.0 * np.exp(-1.0 * x)
model = lambda t, A, k: A * np.exp(-k * t)
# rows 1 & 3 of the issue table must now raise ValueError("...ambiguous..."):
#   bounds=((0.0,10.0),(0.0,10.0))   and   bounds=((None,10.0),(0.0,None))
# rows 2 & 4 (list spelling) must return popt ≈ (2.0, 1.0), matching
#   sopt.curve_fit(model, x, y, p0=[1.0,2.0], bounds=([-np.inf,0.0],[10.0,np.inf]))
# and the #260 repro must STILL match scipy:
#   bounds=([0.0,1.6],[10.0,10.0]) -> popt ≈ [2.42445211, 1.6]
```

Acceptance: all five checks pass, plus
`pounce.minimize(lambda v: (v[0]-3.0)**2, x0=[0.0], bounds=[(float('nan'),10.0)])`
raises.

### 8.4 Adversarial self-review before pushing

Re-run the §5 matrix and ask, for each row: "could this input mean something
else under the other reading, and did we silently pick?" The invariant to
hold: **every accepted input has exactly one valid reading; every input with
two valid readings at n=2 raises.**

## 9. Scope notes

- `curve_fit_minima` reporting `pcov = 0`, `perr = [0, 0]` for a degenerate
  (zero-width) box — "infinite confidence in a wrong answer". Originally
  deferred here as possibly its own issue, but **pulled into this PR** (owner
  decision, PR #269). Fixed by warning — not by changing any number — at the
  single shared result-assembly site in `_solve_fit` (`_curve_fit.py`), which
  all three surfaces route through. The covariance projection is correct,
  intentional, documented behavior (`_projected_covariance` zeros variance
  along active directions; `_active_constraint_jac` supports `lo == hi` as the
  fix-a-parameter idiom); the only defect was that a resulting `perr = 0` was
  silent. Two warnings: a zero-width-bound case (naming the pinned params) and
  an all-params-on-active-bounds fully-degenerate case. Docstring, changelog,
  and tests updated alongside.
- Any change to `pounce.minimize`'s pair-list API itself — it was never
  ambiguous (no scipy tuple form there). Still out of scope.
